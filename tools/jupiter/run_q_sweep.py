#!/usr/bin/env python3
"""
Run a Jupiter q(x,y) calibration sweep on Windows/Linux without relying on
PowerShell pipeline semantics.

Each calibration point is executed in a fresh llama-cli process with stdin
closed, --single-turn, and --simple-io. stdout/stderr are merged into one UTF-8
log, then jupiter_planner.py extracts q samples. Completed CSVs can be reused
with --resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


PLAN_RE = re.compile(
    r"\[JUPITER_SEQUENCE_PLAN\].*?total=(\d+).*?chunks=([0-9,]+)"
)
UBATCH_RE = re.compile(
    r"\[JUPITER_UBATCH\].*?event=BEGIN.*?"
    r"prefix_tokens=(\d+).*?tokens=(\d+)"
)
TASK_TOKENS_RE = re.compile(r"task\.n_tokens\s*=\s*(\d+)")
PROMPT_EVAL_RE = re.compile(r"prompt eval time\s*=.*?/\s*(\d+)\s+tokens")


class PrefillLengthCorrection(Exception):
    """The calibration prefill batch differs from the initial probe."""

    def __init__(self, actual_tokens: int, expected_tokens: int, log_path: Path):
        self.actual_tokens = actual_tokens
        self.expected_tokens = expected_tokens
        self.log_path = log_path
        super().__init__(
            f"prefill batch changed: probe={expected_tokens} "
            f"calibration={actual_tokens}; log={log_path}"
        )


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(int(raw))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer: {raw!r}") from exc
    if not out:
        raise argparse.ArgumentTypeError("list must not be empty")
    return out


def split_feasible_tail(total: int, minimum: int, maximum: int) -> List[int]:
    if total < minimum:
        raise ValueError(f"tail {total} is smaller than min chunk {minimum}")

    count = (total + maximum - 1) // maximum
    while count > 1 and total // count < minimum:
        count -= 1

    if count <= 0:
        raise ValueError(f"cannot split tail {total} into [{minimum},{maximum}]")

    base, extra = divmod(total, count)
    peak = base + (1 if extra else 0)
    if base < minimum or peak > maximum:
        raise ValueError(f"cannot split tail {total} into [{minimum},{maximum}]")

    return [base + (1 if i < extra else 0) for i in range(count)]


def make_sweep_plan(
    total: int,
    x: int,
    minimum: int,
    maximum: int,
) -> List[int]:
    if not minimum <= x <= maximum:
        raise ValueError(f"x={x} is outside [{minimum},{maximum}]")
    if total < minimum:
        raise ValueError(f"target token count {total} is too small")

    n, remainder = divmod(total, x)
    if n == 0:
        return split_feasible_tail(total, minimum, maximum)

    if remainder == 0:
        return [x] * n

    if remainder >= minimum:
        return [x] * n + [remainder]

    # Keep as many exact-x samples as possible. Fold the tiny remainder into
    # the last x-sized chunk, then rebalance only that tail.
    return [x] * (n - 1) + split_feasible_tail(
        x + remainder, minimum, maximum
    )


def print_sweep_plans(
    total: int,
    xs: Sequence[int],
    minimum: int,
    maximum: int,
) -> List[tuple[int, List[int]]]:
    plans: List[tuple[int, List[int]]] = []
    for x in xs:
        chunks = make_sweep_plan(total, x, minimum, maximum)
        if sum(chunks) != total:
            raise RuntimeError(
                f"internal plan error for x={x}: sum={sum(chunks)} target={total}"
            )
        if any(c < minimum or c > maximum for c in chunks):
            raise RuntimeError(
                f"internal plan error for x={x}: chunks={chunks} "
                f"outside [{minimum},{maximum}]"
            )
        plans.append((x, chunks))
        print(
            f"[JUPITER_SWEEP_PLAN] x={x} total={total} "
            f"chunks={','.join(map(str, chunks))}"
        )
    return plans


def should_echo_runtime_line(line: str) -> bool:
    """Keep the console quiet; all runtime lines still go to the log file."""
    return (
        "[JUPITER_SEQUENCE_PLAN]" in line
        or "[JUPITER_SEQUENCE_DECISION]" in line
        or "prompt eval time" in line
        or line.lstrip().startswith(("Error:", "FATAL:", "error:"))
    )


def stream_process(
    cmd: Sequence[str],
    env: dict[str, str],
    log_path: Path,
) -> int:
    """
    Run one llama-cli process.

    Important details:
    - stdin=DEVNULL prevents a calibration process from waiting for a second
      interactive turn.
    - stderr=STDOUT avoids PowerShell's NativeCommandError behavior.
    - bytes are decoded by us, so one malformed model-output byte cannot kill
      the runner.
    """
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        proc = subprocess.Popen(
            list(cmd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        assert proc.stdout is not None

        pending = b""
        while True:
            block = proc.stdout.read(4096)
            if not block:
                break
            pending += block
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").rstrip("\r")
                if should_echo_runtime_line(line):
                    print(line, flush=True)
                log.write(line + "\n")

        if pending:
            line = pending.decode("utf-8", errors="replace").rstrip("\r")
            if should_echo_runtime_line(line):
                print(line, flush=True)
            log.write(line + "\n")

        return proc.wait()


def reconstruct_plan_from_log(
    log_path: Path,
    target_tokens: int,
    min_chunk: int,
) -> tuple[List[int] | None, str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")

    matches = list(PLAN_RE.finditer(text))
    if matches:
        total = int(matches[-1].group(1))
        chunks = parse_int_list(matches[-1].group(2))
        if total == target_tokens:
            return chunks, "SEQUENCE_PLAN"

    # Fallback: rebuild the formal prompt partition from BEGIN records. Warmup
    # batches such as tokens=2 are excluded by min_chunk.
    samples: List[tuple[int, int]] = []
    for match in UBATCH_RE.finditer(text):
        prefix = int(match.group(1))
        tokens = int(match.group(2))
        if tokens >= min_chunk:
            samples.append((prefix, tokens))

    # Search for a contiguous chain beginning at prefix 0 and ending exactly
    # at target_tokens. This remains robust if unrelated warmup/decode records
    # are present before or after the formal prompt.
    for start, (prefix, _tokens) in enumerate(samples):
        if prefix != 0:
            continue
        expected = 0
        chunks: List[int] = []
        for pfx, tok in samples[start:]:
            if pfx != expected:
                break
            chunks.append(tok)
            expected += tok
            if expected == target_tokens:
                return chunks, "UBATCH_RECONSTRUCT"
            if expected > target_tokens:
                break

    return None, "NONE"


def longest_prefill_ubatch_chain(log_text: str) -> int:
    """Recover the actual decode-batch token count from Jupiter runtime traces.

    The server's task.n_tokens may include a token handled separately by the
    prompt cache / generation pipeline. The sequence plan is applied to the
    llama_context::decode() batch, so that batch is the authoritative count.
    Warmup has a very short chain, which is ignored when longer chains exist.
    """
    batches = [
        (int(m.group(1)), int(m.group(2)))
        for m in UBATCH_RE.finditer(log_text)
    ]
    longest = 0
    for i, (prefix, _) in enumerate(batches):
        if prefix != 0:
            continue
        end = 0
        for pfx, tokens in batches[i:]:
            if pfx != end or tokens <= 0:
                break
            end += tokens
        longest = max(longest, end)
    return longest


def infer_runtime_prompt_tokens(log_path: Path) -> tuple[int, str]:
    """Detect the token count passed to the actual prefill decode batch."""
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    task_counts = [
        int(m.group(1)) for m in TASK_TOKENS_RE.finditer(log_text)
    ]
    timing_counts = [
        int(m.group(1)) for m in PROMPT_EVAL_RE.finditer(log_text)
    ]
    ubatch_total = longest_prefill_ubatch_chain(log_text)

    available = []
    if ubatch_total > 2:
        available.append(("ubatch_chain", ubatch_total))
    if task_counts:
        available.append(("task.n_tokens", task_counts[-1]))
    if timing_counts:
        available.append(("prompt_eval", timing_counts[-1]))
    if not available and ubatch_total > 0:
        available.append(("ubatch_chain", ubatch_total))

    if not available:
        raise RuntimeError(
            "token probe produced no prompt length. Check the probe log: "
            f"{log_path}; expected JUPITER_UBATCH BEGIN, task.n_tokens, "
            "or prompt eval time records."
        )

    # The offline plan is matched against balloc.get_n_tokens() in
    # llama_kv_cache::init_batch. That count is equal to the sum of actual
    # JUPITER_UBATCH BEGIN tokens in the prefill decode batch.
    source, total = available[0]
    if total <= 0:
        raise RuntimeError(f"invalid prompt token count {total}: {log_path}")

    report = " ".join(f"{name}={count}" for name, count in available)
    print(f"[JUPITER_SWEEP_TOKEN_SOURCES] {report}", flush=True)

    for name, count in available[1:]:
        if count != total:
            print(
                f"[JUPITER_SWEEP_TOKEN_WARNING] "
                f"{name}={count} differs from {source}={total}; "
                "using actual prefill batch length for the plan.",
                file=sys.stderr, flush=True,
            )
    return total, source


def build_cli_command(
    args: argparse.Namespace, exe: Path, model: Path, prompt_text: str
) -> List[str]:
    # Same CLI options for the token probe and all measured calibrations.
    # Close stdin in stream_process and request exactly one generation turn.
    return [
        str(exe),
        "-m", str(model),
        "--rpc", args.rpc,
        "--split-mode", "layer",
        "-ngl", "all",
        "-ub", str(args.max_chunk),
        "-b", str(args.batch),
        "-c", str(args.context),
        "-n", "1",
        "-t", str(args.threads),
        "--flash-attn", "on",
        # The q-sweep opt-in makes only required Jupiter calibration records
        # visible at -lv 3; other TRACE-level runtime noise stays filtered.
        # Requires the matching common/log.cpp change in the PC binary.
        "-lv", "3",
        "--no-conversation",
        "--single-turn",
        "--simple-io",
        "-p", prompt_text,
        *args.extra_arg,
    ]


def run_checked(cmd: Sequence[str], cwd: Path | None = None) -> None:
    shown = subprocess.list2cmdline(list(cmd)) if os.name == "nt" else " ".join(cmd)
    print(f"[JUPITER_SWEEP_TOOL] {shown}", flush=True)
    proc = subprocess.run(list(cmd), cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tool failed with exit={proc.returncode}: {shown}"
        )


def merge_csvs(csv_paths: Iterable[Path], output: Path) -> tuple[int, int]:
    rows: List[dict[str, str]] = []
    fieldnames: List[str] | None = None

    for path in csv_paths:
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and fieldnames is None:
                fieldnames = list(reader.fieldnames)
            rows.extend(dict(row) for row in reader)

    if not rows or not fieldnames:
        raise RuntimeError("sweep produced no q samples")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    unique_xy = {(row.get("x"), row.get("y")) for row in rows}
    return len(rows), len(unique_xy)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automated Jupiter q(x,y) calibration sweep"
    )
    p.add_argument("--exe", required=True, help="llama-cli executable")
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True)
    p.add_argument("--prompt-file", required=True)

    p.add_argument(
        "--target-tokens", type=int, default=None,
        help="optional hint for a dry run; normal runs use llama-cli's "
             "actual token count and override any mismatched hint",
    )
    p.add_argument(
        "--x-values",
        type=parse_int_list,
        default=parse_int_list("32,48,64,80,96,112,128"),
    )
    p.add_argument("--min-chunk", type=int, default=32)
    p.add_argument("--max-chunk", type=int, default=128)
    p.add_argument("--pc-layers", type=int, default=25)
    p.add_argument("--repeats", type=int, default=1)

    p.add_argument("--out-dir", default="jupiter-q-sweep")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)

    p.add_argument("--run-dp", action="store_true")
    p.add_argument("--plan-output", default="jupiter.plan")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="extra llama-cli argument; repeat this option for multiple args",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.min_chunk <= 0 or args.max_chunk < args.min_chunk:
        raise SystemExit(
            f"invalid chunk range [{args.min_chunk},{args.max_chunk}]"
        )
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    prompt_file = Path(args.prompt_file).resolve()
    if not exe.is_file():
        raise SystemExit(f"llama-cli not found: {exe}")
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    if not prompt_file.is_file():
        raise SystemExit(f"prompt file not found: {prompt_file}")

    script_dir = Path(__file__).resolve().parent
    planner = script_dir / "jupiter_planner.py"
    if not planner.is_file():
        raise SystemExit(f"planner not found: {planner}")

    # Text editors commonly leave a final CR/LF. Passing that newline through
    # -p can change tokenization versus an interactively pasted prompt, which in
    # turn makes LLAMA_JUPITER_SEQUENCE_PLAN silently inapplicable because its
    # chunk sum no longer matches the runtime batch length. Preserve all prompt
    # content except trailing line terminators.
    prompt_text = prompt_file.read_text(encoding="utf-8-sig").rstrip("\r\n")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dry-run intentionally does not launch the model. A numeric hint can be
    # used to preview plans, but real calibration always probes the tokenizer.
    if args.dry_run:
        if args.target_tokens is None:
            print(
                "[JUPITER_SWEEP_DRY_RUN] supply --target-tokens N "
                "to preview plans; a real run detects N automatically."
            )
            return 0
        print_sweep_plans(
            args.target_tokens, args.x_values,
            args.min_chunk, args.max_chunk,
        )
        return 0

    base_env = os.environ.copy()
    base_env["LLAMA_JUPITER_PREFILL"] = "1"
    base_env["LLAMA_JUPITER_PC_LAYERS"] = str(args.pc_layers)
    base_env["LLAMA_JUPITER_WINDOW"] = "1"
    base_env["LLAMA_JUPITER_TRACE"] = "1"
    base_env["LLAMA_JUPITER_PROFILE_STAGE"] = "1"
    base_env["LLAMA_JUPITER_Q_SWEEP"] = "1"
    base_env["LLAMA_JUPITER_LOGITS_TOP10"] = "0"
    # Keep q(x,y) calibration on the original communication contract; a
    # transport A/B needs fresh measurements and should use the bench runner.
    base_env["LLAMA_JUPITER_ASYNC_C2P"] = "0"
    base_env.pop("LLAMA_JUPITER_PLAN_FILE", None)
    base_env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)

    # First run once without a sequence override. The model's actual tokenizer
    # and chat/prompt processing determine the exact prefill length. Without
    # this probe, a one-token difference silently disables the explicit plan.
    probe_log = out_dir / "q-token-probe.log"
    probe_env = base_env.copy()
    probe_env["LLAMA_JUPITER_PROFILE_STAGE"] = "0"

    cmd = build_cli_command(args, exe, model, prompt_text)
    print(
        f"[JUPITER_SWEEP_PROBE] action=BEGIN log={probe_log}",
        flush=True,
    )
    exit_code = stream_process(cmd, probe_env, probe_log)
    if exit_code != 0:
        raise RuntimeError(
            f"token probe failed with exit={exit_code}; log={probe_log}"
        )

    probe_text = probe_log.read_text(encoding="utf-8", errors="replace")
    if not UBATCH_RE.search(probe_text):
        raise RuntimeError(
            "q-sweep prefill UBATCH records are missing at -lv 3. "
            "Update common/log.cpp and rebuild the PC llama-cli so "
            "LLAMA_JUPITER_Q_SWEEP=1 exposes calibration records. "
            f"Probe log: {probe_log}"
        )
    total_tokens, source = infer_runtime_prompt_tokens(probe_log)
    print(
        f"[JUPITER_SWEEP_TOKENS] actual={total_tokens} source={source} "
        f"requested={args.target_tokens if args.target_tokens is not None else 'auto'}",
        flush=True,
    )
    if args.target_tokens is not None and args.target_tokens != total_tokens:
        print(
            f"[JUPITER_SWEEP_TARGET_OVERRIDE] "
            f"requested={args.target_tokens} actual={total_tokens} "
            "action=USE_RUNTIME_TOKEN_COUNT",
            flush=True,
        )

    if total_tokens > args.context or total_tokens > args.batch:
        raise RuntimeError(
            f"actual prompt has {total_tokens} tokens, which exceeds "
            f"--context={args.context} or --batch={args.batch}; "
            "increase the limits so the runtime can apply a single "
            "full-prompt Jupiter sequence plan."
        )

    for correction_attempt in range(2):
        try:
            return run_sweep_for_length(
                args, total_tokens, base_env, out_dir,
                exe, model, prompt_text, planner,
            )
        except PrefillLengthCorrection as exc:
            if correction_attempt != 0:
                raise RuntimeError(
                    "prefill token count remained unstable across retries: "
                    f"{exc}"
                ) from exc
            total_tokens = exc.actual_tokens
            print(
                "[JUPITER_SWEEP_TOKEN_RETRY] "
                f"probe={exc.expected_tokens} "
                f"calibration={total_tokens} "
                "action=REGENERATE_ALL_PARTITIONS",
                flush=True,
            )

    raise RuntimeError("internal error: token correction attempts exhausted")


def run_sweep_for_length(
    args: argparse.Namespace,
    total_tokens: int,
    base_env: dict[str, str],
    out_dir: Path,
    exe: Path,
    model: Path,
    prompt_text: str,
    planner: Path,
) -> int:
    plans = print_sweep_plans(
        total_tokens, args.x_values,
        args.min_chunk, args.max_chunk,
    )

    # Segregate artifacts by actual token count and configuration so --resume
    # will not reuse CSVs from a different prompt, hardware split or length.
    signature_input = "\\n".join([
        prompt_text, str(model), args.rpc,
        str(args.pc_layers), str(args.batch),
        str(args.context), str(args.threads),
        str(args.max_chunk), str(args.min_chunk),
    ])
    signature = hashlib.sha256(
        signature_input.encode("utf-8")
    ).hexdigest()[:12]
    run_dir = out_dir / f"tokens-{total_tokens}-{signature}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[JUPITER_SWEEP_OUTPUT] dir={run_dir}",
        flush=True,
    )

    csv_paths: List[Path] = []

    for x, chunks in plans:
        plan_text = ",".join(map(str, chunks))

        for repeat in range(1, args.repeats + 1):
            stem = f"q-x{x}-r{repeat}"
            log_path = run_dir / f"{stem}.log"
            csv_path = run_dir / f"{stem}.csv"

            if args.resume and log_path.is_file() and csv_path.is_file():
                recorded, _ = reconstruct_plan_from_log(
                    log_path, total_tokens, args.min_chunk,
                )
                with csv_path.open(
                    "r", newline="", encoding="utf-8-sig"
                ) as existing_csv:
                    has_samples = next(csv.DictReader(existing_csv), None) is not None
                if recorded == chunks and has_samples:
                    print(
                        f"[JUPITER_SWEEP_RESUME] x={x} "
                        f"repeat={repeat}/{args.repeats} action=SKIP "
                        f"csv={csv_path}",
                        flush=True,
                    )
                    csv_paths.append(csv_path)
                    continue
                print(
                    f"[JUPITER_SWEEP_RESUME] x={x} "
                    f"repeat={repeat}/{args.repeats} "
                    "action=RERUN reason=INCOMPLETE_OR_MISMATCHED",
                    flush=True,
                )

            env = base_env.copy()
            env["LLAMA_JUPITER_SEQUENCE_PLAN"] = plan_text

            cmd = build_cli_command(args, exe, model, prompt_text)

            print(
                f"\n[JUPITER_SWEEP_RUN] x={x} "
                f"repeat={repeat}/{args.repeats} chunks={plan_text}",
                flush=True,
            )

            exit_code = stream_process(cmd, env, log_path)
            if exit_code != 0:
                raise RuntimeError(
                    f"llama-cli failed for x={x}, repeat={repeat}, "
                    f"exit={exit_code}; log={log_path}"
                )

            actual_chunks, source = reconstruct_plan_from_log(
                log_path,
                total_tokens,
                args.min_chunk,
            )
            if actual_chunks is None:
                log_text = log_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                tail = log_text.splitlines()[-60:]
                task_tokens = [
                    int(m.group(1)) for m in TASK_TOKENS_RE.finditer(log_text)
                ]
                eval_tokens = [
                    int(m.group(1)) for m in PROMPT_EVAL_RE.finditer(log_text)
                ]
                observed_ubatches = [
                    (int(m.group(1)), int(m.group(2)))
                    for m in UBATCH_RE.finditer(log_text)
                ]
                actual_prompt_tokens = task_tokens[-1] if task_tokens else None
                actual_eval_tokens = eval_tokens[-1] if eval_tokens else None

                actual_batch_tokens = longest_prefill_ubatch_chain(log_text)
                print(
                    "[JUPITER_SWEEP_DIAG] "
                    f"probed_tokens={total_tokens} "
                    f"task_tokens={actual_prompt_tokens} "
                    f"prompt_eval_tokens={actual_eval_tokens} "
                    f"actual_batch_tokens={actual_batch_tokens} "
                    f"plan_sum={sum(chunks)} "
                    f"plan_lines={len(PLAN_RE.findall(log_text))} "
                    f"ubatches={observed_ubatches[:24]} "
                    f"log={log_path}",
                    file=sys.stderr, flush=True,
                )
                if actual_batch_tokens > 2 and actual_batch_tokens != total_tokens:
                    raise PrefillLengthCorrection(
                        actual_batch_tokens, total_tokens, log_path
                    )
                if actual_prompt_tokens is not None and actual_prompt_tokens != total_tokens:
                    print(
                        "[JUPITER_SWEEP_DIAG] "
                        f"server_task_tokens={actual_prompt_tokens} "
                        f"decode_batch_tokens={total_tokens}; "
                        "these are different quantities and do not by "
                        "themselves indicate a sequence-plan failure.",
                        file=sys.stderr, flush=True,
                    )

                for line in tail:
                    print(line, file=sys.stderr)
                raise RuntimeError(
                    f"runtime did not apply the expected sequence plan for "
                    f"x={x}, tokens={total_tokens}; "
                    f"inspect [JUPITER_SWEEP_DIAG] and {log_path}"
                )

            actual_text = ",".join(map(str, actual_chunks))
            print(
                f"[JUPITER_SWEEP_PLAN_CHECK] x={x} repeat={repeat} "
                f"source={source} chunks={actual_text}",
                flush=True,
            )
            if actual_chunks != chunks:
                raise RuntimeError(
                    f"runtime plan mismatch for x={x}: "
                    f"expected={plan_text} actual={actual_text} "
                    f"source={source}"
                )

            run_checked([
                sys.executable,
                str(planner),
                "extract-q",
                "--log", str(log_path),
                "--output", str(csv_path),
                "--metric", "compute_ms",
                "--min-tokens", str(args.min_chunk),
            ])
            csv_paths.append(csv_path)

    profile_path = run_dir / "q-profile-all.csv"
    rows, unique_xy = merge_csvs(csv_paths, profile_path)
    print(
        f"\n[JUPITER_SWEEP_DONE] rows={rows} unique_xy={unique_xy} "
        f"profile={profile_path}",
        flush=True,
    )

    if args.run_dp:
        plan_output = Path(args.plan_output).resolve()
        run_checked([
            sys.executable,
            str(planner),
            "sequence",
            "--profile", str(profile_path),
            "--output", str(plan_output),
            "--devices", "2",
            "--min-chunk", str(args.min_chunk),
            "--max-chunk", str(args.max_chunk),
            "--target-length", str(total_tokens),
        ])
        print(f"[JUPITER_SWEEP_DP] plan={plan_output}", flush=True)
        print(plan_output.read_text(encoding="utf-8"), end="")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[JUPITER_SWEEP_STOP] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[JUPITER_SWEEP_ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
