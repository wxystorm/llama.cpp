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
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()

        if pending:
            line = pending.decode("utf-8", errors="replace").rstrip("\r")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

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

    p.add_argument("--target-tokens", type=int, default=358)
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

    prompt_text = prompt_file.read_text(encoding="utf-8-sig")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plans = print_sweep_plans(
        args.target_tokens,
        args.x_values,
        args.min_chunk,
        args.max_chunk,
    )
    if args.dry_run:
        return 0

    base_env = os.environ.copy()
    base_env["LLAMA_JUPITER_PREFILL"] = "1"
    base_env["LLAMA_JUPITER_PC_LAYERS"] = str(args.pc_layers)
    base_env["LLAMA_JUPITER_WINDOW"] = "1"
    base_env["LLAMA_JUPITER_TRACE"] = "1"
    base_env["LLAMA_JUPITER_PROFILE_STAGE"] = "1"
    base_env.pop("LLAMA_JUPITER_PLAN_FILE", None)

    csv_paths: List[Path] = []

    for x, chunks in plans:
        plan_text = ",".join(map(str, chunks))

        for repeat in range(1, args.repeats + 1):
            stem = f"q-x{x}-r{repeat}"
            log_path = out_dir / f"{stem}.log"
            csv_path = out_dir / f"{stem}.csv"

            if args.resume and log_path.is_file() and csv_path.is_file():
                print(
                    f"[JUPITER_SWEEP_RESUME] x={x} "
                    f"repeat={repeat}/{args.repeats} action=SKIP "
                    f"csv={csv_path}",
                    flush=True,
                )
                csv_paths.append(csv_path)
                continue

            env = base_env.copy()
            env["LLAMA_JUPITER_SEQUENCE_PLAN"] = plan_text

            cmd = [
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
                "-lv", "3",
                "--no-conversation",
                "--single-turn",
                "--simple-io",
                "-p", prompt_text,
            ]
            cmd.extend(args.extra_arg)

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
                args.target_tokens,
                args.min_chunk,
            )
            if actual_chunks is None:
                tail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-60:]
                print(
                    "[JUPITER_SWEEP_DIAG] no complete plan could be "
                    f"reconstructed; log={log_path}",
                    file=sys.stderr,
                )
                for line in tail:
                    print(line, file=sys.stderr)
                raise RuntimeError(
                    f"runtime did not expose a complete Jupiter sequence "
                    f"plan for x={x}"
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

    profile_path = out_dir / "q-profile-all.csv"
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
            "--target-length", str(args.target_tokens),
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
