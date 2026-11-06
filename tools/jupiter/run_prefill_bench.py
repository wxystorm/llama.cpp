#!/usr/bin/env python3
"""Repeatable, low-noise Jupiter prefill benchmark.

The calibration runner (run_q_sweep.py) deliberately synchronizes backends to
measure q(x,y). Do not use it to benchmark actual pipeline throughput. This
runner disables that synchronization, runs normal async inference and collects
the server-side "prompt eval time" from each independent llama-cli process.

The benchmark's Default mode deliberately processes the entire prompt as a
single ubatch (traditional PC/Phone layer split). DP and fixed64 continue to
use their own chunked sequence plans with --ubatch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

from run_q_sweep import make_sweep_plan


PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
)
PLAN_RE = re.compile(
    r"\[JUPITER_SEQUENCE_PLAN\].*?total=(\d+).*?chunks=([0-9,]+)"
)
MODES = ("default", "fixed64", "dp")
UBATCH_BEGIN_RE = re.compile(
    r"\[JUPITER_UBATCH\] event=BEGIN chunk=(\d+) "
    r"slot=\d+ prefix_tokens=(\d+) tokens=(\d+)"
)


def single_chunk_ubatch(target: int) -> int:
    """A power-of-two ubatch large enough for one full-prompt microbatch."""
    if target < 1:
        raise ValueError("target must be positive")
    return 1 << (target - 1).bit_length()


def extract_prefill_chunks(log_text: str, target: int) -> list[int] | None:
    """Recover actual prefill ubatches from TRACE; ignore warmup/decode calls."""
    current: list[int] = []
    prefix = 0
    matched = None
    for entry in UBATCH_BEGIN_RE.finditer(log_text):
        index, beginning, count = (int(x) for x in entry.groups())
        if index == 0 and beginning == 0:
            current = []
            prefix = 0
        if index != len(current) or beginning != prefix:
            current = []
            prefix = 0
            continue
        current.append(count)
        prefix += count
        if prefix == target:
            matched = current.copy()
            current = []
            prefix = 0
        elif prefix > target:
            current = []
            prefix = 0
    return matched


def read_plan(path: Path) -> dict[int, list[int]]:
    if not path.is_file():
        return {}
    sequence: dict[int, list[int]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line.startswith("sequence.") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            length = int(key.split(".", 1)[1])
            chunks = [int(x.strip()) for x in value.split(",")]
        except ValueError as exc:
            raise ValueError(f"invalid sequence plan entry: {line!r}") from exc
        if length < 1 or not chunks or any(x <= 0 for x in chunks):
            raise ValueError(f"invalid sequence plan entry: {line!r}")
        if sum(chunks) != length:
            raise ValueError(f"plan sum does not match sequence length: {line}")
        sequence[length] = chunks
    return sequence


def modes_arg(text: str) -> list[str]:
    modes = [part.strip() for part in text.split(",") if part.strip()]
    if not modes or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated distinct modes from {MODES}"
        )
    return modes


def parse_timing(path: Path) -> tuple[float, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(PROMPT_RE.finditer(text))
    if not matches:
        raise RuntimeError(
            "no prompt eval time in log; check CLI verbosity and runtime: "
            f"{path}"
        )
    ms = float(matches[-1].group(1))
    tokens = int(matches[-1].group(2))
    if ms <= 0 or tokens <= 0:
        raise RuntimeError(f"invalid prefill timing in {path}")
    return ms, tokens


def compatible_timing_tokens(actual: int, target: int, tolerance: int) -> bool:
    """Allow the known optional one-token CLI timing counter convention."""
    return actual > 0 and target > 0 and abs(actual - target) <= tolerance


def run_one(
    command: list[str],
    env: dict[str, str],
    log: Path,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as f:
        proc = subprocess.run(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-45:])
        raise RuntimeError(
            f"llama-cli exited {proc.returncode} (0x{proc.returncode & 0xffffffff:08X}); "
            f"inspect log: {log}\n"
            f"--- last 45 lines of llama-cli ---\n{tail}"
        )


def build_cli(
    args: argparse.Namespace,
    exe: Path,
    model: Path,
    prompt: str,
    verbosity: int,
    ubatch: int | None = None,
) -> list[str]:
    return [
        str(exe), "-m", str(model),
        "--rpc", args.rpc, "--split-mode", "layer",
        "-ngl", "all",
        "-ub", str(args.ubatch if ubatch is None else ubatch), "-b", str(args.batch),
        "-c", str(args.context), "-n", "1",
        "-t", str(args.threads),
        "--flash-attn", "on",
        "-lv", str(verbosity),
        "--no-conversation", "--single-turn", "--simple-io",
        "--no-show-timings",  # retain server prompt eval time, hide CLI duplicate
        "-p", prompt,
    ]


def env_for_mode(
    base_env: dict[str, str],
    mode: str,
    plan: Path,
    fixed_chunks: list[int],
) -> dict[str, str]:
    env = base_env.copy()
    env.pop("LLAMA_JUPITER_PLAN_FILE", None)
    env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    if mode == "dp":
        env["LLAMA_JUPITER_PLAN_FILE"] = str(plan)
    elif mode == "fixed64":
        env["LLAMA_JUPITER_SEQUENCE_PLAN"] = ",".join(map(str, fixed_chunks))
    return env


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Low-noise Jupiter prefill A/B benchmark")
    p.add_argument("--exe", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--rpc", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--plan", default="jupiter.plan")
    p.add_argument("--target-tokens", type=int)
    p.add_argument("--pc-layers", type=int, default=25)
    p.add_argument("--cpu-layers", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--timing-token-tolerance", type=int, choices=[0, 1], default=0,
        help="allow at most one token difference between runtime prefill "
             "batch size and CLI timing count (raw count is recorded separately)",
    )
    p.add_argument("--modes", type=modes_arg, default=list(MODES))
    p.add_argument("--out-dir", default="jupiter-bench-formal")
    p.add_argument("--ubatch", type=int, default=128,
                   help="ubatch for chunked DP/fixed64 runs (default: 128)")
    p.add_argument("--default-ubatch", type=int, default=None,
                   help="ubatch for unchunked Default; must be >= target tokens. "
                        "Automatically rounds up to the next power of two "
                        "(366 tokens -> 512) if omitted")
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--verbosity", type=int, default=3)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--async-c2p", action="store_true",
                   help="opt in to the Region Ring-style Jupiter RPC transport worker")
    p.add_argument("--local-graph-reuse", action="store_true",
                   help="opt in to the existing local graph can_reuse path; "
                        "synchronizes before input rebinding and can reduce overlap")
    p.add_argument("--graph-prebuild", action="store_true",
                   help="opt in to PC graph metadata prebuild + Phone GRAPH_REGISTER "
                        "(requires rebuilt Phone RPC server, legacy C2P only)")
    p.add_argument("--comm-timing", action="store_true",
                   help="opt in to per-chunk host/RPC timing (diagnostic runs only)")
    p.add_argument(
        "--skip-verify", action="store_true",
        help="skip the one-time -lv 4 sequence-plan verification before timing",
    )
    p.add_argument("--resume", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    prompt_path = Path(args.prompt_file).resolve()
    plan_path = Path(args.plan).resolve()
    out = Path(args.out_dir).resolve()
    for path in (exe, model, prompt_path):
        if not path.is_file():
            raise RuntimeError(f"input file not found: {path}")
    if (args.repeats < 1 or args.pc_layers < 1 or
            args.cpu_layers < 0 or args.cpu_layers >= args.pc_layers
            or args.window < 1):
        raise ValueError("repeats, pc-layers and window must be positive")
    if args.verbosity < 2 or args.verbosity > 4:
        raise ValueError("use --verbosity 2, 3 or 4; 3 is recommended")
    if args.ubatch <= 0 or args.ubatch > args.batch or args.batch > args.context:
        raise ValueError("require 0 < ubatch <= batch <= context")
    if args.graph_prebuild and (args.async_c2p or args.local_graph_reuse):
        raise ValueError("--graph-prebuild is independent of and currently "
                         "incompatible with --async-c2p / --local-graph-reuse")

    prompt = prompt_path.read_text(encoding="utf-8-sig").rstrip("\r\n")
    if not prompt:
        raise ValueError("prompt file is empty")
    plans = read_plan(plan_path)
    if args.target_tokens is not None:
        target = args.target_tokens
    elif len(plans) == 1:
        target = next(iter(plans))
    else:
        raise ValueError(
            "supply --target-tokens N if the plan has zero or multiple "
            "sequence.N entries"
        )
    if target <= 1 or target > args.batch or target > args.context:
        raise ValueError(f"invalid target prefill token count: {target}")
    default_ubatch = (args.default_ubatch if args.default_ubatch is not None
                      else single_chunk_ubatch(target))
    if "default" in args.modes and not target <= default_ubatch <= args.batch:
        raise ValueError(
            f"Default must run one whole prompt: require {target} <= "
            f"--default-ubatch <= --batch ({args.batch}); got {default_ubatch}"
        )
    mode_ubatch = {
        mode: default_ubatch if mode == "default" else args.ubatch
        for mode in args.modes
    }
    if "dp" in args.modes:
        cpu_lines = [
            line.strip() for line in plan_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip().startswith("model.cpu_layers=")
        ]
        plan_cpu = int(cpu_lines[-1].split("=", 1)[1]) if cpu_lines else 0
        if plan_cpu != args.cpu_layers:
            raise ValueError(
                f"DP plan CPU boundary={plan_cpu} differs from "
                f"--cpu-layers={args.cpu_layers}"
            )
        if target not in plans:
            raise ValueError(f"{plan_path} has no sequence.{target} entry")
        if max(plans[target]) > args.ubatch:
            raise ValueError(
                "DP plan contains chunk larger than --ubatch; "
                "increase --ubatch or regenerate the plan"
            )

    fixed = make_sweep_plan(target, 64, 32, args.ubatch)
    out.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env.update({
        "LLAMA_JUPITER_PREFILL": "1",
        "LLAMA_JUPITER_PC_LAYERS": str(args.pc_layers),
        "LLAMA_JUPITER_CPU_LAYERS": str(args.cpu_layers),
        "LLAMA_JUPITER_WINDOW": str(args.window),
        "LLAMA_JUPITER_PROFILE_STAGE": "0",
        "LLAMA_JUPITER_TRACE": "0",
        "LLAMA_JUPITER_LOGITS_TOP10": "0",
        "LLAMA_JUPITER_ASYNC_C2P": "1" if args.async_c2p else "0",
        "LLAMA_JUPITER_LOCAL_GRAPH_REUSE": "1" if args.local_graph_reuse else "0",
        "LLAMA_JUPITER_GRAPH_PREBUILD": "1" if args.graph_prebuild else "0",
        "LLAMA_GRAPH_REUSE_DISABLE": "0",
        "LLAMA_JUPITER_TIMELINE": "1" if args.comm_timing else "0",
    })
    base_env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    base_env.pop("LLAMA_JUPITER_PLAN_FILE", None)

    config = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "exe": str(exe),
        "exe_size": exe.stat().st_size,
        "exe_mtime_ns": exe.stat().st_mtime_ns,
        "model": str(model),
        "model_size": model.stat().st_size,
        "rpc": args.rpc,
        "target_tokens": target,
        "pc_layers": args.pc_layers,
        "cpu_layers": args.cpu_layers,
        "window": args.window,
        "async_c2p": args.async_c2p,
        "local_graph_reuse": args.local_graph_reuse,
        "graph_prebuild": args.graph_prebuild,
        "comm_timing": args.comm_timing,
        "profile_stage": 0,
        "trace": 0,
        "logits_top10": 0,
        "verbosity": args.verbosity,
        "context": args.context,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "default_ubatch": default_ubatch,
        "mode_ubatch": mode_ubatch,
        "default_chunks": [target] if "default" in args.modes else None,
        "modes": args.modes,
        "repeats": args.repeats,
        "plan": str(plan_path),
        "dp_chunks": plans.get(target),
        "fixed64_chunks": fixed,
    }
    # Keep strict default configs backwards-compatible with earlier runs;
    # an opt-in tolerance must be visible in the experiment identity.
    if args.timing_token_tolerance:
        config["timing_token_tolerance"] = args.timing_token_tolerance
    config_path = out / "configuration.json"
    if args.resume and config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        # Users can resume with a larger repeat count, but never silently mix
        # measurements from different prompts, binaries, plans or parameters.
        comparable = {key: value for key, value in config.items() if key != "repeats"}
        old_comparable = {
            key: value for key, value in previous.items() if key != "repeats"
        }
        if comparable != old_comparable:
            raise RuntimeError(
                f"--resume configuration mismatch in {config_path}; "
                "use a new --out-dir for a new experiment"
            )
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"[JUPITER_BENCH_CONFIG] tokens={target} modes={','.join(args.modes)} "
        f"repeats={args.repeats} lv={args.verbosity} "
        f"window={args.window} async_c2p={int(args.async_c2p)} "
        f"local_graph_reuse={int(args.local_graph_reuse)} "
        f"graph_prebuild={int(args.graph_prebuild)} "
        f"default_ubatch={default_ubatch} chunk_ubatch={args.ubatch} "
        f"comm_timing={int(args.comm_timing)} trace=0 profile_stage=0",
        flush=True,
    )

    if not args.skip_verify:
        # Only verification uses TRACE verbosity. It is not included in the
        # formal results and never enables per-stage synchronous profiling.
        for mode in args.modes:
            wanted = ([target] if mode == "default" else
                      plans[target] if mode == "dp" else fixed)
            check_log = out / f"verify-{mode}.log"
            verify_env = env_for_mode(
                base_env, mode, plan_path, fixed
            )
            verify_env["LLAMA_JUPITER_TRACE"] = "1"
            print(f"[JUPITER_BENCH_VERIFY] mode={mode}", flush=True)
            run_one(
                build_cli(args, exe, model, prompt, 4, mode_ubatch[mode]),
                verify_env, check_log,
            )
            text = check_log.read_text(encoding="utf-8", errors="replace")
            matched = [
                (int(m.group(1)), [int(x) for x in m.group(2).split(",")])
                for m in PLAN_RE.finditer(text)
            ]
            if mode != "default" and not any(
                length == target and chunks == wanted for length, chunks in matched
            ):
                raise RuntimeError(
                    f"{mode} did not report the expected Jupiter sequence plan; "
                    f"inspect {check_log}"
                )
            actual_chunks = extract_prefill_chunks(text, target)
            if actual_chunks != wanted:
                raise RuntimeError(
                    f"{mode}: actual prefill chunks {actual_chunks} != "
                    f"expected {wanted}; inspect {check_log}"
                )
            if args.async_c2p and (
                "[JUPITER_COMM] mode=ASYNC_C2P_NETWORK status=ENABLED" not in text
                or "[JUPITER_COMM] action=C2P_ENQUEUE" not in text
            ):
                raise RuntimeError(
                    f"{mode} did not execute the asynchronous C2P path; "
                    f"inspect {check_log}"
                )
            if args.graph_prebuild:
                prep = re.findall(
                    r"\[JUPITER_GRAPH_PREBUILD\] phase=PREPARE "
                    r"planned=(\d+) ready=(\d+).*status=READY", text,
                )
                hits = text.count("status=HIT")
                registered = "[JUPITER_GRAPH_PREBUILD] phase=PHONE_REGISTER" in text
                if not prep or not registered or hits < int(prep[-1][1]):
                    raise RuntimeError(
                        f"{mode}: graph prebuild did not cover all planned chunks; "
                        f"prepared={prep[-1] if prep else None}, "
                        f"hits={hits}, phone_registered={registered}; "
                        f"inspect {check_log}"
                    )
            verified_chunks = ",".join(map(str, wanted))
            print(
                f"[JUPITER_BENCH_VERIFY] mode={mode} status=PASS "
                f"chunks={verified_chunks} ubatch={mode_ubatch[mode]} "
                f"async_c2p={int(args.async_c2p)}",
                flush=True,
            )

    rows: list[dict[str, object]] = []
    # Rotate mode order so temperature/order drift is less correlated with
    # one policy. Each measured run is a fresh llama-cli process.
    for repeat in range(1, args.repeats + 1):
        offset = (repeat - 1) % len(args.modes)
        order = args.modes[offset:] + args.modes[:offset]
        for mode in order:
            log_path = out / f"{mode}-r{repeat}.log"
            if args.resume and log_path.exists():
                try:
                    ms, n_tokens = parse_timing(log_path)
                    if not compatible_timing_tokens(n_tokens, target, args.timing_token_tolerance):
                        raise ValueError("cached prompt length differs")
                    print(
                        f"[JUPITER_BENCH_RESUME] mode={mode} repeat={repeat} "
                        "action=SKIP",
                        flush=True,
                    )
                except (RuntimeError, ValueError):
                    ms = -1
            else:
                ms = -1

            if ms < 0:
                print(
                    f"[JUPITER_BENCH_RUN] mode={mode} repeat={repeat}/"
                    f"{args.repeats}",
                    flush=True,
                )
                env = env_for_mode(base_env, mode, plan_path, fixed)
                run_one(
                    build_cli(args, exe, model, prompt, args.verbosity, mode_ubatch[mode]),
                    env, log_path,
                )
                ms, n_tokens = parse_timing(log_path)

            if not compatible_timing_tokens(n_tokens, target, args.timing_token_tolerance):
                raise RuntimeError(
                    f"mode={mode} repeat={repeat}: CLI timing tokens "
                    f"{n_tokens}, actual prefill batch tokens {target}; log={log_path}"
                )
            if n_tokens != target:
                print(
                    f"[JUPITER_BENCH_TOKEN_CONVENTION] mode={mode} "
                    f"repeat={repeat} planned_prefill_tokens={target} "
                    f"cli_timing_tokens={n_tokens} difference={n_tokens - target}",
                    flush=True,
                )
            speed = target * 1000.0 / ms
            rows.append({
                "mode": mode, "repeat": repeat, "tokens": target,
                "ubatch": mode_ubatch[mode],
                "timing_tokens": n_tokens,
                "prefill_ms": round(ms, 3),
                "prefill_tok_s": round(speed, 3),
                "log": str(log_path),
            })
            print(
                f"[JUPITER_BENCH_RESULT] mode={mode} repeat={repeat} "
                f"prefill_ms={ms:.3f} tok_s={speed:.3f}",
                flush=True,
            )

    csv_path = out / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "repeat", "tokens", "ubatch", "timing_tokens",
                        "prefill_ms", "prefill_tok_s", "log"],
        )
        writer.writeheader()
        writer.writerows(rows)

    medians: dict[str, float] = {}
    for mode in args.modes:
        values = [float(r["prefill_ms"]) for r in rows if r["mode"] == mode]
        medians[mode] = statistics.median(values)

    summary = out / "summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        for mode in args.modes:
            ms = medians[mode]
            tok_s = target * 1000.0 / ms
            ratio = (
                medians["default"] / ms
                if "default" in medians else float("nan")
            )
            line = (
                f"[JUPITER_BENCH_SUMMARY] mode={mode} "
                f"median_ms={ms:.3f} median_tok_s={tok_s:.3f} "
                f"speedup_vs_default={ratio:.3f}x\n"
            )
            print(line, end="", flush=True)
            f.write(line)
    print(f"[JUPITER_BENCH_DONE] results={csv_path} summary={summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[JUPITER_BENCH_STOP] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[JUPITER_BENCH_ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
