#!/usr/bin/env python3
"""Calibrate Jupiter's minimum efficient prefill chunk on PC GPU + Phone RPC.

Each sample runs an ordinary full prompt, but places the candidate length
first in the explicit sequence plan. Only the first formal (x, prefix=0)
chunk is sampled. Later chunks are large to avoid an expensive x=8 sweep.

Stage service times are synchronized for calibration. They serve as a
throughput-based proxy for accelerator utilization; NOT a GPU occupancy
counter and NOT formal asynchronous prefill latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from min_chunk_profile import partition_feasible, select_min_chunk
from run_q_sweep import (
    infer_runtime_prompt_tokens,
    make_sweep_plan,
    reconstruct_plan_from_log,
    stream_process,
)

STAGE_RE = re.compile(
    r"\[JUPITER_STAGE_PROFILE\].*?backend=(\S+).*?"
    r"input_ms=([0-9.]+) compute_ms=([0-9.]+) wall_ms=([0-9.]+)"
)
BEGIN_RE = re.compile(
    r"\[JUPITER_UBATCH\] event=BEGIN .*?"
    r"prefix_tokens=(\d+) tokens=(\d+)"
)
QUEUED_RE = re.compile(r"\[JUPITER_UBATCH\] event=QUEUED")
SAMPLE_FIELDS = (
    "chunk_tokens", "repeat", "pc_compute_ms", "phone_compute_ms",
    "pc_wall_ms", "phone_wall_ms", "plan", "status", "reason", "log",
)
SUMMARY_FIELDS = (
    "chunk_tokens", "samples", "pc_compute_ms", "phone_compute_ms",
    "pc_tokens_per_s", "phone_tokens_per_s", "pc_relative",
    "phone_relative", "dp_feasible", "selected",
)


def parse_sizes(value: str) -> list[int]:
    try:
        sizes = sorted({int(x.strip()) for x in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("candidate sizes must be integers") from exc
    if not sizes or sizes[0] <= 0:
        raise argparse.ArgumentTypeError("candidate sizes must be positive")
    return sizes


def load_pc_layers(explicit: int | None, model_plan: str | None) -> int:
    if explicit is not None:
        value = explicit
    else:
        if not model_plan:
            raise ValueError("provide --pc-layers or --model-plan")
        plan = Path(model_plan)
        if not plan.is_file():
            raise ValueError(f"model plan not found: {plan}; use --pc-layers N")
        found: str | None = None
        for raw in plan.read_text(encoding="utf-8-sig").splitlines():
            line = raw.split("#", 1)[0].strip()
            if line.startswith("model.pc_layers="):
                found = line.split("=", 1)[1].strip()
        if found is None:
            raise ValueError(f"model.pc_layers missing from {plan}")
        value = int(found)
    if value <= 0:
        raise ValueError("PC layer count must be positive")
    return value


def make_first_chunk_plan(
    total: int, first: int, minimum: int = 8, maximum: int = 128
) -> list[int]:
    """Create a complete partition with an exact first chunk x and no tiny tail."""
    if total <= 0 or first < minimum or first > maximum or first > total:
        raise ValueError("candidate is outside the valid sequence range")
    remainder = total - first
    if remainder == 0:
        return [first]
    if remainder < minimum:
        raise ValueError("insufficient tokens for a valid remainder")
    chunks = [first] + make_sweep_plan(remainder, maximum, minimum, maximum)
    if sum(chunks) != total or max(chunks) > maximum or min(chunks) < minimum:
        raise AssertionError("invalid calibration partition")
    return chunks


def first_chunk_service_times(text: str, tokens: int) -> dict[str, float]:
    """Sum PC accelerator and Phone RPC stage services for (x, prefix=0).

    CPU-only bookkeeping is excluded from PC GPU utilization proxy.
    A matching pair must finish its UBATCH (QUEUED) event; warmups with
    smaller token counts cannot supply the measurement.
    """
    target: tuple[int, int] | None = None
    totals: dict[str, list[float]] = {}
    found: dict[str, int] = {}

    for line in text.splitlines():
        begin = BEGIN_RE.search(line)
        if begin:
            target = (int(begin.group(1)), int(begin.group(2)))
            totals = {"PC": [0.0, 0.0], "PHONE": [0.0, 0.0]}
            found = {"PC": 0, "PHONE": 0}
            continue

        if target != (0, tokens):
            continue

        stage = STAGE_RE.search(line)
        if stage:
            backend = stage.group(1)
            if backend.startswith("RPC"):
                owner = "PHONE"
            elif backend.startswith("CPU"):
                continue
            else:
                owner = "PC"
            compute_ms = float(stage.group(3))
            wall_ms = float(stage.group(4))
            if not (math.isfinite(compute_ms) and compute_ms > 0 and
                    math.isfinite(wall_ms) and wall_ms > 0):
                raise ValueError("invalid calibrated stage duration")
            totals[owner][0] += compute_ms
            totals[owner][1] += wall_ms
            found[owner] += 1
            continue

        if QUEUED_RE.search(line):
            if found["PC"] and found["PHONE"]:
                return {
                    "pc_compute_ms": totals["PC"][0],
                    "phone_compute_ms": totals["PHONE"][0],
                    "pc_wall_ms": totals["PC"][1],
                    "phone_wall_ms": totals["PHONE"][1],
                }
            raise ValueError(
                f"missing calibrated PC GPU/Phone RPC stages for x={tokens}: {found}"
            )

    raise ValueError(f"no complete first prefill chunk (x={tokens}, prefix=0)")


def make_cli(args: argparse.Namespace, prompt: str) -> list[str]:
    return [
        str(Path(args.exe).resolve()),
        "-m", str(Path(args.model).resolve()),
        "--rpc", args.rpc, "--split-mode", "layer", "-ngl", "all",
        "-ub", str(args.max_chunk), "-b", str(args.batch),
        "-c", str(args.context), "-n", "1",
        "-t", str(args.threads), "--flash-attn", "on", "-lv", "3",
        "--no-conversation", "--single-turn", "--simple-io", "-p", prompt,
    ]


def calibration_env(pc_layers: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "LLAMA_JUPITER_PREFILL": "1",
        "LLAMA_JUPITER_PC_LAYERS": str(pc_layers),
        "LLAMA_JUPITER_WINDOW": "1",
        "LLAMA_JUPITER_TRACE": "1",
        "LLAMA_JUPITER_Q_SWEEP": "1",
        "LLAMA_JUPITER_PROFILE_STAGE": "1",
        "LLAMA_JUPITER_LOGITS_TOP10": "0",
        "LLAMA_JUPITER_ASYNC_C2P": "0",
        "LLAMA_JUPITER_TIMELINE": "0",
    })
    env.pop("LLAMA_JUPITER_PLAN_FILE", None)
    env.pop("LLAMA_JUPITER_SEQUENCE_PLAN", None)
    return env


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find Jupiter minimum efficient Chunk b using PC/Phone throughput saturation"
    )
    for option in ("exe", "model", "rpc", "prompt-file"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--pc-layers", type=int, default=None)
    parser.add_argument("--model-plan", default="jupiter-model.plan")
    parser.add_argument(
        "--candidates", type=parse_sizes,
        default=parse_sizes("8,16,24,32,48,64,96,128"),
    )
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="relative throughput plateau threshold (default 0.90)")
    parser.add_argument("--target-tokens", type=int, default=None,
                        help="optional expected prefill tokens; fail if tokenizer differs")
    parser.add_argument("--max-chunk", type=int, default=128)
    parser.add_argument("--min-tail", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--out-dir", default="jupiter-min-chunk")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pc_layers = load_pc_layers(args.pc_layers, args.model_plan)
    if (args.min_tail < 1 or args.max_chunk < args.min_tail or
            any(x < args.min_tail or x > args.max_chunk for x in args.candidates)):
        raise ValueError("candidate sizes must be within [min-tail, max-chunk]")
    if args.repeats < 1 or args.context < args.batch or args.batch < args.max_chunk:
        raise ValueError("require repeats >= 1 and context >= batch >= max-chunk")
    if not 0 < args.threshold <= 1 or not math.isfinite(args.threshold):
        raise ValueError("threshold must be within (0,1]")

    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    prompt_path = Path(args.prompt_file).resolve()
    for path in (exe, model, prompt_path):
        if not path.is_file():
            raise ValueError(f"missing input file: {path}")
    prompt = prompt_path.read_text(encoding="utf-8-sig").rstrip("\r\n")
    if not prompt:
        raise ValueError("prompt file is empty")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Use the exact same tokenizer/runtime settings for token detection and
    # stage measurements. A full plan only applies if its sum matches the
    # actual llama_context prefill batch size.
    env = calibration_env(pc_layers)
    command = make_cli(args, prompt)
    probe_log = out / "token-probe.log"
    probe_env = env.copy()
    probe_env["LLAMA_JUPITER_PROFILE_STAGE"] = "0"
    print(f"[JUPITER_B_PROBE] log={probe_log}", flush=True)
    code = stream_process(command, probe_env, probe_log)
    if code != 0:
        raise RuntimeError(f"token probe exited {code}; log={probe_log}")
    total, source = infer_runtime_prompt_tokens(probe_log)
    if args.target_tokens is not None and total != args.target_tokens:
        raise ValueError(f"runtime token count {total} != --target-tokens {args.target_tokens}")
    if total > args.batch or total > args.context:
        raise ValueError("actual prompt exceeds --batch or --context")

    signature = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "exe": str(exe), "exe_stat": [exe.stat().st_size, exe.stat().st_mtime_ns],
        "model": str(model), "model_stat": [model.stat().st_size, model.stat().st_mtime_ns],
        "rpc": args.rpc, "pc_layers": pc_layers, "tokens": total,
        "candidates": args.candidates, "min_tail": args.min_tail,
        "max_chunk": args.max_chunk, "batch": args.batch,
        "context": args.context, "threads": args.threads, "repeats": args.repeats,
    }
    config_path = out / "configuration.json"
    if args.resume and config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise ValueError("resume configuration differs; use a new --out-dir")
    config_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")

    sample_path = out / "chunk-samples.csv"
    old: dict[tuple[int, int], dict] = {}
    if args.resume and sample_path.is_file():
        with sample_path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "OK":
                    old[(int(row["chunk_tokens"]), int(row["repeat"]))] = row
    samples: list[dict] = []

    print(f"[JUPITER_B_CONFIG] pc_layers={pc_layers} tokens={total} "
          f"source={source} threshold={args.threshold:.3f} "
          f"candidates={','.join(map(str,args.candidates))} repeats={args.repeats}",
          flush=True)
    for x in args.candidates:
        try:
            plan = make_first_chunk_plan(total, x, args.min_tail, args.max_chunk)
            plan_text = ",".join(map(str, plan))
        except ValueError as exc:
            print(f"[JUPITER_B_SKIP] x={x} reason={exc}", flush=True)
            continue

        for repeat in range(1, args.repeats + 1):
            previous = old.get((x, repeat))
            if previous is not None:
                samples.append(previous)
                print(f"[JUPITER_B_RESUME] x={x} repeat={repeat} status=SKIP", flush=True)
                continue

            log_path = out / f"b-x{x}-r{repeat}.log"
            row: dict = {
                "chunk_tokens": x, "repeat": repeat,
                "pc_compute_ms": "", "phone_compute_ms": "",
                "pc_wall_ms": "", "phone_wall_ms": "",
                "plan": plan_text, "status": "FAILED",
                "reason": "", "log": str(log_path),
            }
            print(f"[JUPITER_B_RUN] x={x} repeat={repeat} plan={plan_text}",
                  flush=True)
            try:
                run_env = env.copy()
                run_env["LLAMA_JUPITER_SEQUENCE_PLAN"] = plan_text
                code = stream_process(command, run_env, log_path)
                if code != 0:
                    raise RuntimeError(f"llama-cli exited {code}")
                actual, origin = reconstruct_plan_from_log(
                    log_path, total, args.min_tail
                )
                if actual != plan:
                    raise ValueError(
                        f"sequence plan not applied: expected={plan_text} "
                        f"observed={actual} source={origin}"
                    )
                text = log_path.read_text(encoding="utf-8", errors="replace")
                row.update(first_chunk_service_times(text, x))
                row["status"] = "OK"
                print(f"[JUPITER_B_SAMPLE] x={x} pc_compute_ms="
                      f"{row['pc_compute_ms']:.3f} phone_compute_ms="
                      f"{row['phone_compute_ms']:.3f} status=OK", flush=True)
            except (RuntimeError, ValueError, OSError) as exc:
                row["reason"] = str(exc)
                print(f"[JUPITER_B_REJECT] x={x} reason={exc}", flush=True)
            samples.append(row)
            write_csv(sample_path, SAMPLE_FIELDS, samples)

    if not samples:
        raise ValueError("no feasible calibration candidate")
    # Re-write with resumed samples too; preserve all attempted measurements.
    write_csv(sample_path, SAMPLE_FIELDS, samples)
    selected, measured, status = select_min_chunk(
        samples, args.threshold, min_samples=args.repeats,
        total_tokens=total, max_chunk=args.max_chunk
    )

    report_rows = []
    for item in measured:
        report_rows.append({
            "chunk_tokens": item.tokens, "samples": item.samples,
            "pc_compute_ms": f"{item.pc_ms:.3f}",
            "phone_compute_ms": f"{item.phone_ms:.3f}",
            "pc_tokens_per_s": f"{item.pc_tokens_per_s:.3f}",
            "phone_tokens_per_s": f"{item.phone_tokens_per_s:.3f}",
            "pc_relative": f"{item.pc_relative:.4f}",
            "phone_relative": f"{item.phone_relative:.4f}",
            "dp_feasible": int(partition_feasible(total, item.tokens, args.max_chunk)),
            "selected": 1 if selected.tokens == item.tokens else 0,
        })
        print(f"[JUPITER_B_CANDIDATE] x={item.tokens} "
              f"pc_ratio={item.pc_relative:.3f} phone_ratio={item.phone_relative:.3f} "
              f"pc_tok_s={item.pc_tokens_per_s:.2f} "
              f"phone_tok_s={item.phone_tokens_per_s:.2f}", flush=True)

    write_csv(out / "chunk-summary.csv", SUMMARY_FIELDS, report_rows)
    result = {
        "min_chunk": selected.tokens,
        "status": status,
        "method": "measured_first_chunk_throughput_saturation_proxy",
        "metric": "synchronized_stage_compute_ms",
        "prefix_tokens": 0,
        "threshold": args.threshold,
        "pc_layers": pc_layers,
        "prompt_tokens": total,
        "max_chunk": args.max_chunk,
        "dp_feasible": partition_feasible(total, selected.tokens, args.max_chunk),
        "candidates": report_rows,
        "note": "Not actual GPU/Phone accelerator utilization; validate against normal Prefill A/B.",
    }
    (out / "min-chunk.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[JUPITER_B_SELECTED] b={selected.tokens} status={status} "
          f"pc_relative={selected.pc_relative:.3f} "
          f"phone_relative={selected.phone_relative:.3f} "
          f"report={out / 'min-chunk.json'}", flush=True)
    if status != "BOTH_STAGES_SATURATED":
        print(f"[JUPITER_B_WARNING] b={selected.tokens} status={status} "
              "meaning=FEASIBLE_BELOW_TARGET_THROUGHPUT; "
              "the observed throughput threshold is not fully met",
              flush=True)
    print(f"[JUPITER_B_NEXT] use --min-chunk {selected.tokens} "
          "with jupiter_planner.py sequence and the newly calibrated q(x,y)",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[JUPITER_B_STOP] interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"[JUPITER_B_ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
