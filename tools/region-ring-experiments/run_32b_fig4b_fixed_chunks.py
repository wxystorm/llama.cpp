#!/usr/bin/env python3
"""Fig. 4(b): fixed 32B placement, three fixed chunk widths across prompt lengths.

Reuses run_32b_all40.py (no Planner search), with the *same* two-bank,
async-prefetch, drain-first Cycle runtime for every measurement. Every repeat
has an independent resumable output folder and one unmeasured warmup.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNNER = HERE / "run_32b_all40.py"
LAYOUT = "G8,P6,G8,P6,G8,P6,G8,P6,G8"
GROUPS = (128, 512, 1024)
CHUNKS = (32, 64, 128)
ENV_POLICY = {
    "LLAMA_REGION_RING_GPU_BANK_COUNT": "2",
    "LLAMA_REGION_RING_GPU_BANK_PREFETCH": "1",
    "LLAMA_REGION_RING_DRAIN_FIRST_CYCLE": "1",
    "LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED": "1",
}
RAW_FIELDS = (
    "input_group", "prompt", "chunk", "repeat", "status", "audit_error",
    "prefill_tokens", "context", "prefill_ms", "prefill_tps", "layout",
    "runtime_xc", "runtime_xg", "runtime_xp", "gpu_bank_streaming",
    "gpu_bank_force_small_batch", "bank_refills", "bank_h2d_ms",
    "bank_gpu_wait_ms", "result_file",
)
SUMMARY_FIELDS = (
    "input_group", "prompt", "actual_tokens", "context", "chunk",
    "repeats_pass", "repeats_expected", "median_prefill_ms",
    "normalized_prefill", "best_chunk",
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--prompt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", default=LAYOUT,
                    help="single frozen 64-layer placement for every configuration")
    ap.add_argument("--groups", nargs="+", type=int, choices=GROUPS, default=list(GROUPS))
    ap.add_argument("--chunks", nargs="+", type=int, choices=CHUNKS, default=list(CHUNKS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt-index", type=int, default=1,
                    help="same prompt_N_XX.txt in each length group (default 01)")
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-context", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--context-padding", type=int, default=64)
    ap.add_argument("--inter-prompt-delay-seconds", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=0.0)
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--trace-tasks", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the 3 x 3 x repeats schedule without launching inference")
    args = ap.parse_args(argv)
    if not 1 <= args.repeats <= 30 or not 1 <= args.prompt_index <= 10:
        ap.error("--repeats must be 1..30 and --prompt-index must be 1..10")
    if args.ubatch < 1 or args.batch < args.ubatch:
        ap.error("require --batch >= --ubatch > 0")
    if (args.min_context < 1 or args.max_context < args.min_context
            or args.context_padding < 0 or args.timeout < 0
            or not math.isfinite(args.inter_prompt_delay_seconds)
            or args.inter_prompt_delay_seconds < 0):
        ap.error("invalid context/timeout/cooldown settings")
    args.groups = sorted(set(args.groups))
    args.chunks = sorted(set(args.chunks))
    parts = re.findall(r"([GCP])(\d+)", args.layout)
    if (not parts or ",".join(f"{k}{n}" for k, n in parts) != args.layout
            or sum(int(n) for _, n in parts) != 64
            or sum(k == "G" for k, _ in parts) < 2 or not any(k == "P" for k, _ in parts)):
        ap.error("--layout must be a valid, comma-separated 64-layer G/C/P placement with >=2 GPU regions and a Phone region")
    for key in ("exe", "model", "prompt_dir", "out"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def prompt_name(group: int, index: int) -> str:
    return f"prompt_{group}_{index:02d}.txt"


def case_dir(out: Path, chunk: int, repeat: int) -> Path:
    return out / f"chunk_{chunk:03d}" / f"repeat_{repeat:02d}"


def runner_command(args: argparse.Namespace, chunk: int, repeat: int) -> list[str]:
    """One invocation covers the same selected prompt in each input group."""
    cmd = [
        sys.executable, str(RUNNER),
        "--exe", str(args.exe), "--model", str(args.model),
        "--rpc", args.rpc, "--repo-root", str(ROOT),
        "--prompt-dir", str(args.prompt_dir),
        "--out", str(case_dir(args.out, chunk, repeat)),
        "--model-layers", "64", "--htp-only", "--layout", args.layout,
        "--streaming", "on", "--chunk", str(chunk),
        "--xc", str(chunk), "--xg", str(chunk), "--xp", str(chunk),
        "--batch", str(args.batch), "--ubatch", str(args.ubatch),
        "--threads", str(args.threads), "--n-predict", "1",
        "--min-context", str(args.min_context),
        "--max-context", str(args.max_context),
        "--context-padding", str(args.context_padding),
        "--inter-prompt-delay-seconds", str(args.inter_prompt_delay_seconds),
        "--timeout", str(args.timeout), "--log-verbosity", "4",
        "--max-retries", "0", "--stop-on-fail", "--force-small-batch-mover",
        "--warmup-prompt", prompt_name(args.groups[0], args.prompt_index),
    ]
    for group in args.groups:
        cmd += ["--only-group", str(group),
                "--only-prompt", prompt_name(group, args.prompt_index)]
    for key, value in ENV_POLICY.items():
        cmd += ["--env", f"{key}={value}"]
    if args.no_mmap:
        cmd.append("--no-mmap")
    if args.trace_tasks:
        cmd.append("--trace-tasks")
    return cmd


def read_case(args: argparse.Namespace, chunk: int, repeat: int) -> list[dict]:
    """Read native results and mark unverified execution as failed, not as data."""
    path = case_dir(args.out, chunk, repeat) / "summary.csv"
    actual = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            actual = {r.get("prompt", ""): r for r in csv.DictReader(stream)}
    records = []
    for group in args.groups:
        name = prompt_name(group, args.prompt_index)
        native = actual.get(name, {})
        row = {key: native.get(key, "") for key in RAW_FIELDS}
        row.update({"input_group": group, "prompt": name,
                    "chunk": chunk, "repeat": repeat,
                    "status": native.get("status", "MISSING"),
                    "audit_error": ""})
        errors = []
        if row["status"] != "PASS":
            errors.append(f"native status={row['status']}")
        if native.get("layout") != args.layout:
            errors.append("unexpected runtime layer placement")
        for field in ("runtime_xc", "runtime_xg", "runtime_xp"):
            if str(native.get(field, "")) != str(chunk):
                errors.append(f"{field} not equal to fixed chunk {chunk}")
        for field in ("prefill_tokens", "context"):
            try:
                if int(native.get(field, "")) <= 0:
                    raise ValueError()
            except (ValueError, TypeError):
                errors.append(f"missing or invalid {field}")
        try:
            value = float(native.get("prefill_ms", ""))
            if not math.isfinite(value) or value <= 0:
                raise ValueError()
        except (ValueError, TypeError):
            errors.append("missing or invalid prefill_ms")
        if str(native.get("gpu_bank_streaming", "")).lower() != "true":
            errors.append("async GPU bank mover was not active")
        if str(native.get("gpu_bank_force_small_batch", "")).lower() != "true":
            errors.append("force-small-batch mover was not acknowledged")
        if native.get("plan_mode") != "fixed":
            errors.append("dynamic planner unexpectedly enabled")
        result_path = native.get("result_file")
        if not result_path or not Path(result_path).is_file():
            errors.append("missing native result.json for environment audit")
        else:
            try:
                detail = json.loads(Path(result_path).read_text(encoding="utf-8"))
                env = detail.get("env_overrides", {})
                for key, expected in ENV_POLICY.items():
                    if str(env.get(key, "")) != expected:
                        errors.append(f"{key} not set to {expected}")
            except (ValueError, OSError, TypeError):
                errors.append("unreadable native result.json")
        if errors:
            row["audit_error"] = "; ".join(errors)
            row["status"] = "AUDIT_FAIL" if row["status"] == "PASS" else row["status"]
        records.append(row)
    return records


def write_csv(path: Path, fields, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def aggregate(args: argparse.Namespace) -> tuple[int, int, bool]:
    raw = [row for chunk in args.chunks for rep in range(1, args.repeats + 1)
           for row in read_case(args, chunk, rep)]
    write_csv(args.out / "fig4b_raw.csv", RAW_FIELDS, raw)
    errors = [r for r in raw if r["status"] != "PASS"]
    for row in errors[:12]:
        print(f"[FIG4B_INVALID] group={row['input_group']} chunk={row['chunk']} "
              f"repeat={row['repeat']} {row['audit_error']}", file=sys.stderr)
    if len(errors) > 12:
        print(f"[FIG4B_INVALID] plus {len(errors) - 12} other cases", file=sys.stderr)
    # The same prompt must be tokenized and given the same KV context for all
    # chunks/repeats; otherwise comparisons can reflect prompt-prefix drift.
    for group in args.groups:
        group_rows = [r for r in raw if r["input_group"] == group and r["status"] == "PASS"]
        for field in ("prefill_tokens", "context"):
            if len({str(r[field]) for r in group_rows}) > 1:
                print(f"[FIG4B_INVALID] group={group} inconsistent {field} across configs",
                      file=sys.stderr)
                errors.append({"input_group": group, "status": "INCONSISTENT"})
    summary = []
    for group in args.groups:
        for chunk in args.chunks:
            selected = [r for r in raw if r["input_group"] == group and
                        r["chunk"] == chunk and r["status"] == "PASS"]
            complete = len(selected) == args.repeats
            summary.append({
                "input_group": group,
                "prompt": prompt_name(group, args.prompt_index),
                "actual_tokens": selected[0]["prefill_tokens"] if selected else "",
                "context": selected[0]["context"] if selected else "",
                "chunk": chunk, "repeats_pass": len(selected),
                "repeats_expected": args.repeats,
                "median_prefill_ms": round(statistics.median(float(r["prefill_ms"])
                                       for r in selected), 4) if complete else "",
                "normalized_prefill": "", "best_chunk": "",
            })
    complete = not errors and all(r["median_prefill_ms"] != "" for r in summary)
    if complete:
        for group in args.groups:
            grp = [r for r in summary if r["input_group"] == group]
            best = min(grp, key=lambda r: r["median_prefill_ms"])
            baseline = best["median_prefill_ms"]
            for row in grp:
                row["normalized_prefill"] = round(row["median_prefill_ms"] / baseline, 6)
                row["best_chunk"] = best["chunk"]
    write_csv(args.out / "fig4b_summary.csv", SUMMARY_FIELDS, summary)
    plot = []
    if complete:
        for group in args.groups:
            records = [r for r in summary if r["input_group"] == group]
            plot.append({"input_group": group, "actual_tokens": records[0]["actual_tokens"],
                         **{f"chunk_{r['chunk']}": r["normalized_prefill"] for r in records}})
    plot_fields = ["input_group", "actual_tokens"] + [f"chunk_{c}" for c in args.chunks]
    write_csv(args.out / "fig4b_plot_normalized.csv", plot_fields, plot)
    print(f"[FIG4B_OUTPUT] raw={args.out / 'fig4b_raw.csv'} ")
    print(f"[FIG4B_OUTPUT] summary={args.out / 'fig4b_summary.csv'}")
    if complete:
        print(f"[FIG4B_READY] plot={args.out / 'fig4b_plot_normalized.csv'}")
    else:
        print("[FIG4B_INCOMPLETE] no normalized plot rows emitted; fix failures and resume",
              file=sys.stderr)
    return (len(raw) - len([r for r in raw if r["status"] != "PASS"]),
            len(raw), complete)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not RUNNER.is_file() and not args.dry_run:
        print(f"[FIG4B_ERROR] missing existing runner: {RUNNER}", file=sys.stderr)
        return 2
    expected = len(args.groups) * len(args.chunks) * args.repeats
    print(f"[FIG4B] placement={args.layout} chunks={args.chunks} "
          f"groups={args.groups} repeats={args.repeats} measurements={expected}", flush=True)
    # Rotate chunk execution order across repeats to reduce monotonic thermal
    # drift while keeping exact per-(chunk,repeat) output identities for resume.
    for rep in range(1, args.repeats + 1):
        chunks = args.chunks[(rep - 1) % len(args.chunks):] + args.chunks[:(rep - 1) % len(args.chunks)]
        for chunk in chunks:
            command = runner_command(args, chunk, rep)
            print(f"[FIG4B_CASE] chunk={chunk} repeat={rep} out={case_dir(args.out, chunk, rep)}",
                  flush=True)
            if args.dry_run:
                print(subprocess.list2cmdline(command), flush=True)
                continue
            code = subprocess.call(command, cwd=str(ROOT))
            case_rows = read_case(args, chunk, rep)
            valid = sum(r["status"] == "PASS" for r in case_rows)
            print(f"[FIG4B_PROGRESS] case_valid={valid}/{len(case_rows)} "
                  f"runner_exit={code}", flush=True)
            if code != 0 or valid != len(case_rows):
                aggregate(args)  # preserve a partial raw CSV for diagnosis/resume
                print("[FIG4B_STOP] case failed validation; inspect logs and rerun the same command",
                      file=sys.stderr)
                return code if code != 0 else 3
    if args.dry_run:
        return 0
    passed, total, complete = aggregate(args)
    return 0 if complete and passed == total else 3


if __name__ == "__main__":
    raise SystemExit(main())