#!/usr/bin/env python3
"""Fig. 4(b): fixed 32B placement, three fixed chunk widths across prompt lengths.

Uses the same resident llama-server execution harness as the 32B Full Cycle
50-prompt main study, with native fixed-chunk request hook and GGUF --no-mmap.
Three selected ShareGPT50_short prompts per repetition; no cached prefixes.
Each case has independent resumable server logs and request-level evidence.
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
RUNNER = HERE / "probe_htp_llama_server.py"
LAYOUT = "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7"
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
    "prefill_tokens", "context", "prefill_ms", "prefill_tps", "cache_n",
    "layout", "plan_xc", "plan_xg", "plan_xp", "runtime_xc",
    "runtime_xg", "runtime_xp", "planner_ms", "scheduler_mode",
    "prefetch_mode", "result_file",
)
SUMMARY_FIELDS = (
    "input_group", "prompt", "actual_tokens", "context", "chunk",
    "repeats_pass", "repeats_expected", "median_prefill_ms",
    "normalized_prefill", "best_chunk",
)



def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server-exe", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--prompt-dir", type=Path, required=True)
    ap.add_argument("--profile-source", type=Path, required=True,
                    help="main50 profile directory containing profile.csv/json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout", default=LAYOUT)
    ap.add_argument("--groups", nargs="+", type=int, choices=GROUPS, default=list(GROUPS))
    ap.add_argument("--chunks", nargs="+", type=int, choices=CHUNKS, default=list(CHUNKS))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--prompt-index", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--inter-prompt-delay-seconds", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not 1 <= args.repeats <= 30 or not 1 <= args.prompt_index <= 10:
        ap.error("--repeats must be 1..30 and --prompt-index must be 1..10")
    if not 1024 <= args.port <= 65535 or args.threads != 8:
        ap.error("main50 parity requires threads=8 and a valid server port")
    if (args.inter_prompt_delay_seconds != 5.0 or
            not math.isfinite(args.inter_prompt_delay_seconds)):
        ap.error("main50 parity requires 5-second inter-request cooldown")
    args.groups = sorted(set(args.groups))
    args.chunks = sorted(set(args.chunks))
    if args.groups != list(GROUPS):
        ap.error("Fig. 4(b) requires exactly groups 128 512 1024")
    parts = re.findall(r"([GCP])(\d+)", args.layout)
    if (not parts or ",".join(f"{k}{n}" for k, n in parts) != args.layout
            or sum(int(n) for _, n in parts) != 64
            or args.layout != LAYOUT):
        ap.error("main50 parity requires layout " + LAYOUT)
    for key in ("server_exe", "model", "prompt_dir", "profile_source", "out"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args



def prompt_name(group: int, index: int) -> str:
    return f"prompt_{group}_{index:02d}.txt"


def case_dir(out: Path, chunk: int, repeat: int) -> Path:
    return out / f"chunk_{chunk:03d}" / f"repeat_{repeat:02d}"


def runner_command(args: argparse.Namespace, chunk: int, repeat: int) -> list[str]:
    """A separate resident server for one fixed Chunk and one repetition."""
    command = [
        sys.executable, str(RUNNER),
        "--server-exe", str(args.server_exe),
        "--model", str(args.model), "--model-layers", "64",
        "--rpc", args.rpc, "--prompt-dir", str(args.prompt_dir),
        "--profile-source", str(args.profile_source),
        "--out", str(case_dir(args.out, chunk, repeat)),
        "--port", str(args.port), "--threads", str(args.threads),
        "--locked-layout", args.layout,
        "--all50", "--per-request-chunks",
        "--fixed-chunk", str(chunk), "--no-mmap",
        "--fig4b-prompts",
        prompt_name(1024, args.prompt_index),
        prompt_name(128, args.prompt_index),
        prompt_name(512, args.prompt_index),
        "--inter-prompt-delay-seconds", str(args.inter_prompt_delay_seconds),
    ]
    if (case_dir(args.out, chunk, repeat) / "server-run-config.json").exists():
        command.append("--resume")
    return command



def read_case(args: argparse.Namespace, chunk: int, repeat: int) -> list[dict]:
    """Audit measured server rows and runtime/config provenance."""
    folder = case_dir(args.out, chunk, repeat)
    path = folder / "requests.csv"
    entries = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            entries = {r.get("prompt", ""): r for r in csv.DictReader(stream)}
    config_path = folder / "server-run-config.json"
    # An untouched (Chunk, repeat) is scheduled work, not a failed run.
    if not config_path.exists() and not path.exists() and not folder.exists():
        return [
            {"input_group": group, "prompt": prompt_name(group, args.prompt_index),
             "chunk": chunk, "repeat": repeat, "status": "PENDING",
             "audit_error": "", **{key: "" for key in RAW_FIELDS
                                   if key not in ("input_group", "prompt",
                                                  "chunk", "repeat", "status",
                                                  "audit_error")}}
            for group in args.groups
        ]
    config_errors = []
    if not config_path.is_file():
        config_errors.append("missing server-run-config.json")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            setting = config.get("configuration") or {}
            command = setting.get("server_command") or []
            environment = setting.get("runtime_env") or {}
            required_env = {
                "LLAMA_HYBRID_REGION_RING_LOCKED_LAYOUT": args.layout,
                "LLAMA_REGION_RING_FIXED_CHUNK_ABLATION": str(chunk),
                "LLAMA_REGION_RING_GPU_BANK_FORCE_SMALL_BATCH": "1",
                "LLAMA_REGION_RING_GPU_BANK_COUNT": "2",
                "LLAMA_REGION_RING_GPU_BANK_PREFETCH": "1",
                "LLAMA_REGION_RING_GPU_BANK_SYNC_DEMAND": "0",
                "LLAMA_REGION_RING_DRAIN_FIRST_CYCLE": "1",
                "LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED": "1",
                "LLAMA_REGION_RING_PER_REQUEST_CHUNKS": "1",
                "LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS": "1",
            }
            for key, value in required_env.items():
                if environment.get(key) != value:
                    config_errors.append(f"{key} expected {value}")
            for option in ("--no-mmap", "--no-cache-prompt",
                           "--no-cont-batching", "--no-warmup",
                           "--no-context-shift"):
                if option not in command:
                    config_errors.append("missing server flag " + option)
            for option, value in (("-c", "2048"), ("-b", "2048"),
                                  ("-ub", "2048"), ("--parallel", "1"),
                                  ("--cache-ram", "0"),
                                  ("--ctx-checkpoints", "0"),
                                  ("--split-mode", "tensor"),
                                  ("--flash-attn", "on")):
                if option not in command or command[command.index(option) + 1] != value:
                    config_errors.append(f"{option} expected {value}")
            expected_prompts = [
                prompt_name(g, args.prompt_index) for g in (1024, 128, 512)]
            if setting.get("prompts") != expected_prompts:
                config_errors.append("selected prompts differ from fig4b")
        except (ValueError, OSError, TypeError, KeyError, IndexError) as exc:
            config_errors.append("invalid server config: " + str(exc))
    result = []
    for group in args.groups:
        name = prompt_name(group, args.prompt_index)
        native = entries.get(name, {})
        row = {key: "" for key in RAW_FIELDS}
        row.update({
            "input_group": group, "prompt": name, "chunk": chunk,
            "repeat": repeat, "status": native.get("status", "MISSING"),
            "prefill_tokens": native.get("prompt_n", ""),
            "context": 2048,
            "prefill_ms": native.get("prompt_ms", ""),
            "prefill_tps": native.get("prefill_tps", ""),
            "cache_n": native.get("cache_n", ""),
            "layout": native.get("runtime_layout", ""),
            "planner_ms": native.get("planner_ms", ""),
            "scheduler_mode": native.get("scheduler_mode", ""),
            "prefetch_mode": native.get("prefetch_mode", ""),
            "result_file": str(folder / "requests.jsonl"),
        })
        for key in ("plan_xc", "plan_xg", "plan_xp",
                    "runtime_xc", "runtime_xg", "runtime_xp"):
            row[key] = native.get(key, "")
        errors = list(config_errors)
        if row["status"] != "PASS":
            errors.append("native status=" + str(row["status"]))
        if row["layout"] != args.layout:
            errors.append("runtime placement differs from full Cycle")
        if str(row["cache_n"]) != "0":
            errors.append("prefix cache was not zero")
        if row["scheduler_mode"] != "DRAIN_FIRST":
            errors.append("Drain-First not verified")
        if row["prefetch_mode"] != "ASYNC_DOUBLE_BANK":
            errors.append("async double-bank not verified")
        if native.get("planner_mode") != f"FIXED_{chunk}_NO_REQUEST_SEARCH":
            errors.append("fixed-chunk native hook not verified")
        try:
            tokens = int(row["prefill_tokens"])
            prefill = float(row["prefill_ms"])
            if tokens <= 0 or not math.isfinite(prefill) or prefill <= 0:
                raise ValueError()
            fixed_width = min(chunk, tokens)
            for key in ("plan_xc", "plan_xg", "plan_xp"):
                if int(row[key]) != fixed_width:
                    errors.append(f"{key} != {fixed_width}")
            # Native server probe already validates the actual physical
            # tail-coalesced widths against the log, not against the requested
            # fixed width, since the two may differ.
            # The locked Full Cycle layout has G/P regions only (no CPU
            # compute stages). REGION_RING_SUM may legitimately report XC=0
            # for a non-existent C stage. XG and XP must remain positive.
            has_cpu_stage = any(part.startswith("C") for part in args.layout.split(","))
            xc_actual = int(row["runtime_xc"])
            if xc_actual < 0 or (has_cpu_stage and xc_actual == 0):
                errors.append("invalid runtime_xc for layer placement")
            for key in ("runtime_xg", "runtime_xp"):
                if int(row[key]) <= 0:
                    errors.append("missing runtime width " + key)
        except (ValueError, TypeError):
            errors.append("invalid tokens, prefill time or Chunk widths")
        if errors:
            row["audit_error"] = "; ".join(errors)
            row["status"] = ("AUDIT_FAIL" if row["status"] == "PASS"
                             else row["status"])
        result.append(row)
    return result



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
    # Reports may be emitted after only the first case. Unrun slots are
    # pending, not measurement failures; never flood diagnostics with them.
    errors = [r for r in raw if r["status"] not in ("PASS", "PENDING")]
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