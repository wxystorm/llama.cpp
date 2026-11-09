#!/usr/bin/env python3
"""Compare two recorded Region Ring runs, including per-Chunk GPU execution.

Inputs may be exact Prompt folders (containing result.json) or timestamped
sweep folders containing exactly one Prompt result. Reads files only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

import inspect_gpu_region_timing as diag

SUMMARY = (
    ("actual_prefill_tokens", "Prefill tokens", False),
    ("context", "Context size", False),
    ("prefill_tps", "Prefill tok/s", False),
    ("prefill_ms", "Prefill wall ms", True),
    ("ring_wall_ms", "Ring wall ms", False),
    ("gpu_compute_range_ms", "GPU compute range ms", True),
    ("gpu_bank_wait_ms", "GPU bank wait ms", True),
    ("gpu_bank_h2d_ms", "GPU Bank H2D ms", True),
    ("gpu_bank_prepare_ms", "GPU Bank prepare ms", True),
    ("phone_input_push_wall_ms", "Phone PUSH wall ms", True),
    ("phone_compute_range_ms", "Phone compute range ms", True),
    ("phone_ready_wait_ms", "Phone ready wait ms", True),
    ("pc_final_head_ms", "PC final head ms", True),
    ("phone_final_head_ms", "Phone final head ms", True),
)
ENV_PREFIXES = ("LLAMA_REGION_RING_", "LLAMA_HYBRID_REGION_RING_")


def locate_run(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "result.json").is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"run path not found: {path}")
    matches = [p.parent for p in path.glob("*/result.json")
               if (p.parent / "metrics.jsonl").is_file()]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one Prompt result in {path}; found {len(matches)}. "
            "Pass the exact path to its prompt_*/ directory.")
    return matches[0]


def load(path: Path):
    folder = locate_run(path)
    # The legacy run_prompt_sweep.ps1 does not enable per-Chunk tracing by
    # default. Its result.json still contains useful Ring and Bank aggregates.
    try:
        gpu, banks, result = diag.summarize(folder, folder.name)
        has_chunk_traces = True
    except ValueError as exc:
        if "no GPU prefill Chunk traces" not in str(exc):
            raise
        gpu, banks = [], []
        result = json.loads((folder / "result.json").read_text("utf-8"))
        has_chunk_traces = False
    config = {}
    cfg = folder / "config.json"
    if cfg.is_file():
        config = json.loads(cfg.read_text("utf-8"))
    try:
        events = diag.load_measured(folder)
    except ValueError as exc:
        if "no measurement_start boundary" not in str(exc):
            raise
        events = []
    trace = [row for row in events if row.get("type") == "region_ring_chunk_trace"
             and row.get("device") == "GPU" and int(row.get("tokens") or 0) > 1]
    chunks = {}
    for row in trace:
        chunks.setdefault(int(row["region"]), []).append(row)
    return {"folder": folder, "result": result, "config": config,
            "gpu": {row["region"]: row for row in gpu},
            "banks": {int(row["region"]): row for row in banks},
            "chunks": chunks, "has_chunk_traces": has_chunk_traces}


def fmt(value, precision=2):
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)


def num(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def ratio(a, b):
    """Return slow/fast, not an absolute time difference."""
    a, b = num(a), num(b)
    return b / a if a is not None and b is not None and a > 0 else None


def print_rows(rows, headers):
    widths = [max(len(str(row[i])) for row in [headers, *rows])
              for i in range(len(headers))]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))


def full_chunk_median(run, region):
    """Compare matching max-sized tokens/chunk, excluding first cold Chunk.

    This keeps a 117-Token tail from distorting the median for XG=128.
    """
    rows = run["chunks"].get(region, [])
    if not rows:
        return None, None, 0
    max_tokens = max(int(row["tokens"]) for row in rows)
    eligible = [
        diag.span(row, "lane_compute_begin_ms", "lane_compute_end_ms")
        for row in rows
        if int(row["tokens"]) == max_tokens and int(row.get("chunk") or 0) > 0
    ]
    eligible = [value for value in eligible if value is not None]
    return (median(eligible) if eligible else None,
            max_tokens, len(eligible))


def compare(fast, slow):
    a, b = fast["result"], slow["result"]
    an, bn = num(a.get("actual_prefill_tokens")), num(b.get("actual_prefill_tokens"))
    print("\n=== RUNS ===")
    print(f"FAST: {fast['folder']}")
    print(f"SLOW: {slow['folder']}")
    print(f"Model: {a.get('model', '?')} | {b.get('model', '?')}")
    same_sha = a.get("prompt_sha256") == b.get("prompt_sha256")
    print(f"Same Prompt SHA256: {same_sha} "
          "(different prompts make direct performance comparisons less controlled)")
    print("Layout:", a.get("runtime_layout", "?"), "|",
          b.get("runtime_layout", "?"))
    print("XG/XP:", f"{a.get('xg')}/{a.get('xp')}", "|",
          f"{b.get('xg')}/{b.get('xp')}")
    print("Batch/UBatch:", f"{a.get('batch')}/{a.get('ubatch')}", "|",
          f"{b.get('batch')}/{b.get('ubatch')}")

    print("\n=== OVERALL ===")
    rows = []
    for key, label, scale in SUMMARY:
        av, bv = num(a.get(key)), num(b.get(key))
        ap = av / an if scale and av is not None and an else None
        bp = bv / bn if scale and bv is not None and bn else None
        rows.append([
            label, fmt(av), fmt(bv), fmt(ratio(av, bv), 3),
            fmt(ap, 3), fmt(bp, 3), fmt(ratio(ap, bp), 3),
        ])
    print_rows(rows, ["Metric", "Fast", "Slow", "Raw x",
                      "Fast ms/tok", "Slow ms/tok", "Per-token x"])
    print("Note: counters overlap; GPU bank wait, Phone ready wait and "
          "compute cannot be summed to reconstruct wall time.")

    print("\n=== GPU REGION (actual CUDA lane incl. synchronization) ===")
    if not fast["has_chunk_traces"] or not slow["has_chunk_traces"]:
        missing = []
        if not fast["has_chunk_traces"]:
            missing.append("FAST")
        if not slow["has_chunk_traces"]:
            missing.append("SLOW")
        print(f"No GPU per-Chunk trace for: {', '.join(missing)}.")
        print("Legacy sweep may not enable task traces by default. "
              "For a repeat, set $env:LLAMA_REGION_RING_TASK_TRACE='1' "
              "before running run_prompt_sweep.ps1.")
    regions = sorted(set(fast["gpu"]) | set(slow["gpu"]))
    rows = []
    for r in regions:
        ga, gb = fast["gpu"].get(r, {}), slow["gpu"].get(r, {})
        med_a, tok_a, count_a = full_chunk_median(fast, r)
        med_b, tok_b, count_b = full_chunk_median(slow, r)
        # Normalize to ms per token if effective full Chunk lengths differ.
        per_tok_a = med_a / tok_a if med_a is not None and tok_a else None
        per_tok_b = med_b / tok_b if med_b is not None and tok_b else None
        rows.append([
            str(r),
            ga.get("bank", gb.get("bank", "?")),
            f"{ga.get('chunks', '-')}/{gb.get('chunks', '-')}",
            f"{fmt(med_a,1)}/{fmt(med_b,1)}",
            f"{tok_a or '-'}/{tok_b or '-'}",
            f"{count_a}/{count_b}",
            fmt(ratio(per_tok_a, per_tok_b), 3),
            f"{fmt(ga.get('lane_ms'),1)}/{fmt(gb.get('lane_ms'),1)}",
        ])
    if rows:
        print_rows(rows, ["R", "Bank", "Chunks A/B", "Full-Chunk lane A/B ms",
                          "Chunk tokens A/B", "Samples A/B", "ms/token B/A",
                          "Total lane A/B ms"])
    print("A/B here means FAST/SLOW, not GPU Bank A/B. "
          "Only full-width, non-first Chunk samples are compared.")

    print("\n=== BANK REFILL ===")
    rows = []
    for r in sorted(set(fast["banks"]) | set(slow["banks"])):
        ga, gb = fast["banks"].get(r, {}), slow["banks"].get(r, {})
        rows.append([str(r), fmt(ga.get("total_ms")),
                     fmt(gb.get("total_ms")),
                     fmt(ratio(ga.get("total_ms"), gb.get("total_ms")), 3),
                     fmt(ga.get("queue_wait_ms")),
                     fmt(gb.get("queue_wait_ms"))])
    if rows:
        print_rows(rows, ["Refill R", "Fast ms", "Slow ms", "Ratio B/A",
                          "Fast queue", "Slow queue"])
    else:
        print("No per-Refill traces; use aggregate GPU Bank counters above.")

    print("\n=== RECORDED CONFIG DIFFERENCES ===")
    changed = []
    for k in ("context", "xc", "xg", "xp", "batch", "ubatch",
              "threads", "no_mmap", "gpu_weight_streaming",
              "prompt_sha256", "commit"):
        av = a.get(k)
        bv = b.get(k)
        if av != bv:
            changed.append([k, str(av), str(bv)])
    ea = fast["config"].get("ring_environment", {})
    eb = slow["config"].get("ring_environment", {})
    for key in sorted(set(ea) | set(eb)):
        if not key.startswith(ENV_PREFIXES):
            continue
        if ea.get(key) != eb.get(key):
            changed.append([key, str(ea.get(key, "(unrecorded)")),
                            str(eb.get(key, "(unrecorded)"))])
    if changed:
        print_rows(changed, ["Setting", "Fast", "Slow"])
    else:
        print("No differing fields found in recorded configuration.")
    print("An unrecorded environment variable is NOT proof it was disabled.")
    if not same_sha:
        print("\nWARNING: Different Prompt SHA256 values. Treat the results as "
              "different-workload diagnostics, not a same-input regression test.")
    if an != bn:
        print("WARNING: Token counts differ. Compare GPU full-Chunk time and "
              "per-token columns, not the raw total durations.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", type=Path, required=True,
                    help="fast run timestamp folder or specific Prompt folder")
    ap.add_argument("--slow", type=Path, required=True,
                    help="slow run timestamp folder or specific Prompt folder")
    args = ap.parse_args(argv)
    try:
        fast = load(args.fast)
        slow = load(args.slow)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        ap.error(str(exc))
    compare(fast, slow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
