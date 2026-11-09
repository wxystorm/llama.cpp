#!/usr/bin/env python3
"""Inspect recorded GPU Region timing without rerunning cooperative inference.

Use the last measurement_start boundary. The "lane" interval includes
ggml_backend_sched_compute_split() + ggml_backend_synchronize(); it is NOT
a device-event measurement of kernel-only execution. Similarly the residual
between compute_range, lane and prepare contains gates, CPU splits, and
bookkeeping rather than exclusively a lock wait.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


DEFAULT_PROMPTS = ("prompt_512_05", "prompt_512_10", "prompt_1024_09")
GROUP_FIELDS = (
    "prompt", "region", "bank", "layers", "chunks", "tokens",
    "range_ms", "lane_ms", "prepare_ms", "range_outside_lane_ms",
    "remaining_after_prepare_ms", "copy_call_ms",
    "target_wait_ms", "source_sync_ms", "output_sync_ms",
    "first_lane_ms", "rest_lane_median_ms",
    "max_range_ms", "max_lane_ms",
)


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def span(row, start_key, end_key):
    start, end = number(row.get(start_key)), number(row.get(end_key))
    return end - start if start is not None and end is not None and end > start else None


def field(row, name):
    value = number(row.get(name))
    return value if value is not None else 0.0


def load_measured(folder: Path):
    path = folder / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    events = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    events.append(item)
            except ValueError:
                continue
    markers = [i for i, row in enumerate(events)
               if row.get("type") == "measurement_start"]
    if not markers:
        raise ValueError(f"no measurement_start boundary: {path}")
    return events[markers[-1] + 1:]


def summarize(folder: Path, prompt: str):
    events = load_measured(folder)
    traces = [
        row for row in events
        if row.get("type") == "region_ring_chunk_trace"
        and row.get("device") == "GPU"
        and int(row.get("tokens") or 0) > 1
    ]
    if not traces:
        raise ValueError(f"no GPU prefill Chunk traces: {folder}")
    by_region = {}
    for row in traces:
        region = int(row["region"])
        by_region.setdefault(region, []).append(row)
    groups = []
    for region, items in sorted(by_region.items()):
        items.sort(key=lambda x: (int(x.get("ubatch") or 0),
                                  int(x.get("chunk") or 0)))
        ranges = [
            field(item, "compute_range_ms") or
            (span(item, "compute_range_begin_ms", "compute_range_end_ms") or 0)
            for item in items
        ]
        lanes = [
            span(item, "lane_compute_begin_ms", "lane_compute_end_ms")
            for item in items
        ]
        missing_lane = any(value is None for value in lanes)
        lane_ms = sum(value for value in lanes if value is not None)
        range_ms = sum(ranges)
        prepare_ms = sum(field(x, "split_prepare_ms") for x in items)
        residual = range_ms - lane_ms if not missing_lane else None
        # Do not relabel this as GPU gate time; G0 includes embedding/CPU splits.
        other = residual - prepare_ms if residual is not None else None
        rest = [value for value in lanes[1:] if value is not None]
        groups.append({
            "prompt": prompt,
            "region": region,
            "bank": "A" if region % 4 == 0 else "B",
            "layers": f"{items[0].get('layer_begin', '?')}:"
                      f"{items[0].get('layer_end', '?')}",
            "chunks": len(items),
            "tokens": sum(int(x.get("tokens") or 0) for x in items),
            "range_ms": range_ms,
            "lane_ms": None if missing_lane else lane_ms,
            "prepare_ms": prepare_ms,
            "range_outside_lane_ms": residual,
            "remaining_after_prepare_ms": other,
            "copy_call_ms": sum(field(x, "split_copy_call_ms") for x in items),
            "target_wait_ms": sum(field(x, "split_target_wait_ms") for x in items),
            "source_sync_ms": sum(field(x, "split_source_sync_ms") for x in items),
            "output_sync_ms": sum(field(x, "output_sync_ms") for x in items),
            "first_lane_ms": lanes[0],
            "rest_lane_median_ms": median(rest) if rest else None,
            "max_range_ms": max(ranges),
            "max_lane_ms": max((x for x in lanes if x is not None), default=None),
        })
    bank = [
        row for row in events
        if row.get("type") == "region_ring_bank_trace" and int(row.get("region") or -1) in (4, 6)
    ]
    result_path = folder / "result.json"
    result = json.loads(result_path.read_text("utf-8")) if result_path.exists() else {}
    return groups, bank, result


def print_table(rows, columns):
    widths = {}
    for name, header in columns:
        vals = [header]
        for row in rows:
            value = row.get(name)
            vals.append("-" if value is None else
                        f"{value:.1f}" if isinstance(value, float) else str(value))
        widths[name] = max(len(value) for value in vals)
    print("  ".join(header.ljust(widths[name]) for name, header in columns))
    for row in rows:
        parts = []
        for name, _ in columns:
            value = row.get(name)
            rendered = "-" if value is None else (
                f"{value:.1f}" if isinstance(value, float) else str(value))
            parts.append(rendered.rjust(widths[name]))
        print("  ".join(parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="ShareGPT40 output folder containing per-Prompt folders")
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    parser.add_argument("--csv", type=Path, help="optional path to write per-Region CSV")
    args = parser.parse_args(argv)
    all_rows = []
    errors = 0
    for name in args.prompts:
        stem = Path(name).stem
        folder = args.root / stem
        try:
            rows, banks, result = summarize(folder, stem)
        except (OSError, ValueError, KeyError) as exc:
            print(f"[ERROR] {stem}: {exc}")
            errors += 1
            continue
        all_rows.extend(rows)
        print(f"\n===== {stem}: {result.get('actual_prefill_tokens', '?')} tokens,"
              f" {result.get('prefill_tps', '?')} tok/s =====")
        print_table(rows, [
            ("region", "R"), ("bank", "Bank"), ("layers", "Layers"),
            ("chunks", "N"), ("range_ms", "Range"),
            ("lane_ms", "Lane"), ("prepare_ms", "Prepare"),
            ("range_outside_lane_ms", "NotLane"),
            ("remaining_after_prepare_ms", "Residual"),
            ("copy_call_ms", "Copy"),
            ("first_lane_ms", "FirstLane"),
            ("rest_lane_median_ms", "RestLaneMed"),
        ])
        print("Total GPU CUDA gate wait (all regions):",
              result.get("gpu_cuda_gate_wait_ms", "?"), "ms; "
              "CPU gate wait:", result.get("gpu_cpu_gate_wait_ms", "?"), "ms")
        if banks:
            print("Bank Refill timing:")
            print_table(banks, [
                ("region", "Region"), ("source_region", "From"),
                ("queue_wait_ms", "Queue"),
                ("total_ms", "Refill"),
                ("enqueue_ms", "Enqueue"),
                ("begin_ms", "Begin"),
                ("done_ms", "Done"),
            ])
        else:
            print("No Bank trace records after measurement_start.")
    print("\nInterpretation: Range=compute_range_ms; Lane includes CUDA compute"
          " and synchronize; NotLane=Range-Lane; Residual=NotLane-Prepare."
          " Residual is NOT exclusively Gate wait. Copy is part of Prepare.")
    if args.csv and all_rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=GROUP_FIELDS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"CSV: {args.csv}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
