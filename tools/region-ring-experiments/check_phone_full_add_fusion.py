#!/usr/bin/env python3
"""Check structure of opt-in Phone Full ADD fusion against an unfused run.

This is not numerical parity. It detects a changed layout, changed Phone->PC
return size or missing expected reduction in standalone residual ADD graphs.
"""

import argparse
import json
import sys
from pathlib import Path


def load_run(path):
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.jsonl"
    if not path.is_file():
        raise ValueError(f"Missing metrics.jsonl: {path}")
    records = []
    for line_num, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_num}: {exc}") from exc
    markers = [
        index for index, record in enumerate(records)
        if record.get("type") == "measurement_start"
    ]
    measured = records[markers[-1] + 1:] if markers else records
    runs = [r for r in measured if r.get("type") == "region_ring"]
    if not runs:
        raise ValueError(f"No measured region_ring entries in {path}")
    if not all(r.get("phone_rpc_available") is True for r in runs):
        raise ValueError(f"Phone RPC counters unavailable in {path}")
    return runs


def collect(runs):
    signature = {
        (r.get("layout"), r.get("xc"), r.get("xg"), r.get("xp"))
        for r in runs
    }
    if len(signature) != 1:
        raise ValueError("Multiple different layouts in the same measurement")
    regions = {}
    for record in runs:
        for row in record.get("phone_rpc_regions", []):
            key = (row["region"], row["layer_begin"], row["layer_end"])
            counts = regions.setdefault(key, {})
            for name in ("chunks", "graph_multi_count", "graph_multi_nodes",
                         "graph_single_add_count", "graph_single_other_count",
                         "graph_compute_count", "graph_recompute_count",
                         "graph_snapshot_count", "get_tensor_count",
                         "get_tensor_bytes"):
                counts[name] = counts.get(name, 0) + int(row.get(name, 0))
    if not regions:
        raise ValueError("No Phone Region counters")
    for key, row in regions.items():
        counted = (row["graph_multi_count"] +
                   row["graph_single_add_count"] +
                   row["graph_single_other_count"])
        submitted = (row["graph_compute_count"] +
                     row["graph_recompute_count"] +
                     row["graph_snapshot_count"])
        if counted != submitted:
            raise ValueError(f"Graph classification incomplete for {key}")
    return signature.pop(), regions, sum(float(r.get("wall_ms", 0)) for r in runs)


def compare(baseline_runs, fused_runs):
    base_shape, base, base_ms = collect(baseline_runs)
    new_shape, fused, new_ms = collect(fused_runs)
    if base_shape != new_shape:
        raise ValueError(f"Layout/Chunk config differs: {base_shape} != {new_shape}")
    if base.keys() != fused.keys():
        raise ValueError("Phone Region identity differs between measurements")
    print("Region  Layers    Chunks  ADD before -> after  Main before -> after  GET bytes")
    print("-" * 87)
    saved = 0
    for key in sorted(base):
        b, n = base[key], fused[key]
        if (b["chunks"] != n["chunks"] or
                b["get_tensor_count"] != n["get_tensor_count"] or
                b["get_tensor_bytes"] != n["get_tensor_bytes"]):
            raise ValueError(f"Phone Chunk or return payload changed in region {key}")
        gain = b["graph_single_add_count"] - n["graph_single_add_count"]
        saved += gain
        print(f"{key[0]:>6}  {key[1]:>2}-{key[2]-1:<3}  {b['chunks']:>6}  "
              f"{b['graph_single_add_count']:>3} -> "
              f"{n['graph_single_add_count']:<3}          "
              f"{b['graph_multi_count']:>3} -> "
              f"{n['graph_multi_count']:<3}       "
              f"{n['get_tensor_bytes']}")
    print(f"Saved separate ADD graphs: {saved}")
    print(f"Region wall (baseline -> fused): {base_ms:.3f} -> {new_ms:.3f} ms")
    print("NOTE: Graph/return structure only. Verify numerical output parity separately.")
    if saved <= 0:
        raise ValueError(
            "No separate ADD graphs eliminated; inspect Phone Full fusion "
            "eligibility and GGML_META_PIPELINE_DEBUG messages")
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline metrics.jsonl or run folder")
    parser.add_argument("fused", type=Path, help="Fused metrics.jsonl or run folder")
    args = parser.parse_args(argv)
    try:
        compare(load_run(args.baseline), load_run(args.fused))
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
