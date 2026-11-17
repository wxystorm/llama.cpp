#!/usr/bin/env python3
"""Summarize phone-only HTP per-layer stage / execution profiling."""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

KV = re.compile(r"([a-zA-Z_][a-zA-Z_0-9]*)=([^\s]+)")

def fields(line):
    return {key: val for key, val in KV.findall(line)}

def number(d, key):
    try:
        return float(d.get(key, "0"))
    except ValueError:
        return 0.0

def median(rows, key):
    return statistics.median(r[key] for r in rows) if rows else float("nan")

def main():
    if len(sys.argv) != 4:
        sys.exit("usage: parse_motivation_2b.py LOG OUT_DIR LAYER")
    log, output, layer = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    waiting_weights = defaultdict(list)
    samples = []
    maps = []
    with log.open("r", errors="replace") as fh:
        for line in fh:
            if "[PHONE_MOTI_MAP]" in line:
                maps.append(fields(line))
            if "[PHONE_MOTI_WEIGHT]" in line:
                f = fields(line)
                if int(f.get("layer", "-1")) == layer:
                    waiting_weights[layer].append(f)
            if "[PHONE_MOTI_LAYER]" in line:
                f = fields(line)
                if int(f.get("layer", "-1")) != layer:
                    continue
                weights = waiting_weights[layer]
                row = {
                    "sample": len(samples) + 1,
                    "layer": layer,
                    "splits": int(f.get("splits", "0")),
                    "matmuls": int(f.get("matmuls", "0")),
                    "weight_mib": number(f, "weight_mib"),
                    "stage_ms": number(f, "stage_ms"),
                    "npu_wall_ms": number(f, "npu_wall_ms"),
                    "stage_read_mib": number(f, "stage_read_mib"),
                    "major_faults": int(f.get("major_faults", "0")),
                    "page_touch_ms": sum(number(x, "page_touch_ms") for x in weights),
                    "repack_wall_ms": sum(number(x, "repack_wall_ms") for x in weights),
                    "page_touch_read_mib": sum(number(x, "page_touch_read_mib") for x in weights),
                    "repack_read_mib": sum(number(x, "repack_read_mib") for x in weights),
                    "weight_tensors": len(weights),
                }
                row["stage_to_htp_ratio"] = row["stage_ms"] / row["npu_wall_ms"] if row["npu_wall_ms"] > 0 else float("nan")
                samples.append(row)
                waiting_weights[layer] = []

    if not samples:
        print("[PHONE_MOTI_PARSE] no per-layer samples; check accelerated_matmuls and selected layer")
        return 1
    with (output / "layer_samples.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)

    # A per-buffer map may happen during graph reservation, long before any
    # particular transformer layer executes. Do not add this to per-layer stage.
    with (output / "map_events.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["session", "size_mib", "pinned", "map_ms", "scope"])
        writer.writeheader()
        for m in maps:
            writer.writerow({key: m.get(key, "") for key in writer.fieldnames})

    report = [
        f"layer={layer} samples={len(samples)}",
        f"median_stage_ms={median(samples, 'stage_ms'):.3f}",
        f"median_htp_wall_ms={median(samples, 'npu_wall_ms'):.3f}",
        f"median_stage_to_htp_ratio={median(samples, 'stage_to_htp_ratio'):.2f}",
        f"median_page_touch_ms={median(samples, 'page_touch_ms'):.3f}",
        f"median_repack_wall_ms={median(samples, 'repack_wall_ms'):.3f}",
        f"median_stage_read_mib={median(samples, 'stage_read_mib'):.3f}",
        f"median_stage_major_faults={median(samples, 'major_faults'):.0f}",
        f"buffer_map_events={len(maps)}",
        "NOTE: stage includes page faults + CPU repack + backend copies and waits.",
        "NOTE: HTP wall includes submission / synchronization, not pure DSP kernel time.",
        "NOTE: mapping events are per buffer allocation, NOT necessarily per layer switch.",
    ]
    (output / "summary.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0

if __name__ == "__main__":
    sys.exit(main())
