#!/usr/bin/env python3
"""Extract real transfer-only HYBRID_PROFILE probes into a separate CSV.

The existing C++ Hybrid Profile performs real tensor copies (with an RPC
fence for PC->Phone), independent of Transformer Stage compute. This tool
only exports those measurements. It does not fabricate unmeasured paths
such as Phone NPU->Adreno Host ownership or separately benchmark TCP.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import re
from statistics import median

SAMPLE = re.compile(
    r"\[HYBRID_PROFILE\]\s+"
    r"(gpu_weight_h2d|gpu_to_pc|pc_to_phone|phone_to_pc)\s+"
    r"bytes=(\d+)\s+ms=([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s+confirmed_wall_ms=([0-9]+(?:\.[0-9]+)?))?"
)
LABELS = ("gpu_weight_h2d", "gpu_to_pc", "pc_to_phone", "phone_to_pc")
ROWS = ("path", "bytes", "sample_count", "median_ms",
        "wall_ms", "effective_mibps", "origin", "timing_scope")


def extract(roots: list[Path]) -> list[dict]:
    collected = defaultdict(list)
    seen = set()
    for root in roots:
        files = [root] if root.is_file() else root.rglob("engine.stderr.log")
        for file in files:
            if file in seen:
                continue
            seen.add(file)
            for line in file.read_text("utf-8", errors="replace").splitlines():
                match = SAMPLE.search(line)
                if not match:
                    continue
                label, byte_count, service_ms, wall_ms = match.groups()
                count, ms = int(byte_count), float(service_ms)
                if count < 65536 or ms <= 0.0:
                    continue
                wall = float(wall_ms) if wall_ms else ms
                if wall <= 0:
                    continue
                collected[(label, count)].append((ms, wall))
    rows = []
    for (label, size), values in sorted(collected.items()):
        service = median(t[0] for t in values)
        wall = median(t[1] for t in values)
        rows.append({
            "path": label, "bytes": size, "sample_count": len(values),
            "median_ms": round(service, 6), "wall_ms": round(wall, 6),
            "effective_mibps": round(size / 1048576 / (service / 1000), 6),
            "origin": "REAL_HYBRID_PROFILE_TRANSFER_PROBE",
            "timing_scope": "RPC_FENCED_OR_BACKEND_COPY_AS_IN_CPP",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", type=Path, required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    rows = extract(args.run)
    missing = set(LABELS) - {r["path"] for r in rows}
    if missing:
        ap.error("COMMUNICATION_PROFILE_MISSING_REAL_SAMPLES:" +
                 ",".join(sorted(missing)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=ROWS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[COMMUNICATION_PROFILE] status=PASS paths={len(LABELS)} "
          f"points={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
