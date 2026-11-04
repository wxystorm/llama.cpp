#!/usr/bin/env python3
"""Validate a measured Phone serial-HTP run from the Phone RPC log.

This is *not* a logits parity test. It verifies that HTP, not OpenCL, executed
all requested whole-Layer Chunks and that HTP-owned KV was synchronized for
Decode. Use GPU-only vs mixed run outputs for independent numeric validation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys

EVENT = re.compile(r"\[PHONE_NPU_SERIAL\]\s+(.*)")
FIELD = re.compile(r"([A-Za-z_]+)=([^\s]+)")


def records(path: Path) -> list[dict[str, str]]:
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        found = EVENT.search(line)
        if found is not None:
            result.append(dict(FIELD.findall(found.group(1))))
    return result


def check(rows: list[dict[str, str]], regions: set[int],
          chunks: int, run: int | None = None) -> tuple[bool, str]:
    executions = [r for r in rows if r.get("event") == "EXECUTE"]
    if not executions:
        return False, "NO_HTP_EXECUTION"
    ids = sorted({int(r["run"]) for r in executions if "run" in r})
    if not ids:
        return False, "EXECUTION_RUN_MISSING"
    if run is None:
        # A warmup may produce a shorter NPU run before the measured prompt.
        run = max(ids, key=lambda x: sum(
            int(r.get("tokens", 0)) for r in executions
            if int(r.get("run", -1)) == x))
    chosen = [r for r in executions if int(r.get("run", -1)) == run]
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in chosen:
        try:
            region = int(row["region"])
        except (ValueError, KeyError):
            return False, "REGION_FIELD_INVALID"
        if region not in regions:
            return False, f"UNEXPECTED_HTP_REGION_{region}"
        grouped[region].append(row)
    if set(grouped) != regions:
        return False, f"REGIONS_MISMATCH expected={regions} got={set(grouped)}"
    lines = []
    for region in sorted(regions):
        group = grouped[region]
        try:
            observed_chunks = {int(r["chunk"]) for r in group}
            layers = {int(r["layer"]) for r in group}
            times = [float(r["compute_ms"]) for r in group]
        except (ValueError, KeyError):
            return False, f"INVALID_FIELDS_REGION_{region}"
        if len(group) != chunks or observed_chunks != set(range(chunks)):
            return False, f"CHUNKS_MISMATCH_REGION_{region}: {sorted(observed_chunks)}"
        if len(layers) != 1 or any(r.get("status") != "PASS" for r in group):
            return False, f"HTP_FAILED_OR_LAYER_CHANGED_REGION_{region}"
        if any(r.get("backend") != "HTP0" for r in group):
            return False, f"NOT_HTP_REGION_{region}"
        lines.append(
            f"region={region} layer={next(iter(layers))} "
            f"chunks={len(group)} HTP_compute_sum_ms={sum(times):.3f}")
    kv_sync = [r for r in rows
               if r.get("event") == "KV_SYNC" and
                  r.get("run") == str(run) and
                  r.get("status") == "PASS"]
    if not kv_sync:
        return False, f"KV_SYNC_NOT_CONFIRMED_RUN_{run}"
    return True, f"run={run} NPU_EXECUTION_PASS\n" + "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phone-log", type=Path, required=True)
    parser.add_argument("--regions", default="3,7")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--run", type=int, default=None)
    args = parser.parse_args()
    try:
        regions = {int(x) for x in args.regions.split(",")}
        if not regions or any(x < 0 or x >= 64 for x in regions):
            raise ValueError("Regions must be 0..63")
        if args.chunks <= 0:
            raise ValueError("Chunks must be positive")
        ok, message = check(records(args.phone_log), regions,
                            args.chunks, args.run)
    except (OSError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
