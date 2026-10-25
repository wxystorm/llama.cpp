#!/usr/bin/env python3
"""Validate the opt-in Phone Region chunk-pipeline trace (no parity claim).

Usage: python scripts/check_phone_region_pipeline_log.py region-pair.log

PowerShell redirection may create UTF-16LE logs. The reader accepts both
UTF-8 and UTF-16 with or without BOM.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys


def read_log(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    if raw.count(b"\x00") > max(8, len(raw) // 10):
        return raw.decode("utf-16-le", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=pathlib.Path)
    args = parser.parse_args()
    text = read_log(args.log)

    plan_re = re.compile(
        r"\[PHONE_REGION_PIPE_PLAN\] active=(\d+) window=(\d+) "
        r"chunks=(\d+) stages=(\d+) meta_sgs=(\d+) "
        r"(?:scratch_pool=(\d+) )?reason=([^\s]+)"
    )
    plans = list(plan_re.finditer(text))
    if not plans:
        print("FAIL: missing PHONE_REGION_PIPE_PLAN; check the PC environment flag")
        return 1
    ready = [match for match in plans if match.group(1) == "1"]
    if not ready:
        last = plans[-1]
        print(f"FALLBACK: pipeline inactive, reason={last.group(7)}")
        return 2

    # The final active plan covers one Meta execution. Model warmup can
    # contain other (inactive) plans, so restrict checking to its own block.
    plan = ready[-1]
    remaining_plan = next((p for p in plans if p.start() > plan.start()), None)
    block = text[plan.end():remaining_plan.start() if remaining_plan else len(text)]
    n_chunks = int(plan.group(3))
    if int(plan.group(2)) != 2 or n_chunks < 2:
        print("FAIL: expected window=2 with at least two chunks")
        return 1

    begin_re = re.compile(
        r"\[PHONE_REGION_PIPE_STAGE\].*?chunk=(\d+).*?"
        r"stage=(C2P_NORM|C2P_FFN_REDUCE|PHONE_REGION) event=BEGIN"
    )
    observed = [(int(m.group(1)), m.group(2)) for m in begin_re.finditer(block)]
    expected = [
        (c, stage)
        for c in range(n_chunks)
        for stage in ("C2P_NORM", "C2P_FFN_REDUCE", "PHONE_REGION")
    ]
    if observed != expected:
        print(f"FAIL: stage order mismatch\nexpected={expected}\nobserved={observed}")
        return 1

    for pattern, event in [
        (r"\[REGION_C2P_PREFILL\] layer=34 chunk=(\d+).*?status=OK", "C2P"),
        (r"\[PHONE_REGION_GRAPH\] chunk=(\d+).*?status=DEVICE_LOCAL_READY", "Phone Region"),
        (r"\[PHONE_REGION_GRAPH_DONE\] chunk=(\d+).*?status=0", "Phone submit"),
        (r"\[REGION_P2C_PREFILL\] layer=46 chunk=(\d+).*?status=OK", "P2C"),
    ]:
        found = [int(m.group(1)) for m in re.finditer(pattern, block)]
        if found != list(range(n_chunks)):
            print(f"FAIL: {event} chunks={found}; expected 0..{n_chunks - 1}")
            return 1
    for bad in ("PHONE_REGION_GRAPH_REJECT", "PHONE_REGION_SCRATCH_REJECT", "Compute error"):
        if bad in block:
            print(f"FAIL: found {bad}")
            return 1

    scratch = list(re.finditer(
        r"\[PHONE_REGION_SCRATCH\] chunk=(\d+) slot=(\d+).*?status=BOUND",
        block,
    ))
    if len(scratch) != n_chunks:
        print(f"FAIL: scratch bindings={len(scratch)}; expected {n_chunks}")
        return 1
    slots = [(int(m.group(1)), int(m.group(2))) for m in scratch]
    expected_slots = [(c, c % 2) for c in range(n_chunks)]
    if slots != expected_slots:
        print(f"FAIL: scratch slot order={slots}; expected={expected_slots}")
        return 1

    outputs = [
        int(m.group(1))
        for m in re.finditer(
            r"\[PHONE_REGION_OUTPUT_SLOT\] chunk=(\d+).*?status=BOUND",
            block,
        )
    ]
    if outputs != list(range(n_chunks)):
        print(f"FAIL: output slots={outputs}; expected 0..{n_chunks - 1}")
        return 1

    print(f"PASS: {n_chunks} C2P -> Phone Region stages are interleaved "
          f"(window=2, {plan.group(4)} Meta schedule entries)")
    print("PASS: window-2 Phone scratch slots and persistent chunk outputs are bound")
    print("PASS: corresponding L34, Phone, and L46 stages returned successfully")
    print("NOTE: this does not verify GPU kernel duration, KV parity or logits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
