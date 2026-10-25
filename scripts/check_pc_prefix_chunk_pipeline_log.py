#!/usr/bin/env python3
"""Validate chunk-major PC prefix -> L34 C2P scheduling.

Usage:
    python scripts/check_pc_prefix_chunk_pipeline_log.py pc.log

This checker validates control/dataflow markers only. It does not claim
numeric parity or end-to-end logits correctness.
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

    plans = list(re.finditer(
        r"\[PC_PREFIX_CHUNK_PLAN\] enabled=(\d+) layers=\[0,34\) "
        r"tensor_layer=34 tokens=(\d+) chunks=(\d+) "
        r"order=CHUNK_MAJOR status=(\S+)",
        text,
    ))
    if not plans:
        print("FAIL: missing PC_PREFIX_CHUNK_PLAN")
        return 1
    plan = plans[-1]
    if plan.group(1) != "1" or plan.group(4) != "READY":
        print(
            f"FALLBACK: PC prefix chunking inactive, "
            f"enabled={plan.group(1)} status={plan.group(4)}"
        )
        return 2

    n_chunks = int(plan.group(3))
    if n_chunks < 2:
        print(f"FAIL: expected multi-chunk prefill, got {n_chunks}")
        return 1

    built = [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        for m in re.finditer(
            r"\[PC_PREFIX_CHUNK\] chunk=(\d+) begin=(\d+) "
            r"tokens=(\d+) layers=\[0,34\] c2p_ready=1 status=BUILT",
            text,
        )
    ]
    if len(built) < n_chunks:
        print(f"FAIL: built prefix chunks={built}; expected {n_chunks}")
        return 1
    built = built[-n_chunks:]
    if [c for c, _, _ in built] != list(range(n_chunks)):
        print(f"FAIL: prefix build order={built}")
        return 1
    for i in range(1, len(built)):
        prev_begin, prev_tokens = built[i - 1][1], built[i - 1][2]
        if built[i][1] != prev_begin + prev_tokens:
            print(f"FAIL: non-contiguous prefix chunks={built}")
            return 1

    # Restrict runtime checks to the last active Region pipeline block.
    pipe_plans = list(re.finditer(
        r"\[PHONE_REGION_PIPE_PLAN\] active=(\d+).*?"
        r"phone_main_local=(\d+).*?reason=(\S+)",
        text,
    ))
    active = [m for m in pipe_plans if m.group(1) == "1"]
    if not active:
        print("FAIL: no active PHONE_REGION_PIPE_PLAN")
        return 1
    pipe = active[-1]
    if pipe.group(2) != "1":
        print("FAIL: phone_main_local is not active")
        return 1
    next_plan = next(
        (m for m in pipe_plans if m.start() > pipe.start()),
        None,
    )
    block = text[
        pipe.end():next_plan.start() if next_plan else len(text)
    ]

    # Every chunk must execute at least one PC_PREFIX stage before its C2P norm.
    for chunk in range(n_chunks):
        prefix = re.search(
            rf"\[PHONE_REGION_PIPE_STAGE\].*?chunk={chunk} .*?"
            rf"stage=PC_PREFIX event=BEGIN",
            block,
        )
        norm = re.search(
            rf"\[PHONE_REGION_PIPE_STAGE\].*?chunk={chunk} .*?"
            rf"stage=C2P_NORM event=BEGIN",
            block,
        )
        prep = re.search(
            rf"\[PHONE_MAIN_PC_PREP_ENQUEUE\] "
            rf"layer=34 chunk={chunk} .*?priority=LOW .*?status=QUEUED",
            block,
        )
        if prefix is None or norm is None or prep is None:
            print(
                f"FAIL: missing runtime marker for chunk {chunk}: "
                f"prefix={prefix is not None} norm={norm is not None} "
                f"prep={prep is not None}"
            )
            return 1
        if not (prefix.start() < norm.start() < prep.start()):
            print(
                f"FAIL: chunk {chunk} order is not "
                f"PC_PREFIX -> C2P_NORM -> PC_PREP"
            )
            return 1

    residuals = [
        (int(m.group(1)), m.group(2))
        for m in re.finditer(
            r"\[REGION_C2P_RESIDUAL_STAGE\] "
            r"layer=34 chunk=(\d+) .*?mode=(\S+) status=READY",
            block,
        )
    ]
    expected = [(c, "CHUNK_LOCAL") for c in range(n_chunks)]
    if residuals != expected:
        print(
            f"FAIL: residual staging={residuals}; expected={expected}"
        )
        return 1

    ready = [
        int(m.group(1))
        for m in re.finditer(
            r"\[PHONE_C2P_READY_SEND\] layer=34 chunk=(\d+) .*?"
            r"priority=C2P_FFN status=QUEUED",
            block,
        )
    ]
    if ready != list(range(n_chunks)):
        print(f"FAIL: C2P READY chunks={ready}")
        return 1

    partial = [
        int(m.group(1))
        for m in re.finditer(
            r"\[PHONE_C2P_PARTIAL_ENQUEUE\] layer=34 chunk=(\d+) .*?"
            r"priority=LOW .*?status=QUEUED",
            block,
        )
    ]
    if partial != list(range(n_chunks)):
        print(f"FAIL: C2P PARTIAL chunks={partial}")
        return 1

    for bad in (
        "PHONE_MAIN_LOCAL_REJECT",
        "REGION_C2P_OVERLAP_REJECT",
        "PHONE_REGION_CONTINUATION_REJECT",
        "Compute error",
    ):
        if bad in block:
            print(f"FAIL: found {bad}")
            return 1

    print(
        f"PASS: {n_chunks} chunks execute PC prefix L0-L33 + "
        f"L34 C2P in chunk-major order"
    )
    print("PASS: every L34 residual is snapshotted as CHUNK_LOCAL")
    print("PASS: every chunk publishes C2P READY and queues PARTIAL")
    print("NOTE: verify Phone MAIN logs and numeric parity separately")
    return 0


if __name__ == "__main__":
    sys.exit(main())
