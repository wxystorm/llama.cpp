#!/usr/bin/env python3
"""Offline analysis of a Jupiter Phone-owned sync/async A/B run.

Reads EXISTING logs; does not rerun inference. A Phone RPC can overlap the
next PC ubatch even if it ends before the next Phone enqueue. The PC ubatch
interval includes graph work and scheduling, so this is not a CUDA kernel
timeline. Compares Prepare timings to flag cold/warm confounding.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

UBATCH = re.compile(
    r"\[JUPITER_UBATCH_TIMELINE\]\s+event=(BEGIN|QUEUED)"
    r"\s+chunk=(\d+).*?\bt_us=(\d+)"
)
PHONE = re.compile(
    r"\[JUPITER_PHONE_OWNED_PREFILL\]\s+phase=CHUNK_DONE"
    r"\s+mode=ASYNC\b.*?\bchunk=(\d+).*?"
    r"\bt_begin_us=(\d+)\s+t_done_us=(\d+)"
)
ENQUEUE = re.compile(
    r"\[JUPITER_PHONE_OWNED_PREFILL\]\s+phase=ASYNC_ENQUEUE"
    r"\s+chunk=(\d+).*?\bt_us=(\d+)"
)
PREPARE = re.compile(
    r"\[JUPITER_PHONE_OWNED_PREFILL\]\s+phase=PREPARE_DONE"
    r".*?\bload_ms=([\d.]+)"
    r"\s+build_ms=([\d.]+)"
    r"\s+buffer_ms=([\d.]+)"
    r"\s+precompile_ms=([\d.]+)"
    r"\s+phone_wall_ms=([\d.]+)"
    r"\s+pc_wall_ms=([\d.]+)"
)


def prepare_cost(log: str) -> dict | None:
    matches = PREPARE.findall(log)
    if not matches:
        return None
    names = ("load_ms", "build_ms", "buffer_ms",
             "precompile_ms", "phone_wall_ms", "pc_wall_ms")
    info = dict(zip(names, map(float, matches[-1])))
    info["zero_explicit_prepare_phases"] = all(
        info[k] == 0.0
        for k in ("load_ms", "build_ms", "buffer_ms", "precompile_ms")
    )
    return info


def analyze(sync_log: str, async_log: str) -> dict:
    starts: dict[int, int] = {}
    ends: dict[int, int] = {}
    for event, chunk, us in UBATCH.findall(async_log):
        target = starts if event == "BEGIN" else ends
        target[int(chunk)] = int(us)
    phone = {
        int(chunk): (int(start), int(end))
        for chunk, start, end in PHONE.findall(async_log)
    }
    enqueued = {
        int(chunk): int(us)
        for chunk, us in ENQUEUE.findall(async_log)
    }
    boundaries = []
    for next_chunk in sorted(starts):
        if next_chunk == 0:
            continue
        previous = phone.get(next_chunk - 1)
        pc_begin = starts.get(next_chunk)
        pc_end = ends.get(next_chunk)
        if previous is None or pc_end is None or pc_begin is None:
            boundaries.append({
                "phone_chunk": next_chunk - 1,
                "pc_chunk": next_chunk,
                "status": "MISSING_TIMELINE_EVENTS",
            })
            continue
        phone_start, phone_end = previous
        if phone_end < phone_start or pc_end < pc_begin:
            raise ValueError("non-monotonic PC or Phone timeline")
        overlap_us = max(
            0, min(phone_end, pc_end) - max(phone_start, pc_begin)
        )
        next_enqueue = enqueued.get(next_chunk)
        boundaries.append({
            "phone_chunk": next_chunk - 1,
            "pc_chunk": next_chunk,
            "status": "MEASURED",
            "pc_ubatch_span_ms": round((pc_end - pc_begin) / 1000, 3),
            "phone_rpc_span_ms": round((phone_end - phone_start) / 1000, 3),
            "pc_ubatch_phone_rpc_intersection_ms": round(overlap_us / 1000, 3),
            "next_enqueue_within_previous_phone_rpc": (
                phone_start < next_enqueue < phone_end
                if next_enqueue is not None else None
            ),
        })
    measured = [row for row in boundaries if row["status"] == "MEASURED"]
    sync_prepare = prepare_cost(sync_log)
    async_prepare = prepare_cost(async_log)
    return {
        "synchronous_prepare": sync_prepare,
        "asynchronous_prepare": async_prepare,
        "same_prepare_cost_class": (
            sync_prepare["zero_explicit_prepare_phases"] ==
            async_prepare["zero_explicit_prepare_phases"]
            if sync_prepare and async_prepare else None
        ),
        "pc_ubatch_vs_phone_rpc_positive_windows": sum(
            row["pc_ubatch_phone_rpc_intersection_ms"] > 0 for row in measured
        ),
        "pc_ubatch_vs_phone_rpc_overlap_ms": round(sum(
            row["pc_ubatch_phone_rpc_intersection_ms"] for row in measured
        ), 3),
        "measured_boundaries": len(measured),
        "boundaries": boundaries,
        "interpretation": (
            "PC ubatch vs Phone RPC host-side overlap; not a CUDA/HTP "
            "device-compute overlap proof or a controlled speedup trial."
        ),
    }


def self_test() -> None:
    prep = (
        "[JUPITER_PHONE_OWNED_PREFILL] phase=PREPARE_DONE run=1 graphs=16 "
        "load_ms=0 build_ms=0 buffer_ms=0 precompile_ms=0 "
        "phone_wall_ms=1 pc_wall_ms=2 status=PASS\n"
    )
    text = prep + "\n".join([
        "[JUPITER_UBATCH_TIMELINE] event=BEGIN chunk=0 slot=0 prefix=0 tokens=88 t_us=100",
        "[JUPITER_PHONE_OWNED_PREFILL] phase=ASYNC_ENQUEUE chunk=0 tokens=88 t_us=120 status=QUEUED",
        "[JUPITER_UBATCH_TIMELINE] event=QUEUED chunk=0 slot=0 prefix=0 tokens=88 t_us=130",
        "[JUPITER_UBATCH_TIMELINE] event=BEGIN chunk=1 slot=1 prefix=88 tokens=87 t_us=150",
        "[JUPITER_PHONE_OWNED_PREFILL] phase=CHUNK_DONE mode=ASYNC run=1 chunk=0 tokens=88 "
        "phone_compute_ms=0 phone_upload_ms=0 phone_copy_ms=0 phone_wall_ms=0 "
        "total_bridge_ms=0 t_begin_us=140 t_done_us=200 status=PASS",
        "[JUPITER_PHONE_OWNED_PREFILL] phase=ASYNC_ENQUEUE chunk=1 tokens=87 t_us=250 status=QUEUED",
        "[JUPITER_UBATCH_TIMELINE] event=QUEUED chunk=1 slot=1 prefix=88 tokens=87 t_us=270",
    ])
    outcome = analyze(prep, text)
    row = outcome["boundaries"][0]
    assert row["pc_ubatch_phone_rpc_intersection_ms"] == 0.05
    assert row["next_enqueue_within_previous_phone_rpc"] is False
    assert outcome["pc_ubatch_vs_phone_rpc_positive_windows"] == 1
    assert outcome["same_prepare_cost_class"] is True
    print("[PHONE_OVERLAP_INSPECT] SELF_TEST=PASS")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sync-log", type=Path)
    p.add_argument("--async-log", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.sync_log or not args.async_log:
        p.error("--sync-log and --async-log are required")
    result = analyze(
        args.sync_log.read_text(encoding="utf-8", errors="replace"),
        args.async_log.read_text(encoding="utf-8", errors="replace"),
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
