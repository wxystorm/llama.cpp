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
        r"(?:scratch_pool=(\d+) )?"
        r"(?:continuation=(\d+) )?reason=([^\s]+)"
    )
    plans = list(plan_re.finditer(text))
    if not plans:
        print("FAIL: missing PHONE_REGION_PIPE_PLAN; check the PC environment flag")
        return 1
    ready = [match for match in plans if match.group(1) == "1"]
    if not ready:
        last = plans[-1]
        print(f"FALLBACK: pipeline inactive, reason={last.group(8)}")
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

    if plan.group(7) != "1":
        print("FAIL: expected continuation=1 in PHONE_REGION_PIPE_PLAN")
        return 1

    attn_plans = list(re.finditer(
        r"\[PHONE_TENSOR_ATTN_CHUNK_PLAN\] layer=46 chunks=(\d+) "
        r"tokens=\d+ mode=P2C status=READY",
        text,
    ))
    if not attn_plans or int(attn_plans[-1].group(1)) != n_chunks:
        print(
            "FAIL: missing/mismatched PHONE_TENSOR_ATTN_CHUNK_PLAN "
            f"for {n_chunks} chunks"
        )
        return 1

    attn_chunks = [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        for m in re.finditer(
            r"\[PHONE_TENSOR_ATTN_CHUNK\] layer=46 chunk=(\d+) "
            r"begin=(\d+) tokens=(\d+) "
            r"source=REGION_L45 status=BUILT",
            text,
        )
    ]
    if len(attn_chunks) < n_chunks:
        print(
            f"FAIL: L46 Attention chunk markers={attn_chunks}; "
            f"expected at least {n_chunks}"
        )
        return 1
    attn_chunks = attn_chunks[-n_chunks:]
    if [chunk for chunk, _, _ in attn_chunks] != list(range(n_chunks)):
        print(
            f"FAIL: L46 Attention chunk order={attn_chunks}; "
            f"expected 0..{n_chunks - 1}"
        )
        return 1
    for i in range(1, len(attn_chunks)):
        prev_begin, prev_tokens = attn_chunks[i - 1][1], attn_chunks[i - 1][2]
        if attn_chunks[i][1] != prev_begin + prev_tokens:
            print(f"FAIL: non-contiguous L46 Attention chunks={attn_chunks}")
            return 1

    begin_re = re.compile(
        r"\[PHONE_REGION_PIPE_STAGE\].*?chunk=(\d+).*?"
        r"stage=(C2P_NORM|C2P_FFN_REDUCE|PHONE_REGION|P2C_CONTINUE) "
        r"event=BEGIN"
    )
    observed = [(int(m.group(1)), m.group(2)) for m in begin_re.finditer(block)]

    expected = [
        (c, stage)
        for c in range(n_chunks)
        for stage in (
            "C2P_NORM",
            "C2P_FFN_REDUCE",
            "PHONE_REGION",
            "P2C_CONTINUE",
        )
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
    for bad in (
        "PHONE_REGION_GRAPH_REJECT",
        "PHONE_REGION_SCRATCH_REJECT",
        "PHONE_REGION_CONTINUATION_REJECT",
        "PHONE_REGION_CONTINUATION_DEPENDENCY",
        "CONTINUATION_FUTURE_REGION_INPUT",
        "PHONE_TENSOR_ATTN_OWNER_REJECT",
        "REGION_C2P_PC_PREWORK_REJECT",
        "REGION_C2P_PC_PREWORK_SKIP",
        "PREFILL_INPUT_PARK",
        "Compute error",
    ):
        if bad in block:
            print(f"FAIL: found {bad}")
            return 1

    kept_attn_chunks = [
        int(m.group(1))
        for m in re.finditer(
            r"\[TENSOR_PHONE_KEEP_OWNER\] sg=\d+ layer=46 "
            r"node=phone_tensor_attn_chunk_(\d+)-46 .*?source=subgraph",
            block,
        )
    ]
    if kept_attn_chunks != list(range(n_chunks)):
        print(
            f"FAIL: chunked L46 Attention Phone-owner sequence="
            f"{kept_attn_chunks}; expected 0..{n_chunks - 1}"
        )
        return 1

    prep = list(re.finditer(
        r"\[PHONE_REGION_POOL_PREP\] ready=(\d+) outputs=(\d+) "
        r"(?:output_plans=(\d+) )?.*?"
        r"slot0_mib=([0-9.]+) slot1_mib=([0-9.]+).*?reason=([^\s]+)",
        text,
    ))
    if not prep or prep[-1].group(1) != "1":
        print("FAIL: missing successful PHONE_REGION_POOL_PREP")
        return 1
    if int(prep[-1].group(2)) != n_chunks:
        print(f"FAIL: prewarmed outputs={prep[-1].group(2)}; expected {n_chunks}")
        return 1
    if prep[-1].group(3) is not None and int(prep[-1].group(3)) != n_chunks:
        print(f"FAIL: output bind plans={prep[-1].group(3)}; expected {n_chunks}")
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

    credits = [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        for m in re.finditer(
            r"\[PHONE_REGION_SLOT_CREDIT\] chunk=(\d+) slot=(\d+) "
            r"seq=\d+ marked=(\d+)",
            block,
        )
    ]
    expected_credit_chunks = max(0, n_chunks - 2)
    if len(credits) != expected_credit_chunks:
        print(f"FAIL: slot credits={credits}; expected {expected_credit_chunks}")
        return 1
    if any(marked != 1 for _, _, marked in credits):
        print(f"FAIL: unmarked slot credit: {credits}")
        return 1

    p2c_residuals = [
        (int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)))
        for m in re.finditer(
            r"\[REGION_P2C_RESIDUAL\] layer=46 chunk=(\d+) "
            r"tensor=\S+ bytes=\d+ mode=(\w+) "
            r"residual_offs=(\d+) output_offs=(\d+)",
            block,
        )
    ]
    expected_residuals = [
        (c, "CHUNK_LOCAL", 0)
        for c in range(n_chunks)
    ]
    observed_residuals = [
        (c, mode, residual_offs)
        for c, mode, residual_offs, _ in p2c_residuals
    ]
    if observed_residuals != expected_residuals:
        print(
            f"FAIL: L46 P2C residuals={p2c_residuals}; "
            f"expected chunk-local residual_offs=0 for every chunk"
        )
        return 1

    expected_prework_chunks = list(range(1, n_chunks))
    pc_prework = [
        int(m.group(1))
        for m in re.finditer(
            r"\[REGION_C2P_PC_PREWORK\] layer=34 chunk=(\d+) "
            r".*?state=PC_READY_PHONE_DEFERRED status=OK",
            block,
        )
    ]
    if pc_prework != expected_prework_chunks:
        print(
            f"FAIL: PC prework chunks={pc_prework}; "
            f"expected={expected_prework_chunks}"
        )
        return 1

    norm_stage_consume = [
        int(m.group(1))
        for m in re.finditer(
            r"\[REGION_C2P_NORM_STAGE_CONSUME\] "
            r"layer=34 chunk=(\d+) .*?"
            r"source=PC_PRIVATE_STAGE status=READY",
            block,
        )
    ]
    if norm_stage_consume != expected_prework_chunks:
        print(
            f"FAIL: staged C2P norm chunks={norm_stage_consume}; "
            f"expected={expected_prework_chunks}"
        )
        return 1

    pc_prework_consume = [
        int(m.group(1))
        for m in re.finditer(
            r"\[REGION_C2P_PC_PREWORK_CONSUME\] "
            r"layer=34 chunk=(\d+) .*?status=PC_READY",
            block,
        )
    ]
    if pc_prework_consume != expected_prework_chunks:
        print(
            f"FAIL: PC prework consume chunks={pc_prework_consume}; "
            f"expected={expected_prework_chunks}"
        )
        return 1

    deferred_phone = [
        int(m.group(1))
        for m in re.finditer(
            r"\[REGION_C2P_FFN_DEFERRED_PHONE\] "
            r"phase=PREFILL layer=34 chunk=(\d+) "
            r".*?status=JOIN",
            block,
        )
    ]
    if deferred_phone != expected_prework_chunks:
        print(
            f"FAIL: deferred Phone C2P chunks={deferred_phone}; "
            f"expected={expected_prework_chunks}"
        )
        return 1

    pc_prework_release = [
        int(m.group(1))
        for m in re.finditer(
            r"\[REGION_C2P_PC_PREWORK_RELEASE\] "
            r"layer=34 chunk=(\d+) .*?"
            r"status=RELEASED_AFTER_REDUCE",
            block,
        )
    ]
    if pc_prework_release != expected_prework_chunks:
        print(
            f"FAIL: PC prework release chunks={pc_prework_release}; "
            f"expected={expected_prework_chunks}"
        )
        return 1

    continuation_credits = [
        (int(m.group(1)), int(m.group(2)))
        for m in re.finditer(
            r"\[PHONE_REGION_CONTINUATION_CREDIT\] "
            r"chunk=(\d+) seq=\d+ marked=(\d+) p2c_sg=\d+",
            block,
        )
    ]
    if continuation_credits != [(c, 1) for c in range(n_chunks)]:
        print(
            "FAIL: continuation credits="
            f"{continuation_credits}; expected every chunk marked=1"
        )
        return 1

    completion_watches = [
        int(m.group(1))
        for m in re.finditer(
            r"\[PHONE_REGION_COMPLETION_WATCH\] "
            r"chunk=(\d+) seq=\d+ task=\d+ "
            r"mode=ASYNC_WAIT status=ARMED",
            block,
        )
    ]
    if completion_watches != list(range(n_chunks)):
        print(
            f"FAIL: completion watches={completion_watches}; "
            f"expected 0..{n_chunks - 1}"
        )
        return 1

    completion_events = [
        int(m.group(1))
        for m in re.finditer(
            r"\[PHONE_REGION_COMPLETION_EVENT\] "
            r"chunk=(\d+) seq=\d+ ready=1 status=READY",
            block,
        )
    ]
    if completion_events != list(range(n_chunks)):
        print(
            f"FAIL: completion events={completion_events}; "
            f"expected 0..{n_chunks - 1}"
        )
        return 1

    continuation_waits = [
        (
            int(m.group(1)),
            int(m.group(2)),
            float(m.group(3)),
            float(m.group(4)),
            m.group(5),
            int(m.group(6)),
        )
        for m in re.finditer(
            r"\[PHONE_REGION_CONTINUATION_WAIT\] "
            r"chunk=(\d+) seq=\d+ slot=(\d+) "
            r"wait_ms=([0-9.]+) producer_ms=([0-9.]+) "
            r"wait_mode=(\w+) slot_released=(\d+) status=READY",
            block,
        )
    ]
    if len(continuation_waits) != n_chunks:
        print(
            f"FAIL: continuation waits={continuation_waits}; "
            f"expected {n_chunks} async waits"
        )
        return 1
    for chunk, slot, wait_ms, producer_ms, mode, released in continuation_waits:
        if chunk < 0 or chunk >= n_chunks or slot != chunk % 2:
            print(f"FAIL: invalid continuation wait tuple={continuation_waits}")
            return 1
        if mode != "ASYNC_EVENT" or released != 1:
            print(
                f"FAIL: chunk {chunk} completion mode={mode} "
                f"released={released}; expected ASYNC_EVENT/1"
            )
            return 1
        if producer_ms <= 0:
            print(f"FAIL: chunk {chunk} producer_ms={producer_ms}")
            return 1

    seq_reuse = list(re.finditer(
        r"\[PHONE_REGION_SCRATCH\] chunk=(\d+) slot=(\d+).*?"
        r"reuse_wait_ms=([0-9.]+) wait_mode=(\w+)",
        block,
    ))
    for m in seq_reuse:
        chunk = int(m.group(1))
        if chunk >= 2 and m.group(4) != "NONE":
            print(
                f"FAIL: chunk {chunk} should reuse a slot already released "
                f"by continuation, wait_mode={m.group(4)}"
            )
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

    print(
        f"PASS: {n_chunks} C2P -> Phone Region -> L46 continuations are "
        f"chunk-pipelined (window=2, {plan.group(4)} Meta schedule entries)"
    )
    print("PASS: L46 Attention/QKV/KV/Router/FFN is built per Region chunk")
    print("PASS: Region completion uses async event watches instead of blocking the scheduler")
    print("PASS: next-chunk L34 PC Router/FFN prework is staged across the current Region")
    print("PASS: every Region completion credit is consumed by its own L46 continuation")
    print("PASS: continuation releases the scratch slot before later reuse")
    print("PASS: corresponding L34, Phone, and L46 stages returned successfully")
    print("NOTE: this does not verify GPU kernel duration, KV parity or logits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
