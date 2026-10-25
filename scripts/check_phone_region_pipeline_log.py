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
        r"(?:continuation=(\d+) )?"
        r"(?:pc_overlap=(\d+) )?"
        r"(?:phone_main_local=(\d+) )?"
        r"(?:tail_join_sg=(\d+) )?"
        r"reason=([^\s]+)"
    )
    plans = list(plan_re.finditer(text))
    if not plans:
        print("FAIL: missing PHONE_REGION_PIPE_PLAN; check the PC environment flag")
        return 1
    ready = [match for match in plans if match.group(1) == "1"]
    if not ready:
        last = plans[-1]
        print(f"FALLBACK: pipeline inactive, reason={last.group(11)}")
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

    pc_overlap = plan.group(8) == "1"
    phone_main_local = plan.group(9) == "1"

    if phone_main_local:
        local_plan = list(re.finditer(
            r"\[PHONE_MAIN_LOCAL_PLAN\] requested=1 active=1 "
            r"chunks=(\d+) rpc_patch=15 reason=READY",
            block,
        ))
        if not local_plan or int(local_plan[-1].group(1)) != n_chunks:
            print(
                "FAIL: missing/mismatched successful PHONE_MAIN_LOCAL_PLAN "
                f"for {n_chunks} chunks"
            )
            return 1

        expected_chunks = list(range(n_chunks))

        def chunks_for(pattern: str) -> list[int]:
            return [int(m.group(1)) for m in re.finditer(pattern, block)]

        registered = chunks_for(
            r"\[PHONE_MAIN_PLAN_REGISTERED\] "
            r"chunk=(\d+) .*?status=READY"
        )
        if registered != expected_chunks:
            print(
                f"FAIL: Phone MAIN registrations={registered}; "
                f"expected={expected_chunks}"
            )
            return 1

        armed = chunks_for(
            r"\[PHONE_MAIN_LOCAL_PREP\] "
            r"chunk=(\d+) .*?status=ARMED"
        )
        if armed != expected_chunks:
            print(
                f"FAIL: Phone MAIN armed chunks={armed}; "
                f"expected={expected_chunks}"
            )
            return 1

        c2p_ready_send = chunks_for(
            r"\[PHONE_C2P_READY_SEND\] "
            r"layer=34 chunk=(\d+) .*?"
            r"phone_compute_wait=0 priority=C2P_FFN "
            r"status=QUEUED"
        )
        if c2p_ready_send != expected_chunks:
            print(
                f"FAIL: C2P READY sends={c2p_ready_send}; "
                f"expected={expected_chunks}"
            )
            return 1

        c2p_partial_send = chunks_for(
            r"\[PHONE_C2P_PARTIAL_SEND\] "
            r"layer=34 chunk=(\d+) .*?"
            r"priority=LOW wait_phone=0 .*?status=READY"
        )
        if c2p_partial_send != expected_chunks:
            print(
                f"FAIL: C2P PARTIAL sends={c2p_partial_send}; "
                f"expected={expected_chunks}"
            )
            return 1

        arena_paused = chunks_for(
            r"\[PHONE_MAIN_PC_ARENA_PAUSE\] "
            r"layer=34 chunk=(\d+) sg=\d+ "
            r"reason=C2P_NORM_SNAPSHOT status=PAUSED"
        )
        input_snapshots = chunks_for(
            r"\[PHONE_MAIN_C2P_INPUT_READY\] "
            r"layer=34 chunk=(\d+) task=\d+ "
            r"residual_snapshot=1 status=READY"
        )
        if arena_paused != expected_chunks:
            print(
                f"FAIL: PC arena pauses={arena_paused}; "
                f"expected={expected_chunks}"
            )
            return 1
        if input_snapshots != expected_chunks:
            print(
                f"FAIL: private C2P input/residual snapshots={input_snapshots}; "
                f"expected={expected_chunks}"
            )
            return 1

        local_owned = [
            (int(m.group(1)), m.group(2))
            for m in re.finditer(
                r"\[PHONE_MAIN_LOCAL_OWNED\] "
                r"chunk=(\d+) sg=\d+ "
                r"stage=(PHONE_REGION|P2C_CONTINUE) "
                r"pc_action=SKIP_PHONE_GRAPH "
                r"wait_region=0 status=DELEGATED",
                block,
            )
        ]
        for chunk in expected_chunks:
            stages = {
                stage for c, stage in local_owned if c == chunk
            }
            if "PHONE_REGION" not in stages or "P2C_CONTINUE" not in stages:
                print(
                    f"FAIL: chunk {chunk} local-owned stages={sorted(stages)}; "
                    "expected PHONE_REGION and P2C_CONTINUE"
                )
                return 1

        pc_ffn_enqueued = chunks_for(
            r"\[PHONE_MAIN_PC_FFN_ENQUEUE\] "
            r"chunk=(\d+) .*?"
            r"source=READY_EVENT priority=HIGH .*?"
            r"status=QUEUED"
        )
        pc_ffn_done = chunks_for(
            r"\[PHONE_MAIN_PC_FFN\] "
            r"chunk=(\d+) .*?status=READY"
        )
        partial_rx = chunks_for(
            r"\[PHONE_MAIN_PARTIAL_RX\] "
            r"chunk=(\d+) .*?status=READY"
        )
        pc_reduce_enqueued = chunks_for(
            r"\[PHONE_MAIN_PC_REDUCE_ENQUEUE\] "
            r"chunk=(\d+) .*?"
            r"reason=BOTH_READY priority=HIGH .*?"
            r"status=QUEUED"
        )
        pc_join = chunks_for(
            r"\[PHONE_MAIN_PC_JOIN\] "
            r"chunk=(\d+) .*?status=0"
        )
        for name, found in (
            ("PC FFN enqueue", pc_ffn_enqueued),
            ("PC FFN done", pc_ffn_done),
            ("P2C PARTIAL RX", partial_rx),
            ("PC Reduce enqueue", pc_reduce_enqueued),
            ("PC Reduce join", pc_join),
        ):
            if found != expected_chunks:
                print(
                    f"FAIL: {name} chunks={found}; "
                    f"expected={expected_chunks}"
                )
                return 1

        tail = list(re.finditer(
            r"\[PHONE_MAIN_PC_TAIL_READY\] "
            r"chunks=(\d+) wait_ms=[0-9.]+ "
            r"boundary_sg=\d+ status=READY",
            block,
        ))
        if not tail or int(tail[-1].group(1)) != n_chunks:
            print("FAIL: missing successful PHONE_MAIN_PC_TAIL_READY")
            return 1

        for bad in (
            "PHONE_REGION_CONTINUATION_WAIT",
            "PHONE_REGION_CONTINUATION_REJECT",
            "PHONE_MAIN_LOCAL_REJECT",
            "REGION_C2P_OVERLAP_REJECT",
            "Compute error",
        ):
            if bad in block:
                print(f"FAIL: local-main log contains forbidden {bad}")
                return 1

        # Server-side markers may live in a separate Phone log. If this input
        # contains any of them, require the complete LOW -> MAIN chain.
        server_present = any(
            tag in block for tag in (
                "[PHONE_C2P_READY]",
                "[PHONE_C2P_FFN_DONE]",
                "[PHONE_C2P_PARTIAL_READY]",
                "[PHONE_C2P_REDUCE_ADMIT]",
                "[PHONE_MAIN_CHAIN]",
            )
        )
        if server_present:
            server_checks = (
                ("Phone C2P READY", chunks_for(
                    r"\[PHONE_C2P_READY\] "
                    r"chunk=(\d+) .*?"
                    r"priority=C2P_FFN status=QUEUED"
                )),
                ("Phone C2P FFN", chunks_for(
                    r"\[PHONE_C2P_FFN_DONE\] "
                    r"chunk=(\d+) .*?"
                    r"priority=C2P_FFN status=READY"
                )),
                ("Phone C2P partial", chunks_for(
                    r"\[PHONE_C2P_PARTIAL_READY\] "
                    r"chunk=(\d+) .*?status=READY"
                )),
                ("Phone MAIN admit", chunks_for(
                    r"\[PHONE_C2P_REDUCE_ADMIT\] "
                    r"chunk=(\d+) .*?"
                    r"reason=BOTH_READY priority=MAIN status=QUEUED"
                )),
                ("Phone consume credit", chunks_for(
                    r"\[PHONE_C2P_CONSUME_READY\] "
                    r"chunk=(\d+) .*?"
                    r"phase=REDUCE_DONE status=READY"
                )),
                ("Phone MAIN chain", chunks_for(
                    r"\[PHONE_MAIN_CHAIN\] "
                    r"chunk=(\d+) .*?"
                    r"execute_reduce=1 status=OK"
                )),
            )
            for name, found in server_checks:
                if found != expected_chunks:
                    print(
                        f"FAIL: {name} chunks={found}; "
                        f"expected={expected_chunks}"
                    )
                    return 1

        print(
            f"PASS: local Phone MAIN runtime active for {n_chunks} chunks "
            "(RPC 4.0.15)"
        )
        print(
            "PASS: PC sends C2P READY/PARTIAL without waiting for Phone compute"
        )
        print(
            "PASS: Phone Region/L46 are delegated locally; "
            "no REGION_DONE/CONTINUATION wait is used"
        )
        print(
            "PASS: P2C READY enqueues PC FFN HIGH, PARTIAL joins PC Reduce HIGH"
        )
        if server_present:
            print(
                "PASS: Phone C2P_FFN LOW -> BOTH_READY -> MAIN chain validated"
            )
        else:
            print(
                "NOTE: Phone server log not present; server-side LOW/MAIN "
                "markers were not checked"
            )
        print(
            "NOTE: tail wait is the real L47 data dependency, not a Phone "
            "completion-control wait"
        )
        return 0

    begin_re = re.compile(
        r"\[PHONE_REGION_PIPE_STAGE\].*?chunk=(\d+).*?"
        r"stage=(C2P_NORM|C2P_FFN_REDUCE|C2P_PC_PREP|PHONE_REGION|P2C_CONTINUE) "
        r"event=BEGIN"
    )
    observed = [(int(m.group(1)), m.group(2)) for m in begin_re.finditer(block)]

    if pc_overlap:
        expected = [
            (0, "C2P_NORM"),
            (0, "C2P_FFN_REDUCE"),
            (0, "PHONE_REGION"),
        ]
        for c in range(n_chunks):
            if c + 1 < n_chunks:
                expected.extend([
                    (c + 1, "C2P_NORM"),
                    (c + 1, "C2P_PC_PREP"),
                ])
            expected.append((c, "P2C_CONTINUE"))
            if c + 1 < n_chunks:
                expected.extend([
                    (c + 1, "C2P_FFN_REDUCE"),
                    (c + 1, "PHONE_REGION"),
                ])
    else:
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

    if pc_overlap:
        staged = [
            int(m.group(1))
            for m in re.finditer(
                r"\[REGION_C2P_INPUT_STAGE\] layer=34 chunk=(\d+) "
                r"slot=\d+ task=\d+ bytes=\d+ status=QUEUED",
                block,
            )
        ]
        expected_staged = list(range(1, n_chunks))
        if staged != expected_staged:
            print(
                f"FAIL: overlap input staging chunks={staged}; "
                f"expected={expected_staged}"
            )
            return 1

        prepared = [
            int(m.group(1))
            for m in re.finditer(
                r"\[REGION_C2P_PC_PREP\] layer=34 chunk=(\d+) "
                r"sg=\d+ slot=\d+ .*?status=READY",
                block,
            )
        ]
        if prepared != expected_staged:
            print(
                f"FAIL: PC prep chunks={prepared}; "
                f"expected={expected_staged}"
            )
            return 1

        reused = [
            int(m.group(1))
            for m in re.finditer(
                r"\[REGION_C2P_PC_REUSE\] layer=34 chunk=(\d+) "
                r"sg=\d+ slot=\d+ .*?status=OK",
                block,
            )
        ]
        if reused != expected_staged:
            print(
                f"FAIL: PC reuse chunks={reused}; "
                f"expected={expected_staged}"
            )
            return 1

        released = [
            int(m.group(1))
            for m in re.finditer(
                r"\[REGION_C2P_SLOT_RELEASE\] layer=34 chunk=(\d+) "
                r"slot=\d+ .*?status=FREE",
                block,
            )
        ]
        if released != expected_staged:
            print(
                f"FAIL: overlap slot release chunks={released}; "
                f"expected={expected_staged}"
            )
            return 1

        for c in range(n_chunks - 1):
            region_match = re.search(
                rf"\[PHONE_REGION_PIPE_STAGE\].*?chunk={c} .*?"
                rf"stage=PHONE_REGION event=BEGIN",
                block,
            )
            prep_match = re.search(
                rf"\[REGION_C2P_PC_PREP\] layer=34 chunk={c + 1} "
                rf"sg=\d+ slot=\d+ .*?status=READY",
                block,
            )
            cont_match = re.search(
                rf"\[PHONE_REGION_PIPE_STAGE\].*?chunk={c} .*?"
                rf"stage=P2C_CONTINUE event=BEGIN",
                block,
            )
            if region_match is None or prep_match is None or cont_match is None:
                print(
                    f"FAIL: missing overlap ordering marker for "
                    f"Region{c} -> PC_PREP{c + 1} -> CONTINUE{c}"
                )
                return 1
            if not (
                region_match.start()
                < prep_match.start()
                < cont_match.start()
            ):
                print(
                    f"FAIL: overlap order for chunk {c}: "
                    f"region={region_match.start()} "
                    f"prep={prep_match.start()} "
                    f"continue={cont_match.start()}"
                )
                return 1

        reject = re.search(
            r"\[REGION_C2P_OVERLAP_REJECT\]",
            block,
        )
        if reject:
            print("FAIL: REGION_C2P_OVERLAP_REJECT present")
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

    continuation_waits = [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        for m in re.finditer(
            r"\[PHONE_REGION_CONTINUATION_WAIT\] "
            r"chunk=(\d+) seq=\d+ slot=(\d+) "
            r"wait_ms=[0-9.]+ slot_released=(\d+) status=READY",
            block,
        )
    ]
    expected_waits = [(c, c % 2, 1) for c in range(n_chunks)]
    if continuation_waits != expected_waits:
        print(
            f"FAIL: continuation waits={continuation_waits}; "
            f"expected={expected_waits}"
        )
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

    mode = "pc-overlap" if pc_overlap else "baseline"
    print(
        f"PASS: {n_chunks} C2P -> Phone Region -> L46 continuations are "
        f"chunk-pipelined (window=2, mode={mode}, "
        f"{plan.group(4)} Meta schedule entries)"
    )
    print("PASS: L46 Attention/QKV/KV/Router/FFN is built per Region chunk")
    print("PASS: every Region completion credit is consumed by its own L46 continuation")
    print("PASS: continuation releases the scratch slot before later reuse")
    print("PASS: corresponding L34, Phone, and L46 stages returned successfully")
    print("NOTE: this does not verify GPU kernel duration, KV parity or logits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
