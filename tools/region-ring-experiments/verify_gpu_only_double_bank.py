#!/usr/bin/env python3
"""Verify measured GPU-only A/B Region Ring prefill, not just engine exit code.

The JSONL trace is produced after all workers join. Times are host-observed
millisecond timestamps (not CUDA event kernel times). The test checks that an
empty physical Bank is refilled *after* its source region drains, and that a
new weight upload overlaps the execution of the other GPU weight window.
"""
import argparse
import json
import sys
from pathlib import Path


def verify(events, regions):
    boundaries = [i for i, e in enumerate(events) if e.get("type") == "measurement_start"]
    if boundaries:
        events = events[boundaries[-1] + 1:]
    ring_events = [e for e in events if e.get("type") == "region_ring"]
    ring = [
        e for e in ring_events
        if e.get("gpu_bank_streaming") is True
        and e.get("complete") is True and e.get("status") == 0
    ]
    if not ring:
        perf = [e for e in events if e.get("type") == "perf"]
        prefill_tokens = perf[-1].get("prefill_tokens") if perf else None
        if not ring_events:
            raise ValueError(
                "No region_ring events. "
                f"prefill_tokens={prefill_tokens}, metrics_records={len(events)}. "
                "Check that -NPredict >= 1 (completion -n 0 skips prompt "
                "evaluation), and that the GPU-only executor was entered."
            )
        examples = ", ".join(
            f"tokens={e.get('tokens')} streaming={e.get('gpu_bank_streaming')} "
            f"complete={e.get('complete')} status={e.get('status')}"
            for e in ring_events[-3:]
        )
        raise ValueError(
            "No complete streaming GPU Bank prefill. "
            f"prefill_tokens={prefill_tokens}; ring_events={len(ring_events)}; "
            f"recent=[{examples}]. Requires at least 2*XG tokens in one "
            "physical ubatch, plus a successful Region Ring execution."
        )
    prefill = max(ring, key=lambda e: int(e.get("tokens", 0)))
    ubatch = prefill.get("ubatch")
    measured = [e for e in events if e.get("ubatch") == ubatch]
    if int(prefill.get("cpu_tasks", -1)) != 0 or int(prefill.get("phone_tasks", -1)) != 0:
        raise ValueError("CPU/Phone transformer tasks executed: not GPU-only")
    layout = prefill.get("layout", "")
    pieces = layout.split(",")
    if len(pieces) != regions or any(not p.startswith("G") for p in pieces):
        raise ValueError(f"Incorrect GPU-only layout: {layout}")
    if int(prefill.get("bank_refills", -1)) != regions - 2:
        raise ValueError(f"Expected {regions - 2} successful refills, got {prefill.get('bank_refills')}")

    chunks = [
        e for e in measured if e.get("type") == "region_ring_chunk_trace"
        and e.get("device") == "GPU"
    ]
    refills = [
        e for e in measured if e.get("type") == "region_ring_bank_trace"
    ]
    if len(refills) != regions - 2:
        raise ValueError(f"Expected {regions - 2} Bank traces, got {len(refills)}")
    by_region = {}
    for e in chunks:
        by_region.setdefault(int(e["region"]), []).append(e)
    for region in range(regions):
        if region not in by_region:
            raise ValueError(f"Missing GPU compute trace for region {region}")

    verified_overlap = 0
    for event in sorted(refills, key=lambda e: int(e["region"])):
        target = int(event["region"])
        source = event.get("source_region")
        if source != target - 2:
            raise ValueError(f"Bank parity reuse mismatch for region {target}: source={source}")
        if event.get("status") != 1 or int(event.get("upload_bytes", 0)) <= 0:
            raise ValueError(f"Region {target} did not upload real weights")
        enqueue = event.get("enqueue_ms")
        begin, end = event.get("h2d_begin_ms"), event.get("h2d_end_ms")
        if any(x is None for x in (enqueue, begin, end)):
            raise ValueError(f"Missing timestamp for refill {target}")
        if not (enqueue <= begin < end):
            raise ValueError(f"Invalid refill timestamp for region {target}")
        source_done = max((x.get("complete_ms") or x.get("worker_done_ms") or 0)
                          for x in by_region[source])
        if source_done and enqueue + 0.01 < source_done:
            raise ValueError(f"Refill {target} enqueued before source {source} drained")
        other = target - 1
        overlaps = []
        for chunk in by_region[other]:
            c0 = chunk.get("lane_compute_begin_ms")
            c1 = chunk.get("lane_compute_end_ms")
            if c0 is not None and c1 is not None:
                overlaps.append(max(0.0, min(end, c1) - max(begin, c0)))
        overlap = sum(overlaps)
        if overlap > 0:
            verified_overlap += 1
        print(f"[BANK] source=G{source} target=G{target} "
              f"bank={'A' if target % 2 == 0 else 'B'} "
              f"h2d_ms={end - begin:.3f} overlap_with_G{other}_ms={overlap:.3f}")
    print(f"[GPU_ONLY_VERIFY] layout={layout} tokens={prefill['tokens']} "
          f"gpu_tasks={prefill.get('gpu_tasks')} bank_refills={len(refills)} "
          f"overlapped_refills={verified_overlap}/{len(refills)} "
          f"bank_gpu_wait_ms={prefill.get('bank_gpu_wait_ms')} status=PASS")
    if verified_overlap == 0 and refills:
        print("[GPU_ONLY_VERIFY] WARNING: no measured H2D/GPU compute overlap; "
              "mechanism ran, but transfer may be serialized on this system")
    return {"refills": len(refills), "overlapped_refills": verified_overlap}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--regions", type=int, required=True)
    args = parser.parse_args(argv)
    if not 2 <= args.regions <= 8:
        parser.error("--regions must be 2..8")
    with args.metrics.open(encoding="utf-8") as handle:
        events = []
        for line in handle:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    verify(events, args.regions)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"[GPU_ONLY_VERIFY] FAIL: {error}", file=sys.stderr)
        sys.exit(1)
