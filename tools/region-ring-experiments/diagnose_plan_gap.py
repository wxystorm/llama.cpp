#!/usr/bin/env python3
"""Read-only diagnostic for predicted vs measured Region Ring prefill.

This does not re-run a prompt, change the Profile, or create benchmark data.
The Stage totals overlap; never sum them into a critical-path attribution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


def number(value):
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def fmt(value):
    v = number(value)
    return f"{v:.3f} ms" if v is not None else "N/A"


def load_json(path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError):
        return {}


def read_metrics(path):
    values = []
    if path.is_file():
        with path.open(encoding="utf-8-sig", errors="replace") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        values.append(obj)
                except ValueError:
                    pass
    return values


def last_record(values, kind, min_tokens=None):
    return next((v for v in reversed(values)
                 if v.get("type") == kind
                 and (min_tokens is None
                      or (number(v.get("tokens")) or 0) >= min_tokens)), {})


def last_log_fields(path, tag):
    found = ""
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as file:
            for line in file:
                if f"[{tag}]" in line:
                    found = line
    result = {}
    for key, val in re.findall(r"\b([a-zA-Z_]\w*)=([^\s]+)", found):
        result[key] = val
    return result


def read_refills(path):
    """Read one log pass, retaining the Bank Mover's real per-refill timing."""
    refills = []
    if not path.is_file():
        return refills
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            if "[REGION_RING_GPU_BANK_MOVER]" not in line or "event=REFILL_DONE" not in line:
                continue
            fields = dict(re.findall(r"\b([a-zA-Z_]\w*)=([^\s]+)", line))
            refills.append(fields)
    return refills


def pair_elapsed(row, start, end):
    first, last = number(row.get(start)), number(row.get(end))
    return last - first if first is not None and last is not None and last >= first else None


def value_line(title, value, note=""):
    print(f"  {title:35s} {fmt(value):>15s}" + (f"  {note}" if note else ""))


def inspect(out, prompt, attempt, profile_json=None):
    stem = Path(prompt).stem
    run = out / "runs" / stem
    attempts = sorted(p for p in run.glob("attempt-*") if p.is_dir())
    if not attempts:
        raise FileNotFoundError(f"No attempts found: {run}")
    if attempt:
        selected = run / f"attempt-{int(attempt):04}"
        if selected not in attempts:
            raise FileNotFoundError(f"No such attempt: {selected}")
    else:
        selected = attempts[-1]
    result = load_json(selected / "result.json")
    metrics = read_metrics(selected / "metrics.jsonl")
    ring = last_record(metrics, "region_ring", 2)
    planner = last_record(metrics, "planner")
    perf = last_record(metrics, "perf")
    timeline = [m for m in metrics if m.get("type") == "region_ring_chunk_trace"]
    trace = last_record(metrics, "region_ring_trace_meta", 2)
    log = selected / "engine.stderr.log"
    planned = last_log_fields(log, "REGION_RING_PLAN_SELECTED")
    modeled = last_log_fields(log, "REGION_RING_PLAN_P2C_MODEL")
    transfer = last_log_fields(log, "REGION_RING_PLAN_TRANSFER_DIAG")
    refills = read_refills(log)

    predicted = number(planner.get("predicted_ms"))
    if predicted is None:
        predicted = number(result.get("plan_predicted_ms"))
    prefill = number(perf.get("prefill_ms"))
    if prefill is None:
        prefill = number(result.get("prefill_ms"))
    ring_wall = number(ring.get("wall_ms"))
    layout = planner.get("selected_layout") or ring.get("layout") or result.get("layout") or ""
    tokens = perf.get("prefill_tokens") or result.get("prefill_tokens") or ring.get("tokens")
    print(f"Diagnostic only: {selected}")
    print(f"  prompt={stem} tokens={tokens} layout={layout}")
    print(f"  status={result.get('status', 'UNKNOWN')} plan_source={planner.get('source', 'N/A')}")
    print()
    print("1. Time domains -- do NOT add parallel stage totals")
    value_line("DAG predicted makespan", predicted)
    value_line("Real measured prefill", prefill)
    value_line("Real Region Ring wall", ring_wall)
    if predicted is not None and prefill is not None:
        value_line("Prefill - DAG prediction", prefill - predicted)
        print(f"  observed/predicted = {prefill / predicted:.2f}x"
              if predicted > 0 else "  predicted=0")
    if prefill is not None and ring_wall is not None:
        value_line("Prefill - Ring wall", prefill - ring_wall,
                   "different timer boundaries; investigate if large")

    print("\n2. GPU Bank -- sums overlap GPU work; wait is NOT H2D total")
    for title, val in [
        ("Predicted Bank H2D total", planned.get("weight_h2d_total")),
        ("Predicted exposed Bank H2D", planned.get("weight_h2d_exposed")),
        ("Predicted hidden Bank H2D", planned.get("weight_h2d_hidden")),
        ("Observed Bank H2D", ring.get("bank_h2d_ms")),
        ("Observed Bank CPU stage copy", ring.get("bank_stage_cpu_copy_ms")),
        ("Observed Bank prepare", ring.get("bank_prepare_ms")),
        ("Observed GPU bank wait", ring.get("bank_gpu_wait_ms")),
        ("Observed Bank mover wall", ring.get("bank_mover_wall_ms")),
    ]:
        value_line(title, val)
    print(f"  observed Bank refills={ring.get('bank_refills', 'N/A')}"
          f"  streaming={ring.get('gpu_bank_streaming', 'N/A')}")
    if ring.get("gpu_bank_force_small_batch"):
        print("  NOTE: small-batch bank mover override enabled")
    modeled_h2d = number(planned.get("weight_h2d_total"))
    observed_h2d = number(ring.get("bank_h2d_ms"))
    if modeled_h2d and observed_h2d:
        print(f"  H2D observed/model ratio = {observed_h2d / modeled_h2d:.2f}x")
    print("  H2D Profile: pre-touched PINNED_DOUBLE buffer, no GGUF faults.")
    print("  Runtime default: DIRECT_PAGEABLE/mmap unless")
    print("    GGML_SCHED_WEIGHT_STAGE_PINNED=1 is explicitly set.")
    if number(ring.get("bank_stage_cpu_copy_ms")) == 0:
        print("  stage_cpu_copy_ms=0: consistent with DIRECT_PAGEABLE path.")
    if refills:
        print("  Region  Bank  upload_mib  H2D_ms  MiB/s  prepare_ms")
        for row in refills:
            mib = number(row.get("upload_mib"))
            ms = number(row.get("h2d_ms"))
            speed = mib * 1000 / ms if mib is not None and ms and ms > 0 else None
            print(f"  {row.get('region', '?'):>6}  {row.get('bank', '?'):>4}"
                  f"  {(f'{mib:.1f}' if mib is not None else 'N/A'):>10}"
                  f"  {(f'{ms:.1f}' if ms is not None else 'N/A'):>6}"
                  f"  {(f'{speed:.1f}' if speed is not None else 'N/A'):>7}"
                  f"  {row.get('prepare_ms', 'N/A')}")
    else:
        print("  Per-refill REFILL_DONE logs: unavailable")

    print("\n3. Device lanes -- busy and waits are overlapping demands")
    for dev in ("cpu", "gpu", "phone"):
        print(f"  [{dev.upper()}] tasks={ring.get(dev + '_tasks', 'N/A')}"
              f" resource_deferred_checks="
              f"{ring.get(dev + '_resource_deferred_checks', 'N/A')}")
        for label, suffix in [
            ("backend wall", "backend_wall_ms"),
            ("compute range", "compute_range_ms"),
            ("lane active", "lane_active_ms"),
            ("prepare", "prepare_ms"),
            ("split copy", "split_copy_call_ms"),
            ("source synchronization", "split_source_sync_ms"),
            ("output sync", "output_sync_ms"),
            ("READY wait", "ready_wait_ms"),
        ]:
            value_line("    " + label, ring.get(dev + "_" + suffix))
    print("  READY wait includes scheduler waits; resource-aware Bank"
          " deferrals happen BEFORE GPU tasks are claimed.")
    if number(ring.get("phone_tasks")) == 0 and "P" in layout:
        print("  PHONE tasks=0 may reflect autonomous HTP streaming:"
              " it does NOT prove Phone/NPU compute time is zero.")
    print("\n4. Transfers and setup (not directly additive)")
    for label, key in [
        ("Embedding", "embedding_ms"),
        ("PC final LM head", "pc_final_head_ms"),
        ("Phone graph prebuild", "phone_static_prebuild_ms"),
        ("PC->Phone push elapsed sum", "phone_input_push_wall_ms"),
        ("PC->Phone ACK wait", "phone_input_push_ack_wait_ms"),
        ("Phone host staging (push)", "phone_input_push_phone_host_stage_ms"),
        ("Phone->PC return wait", "phone_snapshot_ready_wait_ms"),
        ("Phone->PC payload receive", "phone_snapshot_payload_recv_ms"),
        ("Meta PC->Phone copy", "meta_pc_to_phone_copy_ms"),
        ("Meta Phone->PC copy", "meta_phone_to_pc_copy_ms"),
        ("PC/Phone compute overlap", "pc_phone_overlap_ms"),
    ]:
        value_line(label, ring.get(key))
    print("  Planned full-stream return:", modeled.get("return", "N/A"),
          "arena_capture:", modeled.get("arena_capture", "N/A"))
    print("  Planned transfer service (PC-local,C2P,P2C):",
          transfer.get("service_ms", "N/A"))

    if trace:
        print("\n5. PC-clock startup landmarks (measured)")
        for label, start, end in [
            ("embedding", "embed_begin_ms", "embed_end_ms"),
            ("Phone owned-plan registration", "owned_plan_begin_ms", "owned_plan_end_ms"),
            ("P2C subscription", "p2c_subscribe_begin_ms", "p2c_subscribe_end_ms"),
            ("worker launch", "workers_launch_begin_ms", "workers_launch_end_ms"),
        ]:
            value_line(label, pair_elapsed(trace, start, end))
        print(f"  Detailed traced Chunk rows={len(timeline)}")
    else:
        print("\n5. No per-Chunk PC-clock trace: the critical-path gap cannot"
              " be localized by Region/Chunk yet.")
        print("  Re-run ONLY an isolated diagnostic with -TraceTasks and"
              " a new -Out, reusing this independent Profile.")

    if timeline:
        per_region = defaultdict(list)
        for row in timeline:
            per_region[(int(row.get("region", -1)), row.get("stage", ""),
                        row.get("layer_begin"), row.get("layer_end"))].append(row)
        print("\n6. Observed Region/Chunk totals (overlap; NOT the critical path)")
        print("  region  stage    layers   chunks  prepare_ms  compute_ms  bank_wait_ms  ready->complete_ms")
        for (region, stage, begin, end), rows in sorted(per_region.items()):
            def total(k):
                return sum(number(r.get(k)) or 0.0 for r in rows)
            print(f"  {region:>6}  {stage:<7} {str(begin)+'..'+str(end):<8}"
                  f" {len(rows):>6}  {total('prepare_total_ms'):>10.1f}"
                  f"  {total('compute_range_ms'):>10.1f}"
                  f"  {total('bank_wait_ms'):>12.1f}"
                  f"  {total('ready_to_complete_ms'):>18.1f}")

    if profile_json is None:
        profile_json = out / "independent-profile" / "profile.json"
    profile = load_json(profile_json)
    stage_points = profile.get("stage_points", [])
    phone_points = [p for p in stage_points if p.get("device") == "PHONE"]
    print("\n7. Independent NPU Profile coverage")
    print("  profile:", profile_json if profile else "NOT FOUND")
    print("  measured NPU points:", len(phone_points))
    depths = sorted({int(p.get("layers", 0)) for p in phone_points})
    chunks = sorted({int(p.get("tokens", 0)) for p in phone_points})
    kvs = sorted({int(p.get("kv_tokens", 0)) for p in phone_points})
    print(f"  measured depths={depths} chunks={chunks} kv={kvs}")
    chosen_depths = [int(x) for x in re.findall(r"P(\d+)", layout)]
    chunk_width = int(number(planner.get("xp")) or number(ring.get("xp")) or 0)
    print(f"  chosen Phone depths={chosen_depths}, XP={chunk_width}")
    missing_depth = sorted(set(chosen_depths) - set(depths))
    missing_comb = sorted({(d, chunk_width) for d in chosen_depths
                           if not any(p.get("layers") == d and
                                      p.get("tokens") == chunk_width
                                      for p in phone_points)})
    if missing_depth:
        print("  WARNING: no measured points for Phone depths", missing_depth,
              "(must infer from other depths)")
    if missing_comb:
        print("  WARNING: no exact Phone (depth,chunk) anchors:", missing_comb)
    if len(phone_points) <= 9:
        print("  NOTE: small NPU grid; an executed layout can depend heavily"
              " on interpolation/extrapolation.")

    print("\n8. Non-benchmark hypotheses to verify")
    if layout and re.search(r"G\d+$", layout):
        print("  * GPU tail: runtime PC final head exists; Planner's"
              " Phone-final head term is zero in this topology.")
    if planned.get("weight_h2d_exposed") is not None and ring.get("bank_gpu_wait_ms") is not None:
        print("  * Compare exposed H2D (pred) vs bank wait (real),"
              " but do not subtract as if their semantics were identical.")
    if missing_comb:
        print("  * NPU Profile lacks exact depth/chunk anchors for chosen layout.")
    if modeled_h2d and observed_h2d and observed_h2d > modeled_h2d * 2:
        print("  * Major H2D model/runtime mismatch: PINNED_DOUBLE synthetic"
              " vs actual pageable GGUF uploads. Check mmap cold pages,"
              " CUDA synchronization and per-refill MiB/s.")
    if number(ring.get("gpu_resource_deferred_checks")):
        print("  * GPU Bank availability is filtered before claim;"
              " the negligible bank_wait_ms excludes this scheduler delay.")
    if prefill is not None and ring_wall is not None and prefill - ring_wall > 250:
        print("  * Significant time lies outside the measured Ring wall.")
    if not timeline:
        print("  * Need -TraceTasks for exact startup and per-Chunk queueing.")
    print("  * No single component is identified as the 5.25s cause"
          " without comparing real timestamps to per-stage DAG predictions.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
                        help="existing (possibly incomplete) smoke result folder")
    parser.add_argument("--prompt", default="prompt_128_01.txt")
    parser.add_argument("--attempt", type=int, default=None,
                        help="attempt number; default newest")
    parser.add_argument("--profile-json", type=Path, default=None)
    args = parser.parse_args()
    return inspect(args.out, args.prompt, args.attempt, args.profile_json)


if __name__ == "__main__":
    raise SystemExit(main())
