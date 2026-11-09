#!/usr/bin/env python3
"""Run each prompt_N.txt exactly once, with quiet stdout/stderr and CSV output.

Example (PowerShell, one line):
python tools/region-ring-experiments/run_prompt_sweep.py --exe build-tensor-v2/bin/Release/llama-completion.exe --model E:/llama/models/Qwen3-14B-Q4_K_M.gguf --rpc 10.39.52.114:50052 --prompt-dir D:/prima.cpp/prompts

Requires the matching Region Ring C++ JSONL instrumentation and Python 3.9+.
The newer llama-cli can be a server client; llama-completion is the local GGUF runner.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

FIELDS = [
    "prompt_file", "prompt_hint", "actual_prefill_tokens", "actual_decode_tokens",
    "prefill_ms", "prefill_tps", "decode_ms", "decode_tps",
    "load_ms", "total_llama_ms", "process_wall_ms", "status", "exit_code",
    "model", "model_size_bytes", "commit", "prompt_sha256",
    "planner_mode", "skip_profile", "plan_applied", "plan_layout", "runtime_layout",
    "gpu_layers", "cpu_layers", "phone_layers", "tensor_layers",
    "xc", "xg", "xp", "context", "batch", "ubatch", "threads", "seed",
    "n_predict", "gpu_weight_streaming", "no_mmap", "predicted_ms", "baseline_predicted_ms",
    "selected_dag_sim_ms", "final_dag_sim_ms", "final_dag_source",
    "link_ab_serial_ms", "link_ab_independent_ms",
    "link_ab_contended_ms", "link_ab_overlap_ms", "link_ab_status",
    "duplex_profile_points", "link_ab_contended_error_ms",
    "dag_plan_tokens", "first_cycle_e", "final_e", "first_cycle_e_pool", "first_cycle_seeds", "prefix_completed",
    "prefix_cycles", "prefix_e_json", "prefix_service_mspt_json",
    "ring_ubatches", "ring_tokens", "ring_wall_ms", "ring_complete",
    "phone_auto_final_enabled", "phone_final_head_ms",
    "cpu_full_meta_queue", "phone_graph_cache", "phone_graph_slots",
    "phone_graph_slot_revisits", "phone_graph_reuses",
    "phone_graph_reuse_count", "phone_graph_reuse_check_ms",
    "phone_graph_reset_ms", "phone_graph_build_ms", "phone_graph_alloc_ms",
    "phone_input_set_ms", "phone_split_prepare_ms",
    "phone_input_push_enabled", "phone_input_push_accepted",
    "phone_input_push_applied", "phone_input_push_fallback",
    "phone_input_push_mib", "phone_input_push_wall_ms",
    "phone_input_push_worker_wait_ms",
    "phone_input_push_socket_lock_ms", "phone_input_push_hash_ms",
    "phone_input_push_pack_ms", "phone_input_push_header_send_ms",
    "phone_input_push_payload_send_ms", "phone_input_push_socket_send_ms",
    "phone_input_push_ack_wait_ms", "phone_input_push_other_ms",
    "phone_copy_records", "phone_copy_call_ms",
    "phone_copy_apply_calls", "phone_copy_apply_ms",
    "phone_copy_apply_misses", "phone_copy_rpc_set_calls",
    "phone_copy_rpc_set_ms", "phone_copy_rpc_set_mib",
    "phone_copy_hash_hit_calls", "phone_copy_local_calls",
    "phone_copy_other_calls",
    "phone_input_push_phone_recv_header_ms",
    "phone_input_push_phone_recv_alloc_ms",
    "phone_input_push_phone_recv_payload_ms",
    "phone_input_push_phone_hash_ms",
    "phone_input_push_phone_slot_wait_ms",
    "phone_input_push_phone_host_stage_ms",
    "phone_input_push_phone_handler_ms",
    "cpu_split_target_wait_ms", "gpu_split_target_wait_ms", "phone_split_target_wait_ms",
    "cpu_split_source_sync_ms", "gpu_split_source_sync_ms", "phone_split_source_sync_ms",
    "cpu_split_copy_call_ms", "gpu_split_copy_call_ms", "phone_split_copy_call_ms",
    "cpu_split_profiled_ms", "gpu_split_profiled_ms", "phone_split_profiled_ms",
    "cpu_split_copy_mib", "gpu_split_copy_mib", "phone_split_copy_mib",
    "cpu_split_copy_calls", "gpu_split_copy_calls", "phone_split_copy_calls",
    "phone_graph_affinity", "phone_graph_affinity_region",
    "phone_graph_affinity_tokens", "phone_target_graph_tasks",
    "phone_target_consecutive", "phone_target_graph_reuses",
    "phone_target_consecutive_reuses",
    "cpu_tasks", "gpu_tasks", "phone_tasks",
    "cpu_compute_range_ms", "gpu_compute_range_ms", "phone_compute_range_ms",
    "cpu_lane_active_ms", "gpu_lane_active_ms", "phone_lane_active_ms",
    "cpu_prepare_ms", "gpu_prepare_ms", "phone_prepare_ms",
    "cpu_output_sync_ms", "gpu_output_sync_ms", "phone_output_sync_ms",
    "cpu_ready_wait_ms", "gpu_ready_wait_ms", "phone_ready_wait_ms",
    "cpu_prepare_wait_ms", "gpu_prepare_wait_ms", "phone_prepare_wait_ms",
    "cpu_cpu_gate_wait_ms", "gpu_cuda_gate_wait_ms", "pc_phone_overlap_ms",
    "gpu_bank_refills", "gpu_bank_upload_mib", "gpu_bank_prepare_ms",
    "gpu_bank_h2d_ms", "gpu_bank_mover_wall_ms", "gpu_bank_wait_ms",
    "gpu_bank_stage_cpu_copy_ms", "bank_initial_upload_mib",
    "bank_initial_load_ms", "meta_available",
    "meta_pc_to_phone_copy_ms", "meta_phone_to_pc_copy_ms",
    "meta_phone_rpc_ms", "meta_pc_backend_ms", "meta_phone_backend_ms",
    "meta_pc_ffn_ms", "meta_attn_ms", "meta_reduce_ms",
    "meta_wait_ms", "meta_graph_ms", "meta_graph_rebuild_ms",
    "meta_graph_execute_ms", "meta_graph_other_ms",
    "meta_phone_full_graph_assemble_ms",
    "meta_graph_calls", "meta_graph_rebuilds", "meta_return_payload_mib",
    "phone_result_get_count", "phone_result_get_mib",
    "phone_result_request_ms", "phone_result_response_wait_ms",
    "phone_result_payload_recv_ms", "phone_result_rpc_total_ms",
    "phone_snapshot_return_count", "phone_snapshot_ready_wait_ms",
    "phone_snapshot_payload_recv_ms",
    "phone_rpc_available", "phone_rpc_regions_json",
    "phone_rpc_set_tensor_count", "phone_rpc_set_tensor_bytes",
    "phone_rpc_graph_compute_count", "phone_rpc_graph_recompute_count",
    "phone_rpc_graph_snapshot_count",
    "phone_rpc_graph_multi_count", "phone_rpc_graph_multi_nodes",
    "phone_rpc_graph_single_add_count", "phone_rpc_graph_single_other_count",
    "phone_rpc_graph_serialize_ms", "phone_rpc_graph_serialized_mib",
    "phone_rpc_graph_compute_cmd_ms", "phone_rpc_graph_recompute_cmd_ms",
    "phone_rpc_get_tensor_count", "phone_rpc_get_tensor_bytes",
    "estimated_gpu_to_cpu_activation_mib", "estimated_cpu_to_gpu_activation_mib",
    "estimated_pc_to_phone_activation_mib", "estimated_phone_to_pc_activation_mib",
    "run_dir",
]
NAME = re.compile(r"^prompt_(\d+)\.txt$", re.I)


def profile_kv_anchors(prompt_hint: int) -> tuple[int, ...]:
    """Use the smallest native KV anchor set covering the sweep.

    File-name token hints are approximate: leave a modest allowance for
    tokenizer/chat-template differences rather than silently asking the
    Planner to extrapolate beyond a measured KV window.
    """
    if prompt_hint < 1:
        raise ValueError("INVALID_PROMPT_HINT")
    needed = prompt_hint + 32
    anchors = [512, 1024]
    for kv in (2048, 4096, 8192, 16384):
        if needed <= anchors[-1]:
            break
        anchors.append(kv)
    if needed > anchors[-1]:
        raise ValueError("PROMPT_EXCEEDS_NATIVE_PROFILE_KV")
    return tuple(anchors)


def records(path: Path):
    result = []
    if path.is_file():
        for i, line in enumerate(path.read_text("utf-8", errors="replace").splitlines(), 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except ValueError:
                    print(f"Warning: malformed telemetry at {path}:{i}", file=sys.stderr)
    return result


def total(rows, key):
    return round(sum(float(r.get(key, 0) or 0) for r in rows), 3)


# One row per Phone Region/Chunk; populated from the native Ring timeline.
# Parse *after* the measurement_start marker so startup/warmup cannot
# contaminate the measured 358-token Push breakdown.
PHONE_PUSH_TRACE_FIELDS = (
    "ubatch", "region", "chunk", "layer_begin", "layer_end", "token_begin",
    "tokens", "push_status", "push_bytes", "push_begin_ms", "push_end_ms",
    "push_wall_ms", "push_socket_lock_ms", "push_hash_ms",
    "push_pack_ms", "push_header_send_ms", "push_payload_send_ms",
    "push_socket_send_ms", "push_ack_wait_ms", "push_other_ms",
    "push_phone_recv_header_ms", "push_phone_recv_alloc_ms",
    "push_phone_recv_payload_ms", "push_phone_hash_ms",
    "push_phone_slot_wait_ms", "push_phone_host_stage_ms",
    "push_phone_handler_ms",
)


def write_phone_input_push_timeline(folder: Path):
    entries = records(folder / "metrics.jsonl")
    markers = [i for i, row in enumerate(entries)
               if row.get("type") == "measurement_start"]
    measured = entries[markers[-1] + 1:] if markers else entries
    rows = [
        {field: row.get(field) for field in PHONE_PUSH_TRACE_FIELDS}
        for row in measured
        if row.get("type") == "region_ring_chunk_trace"
        and row.get("push_status") is not None
    ]
    if not rows:
        return 0
    rows.sort(key=lambda row: (int(row.get("ubatch") or 0),
                               int(row.get("region") or 0),
                               int(row.get("chunk") or 0)))
    path = folder / "phone_input_push_timeline.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PHONE_PUSH_TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# One record per Scheduler tensor_copy_call, annotated by the synchronous
# RPC client setter on the same PC Phone Worker thread. Saved after inference.
PHONE_COPY_TRACE_FIELDS = (
    "ubatch", "region", "chunk", "split", "tensor", "dst_tensor",
    "kind", "bytes", "copy_call_ms", "rpc_apply_calls",
    "rpc_apply_misses", "rpc_set_calls", "rpc_hash_hit_calls",
    "rpc_fused_calls", "rpc_apply_bytes", "rpc_set_bytes",
    "rpc_hash_hit_bytes", "rpc_apply_ms", "rpc_apply_miss_ms",
    "rpc_set_ms", "rpc_hash_hit_ms", "rpc_fused_ms",
    "src_host", "dst_host",
)


def write_phone_copy_timeline(folder: Path):
    entries = records(folder / "metrics.jsonl")
    markers = [i for i, row in enumerate(entries)
               if row.get("type") == "measurement_start"]
    measured = entries[markers[-1] + 1:] if markers else entries
    rows = [
        {field: row.get(field) for field in PHONE_COPY_TRACE_FIELDS}
        for row in measured if row.get("type") == "region_ring_phone_copy"
    ]
    if not rows:
        return 0
    rows.sort(key=lambda row: (
        int(row.get("ubatch") or 0), int(row.get("region") or 0),
        int(row.get("chunk") or 0), int(row.get("split") or 0),
        str(row.get("tensor") or "")))
    path = folder / "phone_copy_timeline.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=PHONE_COPY_TRACE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def match_last(pattern, string):
    matches = re.findall(pattern, string, flags=re.I)
    return matches[-1] if matches else None


def count_layers(layout):
    result = {"G": 0, "C": 0, "P": 0, "C2P": 0, "P2C": 0}
    for key, n in re.findall(r"(C2P|P2C|G|C|P)(\d+)", layout or ""):
        result[key] += int(n)
    return result


def estimated_activation_payloads(ring_rows):
    """Lower-bound FP32 hidden-state handoffs for full G/C/P-only layouts.

    These are modeled activation bytes, NOT socket TCP bytes. Meta/RPC may
    transfer extra tensors. Directional islands use READY/PARTIAL payloads
    with a different shape, so leave their estimates blank.
    """
    keys = (
        "gpu_to_cpu", "cpu_to_gpu", "pc_to_phone", "phone_to_pc",
    )
    totals = {k: 0.0 for k in keys}
    edge_types = {
        ("G", "C"): ("gpu_to_cpu",),
        ("C", "G"): ("cpu_to_gpu",),
        ("C", "P"): ("pc_to_phone",),
        ("P", "C"): ("phone_to_pc",),
        ("G", "P"): ("gpu_to_cpu", "pc_to_phone"),
        ("P", "G"): ("phone_to_pc", "cpu_to_gpu"),
    }
    if not ring_rows:
        return None
    for row in ring_rows:
        layout = str(row.get("layout", ""))
        stages = re.findall(r"(C2P|P2C|G|C|P)(\d+)", layout)
        if any(kind in {"C2P", "P2C"} for kind, _ in stages):
            return None
        if not stages or len(stages) < 2:
            continue
        tokens = int(row.get("tokens", 0) or 0)
        hidden = int(row.get("hidden", 0) or 0)
        if tokens <= 0 or hidden <= 0:
            return None
        mib = tokens * hidden * 4 / (1024.0 * 1024.0)
        for (left, _), (right, _) in zip(stages, stages[1:]):
            for direction in edge_types.get((left, right), ()):
                totals[direction] += mib
    return {key: round(value, 3) for key, value in totals.items()}


def git_head(root):
    try:
        x = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                           text=True, capture_output=True, timeout=5)
        return x.stdout.strip() if x.returncode == 0 else ""
    except OSError:
        return ""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def legacy_profile_cache_file(root: Path, model: Path, exe: Path,
                              rpc: str, threads: int,
                              no_mmap: bool, batch: int,
                              ubatch: int,
                              phone_fingerprint: str = "") -> Path:
    """Cache one measured legacy Hybrid Profile across prompt processes.

    NOTE: This is not the native G/C/N/A matrix. That is stored separately
    by run_v2_stage_matrix.py. The RPC endpoint is only a location key;
    a Phone backend/firmware update requires -ForceProfile until a
    production-RPC device fingerprint API is wired into this coordinator.
    """
    from run_v2_stage_matrix import host_device_id
    material = {
        "schema": 1,
        "model_sha256": file_sha256(model),
        "model_bytes": model.stat().st_size,
        "engine_sha256": file_sha256(exe),
        "pc_id": host_device_id(),
        "rpc_endpoint": rpc,
        "phone_fingerprint": phone_fingerprint,
        "threads": threads,
        "no_mmap": no_mmap,
        "batch": batch,
        "ubatch": ubatch,
    }
    digest = hashlib.sha256(json.dumps(
        material, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", model.stem)[:80]
    return root / "model-profiles" / stem / "hybrid-cache" / (
        digest + ".rrhp")


def result_row(args, folder, prompt, hint, context, code, elapsed, commit):
    entries = records(folder / "metrics.jsonl")
    plan = next((r for r in reversed(entries) if r.get("type") == "planner"), {})
    perf = next((r for r in reversed(entries) if r.get("type") == "perf"), {})
    # The model may execute Region Ring during startup/warmup. Count only
    # ubatches after the final formal llama_perf_context_reset marker.
    markers = [i for i, r in enumerate(entries)
               if r.get("type") == "measurement_start"]
    measured = entries[markers[-1] + 1:] if markers else entries
    ring = [r for r in measured if r.get("type") == "region_ring"]
    phone_copy = [r for r in measured
                  if r.get("type") == "region_ring_phone_copy"]
    logs = ""
    for file in ("engine.stderr.log", "output.stdout.txt"):
        path = folder / file
        if path.is_file():
            logs += path.read_text(encoding="utf-8", errors="replace") + "\n"
    number = r"(\d+(?:\.\d+)?)"
    p = match_last(r"prompt eval time\s*=\s*" + number +
                   r"\s*ms\s*/\s*(\d+)\s*tokens", logs)
    d = match_last(r"(?<!prompt )eval time\s*=\s*" + number +
                   r"\s*ms\s*/\s*(\d+)\s*runs", logs)
    load = match_last(r"\bload time\s*=\s*" + number + r"\s*ms", logs)
    pre_ms = perf.get("prefill_ms", float(p[0]) if p else None)
    pre_n = perf.get("prefill_tokens", int(p[1]) if p else None)
    dec_ms = perf.get("decode_ms", float(d[0]) if d else None)
    dec_n = perf.get("decode_tokens", int(d[1]) if d else None)
    pre_tps = perf.get("prefill_tps")
    dec_tps = perf.get("decode_tps")
    if pre_tps is None and pre_ms and pre_n:
        pre_tps = round(1000 * pre_n / pre_ms, 3)
    if dec_tps is None and dec_ms and dec_n:
        dec_tps = round(1000 * dec_n / dec_ms, 3)
    # The runtime layout (not merely the Planner's proposal) is authoritative.
    runtime_layout = ring[0].get("layout", "") if ring else ""
    layout = runtime_layout or (plan.get("selected_layout", "")
                                if plan.get("applied") else "")
    layers = count_layers(layout)
    # A perf reset may recheck a resident pair and emit ALREADY_READY
    # with zero bytes. Sum real preload events, not simply the last line.
    bank_summaries = re.findall(r"\[REGION_RING_GPU_BANK_LOAD_SUM\][^\n]*", logs)
    bank_summaries = [x for x in bank_summaries if "ALREADY_READY" not in x]
    def bank_sum(field):
        return round(sum(float(m) for line in bank_summaries
                         if (m := match_last(field + number, line))), 3)
    initial_mib = bank_sum(r"upload_mib=") if bank_summaries else None
    initial_ms = bank_sum(r"wall_ms=") if bank_summaries else None
    # predicted_ms is the best candidate, which may NOT have been applied.
    # Record both its DAG makespan and the DAG makespan of the actual plan.
    selected_dag_sim_ms = plan.get("predicted_ms")
    baseline_dag_sim_ms = plan.get("baseline_predicted_ms")
    if plan.get("applied") is True:
        final_dag_sim_ms = selected_dag_sim_ms
        final_dag_source = "selected"
    elif plan.get("applied") is False:
        final_dag_sim_ms = baseline_dag_sim_ms
        final_dag_source = "baseline"
    else:
        final_dag_sim_ms = None
        final_dag_source = None
    if final_dag_sim_ms is None:
        final_dag_source = None
    row = {
        "prompt_file": prompt.name, "prompt_hint": hint,
        "actual_prefill_tokens": pre_n, "actual_decode_tokens": dec_n,
        "prefill_ms": pre_ms, "prefill_tps": pre_tps,
        "decode_ms": dec_ms, "decode_tps": dec_tps,
        "load_ms": perf.get("load_ms", float(load) if load else None),
        "total_llama_ms": perf.get("total_ms"),
        "process_wall_ms": round(elapsed, 3),
        "status": (
            "FAIL" if code != 0 else
            "NO_METRICS" if not perf and not ring else
            "NO_PERF" if not perf else
            "NO_RING" if not ring else
            "INCOMPLETE_RING" if any(
                not x.get("complete") or x.get("status") != 0
                for x in ring) else "PASS"
        ),
        "exit_code": code,
        "model": str(args.model), "model_size_bytes": args.model.stat().st_size,
        "commit": commit,
        "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "planner_mode": args.planner,
        "skip_profile": args.skip_profile,
        "plan_applied": True if args.skip_profile and ring else plan.get("applied"),
        "plan_layout": runtime_layout if args.skip_profile and ring else plan.get("selected_layout"),
        "runtime_layout": runtime_layout,
        "gpu_layers": layers["G"], "cpu_layers": layers["C"],
        "phone_layers": layers["P"],
        "tensor_layers": layers["C2P"] + layers["P2C"],
        "xc": ring[0].get("xc") if ring else plan.get("xc"),
        "xg": ring[0].get("xg") if ring else plan.get("xg"),
        "xp": ring[0].get("xp") if ring else plan.get("xp"),
        "context": context, "batch": args.batch, "ubatch": args.ubatch,
        "threads": args.threads, "seed": args.seed,
        "n_predict": args.n_predict, "gpu_weight_streaming": args.streaming,
        "no_mmap": getattr(args, "no_mmap", False),
        "predicted_ms": selected_dag_sim_ms,
        "baseline_predicted_ms": baseline_dag_sim_ms,
        "selected_dag_sim_ms": selected_dag_sim_ms,
        "final_dag_sim_ms": final_dag_sim_ms,
        "final_dag_source": final_dag_source,
        "link_ab_serial_ms": plan.get("link_ab_serial_ms"),
        "link_ab_independent_ms": (
            plan.get("link_ab_independent_ms")
            if (plan.get("link_ab_independent_ms") or -1) >= 0 else None
        ),
        "link_ab_contended_ms": (
            plan.get("link_ab_contended_ms")
            if (plan.get("link_ab_contended_ms") or -1) >= 0 else None
        ),
        "link_ab_overlap_ms": plan.get("link_ab_overlap_ms"),
        "link_ab_status": plan.get("link_ab_status"),
        "duplex_profile_points": plan.get("duplex_profile_points"),
        "dag_plan_tokens": plan.get("plan_tokens"),
        "first_cycle_e": plan.get("first_cycle_e"), "final_e": plan.get("e"),
        "first_cycle_e_pool": plan.get("first_cycle_e_pool"),
        "first_cycle_seeds": plan.get("first_cycle_seeds"),
        "prefix_completed": plan.get("prefix_completed"),
        "prefix_cycles": plan.get("prefix_cycles"),
        "prefix_e_json": json.dumps(plan.get("prefix_e", [])),
        "prefix_service_mspt_json": json.dumps(
            plan.get("prefix_service_mspt", [])),
        "ring_ubatches": len(ring), "ring_tokens": sum(x.get("tokens", 0) for x in ring),
        "ring_wall_ms": total(ring, "wall_ms"),
        "phone_auto_final_enabled": any(
            x.get("phone_auto_final_enabled") is True for x in ring),
        "phone_final_head_ms": (
            total(ring, "phone_final_head_ms")
            if any("phone_final_head_ms" in x for x in ring) else None),
        "cpu_full_meta_queue": ring[0].get("cpu_full_meta_queue") if ring else None,
        "phone_graph_cache": ring[0].get("phone_graph_cache") if ring else None,
        "phone_graph_slots": ring[0].get("phone_graph_slots") if ring else None,
        "phone_graph_slot_revisits": sum(x.get("phone_graph_slot_revisits", 0) for x in ring) if ring else None,
        "phone_graph_reuses": sum(x.get("phone_graph_reuses", 0) for x in ring) if ring else None,
        "phone_graph_reuse_count": total(ring, "phone_graph_reuse_count") if any("phone_graph_reuse_count" in x for x in ring) else None,
        "phone_graph_affinity": ring[0].get("phone_graph_affinity") if ring else None,
        "phone_graph_affinity_region": ring[0].get("phone_graph_affinity_region") if ring else None,
        "phone_graph_affinity_tokens": ring[0].get("phone_graph_affinity_tokens") if ring else None,
        "phone_target_graph_tasks": sum(x.get("phone_target_graph_tasks", 0) for x in ring) if ring else None,
        "phone_target_consecutive": sum(x.get("phone_target_consecutive", 0) for x in ring) if ring else None,
        "phone_target_graph_reuses": sum(x.get("phone_target_graph_reuses", 0) for x in ring) if ring else None,
        "phone_target_consecutive_reuses": sum(x.get("phone_target_consecutive_reuses", 0) for x in ring) if ring else None,
        "ring_complete": all(x.get("complete", False) for x in ring),
        "pc_phone_overlap_ms": total(ring, "pc_phone_overlap_ms"),
        "gpu_bank_refills": int(total(ring, "bank_refills")),
        "gpu_bank_upload_mib": total(ring, "bank_upload_mib"),
        "gpu_bank_prepare_ms": total(ring, "bank_prepare_ms"),
        "gpu_bank_h2d_ms": total(ring, "bank_h2d_ms"),
        "gpu_bank_mover_wall_ms": total(ring, "bank_mover_wall_ms"),
        "gpu_bank_wait_ms": total(ring, "bank_gpu_wait_ms"),
        "gpu_bank_stage_cpu_copy_ms": total(ring, "bank_stage_cpu_copy_ms"),
        "bank_initial_upload_mib": initial_mib,
        "bank_initial_load_ms": initial_ms,
        "meta_available": any(x.get("meta_available", False) for x in ring),
        "run_dir": str(folder),
    }
    payload = estimated_activation_payloads(ring)
    if payload is not None:
        for direction, mib in payload.items():
            row[f"estimated_{direction}_activation_mib"] = mib
    for dev in ("cpu", "gpu", "phone"):
        for suffix in ("tasks", "compute_range_ms", "lane_active_ms",
                       "prepare_ms", "output_sync_ms", "ready_wait_ms",
                       "prepare_wait_ms",
                        "split_target_wait_ms", "split_source_sync_ms",
                        "split_copy_call_ms", "split_profiled_ms",
                        "split_copy_mib", "split_copy_calls"):
            row[f"{dev}_{suffix}"] = total(ring, f"{dev}_{suffix}")
    for key in ("phone_input_push_accepted", "phone_input_push_applied",
                "phone_input_push_fallback", "phone_input_push_mib",
                "phone_input_push_wall_ms", "phone_input_push_worker_wait_ms",
                "phone_input_push_socket_lock_ms", "phone_input_push_hash_ms",
                "phone_input_push_pack_ms", "phone_input_push_header_send_ms",
                "phone_input_push_payload_send_ms",
                "phone_input_push_socket_send_ms", "phone_input_push_ack_wait_ms",
                "phone_input_push_other_ms",
                "phone_input_push_phone_recv_header_ms",
                "phone_input_push_phone_recv_alloc_ms",
                "phone_input_push_phone_recv_payload_ms",
                "phone_input_push_phone_hash_ms",
                "phone_input_push_phone_slot_wait_ms",
                "phone_input_push_phone_host_stage_ms",
                "phone_input_push_phone_handler_ms"):
        row[key] = total(ring, key)
    row["phone_input_push_enabled"] = any(
        record.get("phone_input_push_enabled") is True for record in ring)
    for key in ("phone_graph_reuse_check_ms", "phone_graph_reset_ms",
                "phone_graph_build_ms", "phone_graph_alloc_ms",
                "phone_input_set_ms", "phone_split_prepare_ms"):
        row[key] = total(ring, key) if any(key in x for x in ring) else None
    for key in (
        "meta_pc_to_phone_copy_ms", "meta_phone_to_pc_copy_ms",
        "meta_phone_rpc_ms", "meta_pc_backend_ms", "meta_phone_backend_ms",
        "meta_pc_ffn_ms", "meta_attn_ms", "meta_reduce_ms",
        "meta_wait_ms", "meta_graph_ms", "meta_graph_rebuild_ms",
        "meta_graph_execute_ms", "meta_graph_other_ms",
        "meta_phone_full_graph_assemble_ms",
        "meta_graph_calls", "meta_graph_rebuilds", "meta_return_payload_mib",
        "phone_result_get_count", "phone_result_get_mib",
        "phone_result_request_ms", "phone_result_response_wait_ms",
        "phone_result_payload_recv_ms", "phone_result_rpc_total_ms",
        "phone_snapshot_return_count", "phone_snapshot_ready_wait_ms",
        "phone_snapshot_payload_recv_ms",
    ):
        row[key] = (
            total(ring, key)
            if row["meta_available"] and any(key in x for x in ring)
            else None
        )
    # These are CPU host-call durations, not Phone GPU kernel time. The
    # logical copy bytes may already exist on Phone via PUSH_APPLY.
    if phone_copy:
        row["phone_copy_records"] = len(phone_copy)
        row["phone_copy_call_ms"] = total(phone_copy, "copy_call_ms")
        row["phone_copy_apply_calls"] = int(sum(
            int(r.get("rpc_apply_calls") or 0) for r in phone_copy))
        row["phone_copy_apply_ms"] = total(phone_copy, "rpc_apply_ms")
        row["phone_copy_apply_misses"] = int(sum(
            int(r.get("rpc_apply_misses") or 0) for r in phone_copy))
        row["phone_copy_rpc_set_calls"] = int(sum(
            int(r.get("rpc_set_calls") or 0) for r in phone_copy))
        row["phone_copy_rpc_set_ms"] = total(phone_copy, "rpc_set_ms")
        row["phone_copy_rpc_set_mib"] = round(
            sum(int(r.get("rpc_set_bytes") or 0) for r in phone_copy)
            / 1048576.0, 3)
        row["phone_copy_hash_hit_calls"] = int(sum(
            int(r.get("rpc_hash_hit_calls") or 0) for r in phone_copy))
        row["phone_copy_local_calls"] = sum(
            r.get("kind") == "LOCAL_COPY" for r in phone_copy)
        row["phone_copy_other_calls"] = sum(
            r.get("kind") == "OTHER_COPY" for r in phone_copy)

    # Region-level Phone Full RPC counters are recorded only for threaded
    # all-Full layouts. Keep the individual UBatch/Region identities in CSV.
    phone_rpc_rows = [
        r for r in ring
        if r.get("phone_rpc_available") is True and
        isinstance(r.get("phone_rpc_regions"), list)
    ]
    row["phone_rpc_available"] = bool(phone_rpc_rows) if ring else None
    if phone_rpc_rows:
        row["phone_rpc_regions_json"] = json.dumps(
            [{"ubatch": r.get("ubatch"), "regions": r["phone_rpc_regions"]}
             for r in phone_rpc_rows], separators=(",", ":"))
        for key in (
            "set_tensor_count", "set_tensor_bytes",
            "graph_compute_count", "graph_recompute_count",
            "graph_snapshot_count", "graph_multi_count", "graph_multi_nodes",
            "graph_single_add_count", "graph_single_other_count",
            "get_tensor_count", "get_tensor_bytes",
        ):
            row[f"phone_rpc_{key}"] = sum(
                int(region.get(key, 0) or 0)
                for record in phone_rpc_rows
                for region in record["phone_rpc_regions"]
                if isinstance(region, dict)
            )
        def rpc_profile_sum(key):
            return sum(int(region.get(key, 0) or 0)
                       for record in phone_rpc_rows
                       for region in record["phone_rpc_regions"]
                       if isinstance(region, dict))
        row["phone_rpc_graph_serialize_ms"] = round(
            rpc_profile_sum("graph_serialize_us") / 1000.0, 3)
        row["phone_rpc_graph_serialized_mib"] = round(
            rpc_profile_sum("graph_serialized_bytes") / 1048576.0, 3)
        row["phone_rpc_graph_compute_cmd_ms"] = round(
            rpc_profile_sum("graph_compute_cmd_us") / 1000.0, 3)
        row["phone_rpc_graph_recompute_cmd_ms"] = round(
            rpc_profile_sum("graph_recompute_cmd_us") / 1000.0, 3)
    if (plan.get("applied") is True and ring and
            row["link_ab_contended_ms"] is not None):
        row["link_ab_contended_error_ms"] = round(
            row["link_ab_contended_ms"] - total(ring, "wall_ms"), 3)
    row["cpu_cpu_gate_wait_ms"] = total(ring, "cpu_cpu_gate_wait_ms")
    row["gpu_cuda_gate_wait_ms"] = total(ring, "gpu_cuda_gate_wait_ms")
    return {field: row.get(field) for field in FIELDS}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rpc", required=True)
    ap.add_argument("--prompt-dir", type=Path, default=Path(r"D:\prima.cpp\prompts"))
    ap.add_argument("--out", type=Path, default=Path("region-ring-results"))
    ap.add_argument("--planner", default="apply-wide")
    ap.add_argument("--streaming", choices=("on", "off"), default="on")
    ap.add_argument("--duplex-profile", choices=("on", "off"), default="off",
                    help="measure solo/overlapped C2P and P2C RPC transfer at startup; diagnostic only")
    ap.add_argument("--log-verbosity", type=int, choices=range(0, 6), default=4,
                    metavar="{0,1,2,3,4,5}",
                    help="llama -lv threshold (default: 4, required for GGML/LLAMA INFO logs; 5 enables DEBUG)")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--baseline-chunk-tokens", type=int, default=64)
    ap.add_argument("--min-context", type=int, default=1024)
    ap.add_argument("--context-padding", type=int, default=256)
    ap.add_argument("--force-layout", default="")
    ap.add_argument("--force-xc", type=int)
    ap.add_argument("--force-xg", type=int)
    ap.add_argument("--force-xp", type=int)
    ap.add_argument("--skip-profile", action="store_true",
                    help="skip Hybrid profiling and Planner search for a forced layout")
    ap.add_argument("--force-profile", action="store_true",
                    help="refresh the model-bound legacy Profile on the first prompt; then reuse it")
    ap.add_argument("--no-profile-cache", action="store_true",
                    help="disable model-bound legacy Profile reuse (diagnostic only)")
    ap.add_argument("--no-v2-profile", action="store_true",
                    help="use legacy G/C/P only, without native GCNA stage profiling")
    ap.add_argument("--v2-profile-dir", type=Path, default=None,
                    help="optional folder containing stage-profile-v2.csv and profile.json")
    ap.add_argument("--v2-profile-phone", default="",
                    help="native Phone profile service HOST:PORT; default RPC host:50177")
    ap.add_argument("--native-profile-exe", type=Path, default=None,
                    help="native isolated G/C/N/A Stage profiler binary")
    ap.add_argument("--no-mmap", action="store_true",
                    help="disable memory mapping when loading the model")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.exe = args.exe.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    args.prompt_dir = args.prompt_dir.expanduser().resolve()
    if not args.exe.is_file() or not args.model.is_file() or not args.prompt_dir.is_dir():
        ap.error("--exe, --model and --prompt-dir must exist")
    if args.batch < args.ubatch or min(args.batch, args.ubatch, args.n_predict) <= 0:
        ap.error("--batch >= --ubatch > 0 and --n-predict > 0 required")
    if args.baseline_chunk_tokens <= 0:
        ap.error("--baseline-chunk-tokens must be positive")
    if args.force_layout and None in (args.force_xc, args.force_xg, args.force_xp):
        ap.error("--force-layout requires all three --force-xc/xg/xp arguments")
    prompts = []
    for path in args.prompt_dir.iterdir():
        m = NAME.fullmatch(path.name)
        if m and path.is_file():
            prompts.append((int(m.group(1)), path))
    prompts.sort(key=lambda x: (x[0], x[1].name))
    if not prompts:
        ap.error(f"no prompt_*.txt files found in {args.prompt_dir}")
    if args.skip_profile and (
            args.planner != "apply-wide" or not args.force_layout or
            not all(isinstance(x, int) and x > 0
                    for x in (args.force_xc, args.force_xg, args.force_xp))):
        ap.error("--skip-profile requires --planner apply-wide and full --force-layout/xc/xg/xp")
    if args.force_profile and (args.skip_profile or args.no_profile_cache):
        ap.error("--force-profile cannot be combined with --skip-profile or --no-profile-cache")
    root = Path(__file__).resolve().parents[2]
    # Add only the longer KV shapes needed by this sweep. Actual token
    # counts are checked at Runtime; filename hints are not exact.
    try:
        required_profile_kvs = profile_kv_anchors(
            max(hint for hint, _ in prompts))
    except ValueError as exc:
        ap.error(str(exc))
    profile_cache = None
    # Native isolated G/C/N/A shapes are sampled at most once per sweep.
    # The cache's PASS-only CSV is consumed by the original C++ apply-wide
    # shortlist/DAG, not by a second production Planner.
    v2_csv = None
    if args.planner == "apply-wide" and not (
            args.skip_profile or args.no_v2_profile):
        from v2_profile_coordinator import prepare_native_profile
        try:
            v2_csv = prepare_native_profile(
                root, args.model, args.rpc, args.threads,
                args.force_profile, args.dry_run,
                directory=args.v2_profile_dir,
                profiler_exe=args.native_profile_exe,
                phone_endpoint=args.v2_profile_phone,
                required_kvs=tuple(required_profile_kvs))
        except (OSError, ValueError, RuntimeError) as exc:
            ap.error(f"native GCNA Profile failed: {exc}")
    phone_identity = ""
    if v2_csv is not None and (v2_csv.parent / "profile.json").is_file():
        info = json.loads((v2_csv.parent / "profile.json").read_text("utf-8"))
        phone = info.get("devices", {}).get("phone", {})
        phone_identity = json.dumps({
            "device_id": phone.get("device_id"),
            "exe_sha256": phone.get("exe_sha256"),
            "runtime_sha256": phone.get("runtime_sha256"),
        }, sort_keys=True)
    if args.planner == "apply-wide" and not (
            args.skip_profile or args.no_profile_cache):
        profile_cache = legacy_profile_cache_file(
            root, args.model, args.exe, args.rpc,
            args.threads, args.no_mmap, args.batch, args.ubatch,
            phone_fingerprint=phone_identity)
        if not args.dry_run:
            profile_cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"[REGION_RING_PROFILE_CACHE] source=LEGACY_HYBRID "
              f"path={profile_cache} exists={int(profile_cache.is_file())}",
              flush=True)
    commit = git_head(root)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (args.out / f"{stamp}-{os.getpid()}").resolve()
    out.mkdir(parents=True)
    rows = []
    for prompt_index, (hint, prompt) in enumerate(prompts):
        folder = out / prompt.stem
        folder.mkdir(exist_ok=False)
        ctx = max(args.min_context,
                  ((hint + args.n_predict + args.context_padding + 255) // 256) * 256)
        env = os.environ.copy()
        for key in ("GGML_META_PIPELINE_DEBUG", "LLAMA_HYBRID_REGION_RING_TRACE",
                    "GGML_CPU_TASK_DEBUG",
                    "LLAMA_HYBRID_REGION_RING_LAYOUT",
                    "LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS",
                    "LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT",
                    "LLAMA_HYBRID_REGION_RING_FORCE_XC",
                    "LLAMA_HYBRID_REGION_RING_FORCE_XG",
                    "LLAMA_HYBRID_REGION_RING_FORCE_XP",
                    "LLAMA_HYBRID_REGION_RING_SKIP_PROFILE",
                    "LLAMA_REGION_RING_PROFILE_CACHE_PATH",
                    "LLAMA_REGION_RING_PROFILE_FORCE",
                    "LLAMA_REGION_RING_V2_PROFILE_CSV",
                    "LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS",
                    "LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP"):
            env.pop(key, None)
        env.update({
            "LLAMA_HYBRID_REGION_RING_V1": "1",
            "LLAMA_HYBRID_REGION_RING_EXEC": "fifo-pc-phone",
            "LLAMA_HYBRID_REGION_RING_PLANNER": args.planner,
            "LLAMA_HYBRID_REGION_RING_SKIP_PROFILE":
                "1" if args.skip_profile else "0",
            "LLAMA_HYBRID_REGION_RING_PLAN_TOKENS": str(hint),
            "LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS":
                str(args.baseline_chunk_tokens),
            "LLAMA_HYBRID_REGION_RING_GPU_WEIGHT_STREAM":
                "1" if args.streaming == "on" else "0",
            "LLAMA_HYBRID_REGION_RING_DUPLEX_PROFILE":
                "1" if args.duplex_profile == "on" else "0",
            "LLAMA_REGION_RING_METRICS_FILE": str(folder / "metrics.jsonl"),
        })
        if profile_cache is not None:
            env["LLAMA_REGION_RING_PROFILE_CACHE_PATH"] = str(profile_cache)
            env["LLAMA_REGION_RING_PROFILE_FORCE"] = (
                "1" if args.force_profile and prompt_index == 0 else "0")
        if v2_csv is not None:
            env.update({
                "LLAMA_REGION_RING_V2_PROFILE_CSV": str(v2_csv),
                # The model-owned Plan supplies HTP Region IDs, so an
                # operator-provided list must never override selected ranks.
                "LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP": "0",
                "LLAMA_REGION_RING_PHONE_GRAPHLESS": "0",
                "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW": "1",
                "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES": "1",
                "GGML_META_PHONE_FULL_REGION_GRAPH": "1",
                "LLAMA_REGION_RING_PHONE_AUTONOMOUS": "1",
                "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL": "1",
                "LLAMA_REGION_RING_PHONE_INPUT_PUSH": "1",
                "LLAMA_REGION_RING_PHONE_INPUT_STREAM": "1",
                "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM": "1",
                "LLAMA_REGION_RING_PHONE_ASYNC_GRAPH_SETUP": "0",
            })
        if env.get("LLAMA_REGION_RING_PHONE_COPY_TRACE") == "1":
            # One opt-in switch collects both the per-copy detail and the
            # reference split breakdown for coverage reconciliation.
            env["LLAMA_REGION_RING_TASK_TRACE"] = "1"
            env["LLAMA_REGION_RING_PREPARE_BREAKDOWN"] = "1"
        if args.force_layout:
            env.update({
                "LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT": args.force_layout,
                "LLAMA_HYBRID_REGION_RING_FORCE_XC": str(args.force_xc),
                "LLAMA_HYBRID_REGION_RING_FORCE_XG": str(args.force_xg),
                "LLAMA_HYBRID_REGION_RING_FORCE_XP": str(args.force_xp),
            })
        command = [
            str(args.exe), "-m", str(args.model), "--rpc", args.rpc,
            "--split-mode", "tensor", "--flash-attn", "on",
            "-c", str(ctx), "-b", str(args.batch), "-ub", str(args.ubatch),
            "-n", str(args.n_predict), "-t", str(args.threads),
            "--seed", str(args.seed), "--no-conversation",
            "-f", str(prompt), "-lv", str(args.log_verbosity),
        ]
        if args.no_mmap:
            command.append("--no-mmap")
        config = {
            "command": command, "cwd": str(root), "repeat": 1,
            "prompt_file": str(prompt), "prompt_hint": hint,
            "commit": commit, "log_verbosity": args.log_verbosity,
            "no_mmap": args.no_mmap,
            "skip_profile": args.skip_profile,
            "profile_cache_file": str(profile_cache) if profile_cache else None,
            "native_v2_profile_csv": str(v2_csv) if v2_csv else None,
            "force_profile": bool(args.force_profile and prompt_index == 0),
            "ring_environment": {k: v for k, v in env.items()
                                 if k.startswith("LLAMA_HYBRID_REGION_RING_")
                                 or k in ("LLAMA_REGION_RING_METRICS_FILE",
                                          "LLAMA_REGION_RING_PROFILE_CACHE_PATH",
                                          "LLAMA_REGION_RING_PROFILE_FORCE",
                                          "LLAMA_REGION_RING_V2_PROFILE_CSV",
                                          "LLAMA_REGION_RING_TASK_TRACE",
                                          "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES",
                                          "LLAMA_REGION_RING_PHONE_AUTONOMOUS",
                                          "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL",
                                          "LLAMA_REGION_RING_PHONE_INPUT_PUSH",
                                          "LLAMA_REGION_RING_PHONE_INPUT_STREAM",
                                          "LLAMA_REGION_RING_PRIORITY_BANK_MOVER",
                                          "LLAMA_REGION_RING_PRIORITY_PHONE_PUSH",
                                          "LLAMA_REGION_RING_PRIORITY_P2C_RECV",
                                          "LLAMA_REGION_RING_PHONE_COPY_TRACE",
                                          "LLAMA_REGION_RING_PREPARE_BREAKDOWN")
                                 or k in ("GGML_META_PHONE_FULL_ADD_FUSION",
                                          "GGML_META_RING_PHONE_FUSION")},
            "phone_full_add_fusion_requested":
                env.get("GGML_META_PHONE_FULL_ADD_FUSION", "0") == "1",
            "notes": [
                "Token count from filename is only a hint; record actual prefill tokens.",
                "Default Ring layout uses actual GGUF depth; inherited layouts are cleared.",
                "Baseline Chunk size applies only without Planner-selected XC/XG/XP.",
                "Device compute_range/prepare/output sync can overlap. Do not sum to wall_ms.",
                "Model load and Planner profiling are outside prefill_ms.",
                "Transfer-only PC/Phone network time needs a separate diagnostic run.",
            ],
        }
        (folder / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        exitcode = -1
        elapsed = 0.0
        if not args.dry_run:
            start = time.perf_counter()
            with (folder / "output.stdout.txt").open("wb") as stdout, \
                 (folder / "engine.stderr.log").open("wb") as stderr:
                try:
                    p = subprocess.run(command, cwd=root, env=env,
                                       stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr,
                                       check=False)
                    exitcode = p.returncode
                except OSError as exc:
                    stderr.write(f"LAUNCH_ERROR: {exc}\n".encode("utf-8"))
            elapsed = 1000.0 * (time.perf_counter() - start)
        row = result_row(args, folder, prompt, hint, ctx, exitcode, elapsed, commit)
        if not args.dry_run:
            write_phone_input_push_timeline(folder)
            write_phone_copy_timeline(folder)
        if args.dry_run:
            row["status"], row["exit_code"] = "DRY_RUN", ""
        (folder / "result.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(row)
        print(f"{prompt.name}: {row['status']}, prefill={row['prefill_tps']} tok/s, "
              f"decode={row['decode_tps']} tok/s, "
              f"final_dag_sim_ms={row['final_dag_sim_ms']}")
    csv_file = out / "summary.csv"
    with csv_file.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary: {csv_file}")
    return 0 if args.dry_run or all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
