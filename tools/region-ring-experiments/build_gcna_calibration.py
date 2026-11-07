#!/usr/bin/env python3
"""Assemble *measured* Phone NPU/Adreno calibration from Patch32 PC logs.

No Android log copy, no device login, no guessed bandwidth, and no
synthetic fallback if the NPU never refilled its third Bank. The full-model
base Profile (CPU/CUDA/network) remains separate and must already exist.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from build_real_stage_profile import case_dirs, profile_lines, number
from gcna_probe_model import _gguf, BLOCK, fingerprint

MIB = 1024 * 1024
NEEDED = (
    "gpu_free_bytes", "gpu_total_bytes", "host_available_bytes",
    "gpu_stage_arena_bytes", "host_stage_budget_bytes",
    "host_stage_bytes", "host_stage_ms", "repack_bytes", "repack_ms",
)

def positive(raw: dict, key: str, case: Path) -> float:
    v = number(raw.get(key))
    if v is None or v <= 0:
        raise ValueError(f"PHONE_PROFILE_RESOURCE_NOT_MEASURED:{key}:{case}")
    return v

def weight_layer_mib(model: Path) -> float:
    reader = _gguf().GGUFReader(model)
    block_bytes = {}
    for t in reader.tensors:
        m = BLOCK.match(t.name)
        if m:
            ix = int(m[1])
            block_bytes[ix] = block_bytes.get(ix, 0) + t.n_bytes
    if not block_bytes:
        raise ValueError("NO_QWEN3_LAYER_WEIGHTS_IN_GGUF")
    return round(max(block_bytes.values()) / MIB, 6)

def assemble(model: Path, profile_dir: Path, runs: Path,
             sample_manifest: Path, stage_csv: Path) -> dict:
    from model_profile import gguf_info
    info = gguf_info(model)
    manifest = json.loads((profile_dir / "model_profile.json").read_text(
        encoding="utf-8"))
    if manifest.get("model") != info:
        raise ValueError("BASE_MODEL_PROFILE_MISMATCH:run existing base Profile first")
    sample = json.loads(sample_manifest.read_text(encoding="utf-8"))
    if sample.get("source_model_fingerprint") != info["fingerprint"] or \
            sample.get("mode") != "COMPACT_GCNA_PROBE_NOT_FULL_INFERENCE" or \
            sample.get("phone_weight_source") != "ORIGINAL_GGUF_FIRST_L_LAYERS":
        raise ValueError("COMPACT_PROBE_NOT_BOUND_TO_SOURCE_MODEL")
    with stage_csv.open(newline="", encoding="utf-8-sig") as src:
        stages = list(csv.DictReader(src))
    classes = {x["device"] for x in stages if float(x["compute_ms"]) > 0}
    if classes != {"CPU", "GPU", "PHONE", "PHONE_GPU"}:
        raise ValueError("FOUR_BACKEND_STAGE_PROFILE_INCOMPLETE")
    successful = []
    for case in case_dirs([runs]):
        result_file = case / "result.json"
        if not result_file.is_file() or json.loads(
                result_file.read_text(encoding="utf-8")).get("status") != "PASS":
            continue
        reports = profile_lines(case / "engine.stderr.log",
                                "[PHONE_PROFILE_RESOURCES]")
        completed = profile_lines(case / "engine.stderr.log",
                                  "[PHONE_STAGE_PROFILE_RPC] event=COMPLETE")
        for report in reports:
            if report.get("source") != "RPC" or report.get("status") != "PASS":
                continue
            if not any(c.get("run") == report.get("run") and
                       c.get("status") == "PASS" for c in completed):
                raise ValueError("PHONE_RESOURCE_RUN_ID_MISMATCH")
            successful.append((case, report))
    if not successful:
        raise ValueError("NO_PHONE_RESOURCE_RPC_SAMPLES")
    measures = {k: [positive(r,k,case) for case,r in successful]
                for k in NEEDED}
    # The Phone's reported free bytes and MemAvailable are observed AFTER
    # sampling. Conservatively reserve 10% extra on both independent and
    # shared budgets; never treat total memory as available free memory.
    gpu_budget = 0.9 * min(measures["gpu_free_bytes"]) / MIB
    shared_budget = 0.9 * min(
        min(measures["gpu_free_bytes"]),
        min(measures["host_available_bytes"])) / MIB
    if min(gpu_budget, shared_budget) <= 0:
        raise ValueError("PHONE_FREE_MEMORY_NOT_MEASURED")
    rate = lambda byte_key,ms_key: statistics.median(
        b / MIB / (ms / 1000.0)
        for b,ms in zip(measures[byte_key], measures[ms_key]))
    result = {
        "schema": 1,
        "model_fingerprint": info["fingerprint"],
        "stage_csv": stage_csv.name,
        "calibration": {
            "phone_gpu_budget_mib": round(gpu_budget, 6),
            "phone_gpu_weight_layer_mib": weight_layer_mib(model),
            # Dual-slot GPU Stage buffer peak, not an invented zero scratch.
            "phone_gpu_scratch_mib": round(
                2.0 * max(measures["gpu_stage_arena_bytes"]) / MIB, 6),
            "phone_shared_budget_mib": round(shared_budget, 6),
            # Host vector N->A ownership transfer has zero Host memcpy.
            # A->N direct-copy candidates are pruned without a measured rate.
            "npu_host_stage_mibps": round(
                rate("host_stage_bytes","host_stage_ms"), 6),
            "npu_bank_repack_mibps": round(
                rate("repack_bytes","repack_ms"), 6),
            "npu_host_stage_budget_mib": round(
                min(measures["host_stage_budget_bytes"]) / MIB, 6),
        },
        "origin": {
            "mode": "PATCH32_PHONE_PROFILE_RPC_AND_COMPACT_GCNA_REAL_STAGE",
            "sampled_layers": sample["layers"],
            "source_layers": sample["source_model_layers"],
            "probe_pattern": sample["pattern"],
            "probe_cycles": sample["cycles"],
            "probe_model_fingerprint": sample["probe_model_fingerprint"],
            "resource_cases": [str(case) for case,_ in successful],
            "host_handoff": "F32_VECTOR_MOVE_ZERO_HOST_COPY_OPENCL_UPLOAD_MEASURED",
            "budget_safety_factor": 0.9,
            "phone_gpu_scratch": "TWO_SLOT_STAGE_ARENA_PEAK",
            "phone_gpu_weights": "EXACT_QUANTIZED_GGUF_LAYER_BYTES",
            "npu_bank_transfer": "REAL_THIRD_BANK_REFILL_ONLY",
        },
    }
    return result

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--profile-dir", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--probe-manifest", type=Path, required=True)
    ap.add_argument("--stage-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()
    if args.output.exists() and not args.update:
        raise FileExistsError(
            "Existing calibration will not be overwritten; pass --update")
    data = assemble(args.model.resolve(strict=True),
                    args.profile_dir.resolve(strict=True),
                    args.run.resolve(strict=True),
                    args.probe_manifest.resolve(strict=True),
                    args.stage_csv.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"[GCNA_CALIBRATION] status=PASS cases={len(data['origin']['resource_cases'])} "
          f"file={args.output}")

if __name__ == "__main__":
    main()
