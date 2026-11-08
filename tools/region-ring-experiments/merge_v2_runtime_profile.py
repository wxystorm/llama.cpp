#!/usr/bin/env python3
"""Attach measured production Phone Bank/Handoff RPC Patch33 telemetry to V2 Profile.

Input is the ordinary PC inference engine stderr: the Phone's structured
Bank-Mover and N->Adreno events are already pulled over RPC, so no
Android log copy or sidecar TCP benchmark is involved.

The output is a NEW merged JSON file. Independent GGUF Stage samples
remain intact. Do not assert planner readiness from this tool:
model/run identity and production activation-transfer DAG have not been
independently established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

PATTERN = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)")
BANK = "[PHONE_BANK_RUNTIME_PROFILE]"
HANDOFF = "[PHONE_LOCAL_HANDOFF_RUNTIME_PROFILE]"
COMPLETE = "[PHONE_STAGE_PROFILE_RPC] event=COMPLETE"


def tags(line: str) -> dict[str, str]:
    return dict(PATTERN.findall(line))


def number(data: dict[str, str], key: str, *, integer: bool = False
           ) -> float | int:
    try:
        v = int(data[key]) if integer else float(data[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("INVALID_RUNTIME_FIELD:" + key) from exc
    if (not math.isfinite(v)) or v < 0:
        raise ValueError("INVALID_RUNTIME_NONFINITE_OR_NEGATIVE:" + key)
    return v


def collect(log: Path) -> list[dict[str, Any]]:
    """Parse bounded structured events only; never evaluate log contents."""
    by_run: dict[int, dict[str, Any]] = {}
    with log.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            kind = ("bank" if BANK in line else
                    "handoff" if HANDOFF in line else
                    "complete" if COMPLETE in line else None)
            if not kind:
                continue
            data = tags(line)
            if data.get("source") != "RPC_PATCH33" and kind != "complete":
                continue
            run = number(data, "run", integer=True)
            if not 0 < run < 2**64:
                raise ValueError("INVALID_RUNTIME_RUN_ID")
            block = by_run.setdefault(run, {"run_id": run})
            if kind == "complete":
                if data.get("status") != "PASS":
                    continue
                block["pull_complete"] = True
                block["stage_samples"] = number(data, "samples", integer=True)
            else:
                # A second same-run summary might represent duplicate log
                # output. Accept byte-for-byte identical fields only.
                prev = block.get(kind)
                if prev is not None and prev != data:
                    raise ValueError("CONFLICTING_RUNTIME_SAMPLE_FOR_RUN")
                block[kind] = data

    records = []
    for run, block in sorted(by_run.items()):
        if not block.get("pull_complete"):
            continue
        bank_raw = block.get("bank")
        handoff_raw = block.get("handoff")
        if bank_raw is None or handoff_raw is None:
            # No synthetic zero fields for missing RPC measurements.
            continue
        bank = {
            "status": bank_raw.get("status"),
            "overlap_state": bank_raw.get("overlap_state"),
            "refills": number(bank_raw, "refills", integer=True),
            "failed": number(bank_raw, "failed", integer=True),
            "inflight": number(bank_raw, "inflight", integer=True),
            "unstaged": number(bank_raw, "unstaged", integer=True),
            "overflow": number(bank_raw, "overflow", integer=True),
            "device_windows": number(bank_raw, "device_windows",
                                     integer=True),
        }
        for field in (
            "queue_ms", "host_stage_wall_ms", "htp_lock_wait_ms",
            "htp_critical_wall_ms", "precompile_ms",
            "host_stage_htp_overlap_ms", "host_stage_adreno_overlap_ms",
            "htp_critical_adreno_overlap_ms",
        ):
            bank[field] = number(bank_raw, field)
        if (bank["host_stage_htp_overlap_ms"] >
                bank["host_stage_wall_ms"] + 0.02 or
                bank["host_stage_adreno_overlap_ms"] >
                bank["host_stage_wall_ms"] + 0.02 or
                bank["htp_critical_adreno_overlap_ms"] >
                bank["htp_critical_wall_ms"] + 0.02):
            raise ValueError("BANK_OVERLAP_EXCEEDS_MEASURED_INTERVAL")
        bank["host_stage_htp_overlap_fraction"] = (
            bank["host_stage_htp_overlap_ms"] /
            bank["host_stage_wall_ms"]
            if bank["host_stage_wall_ms"] > 0 else None)
        bank["htp_critical_adreno_overlap_fraction"] = (
            bank["htp_critical_adreno_overlap_ms"] /
            bank["htp_critical_wall_ms"]
            if bank["htp_critical_wall_ms"] > 0 else None)
        bank["usable_as_runtime_calibration"] = bool(
            bank["status"] == "PASS" and bank["refills"] >= 1 and
            bank["failed"] == 0 and bank["inflight"] == 0 and
            bank["overflow"] == 0 and bank["unstaged"] == 0 and
            bank["device_windows"] > 0 and
            bank["overlap_state"] == "MEASURED")

        handoff = {
            "status": handoff_raw.get("status"),
            "edges": number(handoff_raw, "edges", integer=True),
            "incomplete": number(handoff_raw, "incomplete", integer=True),
            "bytes": number(handoff_raw, "bytes", integer=True),
            "host_move_ms": number(handoff_raw, "host_move_ms"),
            "ready_to_upload_ms": number(handoff_raw, "ready_to_upload_ms"),
            "upload_ms": number(handoff_raw, "upload_ms"),
            "ready_to_device_ms": number(handoff_raw, "ready_to_device_ms"),
            "host_copy_bytes": number(handoff_raw, "host_copy_bytes",
                                      integer=True),
            "intermediate_p2c_bytes": number(
                handoff_raw, "intermediate_p2c_bytes", integer=True),
        }
        handoff["usable_as_runtime_calibration"] = bool(
            handoff["status"] == "PASS" and handoff["edges"] >= 1 and
            handoff["incomplete"] == 0 and handoff["bytes"] > 0 and
            handoff["host_copy_bytes"] == 0 and
            handoff["intermediate_p2c_bytes"] == 0)
        records.append({
            "run_id": run, "stage_samples": block["stage_samples"],
            "bank": bank, "npu_to_adreno": handoff,
            "source": "RPC_PATCH33_PRODUCTION_PHONE_RUNTIME",
            "clock_domain": "PHONE_GGML_TIME_US",
        })
    return records


def merge(profile_path: Path, log_path: Path, out_path: Path) -> dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("schema") != 2 or not profile.get("model_sha256"):
        raise ValueError("V2_PROFILE_SCHEMA_OR_MODEL_FINGERPRINT_MISSING")
    # Accept the PC run root directly. Each engine.stderr.log is parsed
    # independently to preserve per-process RUN IDs and avoid accidental
    # merging of different sessions that reused a numerical run ID.
    files = (sorted(log_path.rglob("engine.stderr.log"))
             if log_path.is_dir() else [log_path])
    if not files:
        raise ValueError("NO_PC_ENGINE_LOG_FILES")
    runs: list[dict[str, Any]] = []
    source_files = []
    for source in files:
        measured = collect(source)
        digest = hashlib.sha256()
        with source.open("rb") as f:
            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                digest.update(block)
        source_files.append({"file": str(source),
                             "sha256": digest.hexdigest(),
                             "complete_runs": len(measured)})
        for record in measured:
            record["pc_log_file"] = str(source)
            record["pc_log_sha256"] = digest.hexdigest()
        runs.extend(measured)
    if not runs:
        raise ValueError("NO_COMPLETE_PATCH33_PRODUCTION_RPC_TELEMETRY")
    profile["runtime_telemetry"] = {
        "source": "PC_ENGINE_LOG_PHONE_RPC_PATCH33",
        "pc_logs": source_files,
        "model_identity_verified_against_runtime": False,
        "runs": runs,
        "bank_measured_run_count": sum(
            r["bank"]["usable_as_runtime_calibration"] for r in runs),
        "npu_to_adreno_measured_run_count": sum(
            r["npu_to_adreno"]["usable_as_runtime_calibration"] for r in runs),
        "production_rpc_activation_transfer_complete": False,
        "planner_eligible": False,
    }
    # A patch33 RPC log by itself lacks a full provenance/shape tie to the
    # independent GGUF-prefix compute samples, memory budgets, and planned
    # scheduling DAG. Do NOT silently mark a Profile as planner-ready.
    profile["planner_ready"] = False
    pending = list(profile.get("pending", []))
    for item in ("runtime_model_shape_identity_verification",
                 "production_rpc_activation_transfer_profile",
                 "runtime_memory_feasibility", "multi_backend_dag"):
        if item not in pending:
            pending.append(item)
    if any(r["bank"]["usable_as_runtime_calibration"] for r in runs):
        pending = [x for x in pending
                   if x != "real_dual_bank_overlapped_refill_profile"]
    profile["pending"] = pending
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(out_path)
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--pc-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        profile = merge(args.profile, args.pc_log, args.out)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("[V2_RUNTIME_PROFILE] status=FAIL reason=" + str(exc),
              file=sys.stderr)
        return 1
    runtime = profile["runtime_telemetry"]
    print("[V2_RUNTIME_PROFILE] status=SAVED"
          f" runs={len(runtime['runs'])}"
          f" real_bank_runs={runtime['bank_measured_run_count']}"
          f" n_to_a_runs={runtime['npu_to_adreno_measured_run_count']}"
          " planner_ready=false"
          f" file={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
