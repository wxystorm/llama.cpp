#!/usr/bin/env python3
"""Immutable rolling calibration from successful Prompt runtime measurements.

Every epoch updates only measured shapes over the initial independently
profiled CPU/CUDA/HTP grid. Never substitute RPC CPU wall time for HTP.
A backend may execute zero layers in a particular five-PASS window; the
independent baseline then supplies its still-valid performance samples.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from build_real_stage_profile import SCHEMA, extract, aggregate


def completed_cases(out: Path) -> list[Path]:
    # One latest PASS per distinct Prompt, matching the main runner's
    # successful-Prompt counter. An earlier PASS followed by an interrupted
    # rerun does not count twice or contaminate the current Profile.
    cases = []
    for prompt_dir in sorted((out / "runs").glob("prompt_*")):
        trials = sorted(prompt_dir.glob("attempt-*/result.json"))
        if not trials:
            continue
        result = trials[-1]
        metrics = result.parent / "metrics.jsonl"
        if not metrics.is_file():
            continue
        try:
            if json.loads(result.read_text(encoding="utf-8")).get("status") == "PASS":
                cases.append(result.parent)
        except (OSError, ValueError):
            pass
    # Snapshot by actual successful completion order, not by filename:
    # groups are executed in order but retries/resumes can interleave.
    def completed_at(folder: Path):
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        return (result.get("end_time", ""),
                str(folder))
    return sorted(cases, key=completed_at)


def snapshot_paths(out: Path, epoch: int) -> tuple[Path, Path]:
    base = out / "profile-epochs" / f"v{epoch:02d}"
    return base / "real-stage-profile.csv", base / "manifest.json"


def profile_snapshot(out: Path, epoch: int, every: int,
                     phone_log: Optional[Path] = None) -> tuple[Path, dict, bool]:
    """Build at exactly every N completed PASS runs, persist immutable CSV.

    If the snapshot already exists (e.g. resume after a crash), verify and
    reuse it. Atomic rename prevents partial profiles from being loaded.
    """
    if epoch < 1 or every < 1:
        raise ValueError("epoch/every must be positive")
    csv_path, meta_path = snapshot_paths(out, epoch)
    independent = out / "independent-profile" / "profile.csv"
    if not independent.is_file():
        raise ValueError("Independent backend Profile is missing; "
                         "never generate a rolling snapshot without a G/C/N baseline")
    base_sha = hashlib.sha256(independent.read_bytes()).hexdigest()
    with independent.open(encoding="utf-8-sig", newline="") as stream:
        baseline = list(csv.DictReader(stream))
    if not baseline or not any(row["device"] == "PHONE" and
                               float(row.get("compute_ms") or 0) > 0
                               for row in baseline):
        raise ValueError("Independent HTP baseline lacks positive service measurements")
    if csv_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if (meta.get("sha256") != actual or meta.get("epoch") != epoch or
                meta.get("independent_sha256") != base_sha):
            raise ValueError("Profile snapshot or independent baseline digest mismatch")
        return csv_path, meta, False

    cases = completed_cases(out)
    expected = every * epoch
    # The wrapper enters an epoch immediately after N completed prompts; a
    # crash and resume can only increase the number of completed cases.
    if len(cases) < expected:
        raise ValueError(f"Need {expected} PASS cases before profile epoch {epoch}")
    # A rolling epoch is exactly the N successful Prompts ending at
    # the epoch boundary. Never average old epochs into later updates.
    selected_cases = cases[expected - every:expected]
    samples, diag = extract(selected_cases, phone_log)
    points = aggregate(samples)
    if not points:
        raise ValueError(
            "No real Stage measurements; enable LLAMA_REGION_RING_TASK_TRACE "
            "and LLAMA_REGION_RING_REAL_STAGE_PROFILE")
    # Runtime measurements supersede independent calibration ONLY for the
    # same (device,depth,Chunk,KV,terminal) shape. All other G/C/N points
    # remain from the independent backend grid. Zero-layer N/CPU/GPU Plans
    # do not invalidate this baseline, and no missing costs become zero.
    def key(p):
        return (p["device"], int(p["layers"]), int(p["tokens"]),
                int(p["kv_tokens"]), int(p["terminal"]))
    updated = {key(p): p for p in baseline}
    for p in points:
        updated[key(p)] = p
    merged = [updated[k] for k in sorted(updated)]
    if not any(p["device"] == "PHONE" and
               float(p["compute_ms"]) > 0 for p in merged):
        raise ValueError("Merged Profile has no real HTP service points")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(merged)
        f.flush()
        os.fsync(f.fileno())
    digest = hashlib.sha256(csv_tmp.read_bytes()).hexdigest()
    meta = {
        "schema": 1, "epoch": epoch, "profile_every": every,
        "passes_at_snapshot": expected, "window_prompts": every,
        "source": "INDEPENDENT_BASELINE_WITH_REAL_STAGE_OVERRIDES",
        "independent_sha256": base_sha,
        "refreshed_points": len(points),
        "point_count": len(merged), "device_points": {
            device: sum(p["device"] == device for p in merged)
            for device in ("CPU", "GPU", "PHONE")
        }, "diagnostic": diag, "sha256": digest,
        "phone_log": str(phone_log) if phone_log else None,
        "fallback": "INDEPENDENT_REAL_HTP_BASELINE_FOR_UNMEASURED_SHAPES",
    }
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    os.replace(csv_tmp, csv_path)
    os.replace(meta_tmp, meta_path)
    return csv_path, meta, True
