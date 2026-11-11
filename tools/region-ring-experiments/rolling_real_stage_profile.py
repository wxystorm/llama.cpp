#!/usr/bin/env python3
"""Rolling real-path Stage Profile snapshots for the verified 32B HTP-only runner.

The existing C++ APPLY_WIDE planner consumes the strict real-stage CSV.
Never synthesize HTP compute from PC RPC wall time. Without a matching Phone
log, only genuine CPU/GPU stage samples are refreshed; the native Phone probe
remains the fallback used by the existing C++ Planner.
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
    return cases


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
    if csv_path.is_file() and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        if meta.get("sha256") != actual or meta.get("epoch") != epoch:
            raise ValueError("Profile snapshot digest mismatch")
        return csv_path, meta, False

    cases = completed_cases(out)
    expected = every * epoch
    # The wrapper enters an epoch immediately after N completed prompts; a
    # crash and resume can only increase the number of completed cases.
    if len(cases) < expected:
        raise ValueError(f"Need {expected} PASS cases before profile epoch {epoch}")
    samples, diag = extract(cases, phone_log)
    points = aggregate(samples)
    if not points:
        raise ValueError(
            "No real Stage measurements; enable LLAMA_REGION_RING_TASK_TRACE "
            "and LLAMA_REGION_RING_REAL_STAGE_PROFILE")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = csv_path.with_name(csv_path.name + ".tmp")
    with csv_tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(points)
        f.flush()
        os.fsync(f.fileno())
    digest = hashlib.sha256(csv_tmp.read_bytes()).hexdigest()
    meta = {
        "schema": 1, "epoch": epoch, "profile_every": every,
        "passes_at_snapshot": len(cases), "source": "REAL_STAGE_CHUNK",
        "point_count": len(points), "device_points": {
            device: sum(p["device"] == device for p in points)
            for device in ("CPU", "GPU", "PHONE")
        }, "diagnostic": diag, "sha256": digest,
        "phone_log": str(phone_log) if phone_log else None,
        "fallback": "NATIVE_PHONE_PROFILE_WHEN_NO_MATCHED_PHONE_SAMPLES",
    }
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    os.replace(csv_tmp, csv_path)
    os.replace(meta_tmp, meta_path)
    return csv_path, meta, True
