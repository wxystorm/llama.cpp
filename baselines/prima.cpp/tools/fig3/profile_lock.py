#!/usr/bin/env python3
"""Safely replay a saved automatic Prima Profile layout for Normal sessions.

This does NOT turn the Profile into a manually planned run. It keeps the
original automatic first session and explicitly labels subsequent Normal
sessions as PROFILE_LOCKED_REPLAY. Do not claim this is re-planned HiGHS
automatic performance.
"""
from __future__ import annotations

import json
from pathlib import Path

from layout_guard import local_layer_counts, parse_fixed, validate_layout


def locked_normal_command(manifest, initial_command, model_layers):
    if not manifest.get("profile"):
        raise ValueError("No completed Profile to lock; Profile must run first")
    if manifest.get("full_command") != initial_command:
        raise ValueError("Profile command differs from requested resume command")
    layout = manifest["profile"].get("layout")
    if not layout or "effective" not in layout:
        raise ValueError("Saved Profile has no post-init layout audit")
    e = layout["effective"]
    if e.get("rank") != 0 or e.get("world") != 2 or e.get("prefetch") != 1:
        raise ValueError(f"Profile is not a two-rank prefetch run: {e}")
    pc, phone, ngl = int(e["lw0"]), int(e["lw1"]), int(e["ngl"])
    parse_fixed(f"{pc},{phone}", model_layers)
    if ngl < 0 or (model_layers and ngl > local_layer_counts(pc, phone, model_layers)[0]):
        raise ValueError(f"Invalid Profile GPU layer count {ngl} for window {pc},{phone}")
    expected = (pc, phone, ngl)
    # Reject conflicting repeated planner stages before freezing.
    validate_layout(layout, expected)
    return initial_command + ["-lw", f"{pc},{phone}", "-ngl", str(ngl)], expected


def quarantine_mismatched_runs(manifest, root, reference):
    """Archive incompatible completed runs; preserve all source evidence.

    A previous version of the collector wrote a mismatched Warmup into
    manifest BEFORE comparing it with Profile. Keep it in rejected_runs
    with its original log file, instead of trusting it or overwriting it.
    If a mismatch occurs, all subsequent runs in that group are also
    archived, because the remaining sequence must be contiguous.
    """
    root = Path(root)
    rejected = []
    for key in ("warmup", "normal"):
        entries = list(manifest.get(key, []))
        kept = []
        rejecting = False
        for index, entry in enumerate(entries, 1):
            effective = (entry.get("layout") or {}).get("effective")
            if effective != reference:
                rejecting = True
            if not rejecting:
                kept.append(entry)
                continue
            old_log = root/f"{key}_{index}.log"
            new_name = None
            if old_log.exists():
                out_dir = root/"rejected_runs"
                out_dir.mkdir(parents=True, exist_ok=True)
                target = out_dir/old_log.name
                if target.exists():
                    raise FileExistsError(
                        f"Cannot archive {old_log}: {target} already exists; "
                        "inspect rejected_runs before retrying."
                    )
                old_log.rename(target)
                new_name = str(target.relative_to(root))
            reason = ("LAYOUT_MISMATCH" if effective != reference
                      else "AFTER_EARLIER_LAYOUT_MISMATCH")
            rejected.append(dict(group=key, index=index, reason=reason,
                                 log=new_name, recorded=entry))
        manifest[key] = kept
    if rejected:
        manifest.setdefault("rejected_sessions", []).extend(rejected)
    return rejected
