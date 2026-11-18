#!/usr/bin/env python3
"""Fail-closed comparison of Prima rank-0 Profile and Normal layer allocation.

The [PRIMA_FIG3_LAYOUT] line is emitted AFTER model initialization on every
run by the instrumented llama-cli. Manual replay uses the native -lw/-ngl
interface (as documented by Prima); it is NOT a HiGHS auto-placement run.
"""
from __future__ import annotations

import re
from pathlib import Path

AUDIT = re.compile(
    r"\[PRIMA_FIG3_LAYOUT\]\s+stage=post_init\s+"
    r"rank=(\d+)\s+world=(\d+)\s+lw0=(\d+)\s+lw1=(\d+)\s+"
    r"ngl=(-?\d+)\s+prefetch=(\d+)"
)
PLAN = re.compile(
    r"-\s*Device Index\s*:\s*(\d+)\s*\n"
    r"\s*-\s*Assignment Set\s*:\s*(\S+)\s*\n"
    r"\s*-\s*N Layer Window\s*:\s*(\d+)\s*\n"
    r"\s*-\s*N GPU Layers\s*:\s*(\d+)",
    re.I,
)
FIELDS = ("rank", "world", "lw0", "lw1", "ngl", "prefetch")


def parse_fixed(value: str, n_model_layers=0):
    if not re.fullmatch(r"[1-9]\d*,[1-9]\d*", value):
        raise ValueError("--fixed-layers must be TWO positive integers, e.g. 39,1")
    pc, phone = (int(x) for x in value.split(","))
    if n_model_layers and pc + phone != n_model_layers:
        raise ValueError(
            f"--fixed-layers {value} total {pc+phone} != model layers {n_model_layers}"
        )
    return pc, phone


def summarize_log(path):
    content = Path(path).read_text(encoding="utf-8", errors="replace")
    raw = list(AUDIT.finditer(content))
    if len(raw) != 1:
        raise ValueError(
            f"{path}: expected exactly ONE [PRIMA_FIG3_LAYOUT] post-init audit "
            f"line, got {len(raw)}. Rebuild PC llama-cli from the Fig.3 branch."
        )
    values = [int(v) for v in raw[0].groups()]
    effective = dict(zip(FIELDS, values))
    stages = []
    current = {}
    for match in PLAN.finditer(content):
        rank = int(match[1])
        record = dict(rank=rank, assignment_set=match[2],
                      layers=int(match[3]), gpu_layers=int(match[4]))
        if current and (rank in current or rank < max(current)):
            stages.append([current[k] for k in sorted(current)])
            current = {}
        current[rank] = record
    if current:
        stages.append([current[k] for k in sorted(current)])
    return dict(effective=effective, planner_stages=stages)


def validate_layout(layout, expected=None):
    """Check actual post-init state, plus planner stages if emitted."""
    effective = layout["effective"]
    if effective["rank"] != 0:
        raise ValueError(f"expected PC rank 0, got {effective}")
    if expected is None:
        # In AUTO mode it is unsafe to combine timing from different
        # planning stages unless the source proves which stage performed
        # Prefill. Reject any observed reassignment (including PC-only).
        stages = layout["planner_stages"]
        signatures = []
        for stage in stages:
            signatures.append(tuple(sorted(
                (row["rank"], row["layers"], row["gpu_layers"])
                for row in stage
            )))
        if len(set(signatures)) > 1:
            raise ValueError(
                "Prima auto-plan REPLANNED within this session: "
                f"{signatures}. Use fixed replay or inspect the exact "
                "Prefill-stage allocation before reporting Fig.3."
            )
    if expected is not None:
        pc, phone, ngl = expected
        required = dict(rank=0, world=2, lw0=pc, lw1=phone,
                        ngl=ngl, prefetch=1)
        mismatched = {k: (effective[k], v) for k, v in required.items()
                      if effective[k] != v}
        if mismatched:
            raise ValueError(
                f"effective Prima allocation differs from fixed replay "
                f"{expected}: {mismatched}; full post-init: {effective}"
            )
        for stage in layout["planner_stages"]:
            by_rank = {x["rank"]: x for x in stage}
            if not by_rank:
                continue
            if len(by_rank) == 1 and 0 in by_rank and by_rank[0]["layers"] == pc+phone:
                raise ValueError("planner removed Phone and assigned all layers to PC")
            if 0 in by_rank and (
                by_rank[0]["layers"] != pc or by_rank[0]["gpu_layers"] != ngl
            ):
                raise ValueError("planner stage contradicts requested PC layout")
            if 1 in by_rank and (
                by_rank[1]["layers"] != phone or by_rank[1]["gpu_layers"] != 0
            ):
                raise ValueError("planner stage contradicts requested Phone layout")
    return effective


def compare_runs(profile, normals):
    """Require identical effective placement in all independently run CLIs."""
    reference = profile["layout"]["effective"]
    for i, normal in enumerate(normals, 1):
        observed = normal["layout"]["effective"]
        if observed != reference:
            mismatched = {k: (reference[k], observed[k]) for k in FIELDS
                          if reference[k] != observed[k]}
            raise ValueError(
                f"Prima Profile/Normal {i} LAYOUT_MISMATCH: {mismatched}. "
                "Do not combine their Prefill timings in Fig.3. "
                "Re-run with --fixed-layers and --pc-ngl."
            )
    return reference
