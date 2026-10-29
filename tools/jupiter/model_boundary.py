"""Measured two-stage Jupiter Model Partition selector.

For PC-prefix/Phone-suffix execution, Eq. (1) reduces to a search over the
measured contiguous boundary: min_p max(T_PC(0:p), T_PHONE(p:L)).  Do not
invent unmeasured per-layer timings from differences between whole runs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
import statistics
from typing import Sequence


@dataclass(frozen=True)
class BoundaryChoice:
    pc_layers: int
    phone_layers: int
    pc_ms: float
    phone_ms: float
    pc_memory_mib: float | None
    phone_memory_mib: float | None
    samples: int

    @property
    def bottleneck_ms(self) -> float:
        return max(self.pc_ms, self.phone_ms)


def _finite_nonnegative(raw: str, field: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {field}={raw}")
    return value


def choose_boundary(
    rows: Sequence[dict[str, str]],
    metric: str = "compute_ms",
    pc_budget_mib: float | None = None,
    phone_budget_mib: float | None = None,
) -> tuple[BoundaryChoice, list[BoundaryChoice], int]:
    """Select the measured feasible minimum-bottleneck boundary.

    Each row is an independently executed synchronized profiling run.  Per
    boundary, use median stage service totals, not the sum of concurrent
    wall times.  Missing memory observations cannot pass an explicit budget.
    """
    if metric not in ("compute_ms", "wall_ms"):
        raise ValueError("metric must be compute_ms or wall_ms")
    for name, value in (("PC", pc_budget_mib), ("PHONE", phone_budget_mib)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"invalid {name} memory budget")

    groups: dict[tuple[int, int], list[dict[str, str]]] = {}
    failures = 0
    for row in rows:
        if row.get("status") != "OK":
            failures += 1
            continue
        pc_layers = int(row["pc_layers"])
        phone_layers = int(row["phone_layers"])
        if pc_layers <= 0 or phone_layers <= 0:
            raise ValueError("both pipeline devices must have at least one layer")
        groups.setdefault((pc_layers, phone_layers), []).append(row)

    total_layers = {pc + phone for pc, phone in groups}
    if len(total_layers) > 1:
        raise ValueError("model boundary profiles contain inconsistent layer counts")

    all_choices: list[BoundaryChoice] = []
    for (pc_layers, phone_layers), samples in sorted(groups.items()):
        def median_value(column: str) -> float:
            return statistics.median(_finite_nonnegative(s[column], column) for s in samples)

        def median_memory(column: str) -> float | None:
            values = [s.get(column, "") for s in samples]
            if not all(v is not None and str(v).strip() for v in values):
                return None
            return statistics.median(_finite_nonnegative(str(v), column) for v in values)

        choice = BoundaryChoice(
            pc_layers, phone_layers,
            median_value(f"pc_{metric}"),
            median_value(f"phone_{metric}"),
            median_memory("pc_memory_mib"),
            median_memory("phone_memory_mib"),
            len(samples),
        )
        all_choices.append(choice)

    # It is unsafe to infer a partition's memory from another boundary: do
    # not count candidates with missing memory as fitting a hard budget.
    feasible = [c for c in all_choices if
                (pc_budget_mib is None or (c.pc_memory_mib is not None and c.pc_memory_mib <= pc_budget_mib))
                and (phone_budget_mib is None or (c.phone_memory_mib is not None and c.phone_memory_mib <= phone_budget_mib))]
    if not feasible:
        raise ValueError("no measured feasible model boundary under the supplied memory budgets")
    best = min(feasible, key=lambda c: (c.bottleneck_ms, c.pc_ms + c.phone_ms, c.pc_layers))
    return best, feasible, failures


def load_boundary_csv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("model boundary profile has no rows")
    return rows
