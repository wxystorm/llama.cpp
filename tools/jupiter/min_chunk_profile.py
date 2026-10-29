"""Select Jupiter's minimum sequence chunk from measured stage throughput.

The INFOCOM'25 procedure uses accelerator utilization. Our llama.cpp
adaptation measures synchronized stage compute time as a throughput-saturation
proxy, not a hardware GPU/NPU occupancy counter.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import math
import statistics
from typing import Sequence


@dataclass(frozen=True)
class ChunkEfficiency:
    tokens: int
    pc_ms: float
    phone_ms: float
    pc_tokens_per_s: float
    phone_tokens_per_s: float
    samples: int
    pc_relative: float = 0.0
    phone_relative: float = 0.0

    @property
    def minimum_relative(self) -> float:
        return min(self.pc_relative, self.phone_relative)


def _positive_number(row: dict, field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be finite and positive: {value}")
    return value


def partition_feasible(total_tokens: int, min_chunk: int, max_chunk: int) -> bool:
    """Check if total can be tiled by integer chunks in [min_chunk, max_chunk]."""
    if total_tokens <= 0 or min_chunk <= 0 or max_chunk < min_chunk:
        return False
    smallest_k = (total_tokens + max_chunk - 1) // max_chunk
    largest_k = total_tokens // min_chunk
    return smallest_k <= largest_k


def select_min_chunk(
    rows: Sequence[dict],
    threshold: float = 0.90,
    min_samples: int = 1,
    total_tokens: int | None = None,
    max_chunk: int | None = None,
) -> tuple[ChunkEfficiency, list[ChunkEfficiency], str]:
    """Pick the smallest chunk where both measured stages have saturated.

    Rates are normalized independently against the maximum median throughput
    observed on each stage. If both stages never meet the threshold at the
    same chunk size, select the best max-min relative throughput instead,
    explicitly flagging that the desired common plateau was NOT observed.

    Use complete synchronized stage measurements at the same prefix (y=0);
    no network transfer or graph-preparation costs are included in the
    throughput proxy. The selected size is a planning heuristic, not a
    measurement of device occupancy. When total_tokens/max_chunk are supplied,
    the full sequence must also admit a strict [b,max_chunk] partition.
    """
    if not 0.0 < threshold <= 1.0 or not math.isfinite(threshold):
        raise ValueError("threshold must be finite and within (0,1]")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")
    if (total_tokens is None) != (max_chunk is None):
        raise ValueError("total_tokens and max_chunk must be supplied together")
    if total_tokens is not None and (total_tokens <= 0 or max_chunk <= 0):
        raise ValueError("total_tokens and max_chunk must be positive")

    by_size: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("status") != "OK":
            continue
        tokens = int(row["chunk_tokens"])
        if tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        by_size.setdefault(tokens, []).append(row)

    measured = []
    for tokens, group in sorted(by_size.items()):
        if len(group) < min_samples:
            continue
        pc_ms = statistics.median(_positive_number(row, "pc_compute_ms") for row in group)
        phone_ms = statistics.median(_positive_number(row, "phone_compute_ms") for row in group)
        measured.append(ChunkEfficiency(
            tokens=tokens,
            pc_ms=pc_ms, phone_ms=phone_ms,
            pc_tokens_per_s=1000.0 * tokens / pc_ms,
            phone_tokens_per_s=1000.0 * tokens / phone_ms,
            samples=len(group),
        ))

    if not measured:
        raise ValueError("no complete PC/Phone chunk measurements")
    peak_pc = max(item.pc_tokens_per_s for item in measured)
    peak_phone = max(item.phone_tokens_per_s for item in measured)
    normalized = [
        replace(item,
                pc_relative=item.pc_tokens_per_s / peak_pc,
                phone_relative=item.phone_tokens_per_s / peak_phone)
        for item in measured
    ]

    feasible = [
        item for item in normalized
        if total_tokens is None or
        partition_feasible(total_tokens, item.tokens, max_chunk)
    ]
    if not feasible:
        raise ValueError(
            "no measured minimum b permits a legal full-prompt partition "
            f"(total={total_tokens}, max_chunk={max_chunk})"
        )

    eligible = [item for item in feasible
                if item.pc_relative + 1e-12 >= threshold
                and item.phone_relative + 1e-12 >= threshold]
    if eligible:
        return eligible[0], normalized, "BOTH_STAGES_SATURATED"

    # The saturating candidate may be impossible to tile with the current
    # prompt and ubatch cap (e.g. b=128, total=357, ubatch=128).
    # Report a feasible, explicitly below-threshold fallback; never emit
    # an unexecutable DP constraint.
    fallback = max(feasible, key=lambda item: (item.minimum_relative, -item.tokens))
    unconstrained_saturated = any(
        item.pc_relative + 1e-12 >= threshold and
        item.phone_relative + 1e-12 >= threshold for item in normalized
    )
    status = "NO_FEASIBLE_PLATEAU" if unconstrained_saturated else "NO_COMMON_PLATEAU"
    return fallback, normalized, status


def read_sample_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as stream:
        data = list(csv.DictReader(stream))
    if not data:
        raise ValueError("minimum chunk sample CSV is empty")
    return data
