#!/usr/bin/env python3
"""
Jupiter offline planner for the llama.cpp adaptation.

Implements the prefill planning equations from:
  Ye et al., "Jupiter: Fast and Resource-Efficient Collaborative Inference
  of Generative LLMs on Edge Devices", INFOCOM 2025.

Sequence partition:
  W(1->y,k) = min_l max(W(1->l,k-1), T*(y-l,l))
  T*(x,y)   = inf if x < b else q(x,y)
  Latency   = sum_i h_i + (|D|-1) * W(1->y,k)

Model partition:
  A(1->y,D_n) = min_l max(A(1->l,D_{n-1}), T(l+1->y,d_n))

The output is a small text plan consumed at runtime via:
  LLAMA_JUPITER_PLAN_FILE=<path>
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from functools import lru_cache
import math
import os
import re
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


INF = float("inf")


def _open_text_auto(path: str):
    # Windows PowerShell 5.x Tee-Object commonly writes UTF-16LE with BOM,
    # while PowerShell 7 and Unix shells generally produce UTF-8. Detect the
    # BOM instead of assuming UTF-8 so calibration logs work on both.
    with open(path, "rb") as f:
        head = f.read(4)

    if head.startswith(b"\\xff\\xfe") or head.startswith(b"\\xfe\\xff"):
        encoding = "utf-16"
    elif head.startswith(b"\\xef\\xbb\\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"

    return open(path, "r", encoding=encoding, errors="replace")


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _load_existing_plan(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _write_plan(path: str, updates: Dict[str, str]) -> None:
    data = _load_existing_plan(path)
    data.update(updates)

    model_keys = sorted(k for k in data if k.startswith("model."))
    seq_keys = sorted(
        (k for k in data if k.startswith("sequence.")),
        key=lambda k: int(k.split(".", 1)[1]),
    )
    other_keys = sorted(k for k in data if k not in set(model_keys + seq_keys))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Jupiter INFOCOM'25 offline plan\n")
        for k in model_keys + seq_keys + other_keys:
            f.write(f"{k}={data[k]}\n")


class QProfile:
    """Median-per-device stage profile with conservative prefix interpolation.

    Jupiter sweeps measure several *fixed chunk sizes*. A prefix may therefore
    have only one observed chunk size. Interpolating x within that prefix would
    silently treat a 48-token observation as the cost of a 128-token chunk.
    First interpolate the prefix axis for the SAME x, then interpolate x.
    Unobserved points remain estimates: use candidate-specific calibration
    before interpreting a DP plan as measured.
    """

    def __init__(self, rows: Sequence[dict], metric: str = "latency_ms"):
        grouped_stage: Dict[Tuple[int, int, str], List[float]] = {}
        for row in rows:
            x = int(row["x"])
            y = int(row["y"])
            if x <= 0 or y < 0:
                raise ValueError(f"invalid q coordinate x={x} y={y}")
            if metric not in row or row[metric] in ("", None):
                raise ValueError(f"q profile is missing metric {metric!r}")
            t = float(row[metric])
            if not math.isfinite(t) or t < 0:
                raise ValueError(f"invalid {metric}={t} for q({x},{y})")
            stage = (row.get("device") or row.get("stage") or "stage").strip()
            grouped_stage.setdefault((x, y, stage), []).append(t)

        if not grouped_stage:
            raise ValueError("q profile is empty")

        per_xy: Dict[Tuple[int, int], List[float]] = {}
        for (x, y, _stage), samples in grouped_stage.items():
            per_xy.setdefault((x, y), []).append(statistics.median(samples))

        self.values = {key: max(timings) for key, timings in per_xy.items()}
        self.ys_by_x: Dict[int, List[int]] = {}
        for x, y in self.values:
            self.ys_by_x.setdefault(x, []).append(y)
        for x in self.ys_by_x:
            self.ys_by_x[x] = sorted(set(self.ys_by_x[x]))
        self.xs = sorted(self.ys_by_x)

    def is_measured(self, x: int, y: int) -> bool:
        """True only for a calibrated (x, prefix) pair, not an interpolation."""
        return (x, y) in self.values

    def sample_status(self, x: int, y: int) -> str:
        if self.is_measured(x, y):
            return "MEASURED"
        if x in self.ys_by_x:
            ys = self.ys_by_x[x]
            return "INTERP_Y" if ys[0] <= y <= ys[-1] else "EXTRAP_Y"
        return "INTERP_XY" if self.xs[0] < x < self.xs[-1] else "EXTRAP_X"

    @lru_cache(maxsize=131072)
    def _interp_y_at_x(self, x: int, y: int) -> float:
        """Interpolate prefix with chunk size fixed (never borrow a smaller x)."""
        ys = self.ys_by_x[x]
        if y <= ys[0]:
            # Do not extrapolate downwards to an artificially cheaper stage.
            return self.values[(x, ys[0])]
        if y >= ys[-1]:
            last = ys[-1]
            value = self.values[(x, last)]
            if len(ys) < 2:
                return value
            before = ys[-2]
            # Conservative non-negative prefix extrapolation; an increasing
            # observed cost is never replaced by a lower one past last.
            slope = max(0.0, (value - self.values[(x, before)]) / (last - before))
            return value + (y - last) * slope

        index = bisect_left(ys, y)
        if ys[index] == y:
            return self.values[(x, y)]
        left, right = ys[index - 1], ys[index]
        left_t = self.values[(x, left)]
        right_t = self.values[(x, right)]
        fraction = (y - left) / (right - left)
        return left_t + fraction * (right_t - left_t)

    @lru_cache(maxsize=131072)
    def q(self, x: int, y: int) -> float:
        if x <= 0 or y < 0:
            raise ValueError(f"invalid q coordinate x={x} y={y}")
        if (x, y) in self.values:
            return self.values[(x, y)]
        if x in self.ys_by_x:
            return self._interp_y_at_x(x, y)

        index = bisect_left(self.xs, x)
        if index == 0:
            # A shorter chunk than the smallest measured x may be cheaper,
            # but a flat upper bound is safer than an optimistic extrapolation.
            return self._interp_y_at_x(self.xs[0], y)
        if index == len(self.xs):
            # No measured x above the query; do not assume the last measured
            # chunk has constant cost as x increases.
            last_x = self.xs[-1]
            return self._interp_y_at_x(last_x, y) * (x / last_x)

        lower_x, upper_x = self.xs[index - 1], self.xs[index]
        lower = self._interp_y_at_x(lower_x, y)
        upper = self._interp_y_at_x(upper_x, y)
        return lower + (x - lower_x) * (upper - lower) / (upper_x - lower_x)


def sequence_points(chunks: Sequence[int]) -> List[Tuple[int, int]]:
    """Return (chunk_size, prefix) coordinates of a sequence partition."""
    points: List[Tuple[int, int]] = []
    prefix = 0
    for x in chunks:
        if x <= 0:
            raise ValueError("sequence chunks must be positive")
        points.append((x, prefix))
        prefix += x
    return points


def sequence_missing_points(profile: QProfile, chunks: Sequence[int]) -> List[Tuple[int, int]]:
    return [(x, y) for x, y in sequence_points(chunks) if not profile.is_measured(x, y)]


@dataclass
class SequenceSolution:
    total: int
    k: int
    chunks: List[int]
    bottleneck_ms: float
    predicted_ms: float


def solve_sequence(
    profile: QProfile,
    total: int,
    devices: int,
    min_chunk: int,
    max_chunk: int | None,
    max_k: int | None,
) -> SequenceSolution:
    if total <= 0:
        raise ValueError("total must be positive")
    if devices <= 0:
        raise ValueError("devices must be positive")

    max_k = max_k or 4 * devices
    max_k = min(max_k, total // min_chunk)
    if max_k <= 0:
        raise ValueError("no feasible partition: total < min_chunk")

    # W[y][k] and predecessor cut l.
    W = [[INF] * (max_k + 1) for _ in range(total + 1)]
    cut = [[-1] * (max_k + 1) for _ in range(total + 1)]

    for y in range(1, total + 1):
        if y >= min_chunk and (max_chunk is None or y <= max_chunk):
            W[y][1] = profile.q(y, 0)
            cut[y][1] = 0

    for k in range(2, max_k + 1):
        for y in range(1, total + 1):
            # Need at least min_chunk tokens in all k pieces.
            l_min = (k - 1) * min_chunk
            l_max = y - min_chunk
            if l_min > l_max:
                continue

            for l in range(l_min, l_max + 1):
                x = y - l
                if max_chunk is not None and x > max_chunk:
                    continue
                prev = W[l][k - 1]
                if not math.isfinite(prev):
                    continue
                cur = max(prev, profile.q(x, l))
                if cur < W[y][k]:
                    W[y][k] = cur
                    cut[y][k] = l

    def backtrack(k: int) -> List[int]:
        chunks: List[int] = []
        y = total
        kk = k
        while kk > 1:
            l = cut[y][kk]
            if l < 0:
                return []
            chunks.append(y - l)
            y = l
            kk -= 1
        if cut[y][1] != 0:
            return []
        chunks.append(y)
        chunks.reverse()
        return chunks

    best: SequenceSolution | None = None
    for k in range(1, max_k + 1):
        if not math.isfinite(W[total][k]):
            continue
        chunks = backtrack(k)
        if not chunks:
            continue

        prefix = 0
        sum_h = 0.0
        for x in chunks:
            sum_h += profile.q(x, prefix)
            prefix += x

        predicted = sum_h + (devices - 1) * W[total][k]
        candidate = SequenceSolution(total, k, chunks, W[total][k], predicted)
        if best is None or candidate.predicted_ms < best.predicted_ms:
            best = candidate

    if best is None:
        raise ValueError("no feasible sequence partition")
    return best


def cmd_sequence(args: argparse.Namespace) -> None:
    profile = QProfile(_read_csv(args.profile), metric=args.metric)
    lengths: Iterable[int]
    if args.target_length is not None:
        lengths = [args.target_length]
    else:
        lengths = range(args.min_total or args.min_chunk, args.smax + 1)

    updates: Dict[str, str] = {}
    for total in lengths:
        try:
            sol = solve_sequence(
                profile,
                total=total,
                devices=args.devices,
                min_chunk=args.min_chunk,
                max_chunk=args.max_chunk,
                max_k=args.max_subsequences,
            )
        except ValueError:
            continue

        missing = sequence_missing_points(profile, sol.chunks)
        print(
            f"[JUPITER_SEQUENCE_COVERAGE] total={total} "
            f"metric={args.metric} measured={len(sol.chunks) - len(missing)} "
            f"chunks={len(sol.chunks)} missing="
            f"{';'.join(f'{x}:{y}' for x, y in missing) or 'NONE'} "
            f"status={'MEASURED' if not missing else 'NEEDS_TARGETED_Q'}"
        )
        if missing and args.require_measured:
            raise SystemExit(
                "DP candidate uses unmeasured q positions; run "
                "refine_sequence_q.py before accepting the plan"
            )
        updates[f"sequence.{total}"] = ",".join(map(str, sol.chunks))
        print(
            f"[JUPITER_SEQUENCE_DP] total={sol.total} k={sol.k} "
            f"chunks={sol.chunks} bottleneck_ms={sol.bottleneck_ms:.3f} "
            f"predicted_ms={sol.predicted_ms:.3f} metric={args.metric}"
        )

    if not updates:
        raise SystemExit("no feasible plans were generated")
    _write_plan(args.output, updates)


def cmd_extract_q(args: argparse.Namespace) -> None:
    begin_re = re.compile(
        r"\[JUPITER_UBATCH\] event=BEGIN .*?prefix_tokens=(\d+) tokens=(\d+)"
    )
    queued_re = re.compile(r"\[JUPITER_UBATCH\] event=QUEUED")
    stage_re = re.compile(
        r"\[JUPITER_STAGE_PROFILE\] .*?backend=([^ ]+) .*?"
        r"input_ms=([0-9.]+) compute_ms=([0-9.]+) wall_ms=([0-9.]+)"
    )

    rows: List[dict] = []
    current: Tuple[int, int] | None = None
    stage_sums: Dict[str, Dict[str, float]] = {}
    begin_count = 0
    stage_count = 0
    queued_count = 0

    def flush() -> None:
        nonlocal current, stage_sums
        if current is None:
            return
        y, x = current
        if x < args.min_tokens:
            current = None
            stage_sums = {}
            return

        metric = args.metric
        for backend, values in sorted(stage_sums.items()):
            latency = values[metric]
            rows.append({
                "x": x,
                "y": y,
                "latency_ms": latency,
                "device": backend,
                "input_ms": values["input_ms"],
                "compute_ms": values["compute_ms"],
                "wall_ms": values["wall_ms"],
            })
        current = None
        stage_sums = {}

    with _open_text_auto(args.log) as f:
        for line in f:
            m = begin_re.search(line)
            if m:
                begin_count += 1
                flush()
                current = (int(m.group(1)), int(m.group(2)))
                stage_sums = {}
                continue

            if current is None:
                continue

            m = stage_re.search(line)
            if m:
                stage_count += 1
                backend = m.group(1)
                values = stage_sums.setdefault(
                    backend,
                    {"input_ms": 0.0, "compute_ms": 0.0, "wall_ms": 0.0},
                )
                values["input_ms"] += float(m.group(2))
                values["compute_ms"] += float(m.group(3))
                values["wall_ms"] += float(m.group(4))
                continue

            if queued_re.search(line):
                queued_count += 1
                flush()

    flush()

    if not rows:
        raise SystemExit(
            "no q samples found; "
            f"matched BEGIN={begin_count} STAGE_PROFILE={stage_count} QUEUED={queued_count}. "
            "If all counts are zero, update jupiter_planner.py to the latest branch "
            "version or convert the PowerShell Tee-Object log to UTF-8."
        )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "x", "y", "latency_ms", "device",
                "input_ms", "compute_ms", "wall_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[JUPITER_Q_EXTRACT] samples={len(rows)} metric={args.metric} "
        f"output={args.output}"
    )


@dataclass
class LayerSample:
    time_ms: float
    memory_bytes: int


def cmd_model(args: argparse.Namespace) -> None:
    rows = _read_csv(args.profile)
    devices = [x.strip() for x in args.device_order.split(",") if x.strip()]
    if len(devices) < 2:
        raise SystemExit("--device-order must contain at least two devices")

    samples: Dict[str, Dict[int, LayerSample]] = {d: {} for d in devices}
    for row in rows:
        dev = row["device"]
        if dev not in samples:
            continue
        layer = int(row["layer"])
        samples[dev][layer] = LayerSample(
            time_ms=float(row["time_ms"]),
            memory_bytes=int(float(row.get("memory_bytes") or 0)),
        )

    common = set(samples[devices[0]])
    for dev in devices[1:]:
        common &= set(samples[dev])
    layers = sorted(common)
    if not layers or layers != list(range(layers[0], layers[-1] + 1)):
        raise SystemExit("profiles must contain the same contiguous layer range for every device")

    n_layers = len(layers)
    base_layer = layers[0]

    budgets: Dict[str, int] = {}
    for item in args.memory_budget_mib or []:
        dev, value = item.split(":", 1)
        budgets[dev] = int(float(value) * 1024 * 1024)

    # Prefix sums per device for O(1) segment cost.
    time_prefix: Dict[str, List[float]] = {}
    mem_prefix: Dict[str, List[int]] = {}
    for dev in devices:
        tp = [0.0]
        mp = [0]
        for layer in layers:
            sample = samples[dev][layer]
            tp.append(tp[-1] + sample.time_ms)
            mp.append(mp[-1] + sample.memory_bytes)
        time_prefix[dev] = tp
        mem_prefix[dev] = mp

    def seg_cost(dev: str, i: int, j: int) -> float:
        # [i,j) in local layer indices.
        mem = mem_prefix[dev][j] - mem_prefix[dev][i]
        if dev in budgets and mem > budgets[dev]:
            return INF
        return time_prefix[dev][j] - time_prefix[dev][i]

    ndev = len(devices)
    A = [[INF] * (n_layers + 1) for _ in range(ndev + 1)]
    cut = [[-1] * (n_layers + 1) for _ in range(ndev + 1)]
    A[0][0] = 0.0

    for n in range(1, ndev + 1):
        dev = devices[n - 1]
        for y in range(1, n_layers + 1):
            for l in range(0, y):
                if not math.isfinite(A[n - 1][l]):
                    continue
                t = seg_cost(dev, l, y)
                cur = max(A[n - 1][l], t)
                if cur < A[n][y]:
                    A[n][y] = cur
                    cut[n][y] = l

    if not math.isfinite(A[ndev][n_layers]):
        raise SystemExit("no feasible model partition under the supplied memory budgets")

    bounds: List[Tuple[int, int]] = []
    y = n_layers
    for n in range(ndev, 0, -1):
        l = cut[n][y]
        if l < 0:
            raise SystemExit("internal DP backtracking failure")
        bounds.append((l, y))
        y = l
    bounds.reverse()

    pretty = []
    for dev, (l, r) in zip(devices, bounds):
        pretty.append(f"{dev}:{base_layer + l}-{base_layer + r - 1}")

    print(
        f"[JUPITER_MODEL_DP] bottleneck_ms={A[ndev][n_layers]:.3f} "
        f"partition={' '.join(pretty)}"
    )

    # Current llama.cpp Jupiter runtime uses exactly two ordered stages and a
    # contiguous repeating-layer boundary.
    updates: Dict[str, str] = {}
    if len(devices) == 2 and base_layer == 0:
        pc_layers = bounds[0][1]
        updates["model.pc_layers"] = str(pc_layers)
        print(f"[JUPITER_MODEL_DP] runtime model.pc_layers={pc_layers}")

    if updates:
        _write_plan(args.output, updates)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Jupiter INFOCOM'25 offline planner")
    sub = p.add_subparsers(dest="cmd", required=True)

    seq = sub.add_parser("sequence", help="Eq.(2)-(4) sequence partition DP")
    seq.add_argument("--profile", required=True, help="CSV: x,y,latency_ms[,device/stage]")
    seq.add_argument("--output", required=True, help="runtime plan file")
    seq.add_argument("--devices", type=int, required=True)
    seq.add_argument("--min-chunk", type=int, required=True, dest="min_chunk")
    seq.add_argument("--max-chunk", type=int, default=None, dest="max_chunk")
    seq.add_argument(
        "--metric", choices=["latency_ms", "compute_ms", "wall_ms"],
        default="latency_ms",
        help="q cost column (latency_ms is the original extracted metric)",
    )
    seq.add_argument(
        "--require-measured", action="store_true",
        help="refuse DP candidates with unmeasured (x,prefix) positions",
    )
    seq.add_argument("--max-subsequences", type=int, default=None, dest="max_subsequences")
    seq.add_argument("--target-length", type=int, default=None, dest="target_length")
    seq.add_argument("--smax", type=int, default=4096)
    seq.add_argument("--min-total", type=int, default=None, dest="min_total")
    seq.set_defaults(func=cmd_sequence)

    extract = sub.add_parser(
        "extract-q",
        help="extract q(x,y) stage samples from a synchronized calibration log",
    )
    extract.add_argument("--log", required=True, help="PC llama-cli calibration log")
    extract.add_argument("--output", required=True, help="q-profile CSV")
    extract.add_argument(
        "--metric",
        choices=["compute_ms", "wall_ms"],
        default="compute_ms",
        help="paper q uses stage inference time; wall_ms additionally includes input handling",
    )
    extract.add_argument("--min-tokens", type=int, default=2, dest="min_tokens")
    extract.set_defaults(func=cmd_extract_q)

    model = sub.add_parser("model", help="Eq.(1) model partition DP")
    model.add_argument("--profile", required=True, help="CSV: device,layer,time_ms[,memory_bytes]")
    model.add_argument("--output", required=True, help="runtime plan file")
    model.add_argument("--device-order", required=True, dest="device_order",
                       help="ordered pipeline devices, e.g. PC,PHONE")
    model.add_argument("--memory-budget-mib", action="append", default=[],
                       help="optional per-device budget, e.g. PC:5500")
    model.set_defaults(func=cmd_model)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
