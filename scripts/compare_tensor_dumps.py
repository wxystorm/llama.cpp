#!/usr/bin/env python3
"""Compare CPU and RPC/NPU tensor snapshots emitted by LLAMA_TENSOR_COMPARE_DIR.

Python standard library only; supports contiguous GGML f32/f16 tensors.
Use identical model, prompt, context, ubatch and attention settings for both
runs. Prefer -n 1 so generated-token divergence cannot affect later inputs.
"""

import argparse
from array import array
from collections import defaultdict
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import struct
import sys


@dataclass(frozen=True)
class Snapshot:
    folder: Path
    capture_id: int
    name: str
    shape: tuple
    occurrence: int
    dtype: str
    op: str
    filename: str
    nbytes: int

    @property
    def key(self):
        return (self.name, self.shape, self.occurrence)


def load_manifest(folder):
    manifest = folder / "manifest.tsv"
    if not manifest.is_file():
        raise FileNotFoundError(f"not found: {manifest}")
    counts = defaultdict(int)
    snapshots = {}
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            shape = tuple(int(row[f"ne{i}"]) for i in range(4))
            name = row["name"]
            base = (name, shape)
            occurrence = counts[base]
            counts[base] += 1
            snap = Snapshot(
                folder=folder, capture_id=int(row["id"]), name=name, shape=shape,
                occurrence=occurrence, dtype=row["type"], op=row["op"],
                filename=row["file"], nbytes=int(row["bytes"]),
            )
            snapshots[snap.key] = snap
    return snapshots


def read_values(snap):
    data = (snap.folder / snap.filename).read_bytes()
    if len(data) != snap.nbytes:
        raise ValueError(f"{snap.filename}: expected {snap.nbytes} bytes, got {len(data)}")
    if snap.dtype == "f32":
        if len(data) % 4:
            raise ValueError(f"invalid f32 bytes: {snap.filename}")
        values = array("f")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        return values
    if snap.dtype == "f16":
        if len(data) % 2:
            raise ValueError(f"invalid f16 bytes: {snap.filename}")
        return [v[0] for v in struct.iter_unpack("<e", data)]
    raise ValueError(f"unsupported dtype {snap.dtype} in {snap.filename}")


STAGES = (
    "attn_norm", "Qcur_normed", "Kcur_normed", "Qcur", "Kcur", "Vcur",
    "kqv_out", "attn_out", "ffn_inp", "ffn_norm", "ffn_up",
    "ffn_gate", "ffn_swiglu", "ffn_down", "ffn_out", "l_out",
)
STAGE_ORDER = {name: i for i, name in enumerate(STAGES)}


def graph_order(key):
    name, shape, occurrence = key
    m = re.match(r"^(.*)-(\d+)$", name)
    if not m:
        return (10**6, 10**6, name, shape, occurrence)
    return (int(m.group(2)), STAGE_ORDER.get(m.group(1), 100),
            name, shape, occurrence)


def metrics(ref, actual, atol, rtol):
    if len(ref) != len(actual):
        raise ValueError(f"element count differs: CPU={len(ref)}, NPU={len(actual)}")
    count = len(ref)
    bad = 0
    nonfinite = 0
    max_abs = 0.0
    max_i = None
    first_i = None
    sum_abs = 0.0
    sum_sq = 0.0
    dot = 0.0
    norm_ref = 0.0
    norm_actual = 0.0

    for i, (a, b) in enumerate(zip(ref, actual)):
        if not math.isfinite(a) or not math.isfinite(b):
            nonfinite += 1
            bad += 1
            if first_i is None:
                first_i = i
            continue
        delta = abs(float(a) - float(b))
        sum_abs += delta
        sum_sq += delta * delta
        dot += float(a) * float(b)
        norm_ref += float(a) * float(a)
        norm_actual += float(b) * float(b)
        if delta > max_abs:
            max_abs = delta
            max_i = i
        if delta > atol + rtol * abs(float(a)):
            bad += 1
            if first_i is None:
                first_i = i

    valid = count - nonfinite
    cosine = dot / math.sqrt(norm_ref * norm_actual) if norm_ref and norm_actual else float("nan")
    return {
        "n": count,
        "bad": bad,
        "nonfinite": nonfinite,
        "max_abs": max_abs,
        "mae": sum_abs / valid if valid else float("nan"),
        "rmse": math.sqrt(sum_sq / valid) if valid else float("nan"),
        "cosine": cosine,
        "max_i": max_i,
        "first_i": first_i,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path, help="CPU snapshot directory")
    parser.add_argument("npu", type=Path, help="RPC/NPU snapshot directory")
    parser.add_argument("--atol", type=float, default=0.01, help="absolute tolerance (default: 0.01)")
    parser.add_argument("--rtol", type=float, default=0.01, help="relative tolerance (default: 0.01)")
    parser.add_argument("--limit", type=int, default=200, help="maximum tensor rows to print")
    args = parser.parse_args()

    cpu = load_manifest(args.cpu)
    npu = load_manifest(args.npu)
    # Preserve real execution order of the CPU baseline. Sorting by layer and
    # tensor shape incorrectly reports late 1-token decode divergence before
    # an earlier 2-token prefill error at a higher-numbered layer.
    shared = sorted(set(cpu) & set(npu), key=lambda key: cpu[key].capture_id)
    only_cpu = sorted(set(cpu) - set(npu), key=lambda key: cpu[key].capture_id)
    only_npu = sorted(set(npu) - set(cpu), key=lambda key: npu[key].capture_id)
    print(f"CPU tensors: {len(cpu)} | NPU tensors: {len(npu)} | comparable: {len(shared)}")
    print("Rows ordered by CPU manifest capture ID, not by layer/shape.")
    print(f"Only CPU: {len(only_cpu)} | Only NPU: {len(only_npu)}")
    if only_cpu:
        print("  CPU-only examples:", ", ".join(k[0] for k in only_cpu[:8]))
    if only_npu:
        print("  NPU-only examples:", ", ".join(k[0] for k in only_npu[:8]))
    if not shared:
        raise SystemExit("No comparable tensor names/shapes found; check the capture regex and input settings.")

    first_bad = None
    print("\nSTATUS   NAME                      SHAPE                 BAD/TOTAL   MAX_ABS      RMSE         COSINE")
    print("-" * 110)
    for index, key in enumerate(shared):
        if index >= args.limit:
            print(f"... {len(shared) - args.limit} additional tensor(s) omitted")
            break
        cs, ns = cpu[key], npu[key]
        ref, actual = read_values(cs), read_values(ns)
        stat = metrics(ref, actual, args.atol, args.rtol)
        is_bad = stat["bad"] > 0
        if is_bad and first_bad is None:
            first_bad = (cs, ns, ref, actual, stat)
        shape_str = "x".join(str(v) for v in cs.shape if v != 1) or "1"
        flag = "DIFF" if is_bad else "OK"
        print(f"{flag:<8} {cs.name:<25} {shape_str:<21} "
              f"{stat['bad']:>5}/{stat['n']:<7} "
              f"{stat['max_abs']:<12.5g} {stat['rmse']:<12.5g} {stat['cosine']:<.7g}")

    if first_bad is not None:
        cs, ns, ref, actual, stat = first_bad
        print(f"\nFirst tolerance mismatch in CPU capture order: {cs.name}")
        print(f"CPU capture ID: {cs.capture_id}, NPU capture ID: {ns.capture_id}")
        print(f"Shape: {cs.shape}, occurrence: {cs.occurrence}, CPU op: {cs.op}, NPU op: {ns.op}")
        print(f"Mismatch: {stat['bad']}/{stat['n']}, nonfinite: {stat['nonfinite']}, "
              f"max_abs: {stat['max_abs']:.7g}, MAE: {stat['mae']:.7g}, RMSE: {stat['rmse']:.7g}")
        if stat["first_i"] is not None:
            i = stat["first_i"]
            print(f"  first mismatching flat index {i}: CPU={ref[i]:.9g}, NPU={actual[i]:.9g}")
        if stat["max_i"] is not None:
            i = stat["max_i"]
            print(f"  largest error flat index {i}: CPU={ref[i]:.9g}, NPU={actual[i]:.9g}")
    else:
        print("\nAll compared tensors passed the requested tolerances.")
    print("\nNote: callback-based capture changes scheduler graph boundaries.")
    print("If the problem disappears with capture enabled, capture-free boundary probes are needed.")
    print("If generated token IDs differ between runs, later 1-token states cannot be compared directly.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc
