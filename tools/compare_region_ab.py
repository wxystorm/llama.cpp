#!/usr/bin/env python3
"""
Compare the logical-element tensor dumps produced by GGML_META_AB_*.

Usage:
    python tools/compare_region_ab.py ab-A ab-B --tokens 358

The C++ diagnostic strips GGML tensor padding and view strides before
writing binary F32/I32 values. This program requires only Python stdlib.
"""
from __future__ import annotations

import argparse
from array import array
import math
from pathlib import Path
import re
import sys

FILE_RE = re.compile(
    r"^prefill_t(?P<tokens>\d+)_c(?P<chunk>full|\d+)_"
    r"(?P<side>pc|phone)_(?P<stage>[a-z0-9_]+)_"
    r"(?P<dtype>f32|i32)\.bin$"
)
STAGES = {
    "ffn_inp": 0,
    "ffn_norm": 1,
    "router_logits": 2,
    "router_topk": 3,
    "router_weights": 4,
    "partial": 5,
    "l_out": 6,
    "l47_input": 7,
}


def scan(directory: Path):
    if not directory.is_dir():
        raise SystemExit(f"Missing dump directory: {directory}")
    found = {}
    for path in directory.glob("prefill_t*_c*_*.bin"):
        m = FILE_RE.match(path.name)
        if m:
            key = (
                int(m.group("tokens")),
                m.group("chunk"),
                m.group("side"),
                m.group("stage"),
                m.group("dtype"),
            )
            if key in found:
                raise SystemExit(f"Duplicate dump: {path.name}")
            found[key] = path
    return found


def read_values(path: Path, dtype: str):
    values = array("f" if dtype == "f32" else "i")
    if values.itemsize != 4:
        raise SystemExit("This script requires 32-bit floats and integers")
    data = path.read_bytes()
    if len(data) % 4:
        raise SystemExit(f"Invalid dump byte count: {path}")
    values.frombytes(data)
    return values


def compare(a, b, dtype, atol, rtol):
    if len(a) != len(b):
        return {"status": "SHAPE", "count_a": len(a), "count_b": len(b)}
    n = len(a)
    worst_delta = 0.0
    worst_idx = 0
    diff2 = 0.0
    dot = 0.0
    aa = 0.0
    bb = 0.0
    mismatch = 0
    nonfinite = 0
    for idx, (av, bv) in enumerate(zip(a, b)):
        if dtype == "f32" and not (math.isfinite(av) and math.isfinite(bv)):
            nonfinite += 1
            if av != bv:
                mismatch += 1
            continue
        delta = abs(float(av) - float(bv))
        diff2 += delta * delta
        aa += float(av) * float(av)
        bb += float(bv) * float(bv)
        dot += float(av) * float(bv)
        if delta > worst_delta:
            worst_delta = delta
            worst_idx = idx
        if dtype == "i32":
            mismatch += av != bv
        else:
            mismatch += delta > atol + rtol * abs(float(bv))
    rms = math.sqrt(diff2 / n) if n else 0.0
    cosine = dot / math.sqrt(aa * bb) if aa > 0 and bb > 0 else float("nan")
    return {
        "status": "PASS" if mismatch == 0 and nonfinite == 0 else "DIFF",
        "n": n,
        "mismatch": mismatch,
        "nonfinite": nonfinite,
        "max_abs": worst_delta,
        "rms": rms,
        "cosine": cosine,
        "worst_idx": worst_idx,
        "a_value": a[worst_idx] if n else None,
        "b_value": b[worst_idx] if n else None,
    }


def display(key, result, label=""):
    tokens, chunk, side, stage, dtype = key
    prefix = f"t={tokens} chunk={chunk:>4} {side:>5} {stage:<15} {dtype}"
    if result["status"] == "SHAPE":
        print(
            f"{label}{prefix} SHAPE "
            f"A_n={result['count_a']} B_n={result['count_b']}"
        )
        return
    print(
        f"{label}{prefix} {result['status']:<4} "
        f"bad={result['mismatch']}/{result['n']} "
        f"max={result['max_abs']:.8g} rms={result['rms']:.8g} "
        f"cos={result['cosine']:.10f} "
        f"idx={result['worst_idx']} "
        f"A={result['a_value']} B={result['b_value']}"
    )


def stage_sort(key):
    tokens, chunk, side, stage, dtype = key
    chunk_num = -1 if chunk == "full" else int(chunk)
    return (
        STAGES.get(stage, 99),
        chunk_num,
        0 if side == "pc" else 1,
        dtype,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("a", type=Path, help="A: new C2P dump directory")
    p.add_argument("b", type=Path, help="B: correct legacy Tensor dump directory")
    p.add_argument("--tokens", type=int, default=None,
                   help="Select full prefill length; defaults to largest shared")
    p.add_argument("--atol", type=float, default=1e-3)
    p.add_argument("--rtol", type=float, default=1e-3)
    args = p.parse_args()

    a_all = scan(args.a)
    b_all = scan(args.b)
    shared_tokens = sorted(
        {k[0] for k in a_all} & {k[0] for k in b_all}
    )
    if not shared_tokens:
        raise SystemExit(
            "No common token count in A/B dumps. Check "
            "GGML_META_AB_DUMP_DIR, GGML_META_AB_MIN_TOKENS, "
            "and use the same input prompt."
        )
    tokens = args.tokens if args.tokens is not None else shared_tokens[-1]
    if tokens not in shared_tokens:
        raise SystemExit(
            f"Token count {tokens} not found in both A and B; "
            f"shared counts: {shared_tokens}"
        )
    a_files = {k: v for k, v in a_all.items() if k[0] == tokens}
    b_files = {k: v for k, v in b_all.items() if k[0] == tokens}
    common = sorted(set(a_files) & set(b_files), key=stage_sort)
    if not common:
        raise SystemExit("No matching stage keys found.")

    print(
        f"A={args.a} ({len(a_files)} dumps) "
        f"B={args.b} ({len(b_files)} dumps) tokens={tokens}"
    )
    print(f"float tolerance atol={args.atol:g} rtol={args.rtol:g}")
    print("Exact matches required for I32 TopK; F32 compares logical elements.")
    print()
    failures = []
    for key in common:
        dtype = key[-1]
        result = compare(
            read_values(a_files[key], dtype),
            read_values(b_files[key], dtype),
            dtype, args.atol, args.rtol,
        )
        display(key, result)
        if result["status"] != "PASS":
            failures.append((key, result))

    missing_in_b = sorted(set(a_files) - set(b_files), key=stage_sort)
    missing_in_a = sorted(set(b_files) - set(a_files), key=stage_sort)
    if missing_in_b or missing_in_a:
        print()
        for k in missing_in_b:
            print(f"ONLY_A t={k[0]} chunk={k[1]} {k[2]} {k[3]} {k[4]}")
        for k in missing_in_a:
            print(f"ONLY_B t={k[0]} chunk={k[1]} {k[2]} {k[3]} {k[4]}")

    # Compare the combined MoE FFN output too: the legacy split may
    # rotate which physical expert channels belong to which backend.
    # Different PC/Phone partials alone do not imply an incorrect sum.
    print()
    print("FFN combined partial parity (PC + Phone):")
    for chunk in sorted(
        {k[1] for k in common if k[3] == "partial" and k[1] != "full"},
        key=int,
    ):
        keys = {
            side: (tokens, chunk, side, "partial", "f32")
            for side in ("pc", "phone")
        }
        if any(k not in a_files or k not in b_files for k in keys.values()):
            print(f"chunk={chunk} SKIP incomplete partials")
            continue
        ap = read_values(a_files[keys["pc"]], "f32")
        ah = read_values(a_files[keys["phone"]], "f32")
        bp = read_values(b_files[keys["pc"]], "f32")
        bh = read_values(b_files[keys["phone"]], "f32")
        if not (len(ap) == len(ah) == len(bp) == len(bh)):
            print(f"chunk={chunk} SHAPE partial lengths differ")
            continue
        a_sum = [float(x) + float(y) for x, y in zip(ap, ah)]
        b_sum = [float(x) + float(y) for x, y in zip(bp, bh)]
        key = (tokens, chunk, "sum", "ffn_combined", "f32")
        result = compare(a_sum, b_sum, "f32", args.atol, args.rtol)
        display(key, result, label="  ")
        if result["status"] != "PASS":
            failures.append((key, result))

    print()
    if failures:
        key, result = failures[0]
        print(
            "FIRST DIFFERENCE (stage order): "
            f"chunk={key[1]} side={key[2]} stage={key[3]} "
            f"status={result['status']}"
        )
        if key[3] == "partial":
            print(
                "NOTE: Check FFN combined parity before blaming one "
                "backend's shard: A/B can use different split rotation."
            )
        print("Compare preceding stages, then check L47 input.")
    else:
        print("All shared stages match within the requested tolerance.")
        if missing_in_a or missing_in_b:
            print("Some stages were missing; do not treat this as full parity.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
