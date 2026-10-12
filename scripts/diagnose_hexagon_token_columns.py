#!/usr/bin/env python3
"""Locate error-concentrated token columns in saved llama.cpp tensor dumps.

Usage (from repository root on Windows):
  py -3 scripts/diagnose_hexagon_token_columns.py ^
    .\hexagon-linear-attn-control\attn\baseline ^
    .\hexagon-linear-attn-control\attn\15-output ^
    --tokens 42 --top 5

GGML snapshots are contiguous, ne0-fastest. For common QKV/attention tensors
whose token dimension is the LAST non-singleton axis, a token occupies one
contiguous slice. This script rejects unsupported layouts rather than silently
mixing tokens.
"""

import argparse
import csv
import math
from pathlib import Path
import re
from compare_tensor_dumps import load_manifest, metrics, read_values


def get_token_extent(shape, n_tokens):
    candidates = [i for i in range(1, 4) if shape[i] == n_tokens and
                  all(x == 1 for x in shape[i + 1:])]
    if not candidates:
        return None
    # A dimension may coincide with the requested token count. Prefer the
    # last axis, as GGML stores the common batch token dimension last.
    axis = max(candidates)
    inner = math.prod(shape[:axis])
    if inner * n_tokens != math.prod(shape):
        return None
    return inner


def compare_columns(baseline, candidate, tokens, selected, atol, rtol):
    ref = load_manifest(baseline)
    npu = load_manifest(candidate)
    found = []
    for key in sorted(ref.keys() & npu.keys(),
                      key=lambda k: ref[k].capture_id):
        a, b = ref[key], npu[key]
        if not selected.fullmatch(a.name):
            continue
        block = get_token_extent(a.shape, tokens)
        if block is None or a.shape != b.shape:
            continue
        va = read_values(a)
        vb = read_values(b)
        if len(va) != len(vb) or len(va) != block * tokens:
            raise ValueError(f"Bad token stride or unmatched values for {a.name}")

        per_name = []
        for i in range(tokens):
            left = va[i * block:(i + 1) * block]
            right = vb[i * block:(i + 1) * block]
            stat = metrics(left, right, atol, rtol)
            rms = math.sqrt(sum(float(v)**2 for v in left if math.isfinite(v)) /
                            len(left))
            nr = stat["rmse"] / max(rms, 1e-6)
            per_name.append({
                "name": a.name,
                "capture_id": a.capture_id,
                "occurrence": a.occurrence,
                "shape": "x".join(map(str, a.shape)),
                "token_column": i,
                "bad": stat["bad"],
                "total": stat["n"],
                "nonfinite": stat["nonfinite"],
                "rmse": stat["rmse"],
                "normalized_rmse": nr,
                "cosine": stat["cosine"],
            })
        found.append(per_name)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tokens", type=int, default=42)
    parser.add_argument("--name-regex", default=
                        r"(attn_norm|Qcur|Kcur|Vcur|kq|kqv_out|attn_out|"
                        r"ffn_inp|ffn_norm|ffn_gate|ffn_up|l_out)-(13|14|15)")
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--csv", type=Path, help="Save all column rows to CSV")
    args = parser.parse_args()

    if args.tokens < 1:
        parser.error("--tokens must be positive")
    groups = compare_columns(args.baseline, args.candidate, args.tokens,
                             re.compile(args.name_regex),
                             args.atol, args.rtol)
    if not groups:
        parser.error("No matching snapshots with the requested token axis")
    all_rows = []
    for group in groups:
        all_rows.extend(group)
        bad = [r for r in group if r["bad"] or r["nonfinite"]]
        if bad:
            order = sorted(bad, key=lambda r: r["normalized_rmse"],
                           reverse=True)
            print(f"{group[0]['name']} capture={group[0]['capture_id']} "
                  f"shape={group[0]['shape']} bad_columns={len(bad)}"
                  f"/{args.tokens} bad_elements={sum(r['bad'] for r in group)}")
            for row in order[:args.top]:
                print("  token_col=%d bad=%d/%d rmse=%.6g nrmse=%.5g "
                      "cosine=%.6g" %
                      (row["token_column"], row["bad"], row["total"],
                       row["rmse"], row["normalized_rmse"], row["cosine"]))
        else:
            print(f"{group[0]['name']} capture={group[0]['capture_id']} "
                  f"shape={group[0]['shape']} all token columns match")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8-sig", newline="") as fp:
            wr = csv.DictWriter(fp, fieldnames=list(all_rows[0]))
            wr.writeheader()
            wr.writerows(all_rows)
        print(f"Saved {len(all_rows)} token rows in {args.csv}")


if __name__ == "__main__":
    main()
