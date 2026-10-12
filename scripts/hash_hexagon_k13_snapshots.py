#!/usr/bin/env python3
"""Hash the exact captured K13 F32 MatMul output bytes for DSP/host/PC comparison.

Read-only: uses existing LLAMA_TENSOR_COMPARE_DIR manifests and dump files.
Run from the repository root on Windows:
    py -3 scripts/hash_hexagon_k13_snapshots.py ^
      .\hexagon-k13-force-flush\attn ^
      --tensor Kcur-13 --ne0 512 --tokens 42 --column 2

Compare fnv64 below with HEX_K13_DSP_PROBE output_flush_fnv64 and the
HEX_K14_TENSOR phase=POST role=output fnv64 line sharing the K13 seq.
In the old host helper, that POST line still says K14_TENSOR for K13.
"""

import argparse
import math
from pathlib import Path
import struct
import sys

from compare_tensor_dumps import load_manifest


def fnv64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def analyze(folder, tensor, ne0, tokens, column):
    snapshots = load_manifest(folder)
    choices = sorted(
        (s for s in snapshots.values()
         if s.name == tensor and s.shape == (ne0, tokens, 1, 1)
         and s.dtype == "f32"),
        key=lambda s: s.capture_id,
    )
    if not choices:
        raise ValueError(f"Missing F32 {tensor} of shape {ne0}x{tokens}x1x1 in {folder}")
    if len(choices) != 1:
        raise ValueError(f"Expected exactly one {tensor} MatMul snapshot in {folder}, got {len(choices)}")
    s = choices[0]
    raw = (folder / s.filename).read_bytes()
    want_bytes = ne0 * tokens * 4
    if s.nbytes != want_bytes or len(raw) != want_bytes:
        raise ValueError(f"Expected {want_bytes} bytes, manifest says {s.nbytes}, read {len(raw)}")
    start = ne0 * column * 4
    values = [v[0] for v in struct.iter_unpack("<f", raw[start:start + ne0 * 4])]
    zeros = sum(v == 0 for v in values)
    nonfinite = sum(not math.isfinite(v) for v in values)
    finite_values = [v for v in values if math.isfinite(v)]
    rms = math.sqrt(sum(v * v for v in finite_values) / len(finite_values)) if finite_values else float("nan")
    return {
        "case": folder.name, "capture": s.capture_id, "tensor": s.name,
        "shape": s.shape, "bytes": len(raw), "fnv64": f"{fnv64(raw):016x}",
        "column": column, "zero_count": zeros, "width": ne0,
        "nonfinite": nonfinite, "column_rms": rms,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage_folder", type=Path, help="Folder containing baseline, 15-output, baseline-repeat")
    ap.add_argument("--tensor", default="Kcur-13")
    ap.add_argument("--ne0", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=42)
    ap.add_argument("--column", type=int, default=2)
    args = ap.parse_args()
    if args.ne0 < 1 or args.tokens < 1 or not (0 <= args.column < args.tokens):
        ap.error("Invalid ne0, tokens, or column index")
    rows = []
    for case in ("baseline", "15-output", "baseline-repeat"):
        case_folder = args.stage_folder / case
        if not case_folder.is_dir():
            ap.error(f"Missing case folder: {case_folder}")
        rows.append(analyze(case_folder, args.tensor, args.ne0, args.tokens, args.column))
    for r in rows:
        print(f"[HEX_PC_SNAPSHOT_HASH] case={r['case']} capture={r['capture']}"
              f" name={r['tensor']} bytes={r['bytes']}"
              f" fnv64={r['fnv64']} column={r['column']}"
              f" zeros={r['zero_count']}/{r['width']}"
              f" nonfinite={r['nonfinite']} rms={r['column_rms']:.9g}")
    unique = {r["fnv64"] for r in rows}
    print(f"distinct_PC_hashes={len(unique)} baseline_repeat_matches="
          f"{rows[0]['fnv64'] == rows[2]['fnv64']}")
    if len(unique) > 1:
        print("PC snapshots differ; compare each PC full hash to the *same run* DSP and Android-host post-exec hashes.")
    else:
        print("PC snapshots identical; check the latest per-token numerical comparison for this MM_SELECT mode.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

