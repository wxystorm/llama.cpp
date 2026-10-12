#!/usr/bin/env python3
"""CPU/NPU tensor comparison for incremental attention-output or FFN offload.

The reference is NOT all-CPU: it includes already-validated QKV on HTP.
For FFN cases it also includes the attention-output MatMuls on HTP.

This diagnostic does not prove actual NPU execution: verify the matching
[HEX_LINEAR_AUDIT] weight name in the phone RPC log for every released weight.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

from compare_tensor_dumps import load_manifest, metrics, read_values

RE_LAYER = re.compile(r"^(?P<stage>.+)-(?P<layer>\d+)$")
# Review NRMSE in these downstream stages, rather than applying a single
# absolute RMSE threshold to MatMuls with different natural scales.
CHECK_STAGES = frozenset({
    "attn_out", "ffn_inp", "ffn_norm", "ffn_up", "ffn_gate",
    "ffn_swiglu", "ffn_down", "ffn_out", "l_out",
})
OUTPUT_STAGES = frozenset({"attn_out", "ffn_out", "l_out"})


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare(ref_dir: Path, npu_dir: Path, case: str, output: Path,
            atol: float, rtol: float, min_cosine: float,
            max_normalized_rmse: float):
    ref = load_manifest(ref_dir)
    npu = load_manifest(npu_dir)
    shared = sorted(ref.keys() & npu.keys(), key=lambda k: ref[k].capture_id)
    missing_ref = sorted(ref.keys() - npu.keys(), key=lambda k: ref[k].capture_id)
    missing_npu = sorted(npu.keys() - ref.keys(), key=lambda k: npu[k].capture_id)

    rows = []
    for key in shared:
        a, b = ref[key], npu[key]
        av = read_values(a)
        bv = read_values(b)
        stats = metrics(av, bv, atol, rtol)
        ref_rms = math.sqrt(sum(float(x) ** 2 for x in av if math.isfinite(x))
                            / len(av)) if av else 0.0
        nrmse = stats["rmse"] / max(ref_rms, 1.0e-6)
        match = RE_LAYER.fullmatch(a.name)
        stage = match["stage"] if match else a.name
        layer = int(match["layer"]) if match else -1
        cosine = stats["cosine"]

        severe = (stats["nonfinite"] != 0 or
                  (stats["bad"] != 0 and not math.isfinite(cosine)) or
                  (math.isfinite(cosine) and cosine < min_cosine) or
                  (stage in CHECK_STAGES and nrmse > max_normalized_rmse))
        level = "CRITICAL" if severe else ("DIFF" if stats["bad"] else "OK")
        rows.append({
            "case": case, "cpu_capture": a.capture_id, "npu_capture": b.capture_id,
            "layer": layer, "name": a.name, "stage": stage,
            "shape": "x".join(str(v) for v in a.shape),
            "occurrence": a.occurrence, "dtype": a.dtype,
            "cpu_op": a.op, "npu_op": b.op,
            "status": level, "bad": stats["bad"], "total": stats["n"],
            "nonfinite": stats["nonfinite"],
            "max_abs": safe(stats["max_abs"]),
            "rmse": safe(stats["rmse"]),
            "reference_rms": safe(ref_rms),
            "normalized_rmse": safe(nrmse),
            "cosine": safe(cosine),
        })

    output.mkdir(parents=True, exist_ok=True)
    fields = ["case", "cpu_capture", "npu_capture", "layer", "name", "stage",
              "shape", "occurrence", "dtype", "cpu_op", "npu_op",
              "status", "bad", "total", "nonfinite", "max_abs",
              "rmse", "reference_rms", "normalized_rmse", "cosine"]
    with (output / "tensor_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    critical = [r for r in rows if r["status"] == "CRITICAL"]
    diff = [r for r in rows if r["status"] == "DIFF"]
    stages_present = {r["stage"] for r in rows}
    # Llama 3.2 source (src/models/llama.cpp) always has these callbacks.
    # ffn_down does not necessarily exist as a callback without down bias.
    mandatory_stages = {"attn_out", "ffn_out", "l_out"}
    missing_stages = sorted(mandatory_stages - stages_present)
    if not shared or missing_ref or missing_npu or missing_stages:
        verdict = "CAPTURE_MISMATCH"
    elif critical:
        verdict = "CRITICAL"
    else:
        verdict = "NUMERICALLY_CLOSE"

    def worst(stages):
        pool = [r for r in rows if r["stage"] in stages
                and r["normalized_rmse"] is not None]
        return max(pool, key=lambda r: r["normalized_rmse"], default=None)

    result = {
        "case": case,
        "status": verdict,
        "cpu_captures": len(ref),
        "npu_captures": len(npu),
        "compared": len(shared),
        "only_cpu": len(missing_ref),
        "only_npu": len(missing_npu),
        "missing_stages": missing_stages,
        "critical": len(critical),
        "diff": len(diff),
        "min_cosine": min((r["cosine"] for r in rows if r["cosine"] is not None),
                          default=None),
        "max_rmse": max((r["rmse"] for r in rows if r["rmse"] is not None),
                        default=None),
        "max_normalized_rmse": max(
            (r["normalized_rmse"] for r in rows
             if r["normalized_rmse"] is not None), default=None),
        "worst_checked_stage": worst(CHECK_STAGES),
        "worst_output_stage": worst(OUTPUT_STAGES),
        "first_critical": critical[0] if critical else None,
        "critical_examples": critical[:10],
        "note": ("This is an exploratory numerical smoke test; passing requires "
                 "the target weight's HTP audit line and capture-free decode."),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")

    print("[HEX_LINEAR_COMPARE] case=%s status=%s compared=%d "
          "only_cpu=%d only_npu=%d critical=%d diff=%d" %
          (case, verdict, len(shared), len(missing_ref),
           len(missing_npu), len(critical), len(diff)))
    if missing_stages:
        print("  required stages not captured:", ", ".join(missing_stages))
    for layer in sorted(set(row["layer"] for row in rows)):
        local = [r for r in rows if r["layer"] == layer]
        targets = [r for r in local if r["stage"] in CHECK_STAGES]
        if not targets:
            continue
        w = max(targets, key=lambda r: r["normalized_rmse"] or 0.0)
        print("  layer=%d checked=%d critical=%d worst=%s "
              "rmse=%.6g nrmse=%.6g cosine=%s" %
              (layer, len(targets),
               sum(r["status"] == "CRITICAL" for r in targets),
               w["name"], w["rmse"] or 0.0,
               w["normalized_rmse"] or 0.0, str(w["cosine"])))
    if critical:
        f = critical[0]
        print("  first CRITICAL in reference execution order: %s "
              "rmse=%s nrmse=%s cosine=%s" %
              (f["name"], f["rmse"], f["normalized_rmse"], f["cosine"]))
    print("  output:", output / "tensor_metrics.csv", output / "summary.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--min-cosine", type=float, default=0.95)
    parser.add_argument("--max-normalized-rmse", type=float, default=0.05)
    args = parser.parse_args()
    result = compare(args.reference, args.candidate, args.case, args.output,
                     args.atol, args.rtol, args.min_cosine,
                     args.max_normalized_rmse)
    return {"NUMERICALLY_CLOSE": 0, "CRITICAL": 2,
            "CAPTURE_MISMATCH": 3}[result["status"]]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit("error: %s" % exc) from exc
