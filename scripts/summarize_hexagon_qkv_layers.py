#!/usr/bin/env python3
"""Compare CPU reference vs an NPU Q/K/V-placement case across layers.

Emits a per-tensor CSV and a per-run JSON summary. This diagnostic is for
correctness only: callback snapshots change graph boundaries and scheduling.

Numerical tolerances classify small expected differences as DIFF, not FAIL.
CRITICAL is a conservative smoke-test heuristic, not a mathematical proof.
Check actual HTP execution via [HEX_QKV_AUDIT] on the phone before accepting
a passing NPU case.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

from compare_tensor_dumps import load_manifest, metrics, read_values

LAYER_NAME = re.compile(r"^(?P<stage>.+)-(?P<layer>\d+)$")
ATTENTION_OUTPUT_STAGES = {"kq_soft_max", "kqv", "kqv_out", "attn_out"}


def summarize(reference: Path, candidate: Path, case: str, out: Path,
              atol: float, rtol: float, min_cosine: float,
              max_attention_rmse: float) -> dict:
    ref = load_manifest(reference)
    npu = load_manifest(candidate)
    common = sorted(ref.keys() & npu.keys(), key=lambda key: ref[key].capture_id)
    only_ref = sorted(ref.keys() - npu.keys(), key=lambda key: ref[key].capture_id)
    only_npu = sorted(npu.keys() - ref.keys(), key=lambda key: npu[key].capture_id)

    rows = []
    critical_rows = []
    warning_rows = []
    for key in common:
        a, b = ref[key], npu[key]
        stat = metrics(read_values(a), read_values(b), atol, rtol)
        match = LAYER_NAME.match(a.name)
        stage = match["stage"] if match else a.name
        layer = int(match["layer"]) if match else -1
        cosine = stat["cosine"]

        severe = (stat["nonfinite"] > 0 or
                  (math.isfinite(cosine) and cosine < min_cosine) or
                  (stage in ATTENTION_OUTPUT_STAGES and
                   stat["rmse"] > max_attention_rmse))
        severity = "CRITICAL" if severe else "DIFF" if stat["bad"] else "OK"
        row = dict(
            case=case, cpu_id=a.capture_id, npu_id=b.capture_id,
            layer=layer, name=a.name, stage=stage,
            shape="x".join(map(str, a.shape)), occurrence=a.occurrence,
            cpu_op=a.op, npu_op=b.op, dtype=a.dtype,
            severity=severity, bad=stat["bad"], total=stat["n"],
            nonfinite=stat["nonfinite"], max_abs=stat["max_abs"],
            rmse=stat["rmse"], cosine=cosine)
        rows.append(row)
        if severe:
            critical_rows.append(row)
        elif stat["bad"]:
            warning_rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    fields = ["case", "cpu_id", "npu_id", "layer", "name", "stage", "shape",
              "occurrence", "cpu_op", "npu_op", "dtype", "severity", "bad",
              "total", "nonfinite", "max_abs", "rmse", "cosine"]
    with (out / "tensor_metrics.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if not common or only_ref or only_npu:
        overall = "CAPTURE_MISMATCH"
    elif critical_rows:
        overall = "CRITICAL"
    else:
        overall = "NUMERICALLY_CLOSE"

    report = {
        "case": case, "status": overall,
        "cpu_captures": len(ref), "npu_captures": len(npu),
        "compared": len(common),
        "only_cpu": len(only_ref), "only_npu": len(only_npu),
        "only_cpu_examples": [k[0] for k in only_ref[:8]],
        "only_npu_examples": [k[0] for k in only_npu[:8]],
        "critical": len(critical_rows), "diff": len(warning_rows),
        "min_cosine": min(
            (r["cosine"] for r in rows if math.isfinite(r["cosine"])),
            default=None),
        "max_rmse": max((r["rmse"] for r in rows), default=None),
        "critical_examples": [
            {k: r[k] for k in ("layer", "name", "shape", "bad",
                              "total", "rmse", "cosine", "nonfinite")}
            for r in critical_rows[:12]],
        "notes": [
            "NUMERICALLY_CLOSE is not bytewise identical and not a proof of full-model correctness.",
            "Confirm the intended QKV weight executed on HTP via the phone [HEX_QKV_AUDIT] log.",
            "Only same-input (before token divergence) snapshots are comparable.",
            "The callback may alter scheduler split boundaries and timing.",
        ],
    }
    with (out / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")

    print(f"[QKV_CHECK] case={case} status={overall}"
          f" compared={len(common)} only_cpu={len(only_ref)} only_npu={len(only_npu)}"
          f" critical={len(critical_rows)} diff={len(warning_rows)}")
    for layer in sorted(set(r["layer"] for r in rows)):
        subset = [r for r in rows if r["layer"] == layer]
        if not subset:
            continue
        worst = min(
            subset,
            key=lambda r: (
                -1 if not math.isfinite(r["cosine"]) and r["nonfinite"] else
                r["cosine"] if math.isfinite(r["cosine"]) else 1))
        worst_attn = max(
            (r for r in subset if r["stage"] in ATTENTION_OUTPUT_STAGES),
            key=lambda r: r["rmse"], default=None)
        critical_count = sum(r["severity"] == "CRITICAL" for r in subset)
        msg = (f"  layer={layer} captures={len(subset)} critical={critical_count}"
               f" worst_cosine={worst['cosine']:.7g} ({worst['name']})")
        if worst_attn:
            msg += (f" max_attn_rmse={worst_attn['rmse']:.6g}"
                    f" ({worst_attn['name']})")
        print(msg)
    if critical_rows:
        first = critical_rows[0]
        print(f"  first critical in CPU capture order:"
              f" id={first['cpu_id']} name={first['name']}"
              f" shape={first['shape']} rmse={first['rmse']:.6g}"
              f" cosine={first['cosine']:.6g}")
    if only_ref:
        print("  CPU-only examples:", ", ".join(report["only_cpu_examples"]))
    if only_npu:
        print("  NPU-only examples:", ", ".join(report["only_npu_examples"]))
    print(f"  saved: {out / 'tensor_metrics.csv'}")
    print(f"  saved: {out / 'summary.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--case", default="case")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--min-cosine", type=float, default=0.95)
    parser.add_argument("--max-attention-rmse", type=float, default=0.05)
    args = parser.parse_args()
    report = summarize(args.reference, args.candidate, args.case, args.output,
                       args.atol, args.rtol, args.min_cosine,
                       args.max_attention_rmse)
    if report["status"] == "CAPTURE_MISMATCH":
        return 3
    if report["status"] == "CRITICAL":
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc
