#!/usr/bin/env python3
"""CPU/NPU tensor comparison for incremental attention-output or FFN offload.

The reference is NOT all-CPU: it includes already-validated QKV on HTP.
For FFN cases it also includes the attention-output MatMuls on HTP.

This diagnostic does not prove actual NPU execution: verify the matching
[HEX_LINEAR_AUDIT] weight name in the phone RPC log for every released weight.
"""
import argparse
from bisect import bisect_right
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
RE_SINGLE_CASE = re.compile(r"^(?P<layer>\d+)-(?P<role>output|gate|up|down|ffn)$")
STAGE_ORDER = [
    "attn_norm", "Qcur", "Kcur", "Vcur", "kq", "kq_soft_max",
    "kqv", "kqv_out", "attn_out", "ffn_inp", "ffn_norm",
    "ffn_up", "ffn_gate", "ffn_swiglu", "ffn_down", "ffn_out", "l_out",
]
STAGE_INDEX = {name: index for index, name in enumerate(STAGE_ORDER)}


def resolve_target(case, expected_layers, focus_layer):
    # Control run: same baseline weights twice, so ALL changed values are
    # suspicious run-to-run drift rather than effects of a migrated MatMul.
    if case == "baseline-control":
        if focus_layer is not None:
            raise ValueError("--focus-layer is not valid for baseline-control")
        return [], max(expected_layers) + 1, "control", "attn_out"

    match = RE_SINGLE_CASE.fullmatch(case)
    if match:
        layer = int(match.group("layer"))
        if layer not in expected_layers:
            raise ValueError(f"case layer {layer} is outside {expected_layers}")
        if focus_layer is not None and focus_layer != layer:
            raise ValueError("--focus-layer must equal the target case layer")
        return [layer], layer, match.group("role"), {
            "output": "attn_out", "gate": "ffn_gate", "up": "ffn_up",
            "down": "ffn_out", "ffn": "ffn_up",
        }[match.group("role")]

    if case not in ("all-output", "all-ffn"):
        raise ValueError(f"unknown case: {case}")
    first = min(expected_layers)
    if focus_layer is not None and focus_layer != first:
        raise ValueError("--focus-layer for all-* must be the first changed layer")
    role = "output" if case == "all-output" else "ffn"
    return list(expected_layers), first, role, (
        "attn_out" if role == "output" else "ffn_up")


def region_of(capture_id, layer, stage, changed_layers,
              direct_stage, role, evaluation_index, target_capture_in_eval):
    # Every new evaluation has its own target barrier. In particular, a
    # 42-token block's Layer 13 occurs BEFORE the 42-token Layer 15 MatMul,
    # even though it occurs AFTER an earlier 2-token Layer 15 MatMul.
    # Later-eval input state may legitimately depend on earlier evals,
    # so separate it from the first evaluation's upstream region.
    if role == "control":
        return "UPSTREAM"
    if target_capture_in_eval is None or capture_id < target_capture_in_eval:
        return "UPSTREAM" if evaluation_index == 0 else "REENTRY_UPSTREAM"
    if layer in changed_layers and (
        stage == direct_stage or
        (role == "ffn" and stage == "ffn_gate")
    ):
        return "TARGET"
    return "DOWNSTREAM"


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def compare(ref_dir: Path, npu_dir: Path, case: str, output: Path,
            atol: float, rtol: float, min_cosine: float,
            max_normalized_rmse: float, expected_layers, focus_layer=None):
    changed_layers, first_layer, role, direct_stage = resolve_target(
        case, expected_layers, focus_layer)
    ref = load_manifest(ref_dir)
    npu = load_manifest(npu_dir)
    shared = sorted(ref.keys() & npu.keys(), key=lambda k: ref[k].capture_id)
    missing_ref = sorted(ref.keys() - npu.keys(), key=lambda k: ref[k].capture_id)
    missing_npu = sorted(npu.keys() - ref.keys(), key=lambda k: npu[k].capture_id)

    # The lowest selected layer's attn_norm marks the beginning of each
    # complete graph evaluation. In the observed CLI run, the two groups
    # correspond to 2-token and 42-token batches. Do not infer they are
    # Decode steps without inspecting PC logs.
    anchor_name = f"attn_norm-{min(expected_layers)}"
    anchors = sorted(
        snap.capture_id for snap in ref.values() if snap.name == anchor_name
    )
    if not anchors:
        # A missing anchor is reported as capture mismatch below; keep
        # a safe fallback for other rows.
        anchors = [min((snap.capture_id for snap in ref.values()), default=0)]

    def evaluation_of(capture_id):
        return max(0, bisect_right(anchors, capture_id) - 1)

    first_target_by_evaluation = {}
    for snap in ref.values():
        match = RE_LAYER.fullmatch(snap.name)
        if (match is not None and int(match["layer"]) in changed_layers and
            (match["stage"] == direct_stage or
             (role == "ffn" and match["stage"] == "ffn_gate"))):
            evaluation = evaluation_of(snap.capture_id)
            first_target_by_evaluation[evaluation] = min(
                snap.capture_id,
                first_target_by_evaluation.get(evaluation, snap.capture_id))
    target_capture_ids = sorted(first_target_by_evaluation.values())
    first_target_capture = target_capture_ids[0] if target_capture_ids else None

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
        evaluation = evaluation_of(a.capture_id)
        region = region_of(a.capture_id, layer, stage, changed_layers,
                           direct_stage, role, evaluation,
                           first_target_by_evaluation.get(evaluation))

        severe = (stats["nonfinite"] != 0 or
                  (stats["bad"] != 0 and not math.isfinite(cosine)) or
                  (math.isfinite(cosine) and cosine < min_cosine) or
                  (stage in CHECK_STAGES and nrmse > max_normalized_rmse))
        level = "CRITICAL" if severe else ("DIFF" if stats["bad"] else "OK")
        rows.append({
            "case": case, "region": region, "evaluation": evaluation,
            "cpu_capture": a.capture_id, "npu_capture": b.capture_id,
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
    fields = ["case", "region", "evaluation", "cpu_capture", "npu_capture", "layer", "name", "stage",
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
    captured_stages = {(r["layer"], r["stage"]) for r in rows}
    # Llama3.2 src/models/llama.cpp emits these callbacks at each layer.
    # ffn_down's callback may be missing on down-bias-free models.
    # 'ffn_out' can be absent from the scheduler callback even when the
    # final residual/layer output 'l_out' exists and is captured.
    # Enforce the direct attention output and the final layer output.
    mandatory_stages = {"attn_out", "l_out"}
    missing_stages = sorted(
        "%s-%d" % (stage, layer)
        for layer in expected_layers
        for stage in mandatory_stages
        if (layer, stage) not in captured_stages
    )
    # Require the output of the actual newly moved MatMul (or its first
    # visible consumer), not merely a collection of unrelated captures.
    target_required = ["%s-%d" % (direct_stage, layer)
                       for layer in changed_layers]
    if role == "ffn":
        target_required += ["ffn_gate-%d" % layer for layer in changed_layers]
    missing_stages = sorted(set(missing_stages) |
                            {label for label in target_required
                             if (int(label.rsplit("-", 1)[1]),
                                 label.rsplit("-", 1)[0]) not in captured_stages})
    if not any(snap.name == anchor_name for snap in ref.values()):
        missing_stages.append(anchor_name)

    upstream = [r for r in rows if r["region"] == "UPSTREAM"]
    reentry  = [r for r in rows if r["region"] == "REENTRY_UPSTREAM"]
    target   = [r for r in rows if r["region"] == "TARGET"]
    downstream = [r for r in rows if r["region"] == "DOWNSTREAM"]
    upstream_diff = [r for r in upstream if r["bad"] > 0]
    reentry_diff = [r for r in reentry if r["bad"] > 0]
    target_critical = [r for r in target if r["status"] == "CRITICAL"]
    downstream_critical = [r for r in downstream if r["status"] == "CRITICAL"]

    # CAPTURE_MISMATCH strictly describes absent snapshots/stages.
    # Differences before the first migrated MatMul must not be dismissed as
    # downstream propagation: they indicate a non-identical starting state.
    if not shared or missing_ref or missing_npu or missing_stages:
        verdict = "CAPTURE_MISMATCH"
    elif upstream_diff:
        verdict = "UPSTREAM_DIVERGENCE"
    elif reentry_diff:
        verdict = "REENTRY_UPSTREAM_DIVERGENCE"
    elif target_critical:
        verdict = "TARGET_CRITICAL"
    elif downstream_critical:
        verdict = "DOWNSTREAM_CRITICAL"
    else:
        verdict = "NUMERICALLY_CLOSE"

    def worst(stages):
        pool = [r for r in rows if r["stage"] in stages
                and r["normalized_rmse"] is not None]
        return max(pool, key=lambda r: r["normalized_rmse"], default=None)

    def region_summary(subset):
        return {
            "compared": len(subset),
            "diff": sum(r["bad"] > 0 for r in subset),
            "critical": sum(r["status"] == "CRITICAL" for r in subset),
            "first_diff": next((r for r in subset if r["bad"] > 0), None),
            "worst": max(
                (r for r in subset if r["normalized_rmse"] is not None),
                key=lambda r: r["normalized_rmse"], default=None),
        }

    region_counts = {
        "upstream": region_summary(upstream),
        "reentry_upstream": region_summary(reentry),
        "target": region_summary(target),
        "downstream": region_summary(downstream),
    }

    result = {
        "case": case,
        "focus_layer": first_layer,
        "first_target_capture_id": first_target_capture,
        "graph_evaluation_starts": anchors,
        "target_capture_ids_by_evaluation": first_target_by_evaluation,
        "changed_layers": changed_layers,
        "target_stage": direct_stage,
        "target_role": role,
        "region_counts": region_counts,
        "worst_upstream": region_counts["upstream"]["worst"],
        "worst_target": region_counts["target"]["worst"],
        "worst_downstream": region_counts["downstream"]["worst"],
        "first_upstream_diff": upstream_diff[0] if upstream_diff else None,
        "first_reentry_diff": reentry_diff[0] if reentry_diff else None,
        "status": verdict,
        "cpu_captures": len(ref),
        "npu_captures": len(npu),
        "compared": len(shared),
        "only_cpu": len(missing_ref),
        "only_npu": len(missing_npu),
        "missing_stages": missing_stages,
        "only_baseline_examples": [str(k) for k in missing_ref[:8]],
        "only_candidate_examples": [str(k) for k in missing_npu[:8]],
        "captured_stages": sorted("%s-%d" % (stage, layer)
                                  for layer, stage in captured_stages),
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
        "note": ("UPSTREAM means before target MatMul output in the FIRST "
                 "graph evaluation. REENTRY_UPSTREAM means before the target "
                 "in a LATER graph evaluation; it may indicate stale KV/"
                 "scheduler state or different subsequent input tokens. "
                 "Capture order is not DSP execution timestamp; do not "
                 "assume a lower layer in a later evaluation is causally "
                 "independent of a previous higher layer. Passing still "
                 "requires HTP audit and capture-free decode."),
    }
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")

    print("[HEX_LINEAR_COMPARE] case=%s status=%s compared=%d "
          "only_cpu=%d only_npu=%d critical=%d diff=%d" %
          (case, verdict, len(shared), len(missing_ref),
           len(missing_npu), len(critical), len(diff)))
    print("  target=%s layers=%s focus=%d graph_starts=%s target_by_eval=%s"
          % (direct_stage, changed_layers, first_layer,
             anchors, first_target_by_evaluation))
    if missing_stages:
        print("  required stages not captured:", ", ".join(missing_stages))
    for region, info in region_counts.items():
        item = info["first_diff"]
        print("  %s compared=%d diff=%d critical=%d first_diff=%s" %
              (region, info["compared"], info["diff"], info["critical"],
               item["name"] if item else "none"))
    if upstream_diff:
        x = upstream_diff[0]
        print("  UPSTREAM_ALERT first=%s occurrence=%s nrmse=%s" %
              (x["name"], x["occurrence"], x["normalized_rmse"]))
    if reentry_diff:
        x = reentry_diff[0]
        print("  REENTRY_ALERT evaluation=%d first=%s occurrence=%s nrmse=%s" %
              (x["evaluation"], x["name"], x["occurrence"],
               x["normalized_rmse"]))
    for layer in sorted(set(row["layer"] for row in rows)):
        local = [r for r in rows if r["layer"] == layer]
        targets = [r for r in local if r["stage"] in CHECK_STAGES]
        if not targets:
            continue
        w = max(targets, key=lambda r: (
            r["normalized_rmse"] if r["normalized_rmse"] is not None else -1))
        print("  layer=%d checked=%d critical=%d worst=%s "
              "rmse=%s nrmse=%s cosine=%s" %
              (layer, len(targets),
               sum(r["status"] == "CRITICAL" for r in targets),
               w["name"], str(w["rmse"]),
               str(w["normalized_rmse"]), str(w["cosine"])))
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
    parser.add_argument("--expected-layers", required=True,
                        help="Comma-separated layer indices, e.g. 13,14,15")
    parser.add_argument("--focus-layer", type=int,
                        help="Target layer to diagnose; defaults to case layer.")
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
    parser.add_argument("--min-cosine", type=float, default=0.95)
    parser.add_argument("--max-normalized-rmse", type=float, default=0.05)
    args = parser.parse_args()
    expected = [int(x) for x in args.expected_layers.split(",") if x.strip()]
    if not expected:
        parser.error("--expected-layers must include at least one layer")
    result = compare(args.reference, args.candidate, args.case, args.output,
                     args.atol, args.rtol, args.min_cosine,
                     args.max_normalized_rmse, expected, args.focus_layer)
    return {"NUMERICALLY_CLOSE": 0, "TARGET_CRITICAL": 2,
            "DOWNSTREAM_CRITICAL": 2, "CAPTURE_MISMATCH": 3,
            "UPSTREAM_DIVERGENCE": 4,
            "REENTRY_UPSTREAM_DIVERGENCE": 4}[result["status"]]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit("error: %s" % exc) from exc
