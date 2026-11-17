#!/usr/bin/env python3
"""PRIMA Fig3: strict two-rank, 6-category, exclusive Prefill accounting.

Raw overlapping source events are never discarded. CUDA timings are real
device durations but legacy CUDA-to-host epoch placement is currently an
upper-bound sync anchor. Accordingly paper_fidelity is PROVISIONAL unless
an absolute GPU timestamp trace has been supplied and integrated.
"""
from __future__ import annotations
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ("run_id", "rank", "pid", "start_ns", "end_ns", "category",
          "kind", "bytes", "detail")
CATEGORIES = ("Phone Compute", "PCIe Copy", "GPU Compute", "CPU Compute",
              "RPC / Coordination", "Waiting / Other")
MAPPING = {"Phone Compute": "Phone Compute", "PCIe Copy": "PCIe Copy",
           "GPU Compute": "GPU Compute", "CPU Compute": "CPU Compute",
           "Coordination": "RPC / Coordination"}
DIAGNOSTICS = ("Host Prepare", "Prefetch", "Host Transfer")


def events(path, run_id):
    result = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            if len(row) != len(FIELDS):
                raise ValueError(f"{path}:{index}: expected {len(FIELDS)} fields, got {len(row)}")
            r = dict(zip(FIELDS, row))
            if r["run_id"] != run_id:
                continue
            for key in ("rank", "pid", "start_ns", "end_ns", "bytes"):
                r[key] = int(r[key])
            if r["start_ns"] > r["end_ns"] or r["start_ns"] <= 0:
                raise ValueError(f"{path}:{index}: invalid interval")
            result.append(r)
    if not result:
        raise ValueError(f"no events for run_id={run_id} in {path}")
    return result


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def calibrate(samples):
    good = sorted([s for s in samples if 0 <= s["uncertainty_ns"] <= 20000000],
                  key=lambda s: s["uncertainty_ns"])
    if len(good) < 8:
        raise ValueError("less than 8 valid Phone clock four-timestamp samples")
    # Average of the 8 tightest four-timestamp estimates; use spread as
    # systematic error check instead of hiding clock drift.
    selected = good[:8]
    offset = int(statistics.median(s["offset_phone_minus_pc_ns"] for s in selected))
    drift = max(s["offset_phone_minus_pc_ns"] for s in selected) - min(
        s["offset_phone_minus_pc_ns"] for s in selected)
    limit = 20000000  # 20 ms
    before = sorted((s for s in samples if s.get("phase") == "before"),
                    key=lambda s: s["uncertainty_ns"])[:4]
    after = sorted((s for s in samples if s.get("phase") == "after"),
                   key=lambda s: s["uncertainty_ns"])[:4]
    if not before or not after:
        raise ValueError("missing before/after phone clock calibration")
    temporal_drift = abs(
        int(statistics.median(s["offset_phone_minus_pc_ns"] for s in before)) -
        int(statistics.median(s["offset_phone_minus_pc_ns"] for s in after)))
    if temporal_drift > limit:
        raise ValueError(f"phone clock drift {temporal_drift/1e6:.3f} ms exceeds 20 ms")
    if drift > limit:
        raise ValueError(f"Phone clock offset spread {drift / 1e6:.3f} ms exceeds 20 ms")
    return offset, min(s["uncertainty_ns"] for s in selected), max(drift, temporal_drift)


def projection(source, start, end):
    """O(N log N) event sweep with mutually-exclusive priority projection."""
    edges = []
    for r in source:
        if r["category"] not in CATEGORIES[:-1]:
            continue
        edges.append((r["a"], 1, r["category"]))
        edges.append((r["b"], -1, r["category"]))
    edges.sort()
    points = sorted({start, end} | {x[0] for x in edges})
    active = Counter()
    totals = defaultdict(int)
    pieces = []
    overlap_ns = 0
    i = 0
    for a, b in zip(points, points[1:]):
        while i < len(edges) and edges[i][0] <= a:
            _, change, cat = edges[i]
            active[cat] += change
            if active[cat] < 0:
                raise ValueError("unbalanced event endpoints")
            i += 1
        if b <= a:
            continue
        if sum(active.values()) > 1:
            overlap_ns += b - a
        cat = next((c for c in CATEGORIES[:-1] if active[c] > 0),
                   "Waiting / Other")
        totals[cat] += b - a
        if pieces and pieces[-1]["category"] == cat and pieces[-1]["b"] == a:
            pieces[-1]["b"] = b
        else:
            pieces.append(dict(a=a, b=b, category=cat))
    if sum(totals.values()) != end - start:
        raise ValueError("projection does not cover all Prefill nanoseconds")
    return pieces, totals, overlap_ns


def inspect(args):
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    run_id = manifest["run_id"]
    pc = events(args.pc, run_id)
    phone = events(args.phone, run_id)
    if any(r["rank"] != 0 for r in pc):
        raise ValueError("PC trace contains events with rank other than 0")
    if any(r["rank"] != 1 for r in phone):
        raise ValueError("Phone trace contains events with rank other than 1")
    prefill = [r for r in pc if r["category"] == "Boundary" and r["kind"] == "PREFILL"]
    if len(prefill) != 1:
        raise ValueError(f"need exactly one complete PC Prefill boundary: got {len(prefill)}")
    t0, t1 = prefill[0]["start_ns"], prefill[0]["end_ns"]
    if t1 <= t0:
        raise ValueError("zero-length Prefill boundary")
    if prefill[0]["bytes"] <= 0:
        raise ValueError("Prefill token count not recorded")

    samples = manifest.get("clock_samples", [])
    offset, uncertainty, spread = calibrate(samples)
    raw = []
    counts = Counter()
    for origin, rows in (("pc", pc), ("phone", phone)):
        for r in rows:
            a = r["start_ns"] - (offset if origin == "phone" else 0)
            b = r["end_ns"] - (offset if origin == "phone" else 0)
            if b <= t0 or a >= t1 or a == b:
                continue
            a, b = max(t0, a), min(t1, b)
            if b <= a:
                continue
            r = dict(r)
            r.update(a=a, b=b, origin=origin)
            r["category"] = MAPPING.get(r["category"], r["category"])
            if r["category"] in CATEGORIES and r["category"] != "Waiting / Other":
                counts[(origin, r["category"])] += 1
            raw.append(r)

    if not counts[("phone", "Phone Compute")]:
        raise ValueError("missing Phone native compute - worker may have 0 layers")
    if not counts[("pc", "GPU Compute")]:
        raise ValueError("missing PC CUDA device events - check CUDA build and tracing")
    if not counts[("pc", "RPC / Coordination")]:
        raise ValueError("missing PC ZeroMQ communication intervals")
    # PCIe can be zero for a particular Prefill; never invent source events.
    # Rank 0 may have zero CPU Graph time if entirely CUDA-assigned.

    projected, totals, overlap = projection(raw, t0, t1)
    envelope = t1 - t0
    normal = manifest.get("normal", [])
    if len(normal) < 3:
        raise ValueError("need 3 separately measured normal Prefill runs")
    normal_ms = statistics.median(x["prefill_ms"] for x in normal)
    norm_tokens = [x["tokens"] for x in normal]
    # Prima may print slightly different token conventions between main
    # chunk loop and internal model timing. Fail on substantial mismatches.
    if any(abs(n - prefill[0]["bytes"]) > 2 for n in norm_tokens):
        raise ValueError("normal/profile token counts differ; check templates")
    rows = []
    for cat in CATEGORIES:
        ns = totals[cat]
        rows.append(dict(category=cat, ms=round(ns / 1e6, 5),
                         pct=round(ns / envelope * 100, 5),
                         raw_overlap_ms=round(sum(
                             x["b"] - x["a"] for x in raw if x["category"] == cat) / 1e6, 5)))

    exclusive = [dict(start_ms=round((x["a"] - t0)/1e6, 6),
                      end_ms=round((x["b"] - t0)/1e6, 6),
                      duration_ms=round((x["b"] - x["a"])/1e6, 6),
                      category=x["category"]) for x in projected]
    source_rows = [dict(start_ms=round((x["a"] - t0)/1e6, 6),
                        end_ms=round((x["b"] - t0)/1e6, 6),
                        duration_ms=round((x["b"] - x["a"])/1e6, 6),
                        rank=x["rank"], origin=x["origin"],
                        category=x["category"], kind=x["kind"],
                        bytes=x["bytes"], quality=x["detail"]) for x in raw]
    write_csv(root/"fig3_prima_source_events.csv", source_rows,
              ("start_ms","end_ms","duration_ms","rank","origin","category","kind","bytes","quality"))
    write_csv(root/"fig3_prima_timeline.csv", exclusive,
              ("start_ms","end_ms","duration_ms","category"))
    write_csv(root/"fig3_prima_breakdown.csv", rows,
              ("category","ms","pct","raw_overlap_ms"))
    waiting_intervals = [x for x in projected if x["category"] == "Waiting / Other"]
    diagnostics = {}
    for cat in DIAGNOSTICS:
        diagnostics[cat] = round(sum(
            max(0, min(a["b"], d["b"]) - max(a["a"], d["a"]))
            for a in waiting_intervals for d in raw if d["category"] == cat)/1e6, 4)
    # Raw Host Prepare, Host Transfer and Prefetch may themselves overlap;
    # diagnostics are non-exclusive and must not be summed.
    cuda_approx = [x for x in raw if x["category"] in ("GPU Compute", "PCIe Copy") and
                   x["detail"] == "DEVICE_DURATION_SYNC_UPPER_BOUND"]
    fidelity = ("PROVISIONAL_GPU_EPOCH" if cuda_approx else "REQUIRES_GPU_QUALITY_REVIEW")
    summary = {
        "baseline": "Prima", "run_id": run_id, "model": manifest["model"],
        "normal_prefill_ms": round(normal_ms, 3),
        "normal_prefill_tps": round(prefill[0]["bytes"] * 1000 / normal_ms, 4),
        "profile_envelope_ms": round(envelope / 1e6, 3),
        "tokens": prefill[0]["bytes"], "clock_samples": len(samples),
        "clock_best_uncertainty_ms": round(uncertainty/1e6, 4),
        "clock_offset_spread_ms": round(spread/1e6, 4),
        "full_coverage": sum(totals.values()) == envelope,
        "overlap_ms": round(overlap/1e6, 3),
        "source_event_count": len(source_rows),
        "exclusive_segments": len(exclusive),
        "event_counts": {f"{k[0]}:{k[1]}": v for k, v in counts.items()},
        "waiting_diagnostics_nonexclusive_ms": diagnostics,
        "gpu_time_placement": "CUDA device durations, host upper-bound sync-anchor, NOT exact device absolute timestamps",
        "paper_fidelity": fidelity,
        "categories": rows,
        "accounting": "priority-exclusive projection, not causal critical path",
    }
    (root/"fig3_prima_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    selected = {row["category"]: row["pct"] for row in rows}
    paper = {
        "baseline": "Prima", "run_id": run_id,
        "normal_prefill_ms": summary["normal_prefill_ms"],
        "profile_envelope_ms": summary["profile_envelope_ms"],
        "tokens": summary["tokens"], "phone_compute_pct": selected["Phone Compute"],
        "gpu_compute_pct": selected["GPU Compute"],
        "cpu_compute_pct": selected["CPU Compute"],
        "pcie_copy_pct": selected["PCIe Copy"],
        "rpc_coordination_pct": selected["RPC / Coordination"],
        "waiting_other_pct": selected["Waiting / Other"],
        "paper_fidelity": fidelity,
    }
    write_csv(root/"fig3_prima_paper_row.csv", [paper], tuple(paper))
    print("[PRIMA_FIG3] source events", len(source_rows),
          "100%-coverage", summary["full_coverage"],
          "GPU timestamp fidelity", fidelity)
    for row in rows:
        print("  %-21s %9.3f ms %8.4f%%" %
              (row["category"], row["ms"], row["pct"]))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pc", required=True)
    ap.add_argument("--phone", required=True)
    ap.add_argument("--out", required=True)
    inspect(ap.parse_args())


if __name__ == "__main__":
    main()
