#!/usr/bin/env python3
"""Extract real Ring Stage service costs from recorded PC and Phone runs.

No inference, model loading, RPC or background work is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
from statistics import median

SCHEMA = ["device", "layers", "tokens", "kv_tokens", "setup_ms",
          "upload_ms", "prepare_ms", "compute_ms", "terminal", "samples"]
FIELDS = re.compile(r"([a-z_]+)=([^\s]+)")


def parse_line(line: str) -> dict[str, str]:
    return dict(FIELDS.findall(line))


def number(value) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def case_dirs(roots: list[Path]) -> list[Path]:
    result: set[Path] = set()
    for root in roots:
        if (root / "metrics.jsonl").exists():
            result.add(root)
        elif root.is_dir():
            result.update(x.parent for x in root.rglob("metrics.jsonl"))
    return sorted(result)


def profile_lines(path: Path, substring: str) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    return [parse_line(line) for line in
            path.read_text(encoding="utf-8", errors="replace").splitlines()
            if substring in line]


def extract(cases: list[Path], rpc_log: Path | None, mixed: bool = False) -> tuple[list[dict], dict]:
    phone = {}
    if rpc_log is not None:
        for x in profile_lines(rpc_log, "[REAL_STAGE_PROFILE]"):
            if x.get("side") == "PHONE" and x.get("phase") == "CHUNK":
                if x.get("status") == "PASS":
                    try:
                        phone[(x["run"], int(x["region"]), int(x["chunk"]))] = x
                    except (KeyError, ValueError):
                        pass
    samples = []
    diag = dict(cases=0, cpu=0, gpu=0, phone=0, phone_gpu=0,
                phone_missing=0, setup_missing=0, trace_missing=0,
                backend_missing=0)
    for case in cases:
        # Only finished PASS probes are valid anchors for ranking. A failed
        # run can contain plausible-looking partial Chunk timing records.
        result_file = case / "result.json"
        if result_file.is_file():
            try:
                if json.loads(result_file.read_text(
                        encoding="utf-8")).get("status") != "PASS":
                    continue
            except (ValueError, OSError):
                continue
        traces = []
        for line in (case / "metrics.jsonl").read_text(
                encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("type") == "region_ring_chunk_trace":
                traces.append(row)
        if not traces:
            diag["trace_missing"] += 1
            continue
        diag["cases"] += 1
        # Match the PC-run session ID to the Phone autonomous job table;
        # otherwise a stale Phone log could contaminate the Profile.
        session_lines = profile_lines(case / "engine.stderr.log",
                                      "[PHONE_AUTO] event=SESSION_READY")
        session = session_lines[0].get("run") if session_lines else None
        setup = {}
        for p in profile_lines(case / "engine.stderr.log",
                               "[REAL_STAGE_PROFILE]"):
            if p.get("side") == "PC" and p.get("phase") == "GRAPH_SETUP":
                try:
                    setup[(int(p["region"]), int(p["chunk"]))] = p
                except (KeyError, ValueError):
                    continue
        final_region = max((int(t["region"]) for t in traces
                            if t.get("device") == "PHONE"), default=-1)
        for t in traces:
            d = t.get("device")
            if d not in ("CPU", "GPU", "PHONE"):
                continue
            try:
                region, chunk = int(t["region"]), int(t["chunk"])
                depth = int(t["layer_end"]) - int(t["layer_begin"])
                tokens, begin = int(t["tokens"]), int(t["token_begin"])
            except (KeyError, ValueError, TypeError):
                continue
            if depth <= 0 or tokens <= 0:
                continue
            s = dict(device=d, layers=depth, tokens=tokens,
                     kv_tokens=begin + tokens, setup_ms=0.0,
                     upload_ms=0.0, prepare_ms=0.0,
                     compute_ms=0.0, terminal=0)
            if d in ("CPU", "GPU"):
                a = number(t.get("prepare_total_ms"))
                b = number(t.get("compute_range_ms"))
                if a is None or b is None or b <= 0:
                    continue
                s.update(prepare_ms=a, compute_ms=b)
            else:
                p = phone.get((session, region, chunk))
                q = setup.get((region, chunk))
                if p is None:
                    diag["phone_missing"] += 1
                    continue
                backend = p.get("backend")
                if mixed:
                    # Preserve distinct Phone HTP and Adreno GPU cost
                    # classes: otherwise the four-backend Planner would
                    # interpolate NPU compute times for GPU candidates.
                    if backend == "HTP0":
                        s["device"] = "PHONE"
                    elif backend == "GPUOpenCL":
                        s["device"] = "PHONE_GPU"
                    else:
                        diag["backend_missing"] += 1
                        continue
                if q is None and not (mixed and backend == "HTP0"):
                    diag["setup_missing"] += 1
                    continue
                val = [number(p.get(k)) for k in
                       ("upload_ms", "prepare_ms", "compute_ms")]
                # Graphless HTP has no OpenCL Slot Prepare: -1 is a log
                # sentinel, not a negative service time.
                if mixed and backend == "HTP0" and p.get("prepare_ms") == "-1.000":
                    val[1] = 0.0
                registration = number(q.get("setup_ms")) if q else 0.0
                if registration is None or any(v is None for v in val):
                    continue
                try:
                    kv = int(q.get("kv_tokens", begin + tokens)) if q else begin + tokens
                    terminal = int(q.get("terminal", int(region == final_region))) if q else int(region == final_region)
                except ValueError:
                    continue
                s.update(upload_ms=val[0], prepare_ms=val[1],
                         compute_ms=val[2], setup_ms=registration,
                         kv_tokens=kv, terminal=terminal)
            diag[s["device"].lower()] += 1
            samples.append(s)
    return samples, diag


def aggregate(samples: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for sample in samples:
        key = (sample["device"], sample["layers"], sample["tokens"],
               sample["kv_tokens"], sample["terminal"])
        grouped[key].append(sample)
    result = []
    for (device, layers, tokens, kv, terminal), arr in sorted(grouped.items()):
        row = dict(device=device, layers=layers, tokens=tokens,
                   kv_tokens=kv, terminal=terminal, samples=len(arr))
        for field in ("setup_ms", "upload_ms", "prepare_ms", "compute_ms"):
            row[field] = round(median(x[field] for x in arr), 6)
        result.append(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--rpc-log", type=Path)
    parser.add_argument("--mixed", action="store_true",
                        help="Require HTP0/GPUOpenCL labels and keep PHONE_GPU distinct")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, diag = extract(case_dirs(args.run), args.rpc_log, mixed=args.mixed)
    points = aggregate(raw)
    if not points:
        parser.error("no real Stage rows; enable task trace and Stage logging")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(points)
    shapes = {d: len({(p["layers"], p["tokens"]) for p in points
                      if p["device"] == d}) for d in
              (("CPU", "GPU", "PHONE", "PHONE_GPU") if args.mixed else
               ("CPU", "GPU", "PHONE"))}
    print(f"[REAL_STAGE_PROFILE] points={len(points)} shapes={shapes} "
          f"raw={len(raw)} diagnostic={diag} file={args.output}")
    if diag["phone_missing"] or diag["setup_missing"] or diag["backend_missing"]:
        print("[REAL_STAGE_PROFILE] WARNING: incomplete Phone records; "
              "Planner will fall back for unsupported shapes")
    if not args.mixed:
        print('$env:LLAMA_REGION_RING_REAL_STAGE_PROFILE_CSV = '
              f'"{args.output.resolve()}"')
    else:
        print("[REAL_STAGE_PROFILE] mode=MIXED_OFFLINE_ONLY; do not pass the "
              "PHONE_GPU CSV to the legacy three-device runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
