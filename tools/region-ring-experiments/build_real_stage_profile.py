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


def extract(cases: list[Path], rpc_log: Path | None) -> tuple[list[dict], dict]:
    # Phone's RPC pull is logged into the matching PC attempt. An archived
    # Phone log remains an optional fallback, never an obligatory copy.
    def collect_phone(path: Path) -> dict:
        result = {}
        for x in profile_lines(path, "[REAL_STAGE_PROFILE]"):
            if x.get("side") != "PHONE" or x.get("phase") != "CHUNK" or x.get("status") != "PASS":
                continue
            try:
                key = (x["run"], int(x["region"]), int(x["chunk"]))
                values = [number(x.get(field)) for field in
                          ("upload_ms", "prepare_ms", "compute_ms")]
                if all(v is not None and v >= 0 for v in values) and values[2] > 0:
                    result[key] = x
            except (KeyError, ValueError, TypeError):
                pass
        return result

    phone = collect_phone(rpc_log) if rpc_log is not None else {}
    samples = []
    diag = dict(cases=0, cpu=0, gpu=0, phone=0,
                phone_missing=0, setup_missing=0, trace_missing=0)
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
        log = case / "engine.stderr.log"
        session_lines = (profile_lines(log, "[PHONE_AUTO] event=SESSION_READY") or
                         profile_lines(log, "[PHONE_AUTO] event=SESSION_PREPARING"))
        session = session_lines[0].get("run") if session_lines else None
        local_phone = collect_phone(log)
        setup = {}
        for p in profile_lines(case / "engine.stderr.log",
                               "[REAL_STAGE_PROFILE]"):
            if p.get("side") == "PC" and p.get("phase") == "GRAPH_SETUP":
                try:
                    setup[(int(p["region"]), int(p["chunk"]))] = p
                except (KeyError, ValueError):
                    continue
        # PHONE is terminal only if the *whole* plan ends on Phone.
        # HTP-only auto plans have a final GPU tail; the last NPU Region
        # is therefore nonterminal even though it is the last Phone one.
        final_region = max((int(t["region"]) for t in traces), default=-1)
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
                key = (session, region, chunk)
                p = local_phone.get(key) or phone.get(key)
                q = setup.get((region, chunk))
                if p is None:
                    diag["phone_missing"] += 1
                    continue
                val = [number(p.get(field)) for field in
                       ("upload_ms", "prepare_ms", "compute_ms")]
                if any(v is None or v < 0 for v in val) or val[2] <= 0:
                    continue
                # In Graphless HTP mode PC GRAPH_SETUP may be absent.
                # Preserve measured compute without fabricating setup cost.
                registration = number(q.get("setup_ms")) if q else None
                if registration is None:
                    diag["setup_missing"] += 1
                try:
                    kv = int(q.get("kv_tokens", begin + tokens)) if q else begin + tokens
                    terminal = int(q.get("terminal", int(region == final_region))) if q else int(region == final_region)
                except ValueError:
                    continue
                s.update(upload_ms=val[0], prepare_ms=val[1],
                         compute_ms=val[2], setup_ms=registration or 0.0,
                         kv_tokens=kv, terminal=terminal)
            diag[d.lower()] += 1
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, diag = extract(case_dirs(args.run), args.rpc_log)
    points = aggregate(raw)
    if not points:
        parser.error("no real Stage rows; enable task trace and Stage logging")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(points)
    shapes = {d: len({(p["layers"], p["tokens"]) for p in points
                      if p["device"] == d}) for d in ("CPU", "GPU", "PHONE")}
    print(f"[REAL_STAGE_PROFILE] points={len(points)} shapes={shapes} "
          f"raw={len(raw)} diagnostic={diag} file={args.output}")
    if diag["phone_missing"] or diag["setup_missing"]:
        print("[REAL_STAGE_PROFILE] WARNING: incomplete Phone records; "
              "Planner will fall back for unsupported shapes")
    print('$env:LLAMA_REGION_RING_REAL_STAGE_PROFILE_CSV = '
          f'"{args.output.resolve()}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
