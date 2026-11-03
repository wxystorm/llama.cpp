#!/usr/bin/env python3
"""Offline audit of Region Ring Planner versus one measured Prefill.

No model, RPC socket or inference process is opened. This script intentionally
reports prediction residuals rather than correcting a DAG wall time by adding
components which may overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def latest_json_record(path: Path, kind: str, tokens: int | None) -> dict[str, Any]:
    if not path.exists():
        return {}
    latest: dict[str, Any] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            if f'"type":"{kind}"' not in raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if row.get("type") != kind:
                continue
            row_tokens = row.get("tokens", row.get("plan_tokens"))
            if tokens is not None and row_tokens is not None:
                if number(row_tokens) != float(tokens):
                    continue
            latest = row
    return latest


def last_tag(path: Path, tag: str, **required: str) -> dict[str, str]:
    if not path.exists():
        return {}
    latest: dict[str, str] = {}
    pattern = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)=([^\s]+)")
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if tag not in line:
                continue
            fields = dict(pattern.findall(line))
            if all(fields.get(k) == v for k, v in required.items()):
                latest = fields
    return latest


def extract(run_dir: Path, case: str, rpc_log: Path | None) -> dict[str, Any]:
    summary_rows = load_csv(run_dir / "summary.csv")
    if not summary_rows:
        raise ValueError(f"summary.csv missing/empty in {run_dir}")
    # The suite may contain multiple prompts. Prefer a matching prompt name,
    # then the latest row with valid measured Prefill tokens.
    matches = [r for r in summary_rows if case in str(r.get("prompt", ""))]
    selected = (matches or [r for r in summary_rows if number(r.get("actual_prefill_tokens")) is not None] or summary_rows)[-1]
    measured_tokens = number(selected.get("actual_prefill_tokens"))
    tokens = int(measured_tokens) if measured_tokens else None
    case_dir = run_dir / case
    metrics_file = case_dir / "metrics.jsonl"
    log_file = case_dir / "engine.stderr.log"
    planner = latest_json_record(metrics_file, "planner", tokens)
    runtime = latest_json_record(metrics_file, "region_ring", tokens)
    setup = last_tag(log_file, "[PHONE_AUTO] event=GRAPH_SETUP_DONE")
    receiver = last_tag(log_file, "[P2C_STREAM] phase=STOP", status="OK")
    phone = last_tag(rpc_log, "[PHONE_AUTO_SUM]", status="PASS") if rpc_log else {}
    phone_input = last_tag(rpc_log, "[PHONE_INPUT_STREAM_SUM]", status="END") if rpc_log else {}
    phone_p2c = (last_tag(rpc_log, "[P2C_STREAM] side=PHONE phase=STOP", status="OK") if rpc_log else {})

    ring_wall = number(runtime.get("wall_ms")) or number(selected.get("ring_wall_ms"))
    predicted = number(planner.get("predicted_ms")) or number(selected.get("final_dag_sim_ms"))
    graph_count = number(setup.get("count"))
    graph_ms = number(setup.get("setup_ms"))
    head_ms = number(runtime.get("phone_final_head_ms")) or number(selected.get("phone_final_head_ms"))
    actual_cpu = number(runtime.get("cpu_compute_range_ms")) or number(selected.get("cpu_compute_range_ms"))
    actual_gpu = number(runtime.get("gpu_compute_range_ms")) or number(selected.get("gpu_compute_range_ms"))
    actual_phone = number(phone.get("compute_ms"))
    result: dict[str, Any] = {
        "status": selected.get("status"),
        "commit": selected.get("commit"),
        "tokens": tokens,
        "layout": selected.get("runtime_layout") or runtime.get("layout"),
        "ring_wall_ms": ring_wall,
        "recorded_planner_predicted_ms": predicted,
        "recorded_prediction_error_ms": (
            ring_wall - predicted if ring_wall is not None and predicted is not None else None
        ),
        "model_note": "Recorded prediction is from its own commit; structural fixes require a new Planner DAG run.",
        "cpu_range_measured_ms": actual_cpu,
        "gpu_range_measured_ms": actual_gpu,
        "phone_compute_measured_ms": actual_phone,
        "cpu_busy_pred_ms": number(planner.get("cpu_busy_pred_ms")),
        "gpu_busy_pred_ms": number(planner.get("gpu_busy_pred_ms")),
        "phone_busy_pred_ms": number(planner.get("phone_busy_pred_ms")),
        "phone_auto_graph_count": graph_count,
        "phone_auto_setup_measured_ms": graph_ms,
        "phone_auto_setup_per_graph_measured_ms": (
            graph_ms / graph_count if graph_ms is not None and graph_count and graph_count > 0 else None
        ),
        "phone_auto_head_measured_ms": head_ms,
        "p2c_frames_pc": number(receiver.get("recv")),
        "p2c_frames_phone": number(phone_p2c.get("frames")),
        "p2c_send_measured_ms": number(phone_p2c.get("send_ms")),
        "phone_input_recv_ms": number(phone_input.get("recv_ms")),
        "phone_input_recv_caution": "Blocking recv_data duration, not link-only service time.",
    }
    return result


def display(result: dict[str, Any]) -> str:
    lines = ["Region Ring offline calibration audit", "=" * 36]
    for key, value in result.items():
        if key not in {"model_note", "phone_input_recv_caution"}:
            lines.append(f"{key}: {value}")
    lines.extend(["", result["model_note"], result["phone_input_recv_caution"]])
    setup = result.get("phone_auto_setup_per_graph_measured_ms")
    head = result.get("phone_auto_head_measured_ms")
    if setup is not None and head is not None:
        lines.extend([
            "", "Optional PowerShell calibration for NEXT Planner run (no network factors applied):",
            f'$env:LLAMA_REGION_RING_PLAN_PHONE_AUTO_SETUP_PER_GRAPH_MS = "{setup:.3f}"',
            f'$env:LLAMA_REGION_RING_PLAN_PHONE_AUTO_HEAD_MS = "{head:.3f}"',
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Existing run folder containing summary.csv")
    parser.add_argument("--case", default="prompt_358",
                        help="Run subfolder containing metrics.jsonl")
    parser.add_argument("--rpc-log", type=Path, default=None,
                        help="Optional copied Phone rpc.log")
    parser.add_argument("--json", action="store_true", help="Machine readable output")
    args = parser.parse_args()
    report = extract(args.run, args.case, args.rpc_log)
    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else display(report))


if __name__ == "__main__":
    main()