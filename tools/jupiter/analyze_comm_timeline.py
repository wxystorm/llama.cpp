#!/usr/bin/env python3
"""Summarize opt-in Jupiter PC/Phone timing without assuming synchronized clocks.

PC log: a run_prefill_bench.py run with --comm-timing.
Phone log (optional): RPC server started with LLAMA_JUPITER_TIMELINE=1.
All per-device timestamps use their local monotonic clock; never subtract
a PC timestamp from a Phone timestamp.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Iterable


EVENT = re.compile(r"\[(JUPITER_[A-Z0-9_]+)\]\s*(.*)")
FIELD = re.compile(r"([a-zA-Z_][a-zA-Z_0-9]*)=([^\s]+)")

METRIC_FIELDS = {
    "JUPITER_C2P_STAGE": (
        "payload_alloc_ms", "gpu_sync_ms", "device_read_ms",
        "enqueue_wait_ms", "stage_ms",
    ),
    "JUPITER_C2P_SYNC_FALLBACK": (
        "sync_wait_ms", "copy_ms", "total_ms",
    ),
    "JUPITER_C2P_SEND": ("queue_ms", "socket_send_ms"),
    "JUPITER_TRANSFER_FENCE": ("wait_ms",),
    "JUPITER_GRAPH_PREPARE": ("control_fence_ms", "graph_prepare_ms"),
    "JUPITER_GRAPH_DISPATCH": (
        "queue_ms", "transfer_fence_ms", "graph_send_ms",
    ),
    "JUPITER_BUFFER_DRAIN": ("wait_ms",),
    "JUPITER_REMOTE_SYNC": ("wait_ms",),
    "JUPITER_GRAPH_HOST": (
        "host_prepare_ms", "graph_build_ms", "graph_alloc_ms",
        "set_inputs_ms", "submit_ms", "other_prepare_ms",
    ),
    "JUPITER_PHONE_TIMELINE": (
        "graph_prepare_ms", "phone_compute_ms", "wall_ms",
    ),
}

STAGE_CSV_FIELDS = [
    "task", "bytes", "src", "dst", "backend",
    "payload_alloc_ms", "gpu_sync_ms", "device_read_ms",
    "enqueue_wait_ms", "stage_ms",
    "socket_queue_ms", "socket_send_ms",
    "stage_t_us", "send_t_us", "post_stage_to_send_ms",
]


def read_timing_events(path: Path) -> list[tuple[str, dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = EVENT.search(line)
        if match:
            result.append((match.group(1), dict(FIELD.findall(match.group(2)))))
    return result


def select_payload_rows(
    events: Iterable[tuple[str, dict[str, str]]],
    minimum_bytes: int,
) -> list[dict[str, str]]:
    """Join C2P staging and socket send by worker task, not by log position."""
    stages: dict[str, dict[str, str]] = {}
    sends: dict[str, dict[str, str]] = {}
    for name, values in events:
        task = values.get("task", "")
        if not task:
            continue
        if name == "JUPITER_C2P_STAGE":
            stages[task] = values
        elif name == "JUPITER_C2P_SEND":
            sends[task] = values

    out = []
    for task, values in stages.items():
        count = int(values.get("bytes", "0"))
        if count < minimum_bytes:
            continue
        send = sends.get(task, {})
        stage_us = values.get("t_us", "")
        send_us = send.get("t_us", "")
        delta = ""
        if stage_us and send_us:
            delta = f"{(int(send_us)-int(stage_us))/1000.0:.3f}"
        out.append({
            "task": task,
            "bytes": str(count),
            "src": values.get("src", ""),
            "dst": values.get("dst", ""),
            "backend": values.get("backend", ""),
            "payload_alloc_ms": values.get("payload_alloc_ms", ""),
            "gpu_sync_ms": values.get("gpu_sync_ms", ""),
            "device_read_ms": values.get("device_read_ms", ""),
            "enqueue_wait_ms": values.get("enqueue_wait_ms", ""),
            "stage_ms": values.get("stage_ms", ""),
            "socket_queue_ms": send.get("queue_ms", ""),
            "socket_send_ms": send.get("socket_send_ms", ""),
            "stage_t_us": stage_us,
            "send_t_us": send_us,
            "post_stage_to_send_ms": delta,
        })
    return sorted(out, key=lambda r: int(r["task"]))


def metric_summary(values: Iterable[float]) -> str:
    nums = list(values)
    if not nums:
        return "count=0"
    return (
        f"count={len(nums)} sum_ms={sum(nums):.3f} "
        f"median_ms={statistics.median(nums):.3f} max_ms={max(nums):.3f}"
    )


def sum_field(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


def graph_coverage(
    pc: list[tuple[str, dict[str, str]]],
    phone: list[tuple[str, dict[str, str]]],
) -> tuple[int, int]:
    """Coverage is by UID, not timestamps (clocks are not synchronized)."""
    pc_uid = {
        values.get("uid") for name, values in pc
        if name == "JUPITER_GRAPH_DISPATCH" and values.get("uid")
    }
    phone_uid = {
        values.get("uid") for name, values in phone
        if name == "JUPITER_PHONE_TIMELINE" and values.get("uid")
    }
    return len(pc_uid & phone_uid), len(pc_uid)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze Jupiter transport and graph timing logs")
    p.add_argument("--pc-log", type=Path, required=True)
    p.add_argument("--phone-log", type=Path, default=None)
    p.add_argument("--min-bytes", type=int, default=1024 * 1024,
                   help="filter tiny C2P transfers in the per-chunk table (default: 1 MiB)")
    p.add_argument("--csv", type=Path, default=None,
                   help="optional detailed C2P task CSV")
    return p


def report(
    pc: list[tuple[str, dict[str, str]]],
    phone: list[tuple[str, dict[str, str]]],
    minimum_bytes: int,
) -> list[dict[str, str]]:
    if minimum_bytes < 0:
        raise ValueError("--min-bytes must be nonnegative")

    payloads = select_payload_rows(pc, minimum_bytes)
    print(f"[JUPITER_TIMELINE_REPORT] pc_events={len(pc)} phone_events={len(phone)} "
          f"payload_tasks={len(payloads)} min_bytes={minimum_bytes}")

    if payloads:
        print("task   MiB    gpu_sync  device_read  socket_send  stage_ms  post_stage_to_send")
        for v in payloads:
            def show(key: str) -> str:
                return v.get(key) or "-"
            print(f"{v['task']:>4}  {int(v['bytes'])/1048576:>5.2f} "
                  f"{show('gpu_sync_ms'):>10} {show('device_read_ms'):>12} "
                  f"{show('socket_send_ms'):>12} {show('stage_ms'):>9} "
                  f"{show('post_stage_to_send_ms'):>19}")
        for field in ("payload_alloc_ms", "gpu_sync_ms", "device_read_ms",
                      "enqueue_wait_ms", "stage_ms", "socket_queue_ms",
                      "socket_send_ms"):
            print(f"[JUPITER_TIMELINE_METRIC] scope=large_c2p metric={field} "
                  + metric_summary(sum_field(payloads, field)))
    else:
        print("[JUPITER_TIMELINE_WARNING] no staged C2P tasks matched. "
              "Check --comm-timing, --async-c2p and --min-bytes.")

    for name, fields in METRIC_FIELDS.items():
        if name in ("JUPITER_C2P_STAGE", "JUPITER_C2P_SEND"):
            continue
        source = phone if name == "JUPITER_PHONE_TIMELINE" else pc
        relevant = [v for n, v in source if n == name]
        if not relevant:
            continue
        for field in fields:
            print(f"[JUPITER_TIMELINE_METRIC] scope={name} metric={field} "
                  + metric_summary(sum_field(relevant, field)))

    if phone:
        common, n_pc = graph_coverage(pc, phone)
        print(f"[JUPITER_TIMELINE_MATCH] common_graph_uids={common} pc_uids={n_pc} "
              "clock_domain=SEPARATE no_cross_machine_timestamp_subtraction=1")
    else:
        print("[JUPITER_TIMELINE_NOTE] Phone compute requires a Phone RPC "
              "server log from a binary built with timeline support, "
              "started with LLAMA_JUPITER_TIMELINE=1.")

    print("[JUPITER_TIMELINE_NOTE] sum_ms represents per-event service totals, "
          "not end-to-end latency; some stages overlap. Socket send completion "
          "is not the same as remote device-write completion.")
    return payloads


def main() -> None:
    args = build_parser().parse_args()
    pc = read_timing_events(args.pc_log)
    phone = read_timing_events(args.phone_log) if args.phone_log else []
    payloads = report(pc, phone, args.min_bytes)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=STAGE_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(payloads)
        print(f"[JUPITER_TIMELINE_CSV] path={args.csv} rows={len(payloads)}")


if __name__ == "__main__":
    main()
