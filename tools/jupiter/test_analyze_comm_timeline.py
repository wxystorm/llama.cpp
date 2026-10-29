#!/usr/bin/env python3
"""Offline regression checks for Jupiter communication profiling reports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analyze_comm_timeline import (
    graph_coverage,
    metric_summary,
    read_timing_events,
    report,
    select_payload_rows,
)


SAMPLE = """
0.20.100 I [JUPITER_UBATCH_TIMELINE] event=BEGIN chunk=0 slot=0 prefix=0 tokens=128 t_us=500
0.20.120 I [JUPITER_C2P_STAGE] task=9 bytes=2621440 src=l_out dst=copy backend=CUDA0 payload_alloc_ms=1.000 gpu_sync_ms=80.000 device_read_ms=17.000 enqueue_wait_ms=2.000 stage_ms=100.000 t_us=1000000
0.20.130 I [JUPITER_C2P_SEND] task=9 bytes=2621440 queue_ms=2.000 socket_send_ms=5.000 t_us=1010000 status=OK
0.20.140 I [JUPITER_C2P_STAGE] task=12 bytes=40960 src=tiny dst=copy backend=CUDA0 payload_alloc_ms=1.000 gpu_sync_ms=1.000 device_read_ms=1.000 enqueue_wait_ms=0.000 stage_ms=3.000 t_us=1020000
0.20.150 I [JUPITER_TRANSFER_FENCE] task=10 uid=123 phase=GRAPH wait_ms=11.500 t_us=1030000 status=OK
0.20.160 I [JUPITER_GRAPH_DISPATCH] task=10 uid=123 slot=0 reused=0 queue_ms=5.000 transfer_fence_ms=11.500 graph_send_ms=1.200 t_us=1040000 status=OK
0.20.170 I [JUPITER_GRAPH_HOST] tokens=128 reused=0 host_prepare_ms=70.000 graph_build_ms=40.000 graph_alloc_ms=20.000 set_inputs_ms=5.000 submit_ms=11.000 other_prepare_ms=5.000 t_us=1050000 status=0
0.20.180 I [JUPITER_BUFFER_DRAIN] reason=FREE bytes=4096 buffer=1234 wait_ms=99.000 t_us=1060000
"""
PHONE = """
0.20.300 I [JUPITER_PHONE_TIMELINE] uid=123 reused=0 device=0 nodes=200 graph_prepare_ms=1.200 phone_compute_ms=60.000 wall_ms=61.200 phone_t_us=9000
"""


class AnalyzeTimelineTests(unittest.TestCase):
    def read(self, content: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.log"
            path.write_text(content, encoding="utf-8")
            return read_timing_events(path)

    def test_parse_and_join_by_task(self) -> None:
        events = self.read(SAMPLE)
        rows = select_payload_rows(events, 1 << 20)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task"], "9")
        self.assertEqual(rows[0]["backend"], "CUDA0")
        self.assertEqual(rows[0]["socket_send_ms"], "5.000")
        self.assertEqual(rows[0]["post_stage_to_send_ms"], "10.000")

    def test_min_bytes_zero_includes_small_activation(self) -> None:
        rows = select_payload_rows(self.read(SAMPLE), 0)
        self.assertEqual([r["task"] for r in rows], ["9", "12"])
        self.assertEqual(rows[1]["socket_send_ms"], "")

    def test_phone_matching_never_uses_absolute_clock(self) -> None:
        pc = self.read(SAMPLE)
        phone = self.read(PHONE)
        self.assertEqual(graph_coverage(pc, phone), (1, 1))
        rows = report(pc, phone, 1 << 20)
        self.assertEqual(len(rows), 1)

    def test_negative_filter_rejected(self) -> None:
        with self.assertRaises(ValueError):
            report(self.read(SAMPLE), [], -1)

    def test_empty_timing_and_missing_fields(self) -> None:
        self.assertEqual(self.read("log without timeline data"), [])
        self.assertEqual(metric_summary([]), "count=0")
        self.assertIn("count=2", metric_summary([5.0, 7.0]))


if __name__ == "__main__":
    unittest.main()
