#!/usr/bin/env python3
"""No-model synthetic Fig3 audit tests. Run: python3 tools/fig3/test_fig3.py"""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from analyze_fig3 import (inspect, projection, calibrate, clock_mapping,
                          normalized_category, PHONE_META_WAIT)


def row(run, rank, start, end, category, kind, tokens=0):
    return [run, rank, 1000 + rank, start, end, category, kind, tokens, "unit"]


def samples(drift=0):
    data = []
    for phase, add in (("before", 0), ("after", drift)):
        center = 9000000000 if phase == "before" else 12000000000
        for i in range(16):
            data.append({
                "phase": phase, "offset_phone_minus_pc_ns": 5000000 + add,
                "uncertainty_ns": 100000 + i * 100,
                "pc_t0": center - 100000,
                "pc_t3": center + 100000,
                "phone_t1": center + 5000000 + add - 100000,
                "phone_t2": center + 5000000 + add + 100000,
            })
    return data


class Fig3Test(unittest.TestCase):
    def test_projection(self):
        source = [
            dict(a=0, b=30, category="GPU Compute"),
            dict(a=10, b=20, category="Phone Compute"),
            dict(a=40, b=60, category="RPC / Coordination"),
        ]
        timeline, counts, overlap = projection(source, 0, 100)
        self.assertEqual(sum(counts.values()), 100)
        self.assertEqual(counts["Phone Compute"], 10)
        self.assertEqual(counts["GPU Compute"], 20)
        self.assertEqual(counts["RPC / Coordination"], 20)
        self.assertEqual(overlap, 10)
        self.assertEqual(timeline[0]["a"], 0)
        self.assertEqual(timeline[-1]["b"], 100)

    def test_phone_blocking_meta_is_not_network(self):
        self.assertEqual(normalized_category("phone", "ZMQ_RECV_META",
                                             "Coordination"), PHONE_META_WAIT)
        self.assertEqual(normalized_category("pc", "ZMQ_RECV_TENSORS",
                                             "Coordination"),
                         "RPC / Coordination")
        self.assertEqual(normalized_category("phone", "CPU_GRAPH",
                                             "Phone Compute"), "Phone Compute")
        source = [
            dict(a=0, b=100, category=PHONE_META_WAIT),
            dict(a=10, b=40, category="GPU Compute"),
            dict(a=55, b=75, category="Phone Compute"),
        ]
        _, counts, _ = projection(source, 0, 100)
        self.assertEqual(counts["RPC / Coordination"], 0)
        self.assertEqual(counts["GPU Compute"], 30)
        self.assertEqual(counts["Phone Compute"], 20)
        self.assertEqual(counts["Waiting / Other"], 50)

    def test_clock_drift(self):
        self.assertEqual(calibrate(samples())[0], 5000000)
        with self.assertRaisesRegex(ValueError, "drift"):
            calibrate(samples(30000000))

    def test_two_point_diagnostic_recovery(self):
        readings = samples(1514126000)
        # Keep the real PC elapsed / phone elapsed monotonic for this fixture.
        for v in readings:
            if v["phase"] == "after":
                v["pc_t0"] = 173733000000 - 100000
                v["pc_t3"] = 173733000000 + 100000
                v["phone_t1"] = 173733000000 + 5000000 + 1514126000 - 100000
                v["phone_t2"] = 173733000000 + 5000000 + 1514126000 + 100000
        with self.assertRaisesRegex(ValueError, "drift"):
            clock_mapping(readings, "strict", 10000000000, 150000000000)
        mapper, detail = clock_mapping(
            readings, "affine-diagnostic", 10000000000, 150000000000)
        self.assertEqual(mapper(9005000000), 9000000000)
        self.assertEqual(mapper(175252126000), 173733000000)
        self.assertEqual(detail["anchor_count"], 2)
        self.assertIn("ASSUMED", detail["mode"])

    def test_dense_clock_diagnostic_flags_real_anomaly(self):
        readings = samples(200_000_000)
        # before PC=9s, after PC=21s: 12s span with dense 1s anchors.
        # Phone clock offset has a 200ms discontinuity at PC=16s.
        for row in readings:
            if row["phase"] == "after":
                row["pc_t0"] = 21_000_000_000 - 100_000
                row["pc_t3"] = 21_000_000_000 + 100_000
                row["phone_t1"] = 21_000_000_000 + 205_000_000 - 100_000
                row["phone_t2"] = 21_000_000_000 + 205_000_000 + 100_000
        for second in range(10, 21):
            center = second * 1_000_000_000
            offset = 205_000_000 if second >= 16 else 5_000_000
            readings.append({
                "phase": "during", "uncertainty_ns": 100_000,
                "pc_t0": center - 100_000,
                "pc_t3": center + 100_000,
                "phone_t1": center + offset - 100_000,
                "phone_t2": center + offset + 100_000,
            })
        with self.assertRaisesRegex(ValueError, "clock discontinuity"):
            clock_mapping(readings, "strict", 10_000_000_000, 20_000_000_000)
        mapper, quality = clock_mapping(
            readings, "piecewise-diagnostic", 10_000_000_000, 20_000_000_000)
        self.assertEqual(quality["clock_anomaly_count"], 1)
        self.assertIn("NOT_VERIFIED", quality["mode"])
        self.assertEqual(mapper(10_005_000_000), 10_000_000_000)
        self.assertEqual(mapper(17_205_000_000), 17_000_000_000)

    def test_synthetic_full(self):
        run = "synthetic-prima"
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pcfile, phonefile = base/"pc.csv", base/"phone.csv"
            pcrows = [
                row(run, 0, 10000000000, 11000000000, "Boundary", "PREFILL", 1041),
                row(run, 0, 10000000000, 10200000000, "GPU Compute", "CUDA_GRAPH"),
                row(run, 0, 10050000000, 10060000000, "PCIe Copy", "H2D"),
                row(run, 0, 10200000000, 10400000000, "Coordination", "ZMQ_RECV_TENSORS"),
                row(run, 0, 10400000000, 10420000000, "CPU Compute", "CPU_GRAPH"),
                row(run, 0, 10200000000, 10300000000, "Host Prepare", "GRAPH_BUILD"),
            ]
            phonerows = [
                row(run, 1, 10250000000 + 5000000, 10550000000 + 5000000,
                    "Phone Compute", "CPU_GRAPH"),
            ]
            for target, rows in ((pcfile, pcrows), (phonefile, phonerows)):
                with target.open("w", newline="") as handle:
                    csv.writer(handle).writerows(rows)
            manifest = base/"manifest.json"
            layout = {"effective": {
                "rank": 0, "world": 2, "lw0": 39, "lw1": 1,
                "ngl": 21, "prefetch": 1}, "planner_stages": []}
            manifest.write_text(json.dumps({
                "run_id": run, "model": "Qwen3-14B",
                "placement_mode": "FIXED_LAYOUT_REPLAY",
                "clock_samples": samples(),
                "profile": {"layout": layout},
                "warmup": [{"layout": layout}],
                "normal": [{"prefill_ms": 870, "tokens": 1041, "layout": layout},
                           {"prefill_ms": 890, "tokens": 1041, "layout": layout},
                           {"prefill_ms": 900, "tokens": 1041, "layout": layout}]
            }))
            args = SimpleNamespace(manifest=str(manifest), pc=str(pcfile),
                                   phone=str(phonefile), out=str(base/"out"))
            audit = inspect(args)
            self.assertTrue(audit["full_coverage"])
            self.assertEqual(audit["tokens"], 1041)
            self.assertEqual(audit["normal_prefill_ms"], 890)
            self.assertGreater(audit["overlap_ms"], 0)
            self.assertEqual(len(audit["categories"]), 6)
            self.assertTrue((base/"out/fig3_prima_paper_row.csv").exists())
            self.assertTrue((base/"out/fig3_prima_source_events.csv").exists())
            self.assertTrue((base/"out/fig3_prima_timeline.csv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
