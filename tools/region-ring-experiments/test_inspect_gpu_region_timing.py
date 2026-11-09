import json
from pathlib import Path
import tempfile
import unittest

import inspect_gpu_region_timing as mod


class InspectGpuRegionTimingTests(unittest.TestCase):
    def test_prefill_measurement_only_and_stage_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = [
                {"type": "region_ring_chunk_trace", "device": "GPU",
                 "region": 0, "tokens": 128, "compute_range_ms": 999.0},
                {"type": "measurement_start"},
                {"type": "region_ring_chunk_trace", "device": "GPU",
                 "region": 0, "chunk": 0, "tokens": 128,
                 "layer_begin": 0, "layer_end": 9,
                 "compute_range_ms": 230.0,
                 "lane_compute_begin_ms": 100.0, "lane_compute_end_ms": 170.0,
                 "split_prepare_ms": 20.0, "split_copy_call_ms": 5.0},
                {"type": "region_ring_chunk_trace", "device": "GPU",
                 "region": 0, "chunk": 1, "tokens": 128,
                 "layer_begin": 0, "layer_end": 9,
                 "compute_range_ms": 180.0,
                 "lane_compute_begin_ms": 300.0, "lane_compute_end_ms": 370.0,
                 "split_prepare_ms": 10.0, "split_copy_call_ms": 3.0},
                {"type": "region_ring_chunk_trace", "device": "GPU",
                 "region": 2, "chunk": 0, "tokens": 128,
                 "layer_begin": 10, "layer_end": 19,
                 "compute_range_ms": 80.0,
                 "lane_compute_begin_ms": 500.0, "lane_compute_end_ms": 575.0,
                 "split_prepare_ms": 3.0},
                {"type": "region_ring_bank_trace", "region": 4,
                 "source_region": 0, "queue_wait_ms": 0.01,
                 "total_ms": 650.0, "enqueue_ms": 300.0,
                 "begin_ms": 300.01, "done_ms": 950.01},
                {"type": "region_ring_chunk_trace", "device": "GPU",
                 "region": 2, "tokens": 1, "compute_range_ms": 100.0},
            ]
            (folder / "metrics.jsonl").write_text(
                "\n".join(json.dumps(x) for x in rows), encoding="utf-8")
            (folder / "result.json").write_text(
                json.dumps({"actual_prefill_tokens": 256, "prefill_tps": 150.0}),
                encoding="utf-8")
            groups, bank, result = mod.summarize(folder, "test")
            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[0]["chunks"], 2)
            self.assertEqual(groups[0]["range_ms"], 410)
            self.assertEqual(groups[0]["lane_ms"], 140)
            self.assertEqual(groups[0]["prepare_ms"], 30)
            self.assertEqual(groups[0]["remaining_after_prepare_ms"], 240)
            self.assertEqual(groups[0]["copy_call_ms"], 8)
            self.assertEqual(groups[0]["bank"], "A")
            self.assertEqual(groups[1]["bank"], "B")
            self.assertEqual(groups[1]["range_ms"], 80)
            self.assertEqual(len(bank), 1)
            self.assertEqual(bank[0]["total_ms"], 650)
            self.assertEqual(result["prefill_tps"], 150)

    def test_missing_measurement_start_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "metrics.jsonl").write_text(
                '{"type":"region_ring_chunk_trace","device":"GPU"}\n',
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "measurement_start"):
                mod.load_measured(folder)

    def test_missing_lane_is_not_mislabeled_as_gate_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "metrics.jsonl").write_text(
                json.dumps({"type": "measurement_start"}) + "\n" +
                json.dumps({"type": "region_ring_chunk_trace", "device": "GPU",
                            "region": 4, "chunk": 0, "tokens": 128,
                            "compute_range_ms": 200.0, "split_prepare_ms": 5.0}),
                encoding="utf-8")
            groups, _, _ = mod.summarize(folder, "test")
            self.assertIsNone(groups[0]["lane_ms"])
            self.assertIsNone(groups[0]["remaining_after_prepare_ms"])


if __name__ == "__main__":
    unittest.main()
