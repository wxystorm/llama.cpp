import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import compare_region_ring_runs as compare


class CompareRegionRingRunsTests(unittest.TestCase):
    def make_run(self, root, name, tokens, context, lane, prompt_sha):
        folder = root / name / "prompt_512"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps({
            "actual_prefill_tokens": tokens,
            "context": context,
            "prefill_tps": 200.0 if tokens > 512 else 330.0,
            "prefill_ms": 1000.0 * tokens / (200.0 if tokens > 512 else 330.0),
            "ring_wall_ms": 1200.0,
            "gpu_compute_range_ms": lane * 2,
            "prompt_sha256": prompt_sha,
            "model": "model.gguf",
            "runtime_layout": "G9,P1,G9,P1,G9,P1,G10",
            "xg": 128, "xp": 128, "batch": 2048, "ubatch": 2048,
        }), encoding="utf-8")
        (folder / "config.json").write_text(json.dumps({
            "ring_environment": {"LLAMA_REGION_RING_PHONE_GRAPHLESS": "1"},
        }), encoding="utf-8")
        rows = [
            {"type": "region_ring_chunk_trace", "device": "GPU",
             "region": 0, "tokens": 128, "compute_range_ms": 999},
            {"type": "measurement_start"},
            {"type": "region_ring_chunk_trace", "device": "GPU",
             "region": 0, "chunk": 0, "tokens": 128, "layer_begin": 0,
             "layer_end": 9, "compute_range_ms": lane,
             "lane_compute_begin_ms": 10, "lane_compute_end_ms": 10+lane},
            {"type": "region_ring_chunk_trace", "device": "GPU",
             "region": 0, "chunk": 1, "tokens": 128, "layer_begin": 0,
             "layer_end": 9, "compute_range_ms": lane,
             "lane_compute_begin_ms": 100, "lane_compute_end_ms": 100+lane},
            {"type": "region_ring_chunk_trace", "device": "GPU",
             "region": 0, "chunk": 2, "tokens": 117, "layer_begin": 0,
             "layer_end": 9, "compute_range_ms": 15,
             "lane_compute_begin_ms": 200, "lane_compute_end_ms": 215},
            {"type": "region_ring_bank_trace", "region": 4,
             "source_region": 0, "total_ms": 400.0,
             "queue_wait_ms": 0.2},
        ]
        (folder / "metrics.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        return folder

    def test_resolve_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            fast = self.make_run(base, "fast", 512, 1024, 50, "sha-a")
            slow = self.make_run(base, "slow", 1013, 4096, 100, "sha-b")
            self.assertEqual(compare.locate_run(base / "fast"), fast)
            self.assertEqual(compare.locate_run(slow), slow)
            f, s = compare.load(base / "fast"), compare.load(base / "slow")
            median_fast, width, count = compare.full_chunk_median(f, 0)
            self.assertEqual((median_fast, width, count), (50, 128, 1))
            self.assertAlmostEqual(compare.ratio(
                median_fast / 128,
                compare.full_chunk_median(s, 0)[0] / 128), 2.0)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                compare.compare(f, s)
            self.assertIn("=== GPU REGION", output.getvalue())
            self.assertIn("=== BANK REFILL", output.getvalue())
            self.assertIn("Different Prompt SHA256", output.getvalue())
            self.assertIn("Token counts differ", output.getvalue())
            self.assertIn("1024", output.getvalue())
            self.assertIn("4096", output.getvalue())

    def test_ambiguous_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_run(root, "run", 512, 1024, 100, "same")
            folder = root / "run"
            other = folder / "other"
            other.mkdir()
            (other / "result.json").write_text("{}", encoding="utf-8")
            (other / "metrics.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                compare.locate_run(folder)


if __name__ == "__main__":
    unittest.main()
