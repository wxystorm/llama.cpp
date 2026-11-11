#!/usr/bin/env python3
"""Local-only rolling-Profile unit tests; no CUDA, Phone, or model needed."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rolling_real_stage_profile import completed_cases, profile_snapshot


class RollingStageProfileTests(unittest.TestCase):
    def make_case(self, root: Path, i: int, complete=True):
        case = root / "runs" / f"prompt_128_{i:02d}" / "attempt-0001"
        case.mkdir(parents=True, exist_ok=True)
        (case / "result.json").write_text(
            json.dumps({"status": "PASS" if complete else "FAIL"}),
            encoding="utf-8")
        # PHONE is deliberately present in the PC task trace, but without
        # matching HTP server records it must NOT become an HTP compute point.
        with (case / "metrics.jsonl").open("w", encoding="utf-8") as f:
            for d, layers, cost in (
                ("GPU", 8, 20.0 + i),
                ("CPU", 2, 5.0 + i),
                ("PHONE", 4, 90.0 + i)):
                f.write(json.dumps({
                    "type": "region_ring_chunk_trace",
                    "device": d, "region": {"GPU": 0, "CPU": 1, "PHONE": 2}[d],
                    "chunk": 0, "layer_begin": 0, "layer_end": layers,
                    "tokens": 96, "token_begin": 0,
                    "prepare_total_ms": 1.0,
                    "compute_range_ms": cost,
                }) + "\n")
        return case

    def test_epoch_is_immutable_and_omits_unverified_htp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(1, 6):
                self.make_case(root, i)
            csv_path, meta, created = profile_snapshot(root, 1, 5)
            self.assertTrue(created)
            self.assertEqual(meta["point_count"], 2)
            self.assertEqual(meta["device_points"], {
                "CPU": 1, "GPU": 1, "PHONE": 0
            })
            with csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            gpu = next(r for r in rows if r["device"] == "GPU")
            self.assertAlmostEqual(float(gpu["compute_ms"]), 23.0)
            self.assertEqual(gpu["samples"], "5")
            csv_path2, meta2, created2 = profile_snapshot(root, 1, 5)
            self.assertFalse(created2)
            self.assertEqual(csv_path2, csv_path)
            self.assertEqual(meta2["sha256"],
                             hashlib.sha256(csv_path.read_bytes()).hexdigest())
            csv_path.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                profile_snapshot(root, 1, 5)

    def test_fail_and_interrupted_cases_do_not_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(1, 5):
                self.make_case(root, i)
            self.make_case(root, 5, complete=False)
            self.assertEqual(len(completed_cases(root)), 4)
            with self.assertRaises(ValueError):
                profile_snapshot(root, 1, 5)


if __name__ == "__main__":
    unittest.main()
