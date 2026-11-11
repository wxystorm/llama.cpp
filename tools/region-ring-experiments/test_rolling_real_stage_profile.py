#!/usr/bin/env python3
"""Host-only tests of immutable, truthful real HTP rolling Stage Profiles."""
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
    def make_case(self, root: Path, i: int, complete=True, htp=False,
                  gpu_tail=False):
        case = root / "runs" / f"prompt_128_{i:02d}" / "attempt-0001"
        case.mkdir(parents=True, exist_ok=True)
        (case / "result.json").write_text(
            json.dumps({"status": "PASS" if complete else "FAIL",
                        "end_time": f"2026-11-11T12:{i:02d}:00+00:00"}),
            encoding="utf-8")
        devices = [("GPU", 0, 8, 20.0 + i),
                   ("CPU", 1, 2, 5.0 + i),
                   ("PHONE", 2, 4, 90.0 + i)]
        if gpu_tail:
            devices.append(("GPU", 3, 1, 10.0))
        with (case / "metrics.jsonl").open("w", encoding="utf-8") as f:
            for d, region, layers, cost in devices:
                f.write(json.dumps({
                    "type": "region_ring_chunk_trace",
                    "device": d, "region": region, "chunk": 0,
                    "layer_begin": 0, "layer_end": layers,
                    "tokens": 96, "token_begin": 0,
                    "prepare_total_ms": 1.0, "compute_range_ms": cost,
                }) + "\n")
        log = "[PHONE_AUTO] event=SESSION_PREPARING run=5678 status=READY\n"
        if htp:
            log += (f"[REAL_STAGE_PROFILE] side=PHONE phase=CHUNK source=RPC_HTP "
                    f"run=5678 region=2 chunk=0 tokens=96 token_begin=0 "
                    f"upload_ms=2.0 prepare_ms=3.0 compute_ms={30 + i}.0 "
                    "status=PASS\n")
        (case / "engine.stderr.log").write_text(log, encoding="utf-8")
        return case

    def test_unverified_phone_is_never_cached_as_htp(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(1, 6):
                self.make_case(root, i)
            with self.assertRaisesRegex(ValueError, "no real Phone HTP"):
                profile_snapshot(root, 1, 5)

    def test_epoch_is_immutable_and_phone_is_measured(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(1, 6):
                self.make_case(root, i, htp=True, gpu_tail=True)
            csv_path, meta, created = profile_snapshot(root, 1, 5)
            self.assertTrue(created)
            self.assertEqual(meta["device_points"]["PHONE"], 1)
            self.assertEqual(meta["window_prompts"], 5)
            with csv_path.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            phone = next(r for r in rows if r["device"] == "PHONE")
            self.assertAlmostEqual(float(phone["compute_ms"]), 33.0)
            self.assertEqual(phone["terminal"], "0")  # GPU tail after HTP
            self.assertEqual(phone["samples"], "5")
            first_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            again, meta2, created2 = profile_snapshot(root, 1, 5)
            self.assertFalse(created2)
            self.assertEqual(again, csv_path)
            self.assertEqual(meta2["sha256"], first_sha)
            csv_path.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                profile_snapshot(root, 1, 5)

    def test_second_epoch_uses_only_recent_five_successes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(1, 11):
                self.make_case(root, i, htp=True)
            first, _, _ = profile_snapshot(root, 1, 5)
            second, meta, _ = profile_snapshot(root, 2, 5)
            with first.open(encoding="utf-8") as f:
                a = next(r for r in csv.DictReader(f) if r["device"] == "PHONE")
            with second.open(encoding="utf-8") as f:
                b = next(r for r in csv.DictReader(f) if r["device"] == "PHONE")
            self.assertEqual(float(a["compute_ms"]), 33.0)
            self.assertEqual(float(b["compute_ms"]), 38.0)
            self.assertEqual(b["samples"], "5")
            self.assertEqual(meta["passes_at_snapshot"], 10)

    def test_fail_and_interrupted_cases_do_not_count(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(1, 5):
                self.make_case(root, i, htp=True)
            self.make_case(root, 5, complete=False, htp=True)
            self.assertEqual(len(completed_cases(root)), 4)
            with self.assertRaisesRegex(ValueError, "Need 5 PASS"):
                profile_snapshot(root, 1, 5)


if __name__ == "__main__":
    unittest.main()
