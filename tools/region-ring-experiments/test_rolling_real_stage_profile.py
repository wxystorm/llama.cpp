#!/usr/bin/env python3
"""Independent baseline + five-PASS runtime-profile regression tests."""
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
    def make_baseline(self, root: Path):
        path = root / "independent-profile" / "profile.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "device", "layers", "tokens", "kv_tokens", "setup_ms",
                "upload_ms", "prepare_ms", "compute_ms", "terminal", "samples"])
            writer.writeheader()
            for d, depth, tokens, kv, compute in (
                    ("CPU", 2, 32, 32, 4.0),
                    ("GPU", 2, 32, 32, 2.0),
                    ("PHONE", 4, 96, 96, 50.0),
                    ("PHONE", 2, 64, 256, 25.0)):
                writer.writerow(dict(device=d, layers=depth,
                                     tokens=tokens, kv_tokens=kv,
                                     setup_ms=0.0, upload_ms=1.0,
                                     prepare_ms=1.0, compute_ms=compute,
                                     terminal=0, samples=2))
        return path

    def make_case(self, root: Path, i: int, complete=True, htp=False,
                  gpu_tail=False):
        case = root / "runs" / f"prompt_128_{i:02d}" / "attempt-0001"
        case.mkdir(parents=True, exist_ok=True)
        (case / "result.json").write_text(
            json.dumps({"status": "PASS" if complete else "FAIL",
                        "end_time": f"2026-11-11T12:{i:02d}:00+00:00"}),
            encoding="utf-8")
        devices = [("GPU", 0, 8, 20.0 + i),
                   ("CPU", 1, 2, 5.0 + i)]
        if htp:
            devices.append(("PHONE", 2, 4, 90.0 + i))
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

    @staticmethod
    def phone_rows(csv_path):
        with csv_path.open(encoding="utf-8", newline="") as f:
            return [p for p in csv.DictReader(f) if p["device"] == "PHONE"]

    def test_no_baseline_must_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(1, 6):
                self.make_case(root, i, htp=True)
            with self.assertRaisesRegex(ValueError, "Independent backend Profile is missing"):
                profile_snapshot(root, 1, 5)

    def test_zero_phone_layers_in_five_pass_window_keep_real_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_baseline(root)
            for i in range(1, 6):
                self.make_case(root, i, htp=False)
            csv_path, meta, created = profile_snapshot(root, 1, 5)
            self.assertTrue(created)
            self.assertEqual(meta["diagnostic"]["phone"], 0)
            phone = self.phone_rows(csv_path)
            self.assertEqual(len(phone), 2)
            self.assertEqual(float(phone[0]["compute_ms"]), 25.0)
            self.assertEqual(float(phone[1]["compute_ms"]), 50.0)

    def test_epoch_overrides_only_matching_shapes_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            baseline = self.make_baseline(root)
            for i in range(1, 6):
                self.make_case(root, i, htp=True, gpu_tail=True)
            csv_path, meta, created = profile_snapshot(root, 1, 5)
            self.assertTrue(created)
            self.assertEqual(meta["device_points"]["PHONE"], 2)
            self.assertEqual(meta["window_prompts"], 5)
            phone = self.phone_rows(csv_path)
            self.assertEqual(len(phone), 2)
            self.assertEqual(float(phone[0]["compute_ms"]), 25.0)
            self.assertAlmostEqual(float(phone[1]["compute_ms"]), 33.0)
            self.assertEqual(phone[1]["terminal"], "0")
            self.assertEqual(phone[1]["samples"], "5")
            first_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
            again, meta2, created2 = profile_snapshot(root, 1, 5)
            self.assertFalse(created2)
            self.assertEqual(again, csv_path)
            self.assertEqual(meta2["sha256"], first_sha)
            csv_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                profile_snapshot(root, 1, 5)
            # Restore profile and verify that a changed independent baseline
            # also invalidates a persisted epoch.
            csv_path.write_bytes(again.read_bytes())
            baseline.write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive service measurements"):
                profile_snapshot(root, 1, 5)

    def test_second_epoch_uses_only_new_five_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_baseline(root)
            for i in range(1, 11):
                self.make_case(root, i, htp=True)
            first, _, _ = profile_snapshot(root, 1, 5)
            second, meta, _ = profile_snapshot(root, 2, 5)
            a = self.phone_rows(first)[1]
            b = self.phone_rows(second)[1]
            self.assertEqual(float(a["compute_ms"]), 33.0)
            self.assertEqual(float(b["compute_ms"]), 38.0)
            self.assertEqual(b["samples"], "5")
            self.assertEqual(meta["passes_at_snapshot"], 10)

    def test_failed_attempts_never_increment_window(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_baseline(root)
            for i in range(1, 5):
                self.make_case(root, i, htp=True)
            self.make_case(root, 5, complete=False, htp=True)
            self.assertEqual(len(completed_cases(root)), 4)
            with self.assertRaisesRegex(ValueError, "Need 5 PASS"):
                profile_snapshot(root, 1, 5)


if __name__ == "__main__":
    unittest.main()
