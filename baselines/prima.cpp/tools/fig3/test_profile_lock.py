#!/usr/bin/env python3
"""No-device regression tests for resuming Prima 32B after HiGHS re-planning."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from profile_lock import locked_normal_command, quarantine_mismatched_runs


def record(pc, phone, gpu):
    return {"layout": {"effective": {
        "rank": 0, "world": 2, "lw0": pc, "lw1": phone,
        "ngl": gpu, "prefetch": 1,
    }, "planner_stages": [[
        {"rank": 0, "assignment_set": "M4", "layers": pc, "gpu_layers": gpu},
        {"rank": 1, "assignment_set": "M4", "layers": phone, "gpu_layers": 0},
    ]]}}


class ProfileLockTest(unittest.TestCase):
    def test_profile_lock_32b_uses_profile_gpu(self):
        original = ["llama-cli", "-m", "Qwen3-32B.gguf", "--prefetch"]
        manifest = {"full_command": original, "profile": record(40, 24, 18)}
        locked, expected = locked_normal_command(manifest, original, 64)
        self.assertEqual(expected, (40, 24, 18))
        self.assertEqual(locked[-4:], ["-lw", "40,24", "-ngl", "18"])
        self.assertEqual(manifest["full_command"], original)
        self.assertEqual(manifest["profile"]["layout"]["effective"]["lw0"], 40)

    def test_reject_old_warmup_without_overwriting_evidence(self):
        old = record(37, 27, 14)
        reference = record(40, 24, 18)["layout"]["effective"]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"pc_events.csv").write_text("PROTECTED_RAW_EVENTS\n")
            old_bytes = b"ORIGINAL_WARMUP_37_27\n"
            (root/"warmup_1.log").write_bytes(old_bytes)
            manifest = {
                "profile": record(40, 24, 18),
                "warmup": [old],
                "normal": [],
            }
            rejected = quarantine_mismatched_runs(manifest, root, reference)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(manifest["warmup"], [])
            self.assertEqual(manifest["normal"], [])
            self.assertFalse((root/"warmup_1.log").exists())
            self.assertEqual(
                (root/"rejected_runs/warmup_1.log").read_bytes(), old_bytes)
            self.assertEqual(
                (root/"pc_events.csv").read_text(), "PROTECTED_RAW_EVENTS\n")
            self.assertEqual(rejected[0]["reason"], "LAYOUT_MISMATCH")
            self.assertEqual(rejected[0]["recorded"]["layout"]["effective"]["lw0"], 37)

    def test_keep_matching_warmup(self):
        manifest = {"warmup": [record(40, 24, 18)], "normal": []}
        with TemporaryDirectory() as folder:
            rejected = quarantine_mismatched_runs(
                manifest, folder, record(40, 24, 18)["layout"]["effective"])
        self.assertEqual(rejected, [])
        self.assertEqual(len(manifest["warmup"]), 1)

    def test_resume_mismatch_rejected(self):
        manifest = {"full_command": ["llama-cli"], "profile": record(40, 24, 18)}
        with self.assertRaisesRegex(ValueError, "differs"):
            locked_normal_command(manifest, ["other-cli"], 64)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            locked_normal_command(manifest, ["llama-cli"], 32)

    def test_cannot_overwrite_prior_quarantine(self):
        manifest = {"warmup": [record(37, 27, 14)], "normal": []}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"warmup_1.log").write_text("NEW")
            (root/"rejected_runs").mkdir()
            (root/"rejected_runs/warmup_1.log").write_text("OLD")
            with self.assertRaises(FileExistsError):
                quarantine_mismatched_runs(
                    manifest, root, record(40, 24, 18)["layout"]["effective"])
            self.assertEqual(
                (root/"rejected_runs/warmup_1.log").read_text(), "OLD")


if __name__ == "__main__":
    unittest.main(verbosity=2)
