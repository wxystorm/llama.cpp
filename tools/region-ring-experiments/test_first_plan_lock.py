#!/usr/bin/env python3
"""Offline tests for cross-Prompt G/C/N layer-lock lifecycle."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name("run_32b_all40.py")
spec = importlib.util.spec_from_file_location("region_ring_runner", SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
LAYOUT = "G13,P7,G12,C1,G7"


def passed(prompt="prompt_128_01.txt", layout=LAYOUT, source="htp_gpu_tail"):
    return {
        "prompt": prompt, "status": "PASS", "plan_applied": True,
        "plan_layout": layout, "layout": layout, "plan_source": source,
        "plan_xc": 32, "plan_xg": 32, "plan_xp": 64,
    }


class FirstPlanLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.prompts = [
            (128, self.out / "prompt_128_01.txt"),
            (128, self.out / "prompt_128_02.txt"),
        ]

    def test_first_pass_creates_and_restores_lock(self):
        first = passed()
        saved = runner.placement_lock_from_pass(first, 40)
        self.assertEqual(saved["layout"], LAYOUT)
        self.assertEqual(saved["max_gpu_region_budget_fraction"], 0.5)
        runner.save_json(self.out / "placement-lock.json", saved)
        restored = runner.restore_placement_lock(self.out, self.prompts, 40)
        self.assertEqual(restored["first_prompt"], first["prompt"])

    def test_resume_recovers_lock_from_pass_if_file_missing(self):
        row = passed()
        path = self.out / "runs" / "prompt_128_01" / "attempt-0001" / "result.json"
        runner.save_json(path, row)
        restored = runner.restore_placement_lock(self.out, self.prompts, 40)
        self.assertEqual(restored["layout"], LAYOUT)
        self.assertTrue((self.out / "placement-lock.json").exists())

    def test_never_lock_failed_or_unapplied_plan(self):
        fail = passed()
        fail["status"] = "FAIL"
        with self.assertRaises(ValueError):
            runner.placement_lock_from_pass(fail, 40)
        fail = passed()
        fail["plan_applied"] = False
        with self.assertRaises(ValueError):
            runner.placement_lock_from_pass(fail, 40)

    def test_locked_runtime_must_acknowledge_chunk_only_mode(self):
        ok = passed("prompt_128_02.txt", source="locked_chunk_only")
        runner.verify_placement_lock(ok, {"layout": LAYOUT})
        self.assertEqual(ok["status"], "PASS")
        wrong_mode = passed("prompt_128_02.txt", source="htp_gpu_tail")
        runner.verify_placement_lock(wrong_mode, {"layout": LAYOUT})
        self.assertEqual(wrong_mode["status"], "PLAN_LOCK_UNVERIFIED")
        wrong_layout = passed("prompt_128_02.txt", layout="G11,C2,P7,G12,C1,G7",
                              source="locked_chunk_only")
        runner.verify_placement_lock(wrong_layout, {"layout": LAYOUT})
        self.assertEqual(wrong_layout["status"], "PLAN_LOCK_MISMATCH")

    def test_corrupt_placement_lock_never_silently_replans(self):
        lock = self.out / "placement-lock.json"
        lock.write_text(json.dumps({"schema": 1, "layout": "G50"}))
        with self.assertRaises(ValueError):
            runner.restore_placement_lock(self.out, self.prompts, 40)


if __name__ == "__main__":
    unittest.main()
