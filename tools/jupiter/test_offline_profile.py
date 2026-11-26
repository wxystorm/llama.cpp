"""Model-free tests for the shared offline DP profile and physical-stage accounting."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import offline_profile as offline
from jupiter_planner import QProfile, solve_sequence, solve_sequence_table, sequence_points
from test_native_calibration import sample


class TestOfflineProfile(unittest.TestCase):
    def test_grid_exact_coverage_without_tiny_tail(self):
        for b in (8, 24, 64, 128):
            jobs = offline.training_jobs(b, 256)
            seen = []
            for cuts, targets in jobs:
                self.assertEqual(sum(cuts), max(x+y for x,y in targets))
                self.assertLessEqual(sum(cuts), 2048)
                self.assertLessEqual(len(cuts), 16)
                self.assertTrue(all(8 <= x <= 256 for x in cuts))
                self.assertTrue(set(targets).issubset(sequence_points(cuts)))
                seen.extend(targets)
            expected = {(x, y) for x in offline.x_grid(256) if x >= b
                        for y in set(offline.BASE_Y + (min(2048-x,7*256),))
                        if y <= min(2048-x,7*256)}
            self.assertEqual(set(seen), expected)
            self.assertEqual(len(seen), len(expected))
            held = offline.holdout_points(b, 256)
            self.assertTrue(held)
            self.assertFalse(set(held) & expected)
            for x, y in held:
                cuts = offline.calibration_plan(x+y, [(x,y)], 256)
                self.assertIn((x,y), sequence_points(cuts))
                self.assertTrue(all(8 <= c <= 256 for c in cuts))

    def test_shared_table_matches_original_single_length_solver(self):
        rows = [{"x": x, "y": y, "device": d, "wall_ms": 2+x*factor+y*.001}
                for x in (8,24,64,128,256) for y in (0,256,512)
                for d, factor in (("PC",.08),("PHONE_NATIVE",.1))]
        profile = QProfile(rows, metric="wall_ms")
        shared = solve_sequence_table(profile, 512, 2, 24, 256, 8)
        for n in (24, 25, 74, 130, 257, 359, 512):
            actual = solve_sequence(profile, n, 2, 24, 256, 8)
            self.assertEqual(shared[n], actual)
            h = [profile.q(x,y) for x,y in sequence_points(actual.chunks)]
            self.assertAlmostEqual(actual.predicted_ms, sum(h)+max(h))

    def test_all_lengths_respect_paper_cap_and_short_exception(self):
        rows = [{"x": x, "y": y, "device": d, "wall_ms": 3+x*factor+y*.001}
                for x in (24,64,128,256) for y in (0,512,1024,1792)
                for d, factor in (("PC",.08),("PHONE_NATIVE",.1))]
        profile = QProfile(rows, metric="wall_ms")
        plans, summary = offline.make_all_plans(profile, 24, 256)
        self.assertEqual(len(plans), 2047)
        self.assertEqual(plans[2], [2])
        self.assertEqual(plans[2048], [256]*8)
        self.assertGreaterEqual(len(plans[1855]), 8)
        for n, cuts in plans.items():
            self.assertEqual(sum(cuts), n)
            self.assertLessEqual(len(cuts), 8)
            self.assertLessEqual(max(cuts), 256)
            if n >= 24:
                self.assertGreaterEqual(min(cuts), 24)
        self.assertEqual(summary[0]["status"], "SHORT_SINGLE_CHUNK")

    def test_b_is_uniform_and_capacity_constrained(self):
        # Only 256 reaches the measured saturation target. Choosing it
        # would leave holes at N=257, etc.; report the constrained proxy.
        rows = [{"x": x, "cuda_compute_ms": 100.0, "phone_htp_ms": 120.0}
                for x in offline.x_grid(256) for _ in range(2)]
        result = offline.choose_b(rows, 256, 2)
        self.assertEqual(result["raw_b"], 256)
        self.assertLessEqual(result["b"], 128)
        self.assertTrue(result["status"].startswith("FEASIBILITY_CONSTRAINED"))
        self.assertTrue(all(offline.feasible(n, result["b"], 256)
                            for n in range(result["b"], 2049)))

    def test_physical_stages_exclude_bridge_wait_and_head(self):
        text = sample().replace("tokens=96\n[JUPITER_NATIVE_BRIDGE_PROFILE]",
                                "tokens=96 admission_ms=1.000 status=QUEUED\n[JUPITER_NATIVE_BRIDGE_PROFILE]")
        # Add timing on both PUSH lines, and the final P2C.
        text = "\n".join(line + " admission_ms=1.000 status=QUEUED"
                         if "phase=PHONE_AUTO_PUSH" in line and "admission_ms" not in line else line
                         for line in text.splitlines())
        text = text.replace("phase=PHONE_AUTO_P2C_DONE run=1 output=REAL_HIDDEN",
                            "phase=PHONE_AUTO_P2C_DONE run=1 chunk=1 tokens=96 phone_compute_ms=5.000 p2c_wait_ms=50.000 output=REAL_HIDDEN")
        text = text.replace("[JUPITER_UBATCH] event=QUEUED chunk=1",
                            "[JUPITER_STAGE_PROFILE] split=3 backend=CUDA0 nodes=2 inputs=1 input_ms=2.000 compute_ms=200.000 wall_ms=202.000 native_bridge=0\n"
                            "[JUPITER_UBATCH] event=QUEUED chunk=1")
        rows = offline.detailed_rows(text, [96,96], 36)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["cpu_compute_ms"], 11.0)
            self.assertEqual(row["cuda_compute_ms"], 8.0)
            self.assertEqual(row["pc_service_ms"], 23.0)
            self.assertEqual(row["phone_service_ms"], 9.0)
            self.assertEqual(row["q_ms"], 23.0)
        self.assertEqual(rows[1]["post_bridge_ms"], 202.0)
        self.assertEqual(rows[1]["p2c_wait_ms"], 50.0)
        table = offline.aggregate_rows(rows)
        profile = QProfile(table, metric="wall_ms")
        self.assertEqual(profile.q(96,96),23.0)
        self.assertTrue(profile.is_measured(96,96))

    def test_exact_synthetic_prompt_cache_and_tamper_rejection(self):
        args = argparse.Namespace(tokenizer="fake", model="fake")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(offline, "count_model_tokens", side_effect=lambda _t,_m,p: len(p.split())+8):
                text = offline.exact_prompt(args, root, 512)
            with patch.object(offline, "count_model_tokens", side_effect=AssertionError("retokenized")):
                self.assertEqual(offline.exact_prompt(args, root, 512), text)
            (root/"prompts"/"synthetic-512.txt").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                offline.exact_prompt(args, root, 512)

    def test_raw_24b_calibration_does_not_use_chat_length_search(self):
        args = argparse.Namespace(tokenizer="fake", model="fake", model_size="24b")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(offline, "count_model_tokens", side_effect=AssertionError("chat count")):
                for total in (160, 192, 2048):
                    text = offline.exact_prompt(args, root, total)
                    self.assertGreater(len(text.split()), total)
                    self.assertEqual(text, offline.exact_prompt(args, root, total))
                    meta = offline.read_json(root/"prompts"/f"synthetic-{total}.json")
                    self.assertEqual(meta["tokens"], total)
                    self.assertEqual(meta["mode"], "RAW_PREFIX")
                (root/"prompts"/"synthetic-160.txt").write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    offline.exact_prompt(args, root, 160)

    def test_ready_profile_fingerprints_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"signature": "test", "config": {"pc_layers":52, "cpu_layers":36, "max_chunk":256}}
            for name in offline.READY_FILES:
                (root/name).write_text("data", encoding="utf-8")
            offline.write_json(root/"min-chunk.json", {"b":24})
            lines = ["model.pc_layers=52", "model.cpu_layers=36"]
            for n in range(2,2049):
                cuts = [n] if n < 24 else offline.balanced_fill(n, 24, 256)
                lines.append(f"sequence.{n}="+",".join(map(str,cuts)))
            offline.atomic_text(root/"sequence.plan", "\n".join(lines)+"\n")
            offline.write_json(root/"profile-ready.json", {
                "signature":"test", "status":"READY", "b":24,
                "sha256":{name:offline.digest_file(root/name) for name in offline.READY_FILES}})
            self.assertEqual(offline.load_ready(root,manifest)["plans"][2048], [256]*8)
            (root/"q-profile.csv").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                offline.load_ready(root,manifest)


if __name__ == "__main__":
    unittest.main()
