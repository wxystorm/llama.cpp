"""Pure offline checks for the Jupiter stage-3 end-to-end glue.

The device execution part is intentionally tested on the real PC+Phone rig.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_sequence_pipeline import (
    choose_refinement_workspace,
    first_chunk_candidates,
    read_complete_plan,
    resolve_stage2,
    verify_runtime,
)


def stage2_report():
    # Same values as the 357-token PC20/Phone20 calibration:
    # 128 would meet 90% on both stages, but cannot tile the prompt.
    return {
        "min_chunk": 128, "prompt_tokens": 357, "pc_layers": 20,
        "max_chunk": 128, "threshold": 0.90,
        "candidates": [
            {"chunk_tokens": 64, "pc_compute_ms": "74.891",
             "phone_compute_ms": "22.785", "samples": 2},
            {"chunk_tokens": 96, "pc_compute_ms": "89.608",
             "phone_compute_ms": "19.776", "samples": 2},
            {"chunk_tokens": 128, "pc_compute_ms": "106.850",
             "phone_compute_ms": "27.459", "samples": 2},
        ],
    }


def q_row(size: int, prefix: int, device: str) -> dict[str, str]:
    return {
        "x": str(size), "y": str(prefix), "device": device,
        "latency_ms": "42", "compute_ms": "42",
        "wall_ms": "45", "input_ms": "3",
    }


class JupiterStage3Tests(unittest.TestCase):
    def test_stale_b_128_is_reselected_without_reprofiling(self):
        pc, total, b, max_chunk, status = resolve_stage2(stage2_report())
        self.assertEqual((pc, total, b, max_chunk), (20, 357, 96, 128))
        self.assertEqual(status, "NO_FEASIBLE_PLATEAU")

    def test_legal_first_chunk_seeds_and_invalid_total(self):
        self.assertEqual(first_chunk_candidates(357, 96, 128), [101, 112, 128])
        with self.assertRaisesRegex(ValueError, "no legal"):
            first_chunk_candidates(357, 128, 128)

    def test_final_plan_exact_q_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "output.plan"
            plan.write_text(
                "model.pc_layers=20\nsequence.357=112,123,122\n",
                encoding="utf-8",
            )
            profile = root / "q-profile-refined.csv"
            rows = []
            for x, y in [(112, 0), (123, 112), (122, 235)]:
                rows += [q_row(x, y, "CUDA0"), q_row(x, y, "RPC0[phone]")]
            with profile.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            self.assertEqual(
                read_complete_plan(plan, 357, 20, 96, 128, profile),
                [112, 123, 122],
            )
            with self.assertRaisesRegex(ValueError, "model boundary"):
                read_complete_plan(plan, 357, 25, 96, 128, profile)
            rows.pop()  # one Phone stage is no longer measured
            with profile.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "unmeasured"):
                read_complete_plan(plan, 357, 20, 96, 128, profile)

    def test_stale_refinement_config_is_isolated_and_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            original = out / "refinement"
            original.mkdir()
            (original / "configuration.json").write_text(
                json.dumps({"profile_sha256": "old-data"}), encoding="utf-8"
            )
            (original / "q-profile-refined.csv").write_text(
                "old measured samples", encoding="utf-8"
            )
            expected = {"profile_sha256": "new-data", "pc_layers": 20}

            new_dir, resume, action = choose_refinement_workspace(out, expected)
            self.assertNotEqual(new_dir, original)
            self.assertTrue(new_dir.name.startswith("refinement-"))
            self.assertFalse(resume)
            self.assertEqual(action, "ISOLATE_OLD_CONFIG_NEW")
            self.assertEqual(
                (original / "q-profile-refined.csv").read_text(encoding="utf-8"),
                "old measured samples",
            )

            new_dir.mkdir()
            (new_dir / "configuration.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            again, resume, action = choose_refinement_workspace(out, expected)
            self.assertEqual(again, new_dir)
            self.assertTrue(resume)
            self.assertEqual(action, "ISOLATE_OLD_CONFIG_REUSE")

    def test_matching_refinement_resumes_and_empty_is_new(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            expected = {"profile_sha256": "stable-q", "pc_layers": 20}
            fresh, resume, action = choose_refinement_workspace(out, expected)
            self.assertEqual(fresh, out / "refinement")
            self.assertFalse(resume)
            self.assertEqual(action, "NEW")

            fresh.mkdir()
            existing, resume, action = choose_refinement_workspace(out, expected)
            self.assertEqual(existing, fresh)
            self.assertFalse(resume)
            (fresh / "configuration.json").write_text(
                json.dumps(expected), encoding="utf-8"
            )
            existing, resume, action = choose_refinement_workspace(out, expected)
            self.assertEqual(existing, fresh)
            self.assertTrue(resume)
            self.assertEqual(action, "RESUME_MATCH")

    def test_unrecognized_hashed_workspace_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            expected = {"profile_sha256": "a"}
            original = out / "refinement"
            original.mkdir()
            (original / "old.txt").write_text("keep me", encoding="utf-8")
            chosen, _, _ = choose_refinement_workspace(out, expected)
            chosen.mkdir()
            (chosen / "old.txt").write_text("other state", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible data"):
                choose_refinement_workspace(out, expected)

    def test_runtime_verification_disables_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            prompt_file = folder / "prompt.txt"
            prompt_file.write_text("hello Jupiter", encoding="utf-8")
            args = argparse.Namespace(
                prompt_file=str(prompt_file), exe=str(prompt_file),
                model=str(prompt_file), rpc="127.0.0.1:10139",
                ubatch=128, batch=512, context=512, threads=8,
            )
            plan = folder / "plan.txt"
            plan.write_text("model.pc_layers=20\nsequence.357=112,123,122\n")
            def fake_run_one(command, env, log):
                self.assertEqual(env["LLAMA_JUPITER_PROFILE_STAGE"], "0")
                self.assertEqual(env["LLAMA_JUPITER_PLAN_FILE"], str(plan))
                self.assertEqual(env["LLAMA_JUPITER_PC_LAYERS"], "20")
                self.assertEqual(env["LLAMA_JUPITER_ASYNC_C2P"], "0")
                log.write_text(
                    "[JUPITER_SEQUENCE_PLAN] mode=PAPER_DP "
                    "total=357 k=3 chunks=112,123,122\n"
                    "prompt eval time = 5000.0 ms / 357 tokens\n",
                    encoding="utf-8",
                )
            with patch("run_sequence_pipeline.run_one", fake_run_one):
                self.assertEqual(
                    verify_runtime(args, plan, [112, 123, 122], 20, 357, folder),
                    5000.0,
                )


if __name__ == "__main__":
    unittest.main()
