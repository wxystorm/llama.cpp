#!/usr/bin/env python3
"""No-inference tests for Prima Fig3's placement consistency checks."""
import tempfile
import unittest
from pathlib import Path
from layout_guard import parse_fixed, summarize_log, validate_layout, compare_runs


def dump(path, rank0=39, rank1=1, ngl=21, world=2, extra_stage=""):
    path.write_text(
        "[PRIMA_FIG3_LAYOUT] stage=post_init rank=0 world=%d lw0=%d "
        "lw1=%d ngl=%d prefetch=1\n" % (world, rank0, rank1, ngl)
        + "- Device Index   : 0\n"
        + "- Assignment Set : M4\n"
        + "- N Layer Window : %d\n" % rank0
        + "- N GPU Layers   : %d\n" % ngl
        + "- Device Index   : 1\n"
        + "- Assignment Set : M4\n"
        + "- N Layer Window : %d\n" % rank1
        + "- N GPU Layers   : 0\n" + extra_stage
        + "llama_perf_context_print: prompt eval time = 5144.86 ms / 1033 tokens\n"
    )


class LayoutGuardTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_fixed("39,1", 40), (39, 1))
        with self.assertRaisesRegex(ValueError, "model layers"):
            parse_fixed("39,1", 64)
        with self.assertRaises(ValueError):
            parse_fixed("39,0")
        with self.assertRaises(ValueError):
            parse_fixed("six,34")

    def test_matching_manual_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.log"
            dump(path)
            layout = summarize_log(path)
            self.assertEqual(validate_layout(layout, (39, 1, 21))["lw0"], 39)
            self.assertEqual(len(layout["planner_stages"]), 1)
            ref = {"layout": layout}
            self.assertEqual(compare_runs(ref, [ref]), layout["effective"])

    def test_old_profile_6_34_is_not_normal_39_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            profile = base / "profile.log"
            normal = base / "normal.log"
            dump(profile, 6, 34, 6)
            dump(normal)
            with self.assertRaisesRegex(ValueError, "LAYOUT_MISMATCH"):
                compare_runs(
                    {"layout": summarize_log(profile)},
                    [{"layout": summarize_log(normal)}],
                )
            with self.assertRaises(ValueError):
                validate_layout(summarize_log(profile), (39, 1, 21))

    def test_pc_only_fallback_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pc40.log"
            dump(path, 40, 0, 21, world=1)
            with self.assertRaises(ValueError):
                validate_layout(summarize_log(path), (39, 1, 21))

    def test_second_planner_stage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replan.log"
            dump(path, extra_stage=(
                "- Device Index   : 0\n"
                "- Assignment Set : M4\n"
                "- N Layer Window : 40\n"
                "- N GPU Layers   : 21\n"
            ))
            with self.assertRaisesRegex(ValueError, "removed Phone"):
                validate_layout(summarize_log(path), (39, 1, 21))

    def test_old_binary_without_marker_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.log"
            path.write_text("- Device Index : 0\n")
            with self.assertRaisesRegex(ValueError, "Rebuild PC"):
                summarize_log(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
