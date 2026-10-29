"""Regression tests for Jupiter's measured minimum efficient Chunk b."""
import tempfile
from pathlib import Path
import unittest

from min_chunk_profile import partition_feasible, select_min_chunk
from run_min_chunk_profile import (
    first_chunk_service_times,
    load_pc_layers,
    make_first_chunk_plan,
    parse_sizes,
)


def sample(size, pc, phone, repeat=1, status="OK"):
    return {
        "chunk_tokens": str(size),
        "pc_compute_ms": str(pc),
        "phone_compute_ms": str(phone),
        "repeat": str(repeat), "status": status,
    }


def synthetic_log(tokens=32):
    lines = []
    for prefix, length, pc_ms, phone_ms in (
        (0, 2, 2.0, 4.0),  # unrelated warmup
        (0, tokens, 10.0, 20.0),
        (tokens, 64, 30.0, 50.0),  # later chunk must not be counted
    ):
        lines.extend([
            f"[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 "
            f"prefix_tokens={prefix} tokens={length} t_us=12",
            f"[JUPITER_STAGE_PROFILE] split=0 backend=CPU0 nodes=3 "
            f"inputs=0 input_ms=0.2 compute_ms=5.000 wall_ms=5.200",
            f"[JUPITER_STAGE_PROFILE] split=1 backend=CUDA0 nodes=44 "
            f"inputs=1 input_ms=1.0 compute_ms={pc_ms:.3f} "
            f"wall_ms={pc_ms + 1:.3f}",
            f"[JUPITER_STAGE_PROFILE] split=2 backend=RPC0[phone] nodes=44 "
            f"inputs=1 input_ms=2.0 compute_ms={phone_ms:.3f} "
            f"wall_ms={phone_ms + 2:.3f}",
            f"[JUPITER_UBATCH] event=QUEUED chunk=0 slot=0 "
            f"prefix_tokens={prefix} tokens={length} t_us=99",
        ])
    return "\n".join(lines)


class MinChunkProfileTest(unittest.TestCase):
    def test_first_chunk_excludes_warmup_and_cpu(self):
        metrics = first_chunk_service_times(synthetic_log(), 32)
        self.assertEqual(metrics["pc_compute_ms"], 10)
        self.assertEqual(metrics["phone_compute_ms"], 20)
        self.assertEqual(metrics["pc_wall_ms"], 11)
        self.assertEqual(metrics["phone_wall_ms"], 22)

    def test_first_chunk_requires_both_devices(self):
        text = synthetic_log().replace("backend=RPC0[phone]", "backend=CPU0")
        with self.assertRaisesRegex(ValueError, "missing calibrated"):
            first_chunk_service_times(text, 32)

    def test_first_chunk_requires_finished_ubatch(self):
        text = synthetic_log().split(
            "[JUPITER_UBATCH] event=QUEUED chunk=0 slot=0 "
            "prefix_tokens=0 tokens=32"
        )[0]
        with self.assertRaisesRegex(ValueError, "no complete"):
            first_chunk_service_times(text, 32)

    def test_complete_plans_with_exact_first_chunk(self):
        for x in (8, 16, 24, 32, 48, 64, 96, 128):
            chunks = make_first_chunk_plan(357, x)
            self.assertEqual(chunks[0], x)
            self.assertEqual(sum(chunks), 357)
            self.assertTrue(all(8 <= v <= 128 for v in chunks))
            self.assertLessEqual(len(chunks), 5)
        self.assertEqual(make_first_chunk_plan(357, 128), [128, 128, 101])
        self.assertEqual(make_first_chunk_plan(357, 8), [8, 128, 128, 93])

    def test_reject_invalid_tail(self):
        with self.assertRaisesRegex(ValueError, "remainder"):
            make_first_chunk_plan(100, 96)

    def test_select_smallest_saturated_on_both(self):
        rows = [
            sample(16, 10, 20),
            sample(32, 18, 22),
            sample(64, 36, 44),
        ]
        chosen, curves, status = select_min_chunk(rows, threshold=0.90)
        self.assertEqual(chosen.tokens, 32)
        self.assertEqual(status, "BOTH_STAGES_SATURATED")
        self.assertEqual(len(curves), 3)

    def test_use_median_across_repeats(self):
        rows = [
            sample(16, 4, 5, 1), sample(16, 200, 300, 2),
            sample(32, 10, 12, 1), sample(32, 10, 12, 2),
        ]
        chosen, curves, _ = select_min_chunk(rows, threshold=0.90, min_samples=2)
        self.assertEqual(chosen.tokens, 32)
        self.assertEqual(curves[0].pc_ms, 102)

    def test_no_common_plateau_is_flagged(self):
        rows = [sample(16, 4, 100), sample(32, 16, 20)]
        selected, _, status = select_min_chunk(rows, threshold=0.95)
        self.assertEqual(selected.tokens, 32)
        self.assertEqual(status, "NO_COMMON_PLATEAU")

    def test_failed_and_incomplete_candidates_not_selected(self):
        rows = [sample(8, 1, 1, status="FAILED"),
                sample(16, 10, 20),
                sample(32, 20, 40), sample(32, 20, 40, repeat=2)]
        chosen, curves, _ = select_min_chunk(rows, min_samples=2)
        self.assertEqual(chosen.tokens, 32)
        self.assertEqual(len(curves), 1)

    def test_model_plan_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.plan"
            path.write_text("# model result\nmodel.pc_layers=20\n", encoding="utf-8")
            self.assertEqual(load_pc_layers(None, str(path)), 20)
            self.assertEqual(load_pc_layers(25, str(path)), 25)
            self.assertEqual(parse_sizes("32, 16,32"), [16, 32])

    def test_reject_invalid_threshold(self):
        with self.assertRaisesRegex(ValueError, "threshold"):
            select_min_chunk([sample(16, 10, 20)], threshold=0)
        with self.assertRaisesRegex(ValueError, "no complete"):
            select_min_chunk([sample(16, 10, 20)], min_samples=2)


    def test_full_prompt_feasibility_changes_selected_b(self):
        # b=128 exceeds the feasible range for 357 tokens / max128.
        # b=96 barely misses 90% PC throughput, but is the measured
        # best feasible candidate and must be flagged as below threshold.
        rows = [
            sample(64, 74.891, 22.785),
            sample(96, 89.608, 19.776),
            sample(128, 106.850, 27.459),
        ]
        selected, measured, status = select_min_chunk(
            rows, threshold=0.90, total_tokens=357, max_chunk=128
        )
        self.assertFalse(partition_feasible(357, 128, 128))
        self.assertTrue(partition_feasible(357, 96, 128))
        self.assertEqual(selected.tokens, 96)
        self.assertEqual(status, "NO_FEASIBLE_PLATEAU")
        self.assertEqual(len(measured), 3)

    def test_reject_entirely_infeasible_calibrated_range(self):
        rows = [sample(96, 89.608, 19.776),
                sample(128, 106.850, 27.459)]
        with self.assertRaisesRegex(ValueError, "no measured minimum b"):
            select_min_chunk(rows, total_tokens=257, max_chunk=128)

if __name__ == "__main__":
    unittest.main()
