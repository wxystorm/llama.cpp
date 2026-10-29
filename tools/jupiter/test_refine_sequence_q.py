#!/usr/bin/env python3
"""Regression tests for q interpolation and targeted Jupiter DP refinement."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from jupiter_planner import (
    QProfile,
    sequence_missing_points,
    sequence_points,
    solve_sequence,
)
from refine_sequence_q import (
    measured_cost,
    target_partition,
    targeted_rows,
)


def row(x: int, y: int, t: float, device: str = "RPC0") -> dict[str, str]:
    return {
        "x": str(x), "y": str(y), "device": device,
        "latency_ms": str(t), "compute_ms": str(t),
        "input_ms": "3", "wall_ms": str(t + 3),
    }


class JupiterInterpolationTests(unittest.TestCase):
    def test_same_x_prefix_interpolation_not_borrowed_from_other_x(self) -> None:
        profile = QProfile([
            row(64, 0, 100),
            row(64, 128, 300),
            row(32, 64, 60),
        ])
        self.assertAlmostEqual(profile.q(64, 64), 200.0)
        self.assertEqual(profile.sample_status(64, 64), "INTERP_Y")
        self.assertFalse(profile.is_measured(64, 64))

    def test_missing_128_token_profile_cannot_reuse_48_token_cost(self) -> None:
        profile = QProfile([
            row(48, 144, 85.97),
            row(128, 0, 129.97),
            row(128, 128, 649.27),
        ])
        # The old row-first interpolation returned ~85.97 ms, because
        # x=48 was the sole observed chunk length at prefix=144.
        self.assertGreaterEqual(profile.q(128, 144), 649.27)
        self.assertEqual(profile.sample_status(128, 144), "EXTRAP_Y")
        self.assertFalse(profile.is_measured(128, 144))

    def test_x_then_y_interpolation(self) -> None:
        profile = QProfile([
            row(64, 0, 100), row(64, 128, 200),
            row(128, 0, 200), row(128, 128, 500),
        ])
        self.assertAlmostEqual(profile.q(96, 64), 250)
        self.assertEqual(profile.sample_status(96, 64), "INTERP_XY")

    def test_mediate_by_device_before_bottleneck(self) -> None:
        profile = QProfile([
            row(64, 0, 100, "CUDA0"),
            row(64, 0, 120, "CUDA0"),
            row(64, 0, 30, "RPC0"),
            row(64, 0, 400, "RPC0"),
            row(64, 0, 40, "RPC0"),
        ])
        # Per backend median: CUDA0=110, RPC0=40, bottleneck=110.
        self.assertAlmostEqual(profile.q(64, 0), 110)

    def test_wall_metric_and_invalid_row(self) -> None:
        rows = [row(32, 0, 40), row(64, 0, 60)]
        self.assertEqual(QProfile(rows, metric="wall_ms").q(32, 0), 43)
        bad = [row(32, 0, 40)]
        bad[0]["wall_ms"] = ""
        with self.assertRaises(ValueError):
            QProfile(bad, metric="wall_ms")

    def test_missing_backend_is_not_full_measurement(self) -> None:
        profile = QProfile([
            row(32, 0, 100, "RPC0"), row(32, 0, 70, "CUDA0"),
            row(64, 0, 80, "CUDA0"),
        ])
        self.assertTrue(profile.is_measured(32, 0))
        self.assertFalse(profile.is_measured(64, 0))

    def test_coverage_exact_only(self) -> None:
        profile = QProfile([row(32, 0, 90), row(128, 128, 649)])
        missing = sequence_missing_points(profile, [32, 112, 128, 85])
        self.assertEqual(
            missing, [(112, 32), (128, 144), (85, 272)]
        )

    def test_dp_returns_valid_partition(self) -> None:
        rows = []
        for x in (32, 64, 96, 128):
            for y in range(0, 320, 32):
                rows.append(row(x, y, x + y * .1))
        profile = QProfile(rows)
        result = solve_sequence(profile, 357, 2, 32, 128, None)
        self.assertEqual(sum(result.chunks), 357)
        self.assertTrue(all(32 <= c <= 128 for c in result.chunks))
        self.assertGreater(measured_cost(profile, result.chunks, 2), 0)


class JupiterTargetedCalibrationTests(unittest.TestCase):
    def test_targeted_plan_preserves_exact_prefix_and_size(self) -> None:
        for x, y in [(112, 32), (128, 144), (85, 272), (87, 0), (101, 256)]:
            with self.subTest(x=x, y=y):
                chunks = target_partition(357, x, y, 32, 128)
                self.assertEqual(sum(chunks), 357)
                self.assertIn((x, y), sequence_points(chunks))
                self.assertTrue(all(32 <= c <= 128 for c in chunks))

    def test_targeted_plan_rejects_impossible_neighbors(self) -> None:
        with self.assertRaises(ValueError):
            target_partition(357, 128, 20, 32, 128)
        with self.assertRaises(ValueError):
            target_partition(357, 140, 32, 32, 128)

    def test_only_targeted_coordinate_is_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=[
                        "x", "y", "latency_ms", "device",
                        "input_ms", "compute_ms", "wall_ms",
                    ],
                )
                writer.writeheader()
                for item in [
                    row(128, 144, 80, "CUDA0"),
                    row(128, 144, 200, "RPC0"),
                    row(72, 0, 50, "CUDA0"),
                ]:
                    writer.writerow(item)
            sampled = targeted_rows(path, 128, 144)
            self.assertEqual(len(sampled), 2)
            self.assertEqual({r["device"] for r in sampled}, {"CUDA0", "RPC0"})
            with self.assertRaises(RuntimeError):
                targeted_rows(path, 72, 0)


if __name__ == "__main__":
    unittest.main()
