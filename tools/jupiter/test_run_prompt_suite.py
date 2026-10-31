"""Offline tests for the Jupiter multi-prompt length experiment runner."""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from run_prompt_suite import (
    case_folder,
    case_identity,
    discover_prompts,
    ineligible_prompt_reason,
    launch_logged,
    main,
    read_benchmark_results,
)


def stage2():
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


class JupiterPromptSuiteTests(unittest.TestCase):
    def test_discovery_deduplicates_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.txt"
            b = root / "b.txt"
            a.write_text("a")
            b.write_text("b")
            self.assertEqual(
                discover_prompts(root, [str(a)], "*.txt", False), [a, b]
            )
            with self.assertRaisesRegex(ValueError, "not found"):
                discover_prompts(root / "missing", [], "*.txt", False)

    def test_prompt_lengths_sort_numerically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("prompt_1024.txt", "prompt_64.txt", "prompt_128.txt"):
                (root / name).write_text("sample", encoding="utf-8")
            found = discover_prompts(root, [], "prompt_*.txt", False)
            self.assertEqual(
                [p.name for p in found],
                ["prompt_64.txt", "prompt_128.txt", "prompt_1024.txt"],
            )

    def test_live_phase_progress_is_echoed_but_runtime_noise_is_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "stage3-run.log"
            code = (
                "print('[JUPITER_SWEEP_RUN] x=64 repeat=1/1', flush=True);"
                "print('[JUPITER_SEQUENCE_DECISION] noisy', flush=True);"
                "print('[JUPITER_REFINE_SAMPLE] target=72:240', flush=True);"
                "print('[JUPITER_REFINE_DONE] status=FULLY_MEASURED', flush=True)"
            )
            visible = io.StringIO()
            with contextlib.redirect_stdout(visible):
                launch_logged([sys.executable, "-c", code], log, "STAGE3")
            shown = visible.getvalue()
            full_log = log.read_text(encoding="utf-8")
            self.assertIn("[JUPITER_SWEEP_RUN]", shown)
            self.assertIn("[JUPITER_REFINE_SAMPLE]", shown)
            self.assertIn("[JUPITER_REFINE_DONE]", shown)
            self.assertNotIn("[JUPITER_SEQUENCE_DECISION]", shown)
            self.assertIn("[JUPITER_SEQUENCE_DECISION]", full_log)

    def test_summary_checks_complete_repeat_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "results.csv"
            def write(lines):
                with p.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "mode", "repeat", "tokens", "prefill_ms"
                    ])
                    writer.writeheader()
                    writer.writerows(lines)
            rows = [
                {"mode": mode, "repeat": repeat, "tokens": 357,
                 "prefill_ms": ms}
                for mode, values in (
                    ("default", [8000, 8200]),
                    ("fixed64", [9000, 9100]),
                    ("dp", [8500, 8600]),
                )
                for repeat, ms in enumerate(values, 1)
            ]
            write(rows)
            metrics = read_benchmark_results(
                p, 357, ["default", "fixed64", "dp"], 2
            )
            self.assertEqual(metrics["default_median_ms"], 8100)
            self.assertEqual(metrics["dp_median_ms"], 8550)
            self.assertAlmostEqual(metrics["speedup_vs_default"], 0.9474)
            write(rows[:-1])
            with self.assertRaisesRegex(ValueError, "expected 2"):
                read_benchmark_results(
                    p, 357, ["default", "fixed64", "dp"], 2
                )

    def test_fixed_b_eligibility_and_no_legal_length_gap(self):
        self.assertIn("fixed_b=96", ineligible_prompt_reason(26, stage2(), 96))
        self.assertIn("fixed_b=96", ineligible_prompt_reason(257, stage2(), 96))
        self.assertIsNone(ineligible_prompt_reason(357, stage2(), 96))
        self.assertIsNone(ineligible_prompt_reason(100, stage2(), 96))
        self.assertIsNone(ineligible_prompt_reason(357, stage2(), None))
        self.assertIn(
            "cannot be partitioned",
            ineligible_prompt_reason(1, stage2(), None),
        )

    def test_two_prompt_pipeline_and_resume_isolated_by_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = root / "prompts"
            prompts.mkdir()
            (prompts / "0alpaca.txt").write_text("synthetic short prompt")
            (prompts / "a357.txt").write_text("synthetic first prompt")
            (prompts / "b1024.txt").write_text("synthetic longer prompt")
            exe = root / "llama-cli.exe"
            model = root / "model.gguf"
            exe.write_text("fake executable")
            model.write_text("fake model")
            stage2_path = root / "min-chunk.json"
            stage2_path.write_text(json.dumps(stage2()))
            out = root / "results"

            commands: list[tuple[str, int]] = []

            def fake_probe(args, prompt, pc_layers, log, resume):
                self.assertEqual(pc_layers, 20)
                return (
                    26 if prompt.stem == "0alpaca"
                    else 357 if prompt.stem == "a357"
                    else 1024
                ), "synthetic"

            def fake_launch(command, log_path, phase):
                def get(name):
                    return command[command.index(name) + 1]
                n = int(get("--target-tokens"))
                commands.append((phase, n))
                if phase == "STAGE3":
                    self.assertIn("--reuse-b-profile-for-new-length", command)
                    self.assertEqual(get("--min-chunk"), "96")
                    plan = Path(get("--plan-output"))
                    chunks = [119, 119, 119] if n == 357 else [128] * 8
                    plan.write_text(
                        "model.pc_layers=20\n"
                        f"sequence.{n}={','.join(map(str, chunks))}\n"
                    )
                    output = Path(get("--out-dir"))
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "sequence-summary.json").write_text(json.dumps({
                        "status": "FULLY_MEASURED_AND_VERIFIED",
                        "pc_layers": 20,
                        "prompt_tokens": n,
                        "min_chunk": 96,
                        "chunks": chunks,
                    }))
                else:
                    self.assertEqual(phase, "BENCH")
                    destination = Path(get("--out-dir"))
                    destination.mkdir(parents=True, exist_ok=True)
                    reps = int(get("--repeats"))
                    with (destination / "results.csv").open(
                        "w", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            "mode", "repeat", "tokens", "prefill_ms"
                        ])
                        writer.writeheader()
                        for mode, t in (
                            ("default", 8000), ("fixed64", 8500), ("dp", 7800)
                        ):
                            for k in range(1, reps + 1):
                                writer.writerow({
                                    "mode": mode, "repeat": k,
                                    "tokens": n, "prefill_ms": t,
                                })

            argv = [
                "run_prompt_suite.py",
                "--exe", str(exe), "--model", str(model),
                "--rpc", "127.0.0.1:10139",
                "--prompts-dir", str(prompts),
                "--min-chunk-json", str(stage2_path),
                "--min-chunk", "96",
                "--bench-repeats", "2",
                "--out-dir", str(out),
            ]
            with patch.object(sys, "argv", argv), \
                 patch("run_prompt_suite.probe_tokens", side_effect=fake_probe), \
                 patch("run_prompt_suite.launch_logged", side_effect=fake_launch):
                self.assertEqual(main(), 0)
            records = list(csv.DictReader(
                (out / "suite-results.csv").open(encoding="utf-8")
            ))
            self.assertEqual({row["actual_tokens"] for row in records},
                             {"26", "357", "1024"})
            self.assertEqual(
                {row["status"] for row in records},
                {"PASS", "SKIPPED_INELIGIBLE"},
            )
            skipped = next(row for row in records if row["actual_tokens"] == "26")
            self.assertIn("fixed_b=96", skipped["skip_reason"])
            self.assertEqual(skipped["error"], "")
            self.assertEqual(len(commands), 4)

            commands.clear()
            with patch.object(sys, "argv", argv + ["--resume"]), \
                 patch("run_prompt_suite.probe_tokens", side_effect=fake_probe), \
                 patch("run_prompt_suite.launch_logged", side_effect=fake_launch):
                self.assertEqual(main(), 0)
            self.assertEqual(commands, [("BENCH", 357), ("BENCH", 1024)])

    def test_case_id_changes_with_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "sample.txt"
            first.write_text("first", encoding="utf-8")
            second = root / "sample2.txt"
            second.write_text("second", encoding="utf-8")
            self.assertNotEqual(
                case_folder(root, first, {"prompt_sha256": "first"}),
                case_folder(root, second, {"prompt_sha256": "second"}),
            )


if __name__ == "__main__":
    unittest.main()
