"""Offline tests for the Fig. 4(b) 32B fixed-Chunk experiment wrapper."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import run_32b_fig4b_fixed_chunks as fig


class Fig4bTests(unittest.TestCase):
    def args(self, root: Path, repeats=2):
        return fig.parse_args([
            "--exe", str(root / "llama-completion.exe"),
            "--model", str(root / "model.gguf"),
            "--rpc", "127.0.0.1:50052",
            "--prompt-dir", str(root / "prompts"),
            "--out", str(root / "results"),
            "--repeats", str(repeats),
        ])

    def test_default_layout_matches_full_cycle_32b(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            self.assertEqual(args.layout, "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7")
            parts = [(p[0], int(p[1:])) for p in args.layout.split(",")]
            self.assertEqual(sum(n for _, n in parts), 64)
            self.assertEqual(sum(n for k, n in parts if k == "G"), 43)
            self.assertEqual(sum(n for k, n in parts if k == "P"), 21)
            self.assertEqual(sum(n for k, n in parts if k == "C"), 0)

    def test_forces_chunk_but_not_planner_and_preserves_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            cmd = fig.runner_command(args, 64, 1)
            for flag in ("--layout", "--chunk", "--xc", "--xg", "--xp"):
                self.assertEqual(cmd[cmd.index(flag) + 1],
                                 fig.LAYOUT if flag == "--layout" else "64")
            self.assertIn("--force-small-batch-mover", cmd)
            self.assertIn("--htp-only", cmd)
            self.assertNotIn("--auto-plan", cmd)
            for key, value in fig.ENV_POLICY.items():
                self.assertIn(f"{key}={value}", cmd)
            self.assertEqual(cmd.count("--only-prompt"), 3)

    def populate(self, args, bad=False):
        expected = {
            128: {32: 100, 64: 105, 128: 120},
            512: {32: 210, 64: 190, 128: 205},
            1024: {32: 390, 64: 360, 128: 310},
        }
        for chunk in args.chunks:
            for repeat in range(1, args.repeats + 1):
                directory = fig.case_dir(args.out, chunk, repeat)
                directory.mkdir(parents=True)
                records = []
                for group in args.groups:
                    name = fig.prompt_name(group, args.prompt_index)
                    result = directory / (name + ".json")
                    result.write_text(json.dumps({"env_overrides": fig.ENV_POLICY}),
                                      encoding="utf-8")
                    records.append({
                        "prompt": name, "status": "PASS", "plan_mode": "fixed",
                        "group": group, "prefill_tokens": group + 15,
                        "context": group + 256, "prefill_ms": expected[group][chunk] + repeat - 1,
                        "layout": args.layout, "runtime_xc": 32 if bad and chunk == 64 and group == 512 else chunk,
                        "runtime_xg": chunk, "runtime_xp": chunk,
                        "gpu_bank_streaming": "True", "gpu_bank_force_small_batch": "True",
                        "result_file": str(result),
                    })
                with (directory / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)

    def test_crossing_curves_and_correct_groupwise_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            self.populate(args)
            passed, total, complete = fig.aggregate(args)
            self.assertEqual((passed, total, complete), (18, 18, True))
            with (args.out / "fig4b_plot_normalized.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            winners = {int(r["input_group"]): min(args.chunks, key=lambda c: float(r[f"chunk_{c}"]))
                       for r in rows}
            self.assertEqual(winners, {128: 32, 512: 64, 1024: 128})
            for group, chunk in winners.items():
                row = next(r for r in rows if int(r["input_group"]) == group)
                self.assertEqual(float(row[f"chunk_{chunk}"]), 1.0)

    def test_incomplete_or_invalid_runs_never_produce_plot_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            self.populate(args, bad=True)
            passed, total, complete = fig.aggregate(args)
            self.assertEqual((passed, total, complete), (16, 18, False))
            with (args.out / "fig4b_plot_normalized.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 0)


if __name__ == "__main__":
    unittest.main()