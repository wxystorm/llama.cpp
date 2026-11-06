#!/usr/bin/env python3
"""Offline tests for the ShareGPT40 Jupiter DP-only sweep."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

from run_sharegpt40_dp import (
    benchmark_ready, discover, existing_rows, feasible_min_chunk,
    parser, save_results, stage3_ready,
)


class ShareGPT40DPTests(unittest.TestCase):
    def test_discover_uses_numeric_group_and_index_and_excludes_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in (
                "prompt_512_10.txt", "prompt_128_02.txt", "prompt_128_01.txt",
                "prompt_1024_03.txt", "prompt_256_01.txt",
                "prompt_2048_01.txt", "prompt_128.txt", "notes.txt",
            ):
                (folder / name).write_text("hello", encoding="utf-8")
            found = discover(folder, [128, 256, 512, 1024])
            self.assertEqual(
                [path.name for _, path in found],
                ["prompt_128_01.txt", "prompt_128_02.txt",
                 "prompt_256_01.txt", "prompt_512_10.txt",
                 "prompt_1024_03.txt"],
            )

    def test_adaptive_min_chunk_keeps_every_length_feasible(self):
        for total in range(60, 1501):
            b = feasible_min_chunk(total, 96, 128)
            self.assertGreater(b, 0)
            self.assertLessEqual(b, 96)
            self.assertGreaterEqual(total // b, (total + 127) // 128)
        self.assertEqual(feasible_min_chunk(128, 96, 128), 96)
        self.assertEqual(feasible_min_chunk(129, 96, 128), 64)
        self.assertEqual(feasible_min_chunk(191, 96, 128), 95)
        self.assertEqual(feasible_min_chunk(257, 96, 128), 85)
        self.assertEqual(feasible_min_chunk(366, 96, 128), 96)
        with self.assertRaises(ValueError):
            feasible_min_chunk(0, 96, 128)

    def test_parser_defaults_to_one_formal_dp_run_no_prebuild(self):
        args = parser().parse_args([])
        self.assertEqual(args.bench_repeats, 1)
        self.assertEqual(args.q_repeats, 1)
        self.assertFalse(args.graph_prebuild)
        self.assertEqual(args.groups, "128,256,512,1024")
        self.assertFalse(args.budgeted_refine)
        self.assertFalse(args.only_incomplete)

    def test_stage3_resume_requires_explicit_fixed_verified_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage3 = root / "stage3"
            stage3.mkdir()
            plan = root / "final.plan"
            plan.write_text("model.pc_layers=24\nsequence.366=128,123,115\n")
            summary = {
                "status": "FULLY_MEASURED_AND_VERIFIED",
                "prompt_tokens": 366, "pc_layers": 24,
                "min_chunk": 96, "max_chunk": 128,
                "setup_source": "FIXED_EXPLICIT",
                "chunks": [128, 123, 115],
            }
            (stage3 / "sequence-summary.json").write_text(json.dumps(summary))
            args = argparse.Namespace(pc_layers=24, max_chunk=128)
            self.assertTrue(stage3_ready(stage3, plan, 366, 96, args))
            self.assertFalse(stage3_ready(stage3, plan, 366, 64, args))
            summary["setup_source"] = "STAGE2_JSON"
            (stage3 / "sequence-summary.json").write_text(json.dumps(summary))
            self.assertFalse(stage3_ready(stage3, plan, 366, 96, args))

    def test_resume_accepts_budgeted_measured_verified_and_preserves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage3 = root / "stage3"
            stage3.mkdir()
            plan = root / "final.plan"
            plan.write_text("model.pc_layers=24\nsequence.1057=" +
                            ",".join(["117"] * 8 + ["121"]) + "\n")
            summary = {
                "status": "BUDGETED_MEASURED_AND_VERIFIED",
                "prompt_tokens": 1057, "pc_layers": 24,
                "min_chunk": 96, "max_chunk": 128,
                "setup_source": "FIXED_EXPLICIT",
                "chunks": [117] * 8 + [121],
            }
            (stage3 / "sequence-summary.json").write_text(json.dumps(summary))
            args = argparse.Namespace(pc_layers=24, max_chunk=128)
            self.assertTrue(stage3_ready(stage3, plan, 1057, 96, args))
            row = {
                "group": 512, "prompt": "prompt_512_01.txt", "status": "PASS",
                "actual_tokens": "600", "prefill_ms": "10000",
                "prefill_tok_s": "60.0", "refinement_policy": "EXACT",
            }
            save_args = argparse.Namespace(prompts_dir=str(root),
                                           bench_repeats=1, graph_prebuild=True)
            save_results(root, [row], save_args)
            loaded = existing_rows(root / "dp-results.csv")
            self.assertEqual(loaded["prompt_512_01.txt"]["prefill_tok_s"], "60.0")
            save_results(root, [loaded["prompt_512_01.txt"]], save_args)
            self.assertEqual(existing_rows(root / "dp-results.csv"), loaded)

    def test_completed_dp_benchmark_never_accepts_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench = root / "bench"
            bench.mkdir()
            exe, model = root / "llama-cli.exe", root / "model.gguf"
            exe.write_bytes(b"exe")
            model.write_bytes(b"model")
            prompt = root / "prompt_256_01.txt"
            prompt.write_text("some text\n", encoding="utf-8")
            plan = root / "final.plan"
            plan.write_text("sequence.366=128,123,115\n")
            args = argparse.Namespace(
                exe=str(exe), model=str(model), rpc="127.0.0.1:10139",
                pc_layers=24, bench_repeats=1, ubatch=128,
                batch=2048, context=2048, window=2, graph_prebuild=False,
            )
            import hashlib
            manifest = {
                "prompt_sha256": hashlib.sha256(b"some text").hexdigest(),
                "exe": str(exe), "model": str(model),
                "target_tokens": 366, "dp_chunks": [128, 123, 115],
                "pc_layers": 24, "cpu_layers": 0,
                "modes": ["dp"], "repeats": 1,
                "ubatch": 128, "batch": 2048, "context": 2048,
                "window": 2, "graph_prebuild": False,
                "rpc": args.rpc, "plan": str(plan),
            }
            (bench / "configuration.json").write_text(json.dumps(manifest))
            with (bench / "results.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "mode", "repeat", "tokens", "prefill_ms"
                ])
                writer.writeheader()
                writer.writerow({
                    "mode": "dp", "repeat": 1, "tokens": 366,
                    "prefill_ms": 9000,
                })
            self.assertEqual(
                benchmark_ready(bench, prompt, plan, 366, [128, 123, 115], args)
                ["dp_median_ms"], 9000,
            )
            manifest["modes"] = ["default", "dp"]
            (bench / "configuration.json").write_text(json.dumps(manifest))
            self.assertIsNone(
                benchmark_ready(bench, prompt, plan, 366, [128, 123, 115], args)
            )


if __name__ == "__main__":
    unittest.main()
