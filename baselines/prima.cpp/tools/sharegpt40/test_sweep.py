#!/usr/bin/env python3
"""Offline smoke tests for metrics parsing and true resume semantics."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import run_sweep


ROOT = Path(__file__).resolve().parent


class ShareGptSweepTests(unittest.TestCase):
    def test_parse_original_prima_style_timings(self) -> None:
        log = """
----- Allocation Strategy (by HiGHS) -----
Parameters:
  - k = 1
  - W = 64

LAPTOP-T6E32F3U:
  - Device Index   : 0
  - Assignment Set : M4
  - N Layer Window : 30
  - N GPU Layers   : 14

24129PN74C:
  - Device Index   : 1
  - Assignment Set : M3
  - N Layer Window : 34
  - N GPU Layers   : 0

llama_perf_context_print: prompt eval time = 1000.00 ms / 512 tokens ( 1.95 ms per token, 512.00 tokens per second)
llama_perf_context_print:        eval time = 1000.00 ms / 32 runs   (31.25 ms per token, 32.00 tokens per second)
"""
        data = run_sweep.parse_perf(log)
        self.assertEqual(data["actual_prefill_tokens"], 512)
        self.assertEqual(data["prefill_ms"], 1000.0)
        self.assertEqual(data["prefill_tps"], 512.0)
        self.assertEqual(data["decode_ms"], 1000.0)
        self.assertEqual(data["decode_tokens"], 32)
        self.assertEqual(data["planner_k"], 1)
        self.assertEqual(data["planner_W"], 64)
        self.assertEqual(data["pc_layers"], 30)
        self.assertEqual(data["pc_gpu_layers"], 14)
        self.assertEqual(data["phone_layers"], 34)
        self.assertEqual(data["phone_gpu_layers"], 0)

    def test_resume_skips_only_completed_cases(self) -> None:
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            prompts = root / "prompts"
            prompts.mkdir()
            for length in run_sweep.LENGTHS:
                for i in range(1, 11):
                    (prompts / f"prompt_{length}_{i:02d}.txt").write_text(
                        f"fixture {length} {i}\n", encoding="utf-8")
            model = root / "fixture.gguf"
            model.write_text("fixture", encoding="utf-8")
            binary = root / "mock-cli.sh"
            binary.write_text(
                "#!/bin/sh\n"
                "printf 'run\\n' >> \"$COUNT_FILE\"\n"
                "echo 'llama_perf_context_print: prompt eval time = 100.00 ms / 128 tokens (0.78 ms per token, 1280.00 tokens per second)'\n"
                "echo 'llama_perf_context_print: eval time = 50.00 ms / 2 runs (25.0 ms per token, 40.00 tokens per second)'\n",
                encoding="utf-8",
            )
            binary.chmod(0o755)
            count_file = root / "invocations"
            env = os.environ.copy()
            env["COUNT_FILE"] = str(count_file)
            output = root / "results"
            cmd = [
                sys.executable, str(ROOT / "run_sweep.py"),
                "--binary", str(binary), "--model", str(model),
                "--prompt-dir", str(prompts), "--out", str(output),
                "--master", "127.0.0.1", "--next", "127.0.0.1",
                "--delay", "0", "--timeout", "10",
            ]

            def run_once() -> None:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        encoding="utf-8", env=env, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout[-3000:])

            run_once()
            self.assertEqual(len(count_file.read_text().splitlines()), 40)
            self.assertEqual(len((output / "summary.csv").read_text().splitlines()), 41)
            run_once()
            self.assertEqual(len(count_file.read_text().splitlines()), 40)
            (output / "cases" / "prompt_256_03.json").unlink()
            run_once()
            self.assertEqual(len(count_file.read_text().splitlines()), 41)


if __name__ == "__main__":
    unittest.main()
