#!/usr/bin/env python3
"""Offline checks for the low-noise Jupiter prefill A/B runner."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from run_prefill_bench import (
    build_cli,
    env_for_mode,
    modes_arg,
    parse_timing,
    read_plan,
)


class PrefillBenchTests(unittest.TestCase):
    def _file(self, content: str) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        file = Path(temporary.name) / "test.txt"
        file.write_text(content, encoding="utf-8")
        return file

    def test_plan_reads_exact_sequence_length(self) -> None:
        path = self._file(
            "# Jupiter plan\nmodel.pc_layers=25\n"
            "sequence.357=64,64,64,64,64,37\n"
        )
        self.assertEqual(
            read_plan(path), {357: [64, 64, 64, 64, 64, 37]}
        )

    def test_plan_rejects_wrong_sum(self) -> None:
        path = self._file("sequence.357=64,64,64,64,64,38\n")
        with self.assertRaises(ValueError):
            read_plan(path)

    def test_timing_uses_last_server_prompt_eval(self) -> None:
        path = self._file(
            "other text\n"
            "0.00 I slot print_timing: prompt eval time = 100.0 ms / 2 tokens\n"
            "0.31 I slot print_timing: prompt eval time = 9739.04 ms / 357 tokens"
            " (27.28 ms per token, 36.66 tokens per second)\n"
        )
        self.assertEqual(parse_timing(path), (9739.04, 357))

    def test_modes_are_distinct(self) -> None:
        self.assertEqual(modes_arg("dp,default"), ["dp", "default"])
        with self.assertRaises(argparse.ArgumentTypeError):
            modes_arg("dp,dp")

    def test_mode_environment_and_profiling_off(self) -> None:
        base = {
            "LLAMA_JUPITER_PREFILL": "1",
            "LLAMA_JUPITER_WINDOW": "2",
            "LLAMA_JUPITER_PROFILE_STAGE": "0",
            "LLAMA_JUPITER_TRACE": "0",
            "LLAMA_JUPITER_LOGITS_TOP10": "0",
            "LLAMA_JUPITER_SEQUENCE_PLAN": "old",
        }
        plan = Path("jupiter.plan")
        chunks = [64, 64, 64, 64, 64, 37]
        self.assertNotIn(
            "LLAMA_JUPITER_SEQUENCE_PLAN",
            env_for_mode(base, "default", plan, chunks),
        )
        self.assertEqual(
            env_for_mode(base, "fixed64", plan, chunks)[
                "LLAMA_JUPITER_SEQUENCE_PLAN"
            ],
            "64,64,64,64,64,37",
        )
        dp = env_for_mode(base, "dp", plan, chunks)
        self.assertEqual(dp["LLAMA_JUPITER_PLAN_FILE"], "jupiter.plan")
        self.assertNotIn("LLAMA_JUPITER_SEQUENCE_PLAN", dp)
        self.assertEqual(dp["LLAMA_JUPITER_PROFILE_STAGE"], "0")

    def test_formal_cli_uses_lv3_single_turn(self) -> None:
        args = argparse.Namespace(
            rpc="127.0.0.1:10139", ubatch=128, batch=512,
            context=512, threads=8,
        )
        command = build_cli(
            args, Path("llama-cli.exe"), Path("qwen.gguf"), "hello", 3
        )
        self.assertEqual(command[command.index("-lv") + 1], "3")
        self.assertIn("--single-turn", command)
        self.assertIn("--simple-io", command)
        self.assertIn("-n", command)
        self.assertEqual(command[command.index("-n") + 1], "1")


if __name__ == "__main__":
    unittest.main()
