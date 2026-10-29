#!/usr/bin/env python3
"""Offline regression tests for Jupiter's auto-sized q(x,y) sweep runner."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from run_q_sweep import (
    build_cli_command,
    infer_runtime_prompt_tokens,
    make_sweep_plan,
    print_sweep_plans,
    reconstruct_plan_from_log,
    should_echo_runtime_line,
)


class JupiterSweepTests(unittest.TestCase):
    def _write_log(self, content: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "test.log"
        path.write_text(content, encoding="utf-8")
        return path

    def test_runtime_token_count_ignores_warmup(self) -> None:
        path = self._write_log(
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=2\n"
            "[JUPITER_UBATCH] event=QUEUED chunk=0 slot=0 prefix_tokens=0 tokens=2\n"
            "I slot new prompt task.n_tokens = 359\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=128\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=1 slot=0 prefix_tokens=128 tokens=128\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=2 slot=0 prefix_tokens=256 tokens=103\n"
            "I slot prompt eval time = 7200.0 ms / 359 tokens\n"
        )
        tokens, source = infer_runtime_prompt_tokens(path)
        self.assertEqual(tokens, 359)
        self.assertEqual(source, "ubatch_chain")

    def test_prefill_batch_overrides_task_token_count(self) -> None:
        # The sequence plan is matched to llama_context::decode() batch size,
        # which can be smaller than the outer server task's token count.
        path = self._write_log(
            "I slot task.n_tokens = 357\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=128\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=1 slot=0 prefix_tokens=128 tokens=128\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=2 slot=0 prefix_tokens=256 tokens=100\n"
            "I slot prompt eval time = 8000 ms / 357 tokens\n"
        )
        tokens, source = infer_runtime_prompt_tokens(path)
        self.assertEqual(tokens, 356)
        self.assertEqual(source, "ubatch_chain")

    def test_prompt_eval_fallback(self) -> None:
        path = self._write_log(
            "I slot prompt eval time = 7924.66 ms / 358 tokens\n"
        )
        self.assertEqual(infer_runtime_prompt_tokens(path), (358, "prompt_eval"))

    def test_ubatch_chain_fallback(self) -> None:
        path = self._write_log(
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=2\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=128\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=1 slot=0 prefix_tokens=128 tokens=128\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=2 slot=0 prefix_tokens=256 tokens=103\n"
        )
        self.assertEqual(infer_runtime_prompt_tokens(path), (359, "ubatch_chain"))

    def test_cli_uses_info_verbosity_for_jupiter_profiles(self) -> None:
        # LLAMA_JUPITER_Q_SWEEP promotes only the calibration marker records
        # to INFO in common/log.cpp; other TRACE messages remain suppressed.
        args = argparse.Namespace(
            rpc="127.0.0.1:10139",
            max_chunk=128,
            batch=512,
            context=512,
            threads=8,
            extra_arg=[],
        )
        cmd = build_cli_command(
            args, Path("llama-cli.exe"), Path("qwen.gguf"), "test prompt"
        )
        self.assertIn("-lv", cmd)
        self.assertEqual(cmd[cmd.index("-lv") + 1], "3")
        self.assertIn("--single-turn", cmd)

    def test_adaptive_chunks_are_valid(self) -> None:
        for total in (358, 359, 371, 401):
            for x in (32, 48, 64, 80, 96, 112, 128):
                with self.subTest(total=total, x=x):
                    chunks = make_sweep_plan(total, x, 32, 128)
                    self.assertEqual(sum(chunks), total)
                    self.assertTrue(all(32 <= c <= 128 for c in chunks))

    def test_high_b_sweep_falls_back_to_legal_first_chunk(self) -> None:
        # Repeated 112 leaves an impossible 133-token tail under max128,
        # but an exact 112-token first chunk can be profiled with [123,122].
        # A 96-token chunk cannot occur in any valid three-chunk 357 plan.
        plans = print_sweep_plans(357, [96, 112, 128], 96, 128)
        self.assertEqual([x for x, _ in plans], [112, 128])
        self.assertEqual(plans[0][1], [112, 123, 122])
        self.assertEqual(plans[1][1], [128, 128, 101])
        self.assertTrue(all(
            sum(chunks) == 357 and all(96 <= c <= 128 for c in chunks)
            for _, chunks in plans
        ))

    def test_console_suppresses_stage_noise_but_keeps_plan_and_errors(self) -> None:
        self.assertFalse(should_echo_runtime_line(
            "[JUPITER_STAGE_PROFILE] backend=RPC0 compute_ms=12.0"
        ))
        self.assertFalse(should_echo_runtime_line(
            "[JUPITER_UBATCH] event=BEGIN prefix_tokens=0 tokens=32"
        ))
        self.assertTrue(should_echo_runtime_line(
            "[JUPITER_SEQUENCE_PLAN] total=357 k=11 chunks=32,32"
        ))
        self.assertTrue(should_echo_runtime_line(
            "slot print_timing: prompt eval time = 8123 ms / 357 tokens"
        ))
        self.assertTrue(should_echo_runtime_line(
            "Error: model evaluation failed"
        ))

    def test_reconstruction_on_runtime_plan(self) -> None:
        expected = make_sweep_plan(359, 32, 32, 128)
        joined = ",".join(map(str, expected))
        path = self._write_log(
            "[JUPITER_SEQUENCE_PLAN] mode=PAPER_DP "
            f"total=359 k={len(expected)} chunks={joined} max_ubatch=128\n"
        )
        actual, source = reconstruct_plan_from_log(path, 359, 32)
        self.assertEqual(actual, expected)
        self.assertEqual(source, "SEQUENCE_PLAN")

    def test_reconstruction_from_ubatches(self) -> None:
        chunks = make_sweep_plan(359, 32, 32, 128)
        log = "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=2\n"
        pos = 0
        for index, size in enumerate(chunks):
            log += (
                f"[JUPITER_UBATCH] event=BEGIN chunk={index} slot=0 "
                f"prefix_tokens={pos} tokens={size}\n"
            )
            pos += size
        actual, source = reconstruct_plan_from_log(
            self._write_log(log), 359, 32
        )
        self.assertEqual(actual, chunks)
        self.assertEqual(source, "UBATCH_RECONSTRUCT")


if __name__ == "__main__":
    unittest.main()
