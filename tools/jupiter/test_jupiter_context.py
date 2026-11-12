#!/usr/bin/env python3
"""Offline unit tests for Jupiter's 14B automatic context sizing."""
from __future__ import annotations

import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jupiter_context import context_for_tokens, model_token_counts, tokenizer_path


class JupiterContextTests(unittest.TestCase):
    def test_matches_region_ring_context_policy(self):
        self.assertEqual(context_for_tokens(167), 1024)
        self.assertEqual(context_for_tokens(587), 1024)
        self.assertEqual(context_for_tokens(1013), 1280)
        self.assertEqual(context_for_tokens(1752), 2048)
        self.assertEqual(context_for_tokens(2050), 2304)
        self.assertEqual(context_for_tokens(2050, minimum=4096), 4096)
        self.assertEqual(context_for_tokens(167, fixed=1024), 1024)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            context_for_tokens(1013, fixed=1024)
        with self.assertRaises(ValueError):
            context_for_tokens(0)

    def test_tokenizer_path_uses_neighbor_binary_or_override(self):
        self.assertEqual(
            tokenizer_path("/tmp/bin/llama-cli", "/tmp/custom-tokenize"),
            Path("/tmp/custom-tokenize").resolve(),
        )

    def test_gguf_count_is_cached_and_invalidated_by_prompt_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, binary = root / "model.gguf", root / "llama-tokenize"
            model.write_bytes(b"model")
            binary.write_bytes(b"tokenizer")
            prompt = root / "prompt_128_01.txt"
            prompt.write_text("first", encoding="utf-8")
            cache = root / "cache" / "prompt-tokens.json"
            completed = subprocess.CompletedProcess(
                [], 0, stdout="Total number of tokens: 167\n", stderr=""
            )
            with patch("jupiter_context.subprocess.run", return_value=completed) as run:
                self.assertEqual(
                    model_token_counts(model, binary, [prompt], cache)[prompt.name],
                    167,
                )
                self.assertEqual(
                    model_token_counts(model, binary, [prompt], cache)[prompt.name],
                    167,
                )
                self.assertEqual(run.call_count, 1)
                prompt.write_text("changed", encoding="utf-8")
                model_token_counts(model, binary, [prompt], cache)
                self.assertEqual(run.call_count, 2)

    def test_cache_write_can_be_disabled_for_memory_check(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, binary, prompt = (root / name for name in
                                     ("model.gguf", "llama-tokenize", "prompt_128_01.txt"))
            for path in (model, binary, prompt):
                path.write_text("test", encoding="utf-8")
            output = root / "unused" / "tokens.json"
            with patch("jupiter_context.subprocess.run", return_value=
                       subprocess.CompletedProcess([], 0,
                           stdout="Total number of tokens: 20", stderr="")):
                self.assertEqual(model_token_counts(model, binary, [prompt],
                                                     output, persist=False)[prompt.name], 20)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
