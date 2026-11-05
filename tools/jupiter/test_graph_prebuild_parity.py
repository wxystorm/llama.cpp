#!/usr/bin/env python3
"""Offline tests for Jupiter graph-prebuild correctness log parsing."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_graph_prebuild_parity import extract, parser


class GraphPrebuildParityTests(unittest.TestCase):
    def test_extract_ignores_decode_and_finds_prompt_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "run.log"
            log.write_text(
                "[JUPITER_LOGITS_HASH] tokens=366 outputs=1 "
                "top1_id=42 top1_logit=2.5 digest=000000000000abcd\n"
                "[JUPITER_LOGITS_HASH] tokens=1 outputs=1 "
                "top1_id=43 top1_logit=3.5 digest=000000000000cdef\n",
                encoding="utf-8",
            )
            self.assertEqual(extract(log, 366), (1, 42, "000000000000abcd"))

    def test_extract_requires_matching_prompt_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "run.log"
            log.write_text(
                "[JUPITER_LOGITS_HASH] tokens=1 outputs=1 "
                "top1_id=1 top1_logit=2 digest=abcd\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing prompt logits digest"):
                extract(log, 366)

    def test_parser_requires_inputs(self) -> None:
        p = parser()
        with self.assertRaises(SystemExit):
            p.parse_args([])
        args = p.parse_args([
            "--exe", "llama-cli.exe", "--model", "model.gguf",
            "--rpc", "127.0.0.1:10139", "--prompt-file", "prompt.txt",
            "--plan", "jupiter.plan", "--target-tokens", "366",
        ])
        self.assertEqual(args.pc_layers, 24)


if __name__ == "__main__":
    unittest.main()
