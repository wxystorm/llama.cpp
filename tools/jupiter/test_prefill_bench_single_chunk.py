#!/usr/bin/env python3
"""Offline regression tests for Jupiter single-chunk Default benchmark."""
from __future__ import annotations

import unittest

from run_prefill_bench import extract_prefill_chunks, single_chunk_ubatch


class JupiterSingleChunkDefaultTests(unittest.TestCase):
    def test_default_ubatch_covers_entire_prompt(self) -> None:
        self.assertEqual(single_chunk_ubatch(366), 512)
        self.assertEqual(single_chunk_ubatch(512), 512)
        self.assertEqual(single_chunk_ubatch(513), 1024)
        with self.assertRaises(ValueError):
            single_chunk_ubatch(0)

    def test_single_chunk_prefill(self) -> None:
        log = (
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=2 t_us=1\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=366 t_us=2\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=1 t_us=3\n"
        )
        self.assertEqual(extract_prefill_chunks(log, 366), [366])

    def test_dp_prefill_and_false_default(self) -> None:
        log = (
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=128 t_us=1\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=1 slot=1 prefix_tokens=128 tokens=123 t_us=2\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=2 slot=0 prefix_tokens=251 tokens=115 t_us=3\n"
        )
        self.assertEqual(extract_prefill_chunks(log, 366), [128, 123, 115])
        self.assertNotEqual(extract_prefill_chunks(log, 366), [366])

    def test_missing_or_broken_trace_fails(self) -> None:
        self.assertIsNone(extract_prefill_chunks("", 366))
        log = (
            "[JUPITER_UBATCH] event=BEGIN chunk=0 slot=0 prefix_tokens=0 tokens=128 t_us=1\n"
            "[JUPITER_UBATCH] event=BEGIN chunk=1 slot=1 prefix_tokens=129 tokens=238 t_us=2\n"
        )
        self.assertIsNone(extract_prefill_chunks(log, 366))


if __name__ == "__main__":
    unittest.main()
