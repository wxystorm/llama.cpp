#!/usr/bin/env python3
"""Regression guard for the PC/Phone INPUT_STREAM backlog safety contract.

The PC and RPC server share a public budget definition, with a lower PC
preflight limit to prevent a NO_ACK sender from blocking a predecessor.
These tests cover the formerly rejected 32B 1855-token / six-Phone layout.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
RPC_HEADER = ROOT / "ggml" / "include" / "ggml-rpc.h"
PC_SOURCE = ROOT / "src" / "llama-context.cpp"
PHONE_SOURCE = ROOT / "ggml" / "src" / "ggml-rpc" / "ggml-rpc.cpp"


def budget_mib(name: str) -> int:
    text = RPC_HEADER.read_text(encoding="utf-8")
    match = re.search(r"^#define\s+" + re.escape(name) +
                      r"\s+(\d+)\s*$", text, re.M)
    if match is None:
        raise AssertionError(f"shared budget constant missing: {name}")
    return int(match.group(1))


def estimated_stream_bytes(tokens: int, hidden: int, regions: int) -> int:
    # Mirrors PC preflight: full-ubatch F32 per Phone Region + frame buffer.
    return tokens * hidden * 4 * regions + 4096 * regions


class PhoneInputStreamBudgetTests(unittest.TestCase):
    def test_32b_long_prompt_passes_with_margin(self):
        budget = budget_mib("GGML_RPC_PHONE_INPUT_STREAM_BUDGET_MIB")
        margin = budget_mib("GGML_RPC_PHONE_INPUT_STREAM_MARGIN_MIB")
        self.assertEqual((budget, margin), (256, 8))
        needed = estimated_stream_bytes(1855, 5120, 6)
        self.assertAlmostEqual(needed / (1024 * 1024), 217.40625, places=3)
        self.assertLess(needed, (budget - margin) * (1024 * 1024))

    def test_32b_2000_tokens_still_fits_safely(self):
        budget = budget_mib("GGML_RPC_PHONE_INPUT_STREAM_BUDGET_MIB")
        margin = budget_mib("GGML_RPC_PHONE_INPUT_STREAM_MARGIN_MIB")
        self.assertLess(estimated_stream_bytes(2000, 5120, 6),
                        (budget - margin) * (1024 * 1024))

    def test_large_plan_is_still_rejected(self):
        budget = budget_mib("GGML_RPC_PHONE_INPUT_STREAM_BUDGET_MIB")
        margin = budget_mib("GGML_RPC_PHONE_INPUT_STREAM_MARGIN_MIB")
        self.assertGreater(estimated_stream_bytes(2304, 5120, 6),
                           (budget - margin) * (1024 * 1024))

    def test_14b_three_phone_regions_also_fits(self):
        budget = budget_mib("GGML_RPC_PHONE_INPUT_STREAM_BUDGET_MIB")
        margin = budget_mib("GGML_RPC_PHONE_INPUT_STREAM_MARGIN_MIB")
        self.assertLess(estimated_stream_bytes(1855, 5120, 3),
                        (budget - margin) * (1024 * 1024))

    def test_pc_and_phone_share_same_constant(self):
        pc = PC_SOURCE.read_text(encoding="utf-8")
        rpc = PHONE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("GGML_RPC_PHONE_INPUT_STREAM_BUDGET_MIB", pc)
        self.assertIn("GGML_RPC_PHONE_INPUT_STREAM_BUDGET_MIB", rpc)
        self.assertIn("GGML_RPC_PHONE_INPUT_STREAM_MARGIN_MIB", pc)
        self.assertIn("phone_stream_expected_bytes", pc)
        self.assertIn("phone_stream_preflight_limit", pc)
        self.assertIn("PHASE_A_BACKLOG_BUDGET", pc)
        self.assertIn("phone_input_stream_held_bytes <=", rpc)
        self.assertIn("phone_input_stream_budget - bytes", rpc)
        self.assertIn("std::chrono::seconds(30)", rpc)
        self.assertIn("BACKPRESSURE_TIMEOUT", rpc)
        self.assertNotIn("BYTE_BUDGET_128M", pc)


if __name__ == "__main__":
    unittest.main()
