#!/usr/bin/env python3
"""Offline checks for the Phone Full residual ADD A/B reporter."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "phone_full_compare", ROOT / "check_phone_full_add_fusion.py")
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


def example(single_add):
    graph_commands = 6 + single_add
    return [{
        "type": "region_ring", "layout": "G11,C1,P2", "xc": 160,
        "xg": 32, "xp": 64, "wall_ms": 1100.0,
        "phone_rpc_regions": [{
            "region": 2, "layer_begin": 12, "layer_end": 14,
            "chunks": 6, "graph_multi_count": 6,
            "graph_multi_nodes": 414 + 6 * (single_add == 0),
            "graph_single_add_count": single_add,
            "graph_single_other_count": 0,
            "graph_compute_count": graph_commands,
            "graph_recompute_count": 0, "graph_snapshot_count": 0,
            "get_tensor_count": 6, "get_tensor_bytes": 7331840,
        }],
    }]


class CompareTests(unittest.TestCase):
    def test_fused_add_eliminated(self):
        self.assertEqual(COMPARE.compare(example(6), example(0)), 6)

    def test_missing_fusion_rejected(self):
        with self.assertRaisesRegex(ValueError, "No separate ADD"):
            COMPARE.compare(example(6), example(6))

    def test_changed_return_rejected(self):
        after = example(0)
        after[0]["phone_rpc_regions"][0]["get_tensor_bytes"] += 4
        with self.assertRaisesRegex(ValueError, "return payload"):
            COMPARE.compare(example(6), after)

    def test_changed_layout_rejected(self):
        after = example(0)
        after[0]["layout"] = "G10,C2,P2"
        with self.assertRaisesRegex(ValueError, "Layout/Chunk config"):
            COMPARE.compare(example(6), after)


if __name__ == "__main__":
    unittest.main()
