"""Offline checks that the quick ShareGPT runner forwards a CPU/GPU/Phone split."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jupiter_memory_partition import DeviceMemory, GGUFWeights, GIB, MIB
from run_sharegpt_quick8 import build_command, choose_layers_by_memory


class JupiterQuickCPUStageTests(unittest.TestCase):
    def test_three_tier_auto_split_and_forwarding(self):
        weights = GGUFWeights("qwen2", (256*MIB,)*64, 0, 0)
        args = argparse.Namespace(
            model="fake.gguf", rpc="127.0.0.1:10139", rpc_device_index=0,
            pc_free_mib=None, phone_free_mib=None, cpu_free_mib=None,
            pc_reserve_mib=0, phone_reserve_mib=0,
            cpu_reserve_mib=0, pc_layers=None,
            cpu_workspace_mib=512, gpu_workspace_mib=1024,
            phone_workspace_mib=1536, memory_safety_percent=10.0,
            kv_bytes_per_element=2.0, kv_per_layer_mib=8.0,
            exe="llama-cli.exe", min_chunk=96, max_chunk=128,
            ubatch=128, batch=2048, context=2048, threads=8, window=2,
            q_repeats=1, refine_repeats=1, bench_repeats=1, max_rounds=1,
            graph_prebuild=True, strict_refine=False,
        )
        with (
            patch("run_sharegpt_quick8.read_gguf_weights", return_value=weights),
            patch("run_sharegpt_quick8.pc_gpu_memory",
                  return_value=DeviceMemory(4*GIB, 6*GIB)),
            patch("run_sharegpt_quick8.phone_rpc_memory",
                  return_value=DeviceMemory(4*GIB, 8*GIB)),
            patch("run_sharegpt_quick8.pc_cpu_memory",
                  return_value=DeviceMemory(16*GIB, 24*GIB)),
        ):
            plan = choose_layers_by_memory(args)
        self.assertGreater(plan.cpu_layers, 0)
        self.assertEqual(plan.cpu_layers + plan.pc_gpu_layers + plan.phone_layers, 64)
        args.pc_layers = plan.pc_layers
        args.cpu_layers = plan.cpu_layers
        with tempfile.TemporaryDirectory() as temp:
            command = build_command(args, Path(temp), Path(temp))
            self.assertEqual(command[command.index("--cpu-layers") + 1],
                             str(plan.cpu_layers))
            self.assertEqual(command[command.index("--pc-layers") + 1],
                             str(plan.pc_layers))
            self.assertIn("--budgeted-refine", command)
            self.assertIn("--graph-prebuild", command)
            args.limit = 1
            args.fail_fast = True
            smoke = build_command(args, Path(temp), Path(temp))
            self.assertEqual(smoke[smoke.index("--limit") + 1], "1")
            self.assertIn("--fail-fast", smoke)


if __name__ == "__main__":
    unittest.main()
