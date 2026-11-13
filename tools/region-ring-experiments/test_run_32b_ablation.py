"""Offline unit tests for Qwen3-32B four-case ablation planning."""
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_32b_ablation as a


class AblationTests(unittest.TestCase):
    @staticmethod
    def options():
        return SimpleNamespace(
            htp_session_mib=2880,
            gpu_reserve_mib=768,
            cpu_reserve_mib=512,
            context=4096,
        )

    def test_independent_profile_native_tokens_column(self):
        rows = (
            "device,layers,tokens,kv_tokens,setup_ms,upload_ms,"
            "prepare_ms,compute_ms,terminal,samples\\n"
            "CPU,2,128,128,0,0,0,400,0,1\\n"
            "GPU,2,128,128,0,0,0,20,0,1\\n"
            "PHONE,2,128,256,0,0,0,70,0,1\\n"
        ).replace("\\\\n", "\\n")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "profile.csv"
            path.write_text(rows, encoding="utf-8")
            sample = a.measured_ms_per_layer(path, 128)
            self.assertEqual(sample, {"C": 200.0, "G": 10.0, "N": 35.0})

    def test_a0_uses_fixed_htp_without_auto_profile(self):
        with tempfile.TemporaryDirectory() as d:
            args = SimpleNamespace(
                model=Path(d) / "model.gguf",
                exe=Path(d) / "llama-completion.exe",
                rpc="127.0.0.1:50052",
                prompt_dir=Path(d),
                out=Path(d),
                chunk=128, threads=8, limit=1, only_prompt="",
                dry_run=False, plan_only=False)
            with patch.object(a.subprocess, "call", return_value=0) as launch:
                self.assertEqual(a.run_one("A0", a.FIXED, args), 0)
            command = launch.call_args.args[0]
            self.assertIn("--htp-only", command)
            self.assertIn("--layout", command)
            self.assertNotIn("--auto-plan", command)
            self.assertNotIn("--force-plan-layout", command)

    def test_gguf_layer_tensor_offsets(self):
        def string(s):
            b = s.encode("utf-8")
            return struct.pack("<Q", len(b)) + b
        prefix = b"GGUF" + struct.pack("<IQQ", 3, 2, 0)
        for i, offset in ((0, 0), (1, 128)):
            prefix += string("blk.%d.attn_q.weight" % i)
            prefix += struct.pack("<I", 1)  # one dim
            prefix += struct.pack("<QIQ", 128, 0, offset)
        prefix += b"\0" * ((-len(prefix)) % 32)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tiny.gguf"
            p.write_bytes(prefix + b"\0" * 256)
            self.assertEqual(a.gguf_layer_weights(p, 2), [128, 128])

    def test_single_bank_counts_only_peak_region(self):
        weights = [100 * a.MIB] * 16
        layout = [("G", 6), ("P", 2), ("G", 6), ("P", 2)]
        self.assertTrue(a.layout_fits(
            layout, weights, 1000, 2000, 2500, 2880,
            100, 100, 512))
        self.assertFalse(a.layout_fits(
            layout, weights, 700, 2000, 2500, 2880,
            100, 100, 512))

    def test_multi_cycle_respects_memory(self):
        weights = [100 * a.MIB] * 64
        opts = self.options()
        layout, details = a.multi_cycle(
            weights, {"G": 4.0, "C": 25.0, "N": 10.0},
            2500, 6000, 4500, opts)
        parts = [(s[0], int(s[1:])) for s in layout.split(",")]
        self.assertEqual(sum(n for _, n in parts), 64)
        self.assertGreaterEqual(sum(k == "G" for k, _ in parts), 2)
        self.assertTrue(a.layout_fits(
            parts, weights, 2500, 6000, 4500,
            opts.htp_session_mib, opts.gpu_reserve_mib,
            opts.cpu_reserve_mib, opts.context))
        self.assertGreater(details["candidates"], 0)

    def test_single_cycle_prioritizes_gpu_then_cpu(self):
        weights = [100 * a.MIB] * 64
        opts = self.options()
        layout, details = a.single_cycle(
            weights, 2500, 6000, 4500, opts)
        kinds = [s[0] for s in layout.split(",")]
        self.assertEqual(kinds, ["G", "C", "P"])
        self.assertEqual(sum(int(s[1:]) for s in layout.split(",")), 64)
        self.assertEqual(details["objective"], "MAX_GPU_THEN_CPU_REMAINDER_TO_PHONE")


if __name__ == "__main__":
    unittest.main()
