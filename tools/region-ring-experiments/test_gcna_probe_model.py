"""Compact G/C/N/A Profile is a real small model, not 40-layer CPU filler."""
import unittest

from gcna_probe_model import describe, create_probe, _gguf
from tempfile import TemporaryDirectory
from pathlib import Path

class GCNAProbeTests(unittest.TestCase):
    def test_default_layout_and_absolute_region_map(self):
        p = describe("G7,C1,N1,A1", 3)
        self.assertEqual(p["layers"], 30)
        self.assertEqual(p["runtime_layout"],
                         ",".join(["G7,C1,P1,P1"]*3))
        self.assertEqual(p["npu_regions"], [2,6,10])
        self.assertEqual(p["adreno_regions"], [3,7,11])
        self.assertEqual(p["sample_layers"],
                         {"G":21,"C":3,"N":3,"A":3})
        self.assertEqual(p["original_layer_indices"], list(range(30)))

    def test_compact_gguf_contains_only_sampled_blocks(self):
        try:
            import numpy as np
            gguf = _gguf()
        except ImportError:
            self.skipTest("numpy is not installed for GGUF roundtrip")
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            full = folder / "full.gguf"
            small = folder / "compact.gguf"
            writer = gguf.GGUFWriter(full, "qwen3")
            writer.add_uint32("qwen3.block_count", 10)
            writer.add_uint32("qwen3.embedding_length", 128)
            writer.add_tensor("token_embd.weight",
                              np.zeros((2, 128), dtype=np.float32))
            for layer in range(10):
                writer.add_tensor(f"blk.{layer}.attn_q.weight",
                                  np.full((2, 2), layer, dtype=np.float16))
            writer.add_tensor("output_norm.weight",
                              np.ones((128,), dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            plan = describe("G1,C1,N1,A1", 2)
            result = create_probe(full, small, plan)
            compact = gguf.GGUFReader(small)
            self.assertEqual(int(compact.get_field(
                "qwen3.block_count").contents()), 8)
            self.assertEqual(result["source_model_layers"], 10)
            self.assertEqual(result["layers"], 8)
            self.assertEqual(
                {t.name for t in compact.tensors if t.name.startswith("blk.")},
                {f"blk.{i}.attn_q.weight" for i in range(8)})
            self.assertIn("token_embd.weight",
                          {t.name for t in compact.tensors})

    def test_skip_optional_cpu(self):
        p = describe("G7,N1,A1",3)
        self.assertEqual(p["runtime_layout"],
                         "G7,P1,P1,G7,P1,P1,G7,P1,P1")
        self.assertEqual(p["npu_regions"], [1,4,7])

    def test_reject_unsupported_shape(self):
        for pattern,cycles in [
            ("G7,C1,N2,A1",3),
            ("G7,C1,A1,N1",3),
            ("G7,C1,N1,A1,G1",3),
            ("G7,C1,N1,A1",1),
            ("G7,C1,N1,A1",5),
            ("G0,C1,N1,A1",3),
            ("C1,N1,A1",3),
            ("G7,C1,N1",3),
            ("G7,C1,A1",3),
            ("G9,C1,N1,A1",3),
        ]:
            with self.subTest(pattern=pattern, cycles=cycles):
                with self.assertRaises(ValueError):
                    describe(pattern, cycles)

if __name__ == "__main__":
    unittest.main()
