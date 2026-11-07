"""Synthetic offline regression: model-specific Profile and plan-only fail-closed."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import struct
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_profile import gguf_info, probe_layouts, fingerprint


def tiny_model(path):
    values = {
        "general.architecture": ("qwen3", 8),
        "qwen3.block_count": (40, 4),
        "qwen3.embedding_length": (5120, 4),
        "qwen3.attention.head_count": (40, 4),
        "qwen3.attention.head_count_kv": (8, 4),
        "qwen3.attention.key_length": (128, 4),
        "qwen3.attention.value_length": (128, 4),
    }
    def s(value):
        return struct.pack("<Q", len(value.encode())) + value.encode()
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(values))
    for key, (value, kind) in values.items():
        data += s(key) + struct.pack("<I", kind)
        data += s(value) if kind == 8 else struct.pack("<I", value)
    path.write_bytes(data)


class ProfileTests(unittest.TestCase):
    def test_internal_htp_p1_geometry(self):
        p = probe_layouts(40)
        self.assertEqual(p["phone_regions"], "2,5,8,11")
        self.assertEqual(p["phone_layers"], "9,19,29,39")
        for layout in p["layouts"]:
            self.assertEqual(sum(int(x[1:]) for x in layout.split(",")), 40)
            self.assertEqual(layout.count("P1"), 4)

    def test_genuine_gguf_metadata_and_plan_fails_without_profile(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "Qwen3-14B-Q4_K_M.gguf"
            tiny_model(model)
            info = gguf_info(model)
            self.assertEqual(info["layers"], 40)
            self.assertEqual(info["hidden"], 5120)
            self.assertEqual(info["kv_bytes_per_token_per_layer"], 4096)
            self.assertEqual(info["fingerprint"], fingerprint(model))
            prompt = root / "prompt.txt"
            prompt.write_text("test")
            cmd = [sys.executable, str(Path(__file__).with_name("model_profile.py")),
                   "plan", "--model", str(model), "--profile-dir", str(root / "missing"),
                   "--prompt", str(prompt), "--tokenizer", str(root / "tokenize"),
                   "--planner", str(root / "planner"), "--out", str(root / "out")]
            p = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("No model Profile", p.stderr)
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
