"""GGUF-verified Qwen3-14B/32B Phone-owned model and layer split tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phone_owned_models as m


class TestModelsAndSplits(unittest.TestCase):
    def mock_gguf(self, layers: int):
        return SimpleNamespace(
            arch="qwen3", n_layers=layers, embedding_length=5120,
            attention_head_count=40 if layers == 40 else 64,
            attention_head_count_kv=8, attention_key_length=128,
            rope_dimension_count=128,
        )

    def test_14b_defaults(self):
        with patch.object(m, "read_gguf_weights",
                          return_value=self.mock_gguf(40)):
            model = m.inspect_model(Path("any.gguf"))
        self.assertEqual((model.family, model.layers), ("14b", 40))
        self.assertEqual(m.resolve_splits(model), ([32], 16))
        self.assertEqual(m.resolve_splits(model, layouts="32,34,39"),
                         ([32, 34, 39], 16))

    def test_32b_defaults_and_overrides(self):
        with patch.object(m, "read_gguf_weights",
                          return_value=self.mock_gguf(64)):
            model = m.inspect_model(Path("any.gguf"))
        self.assertEqual((model.family, model.layers), ("32b", 64))
        self.assertEqual(m.resolve_splits(model), ([56], 44))
        self.assertEqual(m.resolve_splits(
            model, phone_layers=8, cuda_layers=8), ([56], 48))
        self.assertEqual(m.resolve_splits(
            model, layouts="56,58,60,63", cpu_layers=44),
            ([56, 58, 60, 63], 44))
        self.assertEqual(m.resolve_splits(
            model, phone_layers=4, cpu_layers=48), ([60], 48))

    def test_bad_gguf_dimensions_rejected(self):
        model = self.mock_gguf(64)
        with patch.object(m, "read_gguf_weights", return_value=model):
            with self.assertRaises(ValueError):
                m.inspect_model(Path("model.gguf"), "14b")
        model.attention_head_count = 40
        with patch.object(m, "read_gguf_weights", return_value=model):
            with self.assertRaises(ValueError):
                m.inspect_model(Path("model.gguf"))

    def test_bad_layers_and_unsupported_arch_rejected(self):
        model = self.mock_gguf(64)
        model.n_layers = 63
        with patch.object(m, "read_gguf_weights", return_value=model):
            with self.assertRaises(ValueError):
                m.inspect_model(Path("model.gguf"))
        model.arch = "qwen3moe"
        with patch.object(m, "read_gguf_weights", return_value=model):
            with self.assertRaises(ValueError):
                m.inspect_model(Path("model.gguf"))

    def test_invalid_splits_rejected(self):
        model = m.ModelLayout("32b", 64, 5120, 64, 8, 128)
        for options in (
            {"layouts": "55"}, {"layouts": "64"}, {"cpu_layers": 56},
            {"phone_layers": 0}, {"phone_layers": 9},
            {"cuda_layers": 0}, {"cuda_layers": 60},
            {"layouts": "56", "phone_layers": 8},
            {"layouts": "56", "cuda_layers": 12},
            {"cpu_layers": 44, "phone_layers": 8, "cuda_layers": 8},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                m.resolve_splits(model, **options)


if __name__ == "__main__":
    unittest.main()
