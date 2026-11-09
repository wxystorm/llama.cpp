"""Device-free checks for persistent apply-wide Hybrid Profile identity."""
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from run_prompt_sweep import legacy_profile_cache_file


class SweepCacheTests(unittest.TestCase):
    def test_same_model_device_binary_reuses_one_path_across_prompts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "Qwen3-14B.gguf"
            model.write_bytes(b"model v1")
            engine = root / "llama-completion.exe"
            engine.write_bytes(b"engine v1")
            with mock.patch("run_v2_stage_matrix.host_device_id",
                            return_value="pc-device-a"):
                path = legacy_profile_cache_file(
                    root, model, engine, "10.39.52.114:50052",
                    8, False, 512, 512)
                self.assertEqual(path, legacy_profile_cache_file(
                    root, model, engine, "10.39.52.114:50052",
                    8, False, 512, 512))
                self.assertNotEqual(path, legacy_profile_cache_file(
                    root, model, engine, "10.39.52.114:50052",
                    6, False, 512, 512))
                self.assertNotEqual(path, legacy_profile_cache_file(
                    root, model, engine, "10.39.52.115:50052",
                    8, False, 512, 512))
                engine.write_bytes(b"engine v2")
                self.assertNotEqual(path, legacy_profile_cache_file(
                    root, model, engine, "10.39.52.114:50052",
                    8, False, 512, 512))
                engine.write_bytes(b"engine v1")
                model.write_bytes(b"model v2")
                self.assertNotEqual(path, legacy_profile_cache_file(
                    root, model, engine, "10.39.52.114:50052",
                    8, False, 512, 512))
            self.assertEqual(path.suffix, ".rrhp")
            self.assertEqual(path.parent.name, "hybrid-cache")


if __name__ == "__main__":
    unittest.main()
