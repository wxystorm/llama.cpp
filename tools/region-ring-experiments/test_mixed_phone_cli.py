"""End-to-end offline CLI regression with a synthetic model-bound Profile."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import sys
import types
import unittest

from mixed_phone_dag_plan import run_cli
from test_mixed_phone_dag_plan import BASE, MIX


class CliTests(unittest.TestCase):
    def test_model_bound_profile_only_plan(self):
        with TemporaryDirectory() as td:
            d = Path(td)
            model = d / "model.gguf"
            model.write_bytes(b"GGUF")
            prompt = d / "prompt.txt"
            prompt.write_text("test")
            tokenizer = d / "tokenizer.exe"
            tokenizer.write_text("stub")
            folder = d / "profile"
            folder.mkdir()
            info = {"file": model.name, "fingerprint": "sha-test",
                    "layers": 40, "hidden": 5120,
                    "kv_bytes_per_token_per_layer": 4096}
            base = {k: v for k, v in BASE.items() if k not in
                    ("gpu_kv_layer_mib", "npu_kv_layer_mib")}
            mixed = {k: v for k, v in MIX.items() if k != "phone_gpu_kv_layer_mib"}
            (folder / "model_profile.json").write_text(json.dumps({
                "schema": 1, "model": info, "calibration": base,
                "stage_csv": "real-stage-profile.csv"}))
            (folder / "phone_mixed_calibration.json").write_text(json.dumps({
                "schema": 1, "model_fingerprint": "sha-test",
                "stage_csv": "real-stage-mixed.csv", "calibration": mixed}))
            rows = ("device,layers,tokens,kv_tokens,setup_ms,upload_ms,"
                    "prepare_ms,compute_ms,terminal,samples\n")
            for dev, depth, cost in [("CPU", 1, 3), ("GPU", 7, 2),
                                     ("PHONE", 1, 12), ("PHONE_GPU", 1, 43)]:
                for kv in [64, 128, 192, 256]:
                    rows += f"{dev},{depth},64,{kv},0,0,0.8,{cost},0,1\n"
            (folder / "real-stage-mixed.csv").write_text(rows)
            fake = types.ModuleType("model_profile")
            fake.gguf_info = lambda _: info
            fake.tokenize = lambda *args: 256
            with patch.dict(sys.modules, {"model_profile": fake}):
                run_cli(["--model", str(model), "--prompt", str(prompt),
                         "--profile-dir", str(folder),
                         "--tokenizer", str(tokenizer),
                         "--out", str(d / "out")])
            found = json.loads((d / "out" / "plan.json").read_text())
            self.assertEqual(found["schema"], 2)
            self.assertIsInstance(found["runtime_compatible"], bool)
            self.assertGreater(found["dag"]["nodes"], 0)
            self.assertTrue((d / "out" / "candidates.csv").is_file())
            with patch.dict(sys.modules, {"model_profile": fake}):
                run_cli(["--model", str(model), "--prompt", str(prompt),
                         "--profile-dir", str(folder),
                         "--tokenizer", str(tokenizer),
                         "--out", str(d / "out-runtime"),
                         "--runtime-compatible"])
            live = json.loads((d / "out-runtime" / "plan.json").read_text())
            self.assertTrue(live["runtime_compatible"])
            self.assertTrue(live["runtime_experimental"])
            self.assertGreaterEqual(len(live["phone_npu_mapping"]), 2)
            self.assertGreaterEqual(len(live["phone_gpu_mapping"]), 1)
            self.assertIn("HTP_TO_ADRENO",
                          [e["handoff"] for e in live["execution_edges"]])


if __name__ == "__main__":
    unittest.main()
