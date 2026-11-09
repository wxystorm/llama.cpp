"""No-device validation of native G/C/N/A cache readiness."""
from pathlib import Path
import csv
import json
import tempfile
import unittest

from v2_profile_coordinator import validated_profile, select_profile_dir


class NativeProfileCoordinatorTests(unittest.TestCase):
    def test_accepts_only_known_model_and_correct_n1_a1_with_refill(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "gcna"
            folder.mkdir()
            samples = []
            for backend, layers, parity in [
                    ("G", 2, "NOT_REQUESTED"),
                    ("C", 2, "NOT_REQUESTED"),
                    ("N", 1, "PASS"),
                    ("A", 1, "NOT_REQUESTED")]:
                samples.append({
                    "status": "PASS",
                    "key": {"backend": backend, "layers": layers,
                            "chunk": 64, "kv_tokens": 512},
                    "sample": {"compute_median_ms": "2.4",
                               "parity": parity}})
            manifest = {
                "schema": 2,
                "model_sha256": "modelhash",
                "devices": {
                    "pc": {"device_id": "pc", "exe_sha256": "exe"},
                    "phone": {"device_id": "phone",
                              "runtime_sha256": "runtime"}},
                "compute_samples": samples,
                "parity_fail_samples": [
                    {"status": "PARITY_FAIL",
                     "key": {"backend": "N", "layers": 4}}],
                "npu_weight_refill_points": [
                    {"backend": "N", "layers": 1,
                     "refill_median_ms": 4.2}],
            }
            (folder / "profile.json").write_text(json.dumps(manifest))
            with (folder / "stage-profile-v2.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "backend", "layers", "chunk", "kv_tokens",
                    "compute_median_ms", "prepare_median_ms",
                    "weight_bytes", "kv_bytes",
                    "graph_workspace_bytes", "parity", "source"])
                writer.writeheader()
                for record in samples:
                    writer.writerow({
                        "backend": record["key"]["backend"],
                        "layers": record["key"]["layers"],
                        "chunk": 64, "kv_tokens": 512,
                        "compute_median_ms": 2.4,
                        "prepare_median_ms": 0.2,
                        "weight_bytes": 1200, "kv_bytes": 120,
                        "graph_workspace_bytes": 80,
                        "parity": record["sample"]["parity"],
                        "source": "NATIVE_SINGLE_BACKEND_GGUF_PREFIX",
                    })
            self.assertEqual(validated_profile(
                folder, "modelhash", "pc", "exe"), (True, "VERIFIED"))
            self.assertEqual(validated_profile(
                folder, "another_model", "pc", "exe"),
                (False, "MODEL_FINGERPRINT_MISMATCH"))
            self.assertEqual(validated_profile(
                folder, "modelhash", "pc", "new_exe"),
                (False, "PC_BINARY_OR_DEVICE_MISMATCH"))
            del manifest["npu_weight_refill_points"]
            (folder / "profile.json").write_text(json.dumps(manifest))
            self.assertEqual(validated_profile(
                folder, "modelhash", "pc", "exe"),
                (False, "N1_WEIGHT_REFILL_PROFILE_MISSING"))
            self.assertEqual(select_profile_dir(
                root, root / "model.gguf", "modelhash",
                preferred=folder), folder)


if __name__ == "__main__":
    unittest.main()
