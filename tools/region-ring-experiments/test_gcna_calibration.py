"""Fail-closed and model-bound Phone resource calibration using PC RPC logs."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import sys
import unittest
from unittest.mock import patch

from build_gcna_calibration import assemble

RESOURCE = (
    "[PHONE_PROFILE_RESOURCES] side=PHONE source=RPC run=42 "
    "gpu_free_bytes=5242880000 gpu_total_bytes=7500000000 "
    "host_available_bytes=6400000000 gpu_stage_arena_bytes=20971520 "
    "host_stage_budget_bytes=536870912 host_stage_bytes=104857600 "
    "host_stage_ms=25 repack_bytes=104857600 repack_ms=50 status=PASS\n"
)

class CalibrationTests(unittest.TestCase):
    def build_fixture(self, root: Path):
        m = root / "model.gguf"
        m.write_bytes(b"dummy")
        profile = root / "profile"
        profile.mkdir()
        info = {"file": m.name, "fingerprint": "fingerprint"}
        (profile / "model_profile.json").write_text(
            json.dumps({"model":info}))
        probe = root / "probe.json"
        probe.write_text(json.dumps({
            "source_model_fingerprint": "fingerprint",
            "phone_weight_source": "ORIGINAL_GGUF_FIRST_L_LAYERS",
            "mode": "COMPACT_GCNA_PROBE_NOT_FULL_INFERENCE",
            "layers": 30, "source_model_layers":40,
            "pattern":"G7,C1,N1,A1","cycles":3,
            "probe_model_fingerprint":"mini-fp"
        }))
        runs = root / "runs"
        case = runs / "x64" / "20261107-123456"
        case.mkdir(parents=True)
        (case/"result.json").write_text('{"status":"PASS"}')
        (case/"metrics.jsonl").write_text("")
        (case/"engine.stderr.log").write_text(
            "[PHONE_STAGE_PROFILE_RPC] event=COMPLETE run=42 samples=12 status=PASS\n"
            + RESOURCE)
        stages = root / "real-stage-mixed.csv"
        with stages.open("w",newline="",encoding="utf-8") as out:
            w = csv.DictWriter(out,fieldnames=[
                "device","layers","tokens","kv_tokens","setup_ms",
                "upload_ms","prepare_ms","compute_ms","terminal","samples"])
            w.writeheader()
            for dev in ("GPU","CPU","PHONE","PHONE_GPU"):
                w.writerow(dict(device=dev,layers=1,tokens=64,
                    kv_tokens=64,setup_ms=0,upload_ms=1,
                    prepare_ms=1,compute_ms=2,terminal=0,samples=1))
        return m, profile, runs, probe, stages, case

    def test_real_phone_rates_and_model_provenance(self):
        with TemporaryDirectory() as td:
            args = self.build_fixture(Path(td))
            fake = types.ModuleType("model_profile")
            fake.gguf_info = lambda _: {"file":args[0].name,
                                          "fingerprint":"fingerprint"}
            with patch.dict(sys.modules,{"model_profile":fake}), \
                    patch("build_gcna_calibration.weight_layer_mib",
                          return_value=180.5):
                result = assemble(*args[:-1])
            c = result["calibration"]
            self.assertAlmostEqual(c["npu_host_stage_mibps"],4000.)
            self.assertAlmostEqual(c["npu_bank_repack_mibps"],2000.)
            self.assertEqual(c["npu_host_stage_budget_mib"],512.)
            self.assertEqual(c["phone_gpu_scratch_mib"],40.)
            self.assertEqual(c["phone_gpu_weight_layer_mib"],180.5)
            self.assertNotIn("phone_local_mibps",c)
            self.assertLess(c["phone_shared_budget_mib"],
                            c["phone_gpu_budget_mib"]+1000)

    def test_missing_bank_refill_fails_not_faked(self):
        with TemporaryDirectory() as td:
            args = self.build_fixture(Path(td))
            log = args[-1] / "engine.stderr.log"
            log.write_text(log.read_text().replace(
                "host_stage_bytes=104857600","host_stage_bytes=0"))
            fake = types.ModuleType("model_profile")
            fake.gguf_info = lambda _: {"file":args[0].name,
                                          "fingerprint":"fingerprint"}
            with patch.dict(sys.modules,{"model_profile":fake}), \
                    patch("build_gcna_calibration.weight_layer_mib",
                          return_value=180.5):
                with self.assertRaisesRegex(ValueError, "NOT_MEASURED"):
                    assemble(*args[:-1])

if __name__ == "__main__":
    unittest.main()
