#!/usr/bin/env python3
"""No-device tests for production Bank and Phone-local HTP->Adreno telemetry."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from merge_v2_runtime_profile import collect, merge


def log_lines(*, bank_status="PASS", edge_status="PASS",
              incomplete=0, overflow=0, windows=4,
              bank_count=3, edge_count=4, suffix=""):
    return (
        "[PHONE_BANK_RUNTIME_PROFILE] source=RPC_PATCH33 run=818 refills=" +
        str(bank_count) + " failed=0 inflight=0 unstaged=0 overflow=" +
        str(overflow) + " device_windows=" + str(windows) +
        " queue_ms=2.1 host_stage_wall_ms=30.0 htp_lock_wait_ms=1.3"
        " htp_critical_wall_ms=43.0 precompile_ms=5.0"
        " host_stage_htp_overlap_ms=19.0"
        " host_stage_adreno_overlap_ms=11.0"
        " htp_critical_adreno_overlap_ms=24.0"
        " status=" + bank_status + " overlap_state=MEASURED\n"
        "[PHONE_LOCAL_HANDOFF_RUNTIME_PROFILE] source=RPC_PATCH33"
        " run=818 edges=" + str(edge_count) +
        " incomplete=" + str(incomplete) + " bytes=5242880"
        " host_move_ms=0.6 ready_to_upload_ms=3.3"
        " upload_ms=4.2 ready_to_device_ms=13.8"
        " host_copy_bytes=0 intermediate_p2c_bytes=0"
        " status=" + edge_status + "\n"
        "[PHONE_STAGE_PROFILE_RPC] event=COMPLETE"
        " run=818 samples=12 status=PASS\n" + suffix
    )


class RuntimeProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "engine.stderr.log"
        self.profile = self.root / "profile.json"
        self.out = self.root / "profile-with-runtime.json"
        self.profile.write_text(json.dumps({
            "schema": 2, "model_sha256": "f" * 64,
            "compute_samples": [{"sample": {"compute_median_ms": "24.0"}}],
            "planner_ready": False,
            "pending": ["real_dual_bank_overlapped_refill_profile",
                        "runtime_memory_feasibility"],
        }))

    def test_full_run_timed_from_single_phone_clock(self):
        self.log.write_text(log_lines())
        rows = collect(self.log)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], 818)
        self.assertTrue(rows[0]["bank"]["usable_as_runtime_calibration"])
        self.assertTrue(rows[0]["npu_to_adreno"]["usable_as_runtime_calibration"])
        self.assertAlmostEqual(
            rows[0]["bank"]["host_stage_htp_overlap_fraction"], 19/30)
        self.assertEqual(rows[0]["clock_domain"], "PHONE_GGML_TIME_US")
        out = merge(self.profile, self.log, self.out)
        self.assertFalse(out["planner_ready"])
        self.assertFalse(out["runtime_telemetry"]["planner_eligible"])
        self.assertEqual(out["runtime_telemetry"]["bank_measured_run_count"], 1)
        self.assertEqual(
            out["runtime_telemetry"]["npu_to_adreno_measured_run_count"], 1)
        self.assertNotIn("real_dual_bank_overlapped_refill_profile",
                         out["pending"])
        self.assertTrue(self.out.exists())
        self.assertEqual(
            json.loads(self.profile.read_text())["compute_samples"][0]
                ["sample"]["compute_median_ms"], "24.0")

    def test_no_refill_does_not_count_as_dual_bank(self):
        self.log.write_text(
            log_lines(bank_status="NOT_EXERCISED", bank_count=0,
                      windows=0, edge_status="NOT_EXERCISED", edge_count=0))
        out = merge(self.profile, self.log, self.out)
        self.assertFalse(out["runtime_telemetry"]["runs"][0]
                         ["bank"]["usable_as_runtime_calibration"])
        self.assertEqual(out["runtime_telemetry"]["bank_measured_run_count"], 0)
        self.assertIn("real_dual_bank_overlapped_refill_profile",
                      out["pending"])

    def test_overflow_or_incomplete_does_not_qualify(self):
        self.log.write_text(log_lines(
            bank_status="INCOMPLETE", overflow=1,
            edge_status="INCOMPLETE", incomplete=1))
        result = collect(self.log)[0]
        self.assertFalse(result["bank"]["usable_as_runtime_calibration"])
        self.assertFalse(
            result["npu_to_adreno"]["usable_as_runtime_calibration"])

    def test_does_not_invent_missing_phone_rpc_end(self):
        self.log.write_text(log_lines().replace(
            "[PHONE_STAGE_PROFILE_RPC] event=COMPLETE",
            "[PHONE_STAGE_PROFILE_RPC] event=BEGIN"))
        self.assertEqual(collect(self.log), [])
        with self.assertRaisesRegex(ValueError,
                                    "NO_COMPLETE_PATCH33_PRODUCTION_RPC"):
            merge(self.profile, self.log, self.out)

    def test_rejects_overlap_beyond_real_interval(self):
        self.log.write_text(log_lines().replace(
            "host_stage_htp_overlap_ms=19.0",
            "host_stage_htp_overlap_ms=31.0"))
        with self.assertRaisesRegex(ValueError, "BANK_OVERLAP_EXCEEDS"):
            collect(self.log)

    def test_rejects_different_samples_claiming_same_run(self):
        self.log.write_text(log_lines() + log_lines(bank_count=5))
        with self.assertRaisesRegex(ValueError, "CONFLICTING_RUNTIME_SAMPLE"):
            collect(self.log)


if __name__ == "__main__":
    unittest.main()
