#!/usr/bin/env python3
"""Standalone tests of Phone serial NPU log validation; no device needed."""
import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "check_phone_npu_serial", HERE / "check_phone_npu_serial.py")
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class SerialNpuLogTests(unittest.TestCase):
    @staticmethod
    def rows(run: int = 7, n: int = 4):
        result = []
        for region, layer in ((3, 19), (7, 39)):
            for chunk in range(n):
                result.append({
                    "event": "EXECUTE", "run": str(run),
                    "region": str(region), "layer": str(layer),
                    "chunk": str(chunk), "tokens": "128",
                    "backend": "HTP0", "compute_ms": "70.5",
                    "status": "PASS"})
        result.append({"event": "KV_SYNC", "run": str(run),
                       "bytes": "1048576", "status": "PASS"})
        return result

    def test_exact_coverage(self):
        ok, reason = MOD.check(self.rows(), {3, 7}, 4)
        self.assertTrue(ok, reason)
        self.assertIn("NPU_EXECUTION_PASS", reason)

    def test_missing_chunk_rejected(self):
        rows = [x for x in self.rows()
                if not (x["event"] == "EXECUTE" and
                        x["region"] == "3" and x["chunk"] == "2")]
        ok, reason = MOD.check(rows, {3, 7}, 4)
        self.assertFalse(ok)
        self.assertIn("CHUNKS_MISMATCH", reason)

    def test_gpu_fallback_rejected(self):
        rows = self.rows()
        rows[0]["backend"] = "GPUOpenCL"
        ok, reason = MOD.check(rows, {3, 7}, 4)
        self.assertFalse(ok)
        self.assertIn("NOT_HTP", reason)

    def test_kv_sync_required(self):
        rows = self.rows()[:-1]
        ok, reason = MOD.check(rows, {3, 7}, 4)
        self.assertFalse(ok)
        self.assertIn("KV_SYNC_NOT_CONFIRMED", reason)

    def test_warmup_run_excluded(self):
        rows = self.rows(run=4, n=1) + self.rows(run=5, n=4)
        ok, reason = MOD.check(rows, {3, 7}, 4)
        self.assertTrue(ok, reason)
        self.assertIn("run=5", reason)


if __name__ == "__main__":
    unittest.main()
