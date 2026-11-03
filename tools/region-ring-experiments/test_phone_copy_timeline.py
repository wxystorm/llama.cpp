#!/usr/bin/env python3
"""Small synthetic fixture for the opt-in Phone tensor Copy CSV exporter."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "region_ring_sweep", ROOT / "run_prompt_sweep.py")
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class PhoneCopyTimelineTest(unittest.TestCase):
    def test_only_measured_records_and_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            records = [
                {"type": "region_ring_phone_copy", "ubatch": 0,
                 "region": 2, "chunk": 0, "tensor": "warmup",
                 "kind": "RPC_SET", "bytes": 999},
                {"type": "measurement_start"},
                {"type": "region_ring_phone_copy", "ubatch": 1,
                 "region": 5, "chunk": 1, "split": 0,
                 "tensor": "norm-24", "dst_tensor": "norm-24_copy",
                 "kind": "RPC_SET", "bytes": 4096, "copy_call_ms": 4.5,
                 "rpc_set_calls": 1, "rpc_set_bytes": 4096,
                 "rpc_apply_calls": 0},
                {"type": "region_ring_phone_copy", "ubatch": 1,
                 "region": 2, "chunk": 0, "split": 0,
                 "tensor": "inp", "dst_tensor": "inp_copy",
                 "kind": "PUSH_APPLY", "bytes": 1048576,
                 "copy_call_ms": 12.5, "rpc_apply_calls": 1,
                 "rpc_apply_bytes": 1048576, "rpc_set_calls": 0,
                 "rpc_set_bytes": 0},
            ]
            (folder / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8")
            self.assertEqual(SWEEP.write_phone_copy_timeline(folder), 2)
            with (folder / "phone_copy_timeline.csv").open(
                    newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["kind"] for r in rows],
                             ["PUSH_APPLY", "RPC_SET"])
            self.assertEqual(rows[0]["tensor"], "inp")
            self.assertEqual(int(rows[0]["rpc_set_bytes"]), 0)
            self.assertEqual(rows[1]["tensor"], "norm-24")
            self.assertEqual(int(rows[1]["rpc_set_bytes"]), 4096)

    def test_disabled_has_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(SWEEP.write_phone_copy_timeline(folder), 0)
            self.assertFalse((folder / "phone_copy_timeline.csv").exists())


if __name__ == "__main__":
    unittest.main()
