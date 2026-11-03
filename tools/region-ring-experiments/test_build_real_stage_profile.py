import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from build_real_stage_profile import aggregate, case_dirs, extract


class RealStageProfileTests(unittest.TestCase):
    def test_join_real_sessions_and_exclude_queued_time(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            case = root / "run" / "prompt_358"
            case.mkdir(parents=True)
            traces = [
                dict(type="region_ring_chunk_trace", device="GPU", region=0,
                     chunk=0, layer_begin=0, layer_end=10, tokens=16,
                     token_begin=0, compute_range_ms=31, prepare_total_ms=2),
                dict(type="region_ring_chunk_trace", device="CPU", region=1,
                     chunk=0, layer_begin=10, layer_end=11, tokens=16,
                     token_begin=0, compute_range_ms=11, prepare_total_ms=0.5),
                dict(type="region_ring_chunk_trace", device="PHONE", region=2,
                     chunk=0, layer_begin=11, layer_end=14, tokens=64,
                     token_begin=0, compute_range_ms=0),
            ]
            (case / "metrics.jsonl").write_text(
                "\n".join(json.dumps(x) for x in traces), encoding="utf-8")
            (case / "engine.stderr.log").write_text(
                "[PHONE_AUTO] event=SESSION_READY run=123 status=READY\n"
                "[REAL_STAGE_PROFILE] side=PC phase=GRAPH_SETUP region=2 chunk=0 "
                "tokens=64 layers=3 kv_tokens=64 terminal=1 setup_ms=24.043\n",
                encoding="utf-8")
            rpc = root / "rpc.log"
            rpc.write_text(
                "[REAL_STAGE_PROFILE] side=PHONE phase=CHUNK run=123 region=2 "
                "chunk=0 tokens=64 upload_ms=6.5 prepare_ms=3.5 "
                "compute_ms=128.25 compute_queue_ms=400 status=PASS\n",
                encoding="utf-8")
            raw, diag = extract(case_dirs([root]), rpc)
            self.assertEqual(diag["phone"], 1)
            self.assertEqual(diag["cpu"], 1)
            self.assertEqual(diag["gpu"], 1)
            rows = aggregate(raw)
            self.assertEqual(len(rows), 3)
            phone = next(x for x in rows if x["device"] == "PHONE")
            self.assertEqual(phone["setup_ms"], 24.043)
            self.assertEqual(phone["prepare_ms"], 3.5)
            self.assertEqual(phone["upload_ms"], 6.5)
            self.assertEqual(phone["compute_ms"], 128.25)
            self.assertEqual(phone["terminal"], 1)

    def test_no_remote_data_means_no_fabricated_phone_cost(self):
        with TemporaryDirectory() as temp:
            case = Path(temp)
            (case / "metrics.jsonl").write_text(json.dumps(dict(
                type="region_ring_chunk_trace", device="PHONE", region=8,
                chunk=0, tokens=64, token_begin=0,
                layer_begin=39, layer_end=40)), encoding="utf-8")
            rows, diag = extract([case], None)
            self.assertFalse(rows)
            self.assertEqual(diag["phone_missing"], 1)
            self.assertEqual(aggregate(rows), [])


if __name__ == "__main__":
    unittest.main()
