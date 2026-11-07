"""A PC run log containing pulled Patch32 Phone samples needs no phone.log."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from build_real_stage_profile import extract

class RPCProfileExtractorTests(unittest.TestCase):
    def test_phone_records_join_to_pc_chunk_by_run_id(self):
        with TemporaryDirectory() as tmp:
            case = Path(tmp) / "x64" / "run"
            case.mkdir(parents=True)
            (case/"result.json").write_text('{"status":"PASS"}')
            rows = []
            for region,device,first,last in [
                (0,"GPU",0,7), (1,"CPU",7,8),
                (2,"PHONE",8,9), (3,"PHONE",9,10)]:
                rows.append(dict(type="region_ring_chunk_trace",region=region,
                    chunk=0,device=device,layer_begin=first,layer_end=last,
                    token_begin=0,tokens=64,prepare_total_ms=0.4,
                    compute_range_ms=4.5))
            (case/"metrics.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows))
            (case/"engine.stderr.log").write_text(
                "[PHONE_STAGE_PROFILE_RPC] event=COMPLETE run=4242 "
                "samples=2 status=PASS\n"
                # Adreno Graph is prebuilt; no per-Chunk setup record.
                "[REAL_STAGE_PROFILE] side=PHONE phase=CHUNK source=RPC "
                "run=4242 region=2 chunk=0 tokens=64 token_begin=0 "
                "backend=HTP0 upload_ms=-1.000 prepare_ms=-1.000 compute_ms=3 "
                "status=PASS\n"
                "[REAL_STAGE_PROFILE] side=PHONE phase=CHUNK source=RPC "
                "run=4242 region=3 chunk=0 tokens=64 token_begin=0 "
                "backend=GPUOpenCL upload_ms=0.4 prepare_ms=0.5 "
                "compute_ms=2 status=PASS\n"
                # This stale data belongs to another session and cannot be
                # silently included in the Profile.
                "[REAL_STAGE_PROFILE] side=PHONE phase=CHUNK source=RPC "
                "run=4243 region=2 chunk=0 tokens=64 token_begin=0 "
                "backend=HTP0 upload_ms=99 prepare_ms=99 compute_ms=99 "
                "status=PASS\n")
            samples, diag = extract([case], None, mixed=True)
            self.assertEqual(len(samples), 4)
            self.assertEqual({row["device"] for row in samples},
                             {"GPU","CPU","PHONE","PHONE_GPU"})
            htp = next(x for x in samples if x["device"]=="PHONE")
            self.assertEqual(htp["compute_ms"], 3)
            self.assertEqual(htp["prepare_ms"], 0)
            self.assertEqual(htp["upload_ms"], 0)
            self.assertEqual(diag["phone_missing"], 0)
            self.assertEqual(diag["setup_missing"], 0)

if __name__ == "__main__":
    unittest.main()
