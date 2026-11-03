import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from audit_plan_calibration import extract, display


class AuditTests(unittest.TestCase):
    def test_measured_358_tokens_with_no_network(self):
        with TemporaryDirectory() as folder:
            run = Path(folder)
            case = run / "prompt_358"
            case.mkdir()
            with (run / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["status", "commit", "actual_prefill_tokens", "ring_wall_ms", "final_dag_sim_ms", "runtime_layout", "phone_final_head_ms", "cpu_compute_range_ms", "gpu_compute_range_ms"])
                writer.writeheader()
                writer.writerow(dict(status="PASS", commit="abc", actual_prefill_tokens="358", ring_wall_ms="3381.195", final_dag_sim_ms="1929.6708", runtime_layout="G10,C1,P3,G11,C1,P1,G11,C1,P1", phone_final_head_ms="184.839", cpu_compute_range_ms="2079.89", gpu_compute_range_ms="2104.939"))
            (case / "engine.stderr.log").write_text("[PHONE_AUTO] event=GRAPH_SETUP_DONE count=18 setup_ms=432.766 policy=ONE_GRAPH_PER_CHUNK_KV_SAFE status=OK\n[P2C_STREAM] phase=STOP recv=18 pending=0 status=OK\n", encoding="utf-8")
            (case / "metrics.jsonl").write_text("\n".join([json.dumps(dict(type="planner", predicted_ms=1929.6708, cpu_busy_pred_ms=1880, gpu_busy_pred_ms=1990, phone_busy_pred_ms=1200)), json.dumps(dict(type="region_ring", tokens=358, wall_ms=3381.195, phone_final_head_ms=184.839))]) + "\n", encoding="utf-8")
            phone = run / "rpc.log"
            phone.write_text("[PHONE_AUTO_SUM] run=45 registered=18 completed=18 failed=0 compute_ms=1283.637 status=PASS\n[PHONE_INPUT_STREAM_SUM] run=45 frames=18 recv_ms=1455.856 status=END\n[P2C_STREAM] side=PHONE phase=STOP run=2 frames=18 send_ms=296.438 status=OK\n", encoding="utf-8")
            result = extract(run, "prompt_358", phone)
            self.assertEqual(result["p2c_frames_pc"], 18.0)
            self.assertEqual(result["p2c_frames_phone"], 18.0)
            self.assertAlmostEqual(result["recorded_prediction_error_ms"], 1451.5242)
            self.assertAlmostEqual(result["phone_auto_setup_per_graph_measured_ms"], 24.0425556, places=6)
            self.assertIn('PHONE_AUTO_HEAD_MS = "184.839"', display(result))

    def test_missing_phone_log_is_graceful(self):
        with TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "prompt_358").mkdir()
            (run / "summary.csv").write_text("status,actual_prefill_tokens,ring_wall_ms\nPASS,358,3381\n", encoding="utf-8")
            result = extract(run, "prompt_358", None)
            self.assertIsNone(result["phone_compute_measured_ms"])
            self.assertEqual(result["ring_wall_ms"], 3381)


if __name__ == "__main__":
    unittest.main()
