#!/usr/bin/env python3
"""Offline regression test for sweep CSV parsing. No GGUF, RPC or CUDA required."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("run_prompt_sweep.py")
SPEC = importlib.util.spec_from_file_location("run_prompt_sweep", MODULE_PATH)
assert SPEC and SPEC.loader
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class ExperimentReportTests(unittest.TestCase):
    def test_warmup_excluded_and_actual_tokens_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "prompt_128"
            folder.mkdir()
            prompt = root / "prompt_128.txt"
            prompt.write_text("This is not exactly 128 model tokens.", encoding="utf-8")
            model = root / "model.gguf"
            model.write_bytes(b"fake model for parser test")
            rows = [
                {"type": "planner", "applied": True, "selected_layout":
                 "G6,C2,P4,G6,C2,P4,G6,C2,P4,G6,C2,P4", "xc": 32, "xg": 64,
                 "xp": 128, "first_cycle_e": 0.3, "e": 0.2,
                 "predicted_ms": 195.125, "baseline_predicted_ms": 220.5,
                 "plan_tokens": 128,
                 "prefix_e": [0.3, 0.25, 0.22, 0.2],
                 "prefix_service_mspt": [[1, 2, 3], [2, 3, 4]]},
                {"type": "region_ring", "tokens": 512, "complete": True,
                 "status": 0, "wall_ms": 900, "layout": "G4,C4,P4",
                 "phone_rpc_available": True, "phone_rpc_regions": [
                     {"region": 2, "set_tensor_count": 999}]},
                {"type": "measurement_start", "schema": 1},
                {"type": "region_ring", "tokens": 126, "complete": True,
                 "status": 0, "wall_ms": 100.0, "cpu_compute_range_ms": 8.0,
                 "meta_available": True, "meta_pc_to_phone_copy_ms": 7.0,
                 "meta_phone_to_pc_copy_ms": 3.0,
                  "phone_rpc_available": True,
                  "ubatch": 0,
                  "phone_rpc_regions": [
                      {"region": 2, "layer_begin": 8, "layer_end": 12,
                       "chunks": 2, "set_tensor_count": 6,
                       "set_tensor_bytes": 8192, "graph_compute_count": 1,
                       "graph_recompute_count": 1, "graph_snapshot_count": 0,
                       "graph_multi_count": 1, "graph_multi_nodes": 130,
                       "graph_single_add_count": 1, "graph_single_other_count": 0,
                       "get_tensor_count": 2, "get_tensor_bytes": 4096},
                      {"region": 5, "layer_begin": 20, "layer_end": 24,
                       "chunks": 1, "set_tensor_count": 3,
                       "set_tensor_bytes": 4096, "graph_compute_count": 1,
                       "graph_recompute_count": 0, "graph_snapshot_count": 0,
                       "graph_multi_count": 1, "graph_multi_nodes": 85,
                       "graph_single_add_count": 0, "graph_single_other_count": 0,
                       "get_tensor_count": 1, "get_tensor_bytes": 2048}],
                  "phone_result_get_count": 12,
                  "phone_result_get_mib": 8.0,
                  "phone_result_request_ms": 4.0,
                  "phone_result_response_wait_ms": 1200.0,
                  "phone_result_payload_recv_ms": 130.0,
                  "phone_result_rpc_total_ms": 1334.0,
                  "phone_snapshot_return_count": 2,
                  "phone_snapshot_ready_wait_ms": 9.0,
                  "phone_snapshot_payload_recv_ms": 3.0,
                 "cpu_full_meta_queue": False,
                 "phone_graph_cache": False, "phone_graph_slots": 0,
                 "phone_graph_slot_revisits": 0, "phone_graph_reuses": 3,
                  "phone_graph_reuse_count": 3,
                  "phone_graph_reuse_check_ms": 1.5,
                  "phone_graph_reset_ms": 3.5,
                  "phone_graph_build_ms": 90.0,
                  "phone_graph_alloc_ms": 180.0,
                  "phone_input_set_ms": 40.0,
                  "phone_split_prepare_ms": 850.0,
                 "phone_graph_affinity": True,
                 "phone_graph_affinity_region": 2,
                 "phone_graph_affinity_tokens": 96,
                 "phone_target_graph_tasks": 4,
                 "phone_target_consecutive": 2,
                 "phone_target_graph_reuses": 2,
                 "phone_target_consecutive_reuses": 2,
                 "meta_graph_ms": 100.0, "meta_graph_rebuild_ms": 31.0,
                 "meta_graph_execute_ms": 65.0, "meta_graph_other_ms": 4.0,
                 "meta_graph_calls": 2, "meta_graph_rebuilds": 1,
                 "layout": "G6,C2,P4,G6,C2,P4,G6,C2,P4,G6,C2,P4",
                 "xc": 32, "xg": 64, "xp": 128},
                {"type": "perf", "prefill_tokens": 127, "decode_tokens": 8,
                 "prefill_ms": 240.0, "decode_ms": 400.0, "load_ms": 10000,
                 "prefill_tps": 529.1667, "decode_tps": 20.0},
            ]
            (folder / "metrics.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8")
            (folder / "engine.stderr.log").write_text("", encoding="utf-8")
            (folder / "output.stdout.txt").write_text("", encoding="utf-8")
            args = SimpleNamespace(
                model=model, planner="apply-wide", streaming="on",
                batch=512, ubatch=512, threads=8, seed=1234, n_predict=32)
            result = SWEEP.result_row(
                args, folder, prompt, 128, 1024, 0, 20000, "deadbeef")
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["actual_prefill_tokens"], 127)
            self.assertEqual(result["ring_ubatches"], 1)
            self.assertEqual(result["ring_wall_ms"], 100)
            self.assertEqual(result["cpu_compute_range_ms"], 8)
            self.assertEqual(result["meta_pc_to_phone_copy_ms"], 7)
            self.assertEqual(result["phone_result_get_count"], 12)
            self.assertTrue(result["phone_rpc_available"])
            self.assertEqual(result["phone_rpc_set_tensor_count"], 9)
            self.assertEqual(result["phone_rpc_set_tensor_bytes"], 12288)
            self.assertEqual(result["phone_rpc_graph_compute_count"], 2)
            self.assertEqual(result["phone_rpc_graph_recompute_count"], 1)
            self.assertEqual(result["phone_rpc_graph_snapshot_count"], 0)
            self.assertEqual(result["phone_rpc_graph_multi_count"], 2)
            self.assertEqual(result["phone_rpc_graph_multi_nodes"], 215)
            self.assertEqual(result["phone_rpc_graph_single_add_count"], 1)
            self.assertEqual(result["phone_rpc_graph_single_other_count"], 0)
            self.assertEqual(result["phone_rpc_get_tensor_count"], 3)
            self.assertEqual(result["phone_rpc_get_tensor_bytes"], 6144)
            region_records = json.loads(result["phone_rpc_regions_json"])
            self.assertEqual(len(region_records), 1)
            self.assertEqual(region_records[0]["ubatch"], 0)
            self.assertEqual(len(region_records[0]["regions"]), 2)
            self.assertEqual(region_records[0]["regions"][0]["layer_begin"], 8)
            self.assertEqual(result["phone_result_get_mib"], 8)
            self.assertEqual(result["phone_result_response_wait_ms"], 1200)
            self.assertEqual(result["phone_result_payload_recv_ms"], 130)
            self.assertEqual(result["phone_snapshot_ready_wait_ms"], 9)
            self.assertEqual(result["phone_snapshot_payload_recv_ms"], 3)
            self.assertFalse(result["cpu_full_meta_queue"])
            self.assertFalse(result["phone_graph_cache"])
            self.assertEqual(result["phone_graph_slots"], 0)
            self.assertEqual(result["phone_graph_slot_revisits"], 0)
            self.assertEqual(result["phone_graph_reuses"], 3)
            self.assertEqual(result["phone_graph_reuse_count"], 3)
            self.assertEqual(result["phone_graph_reuse_check_ms"], 1.5)
            self.assertEqual(result["phone_graph_reset_ms"], 3.5)
            self.assertEqual(result["phone_graph_build_ms"], 90)
            self.assertEqual(result["phone_graph_alloc_ms"], 180)
            self.assertEqual(result["phone_input_set_ms"], 40)
            self.assertEqual(result["phone_split_prepare_ms"], 850)
            self.assertTrue(result["phone_graph_affinity"])
            self.assertEqual(result["phone_graph_affinity_region"], 2)
            self.assertEqual(result["phone_graph_affinity_tokens"], 96)
            self.assertEqual(result["phone_target_graph_tasks"], 4)
            self.assertEqual(result["phone_target_consecutive"], 2)
            self.assertEqual(result["phone_target_graph_reuses"], 2)
            self.assertEqual(result["phone_target_consecutive_reuses"], 2)
            self.assertEqual(result["meta_graph_rebuild_ms"], 31)
            self.assertEqual(result["meta_graph_execute_ms"], 65)
            self.assertEqual(result["meta_graph_other_ms"], 4)
            self.assertEqual(result["meta_graph_calls"], 2)
            self.assertEqual(result["meta_graph_rebuilds"], 1)
            self.assertEqual(result["gpu_layers"], 24)
            self.assertEqual(result["cpu_layers"], 8)
            self.assertEqual(result["phone_layers"], 16)
            self.assertEqual(result["prefix_e_json"],
                             "[0.3, 0.25, 0.22, 0.2]")
            self.assertEqual(result["selected_dag_sim_ms"], 195.125)
            self.assertEqual(result["final_dag_sim_ms"], 195.125)
            self.assertEqual(result["final_dag_source"], "selected")
            self.assertEqual(result["dag_plan_tokens"], 128)
            # Token count is a planning hint, not necessarily the true
            # tokenizer count used by the official performance report.
            self.assertEqual(result["actual_prefill_tokens"], 127)

    def test_baseline_fallback_records_executed_dag_instead_of_best_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "prompt_64"
            folder.mkdir()
            prompt = root / "prompt_64.txt"
            prompt.write_text("test", encoding="utf-8")
            model = root / "model.gguf"
            model.write_bytes(b"stub")
            records = [
                {"type": "planner", "applied": False, "selected_layout":
                 "G6,C2,P4,G6,C2,P4,G6,C2,P4,G6,C2,P4",
                 "baseline_layout": "G4,C4,P4,G4,C4,P4,G4,C4,P4,G4,C4,P4",
                 "predicted_ms": 100.0, "baseline_predicted_ms": 120.0,
                 "plan_tokens": 64},
                {"type": "measurement_start", "schema": 1},
                {"type": "region_ring", "tokens": 64, "complete": True,
                 "status": 0, "wall_ms": 121.0,
                 "layout": "G4,C4,P4,G4,C4,P4,G4,C4,P4,G4,C4,P4"},
                {"type": "perf", "prefill_tokens": 64,
                 "prefill_ms": 121.0, "decode_tokens": 8,
                 "decode_ms": 100.0},
            ]
            (folder / "metrics.jsonl").write_text(
                "\n".join(json.dumps(r) for r in records) + "\n",
                encoding="utf-8")
            args = SimpleNamespace(
                model=model, planner="apply-wide", streaming="on",
                batch=512, ubatch=512, threads=8, seed=1234, n_predict=32)
            row = SWEEP.result_row(
                args, folder, prompt, 64, 1024, 0, 1000, "deadbeef")
            self.assertEqual(row["status"], "PASS")
            self.assertEqual(row["selected_dag_sim_ms"], 100.0)
            self.assertEqual(row["final_dag_sim_ms"], 120.0)
            self.assertEqual(row["final_dag_source"], "baseline")
            self.assertEqual(row["dag_plan_tokens"], 64)
            self.assertEqual(row["runtime_layout"],
                             "G4,C4,P4,G4,C4,P4,G4,C4,P4,G4,C4,P4")

    def test_activation_estimates_are_full_region_only(self):
        # 64 tokens x 2048 FP32 elements = 0.5 MiB per edge.
        layout = "G6,C2,P4,G6,C2,P4"
        sizes = SWEEP.estimated_activation_payloads([
            {"tokens": 64, "hidden": 2048, "layout": layout},
        ])
        self.assertIsNotNone(sizes)
        self.assertEqual(sizes["gpu_to_cpu"], 1.0)
        self.assertEqual(sizes["pc_to_phone"], 1.0)
        self.assertEqual(sizes["phone_to_pc"], 0.5)
        self.assertEqual(sizes["cpu_to_gpu"], 0.5)
        self.assertIsNone(SWEEP.estimated_activation_payloads([
            {"tokens": 64, "hidden": 2048,
             "layout": "G4,C2,C2P1,P5,P2C1,G3"},
        ]))

    def test_perf_without_region_ring_is_not_a_valid_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "prompt_64"
            folder.mkdir()
            prompt = root / "prompt_64.txt"
            prompt.write_text("test", encoding="utf-8")
            model = root / "model.gguf"
            model.write_bytes(b"stub")
            (folder / "metrics.jsonl").write_text(
                json.dumps({"type": "perf", "prefill_tokens": 64,
                            "prefill_ms": 125, "decode_tokens": 8,
                            "decode_ms": 100}) + "\n", encoding="utf-8")
            args = SimpleNamespace(
                model=model, planner="apply-wide", streaming="on",
                batch=512, ubatch=512, threads=8, seed=1234, n_predict=32)
            row = SWEEP.result_row(
                args, folder, prompt, 64, 1024, 0, 1000, "deadbeef")
            self.assertEqual(row["status"], "NO_RING")
            self.assertIsNone(row["phone_rpc_available"])
            self.assertIsNone(row["phone_rpc_regions_json"])
            self.assertIsNone(row["phone_rpc_set_tensor_count"])
            self.assertIsNone(row["cpu_full_meta_queue"])
            self.assertIsNone(row["phone_graph_cache"])
            self.assertIsNone(row["phone_graph_slots"])
            self.assertIsNone(row["phone_graph_affinity"])
            self.assertIsNone(row["phone_target_graph_reuses"])
            self.assertIsNone(row["meta_graph_rebuild_ms"])
            self.assertIsNone(row["selected_dag_sim_ms"])
            self.assertIsNone(row["final_dag_sim_ms"])
            self.assertIsNone(row["final_dag_source"])

    def test_filename_and_layer_parser(self):
        self.assertIsNotNone(SWEEP.NAME.fullmatch("prompt_358.txt"))
        self.assertIsNone(SWEEP.NAME.fullmatch("something-358.txt"))
        self.assertEqual(SWEEP.count_layers("G4,C2,C2P1,P7,P2C1")["C2P"], 1)
        self.assertEqual(SWEEP.count_layers("G4,C2,C2P1,P7,P2C1")["P2C"], 1)


if __name__ == "__main__":
    unittest.main()
