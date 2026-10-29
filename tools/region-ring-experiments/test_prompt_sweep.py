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
                 "prefix_e": [0.3, 0.25, 0.22, 0.2],
                 "prefix_service_mspt": [[1, 2, 3], [2, 3, 4]]},
                {"type": "region_ring", "tokens": 512, "complete": True,
                 "status": 0, "wall_ms": 900, "layout": "G4,C4,P4"},
                {"type": "measurement_start", "schema": 1},
                {"type": "region_ring", "tokens": 126, "complete": True,
                 "status": 0, "wall_ms": 100.0, "cpu_compute_range_ms": 8.0,
                 "meta_available": True, "meta_pc_to_phone_copy_ms": 7.0,
                 "meta_phone_to_pc_copy_ms": 3.0,
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
            self.assertEqual(result["gpu_layers"], 24)
            self.assertEqual(result["cpu_layers"], 8)
            self.assertEqual(result["phone_layers"], 16)
            self.assertEqual(result["prefix_e_json"],
                             "[0.3, 0.25, 0.22, 0.2]")

    def test_filename_and_layer_parser(self):
        self.assertIsNotNone(SWEEP.NAME.fullmatch("prompt_358.txt"))
        self.assertIsNone(SWEEP.NAME.fullmatch("something-358.txt"))
        self.assertEqual(SWEEP.count_layers("G4,C2,C2P1,P7,P2C1")["C2P"], 1)
        self.assertEqual(SWEEP.count_layers("G4,C2,C2P1,P7,P2C1")["P2C"], 1)


if __name__ == "__main__":
    unittest.main()
