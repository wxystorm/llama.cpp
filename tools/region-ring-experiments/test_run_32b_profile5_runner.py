#!/usr/bin/env python3
"""Host-only regressions for CPU RPC transport and dynamic Plan validation."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_32b_all40 import build_process, parse_metrics


class HtpOnlyAutoPlanTests(unittest.TestCase):
    def test_cpu_transport_graphless_plan_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = SimpleNamespace(
                auto_plan=True, htp_only=True, profile_every=5,
                streaming="on", real_stage_profile_csv=None,
                force_small_batch_mover=False, phone_weight_preload=True,
                boost_bank_mover=False, env=[], exe=root / "completion.exe",
                model=root / "qwen.gguf", rpc="127.0.0.1:50052",
                batch=2048, ubatch=2048, n_predict=1, threads=8,
                seed=1234, log_verbosity=4, no_mmap=False,
                layout="G8,P6,G8,P6,G8,P6,G8,P6,G8",
                xc=128, xg=128, xp=128, chunk=128,
            )
            cmd, env = build_process(
                args, root / "prompt_512_01.txt", 1024, 512,
                root / "metrics.jsonl")
            self.assertIn("--no-warmup", cmd)
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_HTP_ONLY"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_PLAN_CONTROLLED"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_GRAPHLESS"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_AUTONOMOUS"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_INPUT_STREAM"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_REAL_STAGE_PROFILE"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_TASK_TRACE"], "1")
            self.assertNotIn("LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT", env)

    def test_preload_count_uses_selected_not_control_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chosen = "G8,P3,G8,P3,G42"
            with (root / "metrics.jsonl").open("w", encoding="utf-8") as f:
                for obj in (
                    {"type": "planner", "mode": "apply-wide", "applied": True,
                     "selected_layout": chosen, "xc": 64, "xg": 96, "xp": 128},
                    {"type": "perf", "prefill_ms": 1000.0,
                     "prefill_tokens": 512, "prefill_tps": 512.0},
                    {"type": "region_ring", "complete": True, "status": 0,
                     "tokens": 512, "layout": chosen},
                ):
                    f.write(json.dumps(obj) + "\n")
            (root / "engine.stderr.log").write_text("\n".join((
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=START status=RUNNING",
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=REGION region=1 status=PASS",
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=REGION region=3 status=PASS",
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=DONE wall_ms=50 status=PASS",
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=JOIN status=PASS",
            )), encoding="utf-8")
            row = parse_metrics(
                root, 0, 1500.0, auto_plan=True, htp_only=True,
                phone_weight_preload=True,
                fixed_layout="G8,P6,G8,P6,G8,P6,G8,P6,G8")
            self.assertEqual(row["early_weights_status"], "PASS")
            self.assertEqual(row["early_weights_regions"], 2)
            self.assertEqual(row["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
