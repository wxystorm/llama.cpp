#!/usr/bin/env python3
"""Host-only runner tests: no model, Phone RPC, or CUDA required."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_32b_profile5_plan as runner


class RollingPlanTests(unittest.TestCase):
    def test_expected_prompt_order_and_resume_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for group in runner.GROUPS:
                for i in range(1, 11):
                    (root / f"prompt_{group}_{i:02}.txt").write_text("sample", encoding="utf-8")
            items = runner.expected_prompts(root)
            self.assertEqual(len(items), 40)
            self.assertEqual(items[0][1].name, "prompt_128_01.txt")
            self.assertEqual(items[-1][1].name, "prompt_1024_10.txt")
            out = root / "out"
            result = out / "runs" / "prompt_128_01" / "attempt-0001" / "result.json"
            runner.atomic_json(result, {"status": "PASS", "prefill_tps": 99,
                                        "profile_version": 0})
            runner.save_summary(out, items, {})
            rows = list(__import__("csv").DictReader(
                (out / "summary.csv").open(encoding="utf-8-sig", newline="")))
            self.assertEqual(len(rows), 40)
            self.assertEqual(rows[0]["status"], "PASS")
            self.assertEqual(rows[1]["status"], "")
            self.assertEqual(rows[0]["prefill_tps"], "99")

    def test_profile_refresh_after_five_completed_not_attempts(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            args = SimpleNamespace(model=out / "model.gguf", rpc="phone:50052",
                                   threads=8, profile_dir=out / "profile",
                                   native_profile_exe=out / "exe",
                                   refresh_chunks="96,128", profile_every=5)
            seen = []
            def fake_profile(*a, **kwargs):
                seen.append((kwargs["force"], kwargs["refresh_chunks"]))
                folder = out / "profile"
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "stage-profile-v2.csv").write_text("backend,layers\\nN,1\\n")
                (folder / "profile.json").write_text('{"schema":2}')
                return folder / "stage-profile-v2.csv"
            with patch.object(runner, "prepare_native_profile", side_effect=fake_profile):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.ensure_profile(args, out, out, 0), 0)
                    for n in range(1, 5):
                        self.assertEqual(runner.ensure_profile(args, out, out, n), 0)
                    self.assertEqual(runner.ensure_profile(args, out, out, 5), 1)
                    self.assertEqual(runner.ensure_profile(args, out, out, 5), 1)
                    self.assertEqual(runner.ensure_profile(args, out, out, 6), 1)
                    self.assertEqual(runner.ensure_profile(args, out, out, 10), 2)
            self.assertEqual(seen, [
                (False, ()), (True, (96, 128)), (True, (96, 128))
            ])

    def test_terminal_prints_real_selected_plan_and_npu_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompt = root / "prompt_512_09.txt"
            prompt.write_text("example", encoding="utf-8")
            args = SimpleNamespace(exe=root / "llama-completion.exe",
                                   native_profile_exe=root / "stage.exe",
                                   model=root / "qwen.gguf", rpc="phone:50052",
                                   profile_dir=root / "profile", batch=2048,
                                   ubatch=2048, threads=8, n_predict=1,
                                   context_padding=256, no_mmap=True)
            def fake_run(command, cwd, stdout, stderr, check):
                self.assertIn("--planner", command)
                self.assertIn("apply-wide", command)
                self.assertIn("--no-mmap", command)
                output = Path(command[command.index("--out") + 1])
                dest = output / "20261111-000000-123" / "prompt_984"
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "result.json").write_text(json.dumps({
                    "status": "PASS", "prefill_tps": 90.25,
                    "prefill_ms": 10903.0, "plan_layout": "G8,P4,G8,P4",
                    "runtime_layout": "G8,P4,G8,P4", "xc": 64, "xg": 96,
                    "xp": 128, "plan_applied": True,
                    "predicted_ms": 10880.0
                }), encoding="utf-8")
                (dest / "engine.stderr.log").write_text(
                    "[REGION_RING_NPU_MEMORY] policy=STREAMING_2BANK\n",
                    encoding="utf-8")
                return SimpleNamespace(returncode=0)
            capture = io.StringIO()
            with patch.object(runner.subprocess, "run", side_effect=fake_run):
                with contextlib.redirect_stdout(capture):
                    result = runner.run_one(args, root, root / "runs-data",
                                            512, prompt, 984, 1, 1)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["npu_policy"], "STREAMING_2BANK")
            text = capture.getvalue()
            self.assertIn("prompt=prompt_512_09.txt", text)
            self.assertIn("XC=64 XG=96 XN=128", text)
            self.assertIn("npu_weights=STREAMING_2BANK", text)


if __name__ == "__main__":
    unittest.main()
