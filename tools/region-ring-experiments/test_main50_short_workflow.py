"""Offline regression coverage for the ShareGPT50_short two-stage workflow.

No model inference, Android RPC, or CUDA is required; only temp fixtures and
the runner's --dry-run path are executed.
"""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_32b_all40 as runner


class Main50ShortWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = self.root / "sharegpt50_short"
        self.prompts.mkdir()
        self.out = self.root / "main50-new"
        self.exe = self.root / "fake-llama.exe"
        self.model = self.root / "fake-model.gguf"
        self.exe.write_text("stub", encoding="utf-8")
        self.model.write_text("stub", encoding="utf-8")
        with (self.prompts / "manifest.csv").open(
                "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "actual_tokens_tokenizer"])
            for group in runner.MAIN50_SHORT_GROUPS:
                for i in range(1, 11):
                    name = f"prompt_{group}_{i:02}.txt"
                    token_count = (64 + i * 5 if group == 0 else
                                   1855 if name == "prompt_1024_10.txt" else
                                   min(1800, group + i + 10))
                    (self.prompts / name).write_text(
                        "Testing short main 50", encoding="utf-8")
                    writer.writerow([name, token_count])
        self.base = [
            "--exe", str(self.exe), "--model", str(self.model),
            "--rpc", "127.0.0.1:50052",
            "--prompt-dir", str(self.prompts),
            "--repo-root", str(self.root),
            "--out", str(self.out),
            "--model-layers", "64",
            "--auto-plan", "--htp-only", "--streaming", "on",
            "--main50-short", "--max-context", "2048",
            "--context-padding", "0", "--dry-run",
        ]

    def run_main(self, *extra):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = runner.main(self.base + list(extra))
        return result, stream.getvalue()

    def test_first_run_must_be_1024_10_at_context_2048(self):
        with self.assertRaisesRegex(ValueError, "prompt_1024_10.txt"):
            self.run_main()
        result, log = self.run_main(
            "--only-prompt", "prompt_1024_10.txt")
        self.assertEqual(result, 0)
        self.assertIn(
            "prompt=prompt_1024_10.txt group=1024 plan_tokens=1855"
            " source=MANIFEST context=2048", log)
        self.assertEqual(log.count("[DRY_RUN]"), 1)
        self.assertIn("[MAIN50] groups=0,128,256,512,1024 total=50", log)

    def test_resume_with_lock_schedules_all_fifty(self):
        self.out.mkdir()
        runner.save_json(self.out / "placement-lock.json", {
            "schema": 1, "layout": "G8,P6,G8,P6,G8,P6,G8,P6,G8",
            "first_prompt": "prompt_1024_10.txt",
        })
        result, log = self.run_main()
        self.assertEqual(result, 0)
        self.assertEqual(log.count("[DRY_RUN]"), 50)
        self.assertIn("prompt_0_01.txt group=0", log)
        self.assertNotIn("prompt_2048_01.txt", log)

    def test_short_group_is_real_64_to_128_tokens(self):
        manifest = self.prompts / "manifest.csv"
        data = manifest.read_text(encoding="utf-8")
        manifest.write_text(data.replace("prompt_0_01.txt,69",
                                         "prompt_0_01.txt,40"),
                            encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "64..128"):
            self.run_main("--only-prompt", "prompt_1024_10.txt")

    def test_failed_first_plan_can_lock_and_restore(self):
        row = {
            "prompt": "prompt_1024_10.txt", "status": "FAIL",
            "plan_applied": True,
            "plan_layout": "G8,P6,G8,P6,G8,P6,G8,P6,G8",
            "plan_xc": 32, "plan_xg": 128, "plan_xp": 128,
        }
        lock = runner.placement_lock_from_pass(
            row, 64, allow_failed_plan=True)
        self.assertEqual(lock["source"], "FIRST_APPLIED_PLAN")
        self.assertEqual(lock["first_plan_status"], "FAIL")
        folder = self.out / "runs" / "prompt_1024_10" / "attempt-0001"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(
            json.dumps(row), encoding="utf-8")
        prompts = [(1024, self.prompts / "prompt_1024_10.txt")]
        restored = runner.restore_placement_lock(
            self.out, prompts, 64, lock_on_first_plan=True)
        self.assertEqual(restored["layout"], lock["layout"])
        self.assertTrue((self.out / "placement-lock.json").exists())

    def test_fixed_ablation_shares_the_dataset_without_auto_plan(self):
        args = runner.get_args([
            "--exe", str(self.exe), "--model", str(self.model),
            "--rpc", "127.0.0.1:50052",
            "--repo-root", str(self.root),
            "--prompt-dir", str(self.prompts),
            "--out", str(self.root / "ablation"),
            "--sharegpt50-short-dataset", "--max-context", "2048",
            "--chunk", "32", "--no-mmap",
        ])
        self.assertTrue(args.sharegpt50_short_dataset)
        self.assertFalse(args.auto_plan)
        self.assertEqual(args.chunk, 32)


if __name__ == "__main__":
    unittest.main()
