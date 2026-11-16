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

    def test_opt_in_gpu_region_budget_retains_full_memory_guard(self):
        args = runner.get_args(self.base + [
            "--only-prompt", "prompt_1024_10.txt",
            "--gpu-region-budget-fraction", "0.55",
        ])
        self.assertAlmostEqual(args.gpu_region_budget_fraction, 0.55)
        command, env = runner.build_process(
            args, self.prompts / "prompt_1024_10.txt",
            2048, 1855, self.out / "test-metrics.jsonl")
        self.assertIn(str(self.model), command)
        self.assertEqual(
            env["LLAMA_REGION_RING_GPU_REGION_BUDGET_FRACTION"], "0.55")

        default = runner.get_args(self.base + [
            "--only-prompt", "prompt_1024_10.txt"])
        _, default_env = runner.build_process(
            default, self.prompts / "prompt_1024_10.txt",
            2048, 1855, self.out / "default-metrics.jsonl")
        self.assertNotIn(
            "LLAMA_REGION_RING_GPU_REGION_BUDGET_FRACTION", default_env)

        with contextlib.redirect_stderr(io.StringIO()):
            for illegal in ("0.49", "0.61", "nan"):
                with self.assertRaises(SystemExit):
                    runner.get_args(self.base + [
                        "--only-prompt", "prompt_1024_10.txt",
                        "--gpu-region-budget-fraction", illegal])
            with self.assertRaises(SystemExit):
                runner.get_args(self.base + [
                    "--only-prompt", "prompt_1024_10.txt",
                    "--env",
                    "LLAMA_REGION_RING_GPU_REGION_BUDGET_FRACTION=0.80"])

    def test_fixed_layout_locked_short_chunks_omits_x8(self):
        # This is a separate fixed-placement experiment: --main50-short
        # intentionally forbids --locked-layout. Existing short calibration
        # may still contain X=8, but locked Plan selection starts at X=16.
        fixed_args = [
            item for item in self.base if item != "--main50-short"
        ] + [
            "--sharegpt50-short-dataset", "--locked-layout",
            "G7,N4,G7,N4,G8,N5,G7,N4,G7,N4,G7",
            "--locked-short-chunks", "--skip-plan-memory-checks",
        ]
        args = runner.get_args(fixed_args)
        self.assertEqual(
            args.locked_layout, "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7")
        self.assertTrue(args.skip_plan_memory_checks)
        self.assertTrue(args.locked_short_chunks)
        self.assertEqual(runner.SHORT_LOCKED_PLAN_CHUNKS[0], 16)
        self.assertIn(24, runner.SHORT_LOCKED_PLAN_CHUNKS)
        self.assertNotIn(8, runner.SHORT_LOCKED_PLAN_CHUNKS)
        cmd, env = runner.build_process(
            args, self.prompts / "prompt_0_01.txt",
            256, 100, self.out / "fixed-short-metrics.jsonl")
        self.assertIn(str(self.model), cmd)
        self.assertEqual(
            env["LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS"], "1")
        self.assertEqual(
            env["LLAMA_REGION_RING_LOCKED_SKIP_MEMORY_CHECKS"], "1")
        self.assertEqual(
            env["LLAMA_HYBRID_REGION_RING_LOCKED_LAYOUT"],
            "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7")

        # Profile flags are not an escape hatch to an unvalidated layout.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.get_args(
                    fixed_args + ["--main50-short"])
            with self.assertRaises(SystemExit):
                runner.get_args(
                    [a for a in fixed_args if a != "--locked-layout"
                     and a != "G7,N4,G7,N4,G8,N5,G7,N4,G7,N4,G7"])

    def test_x8_not_measured_and_legacy_cache_is_readable(self):
        self.assertEqual(runner.SHORT_PROFILE_CHUNKS[0], 16)
        self.assertNotIn(8, runner.SHORT_PROFILE_CHUNKS)
        self.assertEqual(runner.LEGACY_SHORT_PROFILE_CHUNKS[0], 8)
        self.assertNotIn(8, runner.SHORT_LOCKED_PLAN_CHUNKS)
        profile_dir = self.root / "independent-profile"
        profile_dir.mkdir()
        csv_path = profile_dir / "profile.csv"
        json_path = csv_path.with_suffix(".json")
        columns = [
            "device", "layers", "tokens", "kv_tokens", "setup_ms",
            "upload_ms", "prepare_ms", "compute_ms", "terminal", "samples",
        ]

        def write_profile(widths, cpu_x=16):
            rows = [
                {"device": "CPU", "layers": 2, "tokens": cpu_x,
                 "kv_tokens": cpu_x, "compute_ms": 1.0},
                {"device": "GPU", "layers": 2, "tokens": 24,
                 "kv_tokens": 24, "compute_ms": 2.0},
                {"device": "PHONE", "layers": 2, "tokens": 16,
                 "kv_tokens": 2048, "compute_ms": 3.0},
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            meta = {
                "schema": 1,
                "mode": "INDEPENDENT_BACKEND_PROFILE",
                "backends": {"N": "HTP_STRIPED"},
                "chunk_widths": list(widths),
                "stage_points": rows,
                "transfers": {
                    "pc_to_phone": [{"bytes": 1024, "ms": 1}],
                    "gpu_weight_h2d": [{"bytes": 1024, "ms": 1}],
                    "phone_to_pc": [{"bytes": 1024, "ms": 1}],
                },
            }
            json_path.write_text(
                json.dumps(meta), encoding="utf-8")

        # Both current 16+ and historical 8+ metadata can be reused.
        for widths in (
                runner.SHORT_PROFILE_CHUNKS,
                runner.LEGACY_SHORT_PROFILE_CHUNKS):
            write_profile(widths)
            self.assertEqual(
                runner.verify_independent_profile(
                    csv_path, short_chunks=True), 1)
        # Never accept X8 as the only measured CPU short point.
        write_profile(runner.LEGACY_SHORT_PROFILE_CHUNKS, cpu_x=8)
        with self.assertRaisesRegex(ValueError, "X=16/24"):
            runner.verify_independent_profile(
                csv_path, short_chunks=True)

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
