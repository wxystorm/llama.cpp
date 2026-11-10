"""Pure resumable-sweep tests: no GGUF, GPU, RPC server, or tokenizer needed."""
import json
from pathlib import Path
import tempfile
import unittest

from run_32b_g8p6_resume import (
    CHUNK, LAYOUT, context_size, folders_for, prior_runs, prompt_list,
)


class Resumable32BSweepTests(unittest.TestCase):
    def test_layout_and_context(self):
        self.assertEqual(CHUNK, 128)
        self.assertEqual(sum(int(x[1:]) for x in LAYOUT.split(",")), 64)
        self.assertEqual(context_size(587), 1024)
        self.assertEqual(context_size(1372), 1792)
        self.assertEqual(context_size(1747), 2048)
        self.assertEqual(context_size(1855), 2304)

    def test_both_prompt_filename_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            for filename in (
                "prompt_1024_10.txt", "prompt_512_01.txt",
                "prompt_128.txt", "other.txt",
            ):
                (path / filename).write_text("sample", encoding="utf-8")
            prompts = prompt_list(path)
            self.assertEqual(
                [prompt.name for _, _, prompt in prompts],
                ["prompt_128.txt", "prompt_512_01.txt", "prompt_1024_10.txt"],
            )

    def test_resume_skips_pass_and_rerun_preserves_best(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prompt = out / "prompt_512_01.txt"
            prompt.write_text("sample", encoding="utf-8")
            self.assertEqual(prior_runs(out, prompt, "digest", 0)["status"], "PENDING")
            for number, status, score in (
                (1, "PASS", 124.236),
                (2, "PASS", 95.0),
                (3, "FAIL", None),
            ):
                folder = out / "attempts" / prompt.stem / f"attempt-{number:04d}"
                folder.mkdir(parents=True)
                (folder / "config.json").write_text(
                    json.dumps({"model_tokens": 587}), encoding="utf-8")
                (folder / "result.json").write_text(json.dumps({
                    "status": status, "prefill_tps": score,
                    "prompt_sha256": "digest",
                }), encoding="utf-8")
            previous = prior_runs(out, prompt, "digest", 100)
            self.assertEqual(previous["attempts"], 3)
            self.assertEqual(previous["status"], "PASS")
            self.assertEqual(previous["best"]["prefill_tps"], 124.236)
            self.assertEqual(previous["latest"]["status"], "FAIL")
            self.assertEqual(len(folders_for(out, prompt)), 3)
            self.assertEqual(prior_runs(out, prompt, "digest", 130)["status"], "LOW_TPS")
            self.assertEqual(prior_runs(out, prompt, "changed", 0)["status"], "FAIL")

    def test_runner_legacy_parser_argument_compatibility(self):
        # The newer runner must populate every field read by the shared
        # run_prompt_sweep.result_row() parser, especially n_predict.
        import argparse
        from run_32b_g8p6_resume import inference_env
        from run_32b_g8p6_resume import recover_existing_attempts
        import run_32b_g8p6_resume as runner
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("args.n_predict = args.predict", source)
        self.assertIn("recover_existing_attempts(args, all_prompts, tokens, commit)",
                      source)

    def test_parser_crash_recovery_from_saved_metrics(self):
        import argparse
        from unittest import mock
        from run_32b_g8p6_resume import recover_existing_attempts
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prompts = out / "prompts"
            prompts.mkdir()
            prompt = prompts / "prompt_128_01.txt"
            prompt.write_text("hi", encoding="utf-8")
            folder = out / "attempts" / prompt.stem / "attempt-0001"
            folder.mkdir(parents=True)
            (folder / "config.json").write_text(json.dumps({
                "prompt_sha256": "digest", "context": 1024, "commit": "old"
            }), encoding="utf-8")
            (folder / "metrics.jsonl").write_text(
                '{"type":"region_ring","complete":true,"status":0}\n'
                '{"type":"perf","prefill_ms":100,"prefill_tokens":10}\n',
                encoding="utf-8")
            args = argparse.Namespace(
                out=out, predict=1, padding=256, min_context=1024)
            token_map = {prompt.name: {"sha256": "digest", "tokens": 10}}
            fake_result = {
                "status": "PASS", "prefill_tps": 100.0,
                "prefill_ms": 100.0, "prompt_sha256": "digest",
            }
            with mock.patch(
                "run_32b_g8p6_resume.old.result_row",
                return_value=fake_result,
            ), mock.patch(
                "run_32b_g8p6_resume.old.write_phone_input_push_timeline"
            ), mock.patch(
                "run_32b_g8p6_resume.old.write_phone_copy_timeline"
            ):
                recover_existing_attempts(
                    args, [(128, 1, prompt)], token_map, "new"
                )
            recovered = json.loads(
                (folder / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(recovered["status"], "PASS")
            self.assertTrue(recovered["recovered"])
            self.assertFalse(recovered["process_exit_code_verified"])
            self.assertEqual(prior_runs(out, prompt, "digest", 0)["status"],
                             "PASS")

    def test_reuse_saved_prompt_dir_when_omitted(self):
        from run_32b_g8p6_resume import resolve_prompt_dir
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            initial = out / "sharegpt40-prompts"
            initial.mkdir()
            (out / "sweep-config.json").write_text(
                json.dumps({"prompt_dir": str(initial)}), encoding="utf-8")
            self.assertEqual(resolve_prompt_dir(None, out), initial.resolve())
            explicit = out / "another-prompts"
            explicit.mkdir()
            self.assertEqual(resolve_prompt_dir(explicit, out),
                             explicit.resolve())

    def test_explicit_mismatched_prompt_dir_still_rejected(self):
        from argparse import Namespace
        from run_32b_g8p6_resume import ensure_config, stable_config
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            model = out / "test.gguf"
            model.write_bytes(b"fake model")
            base = {
                "model": model, "exe": out / "llama-completion.exe",
                "tokenizer": out / "llama-tokenize.exe",
                "prompt_dir": out / "prompts1", "rpc": "127.0.0.1:50052",
                "out": out, "batch": 2048, "ubatch": 2048,
                "predict": 1, "threads": 8, "seed": 1234,
                "min_context": 1024, "padding": 256, "log_verbosity": 4,
                "no_ram_preload": False, "no_trace": False,
                "mover_priority": False,
            }
            original = stable_config(Namespace(**base), "old-commit")
            (out / "sweep-config.json").write_text(
                json.dumps(original), encoding="utf-8")
            # Script-only updates are allowed.
            ensure_config(Namespace(**base), "new-commit")
            base["prompt_dir"] = out / "prompts2"
            with self.assertRaisesRegex(RuntimeError, "prompt_dir"):
                ensure_config(Namespace(**base), "new-commit")

    def test_default_all_pending_and_fail_fast(self):
        """Exercise main() with 3 prompts; no model, CUDA or RPC needed."""
        import sys
        from unittest import mock
        import run_32b_g8p6_resume as runner

        for result_status, expected_return, expected_attempts, extra in (
            ("PASS", 0, 3, []),
            ("FAIL", 1, 1, []),
            ("PASS", 0, 1, ["--limit", "1"]),
        ):
            with self.subTest(result_status=result_status, extra=extra):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    model = root / "model.gguf"
                    exe = root / "llama-completion.exe"
                    tokenizer = root / "llama-tokenize.exe"
                    for f in (model, exe, tokenizer):
                        f.write_bytes(b"test")
                    prompts = root / "prompts"
                    prompts.mkdir()
                    for number in range(1, 4):
                        (prompts / f"prompt_128_{number:02d}.txt").write_text(
                            "sample", encoding="utf-8")
                    out = root / "output"
                    argv = [
                        "run_32b_g8p6_resume.py",
                        "--model", str(model), "--exe", str(exe),
                        "--tokenizer", str(tokenizer),
                        "--prompt-dir", str(prompts), "--out", str(out),
                    ] + extra
                    def fake_tokens(_args, selected):
                        return {
                            p.name: {"sha256": "test-hash", "tokens": 167}
                            for _, _, p in selected
                        }

                    with mock.patch.object(sys, "argv", argv), mock.patch(
                        "run_32b_g8p6_resume.old.git_head", return_value="test"
                    ), mock.patch(
                        "run_32b_g8p6_resume.tokens_for_prompts",
                        side_effect=fake_tokens,
                    ), mock.patch(
                        "run_32b_g8p6_resume.run_prompt",
                        return_value={"status": result_status},
                    ) as process:
                        self.assertEqual(runner.main(), expected_return)
                        self.assertEqual(process.call_count, expected_attempts)

    def test_failure_report_captures_engine_error(self):
        from run_32b_g8p6_resume import failure_report
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "engine.stderr.log").write_text(
                "[REGION_RING_GPU_BANK] event=REJECT reason=TEST_FAILURE\\n"
                "test ending\\n", encoding="utf-8")
            output = failure_report(folder, {
                "prompt_file": "prompt_128_01.txt",
                "status": "FAIL", "exit_code": 1, "prefill_tps": None
            })
            self.assertIn("TEST_FAILURE", output)
            self.assertIn("exit_code=1", output)
            self.assertTrue((folder / "failure-summary.txt").exists())

    def test_interrupted_attempt_is_retryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prompt = out / "prompt_128_01.txt"
            folder = out / "attempts" / prompt.stem / "attempt-0001"
            folder.mkdir(parents=True)
            state = prior_runs(out, prompt, "digest", 0)
            self.assertEqual(state["status"], "FAIL")
            self.assertEqual(state["latest"]["status"], "INTERRUPTED")


if __name__ == "__main__":
    unittest.main()
