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
