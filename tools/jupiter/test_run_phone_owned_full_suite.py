"""Model-free coverage for the one-command Jupiter Phone Prefill suite."""
from __future__ import annotations

import argparse
import array
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phone_owned_full_suite as suite


class TestPlanning(unittest.TestCase):
    def test_chunk_plans(self):
        self.assertEqual(suite.choose_chunks(2), [1, 1])
        self.assertEqual(suite.choose_chunks(66), [33, 33])
        self.assertEqual(suite.choose_chunks(175), [88, 87])
        self.assertEqual(suite.choose_chunks(176), [88, 88])
        self.assertEqual(suite.choose_chunks(4096), [256] * 16)
        for n in (2, 66, 88, 171, 175, 300, 512, 1024,
                  1855, 2048, 3072, 4096):
            plan = suite.choose_chunks(n)
            self.assertEqual(sum(plan), n)
            self.assertGreaterEqual(len(plan), 2)
            self.assertLessEqual(len(plan), 16)
            self.assertEqual(plan[:-1], [plan[0]] * (len(plan)-1))
        with self.assertRaises(ValueError):
            suite.choose_chunks(4097)

    def test_context(self):
        self.assertEqual(suite.choose_context(66), 512)
        self.assertEqual(suite.choose_context(175), 512)
        self.assertEqual(suite.choose_context(700), 768)
        self.assertEqual(suite.choose_context(1855), 2048)
        self.assertEqual(suite.choose_context(4096), 4096)
        with self.assertRaises(ValueError):
            suite.choose_context(4097)

    def test_prompt_sorting_and_group(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for name in ("prompt_128_10.txt", "prompt_128_01.txt",
                         "prompt_0_02.txt", "prompt_1024_01.txt",
                         "prompt_2048-4000_01.txt"):
                (base / name).write_text("hello", encoding="utf-8")
            names = [x.name for x in suite.discover(base, "prompt_*_*.txt")]
            self.assertEqual(names, [
                "prompt_0_02.txt", "prompt_128_01.txt",
                "prompt_128_10.txt", "prompt_1024_01.txt",
                "prompt_2048-4000_01.txt"])
            self.assertEqual(
                suite.group_of(base / "prompt_2048-4000_01.txt"),
                "2048-4000")

    def test_aba_selects_different_graph_shape(self):
        a = {"name": "prompt_128_01.txt", "path": "a", "tokens": 175,
             "context": 512, "chunks": [88, 87], "group": "128"}
        b = {"name": "prompt_128_02.txt", "path": "b", "tokens": 175,
             "context": 512, "chunks": [88, 87], "group": "128"}
        c = {"name": "prompt_128_03.txt", "path": "c", "tokens": 179,
             "context": 512, "chunks": [88, 88, 3], "group": "128"}
        sequence = suite.sequence_for_group([a, b, c], revisit=True)
        self.assertEqual([(x["path"], x["kind"]) for x in sequence], [
            ("a", "normal"), ("c", "normal"), ("a", "revisit"), ("b", "normal")])

    def test_csv_and_group_statistics(self):
        rows = []
        for n, speed in ((175, 100), (179, 200)):
            rows.append({
                "layout": 32, "group": "128", "name": f"prompt_128_{n}.txt",
                "kind": "normal", "status": "PASS",
                "tokens": n, "sync_ms": float(speed),
                "autonomous_ms": float(speed/2),
                "speedup": 2.0, "warm_comparable": True,
            })
        grouped = suite.group_summaries(rows)
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0]["passed_pairs"], 2)
        self.assertEqual(grouped[0]["warm_geomean_speedup"], 2.0)
        self.assertEqual(grouped[0]["sync_mean_ms"], 150.0)


class TestResume(unittest.TestCase):
    def test_full_dry_preflight_fake_runs_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            prompts = home / "prompts"
            prompts.mkdir()
            for filename, count in (
                    ("prompt_0_01.txt", 66),
                    ("prompt_128_01.txt", 175),
                    ("prompt_128_02.txt", 179)):
                (prompts / filename).write_text(str(count), encoding="utf-8")
            exe = home / "llama-cli.exe"
            tok = home / "llama-tokenize.exe"
            model = home / "Qwen3-14B.gguf"
            for path in (exe, tok, model):
                path.write_bytes(b"fake")
            out = home / "out"
            args = argparse.Namespace(
                exe=str(exe), tokenizer=str(tok), model=str(model),
                rpc="127.0.0.1:10140", prompts_dir=str(prompts),
                pattern="prompt_*_*.txt", out_dir=str(out),
                layouts="32", cpu_layers=16, sessions=4, chunk_size=88,
                context_floor=512, context_cap=4096, repeats=1,
                expected_count=3, mode_order="alternate",
                no_warmup=False, revisit_first=True,
                resume=False, continue_on_error=False,
                timeout_seconds=60, dry_run=False)

            def fake_run(_args, case, output, position):
                folder = suite.case_directory(output, case)
                path = suite.next_attempt(folder)
                runs = {}
                for mode, ms in (("sync", 100.0), ("autonomous", 90.0)):
                    work = path / mode
                    work.mkdir()
                    logit_path = work / "native-logits.f32"
                    array.array("f", [1.0, 2.0, 3.0]).tofile(
                        logit_path.open("wb"))
                    runs[mode] = {"ms": ms, "cache_hit": True,
                                  "logits": str(logit_path),
                                  "log": str(work / "native-full-prefill.log")}
                return dict(case, status="PASS",
                            runs=runs, sync_ms=100.0, autonomous_ms=90.0,
                            speedup=100/90, warm_comparable=True,
                            first_mode="sync",
                            parity={"top1_same": True, "max_abs": 0.0,
                                    "relative_rmse": 0.0, "cosine": 1.0},
                            case_dir=str(folder), attempt_dir=str(path))

            with patch.object(suite, "parse_args", return_value=args), \
                 patch.object(suite, "count_model_tokens",
                              side_effect=lambda _t, _m, p: int(p)), \
                 patch.object(suite, "perform_case",
                              side_effect=fake_run) as executed:
                self.assertEqual(suite.run(), 0)
                self.assertEqual(executed.call_count, 4)

            summary = json.loads(
                (out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["passed"], 4)
            self.assertEqual(summary["expected_cases"], 4)
            self.assertEqual(summary["warm_comparable_normal_pairs"], 3)

            args.resume = True
            with patch.object(suite, "parse_args", return_value=args), \
                 patch.object(suite, "count_model_tokens",
                              side_effect=AssertionError("retokenized")), \
                 patch.object(suite, "perform_case",
                              side_effect=AssertionError("reexecuted")):
                self.assertEqual(suite.run(), 0)


if __name__ == "__main__":
    unittest.main()
