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
from phone_owned_models import ModelLayout


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
        for tokens in (66, 175, 700, 1855, 2048):
            self.assertEqual(suite.choose_context(tokens), 2048)
        with self.assertRaises(ValueError):
            suite.choose_context(2049)

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

    def test_default_full50_ignores_2048_group_even_when_files_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for group in ("0", "128", "256", "512", "1024", "2048",
                          "2048-4000"):
                for index in range(1, 11):
                    (base / f"prompt_{group}_{index:02d}.txt").write_text(
                        "sample", encoding="utf-8")
            included = suite.discover(
                base, "prompt_*_*.txt",
                groups=set(suite.DEFAULT_GROUPS.split(",")))
            self.assertEqual(len(included), 50)
            self.assertEqual({suite.group_of(x) for x in included},
                             {"0", "128", "256", "512", "1024"})
            self.assertTrue(all(suite.group_of(x) not in
                                {"2048", "2048-4000"} for x in included))
            self.assertEqual(included[0].name, "prompt_0_01.txt")
            self.assertEqual(included[-1].name, "prompt_1024_10.txt")

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


class TestOriginalJupiterDP(unittest.TestCase):
    def test_irregular_measured_dp_plan_is_used_without_repartition(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            case = {
                "layout": 32, "name": "prompt_256_01.txt",
                "path": str(folder / "prompt_256_01.txt"),
                "tokens": 358, "context": 2048,
                "group": "256", "chunks": [], "kind": "normal",
            }
            target = (folder / "calibration" / "l32" / "plans"
                      / "prompt_256_01")
            (target / "stage3").mkdir(parents=True)
            (target / "final.plan").write_text(
                "model.pc_layers=32\nmodel.cpu_layers=16\n"
                "sequence.358=60,60,110,128\n",
                encoding="utf-8",
            )
            (target / "stage3" / "sequence-summary.json").write_text(
                json.dumps({
                    "status": "FULLY_MEASURED_AND_VERIFIED",
                    "pc_layers": 32, "cpu_layers": 16,
                    "prompt_tokens": 358, "min_chunk": 32,
                    "chunks": [60, 60, 110, 128],
                }),
                encoding="utf-8",
            )
            args = argparse.Namespace(cpu_layers=16, max_chunk=128)
            original = suite.exact_jupiter_dp_plan(
                args, case, folder, folder / "stage2.json")
            self.assertEqual(original["chunks"], [60, 60, 110, 128])
            self.assertEqual(original["context"], 2048)
            self.assertEqual(original["plan_source"],
                             "JUPITER_STAGE2_B_QXY_EXACT_DP")
            self.assertEqual(case["chunks"], [])
            args.cpu_layers = 0
            with self.assertRaises(ValueError):
                suite.exact_jupiter_dp_plan(
                    args, case, folder, folder / "stage2.json")

    def test_stage2_boundary_is_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "min-chunk.json"
            file.write_text(json.dumps({
                "pc_layers": 32, "cpu_layers": 16,
                "max_chunk": 128,
                "candidates": [{"samples": 2, "chunk_tokens": 32}],
            }), encoding="utf-8")
            args = argparse.Namespace(
                min_chunk_json=str(file), cpu_layers=16, max_chunk=128)
            self.assertEqual(
                suite.stage2_for_layout(args, [], 32, Path(temp)), file)
            args.cpu_layers = 0
            with self.assertRaises(ValueError):
                suite.stage2_for_layout(args, [], 32, Path(temp))


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
                pattern="prompt_*_*.txt",
                groups="0,128,256,512,1024", out_dir=str(out),
                model_size="auto", layouts="32", cpu_layers=16,
                cuda_layers=None, phone_layers=None,
                sessions=4, chunk_size=88,
                sequence_policy="fixed-control", min_chunk_json=None,
                stage2_repeats=2, q_repeats=2,
                refine_repeats=2, max_rounds=6, max_chunk=128,
                repeats=1,
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
                    with logit_path.open("wb") as file:
                        array.array("f", [1.0, 2.0, 3.0]).tofile(file)
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
                 patch.object(suite, "inspect_model", return_value=ModelLayout(
                     "14b", 40, 5120, 40, 8, 128)), \
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
                 patch.object(suite, "inspect_model", return_value=ModelLayout(
                     "14b", 40, 5120, 40, 8, 128)), \
                 patch.object(suite, "count_model_tokens",
                              side_effect=AssertionError("retokenized")), \
                 patch.object(suite, "perform_case",
                              side_effect=AssertionError("reexecuted")):
                self.assertEqual(suite.run(), 0)


class Test32BFullSuite(unittest.TestCase):
    def test_32b_fixed_2048_one_prompt_dry_run(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            prompts = base / "prompts"
            prompts.mkdir()
            (prompts / "prompt_256_01.txt").write_text(
                "This is a sample.", encoding="utf-8")
            model = base / "Qwen3-32B-Q4_K_M.gguf"
            exe = base / "llama-cli.exe"
            tok = base / "llama-tokenize.exe"
            for p in (model, exe, tok):
                p.write_bytes(b"fake")
            args = argparse.Namespace(
                exe=str(exe), tokenizer=str(tok), model=str(model),
                model_size="32b", rpc="127.0.0.1:10140",
                prompts_dir=str(prompts), pattern="prompt_256_01.txt",
                groups="0,128,256,512,1024",
                out_dir=str(base / "output"), layouts=None,
                cpu_layers=None, cuda_layers=12, phone_layers=8,
                sessions=4, chunk_size=128,
                sequence_policy="jupiter-dp", min_chunk_json=None,
                stage2_repeats=2, q_repeats=2, refine_repeats=2,
                max_rounds=6, max_chunk=128, repeats=1,
                expected_count=1, mode_order="alternate", no_warmup=False,
                revisit_first=False, resume=False,
                continue_on_error=False, timeout_seconds=60, dry_run=True)
            spec = ModelLayout("32b", 64, 5120, 64, 8, 128)
            with patch.object(suite, "parse_args", return_value=args), \
                 patch.object(suite, "inspect_model", return_value=spec), \
                 patch.object(suite, "count_model_tokens", return_value=358):
                self.assertEqual(suite.run(), 0)
            manifest = json.loads(
                (base / "output" / "suite-manifest.json").read_text(
                    encoding="utf-8"))["config"]
            self.assertEqual(manifest["context"], 2048)
            self.assertEqual(manifest["model_size"], "32b")
            self.assertEqual(manifest["model_layers"], 64)
            self.assertEqual(manifest["cpu_layers"], 44)
            self.assertEqual(manifest["layouts"], [56])
            self.assertEqual(manifest["phone_layer_counts"], [8])
            self.assertEqual(manifest["prompts"][0]["tokens"], 358)
            self.assertEqual(manifest["prompts"][0]["context"], 2048)
            # The full-suite dry-run must also admit >16 Phone layers
            # without changing Context or the 16-Chunk DP capacity.
            args.out_dir = str(base / "output-phone24")
            args.phone_layers = 24
            with patch.object(suite, "parse_args", return_value=args), \
                 patch.object(suite, "inspect_model", return_value=spec), \
                 patch.object(suite, "count_model_tokens", return_value=358):
                self.assertEqual(suite.run(), 0)
            extended = json.loads(
                (base / "output-phone24" / "suite-manifest.json").read_text(
                    encoding="utf-8"))["config"]
            self.assertEqual(extended["cpu_layers"], 28)
            self.assertEqual(extended["layouts"], [40])
            self.assertEqual(extended["phone_layer_counts"], [24])
            self.assertEqual(extended["context"], 2048)


if __name__ == "__main__":
    unittest.main()
