#!/usr/bin/env python3
"""Offline tests for the persistent HTP-only llama-server smoke harness."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_htp_llama_server as probe


class ServerProbeTests(unittest.TestCase):
    def test_command_disables_cache_parallelism_and_warmup(self):
        cmd = probe.server_command(
            "llama-server.exe", "model.gguf", "127.0.0.1:50052",
            18080, threads=8)
        for opt in ("--no-cache-prompt", "--no-cont-batching",
                    "--no-warmup", "--no-context-shift"):
            self.assertIn(opt, cmd)
        self.assertEqual(cmd[cmd.index("--parallel") + 1], "1")
        self.assertEqual(cmd[cmd.index("--cache-ram") + 1], "0")
        self.assertEqual(cmd[cmd.index("--ctx-checkpoints") + 1], "0")
        self.assertEqual(cmd[cmd.index("-b") + 1], "2048")
        self.assertEqual(cmd[cmd.index("-ub") + 1], "2048")
        self.assertEqual(cmd[cmd.index("-c") + 1], "2048")

    def test_no_prompt_cache_or_multi_token_decode(self):
        payload = probe.completion_payload("hello world")
        self.assertEqual(payload["prompt"], "hello world")
        self.assertEqual(payload["n_predict"], 1)
        self.assertIs(payload["cache_prompt"], False)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["id_slot"], 0)

    def test_prompt_plan_includes_a_repeated_long_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("prompt_0_01.txt", "prompt_1024_10.txt"):
                (folder / name).write_text("sample", encoding="utf-8")
            paths = probe.prompt_files(folder, all50=False)
            self.assertEqual(
                [p.name for p in paths],
                ["prompt_1024_10.txt", "prompt_0_01.txt",
                 "prompt_1024_10.txt"])

    def test_all50_is_exactly_fifty_unique_prompts_anchor_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for group in probe.runner.MAIN50_SHORT_GROUPS:
                for index in range(1, 11):
                    (folder / f"prompt_{group}_{index:02d}.txt").write_text(
                        "input", encoding="utf-8")
            prompts = probe.prompt_files(folder, all50=True)
            names = [path.name for path in prompts]
            self.assertEqual(len(prompts), 50)
            self.assertEqual(len(set(names)), 50)
            self.assertEqual(names[0], "prompt_1024_10.txt")
            self.assertEqual(sum(name.startswith("prompt_0_") for name in names), 10)
            self.assertEqual(names.count("prompt_1024_10.txt"), 1)

    def test_partial_resume_does_not_duplicate_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "all50"
            prompts = [Path(tmp) / f"prompt_0_0{i}.txt" for i in (1, 2)]
            for path in prompts:
                path.write_text("hello", encoding="utf-8")
            config = {"fingerprint": "same", "configuration": {"schema": 1}}
            new = SimpleNamespace(out=out, resume=False)
            self.assertEqual(
                probe.prepare_sweep_output(new, config, prompts), [])
            records = [{
                "index": 1, "group": 0, "prompt": prompts[0].name,
                "prompt_n": 66, "cache_n": 0,
                "prompt_ms": 100.0, "prefill_tps": 660.0,
                "run_fingerprint": "same", "status": "PASS",
            }]
            (out / "requests.jsonl").write_text(
                json.dumps(records[0]) + "\n", encoding="utf-8")
            resume = SimpleNamespace(out=out, resume=True)
            existing = probe.prepare_sweep_output(resume, config, prompts)
            self.assertEqual(existing, records)
            with self.assertRaisesRegex(ValueError, "overwritten"):
                probe.prepare_sweep_output(new, config, prompts)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                probe.prepare_sweep_output(
                    resume, {"fingerprint": "different"}, prompts)
            self.assertTrue((out / "requests.csv").is_file())
            self.assertTrue((out / "summary.csv").is_file())
            self.assertEqual(
                probe.sweep_log_path(out, resume=True).name,
                "llama-server.log")
            (out / "llama-server.log").write_text("old", encoding="utf-8")
            self.assertEqual(
                probe.sweep_log_path(out, resume=True).name,
                "llama-server-resume-001.log")
            self.assertEqual(
                (out / "llama-server.log").read_text(encoding="utf-8"),
                "old")

    def test_resume_rejects_changed_order_or_duplicate_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "requests.jsonl"
            p = [Path(tmp) / "prompt_1024_10.txt",
                 Path(tmp) / "prompt_0_01.txt"]
            good = {
                "index": 1, "group": 1024,
                "prompt": p[0].name, "cache_n": 0,
                "status": "PASS", "run_fingerprint": "f",
            }
            file.write_text(
                json.dumps(good) + "\n", encoding="utf-8")
            self.assertEqual(
                len(probe.load_completed_requests(file, p, "f")), 1)
            file.write_text(json.dumps(good) + "\n" +
                            json.dumps(good) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row 2"):
                probe.load_completed_requests(file, p, "f")

    def test_aggregate_tps_is_tokens_divided_by_prefill_time(self):
        records = [
            {"group": 0, "prompt_n": 64, "prefill_ms": 1000.0,
             "prefill_tps": 64.0},
            {"group": 0, "prompt_n": 128, "prefill_ms": 1000.0,
             "prefill_tps": 128.0},
        ]
        rows = probe.report_rows(records, probe.runner.MAIN50_SHORT_GROUPS)
        zero = next(row for row in rows if row["group"] == 0)
        self.assertEqual(zero["completed"], 2)
        self.assertEqual(zero["weighted_tps"], 96.0)
        self.assertEqual(zero["median_tps"], 96.0)
        self.assertEqual(zero["status"], "INCOMPLETE")
        self.assertEqual(rows[-1]["total_tokens"], 192)

    def test_sweep_fingerprint_ignores_limit_but_checks_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, binary = root / "model.gguf", root / "server.exe"
            model.write_bytes(b"model")
            binary.write_bytes(b"server")
            prompts = [root / "prompt_1024_10.txt"]
            prompts[0].write_text("the long one", encoding="utf-8")
            (root / "manifest.csv").write_text(
                "file,actual_tokens_tokenizer\n"
                "prompt_1024_10.txt,1855\n", encoding="utf-8")
            profile = root / "profile"
            profile.mkdir()
            (profile / "profile.csv").write_text("test", encoding="utf-8")
            (profile / "profile.json").write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                server_exe=binary, model=model, rpc="127.0.0.1:50052",
                prompt_dir=root, profile_source=profile,
                inter_prompt_delay_seconds=5.0, limit=10)
            first = probe.sweep_config(
                args, ["llama-server", "--parallel", "1"],
                {"LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS": "1"},
                prompts, 1855)
            args.limit = 0
            second = probe.sweep_config(
                args, ["llama-server", "--parallel", "1"],
                {"LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS": "1"},
                prompts, 1855)
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            args.inter_prompt_delay_seconds = 0.0
            third = probe.sweep_config(
                args, ["llama-server", "--parallel", "1"],
                {"LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS": "1"},
                prompts, 1855)
            self.assertNotEqual(first["fingerprint"], third["fingerprint"])

    def test_cache_in_non_cached_request_is_rejected(self):
        import json
        good = {"timings": {
            "prompt_n": 100, "cache_n": 0,
            "prompt_ms": 20.0, "prompt_per_second": 5000.0,
            "predicted_n": 1,
        }}
        with patch.object(probe, "urlopen",
                          return_value=BytesIO(json.dumps(good).encode())):
            result = probe.post_completion(18080, "hello")
        self.assertEqual(result["cache_n"], 0)
        self.assertEqual(result["prompt_n"], 100)

        bad = {"timings": {**good["timings"], "cache_n": 8}}
        with patch.object(probe, "urlopen",
                          return_value=BytesIO(json.dumps(bad).encode())):
            with self.assertRaisesRegex(RuntimeError, "cached tokens"):
                probe.post_completion(18080, "hello")

    def test_earliest_error_comes_before_p2c_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            logfile = Path(tmp) / "llama-server.log"
            logfile.write_text(
                "[REGION_RING_WORKER] event=START\n"
                "[PHONE_INPUT_PUSH] event=FAIL region=7 chunk=0"
                " reason=RPC_INPUT_STREAM_PUSH_REJECTED\n"
                "[P2C_STREAM] phase=STOP recv=174 pending=116 status=FAIL\n"
                "srv decode: Compute error.\n",
                encoding="utf-8",
            )
            found = probe.first_error_markers(logfile)
            self.assertIn("region=7 chunk=0", found)
            self.assertIn("RPC_INPUT_STREAM_PUSH_REJECTED", found)
            self.assertIn("recv=174", found)
            self.assertLess(
                found.index("PHONE_INPUT_PUSH"),
                found.index("P2C_STREAM"))

    def test_original_layout_has_exactly_64_layers(self):
        layers = sum(int(part[1:]) for part in probe.LAYOUT.split(","))
        self.assertEqual(layers, 64)
        self.assertEqual(probe.LAYOUT.split(",")[-1], "G7")


if __name__ == "__main__":
    unittest.main()
