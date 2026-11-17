#!/usr/bin/env python3
"""Offline tests for the persistent HTP-only llama-server smoke harness."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
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
        self.assertEqual(zero["chunk_plan_distribution"], "")
        records[0].update({"plan_xc": 16, "plan_xg": 16,
                           "plan_xp": 24, "planner_ms": 6.0})
        records[1].update({"plan_xc": 24, "plan_xg": 32,
                           "plan_xp": 32, "planner_ms": 4.0})
        with_plans = probe.report_rows(
            records, probe.runner.MAIN50_SHORT_GROUPS)[0]
        self.assertEqual(with_plans["chunk_plan_variants"], 2)
        self.assertIn("XC16-XG16-XP24:1",
                      with_plans["chunk_plan_distribution"])
        self.assertEqual(with_plans["mean_planner_ms"], 5.0)

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
            with_runtime = probe.sweep_config(
                args, ["llama-server", "--parallel", "1"],
                {"LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS": "1",
                 "LLAMA_REGION_RING_PER_REQUEST_CHUNKS": "1"},
                prompts, 1855)
            self.assertNotEqual(third["fingerprint"],
                                with_runtime["fingerprint"])

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

    def test_native_per_request_planner_log_is_required_and_parsed(self):
        layout = "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7"
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "llama-server.log"
            line = (
                "12.345 E [REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"
                f" tokens=66 kv_start=0 layout={layout}"
                " XC=24 XG=16 XP=24"
                " startup_XC=32 startup_XG=32 startup_XP=32"
                " changed=1 planner_ms=5.125"
                " source=LOCKED_CHUNK_ONLY layer_placement=IMMUTABLE"
                " execution=THIS_DECODE_ONLY status=PASS\n")
            runtime = (
                "[REGION_RING_SUM] tokens=66 XC=0 XG=17 XP=25"
                " regions=11 first_region_chunks=4 tasks=44"
                " executed=44 complete=1 wall_ms=9.0"
                " mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0\n")
            log.write_text(
                "[PHONE_HTP_ONLY] event=REJECT reason=REQUIRES_GRAPHLESS_FRESH_PREFILL\n"
                + line + runtime, encoding="utf-8")
            decision, offset = probe.read_request_chunk_plan(log, 0, 66)
            self.assertEqual(
                (decision["plan_xc"], decision["plan_xg"],
                 decision["plan_xp"]), (24, 16, 24))
            self.assertAlmostEqual(decision["planner_ms"], 5.125)
            self.assertEqual((decision["runtime_xc"],
                              decision["runtime_xg"],
                              decision["runtime_xp"]), (0, 17, 25))
            self.assertEqual(offset, log.stat().st_size)
            log.write_text(line, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "execution missing"):
                probe.read_request_chunk_plan(log, 0, 66)
            log.write_text(line + runtime, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
                probe.read_request_chunk_plan(log, offset, 66)
            with self.assertRaisesRegex(RuntimeError, "token mismatch"):
                probe.read_request_chunk_plan(log, 0, 1855)
            log.write_text(line + runtime + line + runtime,
                           encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "matches=2"):
                probe.read_request_chunk_plan(log, 0, 66)

    def test_native_per_request_plan_rejects_wrong_layout_and_bank_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "llama-server.log"
            line = ("[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"
                    " tokens=66 kv_start=0 layout={layout}"
                    " XC=32 XG={xg} XP=16"
                    " startup_XC=32 startup_XG=32 startup_XP=32"
                    " changed=1 planner_ms=0.100"
                    " source=LOCKED_CHUNK_ONLY"
                    " execution=THIS_DECODE_ONLY status=PASS\n")
            correct = "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7"
            runtime = (
                "[REGION_RING_SUM] tokens=66 XC=0 XG=33 XP=17"
                " regions=11 first_region_chunks=2 tasks=22"
                " executed=22 complete=1 wall_ms=9.0"
                " mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0\n")
            log.write_text(
                line.format(layout=correct, xg=32) + runtime,
                encoding="utf-8")
            got, _ = probe.read_request_chunk_plan(log, 0, 66)
            self.assertEqual(got["plan_xg"], 32)
            log.write_text(
                line.format(layout="G8,P4,G7,P4,G8,P5,G7,P4,G7,P4,G6",
                            xg=32), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed fixed layout"):
                probe.read_request_chunk_plan(log, 0, 66)
            # Wider chunks are valid for long prompts up to the native cap.
            long_line = (line.format(layout=correct, xg=256)
                         .replace("tokens=66", "tokens=1855")
                         .replace(" XP=16", " XP=256"))
            long_runtime = (
                "[REGION_RING_SUM] tokens=1855 XC=0 XG=256 XP=256"
                " regions=11 first_region_chunks=8 tasks=88"
                " executed=88 complete=1 wall_ms=100.0"
                " mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0\n")
            log.write_text(long_line + long_runtime, encoding="utf-8")
            got, _ = probe.read_request_chunk_plan(log, 0, 1855)
            self.assertEqual((got["plan_xg"], got["plan_xp"]), (256, 256))
            with self.assertRaisesRegex(RuntimeError, "width bounds"):
                probe.read_request_chunk_plan(
                    log, 0, 1855, max_plan_xg=32, max_plan_xp=32)

            log.write_text(line.format(layout=correct, xg=272),
                           encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "width bounds"):
                probe.read_request_chunk_plan(log, 0, 66)
            log.write_text(line.format(layout=correct, xg=48),
                           encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "insufficient tokens"):
                probe.read_request_chunk_plan(log, 0, 66)

    def test_24b_layout_is_accepted_and_verified_for_resident_server(self):
        layout = "G5,P3,G5,P3,G6,P3,G5,P4,G6"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = root / "llama-server.exe"
            model = root / "Mistral-Small-24B-Q6_K.gguf"
            server.write_bytes(b"stub")
            model.write_bytes(b"stub")
            args = probe.parse_args([
                "--server-exe", str(server), "--model", str(model),
                "--rpc", "10.0.0.1:50052",
                "--prompt-dir", str(root),
                "--profile-source", str(root),
                "--out", str(root / "24b-server-test"),
                "--model-layers", "40",
                "--locked-layout", layout,
                "--per-request-chunks", "--all50",
            ])
            self.assertEqual(args.model_layers, 40)
            self.assertEqual(args.locked_layout, layout)
            self.assertTrue(args.per_request_chunks)
            with self.assertRaises(SystemExit):
                probe.parse_args([
                    "--server-exe", str(server), "--model", str(model),
                    "--rpc", "10.0.0.1:50052",
                    "--prompt-dir", str(root), "--profile-source", str(root),
                    "--out", str(root / "bad"),
                    "--model-layers", "64", "--locked-layout", layout,
                ])
            log = root / "llama-server.log"
            log.write_text(
                "[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"
                " tokens=1855 kv_start=0 layout=" + layout
                + " XC=64 XG=96 XP=128"
                " startup_XC=64 startup_XG=64 startup_XP=64"
                " changed=1 planner_ms=12.125"
                " source=LOCKED_CHUNK_ONLY"
                " execution=THIS_DECODE_ONLY status=PASS\n"
                "[REGION_RING_SUM] tokens=1855 XC=0 XG=96 XP=128"
                " regions=9 first_region_chunks=20 tasks=180"
                " executed=180 complete=1 wall_ms=1234"
                " mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0\n",
                encoding="utf-8")
            planned, _ = probe.read_request_chunk_plan(
                log, 0, 1855, 256, 256, layout)
            self.assertEqual(
                (planned["plan_xg"], planned["plan_xp"]), (96, 128))
            with self.assertRaisesRegex(RuntimeError, "changed fixed layout"):
                probe.read_request_chunk_plan(log, 0, 1855)

    def test_reference_chunk_replay_validates_full_cycle_run(self):
        layout = "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "Qwen3-32B-Q4_K_M.gguf"
            model.write_bytes(b"model")
            profile = root / "independent-profile"
            profile.mkdir()
            (profile / "profile.csv").write_text("test\n", encoding="utf-8")
            (profile / "profile.json").write_text("{}", encoding="utf-8")
            for group in probe.runner.MAIN50_SHORT_GROUPS:
                for index in range(1, 11):
                    (root / f"prompt_{group}_{index:02d}.txt").write_text(
                        f"Sample {group} {index}", encoding="utf-8")
            prompts = probe.prompt_files(root, all50=True)
            config = {"configuration": {
                "prompts": [p.name for p in prompts],
                "prompt_sha256": {
                    p.name: probe.sha256_file(p) for p in prompts},
                "model": probe.file_identity(model),
                "profile_sha256": {
                    name: probe.sha256_file(profile / name)
                    for name in ("profile.csv", "profile.json")},
                "runtime_env": {
                    "LLAMA_REGION_RING_PER_REQUEST_CHUNKS": "1"},
            }}
            baseline = root / "full-cycle"
            baseline.mkdir()
            (baseline / "server-run-config.json").write_text(
                json.dumps(config), encoding="utf-8")
            rows = [{
                "index": i,
                "prompt": p.name,
                "status": "PASS",
                "cache_n": 0,
                "planner_mode": "PER_REQUEST_LOCKED",
                "runtime_layout": layout,
                "prompt_n": 66 if "prompt_0_" in p.name else 1024,
                "plan_xc": 16,
                "plan_xg": 24,
                "plan_xp": 32,
            } for i, p in enumerate(prompts, 1)]
            source = baseline / "requests.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
            decisions = probe.load_reference_chunks(
                source, prompts, layout, model, profile)
            self.assertEqual(len(decisions), 50)
            self.assertEqual(
                (decisions["prompt_0_01.txt"]["plan_xg"],
                 decisions["prompt_0_01.txt"]["plan_xp"]),
                (24, 32))
            sidecar = root / "replay.txt"
            probe.write_replay_sidecar(sidecar, 66, layout, 16, 24, 32)
            self.assertEqual(sidecar.read_text(encoding="ascii"),
                             f"66 {layout} 16 24 32\n")
            probe.write_replay_sidecar(sidecar, 1024, layout, 32, 64, 128)
            self.assertEqual(sidecar.read_text(encoding="ascii"),
                             f"1024 {layout} 32 64 128\n")
            # Fail closed: a missing/non-PASS Prompt or altered Chunk width
            # must not quietly change the ablation's independent variable.
            rows[1]["status"] = "FAIL"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a PASS"):
                probe.load_reference_chunks(
                    source, prompts, layout, model, profile)
            rows[1]["status"] = "PASS"
            rows[1]["plan_xg"] = 64  # 2*64 > 66
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid recorded"):
                probe.load_reference_chunks(
                    source, prompts, layout, model, profile)
            rows[1]["plan_xg"] = 24
            rows.pop()
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "50 PASS rows"):
                probe.load_reference_chunks(
                    source, prompts, layout, model, profile)

    def test_native_replay_log_source_is_mandatory(self):
        layout = "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7"
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "llama-server.log"
            sample = (
                "[REGION_RING_RESOURCE_SCHED] enabled=1 bank_aware=1 status=READY\n"
                "[REGION_RING_CYCLE_SCHED] mode=FIFO rule=EARLIEST_EXECUTABLE_REGION status=READY\n"
                "[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"
                " tokens=66 kv_start=0 layout=" + layout +
                " XC=16 XG=24 XP=32"
                " startup_XC=32 startup_XG=32 startup_XP=32"
                " changed=1 planner_ms=0.200"
                " source=LOCKED_CHUNK_REPLAY"
                " layer_placement=IMMUTABLE"
                " execution=THIS_DECODE_ONLY status=PASS\n"
                "[REGION_RING_SUM] tokens=66 XC=0 XG=24 XP=33"
                " regions=11 first_region_chunks=3 tasks=33"
                " executed=33 complete=1 wall_ms=300"
                " mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0\n")
            log.write_text(sample, encoding="utf-8")
            decision, _ = probe.read_request_chunk_plan(
                log, 0, 66, 256, 256, layout, "LOCKED_CHUNK_REPLAY", "FIFO")
            self.assertEqual(decision["plan_xg"], 24)
            self.assertEqual(decision["plan_xp"], 32)
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout, "LOCKED_CHUNK_ONLY")
            with self.assertRaisesRegex(RuntimeError, "scheduler ablation"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout, "LOCKED_CHUNK_REPLAY",
                    "DRAIN_FIRST")

    def test_cpp_request_replay_hook_checks_tokens_and_layout(self):
        source = (Path(__file__).resolve().parents[2] /
                  "src" / "llama-context.cpp").read_text(encoding="utf-8")
        start = source.index(
            'std::getenv("LLAMA_REGION_RING_REQUEST_CHUNK_REPLAY_FILE")')
        block = source[start:source.index("if (region_ring_v1 &&", start)]
        for phrase in (
            "std::ifstream replay(replay_path)",
            "replay_tokens != n_tokens_all",
            "request_layout != model.hybrid_ring_plan_layout",
            "source=%s",
            '"LOCKED_CHUNK_REPLAY"',
        ):
            self.assertIn(phrase, block)

    def test_cpp_request_hook_reuses_locked_planner_and_stage_builder(self):
        source = (Path(__file__).resolve().parents[2] /
                  "src" / "llama-context.cpp").read_text(
                      encoding="utf-8")
        start = source.index(
            'std::getenv("LLAMA_REGION_RING_PER_REQUEST_CHUNKS")')
        end = source.index("if (region_ring_v1 &&", start)
        hook = source[start:end]
        for phrase in (
            'n_tokens_all >= 32',
            'ring_prefill_input_compatible',
            'model.hybrid_ring_plan_mode != "APPLY_WIDE"',
            '"LLAMA_HYBRID_REGION_RING_LOCKED_LAYOUT"',
            'llama_hybrid_region_ring_planner_apply_wide(',
            'request_layout != model.hybrid_ring_plan_layout',
            'runtime_stages = llama_region_ring_runtime_stages(',
            '"[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"',
        ):
            self.assertIn(phrase, hook)
        self.assertNotIn("model.hybrid_ring_plan_gpu_chunk_tokens =", hook)
        self.assertLess(
            source.index("runtime_stages = llama_region_ring_runtime_stages(",
                         start),
            source.index("process_ubatch_region_ring(", start))

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
