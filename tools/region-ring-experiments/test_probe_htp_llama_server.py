#!/usr/bin/env python3
"""Offline tests for the persistent HTP-only llama-server smoke harness."""
from io import BytesIO
import csv
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
    def test_legacy_fixed64_no_dag_profile_accepts_only_real_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "profile.csv"
            names = ("device", "layers", "tokens", "kv_tokens",
                     "setup_ms", "upload_ms", "prepare_ms",
                     "compute_ms", "terminal", "samples")
            rows = [
                dict(zip(names, (dev, 2, 64, kv, 1, 0, 2, 3, 0, 5)))
                for dev, kv in (("CPU", 1024), ("GPU", 1024),
                                ("PHONE", 2048))
            ]
            metadata = {
                "schema": 1, "mode": "INDEPENDENT_BACKEND_PROFILE",
                "model": {"layers": 64, "hidden": 5120},
                "chunk_widths": [32, 64, 96, 128],
                "backends": {"G": "CUDA", "C": "CPU", "N": "HTP_STRIPED"},
                "stage_points": [],
                "transfers": {
                    "pc_to_phone": [{"bytes": 4096, "ms": 1.1}],
                    "phone_to_pc": [{"bytes": 4096, "ms": 1.2}],
                    "gpu_weight_h2d": [{"bytes": 4096, "ms": 1.3}],
                },
            }

            def save():
                with csv_path.open("w", newline="", encoding="utf-8") as fp:
                    w = csv.DictWriter(fp, fieldnames=names)
                    w.writeheader()
                    w.writerows(rows)
                metadata["stage_points"] = [dict(row) for row in rows]
                csv_path.with_suffix(".json").write_text(
                    json.dumps(metadata), encoding="utf-8")

            save()
            self.assertEqual(
                probe.runner.verify_independent_profile(
                    csv_path, fixed_chunk=64), 1)
            with self.assertRaisesRegex(ValueError, "measured CPU X32"):
                probe.runner.verify_independent_profile(
                    csv_path, fixed_chunk=64, adaptive_no_dag=True)
            rows.extend([
                dict(zip(names, (dev, 2, 32, kv, 1, 0, 2, 3, 0, 5)))
                for dev, kv in (("CPU", 256), ("GPU", 256),
                                ("PHONE", 256))
            ])
            save()
            self.assertEqual(
                probe.runner.verify_independent_profile(
                    csv_path, fixed_chunk=64, adaptive_no_dag=True), 2)
            with self.assertRaisesRegex(ValueError, "Short Chunk run"):
                probe.runner.verify_independent_profile(
                    csv_path, short_chunks=True)
            metadata["chunk_widths"].remove(64)
            save()
            with self.assertRaisesRegex(ValueError, "requires X64"):
                probe.runner.verify_independent_profile(
                    csv_path, fixed_chunk=64)
            metadata["chunk_widths"].append(64)
            rows[2]["kv_tokens"] = 1024
            save()
            with self.assertRaisesRegex(ValueError, "KV=2048"):
                probe.runner.verify_independent_profile(
                    csv_path, fixed_chunk=64)
            rows[2]["kv_tokens"] = 2048
            rows[1]["tokens"] = 128
            save()
            with self.assertRaisesRegex(ValueError, "GPU X64"):
                probe.runner.verify_independent_profile(
                    csv_path, fixed_chunk=64)

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

    def test_wo_async_prefetch_requires_baseline_and_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, model = root / "llama-server.exe", root / "model.gguf"
            server.write_bytes(b"stub")
            model.write_bytes(b"stub")
            reference = root / "full-cycle-requests.jsonl"
            reference.write_text("{}\n", encoding="utf-8")
            base = [
                "--server-exe", str(server), "--model", str(model),
                "--rpc", "127.0.0.1:50052",
                "--prompt-dir", str(root), "--profile-source", str(root),
                "--out", str(root / "ablation"),
                "--all50", "--per-request-chunks",
                "--replay-chunks-from", str(reference),
            ]
            parsed = probe.parse_args(base + ["--no-async-prefetch"])
            self.assertTrue(parsed.no_async_prefetch)
            self.assertFalse(parsed.no_drain_first)
            with self.assertRaises(SystemExit):
                probe.parse_args(
                    base + ["--no-async-prefetch", "--no-drain-first"])
            with self.assertRaises(SystemExit):
                probe.parse_args([
                    "--server-exe", str(server), "--model", str(model),
                    "--rpc", "127.0.0.1:50052",
                    "--prompt-dir", str(root),
                    "--profile-source", str(root),
                    "--out", str(root / "ablation"),
                    "--no-async-prefetch",
                ])

    def test_sync_demand_log_verifies_two_banks_and_drain_first(self):
        layout = "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7"
        prefix = (
            "[REGION_RING_RESOURCE_SCHED] enabled=1 bank_aware=1"
            " phone_push_aware=1 status=READY\n"
            "[REGION_RING_CYCLE_SCHED] mode=DRAIN_FIRST"
            " rule=EARLIEST_EXECUTABLE_REGION status=READY\n")
        selected = (
            "[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"
            f" tokens=66 kv_start=0 layout={layout}"
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
        mode = (
            "[REGION_RING_ABLATION] gpu_banks=2 gpu_prefetch=0"
            " gpu_bank_mover=0 sync_demand=1 status=CONFIGURED\n"
            "[REGION_RING_GPU_BANK_REFILL_MODE] mode=SYNC_ON_DEMAND"
            " banks=2 background_thread=0"
            " same_parity_drain_gate=1 status=READY\n")
        refill_summary = (
            "[REGION_RING_GPU_BANK_MOVER_SUM]"
            " worker=GPU_SYNC_DEMAND started=4 completed=4"
            " failures=0 upload_mib=3456.0 stage_cpu_copy_ms=0"
            " stage_chunks=0 prepare_ms=10 h2d_ms=350 wall_ms=360"
            " gpu_wait_ms=360 thread_started=0 thread_joined=0"
            " status=PASS\n")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text(
                mode + prefix + selected + refill_summary,
                encoding="utf-8")
            result, _ = probe.read_request_chunk_plan(
                log, 0, 66, 256, 256, layout,
                "LOCKED_CHUNK_REPLAY", "DRAIN_FIRST", True)
            self.assertEqual(
                (result["plan_xg"], result["plan_xp"]), (24, 32))
            log.write_text(
                prefix + selected + refill_summary, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "w/o Async Prefetch"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "LOCKED_CHUNK_REPLAY", "DRAIN_FIRST", True)
            log.write_text(
                mode.replace("gpu_banks=2", "gpu_banks=1")
                + prefix + selected + refill_summary, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "w/o Async Prefetch"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "LOCKED_CHUNK_REPLAY", "DRAIN_FIRST", True)
            log.write_text(
                mode + prefix + selected +
                refill_summary.replace("completed=4", "completed=3"),
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "synchronous Bank refills"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "LOCKED_CHUNK_REPLAY", "DRAIN_FIRST", True)
            log.write_text(
                mode + prefix.replace("mode=DRAIN_FIRST", "mode=FIFO")
                + selected + refill_summary,
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "scheduler ablation"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "LOCKED_CHUNK_REPLAY", "DRAIN_FIRST", True)

    def test_strict_sync_demand_reuses_original_streaming_refill_path(self):
        source = (Path(__file__).resolve().parents[2] /
                  "src" / "llama-context.cpp").read_text(encoding="utf-8")
        for token in (
            '"LLAMA_REGION_RING_GPU_BANK_SYNC_DEMAND"',
            "gpu_bank_sync_demand =",
            "gpu_bank_streaming && gpu_bank_sync_demand_requested",
            "gpu_bank_drained_ranks",
            "gpu_bank_streaming && !gpu_bank_sync_demand",
            "refill_gpu_bank(",
            "const bool refilled = refill_gpu_bank(",
            "ready.notify_external_ready();",
            '"[REGION_RING_GPU_BANK_REFILL_MODE] mode=SYNC_ON_DEMAND"',
            '"[REGION_RING_GPU_BANK_SYNC_DEMAND]"',
            "gpu_bank_initial_residency =",
        ):
            self.assertIn(token, source)

    def test_planner_fixed_64_and_128_cli_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = root / "llama-server.exe"
            model = root / "Qwen3-32B-Q4_K_M.gguf"
            server.write_bytes(b"stub")
            model.write_bytes(b"stub")
            baseline = root / "baseline.jsonl"
            baseline.write_text("{}\n", encoding="utf-8")
            args_list = [
                "--server-exe", str(server), "--model", str(model),
                "--rpc", "127.0.0.1:50052",
                "--prompt-dir", str(root), "--profile-source", str(root),
                "--out", str(root / "fixed-128"),
                "--all50", "--per-request-chunks",
                "--model-layers", "64",
                "--locked-layout", probe.DEFAULT_CANONICAL_LAYOUT,
                "--fixed-chunk", "128",
            ]
            args = probe.parse_args(args_list)
            self.assertEqual(args.fixed_chunk, 128)
            self.assertIsNone(args.replay_chunks_from)
            with self.assertRaises(SystemExit):
                probe.parse_args(args_list + [
                    "--replay-chunks-from", str(baseline)])
            with self.assertRaises(SystemExit):
                probe.parse_args(args_list + ["--no-multi-cycle"])
            with self.assertRaises(SystemExit):
                probe.parse_args(args_list + ["--no-async-prefetch"])
            with self.assertRaises(SystemExit):
                probe.parse_args(args_list + ["--no-drain-first"])
            args64 = probe.parse_args(args_list[:-2] + ["--fixed-chunk", "64"])
            self.assertEqual(args64.fixed_chunk, 64)
            with self.assertRaises(SystemExit):
                probe.parse_args(args_list[:-2] + ["--fixed-chunk", "96"])
            new_layout = "G10,P4,G10,P4,G10,P4,G10,P4,G8"
            new_args = args_list[:-4] + [
                "--locked-layout", new_layout, "--fixed-chunk", "64"]
            with self.assertRaises(SystemExit):
                probe.parse_args(new_args)
            with self.assertRaises(SystemExit):
                probe.parse_args(new_args + ["--no-dag-placement"])
            with self.assertRaises(SystemExit):
                probe.parse_args(new_args + [
                    "--no-dag-placement", "--guarded-memory-checks"])
            dynamic_args = args_list[:-4] + [
                "--locked-layout", new_layout]
            with self.assertRaises(SystemExit):
                probe.parse_args(dynamic_args + ["--no-dag-placement"])
            valid_no_dag = probe.parse_args(dynamic_args + [
                "--no-dag-placement", "--guarded-memory-checks"])
            self.assertTrue(valid_no_dag.no_dag_placement)
            self.assertTrue(valid_no_dag.guarded_memory_checks)
            self.assertIsNone(valid_no_dag.fixed_chunk)
            self.assertTrue(valid_no_dag.per_request_chunks)
            self.assertEqual(valid_no_dag.locked_layout, new_layout)
            skip_no_dag = probe.parse_args(dynamic_args + [
                "--no-dag-placement", "--skip-plan-memory-checks"])
            self.assertTrue(skip_no_dag.skip_plan_memory_checks)
            self.assertFalse(skip_no_dag.guarded_memory_checks)
            self.assertEqual(skip_no_dag.locked_layout, new_layout)
            with self.assertRaises(SystemExit):
                probe.parse_args(dynamic_args + [
                    "--no-dag-placement", "--skip-plan-memory-checks",
                    "--guarded-memory-checks"])
            with self.assertRaises(SystemExit):
                probe.parse_args(dynamic_args + [
                    "--skip-plan-memory-checks"])
            with self.assertRaises(SystemExit):
                probe.parse_args(args_list + [
                    "--no-dag-placement", "--guarded-memory-checks"])
            with self.assertRaises(SystemExit):
                probe.parse_args([a for a in dynamic_args if a !=
                                  "--per-request-chunks"] +
                                 ["--no-dag-placement",
                                  "--guarded-memory-checks"])
            with self.assertRaises(SystemExit):
                probe.parse_args([a for a in dynamic_args if a != "--all50"] +
                                 ["--no-dag-placement",
                                  "--guarded-memory-checks"])
            with self.assertRaises(SystemExit):
                without_request = args_list.copy()
                without_request.remove("--per-request-chunks")
                probe.parse_args(without_request)
            with self.assertRaises(SystemExit):
                without_all50 = args_list.copy()
                without_all50.remove("--all50")
                probe.parse_args(without_all50)

    def test_locked_no_dag_memory_skip_is_estimate_only(self):
        source = (Path(__file__).resolve().parents[2] /
                  "src" / "llama-hybrid.cpp").read_text(encoding="utf-8")
        # Guarded new-placement search must remain separate from the
        # already-supported explicit skip on an immutable locked layout.
        no_dag = source.split(
            'const bool no_dag_planner =', 1)[1].split(
            'if (wide_apply && selected_layout != nullptr)', 1)[0]
        self.assertIn('model.linear_no_dag_ablation = true;', no_dag)
        self.assertNotIn('skip_locked_memory || !model.memory_feasible',
                         no_dag)
        self.assertIn('skip_locked_memory ? "ESTIMATES_SKIPPED"', no_dag)
        self.assertIn('if (!skip_locked_memory &&', source)
        self.assertIn('policy=SKIP_ESTIMATES_ONLY', source)

    def test_no_dag_dynamic_chunks_preserve_locked_layout_and_source(self):
        layout = "G8,P4,G6,C1,P5,G8,C1,P5,G6,P6,G8,C1,G5"
        scheduler = (
            "[REGION_RING_RESOURCE_SCHED] enabled=1 bank_aware=1"
            " phone_push_aware=1 status=READY\n"
            "[REGION_RING_CYCLE_SCHED] mode=DRAIN_FIRST"
            " rule=EARLIEST_EXECUTABLE_REGION status=READY\n"
        )

        def request(tokens, xc, xg, xp, source="LINEAR_NO_DAG_CHUNK_ONLY"):
            return (
                f"[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"
                f" tokens={tokens} kv_start=0 layout={layout}"
                f" XC={xc} XG={xg} XP={xp}"
                " startup_XC=64 startup_XG=64 startup_XP=64"
                " changed=1 planner_ms=0.450"
                f" source={source} layer_placement=IMMUTABLE"
                " execution=THIS_DECODE_ONLY status=PASS\n"
                f"[REGION_RING_SUM] tokens={tokens} XC={xc} XG={xg} XP={xp}"
                " regions=13 first_region_chunks=2 tasks=26"
                " executed=26 complete=1 wall_ms=200.0"
                " mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0\n"
            )

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "linear-no-dag.log"
            log.write_text(scheduler + request(128, 64, 32, 64),
                           encoding="utf-8")
            first, _ = probe.read_request_chunk_plan(
                log, 0, 128, 256, 256, layout,
                "LINEAR_NO_DAG_CHUNK_ONLY", "DRAIN_FIRST")
            self.assertEqual(
                (first["plan_xc"], first["plan_xg"], first["plan_xp"]),
                (64, 32, 64))
            log.write_text(scheduler + request(1855, 128, 96, 192),
                           encoding="utf-8")
            second, _ = probe.read_request_chunk_plan(
                log, 0, 1855, 256, 256, layout,
                "LINEAR_NO_DAG_CHUNK_ONLY", "DRAIN_FIRST")
            self.assertEqual(
                (second["plan_xc"], second["plan_xg"], second["plan_xp"]),
                (128, 96, 192))
            self.assertNotEqual(
                (first["plan_xc"], first["plan_xg"], first["plan_xp"]),
                (second["plan_xc"], second["plan_xg"], second["plan_xp"]))
            log.write_text(scheduler + request(
                128, 64, 32, 64, source="LOCKED_CHUNK_ONLY"),
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                probe.read_request_chunk_plan(
                    log, 0, 128, 256, 256, layout,
                    "LINEAR_NO_DAG_CHUNK_ONLY", "DRAIN_FIRST")
            log.write_text(scheduler + request(
                128, 64, 32, 64).replace(layout, "G64"),
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed fixed layout"):
                probe.read_request_chunk_plan(
                    log, 0, 128, 256, 256, layout,
                    "LINEAR_NO_DAG_CHUNK_ONLY", "DRAIN_FIRST")

    def test_planner_fixed_64_and_128_native_single_chunk_and_mover(self):
        layout = probe.DEFAULT_CANONICAL_LAYOUT

        def make_log(n, selected=None, source="FIXED_CHUNK_ABLATION",
                     async_mover=True, actual=None, target=128,
                     layout_override=None, refill_count=4):
            width = min(n, target) if selected is None else selected
            actual_width = width if actual is None else actual
            return (
                "[REGION_RING_RESOURCE_SCHED] enabled=1 bank_aware=1"
                " phone_push_aware=1 status=READY\n"
                "[REGION_RING_CYCLE_SCHED] mode=DRAIN_FIRST"
                " rule=EARLIEST_EXECUTABLE_REGION status=READY\n"
                f"[REGION_RING_FIXED_CHUNK] target={target}"
                f" tokens={n} XC={width} XG={width} XP={width}"
                f" policy={'WHOLE_PROMPT_SINGLE_CHUNK' if n < target else 'FIXED_' + str(target)}"
                " gpu_small_batch_mover=FORCED"
                " profile_queries=0 status=PASS\n"
                "[REGION_RING_ABLATION] gpu_banks=2 gpu_prefetch=1"
                f" gpu_bank_mover={int(async_mover)} sync_demand=0"
                " status=CONFIGURED\n"
                "[REGION_RING_GPU_BANK] phase=INIT banks=2"
                " mapping=FIXED_PARITY canonical=CPU destination=CUDA"
                " mover=CPU_SINGLE min_tokens=67"
                " force_small_batch=1 override_applied=1 status=READY\n"
                "[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"
                f" tokens={n} kv_start=0 layout={layout_override or layout}"
                f" XC={width} XG={width} XP={width}"
                " startup_XC=32 startup_XG=32 startup_XP=32"
                " changed=1 planner_ms=0.010"
                f" source={source}"
                " layer_placement=IMMUTABLE"
                " execution=THIS_DECODE_ONLY status=PASS\n"
                f"[REGION_RING_SUM] tokens={n} XC=0"
                f" XG={actual_width} XP={actual_width}"
                " regions=11 first_region_chunks=1 tasks=11"
                " executed=11 complete=1 wall_ms=300"
                " mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0\n"
                "[REGION_RING_GPU_BANK_MOVER_SUM] worker=CPU_SINGLE"
                f" started={refill_count} completed={refill_count}"
                " failures=0 upload_mib=1234"
                " stage_cpu_copy_ms=0 stage_chunks=0"
                " prepare_ms=50 h2d_ms=100 wall_ms=200"
                " gpu_wait_ms=0 thread_started=1 thread_joined=1"
                " status=PASS\n"
            )

        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            # Shorter than 128 => exactly one unsplit 66-token Stage
            log.write_text(make_log(66), encoding="utf-8")
            result, _ = probe.read_request_chunk_plan(
                log, 0, 66, 256, 256, layout,
                "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                False, False, 128)
            self.assertEqual(
                (result["plan_xc"], result["plan_xg"],
                 result["plan_xp"]), (66, 66, 66))
            self.assertEqual(
                (result["runtime_xg"], result["runtime_xp"]), (66, 66))

            # Exactly one normal 128-token Chunk is also permitted
            log.write_text(make_log(128), encoding="utf-8")
            result, _ = probe.read_request_chunk_plan(
                log, 0, 128, 256, 256, layout,
                "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                False, False, 128)
            self.assertEqual((result["plan_xg"], result["plan_xp"]),
                             (128, 128))

            # Long prompt keeps 128 and may coalesce a tiny tail
            log.write_text(make_log(513, actual=129), encoding="utf-8")
            result, _ = probe.read_request_chunk_plan(
                log, 0, 513, 256, 256, layout,
                "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                False, False, 128)
            self.assertEqual((result["plan_xg"], result["plan_xp"]),
                             (128, 128))
            self.assertEqual(result["runtime_xg"], 129)

            # Fixed-64: a 66-token request must select 64, not 128.
            log.write_text(make_log(66, target=64), encoding="utf-8")
            result, _ = probe.read_request_chunk_plan(
                log, 0, 66, 256, 256, layout,
                "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                False, False, 64)
            self.assertEqual(
                (result["plan_xc"], result["plan_xg"], result["plan_xp"]),
                (64, 64, 64))

            # Shorter than the fixed-64 target remains exactly one Chunk.
            log.write_text(make_log(48, target=64), encoding="utf-8")
            result, _ = probe.read_request_chunk_plan(
                log, 0, 48, 256, 256, layout,
                "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                False, False, 64)
            self.assertEqual(
                (result["plan_xc"], result["plan_xg"], result["plan_xp"]),
                (48, 48, 48))
            self.assertEqual((result["runtime_xg"], result["runtime_xp"]),
                             (48, 48))

            # Different GPU Region count needs a different refill count.
            variable = "G10,P4,G10,P4,G10,P4,G10,P4,G8"
            log.write_text(
                make_log(128, target=64, layout_override=variable,
                         refill_count=3),
                encoding="utf-8")
            result, _ = probe.read_request_chunk_plan(
                log, 0, 128, 256, 256, variable,
                "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                False, False, 64)
            self.assertEqual(result["plan_xg"], 64)

            # A fixed-128 log must not be accepted as a fixed-64 result.
            log.write_text(make_log(66, target=128), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prescribed widths"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                    False, False, 64)

            # A small request must NOT be re-chunked to a legal preset
            log.write_text(make_log(66, selected=32), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prescribed widths"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                    False, False, 128)
            log.write_text(make_log(66, source="LOCKED_CHUNK_ONLY"),
                           encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                    False, False, 128)
            log.write_text(make_log(66, async_mover=False),
                           encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "async mover"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "FIXED_CHUNK_ABLATION", "DRAIN_FIRST",
                    False, False, 128)

    def test_wo_multi_cycle_accepts_only_64layer_g36_p21_g7(self):
        self.assertEqual(probe.GPG_LOGICAL_LAYOUT, "G36,P21,G7")
        self.assertEqual(probe.GPG_PHYSICAL_LAYOUT,
                         "G7,G7,G8,G7,G7,P21,G7")
        segments = probe.GPG_PHYSICAL_LAYOUT.split(",")
        self.assertEqual(
            sum(int(s[1:]) for s in segments if s[0] == "G"), 43)
        self.assertEqual(
            sum(int(s[1:]) for s in segments if s[0] == "P"), 21)
        self.assertEqual(
            [int(s[1:]) for s in segments if s[0] == "G"],
            [7, 7, 8, 7, 7, 7])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server, model = root / "llama-server.exe", root / "32b.gguf"
            server.write_bytes(b"stub")
            model.write_bytes(b"stub")
            reference = root / "baseline.jsonl"
            reference.write_text("{}\n", encoding="utf-8")
            base = [
                "--server-exe", str(server), "--model", str(model),
                "--rpc", "127.0.0.1:50052",
                "--prompt-dir", str(root), "--profile-source", str(root),
                "--out", str(root / "out"),
                "--all50", "--per-request-chunks",
                "--locked-layout", probe.GPG_LOGICAL_LAYOUT,
            ]
            args = probe.parse_args(base + ["--no-multi-cycle"])
            self.assertTrue(args.no_multi_cycle)
            self.assertTrue(args.per_request_chunks)
            self.assertIsNone(args.replay_chunks_from)
            self.assertEqual(args.locked_layout, probe.GPG_LOGICAL_LAYOUT)
            with self.assertRaises(SystemExit):
                probe.parse_args(base + [
                    "--no-multi-cycle", "--replay-chunks-from", str(reference)])
            with self.assertRaises(SystemExit):
                no_per_request = base.copy()
                no_per_request.remove("--per-request-chunks")
                probe.parse_args(no_per_request + ["--no-multi-cycle"])
            with self.assertRaises(SystemExit):
                no_all50 = base.copy()
                no_all50.remove("--all50")
                probe.parse_args(no_all50 + ["--no-multi-cycle"])
            with self.assertRaises(SystemExit):
                probe.parse_args(base + [
                    "--no-multi-cycle", "--no-async-prefetch"])
            with self.assertRaises(SystemExit):
                probe.parse_args(base + [
                    "--no-multi-cycle", "--no-drain-first"])
            invalid = base.copy()
            invalid[invalid.index("--locked-layout") + 1] = (
                probe.DEFAULT_CANONICAL_LAYOUT)
            with self.assertRaises(SystemExit):
                probe.parse_args(invalid + ["--no-multi-cycle"])

    def test_wo_multi_cycle_requires_async_mover_and_native_chunk_plan(self):
        layout = probe.GPG_PHYSICAL_LAYOUT
        full = (
            "[REGION_RING_MULTI_CYCLE_ABLATION] mode=G_P_G"
            " logical=G36,P21,G7"
            " physical=G7,G7,G8,G7,G7,P21,G7"
            " gpu_layers=43 phone_layers=21"
            " gpu_regions=6 phone_regions=1"
            " gpu_banks=2 async_prefetch=1"
            " drain_first=1 status=READY\n"
            "[REGION_RING_ABLATION] gpu_banks=2 gpu_prefetch=1"
            " gpu_bank_mover=1 sync_demand=0 status=CONFIGURED\n"
            "[REGION_RING_RESOURCE_SCHED] enabled=1 bank_aware=1"
            " phone_push_aware=1 status=READY\n"
            "[REGION_RING_CYCLE_SCHED] mode=DRAIN_FIRST"
            " rule=EARLIEST_EXECUTABLE_REGION status=READY\n"
            "[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT"
            f" tokens=66 kv_start=0 layout={layout}"
            " XC=16 XG=24 XP=48"
            " startup_XC=32 startup_XG=32 startup_XP=32"
            " changed=1 planner_ms=0.400"
            " source=LOCKED_CHUNK_ONLY"
            " layer_placement=IMMUTABLE"
            " execution=THIS_DECODE_ONLY status=PASS\n"
            "[REGION_RING_SUM] tokens=66 XC=0 XG=24 XP=48"
            " regions=7 first_region_chunks=3 tasks=21"
            " executed=21 complete=1 wall_ms=300"
            " mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0\n"
            "[REGION_RING_GPU_BANK_MOVER_SUM] worker=CPU_SINGLE"
            " started=4 completed=4 failures=0 upload_mib=1234"
            " stage_cpu_copy_ms=0 stage_chunks=0"
            " prepare_ms=50 h2d_ms=100 wall_ms=200"
            " gpu_wait_ms=0 thread_started=1 thread_joined=1"
            " status=PASS\n")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text(full, encoding="utf-8")
            decision, _ = probe.read_request_chunk_plan(
                log, 0, 66, 256, 256, layout,
                "LOCKED_CHUNK_ONLY", "DRAIN_FIRST", False, True)
            self.assertEqual(
                (decision["plan_xc"], decision["plan_xg"],
                 decision["plan_xp"]), (16, 24, 48))
            # A replay would silently disable native per-Prompt optimization.
            log.write_text(
                full.replace("source=LOCKED_CHUNK_ONLY",
                             "source=LOCKED_CHUNK_REPLAY"),
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "LOCKED_CHUNK_ONLY", "DRAIN_FIRST", False, True)
            log.write_text(
                full.replace("gpu_prefetch=1", "gpu_prefetch=0"),
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "w/o Multi-Cycle"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "LOCKED_CHUNK_ONLY", "DRAIN_FIRST", False, True)
            log.write_text(
                full.replace("completed=4", "completed=3"),
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "asynchronous"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "LOCKED_CHUNK_ONLY", "DRAIN_FIRST", False, True)
            log.write_text(
                full.replace(
                    "[REGION_RING_CYCLE_SCHED] mode=DRAIN_FIRST",
                    "[REGION_RING_CYCLE_SCHED] mode=FIFO"),
                encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "scheduler ablation"):
                probe.read_request_chunk_plan(
                    log, 0, 66, 256, 256, layout,
                    "LOCKED_CHUNK_ONLY", "DRAIN_FIRST", False, True)

    def test_gpg_canonical_parser_preserves_gpu_window_boundaries(self):
        header = (Path(__file__).resolve().parents[2] /
                  "src" / "llama-region-ring.h").read_text(encoding="utf-8")
        context = (Path(__file__).resolve().parents[2] /
                   "src" / "llama-context.cpp").read_text(encoding="utf-8")
        for token in (
            '"LLAMA_REGION_RING_NO_MULTI_CYCLE_GPG"',
            '"G36,P21,G7"',
            '"G7,G7,G8,G7,G7,P21,G7"',
            "llama_region_ring_gpg_physical_regions",
            "llama_region_ring_gpg_physical_compatible",
            "std::array<int, 7> depths = { 7, 7, 8, 7, 7, 21, 7 }",
        ):
            self.assertIn(token, header)
        for token in (
            "llama_region_ring_gpg_ablation_enabled()",
            "gpu_region_order.size() != 6",
            "!gpu_bank_streaming",
            "!drain_first_cycle",
            '"[REGION_RING_MULTI_CYCLE_ABLATION] mode=G_P_G"',
        ):
            self.assertIn(token, context)

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

    def test_memory_proportional_g40_p24_dag_bypasses_only_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            server = root / "llama-server.exe"
            model = root / "Qwen3-32B-Q4_K_M.gguf"
            server.write_bytes(b"stub")
            model.write_bytes(b"stub")
            args = [
                "--server-exe", str(server), "--model", str(model),
                "--model-layers", "64",
                "--rpc", "127.0.0.1:50052",
                "--prompt-dir", str(root),
                "--profile-source", str(root),
                "--out", str(root / "memory-proportional"),
                "--locked-layout", probe.MEMORY_PROPORTIONAL_LAYOUT,
                "--all50", "--per-request-chunks",
                "--memory-proportional-placement",
            ]
            parsed = probe.parse_args(args)
            self.assertEqual(
                probe.MEMORY_PROPORTIONAL_LAYOUT,
                "G7,P5,G7,P5,G7,P5,G7,P5,G7,P4,G5")
            self.assertTrue(parsed.memory_proportional_placement)
            self.assertTrue(parsed.skip_plan_memory_checks)
            self.assertFalse(parsed.no_dag_placement)
            self.assertIsNone(parsed.fixed_chunk)
            self.assertEqual(
                parsed.locked_layout, probe.MEMORY_PROPORTIONAL_LAYOUT)
            for invalid in (
                    ["--no-dag-placement", "--guarded-memory-checks"],
                    ["--fixed-chunk", "64"],
                    ["--guarded-memory-checks"],
                    ["--no-multi-cycle"],
                    ["--no-drain-first"],
                    ["--no-async-prefetch"],
            ):
                with self.assertRaises(SystemExit):
                    probe.parse_args(args + invalid)
            other = args.copy()
            other[other.index(probe.MEMORY_PROPORTIONAL_LAYOUT)] = (
                probe.DEFAULT_CANONICAL_LAYOUT)
            with self.assertRaises(SystemExit):
                probe.parse_args(other)

    def test_memory_proportional_failure_labels_real_oom_only(self):
        import importlib.util
        module_path = (Path(__file__).resolve().parent / "ablations" /
                       "run_32b_memory_proportional.py")
        spec = importlib.util.spec_from_file_location("run_memory_prop",
                                                       module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            log = out / "llama-server.log"
            log.write_text(
                "CUDA_ERROR_OUT_OF_MEMORY failed to allocate buffer\n",
                encoding="utf-8")
            result = module.classify_failure(out, 1)
            self.assertEqual(result["status"], "OOM")
            self.assertEqual(len(result["oom_evidence"]), 1)
            self.assertTrue((out / "failure.json").is_file())
            log.write_text("unrelated CPU crash\n", encoding="utf-8")
            result = module.classify_failure(out, 1)
            self.assertEqual(result["status"], "CRASH_UNCONFIRMED")
            self.assertEqual(result["oom_evidence"], [])
            self.assertEqual(len(
                (out / "failures.jsonl").read_text(
                    encoding="utf-8").splitlines()), 2)

    def test_original_layout_has_exactly_64_layers(self):
        layers = sum(int(part[1:]) for part in probe.LAYOUT.split(","))
        self.assertEqual(layers, 64)
        self.assertEqual(probe.LAYOUT.split(",")[-1], "G7")


if __name__ == "__main__":
    unittest.main()
