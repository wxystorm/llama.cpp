#!/usr/bin/env python3
"""Host-only regressions for CPU RPC transport and dynamic Plan validation."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_32b_all40 import (
    build_process, get_args, parse_metrics, verify_independent_profile,
    htp_epoch_has_runtime_samples, profile_once_resume_compatible,
)


class HtpOnlyAutoPlanTests(unittest.TestCase):
    def test_htp_only_autoplan_accepts_zero_rolling_profiles(self):
        """Regression: the PowerShell entry point pins --profile-every 0.

        A fresh --out directory must pass CLI validation and use only the
        standalone independent Profile, without a second rolling profile.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "Qwen3-32B-Q4_K_M.gguf"
            exe = root / "llama-completion.exe"
            prompts = root / "prompts"
            out = root / "new-32b-results"
            model.touch()
            exe.touch()
            prompts.mkdir()
            args = get_args([
                "--exe", str(exe),
                "--model", str(model),
                "--rpc", "127.0.0.1:50052",
                "--prompt-dir", str(prompts),
                "--repo-root", str(root),
                "--out", str(out),
                "--auto-plan", "--htp-only",
                "--profile-grid", "quick",
                "--profile-every", "0",
                "--batch", "2048", "--ubatch", "2048",
                "--n-predict", "1", "--threads", "8",
                "--stop-on-fail",
            ])
            self.assertTrue(args.htp_only)
            self.assertTrue(args.auto_plan)
            self.assertEqual(args.profile_every, 0)
            self.assertTrue(args.phone_weight_preload)
            self.assertFalse(args.warmup)
            self.assertEqual(args.out, out)

    def test_diagnostic_trace_requires_explicit_switch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "mistral24b.gguf"
            exe = root / "llama-completion.exe"
            prompts = root / "prompts"
            model.touch()
            exe.touch()
            prompts.mkdir()
            base = [
                "--exe", str(exe), "--model", str(model),
                "--rpc", "127.0.0.1:50052",
                "--model-layers", "40", "--prompt-dir", str(prompts),
                "--repo-root", str(root), "--auto-plan", "--htp-only"
            ]
            default = get_args(base)
            _, default_env = build_process(
                default, prompts / "prompt_128_01.txt", 512, 173,
                root / "default.jsonl")
            self.assertNotIn("LLAMA_REGION_RING_TASK_TRACE", default_env)
            diagnostic = get_args(base + ["--trace-tasks"])
            _, diagnostic_env = build_process(
                diagnostic, prompts / "prompt_128_01.txt", 512, 173,
                root / "diagnostic.jsonl")
            self.assertEqual(diagnostic_env["LLAMA_REGION_RING_TASK_TRACE"], "1")

    def test_mistral_small_40_layer_auto_plan_and_forced_layout(self):
        from run_32b_all40 import (default_layout_for_layers,
                                   normalize_layout)
        self.assertEqual(normalize_layout(default_layout_for_layers(40), 40),
                         "G4,C5,P1,G4,C5,P1,G4,C5,P1,G4,C5,P1")
        self.assertEqual(normalize_layout(
            "G4,C5,P1,G4,C5,P1,G4,C5,P1,G4,C5,P1", 40),
            default_layout_for_layers(40))
        with self.assertRaisesRegex(ValueError, "exactly 40 layers"):
            normalize_layout(default_layout_for_layers(64), 40)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q6_K.gguf"
            exe = root / "llama-completion.exe"
            prompts = root / "prompts"
            model.touch()
            exe.touch()
            prompts.mkdir()
            base = [
                "--exe", str(exe), "--model", str(model),
                "--rpc", "127.0.0.1:50052",
                "--model-layers", "40",
                "--prompt-dir", str(prompts), "--repo-root", str(root),
                "--auto-plan", "--htp-only", "--profile-every", "0",
            ]
            auto = get_args(base)
            self.assertEqual(auto.model_layers, 40)
            self.assertEqual(auto.layout, default_layout_for_layers(40))
            self.assertTrue(auto.phone_weight_preload)
            self.assertFalse(auto.warmup)
            command, env = build_process(
                auto, prompts / "prompt_128_01.txt", 512, 167,
                root / "metrics.jsonl")
            self.assertIn("--no-warmup", command)
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_PLANNER"],
                             "apply-wide")
            forced = get_args(base + [
                "--force-plan-layout", "G4,C5,N1,G4,C5,N1,G4,C5,N1,G4,C5,N1",
                "--force-plan-xc", "64",
                "--force-plan-xg", "64",
                "--force-plan-xp", "64",
            ])
            self.assertEqual(forced.force_plan_layout,
                             default_layout_for_layers(40))
            with self.assertRaises(SystemExit):
                get_args(base + [
                    "--force-plan-layout", default_layout_for_layers(64)])

    def test_fixed_htp_forced_plan_keeps_v0_and_memory_guards(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            model = root / "qwen3-32b.gguf"
            exe = root / "llama-completion.exe"
            prompts = root / "prompts"
            prompts.mkdir()
            model.touch()
            exe.touch()
            layout = "G8,N2,G8,N2,G8,N2,G8,N2,G8,N2,G8,N2,G4"
            base = ["--exe", str(exe), "--model", str(model),
                    "--rpc", "127.0.0.1:50052",
                    "--prompt-dir", str(prompts), "--repo-root", str(root),
                    "--auto-plan", "--htp-only"]
            args = get_args(base + [
                "--force-plan-layout", layout,
                "--force-plan-xc", "64", "--force-plan-xg", "64",
                "--force-plan-xp", "64",
            ])
            self.assertEqual(args.force_plan_layout, layout.replace("N", "P"))
            self.assertEqual((args.force_plan_xc, args.force_plan_xg,
                              args.force_plan_xp), (64, 64, 64))
            self.assertEqual(args.profile_every, 0)
            command, env = build_process(
                args, root / "prompt_128_01.txt", 512, 167,
                root / "metrics.jsonl")
            self.assertIn("--no-warmup", command)
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_PLAN_CONTROLLED"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_HTP_ONLY"], "1")
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_SKIP_PROFILE"], "0")
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT"],
                             args.force_plan_layout)
            for key in ("XC", "XG", "XP"):
                self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_" + key],
                                 "64")
            self.assertNotIn("LLAMA_REGION_RING_REAL_STAGE_PROFILE", env)
            self.assertNotIn("LLAMA_REGION_RING_TASK_TRACE", env)
            with self.assertRaises(SystemExit):
                get_args(base + ["--force-plan-layout", layout,
                                 "--force-plan-xp", "63"])
            with self.assertRaises(SystemExit):
                get_args(base + ["--force-plan-xp", "64"])
            with self.assertRaises(SystemExit):
                get_args(base[:-2] + ["--force-plan-layout", layout])

    def test_cpu_transport_graphless_plan_environment(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = SimpleNamespace(
                auto_plan=True, htp_only=True, profile_every=5,
                profile_grid="full", streaming="on",
                real_stage_profile_csv=root / "independent-profile" / "profile.csv",
                force_small_batch_mover=False, phone_weight_preload=True,
                boost_bank_mover=False, env=[], exe=root / "completion.exe",
                model=root / "qwen.gguf", rpc="127.0.0.1:50052",
                batch=2048, ubatch=2048, n_predict=1, threads=8,
                seed=1234, log_verbosity=4, no_mmap=False,
                layout="G8,P6,G8,P6,G8,P6,G8,P6,G8",
                xc=128, xg=128, xp=128, chunk=128,
            )
            cmd, env = build_process(
                args, root / "prompt_512_01.txt", 1024, 512,
                root / "metrics.jsonl")
            self.assertIn("--no-warmup", cmd)
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_HTP_ONLY"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_PLAN_CONTROLLED"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_GRAPHLESS"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_AUTONOMOUS"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_INPUT_STREAM"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_REAL_STAGE_PROFILE"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_TASK_TRACE"], "1")
            self.assertEqual(env["LLAMA_REGION_RING_PROFILE_GRID"], "full")
            self.assertTrue(env["LLAMA_REGION_RING_REAL_STAGE_PROFILE_CSV"].endswith("profile.csv"))
            self.assertNotIn("LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT", env)
            # V0-only mode must not inherit rolling telemetry settings,
            # including when they remain set in the user's shell.
            args.profile_every = 0
            with patch.dict("os.environ", {
                "LLAMA_REGION_RING_REAL_STAGE_PROFILE": "1",
                "LLAMA_REGION_RING_TASK_TRACE": "1",
            }):
                cmd, once_env = build_process(
                    args, root / "prompt_512_01.txt", 1024, 512,
                    root / "metrics.jsonl")
            self.assertIn("--no-warmup", cmd)
            self.assertNotIn("LLAMA_REGION_RING_REAL_STAGE_PROFILE", once_env)
            self.assertNotIn("LLAMA_REGION_RING_TASK_TRACE", once_env)
            self.assertEqual(
                once_env["LLAMA_REGION_RING_REAL_STAGE_PROFILE_CSV"],
                str(args.real_stage_profile_csv))

    def test_profile5_to_once_resume_accepts_only_policy_change(self):
        once = {
            "version": 4,
            "planning_mode": "independent-backend-profile-then-plan",
            "phone_execution": "CPU_TRANSPORT_HTP_ONLY_PLAN_CONTROLLED",
            "real_stage_profile_csv": "AUTO_INDEPENDENT_G_C_N",
            "model": {"path": "same.gguf", "size": 42},
            "exe": {"path": "completion.exe", "size": 17},
            "prompts": {"prompt_128_01.txt": {"sha256": "abc"}},
            "effective_runtime_env": {
                "LLAMA_REGION_RING_REAL_STAGE_PROFILE_CSV": "profile.csv",
                "LLAMA_REGION_RING_PHONE_PLAN_CONTROLLED": "1",
            },
        }
        rolling = json.loads(json.dumps(once))
        rolling.update(rolling_profile_every=5, rolling_phone_log="")
        rolling["effective_runtime_env"].update({
            "LLAMA_REGION_RING_TASK_TRACE": "1",
            "LLAMA_REGION_RING_REAL_STAGE_PROFILE": "1",
        })
        self.assertTrue(profile_once_resume_compatible(rolling, once))
        self.assertEqual(rolling["rolling_profile_every"], 5)
        for change in (
            {"model": {"path": "different.gguf", "size": 42}},
            {"prompts": {"prompt_128_01.txt": {"sha256": "changed"}}},
            {"real_stage_profile_csv": "other.csv"},
        ):
            mismatched = {**once, **change}
            self.assertFalse(profile_once_resume_compatible(rolling, mismatched))
        unknown_env = json.loads(json.dumps(rolling))
        unknown_env["effective_runtime_env"]["MY_UNRELATED_SETTING"] = "changed"
        self.assertFalse(profile_once_resume_compatible(unknown_env, once))
        self.assertFalse(profile_once_resume_compatible(once, rolling))

    def test_official_entrypoint_pins_profile_every_zero(self):
        script = (Path(__file__).resolve().parent /
                  "run_32b_profile5_plan.ps1").read_text(encoding="utf-8")
        self.assertIn("[int]$ProfileEvery = 0", script)
        self.assertIn('"--auto-plan", "--htp-only", "--profile-every", "0"', script)
        self.assertIn('if ($ProfileEvery -ne 0)', script)
        self.assertIn('[string]$ForceLayout = ""', script)
        self.assertIn('"--force-plan-layout", $ForceLayout', script)

    def test_partial_v1_with_36_independent_npu_rows_must_fallback(self):
        # The failing live experiment had 17 C / 59 G / 36 N rows:
        # 36 N were the unchanged independent baseline. The epoch contains
        # zero actual Phone Stage samples and must not displace a working V0.
        v1 = {
            "device_points": {"CPU": 17, "GPU": 59, "PHONE": 36},
            "diagnostic": {
                "cases": 5, "cpu": 48, "gpu": 96,
                "phone": 0, "phone_missing": 12
            }
        }
        self.assertFalse(htp_epoch_has_runtime_samples(v1))
        v1["diagnostic"]["phone"] = 8
        self.assertTrue(htp_epoch_has_runtime_samples(v1))
        self.assertFalse(htp_epoch_has_runtime_samples(
            {"device_points": {"PHONE": 36}}))

    def test_standalone_profile_cache_checks_all_backends_and_links(self):
        import csv
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            csv_file = root / "independent-profile" / "profile.csv"
            csv_file.parent.mkdir(parents=True)
            fields = ["device", "layers", "tokens", "kv_tokens",
                      "setup_ms", "upload_ms", "prepare_ms",
                      "compute_ms", "terminal", "samples"]
            rows = [
                dict(device=device, layers=2, tokens=32, kv_tokens=256,
                     setup_ms=1, upload_ms=1, prepare_ms=1, compute_ms=4,
                     terminal=0, samples=2)
                for device in ("CPU", "GPU", "PHONE")
            ]
            with csv_file.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fields)
                writer.writeheader()
                writer.writerows(rows)
            doc = {
                "schema": 1,
                "mode": "INDEPENDENT_BACKEND_PROFILE",
                "backends": {"G": "CUDA", "C": "CPU",
                             "N": "HTP_STRIPED"},
                "stage_points": [{"device": row["device"]} for row in rows],
                "transfers": {
                    "pc_to_phone": [{"bytes": 100000, "ms": 1.0}],
                    "snapshot_phone_to_pc": [{"bytes": 100000, "ms": 1.0}],
                    "gpu_weight_h2d": [{"bytes": 100000, "ms": 1.0}],
                },
            }
            meta = csv_file.with_suffix(".json")
            with self.assertRaisesRegex(ValueError, "profile.json missing"):
                verify_independent_profile(csv_file)
            meta.write_text(json.dumps(doc), encoding="utf-8")
            self.assertEqual(verify_independent_profile(csv_file), 1)
            doc["stage_points"] = []
            meta.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match Stage CSV"):
                verify_independent_profile(csv_file)
            doc["stage_points"] = [
                {"device": row["device"]} for row in rows]
            doc["transfers"]["snapshot_phone_to_pc"] = []
            meta.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Phone->PC"):
                verify_independent_profile(csv_file)

    def test_forced_plan_rejection_prefers_native_reason_over_generic_tail(self):
        # The original regression: Python displays FORCE_LAYOUT, but
        # llama.cpp emits a generic HTP-only error after forced rejection.
        # Initial zero HTP counters were printed *before* evaluating it.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "engine.stderr.log").write_text("\n".join((
                "[REGION_RING_HTP_MEMORY_DIAG] geometry=0 session_map=0"
                " bank_shape=0 host_ram=0 gpu_vram=0 phone_ram=0"
                " pass=0 source=CANDIDATE_MEMORY_CALLBACK status=COUNTED",
                "[REGION_RING_PLAN_FORCE_MEMORY] status=REJECT"
                " candidate_memory=0 region_memory=0 geometry=0"
                " session_map=0 bank_shape=0 host_ram=0 gpu_vram=1"
                " phone_ram=0 pass=0 phase=FORCED_PLAN",
                "[REGION_RING_PLAN_FORCE_BANK] status=REJECT"
                " bank_A_peak_mib=2500.0 bank_B_peak_mib=2500.0"
                " bank_A_capacity_mib=2400.0 bank_B_capacity_mib=2400.0",
                "[REGION_RING_PLAN_FORCE_HTP] status=REJECT"
                " reason=SHARED_RAM phone_layers=20 shared_total_mib=9100.0"
                " shared_budget_mib=8078.9",
                "[REGION_RING_PLAN_FORCE] mode=APPLY_WIDE"
                " status=REJECT reason=MEMORY_BUDGET"
                " layout=G9,P5,G9,P5,G9,P5,G9,P5,G8"
                " XC=128 XG=128 XP=128",
                "[REGION_RING_PLAN_APPLY] mode=APPLY_WIDE"
                " phase=MODEL_LOAD applied=0 status=FAIL"
                " reason=NO_EXECUTABLE_HTP_ONLY_GPU_TAIL",
            )), encoding="utf-8")
            row = parse_metrics(
                root, 1, 150.0, auto_plan=True, htp_only=True)
            self.assertEqual(row["status"], "FAIL")
            self.assertEqual(row["plan_reject_reason"], "MEMORY_BUDGET")
            self.assertIn("status=REJECT reason=MEMORY_BUDGET",
                          row["plan_force_reject"])
            self.assertIn("gpu_vram=1", row["plan_force_memory_diag"])
            self.assertIn("bank_A_capacity_mib=2400",
                          row["plan_force_bank_diag"])
            self.assertIn("reason=SHARED_RAM",
                          row["plan_force_htp_diag"])
            self.assertIn("shared_total_mib=9100.0",
                          row["plan_force_htp_diag"])
            self.assertIn("pass=0", row["plan_memory_diag"])

    def test_forced_plan_dag_failure_is_not_misreported_as_gpu_tail(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "engine.stderr.log").write_text("\n".join((
                "[REGION_RING_PLAN_FORCE] mode=APPLY_WIDE status=REJECT"
                " reason=PROFILE_OR_DAG_BALANCE"
                " layout=G9,P5,G9,P5,G9,P5,G9,P5,G8",
                "[REGION_RING_PLAN_APPLY] mode=APPLY_WIDE"
                " applied=0 reason=FORCED_PLAN_REJECTED_SEE_PLAN_FORCE",
            )), encoding="utf-8")
            row = parse_metrics(
                root, 1, 100.0, auto_plan=True, htp_only=True)
            self.assertEqual(row["plan_reject_reason"],
                             "PROFILE_OR_DAG_BALANCE")
            self.assertIsNone(row["plan_force_memory_diag"])

    def test_preload_count_uses_selected_not_control_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            chosen = "G8,P3,G8,P3,G42"
            with (root / "metrics.jsonl").open("w", encoding="utf-8") as f:
                for obj in (
                    {"type": "planner", "mode": "apply-wide", "applied": True,
                     "selected_layout": chosen, "xc": 64, "xg": 96, "xp": 128},
                    {"type": "perf", "prefill_ms": 1000.0,
                     "prefill_tokens": 512, "prefill_tps": 512.0},
                    {"type": "region_ring", "complete": True, "status": 0,
                     "tokens": 512, "layout": chosen},
                ):
                    f.write(json.dumps(obj) + "\n")
            (root / "engine.stderr.log").write_text("\n".join((
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=START status=RUNNING",
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=REGION region=1 status=PASS",
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=REGION region=3 status=PASS",
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=DONE wall_ms=50 status=PASS",
                "[REGION_RING_PHONE_EARLY_WEIGHTS] event=JOIN status=PASS",
            )), encoding="utf-8")
            row = parse_metrics(
                root, 0, 1500.0, auto_plan=True, htp_only=True,
                phone_weight_preload=True,
                fixed_layout="G8,P6,G8,P6,G8,P6,G8,P6,G8")
            self.assertEqual(row["early_weights_status"], "PASS")
            self.assertEqual(row["early_weights_regions"], 2)
            self.assertEqual(row["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
