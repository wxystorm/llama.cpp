"""32B runner contract: true 64-layer layout, memory mapping, Context and resume."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_sharegpt40_14b as shared
import run_sharegpt40_32b as mod


class FakePopen:
    calls: list[tuple[list[str], dict]] = []

    def __init__(self, command, cwd, env, stdin, stdout, stderr):
        self.__class__.calls.append((list(command), dict(env)))

    def wait(self): return 0
    def poll(self): return 0
    def kill(self): pass


def fake_result_row(args, folder, prompt, tokens, context, code, elapsed, commit):
    return {
        "status": "PASS" if code == 0 else "FAIL",
        "prompt_file": prompt.name,
        "prompt_sha256": hashlib.sha256(prompt.read_bytes()).hexdigest(),
        "actual_prefill_tokens": tokens,
        "actual_decode_tokens": 1,
        "prefill_ms": 1000.0,
        "prefill_tps": tokens,
        "decode_ms": None,
        "decode_tps": None,
        "commit": commit,
        "exit_code": code,
        "run_dir": str(folder),
    }


class Runner32BTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.exe = self.root / "llama-completion.exe"
        self.exe.touch()
        self.model = self.root / "Qwen3-32B-Q4_K_M.gguf"
        self.model.write_bytes(b"dummy model for runner tests")
        self.prompts = self.root / "sharegpt40"
        self.prompts.mkdir()
        for filename in ("prompt_128_01.txt",
                         "prompt_512_01.txt",
                         "prompt_1024_01.txt"):
            (self.prompts / filename).write_text(filename)
        (self.prompts / "manifest.csv").write_text(
            "file,actual_tokens\n"
            "prompt_128_01.txt,167\n"
            "prompt_512_01.txt,1013\n"
            "prompt_1024_01.txt,1855\n",
            encoding="utf-8")
        self.out = self.root / "results"
        self.args = [
            "--exe", str(self.exe), "--model", str(self.model),
            "--rpc", "local:50052", "--prompt-dir", str(self.prompts),
            "--out", str(self.out), "--token-source", "manifest",
            "--delay", "0",
        ]
        FakePopen.calls.clear()
        self.ps = [
            patch.object(shared.subprocess, "Popen", FakePopen),
            patch.object(shared, "git_head", return_value="test-commit"),
            patch.object(shared, "result_row", side_effect=fake_result_row),
        ]
        for p in self.ps:
            p.start()
            self.addCleanup(p.stop)

    def test_layout_is_64_layers_and_bank_peaks_are_8_9(self):
        self.assertEqual(mod.LAYERS, 64)
        parts = mod.LAYOUT.split(",")
        self.assertEqual(len(parts), 13)
        self.assertEqual(sum(int(x[1:]) for x in parts), 64)
        self.assertEqual(sum(int(x[1:]) for x in parts[::2]), 58)
        self.assertEqual(sum(int(x[1:]) for x in parts[1::2]), 6)
        gpu_regions = [int(x[1:]) for x in parts[::2]]
        self.assertEqual(max(gpu_regions[::2]), 8)
        self.assertEqual(max(gpu_regions[1::2]), 9)
        self.assertEqual(shared.validate_layout(mod.LAYOUT, 64), mod.LAYOUT)
        with self.assertRaisesRegex(ValueError, "64"):
            shared.validate_layout(shared.LAYOUT, 64)

    def test_default_context_mmap_and_no_profile(self):
        self.assertEqual(mod.main(self.args), 0)
        self.assertEqual(len(FakePopen.calls), 3)
        ctxs = {}
        for cmd, env in FakePopen.calls:
            name = Path(cmd[cmd.index("-f") + 1]).name
            ctxs[name] = int(cmd[cmd.index("-c") + 1])
            self.assertEqual(cmd[cmd.index("-n") + 1], "1")
            self.assertNotIn("--no-mmap", cmd)
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT"],
                             mod.LAYOUT)
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_XC"], "128")
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_XG"], "128")
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_XP"], "128")
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_SKIP_PROFILE"], "1")
            self.assertEqual(env["LLAMA_HYBRID_REGION_RING_GPU_WEIGHT_STREAM"], "1")
        self.assertEqual(ctxs, {
            "prompt_128_01.txt": 1024,
            "prompt_512_01.txt": 1280,
            "prompt_1024_01.txt": 2048,
        })
        config = json.loads((self.out / "run-config.json").read_text())
        self.assertEqual(config["model_layers"], 64)
        self.assertEqual(config["chunk_planning"], "FIXED_MANUAL_NO_PROFILE")
        self.assertFalse(config["no_mmap"])
        self.assertEqual(config["n_predict"], 1)
        self.assertFalse((self.out / "profile").exists())
        plan = json.loads((self.out / "prompt_1024_01" / "plan.json").read_text())
        self.assertEqual(plan["context"], 2048)
        self.assertEqual(plan["plan_tokens"], 1855)
        self.assertEqual(plan["layout_fixed"], mod.LAYOUT)

    def test_resume_and_manual_retries(self):
        self.assertEqual(mod.main(self.args + ["--limit", "1"]), 0)
        self.assertEqual(len(FakePopen.calls), 1)
        self.assertEqual(mod.main(self.args), 0)
        self.assertEqual(len(FakePopen.calls), 3)
        self.assertEqual(mod.main(self.args), 0)
        self.assertEqual(len(FakePopen.calls), 3)
        self.assertEqual(mod.main(self.args + [
            "--rerun", "prompt_1024_01"]), 0)
        self.assertEqual(len(FakePopen.calls), 4)
        archived = (self.out / "history" / "prompt_1024_01" /
                    "attempt-001" / "result.json")
        self.assertTrue(archived.is_file())

    def test_per_prompt_chunk_override(self):
        path = self.root / "chunks.json"
        path.write_text(json.dumps({
            "groups": {"1024": {"xp": 96}},
            "prompts": {"prompt_1024_01": {"xg": 112}}
        }))
        self.assertEqual(mod.main(self.args + [
            "--only", "prompt_1024_01",
            "--chunk-overrides", str(path)]), 0)
        self.assertEqual(len(FakePopen.calls), 1)
        cmd, env = FakePopen.calls[0]
        self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_XG"], "112")
        self.assertEqual(env["LLAMA_HYBRID_REGION_RING_FORCE_XP"], "96")
        self.assertEqual(cmd[cmd.index("-c") + 1], "2048")
        self.assertFalse((self.out / "profile").exists())

    def test_no_mmap_optin_and_context_guard(self):
        self.assertEqual(mod.main(self.args + [
            "--no-mmap", "--only", "prompt_128_01"]), 0)
        self.assertIn("--no-mmap", FakePopen.calls[0][0])
        config = json.loads((self.out / "run-config.json").read_text())
        self.assertTrue(config["no_mmap"])
        with self.assertRaises(SystemExit):
            mod.main(self.args)
        self.assertEqual(len(FakePopen.calls), 1)

    def test_rejects_unsafe_context_and_wrong_layout(self):
        with self.assertRaises(SystemExit):
            mod.main(self.args + [
                "--only", "prompt_1024_01", "--context", "1024"])
        with self.assertRaises(SystemExit):
            mod.main(self.args + [
                "--layout", shared.LAYOUT])
        self.assertFalse(FakePopen.calls)

    def test_gpu_memory_summary_from_real_log_shapes(self):
        folder = self.root / "log"
        folder.mkdir()
        (folder / "engine.stderr.log").write_text(
            "llama_kv_cache: CUDA0 KV buffer size = 464.00 MiB\n"
            "[REGION_RING_GPU_BANK_PEAK_SUM] phase=LOAD_TIME banks=2 "
            "enabled=1 future_graphs=5 prepare_ms=100.0 "
            "capacity_a_mib=2400.500 capacity_b_mib=2699.750 "
            "status=READY\n", encoding="utf-8")
        row = shared.gpu_memory_from_log(folder)
        self.assertEqual(row["cuda_kv_mib"], 464.0)
        self.assertEqual(row["gpu_bank_a_mib"], 2400.5)
        self.assertEqual(row["gpu_bank_b_mib"], 2699.75)
        self.assertEqual(row["gpu_banks_plus_kv_mib"], 5564.25)

    def test_profile_and_depth_override_rejected(self):
        for flags in (["--profile-chunks"], ["--update-profile"],
                      ["--model-layers", "40"]):
            with self.assertRaises(SystemExit):
                mod.main(self.args + flags)
        self.assertFalse(FakePopen.calls)


if __name__ == "__main__":
    unittest.main()
