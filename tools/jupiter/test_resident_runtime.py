"""Model-free checks for strict resident request mapping and log isolation."""
import tempfile
import json
import unittest
import urllib.error
from unittest.mock import patch
from pathlib import Path
from resident_runtime import (request_payload, request_log, run_resident_cli,
                              is_request_timeout, timeout_marker, PREFIX)


class TestResident(unittest.TestCase):
    def setup_case(self, root):
        config = dict(model=str(root / "model.gguf"), rpc="127.0.0.1:10140",
                      context=2048, batch=2048, ubatch=128, threads=8,
                      cpu_layers=8, pc_layers=28, sessions=4, native=True, window=2)
        cmd = ["cli", "-m", config["model"], "--rpc", config["rpc"],
               "--split-mode", "layer", "-ngl", "all", "-ub", "95",
               "-c", "2048", "-b", "2048", "-t", "8", "-n", "1",
               "--flash-attn", "on", "-lv", "4", "--no-conversation",
               "--single-turn", "--simple-io", "-p", "A prompt"]
        env = {"LLAMA_JUPITER_SEQUENCE_PLAN": "94,95,77",
               "LLAMA_JUPITER_PHONE_OWNED_NATIVE": "1",
               "LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS": "1",
               "LLAMA_JUPITER_WINDOW": "2",
               "LLAMA_JUPITER_LOGITS_DUMP_FILE": str(root / "logits.f32")}
        return cmd, env, config

    def test_exact_plan_and_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd, env, config = self.setup_case(Path(tmp))
            payload = request_payload(cmd, env, config, "a"*32)
            self.assertEqual(payload["chunks"], [94, 95, 77])
            self.assertEqual(payload["expected_tokens"], 266)
            self.assertTrue(payload["autonomous"])
            self.assertTrue(payload["return_logits"])
            env["LLAMA_JUPITER_PC_LAYERS"] = "20"
            with self.assertRaises(ValueError):
                request_payload(cmd, env, config, "b"*32)
            env.pop("LLAMA_JUPITER_PC_LAYERS")
            env["LLAMA_JUPITER_SEQUENCE_PLAN"] = "96,95,77"
            with self.assertRaises(ValueError):
                request_payload(cmd, env, config, "b"*32)

    def test_legacy_timeout_stops_child_and_preserves_partial_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cmd, env, config = self.setup_case(root)
            config.update(native=False, window=1)
            env.update(LLAMA_JUPITER_PHONE_OWNED_NATIVE="0",
                       LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS="0",
                       LLAMA_JUPITER_WINDOW="1")
            env.pop("LLAMA_JUPITER_LOGITS_DUMP_FILE")
            cmd[cmd.index("-ub") + 1] = "128"
            server_log, sample = root / "server.log", root / "sample.log"
            server_log.write_text("startup\n", encoding="utf-8")
            env.update({PREFIX + "URL": "http://127.0.0.1:1", PREFIX + "KEY": "test",
                        PREFIX + "CONFIG": json.dumps(config), PREFIX + "LOG": str(server_log),
                        PREFIX + "TIMEOUT": "120", PREFIX + "RECOVER_TIMEOUT": "1"})

            def timeout(url, key, payload, seconds):
                with server_log.open("a", encoding="utf-8") as stream:
                    stream.write("partial sequence plan, no completion\n")
                raise TimeoutError("timed out")

            with patch("resident_runtime.http_json", side_effect=timeout):
                with self.assertRaises(SystemExit) as raised:
                    run_resident_cli(cmd, env, sample)
            self.assertEqual(raised.exception.code, 75)
            info = json.loads(timeout_marker(server_log).read_text(encoding="utf-8"))
            self.assertEqual(info["reason"], "HTTP_REQUEST_TIMEOUT")
            self.assertIn("partial sequence plan", Path(info["partial_trace"]).read_text())
            self.assertNotIn("partial sequence plan", sample.read_text())
            self.assertTrue(is_request_timeout(urllib.error.URLError(TimeoutError())))
            self.assertFalse(is_request_timeout(ValueError("timed out")))

    def test_native_calibration_probe_and_exact_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd, env, config = self.setup_case(Path(tmp))
            config["calibration"] = True
            env[PREFIX + "NATIVE_CALIBRATION"] = "1"
            env.pop("LLAMA_JUPITER_LOGITS_DUMP_FILE")
            env.pop("LLAMA_JUPITER_SEQUENCE_PLAN")
            payload = request_payload(cmd, env, config, "a"*32)
            self.assertTrue(payload["calibration"])
            self.assertTrue(payload["autonomous"])
            self.assertEqual(payload["chunks"], [])
            env["LLAMA_JUPITER_PROFILE_STAGE"] = "1"
            with self.assertRaises(ValueError):
                request_payload(cmd, env, config, "a"*32)
            env["LLAMA_JUPITER_SEQUENCE_PLAN"] = "94,95,77"
            payload = request_payload(cmd, env, config, "a"*32)
            self.assertEqual(payload["chunks"], [94,95,77])
            env["LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS"] = "0"
            with self.assertRaises(ValueError):
                request_payload(cmd, env, config, "a"*32)

    def test_raw_prefix_only_for_profile_and_never_formal(self):
        with tempfile.TemporaryDirectory() as tmp:
            cmd, env, config = self.setup_case(Path(tmp))
            config["prompt_mode"] = "RAW"
            payload = request_payload(cmd, env, config, "a"*32)
            self.assertEqual(payload["prompt_mode"], "RAW")
            self.assertEqual(payload["raw_prefix_tokens"], 0)
            env[PREFIX + "RAW_PREFIX_TOKENS"] = "266"
            with self.assertRaises(ValueError):
                request_payload(cmd, env, config, "a"*32)
            config["calibration"] = True
            env[PREFIX + "NATIVE_CALIBRATION"] = "1"
            env.pop("LLAMA_JUPITER_LOGITS_DUMP_FILE")
            env["LLAMA_JUPITER_PROFILE_STAGE"] = "1"
            payload = request_payload(cmd, env, config, "a"*32)
            self.assertEqual(payload["raw_prefix_tokens"], 266)
            env[PREFIX + "RAW_PREFIX_TOKENS"] = "265"
            with self.assertRaises(ValueError):
                request_payload(cmd, env, config, "a"*32)
            env[PREFIX + "RAW_PREFIX_TOKENS"] = "266"
            config["prompt_mode"] = "CHAT"
            with self.assertRaises(ValueError):
                request_payload(cmd, env, config, "a"*32)

    def test_log_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "server.log"
            a, b = "a"*32, "b"*32
            p.write_text(
                f"[JUPITER_RESIDENT] phase=BEGIN id={a} status=RUN\nold\n"
                f"[JUPITER_RESIDENT] phase=END id={a} status=PASS\n"
                f"[JUPITER_RESIDENT] phase=BEGIN id={b} status=RUN\nnew\n"
                f"[JUPITER_RESIDENT] phase=END id={b} status=PASS\n",
                encoding="utf-8")
            self.assertIn(b"new", request_log(p, 0, b))
            self.assertNotIn(b"old", request_log(p, 0, b))
            with self.assertRaises(RuntimeError):
                request_log(p, 0, "c"*32)


if __name__ == "__main__":
    unittest.main()
