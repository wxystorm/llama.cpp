"""Model-free checks for strict resident request mapping and log isolation."""
import tempfile
import unittest
from pathlib import Path
from resident_runtime import request_payload, request_log


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
