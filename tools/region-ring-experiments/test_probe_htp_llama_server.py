#!/usr/bin/env python3
"""Offline tests for the persistent HTP-only llama-server smoke harness."""
from io import BytesIO
from pathlib import Path
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
                "[REGION_RING_WORKER] event=START\\n"
                "[PHONE_INPUT_PUSH] event=FAIL region=7 chunk=0"
                " reason=RPC_INPUT_STREAM_PUSH_REJECTED\\n"
                "[P2C_STREAM] phase=STOP recv=174 pending=116 status=FAIL\\n"
                "srv decode: Compute error.\\n",
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
        import re
        layers = sum(int(n) for _, n in
                     re.findall(r"([GNP]),?(\\d+)", probe.LAYOUT))
        # Recheck without depending on comma parsing.
        layers = sum(int(part[1:]) for part in probe.LAYOUT.split(","))
        self.assertEqual(layers, 64)
        self.assertEqual(probe.LAYOUT.split(",")[-1], "G7")


if __name__ == "__main__":
    unittest.main()
