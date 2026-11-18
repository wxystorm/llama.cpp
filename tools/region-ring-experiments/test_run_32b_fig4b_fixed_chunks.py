"""Offline tests for Fig.4(b) resident no-mmap Full Cycle parity runner."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import run_32b_fig4b_fixed_chunks as fig
import probe_htp_llama_server as server


class Fig4bTests(unittest.TestCase):
    def args(self, root: Path, repeats=2):
        return fig.parse_args([
            "--server-exe", str(root / "llama-server.exe"),
            "--model", str(root / "model.gguf"),
            "--rpc", "127.0.0.1:50052",
            "--prompt-dir", str(root / "prompts"),
            "--profile-source", str(root / "profile"),
            "--out", str(root / "results"),
            "--repeats", str(repeats),
        ])

    def test_server_flags_match_main_and_enforce_no_mmap(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d))
            self.assertEqual(args.layout, server.DEFAULT_CANONICAL_LAYOUT)
            self.assertEqual(args.layout, "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7")
            self.assertEqual(tuple(args.chunks), (64, 128, 256))
            cmd = fig.runner_command(args, 64, 1)
            self.assertEqual(cmd[cmd.index("--fixed-chunk") + 1], "64")
            self.assertIn("--no-mmap", cmd)
            self.assertIn("--all50", cmd)
            self.assertIn("--per-request-chunks", cmd)
            self.assertNotIn("--no-async-prefetch", cmd)
            self.assertNotIn("--no-drain-first", cmd)
            i = cmd.index("--fig4b-prompts")
            self.assertEqual(cmd[i + 1:i + 4],
                             ["prompt_1024_01.txt",
                              "prompt_128_01.txt", "prompt_512_01.txt"])
            resident = server.server_command(
                args.server_exe, args.model, args.rpc, args.port, no_mmap=True)
            self.assertIn("--no-mmap", resident)
            self.assertIn("--no-cache-prompt", resident)
            self.assertIn("--no-warmup", resident)
            self.assertEqual(resident[resident.index("-c") + 1], "2048")

    def populate(self, args, bad=False):
        synthetic = {
            128: {64: 100, 128: 105, 256: 120},
            512: {64: 210, 128: 190, 256: 205},
            1024: {64: 390, 128: 360, 256: 310},
        }
        for chunk in args.chunks:
            for repeat in range(1, args.repeats + 1):
                folder = fig.case_dir(args.out, chunk, repeat)
                folder.mkdir(parents=True)
                env = {
                    "LLAMA_HYBRID_REGION_RING_LOCKED_LAYOUT": args.layout,
                    "LLAMA_REGION_RING_FIXED_CHUNK_ABLATION": str(chunk),
                    "LLAMA_REGION_RING_GPU_BANK_FORCE_SMALL_BATCH": "1",
                    "LLAMA_REGION_RING_GPU_BANK_COUNT": "2",
                    "LLAMA_REGION_RING_GPU_BANK_PREFETCH": "1",
                    "LLAMA_REGION_RING_GPU_BANK_SYNC_DEMAND": "0",
                    "LLAMA_REGION_RING_DRAIN_FIRST_CYCLE": "1",
                    "LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED": "1",
                    "LLAMA_REGION_RING_PER_REQUEST_CHUNKS": "1",
                    "LLAMA_REGION_RING_LOCKED_SHORT_CHUNKS": "1",
                }
                cmd = server.server_command(
                    args.server_exe, args.model, args.rpc, args.port,
                    no_mmap=True)
                manifest = {"configuration": {
                    "runtime_env": env,
                    "server_command": cmd,
                    "prompts": [fig.prompt_name(g, args.prompt_index)
                                for g in (1024, 128, 512)],
                }}
                (folder / "server-run-config.json").write_text(
                    json.dumps(manifest), encoding="utf-8")
                (folder / "requests.jsonl").write_text("{}\n", encoding="utf-8")
                records = []
                for group in args.groups:
                    actual_width = min(chunk, group + 15)
                    records.append({
                        "prompt": fig.prompt_name(group, args.prompt_index),
                        "group": group, "status": "PASS", "cache_n": 0,
                        "prompt_n": group + 15, "prompt_ms": synthetic[group][chunk] + repeat - 1,
                        "prefill_tps": 100.0,
                        "runtime_layout": args.layout,
                        "planner_mode": f"FIXED_{chunk}_NO_REQUEST_SEARCH",
                        "plan_xc": actual_width, "plan_xg": actual_width,
                        "plan_xp": actual_width,
                        # A GPU/Phone-only runtime can report XC=0 because
                        # no CPU compute stage exists in the layer placement.
                        "runtime_xc": 0,
                        "runtime_xg": (0 if bad and chunk == 128 and group == 512
                                       else actual_width),
                        "runtime_xp": actual_width, "planner_ms": 0.1,
                        "scheduler_mode": "DRAIN_FIRST",
                        "prefetch_mode": ("SYNC_DEMAND" if bad and chunk == 128
                                          and group == 512 else "ASYNC_DOUBLE_BANK"),
                    })
                with (folder / "requests.csv").open(
                        "w", encoding="utf-8-sig", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)

    def test_normalized_curve_uses_real_native_server_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d))
            self.populate(args)
            passed, total, complete = fig.aggregate(args)
            self.assertEqual((passed, total, complete), (18, 18, True))
            with (args.out / "fig4b_plot_normalized.csv").open(
                    encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3)
            winners = {int(row["input_group"]):
                       min(args.chunks, key=lambda c: float(row[f"chunk_{c}"]))
                       for row in rows}
            self.assertEqual(winners, {128: 64, 512: 128, 1024: 256})

    def test_256_is_one_full_prompt_chunk_for_short_input(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d))
            self.populate(args)
            records = fig.read_case(args, 256, 1)
            short = next(r for r in records if r["input_group"] == 128)
            # The synthetic 128-group prompt is 143 tokens, so target 256
            # correctly becomes one 143-token chunk, not an invalid >N width.
            self.assertEqual(int(short["prefill_tokens"]), 143)
            self.assertEqual(int(short["plan_xg"]), 143)
            self.assertEqual(int(short["runtime_xg"]), 143)
            self.assertEqual(short["status"], "PASS")

    def test_old_output_dir_with_chunk32_requires_new_out(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "results" / "chunk_032").mkdir(parents=True)
            with self.assertRaises(SystemExit):
                self.args(root)

    def test_unstarted_cases_are_pending_not_false_failures(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d), repeats=3)
            rows = fig.read_case(args, 256, 3)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r["status"] == "PENDING" for r in rows))
            self.assertTrue(all(not r["audit_error"] for r in rows))

    def test_missing_gpu_width_still_rejects_real_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d))
            self.populate(args)
            target = fig.case_dir(args.out, 64, 1) / "requests.csv"
            with target.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            records[0]["runtime_xg"] = "0"
            with target.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=records[0].keys())
                writer.writeheader()
                writer.writerows(records)
            result = fig.read_case(args, 64, 1)
            self.assertEqual(result[0]["status"], "AUDIT_FAIL")
            self.assertIn("runtime_xg", result[0]["audit_error"])
            self.assertEqual(result[1]["status"], "PASS")

    def test_wrong_server_prefetch_policy_rejects_samples(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(Path(d))
            self.populate(args, bad=True)
            passed, total, complete = fig.aggregate(args)
            self.assertEqual((passed, total, complete), (16, 18, False))
            with (args.out / "fig4b_plot_normalized.csv").open(
                    encoding="utf-8-sig", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 0)


if __name__ == "__main__":
    unittest.main()
