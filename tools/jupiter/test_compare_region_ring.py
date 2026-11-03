"""Offline checks for comparing Jupiter/Region Ring results without reruns."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import json

from compare_region_ring import compare, per_prompt


class JupiterRegionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = self.root / "prompt_358"
        self.case.mkdir()
        self.config = {
            "prompt_sha256": "abc123",
            "model": {
                "path": r"E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
                "bytes": 8000000000,
            },
            "threads": 8, "batch": 512,
            "ubatch": 128, "context": 2048,
        }
        self.write_config()
        self.j = [{
            "prompt": r"D:\prima.cpp\prompts\prompt_358.txt",
            "status": "PASS",
            "case_dir": str(self.case),
            "actual_tokens": "358",
            "pc_layers": "20",
            "default_median_ms": "7800",
            "dp_median_ms": "9200",
            "default_tok_s": "45.9",
            "dp_tok_s": "38.9",
            "sample_count": "3",
        }]
        self.rr = [{
            "prompt_file": "prompt_358.txt",
            "status": "PASS",
            "actual_prefill_tokens": "358",
            "prefill_ms": "2300",
            "prefill_tps": "155.65",
            "model": r"E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
            "model_size_bytes": "8000000000",
            "prompt_sha256": "abc123",
            "runtime_layout": "G10,C1,P3,G11,C1,P1,G11,C1,P1",
            "gpu_layers": "32",
            "cpu_layers": "3",
            "phone_layers": "5",
            "batch": "512",
            "ubatch": "128",
            "context": "2048",
            "threads": "8",
            "n_predict": "1",
        }]

    def write_config(self):
        (self.case / "configuration.json").write_text(
            json.dumps(self.config), encoding="utf-8"
        )

    def test_identical_input_and_benchmark_configuration(self):
        row = compare(self.j, self.rr)[0]
        self.assertEqual(row["status"], "MATCHED_STRICT_PARAMS")
        self.assertEqual(row["strict_comparable"], 1)
        self.assertAlmostEqual(row["region_vs_jupiter_dp"], 4.0)
        self.assertAlmostEqual(
            row["region_vs_jupiter_default"], round(7800 / 2300, 4)
        )
        self.assertEqual(row["jupiter_pc_layers"], "20")

    def test_different_prompt_bytes_never_get_ratio(self):
        self.rr[0]["prompt_sha256"] = "other"
        row = compare(self.j, self.rr)[0]
        self.assertEqual(row["status"], "PROMPT_BYTES_DIFFER")
        self.assertNotIn("region_vs_jupiter_dp", row)

    def test_different_actual_token_count_never_get_ratio(self):
        self.rr[0]["actual_prefill_tokens"] = "357"
        row = compare(self.j, self.rr)[0]
        self.assertEqual(row["status"], "TOKEN_COUNT_DIFFERS")
        self.assertNotIn("region_vs_jupiter_dp", row)

    def test_different_model_rejected(self):
        self.rr[0]["model_size_bytes"] = "7999999999"
        row = compare(self.j, self.rr)[0]
        self.assertEqual(row["status"], "MODEL_FILENAME_OR_SIZE_DIFFERS")

    def test_batch_ubatch_context_predict_differences_are_flagged(self):
        self.rr[0]["ubatch"] = "512"
        self.rr[0]["n_predict"] = "32"
        row = compare(self.j, self.rr)[0]
        self.assertEqual(row["status"], "MATCHED_INPUTS_CONFIG_DIFF")
        self.assertEqual(row["strict_comparable"], 0)
        self.assertIn("ubatch:128/512", row["parameter_differences"])
        self.assertIn("n_predict:1/32", row["parameter_differences"])
        self.assertEqual(row["region_vs_jupiter_dp"], 4.0)

    def test_missing_or_incomplete_results_are_not_claimed(self):
        self.assertEqual(compare(self.j, [])[0]["status"], "NO_REGION_RESULT")
        self.assertEqual(compare([], self.rr)[0]["status"], "NO_JUPITER_RESULT")
        self.rr[0]["status"] = "INCOMPLETE_RING"
        self.assertEqual(compare(self.j, self.rr)[0]["status"], "REGION_NOT_PASS")

    def test_duplicate_run_is_rejected_instead_of_cherry_picked(self):
        with self.assertRaisesRegex(ValueError, "duplicate prompt"):
            per_prompt(self.rr + self.rr, "prompt_file", "Region Ring")

    def test_numeric_prompt_order(self):
        more = dict(self.rr[0])
        more["prompt_file"] = "prompt_64.txt"
        data = compare([], [self.rr[0], more])
        self.assertEqual(
            [row["prompt"] for row in data],
            ["prompt_64.txt", "prompt_358.txt"],
        )


if __name__ == "__main__":
    unittest.main()
