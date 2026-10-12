#!/usr/bin/env python3
"""Self-contained numerical regression tests for Hexagon focus-layer diagnostics.

No HTP, RPC, llama-cli, or GGUF model required. Run from repository root:
    py -3 scripts/test_hexagon_linear_focus.py
"""
from contextlib import redirect_stdout
from io import StringIO
import csv
from pathlib import Path
import struct
import tempfile
import unittest

from summarize_hexagon_linear_layers import compare, resolve_target

LAYERS = (13, 14, 15)
STAGES = ("attn_out", "ffn_out", "l_out")
HEADER = ("id", "name", "ne0", "ne1", "ne2", "ne3",
          "type", "op", "file", "bytes")
BASE = (1.0, 2.0, 3.0)
CHANGED = (2.0, 3.0, 4.0)
SLIGHT = (1.01, 2.01, 3.01)


def write_snapshot_directory(folder, changed=None, omit=None):
    """Create valid manifest.tsv + binary snapshots for two successive steps."""
    changed = changed or {}
    omit = omit or set()
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, delimiter="\t")
        writer.writeheader()
        capture = 0
        for step in range(2):
            for layer in LAYERS:
                for stage in STAGES:
                    name = "%s-%d" % (stage, layer)
                    if name in omit:
                        continue
                    filename = "tensor_%03d.bin" % capture
                    values = changed.get((step, name), BASE)
                    raw = struct.pack("<3f", *values)
                    (folder / filename).write_bytes(raw)
                    writer.writerow({
                        "id": capture, "name": name,
                        "ne0": 3, "ne1": 1, "ne2": 1, "ne3": 1,
                        "type": "f32", "op": "MUL_MAT",
                        "file": filename, "bytes": len(raw),
                    })
                    capture += 1


class FocusComparatorTests(unittest.TestCase):
    def check(self, case, change=None, expected="NUMERICALLY_CLOSE",
              omit_both=None, focus=None):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            ref, npu, out = p / "base", p / "npu", p / "compare"
            write_snapshot_directory(ref, omit=omit_both)
            write_snapshot_directory(npu, changed=change, omit=omit_both)
            with redirect_stdout(StringIO()):
                result = compare(ref, npu, case, out, .001, .001, .95,
                                 .05, list(LAYERS), focus)
            self.assertEqual(result["status"], expected)
            self.assertTrue((out / "tensor_metrics.csv").exists())
            self.assertTrue((out / "summary.json").exists())
            return result

    def test_small_target_difference_passes(self):
        r = self.check("15-output", {(0, "attn_out-15"): SLIGHT})
        self.assertEqual(r["region_counts"]["target"]["diff"], 1)

    def test_target_large_difference_is_critical(self):
        r = self.check("15-output", {(0, "attn_out-15"): CHANGED},
                       "TARGET_CRITICAL")
        self.assertEqual(r["region_counts"]["target"]["critical"], 1)

    def test_before_first_target_is_upstream(self):
        r = self.check("15-output", {(0, "attn_out-13"): CHANGED},
                       "UPSTREAM_DIVERGENCE")
        self.assertEqual(r["first_upstream_diff"]["name"], "attn_out-13")

    def test_after_first_target_can_be_later_decode(self):
        # Same lower layer (13) but in step 2, after 15-output in step 1:
        # this is temporally downstream, possibly via a sampled token.
        r = self.check("15-output", {(1, "attn_out-13"): CHANGED},
                       "DOWNSTREAM_CRITICAL")
        self.assertEqual(r["region_counts"]["upstream"]["diff"], 0)
        self.assertEqual(r["region_counts"]["downstream"]["critical"], 1)

    def test_omitted_stage_both_sides_is_capture_mismatch(self):
        r = self.check("15-output", expected="CAPTURE_MISMATCH",
                       omit_both={"ffn_out-14"})
        self.assertEqual(r["only_cpu"], 0)
        self.assertEqual(r["only_npu"], 0)
        self.assertIn("ffn_out-14", r["missing_stages"])

    def test_identical_repeat_is_clean(self):
        r = self.check("baseline-control")
        self.assertEqual(r["region_counts"]["upstream"]["diff"], 0)

    def test_repeat_difference_is_run_drift(self):
        r = self.check("baseline-control", {(1, "attn_out-15"): SLIGHT},
                       "UPSTREAM_DIVERGENCE")
        self.assertEqual(r["region_counts"]["upstream"]["diff"], 1)

    def test_invalid_focus_layer_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_target("15-output", list(LAYERS), 13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
