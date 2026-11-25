"""Model-free tests for native autonomous calibration attribution."""
import os
import unittest
from unittest.mock import patch

from native_calibration import validate_run, profile_rows, PHONE, ENV
from jupiter_planner import QProfile
from run_q_sweep import print_sweep_plans

def sample():
    lines = [
        "[JUPITER_NATIVE_CALIBRATION] mode=PHONE_AUTONOMOUS profile=1",
        "[JUPITER_SEQUENCE_PLAN] mode=PAPER_DP total=192 k=2 chunks=96,96",
        "[JUPITER_PHONE_OWNED_PREFILL] phase=PREPARE_BEGIN run=1 tokens=192 chunk=96 sessions=4 ctx=2048 first_layer=52 layers=12",
        "[JUPITER_PHONE_OWNED_PREFILL] phase=PHONE_AUTO_START run=1",
    ]
    for i in range(2):
        lines += [
            f"[JUPITER_UBATCH] event=BEGIN chunk={i} slot={i} prefix_tokens={i*96} tokens=96",
            "[JUPITER_STAGE_PROFILE] split=0 backend=CPU nodes=2 inputs=1 input_ms=1.000 compute_ms=10.000 wall_ms=11.000 native_bridge=0",
            "[JUPITER_STAGE_PROFILE] split=1 backend=CUDA0 nodes=2 inputs=1 input_ms=2.000 compute_ms=8.000 wall_ms=10.000 native_bridge=0",
            f"[JUPITER_PHONE_OWNED_PREFILL] phase=PHONE_AUTO_PUSH run=1 chunk={i} tokens=96",
        ]
        if i == 1:
            for j in range(2):
                lines.append(f"[JUPITER_NATIVE_PHONE_PROFILE] run=1 chunk={j} prefix={j*96} tokens=96 layers=12 sessions=4 upload_ms=2.000000 htp_ms=5.000000 copy_ms=1.000000 wall_ms=9.000000")
            lines.append("[JUPITER_PHONE_OWNED_PREFILL] phase=PHONE_AUTO_P2C_DONE run=1 output=REAL_HIDDEN status=PASS")
        lines += [
            f"[JUPITER_NATIVE_BRIDGE_PROFILE] chunk={i} tokens=96 bridge_ms=100.000000",
            "[JUPITER_STAGE_PROFILE] split=2 backend=CPU nodes=2 inputs=1 input_ms=1.000 compute_ms=101.000 wall_ms=102.000 native_bridge=1",
            f"[JUPITER_UBATCH] event=QUEUED chunk={i} slot={i} prefix_tokens={i*96} tokens=96",
        ]
    lines.append("[JUPITER_PHONE_OWNED_PREFILL] phase=COMPLETE tokens=192 chunks=2 output=REAL_LOGITS final_head=PC_CUDA status=PASS")
    return "\n".join(lines)

class TestNativeCalibration(unittest.TestCase):
    def test_profile_and_dp(self):
        rows = profile_rows(sample(), require_cpu=True)
        self.assertEqual(len(rows), 6)
        first = {r["device"]: r for r in rows if r["y"] == 0}
        self.assertEqual(first["CPU"]["compute_ms"], 11)
        self.assertEqual(first["CUDA0"]["compute_ms"], 8)
        self.assertEqual(first[PHONE]["compute_ms"], 8)
        with patch.dict(os.environ, {ENV: "1", "LLAMA_JUPITER_HTP_SESSIONS": "4",
                                     "LLAMA_JUPITER_HTP_ACTIVE_SESSIONS": "4"}):
            profile = QProfile(rows, metric="compute_ms", require_cpu=True)
            self.assertTrue(profile.is_measured(96, 0))
            self.assertTrue(profile.is_measured(96, 96))
            self.assertEqual(profile.q(96, 0), 11)
            with self.assertRaises(ValueError):
                QProfile(rows + [dict(rows[0], device="RPC0")], metric="compute_ms")
    def test_missing_footer_and_graph_fallback_rejected(self):
        text = sample()
        with self.assertRaises(ValueError):
            validate_run("\n".join(s for s in text.splitlines()
                                      if "NATIVE_PHONE_PROFILE" not in s))
        with self.assertRaises(ValueError):
            validate_run(text + "\n[RPC_GRAPH] cmd=GRAPH_COMPUTE")
        with self.assertRaises(ValueError):
            profile_rows(text.replace("native_bridge=1", "native_bridge=0"))
    def test_long_sweep_respects_native_wire_capacity(self):
        plans = print_sweep_plans(1855, [64,96,128], 64,128, max_subsequences=16)
        self.assertTrue(plans)
        for x, cuts in plans:
            self.assertEqual(cuts[0], x)
            self.assertEqual(sum(cuts),1855)
            self.assertLessEqual(len(cuts),16)
            self.assertTrue(all(64 <= c <= 128 for c in cuts))

if __name__ == "__main__":
    unittest.main()
