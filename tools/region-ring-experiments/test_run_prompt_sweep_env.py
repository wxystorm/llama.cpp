"""No-device regression tests for forced G/C/P versus native GCNA RPC flags."""

import unittest
from pathlib import Path

from run_prompt_sweep import configure_phone_plan_environment


class PhonePlanSweepEnvironmentTests(unittest.TestCase):
    def stale_plan_environment(self):
        return {
            "LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS": "1,3,5,7",
            "LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP": "1",
            "LLAMA_REGION_RING_PHONE_GRAPHLESS": "1",
            "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW": "1",
            "LLAMA_REGION_RING_PHONE_AUTONOMOUS": "1",
            "UNRELATED_SETTINGS": "preserve",
        }

    def assert_legacy_plan_unset(self, env):
        for key in (
                "LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS",
                "LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP",
                "LLAMA_REGION_RING_PHONE_GRAPHLESS",
                "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW"):
            self.assertNotIn(key, env)
        self.assertEqual(env["UNRELATED_SETTINGS"], "preserve")

    def test_force_layout_with_skip_profile_does_not_inherit_npu_shadow(self):
        env = self.stale_plan_environment()
        self.assertFalse(configure_phone_plan_environment(
            env, v2_csv=None, force_layout=True))
        self.assert_legacy_plan_unset(env)
        self.assertEqual(env["LLAMA_REGION_RING_PHONE_AUTONOMOUS"], "1")

    def test_force_layout_cannot_enable_native_plan_even_with_v2_file(self):
        env = self.stale_plan_environment()
        self.assertFalse(configure_phone_plan_environment(
            env, v2_csv=Path("stage-profile-v2.csv"), force_layout=True))
        self.assert_legacy_plan_unset(env)
        self.assertNotIn("LLAMA_REGION_RING_V2_PROFILE_CSV", env)

    def test_auto_native_gcna_enables_only_model_owned_npu_plan(self):
        env = self.stale_plan_environment()
        self.assertTrue(configure_phone_plan_environment(
            env, v2_csv=Path("stage-profile-v2.csv"), force_layout=False))
        self.assertEqual(
            env["LLAMA_REGION_RING_V2_PROFILE_CSV"], "stage-profile-v2.csv")
        self.assertEqual(env["LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW"], "1")
        self.assertEqual(env["LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP"], "0")
        self.assertEqual(env["LLAMA_REGION_RING_PHONE_GRAPHLESS"], "0")
        self.assertNotIn("LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS", env)

    def test_no_v2_does_not_inherit_stale_plan_on_default_sweep(self):
        env = self.stale_plan_environment()
        self.assertFalse(configure_phone_plan_environment(
            env, v2_csv=None, force_layout=False))
        self.assert_legacy_plan_unset(env)


if __name__ == "__main__":
    unittest.main()
