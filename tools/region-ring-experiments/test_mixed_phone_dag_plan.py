"""Offline test: NPU->PhoneGPU spill, memory reject, bank-mover DAG."""
import unittest
from mixed_phone_dag_plan import Cycle, memory, split_phone, plan, dag_for, planner_backend_mapping

BASE = dict(gpu_budget_mib=500, npu_budget_mib=25,
            gpu_weight_layer_mib=8, npu_weight_layer_mib=10,
            gpu_kv_layer_mib=.1, npu_kv_layer_mib=2,
            gpu_scratch_mib=1, npu_scratch_mib=1,
            gpu_h2d_mibps=250, c2p_mibps=100,
            p2c_mibps=100, local_mibps=1000)
MIX = dict(phone_gpu_budget_mib=300, phone_gpu_weight_layer_mib=5,
           phone_gpu_kv_layer_mib=1, phone_gpu_scratch_mib=1,
           phone_shared_budget_mib=300, phone_local_mibps=500,
           npu_host_stage_mibps=200, npu_bank_repack_mibps=400,
           npu_host_stage_budget_mib=200, pc_head_ms=2)


class FakeProfile:
    devices = {"CPU", "GPU", "PHONE", "PHONE_GPU"}

    def estimate(self, dev, layers, tokens, kv, terminal=False):
        return ({"GPU": 2., "CPU": 3., "PHONE": 12.,
                 "PHONE_GPU": 43.}[dev] * layers, .8)


class MixedPlanTests(unittest.TestCase):
    def test_plan_owns_region_layer_mapping_with_omitted_zero_width(self):
        cycles = [Cycle(7, 1, 1, 1), Cycle(8, 0, 1, 0)]
        npu, phone_gpu = planner_backend_mapping(cycles)
        self.assertEqual(npu, [
            {"region": 2, "layer": 8},
            {"region": 5, "layer": 18},
        ])
        self.assertEqual(phone_gpu, [{
            "region": 3, "first_layer": 9,
            "last_layer_exclusive": 10,
        }])

    def test_spill_instead_of_early_reject(self):
        gcp = [(7, 1, 2)] * 4
        before = [Cycle(g, c, 1, 1) for g, c, _ in gcp]
        self.assertFalse(memory(before, BASE, MIX).fits)
        options = list(split_phone(gcp, BASE, MIX, 1))
        self.assertTrue(options)
        self.assertTrue(all(m.fits for _, m in options))
        self.assertTrue(any(sum(x.n for x in cs) < 4 for cs, _ in options))
        self.assertTrue(all(sum(c.layers for c in cs) == 40 for cs, _ in options))

    def test_phone_gpu_overbudget_reject(self):
        gcp = [(7, 1, 2)] * 4
        mix = dict(MIX, phone_gpu_budget_mib=1.)
        self.assertEqual(list(split_phone(gcp, BASE, mix, 1)), [])

    def test_npu_banks_memory_not_cumulative(self):
        cs = [Cycle(7, 1, 1, 1)] * 4
        m = memory(cs, dict(BASE, npu_budget_mib=100), MIX)
        self.assertEqual(m.npu_bank_mib, (10, 10))
        self.assertEqual(m.npu_used_mib, 29)

    def test_htp_host_staging_budget_is_not_free(self):
        cs = [Cycle(7, 1, 1, 1)] * 4
        base = dict(BASE, npu_budget_mib=100)
        allowed = memory(cs, base, MIX)
        denied = memory(cs, base, dict(MIX, npu_host_stage_budget_mib=9))
        self.assertEqual(allowed.npu_host_stage_peak_mib, 10)
        self.assertTrue(allowed.fits)
        self.assertFalse(denied.fits)
        self.assertGreater(allowed.phone_shared_used_mib,
                           allowed.npu_used_mib + allowed.phone_gpu_used_mib)

    def test_dag_refill_dependencies_and_cost(self):
        cs = [Cycle(7, 1, 1, 1)] * 4
        r = dag_for(cs, 64, 256, 5120, FakeProfile(), BASE, MIX, 2.)
        self.assertIsNotNone(r)
        self.assertIn("N_BANK_HOST_STAGE:2", r["bank_nodes"])
        self.assertIn("N_BANK_REPACK:3", r["bank_nodes"])
        self.assertIn("G_BANK_H2D:2", r["bank_nodes"])
        self.assertGreater(r["npu_bank_repack_ms"], 0)
        self.assertGreater(r["gpu_bank_h2d_ms"], 0)

    def test_dag_refill_heavier_cannot_be_faster(self):
        cs = [Cycle(7, 1, 1, 1)] * 4
        fast = dag_for(cs, 64, 256, 5120, FakeProfile(), BASE, MIX, 2.)
        slow = dag_for(cs, 64, 256, 5120, FakeProfile(), BASE,
                       dict(MIX, npu_host_stage_mibps=1,
                            npu_bank_repack_mibps=1), 2.)
        self.assertGreater(slow["wall_ms"], fast["wall_ms"])

    def test_phone_gpu_linear_projection_from_measured_p1(self):
        from mixed_phone_dag_plan import StageProfile
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.csv"
            path.write_text(
                "device,layers,tokens,kv_tokens,setup_ms,upload_ms,"
                "prepare_ms,compute_ms,terminal,samples\n"
                "PHONE_GPU,1,64,64,0,1,2,40,0,1\n", encoding="utf-8")
            profile = StageProfile(path)
            self.assertEqual(profile.estimate("PHONE_GPU", 3, 64, 64),
                             (120, 9))
            self.assertIsNone(profile.estimate("PHONE", 3, 64, 64))

    def test_auto_search(self):
        opts, diag = plan(dict(layers=40, tokens=256, hidden=5120),
                          FakeProfile(), BASE, MIX)
        self.assertGreater(len(opts), 0)
        self.assertGreater(diag["npu_spill_geometries"], 0)
        self.assertTrue(all(c.layers > 0 for c in opts[0][3]))


if __name__ == "__main__":
    unittest.main()
