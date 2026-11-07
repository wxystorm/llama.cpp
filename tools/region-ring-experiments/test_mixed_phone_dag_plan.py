"""Offline test: NPU->PhoneGPU spill, memory reject, bank-mover DAG."""
import unittest
from mixed_phone_dag_plan import (Cycle, memory, split_phone, plan, dag_for,
                                  planner_backend_mapping, execution_stage_graph,
                                  handoff_route, geometry,
                                  runtime_compatible_cycles)

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

    def test_optional_backends_and_all_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "INVALID_CYCLE"):
            Cycle(0, 0, 0, 0)
        for cycle in (Cycle(0, 1, 1, 0), Cycle(0, 0, 1, 2),
                      Cycle(2, 0, 0, 0), Cycle(0, 2, 0, 0),
                      Cycle(0, 0, 0, 2)):
            self.assertGreater(cycle.layers, 0)

    def test_omitted_regions_have_no_phantom_ids(self):
        cycles = [Cycle(0, 1, 1, 2), Cycle(5, 0, 0, 0), Cycle(0, 1, 0, 1)]
        stages, edges = execution_stage_graph(cycles)
        self.assertEqual([s["backend"] for s in stages], ["C", "N", "A", "G", "C", "A"])
        self.assertEqual([s["region"] for s in stages], list(range(6)))
        self.assertEqual(stages[-1]["last_layer_exclusive"], 11)
        self.assertEqual([e["handoff"] for e in edges],
                         ["C2P", "HTP_TO_ADRENO", "P2C", "PC_LOCAL", "C2P"])
        self.assertEqual(planner_backend_mapping(cycles), (
            [{"region": 1, "layer": 1}],
            [{"region": 2, "first_layer": 2, "last_layer_exclusive": 4},
             {"region": 5, "first_layer": 10, "last_layer_exclusive": 11}]))

    def test_phone_handoff_is_not_assumed_zero_copy(self):
        self.assertEqual(handoff_route("N", "A"), ("HTP_TO_ADRENO", None))
        self.assertEqual(handoff_route("A", "N"), ("PHONE_LOCAL", "phone_local"))
        self.assertEqual(handoff_route("C", "G"), ("PC_CPU_TO_CUDA", "pc_h2d"))
        self.assertEqual(handoff_route("N", "C"), ("P2C", "phone_to_pc"))
        self.assertEqual(handoff_route("G", "G"), ("SAME_PC_BACKEND", None))

    def test_geometry_includes_zero_gpu_cpu_and_phone(self):
        seeds = list(geometry(12, max_cycles=2, max_c=2, max_phone=6))
        self.assertTrue(any(all(g == 0 for g,_,_ in xs) for xs in seeds))
        self.assertTrue(any(all(c == 0 for _,c,_ in xs) for xs in seeds))
        self.assertTrue(any(all(p == 0 for _,_,p in xs) for xs in seeds))

    def test_no_npu_and_no_adreno_both_allowed(self):
        base = dict(BASE, npu_budget_mib=100)
        no_npu = list(split_phone([(5, 1, 1), (5, 1, 1)],
                                  base, MIX, max_npu=0))
        self.assertTrue(no_npu)
        self.assertTrue(all(c.n == 0 for cs,_ in no_npu for c in cs))
        no_adreno = list(split_phone([(5, 1, 1), (5, 1, 1)],
                                     base, MIX, max_npu=1))
        self.assertTrue(any(all(c.a == 0 for c in cs) for cs,_ in no_adreno))

    def test_no_unused_bank_or_scratch_for_absent_engines(self):
        cycles = [Cycle(0, 2, 0, 3), Cycle(0, 2, 0, 3)]
        result = memory(cycles, BASE, MIX)
        self.assertEqual(result.gpu_bank_mib, (0., 0.))
        self.assertEqual(result.npu_bank_mib, (0., 0.))
        self.assertEqual(result.gpu_pc_used_mib, 0.)
        self.assertEqual(result.npu_used_mib, 0.)

    def test_cross_cycle_handoff_uses_next_nonzero_stage(self):
        cycles = [Cycle(7, 1, 1, 1), Cycle(0, 2, 0, 8),
                  Cycle(7, 0, 0, 3)]
        r = dag_for(cycles, 64, 128, 5120, FakeProfile(),
                    dict(BASE, npu_budget_mib=100), MIX, 2.)
        self.assertIsNotNone(r)
        self.assertIn("HTP_TO_ADRENO", r["handoffs"])
        self.assertIn("P2C", r["handoffs"])
        self.assertIn("C2P", r["handoffs"])
        self.assertNotIn("N_CONTROL_P2C", r["handoffs"])

    def test_no_intermediate_htp_frame_for_local_phone_island(self):
        # Two local N->A handoffs, but only a real Adreno->PC crossing and
        # the final Phone->PC Head output consume P2C bandwidth.
        cs = [Cycle(7, 1, 1, 1), Cycle(7, 1, 1, 1)]
        result = dag_for(cs, 64, 128, 5120, FakeProfile(),
                         dict(BASE, npu_budget_mib=100), MIX, 2.)
        self.assertIsNotNone(result)
        self.assertNotIn("N_CONTROL_P2C", result["handoffs"])
        self.assertIn("HTP_TO_ADRENO", result["handoffs"])
        # A of cycle 0 -> next PC, plus final A -> PC Head. No HTP P2C.
        expected_p2c_ms = 2 * (128 * 5120 * 4 / (1024*1024) /
                                BASE["p2c_mibps"] * 1000)
        self.assertAlmostEqual(result["busy_ms"]["phone_to_pc"],
                               expected_p2c_ms, places=5)
        self.assertFalse(any("N_CONTROL" in node
                             for node in result["bank_nodes"]))

    def test_terminal_cuda_requires_measured_d2h(self):
        cycles = [Cycle(5, 0, 0, 0), Cycle(5, 0, 0, 0)]
        self.assertIsNone(dag_for(cycles, 64, 128, 5120,
                                  FakeProfile(), BASE, MIX, 2.))
        measured = dict(BASE, gpu_d2h_mibps=200)
        result = dag_for(cycles, 64, 128, 5120,
                         FakeProfile(), measured, MIX, 2.)
        self.assertIn("FINAL_GPU_TO_CPU", result["handoffs"])

    def test_runtime_candidate_requires_real_adjacent_phone_link(self):
        self.assertTrue(runtime_compatible_cycles([
            Cycle(7, 1, 1, 1), Cycle(7, 1, 1, 1),
            Cycle(8, 1, 0, 1), Cycle(7, 2, 0, 1)]))
        self.assertFalse(runtime_compatible_cycles([
            Cycle(8, 1, 1, 0), Cycle(8, 1, 1, 0)]))
        self.assertFalse(runtime_compatible_cycles([
            Cycle(0, 1, 1, 1), Cycle(7, 1, 1, 1)]))
        self.assertFalse(runtime_compatible_cycles([
            Cycle(7, 1, 0, 1), Cycle(7, 1, 0, 1)]))

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
