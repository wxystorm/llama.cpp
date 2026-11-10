#!/usr/bin/env python3
"""Source-level regression checks for the opt-out Phone HTP scratch rollout.

These guard ownership, admission and dynamic rebind contracts; the real HTP
fixed-UID cache still requires an on-device short/long Prompt smoke test.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
EXEC = ROOT / "ggml" / "src" / "ggml-rpc" / "phone-npu-serial-executor.h"
RPC = ROOT / "ggml" / "src" / "ggml-rpc" / "ggml-rpc.cpp"


class PhoneOwnedScratchReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = EXEC.read_text(encoding="utf-8")
        cls.rpc = RPC.read_text(encoding="utf-8")

    def test_graphless_only_default_and_legacy_optout(self):
        src = self.executor
        self.assertIn('std::getenv("GGML_RPC_PHONE_QWEN3_LOCAL_GRAPHLESS")', src)
        self.assertIn('std::getenv("GGML_RPC_PHONE_QWEN3_SHARED_SCRATCH")', src)
        self.assertIn('std::strcmp(reuse_env, "0") == 0', src)
        self.assertIn('constexpr size_t scratch_cap_bytes = size_t(256) * 1024 * 1024', src)
        self.assertIn('reason = "PHONE_OWNED_HTP_SCRATCH_BUDGET_EXCEEDED"', src)
        self.assertIn('reason = "PHONE_OWNED_HTP_SCRATCH_CAPACITY_CHANGED"', src)
        self.assertIn('item->buffer.reset(ggml_backend_alloc_ctx_tensors(', src)

    def test_one_arena_owned_by_region_kv_not_chunk(self):
        src = self.executor
        owned_kv = src[src.index("struct owned_layer_kv {"):src.index("struct owned_chunk_graph {")]
        self.assertIn("ggml_backend_buffer_ptr scratch_buffer;", owned_kv)
        self.assertIn("ggml_tallocr_new(kv->scratch_buffer.get())", src)
        self.assertIn("ggml_tallocr_alloc(&scratch_alloc, t)", src)
        self.assertIn("ggml_backend_view_init(t)", src)
        self.assertNotIn("ggml_backend_buffer_ptr scratch_buffer;", src[src.index("struct owned_chunk_graph {"):])
        self.assertIn("owned_graphs_.clear();", src)
        self.assertIn("owned_kv_.clear();", src)

    def test_rebind_before_htp_compute_for_both_paths(self):
        src = self.executor
        only = src[src.index("bool run_owned_only_chunk("):src.index("struct local_exec_profile {")]
        shadow = src[src.index("bool run_owned_htp_chunk("):src.index("bool publish_owned_htp_chunk(")]
        for path, input_copy in (
            (only, "ggml_backend_tensor_set(g.input, source, 0, bytes);"),
            (shadow, "ggml_backend_tensor_set(g.input, reference_input_snapshot, 0, bytes);"),
        ):
            rebind = path.index("refresh_owned_chunk_side_inputs(p, chunk, g, reason)")
            upload = path.index(input_copy)
            compute = path.index("ggml_backend_graph_compute(backend_, g.graph)")
            self.assertLess(rebind, upload)
            self.assertLess(upload, compute)
            self.assertIn("state.scratch_buffer", path)
        self.assertIn("ggml_backend_tensor_set(\n            g.positions", src)
        self.assertIn("ggml_backend_tensor_set(\n            g.k_indices", src)
        self.assertIn("ggml_backend_tensor_set(\n            g.v_indices", src)
        self.assertIn("ggml_backend_tensor_set(\n            g.mask", src)

    def test_cache_cleared_before_scratch_teardown(self):
        src = self.executor
        for method_name, next_method in (
            ("void release_owned_htp_graphs()", "static bool capture_weight_sources("),
            ("void reset_kv()", "bool has_backend()"),
        ):
            part = src[src.index(method_name):src.index(next_method)]
            self.assertLess(
                part.index('ggml_backend_hexagon_clear_fixed_graph_cache'),
                part.index('owned_graphs_.clear()'),
            )
            self.assertLess(part.index("owned_graphs_.clear()"),
                            part.index("owned_kv_.clear()"))

    def test_telemetry_records_real_vs_logical_allocation(self):
        src = self.executor
        self.assertIn('event=REGION_PREPARED', src)
        self.assertIn('mode=%s status=PASS', src)
        self.assertIn('stats.graph_logical_storage_bytes += graph_required_bytes', src)
        self.assertIn('stats.graph_storage_bytes +=', src)
        self.assertIn('stats.scratch_slots = 1', src)
        self.assertIn('graph_logical_bytes=%zu scratch_live_bytes=%zu', self.rpc)
        self.assertIn('scratch_slots=%u', self.rpc)
        self.assertIn('session_scratch_live_bytes=%zu', self.rpc)
        self.assertIn('session_scratch_peak_bytes=%zu', self.rpc)
        self.assertIn('owned_scratch_live_bytes_ = 0', src)
        self.assertIn('owned_scratch_peak_bytes_ = 0', src)


if __name__ == "__main__":
    unittest.main()
