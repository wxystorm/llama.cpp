#include "../ggml/src/ggml-rpc/phone-qwen3-local-builder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "check failed: %s at %s:%d\n", \
                 #expr, __FILE__, __LINE__); \
    std::abort(); \
} } while (false)

static ggml_phone_owned::region_plan plan() {
    ggml_phone_owned::region_plan p {};
    p.run_id = 123;
    p.p2c_run_id = 45;
    p.region = 1;
    p.layer = 9;
    p.total_tokens = 512;
    p.chunk_tokens = 128;
    p.n_embd = 5120;
    p.position_base = 0;
    p.kv_capacity = 2048;
    p.n_heads = 40;
    p.n_heads_kv = 8;
    p.head_dim = 128;
    p.n_kv = 512;
    p.rope_type = ggml_phone_owned::k_rope_neox;
    p.n_ctx_orig = 32768;
    p.kv_type = ggml_phone_owned::k_f16;
    p.rms_norm_eps = 1e-6f;
    p.rope_freq_base = 1000000.0f;
    p.rope_freq_scale = 1.0f;
    p.rope_ext_factor = -1.0f;
    p.rope_attn_factor = 1.0f;
    p.rope_beta_fast = 32.0f;
    p.rope_beta_slow = 1.0f;
    return p;
}

static ggml_phone_qwen3::layer_weights weights(
        ggml_context * ctx, int64_t n_heads = 40,
        int64_t ffn_width = 17408) {
    using ggml_phone_qwen3::layer_weights;
    layer_weights w {};
    const int64_t q_width = n_heads * 128;
    w.attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
    w.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, q_width);
    w.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 1024);
    w.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 1024);
    w.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_width, 5120);
    w.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
    w.ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, ffn_width);
    w.ffn_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, ffn_width);
    w.ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn_width, 5120);
    w.q_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    w.k_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    return w;
}

int main() {
    const char * reason = "NONE";
    auto p = plan();
    CHECK(ggml_phone_owned::validate(p, reason));
    ggml_init_params params = {
        ggml_tensor_overhead() * 1024 +
        ggml_graph_overhead_custom(512, false) + 4096,
        nullptr, true
    };
    ggml_context * ctx = ggml_init(params);
    CHECK(ctx != nullptr);
    // Mistral text graph: unequal chunks, Q=32, KV=8, no per-head norm.
    {
        auto lp = p;
        lp.flags |= ggml_phone_owned::k_flag_llama_dense;
        lp.rope_type = ggml_phone_owned::k_rope_norm;
        lp.n_heads = 32;
        lp.schedule_count = 3;
        lp.schedule[0] = 100;
        lp.schedule[1] = 180;
        lp.schedule[2] = 232;
        lp.chunk_tokens = 232;
        ggml_context * lc = ggml_init(params);
        CHECK(lc != nullptr);
        auto lw = weights(lc, 32, 32768);
        lw.q_norm = nullptr;
        lw.k_norm = nullptr;
        CHECK(ggml_phone_qwen3::validate_weights(lp, lw, reason));
        ggml_phone_qwen3::graph_result lg {};
        CHECK(ggml_phone_qwen3::build_graph(lc, lp, 1, lw, lg, reason));
        CHECK(lg.token_begin == 100 && lg.tokens == 180);
        CHECK(lg.output->ne[0] == 5120 && lg.output->ne[1] == 180);
        int norms = 0, ropes = 0, kv_writes = 0;
        for (int i = 0; i < ggml_graph_n_nodes(lg.graph); ++i) {
            const auto * n = ggml_graph_node(lg.graph, i);
            norms += n->op == GGML_OP_RMS_NORM;
            kv_writes += n->op == GGML_OP_SET_ROWS;
            if (n->op == GGML_OP_ROPE) {
                ++ropes;
                CHECK(n->op_params[2] == GGML_ROPE_TYPE_NORMAL);
            }
        }
        CHECK(norms == 2 && ropes == 2 && kv_writes == 2);
        auto bad = lw;
        bad.wq_b = ggml_new_tensor_1d(lc, GGML_TYPE_F32, 5120);
        CHECK(!ggml_phone_qwen3::validate_weights(lp, bad, reason));
        ggml_free(lc);
    }
    const auto w = weights(ctx);
    CHECK(ggml_phone_qwen3::validate_weights(p, w, reason));
    for (uint32_t chunk = 0; chunk < 4; ++chunk) {
        ggml_phone_qwen3::graph_result g {};
        CHECK(ggml_phone_qwen3::build_graph(ctx, p, chunk, w, g, reason));
        CHECK(g.graph && g.output && g.input);
        CHECK(g.tokens == 128 && g.token_begin == chunk * 128);
        CHECK(g.input->type == GGML_TYPE_F32);
        CHECK(g.mask->type == GGML_TYPE_F16);
        CHECK(g.cache_k->type == GGML_TYPE_F16);
        CHECK(g.cache_v->ne[0] == 1024);
        CHECK(g.output->ne[0] == 5120 && g.output->ne[1] == 128);
        int set_rows = 0, flash = 0, rope = 0, glu = 0;
        for (int n = 0; n < ggml_graph_n_nodes(g.graph); ++n) {
            switch (ggml_graph_node(g.graph, n)->op) {
                case GGML_OP_SET_ROWS: ++set_rows; break;
                case GGML_OP_FLASH_ATTN_EXT: ++flash; break;
                case GGML_OP_ROPE: ++rope; break;
                case GGML_OP_GLU: ++glu; break;
                default: break;
            }
        }
        CHECK(set_rows == 2 && flash == 1 && rope == 2 && glu == 1);
        // The HTP stage tracer matches these labels against the standard
        // Qwen3 graph callbacks. Fail model-free tests if a local tensor
        // is renamed, omitted or duplicated.
        const char * stages[] = {
            "attn_norm", "Qcur", "Kcur", "Vcur", "attn_out",
            "ffn_inp", "ffn_norm", "ffn_up", "ffn_gate",
            "ffn_swiglu", "ffn_down"
        };
        for (const char * stage : stages) {
            const std::string name = std::string(stage) +
                "-" + std::to_string(p.layer);
            int matches = 0;
            for (int node = 0; node < ggml_graph_n_nodes(g.graph); ++node) {
                matches += name == ggml_graph_node(g.graph, node)->name;
            }
            CHECK(matches == 1);
        }
        ggml_phone_qwen3::side_inputs side {};
        CHECK(ggml_phone_qwen3::make_side_inputs(p, chunk, side, reason));
        CHECK(side.positions.front() == int32_t(chunk * 128));
        CHECK(side.k_indices.back() == int64_t(chunk * 128 + 127));
        CHECK(side.causal_mask_f16.size() == 512 * 128);
        CHECK(side.causal_mask_f16[0] == 0);
        const size_t future = (size_t) 1 * p.n_kv + (chunk * 128 + 2);
        CHECK(side.causal_mask_f16[future] == 0xfc00);
    }
    // The Jupiter Sequence DP may choose 60,60,110,128 instead of
    // uniform chunks. Graph input dimensions, RoPE positions, and the
    // causal KV mask must follow exact prefix sums, not chunk*width.
    auto dp = p;
    dp.total_tokens = 358;
    dp.chunk_tokens = 128;
    dp.schedule_count = 4;
    dp.schedule = {60, 60, 110, 128};
    CHECK(ggml_phone_owned::validate(dp, reason));
    ggml_init_params dp_params = {
        ggml_tensor_overhead() * 1024 +
        ggml_graph_overhead_custom(512, false) + 4096,
        nullptr, true
    };
    ggml_context * dp_ctx = ggml_init(dp_params);
    CHECK(dp_ctx != nullptr);
    const auto dp_weights = weights(dp_ctx);
    const uint32_t dp_starts[] = {0, 60, 120, 230};
    for (uint32_t i = 0; i < dp.schedule_count; ++i) {
        ggml_phone_qwen3::graph_result dp_graph {};
        CHECK(ggml_phone_qwen3::build_graph(
            dp_ctx, dp, i, dp_weights, dp_graph, reason));
        CHECK(dp_graph.token_begin == dp_starts[i]);
        CHECK(dp_graph.tokens == dp.schedule[i]);
        ggml_phone_qwen3::side_inputs dp_side {};
        CHECK(ggml_phone_qwen3::make_side_inputs(dp, i, dp_side, reason));
        CHECK(dp_side.positions.front() == int32_t(dp_starts[i]));
        CHECK(dp_side.positions.back() ==
              int32_t(dp_starts[i] + dp.schedule[i] - 1));
        CHECK(dp_side.causal_mask_f16.size() ==
              size_t(dp.n_kv) * dp.schedule[i]);
    }
    ggml_free(dp_ctx);
    // Tail graph has a distinct token dimension, but shares one KV span.
    auto tail = p;
    tail.total_tokens = 500;
    ggml_phone_qwen3::side_inputs side {};
    CHECK(ggml_phone_qwen3::make_side_inputs(tail, 3, side, reason));
    CHECK(side.positions.size() == 116);
    CHECK(side.positions.back() == 499);
    CHECK(side.causal_mask_f16[115 * 512 + 499] == 0);
    CHECK(side.causal_mask_f16[115 * 512 + 500] == 0xfc00);
    auto bad = w;
    bad.ffn_up = w.wq;
    CHECK(!ggml_phone_qwen3::validate_weights(p, bad, reason));

    // Check GGML graph shape for Qwen3-32B, not only plan metadata.
    auto wide = p;
    wide.layer = 63;
    wide.region = 15;
    wide.n_heads = 64;
    wide.total_tokens = 2048;
    wide.kv_capacity = 2048;
    wide.n_kv = 2048;
    // Qwen3-32B has 8192 Query width and 25600 FFN hidden width.
    const auto wide_weights = weights(ctx, 64, 25600);
    CHECK(ggml_phone_owned::validate(wide, reason));
    CHECK(ggml_phone_qwen3::validate_weights(wide, wide_weights, reason));
    ggml_phone_qwen3::graph_result wide_graph {};
    CHECK(ggml_phone_qwen3::build_graph(
        ctx, wide, 0, wide_weights, wide_graph, reason));
    CHECK(wide_graph.graph && wide_graph.output);
    CHECK(wide_graph.output->ne[0] == 5120 &&
          wide_graph.output->ne[1] == 128);
    CHECK(wide_weights.ffn_up->ne[1] == 25600);
    CHECK(wide_graph.cache_k->ne[1] == 2048);
    ggml_phone_qwen3::side_inputs wide_side {};
    CHECK(ggml_phone_qwen3::make_side_inputs(wide, 0, wide_side, reason));
    CHECK(wide_side.causal_mask_f16.size() == size_t(2048)*128);
    const ggml_tensor * out_projection = nullptr;
    for (int n = 0; n < ggml_graph_n_nodes(wide_graph.graph); ++n) {
        const ggml_tensor * node = ggml_graph_node(wide_graph.graph, n);
        if (std::strcmp(node->name, "attn_out-63") == 0) {
            out_projection = node;
            break;
        }
    }
    CHECK(out_projection && out_projection->op == GGML_OP_MUL_MAT);
    CHECK(out_projection->src[1] &&
          out_projection->src[1]->ne[0] == 8192 &&
          out_projection->src[1]->ne[1] == 128);
    auto wrong_projection = wide_weights;
    wrong_projection.wo = w.wo; // 14B's narrower Q output projection
    CHECK(!ggml_phone_qwen3::validate_weights(
        wide, wrong_projection, reason));
    // Jupiter can now request a Phone-local Graph with varying prefill KV
    // capacity. Exercise each supported shape with an independently owned
    // no-alloc graph context, rather than relying on the 512-token pilot.
    for (uint32_t context : {256u, 512u, 1024u, 2048u, 4096u}) {
        auto dynamic = p;
        dynamic.layer = 39;
        dynamic.region = 20;
        dynamic.total_tokens = context;
        dynamic.kv_capacity = context;
        dynamic.n_kv = context;
        dynamic.chunk_tokens = context == 4096 ? 256 : 128;
        CHECK(ggml_phone_owned::validate(dynamic, reason));
        CHECK(ggml_phone_owned::chunk_count(dynamic) <= 16);
        ggml_init_params opts = {
            ggml_tensor_overhead() * 1024 +
            ggml_graph_overhead_custom(512, false) + 4096,
            nullptr, true
        };
        ggml_context * dyn_ctx = ggml_init(opts);
        CHECK(dyn_ctx != nullptr);
        const auto dyn_weights = weights(dyn_ctx);
        ggml_phone_qwen3::graph_result dyn_graph {};
        CHECK(ggml_phone_qwen3::build_graph(
            dyn_ctx, dynamic, 0, dyn_weights, dyn_graph, reason));
        CHECK(dyn_graph.graph && dyn_graph.output);
        CHECK(dyn_graph.tokens == dynamic.chunk_tokens);
        CHECK(dyn_graph.cache_k->ne[1] == context);
        ggml_phone_qwen3::side_inputs side {};
        const uint32_t last_chunk =
            ggml_phone_owned::chunk_count(dynamic) - 1;
        CHECK(ggml_phone_qwen3::make_side_inputs(
            dynamic, last_chunk, side, reason));
        CHECK(side.causal_mask_f16.size() ==
              size_t(context) * dynamic.chunk_tokens);
        ggml_free(dyn_ctx);
    }
    ggml_free(ctx);
    std::puts("test-phone-qwen3-local-builder PASS");
    return 0;
}
