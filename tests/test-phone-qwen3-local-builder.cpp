#include "../ggml/src/ggml-rpc/phone-qwen3-local-builder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static ggml_phone_qwen3::layer_weights weights(ggml_context * ctx) {
    using ggml_phone_qwen3::layer_weights;
    layer_weights w {};
    w.attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
    w.wq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 5120);
    w.wk = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 1024);
    w.wv = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 1024);
    w.wo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 5120);
    w.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5120);
    w.ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 17408);
    w.ffn_up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5120, 17408);
    w.ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 17408, 5120);
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
    ggml_free(ctx);
    std::puts("test-phone-qwen3-local-builder PASS");
    return 0;
}
