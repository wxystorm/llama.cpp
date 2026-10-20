#include "llama-hybrid-cycle-graph.h"

#include <cassert>

int main() {
    ggml_init_params params {};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);

    constexpr int64_t hidden = 2048;
    constexpr int64_t tokens = 128;
    constexpr int64_t topk = 8;

    auto ffn = llama_hybrid_cycle_make_ffn_graph_inputs(
        ctx, hidden, tokens, topk);
    assert(ffn.norm && ffn.topk_ids && ffn.topk_weights);
    assert((ffn.norm->flags & GGML_TENSOR_FLAG_INPUT) != 0);
    assert((ffn.topk_ids->flags & GGML_TENSOR_FLAG_INPUT) != 0);
    assert((ffn.topk_weights->flags & GGML_TENSOR_FLAG_INPUT) != 0);
    assert(!llama_hybrid_cycle_ffn_graph_valid(ffn, hidden, tokens, topk));

    // Synthetic FFN output exercises the standalone tensor boundary;
    // the actual Qwen3-MoE graph builder substitutes build_moe_ffn(), using
    // these same three inputs and skipping duplicate Router and residual ADD.
    ffn.ffn_output = ggml_add(ctx, ffn.norm, ffn.norm);
    assert(llama_hybrid_cycle_ffn_graph_valid(ffn, hidden, tokens, topk));
    assert(!llama_hybrid_cycle_ffn_graph_valid(ffn, hidden, tokens - 1, topk));
    assert(!llama_hybrid_cycle_ffn_graph_valid(ffn, hidden, tokens, topk + 1));

    auto bad_type = ffn;
    bad_type.topk_weights = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F16, 1, topk, tokens);
    assert(!llama_hybrid_cycle_ffn_graph_valid(
        bad_type, hidden, tokens, topk));

    auto bad_shape = ffn;
    bad_shape.ffn_output = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hidden, tokens / 2);
    assert(!llama_hybrid_cycle_ffn_graph_valid(
        bad_shape, hidden, tokens, topk));

    auto bad_ids = ffn;
    bad_ids.topk_ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, topk, tokens);
    assert(!llama_hybrid_cycle_ffn_graph_valid(
        bad_ids, hidden, tokens, topk));

    const auto missing = llama_hybrid_cycle_make_ffn_graph_inputs(
        nullptr, hidden, tokens, topk);
    assert(missing.norm == nullptr && missing.topk_ids == nullptr);
    ggml_free(ctx);
    return 0;
}
