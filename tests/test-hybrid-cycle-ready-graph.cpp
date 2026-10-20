#include "llama-hybrid-cycle-graph.h"

#include <cassert>
#include <cstdint>

int main() {
    ggml_init_params p {};
    p.mem_size = 4 * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    assert(ctx != nullptr);

    constexpr int64_t hidden = 2048;
    constexpr int64_t tokens = 128;
    constexpr int64_t topk = 8;
    ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    ggml_tensor * norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, tokens);
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, topk, tokens);

    const llama_hybrid_cycle_ready_graph good {residual, norm, ids, weights};
    assert(llama_hybrid_cycle_ready_graph_valid(good, hidden, tokens, topk));
    assert(!llama_hybrid_cycle_ready_graph_valid(good, hidden, tokens - 1, topk));
    assert(!llama_hybrid_cycle_ready_graph_valid(good, hidden, tokens, topk - 1));
    assert(!llama_hybrid_cycle_ready_graph_valid(good, hidden, 0, topk));

    auto missing = good;
    missing.topk_ids = nullptr;
    assert(!llama_hybrid_cycle_ready_graph_valid(missing, hidden, tokens, topk));

    auto wrong_type = good;
    wrong_type.topk_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, topk, tokens);
    assert(!llama_hybrid_cycle_ready_graph_valid(wrong_type, hidden, tokens, topk));

    auto wrong_shape = good;
    wrong_shape.topk_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, topk, tokens);
    // [topk,tokens] can have identical bytes, but the graph exporter requires
    // the Qwen3-MoE [1,topk,tokens] identity to detect placement reliably.
    assert(!llama_hybrid_cycle_ready_graph_valid(wrong_shape, hidden, tokens, topk));

    auto wrong_hidden = good;
    wrong_hidden.ffn_norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden / 2, tokens);
    assert(!llama_hybrid_cycle_ready_graph_valid(wrong_hidden, hidden, tokens, topk));

    ggml_free(ctx);
    return 0;
}
