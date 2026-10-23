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
    // The production physical FFN helper must build the same intermediate-
    // channel graph used by C2P, P2C and full-Phone replay.
    constexpr int64_t mini_hidden = 32;
    constexpr int64_t mini_channels = 64;
    constexpr int64_t mini_experts = 4;
    constexpr int64_t mini_topk = 2;
    constexpr int64_t mini_tokens = 4;
    auto * mini_gate = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, mini_hidden, mini_channels, mini_experts);
    auto * mini_up = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, mini_hidden, mini_channels, mini_experts);
    auto * mini_down = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, mini_channels, mini_hidden, mini_experts);
    llama_hybrid_cycle_ffn_shard_weights mini_shard {
        mini_up, mini_gate, mini_down
    };
    auto mini_in = llama_hybrid_cycle_make_ffn_graph_inputs(
        ctx, mini_hidden, mini_tokens, mini_topk);
    ggml_tensor * stage_gate = nullptr;
    ggml_tensor * stage_up = nullptr;
    ggml_tensor * stage_swiglu = nullptr;
    ggml_tensor * stage_down = nullptr;
    assert(llama_hybrid_cycle_build_moe_ffn_graph(
        ctx, mini_in, mini_shard, mini_hidden, mini_tokens, mini_topk,
        &stage_gate, &stage_up, &stage_swiglu, &stage_down));
    assert(llama_hybrid_cycle_ffn_graph_valid(
        mini_in, mini_hidden, mini_tokens, mini_topk));
    assert(stage_gate && stage_up && stage_swiglu && stage_down);
    assert(ggml_nelements(mini_in.ffn_output) == mini_hidden * mini_tokens);
    auto * mini_graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(mini_graph, mini_in.ffn_output);
    assert(ggml_graph_n_nodes(mini_graph) > 0);
    assert(!llama_hybrid_cycle_build_moe_ffn_graph(
        nullptr, mini_in, mini_shard,
        mini_hidden, mini_tokens, mini_topk));

    // Output-owner contract: split Reduce is exactly two ADD nodes and
    // does not recursively re-execute the FFN shard.
    auto * residual = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, mini_hidden, mini_tokens);
    auto * partial = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, mini_hidden, mini_tokens);
    auto * split_graph = ggml_new_graph_custom(ctx, 512, false);
    const auto split = llama_hybrid_cycle_build_reduce_graph(
        ctx, split_graph, residual, mini_in.ffn_output,
        partial, false, false);
    assert(split.output && split.reduce_graph);
    assert(ggml_graph_n_nodes(split_graph) > 0);
    assert(ggml_graph_n_nodes(split.reduce_graph) == 2);
    assert(split.output->op == GGML_OP_ADD);
    assert(split.output->src[0] == residual);
    assert(split.output->src[1]->op == GGML_OP_ADD);
    assert((split.output->flags & GGML_TENSOR_FLAG_COMPUTE) != 0);

    // Complete layer uses only residual + FFN, optionally fused into the
    // sole replayable forward graph.
    auto * fused_graph = ggml_new_graph_custom(ctx, 512, false);
    const auto fused = llama_hybrid_cycle_build_reduce_graph(
        ctx, fused_graph, residual, mini_in.ffn_output,
        nullptr, true, true);
    assert(fused.output && !fused.reduce_graph);
    assert(fused.output->src[0] == residual);
    assert(fused.output->src[1] == mini_in.ffn_output);
    assert(ggml_graph_n_nodes(fused_graph) > 0);

    ggml_free(ctx);
    return 0;
}
