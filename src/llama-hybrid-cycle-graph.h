#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

// Qwen3-MoE's P2C AttentionPreGraph exports four separate tensors as the
// SEND_ACTIVATION payload. This contract intentionally contains no FFN output,
// no Reduce, and no wire layout ambiguity.
//
// GGML storage order for [hidden,tokens] is already token-major: dimension 0
// (hidden) is contiguous. Router weights [1,topk,tokens] flatten identically
// to [tokens,topk] in inference-cycle-payload.h.
struct llama_hybrid_cycle_ready_graph {
    ggml_tensor * residual     = nullptr;
    ggml_tensor * ffn_norm     = nullptr;
    ggml_tensor * topk_ids     = nullptr;
    ggml_tensor * topk_weights = nullptr;
};

inline bool llama_hybrid_cycle_ready_graph_valid(
        const llama_hybrid_cycle_ready_graph & graph,
        int64_t hidden, int64_t tokens, int64_t topk) {
    if (hidden <= 0 || tokens <= 0 || topk <= 0 ||
        !graph.residual || !graph.ffn_norm ||
        !graph.topk_ids || !graph.topk_weights) {
        return false;
    }

    const ggml_tensor * r = graph.residual;
    const ggml_tensor * n = graph.ffn_norm;
    const ggml_tensor * i = graph.topk_ids;
    const ggml_tensor * w = graph.topk_weights;
    return r->type == GGML_TYPE_F32 && n->type == GGML_TYPE_F32 &&
           i->type == GGML_TYPE_I32 && w->type == GGML_TYPE_F32 &&
           r->ne[0] == hidden && r->ne[1] == tokens &&
           n->ne[0] == hidden && n->ne[1] == tokens &&
           i->ne[0] == topk && i->ne[1] == tokens &&
           w->ne[0] == 1 && w->ne[1] == topk && w->ne[2] == tokens &&
           r->ne[2] == 1 && r->ne[3] == 1 &&
           n->ne[2] == 1 && n->ne[3] == 1 &&
           i->ne[2] == 1 && i->ne[3] == 1 && w->ne[3] == 1 &&
           ggml_is_contiguous(r) && ggml_is_contiguous(n) &&
           ggml_is_contiguous(i) && ggml_is_contiguous(w);
}


// One-way P2C FFN input boundary. The raw model MoE graph consumes these
// tensors directly; it must NOT recompute the Router or add the residual.
//
// The graph builder selects a specific physical CPU/Phone shard of all three
// MoE expert matrices. The executor must run the result on that same physical
// backend and wait for the output buffer to become device-ready before
// publishing FFN_SOURCE or FFN_DESTINATION completion.
struct llama_hybrid_cycle_ffn_graph {
    ggml_tensor * norm          = nullptr; // F32 [hidden, tokens]
    ggml_tensor * topk_ids      = nullptr; // I32 [topk, tokens]
    ggml_tensor * topk_weights  = nullptr; // F32 [1, topk, tokens]
    ggml_tensor * ffn_output    = nullptr; // F32 [hidden, tokens]
};

inline llama_hybrid_cycle_ffn_graph llama_hybrid_cycle_make_ffn_graph_inputs(
        ggml_context * ctx, int64_t hidden, int64_t tokens, int64_t topk) {
    if (ctx == nullptr || hidden <= 0 || tokens <= 0 || topk <= 0) {
        return {};
    }
    llama_hybrid_cycle_ffn_graph result;
    result.norm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    result.topk_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, tokens);
    result.topk_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, topk, tokens);
    ggml_set_input(result.norm);
    ggml_set_input(result.topk_ids);
    ggml_set_input(result.topk_weights);
    return result;
}

inline bool llama_hybrid_cycle_ffn_graph_valid(
        const llama_hybrid_cycle_ffn_graph & g,
        int64_t hidden, int64_t tokens, int64_t topk) {
    if (hidden <= 0 || tokens <= 0 || topk <= 0 ||
        !g.norm || !g.topk_ids || !g.topk_weights || !g.ffn_output) {
        return false;
    }
    return g.norm->type == GGML_TYPE_F32 &&
           g.topk_ids->type == GGML_TYPE_I32 &&
           g.topk_weights->type == GGML_TYPE_F32 &&
           g.ffn_output->type == GGML_TYPE_F32 &&
           g.norm->ne[0] == hidden && g.norm->ne[1] == tokens &&
           g.topk_ids->ne[0] == topk && g.topk_ids->ne[1] == tokens &&
           g.topk_weights->ne[0] == 1 && g.topk_weights->ne[1] == topk &&
           g.topk_weights->ne[2] == tokens &&
           g.ffn_output->ne[0] == hidden && g.ffn_output->ne[1] == tokens &&
           g.norm->ne[2] == 1 && g.norm->ne[3] == 1 &&
           g.topk_ids->ne[2] == 1 && g.topk_ids->ne[3] == 1 &&
           g.topk_weights->ne[3] == 1 &&
           g.ffn_output->ne[2] == 1 && g.ffn_output->ne[3] == 1 &&
           ggml_is_contiguous(g.norm) &&
           ggml_is_contiguous(g.topk_ids) &&
           ggml_is_contiguous(g.topk_weights) &&
           ggml_is_contiguous(g.ffn_output);
}

 
// Qwen3-MoE Expert weights are partitioned along the intermediate-neuron
// dimension, not expert IDs:
//   gate/up: [hidden, local_ffn, experts]
//   down:    [local_ffn, hidden, experts]
// Thus each device runs the same Top-K expert IDs, but only its own local
// intermediate channels. Their F32 down-projection results ADD to the full
// FFN output. Splitting expert IDs instead would require remapping Top-K.
struct llama_hybrid_cycle_ffn_shard_weights {
    ggml_tensor * up   = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * down = nullptr;

    bool available() const { return up && gate && down; }
};

// Build exactly one physical MoE FFN intermediate-channel shard. Router
// inputs (Norm, TopK ids/weights) are borrowed from Attention; neither
// Router nor residual ADD is recomputed here. Gate/Up/SwiGLU/Down outputs
// remain optional handles for existing numerical diagnostics.
inline bool llama_hybrid_cycle_build_moe_ffn_graph(
        ggml_context * ctx,
        llama_hybrid_cycle_ffn_graph & in,
        const llama_hybrid_cycle_ffn_shard_weights & shard,
        int64_t hidden, int64_t tokens, int64_t topk,
        ggml_tensor ** gate_out = nullptr,
        ggml_tensor ** up_out = nullptr,
        ggml_tensor ** swiglu_out = nullptr,
        ggml_tensor ** down_out = nullptr) {
    if (!ctx || !shard.available() ||
        !in.norm || !in.topk_ids || !in.topk_weights ||
        hidden <= 0 || tokens <= 0 || topk <= 0) {
        return false;
    }
    auto * x = ggml_reshape_3d(ctx, in.norm, hidden, 1, tokens);
    auto * gate = ggml_mul_mat_id(ctx, shard.gate, x, in.topk_ids);
    auto * up = ggml_mul_mat_id(ctx, shard.up, x, in.topk_ids);
    auto * swiglu = ggml_swiglu_split(ctx, gate, up);
    auto * down = ggml_mul_mat_id(ctx, shard.down, swiglu, in.topk_ids);
    auto * weighted = ggml_mul(ctx, down, in.topk_weights);
    ggml_tensor * sum = nullptr;
    for (int64_t k = 0; k < topk; ++k) {
        auto * view = ggml_view_2d(
            ctx, weighted, hidden, tokens, weighted->nb[2],
            size_t(k) * weighted->nb[1]);
        sum = sum ? ggml_add(ctx, sum, view) : view;
    }
    in.ffn_output = ggml_cont(ctx, sum);
    ggml_set_name(in.ffn_output, "cycle_num_phone_ffn_output");
    if (gate_out) *gate_out = gate;
    if (up_out) *up_out = up;
    if (swiglu_out) *swiglu_out = swiglu;
    if (down_out) *down_out = down;
    return true;
}

// Build the ONE output-owner Reduce contract without recomputing FFN.
// A split Tensor layer computes ADD(local FFN, remote FFN) + residual;
// a complete Phone layer computes local FFN + residual. The separate
// Reduce graph intentionally includes only the final ADD nodes, with
// GGML_TENSOR_FLAG_COMPUTE set exactly as in the V5 implementation.
struct llama_hybrid_cycle_reduce_graph {
    ggml_tensor * output = nullptr;
    ggml_cgraph * reduce_graph = nullptr;
};

inline llama_hybrid_cycle_reduce_graph llama_hybrid_cycle_build_reduce_graph(
        ggml_context * ctx, ggml_cgraph * ffn_graph,
        ggml_tensor * residual, ggml_tensor * local_ffn,
        ggml_tensor * remote_ffn, bool full_layer, bool fuse_full) {
    if (!ctx || !ffn_graph || !residual || !local_ffn ||
        (!full_layer && !remote_ffn)) {
        return {};
    }
    auto * ffnsum = full_layer ?
        local_ffn : ggml_add(ctx, local_ffn, remote_ffn);
    auto * output = ggml_add(ctx, residual, ffnsum);
    ggml_set_name(output, full_layer ?
        "cycle_num_phone_full_lout" : "cycle_num_c2p_lout");
    if (fuse_full) {
        ggml_build_forward_expand(ffn_graph, output);
        return {output, nullptr};
    }
    ggml_build_forward_expand(ffn_graph, local_ffn);
    auto * reduce_graph = ggml_new_graph_custom(ctx, 16, false);
    if (!full_layer) {
        ffnsum->flags |= GGML_TENSOR_FLAG_COMPUTE;
        ggml_graph_add_node(reduce_graph, ffnsum);
    }
    output->flags |= GGML_TENSOR_FLAG_COMPUTE;
    ggml_graph_add_node(reduce_graph, output);
    return {output, reduce_graph};
}

inline llama_hybrid_cycle_ffn_shard_weights llama_hybrid_cycle_get_physical_ffn_weights(
        const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * down, size_t device_index) {
    return {
        ggml_backend_meta_get_static_weight_shard(up, device_index),
        ggml_backend_meta_get_static_weight_shard(gate, device_index),
        ggml_backend_meta_get_static_weight_shard(down, device_index),
    };
}

inline bool llama_hybrid_cycle_ffn_shard_weights_valid(
        const llama_hybrid_cycle_ffn_shard_weights & w,
        int64_t hidden, int64_t experts) {
    if (hidden <= 0 || experts <= 0 || !w.available()) {
        return false;
    }
    const int64_t channels = w.up->ne[1];
    if (channels <= 0 ||
        w.up->ne[0] != hidden || w.up->ne[2] != experts ||
        w.gate->ne[0] != hidden || w.gate->ne[1] != channels ||
        w.gate->ne[2] != experts ||
        w.down->ne[0] != channels || w.down->ne[1] != hidden ||
        w.down->ne[2] != experts ||
        w.up->ne[3] != 1 || w.gate->ne[3] != 1 ||
        w.down->ne[3] != 1 ||
        !ggml_is_contiguous(w.up) ||
        !ggml_is_contiguous(w.gate) ||
        !ggml_is_contiguous(w.down)) {
        return false;
    }
    if (!w.up->buffer || !w.gate->buffer || !w.down->buffer) {
        return false;
    }
    // All three weights must share one physical buffer type. Non-default
    // CPU host/mmap/extra buffers may be deviceless but are still valid if
    // the CPU device supports them. GPU/RPC buffers must have a real device.
    const auto up_buft = ggml_backend_buffer_get_type(w.up->buffer);
    const auto gate_buft = ggml_backend_buffer_get_type(w.gate->buffer);
    const auto down_buft = ggml_backend_buffer_get_type(w.down->buffer);
    if (up_buft == nullptr || up_buft != gate_buft || up_buft != down_buft) {
        return false;
    }
    const auto dev = ggml_backend_buft_get_device(up_buft);
    if (dev != nullptr) {
        return ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_META;
    }
    const auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    return cpu_dev != nullptr && ggml_backend_dev_supports_buft(cpu_dev, up_buft);
}


// Ensure two independently borrowed device-local weight sets form a complete
// disjoint intermediate-neuron partition. Both down projections must sum to
// the original hidden dimension, with no missing or repeated channels.
inline bool llama_hybrid_cycle_ffn_partition_valid(
        const ggml_tensor * up, const ggml_tensor * gate,
        const ggml_tensor * down, int64_t hidden, int64_t experts) {
    if (!up || !gate || !down ||
        up->ne[0] != hidden || up->ne[2] != experts ||
        gate->ne[0] != hidden || gate->ne[2] != experts ||
        down->ne[1] != hidden || down->ne[2] != experts ||
        up->ne[1] != gate->ne[1] || up->ne[1] != down->ne[0]) {
        return false;
    }
    const auto pc = llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 0);
    const auto phone = llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 1);
    return llama_hybrid_cycle_ffn_shard_weights_valid(pc, hidden, experts) &&
           llama_hybrid_cycle_ffn_shard_weights_valid(phone, hidden, experts) &&
           pc.up->ne[1] + phone.up->ne[1] == up->ne[1] &&
           pc.gate->ne[1] + phone.gate->ne[1] == gate->ne[1] &&
           pc.down->ne[0] + phone.down->ne[0] == down->ne[0];
}
