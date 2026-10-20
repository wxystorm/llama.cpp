#pragma once

#include "ggml.h"

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
// IMPORTANT: until the per-backend weight-shard binding is implemented, the
// output of the model FFN graph is NOT YET guaranteed to be a *local* partial.
// An executor must isolate the PC/Phone expert weight fragments and choose the
// appropriate backend before publishing FFN_SOURCE or FFN_DESTINATION done.
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
