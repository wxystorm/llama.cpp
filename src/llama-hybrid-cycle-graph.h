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
