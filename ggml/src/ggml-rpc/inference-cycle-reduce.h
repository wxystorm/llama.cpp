#pragma once

#include "inference-cycle-payload.h"

#include <cstddef>

// P2C V1 reference Reduce: the PC CPU is the ONLY owner of the final
// Transformer-layer hidden state. Both FFN contributions are full-hidden
// partials from disjoint expert/intermediate shards.
//
// This routine consumes decoded, HOST-resident F32 data. It is deliberately
// not a generic GPU fallback: C2P must reduce on Phone GPU, not copy its
// partials back to the PC merely to call this function.
//
// The sum is performed in two steps to follow the split-FFN graph semantics:
//   ffn_sum = FFN_PC + FFN_PHONE;
//   l_out   = residual + ffn_sum;
// Accumulation order may differ slightly from a single-device reference.
inline bool inference_cycle_reduce_p2c_pc_cpu(
        const inference_cycle_ready_owned & ready,
        const float * pc_ffn_partial,
        size_t pc_partial_elements,
        const inference_cycle_partial_owned & phone_ffn_partial,
        float * output,
        size_t output_elements) {
    size_t n = 0;
    size_t routes = 0;
    if (!pc_ffn_partial || !output ||
        ready.tokens == 0 || ready.hidden == 0 ||
        ready.tokens != phone_ffn_partial.tokens ||
        ready.hidden != phone_ffn_partial.hidden ||
        !inference_cycle_mul(ready.tokens, ready.hidden, n) ||
        !inference_cycle_mul(ready.tokens, ready.topk, routes) ||
        ready.residual.size() != n ||
        ready.ffn_norm.size() != n ||
        ready.topk_ids.size() != routes ||
        ready.topk_weights.size() != routes ||
        phone_ffn_partial.partial.size() != n ||
        pc_partial_elements != n || output_elements < n) {
        return false;
    }

    for (size_t i = 0; i < n; ++i) {
        // Each device computed one disjoint FFN shard; no double counting.
        const float ffn_sum = pc_ffn_partial[i] + phone_ffn_partial.partial[i];
        output[i] = ready.residual[i] + ffn_sum;
    }
    return true;
}
