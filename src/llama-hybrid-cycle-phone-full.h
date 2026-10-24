#pragma once

#include "llama-hybrid-cycle-phone-executor.h"
#include "llama-hybrid-cycle-resources.h"

#include <cstdint>
#include <functional>

struct llama_model;

// REAL Phone Full execution for one absolute physical layer/chunk. The input
// is the preceding C2P or Phone Full slot's device l_out; no host activation
// transfer is allowed here. Caller keeps the producer slot alive until the
// Attention fence, then retires it to its OWN physical region's idle bank
// through input_consumed while this layer's FFN continues.
//
// The caller owns READY and the output FFN slot. Production Full uses a
// registered server-side fused FFN+residual replay (RPC patch >=28); there is
// no legacy client FFN + separate Reduce fallback. A shape-compatible
// registered slot may be reused, but its output must remain alive until the
// successor Attention consumes l_out. KV commits remain strict Chunk order.
LLAMA_API bool llama_hybrid_cycle_execute_phone_full(
    const llama_model & model,
    const llama_hybrid_cycle_physical_layer_resources & resource,
    ggml_backend_t phone,
    int physical_layer,
    int chunk,
    uint64_t seq,
    int64_t position_start,
    int64_t tokens,
    const ggml_tensor * device_input,
    llama_hybrid_cycle_attention_ready & ready,
    llama_hybrid_cycle_phone_ffn_slot & output_slot,
    const std::function<void(const ggml_tensor *, const ggml_tensor *,
                             const ggml_tensor *)> & stage_probe = {},
    const std::function<bool()> & input_consumed = {});
