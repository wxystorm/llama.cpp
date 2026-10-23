#pragma once

#include "llama-hybrid-cycle-phone-executor.h"
#include "llama-hybrid-cycle-resources.h"

#include <cstdint>
#include <functional>

struct llama_model;

// REAL Phone Full execution for one absolute physical layer/chunk. The input
// is the preceding C2P or Phone Full slot's device l_out; no host activation
// transfer is allowed here. Caller keeps the producer slot alive through
// completion, then may retire it to its OWN physical region's idle bank.
//
// The caller owns READY and the output FFN slot. This function may reuse a
// shape-compatible FFN slot, but MUST NOT move/free the output before the
// successor Attention (or an explicit diagnostic bridge) consumes its l_out.
// The full layer commits its physical KV in strict Chunk order.
LLAMA_API bool llama_hybrid_cycle_execute_phone_full(
    const llama_model & model,
    const llama_hybrid_cycle_physical_layer_resources & resource,
    ggml_backend_t phone,
    int physical_layer,
    int64_t position_start,
    int64_t tokens,
    const ggml_tensor * device_input,
    llama_hybrid_cycle_attention_ready & ready,
    llama_hybrid_cycle_phone_ffn_slot & output_slot,
    const std::function<void(const ggml_tensor *, const ggml_tensor *,
                             const ggml_tensor *)> & stage_probe = {});
