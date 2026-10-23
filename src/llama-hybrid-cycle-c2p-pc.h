#pragma once

#include "llama-hybrid-cycle-attention.h"
#include "llama-hybrid-cycle-ffn-shard.h"
#include "llama-hybrid-cycle-resources.h"

#include <cstdint>
#include <vector>

struct llama_model;

// C2P's real PC half for ONE absolute physical Tensor layer.
// Attention/Router and the PC FFN shard use the loaded CPU weights and
// persistent layer KV; the Phone FFN and its final Reduce are NOT run here.
// The caller owns READY/partial until the SEND and REDUCE dependencies end.
LLAMA_API bool llama_hybrid_cycle_execute_c2p_pc_attention(
    const llama_model & model,
    const llama_hybrid_cycle_physical_layer_resources & resource,
    ggml_backend_t pc_cpu,
    int physical_layer,
    int64_t token_start,
    int64_t tokens,
    const std::vector<float> & input,
    llama_hybrid_cycle_attention_ready & ready);

LLAMA_API bool llama_hybrid_cycle_execute_c2p_pc_ffn_source(
    const llama_model & model,
    const llama_hybrid_cycle_physical_layer_resources & resource,
    ggml_backend_t pc_cpu,
    int physical_layer,
    int64_t tokens,
    const llama_hybrid_cycle_attention_ready & ready,
    std::vector<float> & partial);
