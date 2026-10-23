#pragma once

#include "llama-hybrid-cycle-attention.h"
#include "llama-hybrid-cycle-ffn-shard.h"
#include "llama-hybrid-cycle-resources.h"

#include <cstdint>
#include <vector>

struct llama_model;

// Physical P2C Phone Attention: consumes previous Full layer's GPU l_out,
// executes physical causal Attention/Router and stages a real Phone->PC READY.
// The caller MUST keep GPU READY until its own Phone FFN_SOURCE is complete.
LLAMA_API bool llama_hybrid_cycle_execute_p2c_phone_attention(
    const llama_model & model,
    const llama_hybrid_cycle_physical_layer_resources & resource,
    ggml_backend_t phone, int physical_layer,
    int64_t token_start, int64_t tokens,
    const ggml_tensor * previous_output,
    const llama_hybrid_cycle_live_wire & wire,
    llama_hybrid_cycle_attention_ready & ready,
    const llama_hybrid_cycle_attention_rpc_ops & rpc);

// PC destination FFN consumes the decoded READY from the other device.
// It uses only the physical CPU shard of this P2C Tensor layer.
LLAMA_API bool llama_hybrid_cycle_execute_p2c_pc_ffn(
    const llama_model & model,
    const llama_hybrid_cycle_physical_layer_resources & resource,
    ggml_backend_t cpu, int physical_layer, int64_t tokens,
    const std::vector<float> & norm,
    const std::vector<int32_t> & ids,
    const std::vector<float> & routing,
    std::vector<float> & partial);
