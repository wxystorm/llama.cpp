#pragma once

#include "llama-hybrid-cycle-p2c-ffn.h"

#include <cstdint>
#include <vector>

// Physical MoE FFN operator: Gate, Up, SwiGLU, Down and weighted Top-K
// reduction on one exact backend. Norm/Router are computed by AttentionPre.
// This is shared by the PC CPU source/destination FFN and diagnostic fallback.
struct llama_hybrid_cycle_ffn_stages {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> swiglu;
};

// The Phone P2C SEND_PARTIAL operation lives in a dynamically loaded RPC
// plugin; libllama owns the graph but must not depend on that plugin's link
// symbols. Non-Phone calls leave this hook empty.
struct llama_hybrid_cycle_ffn_rpc_ops {
    bool (*send_partial)(
        ggml_backend_t, int32_t, int32_t, uint64_t, uint64_t,
        const ggml_tensor *) = nullptr;
};

// Any borrowed backend/model shards must outlive this call. The temporary
// graph allocation is released before returning. Phone RPC results are
// transferred only by send_partial, never by a redundant host readback.
LLAMA_API bool llama_hybrid_cycle_execute_ffn_shard(
        ggml_backend_t backend,
        const llama_hybrid_cycle_ffn_shard_weights & shard,
        const std::vector<float> & norm,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        int64_t hidden, int64_t tokens,
        int64_t topk, int64_t experts,
        std::vector<float> & partial,
        std::vector<float> * raw_down_out = nullptr,
        llama_hybrid_cycle_ffn_stages * stages_out = nullptr,
        const llama_hybrid_cycle_live_wire * wire = nullptr,
        const llama_hybrid_cycle_ready_graph * local_phone_ready = nullptr,
        llama_hybrid_cycle_ffn_phase_profile * phase_profile = nullptr,
        const llama_hybrid_cycle_ffn_rpc_ops & rpc_ops = {});
