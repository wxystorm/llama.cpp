#pragma once

#include "llama-hybrid-cycle-phone-executor.h"

struct llama_hybrid_cycle_ffn_phase_profile {
    uint64_t count = 0;
    double graph_build_ms = 0.0;
    double backend_prepare_ms = 0.0;
    double backend_execute_call_ms = 0.0;
};

// P2C's Phone-local, persistent FFN graph slot. It borrows Attention READY
// tensors on Phone GPU; SEND_PARTIAL transfers directly to Phone's TX ring
// after the FFN result is computed. No PC readback or Meta all-reduce.
struct LLAMA_API llama_hybrid_cycle_p2c_ffn_slot {
    ggml_context * ctx = nullptr;
    llama_hybrid_cycle_ffn_executor executor;
    llama_hybrid_cycle_ffn_graph inputs {};
    int64_t prepared_tokens = 0;
    uint64_t registered_graph_uid = 0;
    double graph_build_ms = 0.0;
    double backend_prepare_ms = 0.0;
    double last_execute_call_ms = 0.0;
    double last_send_partial_rpc_ms = 0.0;
    llama_hybrid_cycle_phone_rpc_ops rpc_ops {};

    llama_hybrid_cycle_p2c_ffn_slot() = default;
    llama_hybrid_cycle_p2c_ffn_slot(const llama_hybrid_cycle_p2c_ffn_slot &) = delete;
    llama_hybrid_cycle_p2c_ffn_slot & operator=(const llama_hybrid_cycle_p2c_ffn_slot &) = delete;
    ~llama_hybrid_cycle_p2c_ffn_slot();

    void release();
    bool prepare(ggml_backend_t phone,
                 const llama_hybrid_cycle_ffn_shard_weights & shard,
                 int64_t hidden, int64_t tokens, int64_t topk,
                 int64_t experts, bool phone_local_chain = false);
    bool execute_and_send(
        ggml_backend_t phone,
        const llama_hybrid_cycle_ready_graph & ready,
        const llama_hybrid_cycle_live_wire & wire,
        llama_hybrid_cycle_ffn_phase_profile * profile);
};
