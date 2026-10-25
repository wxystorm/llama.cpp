#pragma once

#include "llama-hybrid-cycle-attention.h"
#include "llama-hybrid-cycle-ffn-executor.h"

// Pluggable RPC Graph/Phone MAIN operations. libllama never directly links
// against the ggml-rpc plugin; the caller binds its loaded plugin functions.
struct llama_hybrid_cycle_phone_rpc_ops {
    bool (*register_graph)(ggml_backend_t, const ggml_cgraph *) = nullptr;
    bool (*send_partial)(ggml_backend_t, int32_t, int32_t, uint64_t, uint64_t,
                         const ggml_tensor *) = nullptr;
    bool (*run_full)(ggml_backend_t, int32_t, int32_t, uint64_t, uint64_t,
                     const ggml_tensor * const[4],
                     const ggml_tensor * const[4]) = nullptr;
    bool (*run_full_chain)(ggml_backend_t, int32_t, int32_t, uint64_t, uint64_t,
                           const ggml_tensor * const[4],
                           const ggml_tensor * const[4],
                           int32_t, uint64_t, uint64_t, uint64_t,
                           const ggml_tensor *, const ggml_tensor *) = nullptr;
    bool (*run_full_pair)(ggml_backend_t, int32_t, int32_t, uint64_t, uint64_t,
                          const ggml_tensor * const[4],
                          const ggml_tensor * const[4],
                          int32_t, uint64_t, uint64_t, uint64_t,
                          const ggml_tensor *, const ggml_tensor *, uint64_t,
                          const ggml_tensor * const[4],
                          const ggml_tensor * const[4]) = nullptr;
};

// One PHYSICAL layer's Phone FFN/Reduce Graph Slot. Attention READY is
// borrowed; the Graph and its compute buffer remain owned by this object
// until downstream consumption/fence. Never release an active GPU lease.
struct LLAMA_API llama_hybrid_cycle_phone_ffn_slot {
    ggml_context * ctx = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * pc_partial = nullptr;
    ggml_tensor * output = nullptr;
    llama_hybrid_cycle_ffn_graph inputs {};
    ggml_cgraph * reduce_graph = nullptr;
    llama_hybrid_cycle_ffn_executor executor;
    double graph_build_ms = 0.0;
    double gpu_alloc_ms = 0.0;
    double ffn_graph_submit_ms = 0.0;
    double reduce_graph_submit_ms = 0.0;
    double reduce_graph_call_ms = 0.0;
    double reduce_backend_sync_ms = 0.0;
    bool is_full_layer = false;
    bool fused_full_graph = false;
    bool rpc_full_replay = false;
    uint64_t registered_full_graph_uid = 0;
    uint64_t registered_c2p_ffn_uid = 0;
    uint64_t registered_c2p_reduce_uid = 0;
    int64_t prepared_tokens = 0;
    llama_hybrid_cycle_phone_rpc_ops rpc_ops {};

    llama_hybrid_cycle_phone_ffn_slot() = default;
    llama_hybrid_cycle_phone_ffn_slot(const llama_hybrid_cycle_phone_ffn_slot &) = delete;
    llama_hybrid_cycle_phone_ffn_slot & operator=(const llama_hybrid_cycle_phone_ffn_slot &) = delete;
    ~llama_hybrid_cycle_phone_ffn_slot();

    void release();
    bool prepare(
            ggml_backend_t phone,
            const llama_hybrid_cycle_ffn_shard_weights & shard,
            int64_t hidden, int64_t tokens, int64_t topk, int64_t experts,
            bool full_layer = false, bool fuse_full = false,
            bool enable_rpc_replay = false, bool enable_phone_full_local = false,
            bool enable_c2p_local = false);
    bool execute_full_local(const llama_hybrid_cycle_attention_ready & ready);
    bool execute_full_server(
            ggml_backend_t phone, const llama_hybrid_cycle_attention_ready & ready,
            int layer, int chunk, uint64_t seq);
    bool execute_full_server_chain(
            ggml_backend_t phone, const llama_hybrid_cycle_attention_ready & ready,
            int layer, int chunk, uint64_t seq,
            int next_layer, uint64_t next_seq, uint64_t next_graph_id,
            const llama_hybrid_cycle_attention_ready & next_ready);
    bool execute_full_server_pair(
            ggml_backend_t phone, const llama_hybrid_cycle_attention_ready & ready,
            int layer, int chunk, uint64_t seq,
            int next_layer, uint64_t next_seq, uint64_t next_graph_id,
            const llama_hybrid_cycle_attention_ready & next_ready,
            const llama_hybrid_cycle_phone_ffn_slot & next_full);
    bool execute_ffn();
    bool execute_reduce(ggml_backend_t phone);
};
