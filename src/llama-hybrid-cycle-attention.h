#pragma once
#include "llama-hybrid-cycle-physical.h"
#include "llama-hybrid-cycle-attention-scratch-pool.h"
#include "llama-hybrid-cycle-attention-graph-cache.h"
#include <cstdint>
#include <functional>
#include <vector>

struct llama_layer;
struct llama_hparams;

// RPC hooks are injected by the caller so libllama never depends on
// ggml-rpc at link time. The physical Graph/READY buffers remain borrowed.
struct llama_hybrid_cycle_attention_rpc_ops {
    bool is_rpc = false;
    bool (*register_graph)(ggml_backend_t, const ggml_cgraph *) = nullptr;
    bool (*send_ready)(ggml_backend_t, int32_t, int32_t, uint64_t, uint64_t,
                       const ggml_tensor *, const ggml_tensor *,
                       const ggml_tensor *, const ggml_tensor *) = nullptr;
};

struct llama_hybrid_cycle_attention_ready {
    std::vector<float> residual;
    std::vector<float> norm;
    std::vector<int32_t> ids;
    std::vector<float> routing;
    // Full pre-TopK Softmax distribution for score-margin diagnosis only.
    // Not serialized in READY and never used by the FFN execution.
    std::vector<float> router_probs;

    // Keep READY's Phone GPU tensors alive through FFN local input copies.
    ggml_context * phone_ctx = nullptr;
    ggml_backend_buffer_t phone_compute_buffer = nullptr;
    cycle_real_attention_scratch_pool * phone_scratch_pool = nullptr;
    cycle_real_attention_graph_cache * phone_graph_cache = nullptr;
    cycle_real_attention_graph_cache::slot * phone_cached_graph = nullptr;
    llama_hybrid_cycle_ready_graph phone_graph {};
    // V2: the next Full layer's Attention was registered but not submitted.
    // These are borrowed only until the Phone worker's completion ACK.
    uint64_t prepared_attention_graph_uid = 0;
    ggml_tensor * prepared_attention_input = nullptr;

    llama_hybrid_cycle_attention_ready() = default;
    llama_hybrid_cycle_attention_ready(const llama_hybrid_cycle_attention_ready &) = delete;
    llama_hybrid_cycle_attention_ready & operator=(const llama_hybrid_cycle_attention_ready &) = delete;
    ~llama_hybrid_cycle_attention_ready() { release_phone_graph(); }

    void release_phone_graph() {
        // FFN already consumed/fenced READY before handing its buffer back.
        // Graph metadata must not outlive the memory it describes.
        if (phone_graph_cache && phone_cached_graph) {
            // This graph is session-owned. Drop tensor bindings so another
            // Chunk can attach a fresh scratch lease without stale RPC
            // pointers. Do not destroy its ggml_context.
            phone_graph_cache->recycle(phone_cached_graph, true);
        } else if (phone_ctx != nullptr) {
            ggml_free(phone_ctx);
        }
        phone_ctx = nullptr;
        phone_graph_cache = nullptr;
        phone_cached_graph = nullptr;
        if (phone_compute_buffer != nullptr) {
            if (phone_scratch_pool) {
                phone_scratch_pool->recycle(phone_compute_buffer);
            } else {
                ggml_backend_buffer_free(phone_compute_buffer);
            }
            phone_compute_buffer = nullptr;
        }
        phone_scratch_pool = nullptr;
        phone_graph = {};
        prepared_attention_graph_uid = 0;
        prepared_attention_input = nullptr;
    }
};

struct llama_hybrid_cycle_live_wire {
    int32_t layer = -1;
    int32_t chunk = 0;
    uint64_t seq = 1;
    uint64_t graph_id = 1;
};

struct llama_hybrid_cycle_attention_phase_profile {
    uint64_t count = 0;
    double graph_build_ms = 0.0;
    double alloc_ms = 0.0;
    double input_copy_ms = 0.0;
    double graph_call_ms = 0.0;
};

// Unique across Phone Attention and FFN/Reduce replay in one process.
LLAMA_API uint64_t llama_hybrid_cycle_graph_replay_uid();

// Physical Qwen3-MoE attention/router, one contiguous causal Chunk.
// Physical K/V may contain up to 512 historical tokens. One Attention call
// still processes a small Chunk (<=16 tokens in native Cycle Forward).
LLAMA_API bool llama_hybrid_cycle_execute_attention_pre(
        ggml_backend_t backend, const llama_layer & layer,
        const llama_hparams & hp, const std::vector<float> & layer_input,
        int64_t hidden, int64_t tokens, int64_t topk, int64_t experts,
        llama_hybrid_cycle_attention_ready & result, double & elapsed_ms,
        const llama_hybrid_cycle_live_wire * wire = nullptr,
        llama_hybrid_cycle_physical_kv_cache * kv_cache = nullptr,
        int64_t kv_token_start = 0, size_t source_device = 1,
        const ggml_tensor * device_input = nullptr,
        bool retain_device_ready = false,
        llama_hybrid_cycle_attention_phase_profile * phase_profile = nullptr,
        const llama_hybrid_cycle_cuda_full_weights * cuda_weights = nullptr,
        cycle_real_attention_scratch_pool * scratch_pool = nullptr,
        cycle_real_attention_graph_cache * graph_cache = nullptr,
        const std::function<bool(const llama_hybrid_cycle_ready_graph &)> &
            before_phone_graph_compute = {},
        bool defer_registered_replay = false,
        const llama_hybrid_cycle_attention_rpc_ops & rpc_ops = {});
