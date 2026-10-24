#pragma once

#include "llama.h"
#include "llama-hybrid-cycle-graph.h"

#include <array>
#include <cstddef>
#include <cstdint>

struct llama_layer;

// Shared physical-weight and KV ownership primitives for the V5 Cycle.
// They are part of libllama and usable by tests and a future llama_decode
// stage executor. Only physical CPU/CUDA/RPC buffers are returned: never a
// Meta parent tensor. Caller must keep the model/backend alive.
LLAMA_API ggml_tensor * llama_hybrid_cycle_resolve_physical_weight(
    const ggml_tensor * weight, ggml_backend_t backend, size_t source_device);

// CUDA mirrors are explicitly per-layer and must be budgeted; the fixed
// Cycle model's CPU weight placement is not silently moved to CUDA.
struct LLAMA_API llama_hybrid_cycle_cuda_full_weights {
    enum slot : size_t {
        ATTN_NORM, WQ, WK, WV, WO, Q_NORM, K_NORM, FFN_NORM, ROUTER,
        FFN_UP, FFN_GATE, FFN_DOWN, COUNT
    };

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::array<ggml_tensor *, COUNT> tensors {};

    llama_hybrid_cycle_cuda_full_weights() = default;
    llama_hybrid_cycle_cuda_full_weights(const llama_hybrid_cycle_cuda_full_weights &) = delete;
    llama_hybrid_cycle_cuda_full_weights & operator=(const llama_hybrid_cycle_cuda_full_weights &) = delete;
    ~llama_hybrid_cycle_cuda_full_weights() { release(); }

    void release();
    bool prepare(ggml_backend_t cuda, ggml_backend_t cpu,
                 const llama_layer & layer, int layer_index);
    llama_hybrid_cycle_ffn_shard_weights full_ffn_shard() const;
};

// Per-PHYSICAL-layer KV allocation. Lifetime is deliberately independent of
// transient Chunk scratch or graph replay objects. The physical causal K/V
// allocation supports up to 512 positions; the per-invocation chunk window
// remains <=64 tokens and <=16 tokens per physical Chunk.
struct LLAMA_API llama_hybrid_cycle_physical_kv_cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * keys = nullptr;
    ggml_tensor * values = nullptr;
    int64_t max_tokens = 0;
    int64_t head_dim = 0;
    int64_t kv_heads = 0;
    int64_t committed_tokens = 0;

    llama_hybrid_cycle_physical_kv_cache() = default;
    llama_hybrid_cycle_physical_kv_cache(const llama_hybrid_cycle_physical_kv_cache &) = delete;
    llama_hybrid_cycle_physical_kv_cache & operator=(const llama_hybrid_cycle_physical_kv_cache &) = delete;
    ~llama_hybrid_cycle_physical_kv_cache() { release(); }

    void release();
    bool init(ggml_backend_t backend, const llama_layer & layer,
              int64_t total_tokens, size_t source_device = 1,
              const llama_hybrid_cycle_cuda_full_weights * cuda_weights = nullptr);
};
