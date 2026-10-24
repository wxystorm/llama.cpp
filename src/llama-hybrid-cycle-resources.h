#pragma once

#include "llama-hybrid-cycle-attention-graph-cache.h"
#include "llama-hybrid-cycle-attention-scratch-pool.h"
#include "llama-hybrid-cycle-physical.h"
#include "llama-hybrid-cycle-runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_model;

// Legacy pure selector retained for policy/unit tests. Production Cycle no
// longer uses the selected bitmap as a CPU-vs-CUDA execution decision:
// every ordinary PC Full layer executes on CUDA. The budget instead sizes a
// bounded CUDA full-weight residency cache; misses synchronously replace one
// resident mirror on the single PC_CUDA FIFO worker.
LLAMA_API bool llama_hybrid_cycle_select_cuda_full_layers(
    const llama_hybrid_cycle_layout & layout,
    const std::vector<size_t> & estimated_mirror_bytes,
    size_t budget_bytes,
    std::vector<uint8_t> & selected,
    size_t & selected_bytes,
    std::string & reason);

struct llama_hybrid_cycle_resource_config {
    int64_t kv_tokens = 64;             // Physical KV supports <=512 tokens.
    int max_inflight = 2;               // Graph/Scratch credit slots per layer.
    size_t cuda_weight_budget_bytes = 0; // Effective residency-cache cap; GPU Full is mandatory.
    size_t cuda_safety_reserve_bytes = 512ULL * 1024 * 1024;
    bool attention_graph_cache = true;
    bool attention_scratch_pool = true;
    // Keep any legacy positional aggregate initializers in their original
    // order above; the new optional knob is deliberately appended last.
    // The Graph-cache OWNER remains present per layer even when max_graphs=0.
    size_t attention_graph_cache_max_graphs = 16;
};

// Backend and model pointers are borrowed. Their owners MUST outlive this
// pool, including clear(). Only the CUDA mirrors, per-layer K/V, host Graph
// metadata and lazy Scratch pools below are pool-owned.
struct llama_hybrid_cycle_physical_layer_resources {
    int physical_layer = -1;
    llama_hybrid_cycle_layer_kind kind =
        llama_hybrid_cycle_layer_kind::PC_CPU;
    ggml_backend_t attn_backend = nullptr;
    // Lazy resident mirror for ordinary PC Full. Ownership moves between
    // physical layers under the pool residency mutex; nullptr means a cache
    // miss, never permission to execute this Full layer on CPU.
    std::unique_ptr<llama_hybrid_cycle_cuda_full_weights> cuda_weights;
    uint64_t cuda_residency_stamp = 0;
    std::unique_ptr<llama_hybrid_cycle_physical_kv_cache> kv;
    // Deferred Phone-region registration may acquire/rebind Attention
    // scratch on a host control worker while the PHONE_GPU ACK path returns
    // the previous Chunk's lease. Protect only these host-side pool/cache
    // metadata mutations; GPU execution itself is never held under this lock.
    std::mutex attention_state_mutex;
    std::unique_ptr<cycle_real_attention_graph_cache> graph_cache;
    std::unique_ptr<cycle_real_attention_scratch_pool> scratch_pool;
};

// Preparing all 48 physical layers does NOT build/compute any model graph,
// does NOT open additional RPC connections and does NOT run llama_decode().
// It validates actual Meta weight placement and physically allocates the
// per-layer F16 KV tensors. Graph metadata/Scratch are created lazily.
// Call clear() only after the five FIFO workers and all GPU/RPC consumers
// have completed, while all three backends and the model are still alive.
class llama_hybrid_cycle_physical_resource_pool {
public:
    LLAMA_API ~llama_hybrid_cycle_physical_resource_pool();

    LLAMA_API bool prepare(
        const llama_model & model, const llama_hybrid_cycle_layout & layout,
        ggml_backend_t cpu, ggml_backend_t phone, ggml_backend_t cuda,
        const llama_hybrid_cycle_resource_config & config,
        std::string & reason);

    LLAMA_API void clear();

    LLAMA_API const llama_hybrid_cycle_layout & execution_layout() const;
    LLAMA_API const llama_hybrid_cycle_physical_layer_resources * layer(
        int physical_layer) const;

    // Ensure one ordinary PC Full layer has CUDA weights resident before its
    // PC_CUDA task executes. There is deliberately no CPU fallback.
    LLAMA_API bool ensure_pc_full_cuda_resident(
        int physical_layer, std::string & reason);

    LLAMA_API size_t cuda_mirror_bytes_planned() const;
    // Number of full-layer mirrors that may be resident simultaneously.
    LLAMA_API int cuda_mirror_layers() const;
    LLAMA_API int cuda_resident_layers() const;
    LLAMA_API uint64_t cuda_residency_loads() const;
    LLAMA_API uint64_t cuda_residency_evictions() const;
    LLAMA_API int prepared_layers() const;

private:
    llama_hybrid_cycle_layout layout_;
    std::vector<std::unique_ptr<llama_hybrid_cycle_physical_layer_resources>>
        layers_;

    const llama_model * model_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
    ggml_backend_t cuda_ = nullptr;

    mutable std::mutex cuda_residency_mutex_;
    uint64_t cuda_residency_clock_ = 0;
    uint64_t cuda_residency_loads_ = 0;
    uint64_t cuda_residency_evictions_ = 0;
    size_t cuda_mirror_bytes_planned_ = 0;
    int cuda_mirror_layers_ = 0;
};
