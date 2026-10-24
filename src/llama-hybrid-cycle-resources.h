#pragma once

#include "llama-hybrid-cycle-attention-graph-cache.h"
#include "llama-hybrid-cycle-attention-scratch-pool.h"
#include "llama-hybrid-cycle-physical.h"
#include "llama-hybrid-cycle-runtime.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// A budget is an upper bound, never a request to allocate all PC layers on
// CUDA. Zero is the safe default: PC full layers execute on the CPU. A
// nonempty input layout.pc_cuda_layers is treated as a REQUIRED selection
// and is rejected if it exceeds the budget.
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
    size_t cuda_weight_budget_bytes = 0; // 0 -> CPU fallback, never mirror all.
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
    std::unique_ptr<llama_hybrid_cycle_cuda_full_weights> cuda_weights;
    std::unique_ptr<llama_hybrid_cycle_physical_kv_cache> kv;
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
    LLAMA_API size_t cuda_mirror_bytes_planned() const;
    LLAMA_API int cuda_mirror_layers() const;
    LLAMA_API int prepared_layers() const;

private:
    llama_hybrid_cycle_layout layout_;
    std::vector<std::unique_ptr<llama_hybrid_cycle_physical_layer_resources>>
        layers_;
    size_t cuda_mirror_bytes_planned_ = 0;
    int cuda_mirror_layers_ = 0;
};
