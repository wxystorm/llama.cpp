#pragma once

#include "llama-hybrid-cycle-attention-graph-cache.h"
#include "llama-hybrid-cycle-attention-scratch-pool.h"
#include "llama-hybrid-cycle-physical.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct llama_model;

struct llama_region_cuda_mirror_timing {
    double attention_ms = 0.0;
    double ffn_ms = 0.0;
    double output_readback_ms = 0.0;
    double output_device_copy_ms = 0.0;
};

// PC Full-only CUDA mirror used by the repeating Region cycle.
//
// Model weights remain canonical on PC CPU. A bounded LRU mirrors complete
// physical PC Full layers into CUDA. Tensor C2P/P2C and Phone Region graphs
// never enter this object and remain owned by the existing Meta/local-MAIN
// path.
class llama_region_cuda_mirror {
public:
    llama_region_cuda_mirror() = default;
    llama_region_cuda_mirror(const llama_region_cuda_mirror &) = delete;
    llama_region_cuda_mirror & operator=(const llama_region_cuda_mirror &) = delete;
    ~llama_region_cuda_mirror();

    bool prepare(
        const llama_model & model,
        ggml_backend_t cpu,
        ggml_backend_t cuda,
        int64_t max_tokens,
        size_t requested_budget_bytes,
        std::string & reason);

    void clear();

    bool execute_block(
        int layer_begin,
        int layer_end,
        int64_t token_start,
        int64_t tokens,
        const std::vector<float> & input,
        std::vector<float> & output,
        std::string & reason,
        int chunk_tokens = 128);

    int resident_capacity() const;
    int resident_layers() const;
    uint64_t residency_loads() const;
    uint64_t residency_evictions() const;
    size_t planned_bytes() const;

private:
    struct layer_resource {
        int layer = -1;
        std::unique_ptr<llama_hybrid_cycle_cuda_full_weights> weights;
        uint64_t stamp = 0;
        std::unique_ptr<llama_hybrid_cycle_physical_kv_cache> kv;
        std::unique_ptr<cycle_real_attention_scratch_pool> scratch;
    };

    bool ensure_resident(int layer, std::string & reason);
    bool execute_layer(
        int layer,
        int64_t token_start,
        int64_t tokens,
        const std::vector<float> & input,
        std::vector<float> & output,
        llama_region_cuda_mirror_timing * timing,
        const ggml_tensor * device_input,
        llama_hybrid_cycle_cuda_activation_slot * device_output,
        std::string & reason);

    const llama_model * model_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
    ggml_backend_t cuda_ = nullptr;
    std::vector<std::unique_ptr<layer_resource>> layers_;

    mutable std::mutex mutex_;
    uint64_t clock_ = 0;
    uint64_t loads_ = 0;
    uint64_t evictions_ = 0;
    int resident_capacity_ = 0;
    size_t planned_bytes_ = 0;
    int64_t max_tokens_ = 0;
};
