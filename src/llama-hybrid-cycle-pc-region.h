#pragma once

#include "llama-hybrid-cycle-resources.h"
#include "llama-hybrid-cycle-execution-session.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// One PC CUDA Region executes a contiguous run of ordinary Full layers as
// ONE backend graph. V2 caches graph topology by (query_tokens, kv_tokens)
// and rebinds borrowed weight/KV aliases to the current physical Region.
// No cached graph owns model weights, so the bounded CUDA LRU may continue
// evicting/reloading mirrors between calls. Only Region input/output cross
// the graph boundary.
struct llama_hybrid_cycle_pc_region_timing {
    double graph_build_ms = 0.0;
    double alloc_ms = 0.0;
    double input_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
    int graph_nodes = 0;
    bool cache_hit = false;
    size_t allocator_bytes = 0;
};

struct llama_hybrid_cycle_pc_region_cache_stats {
    uint64_t hits = 0;
    uint64_t builds = 0;
    uint64_t evictions = 0;
    uint64_t allocator_grows = 0;
    size_t allocator_bytes = 0;
};

class LLAMA_API llama_hybrid_cycle_pc_region_cache {
public:
    explicit llama_hybrid_cycle_pc_region_cache(size_t max_graphs = 2);
    ~llama_hybrid_cycle_pc_region_cache();

    llama_hybrid_cycle_pc_region_cache(
        const llama_hybrid_cycle_pc_region_cache &) = delete;
    llama_hybrid_cycle_pc_region_cache & operator=(
        const llama_hybrid_cycle_pc_region_cache &) = delete;

    uint64_t hits() const;
    uint64_t builds() const;
    uint64_t evictions() const;
    uint64_t allocator_grows() const;
    size_t allocator_bytes() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;

    friend bool llama_hybrid_cycle_execute_pc_full_region_cached(
        const llama_model &,
        llama_hybrid_cycle_physical_resource_pool &,
        llama_hybrid_cycle_pc_region_cache &,
        int, int, int64_t, int64_t,
        const std::vector<float> &, std::vector<float> &,
        std::string &, const std::function<bool()> &,
        llama_hybrid_cycle_pc_region_timing *);
};

LLAMA_API bool llama_hybrid_cycle_execute_pc_full_region_cached(
    const llama_model & model,
    llama_hybrid_cycle_physical_resource_pool & resources,
    llama_hybrid_cycle_pc_region_cache & cache,
    int first_layer,
    int layer_count,
    int64_t token_start,
    int64_t tokens,
    const std::vector<float> & input,
    std::vector<float> & output,
    std::string & reason,
    const std::function<bool()> & input_consumed = {},
    llama_hybrid_cycle_pc_region_timing * timing = nullptr);

LLAMA_API bool llama_hybrid_cycle_execute_pc_full_region(
    const llama_model & model,
    llama_hybrid_cycle_physical_resource_pool & resources,
    int first_layer,
    int layer_count,
    int64_t token_start,
    int64_t tokens,
    const std::vector<float> & input,
    std::vector<float> & output,
    std::string & reason,
    const std::function<bool()> & input_consumed = {},
    llama_hybrid_cycle_pc_region_timing * timing = nullptr);

LLAMA_API bool llama_hybrid_cycle_bind_pc_region_stages(
    llama_hybrid_cycle_stage_dispatcher & stages,
    const llama_model & model,
    llama_hybrid_cycle_physical_resource_pool & resources,
    llama_hybrid_cycle_execution_session<> & session,
    const std::vector<float> & initial_hidden,
    int64_t hidden,
    std::string & reason,
    std::atomic<int> * region_calls = nullptr,
    llama_hybrid_cycle_pc_region_cache_stats * cache_stats = nullptr);
