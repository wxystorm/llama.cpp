#pragma once

#include "llama-hybrid-cycle-resources.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct llama_model;

// One PC CUDA Region executes a contiguous run of ordinary Full layers as
// ONE backend graph. The Region owns no persistent model weights; all CUDA
// mirrors are pinned in the resource-pool LRU for the duration of the call.
// Only the Region input/output cross the graph boundary.
struct llama_hybrid_cycle_pc_region_timing {
    double graph_build_ms = 0.0;
    double alloc_ms = 0.0;
    double input_ms = 0.0;
    double compute_ms = 0.0;
    double readback_ms = 0.0;
    int graph_nodes = 0;
};

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
