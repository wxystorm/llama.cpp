#pragma once

#include "llama-hybrid-cycle-fifo-executor.h"

#include <functional>

// Runtime scheduling contract. Device work is deliberately supplied by the
// caller, so the same five physical FIFO workers can serve a test Cycle today
// and llama_context's full-model physical layer dispatcher later.
//
// All task.layer values are PHYSICAL model indices, never local period slots.
// This plan is an execution description only: model weight placement must
// already match layout.kinds before run() is invoked.
struct llama_hybrid_cycle_runtime_config {
    llama_hybrid_cycle_layout layout;
    int n_chunks = 0;
    int max_inflight = 0;            // 0 = n_chunks (legacy unbounded window)
    bool credit_until_handoff = true;
    bool pc_cpu_reduce_priority = false;
    bool ordered_c2p_reduce = false;
    bool phone_main_priority = false;
};

// Absolute layer identifiers for one repeated PC -> Phone -> PC region.
// The Phone span [phone_begin,phone_end) excludes both Tensor boundaries.
struct llama_hybrid_cycle_runtime_region {
    int cycle_index = -1;
    int begin = -1;
    int c2p_layer = -1;
    int phone_begin = -1;
    int phone_end = -1;
    int p2c_layer = -1;
    int end = -1;
};

class llama_hybrid_cycle_runtime {
public:
    LLAMA_API explicit llama_hybrid_cycle_runtime(llama_hybrid_cycle_runtime_config config);

    // A single invocation schedules ALL physical layers in layout, without
    // inserting a barrier at period boundaries. Returns false for tail layers
    // or layouts without the repeated X/Y shape.
    LLAMA_API bool region_for_layer(
        int physical_layer, llama_hybrid_cycle_runtime_region & out) const;

    // Each callback must return true only when its task's real device/receive
    // dependency is satisfied. Failure terminates the workers; no stage is
    // silently recomputed or converted to a legacy Meta full-model graph.
    LLAMA_API bool run(
        const std::function<bool(const llama_hybrid_cycle_task &)> & execute,
        llama_hybrid_cycle_fifo_profile * profile = nullptr,
        llama_hybrid_cycle_chunk_timeline * timeline = nullptr) const;

    LLAMA_API const llama_hybrid_cycle_runtime_config & config() const;

private:
    llama_hybrid_cycle_runtime_config config_;
};
