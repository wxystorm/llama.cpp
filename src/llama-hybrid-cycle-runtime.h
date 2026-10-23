#pragma once

#include "llama-hybrid-cycle-fifo-executor.h"

#include <array>
#include <functional>
#include <string>
#include <vector>

struct llama_model;

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

// Validate the *loaded physical model* before dispatching any real stage.
// The layout uses absolute layer indices; a test may validate just the first
// period, while the future CLI path validates the full 48-layer sequence.
// Planner layout, Meta allocation and physical MoE shards must all agree.
LLAMA_API bool llama_hybrid_cycle_runtime_validate_model(
    const llama_model & model,
    const llama_hybrid_cycle_layout & layout,
    std::string & reason);

// Typed, physical-layer Stage callback registry. A single instance is
// populated before worker start, then read concurrently by the five FIFO
// workers. Handlers retain ownership of real GPU/CPU/RPC objects and must
// complete device/transfer dependencies before returning true. No graph is
// built or silently routed to Meta by the dispatcher.
//
// Production and the original V5 real-cycle test share this same dispatch:
// FULL_LAYER for PC/Phone, six ordered Tensor stages for C2P/P2C.
class llama_hybrid_cycle_stage_dispatcher {
public:
    using handler = std::function<bool(const llama_hybrid_cycle_task &)>;

    LLAMA_API explicit llama_hybrid_cycle_stage_dispatcher(
        const llama_hybrid_cycle_layout & physical_layout);

    // A physical (layer, stage) can be bound only once. A stage not present
    // in the layer's placement is rejected instead of dropped at runtime.
    LLAMA_API bool bind(int physical_layer, llama_hybrid_cycle_stage stage,
                        handler execute);

    // Check all required Stage handlers before any background workers start.
    LLAMA_API bool validate(std::string & reason) const;
    LLAMA_API bool matches(const llama_hybrid_cycle_layout & other) const;
    LLAMA_API bool execute(const llama_hybrid_cycle_task & task) const;

private:
    static constexpr size_t stages_per_layer =
        static_cast<size_t>(llama_hybrid_cycle_stage::REDUCE) + 1;
    llama_hybrid_cycle_layout layout_;
    std::vector<std::array<handler, stages_per_layer>> handlers_;
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

    // New physical Stage registry: fail closed on an incomplete/foreign
    // layout before starting workers; the legacy callback form is retained.
    LLAMA_API bool run(
        const llama_hybrid_cycle_stage_dispatcher & stages,
        llama_hybrid_cycle_fifo_profile * profile = nullptr,
        llama_hybrid_cycle_chunk_timeline * timeline = nullptr) const;

    LLAMA_API const llama_hybrid_cycle_runtime_config & config() const;

private:
    llama_hybrid_cycle_runtime_config config_;
};
