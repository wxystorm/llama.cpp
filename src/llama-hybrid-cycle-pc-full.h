#pragma once

#include "llama-hybrid-cycle-resources.h"
#include "llama-hybrid-cycle-execution-session.h"
#include "llama-hybrid-cycle-ffn-shard.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

struct llama_model;

// Real, physical PC FULL_LAYER kernel shared by the first PC prefix and by
// every later PC segment in a 48-layer Cycle. It runs the exact physical
// Attention/Router + MoE FFN + residual ADD for one Chunk and commits this
// physical layer's K/V prefix. It never submits Phone RPC/Meta graphs.
//
// A layer placed on CUDA MUST have a budgeted CUDA mirror in the pool;
// otherwise it executes on CPU. No implicit full-model CUDA copies.
struct llama_hybrid_cycle_pc_full_timing {
    double attention_ms = 0.0;
    double ffn_ms = 0.0;
    double output_readback_ms = 0.0;
};

LLAMA_API bool llama_hybrid_cycle_execute_pc_full_layer(
    const llama_model & model,
    const llama_hybrid_cycle_physical_layer_resources & resource,
    ggml_backend_t pc_cpu,
    int physical_layer, int64_t token_start, int64_t tokens,
    const std::vector<float> & input,
    std::vector<float> & output,
    llama_hybrid_cycle_pc_full_timing * timing = nullptr);

// Bind real PC Full stages by ABSOLUTE model layer number. The other
// Tensor/Phone stages are intentionally left for their own real executors;
// callers must bind them before running/validating the full FIFO.
//
// The model, resources, backends, session and initial embedding are BORROWED
// and must remain alive until all five FIFO workers have joined. A producer
// publishes its host output before returning, allowing the next physical
// layer's SAME Chunk to proceed without a period-level barrier.
template <class PhoneSlot, class P2cSlot>
bool llama_hybrid_cycle_bind_pc_full_stages(
    llama_hybrid_cycle_stage_dispatcher & stages,
    const llama_model & model,
    const llama_hybrid_cycle_physical_resource_pool & resources,
    ggml_backend_t pc_cpu,
    llama_hybrid_cycle_execution_session<PhoneSlot, P2cSlot> & session,
    const std::vector<float> & initial_hidden,
    int64_t hidden,
    std::string & reason) {
    reason.clear();
    const auto & layout = resources.execution_layout();
    const auto & state_layout = session.config().layout;
    if (!pc_cpu || !layout.enabled() || hidden <= 0 ||
        layout.kinds != state_layout.kinds ||
        layout.pc_cuda_layers != state_layout.pc_cuda_layers ||
        layout.pc_cuda_full != state_layout.pc_cuda_full ||
        !llama_hybrid_cycle_cuda_layers_valid(layout) ||
        resources.prepared_layers() != int(layout.kinds.size())) {
        reason = "PC_FULL_PHYSICAL_LAYOUT_OR_BACKEND_MISMATCH";
        return false;
    }

    const int begin = session.first_layer();
    const int end = begin + session.layer_count();
    if (begin < 0 || end < begin || end > int(layout.kinds.size())) {
        reason = "PC_FULL_LAYER_RANGE";
        return false;
    }
    for (int l = begin; l < end; ++l) {
        if (layout.kinds[size_t(l)] != llama_hybrid_cycle_layer_kind::PC_CPU) {
            continue;
        }
        const auto * resource = resources.layer(l);
        if (!resource || !resource->kv ||
            !resource->attn_backend ||
            (llama_hybrid_cycle_full_layer_on_cuda(layout, l) !=
             (resource->cuda_weights != nullptr))) {
            reason = "PC_FULL_UNPREPARED_LAYER layer=" + std::to_string(l);
            return false;
        }
        const bool bound = stages.bind(l, llama_hybrid_cycle_stage::FULL_LAYER,
            [l, hidden, pc_cpu, &resources, &session, &model,
             &initial_hidden](const llama_hybrid_cycle_task & task) {
                if (task.layer != l || task.chunk < 0 ||
                    task.chunk >= session.chunks_per_layer() ||
                    task.stage != llama_hybrid_cycle_stage::FULL_LAYER) {
                    return false;
                }
                auto & chunk = session.at(l, task.chunk);
                std::vector<float> input;
                if (l == 0) {
                    const int64_t start = chunk.start -
                        session.config().token_position_base;
                    if (start < 0 || hidden <= 0 ||
                        uint64_t(start) >
                            std::numeric_limits<size_t>::max() /
                                uint64_t(hidden) ||
                        uint64_t(chunk.tokens) >
                            std::numeric_limits<size_t>::max() /
                                uint64_t(hidden)) {
                        return false;
                    }
                    const size_t offset = size_t(start) * size_t(hidden);
                    const size_t n = size_t(chunk.tokens) * size_t(hidden);
                    if (offset > initial_hidden.size() ||
                        n > initial_hidden.size() - offset) {
                        return false;
                    }
                    input.assign(initial_hidden.begin() + offset,
                                 initial_hidden.begin() + offset + n);
                } else if (!session.copy_predecessor_host_input(
                               l, task.chunk, hidden, input)) {
                    std::fprintf(stderr,
                        "[CYCLE_PC_FULL_HANDOFF] layer=%d chunk=%d "
                        "reason=PREDECESSOR_NOT_PUBLISHED status=FAIL\n",
                        l, task.chunk);
                    return false;
                }
                if (!llama_hybrid_cycle_execute_pc_full_layer(
                        model, *resources.layer(l), pc_cpu, l,
                        chunk.start, chunk.tokens, input,
                        chunk.host_output)) {
                    return false;
                }
                return session.publish_host_output(l, task.chunk, hidden);
            });
        if (!bound) {
            reason = "PC_FULL_BIND_FAILED layer=" + std::to_string(l);
            return false;
        }
    }
    return true;
}
