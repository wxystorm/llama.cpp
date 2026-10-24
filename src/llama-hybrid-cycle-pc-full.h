#pragma once

#include "llama-hybrid-cycle-resources.h"
#include "llama-hybrid-cycle-execution-session.h"
#include "llama-hybrid-cycle-ffn-shard.h"
#include "llama-hybrid-cycle-log.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// Real, physical PC FULL_LAYER kernel shared by the first PC prefix and by
// every later PC segment in a 48-layer Cycle. It runs the exact physical
// Attention/Router + MoE FFN + residual ADD for one Chunk and commits this
// physical layer's K/V prefix. It never submits Phone RPC/Meta graphs.
//
// Every ordinary PC Full layer executes on CUDA. Its full-weight mirror may
// be resident already or populated on demand by the bounded resource-pool
// LRU. Missing residency is never permission to execute the Full layer on CPU.
struct llama_hybrid_cycle_pc_full_timing {
    double attention_ms = 0.0;
    double ffn_ms = 0.0;
    double output_readback_ms = 0.0;
    double output_device_copy_ms = 0.0;
};

LLAMA_API bool llama_hybrid_cycle_execute_pc_full_layer(
    const llama_model & model,
    const llama_hybrid_cycle_physical_layer_resources & resource,
    ggml_backend_t pc_cpu,
    int physical_layer, int64_t token_start, int64_t tokens,
    const std::vector<float> & input,
    std::vector<float> & output,
    llama_hybrid_cycle_pc_full_timing * timing = nullptr,
    const std::function<bool()> & input_consumed = {},
    const ggml_tensor * device_input = nullptr,
    llama_hybrid_cycle_cuda_activation_slot * device_output = nullptr);

// Bind real PC Full stages by ABSOLUTE model layer number. The other
// Tensor/Phone stages are intentionally left for their own real executors;
// callers must bind them before running/validating the full FIFO.
//
// The model, resources, backends, session and initial embedding are BORROWED
// and must remain alive until all five FIFO workers have joined. Adjacent CUDA
// Full layers publish a weight-independent device activation slot; only the
// final Full layer before a Tensor boundary materializes Host output.
template <class PhoneSlot, class P2cSlot>
bool llama_hybrid_cycle_bind_pc_full_stages(
    llama_hybrid_cycle_stage_dispatcher & stages,
    const llama_model & model,
    llama_hybrid_cycle_physical_resource_pool & resources,
    ggml_backend_t pc_cpu,
    llama_hybrid_cycle_execution_session<PhoneSlot, P2cSlot> & session,
    const std::vector<float> & initial_hidden,
    int64_t hidden,
    std::string & reason,
    std::function<void(int, int, int64_t, int64_t,
                       const std::vector<float> &)> trace_output = {}) {
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
    // Numerical tracing intentionally preserves the old Host boundary at
    // every PC Full layer. Production (no trace_output) uses device-local
    // handoff across adjacent CUDA Full layers.
    const bool device_local = !trace_output;
    auto cuda_idle_by_layer = std::make_shared<
        std::vector<std::vector<
            std::unique_ptr<llama_hybrid_cycle_cuda_activation_slot>>>>(
                layout.kinds.size());

    for (int l = begin; l < end; ++l) {
        if (layout.kinds[size_t(l)] != llama_hybrid_cycle_layer_kind::PC_CPU) {
            continue;
        }
        const auto * resource = resources.layer(l);
        if (!resource || !resource->kv || !resource->attn_backend ||
            !llama_hybrid_cycle_full_layer_on_cuda(layout, l)) {
            reason = "PC_FULL_UNPREPARED_LAYER layer=" + std::to_string(l);
            return false;
        }
        const bool cuda_predecessor =
            device_local && l > 0 &&
            layout.kinds[size_t(l - 1)] ==
                llama_hybrid_cycle_layer_kind::PC_CPU &&
            llama_hybrid_cycle_full_layer_on_cuda(layout, l - 1);
        const bool cuda_successor =
            device_local && l + 1 < int(layout.kinds.size()) &&
            layout.kinds[size_t(l + 1)] ==
                llama_hybrid_cycle_layer_kind::PC_CPU &&
            llama_hybrid_cycle_full_layer_on_cuda(layout, l + 1);
        const bool bound = stages.bind(l, llama_hybrid_cycle_stage::FULL_LAYER,
            [l, hidden, pc_cpu, &resources, &session, &model,
             &initial_hidden, trace_output, cuda_predecessor,
             cuda_successor, cuda_idle_by_layer](
                const llama_hybrid_cycle_task & task) {
                if (task.layer != l || task.chunk < 0 ||
                    task.chunk >= session.chunks_per_layer() ||
                    task.stage != llama_hybrid_cycle_stage::FULL_LAYER) {
                    return false;
                }
                auto & chunk = session.at(l, task.chunk);
                std::vector<float> input;
                const ggml_tensor * device_input = nullptr;
                if (cuda_predecessor) {
                    device_input =
                        session.cuda_predecessor_output(l, task.chunk);
                    if (!device_input) {
                        std::fprintf(stderr,
                            "[CYCLE_PC_CUDA_HANDOFF] layer=%d chunk=%d "
                            "reason=PREDECESSOR_DEVICE_OUTPUT status=FAIL\n",
                            l, task.chunk);
                        return false;
                    }
                } else if (l == 0) {
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
                std::string residency_reason;
                if (!resources.ensure_pc_full_cuda_resident(
                        l, residency_reason)) {
                    std::fprintf(stderr,
                        "[CYCLE_PC_FULL] layer=%d chunk=%d "
                        "reason=CUDA_RESIDENCY detail=%s status=FAIL\n",
                        l, task.chunk, residency_reason.c_str());
                    return false;
                }
                const auto * resident = resources.layer(l);
                if (!resident || !resident->cuda_weights) {
                    return false;
                }

                std::unique_ptr<llama_hybrid_cycle_cuda_activation_slot>
                    output_slot;
                if (cuda_successor) {
                    auto & idle = (*cuda_idle_by_layer)[size_t(l)];
                    for (auto it = idle.begin(); it != idle.end(); ++it) {
                        if (*it && (*it)->matches(
                                resident->attn_backend, hidden, chunk.tokens)) {
                            output_slot = std::move(*it);
                            idle.erase(it);
                            break;
                        }
                    }
                    if (!output_slot) {
                        output_slot = std::make_unique<
                            llama_hybrid_cycle_cuda_activation_slot>();
                    }
                    if (!output_slot->prepare(
                            resident->attn_backend, hidden, chunk.tokens)) {
                        std::fprintf(stderr,
                            "[CYCLE_PC_CUDA_HANDOFF] layer=%d chunk=%d "
                            "reason=OUTPUT_SLOT_PREPARE status=FAIL\n",
                            l, task.chunk);
                        return false;
                    }
                }

                std::function<bool()> input_consumed = task.input_consumed;
                if (cuda_predecessor) {
                    const auto ring_input_consumed = task.input_consumed;
                    input_consumed =
                        [l, c = task.chunk, &session, cuda_idle_by_layer,
                         ring_input_consumed]() -> bool {
                            auto previous =
                                session.retire_cuda_output_after_handoff(
                                    l - 1, l, c);
                            if (!previous) {
                                std::fprintf(stderr,
                                    "[CYCLE_PC_CUDA_HANDOFF] layer=%d chunk=%d "
                                    "reason=PREDECESSOR_SLOT status=FAIL\n",
                                    l, c);
                                return false;
                            }
                            (*cuda_idle_by_layer)[size_t(l - 1)].push_back(
                                std::move(previous));
                            return !ring_input_consumed ||
                                ring_input_consumed();
                        };
                }

                if (!llama_hybrid_cycle_execute_pc_full_layer(
                        model, *resident, pc_cpu, l,
                        chunk.start, chunk.tokens, input,
                        chunk.host_output, nullptr, input_consumed,
                        device_input, output_slot.get())) {
                    return false;
                }
                if (cuda_successor) {
                    chunk.cuda_output = std::move(output_slot);
                    if (llama_hybrid_cycle_verbose()) {
                        std::printf(
                            "[CYCLE_PC_CUDA_HANDOFF] layer=%d chunk=%d "
                            "bytes=%zu mode=DEVICE_LOCAL status=READY\n",
                            l, task.chunk,
                            ggml_nbytes(chunk.cuda_output->tensor));
                    }
                    return true;
                }

                // Only the final CUDA Full layer before a Tensor boundary
                // materializes Host output. Internal Full layers never do.
                if (trace_output) {
                    trace_output(l, task.chunk, chunk.start, chunk.tokens,
                                 chunk.host_output);
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
