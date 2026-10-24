#include "llama-hybrid-cycle-runtime.h"
#include "llama-hybrid-cycle-graph.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

// Moved verbatim from the inline FIFO header: the five-worker scheduler now
// has a single implementation in libllama. Existing callers keep their ABI,
// ordering, credit, tracing and error-handling contract.
static bool llama_hybrid_cycle_run_fifo_impl(
        const llama_hybrid_cycle_layout & layout,
        int n_chunks,
        const std::function<bool(const llama_hybrid_cycle_task &)> & execute,
        int max_inflight,
        bool credit_until_handoff,
        llama_hybrid_cycle_fifo_profile * profile,
        bool pc_cpu_reduce_priority,
        bool ordered_c2p_reduce,
        bool phone_main_priority,
        llama_hybrid_cycle_chunk_timeline * chunk_timeline,
        bool pc_cuda_chunk_priority) {
    if (!execute || !layout.enabled() || n_chunks <= 0 ||
        max_inflight < 0 || max_inflight > n_chunks) {
        return false;
    }
    llama_hybrid_cycle_task_ring ring(
        layout, n_chunks, max_inflight, credit_until_handoff,
        pc_cpu_reduce_priority, ordered_c2p_reduce,
        phone_main_priority, chunk_timeline != nullptr,
        pc_cuda_chunk_priority);
    ring.start();

    constexpr std::array<llama_hybrid_cycle_queue, 5> queues {
        llama_hybrid_cycle_queue::PC_CPU,
        llama_hybrid_cycle_queue::PHONE_GPU,
        llama_hybrid_cycle_queue::PC_CUDA,
        llama_hybrid_cycle_queue::PC_TO_PHONE,
        llama_hybrid_cycle_queue::PHONE_TO_PC,
    };

    std::atomic<bool> failed {false};
    std::mutex wake_mutex;
    std::condition_variable wake;
    uint64_t wake_epoch = 0;
    const auto signal = [&] {
        {
            std::lock_guard<std::mutex> lock(wake_mutex);
            ++wake_epoch;
        }
        wake.notify_all();
    };
    const auto work = [&](llama_hybrid_cycle_queue queue) {
        // Slots are disjoint: there is one worker per queue, even when the
        // stages in two different queues run concurrently.
        auto * stats = profile ?
            &profile->queues[static_cast<size_t>(queue)] : nullptr;
        auto * trace = chunk_timeline ?
            &chunk_timeline->by_queue[static_cast<size_t>(queue)] : nullptr;
        while (!failed.load() && !ring.finished()) {
            uint64_t seen_epoch = 0;
            {
                std::lock_guard<std::mutex> lock(wake_mutex);
                seen_epoch = wake_epoch;
            }
            llama_hybrid_cycle_task task;
            if (!ring.try_pop(queue, task)) {
                const auto wait_begin = std::chrono::steady_clock::now();
                std::unique_lock<std::mutex> lock(wake_mutex);
                wake.wait(lock, [&] {
                    return wake_epoch != seen_epoch || failed.load() ||
                           ring.finished();
                });
                if (stats) {
                    stats->idle_wait_ms +=
                        std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - wait_begin).count();
                    ++stats->idle_wakes;
                }
                continue;
            }
            // FULL_LAYER can release the predecessor credit before its FFN
            // finishes. The Ring owns the exactly-once accounting; signal()
            // wakes a different worker immediately if that release admits work.
            if (credit_until_handoff &&
                task.stage == llama_hybrid_cycle_stage::FULL_LAYER &&
                task.layer > 0) {
                const int layer = task.layer;
                const int chunk = task.chunk;
                task.input_consumed = [&ring, &signal, layer, chunk] {
                    const bool released =
                        ring.mark_full_input_consumed(layer, chunk);
                    if (released) signal();
                    return released;
                };
            }
            const auto execute_begin = std::chrono::steady_clock::now();
            if (stats && task.queued_at.time_since_epoch().count() != 0) {
                const double queue_wait_ms =
                    std::chrono::duration<double, std::milli>(
                        execute_begin - task.queued_at).count();
                stats->queue_wait_ms += queue_wait_ms;
                stats->max_queue_wait_ms =
                    std::max(stats->max_queue_wait_ms, queue_wait_ms);
            }
            bool ok = false;
            try {
                ok = execute(task);
                if (ok) {
                    ring.complete(task);
                }
            } catch (const std::exception & e) {
                std::fprintf(stderr,
                    "[CYCLE_FIFO] queue=%s stage=%s layer=%d chunk=%d "
                    "status=FAIL exception=%s\n",
                    llama_hybrid_cycle_queue_name(task.queue),
                    llama_hybrid_cycle_stage_name(task.stage),
                    task.layer, task.chunk, e.what());
            } catch (...) {
                std::fprintf(stderr,
                    "[CYCLE_FIFO] queue=%s stage=%s layer=%d chunk=%d "
                    "status=FAIL exception=UNKNOWN\n",
                    llama_hybrid_cycle_queue_name(task.queue),
                    llama_hybrid_cycle_stage_name(task.stage),
                    task.layer, task.chunk);
            }
            if (!ok) {
                std::fprintf(stderr,
                    "[CYCLE_FIFO] queue=%s stage=%s layer=%d chunk=%d "
                    "status=FAIL reason=STAGE_EXECUTION\n",
                    llama_hybrid_cycle_queue_name(task.queue),
                    llama_hybrid_cycle_stage_name(task.stage),
                    task.layer, task.chunk);
                failed.store(true);
            }
            if (stats || trace) {
                const auto execute_end = std::chrono::steady_clock::now();
                if (trace) {
                    trace->push_back({task, execute_begin, execute_end, ok});
                }
                if (stats) {
                    const double elapsed_ms =
                        std::chrono::duration<double, std::milli>(
                            execute_end - execute_begin).count();
                    ++stats->tasks;
                    stats->execute_ms += elapsed_ms;
                    stats->max_execute_ms =
                        std::max(stats->max_execute_ms, elapsed_ms);
                }
            }
            signal();
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(queues.size());
    try {
        for (auto q : queues) {
            workers.emplace_back(work, q);
        }
    } catch (const std::exception & e) {
        // Do not std::terminate() on unwinding a vector containing live
        // std::threads if one of the per-queue worker creations fails.
        std::fprintf(stderr,
            "[CYCLE_FIFO] status=FAIL reason=WORKER_START exception=%s\n",
            e.what());
        failed.store(true);
        signal();
    }
    for (auto & worker : workers) {
        worker.join();
    }
    return !failed.load() && ring.finished();
}

bool llama_hybrid_cycle_run_fifo(
        const llama_hybrid_cycle_layout & layout,
        int n_chunks,
        const std::function<bool(const llama_hybrid_cycle_task &)> & execute,
        int max_inflight,
        bool credit_until_handoff,
        llama_hybrid_cycle_fifo_profile * profile,
        bool pc_cpu_reduce_priority,
        bool ordered_c2p_reduce,
        bool phone_main_priority,
        llama_hybrid_cycle_chunk_timeline * chunk_timeline) {
    return llama_hybrid_cycle_run_fifo_impl(
        layout, n_chunks, execute, max_inflight, credit_until_handoff,
        profile, pc_cpu_reduce_priority, ordered_c2p_reduce,
        phone_main_priority, chunk_timeline, false);
}


bool llama_hybrid_cycle_runtime_validate_model(
        const llama_model & model,
        const llama_hybrid_cycle_layout & layout,
        std::string & reason) {
    reason.clear();
    if (model.arch != LLM_ARCH_QWEN3MOE) {
        reason = "UNSUPPORTED_MODEL_ARCH";
        return false;
    }
    if (model.split_mode() != LLAMA_SPLIT_MODE_TENSOR) {
        reason = "REQUIRES_TENSOR_SPLIT_MODE";
        return false;
    }
    if (!layout.enabled() || layout.cycle_size <= 0 ||
        layout.x <= 0 || layout.y <= 0 ||
        int64_t(layout.x) + layout.y + 2 != layout.cycle_size ||
        layout.kinds.size() > size_t(model.hparams.n_layer()) ||
        layout.kinds.size() > model.layers.size() ||
        layout.kinds.size() > model.hybrid_layer_modes.size() ||
        layout.kinds.size() > model.pc_layer_backends.size()) {
        reason = "PHYSICAL_LAYER_COUNT_OR_LAYOUT_MISMATCH";
        return false;
    }
    for (size_t i = 0; i < layout.kinds.size(); ++i) {
        const auto kind = layout.kinds[i];
        const auto mode = model.hybrid_layer_modes[i];
        bool match = false;
        switch (kind) {
            case llama_hybrid_cycle_layer_kind::PC_CPU:
                // PC CUDA-full weights may be mirrored from this PC_ONLY
                // placement by the independent runtime/session pool.
                match = mode == llama_hybrid_layer_mode::PC_ONLY;
                break;
            case llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL:
                match = mode == llama_hybrid_layer_mode::PC_ONLY &&
                    model.pc_layer_backends[i] ==
                        llama_pc_layer_backend::CUDA;
                break;
            case llama_hybrid_cycle_layer_kind::PHONE_GPU:
                match = mode == llama_hybrid_layer_mode::PHONE_ONLY;
                break;
            case llama_hybrid_cycle_layer_kind::TENSOR_C2P:
                match = mode == llama_hybrid_layer_mode::TENSOR_C2P;
                break;
            case llama_hybrid_cycle_layer_kind::TENSOR_P2C:
                match = mode == llama_hybrid_layer_mode::TENSOR_P2C;
                break;
        }
        if (!match) {
            reason = "DEVICE_ROLE_MISMATCH layer=" + std::to_string(i);
            return false;
        }
        if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ||
            kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
            const auto & lw = model.layers[i];
            if (!llama_hybrid_cycle_ffn_partition_valid(
                    lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps,
                    model.hparams.n_embd, model.hparams.n_expert)) {
                reason = "PHYSICAL_FFN_SHARD_MISMATCH layer=" +
                    std::to_string(i);
                return false;
            }
        }
    }
    return true;
}

namespace {

bool cycle_runtime_stage_applies(
        llama_hybrid_cycle_layer_kind kind, llama_hybrid_cycle_stage stage) {
    const bool tensor = kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ||
                        kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C;
    const auto index = static_cast<size_t>(stage);
    if (index > static_cast<size_t>(llama_hybrid_cycle_stage::REDUCE)) {
        return false;
    }
    return tensor ? stage != llama_hybrid_cycle_stage::FULL_LAYER :
                    stage == llama_hybrid_cycle_stage::FULL_LAYER;
}

llama_hybrid_cycle_queue cycle_runtime_owner_queue(llama_hybrid_cycle_owner owner) {
    switch (owner) {
        case llama_hybrid_cycle_owner::PC_CPU:
            return llama_hybrid_cycle_queue::PC_CPU;
        case llama_hybrid_cycle_owner::PHONE_GPU:
            return llama_hybrid_cycle_queue::PHONE_GPU;
        case llama_hybrid_cycle_owner::PC_CUDA:
            return llama_hybrid_cycle_queue::PC_CUDA;
    }
    return llama_hybrid_cycle_queue::COUNT;
}

bool cycle_runtime_task_route_matches(
        const llama_hybrid_cycle_layout & layout, const llama_hybrid_cycle_task & task) {
    if (task.layer < 0 || static_cast<size_t>(task.layer) >= layout.kinds.size() ||
        task.chunk < 0 || !cycle_runtime_stage_applies(
            layout.kinds[static_cast<size_t>(task.layer)], task.stage)) {
        return false;
    }
    const auto kind = layout.kinds[static_cast<size_t>(task.layer)];
    auto expected_owner = llama_hybrid_cycle_attn_owner(kind);
    if (task.stage == llama_hybrid_cycle_stage::FULL_LAYER &&
        llama_hybrid_cycle_full_layer_on_cuda(layout, task.layer)) {
        expected_owner = llama_hybrid_cycle_owner::PC_CUDA;
    } else if (task.stage == llama_hybrid_cycle_stage::FFN_DESTINATION ||
               task.stage == llama_hybrid_cycle_stage::REDUCE) {
        expected_owner = llama_hybrid_cycle_output_owner(kind);
    }
    auto expected_queue = cycle_runtime_owner_queue(expected_owner);
    if (task.stage == llama_hybrid_cycle_stage::SEND_ACTIVATION ||
        task.stage == llama_hybrid_cycle_stage::SEND_PARTIAL) {
        expected_queue = kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ?
            llama_hybrid_cycle_queue::PC_TO_PHONE :
            llama_hybrid_cycle_queue::PHONE_TO_PC;
    }
    return task.owner == expected_owner && task.queue == expected_queue;
}

} // namespace

llama_hybrid_cycle_stage_dispatcher::llama_hybrid_cycle_stage_dispatcher(
        const llama_hybrid_cycle_layout & physical_layout)
    : layout_(physical_layout), handlers_(physical_layout.kinds.size()) {
    if (!layout_.enabled() || !llama_hybrid_cycle_cuda_layers_valid(layout_)) {
        throw std::invalid_argument("cycle Stage dispatcher needs valid physical layers and CUDA selection");
    }
}

bool llama_hybrid_cycle_stage_dispatcher::bind(
        int physical_layer, llama_hybrid_cycle_stage stage, handler execute) {
    if (physical_layer < 0 ||
        static_cast<size_t>(physical_layer) >= handlers_.size() ||
        !cycle_runtime_stage_applies(layout_.kinds[static_cast<size_t>(physical_layer)],
                                     stage) || !execute) {
        return false;
    }
    auto & slot = handlers_[static_cast<size_t>(physical_layer)]
                           [static_cast<size_t>(stage)];
    if (slot) {
        return false;
    }
    slot = std::move(execute);
    return true;
}

bool llama_hybrid_cycle_stage_dispatcher::validate(std::string & reason) const {
    reason.clear();
    if (handlers_.size() != layout_.kinds.size() || handlers_.empty()) {
        reason = "LAYOUT_SIZE_MISMATCH";
        return false;
    }
    for (size_t layer = 0; layer < handlers_.size(); ++layer) {
        for (size_t stage = 0; stage < stages_per_layer; ++stage) {
            const bool needed = cycle_runtime_stage_applies(
                layout_.kinds[layer], static_cast<llama_hybrid_cycle_stage>(stage));
            if (needed && !handlers_[layer][stage]) {
                reason = "UNBOUND_PHYSICAL_STAGE layer=" + std::to_string(layer) +
                         " stage=" + llama_hybrid_cycle_stage_name(
                             static_cast<llama_hybrid_cycle_stage>(stage));
                return false;
            }
            if (!needed && handlers_[layer][stage]) {
                reason = "INVALID_PHYSICAL_STAGE layer=" + std::to_string(layer);
                return false;
            }
        }
    }
    return true;
}

bool llama_hybrid_cycle_stage_dispatcher::matches(
        const llama_hybrid_cycle_layout & other) const {
    return layout_.kinds == other.kinds && layout_.x == other.x &&
           layout_.y == other.y && layout_.cycle_size == other.cycle_size &&
           layout_.cycles == other.cycles && layout_.tail_begin == other.tail_begin &&
           layout_.pc_cuda_full == other.pc_cuda_full &&
           layout_.pc_cuda_layers == other.pc_cuda_layers;
}

bool llama_hybrid_cycle_stage_dispatcher::execute(
        const llama_hybrid_cycle_task & task) const {
    if (!cycle_runtime_task_route_matches(layout_, task)) {
        return false;
    }
    const auto & execute = handlers_[static_cast<size_t>(task.layer)]
                                    [static_cast<size_t>(task.stage)];
    return execute && execute(task);
}

llama_hybrid_cycle_runtime::llama_hybrid_cycle_runtime(
        llama_hybrid_cycle_runtime_config config)
    : config_(std::move(config)) {
    if (!config_.layout.enabled() ||
        !llama_hybrid_cycle_cuda_layers_valid(config_.layout) ||
        config_.n_chunks <= 0 ||
        config_.max_inflight < 0 ||
        config_.max_inflight > config_.n_chunks) {
        throw std::invalid_argument(
            "cycle runtime needs a nonempty physical layout and valid chunk window");
    }
}

bool llama_hybrid_cycle_runtime::region_for_layer(
        int physical_layer, llama_hybrid_cycle_runtime_region & out) const {
    out = {};
    const auto & layout = config_.layout;
    if (layout.x <= 0 || layout.y <= 0 ||
        layout.cycle_size <= 0 || layout.cycles <= 0 ||
        int64_t(layout.x) + layout.y + 2 != layout.cycle_size ||
        int64_t(layout.cycles) * layout.cycle_size != layout.tail_begin ||
        physical_layer < 0 || physical_layer >= layout.tail_begin ||
        static_cast<size_t>(layout.tail_begin) > layout.kinds.size()) {
        return false;
    }

    const int cycle = physical_layer / layout.cycle_size;
    const int begin = cycle * layout.cycle_size;
    const int c2p = begin + layout.x;
    const int phone_begin = c2p + 1;
    const int phone_end = phone_begin + layout.y;
    const int p2c = phone_end;
    const int end = p2c + 1;

    // Do not invent a V5 region for an arbitrary Planner layout that does
    // not actually have this physical sequence.
    for (int l = begin; l < end; ++l) {
        const auto expected = l < c2p ?
            llama_hybrid_cycle_layer_kind::PC_CPU :
            l == c2p ? llama_hybrid_cycle_layer_kind::TENSOR_C2P :
            l < phone_end ? llama_hybrid_cycle_layer_kind::PHONE_GPU :
            llama_hybrid_cycle_layer_kind::TENSOR_P2C;
        if (layout.kinds[static_cast<size_t>(l)] != expected) {
            return false;
        }
    }
    out.cycle_index = cycle;
    out.begin = begin;
    out.c2p_layer = c2p;
    out.phone_begin = phone_begin;
    out.phone_end = phone_end;
    out.p2c_layer = p2c;
    out.end = end;
    return true;
}

bool llama_hybrid_cycle_runtime::run(
        const std::function<bool(const llama_hybrid_cycle_task &)> & execute,
        llama_hybrid_cycle_fifo_profile * profile,
        llama_hybrid_cycle_chunk_timeline * timeline) const {
    return llama_hybrid_cycle_run_fifo_impl(
        config_.layout, config_.n_chunks, execute, config_.max_inflight,
        config_.credit_until_handoff, profile,
        config_.pc_cpu_reduce_priority, config_.ordered_c2p_reduce,
        config_.phone_main_priority, timeline,
        config_.pc_cuda_chunk_priority);
}

bool llama_hybrid_cycle_runtime::run(
        const llama_hybrid_cycle_stage_dispatcher & stages,
        llama_hybrid_cycle_fifo_profile * profile,
        llama_hybrid_cycle_chunk_timeline * timeline) const {
    std::string reason;
    if (!stages.matches(config_.layout) || !stages.validate(reason)) {
        std::fprintf(stderr,
            "[CYCLE_RUNTIME_STAGE] status=FAIL reason=%s\n",
            !stages.matches(config_.layout) ? "FOREIGN_PHYSICAL_LAYOUT" :
                                                 reason.c_str());
        return false;
    }
    return run([&stages](const llama_hybrid_cycle_task & task) {
        return stages.execute(task);
    }, profile, timeline);
}

const llama_hybrid_cycle_runtime_config & llama_hybrid_cycle_runtime::config() const {
    return config_;
}
