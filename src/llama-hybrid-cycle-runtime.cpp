#include "llama-hybrid-cycle-runtime.h"

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
    if (!execute || !layout.enabled() || n_chunks <= 0 ||
        max_inflight < 0 || max_inflight > n_chunks) {
        return false;
    }
    llama_hybrid_cycle_task_ring ring(
        layout, n_chunks, max_inflight, credit_until_handoff,
        pc_cpu_reduce_priority, ordered_c2p_reduce,
        phone_main_priority, chunk_timeline != nullptr);
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


llama_hybrid_cycle_runtime::llama_hybrid_cycle_runtime(
        llama_hybrid_cycle_runtime_config config)
    : config_(std::move(config)) {
    if (!config_.layout.enabled() || config_.n_chunks <= 0 ||
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
    return llama_hybrid_cycle_run_fifo(
        config_.layout, config_.n_chunks, execute, config_.max_inflight,
        config_.credit_until_handoff, profile,
        config_.pc_cpu_reduce_priority, config_.ordered_c2p_reduce,
        config_.phone_main_priority, timeline);
}

const llama_hybrid_cycle_runtime_config & llama_hybrid_cycle_runtime::config() const {
    return config_;
}
