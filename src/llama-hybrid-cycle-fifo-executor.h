#pragma once

#include "llama-hybrid-cycle-ring.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Opt-in per-worker measurements. Each queue has exactly one owning
// thread, so its slot is written by that worker only and read after join().
// queue_wait_ms: task ready/enqueued -> actual pop, including older FIFO tasks.
// idle_wait_ms: time the worker slept with no ready task. This is not
// automatically a critical-path stall (another queue may be computing).
// execute_ms: callback + dependency completion, not isolated GPU kernel time.
struct llama_hybrid_cycle_queue_profile {
    uint64_t tasks = 0;
    uint64_t idle_wakes = 0;
    double queue_wait_ms = 0.0;
    double max_queue_wait_ms = 0.0;
    double idle_wait_ms = 0.0;
    double execute_ms = 0.0;
    double max_execute_ms = 0.0;
};
struct llama_hybrid_cycle_fifo_profile {
    std::array<llama_hybrid_cycle_queue_profile,
        static_cast<size_t>(llama_hybrid_cycle_queue::COUNT)> queues {};
};

// Bridge the dependency-only Cycle ring to REAL per-device and per-direction
// workers. The callback owns compute buffers, RPC transfers, ACKs, and Reduce.
// A task may be completed only after its true stage dependency is satisfied:
//   SEND_ACTIVATION -> receiver inputs ready
//   SEND_PARTIAL    -> receiver has the source partial
//   REDUCE          -> exclusive output owner has completed l_out
//
// One FIFO worker owns each queue; an unrelated queue can continue while a
// Phone RPC send or PC computation is in flight. With explicit opt-in, the
// PC_CPU worker prefers already-ready P2C Reduce tasks when selecting the
// next task, without preempting its current FFN or changing TX ordering.
// Buffer ownership is independent of this local dequeue policy.
inline bool llama_hybrid_cycle_run_fifo(
        const llama_hybrid_cycle_layout & layout,
        int n_chunks,
        const std::function<bool(const llama_hybrid_cycle_task &)> & execute,
        int max_inflight = 0,
        bool credit_until_handoff = false,
        llama_hybrid_cycle_fifo_profile * profile = nullptr,
        bool pc_cpu_reduce_priority = false) {
    if (!execute || !layout.enabled() || n_chunks <= 0 ||
        max_inflight < 0 || max_inflight > n_chunks) {
        return false;
    }
    llama_hybrid_cycle_task_ring ring(
        layout, n_chunks, max_inflight, credit_until_handoff,
        pc_cpu_reduce_priority);
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
            if (stats) {
                const double elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - execute_begin).count();
                ++stats->tasks;
                stats->execute_ms += elapsed_ms;
                stats->max_execute_ms =
                    std::max(stats->max_execute_ms, elapsed_ms);
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
