#pragma once

#include "llama-hybrid-cycle-ring.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Bridge the dependency-only Cycle ring to REAL per-device and per-direction
// workers. The callback owns compute buffers, RPC transfers, ACKs, and Reduce.
// A task may be completed only after its true stage dependency is satisfied:
//   SEND_ACTIVATION -> receiver inputs ready
//   SEND_PARTIAL    -> receiver has the source partial
//   REDUCE          -> exclusive output owner has completed l_out
//
// One FIFO worker owns each queue; an unrelated queue can continue while a
// Phone RPC send or PC computation is in flight. This contains scheduling
// complexity here, outside the GGML backend mapping/buffer ownership rules.
inline bool llama_hybrid_cycle_run_fifo(
        const llama_hybrid_cycle_layout & layout,
        int n_chunks,
        const std::function<bool(const llama_hybrid_cycle_task &)> & execute) {
    if (!execute || !layout.enabled() || n_chunks <= 0) {
        return false;
    }
    llama_hybrid_cycle_task_ring ring(layout, n_chunks);
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
        while (!failed.load() && !ring.finished()) {
            uint64_t seen_epoch = 0;
            {
                std::lock_guard<std::mutex> lock(wake_mutex);
                seen_epoch = wake_epoch;
            }
            llama_hybrid_cycle_task task;
            if (!ring.try_pop(queue, task)) {
                std::unique_lock<std::mutex> lock(wake_mutex);
                wake.wait(lock, [&] {
                    return wake_epoch != seen_epoch || failed.load() ||
                           ring.finished();
                });
                continue;
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
            signal();
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(queues.size());
    for (auto q : queues) {
        workers.emplace_back(work, q);
    }
    for (auto & worker : workers) {
        worker.join();
    }
    return !failed.load() && ring.finished();
}
