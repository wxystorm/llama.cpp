#pragma once

#include "llama-hybrid-cycle-ring.h"
#include "llama-hybrid-cycle-chunk-profile.h"
#include "llama.h"

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
LLAMA_API bool llama_hybrid_cycle_run_fifo(
        const llama_hybrid_cycle_layout & layout,
        int n_chunks,
        const std::function<bool(const llama_hybrid_cycle_task &)> & execute,
        int max_inflight = 0,
        bool credit_until_handoff = false,
        llama_hybrid_cycle_fifo_profile * profile = nullptr,
        bool pc_cpu_reduce_priority = false,
        bool ordered_c2p_reduce = false,
        bool phone_main_priority = false,
        llama_hybrid_cycle_chunk_timeline * chunk_timeline = nullptr);
