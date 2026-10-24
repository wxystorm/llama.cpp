#include "llama-hybrid-cycle-ring.h"
#include "llama-hybrid-cycle-phone-local.h"
#include "llama-hybrid-cycle-runtime.h"
#include "llama-hybrid-cycle-execution-session.h"
#include "llama-hybrid-cycle-resources.h"
#include "llama-hybrid-cycle-phone-regions.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

static llama_hybrid_cycle_task take(
        llama_hybrid_cycle_task_ring & ring,
        llama_hybrid_cycle_queue queue,
        llama_hybrid_cycle_stage stage,
        int layer,
        int chunk) {
    llama_hybrid_cycle_task task;
    assert(ring.try_pop(queue, task));
    if (task.queue != queue || task.stage != stage ||
        task.layer != layer || task.chunk != chunk) {
        std::fprintf(stderr,
            "[CYCLE_RING_TEST_ORDER] expected=%s/%s L%d C%d "
            "actual=%s/%s L%d C%d\n",
            llama_hybrid_cycle_queue_name(queue),
            llama_hybrid_cycle_stage_name(stage), layer, chunk,
            llama_hybrid_cycle_queue_name(task.queue),
            llama_hybrid_cycle_stage_name(task.stage),
            task.layer, task.chunk);
    }
    assert(task.queue == queue);
    assert(task.stage == stage);
    assert(task.layer == layer);
    assert(task.chunk == chunk);
    return task;
}

static void expect_empty(llama_hybrid_cycle_task_ring & ring, llama_hybrid_cycle_queue queue) {
    llama_hybrid_cycle_task unused;
    assert(!ring.try_pop(queue, unused));
}

// Exercise the two transfer directions with deliberately different completion
// orderings. No Reduce may run before BOTH the remote FFN and Send #2 are ready.
static void test_one_cycle() {
    llama_hybrid_cycle_task_ring ring(llama_hybrid_build_cycle_layout(4, 1, 1), 1);
    ring.start();

    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::FULL_LAYER, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::ATTN_PRE, 1, 0));

    // C2P: the sender finishes its FFN first, but the PC->Phone wire queue
    // still contains Send #1 followed by Send #2.
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::FFN_SOURCE, 1, 0));
    assert(ring.pending(llama_hybrid_cycle_queue::PC_TO_PHONE) == 2);
    expect_empty(ring, llama_hybrid_cycle_queue::PHONE_GPU);
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                       llama_hybrid_cycle_stage::SEND_ACTIVATION, 1, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                       llama_hybrid_cycle_stage::SEND_PARTIAL, 1, 0));
    assert(ring.pending(llama_hybrid_cycle_queue::PHONE_GPU) == 1);
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FFN_DESTINATION, 1, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::REDUCE, 1, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FULL_LAYER, 2, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::ATTN_PRE, 3, 0));

    // P2C: the remote PC FFN finishes before Phone's local FFN sends Send #2.
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                       llama_hybrid_cycle_stage::SEND_ACTIVATION, 3, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::FFN_DESTINATION, 3, 0));
    expect_empty(ring, llama_hybrid_cycle_queue::PC_CPU);
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FFN_SOURCE, 3, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                       llama_hybrid_cycle_stage::SEND_PARTIAL, 3, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::REDUCE, 3, 0));

    assert(ring.finished());
    assert(ring.output_count() == 4);
    expect_empty(ring, llama_hybrid_cycle_queue::PC_CPU);
    expect_empty(ring, llama_hybrid_cycle_queue::PHONE_GPU);
    expect_empty(ring, llama_hybrid_cycle_queue::PC_TO_PHONE);
    expect_empty(ring, llama_hybrid_cycle_queue::PHONE_TO_PC);
}

// A FULL_LAYER may release its predecessor credit as soon as Attention has
// fenced the last input read. Its own output is still unpublished until the
// task completes. Completion must not release the same predecessor twice.
static void test_full_input_consumed_credit_release() {
    auto layout = llama_hybrid_build_cycle_layout(8, 4, 2);
    layout.pc_cuda_full = true;
    llama_hybrid_cycle_task_ring ring(layout, 3, 2, true);
    ring.start();

    auto l0c0 = take(ring, llama_hybrid_cycle_queue::PC_CUDA,
                     llama_hybrid_cycle_stage::FULL_LAYER, 0, 0);
    auto l0c1 = take(ring, llama_hybrid_cycle_queue::PC_CUDA,
                     llama_hybrid_cycle_stage::FULL_LAYER, 0, 1);
    ring.complete(l0c0);
    ring.complete(l0c1);

    auto l1c0 = take(ring, llama_hybrid_cycle_queue::PC_CUDA,
                     llama_hybrid_cycle_stage::FULL_LAYER, 1, 0);
    assert(ring.mark_full_input_consumed(1, 0));
    assert(!ring.mark_full_input_consumed(1, 0));

    // L0 C2 was blocked by window=2. It is admitted immediately after L1 C0
    // consumes L0 C0, even though L1 C0 itself has not completed yet.
    auto l1c1 = take(ring, llama_hybrid_cycle_queue::PC_CUDA,
                     llama_hybrid_cycle_stage::FULL_LAYER, 1, 1);
    auto l0c2 = take(ring, llama_hybrid_cycle_queue::PC_CUDA,
                     llama_hybrid_cycle_stage::FULL_LAYER, 0, 2);
    assert(l1c1.layer == 1 && l1c1.chunk == 1);
    assert(l0c2.layer == 0 && l0c2.chunk == 2);

    // This exercises the completion fallback guard: no credit underflow and
    // no second predecessor release after the early handoff.
    ring.complete(l1c0);
}

// Out-of-order predecessor completions must never enqueue later-layer
// Attention for C2 before C0 and C1: layer-local KV causality is preserved.
static void test_causal_entry_order() {
    llama_hybrid_cycle_task_ring ring(llama_hybrid_build_cycle_layout(4, 1, 1), 3);
    ring.start();
    const auto a = take(ring, llama_hybrid_cycle_queue::PC_CPU,
                        llama_hybrid_cycle_stage::FULL_LAYER, 0, 0);
    const auto b = take(ring, llama_hybrid_cycle_queue::PC_CPU,
                        llama_hybrid_cycle_stage::FULL_LAYER, 0, 1);
    const auto c = take(ring, llama_hybrid_cycle_queue::PC_CPU,
                        llama_hybrid_cycle_stage::FULL_LAYER, 0, 2);
    ring.complete(c);
    expect_empty(ring, llama_hybrid_cycle_queue::PC_CPU);
    ring.complete(a);
    auto next0 = take(ring, llama_hybrid_cycle_queue::PC_CPU,
                      llama_hybrid_cycle_stage::ATTN_PRE, 1, 0);
    ring.complete(b);
    const auto next1 = take(ring, llama_hybrid_cycle_queue::PC_CPU,
                            llama_hybrid_cycle_stage::ATTN_PRE, 1, 1);
    const auto next2 = take(ring, llama_hybrid_cycle_queue::PC_CPU,
                            llama_hybrid_cycle_stage::ATTN_PRE, 1, 2);
    assert(next0.chunk == 0 && next1.chunk == 1 && next2.chunk == 2);
}

static void test_two_cycles_and_cuda_tail() {
    const auto layout = llama_hybrid_build_cycle_layout(48, 10, 8);
    llama_hybrid_cycle_task_ring ring(layout, 4);
    ring.start();

    std::array<int, 7> stage_counts {};
    std::array<int, 5> queue_counts {};
    constexpr std::array<llama_hybrid_cycle_queue, 5> queues {
        llama_hybrid_cycle_queue::PC_CPU,
        llama_hybrid_cycle_queue::PHONE_GPU,
        llama_hybrid_cycle_queue::PC_CUDA,
        llama_hybrid_cycle_queue::PC_TO_PHONE,
        llama_hybrid_cycle_queue::PHONE_TO_PC,
    };

    for (int count = 0; count < 10000 && !ring.finished(); ++count) {
        bool progressed = false;
        for (int i = 0; i < 5; ++i) {
            const auto q = queues[(size_t(count) + i) % queues.size()];
            llama_hybrid_cycle_task task;
            if (!ring.try_pop(q, task)) {
                continue;
            }
            progressed = true;
            ++stage_counts[static_cast<size_t>(task.stage)];
            ++queue_counts[static_cast<size_t>(q)];
            if (task.stage == llama_hybrid_cycle_stage::SEND_ACTIVATION ||
                task.stage == llama_hybrid_cycle_stage::SEND_PARTIAL) {
                assert(task.layer == 10 || task.layer == 19 ||
                       task.layer == 30 || task.layer == 39);
                assert((task.layer == 10 || task.layer == 30) ?
                    task.queue == llama_hybrid_cycle_queue::PC_TO_PHONE :
                    task.queue == llama_hybrid_cycle_queue::PHONE_TO_PC);
            }
            ring.complete(task);
        }
        assert(progressed || ring.finished());
    }

    assert(ring.finished());
    assert(ring.output_count() == size_t(48 * 4));
    assert(stage_counts[static_cast<size_t>(llama_hybrid_cycle_stage::FULL_LAYER)] == 44 * 4);
    assert(stage_counts[static_cast<size_t>(llama_hybrid_cycle_stage::ATTN_PRE)] == 4 * 4);
    assert(stage_counts[static_cast<size_t>(llama_hybrid_cycle_stage::REDUCE)] == 4 * 4);
    assert(queue_counts[static_cast<size_t>(llama_hybrid_cycle_queue::PC_CUDA)] == 8 * 4);
    assert(queue_counts[static_cast<size_t>(llama_hybrid_cycle_queue::PC_TO_PHONE)] == 2 * 4 * 2);
    assert(queue_counts[static_cast<size_t>(llama_hybrid_cycle_queue::PHONE_TO_PC)] == 2 * 4 * 2);
    for (auto q : queues) {
        expect_empty(ring, q);
    }
}

static void test_reject_invalid_completions() {
    llama_hybrid_cycle_task_ring ring(llama_hybrid_build_cycle_layout(4, 1, 1), 1);
    ring.start();
    auto task = take(ring, llama_hybrid_cycle_queue::PC_CPU,
                     llama_hybrid_cycle_stage::FULL_LAYER, 0, 0);
    ring.complete(task);
    try {
        ring.complete(task);
        assert(false && "duplicate completion accepted");
    } catch (const std::logic_error &) {
    }
    try {
        ring.start();
        assert(false && "duplicate start accepted");
    } catch (const std::logic_error &) {
    }
}

// The real P2C diagnostic binds these tasks to Phone GPU and RPC. This
// CPU-only unit test exercises the SAME per-queue worker runtime without
// importing any model, validating the stage dependency boundary.
static void test_p2c_fifo_worker_runtime() {
    llama_hybrid_cycle_layout fragment;
    fragment.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_P2C};
    std::array<std::atomic<int>, 7> completed {};
    for (auto & count : completed) {
        count.store(0);
    }
    const auto done = [&](llama_hybrid_cycle_stage stage) {
        return completed[static_cast<size_t>(stage)].load();
    };
    const bool ok = llama_hybrid_cycle_run_fifo(
        fragment, 1, [&](const llama_hybrid_cycle_task & task) {
            assert(task.layer == 0 && task.chunk == 0);
            switch (task.stage) {
                case llama_hybrid_cycle_stage::ATTN_PRE:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_GPU);
                    break;
                case llama_hybrid_cycle_stage::SEND_ACTIVATION:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_TO_PC);
                    assert(done(llama_hybrid_cycle_stage::ATTN_PRE));
                    break;
                case llama_hybrid_cycle_stage::FFN_SOURCE:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_GPU);
                    assert(done(llama_hybrid_cycle_stage::ATTN_PRE));
                    break;
                case llama_hybrid_cycle_stage::FFN_DESTINATION:
                    assert(task.queue == llama_hybrid_cycle_queue::PC_CPU);
                    assert(done(llama_hybrid_cycle_stage::SEND_ACTIVATION));
                    break;
                case llama_hybrid_cycle_stage::SEND_PARTIAL:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_TO_PC);
                    assert(done(llama_hybrid_cycle_stage::FFN_SOURCE));
                    break;
                case llama_hybrid_cycle_stage::REDUCE:
                    assert(task.queue == llama_hybrid_cycle_queue::PC_CPU);
                    assert(done(llama_hybrid_cycle_stage::FFN_DESTINATION));
                    assert(done(llama_hybrid_cycle_stage::SEND_PARTIAL));
                    break;
                case llama_hybrid_cycle_stage::FULL_LAYER:
                    assert(false && "P2C must not dispatch FULL_LAYER");
                    return false;
            }
            ++completed[static_cast<size_t>(task.stage)];
            return true;
        });
    assert(ok);
    assert(done(llama_hybrid_cycle_stage::REDUCE) == 1);
    assert(done(llama_hybrid_cycle_stage::SEND_ACTIVATION) == 1);
    assert(done(llama_hybrid_cycle_stage::SEND_PARTIAL) == 1);
    assert(done(llama_hybrid_cycle_stage::FFN_SOURCE) == 1);
    assert(done(llama_hybrid_cycle_stage::FFN_DESTINATION) == 1);
}

// P2C must not wait for Phone FFN completion before PC starts its own
// FFN. The Phone GPU worker deliberately blocks its FFN_SOURCE until the
// PC CPU worker has entered FFN_DESTINATION. A single global stage/Fence
// barrier would deadlock this test; independent worker queues must progress.
static void test_p2c_cpu_ffn_runs_while_phone_ffn_pending() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_P2C};
    std::mutex state_mu;
    std::condition_variable state_cv;
    bool phone_ffn_entered = false;
    bool pc_ffn_entered = false;
    bool reduce_ran = false;
    const bool ok = llama_hybrid_cycle_run_fifo(
        layout, 1, [&](const llama_hybrid_cycle_task & task) {
            switch (task.stage) {
                case llama_hybrid_cycle_stage::ATTN_PRE:
                case llama_hybrid_cycle_stage::SEND_ACTIVATION:
                case llama_hybrid_cycle_stage::SEND_PARTIAL:
                    return true;
                case llama_hybrid_cycle_stage::FFN_SOURCE: {
                    std::unique_lock<std::mutex> lock(state_mu);
                    phone_ffn_entered = true;
                    state_cv.notify_all();
                    // The PC CPU must be allowed to execute while the Phone
                    // GPU is still busy with its FFN.
                    return state_cv.wait_for(
                        lock, std::chrono::seconds(2),
                        [&] { return pc_ffn_entered; });
                }
                case llama_hybrid_cycle_stage::FFN_DESTINATION: {
                    std::unique_lock<std::mutex> lock(state_mu);
                    // Wait until the Phone task is actually in flight:
                    // prove overlap rather than a coincidental task order.
                    if (!state_cv.wait_for(
                            lock, std::chrono::seconds(2),
                            [&] { return phone_ffn_entered; })) {
                        return false;
                    }
                    pc_ffn_entered = true;
                    state_cv.notify_all();
                    return true;
                }
                case llama_hybrid_cycle_stage::REDUCE: {
                    std::lock_guard<std::mutex> lock(state_mu);
                    assert(phone_ffn_entered && pc_ffn_entered);
                    reduce_ran = true;
                    return true;
                }
                case llama_hybrid_cycle_stage::FULL_LAYER:
                    return false;
            }
            return false;
        });
    assert(ok);
    assert(reduce_ran);
}

// Four chunks, two GPU allocations/Attention entries max in flight.
// The Phone compute FIFO may start chunk N+2 only after a preceding
// Reduce has returned its entry credit. Each READY and PARTIAL must join
// with the PC FFN of the same chunk before the PC owns final output.
static void test_bounded_multi_chunk_p2c_fifo() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_P2C};
    constexpr int chunks = 4;
    std::array<std::atomic<uint32_t>, chunks> stage_bits {};
    for (auto & flags : stage_bits) {
        flags.store(0);
    }
    std::atomic<int> attention_count {0};
    std::atomic<int> reduce_count {0};
    const auto bit = [](llama_hybrid_cycle_stage stage) -> uint32_t {
        return uint32_t(1) << static_cast<int>(stage);
    };

    const bool ok = llama_hybrid_cycle_run_fifo(
        layout, chunks, [&](const llama_hybrid_cycle_task & task) {
            assert(task.layer == 0 &&
                   task.chunk >= 0 && task.chunk < chunks);
            const uint32_t before = stage_bits[size_t(task.chunk)].load();
            switch (task.stage) {
                case llama_hybrid_cycle_stage::ATTN_PRE:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_GPU);
                    assert(attention_count.fetch_add(1) == task.chunk);
                    assert(task.chunk < reduce_count.load() + 2);
                    break;
                case llama_hybrid_cycle_stage::SEND_ACTIVATION:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_TO_PC);
                    assert(before & bit(llama_hybrid_cycle_stage::ATTN_PRE));
                    break;
                case llama_hybrid_cycle_stage::FFN_SOURCE:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_GPU);
                    assert(before & bit(llama_hybrid_cycle_stage::ATTN_PRE));
                    break;
                case llama_hybrid_cycle_stage::FFN_DESTINATION:
                    assert(task.queue == llama_hybrid_cycle_queue::PC_CPU);
                    assert(before & bit(llama_hybrid_cycle_stage::SEND_ACTIVATION));
                    break;
                case llama_hybrid_cycle_stage::SEND_PARTIAL:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_TO_PC);
                    assert(before & bit(llama_hybrid_cycle_stage::FFN_SOURCE));
                    break;
                case llama_hybrid_cycle_stage::REDUCE:
                    assert(task.queue == llama_hybrid_cycle_queue::PC_CPU);
                    assert(before & bit(llama_hybrid_cycle_stage::FFN_DESTINATION));
                    assert(before & bit(llama_hybrid_cycle_stage::SEND_PARTIAL));
                    ++reduce_count;
                    break;
                case llama_hybrid_cycle_stage::FULL_LAYER:
                    assert(false);
                    return false;
            }
            const uint32_t previous =
                stage_bits[size_t(task.chunk)].fetch_or(bit(task.stage));
            assert(!(previous & bit(task.stage)));
            return true;
        }, 2);
    assert(ok);
    assert(attention_count == chunks && reduce_count == chunks);
    for (const auto & flags : stage_bits) {
        assert((flags.load() & bit(llama_hybrid_cycle_stage::REDUCE)) != 0);
    }
}

// C2P reverses the roles without changing the queue contract. Four real
// chunks must keep Attention causal on PC, send READY before PARTIAL on one
// PC->Phone FIFO, and Reduce ONLY on Phone after both shards are ready.
static void test_bounded_multi_chunk_c2p_fifo() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_C2P};
    constexpr int chunks = 4;
    std::array<std::atomic<uint32_t>, chunks> flags {};
    for (auto & f : flags) {
        f.store(0);
    }
    std::atomic<int> attn {0};
    std::atomic<int> reductions {0};
    const auto bit = [](llama_hybrid_cycle_stage stage) -> uint32_t {
        return uint32_t(1) << static_cast<int>(stage);
    };
    const bool ok = llama_hybrid_cycle_run_fifo(
        layout, chunks, [&](const llama_hybrid_cycle_task & task) {
            assert(task.layer == 0 && task.chunk >= 0 && task.chunk < chunks);
            const auto seen = flags[size_t(task.chunk)].load();
            switch (task.stage) {
                case llama_hybrid_cycle_stage::ATTN_PRE:
                    assert(task.queue == llama_hybrid_cycle_queue::PC_CPU);
                    assert(attn.fetch_add(1) == task.chunk);
                    assert(task.chunk < reductions.load() + 2);
                    break;
                case llama_hybrid_cycle_stage::FFN_SOURCE:
                    assert(task.queue == llama_hybrid_cycle_queue::PC_CPU);
                    assert(seen & bit(llama_hybrid_cycle_stage::ATTN_PRE));
                    break;
                case llama_hybrid_cycle_stage::SEND_ACTIVATION:
                    assert(task.queue == llama_hybrid_cycle_queue::PC_TO_PHONE);
                    assert(seen & bit(llama_hybrid_cycle_stage::ATTN_PRE));
                    break;
                case llama_hybrid_cycle_stage::SEND_PARTIAL:
                    assert(task.queue == llama_hybrid_cycle_queue::PC_TO_PHONE);
                    assert(seen & bit(llama_hybrid_cycle_stage::FFN_SOURCE));
                    assert(seen & bit(llama_hybrid_cycle_stage::SEND_ACTIVATION));
                    break;
                case llama_hybrid_cycle_stage::FFN_DESTINATION:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_GPU);
                    assert(seen & bit(llama_hybrid_cycle_stage::SEND_ACTIVATION));
                    break;
                case llama_hybrid_cycle_stage::REDUCE:
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_GPU);
                    assert(seen & bit(llama_hybrid_cycle_stage::SEND_PARTIAL));
                    assert(seen & bit(llama_hybrid_cycle_stage::FFN_DESTINATION));
                    ++reductions;
                    break;
                default:
                    assert(false && "C2P task unexpectedly on FULL_LAYER");
                    return false;
            }
            assert(!(flags[size_t(task.chunk)].fetch_or(bit(task.stage)) &
                     bit(task.stage)));
            return true;
        }, 2);
    assert(ok);
    assert(attn == chunks && reductions == chunks);
}

// Reduce priority is an opt-in LOCAL dequeue policy, not a bypass of the
// two-FFN join, Phone causal entry order or transfer FIFO ordering. Arrange
// for P2C C0 Reduce to become ready behind C1 FFN on PC CPU.
static void test_p2c_ready_reduce_priority() {
    for (const bool reduce_first : {false, true}) {
        llama_hybrid_cycle_layout layout;
        layout.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_P2C};
        llama_hybrid_cycle_task_ring ring(
            layout, 2, 2, false, reduce_first);
        ring.start();

        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                           llama_hybrid_cycle_stage::ATTN_PRE, 0, 0));
        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                           llama_hybrid_cycle_stage::ATTN_PRE, 0, 1));
        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                           llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, 0));
        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                           llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, 1));

        // Both destination FFNs are ready, but neither phone partial has
        // been received. Reduce must NOT be dispatched early.
        auto pc_ffn0 = take(ring, llama_hybrid_cycle_queue::PC_CPU,
                            llama_hybrid_cycle_stage::FFN_DESTINATION, 0, 0);
        ring.complete(pc_ffn0);
        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                           llama_hybrid_cycle_stage::FFN_SOURCE, 0, 0));
        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                           llama_hybrid_cycle_stage::SEND_PARTIAL, 0, 0));

        const auto first = take(
            ring, llama_hybrid_cycle_queue::PC_CPU,
            reduce_first ? llama_hybrid_cycle_stage::REDUCE :
                           llama_hybrid_cycle_stage::FFN_DESTINATION,
            0, reduce_first ? 0 : 1);
        ring.complete(first);
        const auto second = take(
            ring, llama_hybrid_cycle_queue::PC_CPU,
            reduce_first ? llama_hybrid_cycle_stage::FFN_DESTINATION :
                           llama_hybrid_cycle_stage::REDUCE,
            0, reduce_first ? 1 : 0);
        ring.complete(second);

        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                           llama_hybrid_cycle_stage::FFN_SOURCE, 0, 1));
        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                           llama_hybrid_cycle_stage::SEND_PARTIAL, 0, 1));
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                           llama_hybrid_cycle_stage::REDUCE, 0, 1));
        assert(ring.finished());
        assert(ring.output_count() == 2);
    }
}

// The P2C CPU Reduce is the sole producer of the next PC CUDA layer's
// input. A READY packet and a completed PC FFN alone must NOT enqueue CUDA.
// Completion of BOTH FFN shards and Reduce enqueues the CUDA task, without
// making the CPU wait for that CUDA work to finish.
static void test_p2c_reduce_handoff_to_cuda_queue() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {
        llama_hybrid_cycle_layer_kind::TENSOR_P2C,
        llama_hybrid_cycle_layer_kind::PC_CPU
    };
    layout.pc_cuda_full = true;
    llama_hybrid_cycle_task_ring ring(layout, 2, 2, true, true);
    ring.start();

    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::ATTN_PRE, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::ATTN_PRE, 0, 1));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                       llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                       llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, 1));

    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::FFN_DESTINATION, 0, 0));
    assert(ring.pending(llama_hybrid_cycle_queue::PC_CUDA) == 0);
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FFN_SOURCE, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                       llama_hybrid_cycle_stage::SEND_PARTIAL, 0, 0));
    assert(ring.pending(llama_hybrid_cycle_queue::PC_CUDA) == 0);
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 0));
    assert(ring.pending(llama_hybrid_cycle_queue::PC_CUDA) == 1);
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CUDA,
                       llama_hybrid_cycle_stage::FULL_LAYER, 1, 0));

    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::FFN_DESTINATION, 0, 1));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FFN_SOURCE, 0, 1));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                       llama_hybrid_cycle_stage::SEND_PARTIAL, 0, 1));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 1));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CUDA,
                       llama_hybrid_cycle_stage::FULL_LAYER, 1, 1));
    assert(ring.finished());
    assert(ring.output_count() == 4);
}

// The full X=5,Y=1 period must NOT run its C2P/P2C boundaries as
// independent one-layer tests. The output of EACH preceding physical
// layer's chunk is a dependency on the next layer's matching chunk.
static void test_complete_period_fifo_handoffs() {
    const auto layout = llama_hybrid_build_cycle_layout(48, 5, 1);
    llama_hybrid_cycle_layout period = layout;
    period.kinds.resize(8);
    period.tail_begin = 8;
    constexpr int chunks = 4;
    constexpr int n_layers = 8;
    std::array<std::array<std::atomic<bool>, chunks>, n_layers> done {};
    std::array<std::atomic<int>, n_layers> entered {};
    for (int l = 0; l < n_layers; ++l) {
        entered[l].store(0);
        for (int c = 0; c < chunks; ++c) {
            done[l][c].store(false);
        }
    }
    const bool ok = llama_hybrid_cycle_run_fifo(
        period, chunks, [&](const llama_hybrid_cycle_task & task) {
            const int l = task.layer, c = task.chunk;
            assert(l >= 0 && l < n_layers && c >= 0 && c < chunks);
            if (task.stage == llama_hybrid_cycle_stage::FULL_LAYER ||
                task.stage == llama_hybrid_cycle_stage::ATTN_PRE) {
                assert(entered[l].fetch_add(1) == c);
                // Previous layer's COMPLETE has already returned its chunk
                // output and published it before this task can be enqueued.
                assert(l == 0 || done[l - 1][c].load());
            }
            const auto kind = period.kinds[size_t(l)];
            if (kind == llama_hybrid_cycle_layer_kind::PC_CPU) {
                assert(task.queue == llama_hybrid_cycle_queue::PC_CPU);
                assert(task.stage == llama_hybrid_cycle_stage::FULL_LAYER);
            } else if (kind == llama_hybrid_cycle_layer_kind::PHONE_GPU) {
                assert(task.queue == llama_hybrid_cycle_queue::PHONE_GPU);
                assert(task.stage == llama_hybrid_cycle_stage::FULL_LAYER);
            } else if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ||
                       kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
                const bool c2p = kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P;
                if (task.stage == llama_hybrid_cycle_stage::SEND_ACTIVATION ||
                    task.stage == llama_hybrid_cycle_stage::SEND_PARTIAL) {
                    assert(task.queue == (c2p ?
                        llama_hybrid_cycle_queue::PC_TO_PHONE :
                        llama_hybrid_cycle_queue::PHONE_TO_PC));
                }
            } else {
                assert(false && "unexpected CUDA tail inside first Cycle period");
            }
            if (task.stage == llama_hybrid_cycle_stage::FULL_LAYER ||
                task.stage == llama_hybrid_cycle_stage::REDUCE) {
                assert(!done[l][c].exchange(true));
            }
            return true;
        }, 2, true);
    assert(ok);
    for (int l = 0; l < n_layers; ++l) {
        assert(entered[l] == chunks);
        for (int c = 0; c < chunks; ++c) {
            assert(done[l][c].load());
        }
    }
}


// PC CUDA full layer: one task, entirely on PC_CUDA. Keep the causal
// chunk order and downstream layer credit rules of the original ring.
static void test_pc_cuda_full_fifo() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {
        llama_hybrid_cycle_layer_kind::PC_CPU,
        llama_hybrid_cycle_layer_kind::PC_CPU
    };
    layout.pc_cuda_full = true;
    constexpr int chunks = 3;
    llama_hybrid_cycle_task_ring ring(layout, chunks, 2, true);
    std::array<std::array<bool, chunks>, 2> completed {};
    std::array<int, 2> next {};
    ring.start();
    for (int steps = 0; steps < 100 && !ring.finished(); ++steps) {
        llama_hybrid_cycle_task task;
        assert(ring.try_pop(llama_hybrid_cycle_queue::PC_CUDA, task));
        assert(task.owner == llama_hybrid_cycle_owner::PC_CUDA);
        assert(task.queue == llama_hybrid_cycle_queue::PC_CUDA);
        assert(task.stage == llama_hybrid_cycle_stage::FULL_LAYER);
        const int l = task.layer;
        const int c = task.chunk;
        assert(l >= 0 && l < 2);
        assert(c == next[size_t(l)]++);
        assert(l == 0 || completed[size_t(l-1)][size_t(c)]);
        completed[size_t(l)][size_t(c)] = true;
        ring.complete(task);
    }
    assert(ring.finished());
    assert(ring.output_count() == size_t(2 * chunks));
    for (const auto & row : completed) {
        for (bool done : row) assert(done);
    }
    expect_empty(ring, llama_hybrid_cycle_queue::PC_CUDA);
    expect_empty(ring, llama_hybrid_cycle_queue::PC_CPU);
}

// V4-only guard: C1 may have BOTH FFN branches ready while C0 still
// owns an in-flight Phone FFN. Do not dispatch C1 REDUCE (and therefore
// a premature L5 waiting task) before the C0 REDUCE has finished.
static void test_v4_ordered_c2p_reduce() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_C2P};
    llama_hybrid_cycle_task_ring ring(
        layout, 2, 2, false, false, true);
    ring.start();

    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::ATTN_PRE, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::ATTN_PRE, 0, 1));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::FFN_SOURCE, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::FFN_SOURCE, 0, 1));

    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                       llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                       llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, 1));
    const auto phone_ffn_c0 = take(
        ring, llama_hybrid_cycle_queue::PHONE_GPU,
        llama_hybrid_cycle_stage::FFN_DESTINATION, 0, 0);
    const auto phone_ffn_c1 = take(
        ring, llama_hybrid_cycle_queue::PHONE_GPU,
        llama_hybrid_cycle_stage::FFN_DESTINATION, 0, 1);
    ring.complete(phone_ffn_c1); // C1's Phone FFN finishes first.
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                       llama_hybrid_cycle_stage::SEND_PARTIAL, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                       llama_hybrid_cycle_stage::SEND_PARTIAL, 0, 1));

    // C1 is ready on BOTH branches, but it must not jump ahead of C0.
    expect_empty(ring, llama_hybrid_cycle_queue::PHONE_GPU);
    ring.complete(phone_ffn_c0);
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 1));
    assert(ring.finished());
}

// V5 policy: the PC partial may arrive before or after Phone C2P FFN.
// No unready Reduce may block MAIN; admitted reduces preserve Chunk order,
// but the active Chunk's continuation stays ahead of following reductions.
static void test_v5_phone_main_priority() {
    using sched = llama_hybrid_cycle_phone_local;
    sched ring(3, 2);
    ring.phone_ready(0);
    ring.phone_ready(1);
    ring.phone_ready(2);
    sched::task t {};
    // PC chunk1 is ready before C0, but may not enter MAIN.
    ring.pc_partial_ready(1);
    assert(ring.pop(t) && t.op == sched::stage::C2P_FFN && t.chunk == 0);
    ring.complete(t);
    assert(ring.main_pending() == 0);
    assert(ring.pop(t) && t.op == sched::stage::C2P_FFN && t.chunk == 1);
    ring.complete(t);
    assert(ring.main_pending() == 0);
    ring.pc_partial_ready(0);
    assert(ring.main_pending() == 2);
    assert(ring.pop(t) && t.op == sched::stage::C2P_REDUCE && t.chunk == 0);
    ring.complete(t);
    // C0 MUST beat the already-queued C1 Reduce.
    const sched::stage expected[] = {
        sched::stage::PHONE_FULL, sched::stage::PHONE_FULL,
        sched::stage::P2C_ATTN, sched::stage::P2C_READY_STAGED,
        sched::stage::P2C_FFN, sched::stage::P2C_PARTIAL_STAGED
    };
    for (int step = 0; step < 6; ++step) {
        assert(ring.pop(t) && t.chunk == 0 && t.op == expected[step]);
        if (t.op == sched::stage::PHONE_FULL)
            assert(t.full_layer == step);
        ring.complete(t);
    }
    assert(ring.completed() == 1);
    assert(ring.pop(t) && t.op == sched::stage::C2P_REDUCE && t.chunk == 1);
    ring.complete(t);
    for (auto op : expected) {
        assert(ring.pop(t) && t.chunk == 1 && t.op == op);
        ring.complete(t);
    }
    assert(ring.completed() == 2);
    assert(ring.pop(t) && t.op == sched::stage::C2P_FFN && t.chunk == 2);
    ring.complete(t);
    assert(!ring.pop(t)); // No GPU Worker wait for C2's network data.
    ring.pc_partial_ready(2);
    assert(ring.pop(t) && t.op == sched::stage::C2P_REDUCE && t.chunk == 2);
    ring.complete(t);
    for (auto op : expected) {
        assert(ring.pop(t) && t.chunk == 2 && t.op == op);
        ring.complete(t);
    }
    assert(ring.finished());
}

static void test_v5_phone_rx_first() {
    using sched = llama_hybrid_cycle_phone_local;
    sched ring(1, 0);
    ring.pc_partial_ready(0);
    ring.phone_ready(0);
    sched::task t {};
    assert(ring.pop(t) && t.op == sched::stage::C2P_FFN);
    ring.complete(t);
    assert(ring.pop(t) && t.op == sched::stage::C2P_REDUCE);
    ring.complete(t);
    const sched::stage sequence[] = {
        sched::stage::P2C_ATTN, sched::stage::P2C_READY_STAGED,
        sched::stage::P2C_FFN, sched::stage::P2C_PARTIAL_STAGED
    };
    for (auto op : sequence) {
        assert(ring.pop(t) && t.op == op);
        ring.complete(t);
    }
    assert(ring.finished());
}

// V5 PC proxy: a future C2P Reduce must NOT occupy the only PHONE_GPU
// callback worker ahead of a now-ready MAIN continuation. This also avoids
// blocking on C2P's yet-unregistered window-credit-dependent Region.
static void test_v5_pc_phone_proxy_main_priority() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {
        llama_hybrid_cycle_layer_kind::TENSOR_C2P,
        llama_hybrid_cycle_layer_kind::PHONE_GPU,
        llama_hybrid_cycle_layer_kind::PHONE_GPU,
        llama_hybrid_cycle_layer_kind::TENSOR_P2C
    };
    llama_hybrid_cycle_task_ring ring(
        layout, 2, 2, true, false, true, true);
    ring.start();
    for (int c = 0; c < 2; ++c) {
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                           llama_hybrid_cycle_stage::ATTN_PRE, 0, c));
    }
    for (int c = 0; c < 2; ++c) {
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                           llama_hybrid_cycle_stage::FFN_SOURCE, 0, c));
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                           llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, c));
        ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                           llama_hybrid_cycle_stage::FFN_DESTINATION, 0, c));
    }
    for (int c = 0; c < 2; ++c) {
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                           llama_hybrid_cycle_stage::SEND_PARTIAL, 0, c));
    }
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 0));
    // C1's REDUCE was already queued. V5 instead completes C0's
    // downstream Phone and P2C stages at the next scheduling boundary.
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FULL_LAYER, 1, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FULL_LAYER, 2, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::ATTN_PRE, 3, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FFN_SOURCE, 3, 0));
    // Now the proxy may return to C1 while independent P2C TX progresses.
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 1));
}

// Trace is observational: one producer per queue, no change to credits
// or task ordering. Both entry-ready and worker intervals are monotonic.
static void test_chunk_trace_observational() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {llama_hybrid_cycle_layer_kind::PC_CPU,
                    llama_hybrid_cycle_layer_kind::PC_CPU};
    llama_hybrid_cycle_chunk_timeline timeline;
    std::atomic<int> executed {0};
    const bool ok = llama_hybrid_cycle_run_fifo(
        layout, 2, [&](const llama_hybrid_cycle_task & task) {
            assert(task.stage == llama_hybrid_cycle_stage::FULL_LAYER);
            ++executed;
            return true;
        }, 1, true, nullptr, false, false, false, &timeline);
    assert(ok && executed.load() == 4);
    size_t samples = 0;
    for (const auto & queue : timeline.by_queue) {
        for (const auto & sample : queue) {
            assert(sample.success);
            assert(sample.task.entry_ready_at.time_since_epoch().count() != 0);
            assert(sample.task.entry_ready_at <= sample.task.queued_at);
            assert(sample.task.queued_at <= sample.start);
            assert(sample.start <= sample.finish);
            ++samples;
        }
    }
    assert(samples == 4);
}

// A ready C2P Reduce must start the older Chunk's Phone MAIN ahead of
// the next Chunk's FFN_DESTINATION region preparation on the PC proxy.
// This is an opt-in V5 scheduling policy, not a Phone GPU kernel ordering test.
static void test_v5_ready_c2p_reduce_beats_next_region_prepare() {
    llama_hybrid_cycle_layout layout;
    layout.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_C2P};
    llama_hybrid_cycle_task_ring ring(
        layout, 2, 2, true, false, true, true);
    ring.start();
    // start() queues ATTN_PRE for C0 and C1. Completing C0 Attention
    // appends C0 FFN_SOURCE AFTER C1 Attention in the SAME PC_CPU FIFO.
    // Drain both Attention tasks before any FFN_SOURCE, and likewise
    // preserve SEND_ACTIVATION before SEND_PARTIAL on the PC->Phone FIFO.
    // Only PHONE_GPU has the opt-in V5 priority policy exercised below.
    for (int c = 0; c < 2; ++c) {
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                           llama_hybrid_cycle_stage::ATTN_PRE, 0, c));
    }
    for (int c = 0; c < 2; ++c) {
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                           llama_hybrid_cycle_stage::FFN_SOURCE, 0, c));
    }
    for (int c = 0; c < 2; ++c) {
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                           llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, c));
    }
    for (int c = 0; c < 2; ++c) {
        ring.complete(take(ring, llama_hybrid_cycle_queue::PC_TO_PHONE,
                           llama_hybrid_cycle_stage::SEND_PARTIAL, 0, c));
    }
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FFN_DESTINATION, 0, 0));
    // C1 FFN_DESTINATION is still queued. C0 Reduce is now ready.
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FFN_DESTINATION, 0, 1));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 1));
    assert(ring.finished());
}

// The dispatch path must honor the true backend queue and ownership,
// not merely dispatch any task with a plausible layer/stage index.
static void test_cycle_runtime_physical_route_guard() {
    llama_hybrid_cycle_layout cuda_layout;
    cuda_layout.kinds = {llama_hybrid_cycle_layer_kind::PC_CPU};
    cuda_layout.pc_cuda_full = true;
    llama_hybrid_cycle_stage_dispatcher cuda(cuda_layout);
    int executed = 0;
    assert(cuda.bind(0, llama_hybrid_cycle_stage::FULL_LAYER,
                     [&](const llama_hybrid_cycle_task &) {
                         ++executed;
                         return true;
                     }));
    std::string reason;
    assert(cuda.validate(reason));
    llama_hybrid_cycle_task task {};
    task.layer = 0;
    task.chunk = 0;
    task.stage = llama_hybrid_cycle_stage::FULL_LAYER;
    task.owner = llama_hybrid_cycle_owner::PC_CUDA;
    task.queue = llama_hybrid_cycle_queue::PC_CUDA;
    assert(cuda.execute(task));
    task.queue = llama_hybrid_cycle_queue::PC_CPU;
    assert(!cuda.execute(task));
    assert(executed == 1);

    llama_hybrid_cycle_layout c2p_layout;
    c2p_layout.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_C2P};
    llama_hybrid_cycle_stage_dispatcher c2p(c2p_layout);
    assert(c2p.bind(0, llama_hybrid_cycle_stage::SEND_PARTIAL,
                    [&](const llama_hybrid_cycle_task &) {
                        ++executed;
                        return true;
                    }));
    task.stage = llama_hybrid_cycle_stage::SEND_PARTIAL;
    task.owner = llama_hybrid_cycle_owner::PC_CPU;
    task.queue = llama_hybrid_cycle_queue::PC_TO_PHONE;
    assert(c2p.execute(task));
    task.queue = llama_hybrid_cycle_queue::PHONE_TO_PC;
    assert(!c2p.execute(task));
    task.queue = llama_hybrid_cycle_queue::PC_TO_PHONE;
    task.owner = llama_hybrid_cycle_owner::PHONE_GPU;
    assert(!c2p.execute(task));
    assert(executed == 2);
}

static void test_cycle_runtime_missing_stage_fails_closed() {
    const auto layout = llama_hybrid_build_cycle_layout(8, 4, 2);
    llama_hybrid_cycle_stage_dispatcher stages(layout);
    std::string reason;
    assert(!stages.validate(reason));
    assert(reason.find("UNBOUND_PHYSICAL_STAGE") != std::string::npos);
    assert(!stages.bind(0, llama_hybrid_cycle_stage::SEND_PARTIAL,
                        [](const llama_hybrid_cycle_task &) { return true; }));
    assert(stages.bind(0, llama_hybrid_cycle_stage::FULL_LAYER,
                       [](const llama_hybrid_cycle_task &) { return true; }));
    assert(!stages.bind(0, llama_hybrid_cycle_stage::FULL_LAYER,
                        [](const llama_hybrid_cycle_task &) { return true; }));
    llama_hybrid_cycle_runtime_config config;
    config.layout = layout;
    config.n_chunks = 2;
    llama_hybrid_cycle_runtime runtime(config);
    assert(!runtime.run(stages)); // No workers launch with unbound stages.

    // Fail closed on a foreign physical layout rather than reusing the
    // wrong model's backend placement or a different C2P/P2C direction.
    const auto foreign = llama_hybrid_build_cycle_layout(8, 5, 1);
    llama_hybrid_cycle_stage_dispatcher wrong(foreign);
    assert(!runtime.run(wrong));
}

// Session tests do NOT start GPU/RPC work. They verify exactly the same
// chunk/seq descriptors used by V5, and prove [0,48) has no 8-layer barrier.
static void test_cycle_execution_session_8_and_48() {
    for (const int layers : {8, 48}) {
        llama_hybrid_cycle_execution_session_config cfg;
        cfg.layout = llama_hybrid_build_cycle_layout(layers, 4, 2);
        cfg.layout.pc_cuda_full = true;
        cfg.physical_layer_begin = 0;
        cfg.physical_layer_count = layers;
        cfg.tokens = 64;
        cfg.chunk_tokens = 16;
        cfg.wire_epoch = 1;
        cfg.token_position_base = 0;
        std::string reason;
        assert(llama_hybrid_cycle_validate_session_config(cfg, reason));
        llama_hybrid_cycle_execution_session<> session(cfg);
        assert(session.first_layer() == 0);
        assert(session.layer_count() == layers);
        assert(session.chunks_per_layer() == 4);
        assert(session.states().size() == size_t(layers));

        std::set<uint64_t> ids;
        std::vector<std::unique_ptr<llama_hybrid_cycle_physical_kv_cache>> kv;
        for (int l = 0; l < layers; ++l) {
            auto cache = std::make_unique<llama_hybrid_cycle_physical_kv_cache>();
            cache->committed_tokens = 64;
            assert(session.borrow_kv(l, cache.get()));
            assert(!session.borrow_kv(l, cache.get()));
            kv.push_back(std::move(cache));
            for (int c = 0; c < 4; ++c) {
                const auto & state = session.at(l, c);
                assert(state.layer == l && state.chunk == c);
                assert(state.start == int64_t(c * 16));
                assert(state.tokens == 16);
                assert(state.wire.layer == l);
                assert(state.wire.chunk == c);
                assert(state.wire.graph_id == 1);
                const auto seq = uint64_t(100000) +
                                 uint64_t(l + 1) * 1000 + uint64_t(c + 1);
                assert(state.wire.seq == seq);
                assert(ids.insert(seq).second);
            }
        }
        assert(ids.size() == size_t(layers * 4));
        assert(session.kv_complete(64));
        kv.back()->committed_tokens = 63;
        assert(!session.kv_complete(64));
        kv.back()->committed_tokens = 64;

        std::vector<float> received;
        assert(!session.copy_predecessor_host_input(1, 0, 2, received));
        auto & first = session.at(0, 0);
        first.host_output.assign(size_t(16 * 2), 7.0f);
        assert(!session.publish_host_output(0, 0, 3));  // Wrong width.
        assert(session.publish_host_output(0, 0, 2));
        assert(!session.publish_host_output(0, 0, 2)); // Duplicate publication.
        assert(session.copy_predecessor_host_input(1, 0, 2, received));
        assert(received == first.host_output);
        assert(!session.copy_predecessor_host_input(5, 0, 2, received));
        assert(!session.phone_predecessor_output(1, 0)); // PC -> PC.
        assert(!session.phone_predecessor_output(5, 0)); // Not yet ready.

        // PC CUDA activation ownership mirrors the Phone handoff contract:
        // completion alone does not retire a producer slot; only the adjacent
        // CUDA Full successor may move it after its Attention input fence.
        auto & cuda0 = session.at(0, 1);
        cuda0.cuda_output =
            std::make_unique<llama_hybrid_cycle_cuda_activation_slot>();
        assert(!session.retire_cuda_output_after_handoff(0, 2, 1));
        assert(cuda0.cuda_output);
        auto cuda_slot =
            session.retire_cuda_output_after_handoff(0, 1, 1);
        assert(cuda_slot);
        assert(session.cuda_device_handoffs() == 1);
        assert(!cuda0.cuda_output);
        assert(!session.retire_cuda_output_after_handoff(0, 1, 1));

        if (layers == 48) {
            // Exact P2C/PC CUDA handoff at the first Cycle boundary:
            // no conversion through Phone RPC/CPU oracle, no all-Chunk wait.
            auto & p2c = session.at(7, 0);
            p2c.host_output.assign(size_t(16 * 2), 11.0f);
            assert(!session.copy_predecessor_host_input(8, 0, 2, received));
            assert(session.publish_host_output(7, 0, 2));
            assert(session.copy_predecessor_host_input(8, 0, 2, received));
            assert(received == p2c.host_output);
        }

        // A GPU output is NOT retired by a task completion. Only the
        // consumer-side handoff path can move this Graph Slot.
        auto & c2p = session.at(4, 0);
        c2p.phone_output =
            std::make_unique<llama_hybrid_cycle_phone_ffn_slot>();
        c2p.phone_output->prepared_tokens = 16;
        assert(!session.retire_phone_output_after_handoff(4, 6, 0));
        assert(c2p.phone_output);
        auto slot = session.retire_phone_output_after_handoff(4, 5, 0);
        assert(slot && slot->prepared_tokens == 16);
        assert(!c2p.phone_output);
        assert(!session.retire_phone_output_after_handoff(4, 5, 0));
    }

    // Support a nonzero physical-layer base for later six-region binding,
    // while preserving the original wire IDs and absolute token positions.
    auto cfg = llama_hybrid_cycle_execution_session_config {};
    cfg.layout = llama_hybrid_build_cycle_layout(48, 4, 2);
    cfg.physical_layer_begin = 8;
    cfg.physical_layer_count = 8;
    cfg.tokens = 38;
    cfg.chunk_tokens = 16;
    cfg.wire_epoch = 1;
    cfg.token_position_base = 256;
    llama_hybrid_cycle_execution_session<> sub(cfg);
    assert(sub.chunks_per_layer() == 3);
    assert(sub.at(8, 0).start == 256);
    assert(sub.at(8, 2).start == 288);
    assert(sub.at(8, 2).tokens == 6);
    assert(sub.at(15, 2).wire.seq == 116003);

    cfg.physical_layer_begin = 43;
    std::string reason;
    assert(!llama_hybrid_cycle_validate_session_config(cfg, reason));
    assert(reason == "INVALID_PHYSICAL_LAYER_RANGE");
    cfg.physical_layer_begin = 0;
    cfg.chunk_tokens = 0;
    assert(!llama_hybrid_cycle_validate_session_config(cfg, reason));
    assert(reason == "INVALID_TOKEN_WINDOW");
    assert(llama_hybrid_cycle_session_wire_seq(0, 0, 0) == 1001);
    assert(llama_hybrid_cycle_session_wire_seq(1, 47, 3) == 148004);
    assert(llama_hybrid_cycle_session_wire_seq(0, 99, 0) == 0);

    // Production Prefill: the full 358-token Prompt is one execution session,
    // internally split into five 64-token Chunks plus one 38-token tail.
    llama_hybrid_cycle_execution_session_config whole;
    whole.layout = llama_hybrid_build_cycle_layout(48, 4, 2);
    whole.physical_layer_count = 48;
    whole.tokens = 358;
    whole.chunk_tokens = 64;
    whole.wire_epoch = 0;
    llama_hybrid_cycle_execution_session<> whole_session(whole);
    assert(whole_session.chunks_per_layer() == 6);
    for (int c = 0; c < 5; ++c) {
        assert(whole_session.at(0, c).start == int64_t(c * 64));
        assert(whole_session.at(0, c).tokens == 64);
    }
    assert(whole_session.at(0, 5).start == 320);
    assert(whole_session.at(0, 5).tokens == 38);
    assert(whole_session.at(47, 5).wire.seq == 48006);
}

// Verify the compiled libllama Cycle runtime across both the original
// eight-layer region and all 48 physical model layers. The callback acts as
// a mock backend: no RPC, phone, weights or CPU reference are required.
static void test_cycle_runtime_physical_8_and_48() {
    for (const int layers : {8, 48}) {
        constexpr int chunks = 4;
        llama_hybrid_cycle_runtime_config cfg;
        cfg.layout = llama_hybrid_build_cycle_layout(layers, 4, 2);
        cfg.layout.pc_cuda_full = true;
        cfg.n_chunks = chunks;
        cfg.max_inflight = 2;
        cfg.credit_until_handoff = true;
        cfg.ordered_c2p_reduce = true;
        cfg.phone_main_priority = true;
        cfg.pc_cpu_reduce_priority = true;
        llama_hybrid_cycle_runtime runtime(std::move(cfg));

        for (int cycle = 0; cycle < layers / 8; ++cycle) {
            llama_hybrid_cycle_runtime_region region;
            assert(runtime.region_for_layer(cycle * 8, region));
            assert(runtime.region_for_layer(cycle * 8 + 7, region));
            assert(region.cycle_index == cycle);
            assert(region.begin == cycle * 8);
            assert(region.c2p_layer == cycle * 8 + 4);
            assert(region.phone_begin == cycle * 8 + 5);
            assert(region.phone_end == cycle * 8 + 7);
            assert(region.p2c_layer == cycle * 8 + 7);
            assert(region.end == cycle * 8 + 8);
        }
        llama_hybrid_cycle_runtime_region outside;
        assert(!runtime.region_for_layer(layers, outside));

        std::vector<std::atomic<int>> completed(size_t(layers * chunks));
        for (auto & flag : completed) flag.store(0);
        std::atomic<int> tasks {0};
        std::atomic<int> c2p_sends {0};
        std::atomic<int> p2c_sends {0};
        const auto run_stage = [&](const llama_hybrid_cycle_task & task) {
            assert(task.layer >= 0 && task.layer < layers);
            assert(task.chunk >= 0 && task.chunk < chunks);
            const auto kind = runtime.config().layout.kinds[size_t(task.layer)];
            if (task.stage == llama_hybrid_cycle_stage::FULL_LAYER ||
                task.stage == llama_hybrid_cycle_stage::ATTN_PRE) {
                if (task.layer > 0) {
                    assert(completed[size_t((task.layer - 1) * chunks +
                                           task.chunk)].load() == 1);
                }
            }
            if (task.stage == llama_hybrid_cycle_stage::FULL_LAYER &&
                task.layer > 0) {
                assert(task.input_consumed);
                assert(task.input_consumed());
            }
            if (task.stage == llama_hybrid_cycle_stage::SEND_ACTIVATION ||
                task.stage == llama_hybrid_cycle_stage::SEND_PARTIAL) {
                if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                    assert(task.queue == llama_hybrid_cycle_queue::PC_TO_PHONE);
                    ++c2p_sends;
                } else {
                    assert(kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C);
                    assert(task.queue == llama_hybrid_cycle_queue::PHONE_TO_PC);
                    ++p2c_sends;
                }
            }
            if (task.stage == llama_hybrid_cycle_stage::REDUCE ||
                task.stage == llama_hybrid_cycle_stage::FULL_LAYER) {
                assert(completed[size_t(task.layer * chunks +
                                       task.chunk)].exchange(1) == 0);
            }
            ++tasks;
            return true;
        };

        // Bind all REAL physical Phone stages through independent cycle
        // ledgers. One global Ring still dispatches all 48 layer/chunk tasks.
        llama_hybrid_cycle_stage_dispatcher stages(runtime.config().layout);
        llama_hybrid_cycle_phone_regions phone_regions;
        std::string region_error;
        assert(phone_regions.prepare(
            runtime.config().layout, chunks * 16, 16, 2,
            region_error));
        assert(phone_regions.region_count() == layers / 8);
        assert(phone_regions.bind_phone_stages(
            stages,
            [&](llama_hybrid_cycle_phone_region_progress & region,
                const llama_hybrid_cycle_task & task) {
                assert(task.layer >= region.location.c2p_layer);
                assert(task.layer <= region.location.p2c_layer);
                assert(region.location.cycle_index == task.layer / 8);
                return run_stage(task);
            }, region_error));
        for (int layer = 0; layer < layers; ++layer) {
            const auto kind = runtime.config().layout.kinds[size_t(layer)];
            if (kind == llama_hybrid_cycle_layer_kind::PC_CPU ||
                kind == llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL) {
                assert(stages.bind(
                    layer, llama_hybrid_cycle_stage::FULL_LAYER, run_stage));
            }
        }
        std::string invalid_reason;
        assert(stages.validate(invalid_reason));
        const bool ok = runtime.run(stages);
        assert(ok);
        for (auto & flag : completed) assert(flag.load() == 1);
        assert(tasks.load() == (layers / 8) * chunks * 18);
        assert(c2p_sends.load() == (layers / 8) * chunks * 2);
        assert(p2c_sends.load() == (layers / 8) * chunks * 2);
    }
}

// Full physical model MOCK: no phone, CUDA or weights. Unlike a mere task
// counter, every Stage produces and consumes a typed Chunk activation through
// the same Session handoff API used by V5. The PHONE path remains simulated
// device-local; L7 P2C publishes a HOST result and L8 CUDA consumes it.
// All 48 layers run through ONE global Ring with no per-Cycle join barrier.
static void test_cycle_48_layer_cross_period_handoff() {
    constexpr int n_layers = 48;
    constexpr int n_chunks = 4;
    constexpr int chunk_tokens = 16;
    constexpr int64_t hidden = 2;

    llama_hybrid_cycle_runtime_config cfg;
    cfg.layout = llama_hybrid_build_cycle_layout(n_layers, 4, 2);
    cfg.layout.pc_cuda_full = true;
    cfg.n_chunks = n_chunks;
    cfg.max_inflight = 2;
    cfg.credit_until_handoff = true;
    cfg.ordered_c2p_reduce = true;
    cfg.phone_main_priority = true;
    cfg.pc_cpu_reduce_priority = true;
    llama_hybrid_cycle_runtime runtime(cfg);

    llama_hybrid_cycle_execution_session_config state_cfg;
    state_cfg.layout = cfg.layout;
    state_cfg.physical_layer_count = n_layers;
    state_cfg.tokens = n_chunks * chunk_tokens;
    state_cfg.chunk_tokens = chunk_tokens;
    llama_hybrid_cycle_execution_session<> session(state_cfg);

    const size_t count = size_t(n_layers * n_chunks);
    const size_t width = size_t(chunk_tokens * hidden);
    // Mock GPU values are NEVER read back by the Host path. They are
    // exchanged only by successive simulated Phone execution Stages.
    std::vector<std::vector<float>> phone_values(count);
    std::vector<std::vector<float>> prepared(count);
    std::vector<std::atomic<int>> p2c_finished(size_t(n_layers / 8));
    for (auto & c : p2c_finished) c.store(0);
    std::atomic<int> cross_region_first_chunks {0};
    std::atomic<int> published_host_chunks {0};

    const auto execute = [&](const llama_hybrid_cycle_task & task) -> bool {
        if (task.layer < 0 || task.layer >= n_layers ||
            task.chunk < 0 || task.chunk >= n_chunks) {
            return false;
        }
        const int l = task.layer;
        const int c = task.chunk;
        const auto kind = cfg.layout.kinds[size_t(l)];
        const size_t index = size_t(l * n_chunks + c);
        auto & state = session.at(l, c);

        if (task.stage == llama_hybrid_cycle_stage::FULL_LAYER) {
            if (kind == llama_hybrid_cycle_layer_kind::PC_CPU ||
                kind == llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL) {
                std::vector<float> input;
                if (l == 0) {
                    input.resize(width);
                    for (size_t i = 0; i < width; ++i) {
                        input[i] = float(c * 100 + int(i));
                    }
                } else if (!session.copy_predecessor_host_input(
                               l, c, hidden, input)) {
                    return false;
                }
                // max_inflight=2 cannot retire L7 C2/C3 before L8 C0
                // has actually run. This proves there is NO 8-layer
                // barrier waiting for every Chunk's P2C Reduce.
                if (l > 0 && l % 8 == 0 && c == 0) {
                    const int previous_done =
                        p2c_finished[size_t(l / 8 - 1)].load();
                    if (previous_done < 1 || previous_done >= n_chunks) {
                        return false;
                    }
                    ++cross_region_first_chunks;
                }
                if (input.size() != width) return false;
                state.host_output = std::move(input);
                for (auto & value : state.host_output) value += 1.0f;
                if (!session.publish_host_output(l, c, hidden)) return false;
                ++published_host_chunks;
                return true;
            }
            if (kind == llama_hybrid_cycle_layer_kind::PHONE_GPU && l > 0) {
                const auto & previous = phone_values[
                    size_t((l - 1) * n_chunks + c)];
                if (previous.size() != width) return false;
                phone_values[index] = previous;
                for (auto & value : phone_values[index]) value += 1.0f;
                return true;
            }
            return false;
        }

        if (task.stage == llama_hybrid_cycle_stage::ATTN_PRE) {
            if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                return session.copy_predecessor_host_input(
                    l, c, hidden, prepared[index]);
            }
            if (kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C && l > 0) {
                const auto & previous = phone_values[
                    size_t((l - 1) * n_chunks + c)];
                if (previous.size() != width) return false;
                prepared[index] = previous;
                return true;
            }
            return false;
        }

        if (task.stage == llama_hybrid_cycle_stage::REDUCE) {
            if (prepared[index].size() != width) return false;
            if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                phone_values[index] = prepared[index];
                for (auto & value : phone_values[index]) value += 1.0f;
                return true;
            }
            if (kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
                state.host_output = prepared[index];
                for (auto & value : state.host_output) value += 1.0f;
                if (!session.publish_host_output(l, c, hidden)) return false;
                ++published_host_chunks;
                ++p2c_finished[size_t(l / 8)];
                return true;
            }
            return false;
        }

        // The four intermediate Tensor tasks are separately scheduled by
        // the real Ring. They do not produce an additional transformer layer.
        return kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ||
               kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C;
    };

    llama_hybrid_cycle_stage_dispatcher stages(cfg.layout);
    for (int l = 0; l < n_layers; ++l) {
        const auto kind = cfg.layout.kinds[size_t(l)];
        if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ||
            kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
            for (const auto stage : {
                    llama_hybrid_cycle_stage::ATTN_PRE,
                    llama_hybrid_cycle_stage::SEND_ACTIVATION,
                    llama_hybrid_cycle_stage::FFN_SOURCE,
                    llama_hybrid_cycle_stage::FFN_DESTINATION,
                    llama_hybrid_cycle_stage::SEND_PARTIAL,
                    llama_hybrid_cycle_stage::REDUCE}) {
                assert(stages.bind(l, stage, execute));
            }
        } else {
            assert(stages.bind(
                l, llama_hybrid_cycle_stage::FULL_LAYER, execute));
        }
    }
    std::string reason;
    assert(stages.validate(reason));
    assert(runtime.run(stages));
    assert(cross_region_first_chunks.load() == 5);
    assert(published_host_chunks.load() == 6 * (4 + 1) * n_chunks);
    for (const auto & cycle : p2c_finished) {
        assert(cycle.load() == n_chunks);
    }
    for (int c = 0; c < n_chunks; ++c) {
        const auto & result = session.at(n_layers - 1, c);
        assert(result.host_output_published.load(std::memory_order_acquire));
        assert(result.host_output.size() == width);
        for (size_t i = 0; i < width; ++i) {
            assert(result.host_output[i] ==
                   float(c * 100 + int(i) + n_layers));
        }
    }
}

// Unlike the original 8-layer experiment, a 48-layer physical model must
// not unconditionally mirror all 24 PC full layers into CUDA. The global
// FIFO and Stage owner routing must honor the per-layer residency mask.
static void test_cycle_48_selective_cuda_full_layers() {
    llama_hybrid_cycle_runtime_config cfg;
    cfg.layout = llama_hybrid_build_cycle_layout(48, 4, 2);
    cfg.layout.pc_cuda_full = true; // Legacy flag overridden by explicit map.
    cfg.layout.pc_cuda_layers.assign(48, 0);
    for (const int layer : {0, 8, 24}) {
        cfg.layout.pc_cuda_layers[size_t(layer)] = 1;
    }
    cfg.n_chunks = 4;
    cfg.max_inflight = 2;
    cfg.credit_until_handoff = true;
    assert(llama_hybrid_cycle_cuda_layers_valid(cfg.layout));
    llama_hybrid_cycle_runtime runtime(cfg);
    llama_hybrid_cycle_stage_dispatcher stages(cfg.layout);
    std::atomic<int> pc_cpu_tasks {0};
    std::atomic<int> pc_cuda_tasks {0};
    std::atomic<int> phone_tasks {0};
    const auto execute = [&](const llama_hybrid_cycle_task & task) {
        if (task.stage == llama_hybrid_cycle_stage::FULL_LAYER &&
            cfg.layout.kinds[size_t(task.layer)] ==
                llama_hybrid_cycle_layer_kind::PC_CPU) {
            if (cfg.layout.pc_cuda_layers[size_t(task.layer)] == 1) {
                assert(task.queue == llama_hybrid_cycle_queue::PC_CUDA);
                assert(task.owner == llama_hybrid_cycle_owner::PC_CUDA);
                ++pc_cuda_tasks;
            } else {
                assert(task.queue == llama_hybrid_cycle_queue::PC_CPU);
                assert(task.owner == llama_hybrid_cycle_owner::PC_CPU);
                ++pc_cpu_tasks;
            }
        } else {
            ++phone_tasks;
        }
        return true;
    };
    for (int layer = 0; layer < 48; ++layer) {
        const auto kind = cfg.layout.kinds[size_t(layer)];
        if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ||
            kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
            for (const auto stage : {
                    llama_hybrid_cycle_stage::ATTN_PRE,
                    llama_hybrid_cycle_stage::SEND_ACTIVATION,
                    llama_hybrid_cycle_stage::FFN_SOURCE,
                    llama_hybrid_cycle_stage::FFN_DESTINATION,
                    llama_hybrid_cycle_stage::SEND_PARTIAL,
                    llama_hybrid_cycle_stage::REDUCE}) {
                assert(stages.bind(layer, stage, execute));
            }
        } else {
            assert(stages.bind(layer, llama_hybrid_cycle_stage::FULL_LAYER,
                               execute));
        }
    }
    assert(runtime.run(stages));
    assert(pc_cuda_tasks.load() == 3 * 4);
    assert(pc_cpu_tasks.load() == (24 - 3) * 4);

    // Missing/invalid mask can never schedule a task on an unprepared GPU.
    cfg.layout.pc_cuda_layers[4] = 1; // C2P is not a PC full layer.
    assert(!llama_hybrid_cycle_cuda_layers_valid(cfg.layout));
    cfg.layout.pc_cuda_layers[4] = 0;
    cfg.layout.pc_cuda_layers.pop_back();
    assert(!llama_hybrid_cycle_cuda_layers_valid(cfg.layout));
    assert(!llama_hybrid_cycle_full_layer_on_cuda(cfg.layout, 47));
}

// The real resource pool uses this pure policy before touching CUDA.
// 24 PC FULL layers must never default to 24 CUDA weight mirrors.
static void test_cycle_48_cuda_budget_selection() {
    auto layout = llama_hybrid_build_cycle_layout(48, 4, 2);
    layout.pc_cuda_full = true;
    std::vector<size_t> estimates(48, 0);
    for (int l = 0; l < 48; ++l) {
        if (layout.kinds[size_t(l)] ==
            llama_hybrid_cycle_layer_kind::PC_CPU) {
            estimates[size_t(l)] = 100;
        }
    }
    std::vector<uint8_t> selected;
    size_t bytes = 999;
    std::string reason;
    assert(llama_hybrid_cycle_select_cuda_full_layers(
        layout, estimates, 0, selected, bytes, reason));
    assert(bytes == 0);
    assert(std::count(selected.begin(), selected.end(), uint8_t(1)) == 0);
    layout.pc_cuda_layers = selected;
    assert(!llama_hybrid_cycle_full_layer_on_cuda(layout, 0));

    layout.pc_cuda_layers.clear();
    assert(llama_hybrid_cycle_select_cuda_full_layers(
        layout, estimates, 350, selected, bytes, reason));
    assert(bytes == 300);
    assert(std::count(selected.begin(), selected.end(), uint8_t(1)) == 3);
    // Implicit placement is balanced across the six physical periods.
    assert(selected[8] && selected[24] && selected[40] &&
           !selected[0] && !selected[1] && !selected[2]);

    // A Planner's explicit physical-layer selection is a hard contract:
    // it must not be silently weakened on memory shortfall.
    layout.pc_cuda_layers.assign(48, 0);
    layout.pc_cuda_layers[8] = 1;
    layout.pc_cuda_layers[16] = 1;
    assert(!llama_hybrid_cycle_select_cuda_full_layers(
        layout, estimates, 150, selected, bytes, reason));
    assert(reason.find("CUDA_MIRROR_BUDGET_EXCEEDED") != std::string::npos);
    assert(llama_hybrid_cycle_select_cuda_full_layers(
        layout, estimates, 200, selected, bytes, reason));
    assert(bytes == 200 && selected[8] == 1 && selected[16] == 1);
    assert(selected[0] == 0);
    layout.pc_cuda_layers[7] = 1;
    assert(!llama_hybrid_cycle_select_cuda_full_layers(
        layout, estimates, 2000, selected, bytes, reason));
    assert(reason == "INVALID_LAYOUT_OR_ESTIMATE_VECTOR");
}

// Phone MAIN's L5/L6/L7 command state must be period-local, including
// the READY bitmaps and Graph Slot banks. Simulate independent registration
// before any actual GPU calls, then reject mismatched topology/handler binds.
static void test_cycle_phone_regions_independent_ledgers() {
    const auto layout = llama_hybrid_build_cycle_layout(48, 4, 2);
    llama_hybrid_cycle_phone_regions regions;
    std::string reason;
    llama_hybrid_cycle_phone_regions whole_prompt_regions;
    assert(whole_prompt_regions.prepare(layout, 358, 64, 2, reason));
    assert(whole_prompt_regions.region_count() == 6);
    assert(whole_prompt_regions.n_chunks() == 6);
    for (int i = 0; i < whole_prompt_regions.region_count(); ++i) {
        const auto * ledger = whole_prompt_regions.by_index(i);
        assert(ledger && ledger->n_chunks == 6 &&
               ledger->c2p_ready_chunks.size() == 6);
    }

    // Keep the original 16-token V5 ledger case as a regression.
    assert(regions.prepare(layout, 64, 16, 2, reason));
    assert(regions.region_count() == 6);
    assert(regions.n_chunks() == 4);
    llama_hybrid_cycle_phone_region_slot_banks<
        llama_hybrid_cycle_phone_ffn_slot,
        llama_hybrid_cycle_p2c_ffn_slot> slots;
    slots.prepare(regions);
    assert(slots.size() == 6);
    for (int c = 0; c < 6; ++c) {
        auto * ledger = regions.by_index(c);
        auto * bank = slots.by_index(c);
        assert(ledger && bank);
        assert(ledger->location.cycle_index == c);
        assert(ledger->location.c2p_layer == 8 * c + 4);
        assert(ledger->location.phone_begin == 8 * c + 5);
        assert(ledger->location.phone_end == 8 * c + 7);
        assert(ledger->location.p2c_layer == 8 * c + 7);
        assert(bank->c2p_layer == 8 * c + 4);
        assert(bank->p2c_layer == 8 * c + 7);
        assert(bank->full_idle_by_layer.size() == 48);
        assert(regions.for_layer(8 * c) == ledger);
        assert(regions.for_layer(8 * c + 7) == ledger);
        assert(ledger->c2p_ready_chunks.size() == 4);
        assert(ledger->first_phone_registered_tokens == 0);
        assert(ledger->p2c_registered_tokens == 0);
    }
    assert(regions.by_index(-1) == nullptr);
    assert(regions.for_layer(48) == nullptr);
    assert(slots.by_index(6) == nullptr);
    // Full and P2C Graph Slots are separately owned by EACH period:
    // retiring a C2P output in period 0 cannot be reused in period 1.
    auto * first = slots.by_index(0);
    auto * second = slots.by_index(1);
    first->c2p_idle.push_back(
        std::make_unique<llama_hybrid_cycle_phone_ffn_slot>());
    first->p2c_idle.push_back(
        std::make_unique<llama_hybrid_cycle_p2c_ffn_slot>());
    first->full_idle_by_layer[5].push_back(
        std::make_unique<llama_hybrid_cycle_phone_ffn_slot>());
    assert(second->c2p_idle.empty());
    assert(second->p2c_idle.empty());
    assert(second->full_idle_by_layer[5].empty());
    auto * p0 = regions.by_index(0);
    p0->c2p_ready_chunks[0] = 1;
    p0->next_arm_chunk = 1;
    p0->first_phone_registered_tokens = 16;
    p0->next_attention_registered_tokens = 16;
    p0->p2c_registered_tokens = 16;
    p0->first_phone_completed = 1;
    p0->p2c_ffn_completed = 1;
    for (int i = 1; i < 6; ++i) {
        const auto * peer = regions.by_index(i);
        assert(peer->c2p_ready_chunks[0] == 0);
        assert(peer->next_arm_chunk == 0);
        assert(peer->first_phone_registered_tokens == 0);
        assert(peer->next_attention_registered_tokens == 0);
        assert(peer->p2c_registered_tokens == 0);
        assert(peer->first_phone_completed == 0);
        assert(peer->p2c_ffn_completed == 0);
    }

    // Reject a foreign dispatcher, repeated stage registration and bad
    // physical layer topology before running any FIFO worker.
    auto foreign = llama_hybrid_build_cycle_layout(48, 5, 1);
    llama_hybrid_cycle_stage_dispatcher foreign_stages(foreign);
    assert(!regions.bind_phone_stages(
        foreign_stages,
        [](auto &, const auto &) { return true; }, reason));
    auto corrupt = layout;
    corrupt.kinds[15] = llama_hybrid_cycle_layer_kind::PHONE_GPU;
    assert(!regions.prepare(corrupt, 64, 16, 2, reason));
    assert(reason.find("PHYSICAL_REGION_ROLE_MISMATCH") !=
           std::string::npos);
    assert(!regions.prepare(layout, 64, 16, 5, reason));
    assert(reason == "MAX_INFLIGHT_EXCEEDS_CHUNKS");
}

int main() {
    test_full_input_consumed_credit_release();
    test_cycle_phone_regions_independent_ledgers();
    test_cycle_48_cuda_budget_selection();
    test_cycle_48_selective_cuda_full_layers();
    test_cycle_48_layer_cross_period_handoff();
    test_cycle_execution_session_8_and_48();
    test_cycle_runtime_physical_route_guard();
    test_cycle_runtime_missing_stage_fails_closed();
    test_cycle_runtime_physical_8_and_48();
    test_v5_ready_c2p_reduce_beats_next_region_prepare();
    test_chunk_trace_observational();
    test_v5_pc_phone_proxy_main_priority();
    test_v5_phone_main_priority();
    test_v5_phone_rx_first();
    test_v4_ordered_c2p_reduce();
    test_pc_cuda_full_fifo();
    test_one_cycle();
    test_p2c_fifo_worker_runtime();
    test_p2c_cpu_ffn_runs_while_phone_ffn_pending();
    test_bounded_multi_chunk_p2c_fifo();
    test_p2c_ready_reduce_priority();
    test_p2c_reduce_handoff_to_cuda_queue();
    test_bounded_multi_chunk_c2p_fifo();
    test_complete_period_fifo_handoffs();
    test_causal_entry_order();
    test_two_cycles_and_cuda_tail();
    test_reject_invalid_completions();
    return 0;
}
