#include "llama-hybrid-cycle-ring.h"
#include "llama-hybrid-cycle-phone-local.h"
#include "llama-hybrid-cycle-fifo-executor.h"

#include <array>
#include <cassert>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

static llama_hybrid_cycle_task take(
        llama_hybrid_cycle_task_ring & ring,
        llama_hybrid_cycle_queue queue,
        llama_hybrid_cycle_stage stage,
        int layer,
        int chunk) {
    llama_hybrid_cycle_task task;
    assert(ring.try_pop(queue, task));
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

// An entry-credit stall begins only when the causal head Chunk is
// device-input ready AND max_inflight is exhausted. It ends at admission,
// not at graph completion, so it is distinct from FIFO queue_wait.
static void test_path_entry_credit_profile() {
    llama_hybrid_cycle_layout fragment;
    fragment.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_P2C};
    llama_hybrid_cycle_task_ring ring(
        fragment, 3, 2, false, false, false, false, true);
    ring.start();
    {
        const auto samples = ring.credit_profiles();
        assert(samples.size() == 1);
        assert(samples[0].admitted_chunks == 2);
        assert(samples[0].credit_blocked_chunks == 1);
        assert(samples[0].credit_wait_by_chunk_ms.size() == 3);
    }
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::ATTN_PRE, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                       llama_hybrid_cycle_stage::SEND_ACTIVATION, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_GPU,
                       llama_hybrid_cycle_stage::FFN_SOURCE, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::FFN_DESTINATION, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PHONE_TO_PC,
                       llama_hybrid_cycle_stage::SEND_PARTIAL, 0, 0));
    ring.complete(take(ring, llama_hybrid_cycle_queue::PC_CPU,
                       llama_hybrid_cycle_stage::REDUCE, 0, 0));
    const auto samples = ring.credit_profiles();
    assert(samples[0].admitted_chunks == 3);
    assert(samples[0].credit_blocked_chunks == 1);
    assert(samples[0].credit_wait_by_chunk_ms[0] == 0.0);
    assert(samples[0].credit_wait_by_chunk_ms[1] == 0.0);
    assert(samples[0].credit_wait_by_chunk_ms[2] >= 0.0);
    assert(samples[0].credit_wait_sum_ms >= 0.0);
}

// The real 5-worker bridge copies credit metrics only after workers
// join. Capturing the metric must not alter task order or introduce a
// new GPU/network dependency.
static void test_path_fifo_credit_profile() {
    llama_hybrid_cycle_layout fragment;
    fragment.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_P2C};
    llama_hybrid_cycle_fifo_profile profile {};
    const bool ok = llama_hybrid_cycle_run_fifo(
        fragment, 3, [](const llama_hybrid_cycle_task &) {
            return true;
        }, 2, false, &profile, false, false, false, true);
    assert(ok);
    assert(profile.credit_layers.size() == 1);
    assert(profile.credit_layers[0].admitted_chunks == 3);
    assert(profile.credit_layers[0].credit_blocked_chunks >= 1);
    assert(profile.credit_layers[0].credit_wait_sum_ms >= 0.0);
}

int main() {
    test_path_fifo_credit_profile();
    test_path_entry_credit_profile();
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
