#include "llama-hybrid-cycle-ring.h"

#include <array>
#include <cassert>
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

int main() {
    test_one_cycle();
    test_causal_entry_order();
    test_two_cycles_and_cuda_tail();
    test_reject_invalid_completions();
    return 0;
}
