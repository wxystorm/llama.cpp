#include "../src/llama-sequence-pipeline.h"

#include <cassert>
#include <cstdio>
#include <vector>

static void complete_all_gpu(llama_sequence_pipe_ready & ready) {
    for (;;) {
        if (ready.queue_size(llama_sequence_pipe_resource::GPU) == 0) {
            break;
        }
        auto task = ready.wait_take(llama_sequence_pipe_resource::GPU);
        assert(task.has_value());
        assert(task->kind == llama_sequence_pipe_task_kind::REGION);
        ready.complete(*task);
    }
}

int main() {
    // Minimal prefix of the real runtime layout:
    //   G[0,4) -> S4 -> C[5,11) -> P[11,16)
    //
    // 358 tokens, Attention tile 64, global split:
    //   PC    [0,310)
    //   Phone [310,358)
    //
    // A4=[256,320) contributes 10 Phone tokens.  With min=16 it must not be
    // sent yet.  A5=[320,358) contributes another 38 and the scheduler must
    // aggregate them into one [310,358) task.
    std::vector<llama_sequence_pipe_stage> stages = {
        { llama_sequence_pipe_stage_kind::GPU,          0,  4, 64 },
        { llama_sequence_pipe_stage_kind::SEQUENCE_FFN, 4,  5, 64 },
        { llama_sequence_pipe_stage_kind::CPU,          5, 11, 64 },
        { llama_sequence_pipe_stage_kind::PHONE,       11, 16, 64 },
    };

    llama_sequence_pipe_ready ready(
        std::move(stages),
        358,
        1,
        310,
        16,
        32);

    // Make all upstream GPU intervals available.  Sequence Attention is still
    // strict causal/KV order, so only A0 is initially queued on CPU.
    complete_all_gpu(ready);

    for (int i = 0; i < 6; ++i) {
        auto task = ready.wait_take(llama_sequence_pipe_resource::CPU);
        assert(task.has_value());
        assert(task->kind == llama_sequence_pipe_task_kind::SEQ_ATTN);
        assert(task->token_begin == i * 64);
        ready.complete(*task);

        if (i == 4) {
            assert(ready.phone_pending_tokens() == 10);
            assert(ready.queue_size(
                llama_sequence_pipe_resource::PHONE) == 0);
        }
    }

    // The two Phone fragments were locally aggregated.  No READY/ACK task is
    // present: the only remote work is the actual FFN payload transaction.
    assert(ready.phone_pending_tokens() == 0);
    assert(ready.queue_size(
        llama_sequence_pipe_resource::PHONE) == 1);

    auto phone = ready.wait_take(
        llama_sequence_pipe_resource::PHONE);
    assert(phone.has_value());
    assert(phone->kind ==
        llama_sequence_pipe_task_kind::SEQ_PHONE_FFN);
    assert(phone->token_begin == 310);
    assert(phone->n_tokens == 48);

    // Once Phone has submitted work, CPU priority flips to the oldest causal
    // continuation rather than more lookahead Attention.
    auto pc = ready.wait_take(
        llama_sequence_pipe_resource::CPU);
    assert(pc.has_value());
    assert(pc->kind ==
        llama_sequence_pipe_task_kind::SEQ_PC_FFN);
    assert(pc->token_begin == 0);
    assert(pc->n_tokens == 64);

    std::printf(
        "[SEQ_PIPE_SCHED_TEST] split=310 phone=[310,358) "
        "aggregation=10+38->48 priority=FEED_THEN_OLDEST status=PASS\n");
    return 0;
}
