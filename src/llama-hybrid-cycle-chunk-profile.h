#pragma once

#include "llama-hybrid-cycle-ring.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <map>
#include <tuple>
#include <vector>

// PC-only steady_clock intervals. Each queue vector has one worker writer;
// the printing function runs after all FIFO workers have joined.
struct llama_hybrid_cycle_task_interval {
    llama_hybrid_cycle_task task;
    std::chrono::steady_clock::time_point start {};
    std::chrono::steady_clock::time_point finish {};
    bool success = false;
};
struct llama_hybrid_cycle_chunk_timeline {
    std::array<std::vector<llama_hybrid_cycle_task_interval>,
        static_cast<size_t>(llama_hybrid_cycle_queue::COUNT)> by_queue {};
};

inline double llama_hybrid_cycle_trace_ms(
        std::chrono::steady_clock::time_point a,
        std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Diagnostic only: these PC host spans are NOT a synchronized Phone kernel
// trace. PHONE_GPU FIFO callbacks may contain remote RPC waits. The P2C join
// analysis compares two PC-visible completions on the same monotonic clock.
inline void llama_hybrid_cycle_print_chunk_timeline(
        const llama_hybrid_cycle_layout & layout, int chunks, int window,
        std::chrono::steady_clock::time_point begin,
        std::chrono::steady_clock::time_point end,
        const llama_hybrid_cycle_chunk_timeline & timeline) {
    using interval = llama_hybrid_cycle_task_interval;
    using stage = llama_hybrid_cycle_stage;
    using key = std::tuple<int,int,stage>;
    std::map<key,const interval *> events;
    std::vector<const interval *> ordered;
    for (const auto & q : timeline.by_queue) {
        for (const auto & e : q) {
            events.emplace(key{e.task.layer,e.task.chunk,e.task.stage}, &e);
            ordered.push_back(&e);
        }
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const interval * a, const interval * b) {
            return a->task.queued_at < b->task.queued_at;
        });
    const auto get = [&](int layer,int chunk,stage s) -> const interval * {
        const auto it=events.find(key{layer,chunk,s});
        return it == events.end() ? nullptr : it->second;
    };
    const auto stamp = [&](const interval * e) -> double {
        return e ? llama_hybrid_cycle_trace_ms(begin,e->finish) : -1.0;
    };
    const auto queue_ms = [&](const interval * e) -> double {
        return llama_hybrid_cycle_trace_ms(e->task.queued_at,e->start);
    };

    int c2p = -1, p2c = -1;
    for (size_t i=0;i<layout.kinds.size();++i) {
        if (layout.kinds[i] == llama_hybrid_cycle_layer_kind::TENSOR_C2P)
            c2p = int(i);
        if (layout.kinds[i] == llama_hybrid_cycle_layer_kind::TENSOR_P2C)
            p2c = int(i);
    }
    std::printf(
        "[CYCLE_CHUNK_TRACE_PLAN] side=PC chunks=%d window=%d "
        "c2p=%d p2c=%d tasks=%zu fifo_ms=%.3f "
        "clock=PC_STEADY phone_gpu=RPC_PROXY "
        "phone_kernel_ms=SERVER_ONLY cross_clock_subtraction=FORBIDDEN\n",
        chunks,window,c2p,p2c,ordered.size(),
        llama_hybrid_cycle_trace_ms(begin,end));
    int gated=0; double max_gate=0.0;
    for (const auto * e : ordered) {
        const bool entry=e->task.entry_ready_at.time_since_epoch().count()!=0;
        const double gate=entry ? std::max(0.0,llama_hybrid_cycle_trace_ms(
            e->task.entry_ready_at,e->task.queued_at)) : 0.0;
        if (gate>0.1) { ++gated; max_gate=std::max(max_gate,gate); }
        std::printf(
            "[CYCLE_CHUNK_TASK] layer=%d chunk=%d stage=%s queue=%s "
            "entry_ready_ms=%.3f queued_ms=%.3f start_ms=%.3f "
            "finish_ms=%.3f entry_gate_ms=%.3f queue_wait_ms=%.3f "
            "callback_ms=%.3f status=%s\n",
            e->task.layer,e->task.chunk,
            llama_hybrid_cycle_stage_name(e->task.stage),
            llama_hybrid_cycle_queue_name(e->task.queue),
            entry ? llama_hybrid_cycle_trace_ms(begin,e->task.entry_ready_at) : -1.0,
            llama_hybrid_cycle_trace_ms(begin,e->task.queued_at),
            llama_hybrid_cycle_trace_ms(begin,e->start),
            stamp(e),gate,queue_ms(e),
            llama_hybrid_cycle_trace_ms(e->start,e->finish),
            e->success ? "OK" : "FAIL");
        if (gate > 0.1) {
            std::printf("[CYCLE_CHUNK_ENTRY_GATE] layer=%d chunk=%d "
                "wait_ms=%.3f reason=CREDIT_OR_CAUSAL_ORDER\n",
                e->task.layer,e->task.chunk,gate);
        }
    }
    int complete=0;
    if (c2p>=0 && p2c>=0) {
        for (int c=0;c<chunks;++c) {
            const auto * prefix=get(0,c,stage::FULL_LAYER);
            const auto * first=get(c2p,c,stage::ATTN_PRE);
            const auto * handoff=get(c2p,c,stage::REDUCE);
            const auto * phone_full=get(p2c-1,c,stage::FULL_LAYER);
            const auto * p2c_attn=get(p2c,c,stage::ATTN_PRE);
            const auto * ready=get(p2c,c,stage::SEND_ACTIVATION);
            const auto * pc_ffn=get(p2c,c,stage::FFN_DESTINATION);
            const auto * phone_ffn=get(p2c,c,stage::FFN_SOURCE);
            const auto * partial=get(p2c,c,stage::SEND_PARTIAL);
            const auto * output=get(p2c,c,stage::REDUCE);
            const bool ok=first && handoff && p2c_attn && ready && pc_ffn &&
                partial && output && first->success && handoff->success &&
                p2c_attn->success && ready->success && pc_ffn->success &&
                partial->success && output->success;
            if (ok) ++complete;
            std::printf(
                "[CYCLE_CHUNK_CRITICAL] chunk=%d cuda_prefix_done_ms=%.3f "
                "c2p_attn_done_ms=%.3f c2p_reduce_rpc_done_ms=%.3f "
                "phone_last_rpc_done_ms=%.3f p2c_attn_rpc_done_ms=%.3f "
                "p2c_ready_rx_done_ms=%.3f phone_ffn_rpc_done_ms=%.3f "
                "pc_ffn_done_ms=%.3f partial_rx_done_ms=%.3f "
                "next_cuda_input_ready_ms=%.3f c2p_to_output_ms=%.3f "
                "next_cuda_execution=NOT_OBSERVED status=%s\n",
                c,stamp(prefix),stamp(first),stamp(handoff),
                stamp(phone_full),stamp(p2c_attn),stamp(ready),
                stamp(phone_ffn),stamp(pc_ffn),stamp(partial),stamp(output),
                first && output ? llama_hybrid_cycle_trace_ms(
                    first->task.queued_at,output->finish) : -1.0,
                ok ? "OK" : "INCOMPLETE");
            if (pc_ffn && partial && output) {
                const auto later=std::max(pc_ffn->finish,partial->finish);
                std::printf(
                    "[CYCLE_CHUNK_P2C_JOIN] chunk=%d later=%s "
                    "pc_ffn_ms=%.3f partial_rx_ms=%.3f "
                    "branch_delta_ms=%.3f join_enqueue_gap_ms=%.3f "
                    "reduce_queue_ms=%.3f reduce_callback_ms=%.3f "
                    "tail_after_join_ms=%.3f scope=PC_EVENTS\n",
                    c,partial->finish>pc_ffn->finish ?
                        "PHONE_PARTIAL_RX" : "PC_CPU_FFN",
                    stamp(pc_ffn),stamp(partial),
                    llama_hybrid_cycle_trace_ms(pc_ffn->finish,partial->finish),
                    std::max(0.0,llama_hybrid_cycle_trace_ms(
                        later,output->task.queued_at)),
                    queue_ms(output),
                    llama_hybrid_cycle_trace_ms(output->start,output->finish),
                    std::max(0.0,llama_hybrid_cycle_trace_ms(
                        later,output->finish)));
            }
        }
    }
    auto ranked=ordered;
    std::sort(ranked.begin(),ranked.end(),
        [&](const interval * a,const interval * b) {
            return queue_ms(a)>queue_ms(b);
        });
    for (size_t i=0;i<std::min(size_t(5),ranked.size());++i) {
        const auto * e=ranked[i];
        std::printf("[CYCLE_CHUNK_WAIT_TOP] rank=%zu layer=%d chunk=%d "
            "stage=%s queue=%s queue_wait_ms=%.3f "
            "criticality=NOT_INFERRED_FROM_QUEUE_WAIT\n",
            i+1,e->task.layer,e->task.chunk,
            llama_hybrid_cycle_stage_name(e->task.stage),
            llama_hybrid_cycle_queue_name(e->task.queue),queue_ms(e));
    }
    std::printf("[CYCLE_CHUNK_TRACE_SUM] complete=%d expected=%d "
        "entry_gate_count=%d max_entry_gate_ms=%.3f "
        "queues=OVERLAPPING_NOT_ADDITIVE status=%s\n",
        complete,chunks,gated,max_gate,
        c2p>=0 && p2c>=0 && complete==chunks ? "OK" : "PARTIAL");
    std::fflush(stdout);
}

// Compact aggregate for the official Cycle CLI. The full per-task trace is
// retained only when profiling is explicitly enabled; normal generation
// performs none of this work. A stage duration is a PC-side callback wall
// interval and can include RPC, GPU synchronization, Graph preparation and
// host copies. Stage sums overlap across queues and are NOT on the critical
// path by themselves.
inline void llama_hybrid_cycle_print_stage_summary(
        const llama_hybrid_cycle_chunk_timeline & timeline) {
    constexpr size_t queues = static_cast<size_t>(llama_hybrid_cycle_queue::COUNT);
    constexpr size_t stages = static_cast<size_t>(llama_hybrid_cycle_stage::REDUCE) + 1;
    struct stage_totals {
        size_t tasks = 0;
        double callback_sum_ms = 0;
        double callback_max_ms = 0;
        double queue_wait_sum_ms = 0;
        double queue_wait_max_ms = 0;
        double entry_gate_sum_ms = 0;
        double entry_gate_max_ms = 0;
    };
    std::array<std::array<stage_totals, stages>, queues> totals {};
    std::vector<const llama_hybrid_cycle_task_interval *> ordered;
    for (size_t q = 0; q < queues; ++q) {
        for (const auto & event : timeline.by_queue[q]) {
            const size_t st = static_cast<size_t>(event.task.stage);
            if (st >= stages) continue;
            auto & t = totals[q][st];
            const double callback_ms = std::max(0.0,
                llama_hybrid_cycle_trace_ms(event.start, event.finish));
            const double wait_ms = event.task.queued_at.time_since_epoch().count() ?
                std::max(0.0, llama_hybrid_cycle_trace_ms(
                    event.task.queued_at, event.start)) : 0.0;
            const double gate_ms =
                event.task.entry_ready_at.time_since_epoch().count() &&
                event.task.queued_at.time_since_epoch().count() ?
                std::max(0.0, llama_hybrid_cycle_trace_ms(
                    event.task.entry_ready_at, event.task.queued_at)) : 0.0;
            ++t.tasks;
            t.callback_sum_ms += callback_ms;
            t.callback_max_ms = std::max(t.callback_max_ms, callback_ms);
            t.queue_wait_sum_ms += wait_ms;
            t.queue_wait_max_ms = std::max(t.queue_wait_max_ms, wait_ms);
            t.entry_gate_sum_ms += gate_ms;
            t.entry_gate_max_ms = std::max(t.entry_gate_max_ms, gate_ms);
            ordered.push_back(&event);
        }
    }
    for (size_t q = 0; q < queues; ++q) {
        for (size_t st = 0; st < stages; ++st) {
            const auto & t = totals[q][st];
            if (!t.tasks) continue;
            std::printf(
                "[CYCLE_CORE_STAGE] queue=%s stage=%s tasks=%zu "
                "callback_sum_ms=%.3f callback_max_ms=%.3f "
                "queue_wait_sum_ms=%.3f queue_wait_max_ms=%.3f "
                "entry_gate_sum_ms=%.3f entry_gate_max_ms=%.3f "
                "clock=PC_STEADY scope=HOST_CALLBACK\n",
                llama_hybrid_cycle_queue_name(
                    static_cast<llama_hybrid_cycle_queue>(q)),
                llama_hybrid_cycle_stage_name(
                    static_cast<llama_hybrid_cycle_stage>(st)),
                t.tasks, t.callback_sum_ms, t.callback_max_ms,
                t.queue_wait_sum_ms, t.queue_wait_max_ms,
                t.entry_gate_sum_ms, t.entry_gate_max_ms);
        }
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const auto * a, const auto * b) {
            return llama_hybrid_cycle_trace_ms(a->start,a->finish) >
                   llama_hybrid_cycle_trace_ms(b->start,b->finish);
        });
    for (size_t i = 0; i < std::min(size_t(5), ordered.size()); ++i) {
        const auto & e = *ordered[i];
        std::printf(
            "[CYCLE_CORE_SLOW_TASK] rank=%zu layer=%d chunk=%d "
            "queue=%s stage=%s callback_ms=%.3f "
            "device_kernel_ms=NOT_MEASURED\n",
            i + 1, e.task.layer, e.task.chunk,
            llama_hybrid_cycle_queue_name(e.task.queue),
            llama_hybrid_cycle_stage_name(e.task.stage),
            llama_hybrid_cycle_trace_ms(e.start,e.finish));
    }
}
