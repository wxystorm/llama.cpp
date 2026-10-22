#pragma once

#include "llama-hybrid-cycle.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

// Dependency-only scheduler for the fixed PC -> Phone -> PC layer cycle.
// Backend graph execution and network payload movement are intentionally
// external: a SEND task completes only when its destination buffer is ready.
enum class llama_hybrid_cycle_stage {
    FULL_LAYER,
    ATTN_PRE,
    SEND_ACTIVATION,
    FFN_SOURCE,
    FFN_DESTINATION,
    SEND_PARTIAL,
    REDUCE,
};

enum class llama_hybrid_cycle_queue {
    PC_CPU,
    PHONE_GPU,
    PC_CUDA,
    PC_TO_PHONE,
    PHONE_TO_PC,
    COUNT,
};

struct llama_hybrid_cycle_task {
    int layer = -1;
    int chunk = -1;
    llama_hybrid_cycle_stage stage = llama_hybrid_cycle_stage::FULL_LAYER;
    llama_hybrid_cycle_owner owner = llama_hybrid_cycle_owner::PC_CPU;
    llama_hybrid_cycle_queue queue = llama_hybrid_cycle_queue::PC_CPU;
    // Set only when the task enters its real FIFO. It does not alter
    // scheduling order, dependencies or buffer ownership.
    std::chrono::steady_clock::time_point queued_at {};
};

inline const char * llama_hybrid_cycle_stage_name(llama_hybrid_cycle_stage stage) {
    switch (stage) {
        case llama_hybrid_cycle_stage::FULL_LAYER:       return "FULL_LAYER";
        case llama_hybrid_cycle_stage::ATTN_PRE:         return "ATTN_PRE";
        case llama_hybrid_cycle_stage::SEND_ACTIVATION:  return "SEND_ACTIVATION";
        case llama_hybrid_cycle_stage::FFN_SOURCE:       return "FFN_SOURCE";
        case llama_hybrid_cycle_stage::FFN_DESTINATION:  return "FFN_DESTINATION";
        case llama_hybrid_cycle_stage::SEND_PARTIAL:     return "SEND_PARTIAL";
        case llama_hybrid_cycle_stage::REDUCE:           return "REDUCE";
    }
    return "UNKNOWN";
}

inline const char * llama_hybrid_cycle_queue_name(llama_hybrid_cycle_queue queue) {
    switch (queue) {
        case llama_hybrid_cycle_queue::PC_CPU:        return "PC_CPU";
        case llama_hybrid_cycle_queue::PHONE_GPU:     return "PHONE_GPU";
        case llama_hybrid_cycle_queue::PC_CUDA:       return "PC_CUDA";
        case llama_hybrid_cycle_queue::PC_TO_PHONE:   return "PC_TO_PHONE";
        case llama_hybrid_cycle_queue::PHONE_TO_PC:   return "PHONE_TO_PC";
        case llama_hybrid_cycle_queue::COUNT:         break;
    }
    return "UNKNOWN";
}

// Per-task readiness order:
//
//   ATTN_PRE -> SEND_ACTIVATION -> FFN_DESTINATION ---+
//             \-> FFN_SOURCE -> SEND_PARTIAL ---------+-> REDUCE
//
// SEND_ACTIVATION means that residual, normalized hidden, top-k IDs and
// routing weights (for MoE) are ready on the FFN destination, not just that
// send() returned. SEND_PARTIAL means the source FFN partial has arrived and
// is ready on the Reduce backend. One Reduce owns the final l_out and only
// that completion can make the next layer's input ready.
//
// Per-layer token entry ordering is enforced even if independent FFN/transfer
// completions arrive out of order. This preserves causal KV chunk order.
class llama_hybrid_cycle_task_ring {
public:
    explicit llama_hybrid_cycle_task_ring(
            const llama_hybrid_cycle_layout & layout, int n_chunks,
            int max_inflight = 0, bool credit_until_handoff = false)
        : layout_(layout), n_chunks_(n_chunks),
          max_inflight_(max_inflight == 0 ? n_chunks : max_inflight),
          credit_until_handoff_(credit_until_handoff) {
        if (!layout_.enabled() || n_chunks_ <= 0 ||
            max_inflight_ <= 0 || max_inflight_ > n_chunks_) {
            throw std::invalid_argument(
                "cycle task ring requires positive chunks and valid in-flight window");
        }
        entry_ready_.assign(layout_.kinds.size(), std::vector<uint8_t>(n_chunks_, 0));
        next_chunk_.assign(layout_.kinds.size(), 0);
        layer_inflight_.assign(layout_.kinds.size(), 0);
        tensor_ready_.assign(layout_.kinds.size(), std::vector<tensor_join_state>(n_chunks_));
    }

    void start() {
        std::lock_guard<std::mutex> lock(mu_);
        if (started_) {
            throw std::logic_error("cycle task ring already started");
        }
        started_ = true;
        for (int chunk = 0; chunk < n_chunks_; ++chunk) {
            mark_entry_ready(0, chunk);
        }
    }

    // Every queue is strict FIFO, including the two independent TX directions.
    // Executors should pop without keeping mu_ locked while doing any work.
    bool try_pop(llama_hybrid_cycle_queue queue, llama_hybrid_cycle_task & out) {
        std::lock_guard<std::mutex> lock(mu_);
        const size_t index = static_cast<size_t>(queue);
        if (index >= queues_.size() || queues_[index].empty()) {
            return false;
        }
        out = queues_[index].front();
        queues_[index].pop_front();
        const auto key = task_key(out);
        if (!running_.emplace(key, out).second) {
            throw std::logic_error("cycle task already running");
        }
        return true;
    }

    void complete(const llama_hybrid_cycle_task & task) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto key = task_key(task);
        const auto found = running_.find(key);
        if (found == running_.end() ||
            found->second.owner != task.owner || found->second.queue != task.queue) {
            throw std::logic_error("unknown, duplicate or mismatched cycle task completion");
        }
        running_.erase(found);
        const auto kind = layout_.kinds[task.layer];
        const bool tensor = is_tensor(kind);

        if (!tensor && layout_.pc_cuda_attn &&
            kind == llama_hybrid_cycle_layer_kind::PC_CPU) {
            // A PC full layer is a local two-device FIFO pipeline in this
            // opt-in mode, not one blocking CPU FULL_LAYER callback.
            switch (task.stage) {
                case llama_hybrid_cycle_stage::ATTN_PRE:
                    // CUDA has consumed its predecessor's host output and
                    // synchronized its causal K/V before handing READY to CPU.
                    if (credit_until_handoff_ && task.layer > 0) {
                        release_layer_credit(task.layer - 1);
                    }
                    enqueue(task.layer, task.chunk,
                            llama_hybrid_cycle_stage::FFN_SOURCE);
                    return;
                case llama_hybrid_cycle_stage::FFN_SOURCE:
                    enqueue(task.layer, task.chunk,
                            llama_hybrid_cycle_stage::REDUCE);
                    return;
                case llama_hybrid_cycle_stage::REDUCE:
                    finish_layer(task.layer, task.chunk);
                    return;
                default:
                    throw std::logic_error("unexpected PC CUDA full stage");
            }
        }
        if (!tensor) {
            if (task.stage != llama_hybrid_cycle_stage::FULL_LAYER) {
                throw std::logic_error("unexpected non-Tensor stage");
            }
            // This full layer has now consumed the previous layer's
            // output. Only NOW can the previous owner's GPU/CPU buffer
            // credit be reused to start another Chunk.
            if (credit_until_handoff_ && task.layer > 0) {
                release_layer_credit(task.layer - 1);
            }
            finish_layer(task.layer, task.chunk);
            return;
        }

        tensor_join_state & state = tensor_ready_[task.layer][task.chunk];
        switch (task.stage) {
            case llama_hybrid_cycle_stage::ATTN_PRE:
                // The next Tensor layer has consumed its predecessor's
                // output in AttentionPre. Preserve the predecessor's
                // GPU buffer until THIS point, not just its old Reduce.
                if (credit_until_handoff_ && task.layer > 0) {
                    release_layer_credit(task.layer - 1);
                }
                // Let the sender start its shard without waiting for Send #1.
                enqueue(task.layer, task.chunk, llama_hybrid_cycle_stage::SEND_ACTIVATION);
                enqueue(task.layer, task.chunk, llama_hybrid_cycle_stage::FFN_SOURCE);
                break;
            case llama_hybrid_cycle_stage::SEND_ACTIVATION:
                // Transport completion is defined as receiver device-ready.
                enqueue(task.layer, task.chunk, llama_hybrid_cycle_stage::FFN_DESTINATION);
                break;
            case llama_hybrid_cycle_stage::FFN_SOURCE:
                enqueue(task.layer, task.chunk, llama_hybrid_cycle_stage::SEND_PARTIAL);
                break;
            case llama_hybrid_cycle_stage::FFN_DESTINATION:
                state.destination_ffn_ready = true;
                try_enqueue_reduce(task.layer, task.chunk, state);
                break;
            case llama_hybrid_cycle_stage::SEND_PARTIAL:
                state.source_partial_ready = true;
                try_enqueue_reduce(task.layer, task.chunk, state);
                break;
            case llama_hybrid_cycle_stage::REDUCE:
                finish_layer(task.layer, task.chunk);
                break;
            case llama_hybrid_cycle_stage::FULL_LAYER:
                throw std::logic_error("Tensor cannot use FULL_LAYER stage");
        }
    }

    bool finished() const {
        std::lock_guard<std::mutex> lock(mu_);
        return started_ && layer_outputs_done_ == layout_.kinds.size() * size_t(n_chunks_);
    }

    size_t output_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        return layer_outputs_done_;
    }

    size_t pending(llama_hybrid_cycle_queue queue) const {
        std::lock_guard<std::mutex> lock(mu_);
        const size_t i = static_cast<size_t>(queue);
        return i < queues_.size() ? queues_[i].size() : 0;
    }

private:
    struct tensor_join_state {
        bool destination_ffn_ready = false;
        bool source_partial_ready = false;
        bool reduce_queued = false;
    };
    using task_id = std::tuple<int, int, llama_hybrid_cycle_stage>;
    static constexpr size_t queue_count = static_cast<size_t>(llama_hybrid_cycle_queue::COUNT);

    static bool is_tensor(llama_hybrid_cycle_layer_kind kind) {
        return kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ||
               kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C;
    }

    static task_id task_key(const llama_hybrid_cycle_task & task) {
        return {task.layer, task.chunk, task.stage};
    }

    static llama_hybrid_cycle_queue compute_queue(llama_hybrid_cycle_owner owner) {
        switch (owner) {
            case llama_hybrid_cycle_owner::PC_CPU:    return llama_hybrid_cycle_queue::PC_CPU;
            case llama_hybrid_cycle_owner::PHONE_GPU: return llama_hybrid_cycle_queue::PHONE_GPU;
            case llama_hybrid_cycle_owner::PC_CUDA:   return llama_hybrid_cycle_queue::PC_CUDA;
        }
        throw std::logic_error("unrecognized cycle compute owner");
    }

    llama_hybrid_cycle_task make_task(int layer, int chunk, llama_hybrid_cycle_stage stage) const {
        const auto kind = layout_.kinds[layer];
        const auto source = llama_hybrid_cycle_attn_owner(kind);
        const auto destination = llama_hybrid_cycle_output_owner(kind);
        llama_hybrid_cycle_task task {layer, chunk, stage, source, compute_queue(source)};

        if (layout_.pc_cuda_attn &&
            kind == llama_hybrid_cycle_layer_kind::PC_CPU) {
            switch (stage) {
                case llama_hybrid_cycle_stage::ATTN_PRE:
                    task.owner = llama_hybrid_cycle_owner::PC_CUDA;
                    task.queue = llama_hybrid_cycle_queue::PC_CUDA;
                    return task;
                case llama_hybrid_cycle_stage::FFN_SOURCE:
                case llama_hybrid_cycle_stage::REDUCE:
                    task.owner = llama_hybrid_cycle_owner::PC_CPU;
                    task.queue = llama_hybrid_cycle_queue::PC_CPU;
                    return task;
                default:
                    throw std::logic_error("unexpected PC CUDA task stage");
            }
        }

        switch (stage) {
            case llama_hybrid_cycle_stage::FULL_LAYER:
            case llama_hybrid_cycle_stage::ATTN_PRE:
            case llama_hybrid_cycle_stage::FFN_SOURCE:
                break;
            case llama_hybrid_cycle_stage::FFN_DESTINATION:
            case llama_hybrid_cycle_stage::REDUCE:
                task.owner = destination;
                task.queue = compute_queue(destination);
                break;
            case llama_hybrid_cycle_stage::SEND_ACTIVATION:
            case llama_hybrid_cycle_stage::SEND_PARTIAL:
                task.queue = kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ?
                    llama_hybrid_cycle_queue::PC_TO_PHONE :
                    llama_hybrid_cycle_queue::PHONE_TO_PC;
                break;
        }
        return task;
    }

    void enqueue(int layer, int chunk, llama_hybrid_cycle_stage stage) {
        auto task = make_task(layer, chunk, stage);
        task.queued_at = std::chrono::steady_clock::now();
        if (!seen_.insert(task_key(task)).second) {
            throw std::logic_error("duplicate cycle task enqueue");
        }
        queues_[static_cast<size_t>(task.queue)].push_back(task);
    }

    void try_enqueue_reduce(int layer, int chunk, tensor_join_state & state) {
        if (state.destination_ffn_ready && state.source_partial_ready && !state.reduce_queued) {
            state.reduce_queued = true;
            enqueue(layer, chunk, llama_hybrid_cycle_stage::REDUCE);
        }
    }

    void mark_entry_ready(int layer, int chunk) {
        if (layer == static_cast<int>(layout_.kinds.size())) {
            return;
        }
        if (entry_ready_[layer][chunk] != 0) {
            throw std::logic_error("duplicate cycle layer entry");
        }
        entry_ready_[layer][chunk] = 1;
        drain_entry_window(layer);
    }

    // Apply bounded credits separately at every layer. A late chunk cannot
    // jump ahead of an earlier KV position and may not allocate a new Phone
    // Attention graph until one earlier chunk's final Reduce has freed its
    // entry credit. Default (max_inflight == n_chunks) is legacy behavior.
    void drain_entry_window(int layer) {
        while (next_chunk_[layer] < n_chunks_ &&
               entry_ready_[layer][next_chunk_[layer]] != 0 &&
               layer_inflight_[layer] < max_inflight_) {
            const int ready_chunk = next_chunk_[layer]++;
            ++layer_inflight_[layer];
            enqueue(layer, ready_chunk,
                    (is_tensor(layout_.kinds[layer]) ||
                     (layout_.pc_cuda_attn &&
                      layout_.kinds[layer] == llama_hybrid_cycle_layer_kind::PC_CPU)) ?
                        llama_hybrid_cycle_stage::ATTN_PRE :
                        llama_hybrid_cycle_stage::FULL_LAYER);
        }
    }

    void release_layer_credit(int layer) {
        if (layer_inflight_[layer] <= 0) {
            throw std::logic_error("cycle layer credit underflow");
        }
        --layer_inflight_[layer];
        drain_entry_window(layer);
    }

    void finish_layer(int layer, int chunk) {
        ++layer_outputs_done_;
        // For the full multi-layer period, output buffers remain owned
        // by this layer until the successor's FULL_LAYER or ATTN_PRE
        // actually consumes the tensor. Otherwise Reduce could allow
        // unbounded pending Phone GPU outputs despite max_inflight=2.
        // On a single-layer diagnostic or the final layer, the normal
        // immediate credit release is still correct.
        if (!credit_until_handoff_ ||
            layer + 1 == int(layout_.kinds.size())) {
            release_layer_credit(layer);
        }
        mark_entry_ready(layer + 1, chunk);
    }

    llama_hybrid_cycle_layout layout_;
    int n_chunks_ = 0;
    int max_inflight_ = 0;
    bool credit_until_handoff_ = false;
    bool started_ = false;
    size_t layer_outputs_done_ = 0;

    mutable std::mutex mu_;
    std::array<std::deque<llama_hybrid_cycle_task>, queue_count> queues_;
    std::map<task_id, llama_hybrid_cycle_task> running_;
    std::set<task_id> seen_;
    std::vector<std::vector<uint8_t>> entry_ready_;
    std::vector<int> next_chunk_;
    std::vector<int> layer_inflight_;
    std::vector<std::vector<tensor_join_state>> tensor_ready_;
};
