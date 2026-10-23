#pragma once

#include "llama-hybrid-cycle-p2c-ffn.h"
#include "llama-hybrid-cycle-wire.h"
#include "llama-hybrid-cycle-runtime.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Physical session spans one or more consecutive Cycle regions in the
// LOADED model. The V5 probe currently exercises [0,8); full-model callers
// may use [0,48) without constructing a barrier at every eighth layer.
//
// A session is per invocation, NOT a hidden graph or GPU memory allocator.
// External CUDA/Phone Graph caches and physical KV buffers keep their
// existing persistent owners; this session only borrows their pointers.
struct llama_hybrid_cycle_execution_session_config {
    llama_hybrid_cycle_layout layout;
    int physical_layer_begin = 0;
    int physical_layer_count = 0;
    int64_t tokens = 0;
    int chunk_tokens = 0;
    uint64_t wire_epoch = 0;
    int64_t token_position_base = 0;
};

LLAMA_API bool llama_hybrid_cycle_validate_session_config(
    const llama_hybrid_cycle_execution_session_config & config,
    std::string & reason);

// Legacy V5 wire IDs are preserved byte-for-byte for layer 0..47 and epoch
// 0/1. The generator explicitly rejects positions outside the collision-free
// (layer<100, chunks<1000) identity space, rather than wrapping a sequence.
LLAMA_API uint64_t llama_hybrid_cycle_session_wire_seq(
    uint64_t epoch, int physical_layer, int chunk);

// Optional observational PC timestamps. Each field is written by the same
// stage/worker that wrote it in the old V5 test; no new cross-queue lock.
struct llama_hybrid_cycle_c2p_path_timing {
    double pc_attn_call_ms = 0.0;
    double phone_slot_prepare_ms = 0.0;
    double pc_ffn_call_ms = 0.0;
    double phone_ffn_submit_rpc_ms = 0.0;
    llama_hybrid_cycle_wire_submit_timing ready_tx {};
    double ready_apply_rpc_ms = 0.0;
    double ready_ack_wait_ms = 0.0;
    llama_hybrid_cycle_wire_submit_timing partial_tx {};
    double partial_apply_rpc_ms = 0.0;
    double reduce_graph_rpc_ms = 0.0;
    double reduce_backend_sync_ms = 0.0;
    double reduce_finish_rpc_ms = 0.0;
    double reduce_ack_wait_ms = 0.0;
};

struct llama_hybrid_cycle_p2c_path_timing {
    size_t ready_rx_bytes = 0;
    size_t partial_rx_bytes = 0;
    double ready_receive_decode_ms = 0.0;
    double ready_ack_rpc_ms = 0.0;
    double phone_ffn_prepare_ms = 0.0;
    double phone_ffn_submit_rpc_ms = 0.0;
    double phone_send_partial_rpc_ms = 0.0;
    // Fallback without FFN pool includes the output-send RPC.
    double phone_ffn_fallback_call_ms = 0.0;
    double pc_ffn_call_ms = 0.0;
    double partial_receive_decode_ms = 0.0;
    double pc_reduce_call_ms = 0.0;
    double partial_ack_rpc_ms = 0.0;
};

struct llama_hybrid_cycle_p2c_overlap_events {
    std::chrono::steady_clock::time_point ready_received {};
    std::chrono::steady_clock::time_point cpu_ffn_started {};
    std::chrono::steady_clock::time_point cpu_ffn_finished {};
    std::chrono::steady_clock::time_point partial_received {};
};

// Per-physical-layer, per-Chunk activation/transfer owner. Device Graph
// buffers are held by the typed slots; they must NOT be recycled when a
// compute callback merely finishes enqueueing work.
template<class PhoneSlot, class P2cSlot>
struct llama_hybrid_cycle_chunk_state {
    llama_hybrid_cycle_c2p_path_timing c2p_timing {};
    llama_hybrid_cycle_p2c_path_timing p2c_timing {};
    llama_hybrid_cycle_p2c_overlap_events p2c_overlap {};
    int layer = 0;
    int chunk = 0;
    int64_t start = 0;
    int64_t tokens = 0;
    llama_hybrid_cycle_live_wire wire {};
    llama_hybrid_cycle_attention_ready ready;
    std::vector<float> partial;
    std::vector<float> host_output;
    inference_cycle_message recv_ready;
    inference_cycle_message recv_partial;
    std::unique_ptr<PhoneSlot> phone_output;
    // Patch 33: each pre-registered L7 FFN owns its own buffer slot.
    // A single shared P2C FFN slot would be unsafe with window=2 while
    // the next Chunk is registered before the previous FFN completed.
    std::unique_ptr<P2cSlot> p2c_main_ffn;
};

template<class PhoneSlot = llama_hybrid_cycle_phone_ffn_slot,
         class P2cSlot = llama_hybrid_cycle_p2c_ffn_slot>
class llama_hybrid_cycle_execution_session {
public:
    using chunk_state = llama_hybrid_cycle_chunk_state<PhoneSlot, P2cSlot>;
    using state_ptr = std::unique_ptr<chunk_state>;
    using storage_type = std::vector<std::vector<state_ptr>>;

    explicit llama_hybrid_cycle_execution_session(
            llama_hybrid_cycle_execution_session_config config)
        : config_(std::move(config)) {
        std::string reason;
        if (!llama_hybrid_cycle_validate_session_config(config_, reason)) {
            throw std::invalid_argument("cycle execution session: " + reason);
        }
        n_chunks_ = int((config_.tokens - 1) / config_.chunk_tokens + 1);
        states_.resize(static_cast<size_t>(config_.physical_layer_count));
        kv_borrowed_.resize(static_cast<size_t>(config_.physical_layer_count),
                            nullptr);
        for (int offset = 0; offset < config_.physical_layer_count; ++offset) {
            const int layer = config_.physical_layer_begin + offset;
            auto & row = states_[static_cast<size_t>(offset)];
            row.reserve(static_cast<size_t>(n_chunks_));
            for (int c = 0; c < n_chunks_; ++c) {
                auto st = std::make_unique<chunk_state>();
                st->layer = layer;
                st->chunk = c;
                st->start = config_.token_position_base +
                            int64_t(c) * config_.chunk_tokens;
                st->tokens = std::min(
                    int64_t(config_.chunk_tokens),
                    config_.tokens - int64_t(c) * config_.chunk_tokens);
                st->wire = {
                    layer, c,
                    llama_hybrid_cycle_session_wire_seq(
                        config_.wire_epoch, layer, c), 1
                };
                row.push_back(std::move(st));
            }
        }
    }

    llama_hybrid_cycle_execution_session(
        const llama_hybrid_cycle_execution_session &) = delete;
    llama_hybrid_cycle_execution_session & operator=(
        const llama_hybrid_cycle_execution_session &) = delete;

    int first_layer() const { return config_.physical_layer_begin; }
    int layer_count() const { return config_.physical_layer_count; }
    int chunks_per_layer() const { return n_chunks_; }
    const llama_hybrid_cycle_execution_session_config & config() const {
        return config_;
    }

    // Returned state storage is stable for the entire invocation. The same
    // layer/chunk address is observed by independent compute and TX workers.
    storage_type & states() { return states_; }
    const storage_type & states() const { return states_; }

    chunk_state & at(int physical_layer, int chunk) {
        return *states_.at(row_index(physical_layer))
                       .at(static_cast<size_t>(checked_chunk(chunk)));
    }
    const chunk_state & at(int physical_layer, int chunk) const {
        return *states_.at(row_index(physical_layer))
                       .at(static_cast<size_t>(checked_chunk(chunk)));
    }

    // The existing model/session owns actual KV allocation. Borrow only
    // until the FIFO workers have joined and before KV is returned to Graph
    // caches. Never free or reset an externally-owned KV tensor here.
    bool borrow_kv(int physical_layer,
                   llama_hybrid_cycle_physical_kv_cache * kv) {
        if (!kv || !contains_layer(physical_layer)) return false;
        auto & slot = kv_borrowed_[row_index(physical_layer)];
        if (slot != nullptr) return false;
        slot = kv;
        return true;
    }
    bool kv_complete(int64_t expected_tokens) const {
        if (expected_tokens <= 0) return false;
        for (const auto * kv : kv_borrowed_) {
            if (!kv || kv->committed_tokens != expected_tokens) {
                return false;
            }
        }
        return true;
    }

    // Caller MUST invoke this only after the successor has consumed the
    // producer's GPU output (Attention fence / ordered input copy / TX
    // staging ACK). Moving the slot to the existing idle pool is explicitly
    // separate from the task completion and does not add a backend sync.
    std::unique_ptr<PhoneSlot> retire_phone_output_after_handoff(
            int producer_layer, int successor_layer, int chunk) {
        // Validate both indices BEFORE testing adjacency: producer_layer
        // comes from the caller and must never overflow an int increment.
        if (!contains_layer(producer_layer) ||
            !contains_layer(successor_layer) ||
            int64_t(successor_layer) != int64_t(producer_layer) + 1 ||
            chunk < 0 || chunk >= n_chunks_) {
            return {};
        }
        return std::move(at(producer_layer, chunk).phone_output);
    }

private:
    bool contains_layer(int layer) const {
        return layer >= config_.physical_layer_begin &&
               int64_t(layer) < int64_t(config_.physical_layer_begin) +
                                config_.physical_layer_count;
    }
    size_t row_index(int physical_layer) const {
        if (!contains_layer(physical_layer)) {
            throw std::out_of_range("cycle physical layer outside session");
        }
        return static_cast<size_t>(
            physical_layer - config_.physical_layer_begin);
    }
    int checked_chunk(int c) const {
        if (c < 0 || c >= n_chunks_) {
            throw std::out_of_range("cycle chunk outside session");
        }
        return c;
    }

    llama_hybrid_cycle_execution_session_config config_;
    int n_chunks_ = 0;
    storage_type states_;
    std::vector<llama_hybrid_cycle_physical_kv_cache *> kv_borrowed_;
};
