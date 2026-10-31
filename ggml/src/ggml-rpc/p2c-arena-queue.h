#pragma once

// Region Ring P2C stream staging. A completed producer publishes a full
// immutable chunk before it becomes visible to the single network sender.
// This component does not own a socket; the RPC transport sends a header
// followed by exactly payload_bytes bytes, then calls complete_send().
#include "ggml-rpc.h"

#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <vector>

class ggml_rpc_p2c_arena_queue {
public:
    struct item {
        ggml_backend_rpc_p2c_frame frame {};
        const uint8_t * data = nullptr;
    };

    bool initialize(uint64_t run_id, uint32_t n_regions,
                    uint32_t n_tokens, uint32_t n_embd) {
        if (!n_regions || !n_tokens || !n_embd) {
            return false;
        }
        const uint64_t stride = uint64_t(n_tokens) * n_embd * sizeof(float);
        if (stride > SIZE_MAX / n_regions) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ || in_flight_) {
            return false;
        }
        arena_.assign(size_t(stride) * n_regions, 0);
        published_.assign(n_regions, std::vector<uint8_t>(n_tokens, 0));
        region_stride_ = size_t(stride);
        n_tokens_ = n_tokens;
        n_embd_ = n_embd;
        run_id_ = run_id;
        last_seq_ = 0;
        stopped_ = false;
        active_ = true;
        return true;
    }

    // Caller must have completed the device->host read before publishing.
    // Copying into disjoint Arena intervals and enqueueing the descriptor are
    // one atomic publication operation. The sender cannot see partial data.
    bool publish(const ggml_backend_rpc_p2c_frame & frame,
                 const void * completed_host_data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_ || stopped_ || completed_host_data == nullptr ||
            frame.run_id != run_id_ || frame.seq <= last_seq_ ||
            frame.region_id >= published_.size() ||
            frame.n_tokens == 0 ||
            frame.token_begin > n_tokens_ ||
            frame.n_tokens > n_tokens_ - frame.token_begin ||
            frame.payload_bytes != size ||
            size != size_t(frame.n_tokens) * n_embd_ * sizeof(float) ||
            frame.kind != GGML_BACKEND_RPC_P2C_FULL) {
            return false;
        }
        auto & marks = published_[frame.region_id];
        for (uint32_t t = frame.token_begin;
             t < frame.token_begin + frame.n_tokens; ++t) {
            if (marks[t]) {
                return false;
            }
        }
        const size_t offset = size_t(frame.region_id) * region_stride_ +
                              size_t(frame.token_begin) * n_embd_ * sizeof(float);
        std::memcpy(arena_.data() + offset, completed_host_data, size);
        for (uint32_t t = frame.token_begin;
             t < frame.token_begin + frame.n_tokens; ++t) {
            marks[t] = 1;
        }
        last_seq_ = frame.seq;
        pending_.push_back(frame);
        cv_.notify_one();
        return true;
    }

    // Exactly one sender calls take() and complete_send().
    // A returned data pointer remains stable until complete_send() because
    // initialize/reset is forbidden while the stream is active.
    bool take(item & out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stopped_ || !pending_.empty(); });
        if (in_flight_ || pending_.empty()) {
            return false;
        }
        out.frame = pending_.front();
        pending_.pop_front();
        out.data = arena_.data() +
                   size_t(out.frame.region_id) * region_stride_ +
                   size_t(out.frame.token_begin) * n_embd_ * sizeof(float);
        in_flight_ = true;
        return true;
    }

    void complete_send(bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        in_flight_ = false;
        if (!success) {
            stopped_ = true;
        }
        cv_.notify_all();
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        cv_.notify_all();
    }

    bool drained() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.empty() && !in_flight_;
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<uint8_t> arena_;
    std::vector<std::vector<uint8_t>> published_;
    std::deque<ggml_backend_rpc_p2c_frame> pending_;
    size_t region_stride_ = 0;
    uint64_t run_id_ = 0;
    uint64_t last_seq_ = 0;
    uint32_t n_tokens_ = 0;
    uint32_t n_embd_ = 0;
    bool active_ = false;
    bool stopped_ = false;
    bool in_flight_ = false;
};
