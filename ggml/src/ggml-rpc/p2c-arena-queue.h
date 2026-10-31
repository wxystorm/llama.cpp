#pragma once

// A whole-prefill, per-Region Phone host Arena. Every token interval is
// written once, and becomes sendable ONLY after GPU->host read completes.
// One network sender serializes all ready frames. GPU/Phone compute never
// waits for the TCP sender to drain this queue.
#include "ggml-rpc.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
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
        if (run_id == 0 || n_regions == 0 || n_tokens == 0 || n_embd == 0) {
            return false;
        }
        const uint64_t bytes_per_region =
            uint64_t(n_tokens) * n_embd * sizeof(float);
        if (bytes_per_region > SIZE_MAX / n_regions ||
            bytes_per_region > SIZE_MAX) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_) {
            return false;
        }
        try {
            arena_.assign(size_t(bytes_per_region) * n_regions, 0);
            marks_.assign(n_regions, std::vector<uint8_t>(n_tokens, 0));
        } catch (...) {
            return false;
        }
        queue_.clear();
        stride_ = size_t(bytes_per_region);
        n_tokens_ = n_tokens;
        n_embd_ = n_embd;
        run_id_ = run_id;
        last_seq_ = 0;
        in_flight_ = false;
        finishing_ = false;
        failed_ = false;
        active_ = true;
        return true;
    }

    // Read directly into the preallocated Arena. The reservation lock is
    // released before the device read, so an earlier READY chunk may be sent
    // concurrently. The reader MUST complete the GPU->host read synchronously.
    template<typename Reader>
    bool capture(ggml_backend_rpc_p2c_frame frame, Reader reader) {
        uint8_t * destination = nullptr;
        size_t size = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!active_ || finishing_ || failed_ ||
                frame.run_id != run_id_ ||
                frame.region_id >= marks_.size() ||
                frame.kind != GGML_BACKEND_RPC_P2C_FULL ||
                frame.reserved != 0 ||
                frame.n_tokens == 0 ||
                frame.token_begin > n_tokens_ ||
                frame.n_tokens > n_tokens_ - frame.token_begin ||
                frame.payload_bytes !=
                    uint64_t(frame.n_tokens) * n_embd_ * sizeof(float)) {
                return false;
            }
            auto & marks = marks_[frame.region_id];
            for (uint32_t i = frame.token_begin;
                 i < frame.token_begin + frame.n_tokens; ++i) {
                if (marks[i] != 0) {
                    return false;
                }
            }
            for (uint32_t i = frame.token_begin;
                 i < frame.token_begin + frame.n_tokens; ++i) {
                marks[i] = 1; // FILLING, never visible to Sender
            }
            size = size_t(frame.payload_bytes);
            destination = arena_.data() +
                size_t(frame.region_id) * stride_ +
                size_t(frame.token_begin) * n_embd_ * sizeof(float);
        }

        bool ok = false;
        try {
            ok = reader(destination, size);
        } catch (...) {
            ok = false;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (ok && !failed_) {
                frame.seq = last_seq_ + 1; // assign publication order
                try {
                    queue_.push_back(frame);
                    last_seq_ = frame.seq;
                    auto & marks = marks_[frame.region_id];
                    for (uint32_t i = frame.token_begin;
                         i < frame.token_begin + frame.n_tokens; ++i) {
                        marks[i] = 2; // READY, immutable until stream shutdown
                    }
                } catch (...) {
                    ok = false;
                    failed_ = true;
                    finishing_ = true;
                }
            } else {
                ok = false;
                failed_ = true;
                finishing_ = true;
            }
        }
        cv_.notify_all();
        return ok;
    }

    // Called by ONE Sender thread. Returns false after FINISH and drain.
    bool take(item & out) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return finishing_ || !queue_.empty(); });
        if (in_flight_ || failed_ || queue_.empty()) {
            return false;
        }
        out.frame = queue_.front();
        queue_.pop_front();
        out.data = arena_.data() +
            size_t(out.frame.region_id) * stride_ +
            size_t(out.frame.token_begin) * n_embd_ * sizeof(float);
        in_flight_ = true;
        return true;
    }

    // Bounded equivalent for tests and diagnostics. The production Sender
    // retains take(), which waits until a READY frame or FINISH.
    bool take_for(item & out, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, timeout,
                          [&] { return finishing_ || failed_ || !queue_.empty(); })) {
            return false;
        }
        if (in_flight_ || failed_ || queue_.empty()) {
            return false;
        }
        out.frame = queue_.front();
        queue_.pop_front();
        out.data = arena_.data() +
            size_t(out.frame.region_id) * stride_ +
            size_t(out.frame.token_begin) * n_embd_ * sizeof(float);
        in_flight_ = true;
        return true;
    }

    void complete_send(bool ok) {
        std::lock_guard<std::mutex> lock(mutex_);
        in_flight_ = false;
        if (!ok) {
            failed_ = true;
            finishing_ = true;
        }
        cv_.notify_all();
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        finishing_ = true;
        cv_.notify_all();
    }

    bool failed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return failed_;
    }

    uint64_t last_seq() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_seq_;
    }

    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Caller owns the single Sender and has joined/drained it first.
        active_ = false;
        queue_.clear();
        marks_.clear();
        arena_.clear();
        in_flight_ = false;
        finishing_ = false;
        failed_ = false;
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<uint8_t> arena_;
    std::vector<std::vector<uint8_t>> marks_;
    std::deque<ggml_backend_rpc_p2c_frame> queue_;
    size_t stride_ = 0;
    uint64_t run_id_ = 0;
    uint64_t last_seq_ = 0;
    uint32_t n_tokens_ = 0;
    uint32_t n_embd_ = 0;
    bool active_ = false;
    bool finishing_ = false;
    bool failed_ = false;
    bool in_flight_ = false;
};
