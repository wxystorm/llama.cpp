#pragma once

// Phone-owned native Prefill task stream. No PC-authored GGML Graphs.
// One bounded FIFO worker serializes Chunk execution and private HTP KV.
// Only the final Chunk response is retained for the PC LM Head.
//
// This header deliberately knows nothing about RPC sockets or HTP tensor
// pointers. The same queue contract can be tested without a real Phone.

#include <cstddef>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ggml_phone_autonomous {

class prefill_queue {
public:
    using runner = std::function<bool(
        uint32_t, const std::vector<uint8_t> &, std::vector<uint8_t> &)>;

    prefill_queue(uint32_t n_chunks, size_t capacity, runner execute)
        : n_chunks_(n_chunks), capacity_(capacity), execute_(std::move(execute)),
          worker_([this] { run(); }) {}

    ~prefill_queue() { stop(); }

    prefill_queue(const prefill_queue &) = delete;
    prefill_queue & operator=(const prefill_queue &) = delete;

    // The caller owns a complete, validated RPC payload. Admission ACK
    // means accepted in the Phone queue, NOT that HTP has completed.
    bool push(uint32_t chunk, std::vector<uint8_t> payload) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::seconds(120), [&] {
                return stopped_ || failed_ || outstanding_ < capacity_;
            })) {
            fail_locked("INPUT_BACKPRESSURE_TIMEOUT");
            return false;
        }
        if (stopped_ || failed_ || chunk != accepted_ ||
                accepted_ >= n_chunks_ || payload.empty()) {
            fail_locked("INVALID_INPUT_FIFO_OR_STREAM_CLOSED");
            return false;
        }
        jobs_.push_back({chunk, std::move(payload)});
        last_progress_ = std::chrono::steady_clock::now();
        ++accepted_;
        ++outstanding_;
        cv_.notify_all();
        return true;
    }

    // Event-driven, independent P2C response: the Phone worker publishes
    // the final hidden-state payload when it finishes, and the waiting RPC
    // subscriber is released immediately. No GET_TENSOR polling.
    bool wait_final(std::vector<uint8_t> & response,
                    std::string & error) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, std::chrono::seconds(120), [&] {
                return finished_ || failed_ || stopped_;
            })) {
            fail_locked("P2C_COMPLETION_TIMEOUT");
        }
        if (!finished_ || failed_) {
            error = error_;
            if (error.empty()) { error = "STREAM_ABORTED"; }
            return false;
        }
        response = final_response_;
        return !response.empty();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!finished_ && !failed_) {
                failed_ = true;
                error_ = "STREAM_CANCELLED";
            }
            stopped_ = true;
            jobs_.clear();
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // A restarted PC must not leave a lost stream holding the Phone's only
    // active HTP Graph/KV generation forever. Used solely at the next
    // PREPARE boundary; never interrupts a recently active request.
    bool abandoned() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return failed_ || stopped_ ||
            (!finished_ && std::chrono::steady_clock::now() -
                last_progress_ > std::chrono::minutes(5));
    }

    uint32_t accepted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return accepted_;
    }

    uint32_t completed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return completed_;
    }

private:
    struct job {
        uint32_t chunk;
        std::vector<uint8_t> payload;
    };

    void fail_locked(const char * reason) {
        failed_ = true;
        error_ = reason;
        jobs_.clear();
        cv_.notify_all();
    }

    void run() {
        for (;;) {
            job next {};
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&] { return stopped_ || failed_ || !jobs_.empty(); });
                if (stopped_ || failed_) { return; }
                next = std::move(jobs_.front());
                jobs_.pop_front();
            }
            std::vector<uint8_t> response;
            bool ok = false;
            try {
                ok = execute_(next.chunk, next.payload, response);
            } catch (...) {
                ok = false;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (stopped_ || failed_) { return; }
                if (!ok || next.chunk != completed_) {
                    fail_locked("PHONE_NATIVE_HTP_EXECUTION_FAILED");
                    return;
                }
                ++completed_;
                last_progress_ = std::chrono::steady_clock::now();
                --outstanding_;
                if (completed_ == n_chunks_) {
                    if (response.empty()) {
                        fail_locked("PHONE_P2C_FINAL_OUTPUT_MISSING");
                        return;
                    }
                    final_response_ = std::move(response);
                    finished_ = true;
                }
            }
            cv_.notify_all();
        }
    }

    const uint32_t n_chunks_;
    const size_t capacity_;
    const runner execute_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<job> jobs_;
    uint32_t accepted_ = 0;
    uint32_t completed_ = 0;
    size_t outstanding_ = 0;
    bool failed_ = false;
    bool finished_ = false;
    bool stopped_ = false;
    std::string error_;
    std::chrono::steady_clock::time_point last_progress_ =
        std::chrono::steady_clock::now();
    std::vector<uint8_t> final_response_;
    std::thread worker_;
};

} // namespace ggml_phone_autonomous
