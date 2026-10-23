#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <vector>

// Phone-only dispatch policy for a fixed C2P -> Phone region -> P2C cycle.
// There is ONE GPU execution lane and two LOGICAL queues. A worker must
// never sleep waiting for a producer while holding the GPU gate.
//
// C2P FFN is background work. A Chunk joins MAIN only after its local FFN
// and the remote PC partial have BOTH become device-consumable.
// The ready-to-run MAIN continuation has priority over every queued FFN.
// Completion/ownership ACKs are deliberately outside this scheduler.
class llama_hybrid_cycle_phone_local {
public:
    enum class stage : uint8_t {
        C2P_FFN, C2P_REDUCE, PHONE_FULL, P2C_ATTN,
        P2C_READY_STAGED, P2C_FFN, P2C_PARTIAL_STAGED
    };
    struct task {
        stage op = stage::C2P_FFN;
        int chunk = -1;
        int full_layer = -1;
    };

    explicit llama_hybrid_cycle_phone_local(int chunks, int phone_full_layers)
        : state_(size_t(chunks)),
          phone_full_layers_(phone_full_layers) {
        if (chunks <= 0 || phone_full_layers < 0) {
            throw std::invalid_argument("invalid Phone-local cycle topology");
        }
    }

    // Data-plane READY has been applied to the actual Phone GPU buffers.
    // Do NOT run the FFN merely because a Chunk was registered.
    void phone_ready(int chunk) {
        std::lock_guard<std::mutex> lock(mu_);
        auto & s = get(chunk);
        if (s.ready) throw std::logic_error("duplicate Phone READY");
        s.ready = true;
        ffn_queue_.push_back({stage::C2P_FFN, chunk, -1});
    }

    // Same admission call may originate from RX or FFN completion; no
    // producer should have to wait for the other to arrive.
    void pc_partial_ready(int chunk) {
        std::lock_guard<std::mutex> lock(mu_);
        auto & s = get(chunk);
        if (s.pc_partial) throw std::logic_error("duplicate PC partial");
        s.pc_partial = true;
        admit_ready_reduces();
    }

    bool pop(task & result) {
        std::lock_guard<std::mutex> lock(mu_);
        if (running_) return false;
        if (!main_queue_.empty()) {
            result = main_queue_.front();
            main_queue_.pop_front();
        } else if (!ffn_queue_.empty()) {
            result = ffn_queue_.front();
            ffn_queue_.pop_front();
        } else {
            return false;
        }
        running_ = true;
        current_ = result;
        return true;
    }

    // One GPU task completes before another is selected. Follow-up MAIN
    // tasks go at the FRONT, so one Chunk returns to PC promptly rather
    // than being delayed by a batch of other Chunk reduces.
    void complete(const task & t) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!running_ || t.op != current_.op ||
            t.chunk != current_.chunk || t.full_layer != current_.full_layer) {
            throw std::logic_error("unexpected Phone task completion");
        }
        auto & s = get(t.chunk);
        running_ = false;
        switch (t.op) {
            case stage::C2P_FFN:
                if (s.ffn_done) throw std::logic_error("duplicate FFN");
                s.ffn_done = true;
                admit_ready_reduces();
                break;
            case stage::C2P_REDUCE:
                main_queue_.push_front(phone_full_layers_ ?
                    task{stage::PHONE_FULL, t.chunk, 0} :
                    task{stage::P2C_ATTN, t.chunk, -1});
                break;
            case stage::PHONE_FULL:
                main_queue_.push_front(t.full_layer + 1 < phone_full_layers_ ?
                    task{stage::PHONE_FULL, t.chunk, t.full_layer + 1} :
                    task{stage::P2C_ATTN, t.chunk, -1});
                break;
            case stage::P2C_ATTN:
                main_queue_.push_front({stage::P2C_READY_STAGED, t.chunk, -1});
                break;
            case stage::P2C_READY_STAGED:
                main_queue_.push_front({stage::P2C_FFN, t.chunk, -1});
                break;
            case stage::P2C_FFN:
                main_queue_.push_front({stage::P2C_PARTIAL_STAGED, t.chunk, -1});
                break;
            case stage::P2C_PARTIAL_STAGED:
                if (s.complete) throw std::logic_error("duplicate completion");
                s.complete = true;
                ++completed_;
                break;
        }
    }

    bool finished() const {
        std::lock_guard<std::mutex> lock(mu_);
        return completed_ == int(state_.size()) &&
               !running_ && main_queue_.empty() && ffn_queue_.empty();
    }

    int completed() const {
        std::lock_guard<std::mutex> lock(mu_);
        return completed_;
    }

    size_t main_pending() const {
        std::lock_guard<std::mutex> lock(mu_);
        return main_queue_.size();
    }
private:
    struct chunk_state {
        bool ready = false;
        bool ffn_done = false;
        bool pc_partial = false;
        bool reduce_admitted = false;
        bool complete = false;
    };
    chunk_state & get(int chunk) {
        if (chunk < 0 || size_t(chunk) >= state_.size())
            throw std::out_of_range("invalid chunk");
        return state_[size_t(chunk)];
    }
    void admit_ready_reduces() {
        while (next_reduce_chunk_ < int(state_.size())) {
            auto & s = state_[size_t(next_reduce_chunk_)];
            if (!s.ffn_done || !s.pc_partial) break;
            if (s.reduce_admitted) throw std::logic_error("duplicate reduce");
            s.reduce_admitted = true;
            main_queue_.push_back({stage::C2P_REDUCE, next_reduce_chunk_, -1});
            ++next_reduce_chunk_;
        }
    }

    mutable std::mutex mu_;
    std::vector<chunk_state> state_;
    std::deque<task> main_queue_;
    std::deque<task> ffn_queue_;
    int phone_full_layers_ = 0;
    int next_reduce_chunk_ = 0;
    int completed_ = 0;
    bool running_ = false;
    task current_ {};
};
