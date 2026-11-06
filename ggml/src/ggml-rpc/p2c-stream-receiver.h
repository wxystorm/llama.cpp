#pragma once

// PC-side receiver for the dedicated Phone->PC stream socket.
// RPC commands must never share this socket: it carries only framed chunks.
// A single receive loop gives deterministic frame boundaries and naturally
// serializes the P2C payload traffic.
#include "ggml-rpc.h"
#include "transport.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <vector>

class ggml_rpc_p2c_stream_receiver {
public:
    using clock_us_fn = int64_t (*)();

    // The production RPC caller injects ggml_time_us so these timestamps
    // share the exact PC_GGML_TIME_US origin of region_ring_chunk_trace.
    // Standalone unit tests need no ggml-base linkage and use steady_clock.
    ggml_rpc_p2c_stream_receiver(
            uint64_t run_id, uint32_t n_regions, uint32_t n_tokens,
            uint32_t n_embd, ggml_backend_rpc_p2c_resolve_t resolve,
            ggml_backend_rpc_p2c_complete_t complete, void * userdata,
            clock_us_fn clock_us = nullptr)
        : run_id_(run_id), n_regions_(n_regions), n_tokens_(n_tokens),
          n_embd_(n_embd), resolve_(resolve),
          complete_(complete), userdata_(userdata),
          clock_us_(clock_us) {}

    ~ggml_rpc_p2c_stream_receiver() {
        // Flush only after the receive loop exits. Never write trace logs
        // between a frame header and its payload or while frames are queued.
        if (!trace_enabled_) { return; }
        for (const auto & t : trace_) {
            std::fprintf(stderr,
                "[PC_CHUNK_P2C_RECV] clock=PC_GGML_TIME_US "
                "p2c_run=%" PRIu64 " region=%u chunk=%u"
                " token_begin=%u tokens=%u seq=%" PRIu64
                " bytes=%" PRIu64
                " frame_wait_begin_us=%" PRId64
                " frame_header_end_us=%" PRId64
                " resolve_begin_us=%" PRId64
                " resolve_end_us=%" PRId64
                " payload_begin_us=%" PRId64
                " payload_end_us=%" PRId64
                " callback_begin_us=%" PRId64
                " callback_end_us=%" PRId64
                " status=%s\n",
                run_id_, t.region, t.chunk, t.token_begin,
                t.n_tokens, t.seq, t.bytes,
                t.frame_wait_begin_us, t.frame_header_end_us,
                t.resolve_begin_us, t.resolve_end_us,
                t.payload_begin_us, t.payload_end_us,
                t.callback_begin_us, t.callback_end_us,
                t.ok ? "PASS" : "FAIL");
        }
    }

    uint64_t frames_received() const { return frames_; }
    uint64_t payload_bytes_received() const { return payload_bytes_; }
    int64_t payload_recv_us() const { return payload_recv_us_; }

    // Runs only on the independent network thread. The caller owns the
    // socket, session lifecycle and cancellation policy.
    template <typename SocketPtr>
    bool receive(SocketPtr socket) {
        if (!socket || !resolve_ || !complete_ ||
            n_regions_ == 0 || n_tokens_ == 0 || n_embd_ == 0) {
            return false;
        }
        const char * env = std::getenv("GGML_RPC_P2C_CHUNK_TIMELINE");
        trace_enabled_ = env != nullptr && std::strcmp(env, "1") == 0;
        if (trace_enabled_) { trace_.reserve(64); }
        for (;;) {
            ggml_backend_rpc_p2c_frame frame {};
            const int64_t header_begin_us =
                trace_enabled_ ? trace_now_us() : 0;
            if (!socket->recv_data(&frame, sizeof(frame))) {
                return false;
            }
            const int64_t header_end_us =
                trace_enabled_ ? trace_now_us() : 0;
            if (frame.run_id != run_id_ || frame.reserved != 0 ||
                frame.seq <= last_seq_) {
                return false;
            }
            if (frame.kind == GGML_BACKEND_RPC_P2C_END) {
                return frame.payload_bytes == 0;
            }
            if (frame.kind == GGML_BACKEND_RPC_P2C_ERROR) {
                return false;
            }
            // The FULL output is a contiguous FP32 hidden-state slab. Future
            // Tensor partial payloads need their own explicitly typed shape
            // validation; do not silently interpret them as FULL.
            if (frame.kind != GGML_BACKEND_RPC_P2C_FULL ||
                frame.region_id >= n_regions_ ||
                frame.n_tokens == 0 || frame.token_begin > n_tokens_ ||
                frame.n_tokens > n_tokens_ - frame.token_begin ||
                frame.payload_bytes !=
                    uint64_t(frame.n_tokens) * n_embd_ * sizeof(float) ||
                frame.payload_bytes > SIZE_MAX) {
                return false;
            }
            const int64_t resolve_begin_us =
                trace_enabled_ ? trace_now_us() : 0;
            void * destination = resolve_(userdata_, &frame);
            const int64_t resolve_end_us =
                trace_enabled_ ? trace_now_us() : 0;
            if (destination == nullptr) {
                complete_(userdata_, &frame, false);
                return false;
            }
            const int64_t payload_begin_us =
                trace_enabled_ ? trace_now_us() : 0;
            const auto recv_begin = std::chrono::steady_clock::now();
            const bool ok = socket->recv_data(
                destination, size_t(frame.payload_bytes));
            const auto recv_end = std::chrono::steady_clock::now();
            const int64_t payload_end_us =
                trace_enabled_ ? trace_now_us() : 0;
            payload_recv_us_ +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    recv_end - recv_begin).count();
            if (ok) {
                ++frames_;
                payload_bytes_ += frame.payload_bytes;
            }
            const int64_t callback_begin_us =
                trace_enabled_ ? trace_now_us() : 0;
            complete_(userdata_, &frame, ok);
            if (trace_enabled_) {
                frame_trace t {};
                t.region = frame.region_id;
                t.chunk = frame.chunk_id;
                t.token_begin = frame.token_begin;
                t.n_tokens = frame.n_tokens;
                t.seq = frame.seq;
                t.bytes = frame.payload_bytes;
                t.frame_wait_begin_us = header_begin_us;
                t.frame_header_end_us = header_end_us;
                t.resolve_begin_us = resolve_begin_us;
                t.resolve_end_us = resolve_end_us;
                t.payload_begin_us = payload_begin_us;
                t.payload_end_us = payload_end_us;
                t.callback_begin_us = callback_begin_us;
                t.callback_end_us = trace_now_us();
                t.ok = ok;
                trace_.push_back(t);
            }
            if (!ok) {
                return false;
            }
            last_seq_ = frame.seq;
        }
    }

private:
    int64_t trace_now_us() const {
        if (clock_us_ != nullptr) { return clock_us_(); }
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    struct frame_trace {
        uint32_t region = 0;
        uint32_t chunk = 0;
        uint32_t token_begin = 0;
        uint32_t n_tokens = 0;
        uint64_t seq = 0;
        uint64_t bytes = 0;
        int64_t frame_wait_begin_us = 0;
        int64_t frame_header_end_us = 0;
        int64_t resolve_begin_us = 0;
        int64_t resolve_end_us = 0;
        int64_t payload_begin_us = 0;
        int64_t payload_end_us = 0;
        int64_t callback_begin_us = 0;
        int64_t callback_end_us = 0;
        bool ok = false;
    };
    bool trace_enabled_ = false;
    std::vector<frame_trace> trace_;
    uint64_t run_id_;
    uint64_t last_seq_ = 0;
    uint32_t n_regions_;
    uint32_t n_tokens_;
    uint32_t n_embd_;
    ggml_backend_rpc_p2c_resolve_t resolve_;
    ggml_backend_rpc_p2c_complete_t complete_;
    void * userdata_;
    clock_us_fn clock_us_ = nullptr;
    uint64_t frames_ = 0;
    uint64_t payload_bytes_ = 0;
    int64_t payload_recv_us_ = 0;
};
