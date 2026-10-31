#pragma once

// PC-side receiver for the dedicated Phone->PC stream socket.
// RPC commands must never share this socket: it carries only framed chunks.
// A single receive loop gives deterministic frame boundaries and naturally
// serializes the P2C payload traffic.
#include "ggml-rpc.h"
#include "transport.h"

#include <cstdint>
#include <limits>

class ggml_rpc_p2c_stream_receiver {
public:
    ggml_rpc_p2c_stream_receiver(
            uint64_t run_id, uint32_t n_regions, uint32_t n_tokens,
            uint32_t n_embd, ggml_backend_rpc_p2c_resolve_t resolve,
            ggml_backend_rpc_p2c_complete_t complete, void * userdata)
        : run_id_(run_id), n_regions_(n_regions), n_tokens_(n_tokens),
          n_embd_(n_embd), resolve_(resolve),
          complete_(complete), userdata_(userdata) {}

    // Runs only on the independent network thread. The caller owns the
    // socket, session lifecycle and cancellation policy.
    template <typename SocketPtr>
    bool receive(SocketPtr socket) {
        if (!socket || !resolve_ || !complete_ ||
            n_regions_ == 0 || n_tokens_ == 0 || n_embd_ == 0) {
            return false;
        }
        for (;;) {
            ggml_backend_rpc_p2c_frame frame {};
            if (!socket->recv_data(&frame, sizeof(frame))) {
                return false;
            }
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
            void * destination = resolve_(userdata_, &frame);
            if (destination == nullptr) {
                complete_(userdata_, &frame, false);
                return false;
            }
            const bool ok = socket->recv_data(
                destination, size_t(frame.payload_bytes));
            complete_(userdata_, &frame, ok);
            if (!ok) {
                return false;
            }
            last_seq_ = frame.seq;
        }
    }

private:
    uint64_t run_id_;
    uint64_t last_seq_ = 0;
    uint32_t n_regions_;
    uint32_t n_tokens_;
    uint32_t n_embd_;
    ggml_backend_rpc_p2c_resolve_t resolve_;
    ggml_backend_rpc_p2c_complete_t complete_;
    void * userdata_;
};
