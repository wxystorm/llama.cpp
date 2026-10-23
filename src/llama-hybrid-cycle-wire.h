#pragma once

#include "llama-hybrid-cycle-attention.h"
#include "../ggml/src/ggml-rpc/inference-cycle-transport.h"

#include <cstdint>
#include <vector>

// Encoded PC-to-Phone frames are submitted to the existing inference
// transport backend via its registered proc, never through a new socket.
// Timings are host-side queueing only, not measured wire latency.
struct llama_hybrid_cycle_wire_submit_timing {
    size_t payload_bytes = 0;
    double encode_ms = 0.0;
    double enqueue_call_ms = 0.0;
};

LLAMA_API bool llama_hybrid_cycle_receive_wire(
        ggml_backend_t phone, const llama_hybrid_cycle_live_wire & wire,
        inference_cycle_payload_kind kind,
        uint32_t tokens, uint32_t hidden, uint32_t topk,
        inference_cycle_message & output);

LLAMA_API bool llama_hybrid_cycle_send_c2p_frame(
        ggml_backend_t phone, const llama_hybrid_cycle_live_wire & wire,
        inference_cycle_payload_kind kind,
        const llama_hybrid_cycle_attention_ready & ready,
        const std::vector<float> & partial,
        uint32_t tokens, uint32_t hidden, uint32_t topk,
        llama_hybrid_cycle_wire_submit_timing * timing = nullptr);
