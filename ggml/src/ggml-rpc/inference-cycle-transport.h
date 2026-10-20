#pragma once

#include "inference-cycle-payload.h"
#include "inference-transport.h"

#include <cstdint>
#include <utility>

struct inference_cycle_message {
    inference_cycle_direction direction = INFERENCE_CYCLE_PHONE_TO_PC;
    inference_cycle_payload_kind kind = INFERENCE_CYCLE_READY;
    int32_t layer = -1;
    int32_t chunk = -1;
    uint64_t seq = 0;
    uint64_t graph_id = 0;
    inference_cycle_ready_owned ready;
    inference_cycle_partial_owned partial;
};

// The caller must submit from a transport worker, not a compute worker:
// acquire_tx() can block when all TX slots are in flight. The data is copied
// into the dedicated preallocated TX slot before this function returns.
inline bool inference_cycle_send_ready(
        inference_transport & transport,
        inference_cycle_direction direction,
        int32_t layer, int32_t chunk, uint64_t seq, uint64_t graph_id,
        const inference_cycle_ready_view & source) {
    size_t payload_bytes = 0;
    if (layer < 0 || chunk < 0 || seq == 0 ||
        (direction != INFERENCE_CYCLE_PC_TO_PHONE &&
         direction != INFERENCE_CYCLE_PHONE_TO_PC) ||
        !inference_cycle_payload_size(
            INFERENCE_CYCLE_READY, source.tokens, source.hidden, source.topk,
            payload_bytes)) {
        return false;
    }
    inference_tx_buffer tx = transport.acquire_tx(payload_bytes);
    if (!tx) {
        return false;
    }
    size_t written = 0;
    if (!inference_cycle_write_ready(tx.data, tx.capacity, source, written) ||
        written != payload_bytes) {
        transport.release_tx(std::move(tx));
        return false;
    }
    inference_frame_header header = inference_make_header(INFERENCE_FRAME_DATA);
    header.stage = INFERENCE_STAGE_CYCLE_READY;
    header.tensor_role = direction;
    header.layer = layer;
    header.chunk = chunk;
    header.seq = seq;
    header.graph_id = graph_id;
    return transport.submit_tx(std::move(tx), header, written);
}

inline bool inference_cycle_send_partial(
        inference_transport & transport,
        inference_cycle_direction direction,
        int32_t layer, int32_t chunk, uint64_t seq, uint64_t graph_id,
        const inference_cycle_partial_view & source) {
    size_t payload_bytes = 0;
    if (layer < 0 || chunk < 0 || seq == 0 ||
        (direction != INFERENCE_CYCLE_PC_TO_PHONE &&
         direction != INFERENCE_CYCLE_PHONE_TO_PC) ||
        !inference_cycle_payload_size(
            INFERENCE_CYCLE_PARTIAL, source.tokens, source.hidden, 0,
            payload_bytes)) {
        return false;
    }
    inference_tx_buffer tx = transport.acquire_tx(payload_bytes);
    if (!tx) {
        return false;
    }
    size_t written = 0;
    if (!inference_cycle_write_partial(tx.data, tx.capacity, source, written) ||
        written != payload_bytes) {
        transport.release_tx(std::move(tx));
        return false;
    }
    inference_frame_header header = inference_make_header(INFERENCE_FRAME_DATA);
    header.stage = INFERENCE_STAGE_CYCLE_PARTIAL;
    header.tensor_role = direction;
    header.layer = layer;
    header.chunk = chunk;
    header.seq = seq;
    header.graph_id = graph_id;
    return transport.submit_tx(std::move(tx), header, written);
}

// Decode a cycle frame AFTER an outer RPC dispatcher selects its stage, or
// on a dedicated cycle-only socket. Do not create a second receive loop on
// the existing shared RPC inference socket: it already has one owner.
inline bool inference_cycle_decode_frame(
        const inference_frame_header & header,
        const uint8_t * data, size_t size,
        inference_cycle_message & output) {
    if (header.magic != INFERENCE_TRANSPORT_MAGIC ||
        header.version != INFERENCE_TRANSPORT_VERSION ||
        header.header_bytes != sizeof(inference_frame_header) ||
        header.type != INFERENCE_FRAME_DATA ||
        header.layer < 0 || header.chunk < 0 || header.seq == 0 ||
        header.payload_bytes != size ||
        (header.tensor_role != INFERENCE_CYCLE_PC_TO_PHONE &&
         header.tensor_role != INFERENCE_CYCLE_PHONE_TO_PC)) {
        return false;
    }
    inference_cycle_message msg;
    msg.direction = static_cast<inference_cycle_direction>(header.tensor_role);
    msg.layer = header.layer;
    msg.chunk = header.chunk;
    msg.seq = header.seq;
    msg.graph_id = header.graph_id;
    if (header.stage == INFERENCE_STAGE_CYCLE_READY) {
        msg.kind = INFERENCE_CYCLE_READY;
        if (!inference_cycle_read_ready(data, size, msg.ready)) {
            return false;
        }
    } else if (header.stage == INFERENCE_STAGE_CYCLE_PARTIAL) {
        msg.kind = INFERENCE_CYCLE_PARTIAL;
        if (!inference_cycle_read_partial(data, size, msg.partial)) {
            return false;
        }
    } else {
        return false;
    }
    output = std::move(msg);
    return true;
}

// ACKs are small reverse-direction control frames, NOT an additional Tensor
// payload transfer. The receiver sends an ACK only after the destination
// compute backend can safely consume the received activation or partial.
// A successful submit_tx() or receive() is NOT a device-ready completion.
enum inference_cycle_ack_status : uint32_t {
    INFERENCE_CYCLE_ACK_DEVICE_READY = 1,
    INFERENCE_CYCLE_ACK_DEVICE_FAILED = 2,
};

struct inference_cycle_ack {
    inference_cycle_direction direction = INFERENCE_CYCLE_PHONE_TO_PC;
    inference_cycle_payload_kind kind = INFERENCE_CYCLE_READY;
    int32_t layer = -1;
    int32_t chunk = -1;
    uint64_t seq = 0;
    uint64_t graph_id = 0;
    bool device_ready = false;
};

// Safe to call only after the destination's upload/host copy and required
// backend fence are complete. For a failed upload, set device_ready=false so
// the source does not release/reuse its dependency as successful.
inline bool inference_cycle_send_ack(
        inference_transport & transport,
        const inference_cycle_message & received,
        bool device_ready) {
    if (received.layer < 0 || received.chunk < 0 || received.seq == 0 ||
        (received.direction != INFERENCE_CYCLE_PC_TO_PHONE &&
         received.direction != INFERENCE_CYCLE_PHONE_TO_PC) ||
        (received.kind != INFERENCE_CYCLE_READY &&
         received.kind != INFERENCE_CYCLE_PARTIAL)) {
        return false;
    }
    inference_tx_buffer tx = transport.acquire_tx(0);
    if (!tx) {
        return false;
    }
    inference_frame_header header = inference_make_header(INFERENCE_FRAME_ACK);
    header.stage = received.kind == INFERENCE_CYCLE_READY ?
        INFERENCE_STAGE_CYCLE_READY : INFERENCE_STAGE_CYCLE_PARTIAL;
    // tensor_role describes the ORIGINAL tensor direction, not the small
    // reverse ACK direction. The origin can therefore match its send task.
    header.tensor_role = received.direction;
    header.flags = device_ready ?
        INFERENCE_CYCLE_ACK_DEVICE_READY : INFERENCE_CYCLE_ACK_DEVICE_FAILED;
    header.layer = received.layer;
    header.chunk = received.chunk;
    header.seq = received.seq;
    header.graph_id = received.graph_id;
    return transport.submit_tx(std::move(tx), header, 0);
}

inline bool inference_cycle_decode_ack(
        const inference_frame_header & header,
        size_t payload_bytes,
        inference_cycle_ack & output) {
    if (header.magic != INFERENCE_TRANSPORT_MAGIC ||
        header.version != INFERENCE_TRANSPORT_VERSION ||
        header.header_bytes != sizeof(inference_frame_header) ||
        header.type != INFERENCE_FRAME_ACK ||
        header.payload_bytes != 0 || payload_bytes != 0 ||
        header.layer < 0 || header.chunk < 0 || header.seq == 0 ||
        (header.tensor_role != INFERENCE_CYCLE_PC_TO_PHONE &&
         header.tensor_role != INFERENCE_CYCLE_PHONE_TO_PC) ||
        (header.stage != INFERENCE_STAGE_CYCLE_READY &&
         header.stage != INFERENCE_STAGE_CYCLE_PARTIAL) ||
        (header.flags != INFERENCE_CYCLE_ACK_DEVICE_READY &&
         header.flags != INFERENCE_CYCLE_ACK_DEVICE_FAILED)) {
        return false;
    }

    inference_cycle_ack result;
    result.direction = static_cast<inference_cycle_direction>(header.tensor_role);
    result.kind = header.stage == INFERENCE_STAGE_CYCLE_READY ?
        INFERENCE_CYCLE_READY : INFERENCE_CYCLE_PARTIAL;
    result.layer = header.layer;
    result.chunk = header.chunk;
    result.seq = header.seq;
    result.graph_id = header.graph_id;
    result.device_ready = header.flags == INFERENCE_CYCLE_ACK_DEVICE_READY;
    output = result;
    return true;
}

// Dedicated cycle-only socket convenience. Successfully decoding a frame
// means that its data is HOST-ready. Ring task completion must be delayed
// until the destination CPU tensor or Phone GPU input buffer is DEVICE-ready.
inline bool inference_cycle_receive(
        inference_transport & transport,
        inference_cycle_message & output, int timeout_ms = -1) {
    inference_rx_frame frame;
    if (!transport.receive(frame, timeout_ms)) {
        return false;
    }
    const bool ok = inference_cycle_decode_frame(
        frame.header, frame.data, frame.size, output);
    transport.release_rx(std::move(frame));
    return ok;
}
