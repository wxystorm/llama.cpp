#pragma once

#include <cstdint>

// V1 wire format. PC x86-64 and Android arm64 are both little-endian.
// Keep this header fixed-size so framing does not depend on RPC structs.
static constexpr uint32_t INFERENCE_TRANSPORT_MAGIC   = 0x4954524eU; // "ITRN"
static constexpr uint16_t INFERENCE_TRANSPORT_VERSION = 1;

enum inference_frame_type : uint16_t {
    INFERENCE_FRAME_DATA = 1,
    INFERENCE_FRAME_ACK  = 2,
    INFERENCE_FRAME_PING = 3,
};

enum inference_stage : uint16_t {
    INFERENCE_STAGE_UNSPECIFIED = 0,
    INFERENCE_STAGE_ATTN_OUT    = 1,
    INFERENCE_STAGE_FFN_IN      = 2,
    INFERENCE_STAGE_FFN_OUT     = 3,
    INFERENCE_STAGE_RETURN      = 4,
    INFERENCE_STAGE_REDUCE_IN   = 5,
};

struct inference_frame_header {
    uint32_t magic;
    uint16_t version;
    uint16_t type;

    uint32_t flags;

    uint16_t header_bytes;
    uint16_t stage;
    uint16_t tensor_role;
    uint16_t priority;

    int32_t  layer;
    int32_t  chunk;
    uint32_t stream_id;

    uint64_t seq;
    uint64_t graph_id;
    uint64_t payload_bytes;

    uint32_t reserved0;
    uint32_t reserved1;
};

static_assert(sizeof(inference_frame_header) == 64, "inference frame header must stay fixed-size");

inline inference_frame_header inference_make_header(uint16_t type = INFERENCE_FRAME_DATA) {
    inference_frame_header h = {};
    h.magic        = INFERENCE_TRANSPORT_MAGIC;
    h.version      = INFERENCE_TRANSPORT_VERSION;
    h.type         = type;
    h.header_bytes = sizeof(inference_frame_header);
    h.layer        = -1;
    h.chunk        = -1;
    return h;
}
