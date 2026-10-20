#pragma once

#include "inference-transport-protocol.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// A fixed, little-endian V1 payload for the two data messages of a one-way
// Tensor block. This is independent of GGML tensor allocation/layout and the
// old phone-primary Route/Return machinery.
//
// SEND #1 (READY): residual [tokens,hidden] f32,
//                  ffn_norm [tokens,hidden] f32,
//                  topk_ids [tokens,topk] i32,
//                  topk_weights [tokens,topk] f32.
// SEND #2 (PARTIAL): source FFN contribution [tokens,hidden] f32.
// All arrays are contiguous, with hidden/topk dimension varying fastest.
//
// The receiver owns the full residual and alone performs the final ADD:
//   l_out = residual + FFN_destination + FFN_source.
// No residual is added by the sender in this V1 protocol.
enum inference_cycle_direction : uint16_t {
    INFERENCE_CYCLE_PC_TO_PHONE = 1,
    INFERENCE_CYCLE_PHONE_TO_PC = 2,
};

enum inference_cycle_payload_kind : uint16_t {
    INFERENCE_CYCLE_READY = 1,
    INFERENCE_CYCLE_PARTIAL = 2,
};

static constexpr uint32_t INFERENCE_CYCLE_PAYLOAD_MAGIC = 0x4c435943U; // "CYCL"
static constexpr uint16_t INFERENCE_CYCLE_PAYLOAD_VERSION = 1;

struct inference_cycle_payload_header {
    uint32_t magic = INFERENCE_CYCLE_PAYLOAD_MAGIC;
    uint16_t version = INFERENCE_CYCLE_PAYLOAD_VERSION;
    uint16_t kind = 0;
    uint32_t tokens = 0;
    uint32_t hidden = 0;
    uint32_t topk = 0;
    uint32_t reserved = 0;
};
static_assert(sizeof(inference_cycle_payload_header) == 24, "fixed inference cycle payload header");

struct inference_cycle_ready_view {
    uint32_t tokens = 0;
    uint32_t hidden = 0;
    uint32_t topk = 0;
    const float * residual = nullptr;
    const float * ffn_norm = nullptr;
    const int32_t * topk_ids = nullptr;
    const float * topk_weights = nullptr;
};

struct inference_cycle_partial_view {
    uint32_t tokens = 0;
    uint32_t hidden = 0;
    const float * partial = nullptr;
};

struct inference_cycle_ready_owned {
    uint32_t tokens = 0;
    uint32_t hidden = 0;
    uint32_t topk = 0;
    std::vector<float> residual;
    std::vector<float> ffn_norm;
    std::vector<int32_t> topk_ids;
    std::vector<float> topk_weights;
};

struct inference_cycle_partial_owned {
    uint32_t tokens = 0;
    uint32_t hidden = 0;
    std::vector<float> partial;
};

inline bool inference_cycle_mul(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

inline bool inference_cycle_add(size_t a, size_t b, size_t & result) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    result = a + b;
    return true;
}

inline bool inference_cycle_payload_size(
        inference_cycle_payload_kind kind, uint32_t tokens, uint32_t hidden,
        uint32_t topk, size_t & bytes) {
    if (tokens == 0 || hidden == 0 ||
        (kind != INFERENCE_CYCLE_READY && kind != INFERENCE_CYCLE_PARTIAL) ||
        (kind == INFERENCE_CYCLE_PARTIAL && topk != 0)) {
        return false;
    }
    size_t activations = 0;
    size_t routing = 0;
    if (!inference_cycle_mul(tokens, hidden, activations) ||
        !inference_cycle_mul(tokens, topk, routing)) {
        return false;
    }

    size_t count = activations;
    if (kind == INFERENCE_CYCLE_READY) {
        // Two f32 activation arrays plus an i32 and f32 routing array.
        if (!inference_cycle_mul(activations, 2, count) ||
            !inference_cycle_mul(routing, 2, routing) ||
            !inference_cycle_add(count, routing, count)) {
            return false;
        }
    }
    if (!inference_cycle_mul(count, sizeof(float), count) ||
        !inference_cycle_add(count, sizeof(inference_cycle_payload_header), bytes)) {
        return false;
    }
    return true;
}

inline bool inference_cycle_write_ready(
        uint8_t * dst, size_t capacity,
        const inference_cycle_ready_view & value, size_t & written) {
    size_t required = 0;
    if (!dst || !value.residual || !value.ffn_norm ||
        (value.topk != 0 && (!value.topk_ids || !value.topk_weights)) ||
        !inference_cycle_payload_size(
            INFERENCE_CYCLE_READY, value.tokens, value.hidden, value.topk, required) ||
        required > capacity) {
        return false;
    }
    const inference_cycle_payload_header h {
        INFERENCE_CYCLE_PAYLOAD_MAGIC,
        INFERENCE_CYCLE_PAYLOAD_VERSION,
        INFERENCE_CYCLE_READY,
        value.tokens, value.hidden, value.topk, 0,
    };
    size_t offset = 0;
    auto write = [&](const void * src, size_t n) {
        if (n != 0) {
            std::memcpy(dst + offset, src, n);
        }
        offset += n;
    };
    const size_t activation_bytes = size_t(value.tokens) * value.hidden * sizeof(float);
    const size_t routing_bytes = size_t(value.tokens) * value.topk * sizeof(float);
    write(&h, sizeof(h));
    write(value.residual, activation_bytes);
    write(value.ffn_norm, activation_bytes);
    write(value.topk_ids, routing_bytes);
    write(value.topk_weights, routing_bytes);
    written = offset;
    return offset == required;
}

inline bool inference_cycle_write_partial(
        uint8_t * dst, size_t capacity,
        const inference_cycle_partial_view & value, size_t & written) {
    size_t required = 0;
    if (!dst || !value.partial ||
        !inference_cycle_payload_size(
            INFERENCE_CYCLE_PARTIAL, value.tokens, value.hidden, 0, required) ||
        required > capacity) {
        return false;
    }
    const inference_cycle_payload_header h {
        INFERENCE_CYCLE_PAYLOAD_MAGIC,
        INFERENCE_CYCLE_PAYLOAD_VERSION,
        INFERENCE_CYCLE_PARTIAL,
        value.tokens, value.hidden, 0, 0,
    };
    std::memcpy(dst, &h, sizeof(h));
    std::memcpy(dst + sizeof(h), value.partial, required - sizeof(h));
    written = required;
    return true;
}

inline bool inference_cycle_parse_payload_header(
        const uint8_t * src, size_t size,
        inference_cycle_payload_kind expected,
        inference_cycle_payload_header & out) {
    if (!src || size < sizeof(out)) {
        return false;
    }
    std::memcpy(&out, src, sizeof(out));
    size_t required = 0;
    return out.magic == INFERENCE_CYCLE_PAYLOAD_MAGIC &&
           out.version == INFERENCE_CYCLE_PAYLOAD_VERSION &&
           out.kind == expected && out.reserved == 0 &&
           inference_cycle_payload_size(expected, out.tokens, out.hidden, out.topk, required) &&
           required == size;
}

inline bool inference_cycle_read_ready(
        const uint8_t * src, size_t size,
        inference_cycle_ready_owned & output) {
    inference_cycle_payload_header h {};
    if (!inference_cycle_parse_payload_header(src, size, INFERENCE_CYCLE_READY, h)) {
        return false;
    }
    // Decode by memcpy: a byte-buffer's alignment is not an f32/i32 contract.
    inference_cycle_ready_owned result;
    result.tokens = h.tokens;
    result.hidden = h.hidden;
    result.topk = h.topk;
    const size_t n_activation = size_t(h.tokens) * h.hidden;
    const size_t n_routing = size_t(h.tokens) * h.topk;
    result.residual.resize(n_activation);
    result.ffn_norm.resize(n_activation);
    result.topk_ids.resize(n_routing);
    result.topk_weights.resize(n_routing);

    size_t offset = sizeof(h);
    auto read = [&](void * dst, size_t n) {
        if (n != 0) {
            std::memcpy(dst, src + offset, n);
        }
        offset += n;
    };
    read(result.residual.data(), n_activation * sizeof(float));
    read(result.ffn_norm.data(), n_activation * sizeof(float));
    read(result.topk_ids.data(), n_routing * sizeof(int32_t));
    read(result.topk_weights.data(), n_routing * sizeof(float));
    output = std::move(result);
    return offset == size;
}

inline bool inference_cycle_read_partial(
        const uint8_t * src, size_t size,
        inference_cycle_partial_owned & output) {
    inference_cycle_payload_header h {};
    if (!inference_cycle_parse_payload_header(src, size, INFERENCE_CYCLE_PARTIAL, h)) {
        return false;
    }
    inference_cycle_partial_owned result;
    result.tokens = h.tokens;
    result.hidden = h.hidden;
    result.partial.resize(size_t(h.tokens) * h.hidden);
    std::memcpy(result.partial.data(), src + sizeof(h),
                result.partial.size() * sizeof(float));
    output = std::move(result);
    return true;
}
