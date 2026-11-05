#pragma once

// Qwen3 single-sequence prefill plan v1. This file intentionally depends on
// no GGML or llama symbols so the PC planner, Android RPC server and tests
// validate exactly the same wire contract.
//
// A PLAN_ACCEPTED response is NOT a GRAPH_READY response: execution must
// remain on the validated legacy graph until an independent Phone graph
// compiler has produced and verified every fixed Chunk graph.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace ggml_phone_owned_qwen3 {

static constexpr uint32_t magic = 0x46503351u;  // "Q3PF", little endian
static constexpr uint32_t version = 1;
static constexpr uint32_t arch_qwen3_dense = 1;
static constexpr uint32_t max_tokens = 65536;
static constexpr uint32_t max_chunks = 1024;
static constexpr uint32_t max_regions = 64;
static constexpr size_t max_wire_bytes = 512 * 1024;

struct chunk {
    uint32_t region = 0;
    uint32_t layer = 0;
    uint32_t index = 0;
    uint32_t token_begin = 0;
    uint32_t n_tokens = 0;
};

struct plan {
    uint64_t run_id = 0;
    uint32_t arch = arch_qwen3_dense;
    uint32_t n_ctx = 0;
    uint32_t n_tokens = 0;
    uint32_t n_embd = 0;
    uint32_t n_layer = 0;
    uint32_t n_head = 0;
    uint32_t n_head_kv = 0;
    uint32_t head_dim = 0;
    uint32_t n_ff = 0;
    uint32_t seq_id = 0;
    uint32_t rope_type = 0;
    float rms_eps = 0.0f;
    float rope_base = 0.0f;
    float rope_scale = 0.0f;
    std::vector<int32_t> positions;
    std::vector<chunk> chunks;
};

inline bool reject(std::string & why, const char * message) {
    why = message;
    return false;
}

// Every Region owns its own complete token interval partition. Thus plans
// may mix sizes (16/32/64/128/...) and have arbitrary positive tail lengths.
inline bool validate(const plan & p, std::string & why) {
    why.clear();
    if (p.arch != arch_qwen3_dense ||
            p.run_id == 0 || p.n_ctx == 0 || p.n_ctx > (1u << 20) ||
            p.n_tokens == 0 || p.n_tokens > max_tokens ||
            p.n_tokens > p.n_ctx ||
            p.n_embd == 0 || p.n_embd > (1u << 16) ||
            p.n_layer == 0 || p.n_layer > 512 ||
            p.n_head == 0 || p.n_head > 512 ||
            p.n_head_kv == 0 || p.n_head_kv > p.n_head ||
            p.n_head % p.n_head_kv != 0 ||
            p.head_dim == 0 || p.head_dim > 512 ||
            p.n_ff == 0 || p.n_ff > (1u << 18) ||
            uint64_t(p.n_head) * p.head_dim != p.n_embd ||
            !std::isfinite(p.rms_eps) || p.rms_eps <= 0.0f ||
            !std::isfinite(p.rope_base) || p.rope_base <= 0.0f ||
            !std::isfinite(p.rope_scale) || p.rope_scale <= 0.0f) {
        return reject(why, "INVALID_QWEN3_PREFILL_HEADER");
    }
    if (p.positions.size() != p.n_tokens) {
        return reject(why, "POSITION_COUNT_MISMATCH");
    }
    // The first local-builder implementation is single-sequence causal
    // prefill. Non-consecutive position maps require explicit KV remapping.
    const int64_t base = p.positions.front();
    if (base < 0 || base + p.n_tokens > p.n_ctx) {
        return reject(why, "POSITION_OUT_OF_RANGE");
    }
    for (size_t i = 0; i < p.positions.size(); ++i) {
        if (p.positions[i] != base + int64_t(i)) {
            return reject(why, "NONCONTIGUOUS_PREFILL_POSITIONS");
        }
    }
    if (p.chunks.empty() || p.chunks.size() > max_chunks) {
        return reject(why, "INVALID_CHUNK_COUNT");
    }
    std::set<uint32_t> layers;
    std::set<uint32_t> regions;
    size_t i = 0;
    uint32_t previous_region = 0;
    bool first_region = true;
    while (i < p.chunks.size()) {
        const chunk & head = p.chunks[i];
        if (head.region == 0 || head.region >= max_regions ||
                head.layer >= p.n_layer ||
                (!first_region && head.region <= previous_region) ||
                !layers.insert(head.layer).second ||
                !regions.insert(head.region).second ||
                regions.size() > max_regions) {
            return reject(why, "INVALID_REGION_OR_LAYER");
        }
        uint32_t expected_begin = 0;
        uint32_t expected_index = 0;
        const uint32_t region = head.region;
        const uint32_t layer = head.layer;
        while (i < p.chunks.size() && p.chunks[i].region == region) {
            const chunk & c = p.chunks[i++];
            if (c.layer != layer || c.index != expected_index ||
                    c.n_tokens == 0 || c.n_tokens > p.n_tokens ||
                    c.token_begin != expected_begin ||
                    c.n_tokens > p.n_tokens - expected_begin ||
                    uint64_t(c.n_tokens) * p.n_embd * sizeof(float) >
                        64ull * 1024 * 1024) {
                return reject(why, "INVALID_CHUNK_PARTITION");
            }
            expected_begin += c.n_tokens;
            ++expected_index;
        }
        if (expected_begin != p.n_tokens) {
            return reject(why, "INCOMPLETE_REGION_PARTITION");
        }
        previous_region = region;
        first_region = false;
    }
    return true;
}

inline void append_u32(std::vector<uint8_t> & out, uint32_t n) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(uint8_t(n >> (i * 8)));
    }
}

inline void append_u64(std::vector<uint8_t> & out, uint64_t n) {
    append_u32(out, uint32_t(n));
    append_u32(out, uint32_t(n >> 32));
}

inline uint32_t float_bits(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

inline float bits_float(uint32_t bits) {
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline bool read_u32(const uint8_t * data, size_t size,
                     size_t & offset, uint32_t & out) {
    if (size - offset < 4) { return false; }
    out = 0;
    for (int i = 0; i < 4; ++i) {
        out |= uint32_t(data[offset++]) << (i * 8);
    }
    return true;
}

inline bool read_u64(const uint8_t * data, size_t size,
                     size_t & offset, uint64_t & out) {
    uint32_t lo = 0, hi = 0;
    if (!read_u32(data, size, offset, lo) ||
            !read_u32(data, size, offset, hi)) {
        return false;
    }
    out = uint64_t(lo) | (uint64_t(hi) << 32);
    return true;
}

inline bool encode(const plan & p, std::vector<uint8_t> & out,
                   std::string & why) {
    out.clear();
    if (!validate(p, why)) { return false; }
    // 20 header dwords (including the 64-bit run_id) + variable arrays.
    const size_t size = 80 + 4 * p.positions.size() +
        20 * p.chunks.size();
    if (size > max_wire_bytes) {
        return reject(why, "PLAN_TOO_LARGE");
    }
    out.reserve(size);
    append_u32(out, magic);
    append_u32(out, version);
    append_u64(out, p.run_id);
    append_u32(out, p.arch);
    append_u32(out, p.n_ctx);
    append_u32(out, p.n_tokens);
    append_u32(out, p.n_embd);
    append_u32(out, p.n_layer);
    append_u32(out, p.n_head);
    append_u32(out, p.n_head_kv);
    append_u32(out, p.head_dim);
    append_u32(out, p.n_ff);
    append_u32(out, p.seq_id);
    append_u32(out, p.rope_type);
    append_u32(out, float_bits(p.rms_eps));
    append_u32(out, float_bits(p.rope_base));
    append_u32(out, float_bits(p.rope_scale));
    append_u32(out, uint32_t(p.positions.size()));
    append_u32(out, uint32_t(p.chunks.size()));
    for (int32_t pos : p.positions) {
        append_u32(out, uint32_t(pos));
    }
    for (const chunk & c : p.chunks) {
        append_u32(out, c.region);
        append_u32(out, c.layer);
        append_u32(out, c.index);
        append_u32(out, c.token_begin);
        append_u32(out, c.n_tokens);
    }
    return true;
}

inline bool decode(const uint8_t * data, size_t size,
                   plan & p, std::string & why) {
    p = {};
    why.clear();
    if (data == nullptr || size < 80 || size > max_wire_bytes) {
        return reject(why, "INVALID_PLAN_SIZE");
    }
    size_t offset = 0;
    uint32_t got_magic = 0, got_version = 0;
    uint32_t eps = 0, base = 0, scale = 0, positions = 0, chunks = 0;
    if (!read_u32(data, size, offset, got_magic) ||
            !read_u32(data, size, offset, got_version) ||
            !read_u64(data, size, offset, p.run_id) ||
            !read_u32(data, size, offset, p.arch) ||
            !read_u32(data, size, offset, p.n_ctx) ||
            !read_u32(data, size, offset, p.n_tokens) ||
            !read_u32(data, size, offset, p.n_embd) ||
            !read_u32(data, size, offset, p.n_layer) ||
            !read_u32(data, size, offset, p.n_head) ||
            !read_u32(data, size, offset, p.n_head_kv) ||
            !read_u32(data, size, offset, p.head_dim) ||
            !read_u32(data, size, offset, p.n_ff) ||
            !read_u32(data, size, offset, p.seq_id) ||
            !read_u32(data, size, offset, p.rope_type) ||
            !read_u32(data, size, offset, eps) ||
            !read_u32(data, size, offset, base) ||
            !read_u32(data, size, offset, scale) ||
            !read_u32(data, size, offset, positions) ||
            !read_u32(data, size, offset, chunks)) {
        return reject(why, "TRUNCATED_PLAN_HEADER");
    }
    if (got_magic != magic || got_version != version) {
        return reject(why, "UNSUPPORTED_PLAN_VERSION");
    }
    if (positions == 0 || positions > max_tokens ||
            chunks == 0 || chunks > max_chunks ||
            size != offset + 4ull * positions + 20ull * chunks) {
        return reject(why, "PLAN_PAYLOAD_LENGTH_MISMATCH");
    }
    p.rms_eps = bits_float(eps);
    p.rope_base = bits_float(base);
    p.rope_scale = bits_float(scale);
    p.positions.reserve(positions);
    p.chunks.reserve(chunks);
    for (uint32_t i = 0; i < positions; ++i) {
        uint32_t value = 0;
        if (!read_u32(data, size, offset, value)) {
            return reject(why, "TRUNCATED_POSITIONS");
        }
        p.positions.push_back(int32_t(value));
    }
    for (uint32_t i = 0; i < chunks; ++i) {
        chunk c {};
        if (!read_u32(data, size, offset, c.region) ||
                !read_u32(data, size, offset, c.layer) ||
                !read_u32(data, size, offset, c.index) ||
                !read_u32(data, size, offset, c.token_begin) ||
                !read_u32(data, size, offset, c.n_tokens)) {
            return reject(why, "TRUNCATED_CHUNKS");
        }
        p.chunks.push_back(c);
    }
    return validate(p, why);
}

}  // namespace ggml_phone_owned_qwen3
