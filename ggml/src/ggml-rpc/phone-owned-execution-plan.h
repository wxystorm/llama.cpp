#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>

// A small, backend-independent contract for the first Phone-owned execution
// phase. The PC sends one immutable descriptor per single-layer Phone Region.
// Chunk intervals are derived locally; no per-Chunk Graph pointers or KV
// tensor addresses cross the RPC boundary.
//
// This phase is intentionally a shadow contract: it must never claim that a
// local HTP graph is executable until a Phone-local graph builder exists.
namespace ggml_phone_owned {

static constexpr uint32_t k_plan_version = 1;
static constexpr uint32_t k_slots = 2;
static constexpr uint32_t k_max_regions = 64;
static constexpr uint32_t k_max_chunks = 4096;
static constexpr uint64_t k_max_activation_bytes = 128ULL * 1024 * 1024;

struct region_plan {
    uint32_t version = k_plan_version;
    uint32_t flags = 0;  // v1: dense Qwen3, F32 activation, single sequence
    uint64_t run_id = 0;
    uint64_t p2c_run_id = 0;
    uint32_t region = 0;
    uint32_t layer = 0;
    uint32_t total_tokens = 0;
    uint32_t chunk_tokens = 0;
    uint32_t n_embd = 0;
    uint32_t position_base = 0;
    uint32_t kv_capacity = 0;
    uint32_t n_heads = 0;
    uint32_t n_heads_kv = 0;
    uint32_t head_dim = 0;
};

inline bool equal(const region_plan & a, const region_plan & b) {
    return a.version == b.version && a.flags == b.flags &&
        a.run_id == b.run_id && a.p2c_run_id == b.p2c_run_id &&
        a.region == b.region && a.layer == b.layer &&
        a.total_tokens == b.total_tokens &&
        a.chunk_tokens == b.chunk_tokens && a.n_embd == b.n_embd &&
        a.position_base == b.position_base &&
        a.kv_capacity == b.kv_capacity &&
        a.n_heads == b.n_heads && a.n_heads_kv == b.n_heads_kv &&
        a.head_dim == b.head_dim;
}

inline bool validate(const region_plan & p, const char * & reason) {
    reason = "NONE";
    if (p.version != k_plan_version || p.flags != 0) {
        reason = "UNSUPPORTED_PLAN_VERSION_OR_FLAGS";
        return false;
    }
    if (!p.run_id || !p.p2c_run_id ||
            p.region >= k_max_regions || p.layer >= 256) {
        reason = "INVALID_ID_OR_REGION";
        return false;
    }
    if (!p.total_tokens || !p.chunk_tokens ||
            p.chunk_tokens > p.total_tokens ||
            p.chunk_tokens > 4096 ||
            (uint64_t(p.total_tokens) + p.chunk_tokens - 1) /
                p.chunk_tokens > k_max_chunks) {
        reason = "INVALID_CHUNK_SCHEDULE";
        return false;
    }
    if (!p.n_embd || p.n_embd > 65536 ||
            !p.n_heads || !p.n_heads_kv ||
            p.n_heads_kv > p.n_heads || !p.head_dim ||
            p.head_dim > 512 || p.n_embd % p.n_heads != 0 ||
            p.n_embd / p.n_heads != p.head_dim) {
        reason = "INVALID_QWEN3_SHAPE";
        return false;
    }
    if (uint64_t(p.position_base) + p.total_tokens >
            p.kv_capacity || !p.kv_capacity) {
        reason = "INVALID_KV_POSITION_RANGE";
        return false;
    }
    if (uint64_t(p.n_embd) * p.chunk_tokens * sizeof(float) >
            k_max_activation_bytes) {
        reason = "ACTIVATION_BUDGET_EXCEEDED";
        return false;
    }
    return true;
}

inline uint32_t chunk_count(const region_plan & p) {
    return (uint64_t(p.total_tokens) + p.chunk_tokens - 1) /
        p.chunk_tokens;
}

inline bool matches_chunk(const region_plan & p, uint32_t chunk,
                          uint32_t begin, uint32_t tokens,
                          uint64_t p2c_run_id,
                          uint64_t input_bytes, uint64_t output_bytes) {
    if (chunk >= chunk_count(p)) { return false; }
    const uint64_t expected_begin = uint64_t(chunk) * p.chunk_tokens;
    const uint32_t expected_tokens =
        std::min<uint64_t>(p.chunk_tokens, p.total_tokens - expected_begin);
    const uint64_t expected_bytes =
        uint64_t(expected_tokens) * p.n_embd * sizeof(float);
    return begin == expected_begin && tokens == expected_tokens &&
        p2c_run_id == p.p2c_run_id &&
        input_bytes == expected_bytes && output_bytes == expected_bytes;
}

class registry {
public:
    bool add(const region_plan & plan, const char * & reason) {
        if (!validate(plan, reason)) { return false; }
        const auto key = std::make_pair(plan.run_id, plan.region);
        const auto it = regions_.find(key);
        if (it != regions_.end()) {
            if (equal(it->second, plan)) { return true; }
            reason = "CONFLICTING_REGION_PLAN";
            return false;
        }
        regions_.emplace(key, plan);
        return true;
    }

    const region_plan * find(uint64_t run_id, uint32_t region) const {
        const auto it = regions_.find(std::make_pair(run_id, region));
        return it == regions_.end() ? nullptr : &it->second;
    }

    void erase_run(uint64_t run_id) {
        for (auto it = regions_.begin(); it != regions_.end();) {
            if (it->first.first == run_id) {
                it = regions_.erase(it);
            } else {
                ++it;
            }
        }
    }

    size_t size() const { return regions_.size(); }

private:
    std::map<std::pair<uint64_t, uint32_t>, region_plan> regions_;
};

} // namespace ggml_phone_owned
