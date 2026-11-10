#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

// Immutable, backend-independent Phone-owned execution descriptor.
// No model weight, Graph, Tensor or KV pointer is transferred from the PC.
// v2 carries all scalars needed to reproduce a single Qwen3 Layer's math.
// Runtime cutover is deliberately separate from this plan/Graph shadow path.
namespace ggml_phone_owned {

static constexpr uint32_t k_plan_version = 2;
static constexpr uint32_t k_multi_plan_version = 3;
// Fresh prefill: HTP owns all Phone KV and immutable weights. There is no
// canonical RPC/OpenCL Decode KV binding or PC-built Phone graph.
static constexpr uint32_t k_htp_only_plan_version = 4;
static constexpr uint32_t k_layer_count_shift = 16;
static constexpr uint32_t k_max_phone_layers_per_region = 64;
static constexpr uint32_t k_wire_region_limit = 64;
static constexpr uint32_t k_internal_region_base = k_wire_region_limit;
static constexpr uint32_t k_flag_flash_attn = 1;
static constexpr uint32_t k_rope_neox = 2; // GGML_ROPE_TYPE_NEOX
static constexpr uint32_t k_f16 = 1;      // GGML_TYPE_F16
static constexpr uint32_t k_slots = 2;
static constexpr uint32_t k_max_regions =
    k_internal_region_base +
    k_wire_region_limit * k_max_phone_layers_per_region;
static constexpr uint32_t k_max_chunks = 4096;
static constexpr uint64_t k_max_activation_bytes = 128ULL * 1024 * 1024;

struct region_plan {
    uint32_t version = k_plan_version;
    uint32_t flags = k_flag_flash_attn;
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
    // The mask is generated for the fixed padded KV read span. v2 supports
    // fresh single-sequence prefill only; no prefix reuse/KV shift yet.
    uint32_t n_kv = 0;
    uint32_t rope_type = k_rope_neox;
    uint32_t n_ctx_orig = 0;
    uint32_t kv_type = k_f16;
    float rms_norm_eps = 0.0f;
    float rope_freq_base = 0.0f;
    float rope_freq_scale = 0.0f;
    float rope_ext_factor = 0.0f;
    float rope_attn_factor = 0.0f;
    float rope_beta_fast = 0.0f;
    float rope_beta_slow = 0.0f;
};

// Plan v3 encodes the consecutive NPU layer count in the upper 16
// bits of flags, keeping the 108-byte RPC descriptor ABI unchanged.
// The lower bits retain the exact v2 Flash-Attention contract.
inline uint32_t layer_count(const region_plan & p) {
    return p.version == k_multi_plan_version ||
            p.version == k_htp_only_plan_version ?
        p.flags >> k_layer_count_shift : 1;
}
inline uint32_t flags_for_layers(uint32_t n_layers) {
    return k_flag_flash_attn |
        (n_layers << k_layer_count_shift);
}
inline uint32_t internal_layer_region(
        const region_plan & parent, uint32_t offset) {
    return layer_count(parent) == 1 ? parent.region :
        k_internal_region_base +
        parent.region * k_max_phone_layers_per_region + offset;
}
inline region_plan single_layer_plan(
        const region_plan & parent, uint32_t offset) {
    region_plan child = parent;
    child.version = k_plan_version;
    child.flags = k_flag_flash_attn;
    child.region = internal_layer_region(parent, offset);
    child.layer = parent.layer + offset;
    return child;
}

inline bool equal(const region_plan & a, const region_plan & b) {
    return a.version == b.version && a.flags == b.flags &&
        a.run_id == b.run_id && a.p2c_run_id == b.p2c_run_id &&
        a.region == b.region && a.layer == b.layer &&
        a.total_tokens == b.total_tokens &&
        a.chunk_tokens == b.chunk_tokens && a.n_embd == b.n_embd &&
        a.position_base == b.position_base &&
        a.kv_capacity == b.kv_capacity &&
        a.n_heads == b.n_heads && a.n_heads_kv == b.n_heads_kv &&
        a.head_dim == b.head_dim && a.n_kv == b.n_kv &&
        a.rope_type == b.rope_type && a.n_ctx_orig == b.n_ctx_orig &&
        a.kv_type == b.kv_type &&
        a.rms_norm_eps == b.rms_norm_eps &&
        a.rope_freq_base == b.rope_freq_base &&
        a.rope_freq_scale == b.rope_freq_scale &&
        a.rope_ext_factor == b.rope_ext_factor &&
        a.rope_attn_factor == b.rope_attn_factor &&
        a.rope_beta_fast == b.rope_beta_fast &&
        a.rope_beta_slow == b.rope_beta_slow;
}

inline bool validate(const region_plan & p, const char * & reason) {
    reason = "NONE";
    const uint32_t layers = layer_count(p);
    const bool legacy = p.version == k_plan_version &&
        p.flags == k_flag_flash_attn;
    const bool multi = p.version == k_multi_plan_version &&
        (p.flags & 0xffffu) == k_flag_flash_attn &&
        layers >= 2 && layers <= k_max_phone_layers_per_region;
    // Protocol version 4 is an explicit fail-closed capability contract.
    // It preserves the wire ABI and supports both P1 and consecutive Pn.
    const bool htp_only = p.version == k_htp_only_plan_version &&
        (p.flags & 0xffffu) == k_flag_flash_attn &&
        layers >= 1 && layers <= k_max_phone_layers_per_region;
    if ((!legacy && !multi && !htp_only) ||
            p.layer >= 256 || layers > 256 - p.layer ||
            p.rope_type != k_rope_neox ||
            p.kv_type != k_f16) {
        reason = "UNSUPPORTED_QWEN3_PLAN_OR_FORMAT";
        return false;
    }
    if (!p.run_id || !p.p2c_run_id ||
            p.region >= k_max_regions) {
        reason = "INVALID_ID_OR_REGION";
        return false;
    }
    if (!p.total_tokens || !p.chunk_tokens ||
            p.chunk_tokens > p.total_tokens || p.chunk_tokens > 4096 ||
            (uint64_t(p.total_tokens) + p.chunk_tokens - 1) /
                p.chunk_tokens > k_max_chunks) {
        reason = "INVALID_CHUNK_SCHEDULE";
        return false;
    }
    if (!p.n_embd || p.n_embd > 65536 ||
            !p.n_heads || !p.n_heads_kv ||
            p.n_heads_kv > p.n_heads ||
            p.n_heads % p.n_heads_kv != 0 || !p.head_dim ||
            p.head_dim > 512 || (p.head_dim & 1) != 0 ||
            // Qwen3-32B: 64*128 Q features != 5120 hidden features.
            // The output projection maps Q width back into hidden size.
            uint64_t(p.n_heads) * p.head_dim > 65536) {
        reason = "INVALID_QWEN3_SHAPE";
        return false;
    }
    if (p.position_base != 0 || !p.kv_capacity ||
            uint64_t(p.position_base) + p.total_tokens > p.kv_capacity ||
            p.n_kv < p.total_tokens || p.n_kv > p.kv_capacity ||
            (p.n_kv % 256 != 0 && p.n_kv != p.kv_capacity)) {
        reason = "UNSUPPORTED_KV_POSITION_OR_READ_SPAN";
        return false;
    }
    if (uint64_t(p.n_embd) * p.chunk_tokens * sizeof(float) >
            k_max_activation_bytes) {
        reason = "ACTIVATION_BUDGET_EXCEEDED";
        return false;
    }
    if (!p.n_ctx_orig ||
            !std::isfinite(p.rms_norm_eps) || p.rms_norm_eps <= 0 ||
            !std::isfinite(p.rope_freq_base) || p.rope_freq_base <= 0 ||
            !std::isfinite(p.rope_freq_scale) || p.rope_freq_scale <= 0 ||
            !std::isfinite(p.rope_ext_factor) ||
            !std::isfinite(p.rope_attn_factor) ||
            !std::isfinite(p.rope_beta_fast) ||
            !std::isfinite(p.rope_beta_slow)) {
        reason = "INVALID_NUMERIC_GRAPH_PARAMS";
        return false;
    }
    return true;
}

inline uint32_t chunk_count(const region_plan & p) {
    return (uint64_t(p.total_tokens) + p.chunk_tokens - 1) /
        p.chunk_tokens;
}

// Guard the fresh 0-based KV cache contract before HTP-only compute.
// RoPE positions are not sufficient to prove a physical Decode KV slot
// layout: the PC allocator can recycle cells. Require the actual I64
// indices supplied to the original SET_ROWS nodes to be contiguous.
inline bool identity_kv_slots(const int64_t * indices,
                             size_t count, uint64_t first) {
    if (indices == nullptr || count == 0 ||
            count > uint64_t(INT64_MAX) ||
            first > uint64_t(INT64_MAX) - count) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (indices[i] != static_cast<int64_t>(first + i)) {
            return false;
        }
    }
    return true;
}

// The verified PUSH payload is F32 activation data. A host-direct handoff
// must preserve the byte-exact Chunk contract before HTP upload.
inline bool valid_direct_host_bytes(uint64_t expected, size_t actual) {
    return expected != 0 &&
           expected <= k_max_activation_bytes &&
           expected % sizeof(float) == 0 &&
           expected == actual;
}

// Reserved UID namespace: Phone-owned tasks do not represent PC Graphs.
inline uint64_t owned_task_uid(uint32_t region, uint32_t chunk) {
    return (uint64_t(2) << 62) |
           (uint64_t(region) << 32) | (uint64_t(chunk) + 1);
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

    size_t count_run(uint64_t run_id) const {
        size_t count = 0;
        for (const auto & entry : regions_) {
            if (entry.first.first == run_id) { ++count; }
        }
        return count;
    }

private:
    std::map<std::pair<uint64_t, uint32_t>, region_plan> regions_;
};

} // namespace ggml_phone_owned
