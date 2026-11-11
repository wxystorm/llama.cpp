#pragma once

// Read-only capacity check for Phone-owned Qwen3 Graphless HTP plans.
// Mirror the RPC server's per-REGION starting session followed by per-LAYER
// round-robin assignment. The four FastRPC sessions have separate virtual
// mappings but draw from the SAME physical Phone RAM.
#include "llama-region-ring.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace llama_ring_planner {

static constexpr long double HTP_MEMORY_MIB = 1048576.0L;

struct htp_memory_limits {
    // opt_vmem = 3355443200 bytes (3200 MiB) in the observed HTP0..HTP3
    // driver logs. Keep a 10% VA/queue margin; this is NOT four RAM pools.
    long double session_budget_bytes = 2880.0L * HTP_MEMORY_MIB;
    // The RPC-visible CPU transport must report Android MemAvailable, not
    // MemTotal. The caller applies a physical-RAM safety fraction once.
    long double shared_budget_bytes = 0.0L;
    // Quantized HTP tensors are allocated in the repack buft, not simply
    // mmap'd from GGUF. This is deliberately an upper-bound multiplier.
    long double weight_allocation_factor = 1.15L;
    // Per-session HTP queue, graph metadata, scratch mapping headroom.
    long double per_session_reserve_bytes = 64.0L * HTP_MEMORY_MIB;
    // Host-side RPC activations and FastRPC staging share Phone RAM too.
    long double shared_host_reserve_bytes = 256.0L * HTP_MEMORY_MIB;
    int session_count = 4;
    int tokens = 0;
    int kv_start = 0;
    int xp = 0;
    int n_embd = 0;
    int n_ff = 0;
    long double kv_bytes_per_token_per_layer = 0.0L;
};

enum class htp_memory_reject {
    NONE, INVALID, SESSION_MAPPING, SHARED_RAM,
};

struct htp_memory_usage {
    std::array<long double, 4> session_weights_bytes {};
    std::array<long double, 4> session_kv_bytes {};
    std::array<long double, 4> session_scratch_bytes {};
    std::array<long double, 4> session_total_bytes {};
    std::array<int, 4> session_layers {};
    long double scratch_bytes_per_layer = 0.0L;
    long double shared_total_bytes = 0.0L;
    int phone_layers = 0;
    htp_memory_reject reject = htp_memory_reject::INVALID;
};

// A single Qwen3 HTP layer creates separate persistent KV, immutable repacked
// weights and ONE scratch arena per layer (NOT per Region). Graphless reuses
// this scratch for all Chunks of the layer, but different layers/Regions must
// keep their arenas simultaneously. F32 intermediates scale with the wider
// of the FFN and attention activation shapes. The formula is a conservative
// architecture-based bound, not a claim of exact allocator byte parity:
// (8*n_ff + 4*n_embd)*sizeof(float)*max_chunk + 1 MiB per layer.
// Observed 32B scratch at chunk=167: 121.195 MiB; this bound is ~144 MiB.
inline long double htp_graphless_layer_scratch_bytes(
        int n_embd, int n_ff, int chunk_tokens) {
    if (n_embd <= 0 || n_ff <= 0 || chunk_tokens <= 0) {
        return 0.0L;
    }
    const long double f32_elements = 8.0L * n_ff + 4.0L * n_embd;
    return f32_elements * sizeof(float) * chunk_tokens + HTP_MEMORY_MIB;
}

template <typename LayerBytesFn>
inline bool htp_plan_memory_feasible(
        const std::vector<llama_region_ring_region> & regions,
        const htp_memory_limits & limits,
        LayerBytesFn layer_bytes,
        htp_memory_usage * measured = nullptr) {
    htp_memory_usage usage;
    const auto finish = [&](htp_memory_reject why) {
        usage.reject = why;
        if (measured) { *measured = usage; }
        return why == htp_memory_reject::NONE;
    };
    if (limits.session_count != 4 || limits.tokens <= 0 ||
            limits.kv_start < 0 || limits.xp <= 0 ||
            !std::isfinite(limits.session_budget_bytes) ||
            !(limits.session_budget_bytes > 0.0L) ||
            !std::isfinite(limits.shared_budget_bytes) ||
            !(limits.shared_budget_bytes > 0.0L) ||
            !std::isfinite(limits.weight_allocation_factor) ||
            limits.weight_allocation_factor < 1.0L ||
            !std::isfinite(limits.kv_bytes_per_token_per_layer) ||
            limits.kv_bytes_per_token_per_layer < 0.0L ||
            !std::isfinite(limits.per_session_reserve_bytes) ||
            limits.per_session_reserve_bytes < 0.0L ||
            !std::isfinite(limits.shared_host_reserve_bytes) ||
            limits.shared_host_reserve_bytes < 0.0L) {
        return finish(htp_memory_reject::INVALID);
    }
    // For one final Chunk the actual size can equal all remaining tokens;
    // the coalescer can increase Xp by up to 25% for a tiny tail.
    const int effective_xp = llama_region_ring_coalesced_chunk_width(
        limits.tokens, limits.xp);
    const int max_chunk = std::min(limits.tokens,
                                   std::max(limits.xp, effective_xp));
    usage.scratch_bytes_per_layer = htp_graphless_layer_scratch_bytes(
        limits.n_embd, limits.n_ff, max_chunk);
    if (!(usage.scratch_bytes_per_layer > 0.0L)) {
        return finish(htp_memory_reject::INVALID);
    }
    // Real Phone HTP KV is F16 and padded to 256 token boundaries. The
    // registered max context often has a 512-token floor for short prompts.
    const long double padded_kv = std::max(
        512.0L, std::ceil(((long double) limits.kv_start + limits.tokens) /
                         256.0L) * 256.0L);
    const long double kv_per_layer = padded_kv *
                                    limits.kv_bytes_per_token_per_layer;
    if (!std::isfinite(kv_per_layer)) {
        return finish(htp_memory_reject::INVALID);
    }
    size_t phone_region_rank = 0;
    for (const auto & region : regions) {
        if (region.device != llama_region_ring_device::PHONE) { continue; }
        if (region.layer_begin < 0 || region.layer_end <= region.layer_begin) {
            return finish(htp_memory_reject::INVALID);
        }
        const size_t first_session = phone_region_rank++ % 4;
        for (int l = region.layer_begin; l < region.layer_end; ++l) {
            const size_t session =
                (first_session + (size_t) (l - region.layer_begin)) % 4;
            const size_t raw_weight_bytes = layer_bytes(l);
            if (raw_weight_bytes == 0) {
                return finish(htp_memory_reject::INVALID);
            }
            usage.session_layers[session] += 1;
            usage.phone_layers += 1;
            usage.session_weights_bytes[session] +=
                (long double) raw_weight_bytes *
                limits.weight_allocation_factor;
            usage.session_kv_bytes[session] += kv_per_layer;
            usage.session_scratch_bytes[session] +=
                usage.scratch_bytes_per_layer;
        }
    }
    if (usage.phone_layers == 0) {
        usage.shared_total_bytes = 0.0L;
        return finish(htp_memory_reject::NONE);
    }
    usage.shared_total_bytes = limits.shared_host_reserve_bytes;
    for (size_t s = 0; s < 4; ++s) {
        if (usage.session_layers[s] == 0) { continue; }
        usage.session_total_bytes[s] =
            usage.session_weights_bytes[s] +
            usage.session_kv_bytes[s] +
            usage.session_scratch_bytes[s] +
            limits.per_session_reserve_bytes;
        if (!std::isfinite(usage.session_total_bytes[s]) ||
                usage.session_total_bytes[s] >
                    limits.session_budget_bytes) {
            return finish(htp_memory_reject::SESSION_MAPPING);
        }
        usage.shared_total_bytes += usage.session_total_bytes[s];
    }
    if (!std::isfinite(usage.shared_total_bytes) ||
            usage.shared_total_bytes > limits.shared_budget_bytes) {
        return finish(htp_memory_reject::SHARED_RAM);
    }
    return finish(htp_memory_reject::NONE);
}

} // namespace llama_ring_planner
