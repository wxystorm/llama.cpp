#pragma once

#include "phone-owned-execution-plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Model-independent, conservative admission policy for the Phone-owned
// fixed-UID HTP graph cache. The Hexagon backend currently only offers a
// session-wide cache flush; removing one entry's live tensor storage could
// leave other compiled graph nodes holding dangling addresses. Consequently,
// capacity/budget rollover evicts the *whole generation* before new graphs
// are built, never one live template in isolation.
namespace ggml_phone_owned {

inline uint64_t estimate_resident_bytes(
        const region_plan & cached, uint64_t cached_bytes,
        const region_plan & requested) {
    if (cached_bytes == 0 || cached.chunk_tokens == 0 ||
            cached.n_kv == 0 || chunk_count(cached) == 0) {
        return UINT64_MAX;
    }
    // Graph storage is approximately proportional to the Chunk count,
    // active Chunk width and fixed KV read span. Overestimate when the
    // requested geometry grows, with a 25% headroom for allocator padding.
    const double chunks = std::max(
        1.0, double(chunk_count(requested))/chunk_count(cached));
    const double width = std::max(
        1.0, double(requested.chunk_tokens)/cached.chunk_tokens);
    const double kv = std::max(
        1.0, double(requested.n_kv)/cached.n_kv);
    const double estimate = std::ceil(
        double(cached_bytes)*chunks*width*kv*1.25);
    if (!std::isfinite(estimate) ||
            estimate >= double(UINT64_MAX)) {
        return UINT64_MAX;
    }
    return static_cast<uint64_t>(estimate);
}

inline bool must_evict_generation(
        size_t standby_entries, size_t max_entries,
        uint64_t standby_bytes, uint64_t incoming_estimate_bytes,
        uint64_t budget_bytes) {
    if (standby_entries == 0) { return false; }
    if (max_entries == 0 || standby_entries >= max_entries ||
            standby_bytes >= budget_bytes ||
            incoming_estimate_bytes > budget_bytes - standby_bytes) {
        return true;
    }
    return false;
}

} // namespace ggml_phone_owned
