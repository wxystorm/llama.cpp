#pragma once

#include <cstdint>

namespace ggml_phone_auto {

// The next expected Chunk can be prepared, as can one following Chunk.
// Later Chunks stay in the bounded Host queue. Keep the subtraction safe if
// a session unexpectedly reaches UINT32_MAX.
constexpr bool can_prepare(uint32_t next_kv_chunk, uint32_t chunk) {
    return chunk >= next_kv_chunk && chunk - next_kv_chunk <= 1;
}

// Attention consumes KV in sequence. Never submit Chunk N+1 before the
// compute and KV writes for Chunk N have completed.
constexpr bool can_compute(uint32_t next_kv_chunk, uint32_t chunk) {
    return next_kv_chunk == chunk;
}

} // namespace ggml_phone_auto
