#pragma once

#include <cstdint>
#include <cstring>

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

// Meta names its RPC split inputs (e.g. Meta(CPU,RPC0)#inp_stage#0).
// Accept the original leaf or that exact namespaced leaf, not shape-only
// guesses, partial suffixes, or similarly named unrelated inputs.
inline bool is_stage_input_name(const char * name) {
    if (name == nullptr) { return false; }
    if (std::strcmp(name, "inp_stage") == 0) { return true; }
    if (std::strncmp(name, "Meta(", 5) != 0) { return false; }
    const char * close = std::strchr(name + 5, ')');
    return close != nullptr && close > name + 5 &&
        std::strcmp(close + 1, "#inp_stage#0") == 0;
}

} // namespace ggml_phone_auto
