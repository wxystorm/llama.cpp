#pragma once

#include <cstdint>
#include <cstddef>
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

// The native Phone backend holds actual OpenCL buffer/data addresses.
// The generic RPC client serializer blanks these fields for non-RPC buffers;
// they must be restored before server-local APPLY or overlap validation.
inline bool bind_native_storage(uint64_t & wire_buffer, uint64_t & wire_data,
                                const void * buffer, const void * data) {
    if (buffer == nullptr || data == nullptr) { return false; }
    wire_buffer = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(buffer));
    wire_data = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data));
    return true;
}

// Fail-closed capture admission: only the exact registered, single F32
// contiguous output of the same Chunk may be exposed to the P2C receiver.
constexpr bool valid_capture_contract(uint32_t registered_tokens,
        uint32_t requested_tokens, size_t outputs, uint64_t output_bytes,
        uint64_t expected_bytes, bool f32, bool contiguous) {
    return registered_tokens > 0 &&
        registered_tokens == requested_tokens && outputs == 1 &&
        output_bytes > 0 && output_bytes == expected_bytes &&
        f32 && contiguous;
}

} // namespace ggml_phone_auto
