#pragma once

#include <cstddef>
#include <utility>
#include <vector>

// Phone-owned NPU->Adreno activation handoff. Both stages consume exactly
// the same Host F32 allocation; only the later Host->OpenCL device upload
// is required. This helper intentionally never hashes or copies the payload.
namespace ggml_phone_local_host {

inline bool move_f32(std::vector<float> & source,
                     std::vector<float> & destination,
                     std::size_t expected_bytes) {
    if (expected_bytes == 0 ||
            expected_bytes % sizeof(float) != 0 ||
            source.size() != expected_bytes / sizeof(float) ||
            !destination.empty()) {
        return false;
    }
    const float * const original = source.data();
    destination = std::move(source);
    return destination.size() == expected_bytes / sizeof(float) &&
        destination.data() == original;
}

} // namespace ggml_phone_local_host
