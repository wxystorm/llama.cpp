#pragma once

// Host-only scheduling/measurement policy for independent Phone HTP and
// OpenCL compute lanes. No RPC, OpenCL, Hexagon or GGUF dependency.
#include <algorithm>
#include <cstdint>
#include <vector>

namespace ggml_phone_mixed_pipeline {

inline bool lane_accepts(bool dual_lane, bool worker_is_htp,
                         bool task_is_htp) {
    // The default (one worker) preserves the existing serialized execution.
    // Dual lane is strict: each worker owns only its physical compute unit.
    return !dual_lane || worker_is_htp == task_is_htp;
}

struct interval {
    int64_t begin_us = 0;
    int64_t end_us = 0;
};

// Actual HTP/OpenCL overlap in the *same Phone monotonic clock*. Jobs of
// each lane cannot overlap one another, so interval intersections are not
// double counted. Negative/empty intervals are ignored.
inline uint64_t overlap_us(
        std::vector<interval> htp,
        std::vector<interval> opencl) {
    const auto valid = [](const interval & i) {
        return i.begin_us > 0 && i.end_us > i.begin_us;
    };
    htp.erase(std::remove_if(htp.begin(), htp.end(),
                            [&](const interval & i) { return !valid(i); }),
              htp.end());
    opencl.erase(std::remove_if(opencl.begin(), opencl.end(),
                               [&](const interval & i) { return !valid(i); }),
                 opencl.end());
    const auto by_begin = [](const interval & a, const interval & b) {
        return a.begin_us < b.begin_us;
    };
    std::sort(htp.begin(), htp.end(), by_begin);
    std::sort(opencl.begin(), opencl.end(), by_begin);
    size_t i = 0, j = 0;
    uint64_t total = 0;
    while (i < htp.size() && j < opencl.size()) {
        const int64_t lo = std::max(htp[i].begin_us, opencl[j].begin_us);
        const int64_t hi = std::min(htp[i].end_us, opencl[j].end_us);
        if (hi > lo) {
            total += static_cast<uint64_t>(hi - lo);
        }
        if (htp[i].end_us <= opencl[j].end_us) {
            if (htp[i].end_us == opencl[j].end_us) { ++j; }
            ++i;
        } else {
            ++j;
        }
    }
    return total;
}

} // namespace ggml_phone_mixed_pipeline
