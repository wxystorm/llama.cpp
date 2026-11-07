#pragma once

// Pure host-side checks for a Planner-selected, immutable Phone HTP map.
// The identical invariants are required on both RPC Client and Phone.
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ggml_phone_plan {

using region_layer = std::pair<uint32_t, uint32_t>;

inline bool valid_htp_map(const std::vector<region_layer> & order) {
    if (order.size() < 2 || order.size() > 64) {
        return false; // Physical two-Bank HTP path.
    }
    for (std::size_t i = 0; i < order.size(); ++i) {
        if (order[i].first >= 64 || order[i].second >= 256 ||
                (i != 0 &&
                 (order[i-1].first >= order[i].first ||
                  order[i-1].second >= order[i].second))) {
            return false;
        }
    }
    return true;
}

// Zero means ordinary Phone->PC boundary. Nonzero successors must be the
// *immediately adjacent* OpenCL Region; no cross-Cycle or arbitrary GPU
// mapping is silently inferred. Both endpoints use the same Region ID space.
inline bool valid_local_successors(
        const std::vector<region_layer> & order,
        const std::vector<uint32_t> & successor) {
    if (!valid_htp_map(order) || successor.size() != order.size()) {
        return false;
    }
    for (std::size_t i = 0; i < order.size(); ++i) {
        const uint32_t src = order[i].first;
        const uint32_t dst = successor[i];
        if (dst == 0) { continue; }
        if (src >= 63 || dst != src + 1) { return false; }
        for (const auto & entry : order) {
            if (entry.first == dst) { return false; }
        }
    }
    return true;
}

} // namespace ggml_phone_plan
