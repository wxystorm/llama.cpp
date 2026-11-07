#pragma once

// Pure host-side checks for a Planner-selected, immutable Phone HTP map.
// The identical invariants are required on both RPC Client and Phone.
#include <cstdint>
#include <utility>
#include <vector>

namespace ggml_phone_plan {

using region_layer = std::pair<uint32_t, uint32_t>;

inline bool valid_htp_map(const std::vector<region_layer> & order) {
    if (order.size() < 2 || order.size() > 64) {
        return false; // Physical two-Bank HTP path.
    }
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i].first >= 64 || order[i].second >= 256 ||
                (i != 0 &&
                 (order[i-1].first >= order[i].first ||
                  order[i-1].second >= order[i].second))) {
            return false;
        }
    }
    return true;
}

} // namespace ggml_phone_plan
