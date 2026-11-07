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

// V2 mapping: a Phone NPU Stage owns a contiguous *range* of Qwen3 layers.
// This does not change the legacy one-layer region-layer RPC ABI. A future
// versioned RPC message can transport these descriptors without ever
// interpreting N2 as two independent cross-device Regions.
struct stage_region_range {
    uint32_t region = 0;
    uint32_t layer_begin = 0;
    uint32_t layer_end = 0; // exclusive
};

// Empty is valid: a Plan may assign zero layers to HTP. One Stage is valid
// as well: the physical two-Bank mover is optional when only one rank exists.
inline bool valid_htp_stage_map(
        const std::vector<stage_region_range> & stages,
        uint32_t model_layers) {
    if (model_layers == 0 || model_layers > 256 || stages.size() > 64) {
        return false;
    }
    for (std::size_t i = 0; i < stages.size(); ++i) {
        const auto & s = stages[i];
        if (s.region >= 64 || s.layer_begin >= s.layer_end ||
                s.layer_end > model_layers) {
            return false;
        }
        if (i != 0 &&
                (stages[i - 1].region >= s.region ||
                 stages[i - 1].layer_end > s.layer_begin)) {
            return false;
        }
    }
    return true;
}

inline bool valid_htp_stage_successors(
        const std::vector<stage_region_range> & stages,
        const std::vector<uint32_t> & successor,
        uint32_t model_layers) {
    if (!valid_htp_stage_map(stages, model_layers) ||
            stages.size() != successor.size()) {
        return false;
    }
    for (std::size_t i = 0; i < stages.size(); ++i) {
        const uint32_t src = stages[i].region;
        const uint32_t dst = successor[i];
        if (dst == 0) { continue; }
        if (src >= 63 || dst != src + 1) { return false; }
        for (const auto & s : stages) {
            if (s.region == dst) { return false; }
        }
    }
    return true;
}

} // namespace ggml_phone_plan
