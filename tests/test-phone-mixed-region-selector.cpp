// Host-only regression for selecting whole Transformer Phone Regions by
// physical backend. No GGUF, GPU, NPU, Android or RPC required.
// Compile: c++ -std=c++17 -Isrc tests/test-phone-mixed-region-selector.cpp -o test-phone-mixed-region-selector
#include "llama-region-ring.h"
#include <cassert>
#include <cstdint>
#include <set>
#include <string>

int main() {
    std::set<uint32_t> htp;
    assert(llama_region_ring_parse_region_ids(nullptr, htp) && htp.empty());
    assert(llama_region_ring_parse_region_ids("", htp) && htp.empty());
    assert(llama_region_ring_parse_region_ids("2,5", htp));
    assert((htp == std::set<uint32_t>{2,5}));
    assert(!llama_region_ring_parse_region_ids("2,2", htp));
    assert(!llama_region_ring_parse_region_ids("-1", htp));
    assert(!llama_region_ring_parse_region_ids("64", htp));
    assert(!llama_region_ring_parse_region_ids("2,", htp));
    assert(!llama_region_ring_parse_region_ids(",2", htp));
    assert(!llama_region_ring_parse_region_ids("2, 5", htp));
    assert(!llama_region_ring_parse_region_ids("2;5", htp));
    assert(!llama_region_ring_parse_region_ids("0x2", htp));
    assert(!llama_region_ring_parse_region_ids("999999999999999999", htp));

    assert(llama_region_ring_parse_region_ids("2,5,8", htp));
    const auto stages = llama_region_ring_parse_apply_wide_layout(
        "G8,C1,P1,G8,C1,P1,G8,C1,P1,G7,C2,P1", 40);
    assert(stages.size() == 12);
    assert(llama_region_ring_apply_topology_compatible(stages, 40));
    int htp_layers = 0, opencl_layers = 0, pc_layers = 0;
    for (size_t i = 0; i < stages.size(); ++i) {
        const auto & stage = stages[i];
        const int n = stage.layer_end - stage.layer_begin;
        if (stage.device != llama_region_ring_device::PHONE) {
            assert(htp.count((uint32_t) i) == 0);
            pc_layers += n;
        } else if (htp.count((uint32_t) i)) {
            assert(n == 1);
            const int expected = i == 2 ? 9 : 19;
            assert(stage.layer_begin == expected);
            ++htp_layers;
        } else {
            assert((i == 8 && stage.layer_begin == 29 && n == 1) ||
                   (i == 11 && stage.layer_begin == 39 && n == 1));
            opencl_layers += n;
        }
    }
    assert(pc_layers == 36);
    assert(htp_layers == 2);
    assert(opencl_layers == 2);
    assert(pc_layers + htp_layers + opencl_layers == 40);
}
