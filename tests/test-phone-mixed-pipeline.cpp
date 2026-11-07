// Host-only policy test for mixed Phone NPU/GPU Chunk pipeline.
// No GGUF, Phone, HTP SDK, OpenCL runtime or RPC needed.
#include "../ggml/src/ggml-rpc/phone-mixed-pipeline-policy.h"
#include <cassert>
#include <vector>

int main() {
    using namespace ggml_phone_mixed_pipeline;

    // Default: one worker may execute either backend.
    assert(lane_accepts(false, false, true));
    assert(lane_accepts(false, false, false));

    // Dual lane: NPU and GPU are separate, but each is exactly one lane.
    assert(lane_accepts(true, true, true));
    assert(!lane_accepts(true, true, false));
    assert(lane_accepts(true, false, false));
    assert(!lane_accepts(true, false, true));

    const std::vector<interval> htp {{10, 30}, {45, 55}};
    const std::vector<interval> gpu {{20, 35}, {40, 50}};
    assert(overlap_us(htp, gpu) == 15); // 10 us + 5 us
    assert(overlap_us({{10, 20}}, {{20, 30}}) == 0);
    assert(overlap_us({{10, 10}, {0, 100}}, {{20, 30}}) == 0);
    assert(overlap_us({}, gpu) == 0);
    // Out-of-order vectors are deliberately sorted, no double counting.
    assert(overlap_us({{45, 55}, {10, 30}},
                      {{40, 50}, {20, 35}}) == 15);
    assert(overlap_us({{1, 5}, {6, 10}}, {{2, 7}}) == 4);
    return 0;
}
