#include "../ggml/src/ggml-rpc/phone-plan-region-map.h"
#include <cassert>

#include <stdexcept>

static void check(bool ok, const char * what) {
    if (!ok) { throw std::runtime_error(what); }
}

static void test_v2_multi_layer_stage_map() {
    using ggml_phone_plan::stage_region_range;
    using ggml_phone_plan::valid_htp_stage_map;
    using ggml_phone_plan::valid_htp_stage_successors;
    check(valid_htp_stage_map({}, 40), "N0 must be legal");
    check(valid_htp_stage_map({{2,0,4}}, 40), "N4 must be one physical Stage");
    check(valid_htp_stage_map({{2,0,4},{5,12,15}}, 40),
          "noncontiguous Stage locations are legitimate");
    check(!valid_htp_stage_map({{2,0,4},{5,3,6}}, 40),
          "overlapping multi-layer Stages cannot be accepted");
    check(!valid_htp_stage_map({{2,0,0}}, 40),
          "zero-layer NPU Stage cannot own a physical Region");
    check(!valid_htp_stage_map({{2,39,41}}, 40),
          "model layer overflow");
    check(!valid_htp_stage_map({{5,0,4},{2,4,8}}, 40),
          "reversed Region ID order");
    check(valid_htp_stage_successors({{2,0,4},{5,12,15}}, {3,0}, 40),
          "N4->A local handoff must be legal");
    check(!valid_htp_stage_successors({{2,0,4},{3,12,15}}, {3,0}, 40),
          "HTP successor must not be another HTP Region");
    check(!valid_htp_stage_successors({{2,0,4}}, {5}, 40),
          "successor must be adjacent");
}

int main() {
    test_v2_multi_layer_stage_map();
    using ggml_phone_plan::valid_htp_map;
    using ggml_phone_plan::valid_local_successors;
    using ggml_phone_plan::region_layer;
    assert(valid_htp_map({{2,9},{5,19}}));
    assert(valid_htp_map({{2,9},{5,19},{8,29},{11,39}}));
    assert(!valid_htp_map({}));
    assert(!valid_htp_map({{2,9}}));
    assert(!valid_htp_map({{2,9},{2,19}}));
    assert(!valid_htp_map({{5,9},{2,19}}));
    assert(!valid_htp_map({{2,19},{5,9}}));
    assert(!valid_htp_map({{64,9},{65,19}}));
    assert(!valid_htp_map({{2,255},{5,256}}));
    assert(valid_local_successors({{2,9},{5,19}}, {3,0}));
    assert(valid_local_successors({{2,9},{5,19}}, {3,6}));
    assert(valid_local_successors({{2,9},{5,19}}, {0,0}));
    assert(!valid_local_successors({{2,9},{5,19}}, {3}));
    assert(!valid_local_successors({{2,9},{5,19}}, {4,0}));
    assert(!valid_local_successors({{2,9},{3,19}}, {3,0}));
    assert(!valid_local_successors({{62,9},{63,19}}, {63,0}));
    assert(!valid_local_successors({{2,9},{63,19}}, {0,64}));
    return 0;
}
