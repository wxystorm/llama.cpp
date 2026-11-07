#include "../ggml/src/ggml-rpc/phone-plan-region-map.h"
#include <cassert>

int main() {
    using ggml_phone_plan::valid_htp_map;
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
    return 0;
}
