// Standalone (host only):
//   c++ -std=c++17 -O2 -Wall -Wextra -Werror test_phone_npu_serial_policy.cpp
// No Phone, HTP SDK, model file or RPC needed.
#include "../../ggml/src/ggml-rpc/phone-npu-serial-policy.h"

#include <cassert>
#include <set>
#include <string>

int main() {
    using namespace ggml_phone_npu_serial;
    std::set<uint32_t> selected;
    assert(parse_regions(nullptr, selected) && selected.empty());
    assert(parse_regions("", selected) && selected.empty());
    assert(parse_regions("3,7", selected));
    assert(selected.size() == 2 && htp_region(selected, 3));
    assert(htp_region(selected, 7) && !htp_region(selected, 1));
    assert(parse_regions("0,63", selected));
    assert(htp_region(selected, 0) && htp_region(selected, 63));
    assert(!parse_regions("64", selected));
    assert(!parse_regions("-1", selected));
    assert(!parse_regions("3,3", selected));
    assert(!parse_regions("3,", selected));
    assert(!parse_regions(",3", selected));
    assert(!parse_regions("1, 3", selected));
    assert(!parse_regions("3;7", selected));
    assert(!parse_regions("0x3", selected));
    assert(layer_from_tensor_name("blk.19.attn_k.weight") == 19);
    assert(layer_from_tensor_name("Meta(CPU,RPC0)#blk.39.ffn_up.weight#0") == 39);
    assert(layer_from_tensor_name("blk.1.attn_k.weight") == 1);
    assert(layer_from_tensor_name("blk.10.attn_k.weight") == 10);
    assert(layer_from_tensor_name("blk.1x.attn_k.weight") == -1);
    assert(layer_from_tensor_name("inp_stage") == -1);
    assert(layer_from_tensor_name(nullptr) == -1);
    return 0;
}
