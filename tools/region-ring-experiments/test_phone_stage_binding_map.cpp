// Standalone: c++ -std=c++17 -O2 -Wall -Wextra -I.
// tools/region-ring-experiments/test_phone_stage_binding_map.cpp -o /tmp/test_phone_stage_binding_map
#include "../../ggml/src/ggml-rpc/phone-stage-binding-map.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <unordered_map>

using namespace ggml_phone_stage;

int main() {
    stage_binding_map map(321650688);
    // Real 64-token full-stage high-water extent, including input/output.
    assert(map.add({1, 0, 1310720, 0, 0, true}) == status::ok);
    assert(map.add({2, 16057600, 1310720, 0, 0, true}) == status::ok);
    assert(map.add({3, 131072, 262144, 1, 131072, false}) == status::ok);
    assert(map.add({4, 196608, 65536, 3, 65536, false}) == status::ok);
    assert(map.add({5, 0, 4096, 0, 0, false}) == status::ok); // allocator reuse
    assert(map.validate() == status::ok);
    assert(map.entries().size() == 5);
    assert(map.view_count() == 2);
    assert(map.protected_count() == 2);
    assert(map.required_bytes() == 17368320);
    stage_binding_map changed(321650688);
    assert(changed.add({1, 0, 1310720, 0, 0, true}) == status::ok);
    assert(changed.add({2, 16057664, 1310720, 0, 0, true}) == status::ok);
    assert(map.layout_signature() == stage_binding_map(map).layout_signature());
    assert(map.layout_signature() != changed.layout_signature());
    using bindings = std::array<std::unordered_map<uint32_t, size_t>, k_slot_count>;
    bindings output;
    const std::array<stage_slot_address, k_slot_count> slots {{
        {0x10000000, 17368320}, {0x20000000, 17368320}
    }};
    assert(map.resolve_slots(slots, output) == status::ok);
    assert(output[0].at(2) == 0x10000000 + 16057600);
    assert(output[1].at(3) == 0x20000000 + 131072);
    assert(output[0].at(5) == output[0].at(1));
    const auto saved = output;
    auto too_small = slots;
    too_small[1].capacity = 17368319;
    assert(map.resolve_slots(too_small, output) == status::out_of_memory);
    assert(output == saved);
    auto overlapping = slots;
    overlapping[1].base = overlapping[0].base + 1;
    assert(map.resolve_slots(overlapping, output) == status::invalid);
    assert(output == saved);
    auto wrapping = slots;
    wrapping[1].base = std::numeric_limits<size_t>::max() - 100;
    assert(map.resolve_slots(wrapping, output) == status::overflow);
    assert(output == saved);
    auto null_slot = slots;
    null_slot[0].base = 0;
    assert(map.resolve_slots(null_slot, output) == status::invalid);
    assert(output == saved);

    // OpenCL returns a shared synthetic data pointer for independent buffers.
    // A tensor is identified by (buffer, data), never by data alone.
    bindings cl_output;
    const std::array<stage_slot_address, k_slot_count> cl_slots {{
        {0x1000, 17368320, 0x1111, true},
        {0x1000, 17368320, 0x2222, true},
    }};
    assert(map.resolve_slots(cl_slots, cl_output) == status::ok);
    assert(cl_output[0].at(2) == cl_output[1].at(2));
    assert(cl_output[0].at(2) == 0x1000 + 16057600);
    const auto cl_saved = cl_output;
    auto same_buffer = cl_slots;
    same_buffer[1].buffer_identity = same_buffer[0].buffer_identity;
    assert(map.resolve_slots(same_buffer, cl_output) == status::invalid);
    assert(cl_output == cl_saved);
    auto no_identity = cl_slots;
    no_identity[0].buffer_identity = 0;
    assert(map.resolve_slots(no_identity, cl_output) == status::invalid);
    assert(cl_output == cl_saved);
    auto mixed_address_spaces = cl_slots;
    mixed_address_spaces[0].opaque_address_space = false;
    assert(map.resolve_slots(mixed_address_spaces, cl_output) == status::invalid);
    assert(cl_output == cl_saved);
    auto host_same_base = slots;
    host_same_base[1].base = host_same_base[0].base;
    assert(map.resolve_slots(host_same_base, cl_output) == status::invalid);
    assert(cl_output == cl_saved);

    stage_binding_map bounds(128);
    assert(bounds.add({1, 120, 16, 0, 0, false}) == status::invalid);
    assert(bounds.entries().empty());
    assert(bounds.add({0, 0, 16, 0, 0, false}) == status::invalid);
    assert(bounds.add({1, 0, 64, 0, 0, false}) == status::ok);
    assert(bounds.add({1, 64, 64, 0, 0, false}) == status::invalid);
    assert(bounds.entries().size() == 1);

    stage_binding_map broken_view(128);
    assert(broken_view.add({2, 16, 16, 99, 16, false}) == status::ok);
    uint32_t failed_id = 0;
    assert(broken_view.validate(&failed_id) == status::invalid); // missing parent
    assert(failed_id == 2);
    stage_binding_map wrong_offset(128);
    assert(wrong_offset.add({1, 0, 64, 0, 0, false}) == status::ok);
    assert(wrong_offset.add({2, 32, 16, 1, 16, false}) == status::ok);
    assert(wrong_offset.validate() == status::invalid);
    stage_binding_map view_overrun(128);
    assert(view_overrun.add({1, 0, 64, 0, 0, false}) == status::ok);
    assert(view_overrun.add({2, 60, 16, 1, 60, false}) == status::ok);
    assert(view_overrun.validate() == status::invalid);
    stage_binding_map cycle(128);
    assert(cycle.add({1, 0, 64, 2, 0, false}) == status::ok);
    assert(cycle.add({2, 0, 64, 1, 0, false}) == status::ok);
    assert(cycle.validate() == status::invalid);
    return 0;
}
