#include "../ggml/src/ggml-rpc/phone-autonomous-policy.h"

// Keep assertions active under Release/MSVC NDEBUG builds.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdint>
#include <limits>

static void test_one_chunk_lookahead() {
    using ggml_phone_auto::can_prepare;
    assert(can_prepare(0, 0));
    assert(can_prepare(0, 1));
    assert(!can_prepare(0, 2));
    assert(!can_prepare(1, 0));
    assert(can_prepare(1, 1));
    assert(can_prepare(1, 2));
    assert(!can_prepare(1, 3));
    constexpr uint32_t limit = std::numeric_limits<uint32_t>::max();
    assert(can_prepare(limit - 1, limit));
    assert(!can_prepare(limit - 2, limit));
    assert(can_prepare(limit, limit));
}

static void test_strict_kv_fifo() {
    using ggml_phone_auto::can_compute;
    assert(can_compute(0, 0));
    assert(!can_compute(0, 1));
    assert(!can_compute(1, 0));
    assert(can_compute(1, 1));
    assert(!can_compute(1, 2));
    assert(can_compute(38, 38));
}

static void test_stage_input_name_contract() {
    using ggml_phone_auto::is_stage_input_name;
    assert(is_stage_input_name("inp_stage"));
    assert(is_stage_input_name("Meta(CPU,RPC0)#inp_stage#0"));
    assert(is_stage_input_name("Meta(RPC0)#inp_stage#0"));
    assert(!is_stage_input_name(nullptr));
    assert(!is_stage_input_name(""));
    assert(!is_stage_input_name("inp_stage_copy"));
    assert(!is_stage_input_name("Meta(CPU,RPC0)#inp_stage#1"));
    assert(!is_stage_input_name("Meta(CPU,RPC0)#inp_stage#0_extra"));
    assert(!is_stage_input_name("Meta(CPU,RPC0)#other_inp_stage#0"));
    assert(!is_stage_input_name("Other(CPU,RPC0)#inp_stage#0"));
    assert(!is_stage_input_name("Meta()#inp_stage#0"));
    assert(!is_stage_input_name("Meta(CPU,RPC0)inp_stage"));
}

static void test_server_native_storage_binding() {
    using ggml_phone_auto::bind_native_storage;
    alignas(16) uint8_t first[64] = {};
    alignas(16) uint8_t second[64] = {};
    uint8_t buffer_a = 0;
    uint8_t buffer_b = 0;
    uint64_t wire_buffer = 0;
    uint64_t wire_data = 0;
    assert(!bind_native_storage(wire_buffer, wire_data, nullptr, first));
    assert(wire_buffer == 0 && wire_data == 0);
    assert(!bind_native_storage(wire_buffer, wire_data, &buffer_a, nullptr));
    assert(wire_buffer == 0 && wire_data == 0);
    // Regression: serialize_tensor() blanks native OpenCL storage, so the
    // Phone must explicitly restore both fields to distinct real addresses.
    assert(bind_native_storage(wire_buffer, wire_data, &buffer_a, first));
    assert(wire_buffer ==
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&buffer_a)));
    assert(wire_data ==
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(first)));
    const uint64_t first_buffer = wire_buffer;
    const uint64_t first_data = wire_data;
    assert(bind_native_storage(wire_buffer, wire_data, &buffer_b, second));
    assert(wire_buffer != first_buffer);
    assert(wire_data != first_data);
}

int main() {
    test_one_chunk_lookahead();
    test_strict_kv_fifo();
    test_stage_input_name_contract();
    test_server_native_storage_binding();
    return 0;
}
