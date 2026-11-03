#include "../ggml/src/ggml-rpc/phone-autonomous-policy.h"

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

int main() {
    test_one_chunk_lookahead();
    test_strict_kv_fifo();
    return 0;
}
