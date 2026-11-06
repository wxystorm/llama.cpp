#include "phone-npu-weight-banks.h"

#include <cassert>
#include <string>
#include <vector>

using namespace ggml_phone_npu_serial;
static constexpr size_t MiB = 1024 * 1024;

int main() {
    const std::vector<weight_bank_region> regions = {
        {1, 9, 10, 4 * MiB}, {3, 19, 20, 6 * MiB},
        {5, 29, 30, 8 * MiB}, {7, 39, 40, 7 * MiB},
    };
    std::string reason;
    two_weight_banks banks;
    // Even rank max 8MiB + odd rank max 7MiB + KV 2MiB + scratch 1MiB.
    assert(!banks.configure(regions, 2 * MiB, 1 * MiB, 17 * MiB, reason));
    assert(reason == "HTP_TWO_BANK_MEMORY_EXCEEDED");
    assert(banks.configure(regions, 2 * MiB, 1 * MiB, 18 * MiB, reason));
    assert(banks.total_required_bytes() == 18 * MiB);
    assert(banks.kv_resident_bytes() == 2 * MiB);
    assert(banks.bank_capacity(0) == 8 * MiB);
    assert(banks.bank_capacity(1) == 7 * MiB);
    assert(!banks.begin_chunk(0, 0)); // no compute before initial load
    assert(banks.prime_complete(0) && banks.prime_complete(1));
    assert(!banks.prime_complete(0));
    assert(banks.begin_chunk(0, 0));
    assert(!banks.begin_chunk(1, 0)); // HTP compute is globally serial
    const auto not_last = banks.finish_chunk(0, 0, false);
    assert(!not_last);
    assert(!banks.begin_chunk(0, 0)); // FIFO: no duplicate chunk
    assert(banks.begin_chunk(0, 1));
    const auto a_to2 = banks.finish_chunk(0, 1, true);
    assert(a_to2 && a_to2->bank == 0 && a_to2->from_rank == 0 &&
           a_to2->to_rank == 2 && a_to2->to_region == 5);
    assert(!banks.begin_refill(*a_to2)); // cached Graph still references A
    assert(banks.graphs_retired(0));
    assert(banks.begin_refill(*a_to2));
    assert(!banks.begin_chunk(2, 0)); // refill not yet finished
    assert(banks.refill_complete(2, true));
    assert(banks.begin_chunk(2, 0));
    assert(!banks.begin_refill(*a_to2));
    assert(banks.finish_chunk(2, 0, true) == std::nullopt);

    // Bank B can be independently reused only after rank 1 drains.
    assert(banks.begin_chunk(1, 0));
    const auto b_to3 = banks.finish_chunk(1, 0, true);
    assert(b_to3 && b_to3->bank == 1 && b_to3->to_rank == 3);
    assert(!banks.begin_refill(*b_to3));
    assert(banks.graphs_retired(1));
    assert(banks.begin_refill(*b_to3));
    assert(!banks.refill_complete(3, false));
    assert(!banks.begin_chunk(3, 0)); // failure never becomes READY
    assert(banks.kv_resident_bytes() == 2 * MiB); // KV never reused/evicted

    auto invalid = regions;
    invalid[1].region = invalid[0].region;
    assert(!banks.configure(invalid, 0, 0, 100 * MiB, reason));
    assert(reason == "HTP_TWO_BANK_REGION_ORDER_OR_SIZE");
    return 0;
}
