#include "phone-npu-weight-banks.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "HTP_TWO_BANK_TEST_FAIL line=%d expr=%s\n", \
                 __LINE__, #expr); \
    std::abort(); \
} } while (0)

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
    CHECK(!banks.configure(regions, 2 * MiB, 1 * MiB, 17 * MiB, reason));
    CHECK(reason == "HTP_TWO_BANK_MEMORY_EXCEEDED");
    CHECK(banks.configure(regions, 2 * MiB, 1 * MiB, 18 * MiB, reason));
    CHECK(banks.total_required_bytes() == 18 * MiB);
    CHECK(banks.kv_resident_bytes() == 2 * MiB);
    CHECK(banks.bank_capacity(0) == 8 * MiB);
    CHECK(banks.bank_capacity(1) == 7 * MiB);
    CHECK(!banks.begin_chunk(0, 0, banks.bank(0).generation)); // no compute before initial load
    CHECK(banks.prime_complete(0) && banks.prime_complete(1));
    CHECK(!banks.prime_complete(0));
    CHECK(banks.begin_chunk(0, 0, banks.bank(0).generation));
    CHECK(!banks.begin_chunk(1, 0, banks.bank(1).generation)); // HTP compute is globally serial
    const auto not_last = banks.finish_chunk(0, 0, false);
    CHECK(!not_last);
    CHECK(!banks.begin_chunk(0, 0, banks.bank(0).generation)); // FIFO: no duplicate chunk
    CHECK(banks.begin_chunk(0, 1, banks.bank(0).generation));
    const auto a_to2 = banks.finish_chunk(0, 1, true);
    CHECK(a_to2 && a_to2->bank == 0 && a_to2->from_rank == 0 &&
           a_to2->to_rank == 2 && a_to2->to_region == 5);
    CHECK(!banks.begin_refill(*a_to2)); // cached Graph still references A
    CHECK(banks.graphs_retired(0));
    CHECK(banks.begin_refill(*a_to2));
    CHECK(!banks.begin_chunk(2, 0, banks.bank(0).generation)); // refill not yet finished
    CHECK(banks.refill_complete(2, true));
    CHECK(!banks.begin_chunk(2, 0, banks.bank(0).generation - 1));
    CHECK(banks.begin_chunk(2, 0, banks.bank(0).generation));
    CHECK(!banks.begin_refill(*a_to2));
    CHECK(banks.finish_chunk(2, 0, true) == std::nullopt);

    // Bank B can be independently reused only after rank 1 drains.
    CHECK(banks.begin_chunk(1, 0, banks.bank(1).generation));
    const auto b_to3 = banks.finish_chunk(1, 0, true);
    CHECK(b_to3 && b_to3->bank == 1 && b_to3->to_rank == 3);
    CHECK(!banks.begin_refill(*b_to3));
    CHECK(banks.graphs_retired(1));
    CHECK(banks.begin_refill(*b_to3));
    CHECK(!banks.refill_complete(3, false));
    CHECK(!banks.begin_chunk(3, 0, banks.bank(1).generation)); // failure never becomes READY
    CHECK(banks.kv_resident_bytes() == 2 * MiB); // KV never reused/evicted

    auto invalid = regions;
    invalid[1].region = invalid[0].region;
    CHECK(!banks.configure(invalid, 0, 0, 100 * MiB, reason));
    CHECK(reason == "HTP_TWO_BANK_REGION_ORDER_OR_SIZE");
    return 0;
}
