// Standalone: c++ -std=c++17 -O2 -Wall -Wextra -Werror test_phone_stage_dual_slot.cpp
#include "../../ggml/src/ggml-rpc/phone-stage-binding-pool.h"

#include <cassert>

using namespace ggml_phone_stage;

int main() {
    two_slot_pool pool;
    template_plan plan;
    plan.layer_begin = 11;
    plan.layer_end = 14;
    plan.tokens = 64;
    plan.topology_signature = 1;
    plan.stage_arena_required_bytes = 17368320;
    assert(pool.reserve({plan}, 128) == status::ok);
    const uint64_t epoch = pool.epoch();
    assert(pool.bytes_per_slot() >= 17368320);

    size_t a = k_slot_count;
    size_t b = k_slot_count;
    size_t unused = k_slot_count;
    assert(pool.acquire_next_preparing(1, epoch, a) == status::ok);
    assert(a == 0 && pool.busy(a));
    assert(pool.acquire_next_preparing(2, epoch, b) == status::ok);
    assert(b == 1 && pool.busy(b));
    assert(pool.acquire_next_preparing(3, epoch, unused) == status::busy);
    assert(unused == k_slot_count);
    assert(pool.start_compute(a, 3) == status::stale);
    assert(pool.start_compute(a, 1) == status::ok);
    assert(pool.output_published(a, 1) == status::stale);
    assert(pool.reserve({plan}, 128) == status::busy);
    assert(pool.compute_complete(a, 1) == status::ok);
    assert(pool.output_published(a, 1) == status::ok);
    assert(!pool.busy(a));
    assert(pool.cancel_preparing(b, 2) == status::ok);
    assert(!pool.busy(b));

    assert(pool.acquire_next_preparing(3, epoch, a) == status::ok);
    assert(a == 0);
    assert(pool.cancel_preparing(a, 2) == status::stale);
    assert(pool.cancel_preparing(a, 3) == status::ok);
    assert(pool.reserve({plan}, 128) == status::ok);
    assert(pool.epoch() != epoch);
    assert(pool.acquire_next_preparing(4, epoch, a) == status::stale);
    assert(pool.acquire_next_preparing(4, pool.epoch(), a) == status::ok);
    assert(a == 1);
    assert(pool.start_compute(a, 4) == status::ok);
    assert(pool.compute_complete(a, 4) == status::ok);
    assert(pool.output_published(a, 4) == status::ok);
    return 0;
}
