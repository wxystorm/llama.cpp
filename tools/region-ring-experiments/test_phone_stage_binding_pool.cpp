// Standalone host test: c++ -std=c++17 -O2 -Wall -Wextra
//   -I. tools/region-ring-experiments/test_phone_stage_binding_pool.cpp
//   -o /tmp/test_phone_stage_binding_pool && /tmp/test_phone_stage_binding_pool
#include "../../ggml/src/ggml-rpc/phone-stage-binding-pool.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

using namespace ggml_phone_stage;

static template_plan plan(uint32_t tokens, size_t bytes) {
    template_plan p;
    p.layer_begin = 11;
    p.layer_end = 15;
    p.tokens = tokens;
    p.topology_signature = (uint64_t) tokens + 1;
    assert(p.scratch.build({{1, bytes, 64, 0, 2}}, {}) == status::ok);
    return p;
}

int main() {
    // Reproduce 64-token Phone Stage addresses: source gallocr backend
    // buffer is 321,650,688 bytes, but the Stage only references its
    // first 17,368,320 bytes. A partial scratch-only estimate of
    // 15,990,784 must never become a binding-capacity guarantee.
    arena_extent stage;
    const size_t base = 0x100000;
    assert(stage.add(base, 321650688, base, 1310720));
    assert(stage.add(base, 321650688, base + 16057600, 1310720));
    assert(stage.valid());
    assert(stage.first_offset == 0);
    assert(stage.last_offset == 17368320);
    assert(stage.span_bytes() == 17368320);
    const auto count = stage.tensor_count;
    assert(!stage.add(base, 321650688, base + 321650688 - 8, 16));
    assert(!stage.add(base, 321650688, base - 1, 1));
    assert(stage.tensor_count == count && stage.last_offset == 17368320);
    assert(!stage.add(base, 321650688,
                      std::numeric_limits<size_t>::max(), 1024));

    binding_layout a;
    // Reuse is allowed only AFTER the previous consumer has finished.
    assert(a.build({
        {1, 4096, 256, 0, 3},
        {2, 2048, 256, 4, 5},
        {3, 1024, 256, 3, 5},
    }, {{4, 1, 64, 128}}) == status::ok);
    assert(a.roots.size() == 3);
    assert(a.roots[0].offset == 0);
    assert(a.roots[1].id == 3 && a.roots[1].offset >= 4096); // live at node 3
    assert(a.roots[2].id == 2 && a.roots[2].offset == 0); // root 1 is free by node 4

    std::unordered_map<uint32_t, size_t> bound = {{123, 777}};
    assert(a.resolve(0x100000, a.required_bytes, bound) == status::ok);
    assert(bound.at(4) == bound.at(1) + 64);
    assert(bound.at(2) == bound.at(1));
    const auto saved = bound;
    assert(a.resolve(0x100000, a.required_bytes - 1, bound) ==
        status::out_of_memory);
    assert(bound == saved); // Failed binding leaves prior result intact
    assert(a.resolve(std::numeric_limits<size_t>::max() - 100,
                     a.required_bytes, bound) == status::overflow);
    assert(bound == saved); // Overflow also leaves prior result intact

    binding_layout invalid;
    assert(invalid.build({{1, 64, 0, 0, 0}}, {}) == status::invalid);
    assert(invalid.build({{1, 64, 32, 0, 0}},
                         {{2, 1, 60, 5}}) == status::invalid);
    assert(invalid.build({{1, 64, 32, 0, 0},
                          {1, 64, 32, 1, 2}}, {}) == status::invalid);
    assert(invalid.build({{1, 64, 32, 0, 0}},
                         {{1, 1, 0, 16}}) == status::invalid);
    assert(!checked_add(std::numeric_limits<size_t>::max(), 1,
                        bound[123]));

    // One shared two-slot budget is MAX(template requirements), not SUM.
    auto main = plan(64, 1024);
    auto tail = plan(45, 512);
    two_slot_pool pool;
    assert(pool.reserve({main, tail}, 256, 128) == status::ok);
    assert(pool.bytes_per_slot() == 1280);
    const uint64_t epoch = pool.epoch();
    assert(pool.acquire(0, 100, epoch) == status::ok);
    assert(pool.acquire(1, 101, epoch) == status::ok);
    assert(pool.acquire(0, 102, epoch) == status::busy);
    assert(pool.output_published(0, 100) == status::stale);
    assert(pool.compute_complete(0, 100) == status::ok);
    assert(pool.acquire(0, 102, epoch) == status::busy); // output not safe
    assert(pool.reserve({main}, 256) == status::busy);
    assert(pool.output_published(0, 100) == status::ok);
    assert(pool.acquire(0, 102, epoch) == status::ok);
    assert(pool.compute_complete(1, 101) == status::ok);
    assert(pool.output_published(1, 101) == status::ok);
    assert(pool.compute_complete(0, 102) == status::ok);
    assert(pool.output_published(0, 102) == status::ok);
    assert(pool.reserve({main, tail}, 256) == status::ok);
    assert(pool.acquire(0, 103, epoch) == status::stale); // invalidated epoch

    // The 105-node Stage may alias its input and final output into the
    // same original gallocr arena. A scratch-only peak of ~15.25 MiB
    // MUST NOT allow allocating a smaller slot than the true 16.56 MiB
    // observed span (17,368,320 bytes from the original arena base).
    main.stage_arena_required_bytes = 17368320;
    tail.stage_arena_required_bytes = 10312704;
    assert(pool.reserve({main, tail}, 256) == status::ok);
    assert(pool.bytes_per_slot() == 17368320);
    assert(pool.bytes_per_slot() * k_slot_count == 34736640);

    // Pure scratch-only tests still use scratch peak when larger.
    tail.scratch.required_bytes = 20000000;
    assert(pool.reserve({main, tail}, 256) == status::ok);
    assert(pool.bytes_per_slot() == 20000000);
    return 0;
}
