// Standalone, deterministic Planner DAG tests; no model, CUDA or Phone needed.
#include "llama-region-ring-planner.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace llama_ring_planner;

static model mock_profile() {
    model m;
    m.layers = 48;
    m.tokens = 358;
    m.hidden = 2048;
    m.window = 2;
    m.compute_ms = [](device d, int layers, int tokens, int kv, double & ms) {
        const double per = d == device::CPU ? 0.07 :
                           d == device::GPU ? 0.025 : 0.055;
        ms = layers * (0.3 + per * tokens) + 0.00003 * layers * tokens * kv;
        return true;
    };
    m.transfer_ms = [](device a, device b, size_t bytes, double & ms) {
        ms = (a == device::PHONE || b == device::PHONE ? 0.4 : 0.05) +
             (double) bytes / (60 * 1024 * 1024) * 1000.0;
        return true;
    };
    return m;
}

int main() {
    model m = mock_profile();
    candidate uniform;
    uniform.cycles = 3;
    uniform.c = 7;
    uniform.g = 4;
    uniform.p = 5;
    std::vector<llama_region_ring_region> regions;
    std::string reason;
    assert(make_regions(uniform, m.layers, regions));
    assert(regions.size() == 9);
    assert(simulate(m, regions, uniform, &reason));
    assert(uniform.tasks == 54 && uniform.max_live_slots <= 2);
    assert(uniform.wall_ms > 0.0);

    // One PC Chunk splits into two or four Phone Chunks. The output Slot
    // is released when all transfers have entered downstream staging.
    candidate smaller_phone = uniform;
    smaller_phone.xc = 64;
    smaller_phone.xg = 32;
    smaller_phone.xp = 16;
    assert(simulate(m, regions, smaller_phone, &reason));
    assert(smaller_phone.tasks > uniform.tasks);
    assert(smaller_phone.max_live_slots <= 2);

    // Many Phone producers feed a larger PC Chunk: test interval gather and
    // exact-coverage prerequisites without assuming identical Chunk IDs.
    candidate larger_pc = uniform;
    larger_pc.xc = 128;
    larger_pc.xg = 64;
    larger_pc.xp = 32;
    assert(simulate(m, regions, larger_pc, &reason));
    assert(larger_pc.max_live_slots <= 2);

    // A non-dividing three-cycle width gives a PC-only final suffix.
    candidate tail = uniform;
    tail.p = 4;
    tail.tail_c = 3;
    assert(make_regions(tail, m.layers, regions));
    assert(regions.size() == 10 && regions.back().device == device::CPU);
    assert(simulate(m, regions, tail, &reason));

    // A suffix may end on CUDA instead of CPU.
    tail.tail_c = 1;
    tail.tail_g = 2;
    assert(make_regions(tail, m.layers, regions));
    assert(regions.back().device == device::GPU);
    assert(simulate(m, regions, tail, &reason));

    // If required Profile data is absent, reject the estimate, never invent
    // a speed. This is DRY_RUN: no real Ring runtime state is changed.
    model incomplete = mock_profile();
    incomplete.compute_ms = [](device d, int, int, int, double & ms) {
        if (d == device::PHONE) return false;
        ms = 1.0;
        return true;
    };
    assert(!simulate(incomplete, regions, tail, &reason));
    assert(reason == "COMPUTE_PROFILE_MISSING");

    model limited = mock_profile();
    limited.memory_feasible = [](const candidate & p) {
        return p.cycles * p.p <= 15 && p.cycles * p.g <= 12;
    };
    const result selected = search(limited, 80);
    assert(selected.valid && selected.simulated > 0);
    assert(selected.selected.wall_ms > 0.0);
    assert(selected.selected.cycles * selected.selected.p <= 15);
    assert(selected.selected.cycles * selected.selected.g <= 12);
    assert(selected.selected.cycles *
               (selected.selected.c + selected.selected.g + selected.selected.p) +
           selected.selected.tail_c + selected.selected.tail_g == m.layers);
    assert(selected.top.size() <= 5);

    std::printf("REGION_RING_PLANNER_TEST status=PASS baseline_ms=%.3f "
                "hetero_ms=%.3f selected_ms=%.3f "
                "simulated=%zu coarse=%zu\n",
                uniform.wall_ms, smaller_phone.wall_ms,
                selected.selected.wall_ms, selected.simulated, selected.profiled);
    return 0;
}
