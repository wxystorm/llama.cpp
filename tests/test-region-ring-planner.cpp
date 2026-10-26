// Standalone, deterministic Planner DAG tests; no model, CUDA or Phone needed.
#include "llama-region-ring-planner.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
using namespace llama_ring_planner;
#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "REGION_RING_PLANNER_TEST_FAIL line=%d expr=%s\n", \
                     __LINE__, #expr); \
        std::abort(); \
    } \
} while (0)

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
    REQUIRE(make_regions(uniform, m.layers, regions));
    REQUIRE(regions.size() == 9);
    REQUIRE(simulate(m, regions, uniform, &reason));
    REQUIRE(uniform.tasks == 54 && uniform.max_live_slots <= 2);
    REQUIRE(uniform.wall_ms > 0.0);
    // PC CPU and CUDA share the real execution gate: their aggregate
    // compute demand cannot exceed elapsed wall time in the gated model.
    REQUIRE(uniform.pc_gate_busy_ms <= uniform.wall_ms + 1e-6);
    REQUIRE(std::fabs(uniform.pc_gate_busy_ms -
                     uniform.busy_ms[0] - uniform.busy_ms[1]) < 1e-6);

    // The counterfactual independent-device model is retained for future
    // backend ownership experiments, but must be opt-in.
    model independent_pc = mock_profile();
    independent_pc.pc_compute_gate = false;
    candidate independent_uniform = uniform;
    REQUIRE(simulate(independent_pc, regions, independent_uniform, &reason));
    REQUIRE(independent_uniform.wall_ms > 0.0);

    // One PC Chunk splits into two or four Phone Chunks. The output Slot
    // is released when all transfers have entered downstream staging.
    candidate smaller_phone = uniform;
    smaller_phone.xc = 64;
    smaller_phone.xg = 32;
    smaller_phone.xp = 16;
    REQUIRE(simulate(m, regions, smaller_phone, &reason));
    REQUIRE(smaller_phone.tasks > uniform.tasks);
    REQUIRE(smaller_phone.max_live_slots <= 2);
    REQUIRE(smaller_phone.pc_gate_busy_ms <= smaller_phone.wall_ms + 1e-6);

    // Many Phone producers feed a larger PC Chunk: test interval gather and
    // exact-coverage prerequisites without assuming identical Chunk IDs.
    candidate larger_pc = uniform;
    larger_pc.xc = 128;
    larger_pc.xg = 64;
    larger_pc.xp = 32;
    REQUIRE(simulate(m, regions, larger_pc, &reason));
    REQUIRE(larger_pc.max_live_slots <= 2);

    // A non-dividing three-cycle width gives a PC-only final suffix.
    candidate tail = uniform;
    tail.p = 4;
    tail.tail_c = 3;
    REQUIRE(make_regions(tail, m.layers, regions));
    REQUIRE(regions.size() == 10 && regions.back().device == device::CPU);
    REQUIRE(simulate(m, regions, tail, &reason));

    // A suffix may end on CUDA instead of CPU.
    tail.tail_c = 1;
    tail.tail_g = 2;
    REQUIRE(make_regions(tail, m.layers, regions));
    REQUIRE(regions.back().device == device::GPU);
    REQUIRE(simulate(m, regions, tail, &reason));

    // If required Profile data is absent, reject the estimate, never invent
    // a speed. This is DRY_RUN: no real Ring runtime state is changed.
    model incomplete = mock_profile();
    incomplete.compute_ms = [](device d, int, int, int, double & ms) {
        if (d == device::PHONE) return false;
        ms = 1.0;
        return true;
    };
    REQUIRE(!simulate(incomplete, regions, tail, &reason));
    REQUIRE(reason == "COMPUTE_PROFILE_MISSING");

    model limited = mock_profile();
    limited.memory_feasible = [](const candidate & p) {
        return p.cycles * p.p <= 15 && p.cycles * p.g <= 12;
    };
    const result selected = search(limited, 80);
    REQUIRE(selected.valid && selected.simulated > 0);
    REQUIRE(selected.selected.wall_ms > 0.0);
    REQUIRE(selected.selected.cycles * selected.selected.p <= 15);
    REQUIRE(selected.selected.cycles * selected.selected.g <= 12);
    REQUIRE(selected.selected.cycles *
               (selected.selected.c + selected.selected.g + selected.selected.p) +
           selected.selected.tail_c + selected.selected.tail_g == m.layers);
    REQUIRE(selected.top.size() <= 5);

    std::printf("REGION_RING_PLANNER_TEST status=PASS baseline_ms=%.3f "
                "hetero_ms=%.3f selected_ms=%.3f "
                "simulated=%zu coarse=%zu\n",
                uniform.wall_ms, smaller_phone.wall_ms,
                selected.selected.wall_ms, selected.simulated, selected.profiled);
    return 0;
}
