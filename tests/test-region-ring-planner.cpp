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

    // General topology may change Phone P5 to P8, the number of cycles,
    // and the final logits owner; but Chunk64 and Window=2 remain fixed.
    model general_profile = mock_profile();
    const result general = search_supported_topology(general_profile,64,96);
    REQUIRE(general.valid && general.simulated > 0);
    REQUIRE(general.selected.xc == 64 &&
            general.selected.xg == 64 &&
            general.selected.xp == 64);
    std::vector<llama_region_ring_region> general_regions;
    REQUIRE(make_regions(general.selected,48,general_regions));
    REQUIRE(llama_region_ring_apply_topology_compatible(
        general_regions,48));
    REQUIRE(general.selected.max_live_slots <= 2);
    REQUIRE(!search_supported_topology(general_profile,128).valid);
    const result executable = search_supported_topology(
        general_profile,64,96,true);
    REQUIRE(executable.valid && executable.simulated > 0);
    std::vector<llama_region_ring_region> executable_regions;
    REQUIRE(make_regions(executable.selected,48,executable_regions));
    REQUIRE(executable_regions.back().device != device::PHONE);
    REQUIRE(executable.selected.tail_c + executable.selected.tail_g > 0);
    REQUIRE(executable.selected.max_live_slots <= 2);
    REQUIRE(executable.selected.tasks ==
            (int) executable_regions.size() * 6);
    candidate example;
    example.cycles = 2;
    example.c = 6;
    example.g = 4;
    example.p = 8;
    example.tail_c = 8;
    example.tail_g = 4;
    example.xc = example.xg = example.xp = 64;
    std::vector<llama_region_ring_region> example_regions;
    REQUIRE(make_regions(example,48,example_regions));
    REQUIRE(example_regions.size() == 8);
    REQUIRE(example_regions.back().device == device::GPU);
    REQUIRE(simulate(general_profile,example_regions,example,&reason));
    REQUIRE(example.tasks == 48);
    REQUIRE(example.max_live_slots <= 2);

    // Runtime APPLY-safe uses a separate constrained search. It may ONLY
    // move the C/G boundaries within the three existing 11-layer PC windows;
    // it must preserve all three Phone P5 regions, CUDA11 total and Chunk64.
    const auto baseline_regions = llama_region_ring_parse("", 48);
    model apply_profile = mock_profile();
    const supported_result supported =
        search_supported_fixed_phone(apply_profile,baseline_regions,64);
    REQUIRE(supported.baseline_valid && supported.valid);
    REQUIRE(supported.enumerated > 0 && supported.simulated > 0);
    REQUIRE(supported.selected.estimate.wall_ms > 0.0);
    REQUIRE(supported.selected.estimate.wall_ms <=
            supported.baseline.estimate.wall_ms + 1e-6);
    REQUIRE(supported.selected.estimate.xc == 64 &&
            supported.selected.estimate.xg == 64 &&
            supported.selected.estimate.xp == 64);
    REQUIRE(llama_region_ring_apply_safe_phone_shape(
            supported.selected.regions));
    REQUIRE(llama_region_ring_apply_safe_compatible(
            baseline_regions,supported.selected.regions));
    REQUIRE(llama_region_ring_layer_counts(
            supported.selected.regions) ==
            llama_region_ring_layer_counts(baseline_regions));
    REQUIRE(supported.top.size() <= 5);
    for (const auto & choice : supported.top) {
        REQUIRE(llama_region_ring_apply_safe_compatible(
                baseline_regions,choice.regions));
        REQUIRE(llama_region_ring_apply_safe_phone_shape(choice.regions));
        REQUIRE(choice.estimate.tasks == 54);
        REQUIRE(choice.estimate.max_live_slots <= 2);
    }

    model only_baseline = mock_profile();
    only_baseline.region_memory_feasible = [&](
            const std::vector<llama_region_ring_region> & r) {
        return llama_region_ring_layout_string(r) ==
               llama_region_ring_layout_string(baseline_regions);
    };
    const auto single_supported =
        search_supported_fixed_phone(only_baseline,baseline_regions,64);
    REQUIRE(single_supported.valid && single_supported.simulated == 1);
    REQUIRE(llama_region_ring_layout_string(
            single_supported.selected.regions) ==
            llama_region_ring_layout_string(baseline_regions));

    model no_memory = mock_profile();
    no_memory.region_memory_feasible = [](
            const std::vector<llama_region_ring_region> &) {
        return false;
    };
    const auto none_supported =
        search_supported_fixed_phone(no_memory,baseline_regions,64);
    REQUIRE(none_supported.baseline_valid && !none_supported.valid);
    REQUIRE(none_supported.rejected_memory == none_supported.enumerated);

    model unsupported_window = mock_profile();
    unsupported_window.window = 3;
    REQUIRE(!search_supported_fixed_phone(
            unsupported_window,baseline_regions,64).valid);
    REQUIRE(!search_supported_fixed_phone(
            apply_profile,baseline_regions,128).valid);
    REQUIRE(!search_supported_fixed_phone(
            apply_profile,llama_region_ring_parse(
                "C11,G6,P7,C11,G5,P8",48),64).valid);

    std::printf("REGION_RING_PLANNER_TEST status=PASS baseline_ms=%.3f "
                "hetero_ms=%.3f selected_ms=%.3f "
                "simulated=%zu coarse=%zu\n",
                uniform.wall_ms, smaller_phone.wall_ms,
                selected.selected.wall_ms, selected.simulated, selected.profiled);
    return 0;
}
