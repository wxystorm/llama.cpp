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
    m.pc_compute_gate = false;
    m.tokens = 358;
    m.hidden = 2048;
    m.slot_pool = (int) LLAMA_REGION_RING_SLOT_POOL_SIZE;
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
    m.memory_feasible = [](const candidate &) { return true; };
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
    REQUIRE(uniform.tasks == 54 && uniform.max_live_slots <= m.slot_pool);
    REQUIRE(uniform.wall_ms > 0.0);
    // Current runtime: CPU and CUDA are independent compute lanes.
    REQUIRE(uniform.pc_gate_busy_ms == 0.0);
    REQUIRE(std::fabs(uniform.pc_compute_bound_ms -
                     std::max(uniform.busy_ms[0], uniform.busy_ms[1])) < 1e-6);
    REQUIRE(uniform.service_ms_per_token[0] > 0.0);
    REQUIRE(uniform.service_ms_per_token[1] > 0.0);
    REQUIRE(uniform.service_ms_per_token[2] > 0.0);

    // Handwritten minimax ring equations. Construct A=4, B=4, D=3,
    // TP2C=1, TC2P=1 ms/token, so e1=0, e2=0, e3=2/5 and E=0.4.
    model equation_profile = mock_profile();
    equation_profile.phone_register_ms = 0.0;
    equation_profile.compute_ms = [](device d, int layers, int tokens,
                                     int, double & ms) {
        const double per_layer_mspt =
            d == device::CPU ? 1.0 :
            d == device::GPU ? 0.5 : 0.75;
        ms = per_layer_mspt * layers * tokens;
        return true;
    };
    equation_profile.transfer_ms = [](device a, device b, size_t bytes,
                                      double & ms) {
        const double tokens =
            (double) bytes / (2048.0 * sizeof(float));
        const double per_token =
            (a == device::PHONE || b == device::PHONE) ? 1.0 : 0.1;
        ms = tokens * per_token;
        return true;
    };
    candidate equation;
    equation.cycles = 3;
    equation.c = 4;
    equation.g = 8;
    equation.p = 4;
    equation.xc = equation.xg = equation.xp = 64;
    REQUIRE(estimate_balance(equation_profile, equation));
    REQUIRE(std::fabs(equation.ring_stage_ms_per_token[0] - 4.0) < 1e-9);
    REQUIRE(std::fabs(equation.ring_stage_ms_per_token[1] - 4.0) < 1e-9);
    REQUIRE(std::fabs(equation.ring_stage_ms_per_token[2] - 3.0) < 1e-9);
    REQUIRE(std::fabs(equation.ring_p2c_ms_per_token - 1.0) < 1e-9);
    REQUIRE(std::fabs(equation.ring_c2p_ms_per_token - 1.0) < 1e-9);
    REQUIRE(std::fabs(equation.ring_error[0]) < 1e-9);
    REQUIRE(std::fabs(equation.ring_error[1]) < 1e-9);
    REQUIRE(std::fabs(equation.ring_error[2] - 0.4) < 1e-9);
    REQUIRE(std::fabs(equation.ring_error_max - 0.4) < 1e-9);
    REQUIRE(std::fabs(equation.balance - equation.ring_error_max) < 1e-12);

    candidate nonuniform;
    nonuniform.xc = nonuniform.xg = nonuniform.xp = 64;
    const auto fixed_regions = llama_region_ring_parse("", 48);
    REQUIRE(estimate_region_ring_balance(
        equation_profile, fixed_regions, nonuniform));
    // Baseline C widths 7,7,8 average to 22/3; G widths 4,4,3 to 11/3.
    REQUIRE(std::fabs(nonuniform.ring_stage_ms_per_token[0] -
                     22.0 / 3.0) < 1e-9);
    REQUIRE(std::fabs(nonuniform.ring_stage_ms_per_token[1] -
                     11.0 / 6.0) < 1e-9);
    REQUIRE(std::fabs(nonuniform.ring_stage_ms_per_token[2] -
                     3.75) < 1e-9);
    REQUIRE(std::isfinite(nonuniform.ring_error_max));

    // Keep the historical gated model only as a regression comparison.
    model gated_pc = mock_profile();
    gated_pc.pc_compute_gate = true;
    candidate gated_uniform = uniform;
    REQUIRE(simulate(gated_pc, regions, gated_uniform, &reason));
    REQUIRE(gated_uniform.pc_gate_busy_ms <= gated_uniform.wall_ms + 1e-6);
    REQUIRE(std::fabs(gated_uniform.pc_gate_busy_ms -
                     gated_uniform.busy_ms[0] - gated_uniform.busy_ms[1]) < 1e-6);
    REQUIRE(gated_uniform.wall_ms + 1e-6 >= uniform.wall_ms);

    // One PC Chunk splits into two or four Phone Chunks. The producer Slot
    // remains live until all downstream consumers have finished reading it.
    candidate smaller_phone = uniform;
    smaller_phone.xc = 64;
    smaller_phone.xg = 32;
    smaller_phone.xp = 16;
    REQUIRE(simulate(m, regions, smaller_phone, &reason));
    REQUIRE(smaller_phone.tasks > uniform.tasks);
    REQUIRE(smaller_phone.max_live_slots <= m.slot_pool);
    REQUIRE(smaller_phone.pc_gate_busy_ms <= smaller_phone.wall_ms + 1e-6);

    // Many Phone producers feed a larger PC Chunk: test interval gather and
    // exact-coverage prerequisites without assuming identical Chunk IDs.
    candidate larger_pc = uniform;
    larger_pc.xc = 128;
    larger_pc.xg = 64;
    larger_pc.xp = 32;
    REQUIRE(simulate(m, regions, larger_pc, &reason));
    REQUIRE(larger_pc.max_live_slots <= m.slot_pool);

    // Production Ring uses slot_pool=0: a large downstream chunk may gather
    // any number of smaller upstream intervals because every boundary owns a
    // full-ubatch activation buffer. The old bounded mode is kept only as a
    // regression check and must still expose the historical credit deadlock.
    candidate unbounded_gather = uniform;
    unbounded_gather.xc = 128;
    unbounded_gather.xg = 16;
    unbounded_gather.xp = 128;
    REQUIRE(m.slot_pool == 0);
    REQUIRE(simulate(m, regions, unbounded_gather, &reason));

    model bounded_slots = m;
    bounded_slots.slot_pool = 4;
    candidate bounded_gather = unbounded_gather;
    reason.clear();
    REQUIRE(!simulate(bounded_slots, regions, bounded_gather, &reason));
    REQUIRE(reason == "SLOT_OR_KV_DEADLOCK");

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
    const result selected = search(limited, 80, 0.08, 16);
    REQUIRE(selected.valid && selected.simulated > 0);
    REQUIRE(selected.shortlisted == selected.simulated);
    REQUIRE(selected.shortlisted <= 80);
    REQUIRE(std::isfinite(selected.coarse_e_min));
    REQUIRE(selected.coarse_e_limit + 1e-12 >= selected.coarse_e_min);
    REQUIRE(selected.selected.wall_ms > 0.0);
    REQUIRE(selected.selected.cycles * selected.selected.p <= 15);
    REQUIRE(selected.selected.cycles * selected.selected.g <= 12);
    REQUIRE(selected.selected.cycles *
               (selected.selected.c + selected.selected.g + selected.selected.p) +
           selected.selected.tail_c + selected.selected.tail_g == m.layers);
    REQUIRE(selected.top.size() <= 10);

    // General topology may change Phone P5 to P8, the number of cycles,
    // and the final logits owner; Chunk64 and the shared dynamic slot pool remain fixed.
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
    REQUIRE(general.selected.max_live_slots <= m.slot_pool);
    REQUIRE(!search_supported_topology(general_profile,128).valid);

    // APPLY_CGP frees exactly the three repeated region lengths while
    // keeping three C->G->P cycles, Chunk64, the shared dynamic slot pool and
    // final Phone ownership. CPU/CUDA/Phone are independent service lanes.
    // For 48 layers: C+G+P=16, P2..P8 => 70 layouts.
    const auto baseline_regions = llama_region_ring_parse("", 48);
    model free_cgp_profile = mock_profile();
    const result free_cgp =
        search_three_cycle_cgp(free_cgp_profile, baseline_regions, 64);
    REQUIRE(free_cgp.baseline_valid && free_cgp.valid);
    REQUIRE(free_cgp.enumerated == 70);
    // Small safe space: the minimax equation score ranks/explains candidates,
    // but no C/G/P geometry is discarded before exact simulation.
    REQUIRE(free_cgp.profiled == 70);
    REQUIRE(free_cgp.simulated == 70);
    REQUIRE(std::isfinite(free_cgp.selected.ring_error_max));
    REQUIRE(free_cgp.selected.ring_error_max >= 0.0 &&
            free_cgp.selected.ring_error_max <= 1.0);
    REQUIRE(free_cgp.top.size() <= 5);
    REQUIRE(free_cgp.selected.cycles == 3);
    REQUIRE(free_cgp.selected.tail_c == 0 &&
            free_cgp.selected.tail_g == 0);
    REQUIRE(free_cgp.selected.c + free_cgp.selected.g +
            free_cgp.selected.p == 16);
    REQUIRE(free_cgp.selected.c >= 1 &&
            free_cgp.selected.g >= 1 &&
            free_cgp.selected.p >= 2 &&
            free_cgp.selected.p <= 8);
    std::vector<llama_region_ring_region> free_cgp_regions;
    REQUIRE(make_regions(free_cgp.selected,48,free_cgp_regions));
    REQUIRE(free_cgp_regions.size() == 9);
    REQUIRE(free_cgp_regions.back().device == device::PHONE);
    REQUIRE(llama_region_ring_apply_topology_compatible(
        free_cgp_regions,48));
    REQUIRE(!search_three_cycle_cgp(
        free_cgp_profile, baseline_regions, 128).valid);
    model cgp_no_memory = mock_profile();
    cgp_no_memory.memory_feasible = {};
    REQUIRE(!search_three_cycle_cgp(
        cgp_no_memory, baseline_regions, 64).valid);

    // Runtime APPLY-safe uses a separate constrained search. It may ONLY
    // move the C/G boundaries within the three existing 11-layer PC windows;
    // it must preserve all three Phone P5 regions, CUDA11 total and Chunk64.
    model apply_profile = mock_profile();
    const supported_result supported =
        search_supported_fixed_phone(apply_profile,baseline_regions,64);
    REQUIRE(supported.baseline_valid && supported.valid);
    // Three 11-layer PC windows, GPU total fixed at 11, with at least one
    // CPU and one CUDA layer in each window:
    // g0 + g1 + g2 = 11, gi >= 1 => C(10,2) = 45 safe layouts.
    REQUIRE(supported.enumerated == 45);
    REQUIRE(supported.simulated == 45);
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
    const auto supported_counts =
        llama_region_ring_layer_counts(supported.selected.regions);
    REQUIRE(supported_counts[0] == 22);
    REQUIRE(supported_counts[1] == 11);
    REQUIRE(supported_counts[2] == 15);
    REQUIRE(supported_counts ==
            llama_region_ring_layer_counts(baseline_regions));
    REQUIRE(supported.top.size() <= 5);
    for (const auto & choice : supported.top) {
        REQUIRE(llama_region_ring_apply_safe_compatible(
                baseline_regions,choice.regions));
        REQUIRE(llama_region_ring_apply_safe_phone_shape(choice.regions));
        REQUIRE(choice.estimate.tasks == 54);
        REQUIRE(choice.estimate.max_live_slots <= m.slot_pool);
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

    model unsupported_slot_pool = mock_profile();
    unsupported_slot_pool.slot_pool =
        (int) LLAMA_REGION_RING_SLOT_POOL_SIZE - 1;
    REQUIRE(!search_supported_fixed_phone(
            unsupported_slot_pool,baseline_regions,64).valid);
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
