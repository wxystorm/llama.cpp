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
    m.directional_ms =
        [](llama_region_ring_stage_kind kind,
           int tokens, int kv, double & ms) {
            const double per =
                kind == llama_region_ring_stage_kind::C2P ?
                    0.050 : 0.060;
            ms = 0.5 + per * tokens +
                0.00002 * tokens * kv;
            return true;
        };
    m.directional_prepare_ms =
        [](llama_region_ring_stage_kind,
           int, int, double & ms) {
            ms = 0.1;
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

    // Unequal Cycle boundaries reuse the exact Region list instead of
    // reconstructing the first Cycle's widths. This is the MoE APPLY_SAFE
    // concrete-region capability, now part of APPLY_WIDE candidate scoring.
    candidate uneven = uniform;
    uneven.concrete_regions = llama_region_ring_parse(
        "G4,C8,P4,G5,C7,P4,G3,C8,P5", 48);
    std::vector<llama_region_ring_region> uneven_regions;
    REQUIRE(make_regions(uneven, m.layers, uneven_regions));
    REQUIRE(llama_region_ring_layout_string(uneven_regions) ==
            "G4,C8,P4,G5,C7,P4,G3,C8,P5");
    REQUIRE(estimate_region_ring_balance(
        m, uneven_regions, uneven));
    REQUIRE(simulate(m, uneven_regions, uneven, &reason));
    REQUIRE(uneven.wall_ms > 0.0);
    REQUIRE(uneven.concrete_regions.size() == 9);
    // A deliberately malformed explicit plan must not silently fall back
    // to its unrelated legacy uniform C/G/P fields.
    candidate invalid_uneven = uneven;
    invalid_uneven.concrete_regions[1].layer_end++;
    REQUIRE(!make_regions(
        invalid_uneven, m.layers, uneven_regions));

    // Dense RPC graph costs are based on actual Stage kind and whether
    // Phone Full is inside a C2P->P2C Meta island. A non-directional Phone
    // Region is already one direct RPC graph; an unfused embedded one needs
    // multiple subgraph submissions. This callback must participate in both
    // the full event simulation and the coarse balance equations.
    model dense_registration = mock_profile();
    bool saw_embedded = false;
    bool saw_direct = false;
    bool saw_c2p = false;
    bool saw_p2c = false;
    dense_registration.phone_registration_cost =
        [&](llama_region_ring_stage_kind kind, int layers,
            int tokens, bool embedded, double & ms) {
            REQUIRE(layers > 0 && tokens > 0);
            saw_embedded |=
                kind == llama_region_ring_stage_kind::PHONE && embedded;
            saw_direct |=
                kind == llama_region_ring_stage_kind::PHONE && !embedded;
            saw_c2p |= kind == llama_region_ring_stage_kind::C2P;
            saw_p2c |= kind == llama_region_ring_stage_kind::P2C;
            ms = kind == llama_region_ring_stage_kind::PHONE ?
                 (embedded ? 1.0 : 2.0) : 0.5;
            return true;
        };
    const auto dense_island = llama_region_ring_parse(
        "G4,C6,C2P1,P4,P2C1,G4,C7,P5,G3,C8,P5", 48);
    candidate dense_island_est = uniform;
    REQUIRE(simulate(
        dense_registration, dense_island,
        dense_island_est, &reason));
    REQUIRE(saw_embedded && saw_direct && saw_c2p && saw_p2c);
    REQUIRE(dense_island_est.wall_ms > 0.0);

    model invalid_dense_registration = mock_profile();
    invalid_dense_registration.phone_registration_cost =
        [](llama_region_ring_stage_kind, int, int, bool, double &) {
            return false;
        };
    candidate invalid_dense_est = uniform;
    REQUIRE(!simulate(
        invalid_dense_registration, regions,
        invalid_dense_est, &reason));
    REQUIRE(reason == "PHONE_REGISTRATION_PROFILE_MISSING");

    // Simplified full-prompt prepare model: every claimed task queues on one
    // shared PREPARE lane before entering its independent device compute lane.
    model prepared = mock_profile();
    prepared.prepare_ms =
        [](device, int, int, int, double & ms) {
            ms = 1.0;
            return true;
        };
    candidate prepared_uniform = uniform;
    REQUIRE(simulate(
        prepared, regions, prepared_uniform, &reason));
    REQUIRE(std::fabs(
        prepared_uniform.prepare_busy_ms -
        (double) prepared_uniform.tasks) < 1e-9);
    REQUIRE(prepared_uniform.wall_ms + 1e-9 >= uniform.wall_ms);

    // Handwritten G->C->P minimax equations. Construct A=4, B=4, D=3,
    // TP2G=1, TC2P=1 and TG2C=0.1 ms/token. Then e1=0,
    // e2=0.1/4.1, e3=2/5 and E=0.4.
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
    REQUIRE(std::fabs(equation.ring_error[1] - (0.1 / 4.1)) < 1e-9);
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

    // A non-dividing three-cycle width may end with a GPU-only tail.
    candidate tail = uniform;
    tail.p = 4;
    tail.tail_g = 3;
    REQUIRE(make_regions(tail, m.layers, regions));
    REQUIRE(regions.size() == 10 && regions.back().device == device::GPU);
    REQUIRE(simulate(m, regions, tail, &reason));

    // Or continue the partial tail GPU -> CPU.
    tail.tail_g = 1;
    tail.tail_c = 2;
    REQUIRE(make_regions(tail, m.layers, regions));
    REQUIRE(regions.back().device == device::CPU);
    REQUIRE(simulate(m, regions, tail, &reason));

    candidate invalid_cpu_only_tail = tail;
    invalid_cpu_only_tail.tail_g = 0;
    invalid_cpu_only_tail.tail_c = 3;
    REQUIRE(!make_regions(invalid_cpu_only_tail, m.layers, regions));

    // make_regions() clears its output before validating a candidate, so
    // rebuild the last valid G->C->P topology before the profile-missing test.
    REQUIRE(make_regions(tail, m.layers, regions));

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
    // and the final logits owner; Chunk64 and the full-batch boundary buffer remain fixed.
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

    // Two-Bank GPU weight streaming: logical GPU ownership is not a
    // residency limit. Four GPU Regions may own far more total weight than
    // either Bank while only the largest even/odd GPU Region must coexist.
    {
        model streamed = mock_profile();
        streamed.gpu_weight_streaming = true;
        streamed.gpu_region_weight_bytes =
            [](int begin, int end, size_t & bytes) {
                if (begin < 0 || end <= begin) {
                    return false;
                }
                // Deliberately make later Regions larger so parity matters.
                bytes = (size_t) (end - begin) * 1000 +
                        (size_t) begin * 10;
                return true;
            };
        streamed.gpu_weight_h2d_ms =
            [](size_t bytes, double & ms) {
                ms = (double) bytes / 1000.0;
                return true;
            };
        streamed.prepare_ms =
            [](device d, int, int, int, double & ms) {
                ms = d == device::GPU ? 2.0 : 0.5;
                return true;
            };

        candidate streamed_plan;
        streamed_plan.cycles = 4;
        streamed_plan.c = 4;
        streamed_plan.g = 4;
        streamed_plan.p = 4;
        streamed_plan.xc = 64;
        streamed_plan.xg = 16;
        streamed_plan.xp = 128;

        std::vector<llama_region_ring_region> streamed_regions;
        REQUIRE(make_regions(
            streamed_plan, streamed.layers, streamed_regions));
        std::array<size_t, 2> bank_peaks {};
        size_t total_gpu_weight = 0;
        REQUIRE(gpu_weight_bank_peaks(
            streamed, streamed_regions, bank_peaks,
            &total_gpu_weight));
        REQUIRE(bank_peaks[0] > 0 && bank_peaks[1] > 0);
        REQUIRE(total_gpu_weight >
                bank_peaks[0] + bank_peaks[1]);

        double expected_h2d = 0.0;
        REQUIRE(gpu_weight_h2d_total(
            streamed, streamed_regions, expected_h2d));
        REQUIRE(expected_h2d > 0.0);

        std::string streamed_reason;
        REQUIRE(simulate(
            streamed, streamed_regions, streamed_plan,
            &streamed_reason));
        REQUIRE(std::fabs(
            streamed_plan.weight_h2d_busy_ms -
            expected_h2d) < 1e-9);
        // rank0/rank1 are load-time resident; prompt simulation begins
        // with Bank A/B ready and pays only rank2/rank3 rolling costs.
        REQUIRE(std::fabs(
            streamed_plan.weight_h2d_startup_ms) < 1e-9);
        REQUIRE(std::fabs(
            streamed_plan.weight_prepare_busy_ms - 4.0) < 1e-9);
        REQUIRE(streamed_plan.prepare_busy_ms + 1e-9 >=
                streamed_plan.weight_prepare_busy_ms);
        REQUIRE(streamed_plan.weight_h2d_exposed_ms >= 0.0);
        REQUIRE(streamed_plan.weight_h2d_hidden_ms >= 0.0);
        REQUIRE(streamed_plan.weight_h2d_hidden_ms <=
                streamed_plan.weight_h2d_busy_ms + 1e-9);

        // APPLY_WIDE now uses E as the first-stage physical filter. Every
        // profile-feasible candidate in E_min + margin is simulated; there is
        // no topology-family quota and no max_simulations truncation.
        model value_profile = mock_profile();
        value_profile.gpu_weight_streaming = true;
        value_profile.compute_ms =
            [](device d, int layers, int tokens, int, double & ms) {
                const double per =
                    d == device::CPU ? 1.0 :
                    d == device::GPU ? 0.01 : 0.05;
                ms = per * layers * tokens;
                return true;
            };
        value_profile.transfer_ms =
            [](device, device, size_t, double & ms) {
                ms = 0.001;
                return true;
            };
        value_profile.gpu_region_weight_bytes =
            [](int begin, int end, size_t & bytes) {
                if (begin < 0 || end <= begin) {
                    return false;
                }
                bytes = (size_t) (end - begin) * 1024;
                return true;
            };
        value_profile.gpu_weight_h2d_ms =
            [](size_t bytes, double & ms) {
                ms = (double) bytes / (1024.0 * 1024.0);
                return true;
            };
        value_profile.memory_feasible =
            [](const candidate &) { return true; };

        const auto value_baseline = llama_region_ring_parse("", 48);
        const result value_result = search_runtime_applicable(
            value_profile, value_baseline, 64, 16, 0.50, 0);
        REQUIRE(value_result.valid);
        REQUIRE(value_result.shortlisted == value_result.simulated);
        REQUIRE(value_result.nonuniform_profiled > 0);
        REQUIRE(value_result.nonuniform_simulated > 0);
        REQUIRE(value_result.simulated > 16);
        REQUIRE(value_result.selected.ring_error_max <=
                value_result.shortlist_e_limit + 1e-12);
        REQUIRE(std::fabs(
            value_result.coarse_e_limit -
            std::min(1.0, value_result.coarse_e_min + 0.50)) < 1e-12);

        // A narrow E window must not starve the full-prompt simulator.
        // Keep all natural E-window candidates, then extend the E-sorted
        // prefix to the requested floor without reintroducing a hard cap.
        const result floor_result = search_runtime_applicable(
            value_profile, value_baseline, 64, 1, 0.0, 64);
        REQUIRE(floor_result.valid);
        REQUIRE(floor_result.shortlisted == floor_result.simulated);
        const size_t expected_floor =
            std::min<size_t>(floor_result.profiled, 64);
        REQUIRE(floor_result.shortlisted >= expected_floor);
        REQUIRE(floor_result.shortlisted >=
                floor_result.e_window_shortlisted);
        if (floor_result.e_window_shortlisted < expected_floor) {
            REQUIRE(floor_result.shortlisted >= expected_floor);
            REQUIRE(floor_result.shortlist_e_limit + 1e-12 >=
                    floor_result.coarse_e_limit);
        } else {
            REQUIRE(floor_result.shortlisted >=
                    floor_result.e_window_shortlisted);
        }
        REQUIRE(floor_result.nonuniform_profiled > 0);
        REQUIRE(floor_result.selected.ring_error_max <=
                floor_result.shortlist_e_limit + 1e-12);
    }

    // The manual A/B override uses the same candidate contract as
    // APPLY_WIDE. Keep the concrete G5/C3/P4 x4 experiment scoreable with
    // independent XC/XG/XP so runtime forcing cannot drift from the planner.
    {
        model forced_profile = mock_profile();
        forced_profile.gpu_weight_streaming = true;
        forced_profile.gpu_region_weight_bytes =
            [](int begin, int end, size_t & bytes) {
                if (begin < 0 || end <= begin) {
                    return false;
                }
                bytes = (size_t) (end - begin) * 1024;
                return true;
            };
        forced_profile.gpu_weight_h2d_ms =
            [](size_t bytes, double & ms) {
                ms = (double) bytes / (1024.0 * 1024.0);
                return true;
            };
        forced_profile.memory_feasible =
            [](const candidate &) { return true; };

        const auto forced_regions = llama_region_ring_parse(
            "G5,C3,P4,G5,C3,P4,G5,C3,P4,G5,C3,P4", 48);
        candidate forced;
        forced.cycles = 4;
        forced.c = 3;
        forced.g = 5;
        forced.p = 4;
        forced.xc = 64;
        forced.xg = 16;
        forced.xp = 128;

        REQUIRE(llama_region_ring_apply_topology_compatible(
            forced_regions, forced_profile.layers));
        REQUIRE(estimate_region_ring_balance(
            forced_profile, forced_regions, forced));
        std::string forced_reason;
        REQUIRE(simulate(
            forced_profile, forced_regions, forced, &forced_reason));
        REQUIRE(forced.tasks > 0);
        REQUIRE(forced.edges > 0);
        REQUIRE(std::isfinite(forced.wall_ms));
        REQUIRE(forced.wall_ms > 0.0);
    }

    // APPLY_WIDE may change cycle count and independent chunks, but every
    // selected candidate must end on Phone with no tail so the current final
    // output owner remains canonical.
    {
        const auto baseline = llama_region_ring_parse("", 48);
        const result applicable =
            search_runtime_applicable(m, baseline, 64, 64, 0.20, 16);
        REQUIRE(applicable.valid);
        REQUIRE(applicable.simulated > 0);
        REQUIRE(applicable.selected.tail_c == 0);
        REQUIRE(applicable.selected.tail_g == 0);
        REQUIRE(applicable.selected.xc > 0);
        REQUIRE(applicable.selected.xg > 0);
        REQUIRE(applicable.selected.xp > 0);
        std::vector<llama_region_ring_region> applied_regions;
        REQUIRE(make_regions(
            applicable.selected, m.layers, applied_regions));
        REQUIRE(!applied_regions.empty());
        REQUIRE(applied_regions.back().device == device::PHONE);
        REQUIRE(llama_region_ring_apply_topology_compatible(
            applied_regions, m.layers));
    }

    // APPLY_CGP frees exactly the three repeated region lengths while
    // keeping three G->C->P cycles, Chunk64, the full-batch boundary buffer and
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
                "G6,C11,P7,G5,C11,P8",48),64).valid);


    // Boundary Refiner V1: the first-stage Cycle result remains a candidate.
    // Tensor refinement is odd-width only and alternates ownership direction.
    {
        candidate cycle;
        cycle.cycles = 3;
        cycle.g = 4;
        cycle.c = 7;
        cycle.p = 5;
        cycle.xc = cycle.xg = cycle.xp = 64;

        std::vector<llama_region_ring_region> cycle_regions;
        REQUIRE(make_regions(cycle, 48, cycle_regions));
        const auto anchors = boundary_anchors(cycle_regions);
        // Three PC->Phone entries and two Phone->PC/GPU exits.
        REQUIRE(anchors.size() == 5);

        const auto legal = boundary_legal_widths(anchors[0]);
        REQUIRE(!legal.empty() && legal.front() == 0);
        for (size_t i = 1; i < legal.size(); ++i) {
            REQUIRE((legal[i] & 1) != 0);
        }

        std::vector<int> widths(anchors.size(), 0);
        widths[0] = 3;
        std::vector<llama_region_ring_region> t3_regions;
        REQUIRE(build_boundary_regions(
            cycle_regions, anchors, widths, 48, t3_regions));
        REQUIRE(llama_region_ring_has_directional_stages(t3_regions));
        REQUIRE(llama_region_ring_directional_topology_compatible(
            t3_regions, 48));
        REQUIRE(llama_region_ring_layout_string(t3_regions).find(
            "C2P1,P2C1,C2P1") != std::string::npos);

        // Even width can never complete one net ownership handoff.
        widths[0] = 2;
        REQUIRE(!build_boundary_regions(
            cycle_regions, anchors, widths, 48, t3_regions));

        model no_tensor_profile = mock_profile();
        no_tensor_profile.region_memory_feasible =
            [](const std::vector<llama_region_ring_region> &) {
                return true;
            };
        no_tensor_profile.directional_ms =
            [](llama_region_ring_stage_kind,
               int, int, double & ms) {
                ms = 10000.0;
                return true;
            };
        candidate cycle_score = cycle;
        REQUIRE(simulate(
            no_tensor_profile, cycle_regions,
            cycle_score, &reason));
        const auto no_tensor = refine_optional_boundaries(
            no_tensor_profile, cycle_regions, cycle_score);
        REQUIRE(no_tensor.valid && no_tensor.baseline_valid);
        REQUIRE(no_tensor.selected.tensor_layers == 0);
        REQUIRE(llama_region_ring_layout_string(
            no_tensor.selected.regions) ==
            llama_region_ring_layout_string(cycle_regions));

        model tensor_profile = mock_profile();
        tensor_profile.region_memory_feasible =
            [](const std::vector<llama_region_ring_region> &) {
                return true;
            };
        tensor_profile.directional_ms =
            [](llama_region_ring_stage_kind,
               int, int, double & ms) {
                ms = 0.001;
                return true;
            };
        tensor_profile.directional_prepare_ms =
            [](llama_region_ring_stage_kind,
               int, int, double & ms) {
                ms = 0.0;
                return true;
            };
        candidate tensor_cycle_score = cycle;
        REQUIRE(simulate(
            tensor_profile, cycle_regions,
            tensor_cycle_score, &reason));
        const auto tensor_selected = refine_optional_boundaries(
            tensor_profile, cycle_regions, tensor_cycle_score);
        REQUIRE(tensor_selected.valid);
        REQUIRE(tensor_selected.selected.estimate.wall_ms <=
                tensor_selected.baseline.estimate.wall_ms + 1e-9);
        REQUIRE(tensor_selected.selected.tensor_layers > 0);
        REQUIRE(llama_region_ring_directional_topology_compatible(
            tensor_selected.selected.regions, 48));
        for (int width : tensor_selected.selected.widths) {
            REQUIRE(width == 0 || (width & 1) != 0);
        }
    }

    std::printf("REGION_RING_PLANNER_TEST status=PASS baseline_ms=%.3f "
                "hetero_ms=%.3f selected_ms=%.3f "
                "simulated=%zu coarse=%zu\n",
                uniform.wall_ms, smaller_phone.wall_ms,
                selected.selected.wall_ms, selected.simulated, selected.profiled);
    return 0;
}
