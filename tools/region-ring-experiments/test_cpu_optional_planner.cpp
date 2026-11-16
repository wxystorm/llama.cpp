// Offline, backend-free regression tests for CPU-optional APPLY_WIDE plans.
// Windows (Developer PowerShell): cl /nologo /std:c++17 /EHsc /I src /Fe:test_cpu_optional_planner.exe tools/region-ring-experiments/test_cpu_optional_planner.cpp
// Linux: c++ -std=c++17 -O2 -Isrc tools/region-ring-experiments/test_cpu_optional_planner.cpp -o /tmp/test_cpu_optional_planner
#include "llama-region-ring-planner.h"
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

using device = llama_region_ring_device;
using namespace llama_ring_planner;

static bool rejects(const char * text) {
    try {
        (void) llama_region_ring_parse_apply_wide_layout(text, 40);
        return false;
    } catch (const std::exception &) {
        return true;
    }
}

static void test_layouts() {
    const auto baseline = llama_region_ring_parse_apply_wide_layout(
        "G11,C1,P2,G10,C1,P2,G11,C1,P1", 40);
    assert(baseline.size() == 9);
    const auto gp = llama_region_ring_parse_apply_wide_layout(
        "G12,C0,P2,G11,C0,P2,G11,P2", 40);
    assert(gp.size() == 6);
    assert(llama_region_ring_apply_topology_compatible(gp, 40));
    assert(llama_region_ring_layout_string(gp) ==
           "G12,P2,G11,P2,G11,P2");
    const auto mixed = llama_region_ring_parse_apply_wide_layout(
        "G11,C1,P2,G12,C0,P2,G11,P1", 40);
    assert(mixed.size() == 8);
    assert(llama_region_ring_apply_topology_compatible(mixed, 40));
    assert(llama_region_ring_layout_string(mixed) ==
           "G11,C1,P2,G12,P2,G11,P1");
    const auto gp_no_explicit_zero = llama_region_ring_parse_apply_wide_layout(
        "G12,P2,G11,P2,G11,P2", 40);
    assert(gp_no_explicit_zero.size() == 6);
    assert(rejects("G0,P2,G11,P2,G23,P2"));
    assert(rejects("G12,P0,G12,P2,G12,P2"));
    assert(rejects("C2,G10,P2,G10,P2,G14"));
    assert(rejects("G12,P2,G11,P2,G11,P1"));
}

static model make_test_model() {
    model m;
    m.layers = 40;
    m.tokens = 128;
    m.hidden = 64;
    m.kv_start = 0;
    m.phone_full_return_stream = true;
    m.compute_ms = [](device d, int layers, int tokens, int, double & ms) {
        if (layers < 1 || tokens < 1) { return false; }
        // CPU is deliberately much slower, to favor deleting CPU regions.
        ms = layers * (d == device::CPU ? 100.0 :
                       d == device::GPU ? 8.0 : 6.0) * tokens / 64.0;
        return true;
    };
    m.prepare_ms = [](device, int, int, int, double & ms) {
        ms = 0.5;
        return true;
    };
    m.transfer_ms = [](device, device, size_t bytes, double & ms) {
        ms = 0.2 + bytes / (1.0 * 1024 * 1024);
        return true;
    };
    m.memory_feasible = [](const candidate &) { return true; };
    return m;
}

static void test_optional_cpu_dag_and_memory() {
    model m = make_test_model();
    const auto baseline = llama_region_ring_parse_apply_wide_layout(
        "G11,C1,P2,G10,C1,P2,G11,C1,P1", 40);
    candidate original;
    original.cycles = 3;
    original.g = 11;
    original.c = 1;
    original.p = 2;
    original.xc = original.xg = original.xp = 64;
    original.concrete_regions = baseline;
    assert(estimate_region_ring_balance(m, baseline, original));
    assert(simulate(m, baseline, original));

    result out;
    out.top.push_back(original);
    std::unordered_set<std::string> scored;
    search_runtime_cpu_optional(m, out, scored);
    assert(out.cpu_optional_enumerated > 0);
    assert(out.cpu_optional_profiled > 0);
    assert(out.cpu_optional_simulated > 0);
    bool found_cpu_free = false;
    bool found_partial = false;
    for (const candidate & c : out.top) {
        if (!c.cpu_optional_seed) { continue; }
        std::vector<llama_region_ring_region> regions;
        assert(make_regions(c, 40, regions));
        assert(llama_region_ring_apply_topology_compatible(regions, 40));
        assert(regions.back().device == device::PHONE);
        const bool no_cpu = std::none_of(regions.begin(), regions.end(),
            [](const llama_region_ring_region & r) {
                return r.device == device::CPU;
            });
        found_cpu_free |= no_cpu;
        found_partial |= !no_cpu;
        assert(c.transfer_event_count[1] > 0); // Direct G -> Phone.
        assert(c.wall_ms > 0);
        for (const auto & r : regions) {
            assert(r.layer_end > r.layer_begin); // No zero-length tasks.
        }
    }
    assert(found_cpu_free && found_partial);

    // Memory guard must be applied before the GPU/Phone DAG is scored.
    m.memory_feasible = [](const candidate & c) {
        return std::any_of(c.concrete_regions.begin(),
                           c.concrete_regions.end(),
            [](const llama_region_ring_region & r) {
                return r.device == device::CPU;
            });
    };
    result limited;
    limited.top.push_back(original);
    scored.clear();
    search_runtime_cpu_optional(m, limited, scored);
    assert(limited.rejected_memory > 0);
    for (const candidate & c : limited.top) {
        if (!c.cpu_optional_seed) { continue; }
        assert(std::any_of(c.concrete_regions.begin(), c.concrete_regions.end(),
            [](const llama_region_ring_region & r) {
                return r.device == device::CPU;
            }));
    }
}

// Final CPU shrink is an optional neighborhood around already-scored
// plans. It must be able to remove C1 and partially shrink C2, and it must
// reject memory-infeasible neighbors before their DAG simulation.
static void test_final_cpu_shrink() {
    for (int cpu_depth : {1, 2}) {
        model m = make_test_model();
        const auto base = llama_region_ring_parse_apply_wide_layout(
            cpu_depth == 1 ?
                "G11,C1,P2,G10,C1,P2,G11,C1,P1" :
                "G11,C2,P1,G10,C2,P1,G10,C2,P1", 40);
        assert(runtime_cycle_micro_adjusted(base));
        candidate origin;
        origin.cycles = 3;
        origin.g = 11;
        origin.c = cpu_depth;
        origin.p = cpu_depth == 1 ? 2 : 1;
        origin.xc = origin.xg = origin.xp = 64;
        origin.concrete_regions = base;
        assert(estimate_region_ring_balance(m, base, origin));
        assert(simulate(m, base, origin));

        result improved;
        improved.top.push_back(origin);
        std::unordered_set<std::string> seen;
        refine_runtime_cpu_shrink(m, improved, seen);
        assert(improved.cpu_shrink_anchors == 1);
        assert(improved.cpu_shrink_generated > 0);
        assert(improved.cpu_shrink_profiled > 0);
        assert(improved.cpu_shrink_simulated > 0);
        const int original_cpu_layers = cpu_depth * 3;
        bool saw_cpu_drop = false;
        bool saw_cpu_remove = false;
        bool saw_partial = false;
        for (const auto & plan : improved.top) {
            std::vector<llama_region_ring_region> layout;
            assert(make_regions(plan, 40, layout));
            assert(llama_region_ring_apply_topology_compatible(layout, 40));
            assert(runtime_cycle_micro_adjusted(layout));
            int cpu_layers = 0;
            int cpu_stages = 0;
            for (const auto & region : layout) {
                if (region.device == device::CPU) {
                    cpu_layers += region.layer_end - region.layer_begin;
                    ++cpu_stages;
                }
            }
            if (cpu_layers < original_cpu_layers) { saw_cpu_drop = true; }
            if (cpu_stages < 3) { saw_cpu_remove = true; }
            if (cpu_depth == 2 &&
                    cpu_layers == original_cpu_layers - 1) {
                saw_partial = true;
            }
        }
        assert(saw_cpu_drop);
        assert(saw_cpu_remove);
        if (cpu_depth == 2) { assert(saw_partial); }

        // The same previously-scored source is retained, but all smaller
        // CPU candidates fail the real callback and are never simulated.
        m.memory_feasible = [original_cpu_layers](const candidate & plan) {
            int count = 0;
            for (const auto & region : plan.concrete_regions) {
                if (region.device == device::CPU) {
                    count += region.layer_end - region.layer_begin;
                }
            }
            return count >= original_cpu_layers;
        };
        result rejected;
        rejected.top.push_back(origin);
        seen.clear();
        refine_runtime_cpu_shrink(m, rejected, seen);
        assert(rejected.cpu_shrink_memory_rejected > 0);
        assert(rejected.cpu_shrink_simulated == 0);
        assert(rejected.top.size() == 1);
    }
}

// A previous 512-token UBatch can leave A/B holding later GPU Regions.
// A stale "initial banks loaded" flag must not be considered sufficient.
static void test_gpu_bank_pair_after_ubatch() {
    const std::vector<size_t> gpu_regions = {0, 2, 4, 6};
    std::array<bool, 2> ready = {{true, true}};
    std::array<size_t, 2> owners = {{4, 6}};
    assert(!llama_region_ring_gpu_initial_banks_match(gpu_regions, ready, owners));
    owners = {{0, 2}}; // the boundary restore loaded the first two Regions
    assert(llama_region_ring_gpu_initial_banks_match(gpu_regions, ready, owners));
    ready[1] = false;
    assert(!llama_region_ring_gpu_initial_banks_match(gpu_regions, ready, owners));
}

// Independent CUDA compute schedulers coexist with both weight Banks.
// A layout that passed the old weight+KV-only check must be rejected when
// measured activation storage and a separate CUDA runtime reserve are charged.
static void test_gpu_cuda_memory_reserve() {
    constexpr size_t MiB = 1024 * 1024;
    model m = make_test_model();
    m.gpu_weight_streaming = true;
    m.slot_pool = 2;
    m.gpu_kv_bytes_per_layer = (long double) MiB / 2;
    m.gpu_region_weight_bytes =
        [](int begin, int end, size_t & bytes) {
            if (begin < 0 || end <= begin || end > 40) { return false; }
            bytes = (size_t) (end - begin) * 10 * MiB;
            return true;
        };
    m.gpu_compute_buffer_bytes =
        [](int xg, size_t & bytes) {
            if (xg != 32) { return false; }
            bytes = 64 * MiB;
            return true;
        };
    m.gpu_cuda_other_reserve_bytes = 40 * MiB;
    assert(gpu_cuda_additional_reserve_bytes(m, 32) == 168 * MiB);
    assert(gpu_cuda_additional_reserve_bytes(m, 32, 1) == 104 * MiB);

    const auto layout = llama_region_ring_parse_apply_wide_layout(
        "G13,P7,G12,C1,G7", 40);
    m.gpu_memory_budget_bytes = 400 * MiB;
    assert(!gpu_streamed_banks_fit(m, layout, 32));
    m.gpu_memory_budget_bytes = 480 * MiB;
    assert(gpu_streamed_banks_fit(m, layout, 32));
    // A one-Region CUDA layout needs only one physical scheduler buffer.
    const std::vector<llama_region_ring_region> one_gpu = {
        {device::GPU, 0, 13}, {device::PHONE, 13, 40},
    };
    m.gpu_memory_budget_bytes = 280 * MiB;
    assert(gpu_streamed_banks_fit(m, one_gpu, 32));
    m.gpu_memory_budget_bytes = 480 * MiB;
    m.gpu_compute_buffer_bytes =
        [](int, size_t &) { return false; };
    assert(!gpu_streamed_banks_fit(m, layout, 32));
    m.gpu_compute_buffer_bytes =
        [](int xg, size_t & bytes) {
            if (xg != 32) { return false; }
            bytes = 64 * MiB;
            return true;
        };
    // No CUDA allocation should be reserved for CPU+Phone-only plans.
    const std::vector<llama_region_ring_region> zero_gpu = {
        {device::CPU, 0, 20}, {device::PHONE, 20, 40},
    };
    m.gpu_memory_budget_bytes = 0.0L;
    assert(gpu_streamed_banks_fit(m, zero_gpu, 32));

    const std::vector<llama_region_ring_region> prefix = {
        {device::GPU, 0, 13},
        {device::CPU, 13, 14},
        {device::PHONE, 14, 21},
    };
    m.gpu_memory_budget_bytes = 300 * MiB;
    assert(!gpu_prefix_memory_feasible(m, prefix, 3, 32));
    // Legacy zero-reserve models retain their former optimistic bound.
    m.gpu_compute_buffer_bytes = nullptr;
    m.gpu_cuda_other_reserve_bytes = 0.0L;
    assert(gpu_prefix_memory_feasible(m, prefix, 3, 32));
}

static void test_gpu_region_half_budget_cap() {
    constexpr size_t MiB = 1024 * 1024;
    model m = make_test_model();
    m.gpu_weight_streaming = true;
    m.slot_pool = 0;
    m.gpu_kv_bytes_per_layer = 0.0L;
    m.gpu_memory_budget_bytes = 400 * MiB;
    m.gpu_region_weight_bytes = [](int begin, int end, size_t & bytes) {
        if (begin < 0 || end <= begin || end > 40) { return false; }
        bytes = (size_t) (end - begin) * 16 * MiB;
        return true;
    };
    const auto layout = llama_region_ring_parse_apply_wide_layout(
        "G13,P7,G12,C1,G7", 40);
    assert(!gpu_streamed_region_within_limit(m, 13 * 16 * MiB));
    assert(!gpu_streamed_banks_fit(m, layout, 32));
    // 13*16 = 208MiB < half of 420MiB. Combined A+B=400MiB.
    m.gpu_memory_budget_bytes = 420 * MiB;
    assert(gpu_streamed_region_within_limit(m, 13 * 16 * MiB));
    assert(gpu_streamed_banks_fit(m, layout, 32));
    m.gpu_max_region_budget_fraction = 0.40L;
    assert(!gpu_streamed_banks_fit(m, layout, 32));
}

int main() {
    test_layouts();
    test_optional_cpu_dag_and_memory();
    test_final_cpu_shrink();
    test_gpu_bank_pair_after_ubatch();
    test_gpu_cuda_memory_reserve();
    test_gpu_region_half_budget_cap();
    std::cout << "CPU optional Planner offline tests: PASS\n";
}
