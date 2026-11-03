// Offline, backend-free regression tests for CPU-optional APPLY_WIDE plans.
// Windows (Developer PowerShell): cl /nologo /std:c++17 /EHsc /I src /Fe:test_cpu_optional_planner.exe tools/region-ring-experiments/test_cpu_optional_planner.cpp
// Linux: c++ -std=c++17 -O2 -Isrc tools/region-ring-experiments/test_cpu_optional_planner.cpp -o /tmp/test_cpu_optional_planner
#include "llama-region-ring-planner.h"
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

int main() {
    test_layouts();
    test_optional_cpu_dag_and_memory();
    std::cout << "CPU optional Planner offline tests: PASS\n";
}
