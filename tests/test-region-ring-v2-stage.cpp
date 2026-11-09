#include "../src/llama-region-ring-v2.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Release-safe checks: CI builds with NDEBUG, so do not use assert().
static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

using namespace llama_ring_v2;

static plan_spec base(int n, std::vector<cycle> cycles) {
    plan_spec p {};
    p.model_layers = n;
    p.chunk_tokens = 64;
    p.cpu_chunk_tokens = 32;
    p.cycles = std::move(cycles);
    return p;
}

static void test_four_backend_layouts() {
    const auto p = base(24, {
        {4, 0, 2, 0},  // G4,N2
        {3, 2, 3, 0},  // G3,C2,N3
        {4, 0, 1, 2},  // G4,N1,A2
        {2, 1, 0, 0},  // G2,C1, CPU tail
    });
    const auto x = compile(p);
    check(x.valid(), "valid four-backend plan was rejected");
    check(x.stages.size() == 10, "zero stages must be omitted");
    check(x.boundaries.size() == 9, "boundary count incorrect");
    check(x.stages.front().layer_begin == 0, "first layer is not zero");
    check(x.stages.back().layer_end == 24, "missing a model layer");
    check(x.final_backend() == backend::C, "CPU tail rejected");
    check(!x.final_head_needs_p2c() && !x.final_head_needs_d2h(),
          "CPU tail must not pay final phone/CUDA return");
    bool saw_na = false;
    bool saw_pg = false;
    bool saw_across_cycle = false;
    for (const auto & edge : x.boundaries) {
        check(edge.from + 1 == edge.to, "nonadjacent boundary");
        const auto & src = x.stages[edge.from];
        const auto & dst = x.stages[edge.to];
        check(src.layer_end == dst.layer_begin, "missing/duplicated layer");
        if (edge.kind == handoff::HTP_TO_ADRENO) { saw_na = true; }
        if (edge.kind == handoff::PHONE_TO_PC) { saw_pg = true; }
        if (src.cycle_id != dst.cycle_id) { saw_across_cycle = true; }
    }
    check(saw_na && saw_pg && saw_across_cycle, "handoff classification incorrect");
    check(x.stages.back().chunk_tokens == 32, "XC override ignored");
}

static void test_all_four_tail_options_and_optional_backends() {
    const auto g = compile(base(4, {{4, 0, 0, 0}}));
    const auto c = compile(base(4, {{0, 4, 0, 0}}));
    const auto n = compile(base(4, {{0, 0, 4, 0}}));
    const auto a = compile(base(4, {{0, 0, 0, 4}}));
    check(g.valid() && c.valid() && n.valid() && a.valid(),
          "a single backend must be legal in the Stage contract");
    check(g.final_head_needs_d2h() && !g.final_head_needs_p2c(), "CUDA Head return");
    check(!c.final_head_needs_d2h() && !c.final_head_needs_p2c(), "CPU Head return");
    check(n.final_head_needs_p2c() && a.final_head_needs_p2c(),
          "Phone must return hidden activation for CPU Head");
    check(compile(base(4, {{0, 0, 2, 0}, {0, 0, 0, 2}})).valid(),
          "N->A across cycles is a valid stage ordering");
    check(compile(base(4, {{2, 0, 0, 0}, {0, 2, 0, 0}})).valid(),
          "G->C across cycles is a valid final tail");
    check(classify_handoff(backend::C, backend::N) == handoff::PC_TO_PHONE,
          "C->N route missing");
    check(classify_handoff(backend::G, backend::N) == handoff::PC_TO_PHONE,
          "G->N route missing");
    check(classify_handoff(backend::N, backend::A) == handoff::HTP_TO_ADRENO,
          "N->A local handoff missing");
    check(classify_handoff(backend::A, backend::N) == handoff::ADRENO_TO_HTP,
          "A->N must not be silently zero copy");
}

static void test_invalid_shapes() {
    auto bad = base(8, {{3, 0, 0, 0}});
    check(!compile(bad).valid(), "incomplete coverage accepted");
    bad = base(8, {{9, 0, 0, 0}});
    check(!compile(bad).valid(), "overrun accepted");
    bad = base(8, {{4, 0, 0, 0}, {0, 0, 0, 0}, {4, 0, 0, 0}});
    check(!compile(bad).valid(), "empty Cycle accepted");
    bad = base(8, {{4, -1, 0, 0}, {5, 0, 0, 0}});
    check(!compile(bad).valid(), "negative backend depth accepted");
    bad = base(8, {{8, 0, 0, 0}});
    bad.version = 1;
    check(!compile(bad).valid(), "legacy contract version accepted");
}

static void test_independent_profile_prefix() {
    for (auto b : {backend::G, backend::C, backend::N, backend::A}) {
        profile_request request {};
        request.device = b;
        request.model_layers = 40;
        request.layers = 4;
        request.chunk_tokens = 64;
        request.kv_tokens = 512;
        request.repeats = 10;
        check(validate(request).empty(), "independent first-N probe rejected");
        request.layers = 41;
        check(!validate(request).empty(), "out-of-model prefix accepted");
        request.layers = 4;
        request.kv_tokens = 16;
        check(!validate(request).empty(), "unreal KV read span accepted");
        request.kv_tokens = 512;
        request.repeats = 0;
        check(!validate(request).empty(), "zero repeat accepted");
    }
}

static void test_adaptive_npu_memory() {
    npu_memory_request m {};
    m.phone_budget_bytes = 1500;
    m.max_htp_allocation_bytes = 300;
    m.npu_kv_bytes = 100;
    m.npu_graph_workspace_bytes = 60;
    m.host_stage_bytes = 30;
    m.adreno_resident_bytes = 80;
    m.other_shared_bytes = 30;
    m.stage_weight_bytes = {210, 215, 200, 205};
    auto d = choose_npu_weight_policy(m);
    check(d.feasible && d.policy == npu_weight_policy::RESIDENT,
          "four HTP stages should be fully resident when memory permits");
    check(d.all_npu_weight_bytes == 830,
          "resident bytes must include every selected HTP stage");
    check(d.resident_total_bytes == 1130 &&
          d.streaming_total_bytes == 730,
          "memory footprint must charge shared KV and two bank maxima");

    m.phone_budget_bytes = 900;
    d = choose_npu_weight_policy(m);
    check(d.feasible && d.policy == npu_weight_policy::STREAMING_2BANK,
          "full residency must spill to two banks only if needed");
    check(d.selected_total_bytes == 730 &&
          d.bank_weight_bytes[0] == 210 &&
          d.bank_weight_bytes[1] == 215,
          "streaming must use rank-parity bank maxima");

    m.phone_budget_bytes = 729;
    d = choose_npu_weight_policy(m);
    check(!d.feasible && d.reason == std::string("PHONE_BUDGET_EXCEEDED"),
          "Plan must reject a joint Phone memory budget overrun");

    m.phone_budget_bytes = 1500;
    m.max_htp_allocation_bytes = 209;
    d = choose_npu_weight_policy(m);
    check(!d.feasible &&
          d.reason == std::string("HTP_STAGE_ALLOCATION_EXCEEDED"),
          "single-region HTP allocator limit must be checked");

    m.max_htp_allocation_bytes = 300;
    m.stage_weight_bytes = {210};
    d = choose_npu_weight_policy(m);
    check(d.feasible && d.policy == npu_weight_policy::RESIDENT,
          "one NPU Region must not require dual Banks");

    m.stage_weight_bytes.clear();
    d = choose_npu_weight_policy(m);
    check(d.feasible && d.policy == npu_weight_policy::DISABLED,
          "NPU0 must not allocate any HTP weight Bank");

    m.phone_budget_bytes = 1;
    d = choose_npu_weight_policy(m);
    check(!d.feasible, "NPU0 still shares the Phone's non-NPU budget");
}

int main() {
    test_four_backend_layouts();
    test_all_four_tail_options_and_optional_backends();
    test_invalid_shapes();
    test_independent_profile_prefix();
    test_adaptive_npu_memory();
    std::puts("test-region-ring-v2-stage PASS");
    return 0;
}
