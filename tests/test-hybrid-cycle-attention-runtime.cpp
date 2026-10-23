#include "llama-hybrid-cycle-attention.h"
#include "llama-hybrid-cycle-wire.h"
#include "llama-model.h"

#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

int main() {
    // Exercise the ABI provided by libllama rather than the old inline V5
    // test implementation. UIDs must never repeat within a Phone RPC session.
    std::set<uint64_t> ids;
    for (int i = 0; i < 1024; ++i) {
        const auto uid = llama_hybrid_cycle_graph_replay_uid();
        assert(uid != 0);
        assert(ids.insert(uid).second);
    }

    llama_layer layer {};
    llama_hparams hp {};
    llama_hybrid_cycle_attention_ready ready;
    double call_ms = 0.0;
    const std::vector<float> no_input;
    assert(!llama_hybrid_cycle_execute_attention_pre(
        nullptr, layer, hp, no_input, 2048, 16, 8, 128,
        ready, call_ms));
    assert(ready.phone_compute_buffer == nullptr);
    ready.release_phone_graph();

    // Invalid READY shape must be rejected before any transport backend
    // proc is invoked (including when Phone is unavailable).
    llama_hybrid_cycle_live_wire wire {4, 0, 1, 1};
    std::vector<float> no_partial;
    assert(!llama_hybrid_cycle_send_c2p_frame(
        nullptr, wire, INFERENCE_CYCLE_READY, ready, no_partial,
        0, 0, 0));
    assert(!llama_hybrid_cycle_send_c2p_frame(
        nullptr, wire, INFERENCE_CYCLE_PARTIAL, ready, no_partial,
        0, 0, 0));
    return 0;
}
