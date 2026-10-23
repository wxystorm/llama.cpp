#include "llama-hybrid-cycle-attention.h"
#include "llama-hybrid-cycle-wire.h"
#include "llama-hybrid-cycle-phone-executor.h"
#include "llama-hybrid-cycle-p2c-ffn.h"
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

    // Link and exercise the actual Phone MAIN/FFN/Reduce slot from
    // libllama. Invalid/empty slots must fail closed without running RPC.
    llama_hybrid_cycle_phone_ffn_slot phone_slot;
    llama_hybrid_cycle_phone_ffn_slot next_slot;
    assert(!phone_slot.execute_full_local(ready));
    assert(!phone_slot.execute_full_server(
        nullptr, ready, 5, 0, 1));
    assert(!phone_slot.execute_full_server_chain(
        nullptr, ready, 5, 0, 1, 6, 2, 3, ready));
    assert(!phone_slot.execute_full_server_pair(
        nullptr, ready, 5, 0, 1, 6, 2, 3, ready, next_slot));
    assert(!phone_slot.execute_reduce(nullptr));
    phone_slot.registered_full_graph_uid = 123;
    phone_slot.release();
    assert(phone_slot.registered_full_graph_uid == 0);
    assert(phone_slot.output == nullptr);
    llama_hybrid_cycle_p2c_ffn_slot p2c_slot;
    llama_hybrid_cycle_ready_graph no_ready {};
    llama_hybrid_cycle_live_wire p2c_wire {7, 0, 1, 1};
    assert(!p2c_slot.execute_and_send(
        nullptr, no_ready, p2c_wire, nullptr));
    p2c_slot.registered_graph_uid = 321;
    p2c_slot.release();
    assert(p2c_slot.registered_graph_uid == 0);


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
