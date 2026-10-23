#include "llama-hybrid-cycle-p2c-ffn.h"

#include <chrono>
#include <cstdio>
#include <cstdint>

llama_hybrid_cycle_p2c_ffn_slot::~llama_hybrid_cycle_p2c_ffn_slot() {
    release();
}

// The original V5 P2C Phone FFN and direct Phone TX slot path, compiled as a
// physical libllama executor. RPC registration/TX are injected plugin hooks.
void llama_hybrid_cycle_p2c_ffn_slot::release() {
        executor.reset(); // Free the remote buffer before its graph context.
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        inputs = {};
        prepared_tokens = 0;
        registered_graph_uid = 0;
        graph_build_ms = 0.0;
        backend_prepare_ms = 0.0;
        last_execute_call_ms = 0.0;
        last_send_partial_rpc_ms = 0.0;
    }

bool llama_hybrid_cycle_p2c_ffn_slot::prepare(ggml_backend_t phone,
                 const llama_hybrid_cycle_ffn_shard_weights & shard,
                 int64_t hidden, int64_t tokens, int64_t topk,
                 int64_t experts, bool phone_local_chain) {
        release();
        const auto build_begin = std::chrono::steady_clock::now();
        ggml_init_params params {};
        params.mem_size = 2 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) return false;

        // Exactly the same physical FFN shard graph as the uncached
        // cycle_real_execute_ffn() path (no residual ADD / Meta allreduce).
        inputs = llama_hybrid_cycle_make_ffn_graph_inputs(
            ctx, hidden, tokens, topk);
        if (!llama_hybrid_cycle_build_moe_ffn_graph(
                ctx, inputs, shard, hidden, tokens, topk)) {
            release();
            return false;
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
        ggml_build_forward_expand(graph, inputs.ffn_output);
        const uint64_t phone_local_graph_uid = phone_local_chain ?
            llama_hybrid_cycle_graph_replay_uid() : 0;
        if (phone_local_chain) {
            // A fresh UID owns precisely this retained Graph/Buffer Slot.
            ggml_graph_set_uid(graph, phone_local_graph_uid);
        }
        const auto build_end = std::chrono::steady_clock::now();
        const bool ok = executor.prepare(
            phone, ctx, graph, inputs, shard, hidden, tokens, topk, experts);
        const auto prepare_end = std::chrono::steady_clock::now();
        graph_build_ms = std::chrono::duration<double, std::milli>(
            build_end - build_begin).count();
        backend_prepare_ms = std::chrono::duration<double, std::milli>(
            prepare_end - build_end).count();
        if (!ok) {
            release();
            return false;
        }
        if (phone_local_chain) {
            if (!rpc_ops.register_graph || !rpc_ops.register_graph(phone, graph)) {
                release();
                return false;
            }
            registered_graph_uid = phone_local_graph_uid;
        }
        prepared_tokens = tokens;
        return true;
    }

bool llama_hybrid_cycle_p2c_ffn_slot::execute_and_send(ggml_backend_t phone,
                          const llama_hybrid_cycle_ready_graph & ready,
                          const llama_hybrid_cycle_live_wire & wire,
                          llama_hybrid_cycle_ffn_phase_profile * profile) {
        if (!ctx || prepared_tokens <= 0 || !inputs.ffn_output) return false;
        // Each chunk has fresh Norm, TopK ids and TopK weights. The executor
        // copies these from the corresponding Phone Attention buffers.
        const auto execute_begin = std::chrono::steady_clock::now();
        const bool computed = executor.execute(
            nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, &ready);
        const auto execute_done = std::chrono::steady_clock::now();
        last_execute_call_ms =
            std::chrono::duration<double, std::milli>(
                execute_done - execute_begin).count();
        if (profile) {
            profile->backend_execute_call_ms += last_execute_call_ms;
        }
        last_send_partial_rpc_ms = 0.0;
        if (!computed) return false;
        // This RPC command synchronously reads GPU output and copies it
        // into the independent Phone TX slot before acknowledging success.
        // The slot can then be overwritten for another chunk while TX runs.
        const bool sent = rpc_ops.send_partial && rpc_ops.send_partial(
            phone, wire.layer, wire.chunk, wire.seq, wire.graph_id,
            inputs.ffn_output);
        last_send_partial_rpc_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - execute_done).count();
        return sent;
    }
