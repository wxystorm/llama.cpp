#include "llama-hybrid-cycle-phone-executor.h"
#include "llama-hybrid-cycle-log.h"

#include "../ggml/src/ggml-backend-impl.h" // GPU local tensor-to-tensor copy

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>

llama_hybrid_cycle_phone_ffn_slot::~llama_hybrid_cycle_phone_ffn_slot() {
    release();
}

// Original V5 Phone Full and C2P graph execution; only RPC calls are
// injected so libllama can be linked without the ggml-rpc plugin.
void llama_hybrid_cycle_phone_ffn_slot::release() {
        executor.reset();
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        residual = nullptr;
        pc_partial = nullptr;
        output = nullptr;
        reduce_graph = nullptr;
        inputs = {};
        is_full_layer = false;
        fused_full_graph = false;
        rpc_full_replay = false;
        registered_full_graph_uid = 0;
        registered_c2p_ffn_uid = 0;
        registered_c2p_reduce_uid = 0;
        prepared_tokens = 0;
        reduce_graph_call_ms = 0.0;
        reduce_backend_sync_ms = 0.0;
    }

bool llama_hybrid_cycle_phone_ffn_slot::prepare(
            ggml_backend_t phone,
            const llama_hybrid_cycle_ffn_shard_weights & shard,
            int64_t hidden, int64_t tokens, int64_t topk, int64_t experts,
            bool full_layer, bool fuse_full,
            bool enable_rpc_replay,
            bool enable_phone_full_local,
            bool enable_c2p_local) {
        const auto build_begin = std::chrono::steady_clock::now();
        release();
        ggml_init_params params {};
        params.mem_size = 2 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }
        is_full_layer = full_layer;
        fused_full_graph = full_layer && fuse_full;
        rpc_full_replay = (full_layer || enable_c2p_local) &&
            (enable_rpc_replay || enable_c2p_local);
        if (enable_c2p_local && full_layer) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_V5] status=FAIL reason=C2P_NOT_FULL_LAYER\n");
            release();
            return false;
        }
        if (enable_phone_full_local &&
            (!is_full_layer || !fused_full_graph || !rpc_full_replay)) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_FULL_LOCAL] phase=PREPARE "
                "status=FAIL reason=REQUIRES_FUSED_FULL_GRAPH_AND_REPLAY\n");
            release();
            return false;
        }
        inputs = llama_hybrid_cycle_make_ffn_graph_inputs(ctx, hidden, tokens, topk);
        residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        if (!full_layer) {
            pc_partial = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        }
        if (!llama_hybrid_cycle_build_moe_ffn_graph(
                ctx, inputs, shard, hidden, tokens, topk)) {
            release();
            return false;
        }
        auto * ffn_graph = ggml_new_graph_custom(ctx, 512, false);

        // One physical owner builds the SAME V5 Reduce graph through
        // libllama; the split Reduce must never recompute FFN.
        const auto reduced = llama_hybrid_cycle_build_reduce_graph(
            ctx, ffn_graph, residual, inputs.ffn_output, pc_partial,
            full_layer, fused_full_graph);
        if (!reduced.output) {
            release();
            return false;
        }
        output = reduced.output;
        reduce_graph = reduced.reduce_graph;
        const uint64_t phone_full_graph_uid = rpc_full_replay ?
            llama_hybrid_cycle_graph_replay_uid() : 0;
        const uint64_t c2p_reduce_graph_uid = rpc_full_replay && reduce_graph ?
            llama_hybrid_cycle_graph_replay_uid() : 0;
        if (rpc_full_replay) {
            // Replayed graph references exactly these persistent FFN,
            // residual and output buffers. A freed graph must NEVER reuse
            // its UID; only this object's subsequent Chunk may replay it.
            ggml_graph_set_uid(ffn_graph, phone_full_graph_uid);
            if (reduce_graph) {
                ggml_graph_set_uid(reduce_graph, c2p_reduce_graph_uid);
            }
        }
        const auto build_end = std::chrono::steady_clock::now();
        if (!executor.prepare(
                phone, ctx, ffn_graph, inputs, shard,
                hidden, tokens, topk, experts)) {
            std::fprintf(stderr,
                "[CYCLE_C2P_GPU] status=FAIL reason=FFN_PREPARE\n");
            release();
            return false;
        }
        // Registration creates the server-side FFN/Reduce graph ONCE per
        // physical slot/shape. It does not run with uninitialized inputs.
        // The existing pool keeps both graph UID and GPU buffers alive
        // until the final downstream Attention has consumed l_out.
        if (enable_phone_full_local) {
            if (!rpc_ops.register_graph || !rpc_ops.register_graph(phone, ffn_graph)) {
                std::fprintf(stderr,
                    "[CYCLE_PHONE_FULL_LOCAL] phase=REGISTER "
                    "status=FAIL reason=REMOTE_GRAPH_REGISTER\n");
                release();
                return false;
            }
            registered_full_graph_uid = phone_full_graph_uid;
        }
        if (enable_c2p_local) {
            // Register once per physical slot; both graphs reference the
            // existing retained buffers. Registration does not execute.
            if (!reduce_graph || !rpc_ops.register_graph ||
                !rpc_ops.register_graph(phone, ffn_graph) ||
                !rpc_ops.register_graph(phone, reduce_graph)) {
                std::fprintf(stderr,
                    "[CYCLE_PHONE_V5] phase=REGISTER_C2P status=FAIL\n");
                release();
                return false;
            }
            registered_c2p_ffn_uid = phone_full_graph_uid;
            registered_c2p_reduce_uid = c2p_reduce_graph_uid;
        }
        const auto alloc_end = std::chrono::steady_clock::now();
        graph_build_ms = std::chrono::duration<double, std::milli>(
            build_end - build_begin).count();
        gpu_alloc_ms = std::chrono::duration<double, std::milli>(
            alloc_end - build_end).count();
        prepared_tokens = tokens;
        if (llama_hybrid_cycle_verbose()) std::printf(
            "[CYCLE_GPU_PREPARE] phase=PHONE_FFN graph_build_ms=%.3f "
            "gpu_alloc_ms=%.3f source=PC_CLIENT status=OK\n",
            graph_build_ms, gpu_alloc_ms);
        return true;
    }

bool llama_hybrid_cycle_phone_ffn_slot::execute_full_local(const llama_hybrid_cycle_attention_ready & ready) {
        if (!is_full_layer || !ready.phone_graph.residual ||
            !ready.phone_graph.ffn_norm ||
            !ready.phone_graph.topk_ids ||
            !ready.phone_graph.topk_weights ||
            !residual || !residual->buffer ||
            !ggml_backend_buffer_copy_tensor(
                ready.phone_graph.residual, residual)) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_FULL] stage=INPUT status=FAIL "
                "reason=LOCAL_GPU_HANDOFF\n");
            return false;
        }
        const std::vector<float> empty_f32;
        const std::vector<int32_t> empty_i32;
        return executor.execute(
            empty_f32.data(), 0, empty_i32.data(), 0,
            empty_f32.data(), 0, nullptr, 0, &ready.phone_graph);
    }

bool llama_hybrid_cycle_phone_ffn_slot::execute_full_server(
            ggml_backend_t phone,
            const llama_hybrid_cycle_attention_ready & ready,
            int layer, int chunk, uint64_t seq) {
        if (!is_full_layer || !fused_full_graph ||
            registered_full_graph_uid == 0 || !residual ||
            !inputs.norm || !inputs.topk_ids || !inputs.topk_weights) {
            return false;
        }
        const ggml_tensor * const sources[4] = {
            ready.phone_graph.residual,
            ready.phone_graph.ffn_norm,
            ready.phone_graph.topk_ids,
            ready.phone_graph.topk_weights,
        };
        const ggml_tensor * const destinations[4] = {
            residual, inputs.norm, inputs.topk_ids, inputs.topk_weights,
        };
        return rpc_ops.run_full && rpc_ops.run_full(
            phone, layer, chunk, seq, registered_full_graph_uid,
            sources, destinations);
    }

bool llama_hybrid_cycle_phone_ffn_slot::execute_full_server_chain(
            ggml_backend_t phone,
            const llama_hybrid_cycle_attention_ready & ready,
            int layer, int chunk, uint64_t seq,
            int next_layer, uint64_t next_seq, uint64_t next_graph_id,
            const llama_hybrid_cycle_attention_ready & next_ready) {
        if (!is_full_layer || !fused_full_graph ||
            registered_full_graph_uid == 0 || !output ||
            next_ready.prepared_attention_graph_uid == 0 ||
            !next_ready.prepared_attention_input ||
            !residual || !inputs.norm || !inputs.topk_ids ||
            !inputs.topk_weights) {
            return false;
        }
        const ggml_tensor * const sources[4] = {
            ready.phone_graph.residual,
            ready.phone_graph.ffn_norm,
            ready.phone_graph.topk_ids,
            ready.phone_graph.topk_weights,
        };
        const ggml_tensor * const destinations[4] = {
            residual, inputs.norm, inputs.topk_ids, inputs.topk_weights,
        };
        return rpc_ops.run_full_chain && rpc_ops.run_full_chain(
            phone, layer, chunk, seq, registered_full_graph_uid,
            sources, destinations,
            next_layer, next_seq, next_graph_id,
            next_ready.prepared_attention_graph_uid, output,
            next_ready.prepared_attention_input);
    }

bool llama_hybrid_cycle_phone_ffn_slot::execute_full_server_pair(
            ggml_backend_t phone,
            const llama_hybrid_cycle_attention_ready & ready,
            int layer, int chunk, uint64_t seq,
            int next_layer, uint64_t next_seq, uint64_t next_graph_id,
            const llama_hybrid_cycle_attention_ready & next_ready,
            const llama_hybrid_cycle_phone_ffn_slot & next_full) {
        if (!is_full_layer || !fused_full_graph ||
            registered_full_graph_uid == 0 || !output ||
            !next_full.is_full_layer || !next_full.fused_full_graph ||
            next_full.registered_full_graph_uid == 0 ||
            next_ready.prepared_attention_graph_uid == 0 ||
            !next_ready.prepared_attention_input ||
            !residual || !inputs.norm || !inputs.topk_ids ||
            !inputs.topk_weights || !next_full.residual ||
            !next_full.inputs.norm || !next_full.inputs.topk_ids ||
            !next_full.inputs.topk_weights) {
            return false;
        }
        const ggml_tensor * const sources[4] = {
            ready.phone_graph.residual,
            ready.phone_graph.ffn_norm,
            ready.phone_graph.topk_ids,
            ready.phone_graph.topk_weights,
        };
        const ggml_tensor * const destinations[4] = {
            residual, inputs.norm, inputs.topk_ids, inputs.topk_weights,
        };
        const ggml_tensor * const next_sources[4] = {
            next_ready.phone_graph.residual,
            next_ready.phone_graph.ffn_norm,
            next_ready.phone_graph.topk_ids,
            next_ready.phone_graph.topk_weights,
        };
        const ggml_tensor * const next_destinations[4] = {
            next_full.residual, next_full.inputs.norm,
            next_full.inputs.topk_ids, next_full.inputs.topk_weights,
        };
        return rpc_ops.run_full_pair && rpc_ops.run_full_pair(
            phone, layer, chunk, seq, registered_full_graph_uid,
            sources, destinations, next_layer, next_seq, next_graph_id,
            next_ready.prepared_attention_graph_uid, output,
            next_ready.prepared_attention_input,
            next_full.registered_full_graph_uid,
            next_sources, next_destinations);
    }

bool llama_hybrid_cycle_phone_ffn_slot::execute_ffn() {
        const auto begin = std::chrono::steady_clock::now();
        const bool ok = executor.execute_device_ready();
        ffn_graph_submit_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        if (!ok || llama_hybrid_cycle_verbose()) std::printf("[CYCLE_GPU_SUBMIT] phase=PHONE_FFN "
                    "graph_submit_ms=%.3f gpu_compute_ms=SERVER_ONLY "
                    "status=%s\n",
                    ffn_graph_submit_ms, ok ? "OK" : "FAIL");
        return ok;
    }

bool llama_hybrid_cycle_phone_ffn_slot::execute_reduce(ggml_backend_t phone) {
        // A full Phone GPU layer has no PC FFN shard and deliberately does
        // not allocate pc_partial. C2P needs the PC shard for its two-input
        // FFN sum; FULL_PHONE reduces only its local FFN and residual.
        if (!phone || !reduce_graph || !output ||
            (!is_full_layer && !pc_partial) ||
            !inputs.ffn_output || !residual) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_REDUCE] mode=%s status=FAIL "
                "reason=MISSING_GRAPH_INPUT pc_partial=%d output=%d\n",
                is_full_layer ? "PHONE_FULL" : "TENSOR_C2P",
                pc_partial != nullptr ? 1 : 0, output != nullptr ? 1 : 0);
            return false;
        }
        const auto begin = std::chrono::steady_clock::now();
        const bool ok = ggml_backend_graph_compute(phone, reduce_graph) ==
                        GGML_STATUS_SUCCESS;
        const auto graph_done = std::chrono::steady_clock::now();
        if (ok) {
            ggml_backend_synchronize(phone);
        }
        const auto done = std::chrono::steady_clock::now();
        reduce_graph_call_ms =
            std::chrono::duration<double, std::milli>(
                graph_done - begin).count();
        reduce_backend_sync_ms =
            std::chrono::duration<double, std::milli>(
                done - graph_done).count();
        reduce_graph_submit_ms = reduce_graph_call_ms +
            reduce_backend_sync_ms;
        if (!ok || llama_hybrid_cycle_verbose()) std::printf("[CYCLE_GPU_SUBMIT] phase=PHONE_REDUCE mode=%s "
                    "graph_submit_ms=%.3f gpu_compute_ms=SERVER_ONLY "
                    "status=%s\n",
                    is_full_layer ? "PHONE_FULL" : "TENSOR_C2P",
                    reduce_graph_submit_ms, ok ? "OK" : "FAIL");
        return ok;
    }
