#include "llama-hybrid-cycle-ffn-shard.h"

#include "ggml-alloc.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Extracted from the physical V5 probe; the graph, FFN executor, diagnostic
// nodes, and backend synchronization remain identical. Only the RPC send is
// injected rather than statically linked.
bool llama_hybrid_cycle_execute_ffn_shard(
        ggml_backend_t backend,
        const llama_hybrid_cycle_ffn_shard_weights & shard,
        const std::vector<float> & norm,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        int64_t hidden, int64_t tokens,
        int64_t topk, int64_t experts,
        std::vector<float> & partial,
        std::vector<float> * raw_down_out,
        llama_hybrid_cycle_ffn_stages * stages_out,
        const llama_hybrid_cycle_live_wire * wire,
        const llama_hybrid_cycle_ready_graph * local_phone_ready,
        llama_hybrid_cycle_ffn_phase_profile * phase_profile,
        const llama_hybrid_cycle_ffn_rpc_ops & rpc_ops) {
    if (!backend || hidden <= 0 || tokens <= 0 || topk <= 0 ||
        experts < topk || !shard.available() ||
        (wire && !rpc_ops.send_partial)) {
        return false;
    }
    const auto graph_build_begin = std::chrono::steady_clock::now();
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    auto in = llama_hybrid_cycle_make_ffn_graph_inputs(
        ctx, hidden, tokens, topk);
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * activation = nullptr;
    ggml_tensor * down = nullptr;
    if (!llama_hybrid_cycle_build_moe_ffn_graph(
            ctx, in, shard, hidden, tokens, topk,
            &gate, &up, &activation, &down)) {
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, in.ffn_output);
    const auto graph_build_done = std::chrono::steady_clock::now();
    if (std::getenv("LLAMA_HYBRID_CYCLE_STAGE_PROFILE") != nullptr) {
        std::printf("[CYCLE_FFN_GRAPH_BUILD] backend=%s "
                    "graph_build_ms=%.3f nodes=%d\n",
                    ggml_backend_name(backend),
                    std::chrono::duration<double, std::milli>(
                        graph_build_done - graph_build_begin).count(),
                    ggml_graph_n_nodes(graph));
    }

    bool ok = false;
    {
        const char * backend_name = ggml_backend_name(backend);
        std::printf("[CYCLE_REAL_FFN_SHARD] backend=%s channels=%lld nodes=%d "
                    "stage=PREPARE_BEGIN\n",
                    backend_name, (long long)shard.up->ne[1], ggml_graph_n_nodes(graph));
        llama_hybrid_cycle_ffn_executor executor;
        const auto prepare_begin = std::chrono::steady_clock::now();
        const bool prepared = executor.prepare(
            backend, ctx, graph, in, shard, hidden, tokens, topk, experts);
        const auto prepare_done = std::chrono::steady_clock::now();
        if (phase_profile) {
            ++phase_profile->count;
            phase_profile->graph_build_ms +=
                std::chrono::duration<double, std::milli>(
                    graph_build_done - graph_build_begin).count();
            phase_profile->backend_prepare_ms +=
                std::chrono::duration<double, std::milli>(
                    prepare_done - prepare_begin).count();
        }
        if (!prepared) {
            std::fprintf(stderr,
                "[CYCLE_REAL_FFN_SHARD] backend=%s stage=PREPARE_FAIL "
                "weight_buft=%s compute_buft=%s\n",
                backend_name,
                shard.up->buffer != nullptr ?
                    ggml_backend_buft_name(ggml_backend_buffer_get_type(shard.up->buffer)) :
                    "(unallocated)",
                ggml_backend_buft_name(ggml_backend_get_default_buffer_type(backend)));
        } else {
            if (wire == nullptr) {
                partial.resize(size_t(hidden * tokens));
            }
            const auto execute_begin = std::chrono::steady_clock::now();
            ok = executor.execute(
                norm.data(), norm.size(),
                ids.data(), ids.size(),
                routing.data(), routing.size(),
                wire != nullptr ? nullptr : partial.data(),
                wire != nullptr ? 0 : partial.size(),
                local_phone_ready);
            if (phase_profile) {
                phase_profile->backend_execute_call_ms +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - execute_begin).count();
            }
            if (ok && stages_out != nullptr) {
                const int64_t elems = shard.up->ne[1] * topk * tokens;
                const auto read_stage = [&](const char * stage,
                                            ggml_tensor * node,
                                            std::vector<float> & dst) {
                    if (!node || node->type != GGML_TYPE_F32 ||
                        !ggml_is_contiguous(node) ||
                        ggml_nelements(node) != elems) {
                        std::fprintf(stderr,
                            "[CYCLE_FFN_STAGE_CAPTURE] backend=%s stage=%s "
                            "status=FAIL reason=TYPE_OR_SHAPE\n",
                            backend_name, stage);
                        return false;
                    }
                    dst.resize(size_t(elems));
                    ggml_backend_tensor_get(
                        node, dst.data(), 0, ggml_nbytes(node));
                    std::printf(
                        "[CYCLE_FFN_STAGE_CAPTURE] backend=%s stage=%s "
                        "elements=%lld status=OK\n",
                        backend_name, stage, (long long)elems);
                    return true;
                };
                ok = read_stage("GATE", gate, stages_out->gate) &&
                     read_stage("UP", up, stages_out->up) &&
                     read_stage("SWIGLU", activation, stages_out->swiglu);
            }
            // Diagnostic only: capture the unweighted per-expert Down
            // projection [hidden, topk, tokens] from this physical shard.
            // This is the exact same tensor as canonical ffn_moe_down
            // before Router weights and the expert reduction.
            if (ok && raw_down_out != nullptr) {
                if (down->type != GGML_TYPE_F32 ||
                    !ggml_is_contiguous(down) ||
                    ggml_nelements(down) != hidden * topk * tokens) {
                    std::fprintf(stderr,
                        "[CYCLE_FFN_INTERMEDIATE] backend=%s status=FAIL "
                        "reason=DOWN_OUTPUT_SHAPE\n", backend_name);
                    ok = false;
                } else {
                    raw_down_out->resize(size_t(hidden * topk * tokens));
                    ggml_backend_tensor_get(
                        down, raw_down_out->data(), 0, ggml_nbytes(down));
                }
            }
            if (ok && wire != nullptr) {
                // The remote output has no ordinary RPC readback: the Phone
                // owns it until its Cycle TX worker copies Send #2.
                ok = rpc_ops.send_partial &&
                    rpc_ops.send_partial(
                        backend, wire->layer, wire->chunk,
                        wire->seq, wire->graph_id, in.ffn_output);
                if (!ok) {
                    std::fprintf(stderr,
                        "[CYCLE_REAL_WIRE] stage=PARTIAL status=FAIL "
                        "reason=PHONE_REMOTE_SEND\n");
                }
            }
            std::printf("[CYCLE_REAL_FFN_SHARD] backend=%s stage=COMPUTE_END "
                        "elements=%zu status=%s\n",
                        backend_name, partial.size(), ok ? "OK" : "FAIL");
        }
    }
    ggml_free(ctx);
    return ok;
}
