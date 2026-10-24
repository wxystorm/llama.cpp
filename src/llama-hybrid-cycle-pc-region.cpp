#include "llama-hybrid-cycle-pc-region.h"

#include "llama-hybrid-cycle-log.h"
#include "llama-model.h"
#include "llama-hparams.h"

#include "ggml-alloc.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

bool llama_hybrid_cycle_execute_pc_full_region(
        const llama_model & model,
        llama_hybrid_cycle_physical_resource_pool & resources,
        int first_layer,
        int layer_count,
        int64_t token_start,
        int64_t tokens,
        const std::vector<float> & input,
        std::vector<float> & output,
        std::string & reason,
        const std::function<bool()> & input_consumed,
        llama_hybrid_cycle_pc_region_timing * timing) {
    output.clear();
    reason.clear();

    const auto & layout = resources.execution_layout();
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    const int64_t experts = model.hparams.n_expert;
    if (!layout.enabled() || first_layer < 0 || layer_count < 2 ||
        int64_t(first_layer) + layer_count > int64_t(layout.kinds.size()) ||
        int64_t(first_layer) + layer_count > int64_t(model.layers.size()) ||
        tokens < 1 || tokens > 128 || token_start < 0 ||
        hidden <= 0 || topk <= 0 || experts < topk ||
        uint64_t(hidden) > std::numeric_limits<size_t>::max() /
                           uint64_t(tokens) ||
        input.size() != size_t(hidden) * size_t(tokens)) {
        reason = "PC_REGION_INVALID_SHAPE_OR_RANGE";
        return false;
    }

    const auto & hp = model.hparams;
    if (hp.f_norm_rms_eps <= 0.0f ||
        hp.rope_type != LLAMA_ROPE_TYPE_NEOX ||
        (hp.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_NONE &&
         hp.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_LINEAR) ||
        std::fabs(hp.rope_freq_scale_train - 1.0f) > 1.0e-6f ||
        hp.yarn_ext_factor > 0.0f ||
        std::fabs(hp.rope_attn_factor - 1.0f) > 1.0e-6f ||
        hp.use_alibi || hp.f_clamp_kqv != 0.0f ||
        hp.attn_soft_cap || hp.f_max_alibi_bias != 0.0f) {
        reason = "PC_REGION_UNSUPPORTED_ATTENTION_CONFIG";
        return false;
    }

    // Pin the whole contiguous Region in the bounded LRU before building any
    // graph. The PC_CUDA queue has one worker, so no other residency fault can
    // evict these mirrors until this Region call returns.
    std::vector<const llama_hybrid_cycle_physical_layer_resources *> region;
    region.reserve(size_t(layer_count));
    ggml_backend_t cuda = nullptr;
    for (int offset = 0; offset < layer_count; ++offset) {
        const int layer = first_layer + offset;
        if (layout.kinds[size_t(layer)] != llama_hybrid_cycle_layer_kind::PC_CPU ||
            !llama_hybrid_cycle_full_layer_on_cuda(layout, layer)) {
            reason = "PC_REGION_NON_CUDA_FULL layer=" + std::to_string(layer);
            return false;
        }
        std::string residency_reason;
        if (!resources.ensure_pc_full_cuda_resident(layer, residency_reason)) {
            reason = "PC_REGION_RESIDENCY layer=" + std::to_string(layer) +
                     " detail=" + residency_reason;
            return false;
        }
        const auto * resource = resources.layer(layer);
        if (!resource || !resource->attn_backend || !resource->cuda_weights ||
            !resource->kv ||
            resource->physical_layer != layer ||
            resource->kv->committed_tokens != token_start ||
            token_start + tokens > resource->kv->max_tokens) {
            reason = "PC_REGION_RESOURCE_OR_KV layer=" + std::to_string(layer);
            return false;
        }
        if (!cuda) cuda = resource->attn_backend;
        if (resource->attn_backend != cuda) {
            reason = "PC_REGION_BACKEND_MISMATCH";
            return false;
        }
        region.push_back(resource);
    }
    if (!cuda) {
        reason = "PC_REGION_CUDA_MISSING";
        return false;
    }

    const auto graph_build_begin = std::chrono::steady_clock::now();
    ggml_init_params params {};
    params.mem_size = 64ULL * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        reason = "PC_REGION_CONTEXT_ALLOC";
        return false;
    }
    ggml_backend_buffer_t compute_buffer = nullptr;
    const auto cleanup = [&] {
        if (compute_buffer) {
            ggml_backend_buffer_free(compute_buffer);
            compute_buffer = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
    };
    const auto fail = [&](const std::string & why) {
        reason = why;
        cleanup();
        output.clear();
        return false;
    };

    const int64_t kv_tokens = token_start + tokens;
    ggml_tensor * x = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hidden, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, tokens);
    ggml_tensor * causal_mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, kv_tokens, tokens);
    if (!x || !positions || !causal_mask) {
        return fail("PC_REGION_INPUT_TENSORS");
    }
    ggml_set_input(x);
    ggml_set_input(positions);
    ggml_set_input(causal_mask);
    ggml_set_name(x, "cycle_pc_region_input");
    ggml_set_name(positions, "cycle_pc_region_positions");
    ggml_set_name(causal_mask, "cycle_pc_region_causal_mask");

    std::vector<ggml_tensor *> ordered_kv_writes;
    ordered_kv_writes.reserve(size_t(layer_count) * 2);
    ggml_tensor * current = x;

    const int ctx_orig = hp.n_ctx_orig_yarn > 0 ?
        int(hp.n_ctx_orig_yarn) : int(hp.n_ctx_train);
    const float rope_base = hp.rope_freq_base_train > 0 ?
        hp.rope_freq_base_train : 10000.0f;
    const float rope_scale = hp.rope_freq_scale_train > 0 ?
        hp.rope_freq_scale_train : 1.0f;

    for (int offset = 0; offset < layer_count; ++offset) {
        const int layer_index = first_layer + offset;
        const auto & lw = model.layers[size_t(layer_index)];
        if (lw.wqkv || lw.wq_b || lw.wk_b || lw.wv_b ||
            lw.wo_b || lw.wq_s || lw.wk_s || lw.wv_s || lw.wo_s) {
            return fail("PC_REGION_UNSUPPORTED_LAYER layer=" +
                        std::to_string(layer_index));
        }

        const auto * resource = region[size_t(offset)];
        const auto & cw = *resource->cuda_weights;
        ggml_tensor * w_attn_norm =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::ATTN_NORM];
        ggml_tensor * wq =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::WQ];
        ggml_tensor * wk =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::WK];
        ggml_tensor * wv =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::WV];
        ggml_tensor * wo =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::WO];
        ggml_tensor * wq_norm =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::Q_NORM];
        ggml_tensor * wk_norm =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::K_NORM];
        ggml_tensor * w_ffn_norm =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::FFN_NORM];
        ggml_tensor * w_router =
            cw.tensors[llama_hybrid_cycle_cuda_full_weights::ROUTER];
        const auto shard = cw.full_ffn_shard();

        if (!w_attn_norm || !wq || !wk || !wv || !wo ||
            !wq_norm || !wk_norm || !w_ffn_norm || !w_router ||
            !shard.available()) {
            return fail("PC_REGION_WEIGHT_MISSING layer=" +
                        std::to_string(layer_index));
        }

        const int64_t head_dim = wq_norm->ne[0];
        if (head_dim <= 0 || wk_norm->ne[0] != head_dim ||
            wq->ne[0] != hidden || wk->ne[0] != hidden ||
            wv->ne[0] != hidden || wq->ne[1] % head_dim != 0 ||
            wk->ne[1] % head_dim != 0 ||
            wv->ne[1] != wk->ne[1] ||
            wo->ne[0] != wq->ne[1] || wo->ne[1] != hidden ||
            w_router->ne[0] != hidden || w_router->ne[1] != experts ||
            w_attn_norm->ne[0] != hidden ||
            w_ffn_norm->ne[0] != hidden ||
            hp.n_rot_full != uint32_t(head_dim)) {
            return fail("PC_REGION_WEIGHT_SHAPE layer=" +
                        std::to_string(layer_index));
        }
        const int64_t n_heads = wq->ne[1] / head_dim;
        const int64_t n_kv_heads = wk->ne[1] / head_dim;
        if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0 ||
            resource->kv->head_dim != head_dim ||
            resource->kv->kv_heads != n_kv_heads) {
            return fail("PC_REGION_GQA_OR_KV_SHAPE layer=" +
                        std::to_string(layer_index));
        }

        auto * attn_norm = ggml_mul(
            ctx, ggml_rms_norm(ctx, current, hp.f_norm_rms_eps),
            w_attn_norm);
        auto * q = ggml_reshape_3d(
            ctx, ggml_mul_mat(ctx, wq, attn_norm),
            head_dim, n_heads, tokens);
        auto * k = ggml_reshape_3d(
            ctx, ggml_mul_mat(ctx, wk, attn_norm),
            head_dim, n_kv_heads, tokens);
        auto * v = ggml_reshape_3d(
            ctx, ggml_mul_mat(ctx, wv, attn_norm),
            head_dim, n_kv_heads, tokens);
        q = ggml_mul(
            ctx, ggml_rms_norm(ctx, q, hp.f_norm_rms_eps), wq_norm);
        k = ggml_mul(
            ctx, ggml_rms_norm(ctx, k, hp.f_norm_rms_eps), wk_norm);
        q = ggml_rope_ext(
            ctx, q, positions, nullptr,
            int(head_dim), GGML_ROPE_TYPE_NEOX, ctx_orig,
            rope_base, rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(
            ctx, k, positions, nullptr,
            int(head_dim), GGML_ROPE_TYPE_NEOX, ctx_orig,
            rope_base, rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_cast(ctx, k, GGML_TYPE_F16);
        v = ggml_cast(ctx, v, GGML_TYPE_F16);

        auto * dst_k = ggml_view_3d(
            ctx, resource->kv->keys,
            head_dim, n_kv_heads, tokens,
            resource->kv->keys->nb[1],
            resource->kv->keys->nb[2],
            size_t(token_start) * resource->kv->keys->nb[2]);
        auto * dst_v = ggml_view_3d(
            ctx, resource->kv->values,
            head_dim, n_kv_heads, tokens,
            resource->kv->values->nb[1],
            resource->kv->values->nb[2],
            size_t(token_start) * resource->kv->values->nb[2]);
        auto * kv_k_write = ggml_cpy(ctx, k, dst_k);
        auto * kv_v_write = ggml_cpy(ctx, v, dst_v);
        ordered_kv_writes.push_back(kv_k_write);
        ordered_kv_writes.push_back(kv_v_write);

        k = ggml_view_3d(
            ctx, resource->kv->keys,
            head_dim, n_kv_heads, kv_tokens,
            resource->kv->keys->nb[1],
            resource->kv->keys->nb[2], 0);
        v = ggml_view_3d(
            ctx, resource->kv->values,
            head_dim, n_kv_heads, kv_tokens,
            resource->kv->values->nb[1],
            resource->kv->values->nb[2], 0);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);
        auto * kq = ggml_mul_mat(ctx, k, q);
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
        auto * probs = ggml_soft_max_ext(
            ctx, kq, causal_mask,
            1.0f / std::sqrt(float(head_dim)), 0.0f);
        auto * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
        auto * kqv = ggml_mul_mat(ctx, vt, probs);
        const int64_t attention_width = wq->ne[1];
        auto * attn = ggml_cont_2d(
            ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3),
            attention_width, tokens);
        auto * attn_output = ggml_mul_mat(ctx, wo, attn);
        auto * residual = ggml_add(ctx, attn_output, current);

        auto * norm = ggml_mul(
            ctx, ggml_rms_norm(ctx, residual, hp.f_norm_rms_eps),
            w_ffn_norm);
        auto * router_logits = ggml_mul_mat(ctx, w_router, norm);
        auto * router_probs = ggml_soft_max(ctx, router_logits);
        auto * ids = ggml_cont(
            ctx, ggml_argsort_top_k(ctx, router_probs, int(topk)));
        auto * router_p3 =
            ggml_reshape_3d(ctx, router_probs, 1, experts, tokens);
        auto * routing = ggml_get_rows(ctx, router_p3, ids);
        routing = ggml_reshape_2d(ctx, routing, topk, tokens);
        auto * sum = ggml_sum_rows(ctx, routing);
        sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
        routing = ggml_div(ctx, routing, sum);
        routing = ggml_reshape_3d(ctx, routing, 1, topk, tokens);
        if (hp.expert_weights_scale != 0.0f &&
            hp.expert_weights_scale != 1.0f) {
            routing = ggml_scale(ctx, routing, hp.expert_weights_scale);
        }
        routing = ggml_cont(ctx, routing);

        llama_hybrid_cycle_ffn_graph ffn {};
        ffn.norm = norm;
        ffn.topk_ids = ids;
        ffn.topk_weights = routing;
        if (!llama_hybrid_cycle_build_moe_ffn_graph(
                ctx, ffn, shard, hidden, tokens, topk)) {
            return fail("PC_REGION_FFN_GRAPH layer=" +
                        std::to_string(layer_index));
        }
        current = ggml_cont(ctx, ggml_add(ctx, residual, ffn.ffn_output));
        ggml_set_name(current, "cycle_pc_region_l_out");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    if (!graph || !current) {
        return fail("PC_REGION_GRAPH_ALLOC");
    }
    for (auto * write : ordered_kv_writes) {
        ggml_build_forward_expand(graph, write);
    }
    ggml_build_forward_expand(graph, current);

    int unsupported = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_NONE &&
            !ggml_backend_supports_op(cuda, node)) {
            ++unsupported;
            std::fprintf(stderr,
                "[CYCLE_PC_CUDA_REGION_OP] layer_begin=%d node=%d "
                "op=%s name=%s supported=0\n",
                first_layer, i, ggml_op_name(node->op), node->name);
        }
    }
    if (unsupported != 0) {
        return fail("PC_REGION_UNSUPPORTED_OP count=" +
                    std::to_string(unsupported));
    }

    const auto graph_build_done = std::chrono::steady_clock::now();
    const auto alloc_begin = graph_build_done;
    compute_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(cuda));
    const auto alloc_done = std::chrono::steady_clock::now();
    if (!compute_buffer) {
        return fail("PC_REGION_COMPUTE_BUFFER");
    }

    std::vector<int32_t> position_data(size_t(tokens), 0);
    std::vector<ggml_fp16_t> mask_data(size_t(tokens * kv_tokens));
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t query = 0; query < tokens; ++query) {
        const int64_t global_pos = token_start + query;
        position_data[size_t(query)] = int32_t(global_pos);
        for (int64_t key = 0; key < kv_tokens; ++key) {
            mask_data[size_t(query * kv_tokens + key)] =
                key > global_pos ? drop : keep;
        }
    }

    const auto input_begin = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(
        x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(
        positions, position_data.data(), 0, ggml_nbytes(positions));
    ggml_backend_tensor_set(
        causal_mask, mask_data.data(), 0, ggml_nbytes(causal_mask));
    const auto input_done = std::chrono::steady_clock::now();

    // The Host predecessor remains session-owned, but its Ring credit can be
    // released once the Region has staged the immutable input tensor.
    if (input_consumed && !input_consumed()) {
        return fail("PC_REGION_INPUT_CONSUMED_NOTIFY");
    }

    const auto compute_begin = std::chrono::steady_clock::now();
    const bool computed =
        ggml_backend_graph_compute(cuda, graph) == GGML_STATUS_SUCCESS;
    if (computed) {
        ggml_backend_synchronize(cuda);
    }
    const auto compute_done = std::chrono::steady_clock::now();
    if (!computed) {
        return fail("PC_REGION_GRAPH_COMPUTE");
    }

    for (const auto * resource : region) {
        resource->kv->committed_tokens = kv_tokens;
    }

    if (current->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(current) ||
        ggml_nelements(current) != hidden * tokens) {
        return fail("PC_REGION_OUTPUT_SHAPE");
    }
    const auto readback_begin = std::chrono::steady_clock::now();
    output.resize(size_t(hidden) * size_t(tokens));
    ggml_backend_tensor_get(
        current, output.data(), 0, output.size() * sizeof(float));
    const auto readback_done = std::chrono::steady_clock::now();

    for (float value : output) {
        if (!std::isfinite(value)) {
            return fail("PC_REGION_NONFINITE_OUTPUT");
        }
    }

    if (timing) {
        timing->graph_build_ms =
            std::chrono::duration<double, std::milli>(
                graph_build_done - graph_build_begin).count();
        timing->alloc_ms =
            std::chrono::duration<double, std::milli>(
                alloc_done - alloc_begin).count();
        timing->input_ms =
            std::chrono::duration<double, std::milli>(
                input_done - input_begin).count();
        timing->compute_ms =
            std::chrono::duration<double, std::milli>(
                compute_done - compute_begin).count();
        timing->readback_ms =
            std::chrono::duration<double, std::milli>(
                readback_done - readback_begin).count();
        timing->graph_nodes = ggml_graph_n_nodes(graph);
    }

    if (llama_hybrid_cycle_verbose()) {
        std::printf(
            "[CYCLE_PC_CUDA_REGION] layers=[%d,%d) tokens=%lld "
            "kv_tokens=%lld graph_nodes=%d input_bytes=%zu output_bytes=%zu "
            "status=PASS\n",
            first_layer, first_layer + layer_count,
            (long long)tokens, (long long)kv_tokens,
            ggml_graph_n_nodes(graph),
            input.size() * sizeof(float),
            output.size() * sizeof(float));
    }

    cleanup();
    return true;
}


bool llama_hybrid_cycle_bind_pc_region_stages(
        llama_hybrid_cycle_stage_dispatcher & stages,
        const llama_model & model,
        llama_hybrid_cycle_physical_resource_pool & resources,
        llama_hybrid_cycle_execution_session<> & session,
        const std::vector<float> & initial_hidden,
        int64_t hidden,
        std::string & reason,
        std::atomic<int> * region_calls) {
    reason.clear();
    const auto & layout = resources.execution_layout();
    const auto & session_layout = session.config().layout;
    if (!layout.enabled() ||
        layout.kinds != session_layout.kinds ||
        layout.pc_cuda_layers != session_layout.pc_cuda_layers ||
        layout.pc_cuda_full != session_layout.pc_cuda_full ||
        layout.x <= 1 || layout.cycle_size <= layout.x ||
        layout.cycles <= 0 ||
        layout.tail_begin != int(layout.kinds.size()) ||
        hidden <= 0 ||
        resources.prepared_layers() != int(layout.kinds.size())) {
        reason = "PC_REGION_BIND_LAYOUT";
        return false;
    }

    for (int cycle = 0; cycle < layout.cycles; ++cycle) {
        const int begin = cycle * layout.cycle_size;
        const int end = begin + layout.x;
        if (begin < session.first_layer() ||
            end > session.first_layer() + session.layer_count()) {
            reason = "PC_REGION_BIND_SESSION_RANGE";
            return false;
        }
        for (int l = begin; l < end; ++l) {
            if (layout.kinds[size_t(l)] !=
                    llama_hybrid_cycle_layer_kind::PC_CPU ||
                !llama_hybrid_cycle_full_layer_on_cuda(layout, l)) {
                reason = "PC_REGION_BIND_NON_CUDA_FULL layer=" +
                         std::to_string(l);
                return false;
            }
        }

        const bool start_bound = stages.bind(
            begin, llama_hybrid_cycle_stage::FULL_LAYER,
            [begin, end, hidden, &model, &resources, &session,
             &initial_hidden, region_calls](
                const llama_hybrid_cycle_task & task) -> bool {
                if (task.layer != begin ||
                    task.stage != llama_hybrid_cycle_stage::FULL_LAYER ||
                    task.chunk < 0 ||
                    task.chunk >= session.chunks_per_layer()) {
                    return false;
                }

                auto & start_state = session.at(begin, task.chunk);
                auto & tail_state = session.at(end - 1, task.chunk);
                std::vector<float> input;
                if (begin == session.first_layer()) {
                    const int64_t relative_start =
                        start_state.start -
                        session.config().token_position_base;
                    if (relative_start < 0 ||
                        uint64_t(relative_start) >
                            std::numeric_limits<size_t>::max() /
                                uint64_t(hidden) ||
                        uint64_t(start_state.tokens) >
                            std::numeric_limits<size_t>::max() /
                                uint64_t(hidden)) {
                        return false;
                    }
                    const size_t offset =
                        size_t(relative_start) * size_t(hidden);
                    const size_t count =
                        size_t(start_state.tokens) * size_t(hidden);
                    if (offset > initial_hidden.size() ||
                        count > initial_hidden.size() - offset) {
                        return false;
                    }
                    input.assign(
                        initial_hidden.begin() + offset,
                        initial_hidden.begin() + offset + count);
                } else if (!session.copy_predecessor_host_input(
                               begin, task.chunk, hidden, input)) {
                    std::fprintf(stderr,
                        "[CYCLE_PC_CUDA_REGION] layers=[%d,%d) chunk=%d "
                        "reason=PREDECESSOR_HOST status=FAIL\n",
                        begin, end, task.chunk);
                    return false;
                }

                std::string execute_reason;
                llama_hybrid_cycle_pc_region_timing timing;
                const bool ok = llama_hybrid_cycle_execute_pc_full_region(
                    model, resources, begin, end - begin,
                    start_state.start, start_state.tokens,
                    input, tail_state.host_output, execute_reason,
                    task.input_consumed, &timing);
                if (!ok) {
                    std::fprintf(stderr,
                        "[CYCLE_PC_CUDA_REGION] layers=[%d,%d) chunk=%d "
                        "reason=%s status=FAIL\n",
                        begin, end, task.chunk, execute_reason.c_str());
                    return false;
                }
                if (!session.publish_host_output(
                        end - 1, task.chunk, hidden)) {
                    std::fprintf(stderr,
                        "[CYCLE_PC_CUDA_REGION] layers=[%d,%d) chunk=%d "
                        "reason=TAIL_PUBLISH status=FAIL\n",
                        begin, end, task.chunk);
                    return false;
                }
                if (region_calls) {
                    region_calls->fetch_add(1, std::memory_order_relaxed);
                }
                if (llama_hybrid_cycle_verbose()) {
                    std::printf(
                        "[CYCLE_PC_CUDA_REGION_PROFILE] layers=[%d,%d) "
                        "chunk=%d graph_build_ms=%.3f alloc_ms=%.3f "
                        "input_ms=%.3f compute_ms=%.3f readback_ms=%.3f "
                        "nodes=%d status=PASS\n",
                        begin, end, task.chunk,
                        timing.graph_build_ms, timing.alloc_ms,
                        timing.input_ms, timing.compute_ms,
                        timing.readback_ms, timing.graph_nodes);
                }
                return true;
            });
        if (!start_bound) {
            reason = "PC_REGION_BIND_START layer=" + std::to_string(begin);
            return false;
        }

        for (int l = begin + 1; l < end; ++l) {
            const bool bound = stages.bind(
                l, llama_hybrid_cycle_stage::FULL_LAYER,
                [l, end, &session](
                    const llama_hybrid_cycle_task & task) -> bool {
                    if (task.layer != l ||
                        task.stage != llama_hybrid_cycle_stage::FULL_LAYER ||
                        task.chunk < 0 ||
                        task.chunk >= session.chunks_per_layer()) {
                        return false;
                    }
                    // Region compute already consumed every internal edge.
                    // These tasks only advance the original physical-layer
                    // dependency graph and release its bounded credits.
                    if (task.input_consumed &&
                        !task.input_consumed()) {
                        return false;
                    }
                    if (l == end - 1) {
                        const auto & tail =
                            session.at(end - 1, task.chunk);
                        if (!tail.host_output_published.load(
                                std::memory_order_acquire)) {
                            std::fprintf(stderr,
                                "[CYCLE_PC_CUDA_REGION_CONT] layer=%d "
                                "chunk=%d reason=TAIL_NOT_PUBLISHED "
                                "status=FAIL\n",
                                l, task.chunk);
                            return false;
                        }
                    }
                    return true;
                });
            if (!bound) {
                reason = "PC_REGION_BIND_CONT layer=" + std::to_string(l);
                return false;
            }
        }
    }
    return true;
}
