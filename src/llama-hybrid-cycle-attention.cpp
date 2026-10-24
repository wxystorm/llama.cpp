#include "llama-hybrid-cycle-attention.h"
#include "llama-hybrid-cycle-log.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h" // physical GPU-to-GPU copy

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

uint64_t llama_hybrid_cycle_graph_replay_uid() {
    static std::atomic<uint64_t> next_uid {
        uint64_t(std::chrono::steady_clock::now()
            .time_since_epoch().count()) | UINT64_C(1)
    };
    const uint64_t uid = next_uid.fetch_add(1, std::memory_order_relaxed);
    return uid != 0 ? uid : 1;
}

// The numerical graph, physical KV ordering, scratch leases and completion
// fences match the V5 probe. Only direct RPC calls become injected hooks.
bool llama_hybrid_cycle_execute_attention_pre(
        ggml_backend_t backend, const llama_layer & layer,
        const llama_hparams & hp,
        const std::vector<float> & layer_input,
        int64_t hidden, int64_t tokens,
        int64_t topk, int64_t experts,
        llama_hybrid_cycle_attention_ready & result,
        double & elapsed_ms,
        const llama_hybrid_cycle_live_wire * wire,
        llama_hybrid_cycle_physical_kv_cache * kv_cache,
        int64_t kv_token_start,
        size_t source_device,
        const ggml_tensor * device_input,
        bool retain_device_ready,
        llama_hybrid_cycle_attention_phase_profile * phase_profile,
        const llama_hybrid_cycle_cuda_full_weights * cuda_weights,
        cycle_real_attention_scratch_pool * scratch_pool,
        cycle_real_attention_graph_cache * graph_cache,
        const std::function<bool(const llama_hybrid_cycle_ready_graph &)> &
            before_phone_graph_compute,
        bool defer_registered_replay,
        const llama_hybrid_cycle_attention_rpc_ops & rpc_ops) {
    if (defer_registered_replay &&
        (wire != nullptr || !retain_device_ready ||
         device_input == nullptr || !rpc_ops.is_rpc)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_V2] status=FAIL reason=DEFERRED_GRAPH_MODE\n");
        return false;
    }
    if (!backend || ggml_backend_is_meta(backend) ||
        hidden <= 0 || tokens < 1 || tokens > 128 ||
        kv_token_start < 0 || kv_token_start + tokens > 512 ||
        (kv_cache == nullptr && kv_token_start != 0) ||
        (device_input == nullptr ?
            layer_input.size() != size_t(hidden * tokens) :
            (device_input->type != GGML_TYPE_F32 ||
             ggml_nelements(device_input) != hidden * tokens ||
             device_input->buffer == nullptr)) ||
        hp.f_norm_rms_eps <= 0.0f ||
        topk <= 0 || experts < topk ||
        hp.rope_type != LLAMA_ROPE_TYPE_NEOX ||
        // The loader defaults to "linear" even when no RoPE scale key is
        // present. Linear with factor 1 is numerically identical to NONE.
        (hp.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_NONE &&
         hp.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_LINEAR) ||
        std::fabs(hp.rope_freq_scale_train - 1.0f) > 1.0e-6f ||
        hp.yarn_ext_factor > 0.0f ||
        std::fabs(hp.rope_attn_factor - 1.0f) > 1.0e-6f ||
        hp.use_alibi ||
        layer.wqkv || layer.wq_b || layer.wk_b || layer.wv_b ||
        layer.wo_b || layer.wq_s || layer.wk_s || layer.wv_s ||
        layer.wo_s || hp.f_clamp_kqv != 0.0f ||
        hp.attn_soft_cap || hp.f_max_alibi_bias != 0.0f) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL "
            "reason=UNSUPPORTED_CONTIGUOUS_CAUSAL_PREFILL\n");
        return false;
    }

    // PC complete layers use their native CPU model weights. Tensor and
    // Phone layers still borrow physical Meta shards without host transfers.
    auto get_weight = [backend, source_device](const ggml_tensor * t) -> ggml_tensor * {
        return llama_hybrid_cycle_resolve_physical_weight(t, backend, source_device);
    };
    ggml_tensor * w_attn_norm = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::ATTN_NORM] :
        get_weight(layer.attn_norm);
    ggml_tensor * wq = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::WQ] :
        get_weight(layer.wq);
    ggml_tensor * wk = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::WK] :
        get_weight(layer.wk);
    ggml_tensor * wv = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::WV] :
        get_weight(layer.wv);
    ggml_tensor * wo = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::WO] :
        get_weight(layer.wo);
    ggml_tensor * wq_norm = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::Q_NORM] :
        get_weight(layer.attn_q_norm);
    ggml_tensor * wk_norm = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::K_NORM] :
        get_weight(layer.attn_k_norm);
    ggml_tensor * w_ffn_norm = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::FFN_NORM] :
        get_weight(layer.ffn_norm);
    ggml_tensor * w_router = cuda_weights ?
        cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::ROUTER] :
        get_weight(layer.ffn_gate_inp);

    if (!w_attn_norm || !wq || !wk || !wv || !wo ||
        !wq_norm || !wk_norm || !w_ffn_norm || !w_router) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=PHONE_WEIGHT_MISSING\n");
        return false;
    }
    const int64_t head_dim = wq_norm->ne[0];
    if (head_dim <= 0 || wk_norm->ne[0] != head_dim ||
        wq->ne[0] != hidden || wk->ne[0] != hidden ||
        wv->ne[0] != hidden || wq->ne[1] % head_dim != 0 ||
        wk->ne[1] % head_dim != 0 ||
        wv->ne[1] != wk->ne[1] ||
        wo->ne[0] != wq->ne[1] || wo->ne[1] != hidden ||
        w_router->ne[0] != hidden || w_router->ne[1] != experts ||
        w_attn_norm->ne[0] != hidden || w_ffn_norm->ne[0] != hidden ||
        hp.n_rot_full != uint32_t(head_dim)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=QKV_SHAPE\n");
        return false;
    }
    const int64_t n_heads = wq->ne[1] / head_dim;
    const int64_t n_kv_heads = wk->ne[1] / head_dim;
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=GQA_HEADS\n");
        return false;
    }

    const int64_t kv_tokens = kv_token_start + tokens;
    if (kv_cache != nullptr &&
        (!kv_cache->keys || !kv_cache->values ||
         kv_cache->head_dim != head_dim ||
         kv_cache->kv_heads != n_kv_heads ||
         kv_tokens > kv_cache->max_tokens ||
         // Deferred registration does not read the mutable host
         // committed-prefix counter. Protocol admission + the global Ring
         // enforce execution order; only live execution requires an exact
         // prefix match. This also keeps async registration race-free.
         (!defer_registered_replay &&
          kv_cache->committed_tokens != kv_token_start))) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_KV_PERSISTENT] status=FAIL "
            "reason=BAD_PREFIX_OR_OUT_OF_ORDER start=%lld "
            "committed=%lld mode=%s\n",
            (long long)kv_token_start,
            (long long)(kv_cache ? kv_cache->committed_tokens : -1),
            defer_registered_replay ? "REGISTER" : "EXECUTE");
        return false;
    }
    const auto graph_build_begin = std::chrono::steady_clock::now();
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * causal_mask = nullptr;
    ggml_tensor * router_probs = nullptr;
    llama_hybrid_cycle_ready_graph ready {};
    cycle_real_attention_graph_cache::slot * graph_slot = nullptr;
    const bool cache_candidate = graph_cache != nullptr && kv_cache != nullptr;
    if (cache_candidate) {
        graph_slot = graph_cache->checkout(tokens, kv_tokens, kv_cache);
        if (graph_slot) {
            // The compiled graph, physical KV views and weight pointers are
            // identical to the Warmup graph. Its GPU scratch was detached
            // after the previous FFN consumer fence.
            ctx = graph_slot->ctx;
            graph = graph_slot->graph;
            x = graph_slot->input;
            positions = graph_slot->positions;
            causal_mask = graph_slot->causal_mask;
            router_probs = graph_slot->router_probs;
            ready = graph_slot->ready;
        }
    }
    if (!graph_slot) {
    ggml_init_params p {};
    p.mem_size = 16 * 1024 * 1024;
    p.no_alloc = true;
    ctx = ggml_init(p);
    if (!ctx) {
        return false;
    }

    x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    // Keys include all PAST tokens and the current causal chunk.
    causal_mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, kv_tokens, tokens);
    ggml_set_input(x);
    ggml_set_input(positions);
    ggml_set_input(causal_mask);
    ggml_set_name(x, "cycle_phone_attn_input");
    ggml_set_name(positions, "cycle_phone_positions");
    ggml_set_name(causal_mask, "cycle_phone_causal_kq_mask");

    auto * attn_norm = ggml_mul(
        ctx, ggml_rms_norm(ctx, x, hp.f_norm_rms_eps), w_attn_norm);
    auto * q = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, wq, attn_norm), head_dim, n_heads, tokens);
    auto * k = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, wk, attn_norm), head_dim, n_kv_heads, tokens);
    auto * v = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, wv, attn_norm), head_dim, n_kv_heads, tokens);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.f_norm_rms_eps), wq_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp.f_norm_rms_eps), wk_norm);

    // RoPE is now essential: position 1..N-1 must not be treated as 0.
    // The standalone Qwen3-MoE check rejects nondefault YaRN/scaling above.
    const int ctx_orig = hp.n_ctx_orig_yarn > 0 ?
        int(hp.n_ctx_orig_yarn) : int(hp.n_ctx_train);
    const float rope_base = hp.rope_freq_base_train > 0 ?
        hp.rope_freq_base_train : 10000.0f;
    const float rope_scale = hp.rope_freq_scale_train > 0 ?
        hp.rope_freq_scale_train : 1.0f;
    q = ggml_rope_ext(ctx, q, positions, nullptr,
        int(head_dim), GGML_ROPE_TYPE_NEOX, ctx_orig,
        rope_base, rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr,
        int(head_dim), GGML_ROPE_TYPE_NEOX, ctx_orig,
        rope_base, rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);

    // Same no-Flash GQA layout as llm_graph_context::build_attn_mha():
    // [head_dim, tokens, heads, stream]. The reference cache stores K/V
    // in F16; preserve that rounding before the attention projection.
    // In this single-batch test, K/V at positions [0,N) are the entire
    // transient cache. The mask below implements the same causal rule
    // as llama_kv_cache::set_input_kq_mask (key_pos > query_pos -> -inf).
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
    v = ggml_cast(ctx, v, GGML_TYPE_F16);
    ggml_tensor * kv_k_write = nullptr;
    ggml_tensor * kv_v_write = nullptr;
    if (kv_cache != nullptr) {
        // NEW K/V are contiguous [head_dim,kv_heads,chunk_tokens].
        // Persist them in the same Phone GPU buffer before reading all
        // historical keys; no historical K/V is ever sent to the PC.
        auto * dst_k = ggml_view_3d(
            ctx, kv_cache->keys, head_dim, n_kv_heads, tokens,
            kv_cache->keys->nb[1], kv_cache->keys->nb[2],
            size_t(kv_token_start) * kv_cache->keys->nb[2]);
        auto * dst_v = ggml_view_3d(
            ctx, kv_cache->values, head_dim, n_kv_heads, tokens,
            kv_cache->values->nb[1], kv_cache->values->nb[2],
            size_t(kv_token_start) * kv_cache->values->nb[2]);
        kv_k_write = ggml_cpy(ctx, k, dst_k);
        kv_v_write = ggml_cpy(ctx, v, dst_v);
        k = ggml_view_3d(
            ctx, kv_cache->keys, head_dim, n_kv_heads, kv_tokens,
            kv_cache->keys->nb[1], kv_cache->keys->nb[2], 0);
        v = ggml_view_3d(
            ctx, kv_cache->values, head_dim, n_kv_heads, kv_tokens,
            kv_cache->values->nb[1], kv_cache->values->nb[2], 0);
    }
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

    // The concatenated attention width is n_heads * head_dim (4096 on
    // Qwen3-30B-A3B), NOT the residual/hidden width (2048). The WO
    // projection is responsible for mapping 4096 -> 2048. Previously,
    // requesting [hidden,1] in ggml_cont_2d aborted at ggml.c:3594.
    const int64_t attention_width = wq->ne[1];
    if (ggml_nelements(kqv) != attention_width * tokens) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=ATTN_WIDTH "
            "kqv_elements=%lld expected=%lld hidden=%lld\n",
            (long long)ggml_nelements(kqv),
            (long long)(attention_width * tokens), (long long)hidden);
        ggml_free(ctx);
        return false;
    }
    auto * attn = ggml_cont_2d(ctx,
        ggml_permute(ctx, kqv, 0, 2, 1, 3), attention_width, tokens);

    if (llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_PHONE_ATTN_SHAPE] hidden=%lld head_dim=%lld "
        "heads=%lld kv_heads=%lld attn_width=%lld wo_out=%lld "
        "query_tokens=%lld kv_tokens=%lld status=PASS\n",
        (long long)hidden, (long long)head_dim,
        (long long)n_heads, (long long)n_kv_heads,
        (long long)attention_width, (long long)wo->ne[1],
        (long long)tokens, (long long)kv_tokens);

    auto * attn_output = ggml_mul_mat(ctx, wo, attn);
    auto * residual = ggml_add(ctx, attn_output, x);
    ggml_set_name(residual, "cycle_num_attn_residual");
    auto * norm = ggml_mul(
        ctx, ggml_rms_norm(ctx, residual, hp.f_norm_rms_eps), w_ffn_norm);
    ggml_set_name(norm, "cycle_num_attn_ffn_norm");
    auto * router_logits = ggml_mul_mat(ctx, w_router, norm);
    router_probs = ggml_soft_max(ctx, router_logits);
    auto * ids = ggml_cont(ctx, ggml_argsort_top_k(
        ctx, router_probs, int(topk)));
    auto * router_p3 = ggml_reshape_3d(ctx, router_probs, 1, experts, tokens);
    auto * weights = ggml_get_rows(ctx, router_p3, ids);
    weights = ggml_reshape_2d(ctx, weights, topk, tokens);
    auto * sum = ggml_sum_rows(ctx, weights);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, sum);
    weights = ggml_reshape_3d(ctx, weights, 1, topk, tokens);
    if (hp.expert_weights_scale != 0.0f &&
        hp.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, hp.expert_weights_scale);
    }
    weights = ggml_cont(ctx, weights);

    ready = {residual, norm, ids, weights};
    if (!llama_hybrid_cycle_ready_graph_valid(ready, hidden, tokens, topk)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=READY_SHAPES\n");
        ggml_free(ctx);
        return false;
    }
    graph = ggml_new_graph_custom(ctx, 512, false);
    // Enforce GPU cache writes before subsequent prefix attention reads.
    if (kv_cache != nullptr) {
        ggml_build_forward_expand(graph, kv_k_write);
        ggml_build_forward_expand(graph, kv_v_write);
    }
    ggml_build_forward_expand(graph, ready.residual);
    ggml_build_forward_expand(graph, ready.ffn_norm);
    ggml_build_forward_expand(graph, ready.topk_ids);
    ggml_build_forward_expand(graph, ready.topk_weights);
    int unsupported = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_NONE &&
            !ggml_backend_supports_op(backend, node)) {
            ++unsupported;
            std::fprintf(stderr,
                "[CYCLE_PHONE_ATTN_PRE_OP] node=%d op=%s "
                "name=%s supported=0\n",
                i, ggml_op_name(node->op), node->name);
        }
    }
    if (unsupported != 0) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL "
            "reason=UNSUPPORTED_PHONE_OP count=%d\n", unsupported);
        ggml_free(ctx);
        return false;
    }

    }
    const auto graph_build_done = std::chrono::steady_clock::now();

    const auto buft = ggml_backend_get_default_buffer_type(backend);
    const auto alloc_begin = std::chrono::steady_clock::now();
    ggml_backend_buffer_t compute_buffer = nullptr;
    bool scratch_pooled = false;
    const bool allocated = scratch_pool ?
        scratch_pool->acquire(ctx, compute_buffer, scratch_pooled) :
        (compute_buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)) != nullptr;
    const auto alloc_done = std::chrono::steady_clock::now();
    if (!allocated || !compute_buffer) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=ALLOC_PHONE_GRAPH\n");
        if (graph_slot) {
            graph_cache->recycle(graph_slot, false);
        } else {
            ggml_free(ctx);
        }
        return false;
    }

    const auto release_compute_buffer = [&] {
        if (scratch_pooled && scratch_pool) {
            scratch_pool->recycle(compute_buffer);
        } else {
            ggml_backend_buffer_free(compute_buffer);
        }
    };

    // Input layout: token-major hidden vectors; positions increment 0..N-1.
    // The F16 mask's dimension 0 is the KV index and dimension 1 is query.
    std::vector<int32_t> position_data(size_t(tokens), 0);
    std::vector<ggml_fp16_t> mask_data(size_t(tokens * kv_tokens));
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    int64_t masked_future = 0;
    for (int64_t query = 0; query < tokens; ++query) {
        const int64_t global_pos = kv_token_start + query;
        position_data[size_t(query)] = int32_t(global_pos);
        for (int64_t key = 0; key < kv_tokens; ++key) {
            const bool future = key > global_pos;
            mask_data[size_t(query * kv_tokens + key)] = future ? drop : keep;
            masked_future += future ? 1 : 0;
        }
    }
    if (masked_future != tokens * (tokens - 1) / 2) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_KV] status=FAIL reason=MASK_VALIDATION\n");
        if (graph_slot) {
            graph_cache->recycle(graph_slot, false);
        } else {
            ggml_free(ctx);
        }
        release_compute_buffer();
        return false;
    }
    if (llama_hybrid_cycle_verbose()) std::printf("[CYCLE_PHONE_KV] query_tokens=%lld kv_tokens=%lld "
                "position_start=%lld position_end=%lld "
                "masked_future=%lld mask_type=F16 scope=%s "
                "status=PASS\n",
                (long long)tokens, (long long)kv_tokens,
                (long long)kv_token_start, (long long)(kv_tokens - 1),
                (long long)masked_future,
                kv_cache == nullptr ? "SINGLE_UBATCH" :
            source_device == 0 ? "PC_PERSISTENT" : "PHONE_PERSISTENT");

    // Arm the Phone continuation before the ordered Attention Graph and
    // READY stage. Failure must return both Graph and Scratch leases.
    if (before_phone_graph_compute &&
        !before_phone_graph_compute(ready)) {
        if (graph_slot) {
            graph_cache->recycle(graph_slot, false);
        } else {
            ggml_free(ctx);
        }
        release_compute_buffer();
        return false;
    }
    const auto begin = std::chrono::steady_clock::now();
    if (device_input != nullptr) {
        // Both immediate and deferred Phone Full handoffs must validate
        // exactly the same shape/strides. In V2 the source l_out is NOT yet
        // produced: the Phone worker will COPY after the producer FFN+ADD.
        // Never upload the empty host vector as an Attention input.
        bool layout_ok = ggml_are_same_shape(device_input, x) &&
                         device_input->type == x->type;
        for (int d = 0; d < GGML_MAX_DIMS && layout_ok; ++d) {
            layout_ok = device_input->nb[d] == x->nb[d];
        }
        if (!layout_ok ||
            (!defer_registered_replay &&
             !ggml_backend_buffer_copy_tensor(device_input, x))) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_LOCAL_HANDOFF] status=FAIL "
                "reason=%s\n",
                !layout_ok ? "TENSOR_LAYOUT_MISMATCH" :
                             "DEVICE_COPY_UNSUPPORTED");
            if (graph_slot) {
                graph_cache->recycle(graph_slot, false);
            } else {
                ggml_free(ctx);
            }
            release_compute_buffer();
            return false;
        }
        if (llama_hybrid_cycle_verbose()) std::printf(
            "[CYCLE_PHONE_LOCAL_HANDOFF] backend=%s "
            "bytes=%zu source=PHONE_GPU_PREVIOUS_LAYER "
            "destination=PHONE_GPU_ATTENTION "
            "action=%s status=OK\n",
            ggml_backend_name(backend), ggml_nbytes(x),
            defer_registered_replay ?
                "DEFER_TO_PHONE_FIFO" : "DEVICE_COPY");
    } else {
        // Host upload is valid only for a genuinely host-resident input;
        // deferred replay requires a non-null physical GPU producer.
        ggml_backend_tensor_set(x, layer_input.data(), 0, ggml_nbytes(x));
    }
    ggml_backend_tensor_set(positions, position_data.data(), 0,
                            ggml_nbytes(positions));
    ggml_backend_tensor_set(causal_mask, mask_data.data(), 0,
                            ggml_nbytes(causal_mask));
    const auto input_done = std::chrono::steady_clock::now();
    uint64_t deferred_graph_uid = 0;
    bool graph_submitted = false;
    if (defer_registered_replay) {
        deferred_graph_uid = llama_hybrid_cycle_graph_replay_uid();
        ggml_graph_set_uid(graph, deferred_graph_uid);
        graph_submitted = rpc_ops.register_graph != nullptr &&
            rpc_ops.register_graph(backend, graph);
        if (!graph_submitted || llama_hybrid_cycle_verbose()) std::printf(
            "[CYCLE_PHONE_V2] side=PC event=REGISTER_NEXT_ATTN "
            "uid=%" PRIu64 " tokens=%lld prefix=%lld status=%s\n",
            deferred_graph_uid, (long long)tokens,
            (long long)kv_token_start,
            graph_submitted ? "OK" : "FAIL");
    } else {
        graph_submitted = ggml_backend_graph_compute(backend, graph) ==
            GGML_STATUS_SUCCESS;
    }
    const auto submit_done = std::chrono::steady_clock::now();
    if (phase_profile) {
        ++phase_profile->count;
        phase_profile->graph_build_ms +=
            std::chrono::duration<double, std::milli>(
                graph_build_done - graph_build_begin).count();
        phase_profile->alloc_ms +=
            std::chrono::duration<double, std::milli>(
                alloc_done - alloc_begin).count();
        phase_profile->input_copy_ms +=
            std::chrono::duration<double, std::milli>(
                input_done - begin).count();
        phase_profile->graph_call_ms +=
            std::chrono::duration<double, std::milli>(
                submit_done - input_done).count();
    }
    if (std::getenv("LLAMA_HYBRID_CYCLE_STAGE_PROFILE") != nullptr) {
        std::printf(
            "[CYCLE_ATTN_PROFILE] backend=%s "
            "graph_build_ms=%.3f gpu_alloc_ms=%.3f "
            "input_copy_ms=%.3f graph_submit_or_compute_ms=%.3f "
            "remote_gpu_compute_ms=SEE_CYCLE_GPU_SERVER\n",
            ggml_backend_name(backend),
            std::chrono::duration<double, std::milli>(
                graph_build_done - graph_build_begin).count(),
            std::chrono::duration<double, std::milli>(
                alloc_done - alloc_begin).count(),
            std::chrono::duration<double, std::milli>(
                input_done - begin).count(),
            std::chrono::duration<double, std::milli>(
                submit_done - input_done).count());
    }
    bool ok = graph_submitted;
    if (ok && !defer_registered_replay) {
        ggml_backend_synchronize(backend);
        if (kv_cache != nullptr) {
            kv_cache->committed_tokens = kv_tokens;
            if (llama_hybrid_cycle_verbose()) std::printf(
                "[CYCLE_PHONE_KV_PERSISTENT] chunk=%d "
                "position_start=%lld committed=%lld "
                "total=%lld owner=%s status=OK\n",
                wire != nullptr ? wire->chunk : -1,
                (long long)kv_token_start,
                (long long)kv_cache->committed_tokens,
                (long long)kv_cache->max_tokens,
                cuda_weights ? "PC_CUDA" :
                    source_device == 0 ? "PC_CPU" : "PHONE_GPU");
        }
        if (wire != nullptr) {
            // GPU tensors remain valid until compute_buffer is destroyed.
            // The Phone RPC server copies directly into its transport TX
            // slot; no PC-side retransmission is involved.
            ok = rpc_ops.send_ready != nullptr &&
                rpc_ops.send_ready(
                    backend, wire->layer, wire->chunk,
                    wire->seq, wire->graph_id,
                    ready.residual, ready.ffn_norm,
                    ready.topk_ids, ready.topk_weights);
            if (!ok) {
                std::fprintf(stderr,
                    "[CYCLE_REAL_WIRE] stage=READY status=FAIL "
                    "reason=PHONE_REMOTE_SEND\n");
            }
        }
    }
    if (ok && wire == nullptr && !retain_device_ready) {
        result.residual.resize(size_t(hidden * tokens));
        result.norm.resize(size_t(hidden * tokens));
        result.ids.resize(size_t(topk * tokens));
        result.routing.resize(size_t(topk * tokens));
        result.router_probs.resize(size_t(experts * tokens));
        ggml_backend_tensor_get(ready.residual,
            result.residual.data(), 0, ggml_nbytes(ready.residual));
        ggml_backend_tensor_get(ready.ffn_norm,
            result.norm.data(), 0, ggml_nbytes(ready.ffn_norm));
        ggml_backend_tensor_get(ready.topk_ids,
            result.ids.data(), 0, ggml_nbytes(ready.topk_ids));
        ggml_backend_tensor_get(ready.topk_weights,
            result.routing.data(), 0, ggml_nbytes(ready.topk_weights));
        ggml_backend_tensor_get(router_probs,
            result.router_probs.data(), 0, ggml_nbytes(router_probs));
    }
    elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    if (!ok || llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_PHONE_ATTN_PRE] backend=%s "
        "positions=[%lld,%lld] tokens=%lld kv_tokens=%lld "
        "heads=%lld kv_heads=%lld graph_nodes=%d "
        "wall_ms=%.3f status=%s\n",
        ggml_backend_name(backend),
        (long long)kv_token_start, (long long)(kv_tokens - 1),
        (long long)tokens, (long long)kv_tokens,
        (long long)n_heads, (long long)n_kv_heads,
        ggml_graph_n_nodes(graph), elapsed_ms,
        ok ? "OK" : "FAIL");

    if (ok && (wire != nullptr || retain_device_ready)) {
        // Borrow original Attention tensors until the local Phone FFN
        // consumes them, preventing a hidden PC->Phone activation upload.
        if (cache_candidate && !graph_slot) {
            // Store metadata only. READY still owns the temporary GPU
            // scratch buffer; the graph remains busy until FFN consumes it.
            graph_slot = graph_cache->retain(
                tokens, kv_tokens, kv_cache, ctx, graph,
                x, positions, causal_mask, router_probs, ready);
        }
        result.phone_ctx = graph_slot ? nullptr : ctx;
        result.phone_compute_buffer = compute_buffer;
        result.phone_scratch_pool = scratch_pooled ? scratch_pool : nullptr;
        result.phone_graph_cache = graph_slot ? graph_cache : nullptr;
        result.phone_cached_graph = graph_slot;
        result.phone_graph = ready;
        result.prepared_attention_graph_uid = deferred_graph_uid;
        result.prepared_attention_input =
            defer_registered_replay ? x : nullptr;
        return true;
    }
    if (graph_slot) {
        graph_cache->recycle(graph_slot, false);
    } else {
        ggml_free(ctx);
    }
    release_compute_buffer();
    return ok;
}
