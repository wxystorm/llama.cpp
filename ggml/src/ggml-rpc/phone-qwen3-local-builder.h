#pragma once

#include "ggml.h"
#include "phone-owned-execution-plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Phone-local, metadata-only Qwen3 single-Layer graph builder. This copies
// the operator order from src/models/qwen3.cpp and the standard flash-attn
// KV path in src/llama-graph.cpp and src/llama-kv-cache.cpp.
// The Graph has no input/storage bindings until a separate HTP allocation
// phase. Merely building it does NOT make a Chunk executable.
namespace ggml_phone_qwen3 {

struct layer_weights {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
};

struct graph_result {
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * positions = nullptr;
    ggml_tensor * k_indices = nullptr;
    ggml_tensor * v_indices = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * cache_k = nullptr;
    ggml_tensor * cache_v = nullptr;
    ggml_tensor * output = nullptr;
    uint32_t tokens = 0;
    uint32_t token_begin = 0;
};

struct side_inputs {
    std::vector<int32_t> positions;
    std::vector<int64_t> k_indices;
    std::vector<int64_t> v_indices;
    std::vector<uint16_t> causal_mask_f16;
};

inline bool make_side_inputs(const ggml_phone_owned::region_plan & p,
                             uint32_t chunk, side_inputs & out,
                             const char * & reason) {
    if (!ggml_phone_owned::validate(p, reason) ||
            chunk >= ggml_phone_owned::chunk_count(p)) {
        reason = "INVALID_SIDE_INPUT_CHUNK";
        return false;
    }
    const uint32_t begin = chunk * p.chunk_tokens;
    const uint32_t tokens = std::min(p.chunk_tokens, p.total_tokens - begin);
    if (uint64_t(p.n_kv) * tokens * sizeof(uint16_t) >
            ggml_phone_owned::k_max_activation_bytes) {
        reason = "CAUSAL_MASK_BUDGET_EXCEEDED";
        return false;
    }
    out.positions.resize(tokens);
    out.k_indices.resize(tokens);
    out.v_indices.resize(tokens);
    out.causal_mask_f16.resize(uint64_t(p.n_kv) * tokens);
    for (uint32_t t = 0; t < tokens; ++t) {
        const uint32_t position = p.position_base + begin + t;
        out.positions[t] = int32_t(position);
        out.k_indices[t] = int64_t(position);
        out.v_indices[t] = int64_t(position);
        for (uint32_t k = 0; k < p.n_kv; ++k) {
            // F16 zero is 0x0000, F16 -infinity is 0xfc00.
            out.causal_mask_f16[uint64_t(t) * p.n_kv + k] =
                k <= position ? uint16_t(0) : uint16_t(0xfc00);
        }
    }
    reason = "NONE";
    return true;
}

inline bool validate_weights(const ggml_phone_owned::region_plan & p,
                              const layer_weights & w,
                              const char * & reason) {
    const int64_t d = p.n_embd;
    const int64_t hk = int64_t(p.n_heads_kv) * p.head_dim;
    const int64_t hq = int64_t(p.n_heads) * p.head_dim;
    const auto vec = [](const ggml_tensor * t, int64_t n) {
        return t && t->ne[0] == n && t->ne[1] == 1 &&
            t->ne[2] == 1 && t->ne[3] == 1;
    };
    const auto mat = [](const ggml_tensor * t, int64_t n0, int64_t n1) {
        return t && t->ne[0] == n0 && t->ne[1] == n1 &&
            t->ne[2] == 1 && t->ne[3] == 1;
    };
    if (!vec(w.attn_norm, d) || !mat(w.wq, d, hq) ||
            !mat(w.wk, d, hk) || !mat(w.wv, d, hk) ||
            !mat(w.wo, hq, d) || !vec(w.ffn_norm, d) ||
            !vec(w.q_norm, p.head_dim) ||
            !vec(w.k_norm, p.head_dim)) {
        reason = "QWEN3_ATTENTION_WEIGHT_LAYOUT";
        return false;
    }
    const int64_t n_ff = w.ffn_gate ? w.ffn_gate->ne[1] : 0;
    if (n_ff <= 0 || n_ff > 131072 ||
            !mat(w.ffn_gate, d, n_ff) ||
            !mat(w.ffn_up, d, n_ff) ||
            !mat(w.ffn_down, n_ff, d)) {
        reason = "QWEN3_FFN_WEIGHT_LAYOUT";
        return false;
    }
    reason = "NONE";
    return true;
}

inline bool build_graph(ggml_context * ctx,
                        const ggml_phone_owned::region_plan & p,
                        uint32_t chunk,
                        const layer_weights & w,
                        graph_result & out,
                        const char * & reason,
                        ggml_tensor * persistent_k = nullptr,
                        ggml_tensor * persistent_v = nullptr) {
    out = {};
    if (!ctx || !ggml_phone_owned::validate(p, reason) ||
            chunk >= ggml_phone_owned::chunk_count(p) ||
            !validate_weights(p, w, reason)) {
        if (reason == nullptr || std::string(reason) == "NONE") {
            reason = "INVALID_LOCAL_GRAPH_INPUT";
        }
        return false;
    }
    const uint32_t begin = chunk * p.chunk_tokens;
    const uint32_t tokens = std::min(p.chunk_tokens, p.total_tokens - begin);
    const int64_t kv_width = int64_t(p.n_heads_kv) * p.head_dim;
    if (uint64_t(p.n_kv) * tokens * sizeof(uint16_t) >
            ggml_phone_owned::k_max_activation_bytes) {
        reason = "CAUSAL_MASK_BUDGET_EXCEEDED";
        return false;
    }
    // Two graphs can share one persistent layer KV allocation. Ownership
    // and KV reset are the caller's responsibility; shadow mode uses leaves.
    if ((persistent_k == nullptr) != (persistent_v == nullptr)) {
        reason = "INCOMPLETE_PERSISTENT_KV_PAIR";
        return false;
    }
    ggml_tensor * cache_k = persistent_k;
    ggml_tensor * cache_v = persistent_v;
    if (!cache_k) {
        cache_k = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F16, kv_width, p.kv_capacity);
        cache_v = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F16, kv_width, p.kv_capacity);
    }
    if (!cache_k || !cache_v ||
            cache_k->type != GGML_TYPE_F16 ||
            cache_v->type != GGML_TYPE_F16 ||
            cache_k->ne[0] != kv_width ||
            cache_v->ne[0] != kv_width ||
            cache_k->ne[1] != p.kv_capacity ||
            cache_v->ne[1] != p.kv_capacity ||
            !ggml_is_contiguous(cache_k) ||
            !ggml_is_contiguous(cache_v)) {
        reason = "INVALID_PHONE_OWNED_KV_LAYOUT";
        return false;
    }
    char name[GGML_MAX_NAME] = {};
    std::snprintf(name, sizeof(name), "cache_k_l%u", p.layer);
    ggml_set_name(cache_k, name);
    std::snprintf(name, sizeof(name), "cache_v_l%u", p.layer);
    ggml_set_name(cache_v, name);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    if (!graph) { reason = "LOCAL_GRAPH_ALLOC"; return false; }
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, p.n_embd, tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, tokens);
    ggml_tensor * ki = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I64, tokens);
    ggml_tensor * vi = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I64, tokens);
    ggml_tensor * mask = ggml_new_tensor_4d(
        ctx, GGML_TYPE_F16, p.n_kv, tokens, 1, 1);
    if (!input || !pos || !ki || !vi || !mask) {
        reason = "LOCAL_GRAPH_INPUT_ALLOC";
        return false;
    }
    ggml_set_name(input, "inp_stage");
    ggml_set_name(pos, "inp_pos");
    ggml_set_name(ki, "kv_k_indices");
    ggml_set_name(vi, "kv_v_indices");
    ggml_set_name(mask, "attn_inp_kq_mask");
    ggml_set_input(input);
    ggml_set_input(pos);
    ggml_set_input(ki);
    ggml_set_input(vi);
    ggml_set_input(mask);

    // Names mirror llama.cpp's cb(stage, layer) labels for optional
    // HTP-to-HTP stage parity. These do not alter the operator graph.
    const auto mark = [&](ggml_tensor * t, const char * stage) {
        std::snprintf(name, sizeof(name), "%s-%u", stage, p.layer);
        ggml_set_name(t, name);
    };

    // Qwen3: RMSNorm -> QKV matmuls -> per-head Q/K RMSNorm -> NeoX RoPE.
    ggml_tensor * x_norm = ggml_mul(ctx,
        ggml_rms_norm(ctx, input, p.rms_norm_eps), w.attn_norm);
    mark(x_norm, "attn_norm");
    ggml_tensor * q = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, w.wq, x_norm), p.head_dim, p.n_heads, tokens);
    ggml_tensor * k = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, w.wk, x_norm), p.head_dim, p.n_heads_kv, tokens);
    ggml_tensor * v = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, w.wv, x_norm), p.head_dim, p.n_heads_kv, tokens);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, p.rms_norm_eps), w.q_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, p.rms_norm_eps), w.k_norm);
    q = ggml_rope_ext(ctx, q, pos, nullptr, p.head_dim, p.rope_type,
                       p.n_ctx_orig, p.rope_freq_base, p.rope_freq_scale,
                       p.rope_ext_factor, p.rope_attn_factor,
                       p.rope_beta_fast, p.rope_beta_slow);
    k = ggml_rope_ext(ctx, k, pos, nullptr, p.head_dim, p.rope_type,
                       p.n_ctx_orig, p.rope_freq_base, p.rope_freq_scale,
                       p.rope_ext_factor, p.rope_attn_factor,
                       p.rope_beta_fast, p.rope_beta_slow);

    mark(q, "Qcur");
    mark(k, "Kcur");
    mark(v, "Vcur");
    // Preserve original KV write order. set_rows aliases the persistent
    // KV base; the flash-attn read nodes then see all committed prior chunks.
    ggml_build_forward_expand(graph, q);
    ggml_build_forward_expand(graph, v);
    ggml_build_forward_expand(graph, k);
    ggml_tensor * k_rows = ggml_view_2d(
        ctx, k, kv_width, tokens, k->nb[2], 0);
    ggml_tensor * v_rows = ggml_view_2d(
        ctx, v, kv_width, tokens, v->nb[2], 0);
    ggml_build_forward_expand(
        graph, ggml_set_rows(ctx, cache_k, k_rows, ki));
    ggml_build_forward_expand(
        graph, ggml_set_rows(ctx, cache_v, v_rows, vi));

    // Flash Attention takes Q/K/V as [head_dim, tokens, heads, streams].
    // V is not transposed: this is the regular F16 KV layout used by FA.
    ggml_tensor * q4 = ggml_view_4d(
        ctx, q, p.head_dim, p.n_heads, tokens, 1,
        q->nb[1], q->nb[2], q->nb[3], 0);
    ggml_tensor * k4 = ggml_view_4d(
        ctx, cache_k, p.head_dim, p.n_heads_kv, p.n_kv, 1,
        ggml_row_size(cache_k->type, p.head_dim),
        ggml_row_size(cache_k->type, kv_width),
        ggml_row_size(cache_k->type, kv_width * p.kv_capacity), 0);
    ggml_tensor * v4 = ggml_view_4d(
        ctx, cache_v, p.head_dim, p.n_heads_kv, p.n_kv, 1,
        ggml_row_size(cache_v->type, p.head_dim),
        ggml_row_size(cache_v->type, kv_width),
        ggml_row_size(cache_v->type, kv_width * p.kv_capacity), 0);
    ggml_tensor * attn = ggml_flash_attn_ext(
        ctx, ggml_permute(ctx, q4, 0, 2, 1, 3),
             ggml_permute(ctx, k4, 0, 2, 1, 3),
             ggml_permute(ctx, v4, 0, 2, 1, 3),
             mask, 1.0f / std::sqrt(float(p.head_dim)), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(ctx, attn, p.n_embd, tokens);
    ggml_tensor * attn_out = ggml_mul_mat(ctx, w.wo, attn);
    mark(attn_out, "attn_out");
    ggml_tensor * ffn_inp = ggml_add(ctx, attn_out, input);
    mark(ffn_inp, "ffn_inp");

    ggml_tensor * ffn_norm = ggml_mul(ctx,
        ggml_rms_norm(ctx, ffn_inp, p.rms_norm_eps), w.ffn_norm);
    mark(ffn_norm, "ffn_norm");
    ggml_tensor * up = ggml_mul_mat(ctx, w.ffn_up, ffn_norm);
    mark(up, "ffn_up");
    ggml_tensor * gate = ggml_mul_mat(ctx, w.ffn_gate, ffn_norm);
    mark(gate, "ffn_gate");
    ggml_tensor * glu = ggml_swiglu_split(ctx, gate, up);
    mark(glu, "ffn_swiglu");
    ggml_tensor * ffn = ggml_mul_mat(ctx, w.ffn_down, glu);
    mark(ffn, "ffn_down");
    ggml_tensor * result = ggml_add(ctx, ffn, ffn_inp);
    mark(result, "ffn_out");
    std::snprintf(name, sizeof(name), "l_out-%u", p.layer);
    ggml_set_name(result, name);
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);

    if (ggml_graph_n_nodes(graph) < 10) {
        reason = "LOCAL_GRAPH_EMPTY";
        return false;
    }
    out.graph = graph;
    out.input = input;
    out.positions = pos;
    out.k_indices = ki;
    out.v_indices = vi;
    out.mask = mask;
    out.cache_k = cache_k;
    out.cache_v = cache_v;
    out.output = result;
    out.tokens = tokens;
    out.token_begin = begin;
    reason = "NONE";
    return true;
}

} // namespace ggml_phone_qwen3
