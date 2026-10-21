// Opt-in real-Qwen3-MoE P2C numerical comparison, not a production Cycle
// executor. A separate CPU-only GGUF model provides the independent full
// Attention/Norm/Router/FFN/l_out reference, and is freed before the Cycle
// model is loaded. PC CPU + Phone GPU then run *real physical FFN shards*
// independently. Only PC sums both partials and adds the captured residual.
// Phone GPU executes a real causal AttentionPre over the prompt's KV tokens
// and we compare its four READY outputs against that CPU reference before
// using them for either FFN shard. The Phone graph now supports contiguous
// multi-token causal KV within one prompt ubatch, but not persistent KV
// across calls/chunks.
//
// READY/PARTIAL are serialized/deserialized IN PROCESS as wire-format
// checks only. The Phone FFN input/output still travel through ordinary
// RPC backend tensor set/get; the dedicated P2C network queues, actual two
// sender messages and FIFO stage executor are NOT active.
//
// Usage (choose a TENSOR_P2C layer in the fixed cycle):
//   test-hybrid-cycle-real-p2c <Qwen3-MoE.gguf> <phone:rpc-port> <layer> [prompt] [n_tokens]
// n_tokens=1..128 creates a deterministic token-count case by repeating
// and truncating the prompt, keeping one causal ubatch with positions 0..N-1.
// Set LLAMA_HYBRID_CYCLE_PC_LAYERS and LLAMA_HYBRID_CYCLE_PHONE_LAYERS
// before launching (e.g. 5 and 1, then the first P2C layer is 7).
#include "llama.h"
#include "llama-model.h"
#include "llama-hybrid-cycle.h"
#include "llama-hybrid-cycle-ffn-executor.h"
#include "inference-cycle-transport.h"
#include "inference-cycle-reduce.h"
#include "ggml-rpc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct cycle_real_capture_value {
    std::vector<uint8_t> bytes;
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[GGML_MAX_DIMS] = {};
    int count = 0;
};

struct cycle_real_capture {
    std::map<std::string, cycle_real_capture_value> values;
    bool error = false;
};

// This is the ordinary llama_context scheduler callback. On ask=false,
// the scheduler has already synchronized the node's producer backend.
static bool cycle_real_capture_cb(ggml_tensor * tensor, bool ask, void * userdata) {
    auto & state = *static_cast<cycle_real_capture *>(userdata);
    if (!tensor) {
        state.error = true;
        return false;
    }
    auto it = state.values.find(tensor->name);
    if (it == state.values.end()) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    if (!tensor->buffer || !ggml_is_contiguous(tensor)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=NO_BUFFER_OR_NONCONTIG\n",
            tensor->name);
        state.error = true;
        return false;
    }
    auto & output = it->second;
    if (output.count != 0) {
        // A single prompt ubatch should deliver exactly one complete tensor.
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=DUPLICATE_CAPTURE\n",
            tensor->name);
        state.error = true;
        return false;
    }
    output.type = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        output.ne[i] = tensor->ne[i];
    }
    output.bytes.resize(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, output.bytes.data(), 0, output.bytes.size());
    ++output.count;
    std::printf("[CYCLE_REAL_CAPTURE] tensor=%s type=%s bytes=%zu "
                "ne=[%lld,%lld,%lld] status=OK\n",
                tensor->name, ggml_type_name(tensor->type), output.bytes.size(),
                (long long)tensor->ne[0], (long long)tensor->ne[1],
                (long long)tensor->ne[2]);
    return true;
}

static bool cycle_real_set_compare_layer(int layer) {
    const std::string value = std::to_string(layer);
#ifdef _WIN32
    return _putenv_s("LLAMA_HYBRID_CYCLE_REAL_COMPARE_LAYER", value.c_str()) == 0;
#else
    return setenv("LLAMA_HYBRID_CYCLE_REAL_COMPARE_LAYER", value.c_str(), 1) == 0;
#endif
}

// The fixed Cycle placement variables are process-global and the production
// llama_model_load path intentionally rejects them with split_mode=LAYER.
// Hide only the placement variables while running the independent CPU
// reference. Keep LLAMA_HYBRID_CYCLE_REAL_COMPARE_LAYER active so the full
// Qwen3-MoE graph still exposes its captured Router/FFN outputs.
//
// Both original values are copied before mutating the environment and are
// restored on scope exit, including on any early return.
static bool cycle_real_set_env(const char * name, const char * value) {
#ifdef _WIN32
    return _putenv_s(name, value ? value : "") == 0;
#else
    return value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0;
#endif
}

class cycle_real_cpu_reference_env {
public:
    cycle_real_cpu_reference_env() {
        const char * pc = std::getenv("LLAMA_HYBRID_CYCLE_PC_LAYERS");
        const char * phone = std::getenv("LLAMA_HYBRID_CYCLE_PHONE_LAYERS");
        had_pc_ = pc != nullptr;
        had_phone_ = phone != nullptr;
        if (had_pc_) {
            saved_pc_ = pc;
        }
        if (had_phone_) {
            saved_phone_ = phone;
        }
        const bool pc_cleared =
            cycle_real_set_env("LLAMA_HYBRID_CYCLE_PC_LAYERS", nullptr);
        const bool phone_cleared =
            cycle_real_set_env("LLAMA_HYBRID_CYCLE_PHONE_LAYERS", nullptr);
        disabled_ = pc_cleared && phone_cleared &&
            !llama_hybrid_cycle_requested();
    }

    cycle_real_cpu_reference_env(const cycle_real_cpu_reference_env &) = delete;
    cycle_real_cpu_reference_env & operator=(const cycle_real_cpu_reference_env &) = delete;

    ~cycle_real_cpu_reference_env() {
        const bool pc_restored = cycle_real_set_env(
            "LLAMA_HYBRID_CYCLE_PC_LAYERS",
            had_pc_ ? saved_pc_.c_str() : nullptr);
        const bool phone_restored = cycle_real_set_env(
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS",
            had_phone_ ? saved_phone_.c_str() : nullptr);
        if (!pc_restored || !phone_restored) {
            std::fprintf(stderr,
                "[CYCLE_REAL_REFERENCE] status=FAIL reason=RESTORE_CYCLE_ENV\n");
        }
    }

    bool disabled() const { return disabled_; }

private:
    std::string saved_pc_;
    std::string saved_phone_;
    bool had_pc_ = false;
    bool had_phone_ = false;
    bool disabled_ = false;
};

// The chosen P2C layer must keep Attention, both norms and MoE Router
// exclusively on the sender. A numerically matching FFN is not sufficient
// if the reference graph silently computes its Attention on the PC.
static bool cycle_real_check_phone_weight(
        const char * role,
        const ggml_tensor * tensor,
        ggml_backend_dev_t phone_device) {
    if (!tensor || !phone_device) {
        std::fprintf(stderr,
            "[CYCLE_REAL_OWNER] role=%s status=FAIL reason=MISSING_WEIGHT\n", role);
        return false;
    }
    // The Meta shard accessor deliberately returns nullptr for an empty
    // physical shard. A P2C source-only weight must have NO PC allocation,
    // while its Phone shard must contain the whole original tensor.
    const auto * pc = ggml_backend_meta_get_static_weight_shard(tensor, 0);
    const auto * phone = ggml_backend_meta_get_static_weight_shard(tensor, 1);
    bool same_shape = phone != nullptr && phone->type == tensor->type;
    if (same_shape) {
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            same_shape = same_shape && phone->ne[d] == tensor->ne[d];
        }
    }
    const bool valid =
        pc == nullptr && phone != nullptr && phone->buffer != nullptr &&
        same_shape &&
        ggml_backend_buft_get_device(
            ggml_backend_buffer_get_type(phone->buffer)) == phone_device;
    std::printf(
        "[CYCLE_REAL_OWNER] role=%s source=PHONE_GPU "
        "pc_shard=%s phone_elements=%lld expected=%lld status=%s\n",
        role,
        pc == nullptr ? "EMPTY_EXPECTED" : "UNEXPECTED_PRESENT",
        (long long)(phone ? ggml_nelements(phone) : -1),
        (long long)ggml_nelements(tensor),
        valid ? "PASS" : "FAIL");
    return valid;
}

static bool cycle_real_check_capture(
        const cycle_real_capture & state,
        const std::string & name,
        ggml_type type,
        int64_t d0, int64_t d1, int64_t d2) {
    const auto it = state.values.find(name);
    if (it == state.values.end() || it->second.count != 1) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=NOT_CAPTURED\n",
            name.c_str());
        return false;
    }
    const auto & entry = it->second;
    const size_t expected = size_t(d0) * size_t(d1) * size_t(d2) *
        (type == GGML_TYPE_F32 ? sizeof(float) : sizeof(int32_t));
    const bool ok =
        entry.type == type &&
        entry.ne[0] == d0 && entry.ne[1] == d1 &&
        entry.ne[2] == d2 && entry.ne[3] == 1 &&
        entry.bytes.size() == expected;
    if (!ok) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL "
            "reason=TYPE_OR_SHAPE actual=[%lld,%lld,%lld,%lld] bytes=%zu\n",
            name.c_str(), (long long)entry.ne[0], (long long)entry.ne[1],
            (long long)entry.ne[2], (long long)entry.ne[3],
            entry.bytes.size());
    }
    return ok;
}

template <typename T>
static std::vector<T> cycle_real_read(
        const cycle_real_capture & state, const std::string & name) {
    const auto & bytes = state.values.at(name).bytes;
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

// Correctness-first Phone AttentionPre for ONE contiguous causal prompt
// ubatch (positions [0, n_tokens)). The current Qwen3-MoE K/V values become
// an F16 transient KV block in the same graph. Each token attends ONLY
// positions <= itself, following the same Q/K/V, head-RMSNorm, RoPE, GQA,
// KQ-mask and non-Flash attention operations as the canonical Qwen3 graph.
//
// No historical KV is persisted between calls yet: this validates multi-KV
// *within one batch*, not cross-chunk KV handoff or production scheduling.
struct cycle_real_phone_ready {
    std::vector<float> residual;
    std::vector<float> norm;
    std::vector<int32_t> ids;
    std::vector<float> routing;
    // Full pre-TopK Softmax distribution for score-margin diagnosis only.
    // Not serialized in READY and never used by the FFN execution.
    std::vector<float> router_probs;
};

static bool cycle_real_execute_phone_attention_pre(
        ggml_backend_t backend, const llama_layer & layer,
        const llama_hparams & hp,
        const std::vector<float> & layer_input,
        int64_t hidden, int64_t tokens,
        int64_t topk, int64_t experts,
        cycle_real_phone_ready & result,
        double & elapsed_ms) {
    if (!backend || ggml_backend_is_meta(backend) ||
        hidden <= 0 || tokens < 1 || tokens > 128 ||
        layer_input.size() != size_t(hidden * tokens) ||
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

    // Meta gives a borrowed tensor on the REAL Phone buffer, not a
    // retransmitted or re-quantized copy. Empty PC shards are expected.
    auto get_weight = [](const ggml_tensor * t) -> ggml_tensor * {
        return ggml_backend_meta_get_static_weight_shard(t, 1);
    };
    ggml_tensor * w_attn_norm = get_weight(layer.attn_norm);
    ggml_tensor * wq = get_weight(layer.wq);
    ggml_tensor * wk = get_weight(layer.wk);
    ggml_tensor * wv = get_weight(layer.wv);
    ggml_tensor * wo = get_weight(layer.wo);
    ggml_tensor * wq_norm = get_weight(layer.attn_q_norm);
    ggml_tensor * wk_norm = get_weight(layer.attn_k_norm);
    ggml_tensor * w_ffn_norm = get_weight(layer.ffn_norm);
    ggml_tensor * w_router = get_weight(layer.ffn_gate_inp);

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

    ggml_init_params p {};
    p.mem_size = 16 * 1024 * 1024;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        return false;
    }

    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    // Match the canonical KV-cache mask: [kv_tokens, query_tokens], F16.
    // A single sequence at positions [0,N) needs N usable KV slots.
    auto * causal_mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, tokens, tokens);
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

    std::printf(
        "[CYCLE_PHONE_ATTN_SHAPE] hidden=%lld head_dim=%lld "
        "heads=%lld kv_heads=%lld attn_width=%lld wo_out=%lld "
        "query_tokens=%lld kv_tokens=%lld status=PASS\n",
        (long long)hidden, (long long)head_dim,
        (long long)n_heads, (long long)n_kv_heads,
        (long long)attention_width, (long long)wo->ne[1],
        (long long)tokens, (long long)tokens);

    auto * attn_output = ggml_mul_mat(ctx, wo, attn);
    auto * residual = ggml_add(ctx, attn_output, x);
    auto * norm = ggml_mul(
        ctx, ggml_rms_norm(ctx, residual, hp.f_norm_rms_eps), w_ffn_norm);
    auto * router_logits = ggml_mul_mat(ctx, w_router, norm);
    auto * router_probs = ggml_soft_max(ctx, router_logits);
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

    const llama_hybrid_cycle_ready_graph ready {
        residual, norm, ids, weights
    };
    if (!llama_hybrid_cycle_ready_graph_valid(ready, hidden, tokens, topk)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=READY_SHAPES\n");
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
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

    const auto buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_t compute_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!compute_buffer) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=ALLOC_PHONE_GRAPH\n");
        ggml_free(ctx);
        return false;
    }

    // Input layout: token-major hidden vectors; positions increment 0..N-1.
    // The F16 mask's dimension 0 is the KV index and dimension 1 is query.
    std::vector<int32_t> position_data(size_t(tokens), 0);
    std::vector<ggml_fp16_t> mask_data(size_t(tokens * tokens));
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    int64_t masked_future = 0;
    for (int64_t query = 0; query < tokens; ++query) {
        position_data[size_t(query)] = int32_t(query);
        for (int64_t key = 0; key < tokens; ++key) {
            const bool future = key > query;
            mask_data[size_t(query * tokens + key)] = future ? drop : keep;
            masked_future += future ? 1 : 0;
        }
    }
    if (masked_future != tokens * (tokens - 1) / 2) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_KV] status=FAIL reason=MASK_VALIDATION\n");
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        return false;
    }
    std::printf("[CYCLE_PHONE_KV] query_tokens=%lld kv_tokens=%lld "
                "position_start=0 position_end=%lld "
                "masked_future=%lld mask_type=F16 scope=SINGLE_UBATCH "
                "status=PASS\n",
                (long long)tokens, (long long)tokens,
                (long long)(tokens - 1), (long long)masked_future);

    const auto begin = std::chrono::steady_clock::now();
    ggml_backend_tensor_set(x, layer_input.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(positions, position_data.data(), 0,
                            ggml_nbytes(positions));
    ggml_backend_tensor_set(causal_mask, mask_data.data(), 0,
                            ggml_nbytes(causal_mask));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_synchronize(backend);
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
    std::printf(
        "[CYCLE_PHONE_ATTN_PRE] backend=%s "
        "positions=[0,%lld] tokens=%lld kv_tokens=%lld "
        "heads=%lld kv_heads=%lld graph_nodes=%d "
        "wall_ms=%.3f status=%s\n",
        ggml_backend_name(backend),
        (long long)(tokens - 1), (long long)tokens, (long long)tokens,
        (long long)n_heads, (long long)n_kv_heads,
        ggml_graph_n_nodes(graph), elapsed_ms,
        ok ? "OK" : "FAIL");

    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    return ok;
}

static bool cycle_real_execute_ffn(
        ggml_backend_t backend,
        const llama_hybrid_cycle_ffn_shard_weights & shard,
        const std::vector<float> & norm,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        int64_t hidden, int64_t tokens,
        int64_t topk, int64_t experts,
        std::vector<float> & partial) {
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    auto in = llama_hybrid_cycle_make_ffn_graph_inputs(
        ctx, hidden, tokens, topk);
    auto * x = ggml_reshape_3d(ctx, in.norm, hidden, 1, tokens);
    auto * gate = ggml_mul_mat_id(ctx, shard.gate, x, in.topk_ids);
    auto * up = ggml_mul_mat_id(ctx, shard.up, x, in.topk_ids);
    auto * activation = ggml_swiglu_split(ctx, gate, up);
    auto * down = ggml_mul_mat_id(ctx, shard.down, activation, in.topk_ids);
    auto * weighted = ggml_mul(ctx, down, in.topk_weights);

    ggml_tensor * sum = nullptr;
    for (int64_t k = 0; k < topk; ++k) {
        auto * view = ggml_view_2d(ctx, weighted, hidden, tokens,
                                  weighted->nb[2], size_t(k) * weighted->nb[1]);
        sum = sum ? ggml_add(ctx, sum, view) : view;
    }
    in.ffn_output = ggml_cont(ctx, sum);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, in.ffn_output);

    bool ok = false;
    {
        const char * backend_name = ggml_backend_name(backend);
        std::printf("[CYCLE_REAL_FFN_SHARD] backend=%s channels=%lld nodes=%d "
                    "stage=PREPARE_BEGIN\n",
                    backend_name, (long long)shard.up->ne[1], ggml_graph_n_nodes(graph));
        llama_hybrid_cycle_ffn_executor executor;
        if (!executor.prepare(backend, ctx, graph, in, shard,
                              hidden, tokens, topk, experts)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_FFN_SHARD] backend=%s stage=PREPARE_FAIL "
                "weight_buft=%s compute_buft=%s\n",
                backend_name,
                shard.up->buffer != nullptr ?
                    ggml_backend_buft_name(ggml_backend_buffer_get_type(shard.up->buffer)) :
                    "(unallocated)",
                ggml_backend_buft_name(ggml_backend_get_default_buffer_type(backend)));
        } else {
            partial.resize(size_t(hidden * tokens));
            ok = executor.execute(
                norm.data(), norm.size(),
                ids.data(), ids.size(),
                routing.data(), routing.size(),
                partial.data(), partial.size());
            std::printf("[CYCLE_REAL_FFN_SHARD] backend=%s stage=COMPUTE_END "
                        "elements=%zu status=%s\n",
                        backend_name, partial.size(), ok ? "OK" : "FAIL");
        }
    }
    ggml_free(ctx);
    return ok;
}

struct cycle_real_diff {
    double max_abs = 0;
    double max_ref = 0;
    double rms = 0;
    double cosine = 0;
    double tolerance = 0;
    bool valid = false;
    bool pass = false;
};

static cycle_real_diff cycle_real_compare(
        const std::vector<float> & actual,
        const std::vector<float> & reference) {
    cycle_real_diff result {};
    if (actual.empty() || actual.size() != reference.size()) {
        return result;
    }
    double sum_sq = 0;
    double sum_x_sq = 0;
    double sum_y_sq = 0;
    double dot = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double x = actual[i], y = reference[i];
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return result;
        }
        const double err = std::fabs(x - y);
        result.max_abs = std::max(result.max_abs, err);
        result.max_ref = std::max(result.max_ref, std::fabs(y));
        sum_sq += err * err;
        sum_x_sq += x * x;
        sum_y_sq += y * y;
        dot += x * y;
    }
    result.rms = std::sqrt(sum_sq / actual.size());
    result.cosine = (sum_x_sq > 0 && sum_y_sq > 0) ?
        dot / std::sqrt(sum_x_sq * sum_y_sq) : 0;
    result.tolerance = 0.001 + 0.005 * result.max_ref;
    result.valid = true;
    result.pass = result.max_abs <= result.tolerance;
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 6 || !llama_hybrid_cycle_requested()) {
        std::fprintf(stderr,
            "Usage: test-hybrid-cycle-real-p2c <model.gguf> <phone-ip:port> "
            "<P2C-layer> [prompt] [n_tokens=1..128]\n"
            "Set LLAMA_HYBRID_CYCLE_PC_LAYERS and "
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS first.\n");
        return 2;
    }
    char * end = nullptr;
    const long layer_long = std::strtol(argv[3], &end, 10);
    if (!end || *end != '\0' || layer_long < 0 || layer_long >= 512) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=BAD_LAYER\n");
        return 2;
    }
    const int layer = int(layer_long);
    int32_t target_tokens = 0;
    if (argc == 6) {
        char * token_end = nullptr;
        const long parsed = std::strtol(argv[5], &token_end, 10);
        if (token_end == argv[5] || !token_end || *token_end != '\0' ||
            parsed < 1 || parsed > 128) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL "
                "reason=BAD_TARGET_TOKENS allowed=[1,128]\n");
            return 2;
        }
        target_tokens = int32_t(parsed);
    }
    if (!cycle_real_set_compare_layer(layer)) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=SET_ENV\n");
        return 2;
    }

    llama_backend_init();
    const auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=CPU_DEVICE_MISSING\n");
        return 1;
    }

    const std::string suffix = "-" + std::to_string(layer);
    // The previous layer's complete output is the exact input of Layer 7
    // Attention (Qwen3-MoE's inpL). Capture it from the independent CPU
    // baseline; do not substitute an arbitrary activation.
    const std::string attn_input_name =
        "l_out-" + std::to_string(layer - 1);
    const std::string residual_name = "ffn_inp" + suffix;
    const std::string norm_name = "ffn_norm" + suffix;
    const std::string ids_name = "cycle_real_topk_ids" + suffix;
    const std::string routing_name = "cycle_real_topk_weights" + suffix;
    // Existing canonical Router callback before argsort/topk.
    const std::string probs_name = "ffn_moe_probs" + suffix;
    const std::string ffn_name = "ffn_moe_out" + suffix;
    const std::string output_name = "l_out" + suffix;
    cycle_real_capture capture;
    for (const auto & name : {attn_input_name, residual_name, norm_name,
                             ids_name, routing_name, probs_name,
                             ffn_name, output_name}) {
        capture.values.emplace(name, cycle_real_capture_value {});
    }

    const std::string prompt = argc >= 5 ? argv[4] :
        "The quick brown fox jumps over the lazy dog. Please explain a transformer.";
    int32_t ntokens = 0;
    int32_t reference_layers = 0;
    int64_t reference_hidden = 0;
    int64_t reference_topk = 0;
    int64_t reference_experts = 0;
    double forward_ms = 0.0;

    // The Cycle model has a placement-only C2P/P2C policy and MUST NOT run
    // through the legacy Meta full-model scheduler. Establish an independent
    // complete-layer reference on ordinary CPU first. Its GGUF mmap and KV
    // resources are freed before loading the physical PC/Phone Cycle shards,
    // so the two copies of a 30B model never coexist in memory.
    {
        cycle_real_cpu_reference_env cpu_reference_env;
        if (!cpu_reference_env.disabled()) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=DISABLE_CYCLE_ENV\n");
            return 1;
        }
        std::printf("[CYCLE_REAL_REFERENCE] cycle_placement_env=DISABLED "
                    "router_capture_env=PRESERVED status=OK\n");
        ggml_backend_dev_t reference_devices[2] = {cpu_dev, nullptr};
        llama_model_params reference_params = llama_model_default_params();
        reference_params.devices = reference_devices;
        reference_params.n_gpu_layers = 0;
        reference_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        reference_params.hybrid_auto = false;
        reference_params.use_mmap = true;
        // A complete 30B Q4_K CPU reference is loaded only to capture
        // numerical intermediates. CPU_REPACK duplicates many mapped weights
        // (~13 GiB in this GGUF) and is unnecessary for this correctness
        // probe. Keep the model's original quantization and mmap buffers.
        // This does not change the second PC+Phone Cycle model's policy.
        reference_params.use_extra_bufts = false;
        std::printf("[CYCLE_REAL_REFERENCE] backend=CPU_ONLY "
                    "split_mode=LAYER cpu_repack=DISABLED model_load=BEGIN\n");
        std::unique_ptr<llama_model, decltype(&llama_model_free)> reference_model(
            llama_model_load_from_file(argv[1], reference_params), &llama_model_free);
        reference_layers = reference_model ?
            llama_model_n_layer(reference_model.get()) : 0;
        if (!reference_model ||
            reference_model->arch != LLM_ARCH_QWEN3MOE ||
            reference_layers <= 1 || layer >= reference_layers - 1) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=REFERENCE_MODEL_OR_LAYER\n");
            return 1;
        }
        reference_hidden = reference_model->hparams.n_embd;
        reference_topk = reference_model->hparams.n_expert_used;
        reference_experts = reference_model->hparams.n_expert;

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.cb_eval = cycle_real_capture_cb;
        cp.cb_eval_user_data = &capture;
        std::unique_ptr<llama_context, decltype(&llama_free)> reference_context(
            llama_init_from_model(reference_model.get(), cp), &llama_free);
        if (!reference_context) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=REFERENCE_CONTEXT_INIT\n");
            return 1;
        }
        // A fixed token-count case is more reproducible than relying on
        // how many BPE tokens a short user-supplied prompt happens to use.
        // Repeat the seed only when necessary; truncate the resulting token
        // sequence to exactly the requested length before llama_decode.
        std::vector<llama_token> tokens(4096);
        const auto * vocab = llama_model_get_vocab(reference_model.get());
        std::string sample = prompt.empty() ? "Hello" : prompt;
        for (int attempt = 0; attempt < 256; ++attempt) {
            ntokens = llama_tokenize(
                vocab, sample.c_str(), int(sample.size()),
                tokens.data(), int32_t(tokens.size()), true, true);
            if (ntokens < 0) {
                tokens.resize(size_t(-ntokens));
                ntokens = llama_tokenize(
                    vocab, sample.c_str(), int(sample.size()),
                    tokens.data(), int32_t(tokens.size()), true, true);
            }
            if (ntokens < 0 ||
                target_tokens == 0 || ntokens >= target_tokens) {
                break;
            }
            if (sample.size() > 65536) {
                break;
            }
            sample.append(" ");
            sample.append(prompt.empty() ? "Hello" : prompt);
        }
        if (target_tokens > 0 && ntokens >= target_tokens) {
            ntokens = target_tokens;
        }
        if (ntokens < 1 || ntokens > int32_t(cp.n_batch) ||
            ntokens > int32_t(cp.n_ctx) ||
            (target_tokens > 0 && ntokens != target_tokens)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=TOKEN_COUNT "
                "tokens=%d max=%u\n",
                int(ntokens), cp.n_batch);
            return 1;
        }

        llama_batch batch = llama_batch_init(ntokens, 0, 1);
        batch.n_tokens = ntokens;
        for (int i = 0; i < ntokens; ++i) {
            batch.token[i] = tokens[size_t(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == ntokens - 1) ? 1 : 0;
        }
        std::printf(
            "[CYCLE_REAL_FORWARD] layer=%d prompt_tokens=%d "
            "target_tokens=%d positions=[0,%d] "
            "source=CPU_ONLY_REFERENCE stage=BEGIN\n",
            layer, int(ntokens), int(target_tokens), int(ntokens - 1));
        const auto forward_begin = std::chrono::steady_clock::now();
        const int decode_status =
            llama_decode(reference_context.get(), batch);
        llama_synchronize(reference_context.get());
        forward_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - forward_begin).count();
        llama_batch_free(batch);
        if (decode_status != 0 || capture.error) {
            std::fprintf(stderr,
                "[CYCLE_REAL_FORWARD] status=FAIL "
                "reason=CPU_REFERENCE_DECODE_OR_CAPTURE code=%d\n",
                decode_status);
            return 1;
        }
        std::printf(
            "[CYCLE_REAL_FORWARD] source=CPU_ONLY_REFERENCE "
            "stage=END wall_ms=%.3f status=PASS\n",
            forward_ms);
    }
    std::printf("[CYCLE_REAL_REFERENCE] model=CPU_ONLY "
                "status=RELEASED_BEFORE_CYCLE_LOAD\n");

    // All N tokens are computed together at positions [0,N); the Phone
    // stage builds a causal F16 mask and may read earlier tokens' K/V.
    // This is NOT yet a persistent KV cache across multiple separate
    // llama_decode calls or distributed Cycle chunks.
    std::printf("[CYCLE_REAL_KV_SCOPE] tokens=%d position_end=%d "
                "kind=CONTIGUOUS_PROMPT_CACHE persistent=0 status=OK\n",
                int(ntokens), int(ntokens - 1));
    // The RAII guard restored the physical Cycle placement policy. Verify
    // the requested P2C layer before loading the distributed model.
    llama_hybrid_cycle_layout expected_layout;
    if (!llama_hybrid_cycle_layout_from_env(
            reference_layers, expected_layout) ||
        expected_layout.kinds[size_t(layer)] !=
            llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=RESTORED_LAYOUT_OR_NOT_P2C\n");
        return 1;
    }
    std::printf("[CYCLE_REAL_REFERENCE] cycle_placement_env=RESTORED "
                "target_layer=%d kind=P2C status=OK\n", layer);

    // Fail on a malformed/uncaptured CPU baseline before allocating the
    // much larger distributed PC/Phone weight buffers.
    if (!cycle_real_check_capture(
            capture, attn_input_name, GGML_TYPE_F32, reference_hidden, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, residual_name, GGML_TYPE_F32, reference_hidden, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, norm_name, GGML_TYPE_F32, reference_hidden, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, ids_name, GGML_TYPE_I32, reference_topk, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, routing_name, GGML_TYPE_F32, 1, reference_topk, ntokens) ||
        !cycle_real_check_capture(
            capture, probs_name, GGML_TYPE_F32, reference_experts, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, ffn_name, GGML_TYPE_F32, reference_hidden, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, output_name, GGML_TYPE_F32, reference_hidden, ntokens, 1)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=CPU_REFERENCE_CAPTURE_MISSING\n");
        return 1;
    }
    std::printf("[CYCLE_REAL_REFERENCE] "
                "capture_fields=8 status=PASS\n");

    // A new model instance holds only the *real* physical Cycle weights:
    // PC CPU FFN shard + Phone OpenCL FFN shard, with Attention/Router
    // weights resident on Phone. No llama_context/full-graph Meta execution
    // is permitted for this placement-only model.
    const auto reg = ggml_backend_rpc_add_server(argv[2]);
    if (!reg || ggml_backend_reg_dev_count(reg) != 1) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=RPC_DEVICE_COUNT\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(reg, 0);
    if (!phone_dev ||
        ggml_backend_dev_type(phone_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=NOT_PHONE_GPU\n");
        return 1;
    }
    ggml_backend_dev_t devices[3] = {cpu_dev, phone_dev, nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.devices = devices;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.hybrid_auto = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), &llama_model_free);
    const int32_t model_layers =
        model ? llama_model_n_layer(model.get()) : 0;
    if (!model || model->arch != LLM_ARCH_QWEN3MOE ||
        model_layers != reference_layers ||
        layer >= model_layers - 1 ||
        model->hybrid_layer_modes.size() != size_t(model_layers) ||
        model->hybrid_layer_modes[size_t(layer)] !=
            llama_hybrid_layer_mode::TENSOR_P2C) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL "
            "reason=MODEL_OR_NOT_P2C_LAYER layer=%d\n", layer);
        return 1;
    }

    const auto & lw = model->layers[size_t(layer)];
    if (!cycle_real_check_phone_weight("ATTN_NORM", lw.attn_norm, phone_dev) ||
        !cycle_real_check_phone_weight("ATTN_Q", lw.wq, phone_dev) ||
        !cycle_real_check_phone_weight("ATTN_K", lw.wk, phone_dev) ||
        !cycle_real_check_phone_weight("ATTN_V", lw.wv, phone_dev) ||
        !cycle_real_check_phone_weight("ATTN_Q_NORM", lw.attn_q_norm, phone_dev) ||
        !cycle_real_check_phone_weight("ATTN_K_NORM", lw.attn_k_norm, phone_dev) ||
        !cycle_real_check_phone_weight("ATTN_OUT", lw.wo, phone_dev) ||
        !cycle_real_check_phone_weight("FFN_NORM", lw.ffn_norm, phone_dev) ||
        !cycle_real_check_phone_weight("MOE_ROUTER", lw.ffn_gate_inp, phone_dev)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL "
            "reason=ATTENTION_ROUTER_NOT_PHONE_OWNED\n");
        return 1;
    }
    const int64_t hidden = model->hparams.n_embd;
    const int64_t topk = model->hparams.n_expert_used;
    const int64_t experts = model->hparams.n_expert;
    if (hidden != reference_hidden || topk != reference_topk ||
        experts != reference_experts) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL "
            "reason=REFERENCE_CYCLE_HPARAM_MISMATCH\n");
        return 1;
    }
    // The first real comparison covers ordinary Qwen3-MoE SwiGLU, without
    // extra per-expert scales or clamping. Reject unsupported formulas.
    if (model->hparams.swiglu_clamp_exp[size_t(layer)] > 1.0e-6f ||
        lw.ffn_gate_exps_s || lw.ffn_up_exps_s || lw.ffn_down_exps_s ||
        !llama_hybrid_cycle_ffn_partition_valid(
            lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps,
            hidden, experts)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=WEIGHT_SHARD_OR_SCALES\n");
        return 1;
    }
    const auto pc_weights = llama_hybrid_cycle_get_physical_ffn_weights(
        lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps, 0);
    const auto phone_weights = llama_hybrid_cycle_get_physical_ffn_weights(
        lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps, 1);
    std::printf("[CYCLE_REAL_MODEL] layer=%d hidden=%lld topk=%lld "
                "experts=%lld pc_channels=%lld phone_channels=%lld "
                "type=%s status=OK\n",
                layer, (long long)hidden, (long long)topk,
                (long long)experts, (long long)pc_weights.up->ne[1],
                (long long)phone_weights.up->ne[1],
                ggml_type_name(lw.ffn_up_exps->type));

    const auto ref_input = cycle_real_read<float>(capture, attn_input_name);
    const auto ref_residual = cycle_real_read<float>(capture, residual_name);
    const auto ref_norm = cycle_real_read<float>(capture, norm_name);
    const auto ref_ids = cycle_real_read<int32_t>(capture, ids_name);
    const auto ref_routing = cycle_real_read<float>(capture, routing_name);
    const auto ref_router_probs = cycle_real_read<float>(capture, probs_name);
    const auto baseline_ffn = cycle_real_read<float>(capture, ffn_name);
    const auto baseline_output = cycle_real_read<float>(capture, output_name);

    // Sanity-check the *reference* before comparing any new execution.
    // For ordinary Qwen3-MoE with no adapters this layer must implement
    // l_out = AttentionResidual + CompleteFFN. Otherwise a capture label
    // or a prior graph transformation is not comparing like with like.
    if (baseline_ffn.size() != ref_residual.size() ||
        baseline_output.size() != ref_residual.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=BASELINE_SIZE\n");
        return 1;
    }
    std::vector<float> baseline_reconstructed(ref_residual.size());
    for (size_t i = 0; i < ref_residual.size(); ++i) {
        baseline_reconstructed[i] = ref_residual[i] + baseline_ffn[i];
    }
    const cycle_real_diff baseline_check =
        cycle_real_compare(baseline_reconstructed, baseline_output);
    std::printf("[CYCLE_REAL_COMPARE] target=BASELINE_INTERNAL layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, baseline_check.max_abs, baseline_check.rms,
                baseline_check.cosine, baseline_check.tolerance,
                baseline_check.valid && baseline_check.pass ? "PASS" : "FAIL");
    if (!baseline_check.valid || !baseline_check.pass) {
        return 1;
    }
    // The CPU reference is read-only from this point on. The READY wire and
    // the two physical FFN shards must consume Phone-generated arrays only.
    ggml_backend_t phone = ggml_backend_dev_init(phone_dev, nullptr);
    if (!phone || !ggml_backend_is_rpc(phone)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=PHONE_RPC_BACKEND\n");
        return 1;
    }
    cycle_real_phone_ready phone_ready;
    double phone_attn_ms = 0.0;
    if (!cycle_real_execute_phone_attention_pre(
            phone, lw, model->hparams, ref_input,
            hidden, ntokens, topk, experts, phone_ready, phone_attn_ms)) {
        ggml_backend_free(phone);
        return 1;
    }

    const cycle_real_diff residual_diff =
        cycle_real_compare(phone_ready.residual, ref_residual);
    const cycle_real_diff norm_diff =
        cycle_real_compare(phone_ready.norm, ref_norm);
    const cycle_real_diff routing_slot_diff =
        cycle_real_compare(phone_ready.routing, ref_routing);
    const cycle_real_diff probs_diff =
        cycle_real_compare(phone_ready.router_probs, ref_router_probs);
    const bool topk_equal = phone_ready.ids == ref_ids;

    // Experts are a weighted SET, not an ordered semantic vector. argsort
    // can permute two close-scoring experts between CPU and OpenCL without
    // changing the selected expert set. Every weight is attached to its
    // Expert ID: never compare or reorder weights independently of IDs.
    const size_t expected_slots = size_t(ntokens) * size_t(topk);
    const size_t expected_probs = size_t(ntokens) * size_t(experts);
    if (phone_ready.ids.size() != expected_slots ||
        ref_ids.size() != expected_slots ||
        phone_ready.routing.size() != expected_slots ||
        ref_routing.size() != expected_slots ||
        phone_ready.router_probs.size() != expected_probs ||
        ref_router_probs.size() != expected_probs) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ROUTER_SEMANTIC] status=FAIL reason=SHAPE_MISMATCH\n");
        ggml_backend_free(phone);
        return 1;
    }
    std::vector<float> aligned_phone_weights;
    std::vector<float> aligned_ref_weights;
    aligned_phone_weights.reserve(expected_slots);
    aligned_ref_weights.reserve(expected_slots);
    bool topk_set_equal = true;
    bool topk_ids_valid = true;
    int32_t permuted_tokens = 0;
    size_t changed_slots = 0;
    constexpr size_t max_printed_router_diff = 16;
    for (int32_t token = 0; token < ntokens; ++token) {
        std::vector<std::pair<int32_t, float>> ref_pairs;
        std::vector<std::pair<int32_t, float>> phone_pairs;
        ref_pairs.reserve(size_t(topk));
        phone_pairs.reserve(size_t(topk));
        bool token_reordered = false;
        for (int64_t slot = 0; slot < topk; ++slot) {
            const size_t index = size_t(token) * size_t(topk) + size_t(slot);
            const int32_t ref_id = ref_ids[index];
            const int32_t phone_id = phone_ready.ids[index];
            const float ref_weight = ref_routing[index];
            const float phone_weight = phone_ready.routing[index];
            if (ref_id < 0 || ref_id >= experts ||
                phone_id < 0 || phone_id >= experts ||
                !std::isfinite(ref_weight) || !std::isfinite(phone_weight)) {
                topk_ids_valid = false;
                continue;
            }
            ref_pairs.emplace_back(ref_id, ref_weight);
            phone_pairs.emplace_back(phone_id, phone_weight);
            if (ref_id != phone_id) {
                token_reordered = true;
                if (changed_slots < max_printed_router_diff) {
                    // Show whether the scores are nearly tied or genuinely
                    // rank-inverted. Softmax distributions have [expert,token]
                    // layout; use the FULL scores, not normalized TopK slots.
                    const size_t off = size_t(token) * size_t(experts);
                    const float ref_a = ref_router_probs[off + size_t(ref_id)];
                    const float ref_b = ref_router_probs[off + size_t(phone_id)];
                    const float ph_a = phone_ready.router_probs[off + size_t(ref_id)];
                    const float ph_b = phone_ready.router_probs[off + size_t(phone_id)];
                    std::printf(
                        "[CYCLE_PHONE_ROUTER_RANK] pos=%d slot=%lld "
                        "reference_id=%d phone_id=%d "
                        "ref_gap=%.8g phone_gap=%.8g "
                        "ref_scores=[%.8g,%.8g] phone_scores=[%.8g,%.8g]\n",
                        token, (long long)slot, ref_id, phone_id,
                        double(ref_a) - double(ref_b),
                        double(ph_a) - double(ph_b),
                        ref_a, ref_b, ph_a, ph_b);
                }
                ++changed_slots;
            }
        }
        if (token_reordered) {
            ++permuted_tokens;
        }
        if (ref_pairs.size() != size_t(topk) ||
            phone_pairs.size() != size_t(topk)) {
            topk_set_equal = false;
            continue;
        }
        const auto by_id = [](const std::pair<int32_t, float> & a,
                              const std::pair<int32_t, float> & b) {
            return a.first < b.first;
        };
        std::sort(ref_pairs.begin(), ref_pairs.end(), by_id);
        std::sort(phone_pairs.begin(), phone_pairs.end(), by_id);
        bool this_set_equal = true;
        for (size_t slot = 0; slot < size_t(topk); ++slot) {
            // Duplicate IDs make an expert set ambiguous; fail closed.
            if (slot > 0 && (ref_pairs[slot].first == ref_pairs[slot - 1].first ||
                             phone_pairs[slot].first == phone_pairs[slot - 1].first)) {
                topk_ids_valid = false;
            }
            if (ref_pairs[slot].first != phone_pairs[slot].first) {
                this_set_equal = false;
            }
        }
        if (!this_set_equal) {
            topk_set_equal = false;
            std::fprintf(stderr,
                "[CYCLE_PHONE_TOPK_SET] pos=%d status=FAIL reason=CHANGED_EXPERT_SET\n",
                token);
            continue;
        }
        // Compare matched Expert ID + normalized weight, in sorted-ID order
        // only for numerical diagnosis. FFN still consumes phone-original
        // ID/weight pairs without modifying their original slot order.
        for (size_t slot = 0; slot < size_t(topk); ++slot) {
            aligned_ref_weights.push_back(ref_pairs[slot].second);
            aligned_phone_weights.push_back(phone_pairs[slot].second);
        }
    }
    const cycle_real_diff routing_aligned_diff =
        cycle_real_compare(aligned_phone_weights, aligned_ref_weights);

    // Print which *position* diverges, not just one pooled statistic.
    // In a causal multi-KV test, a future-mask or RoPE error can be local
    // to later tokens even when the first token matches perfectly.
    const auto report_token_errors = [&](const char * field,
                                         const std::vector<float> & actual,
                                         const std::vector<float> & baseline,
                                         int64_t elems_per_token) {
        if (actual.size() != baseline.size() ||
            actual.size() != size_t(elems_per_token * ntokens)) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_READY_TOKEN] field=%s status=FAIL "
                "reason=SHAPE_MISMATCH\n", field);
            return;
        }
        double max_abs = 0.0;
        int64_t worst_token = -1;
        int64_t worst_element = -1;
        for (int32_t token = 0; token < ntokens; ++token) {
            for (int64_t j = 0; j < elems_per_token; ++j) {
                const size_t idx = size_t(token * elems_per_token + j);
                const double err = std::fabs(
                    double(actual[idx]) - double(baseline[idx]));
                if (err > max_abs || worst_token < 0) {
                    max_abs = err;
                    worst_token = token;
                    worst_element = j;
                }
            }
        }
        std::printf("[CYCLE_PHONE_READY_TOKEN] field=%s tokens=%d "
                    "worst_pos=%lld worst_element=%lld max_abs=%.8g "
                    "status=MEASURED\n",
                    field, int(ntokens),
                    (long long)worst_token, (long long)worst_element,
                    max_abs);
    };
    report_token_errors(
        "RESIDUAL", phone_ready.residual, ref_residual, hidden);
    report_token_errors(
        "FFN_NORM", phone_ready.norm, ref_norm, hidden);
    report_token_errors(
        "ROUTING_WEIGHTS_SLOT_ORDER",
        phone_ready.routing, ref_routing, topk);

    // Do NOT reuse the relaxed full-layer tolerance here. The residual
    // contains large activations; requiring its comparison to pass only
    // via 0.005*max_ref would hide significant Attention errors.
    const auto strict_ok = [](const cycle_real_diff & d,
                              double floor, double relative) {
        return d.valid && d.max_abs <= floor + relative * d.max_ref;
    };
    const bool residual_ok = strict_ok(residual_diff, 0.02, 0.00001);
    const bool norm_ok = strict_ok(norm_diff, 0.01, 0.001);
    const bool routing_ok =
        topk_set_equal && topk_ids_valid &&
        aligned_ref_weights.size() == expected_slots &&
        strict_ok(routing_aligned_diff, 0.003, 0.001);

    std::printf("[CYCLE_PHONE_READY_COMPARE] target=RESIDUAL "
                "max_abs=%.8g rms=%.8g cosine=%.8g "
                "strict_tolerance=%.8g status=%s\n",
                residual_diff.max_abs, residual_diff.rms,
                residual_diff.cosine,
                0.02 + 0.00001 * residual_diff.max_ref,
                residual_ok ? "PASS" : "FAIL");
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=FFN_NORM "
                "max_abs=%.8g rms=%.8g cosine=%.8g "
                "strict_tolerance=%.8g status=%s\n",
                norm_diff.max_abs, norm_diff.rms,
                norm_diff.cosine, 0.01 + 0.001 * norm_diff.max_ref,
                norm_ok ? "PASS" : "FAIL");
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=TOPK "
                "ids_exact=%d set_equal=%d valid_ids=%d "
                "permuted_tokens=%d changed_slots=%zu status=%s\n",
                topk_equal ? 1 : 0, topk_set_equal ? 1 : 0,
                topk_ids_valid ? 1 : 0, int(permuted_tokens), changed_slots,
                topk_set_equal && topk_ids_valid ? "PASS" : "FAIL");
    if (!topk_equal) {
        size_t mismatch_count = 0;
        constexpr size_t max_printed = 16;
        for (size_t j = 0; j < ref_ids.size(); ++j) {
            if (ref_ids[j] == phone_ready.ids[j]) {
                continue;
            }
            if (mismatch_count < max_printed) {
                std::fprintf(stderr,
                    "[CYCLE_PHONE_TOPK_MISMATCH] idx=%zu "
                    "pos=%zu expert_slot=%zu reference=%d phone=%d\n",
                    j, j / size_t(topk), j % size_t(topk),
                    ref_ids[j], phone_ready.ids[j]);
            }
            ++mismatch_count;
        }
        std::fprintf(stderr,
            "[CYCLE_PHONE_TOPK_MISMATCH_SUM] tokens=%d "
            "total=%zu displayed_max=%zu\n",
            int(ntokens), mismatch_count, max_printed);
    }
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=ROUTER_PROBS "
                "max_abs=%.8g rms=%.8g status=MEASURED\n",
                probs_diff.max_abs, probs_diff.rms);
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=ROUTING_WEIGHTS_SLOT_ORDER "
                "max_abs=%.8g rms=%.8g status=DIAGNOSTIC_ONLY\n",
                routing_slot_diff.max_abs, routing_slot_diff.rms);
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=ROUTING_WEIGHTS_BY_EXPERT "
                "max_abs=%.8g rms=%.8g cosine=%.8g "
                "strict_tolerance=%.8g status=%s\n",
                routing_aligned_diff.max_abs, routing_aligned_diff.rms,
                routing_aligned_diff.cosine,
                0.003 + 0.001 * routing_aligned_diff.max_ref,
                routing_ok ? "PASS" : "FAIL");
    if (!residual_ok || !norm_ok || !probs_diff.valid ||
        !topk_set_equal || !topk_ids_valid || !routing_ok) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL "
            "reason=CPU_REFERENCE_MISMATCH\n");
        ggml_backend_free(phone);
        return 1;
    }

    const auto & residual = phone_ready.residual;
    const auto & norm = phone_ready.norm;
    const auto & ids = phone_ready.ids;
    const auto & routing = phone_ready.routing;
    for (int32_t id : ids) {
        if (id < 0 || id >= experts) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=INVALID_TOPK id=%d\n", id);
            ggml_backend_free(phone);
            return 1;
        }
    }
    for (float v : norm) {
        if (!std::isfinite(v)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=NONFINITE_NORM\n");
            ggml_backend_free(phone);
            return 1;
        }
    }
    for (float w : routing) {
        if (!std::isfinite(w)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=NONFINITE_ROUTING\n");
            ggml_backend_free(phone);
            return 1;
        }
    }
    std::printf("[CYCLE_REAL_READY] layer=%d tokens=%d "
                "fields=RESIDUAL,NORM,TOPK,WEIGHTS "
                "source=PHONE_GPU_ATTN_PRE status=OK\n", layer, int(ntokens));

    // Exercise the V1 P2C READY wire representation with the *real* tensor
    // values, including frame direction and seq. This is an in-process
    // encode/decode check, NOT a claim that the new network sender is wired.
    const inference_cycle_ready_view ready_source {
        uint32_t(ntokens), uint32_t(hidden), uint32_t(topk),
        residual.data(), norm.data(), ids.data(), routing.data(),
    };
    size_t ready_bytes = 0;
    if (!inference_cycle_payload_size(
            INFERENCE_CYCLE_READY, uint32_t(ntokens), uint32_t(hidden),
            uint32_t(topk), ready_bytes)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_SIZE\n");
        return 1;
    }
    std::vector<uint8_t> ready_wire(ready_bytes);
    size_t ready_written = 0;
    if (!inference_cycle_write_ready(
            ready_wire.data(), ready_wire.size(), ready_source, ready_written) ||
        ready_written != ready_wire.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_ENCODE\n");
        return 1;
    }
    inference_frame_header ready_header = inference_make_header(INFERENCE_FRAME_DATA);
    ready_header.stage = INFERENCE_STAGE_CYCLE_READY;
    ready_header.tensor_role = INFERENCE_CYCLE_PHONE_TO_PC;
    ready_header.layer = layer;
    ready_header.chunk = 0;
    ready_header.seq = 1;
    ready_header.graph_id = 1;
    ready_header.payload_bytes = ready_wire.size();
    inference_cycle_message ready_msg;
    if (!inference_cycle_decode_frame(
            ready_header, ready_wire.data(), ready_wire.size(), ready_msg) ||
        ready_msg.kind != INFERENCE_CYCLE_READY ||
        ready_msg.direction != INFERENCE_CYCLE_PHONE_TO_PC ||
        ready_msg.layer != layer || ready_msg.seq != 1 ||
        ready_msg.ready.topk_ids != ids ||
        ready_msg.ready.topk_weights != routing ||
        ready_msg.ready.ffn_norm != norm ||
        ready_msg.ready.residual != residual) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_DECODE\n");
        return 1;
    }
    std::printf("[CYCLE_REAL_WIRE] stage=READY direction=P2C "
                "bytes=%zu mode=IN_PROCESS_CODEC status=PASS\n", ready_bytes);

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PHYSICAL_BACKENDS\n");
        return 1;
    }

    const auto parts_begin = std::chrono::steady_clock::now();
    auto phone_task = std::async(std::launch::async, [&] {
        std::vector<float> part;
        if (!cycle_real_execute_ffn(
                phone, phone_weights, norm, ids, routing,
                hidden, ntokens, topk, experts, part)) {
            part.clear();
        }
        return part;
    });
    std::vector<float> pc_part;
    // PC FFN consumes the decoded Phone READY while the Phone FFN runs on
    // the same Phone backend after its own AttentionPre is complete.
    // The dedicated network two-send FIFO remains a separate next step.
    const bool pc_ok = cycle_real_execute_ffn(
        cpu, pc_weights,
        ready_msg.ready.ffn_norm, ready_msg.ready.topk_ids,
        ready_msg.ready.topk_weights,
        hidden, ntokens, topk, experts, pc_part);
    const auto phone_part = phone_task.get();
    const double parts_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - parts_begin).count();

    ggml_backend_free(phone);
    ggml_backend_free(cpu);

    if (!pc_ok || pc_part.size() != residual.size() ||
        phone_part.size() != residual.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PHYSICAL_FFN_COMPUTE\n");
        return 1;
    }

    // SEND #2: serialize the actual Phone GPU FFN contribution and decode it
    // as the PC receiver would. Both payload frames have the SAME direction.
    const inference_cycle_partial_view partial_source {
        uint32_t(ntokens), uint32_t(hidden), phone_part.data(),
    };
    size_t partial_bytes = 0;
    if (!inference_cycle_payload_size(
            INFERENCE_CYCLE_PARTIAL, uint32_t(ntokens), uint32_t(hidden),
            0, partial_bytes)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_SIZE\n");
        return 1;
    }
    std::vector<uint8_t> partial_wire(partial_bytes);
    size_t partial_written = 0;
    if (!inference_cycle_write_partial(
            partial_wire.data(), partial_wire.size(),
            partial_source, partial_written) ||
        partial_written != partial_wire.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_ENCODE\n");
        return 1;
    }
    inference_frame_header partial_header = inference_make_header(INFERENCE_FRAME_DATA);
    partial_header.stage = INFERENCE_STAGE_CYCLE_PARTIAL;
    partial_header.tensor_role = INFERENCE_CYCLE_PHONE_TO_PC;
    partial_header.layer = layer;
    partial_header.chunk = 0;
    partial_header.seq = ready_header.seq;
    partial_header.graph_id = ready_header.graph_id;
    partial_header.payload_bytes = partial_wire.size();
    inference_cycle_message partial_msg;
    if (!inference_cycle_decode_frame(
            partial_header, partial_wire.data(), partial_wire.size(), partial_msg) ||
        partial_msg.kind != INFERENCE_CYCLE_PARTIAL ||
        partial_msg.direction != ready_msg.direction ||
        partial_msg.layer != ready_msg.layer ||
        partial_msg.seq != ready_msg.seq ||
        partial_msg.graph_id != ready_msg.graph_id ||
        partial_msg.partial.partial != phone_part) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_DECODE\n");
        return 1;
    }
    std::printf("[CYCLE_REAL_WIRE] stage=PARTIAL direction=P2C "
                "bytes=%zu mode=IN_PROCESS_CODEC status=PASS\n", partial_bytes);

    std::vector<float> combined_ffn(residual.size());
    std::vector<float> combined_output(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        combined_ffn[i] = pc_part[i] + partial_msg.partial.partial[i];
    }
    if (!inference_cycle_reduce_p2c_pc_cpu(
            ready_msg.ready, pc_part.data(), pc_part.size(),
            partial_msg.partial, combined_output.data(), combined_output.size())) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PC_REDUCE\n");
        return 1;
    }
    const cycle_real_diff ffn_diff = cycle_real_compare(combined_ffn, baseline_ffn);
    const cycle_real_diff out_diff = cycle_real_compare(combined_output, baseline_output);
    report_token_errors("FFN_PARTIAL_SUM", combined_ffn, baseline_ffn, hidden);
    report_token_errors("LAYER_OUTPUT", combined_output, baseline_output, hidden);
    std::printf("[CYCLE_REAL_COMPARE] target=FFN_PARTIAL_SUM layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, ffn_diff.max_abs, ffn_diff.rms, ffn_diff.cosine,
                ffn_diff.tolerance, ffn_diff.pass ? "PASS" : "FAIL");
    std::printf("[CYCLE_REAL_COMPARE] target=LAYER_OUTPUT layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, out_diff.max_abs, out_diff.rms, out_diff.cosine,
                out_diff.tolerance, out_diff.pass ? "PASS" : "FAIL");
    // The generic layer metric has a deliberately relaxed scale-based
    // tolerance inherited from older comparisons; it can exceed 1.0 for
    // large residuals. A real multi-KV correctness gate must also require
    // an independent absolute/relative layer bound.
    const double strict_output_tolerance = 0.02 + 0.00002 * out_diff.max_ref;
    const bool strict_output_ok = out_diff.valid &&
        out_diff.max_abs <= strict_output_tolerance;
    std::printf("[CYCLE_REAL_COMPARE] target=LAYER_OUTPUT_STRICT "
                "layer=%d max_abs=%.8g tolerance=%.8g status=%s\n",
                layer, out_diff.max_abs, strict_output_tolerance,
                strict_output_ok ? "PASS" : "FAIL");
    const bool pass = ffn_diff.valid && out_diff.valid &&
                      ffn_diff.pass && out_diff.pass && strict_output_ok;
    std::printf("[CYCLE_REAL_P2C_SUM] layer=%d tokens=%d "
                "forward_ms=%.3f ffn_parallel_ms=%.3f "
                "real_attention=PHONE_GPU_CAUSAL_MULTI_KV phone_attention_executed=1 "
                "phone_attn_ms=%.3f kv_scope=ONE_UBATCH kv_persistent=0 "
                "real_ffn_weights=1 "
                "ffn_shards=PC_CPU+PHONE_GPU reduce_owner=PC_CPU "
                "reference=CPU_FULL_QWEN3_LAYER data_plane=RPC_BACKEND_ONLY "
                "cycle_wire=IN_PROCESS_CODEC_ONLY status=%s\n",
                layer, int(ntokens), forward_ms, parts_ms,
                phone_attn_ms, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
