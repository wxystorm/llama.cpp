#pragma once

// One-off test-only GPU A/B replay on EXACTLY the CPU oracle's input.
// The normal 48-layer FIFO is complete before this runs. The CPU reference
// after_layer[L-1] is explicitly uploaded once to Phone GPU, so upstream
// numerical drift is excluded. No model Graph/Buffer/transport is borrowed
// from the FIFO session; this probe owns its Attention scratch and FFN slot.
// Single 16-token causal chunk only, no previous-layer KV dependency.

#include "hybrid-cycle-full-reference.h"
#include "llama-hybrid-cycle-attention.h"
#include "llama-hybrid-cycle-ffn-shard.h"
#include "llama-hybrid-cycle-phone-executor.h"
#include "llama-hybrid-cycle-physical.h"
#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace cycle_same_input {

struct comparison {
    bool valid = false;
    bool strict = false;
    double max_abs = 0.0;
    double rms = 0.0;
    double cosine = 0.0;
};

inline comparison compare(
        int layer, const char * point,
        const std::vector<float> & actual,
        const std::vector<float> & reference) {
    comparison result {};
    if (actual.empty() || actual.size() != reference.size()) {
        std::printf(
            "[CYCLE_SAME_INPUT_PARITY] layer=%d point=%s "
            "actual=%zu reference=%zu status=SHAPE_FAIL\n",
            layer, point, actual.size(), reference.size());
        return result;
    }
    double max_ref = 0;
    double error2 = 0;
    double dot = 0, a2 = 0, r2 = 0;
    size_t worst = 0;
    bool finite = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = reference[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            finite = false;
            worst = i;
            break;
        }
        const double e = std::fabs(a - b);
        if (e > result.max_abs) {
            result.max_abs = e;
            worst = i;
        }
        max_ref = std::max(max_ref, std::fabs(b));
        error2 += e * e;
        dot += a * b;
        a2 += a * a;
        r2 += b * b;
    }
    const double threshold = 0.02 + 0.00002 * max_ref;
    result.valid = finite;
    result.strict = finite && result.max_abs <= threshold;
    result.rms = finite ?
        std::sqrt(error2 / double(actual.size())) : 0.0;
    result.cosine = finite && a2 > 0 && r2 > 0 ?
        dot / std::sqrt(a2 * r2) : 0.0;
    std::printf(
        "[CYCLE_SAME_INPUT_PARITY] layer=%d point=%s "
        "source=CPU_ORACLE_L_OUT_PREDECESSOR backend=PHONE_GPU "
        "elements=%zu max_abs=%.9g rms=%.9g cosine=%.9g "
        "max_ref=%.9g strict_tolerance=%.9g "
        "worst_token=%zu worst_channel=%zu "
        "worst_actual=%.9g worst_reference=%.9g status=%s\n",
        layer, point, actual.size(), result.max_abs,
        result.rms, result.cosine, max_ref, threshold,
        worst / 2048, worst % 2048,
        actual[worst], reference[worst],
        result.strict ? "PASS" : "FAIL");
    return result;
}

inline bool gpu_get_f32(
        const ggml_tensor * src, int64_t hidden, int tokens,
        std::vector<float> & out) {
    out.clear();
    if (!src || !src->buffer || hidden <= 0 || tokens <= 0 ||
        src->type != GGML_TYPE_F32 || !ggml_is_contiguous(src) ||
        src->ne[0] != hidden || src->ne[1] != tokens ||
        ggml_nelements(src) != hidden * tokens) {
        return false;
    }
    out.resize(size_t(hidden) * size_t(tokens));
    ggml_backend_tensor_get(src, out.data(), 0,
                            out.size() * sizeof(float));
    for (float v : out) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

struct results {
    bool execution_ok = false;
    int layers_run = 0;
    int stages_compared = 0;
    int strict_failures = 0;
};

inline results run(const llama_model & model, ggml_backend_t phone,
                   const cycle_full_reference::baseline & ref, int tokens) {
    results out {};
    constexpr std::array<int, 2> targets = {5, 46};
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    const int64_t experts = model.hparams.n_expert;
    if (!phone || tokens != 16 || ref.n_tokens != 16 ||
        ref.hidden != hidden || hidden != 2048 ||
        model.layers.size() != 48 ||
        model.arch != LLM_ARCH_QWEN3MOE) {
        std::fprintf(stderr,
            "[CYCLE_SAME_INPUT_SUM] status=FAIL reason=REQUIRES_QWEN3_MOE_16\n");
        return out;
    }
    for (const int layer : targets) {
        const auto & input = ref.after_layer[size_t(layer - 1)];
        const auto & oracle_attn = ref.after_attention[size_t(layer)];
        const auto & oracle_norm = ref.after_ffn_norm[size_t(layer)];
        const auto & oracle_ffn  = ref.after_ffn[size_t(layer)];
        const auto & oracle_out  = ref.after_layer[size_t(layer)];
        const size_t expected = size_t(tokens) * size_t(hidden);
        if (input.size() != expected || oracle_attn.size() != expected ||
            oracle_norm.size() != expected || oracle_ffn.size() != expected ||
            oracle_out.size() != expected) {
            std::fprintf(stderr,
                "[CYCLE_SAME_INPUT] layer=%d status=FAIL "
                "reason=CPU_REFERENCE_STAGE_MISSING\n", layer);
            return out;
        }

        // Deliberate host->GPU input upload is for diagnostic A/B ONLY.
        // F16 causal K/V and mask are constructed from this same 16-token
        // input inside the real attention kernel, without touching the
        // physical K/V cache already committed by the global FIFO.
        llama_hybrid_cycle_attention_ready ready;
        double attn_ms = 0.0;
        const auto & w = model.layers[size_t(layer)];
        bool ok = llama_hybrid_cycle_execute_attention_pre(
            phone, w, model.hparams, input, hidden, tokens, topk, experts,
            ready, attn_ms, nullptr, nullptr, 0, 1,
            nullptr, true);
        if (!ok) {
            std::fprintf(stderr,
                "[CYCLE_SAME_INPUT] layer=%d status=FAIL "
                "reason=GPU_ATTENTION_EXEC\n", layer);
            ggml_backend_synchronize(phone);
            return out;
        }

        const llama_hybrid_cycle_ffn_shard_weights shard {
            llama_hybrid_cycle_resolve_physical_weight(
                w.ffn_up_exps, phone, 1),
            llama_hybrid_cycle_resolve_physical_weight(
                w.ffn_gate_exps, phone, 1),
            llama_hybrid_cycle_resolve_physical_weight(
                w.ffn_down_exps, phone, 1),
        };
        llama_hybrid_cycle_phone_ffn_slot slot;
        ok = shard.available() &&
            slot.prepare(phone, shard, hidden, tokens, topk, experts,
                         true, false) &&
            slot.execute_full_local(ready) &&
            slot.execute_reduce(phone);
        if (!ok) {
            ggml_backend_synchronize(phone);
            std::fprintf(stderr,
                "[CYCLE_SAME_INPUT] layer=%d status=FAIL "
                "reason=GPU_FFN_OR_REDUCE_EXEC\n", layer);
            return out;
        }

        const std::array<const ggml_tensor *, 4> gpu_points {
            ready.phone_graph.residual,
            ready.phone_graph.ffn_norm,
            slot.inputs.ffn_output,
            slot.output,
        };
        const std::array<const std::vector<float> *, 4> refs {
            &oracle_attn, &oracle_norm, &oracle_ffn, &oracle_out,
        };
        constexpr std::array<const char *, 4> names {
            "ATTN_RESIDUAL", "FFN_NORM", "FFN_OUTPUT", "L_OUT",
        };
        int passed = 0;
        for (size_t index = 0; index < gpu_points.size(); ++index) {
            std::vector<float> actual;
            if (!gpu_get_f32(gpu_points[index], hidden, tokens, actual)) {
                std::fprintf(stderr,
                    "[CYCLE_SAME_INPUT] layer=%d point=%s "
                    "status=FAIL reason=GPU_OUTPUT_READBACK\n",
                    layer, names[index]);
                return out;
            }
            const comparison diff =
                compare(layer, names[index], actual, *refs[index]);
            if (!diff.valid) return out;
            ++out.stages_compared;
            if (diff.strict) ++passed;
            else ++out.strict_failures;
        }
        ++out.layers_run;
        std::printf(
            "[CYCLE_SAME_INPUT] layer=%d source=CPU_ORACLE_L_OUT_%d "
            "input_upload=DIAGNOSTIC_ONLY "
            "gpu_attention_ms=%.3f strict_pass=%d stages=4 "
            "status=COMPLETE\n",
            layer, layer - 1, attn_ms, passed);
    }
    out.execution_ok = out.layers_run == 2 && out.stages_compared == 8;
    std::printf(
        "[CYCLE_SAME_INPUT_SUM] targets=5,46 "
        "executed=%d expected=2 compared=%d expected_stages=8 "
        "strict_failed=%d all_strict=%s "
        "gpu_upload=DIAGNOSTIC_ONLY status=%s\n",
        out.layers_run, out.stages_compared, out.strict_failures,
        out.strict_failures == 0 ? "PASS" : "FAIL",
        out.execution_ok ? "COMPLETE" : "FAIL");
    return out;
}

} // namespace cycle_same_input
