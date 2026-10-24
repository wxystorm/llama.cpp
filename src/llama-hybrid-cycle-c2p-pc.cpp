#include "llama-hybrid-cycle-c2p-pc.h"
#include "llama-hybrid-cycle-log.h"

#include "llama-model.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

bool valid_c2p_pc(
        const llama_model & model,
        const llama_hybrid_cycle_physical_layer_resources & resource,
        ggml_backend_t cpu, int layer, int64_t tokens) {
    return cpu && layer >= 0 &&
           size_t(layer) < model.layers.size() &&
           resource.physical_layer == layer &&
           resource.kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P &&
           resource.attn_backend == cpu && resource.kv &&
           tokens > 0 && tokens <= 128 &&
           model.hparams.n_embd > 0 &&
           model.hparams.n_expert_used > 0 &&
           model.hparams.n_expert_used <= model.hparams.n_expert &&
           uint64_t(tokens) <=
               std::numeric_limits<size_t>::max() /
               uint64_t(model.hparams.n_embd);
}

bool all_finite(const std::vector<float> & values) {
    for (float v : values) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

} // namespace

bool llama_hybrid_cycle_execute_c2p_pc_attention(
        const llama_model & model,
        const llama_hybrid_cycle_physical_layer_resources & resource,
        ggml_backend_t pc_cpu,
        int physical_layer, int64_t token_start, int64_t tokens,
        const std::vector<float> & input,
        llama_hybrid_cycle_attention_ready & ready) {
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    if (!valid_c2p_pc(model, resource, pc_cpu, physical_layer, tokens) ||
        token_start < 0 ||
        tokens > resource.kv->max_tokens ||
        token_start > resource.kv->max_tokens - tokens ||
        resource.kv->committed_tokens != token_start ||
        input.size() != size_t(tokens) * size_t(hidden)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_C2P_PC] layer=%d stage=ATTN_PRE "
            "status=FAIL reason=PHYSICAL_OR_KV_PREFIX\n", physical_layer);
        return false;
    }
    double attn_ms = 0.0;
    const bool calculated = llama_hybrid_cycle_execute_attention_pre(
        pc_cpu, model.layers[size_t(physical_layer)], model.hparams,
        input, hidden, tokens, topk, model.hparams.n_expert,
        ready, attn_ms, nullptr, resource.kv.get(), token_start, 0);
    const bool ok = calculated &&
        resource.kv->committed_tokens == token_start + tokens &&
        ready.residual.size() == size_t(tokens) * size_t(hidden) &&
        ready.norm.size() == size_t(tokens) * size_t(hidden) &&
        ready.ids.size() == size_t(tokens) * size_t(topk) &&
        ready.routing.size() == size_t(tokens) * size_t(topk) &&
        all_finite(ready.residual) && all_finite(ready.norm) &&
        all_finite(ready.routing);
    if (!ok || llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_REAL_C2P_PC] layer=%d stage=ATTN_PRE "
        "position_start=%lld tokens=%lld kv_committed=%lld "
        "attn_ms=%.3f status=%s\n",
        physical_layer, (long long)token_start, (long long)tokens,
        (long long)resource.kv->committed_tokens,
        attn_ms, ok ? "PASS" : "FAIL");
    return ok;
}

bool llama_hybrid_cycle_execute_c2p_pc_ffn_source(
        const llama_model & model,
        const llama_hybrid_cycle_physical_layer_resources & resource,
        ggml_backend_t pc_cpu,
        int physical_layer, int64_t tokens,
        const llama_hybrid_cycle_attention_ready & ready,
        std::vector<float> & partial) {
    partial.clear();
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    if (!valid_c2p_pc(model, resource, pc_cpu, physical_layer, tokens) ||
        ready.norm.size() != size_t(tokens) * size_t(hidden) ||
        ready.ids.size() != size_t(tokens) * size_t(topk) ||
        ready.routing.size() != size_t(tokens) * size_t(topk)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_C2P_PC] layer=%d stage=FFN_SOURCE "
            "status=FAIL reason=INVALID_REAL_READY\n", physical_layer);
        return false;
    }
    const auto & lw = model.layers[size_t(physical_layer)];
    const llama_hybrid_cycle_ffn_shard_weights shard {
        llama_hybrid_cycle_resolve_physical_weight(lw.ffn_up_exps, pc_cpu, 0),
        llama_hybrid_cycle_resolve_physical_weight(lw.ffn_gate_exps, pc_cpu, 0),
        llama_hybrid_cycle_resolve_physical_weight(lw.ffn_down_exps, pc_cpu, 0),
    };
    if (!shard.available()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_C2P_PC] layer=%d stage=FFN_SOURCE "
            "status=FAIL reason=MISSING_PC_SHARD\n", physical_layer);
        return false;
    }
    const bool calculated = llama_hybrid_cycle_execute_ffn_shard(
        pc_cpu, shard, ready.norm, ready.ids, ready.routing,
        hidden, tokens, topk, model.hparams.n_expert, partial);
    const bool ok = calculated &&
        partial.size() == size_t(tokens) * size_t(hidden) &&
        all_finite(partial);
    if (!ok || llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_REAL_C2P_PC] layer=%d stage=FFN_SOURCE "
        "tokens=%lld elements=%zu status=%s\n",
        physical_layer, (long long)tokens, partial.size(),
        ok ? "PASS" : "FAIL");
    if (!ok) partial.clear();
    return ok;
}
