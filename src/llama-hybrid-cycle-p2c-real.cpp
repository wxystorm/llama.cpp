#include "llama-hybrid-cycle-p2c-real.h"
#include "llama-hybrid-cycle-log.h"

#include "llama-model.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

bool valid_p2c(const llama_model & model,
               const llama_hybrid_cycle_physical_layer_resources & r,
               int layer, int64_t tokens) {
    return layer >= 0 && size_t(layer) < model.layers.size() &&
           r.physical_layer == layer &&
           r.kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C &&
           tokens > 0 && tokens <= 128 && model.hparams.n_embd > 0 &&
           model.hparams.n_expert_used > 0 &&
           model.hparams.n_expert_used <= model.hparams.n_expert &&
           uint64_t(tokens) <= std::numeric_limits<size_t>::max() /
               uint64_t(model.hparams.n_embd);
}
bool all_finite(const std::vector<float> & values) {
    for (float x : values) if (!std::isfinite(x)) return false;
    return true;
}

} // namespace

bool llama_hybrid_cycle_execute_p2c_phone_attention(
        const llama_model & model,
        const llama_hybrid_cycle_physical_layer_resources & resource,
        ggml_backend_t phone, int physical_layer,
        int64_t token_start, int64_t tokens,
        const ggml_tensor * previous_output,
        const llama_hybrid_cycle_live_wire & wire,
        llama_hybrid_cycle_attention_ready & ready,
        const llama_hybrid_cycle_attention_rpc_ops & rpc) {
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    const bool physical_ok = phone &&
        valid_p2c(model, resource, physical_layer, tokens) &&
        resource.attn_backend == phone && resource.kv &&
        resource.graph_cache && resource.scratch_pool &&
        token_start >= 0 && tokens <= resource.kv->max_tokens &&
        token_start <= resource.kv->max_tokens - tokens &&
        resource.kv->committed_tokens == token_start &&
        wire.layer == physical_layer && wire.chunk >= 0 &&
        wire.seq != 0 && wire.graph_id != 0 &&
        rpc.is_rpc && rpc.send_ready &&
        previous_output && previous_output->buffer &&
        previous_output->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(previous_output) &&
        ggml_nelements(previous_output) == tokens * hidden;
    if (!physical_ok) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C_ATTN] layer=%d status=FAIL "
            "reason=DEVICE_SOURCE_OR_KV_PREFIX\n", physical_layer);
        return false;
    }
    const std::vector<float> empty;
    double attn_ms = 0.0;
    const bool computed = llama_hybrid_cycle_execute_attention_pre(
        phone, model.layers[size_t(physical_layer)], model.hparams, empty,
        hidden, tokens, topk, model.hparams.n_expert,
        ready, attn_ms, &wire, resource.kv.get(), token_start, 1,
        previous_output, true, nullptr, nullptr,
        resource.scratch_pool.get(), resource.graph_cache.get(),
        {}, false, rpc);
    const bool ok = computed &&
        resource.kv->committed_tokens == token_start + tokens &&
        ready.phone_graph.residual && ready.phone_graph.ffn_norm &&
        ready.phone_graph.topk_ids && ready.phone_graph.topk_weights &&
        ready.phone_compute_buffer;
    if (computed && !ok) ready.release_phone_graph();
    if (!ok || llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_REAL_P2C_ATTN] layer=%d position_start=%lld "
        "tokens=%lld kv_committed=%lld attn_ms=%.3f "
        "input=PHONE_GPU_PREVIOUS_L_OUT ready=PHONE_TO_PC_STAGED "
        "status=%s\n",
        physical_layer, (long long) token_start, (long long) tokens,
        (long long)(resource.kv ? resource.kv->committed_tokens : -1),
        attn_ms, ok ? "PASS" : "FAIL");
    return ok;
}

bool llama_hybrid_cycle_execute_p2c_pc_ffn(
        const llama_model & model,
        const llama_hybrid_cycle_physical_layer_resources & resource,
        ggml_backend_t cpu, int physical_layer, int64_t tokens,
        const std::vector<float> & norm,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        std::vector<float> & partial) {
    partial.clear();
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    if (!cpu || !valid_p2c(model, resource, physical_layer, tokens) ||
        resource.attn_backend == cpu ||
        norm.size() != size_t(tokens) * size_t(hidden) ||
        ids.size() != size_t(tokens) * size_t(topk) ||
        routing.size() != size_t(tokens) * size_t(topk) ||
        !all_finite(norm) || !all_finite(routing)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C_PC_FFN] layer=%d status=FAIL "
            "reason=PHYSICAL_SHARD_OR_READY\n", physical_layer);
        return false;
    }
    const auto & w = model.layers[size_t(physical_layer)];
    const llama_hybrid_cycle_ffn_shard_weights shard {
        llama_hybrid_cycle_resolve_physical_weight(w.ffn_up_exps, cpu, 0),
        llama_hybrid_cycle_resolve_physical_weight(w.ffn_gate_exps, cpu, 0),
        llama_hybrid_cycle_resolve_physical_weight(w.ffn_down_exps, cpu, 0),
    };
    const bool computed = shard.available() &&
        llama_hybrid_cycle_execute_ffn_shard(
            cpu, shard, norm, ids, routing,
            hidden, tokens, topk, model.hparams.n_expert, partial);
    const bool ok = computed &&
        partial.size() == size_t(tokens) * size_t(hidden) &&
        all_finite(partial);
    if (!ok || llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_REAL_P2C_PC_FFN] layer=%d tokens=%lld elements=%zu "
        "status=%s\n", physical_layer, (long long) tokens, partial.size(),
        ok ? "PASS" : "FAIL");
    if (!ok) partial.clear();
    return ok;
}
