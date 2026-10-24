#include "llama-hybrid-cycle-phone-full.h"
#include "llama-hybrid-cycle-log.h"

#include "llama-model.h"

#include <chrono>
#include <cstdio>
#include <vector>

bool llama_hybrid_cycle_execute_phone_full(
        const llama_model & model,
        const llama_hybrid_cycle_physical_layer_resources & resource,
        ggml_backend_t phone,
        int physical_layer, int chunk, uint64_t seq,
        int64_t position_start, int64_t tokens,
        const ggml_tensor * device_input,
        llama_hybrid_cycle_attention_ready & ready,
        llama_hybrid_cycle_phone_ffn_slot & output_slot,
        const std::function<void(const ggml_tensor *, const ggml_tensor *,
                                 const ggml_tensor *)> & stage_probe) {
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    const int64_t experts = model.hparams.n_expert;
    const bool inputs_valid =
        phone && physical_layer >= 0 && chunk >= 0 && seq != 0 &&
        size_t(physical_layer) < model.layers.size() &&
        resource.physical_layer == physical_layer &&
        resource.kind == llama_hybrid_cycle_layer_kind::PHONE_GPU &&
        resource.attn_backend == phone &&
        resource.kv && resource.graph_cache && resource.scratch_pool &&
        position_start >= 0 && tokens > 0 &&
        position_start + tokens <= resource.kv->max_tokens &&
        resource.kv->committed_tokens == position_start &&
        device_input && device_input->buffer &&
        device_input->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(device_input) &&
        ggml_nelements(device_input) == hidden * tokens &&
        hidden > 0 && topk > 0 && experts >= topk &&
        (!output_slot.prepared_tokens ||
         (output_slot.is_full_layer &&
          output_slot.prepared_tokens == tokens &&
          output_slot.output && output_slot.fused_full_graph &&
          output_slot.registered_full_graph_uid != 0));
    if (!inputs_valid) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PHONE_FULL] layer=%d start=%lld tokens=%lld "
            "status=FAIL reason=PHYSICAL_INPUT_OR_KV_PREFIX\n",
            physical_layer, (long long)position_start, (long long)tokens);
        return false;
    }
    const auto & weights = model.layers[size_t(physical_layer)];
    if (!output_slot.prepared_tokens) {
        const llama_hybrid_cycle_ffn_shard_weights shard {
            llama_hybrid_cycle_resolve_physical_weight(
                weights.ffn_up_exps, phone, 1),
            llama_hybrid_cycle_resolve_physical_weight(
                weights.ffn_gate_exps, phone, 1),
            llama_hybrid_cycle_resolve_physical_weight(
                weights.ffn_down_exps, phone, 1),
        };
        if (!shard.available() ||
            !output_slot.prepare(phone, shard, hidden, tokens,
                                 topk, experts,
                                 true,  // full_layer
                                 true,  // fuse FFN + residual ADD
                                 true,  // registered RPC replay
                                 true)) { // Phone Full local replay
            std::fprintf(stderr,
                "[CYCLE_REAL_PHONE_FULL] layer=%d "
                "status=FAIL reason=FULL_FFN_SLOT_PREPARE\n",
                physical_layer);
            return false;
        }
    }

    const auto start = std::chrono::steady_clock::now();
    const std::vector<float> no_host_input;
    double attn_ms = 0.0;
    // The persistent KV, Attention Graph and Scratch remain owned by the
    // absolute physical layer. The previous GPU output is copied directly
    // inside the same RPC backend and stays alive until the caller retires it.
    bool ok = llama_hybrid_cycle_execute_attention_pre(
        phone, weights, model.hparams, no_host_input,
        hidden, tokens, topk, experts, ready, attn_ms,
        nullptr, resource.kv.get(), position_start, 1,
        device_input, true, nullptr, nullptr,
        resource.scratch_pool.get(), resource.graph_cache.get());
    const bool attention_done = ok &&
        resource.kv->committed_tokens == position_start + tokens &&
        ready.phone_graph.residual && ready.phone_graph.ffn_norm &&
        ready.phone_graph.topk_ids && ready.phone_graph.topk_weights;
    if (attention_done) {
        // Protocol 28: READY remains entirely on Phone. The server copies the
        // four READY tensors into the persistent registered Full slot and
        // replays one fused FFN + residual graph. The ACK is the l_out device
        // ownership boundary: no separate client FFN submit, Reduce submit,
        // or fence is required on the success path.
        ok = output_slot.execute_full_server(
            phone, ready, physical_layer, chunk, seq);
    } else {
        ok = false;
    }
    if (!ok) {
        // Failed remote work may still reference READY scratch.
        ggml_backend_synchronize(phone);
    }
    if (ok && stage_probe) {
        // Explicit numerical trace only. Protocol-28 RUN returns only after
        // fused FFN+ADD is device-complete, so READY/scratch and the registered
        // slot are still valid here without another client fence.
        stage_probe(ready.phone_graph.residual,
                    ready.phone_graph.ffn_norm,
                    output_slot.inputs.ffn_output);
    }
    // Fused Full is complete (or the failure path fenced). No consumer still
    // needs this layer's Attention scratch/Graph lease.
    ready.release_phone_graph();
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_REAL_PHONE_FULL] layer=%d "
        "position_start=%lld tokens=%lld kv_committed=%lld "
        "attn_ms=%.3f wall_ms=%.3f "
        "input=PHONE_DEVICE_L_OUT ffn=REGISTERED_REPLAY reduce=FUSED_ADD "
        "status=%s\n",
        physical_layer, (long long)position_start, (long long)tokens,
        (long long)resource.kv->committed_tokens, attn_ms, wall_ms,
        ok ? "PASS" : "FAIL");
    return ok;
}
