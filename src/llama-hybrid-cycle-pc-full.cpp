#include "llama-hybrid-cycle-pc-full.h"

#include "llama-model.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

bool llama_hybrid_cycle_execute_pc_full_layer(
        const llama_model & model,
        const llama_hybrid_cycle_physical_layer_resources & resource,
        ggml_backend_t pc_cpu,
        int physical_layer, int64_t token_start, int64_t tokens,
        const std::vector<float> & input,
        std::vector<float> & output,
        llama_hybrid_cycle_pc_full_timing * timing,
        const std::function<bool()> & input_consumed,
        const ggml_tensor * device_input,
        llama_hybrid_cycle_cuda_activation_slot * device_output) {
    output.clear();
    (void) pc_cpu; // Tensor layers still use CPU; ordinary Full never does.
    if (!resource.attn_backend || !resource.kv || !resource.cuda_weights ||
        physical_layer < 0 ||
        size_t(physical_layer) >= model.layers.size() ||
        resource.physical_layer != physical_layer ||
        resource.kind != llama_hybrid_cycle_layer_kind::PC_CPU ||
        tokens < 1 || tokens > 128 || token_start < 0 ||
        tokens > resource.kv->max_tokens ||
        token_start > resource.kv->max_tokens - tokens ||
        resource.kv->committed_tokens != token_start) {
        std::fprintf(stderr,
            "[CYCLE_PC_FULL] layer=%d reason=INVALID_PHYSICAL_KV "
            "status=FAIL\n", physical_layer);
        return false;
    }
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    const int64_t experts = model.hparams.n_expert;
    if (hidden <= 0 || topk <= 0 || topk > experts ||
        uint64_t(hidden) > std::numeric_limits<size_t>::max() /
                           uint64_t(tokens)) {
        return false;
    }
    const size_t activation_elements = size_t(hidden) * size_t(tokens);
    const bool input_ok = device_input ?
        (device_input->type == GGML_TYPE_F32 &&
         device_input->buffer != nullptr &&
         ggml_is_contiguous(device_input) &&
         device_input->ne[0] == hidden &&
         device_input->ne[1] == tokens &&
         ggml_nelements(device_input) == hidden * tokens) :
        input.size() == activation_elements;
    if (!input_ok) {
        return false;
    }

    // GPU-only invariant: residency must have been faulted in by the PC_CUDA
    // worker before entering this kernel. There is no CPU Full fallback.
    const auto & lw = model.layers[size_t(physical_layer)];
    const auto shard = resource.cuda_weights->full_ffn_shard();
    if (!shard.available()) {
        std::fprintf(stderr,
            "[CYCLE_PC_FULL] layer=%d reason=CUDA_FFN_WEIGHTS_MISSING "
            "status=FAIL\n", physical_layer);
        return false;
    }

    llama_hybrid_cycle_attention_ready ready;
    double attn_ms = 0.0;
    const auto attn_begin = std::chrono::steady_clock::now();
    // CUDA retains READY on-device until the local FFN consumes it.
    const bool attention_ok = llama_hybrid_cycle_execute_attention_pre(
        resource.attn_backend, lw, model.hparams, input,
        hidden, tokens, topk, experts, ready, attn_ms,
        nullptr, resource.kv.get(), token_start, 0,
        device_input, true, nullptr,
        resource.cuda_weights.get(),
        resource.scratch_pool.get(),
        resource.graph_cache.get());
    if (timing) {
        timing->attention_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - attn_begin).count();
    }
    if (!attention_ok) {
        return false;
    }
    // execute_attention_pre() fences the backend before returning on this
    // non-deferred CUDA path. The predecessor input can therefore be reused
    // while this layer continues through FFN + residual + readback.
    if (input_consumed && !input_consumed()) {
        std::fprintf(stderr,
            "[CYCLE_PC_FULL_HANDOFF] layer=%d reason=INPUT_CONSUMED_NOTIFY "
            "status=FAIL\n", physical_layer);
        ready.release_phone_graph();
        return false;
    }

    const auto ffn_begin = std::chrono::steady_clock::now();
    bool ok = false;
    llama_hybrid_cycle_phone_ffn_slot local_ffn;
    const bool prepared = local_ffn.prepare(
        resource.attn_backend, shard, hidden, tokens, topk, experts,
        true, true, false);
    if (prepared) {
        // The Full task owns the one PC_CUDA worker. Synchronize before the
        // residency cache may evict/reuse this layer's mirror.
        ok = local_ffn.execute_full_local(ready);
        ggml_backend_synchronize(resource.attn_backend);
    }
    if (ok && local_ffn.output &&
        local_ffn.output->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(local_ffn.output) &&
        ggml_nelements(local_ffn.output) == hidden * tokens) {
        if (device_output) {
            const auto copy_begin = std::chrono::steady_clock::now();
            ok = device_output->prepare(
                     resource.attn_backend, hidden, tokens) &&
                 device_output->store(local_ffn.output);
            if (timing) {
                timing->output_device_copy_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - copy_begin)
                        .count();
            }
        } else {
            const auto readback_begin = std::chrono::steady_clock::now();
            output.resize(activation_elements);
            ggml_backend_tensor_get(
                local_ffn.output, output.data(), 0,
                output.size() * sizeof(float));
            if (timing) {
                timing->output_readback_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - readback_begin)
                        .count();
            }
        }
    } else {
        ok = false;
    }
    // Destroy/release graph references before another Full task can evict
    // this mirror from the bounded residency cache.
    ready.release_phone_graph();
    if (timing) {
        timing->ffn_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ffn_begin).count();
    }
    if (ok && !device_output) {
        for (float value : output) {
            if (!std::isfinite(value)) {
                ok = false;
                break;
            }
        }
    }
    if (!ok) {
        output.clear();
        std::fprintf(stderr,
            "[CYCLE_PC_FULL] layer=%d tokens=%lld "
            "backend=%s status=FAIL\n",
            physical_layer, (long long)tokens,
            ggml_backend_name(resource.attn_backend));
    }
    return ok;
}
