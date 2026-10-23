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
        llama_hybrid_cycle_pc_full_timing * timing) {
    output.clear();
    if (!pc_cpu || !resource.attn_backend || !resource.kv ||
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
                           uint64_t(tokens) ||
        input.size() != size_t(hidden) * size_t(tokens)) {
        return false;
    }

    const bool cuda_mirror = resource.cuda_weights != nullptr;
    if (!cuda_mirror && resource.attn_backend != pc_cpu) {
        std::fprintf(stderr,
            "[CYCLE_PC_FULL] layer=%d reason=CUDA_MIRROR_REQUIRED "
            "status=FAIL\n", physical_layer);
        return false;
    }

    const auto & lw = model.layers[size_t(physical_layer)];
    const auto cpu_weight = [&](const ggml_tensor * tensor) {
        return llama_hybrid_cycle_resolve_physical_weight(
            tensor, pc_cpu, 0);
    };
    const auto shard = cuda_mirror ?
        resource.cuda_weights->full_ffn_shard() :
        llama_hybrid_cycle_ffn_shard_weights {
            cpu_weight(lw.ffn_up_exps),
            cpu_weight(lw.ffn_gate_exps),
            cpu_weight(lw.ffn_down_exps),
        };
    if (!shard.available()) {
        std::fprintf(stderr,
            "[CYCLE_PC_FULL] layer=%d reason=FFN_WEIGHTS "
            "status=FAIL\n", physical_layer);
        return false;
    }

    llama_hybrid_cycle_attention_ready ready;
    double attn_ms = 0.0;
    const auto attn_begin = std::chrono::steady_clock::now();
    // CPU follows the original V5 full-layer code path (host READY);
    // mirrored CUDA retains the device READY until its local FFN consumes it.
    const bool attention_ok = llama_hybrid_cycle_execute_attention_pre(
        resource.attn_backend, lw, model.hparams, input,
        hidden, tokens, topk, experts, ready, attn_ms,
        nullptr, resource.kv.get(), token_start, 0,
        nullptr, cuda_mirror, nullptr,
        resource.cuda_weights.get(),
        cuda_mirror ? resource.scratch_pool.get() : nullptr,
        cuda_mirror ? resource.graph_cache.get() : nullptr);
    if (timing) {
        timing->attention_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - attn_begin).count();
    }
    if (!attention_ok) {
        return false;
    }

    const auto ffn_begin = std::chrono::steady_clock::now();
    bool ok = false;
    if (cuda_mirror) {
        llama_hybrid_cycle_phone_ffn_slot local_ffn;
        const bool prepared = local_ffn.prepare(
            resource.attn_backend, shard, hidden, tokens, topk, experts,
            true, true, false);
        if (prepared) {
            // Even a failed submission may have outstanding GPU operations.
            ok = local_ffn.execute_full_local(ready);
            ggml_backend_synchronize(resource.attn_backend);
        }
        if (ok && local_ffn.output &&
            local_ffn.output->type == GGML_TYPE_F32 &&
            ggml_nelements(local_ffn.output) == hidden * tokens) {
            const auto readback_begin = std::chrono::steady_clock::now();
            output.resize(size_t(hidden) * size_t(tokens));
            ggml_backend_tensor_get(
                local_ffn.output, output.data(), 0,
                output.size() * sizeof(float));
            if (timing) {
                timing->output_readback_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - readback_begin)
                        .count();
            }
        } else {
            ok = false;
        }
        // The local FFN slot is destroyed before the borrowed READY graph.
        ready.release_phone_graph();
    } else {
        std::vector<float> partial;
        ok = llama_hybrid_cycle_execute_ffn_shard(
            pc_cpu, shard, ready.norm, ready.ids, ready.routing,
            hidden, tokens, topk, experts, partial);
        if (ok && partial.size() == ready.residual.size() &&
            partial.size() == size_t(hidden) * size_t(tokens)) {
            output = std::move(partial);
            for (size_t i = 0; i < output.size(); ++i) {
                output[i] += ready.residual[i];
            }
        } else {
            ok = false;
        }
    }
    if (timing) {
        timing->ffn_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ffn_begin).count();
    }
    if (ok) {
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
