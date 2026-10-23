#include "llama-hybrid-cycle-physical.h"

#include "llama-model.h"

#include "ggml-alloc.h"
#include "../ggml/src/ggml-backend-impl.h" // ggml_backend_buffer_is_meta

#include <cstdio>
#include <cstdlib>

ggml_tensor * llama_hybrid_cycle_resolve_physical_weight(
        const ggml_tensor * weight, ggml_backend_t backend,
        size_t source_device) {
    if (!weight || !weight->buffer || !backend ||
        weight->op != GGML_OP_NONE || weight->view_src != nullptr) {
        return nullptr;
    }
    const auto weight_buft = ggml_backend_buffer_get_type(weight->buffer);
    const auto native_buft = ggml_backend_get_default_buffer_type(backend);
    const auto device = ggml_backend_get_device(backend);
    const bool pc_native_compatible =
        source_device == 0 && device &&
        ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU &&
        !ggml_backend_buffer_is_meta(weight->buffer) &&
        ggml_backend_dev_supports_buft(device, weight_buft);
    if ((native_buft && weight_buft == native_buft) ||
        pc_native_compatible) {
        if (pc_native_compatible && weight_buft != native_buft &&
            std::getenv("GGML_CYCLE_SHARD_DIAG") != nullptr) {
            std::fprintf(stderr,
                "[CYCLE_WEIGHT_RESOLVE] weight=%s index=%zu "
                "backend=%s weight_buft=%s default_buft=%s "
                "mode=CPU_COMPATIBLE status=OK\n",
                weight->name, source_device, ggml_backend_name(backend),
                ggml_backend_buft_name(weight_buft),
                native_buft ? ggml_backend_buft_name(native_buft) : "(null)");
        }
        // Borrowed model tensor: it remains owned by the model until unload.
        return const_cast<ggml_tensor *>(weight);
    }
    return ggml_backend_meta_get_static_weight_shard(weight, source_device);
}

void llama_hybrid_cycle_cuda_full_weights::release() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        tensors.fill(nullptr);
    }

bool llama_hybrid_cycle_cuda_full_weights::prepare(ggml_backend_t cuda, ggml_backend_t cpu,
                 const llama_layer & layer, int layer_index) {
        release();
        if (!cuda || !cpu) {
            return false;
        }
        const ggml_tensor * logical[COUNT] = {
            layer.attn_norm, layer.wq, layer.wk, layer.wv, layer.wo,
            layer.attn_q_norm, layer.attn_k_norm, layer.ffn_norm,
            layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps,
            layer.ffn_down_exps
        };
        std::array<const ggml_tensor *, COUNT> physical {};
        for (size_t i = 0; i < COUNT; ++i) {
            physical[i] = llama_hybrid_cycle_resolve_physical_weight(
                logical[i], cpu, 0);
            if (!physical[i] || !physical[i]->buffer ||
                !ggml_is_contiguous(physical[i])) {
                std::fprintf(stderr,
                    "[CYCLE_PC_CUDA_WEIGHT] layer=%d slot=%zu "
                    "status=FAIL reason=PC_PHYSICAL_WEIGHT_UNAVAILABLE\n",
                    layer_index, i);
                return false;
            }
        }
        ggml_init_params params {};
        params.mem_size = 2 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }
        for (size_t i = 0; i < COUNT; ++i) {
            tensors[i] = ggml_dup_tensor(ctx, physical[i]);
            if (!tensors[i]) {
                release();
                return false;
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(cuda));
        if (!buffer) {
            release();
            return false;
        }
        size_t upload_bytes = 0;
        for (size_t i = 0; i < COUNT; ++i) {
            // One-time CPU -> CUDA copy; never re-upload model weights per
            // Chunk. Only final full-layer output is read back to the host.
            ggml_backend_tensor_copy(physical[i], tensors[i]);
            upload_bytes += ggml_nbytes(physical[i]);
        }
        ggml_backend_synchronize(cuda);
        std::printf(
            "[CYCLE_PC_CUDA_WEIGHT] layer=%d weights=%zu upload_mib=%.3f "
            "owner=PC_CUDA lifetime=PERIOD status=READY\n",
            layer_index, size_t(COUNT),
            double(upload_bytes) / (1024.0 * 1024.0));
        return true;
    }

llama_hybrid_cycle_ffn_shard_weights llama_hybrid_cycle_cuda_full_weights::full_ffn_shard() const {
        return {tensors[FFN_UP], tensors[FFN_GATE], tensors[FFN_DOWN]};
    }

void llama_hybrid_cycle_physical_kv_cache::release() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        keys = nullptr;
        values = nullptr;
        max_tokens = 0;
        head_dim = 0;
        kv_heads = 0;
        committed_tokens = 0;
    }

bool llama_hybrid_cycle_physical_kv_cache::init(ggml_backend_t backend,
              const llama_layer & layer, int64_t total_tokens,
              size_t source_device,
              const llama_hybrid_cycle_cuda_full_weights * cuda_weights) {
        release();
        const auto * norm = cuda_weights ?
            cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::K_NORM] :
            llama_hybrid_cycle_resolve_physical_weight(
                layer.attn_k_norm, backend, source_device);
        const auto * wk = cuda_weights ?
            cuda_weights->tensors[llama_hybrid_cycle_cuda_full_weights::WK] :
            llama_hybrid_cycle_resolve_physical_weight(
                layer.wk, backend, source_device);
        if (!backend || !norm || !wk ||
            total_tokens <= 0 || total_tokens > 128 ||
            norm->ne[0] <= 0 || wk->ne[1] % norm->ne[0] != 0) {
            return false;
        }
        head_dim = norm->ne[0];
        kv_heads = wk->ne[1] / head_dim;
        max_tokens = total_tokens;
        ggml_init_params params {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }
        keys = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F16, head_dim, kv_heads, max_tokens);
        values = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F16, head_dim, kv_heads, max_tokens);
        if (!keys || !values) {
            release();
            return false;
        }
        ggml_set_name(keys, "cycle_phone_k_cache_f16");
        ggml_set_name(values, "cycle_phone_v_cache_f16");
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(backend));
        if (!buffer) {
            release();
            return false;
        }
        std::printf(
            "[CYCLE_PHONE_KV_PERSISTENT] total_tokens=%lld head_dim=%lld "
            "kv_heads=%lld cache_bytes=%zu residency=%s "
            "status=READY\n",
            (long long)max_tokens, (long long)head_dim,
            (long long)kv_heads, ggml_nbytes(keys) + ggml_nbytes(values),
            cuda_weights ? "PC_CUDA_F16" :
                source_device == 0 ? "PC_CPU_F16" : "PHONE_GPU_F16");
        return true;
    }
