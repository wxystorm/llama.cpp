#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-hybrid.h"
#include "llama-region-ring.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

//
// llama_context
//

static llm_graph_type ctx_type_to_graph_type(llama_context_type ctx_type) {
    switch (ctx_type) {
        case LLAMA_CONTEXT_TYPE_DEFAULT: return LLM_GRAPH_TYPE_DEFAULT;
        case LLAMA_CONTEXT_TYPE_MTP    : return LLM_GRAPH_TYPE_DECODER_MTP;
    }
    throw std::runtime_error("Unsupported ctx type");
}

static const char * llama_hybrid_runtime_stage_name(llama_hybrid_runtime_stage_kind kind) {
    switch (kind) {
        case llama_hybrid_runtime_stage_kind::GPU:    return "GPU";
        case llama_hybrid_runtime_stage_kind::CPU:    return "CPU";
        case llama_hybrid_runtime_stage_kind::TENSOR: return "TENSOR";
        case llama_hybrid_runtime_stage_kind::PHONE:  return "PHONE";
    }
    return "UNKNOWN";
}

static const char * llama_hybrid_boundary_action_name(llama_hybrid_boundary_action action) {
    switch (action) {
        case llama_hybrid_boundary_action::PASS:       return "PASS";
        case llama_hybrid_boundary_action::SPLIT:      return "SPLIT";
        case llama_hybrid_boundary_action::ACCUMULATE: return "ACCUMULATE";
    }
    return "UNKNOWN";
}

static void llama_hybrid_log_formal_prepare_prediction(
        uint32_t tokens,
        double   prepare_ms) {
    llama_hybrid_plan runtime_plan;
    llama_hybrid_wave_calibration calibration;
    llama_hybrid_full_prefill_prediction full_prediction;
    if (!llama_hybrid_runtime_plan_get(runtime_plan) ||
        !llama_hybrid_runtime_wave_calibration_get(calibration) ||
        !llama_hybrid_runtime_predict_full_prefill((int) tokens, full_prediction)) {
        return;
    }

    if (tokens <= 1 ||
        calibration.tokens != (int) tokens ||
        calibration.tensor_layers <= 0 ||
        runtime_plan.tensor_layers != full_prediction.tensor_layers ||
        runtime_plan.tensor_chunk_tokens != full_prediction.tensor_chunk_tokens ||
        calibration.tensor_chunk_tokens != full_prediction.tensor_chunk_tokens ||
        calibration.attn_group_chunks != full_prediction.attn_group_chunks ||
        std::abs(calibration.tensor_pc_ratio - full_prediction.tensor_pc_ratio) >= 1e-4f ||
        calibration.wave_ii_count <= 0 ||
        calibration.wave_ii_median_ms <= 0.0) {
        return;
    }

    calibration.prepare_target_tensor_layers = full_prediction.tensor_layers;
    calibration.prepare_target_ms            = prepare_ms;
    calibration.prepare_target_valid         = true;
    llama_hybrid_runtime_wave_calibration_set(calibration);

    const int target_intervals = std::max(0, full_prediction.tensor_layers - 1);
    const int observed_intervals = std::min(target_intervals, calibration.wave_ii_count);
    const int remaining_intervals = std::max(0, target_intervals - observed_intervals);

    const double observed_ii_sum_ms =
        calibration.wave_ii_mean_ms * observed_intervals;
    const double late_ii_ms =
        calibration.wave_late_ii_median_ms > 0.0 ?
            calibration.wave_late_ii_median_ms :
            calibration.wave_ii_median_ms;

    // Primary drift-aware model: preserve the exact probe intervals we
    // actually observed, then assume the late probe rate is the steady state
    // for the remaining Tensor layers. This uses the slowdown trend without
    // extrapolating a noisy linear slope indefinitely.
    const double wave_span_drift_ms =
        calibration.wave_fill_ms +
        observed_ii_sum_ms +
        remaining_intervals * late_ii_ms +
        calibration.wave_drain_ms;
    const double runtime_drift_ms =
        calibration.wave_outer_runtime_ms + wave_span_drift_ms;
    const double tensor_drift_ms = prepare_ms + runtime_drift_ms;
    const double total_drift_ms = full_prediction.gpu_ms + tensor_drift_ms;

    // Keep the old constant-median extrapolation beside it for comparison.
    const double wave_span_median_ms =
        calibration.wave_fill_ms +
        target_intervals * calibration.wave_ii_median_ms +
        calibration.wave_drain_ms;
    const double runtime_median_ms =
        calibration.wave_outer_runtime_ms + wave_span_median_ms;
    const double total_median_ms =
        full_prediction.gpu_ms + prepare_ms + runtime_median_ms;

    LLAMA_LOG_DEBUG(
        "[WAVE_FORMAL_PREP] tokens=%u target_tensor_layers=%d prepare_ms=%.3f reuse=1\n",
        tokens, full_prediction.tensor_layers, prepare_ms);

    LLAMA_LOG_DEBUG(
        "[PRED_WAVE_II] mode=late_hold probe_layers=%d target_layers=%d "
        "prepare_ms=%.3f observed_intervals=%d observed_ii_sum_ms=%.3f "
        "remaining_intervals=%d late_ii_ms=%.3f ii_drift_ratio=%.4f "
        "fill_ms=%.3f drain_ms=%.3f outer_runtime_ms=%.3f "
        "wave_span_ms=%.3f runtime_ms=%.3f tensor_ms=%.3f gpu_ms=%.3f "
        "total_ms=%.3f constant_median_total_ms=%.3f\n",
        calibration.tensor_layers,
        full_prediction.tensor_layers,
        prepare_ms,
        observed_intervals,
        observed_ii_sum_ms,
        remaining_intervals,
        late_ii_ms,
        calibration.wave_ii_drift_ratio,
        calibration.wave_fill_ms,
        calibration.wave_drain_ms,
        calibration.wave_outer_runtime_ms,
        wave_span_drift_ms,
        runtime_drift_ms,
        tensor_drift_ms,
        full_prediction.gpu_ms,
        total_drift_ms,
        total_median_ms);
}

static std::vector<llama_hybrid_runtime_stage> llama_hybrid_build_runtime_stages(const llama_hybrid_plan & plan) {
    std::vector<llama_hybrid_runtime_stage> stages;

    const int gpu_end    = plan.gpu_pc_layers;
    const int cpu_end    = plan.pc_layers;
    const int tensor_end = cpu_end + plan.tensor_layers;
    const int phone_end  = tensor_end + plan.phone_layers;

    int upstream_macro_tokens = 0;
    if (gpu_end > 0) {
        upstream_macro_tokens = plan.gpu_chunk_tokens;
        stages.push_back({ llama_hybrid_runtime_stage_kind::GPU, 0, gpu_end, upstream_macro_tokens, 0 });
    }
    if (cpu_end > gpu_end) {
        upstream_macro_tokens = plan.cpu_chunk_tokens;
        stages.push_back({ llama_hybrid_runtime_stage_kind::CPU, gpu_end, cpu_end, upstream_macro_tokens, 0 });
    }
    if (tensor_end > cpu_end) {
        stages.push_back({ llama_hybrid_runtime_stage_kind::TENSOR, cpu_end, tensor_end,
                           upstream_macro_tokens, plan.tensor_chunk_tokens });
    }
    if (phone_end > tensor_end) {
        stages.push_back({ llama_hybrid_runtime_stage_kind::PHONE, tensor_end, phone_end,
                           plan.phone_chunk_tokens, 0 });
    }

    return stages;
}

static std::vector<llama_hybrid_runtime_stage> llama_region_ring_runtime_stages(
        const std::vector<llama_region_ring_region> & regions,
        int cpu_chunk_tokens,
        int gpu_chunk_tokens,
        int phone_chunk_tokens) {
    if (cpu_chunk_tokens <= 0 || gpu_chunk_tokens <= 0 ||
            phone_chunk_tokens <= 0) {
        throw std::runtime_error(
            "region ring per-device chunks must be positive");
    }
    std::vector<llama_hybrid_runtime_stage> stages;
    stages.reserve(regions.size());
    for (const auto & region : regions) {
        const auto kind =
            region.device == llama_region_ring_device::CPU ?
                llama_hybrid_runtime_stage_kind::CPU :
            region.device == llama_region_ring_device::GPU ?
                llama_hybrid_runtime_stage_kind::GPU :
                llama_hybrid_runtime_stage_kind::PHONE;
        const int chunk_tokens =
            region.device == llama_region_ring_device::CPU ?
                cpu_chunk_tokens :
            region.device == llama_region_ring_device::GPU ?
                gpu_chunk_tokens : phone_chunk_tokens;
        stages.push_back({
            kind, region.layer_begin, region.layer_end, chunk_tokens, 0});
    }
    return stages;
}

static llama_ubatch llama_hybrid_slice_ubatch(
        const llama_ubatch & source,
        uint32_t             token_begin,
        uint32_t             n_tokens,
        float *              hidden) {
    GGML_ASSERT(source.n_pos == 1);
    GGML_ASSERT(!source.equal_seqs());
    GGML_ASSERT(token_begin + n_tokens <= source.n_tokens);

    llama_ubatch result = source;
    result.n_tokens     = n_tokens;
    result.n_seq_tokens = 1;
    result.n_seqs       = n_tokens;
    result.token        = hidden == nullptr && source.token != nullptr ? source.token + token_begin : nullptr;
    result.embd         = hidden;
    result.pos           = source.pos + token_begin;
    result.n_seq_id      = source.n_seq_id + token_begin;
    result.seq_id        = source.seq_id + token_begin;
    result.output        = source.output + token_begin;
    return result;
}

struct llm_fused_op_probe {
    llm_fused_op op;
    const char * name;
    uint32_t n_tokens_per_seq;
};

static const llm_fused_op_probe llm_fused_op_flash_attn_probe = {
    /*.op               =*/ LLM_FUSED_OP_FLASH_ATTN,
    /*.name             =*/ "Flash Attention",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ar_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_AR,
    /*.name             =*/ "fused Gated Delta Net (autoregressive)",
    /*.n_tokens_per_seq =*/ 1,
};

static const llm_fused_op_probe llm_fused_op_gdn_ch_probe = {
    /*.op               =*/ LLM_FUSED_OP_GDN_CH,
    /*.name             =*/ "fused Gated Delta Net (chunked)",
    /*.n_tokens_per_seq =*/ 16,
};

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_rs_seq = params.n_rs_seq;
    if (cparams.n_rs_seq > 0 && !llm_arch_supports_rs_rollback(model.arch)) {
        LLAMA_LOG_DEBUG("%s: n_rs_seq=%u requested but model arch does not support recurrent partial rollback; clamping to 0\n",
                        __func__, cparams.n_rs_seq);
        cparams.n_rs_seq = 0;
    }

    cparams.n_threads               = params.n_threads;
    cparams.n_threads_batch         = params.n_threads_batch;
    cparams.yarn_ext_factor         = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor        = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast          = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow          = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings              = params.embeddings;
    cparams.embeddings_nextn        = false;
    cparams.embeddings_nextn_masked = false;
    cparams.offload_kqv             = params.offload_kqv;
    cparams.no_perf                 = params.no_perf;
    cparams.warmup                  = false;

    cparams.embeddings_layer_inp.resize(hparams.n_layer(), false);
    embd_layer_inp.resize(hparams.n_layer());

    cparams.ctx_type     = params.ctx_type;
    cparams.pooling_type = params.pooling_type;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    cparams.ctx_other = nullptr;

    // TODO: more generic
    if (model.arch == LLM_ARCH_GEMMA4_ASSISTANT) {
        if (params.ctx_other == nullptr) {
            // TODO: change from runtime_error to llama_exception to avoid printing error message
            throw std::runtime_error("Gemma4Assistant requires ctx_other to be set (this warning is normal during memory fitting)");
        }

        cparams.ctx_other = params.ctx_other;
    }

    if (model.arch == LLM_ARCH_EAGLE3 || model.arch == LLM_ARCH_DFLASH) {
        if (model.tok_embd == nullptr || model.output == nullptr) {
            if (params.ctx_other == nullptr) {
                throw std::runtime_error(model.arch_name() + " requires ctx_other to be set (this warning is normal during memory fitting)");
            }
            cparams.ctx_other = params.ctx_other;
        }
    }

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    cparams.n_outputs_max = params.n_outputs_max == 0 || llama_model_has_encoder(&model) ? cparams.n_batch : params.n_outputs_max;

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);
    LLAMA_LOG_INFO("%s: n_rs_seq      = %u\n",   __func__, cparams.n_rs_seq);
    LLAMA_LOG_INFO("%s: n_outputs_max = %u\n",   __func__, cparams.n_outputs_max);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_INFO("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k    =*/ params.type_k,
            /*.type_v    =*/ params.type_v,
            /*.swa_full  =*/ params.swa_full,
            /*.ctx_type  =*/ cparams.ctx_type,
            /*.mem_other =*/ llama_get_memory(cparams.ctx_other),
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer_all &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    // Auxiliary schedulers reference backend_ptrs; tear them down while their
    // underlying backend instances are still alive.
    for (auto & worker : region_ring_worker_resources) {
        if (worker.sched) {
            ggml_backend_sched_synchronize(worker.sched.get());
        }
        worker.res.reset();
        worker.sched.reset();
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::resolve_fused_ops(const llama_memory_context_i * mctx, uint32_t n_seqs) {
    const char * func = __func__;
    auto resolve = [&](const llm_fused_op_probe & probe, bool & enabled) {
        if (!enabled) {
            return;
        }

        const uint32_t n_tokens_probe = probe.n_tokens_per_seq*n_seqs;

        auto * gf = graph_reserve(n_tokens_probe, n_seqs, n_tokens_probe, mctx, true);
        if (!gf) {
            throw std::runtime_error(std::string("failed to reserve graph for ") + probe.name + " check");
        }

        bool device_mismatch = false;
        for (const auto & node : get_gf_res_reserve()->get_fused_nodes()) {
            if (node.op != probe.op) {
                continue;
            }

            GGML_ASSERT(node.il >= 0);

            ggml_backend_t backend_fused = ggml_backend_sched_get_tensor_backend(sched.get(), node.tensor);
            ggml_backend_dev_t device_fused = backend_fused ? ggml_backend_get_device(backend_fused) : nullptr;

            // TODO: make this descriptor-specific; model.dev_layer() preserves the current behavior,
            // but is still wrong for cases like --no-kv-offload.
            ggml_backend_dev_t device_layer = model.dev_layer(node.il);

            if (device_fused != device_layer) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but %s "
                        "is assigned to device %s (usually due to missing support)\n",
                        func, node.il,
                        device_layer ? ggml_backend_dev_name(device_layer) : "none",
                        probe.name,
                        device_fused ? ggml_backend_dev_name(device_fused) : "none");
                device_mismatch = true;
                break;
            }
        }

        if (device_mismatch) {
            enabled = false;
            LLAMA_LOG_WARN("%s: %s not supported, set to disabled\n", func, probe.name);
        } else {
            enabled = true;
            LLAMA_LOG_INFO("%s: %s enabled\n", func, probe.name);
        }
    };

    if (cparams.auto_fa) {
        resolve(llm_fused_op_flash_attn_probe, cparams.flash_attn);
        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", func);
        resolve(llm_fused_op_gdn_ar_probe, cparams.fused_gdn_ar);
        resolve(llm_fused_op_gdn_ch_probe, cparams.fused_gdn_ch);
        cparams.auto_fgdn = false;
    }
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    // Recreate auxiliary Ring schedulers whenever the normal scheduler is
    // re-reserved; they must not retain graph allocators from an old context.
    for (auto & worker : region_ring_worker_resources) {
        if (worker.sched) {
            ggml_backend_sched_synchronize(worker.sched.get());
        }
        worker.res.reset();
        worker.sched.reset();
        worker.cached_region = std::numeric_limits<size_t>::max();
        worker.last_use = 0;
    }
    region_ring_gpu_weight_bank_epoch = 0;

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs   = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("[GRAPH_CAP] tokens=%u max_nodes=%zu\n", n_tokens, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_pipe.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes,
                                       cparams.pipeline_parallel, cparams.op_offload));
    sched_pipe.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes,
                                            false, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    resolve_fused_ops(mctx.get(), n_seqs);

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    const bool decode_profile_pending =
        decode_runtime_profile.pending &&
        n_queued_tokens == 1;
    const int64_t sync_begin_us =
        decode_profile_pending ? ggml_time_us() : 0;

    ggml_backend_sched_synchronize(sched.get());

    if (decode_profile_pending) {
        const int64_t sync_us =
            ggml_time_us() - sync_begin_us;
        const int64_t total_us =
            t_compute_start_us > 0 ?
                ggml_time_us() - t_compute_start_us : 0;

        ggml_backend_meta_tensor_profile meta {};
        const int n_backends =
            ggml_backend_sched_get_n_backends(sched.get());
        for (int ib = 0; ib < n_backends; ++ib) {
            ggml_backend_meta_tensor_profile cur {};
            if (!ggml_backend_meta_tensor_profile_get(
                    ggml_backend_sched_get_backend(
                        sched.get(), ib),
                    &cur)) {
                continue;
            }
            meta.attn_us += cur.attn_us;
            meta.pc_ffn_us += cur.pc_ffn_us;
            meta.h2d_us += cur.h2d_us;
            meta.phone_us += cur.phone_us;
            meta.d2h_us += cur.d2h_us;
            meta.reduce_us += cur.reduce_us;
            meta.wait_us += cur.wait_us;
            meta.compute_wall_us += cur.compute_wall_us;
            meta.reduce_wall_us += cur.reduce_wall_us;
            meta.meta_total_us += cur.meta_total_us;
            meta.other_main_us += cur.other_main_us;
            meta.graph_compute_count +=
                cur.graph_compute_count;
            meta.graph_rebuild_count +=
                cur.graph_rebuild_count;
            meta.graph_total_us += cur.graph_total_us;
            meta.graph_rebuild_us +=
                cur.graph_rebuild_us;
            meta.graph_execute_us +=
                cur.graph_execute_us;
            meta.graph_other_us += cur.graph_other_us;
            meta.simple_backend_count =
                std::max(
                    meta.simple_backend_count,
                    cur.simple_backend_count);
            for (int j = 0;
                 j < GGML_BACKEND_META_MAX_DEVICES;
                 ++j) {
                meta.simple_backend_compute_us[j] +=
                    cur.simple_backend_compute_us[j];
                meta.simple_backend_compute_calls[j] +=
                    cur.simple_backend_compute_calls[j];
            }

            for (int64_t ie = 0;
                 ie < cur.layer_timing_entries &&
                 ie < GGML_BACKEND_META_MAX_LAYER_TIMINGS;
                 ++ie) {
                const int64_t dst = meta.layer_timing_entries;
                if (dst >= GGML_BACKEND_META_MAX_LAYER_TIMINGS) {
                    break;
                }
                meta.layer_timing_first[dst] =
                    cur.layer_timing_first[ie];
                meta.layer_timing_last[dst] =
                    cur.layer_timing_last[ie];
                meta.layer_timing_pc_compute_us[dst] =
                    cur.layer_timing_pc_compute_us[ie];
                meta.layer_timing_phone_compute_us[dst] =
                    cur.layer_timing_phone_compute_us[ie];
                meta.layer_timing_copy_0to1_us[dst] =
                    cur.layer_timing_copy_0to1_us[ie];
                meta.layer_timing_copy_1to0_us[dst] =
                    cur.layer_timing_copy_1to0_us[ie];
                meta.layer_timing_wall_us[dst] =
                    cur.layer_timing_wall_us[ie];
                meta.layer_timing_entries = dst + 1;
            }
        }

        const int64_t timeline_accounted_us =
            decode_runtime_profile.prepare_us +
            decode_runtime_profile.split_prepare_us +
            decode_runtime_profile.split_submit_us +
            decode_runtime_profile.output_submit_us +
            sync_us;
        const int64_t residual_us =
            std::max<int64_t>(
                0, total_us - timeline_accounted_us);

        LLAMA_LOG_DEBUG(
            "[DECODE_RUNTIME] sample=%" PRId64
            " total_ms=%.3f prepare_ms=%.3f "
            "split_prepare_ms=%.3f split_submit_ms=%.3f "
            "output_submit_ms=%.3f sync_ms=%.3f "
            "residual_ms=%.3f splits=%d\n",
            decode_runtime_profile.sample_index,
            total_us / 1000.0,
            decode_runtime_profile.prepare_us / 1000.0,
            decode_runtime_profile.split_prepare_us /
                1000.0,
            decode_runtime_profile.split_submit_us /
                1000.0,
            decode_runtime_profile.output_submit_us /
                1000.0,
            sync_us / 1000.0,
            residual_us / 1000.0,
            decode_runtime_profile.n_splits);

        const double simple0_ms =
            meta.simple_backend_count > 0 ?
                meta.simple_backend_compute_us[0] / 1000.0 : 0.0;
        const double simple1_ms =
            meta.simple_backend_count > 1 ?
                meta.simple_backend_compute_us[1] / 1000.0 : 0.0;
        const int64_t simple0_calls =
            meta.simple_backend_count > 0 ?
                meta.simple_backend_compute_calls[0] : 0;
        const int64_t simple1_calls =
            meta.simple_backend_count > 1 ?
                meta.simple_backend_compute_calls[1] : 0;
        const double simple_max_ms =
            std::max(simple0_ms, simple1_ms);
        const double meta_uncovered_ms =
            std::max(
                0.0,
                meta.graph_execute_us / 1000.0 -
                    simple_max_ms);

        LLAMA_LOG_DEBUG(
            "[DECODE_META] sample=%" PRId64
            " graph_calls=%" PRId64
            " rebuilds=%" PRId64
            " graph_total_ms=%.3f rebuild_ms=%.3f "
            "execute_ms=%.3f other_ms=%.3f "
            "simple_count=%" PRId64
            " simple0_ms=%.3f simple0_calls=%" PRId64
            " simple1_ms=%.3f simple1_calls=%" PRId64
            " execute_minus_simple_max_ms=%.3f "
            "attn_ms=%.3f pc_ffn_ms=%.3f "
            "h2d_ms=%.3f phone_ms=%.3f d2h_ms=%.3f "
            "reduce_ms=%.3f wait_ms=%.3f\n",
            decode_runtime_profile.sample_index,
            meta.graph_compute_count,
            meta.graph_rebuild_count,
            meta.graph_total_us / 1000.0,
            meta.graph_rebuild_us / 1000.0,
            meta.graph_execute_us / 1000.0,
            meta.graph_other_us / 1000.0,
            meta.simple_backend_count,
            simple0_ms,
            simple0_calls,
            simple1_ms,
            simple1_calls,
            meta_uncovered_ms,
            meta.attn_us / 1000.0,
            meta.pc_ffn_us / 1000.0,
            meta.h2d_us / 1000.0,
            meta.phone_us / 1000.0,
            meta.d2h_us / 1000.0,
            meta.reduce_us / 1000.0,
            meta.wait_us / 1000.0);

        int64_t classified_layers = 0;
        int64_t pc_only_layers = 0;
        int64_t phone_only_layers = 0;
        int64_t tensor_layers = 0;
        int64_t mixed_entries = 0;

        int64_t pc_only_compute_us = 0;
        int64_t pc_only_wall_us = 0;
        int64_t phone_only_submit_us = 0;
        int64_t phone_only_wall_us = 0;
        int64_t tensor_pc_compute_us = 0;
        int64_t tensor_phone_submit_us = 0;
        int64_t tensor_wall_us = 0;
        int64_t copy_0to1_us = 0;
        int64_t copy_1to0_us = 0;
        int64_t orchestration_us = 0;
        int64_t layer_wall_us = 0;

        for (int64_t ie = 0;
             ie < meta.layer_timing_entries &&
             ie < GGML_BACKEND_META_MAX_LAYER_TIMINGS;
             ++ie) {
            const int first_layer =
                meta.layer_timing_first[ie];
            const int last_layer =
                meta.layer_timing_last[ie];
            if (first_layer < 0 || last_layer < first_layer) {
                continue;
            }

            int n_pc = 0;
            int n_phone = 0;
            int n_tensor = 0;
            for (int il = first_layer; il <= last_layer; ++il) {
                switch (model.hybrid_layer_mode(il)) {
                    case llama_hybrid_layer_mode::PC_ONLY:
                        ++n_pc;
                        break;
                    case llama_hybrid_layer_mode::PHONE_ONLY:
                        ++n_phone;
                        break;
                    case llama_hybrid_layer_mode::TENSOR_SPLIT:
                    case llama_hybrid_layer_mode::TENSOR_C2P:
                    case llama_hybrid_layer_mode::TENSOR_P2C:
                    case llama_hybrid_layer_mode::TENSOR_PHONE_PRIMARY:
                        ++n_tensor;
                        break;
                }
            }

            const int n_layers =
                n_pc + n_phone + n_tensor;
            if (n_layers <= 0) {
                continue;
            }
            classified_layers += n_layers;

            const int64_t pc_us =
                meta.layer_timing_pc_compute_us[ie];
            const int64_t phone_us =
                meta.layer_timing_phone_compute_us[ie];
            const int64_t c01_us =
                meta.layer_timing_copy_0to1_us[ie];
            const int64_t c10_us =
                meta.layer_timing_copy_1to0_us[ie];
            const int64_t wall_us =
                meta.layer_timing_wall_us[ie];
            const int64_t orchestration_entry_us =
                std::max<int64_t>(
                    0,
                    wall_us -
                        std::max(pc_us, phone_us) -
                        c01_us - c10_us);

            copy_0to1_us += c01_us;
            copy_1to0_us += c10_us;
            orchestration_us += orchestration_entry_us;
            layer_wall_us += wall_us;

            if (n_pc == n_layers) {
                pc_only_layers += n_layers;
                pc_only_compute_us += pc_us;
                pc_only_wall_us += wall_us;
            } else if (n_phone == n_layers) {
                phone_only_layers += n_layers;
                phone_only_submit_us += phone_us;
                phone_only_wall_us += wall_us;
            } else if (n_tensor == n_layers) {
                tensor_layers += n_layers;
                tensor_pc_compute_us += pc_us;
                tensor_phone_submit_us += phone_us;
                tensor_wall_us += wall_us;
            } else {
                // Fused timing ranges should normally stay within one placement
                // region. Keep the entry visible instead of silently assigning
                // it to the wrong region.
                ++mixed_entries;
            }
        }

        const double pc_only_per_layer_ms =
            pc_only_layers > 0 ?
                pc_only_compute_us /
                    1000.0 / pc_only_layers : 0.0;
        const double phone_only_wall_per_layer_ms =
            phone_only_layers > 0 ?
                phone_only_wall_us /
                    1000.0 / phone_only_layers : 0.0;
        const double non_layer_execute_ms =
            std::max(
                0.0,
                meta.graph_execute_us / 1000.0 -
                    layer_wall_us / 1000.0);

        LLAMA_LOG_DEBUG(
            "[DECODE_META_LAYER] sample=%" PRId64
            " entries=%" PRId64
            " layers=%" PRId64
            " mixed_entries=%" PRId64
            " pc_only_layers=%" PRId64
            " pc_only_compute_ms=%.3f pc_only_per_layer_ms=%.3f "
            "pc_only_wall_ms=%.3f "
            "phone_only_layers=%" PRId64
            " phone_submit_ms=%.3f phone_wall_ms=%.3f "
            "phone_wall_per_layer_ms=%.3f "
            "tensor_layers=%" PRId64
            " tensor_pc_ms=%.3f tensor_phone_submit_ms=%.3f "
            "tensor_wall_ms=%.3f "
            "copy_0to1_ms=%.3f copy_1to0_ms=%.3f "
            "orchestration_ms=%.3f layer_wall_ms=%.3f "
            "non_layer_execute_ms=%.3f\n",
            decode_runtime_profile.sample_index,
            meta.layer_timing_entries,
            classified_layers,
            mixed_entries,
            pc_only_layers,
            pc_only_compute_us / 1000.0,
            pc_only_per_layer_ms,
            pc_only_wall_us / 1000.0,
            phone_only_layers,
            phone_only_submit_us / 1000.0,
            phone_only_wall_us / 1000.0,
            phone_only_wall_per_layer_ms,
            tensor_layers,
            tensor_pc_compute_us / 1000.0,
            tensor_phone_submit_us / 1000.0,
            tensor_wall_us / 1000.0,
            copy_0to1_us / 1000.0,
            copy_1to0_us / 1000.0,
            orchestration_us / 1000.0,
            layer_wall_us / 1000.0,
            non_layer_execute_ms);

        decode_runtime_profile.pending = false;
    }

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();
        gf_res_pipe->reset();
        // The same KV cache is shared by Ring's auxiliary Graph Results.
        // Invalidate their reuse state after any memory module update.
        for (auto & worker : region_ring_worker_resources) {
            if (worker.res) {
                worker.res->reset();
            }
        }

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        const uint32_t n_outputs_max = std::min(n_tokens, cparams.n_outputs_max);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_max, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

float * llama_context::get_embeddings_nextn() {
    output_reorder();

    return embd_nextn.data;
}

float * llama_context::get_embeddings_nextn_ith(int32_t i) {
    output_reorder();

    try {
        if (embd_nextn.data == nullptr) {
            throw std::runtime_error("no nextn embeddings");
        }

        const uint32_t n_embd = model.hparams.n_embd_out();

        if (!cparams.embeddings_nextn_masked) {
            // unmasked: nextn rows are stored densely, indexed by raw token position.
            if (i < 0 || (size_t)(i + 1) * n_embd > embd_nextn.size) {
                throw std::runtime_error(format("out of range [0, %zu)", embd_nextn.size / n_embd));
            }
            return embd_nextn.data + (size_t) i * n_embd;
        }

        const int64_t j = output_resolve_row(i);
        return embd_nextn.data + j*n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid nextn embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_layer_inp(uint32_t lid) {
    output_reorder();

    GGML_ASSERT(lid < embd_layer_inp.size() && embd_layer_inp[lid].has_data());

    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_embeddings_nextn(bool value, bool masked) {
    LLAMA_LOG_DEBUG("%s: value = %d, masked = %d\n", __func__, value, masked);

    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid < model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

void llama_context::set_nextn_layer_offset(int32_t offset) {
    cparams.nextn_layer_offset = offset;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    if (sampler && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        static bool warned = false;
        if (!warned) {
            LLAMA_LOG_WARN("%s: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n", __func__);
            warned = true;
        }
        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }
        sampling.samplers.erase(seq_id);
        return false;
    }

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::prepare_ubatch(llm_graph_result *       res,
                                                 ggml_backend_sched_t     sched_use,
                                                 const llama_ubatch &     ubatch,
                                                 llm_graph_type           gtype,
                                                 llama_memory_context_i * mctx,
                                                 ggml_status &            ret,
                                                 bool                     apply_mctx,
                                                 const llama_hybrid_runtime_stage * stage,
                                                 int32_t                  outputs_override,
                                                 bool                     force_hidden_input,
                                                 bool                     embedding_only,
                                                 bool                     preserve_weight_cache_on_reset) {
    const bool log_formal_prepare =
        ubatch.n_tokens > 1 &&
        stage == nullptr &&
        std::getenv("LLAMA_HYBRID_RETURN_WAVEFRONT") != nullptr;
    const bool log_stage_prepare =
        ubatch.n_tokens > 1 &&
        stage != nullptr;
    const bool log_prepare_breakdown = log_formal_prepare || log_stage_prepare;

    const int64_t prepare_total_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;
    int64_t apply_mctx_us    = 0;
    int64_t graph_params_us  = 0;
    int64_t reuse_check_us   = 0;
    int64_t reuse_sync_us    = 0;
    int64_t graph_reset_us   = 0;
    int64_t graph_build_us   = 0;
    int64_t graph_alloc_us   = 0;
    int64_t set_inputs_us    = 0;
    bool reused_graph = false;

    const int64_t apply_mctx_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;
    if (apply_mctx && mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (log_prepare_breakdown) {
        apply_mctx_us = ggml_time_us() - apply_mctx_begin_us;
    }

    auto * gf = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const int64_t graph_params_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;
    // The optional explicit task output count avoids writing shared n_outputs.
    auto gparams = graph_params(res, ubatch, mctx, gtype, sched_use, outputs_override);
    if (stage != nullptr) {
        const int n_layer = model.hparams.n_layer();
        GGML_ASSERT(stage->layer_begin >= 0);
        GGML_ASSERT(stage->layer_begin < stage->layer_end);
        GGML_ASSERT(stage->layer_end <= n_layer);

        gparams.hybrid_layer_begin   = stage->layer_begin;
        gparams.hybrid_layer_end     = stage->layer_end;
        gparams.hybrid_hidden_input  =
            stage->layer_begin > 0 || force_hidden_input;
        gparams.hybrid_output_head   =
            !embedding_only && stage->layer_end == n_layer;
        gparams.hybrid_wave_probe    = stage->wave_probe;
        gparams.hybrid_embedding_only = embedding_only;
    }
    if (log_prepare_breakdown) {
        graph_params_us = ggml_time_us() - graph_params_begin_us;
    }

    const int64_t reuse_check_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;
    const bool can_reuse_graph = !graph_reuse_disable && res->can_reuse(gparams);
    if (log_prepare_breakdown) {
        reuse_check_us = ggml_time_us() - reuse_check_begin_us;
    }

    if (can_reuse_graph) {
        reused_graph = true;
        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            const int64_t reuse_sync_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;
            ggml_backend_sched_synchronize(sched_use);
            if (log_prepare_breakdown) {
                reuse_sync_us = ggml_time_us() - reuse_sync_begin_us;
            }
        }

        n_reused++;
    } else {
        const int64_t graph_reset_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;
        res->reset();

        if (preserve_weight_cache_on_reset &&
                ggml_backend_sched_get_weight_cache(sched_use)) {
            ggml_backend_sched_reset_preserve_weight_cache(
                sched_use);
            if (stage != nullptr) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT] event=GRAPH_REBUILD_KEEP_CACHE "
                    "layers=[%d,%d) tokens=%u status=PRESERVED\n",
                    stage->layer_begin, stage->layer_end,
                    ubatch.n_tokens);
            }
        } else {
            ggml_backend_sched_reset(sched_use);
        }
        ggml_backend_sched_set_eval_callback(sched_use, cparams.cb_eval, cparams.cb_eval_user_data);
        if (log_prepare_breakdown) {
            graph_reset_us = ggml_time_us() - graph_reset_begin_us;
        }

        const int64_t graph_build_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;
        gf = model.build_graph(gparams);
        if (log_prepare_breakdown) {
            graph_build_us = ggml_time_us() - graph_build_begin_us;
        }

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        const int64_t graph_alloc_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;
        if (!ggml_backend_sched_alloc_graph(sched_use, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
        if (log_prepare_breakdown) {
            graph_alloc_us = ggml_time_us() - graph_alloc_begin_us;
        }
    }

    // set the input data for the input tensors
    {
        const int64_t set_inputs_begin_us = log_prepare_breakdown ? ggml_time_us() : 0;

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        if (log_prepare_breakdown) {
            set_inputs_us = ggml_time_us() - set_inputs_begin_us;
        }
    }

    if (log_prepare_breakdown) {
        const int64_t prepare_total_us = ggml_time_us() - prepare_total_begin_us;

        if (log_formal_prepare) {
            LLAMA_LOG_DEBUG(
                "[FORMAL_PREPARE_BREAKDOWN] tokens=%u reused=%d total_ms=%.3f "
                "apply_mctx_ms=%.3f graph_params_ms=%.3f build_graph_ms=%.3f "
                "alloc_graph_ms=%.3f set_inputs_ms=%.3f unaccounted_ms=%.3f\n",
                ubatch.n_tokens,
                reused_graph ? 1 : 0,
                prepare_total_us / 1000.0,
                apply_mctx_us / 1000.0,
                graph_params_us / 1000.0,
                graph_build_us / 1000.0,
                graph_alloc_us / 1000.0,
                set_inputs_us / 1000.0,
                std::max<int64_t>(
                    0,
                    prepare_total_us - apply_mctx_us - graph_params_us -
                        graph_build_us - graph_alloc_us - set_inputs_us) / 1000.0);
        }

        if (log_stage_prepare) {
            const int graph_nodes = gf != nullptr ? ggml_graph_n_nodes(gf) : 0;
            const int graph_splits = ggml_backend_sched_get_n_splits(sched_use);
            const int64_t accounted_us =
                apply_mctx_us + graph_params_us + reuse_check_us + reuse_sync_us +
                graph_reset_us + graph_build_us + graph_alloc_us + set_inputs_us;
            LLAMA_LOG_DEBUG(
                "[STAGE_PREPARE_BREAKDOWN] stage=%s layers=[%d,%d) tokens=%u outputs=%d "
                "reused=%d nodes=%d splits=%d total_ms=%.3f "
                "apply_mctx_ms=%.3f graph_params_ms=%.3f reuse_check_ms=%.3f "
                "reuse_sync_ms=%.3f reset_ms=%.3f build_graph_ms=%.3f "
                "alloc_graph_ms=%.3f set_inputs_ms=%.3f unaccounted_ms=%.3f\n",
                llama_hybrid_runtime_stage_name(stage->kind),
                stage->layer_begin,
                stage->layer_end,
                ubatch.n_tokens,
                gparams.n_outputs,
                reused_graph ? 1 : 0,
                graph_nodes,
                graph_splits,
                prepare_total_us / 1000.0,
                apply_mctx_us / 1000.0,
                graph_params_us / 1000.0,
                reuse_check_us / 1000.0,
                reuse_sync_us / 1000.0,
                graph_reset_us / 1000.0,
                graph_build_us / 1000.0,
                graph_alloc_us / 1000.0,
                set_inputs_us / 1000.0,
                std::max<int64_t>(0, prepare_total_us - accounted_us) / 1000.0);
        }
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

bool llama_context::prepare_pipe_slot(llama_prefill_pipe_slot & slot,
                                      const llama_ubatch &      ubatch,
                                      llm_graph_type            gtype,
                                      llama_memory_context_i *  mctx,
                                      int                       ubatch_id,
                                      bool                      stage_queue,
                                      ggml_status &             ret) {
    GGML_ASSERT(!slot.active);

    slot.res = prepare_ubatch(slot.res, slot.sched, ubatch, gtype, mctx, ret);
    if (slot.res == nullptr) {
        return false;
    }

    const int n_splits = ggml_backend_sched_get_n_splits(slot.sched);

    int gpu1_begin = -1;
    int gpu1_end   = -1;
    int gpu2_begin = -1;
    int gpu2_end   = -1;

    int  gpu_region = 0;
    bool in_gpu     = false;

    for (int split_id = 0; split_id < n_splits; ++split_id) {
        ggml_backend_t backend = ggml_backend_sched_get_split_backend(slot.sched, split_id);
        const bool is_gpu = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU;

        if (is_gpu && !in_gpu) {
            ++gpu_region;
            in_gpu = true;

            if (gpu_region == 1) {
                gpu1_begin = split_id;
            } else if (gpu_region == 2) {
                gpu2_begin = split_id;
            }
        }

        if (!is_gpu && in_gpu) {
            if (gpu_region == 1) {
                gpu1_end = split_id;
            } else if (gpu_region == 2) {
                gpu2_end = split_id;
            }

            in_gpu = false;
        }
    }

    if (in_gpu) {
        if (gpu_region == 1) {
            gpu1_end = n_splits;
        } else if (gpu_region == 2) {
            gpu2_end = n_splits;
        }
    }

    const bool topo_stage_queue =
        gpu_region == 1 && gpu1_begin >= 0 && gpu1_end > gpu1_begin && gpu1_end < n_splits;
    const bool topo_one_gpu =
        gpu_region == 1 && gpu1_begin > 0 && gpu1_end > gpu1_begin && gpu1_end < n_splits;
    const bool topo_two_gpu = gpu_region == 2 && gpu1_begin > 0 && gpu1_end > gpu1_begin &&
                              gpu2_begin > gpu1_end && gpu2_end == n_splits;
    const bool topo_ok = stage_queue ? topo_stage_queue : (topo_one_gpu || topo_two_gpu);

    if (!topo_ok) {
        printf("[PIPE2_TOPO_SUM] ub=%d n_splits=%d gpu_regions=%d gpu1=[%d,%d) gpu2=[%d,%d)\n", ubatch_id,
               n_splits, gpu_region, gpu1_begin, gpu1_end, gpu2_begin, gpu2_end);

        for (int split_id = 0; split_id < n_splits; ++split_id) {
            ggml_backend_t backend = ggml_backend_sched_get_split_backend(slot.sched, split_id);
            printf("[PIPE2_TOPO] ub=%d split=%d backend=%s type=%d\n", ubatch_id, split_id,
                   ggml_backend_name(backend),
                   (int) ggml_backend_dev_type(ggml_backend_get_device(backend)));
        }

        LLAMA_LOG_WARN("%s: ubatch %d does not support PIPE2 topology; falling back to serial execution\n", __func__,
                       ubatch_id);
        ret = GGML_STATUS_SUCCESS;
        return false;
    }

    slot.ubatch_id     = ubatch_id;
    slot.n_tokens      = ubatch.n_tokens;
    slot.pre_begin     = 0;
    slot.pre_end       = gpu1_begin;
    slot.gpu_begin     = gpu1_begin;
    slot.gpu_end       = gpu1_end;
    slot.post_begin    = gpu1_end;
    slot.post_end      = !stage_queue && topo_two_gpu ? gpu2_begin : n_splits;
    slot.tail_begin    = !stage_queue && topo_two_gpu ? gpu2_begin : n_splits;
    slot.tail_end      = !stage_queue && topo_two_gpu ? gpu2_end : n_splits;
    slot.active        = true;
    slot.post_prepared = false;

    printf("[PIPE2_STAGE] ub=%d PRE=[%d,%d) GPU=[%d,%d) POST=[%d,%d) TAIL=[%d,%d)\n", slot.ubatch_id,
           slot.pre_begin, slot.pre_end, slot.gpu_begin, slot.gpu_end, slot.post_begin, slot.post_end,
           slot.tail_begin, slot.tail_end);

    return true;
}

ggml_status llama_context::pipe_run_pre(llama_prefill_pipe_slot & slot) {
    GGML_ASSERT(slot.active);
    slot.t_pre_begin = ggml_time_us();

    if (slot.pre_begin == slot.pre_end) {
        slot.t_pre_end = slot.t_pre_begin;
        return GGML_STATUS_SUCCESS;
    }

    GGML_ASSERT(slot.pre_begin < slot.pre_end);
    printf("[PIPE2] ub=%d PRE_BEGIN t=%" PRId64 "\n", slot.ubatch_id, slot.t_pre_begin);

    const ggml_status status = graph_compute_range(slot.sched, slot.pre_begin, slot.pre_end, true);
    if (status == GGML_STATUS_SUCCESS) {
        slot.t_pre_end = ggml_time_us();
        printf("[PIPE2] ub=%d PRE_END t=%" PRId64 " dur=%.3f ms\n", slot.ubatch_id, slot.t_pre_end,
               (slot.t_pre_end - slot.t_pre_begin) / 1000.0);
    }
    return status;
}

ggml_status llama_context::pipe_run_gpu(llama_prefill_pipe_slot & slot) {
    GGML_ASSERT(slot.active);
    slot.t_gpu_begin = ggml_time_us();
    printf("[PIPE2] ub=%d GPU_BEGIN t=%" PRId64 "\n", slot.ubatch_id, slot.t_gpu_begin);

    const ggml_status status = graph_compute_range(slot.sched, slot.gpu_begin, slot.gpu_end, true);
    if (status == GGML_STATUS_SUCCESS) {
        slot.t_gpu_submit = ggml_time_us();
        printf("[PIPE2] ub=%d GPU_SUBMITTED t=%" PRId64 " host_submit=%.3f ms\n", slot.ubatch_id,
               slot.t_gpu_submit, (slot.t_gpu_submit - slot.t_gpu_begin) / 1000.0);
    }
    return status;
}

ggml_status llama_context::pipe_prepare_post(llama_prefill_pipe_slot & slot) {
    GGML_ASSERT(slot.active);
    GGML_ASSERT(!slot.post_prepared);
    GGML_ASSERT(slot.post_begin < slot.post_end);

    printf("[PIPE2] ub=%d POST_PREP_BEGIN\n", slot.ubatch_id);
    const ggml_status status = ggml_backend_sched_prepare_split(slot.sched, slot.post_begin);
    if (status == GGML_STATUS_SUCCESS) {
        slot.post_prepared = true;
        printf("[PIPE2] ub=%d POST_INPUT_READY\n", slot.ubatch_id);
    }
    return status;
}

ggml_status llama_context::pipe_run_post_prepared(llama_prefill_pipe_slot & slot) {
    GGML_ASSERT(slot.active);
    GGML_ASSERT(slot.post_prepared);

    slot.t_post_begin = ggml_time_us();
    printf("[PIPE2] ub=%d POST_BEGIN t=%" PRId64 "\n", slot.ubatch_id, slot.t_post_begin);

    ggml_status status = ggml_backend_sched_compute_split(slot.sched, slot.post_begin);
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }

    if (slot.post_begin + 1 < slot.post_end) {
        status = graph_compute_range(slot.sched, slot.post_begin + 1, slot.post_end, true);
    }

    if (status == GGML_STATUS_SUCCESS) {
        slot.t_post_end = ggml_time_us();
        printf("[PIPE2] ub=%d POST_END t=%" PRId64 " dur=%.3f ms\n", slot.ubatch_id, slot.t_post_end,
               (slot.t_post_end - slot.t_post_begin) / 1000.0);
    }

    return status;
}

void llama_context::pipe_sync_post(llama_prefill_pipe_slot & slot) {
    GGML_ASSERT(slot.active);
    GGML_ASSERT(slot.post_prepared);

    ggml_backend_t prev = nullptr;
    for (int split_id = slot.post_begin; split_id < slot.post_end; ++split_id) {
        ggml_backend_t backend = ggml_backend_sched_get_split_backend(slot.sched, split_id);
        GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_GPU);

        if (backend != prev) {
            ggml_backend_synchronize(backend);
            prev = backend;
        }
    }

    slot.post_prepared = false;
}

ggml_status llama_context::pipe_run_tail(llama_prefill_pipe_slot & slot) {
    GGML_ASSERT(slot.active);
    GGML_ASSERT(!slot.post_prepared);

    if (slot.tail_begin == slot.tail_end) {
        slot.t_tail_begin = ggml_time_us();
        slot.t_tail_end   = slot.t_tail_begin;
        slot.active        = false;
        slot.post_prepared = false;
        return GGML_STATUS_SUCCESS;
    }

    GGML_ASSERT(slot.tail_begin < slot.tail_end);

    slot.t_tail_begin = ggml_time_us();
    printf("[PIPE2] ub=%d TAIL_BEGIN t=%" PRId64 "\n", slot.ubatch_id, slot.t_tail_begin);

    const ggml_status status = graph_compute_range(slot.sched, slot.tail_begin, slot.tail_end, true);
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }

    ggml_backend_t prev_backend = nullptr;
    for (int split_id = slot.tail_begin; split_id < slot.tail_end; ++split_id) {
        ggml_backend_t backend = ggml_backend_sched_get_split_backend(slot.sched, split_id);
        GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU);

        if (backend != prev_backend) {
            ggml_backend_synchronize(backend);
            prev_backend = backend;
        }
    }

    slot.t_tail_end = ggml_time_us();
    printf("[PIPE2] ub=%d TAIL_END t=%" PRId64 " dur=%.3f ms\n", slot.ubatch_id, slot.t_tail_end,
           (slot.t_tail_end - slot.t_tail_begin) / 1000.0);
    printf("[PIPE2_SUM] ub=%d PRE=%.3f GPU_HOST=%.3f POST=%.3f TAIL=%.3f ms\n", slot.ubatch_id,
           (slot.t_pre_end - slot.t_pre_begin) / 1000.0, (slot.t_gpu_submit - slot.t_gpu_begin) / 1000.0,
           (slot.t_post_end - slot.t_post_begin) / 1000.0, (slot.t_tail_end - slot.t_tail_begin) / 1000.0);

    slot.active        = false;
    slot.post_prepared = false;

    return GGML_STATUS_SUCCESS;
}

bool llama_context::copy_hybrid_stage_output(
        llm_graph_result *   res,
        ggml_backend_sched_t sched_use,
        float *              output,
        uint32_t             n_tokens,
        int                  ubatch_id,
        size_t               stage_index,
        size_t               block_index,
        bool                 graph_already_synchronized) {
    if (!graph_already_synchronized) {
        ggml_backend_sched_synchronize(sched_use);
    }

    ggml_tensor * stage_output = res->get_stage_output();
    const int64_t n_embd = model.hparams.n_embd;
    if (stage_output == nullptr || stage_output->type != GGML_TYPE_F32 || !ggml_is_contiguous(stage_output) ||
        stage_output->ne[0] != n_embd || stage_output->ne[1] != n_tokens) {
        LLAMA_LOG_INFO(
            "[HYBRID_EXEC] ub=%d stage=%zu block=%zu invalid stage output\n",
            ubatch_id, stage_index, block_index);
        return false;
    }

    ggml_backend_tensor_get(stage_output, output, 0, (size_t) n_tokens * n_embd * sizeof(float));

    // Diagnostics only. The hidden-state copy is already required by the
    // stage executor, so inspecting it does not add a backend synchronization.
    if (llama_region_ring_enabled() &&
        std::getenv("LLAMA_HYBRID_REGION_RING_TRACE") != nullptr) {
        const size_t count = (size_t) n_tokens * (size_t) n_embd;
        double sum = 0.0;
        double sq_sum = 0.0;
        double max_abs = 0.0;
        size_t nonfinite = 0;
        for (size_t i = 0; i < count; ++i) {
            const double value = (double) output[i];
            if (!std::isfinite(value)) {
                ++nonfinite;
                continue;
            }
            sum += value;
            sq_sum += value * value;
            max_abs = std::max(max_abs, std::abs(value));
        }
        LLAMA_LOG_ERROR(
            "[REGION_RING_BOUNDARY] stage=%zu block=%zu tokens=%u "
            "elements=%zu sum=%.9f l2=%.9f max_abs=%.9f "
            "nonfinite=%zu v0=%.9f\n",
            stage_index, block_index, n_tokens, count,
            sum, std::sqrt(sq_sum), max_abs, nonfinite,
            count > 0 ? (double) output[0] : 0.0);
    }
    return true;
}

llm_graph_result * llama_context::run_hybrid_stage_block(
        const llama_ubatch &               ubatch,
        const llama_hybrid_runtime_stage & stage,
        uint32_t                           token_begin,
        uint32_t                           block_tokens,
        float *                            stage_input,
        float *                            stage_output,
        llm_graph_type                     gtype,
        llama_memory_context_i *          mctx,
        ggml_backend_sched_t               sched_use,
        llm_graph_result *                 res_use,
        int                                ubatch_id,
        size_t                             stage_index,
        size_t                             block_index,
        llama_hybrid_boundary_action       action,
        int32_t                            block_outputs,
        bool &                             apply_mctx,
        bool                               synchronize,
        ggml_status &                      ret,
        llama_hybrid_stage_timing *        timing,
        bool                               task_local_graph_outputs,
        bool                               pc_phone_overlap,
        std::unique_lock<std::mutex> *     ring_prepare_gate,
        int64_t *                          lane_compute_begin_us,
        int64_t *                          lane_compute_end_us,
        ggml_backend_t                     strict_compute_backend,
        int                                ring_lane,
        std::unique_lock<std::mutex> *     ring_cpu_gate,
        std::unique_lock<std::mutex> *     ring_cuda_gate,
        int64_t *                          ring_prepare_wait_us,
        int64_t *                          ring_cpu_wait_us,
        int64_t *                          ring_cuda_wait_us,
        bool                               preserve_weight_cache_on_graph_reset) {
    auto * kv_mctx = dynamic_cast<llama_kv_cache_context *>(mctx);
    if (kv_mctx == nullptr || !kv_mctx->set_stage_range(ubatch_id, token_begin, block_tokens)) {
        LLAMA_LOG_INFO(
            "[HYBRID_EXEC] ub=%d stage=%zu block=%zu invalid KV range=[%u,%u)\n",
            ubatch_id, stage_index, block_index, token_begin, token_begin + block_tokens);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    llama_ubatch stage_ubatch = llama_hybrid_slice_ubatch(
        ubatch, token_begin, block_tokens, stage_input);
    if (block_outputs < 0) {
        LLAMA_LOG_ERROR("[REGION_RING_TASK_STATE] status=REJECT reason=NEGATIVE_OUTPUTS\n");
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    // Old stage executors retain their original shared-field behavior.
    // fifo-state instead passes this value directly into llm_graph_params.
    if (!task_local_graph_outputs) {
        n_outputs = block_outputs;
    }

    LLAMA_LOG_DEBUG(
        "[HYBRID_EXEC] ub=%d stage=%zu kind=%s layers=[%d,%d) block=%zu action=%s tokens=[%u,%u)\n",
        ubatch_id, stage_index, llama_hybrid_runtime_stage_name(stage.kind), stage.layer_begin, stage.layer_end,
        block_index, llama_hybrid_boundary_action_name(action), token_begin, token_begin + block_tokens);

    const bool split_backend_gates =
        pc_phone_overlap && ring_prepare_gate != nullptr && ring_lane >= 0;
    if (split_backend_gates) {
        const bool lane_valid = ring_lane >= 0 && ring_lane <= 2;
        const bool prepare_owned = ring_prepare_gate->owns_lock();
        const bool cpu_associated =
            ring_cpu_gate != nullptr && ring_cpu_gate->mutex() != nullptr;
        const bool cuda_associated =
            ring_cuda_gate != nullptr && ring_cuda_gate->mutex() != nullptr;
        const bool cpu_owned =
            ring_cpu_gate != nullptr && ring_cpu_gate->owns_lock();
        const bool cuda_owned =
            ring_cuda_gate != nullptr && ring_cuda_gate->owns_lock();
        const bool resources_ok =
            cpu_associated &&
            (ring_lane != 1 || cuda_associated) &&
            !cpu_owned && !cuda_owned;
        if (!lane_valid || !prepare_owned || !resources_ok) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_GATE] stage=%zu block=%zu lane=%d "
                "prepare=%d cpu_assoc=%d cpu_owned=%d "
                "cuda_assoc=%d cuda_owned=%d status=REJECT "
                "reason=LOCK_CONTRACT\n",
                stage_index, block_index, ring_lane,
                prepare_owned ? 1 : 0,
                cpu_associated ? 1 : 0, cpu_owned ? 1 : 0,
                cuda_associated ? 1 : 0, cuda_owned ? 1 : 0);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
    }

    const int64_t prepare_begin_us = timing != nullptr ? ggml_time_us() : 0;
    llm_graph_result * result = prepare_ubatch(
        res_use, sched_use, stage_ubatch, gtype, mctx, ret, apply_mctx,
        &stage, task_local_graph_outputs ? block_outputs : -1,
        stage_input != nullptr, false,
        preserve_weight_cache_on_graph_reset);
    if (timing != nullptr) {
        timing->prepare_us += ggml_time_us() - prepare_begin_us;
    }
    apply_mctx = false;
    if (result == nullptr || ret != GGML_STATUS_SUCCESS) {
        return nullptr;
    }

    const int n_splits = ggml_backend_sched_get_n_splits(sched_use);
    const bool gpu_weight_streaming =
        ring_lane == 1 &&
        model.hybrid_ring_gpu_weight_streaming &&
        ggml_backend_sched_get_weight_cache(sched_use);

    // A GPU Scheduler retains CPU as its final fallback backend. GPU-first
    // Region Ring has one intentional exception to CUDA-only compute: stage 0
    // may start with a CPU embedding split because Q4_K GET_ROWS is not
    // supported by CUDA. Only the exact CPU-prefix -> CUDA pattern is legal.
    int split_backend_count = 0;
    ggml_backend_t local_cpu = nullptr;
    if (split_backend_gates) {
        split_backend_count = ggml_backend_sched_get_n_backends(sched_use);
        if (split_backend_count <= 0 || split_backend_count >= 64) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_GATE] stage=%zu block=%zu lane=%d "
                "backends=%d status=REJECT reason=INVALID_BACKEND_COUNT\n",
                stage_index, block_index, ring_lane, split_backend_count);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        local_cpu = ggml_backend_sched_get_backend(
            sched_use, split_backend_count - 1);
        if (ggml_backend_dev_type(ggml_backend_get_device(local_cpu)) !=
                GGML_BACKEND_DEVICE_TYPE_CPU) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_GATE] stage=%zu block=%zu lane=%d "
                "status=REJECT reason=CPU_SENTINEL_MISSING\n",
                stage_index, block_index, ring_lane);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
    }

    bool gpu_cpu_embedding_prefix = false;
    if (strict_compute_backend != nullptr) {
        bool strict_ok = true;
        for (int split = 0; split < n_splits; ++split) {
            ggml_backend_t target =
                ggml_backend_sched_get_split_backend(sched_use, split);
            if (target == strict_compute_backend) {
                continue;
            }

            const bool allowed_embedding_prefix =
                split_backend_gates &&
                ring_lane == 1 &&
                stage_index == 0 &&
                stage.layer_begin == 0 &&
                n_splits == 2 &&
                split == 0 &&
                target == local_cpu &&
                ggml_backend_sched_get_split_backend(sched_use, 1) ==
                    strict_compute_backend;
            if (allowed_embedding_prefix) {
                gpu_cpu_embedding_prefix = true;
                continue;
            }

            LLAMA_LOG_ERROR(
                "[REGION_RING_STRICT_BACKEND] stage=%zu block=%zu "
                "split=%d/%d expected=%s actual=%s "
                "first_node=%s first_op=%s "
                "status=REJECT reason=COMPUTE_BACKEND_FALLBACK\n",
                stage_index, block_index, split, n_splits,
                ggml_backend_name(strict_compute_backend),
                target ? ggml_backend_name(target) : "NULL",
                ggml_backend_sched_get_split_first_node_name(
                    sched_use, split),
                ggml_op_name(
                    ggml_backend_sched_get_split_first_node_op(
                        sched_use, split)));
            strict_ok = false;
            break;
        }
        if (!strict_ok) {
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        if (gpu_cpu_embedding_prefix) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_GPU_PREFIX] stage=%zu block=%zu "
                "splits=2 prefix=CPU_EMBEDDING main=CUDA "
                "status=ALLOWED\n",
                stage_index, block_index);
        } else {
            LLAMA_LOG_ERROR(
                "[REGION_RING_STRICT_BACKEND] stage=%zu block=%zu "
                "splits=%d compute=%s weight_stream=%d status=PASS\n",
                stage_index, block_index, n_splits,
                ggml_backend_name(strict_compute_backend),
                gpu_weight_streaming ? 1 : 0);
        }
    }

    const int64_t compute_begin_us = timing != nullptr ? ggml_time_us() : 0;
    bool graph_already_synchronized = false;
    if (split_backend_gates) {
        if (ring_lane == 0) {
            if (n_splits != 1 ||
                    ggml_backend_sched_get_split_backend(sched_use, 0) !=
                        ggml_backend_sched_get_backend(sched_use, 0)) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GATE] stage=%zu block=%zu lane=CPU "
                    "splits=%d status=REJECT reason=CPU_TARGET_MISMATCH\n",
                    stage_index, block_index, n_splits);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
        } else if (ring_lane == 1) {
            if (n_splits != 1 &&
                    !gpu_cpu_embedding_prefix &&
                    !gpu_weight_streaming) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GATE] stage=%zu block=%zu lane=GPU "
                    "splits=%d status=REJECT "
                    "reason=MULTI_SPLIT_UNVALIDATED\n",
                    stage_index, block_index, n_splits);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            if (gpu_weight_streaming && n_splits <= 0) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT] stage=%zu block=%zu "
                    "status=REJECT reason=NO_CUDA_SPLITS\n",
                    stage_index, block_index);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
        } else if (ring_lane == 2) {
            if (n_splits != 1) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_PC_PHONE] stage=%zu block=%zu "
                    "event=REJECT reason=UNSUPPORTED_PHONE_SPLIT "
                    "splits=%d\n",
                    stage_index, block_index, n_splits);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            ggml_backend_t target =
                ggml_backend_sched_get_split_backend(sched_use, 0);
            if (split_backend_count != 3 || !ggml_backend_is_meta(target) ||
                    target != ggml_backend_sched_get_backend(sched_use, 0)) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_PC_PHONE] stage=%zu block=%zu "
                    "event=REJECT reason=UNSUPPORTED_PHONE_SPLIT "
                    "backends=%d target=%s\n",
                    stage_index, block_index, split_backend_count,
                    target ? ggml_backend_name(target) : "NULL");
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            const uint64_t mask =
                ggml_backend_sched_get_split_resource_mask(sched_use, 0);
            const uint64_t permitted =
                (UINT64_C(1) << 0) |
                (UINT64_C(1) << (split_backend_count - 1));
            if ((mask & ~permitted) != 0) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_PC_PHONE] stage=%zu block=%zu "
                    "event=REJECT reason=PHONE_RESOURCE_CONFLICT "
                    "mask=0x%016" PRIx64 "\n",
                    stage_index, block_index, mask);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
        }

        const int cpu_backend_index = split_backend_count - 1;
        const uint64_t cpu_bit = UINT64_C(1) << cpu_backend_index;

        // prepare_ubatch() above entered with PREPARE held. Every split below
        // follows CPU -> CUDA -> PREPARE lock order. Stage-0 GPU work may use
        // CPU only for the embedding prefix; later CPU and CUDA Region work can
        // still overlap once that short prefix/staging section has drained.
        if (ring_prepare_gate->owns_lock()) {
            ring_prepare_gate->unlock();
        }

        bool all_splits_synchronized = true;
        for (int split = 0; split < n_splits; ++split) {
            ggml_backend_t target =
                ggml_backend_sched_get_split_backend(sched_use, split);
            const bool target_is_cpu = target == local_cpu;
            const bool target_is_cuda =
                ring_lane == 1 && target == strict_compute_backend;

            const uint64_t prepare_sync_mask =
                gpu_weight_streaming ?
                    ggml_backend_sched_get_split_mutable_sync_mask(
                        sched_use, split) :
                    ggml_backend_sched_get_split_prepare_sync_mask(
                        sched_use, split);
            if ((prepare_sync_mask >> 63) != 0) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_PREPARE_SYNC] stage=%zu block=%zu "
                    "lane=%d split=%d/%d sync_mask=0x%016" PRIx64
                    " status=REJECT reason=UNRESOLVED_SOURCE_BACKEND\n",
                    stage_index, block_index, ring_lane, split, n_splits,
                    prepare_sync_mask);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
            const bool prepare_needs_cpu =
                (prepare_sync_mask & cpu_bit) != 0;

            if (target_is_cpu || prepare_needs_cpu) {
                const int64_t wait_begin_us = ggml_time_us();
                ring_cpu_gate->lock();
                if (ring_cpu_wait_us != nullptr) {
                    *ring_cpu_wait_us += ggml_time_us() - wait_begin_us;
                }
            }
            if (target_is_cuda) {
                const int64_t wait_begin_us = ggml_time_us();
                ring_cuda_gate->lock();
                if (ring_cuda_wait_us != nullptr) {
                    *ring_cuda_wait_us += ggml_time_us() - wait_begin_us;
                }
            }

            {
                const int64_t wait_begin_us = ggml_time_us();
                ring_prepare_gate->lock();
                if (ring_prepare_wait_us != nullptr) {
                    *ring_prepare_wait_us += ggml_time_us() - wait_begin_us;
                }
            }

            if (block_index == 0 || prepare_needs_cpu ||
                    gpu_cpu_embedding_prefix) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_PREPARE_SYNC] stage=%zu block=%zu "
                    "lane=%d split=%d/%d backend=%s "
                    "sync_mask=0x%016" PRIx64 " cpu_required=%d "
                    "weight_stream=%d status=READY\n",
                    stage_index, block_index, ring_lane, split, n_splits,
                    target ? ggml_backend_name(target) : "NULL",
                    prepare_sync_mask, prepare_needs_cpu ? 1 : 0,
                    gpu_weight_streaming ? 1 : 0);
            }

            const int64_t split_prepare_begin_us =
                timing != nullptr ? ggml_time_us() : 0;
            ret = ggml_backend_sched_prepare_split(sched_use, split);
            if (timing != nullptr) {
                const int64_t split_prepare_us =
                    ggml_time_us() - split_prepare_begin_us;
                timing->prepare_us += split_prepare_us;
                if (gpu_weight_streaming) {
                    timing->weight_stage_us += split_prepare_us;
                }
                timing->weight_upload_bytes +=
                    ggml_backend_sched_get_last_weight_upload_bytes(
                        sched_use);
                timing->weight_cache_hit_bytes +=
                    ggml_backend_sched_get_last_weight_cache_hit_bytes(
                        sched_use);
            }
            if (ret != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GATE] stage=%zu block=%zu lane=%d "
                    "split=%d/%d event=FAIL reason=SPLIT_PREPARE "
                    "status=%d\n",
                    stage_index, block_index, ring_lane, split, n_splits,
                    (int) ret);
                return nullptr;
            }

            if (gpu_weight_streaming) {
                const size_t upload_bytes =
                    ggml_backend_sched_get_last_weight_upload_bytes(
                        sched_use);
                const size_t hit_bytes =
                    ggml_backend_sched_get_last_weight_cache_hit_bytes(
                        sched_use);
                if (block_index == 0 || upload_bytes > 0) {
                    LLAMA_LOG_ERROR(
                        "[REGION_RING_GPU_WEIGHT] event=SPLIT_READY "
                        "stage=%zu block=%zu split=%d/%d "
                        "upload_mib=%.3f cache_hit_mib=%.3f "
                        "cpu_gate=%d status=READY\n",
                        stage_index, block_index, split, n_splits,
                        upload_bytes / 1048576.0,
                        hit_bytes / 1048576.0,
                        prepare_needs_cpu ? 1 : 0);
                }
            }

            ring_prepare_gate->unlock();
            if (!target_is_cpu && ring_cpu_gate->owns_lock()) {
                ring_cpu_gate->unlock();
            }

            const bool main_lane_split =
                ring_lane == 0 ? target_is_cpu :
                ring_lane == 1 ? target_is_cuda :
                true;
            const int64_t lane_begin_us = ggml_time_us();
            if (main_lane_split && lane_compute_begin_us != nullptr &&
                    *lane_compute_begin_us == 0) {
                *lane_compute_begin_us = lane_begin_us;
            }

            LLAMA_LOG_DEBUG(
                "[REGION_RING_GATE] stage=%zu block=%zu lane=%d "
                "split=%d/%d event=COMPUTE_BEGIN backend=%s\n",
                stage_index, block_index, ring_lane, split, n_splits,
                target ? ggml_backend_name(target) : "NULL");

            ret = ggml_backend_sched_compute_split(sched_use, split);
            if (ret == GGML_STATUS_SUCCESS) {
                ggml_backend_synchronize(target);
            } else {
                all_splits_synchronized = false;
            }

            const int64_t lane_end_us = ggml_time_us();
            if (main_lane_split && lane_compute_end_us != nullptr) {
                *lane_compute_end_us = lane_end_us;
            }

            LLAMA_LOG_DEBUG(
                "[REGION_RING_GATE] stage=%zu block=%zu lane=%d "
                "split=%d/%d event=COMPUTE_END backend=%s "
                "dur_ms=%.3f status=%d\n",
                stage_index, block_index, ring_lane, split, n_splits,
                target ? ggml_backend_name(target) : "NULL",
                (lane_end_us - lane_begin_us) / 1000.0, (int) ret);

            if (target_is_cuda && ring_cuda_gate->owns_lock()) {
                ring_cuda_gate->unlock();
            }
            if (target_is_cpu && ring_cpu_gate->owns_lock()) {
                ring_cpu_gate->unlock();
            }

            if (ret != GGML_STATUS_SUCCESS) {
                return nullptr;
            }
        }
        graph_already_synchronized = all_splits_synchronized;
    } else if (pc_phone_overlap) {
        // Compatibility fallback for callers not yet using split backend gates.
        ret = ggml_backend_sched_compute_range(sched_use, 0, n_splits);
    } else {
        ret = graph_compute_range(sched_use, 0, n_splits, block_tokens > 1);
    }
    if (timing != nullptr) {
        timing->compute_range_us += ggml_time_us() - compute_begin_us;
    }
    if (ret != GGML_STATUS_SUCCESS) {
        return nullptr;
    }

    if (stage_output != nullptr) {
        const int64_t sync_begin_us = timing != nullptr ? ggml_time_us() : 0;
        if (!copy_hybrid_stage_output(
                result, sched_use, stage_output, block_tokens, ubatch_id,
                stage_index, block_index, graph_already_synchronized)) {
            if (timing != nullptr) {
                timing->sync_us += ggml_time_us() - sync_begin_us;
            }
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        if (timing != nullptr) {
            timing->sync_us += ggml_time_us() - sync_begin_us;
        }
    } else if (synchronize && !graph_already_synchronized) {
        const int64_t sync_begin_us = timing != nullptr ? ggml_time_us() : 0;
        ggml_backend_sched_synchronize(sched_use);
        if (timing != nullptr) {
            timing->sync_us += ggml_time_us() - sync_begin_us;
        }
    }

    return result;
}

bool llama_context::run_hybrid_wave_probe(
        uint32_t                  probe_tokens,
        int                       probe_layers,
        const llama_hybrid_plan & runtime_plan) {
    if (probe_tokens <= 1 || probe_layers <= 0 || memory == nullptr ||
        model.arch != LLM_ARCH_QWEN2 ||
        model.split_mode() != LLAMA_SPLIT_MODE_TENSOR ||
        runtime_plan.tensor_layers <= 0 ||
        runtime_plan.phone_layers != 0 ||
        runtime_plan.pc_layers != runtime_plan.gpu_pc_layers) {
        return false;
    }

    const int tensor_begin = runtime_plan.pc_layers;
    const int tensor_end_full = tensor_begin + runtime_plan.tensor_layers;
    const int tensor_end_probe = std::min(tensor_end_full, tensor_begin + probe_layers);
    if (tensor_begin < 0 || tensor_end_probe <= tensor_begin) {
        return false;
    }

    ggml_backend_t meta_backend = nullptr;
    for (ggml_backend_t backend : backend_ptrs) {
        if (backend != nullptr && ggml_backend_meta_tensor_profile_reset(backend)) {
            meta_backend = backend;
            break;
        }
    }
    if (meta_backend == nullptr) {
        LLAMA_LOG_WARN("[WAVE_PROBE] skipped: Meta backend not found\n");
        return false;
    }

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;
    const int64_t n_embd = model.hparams.n_embd;
    if (n_embd <= 0 || probe_tokens > cparams.n_ubatch) {
        LLAMA_LOG_WARN(
            "[WAVE_PROBE] skipped: tokens=%u exceeds n_ubatch=%u or invalid n_embd=%" PRId64 "\n",
            probe_tokens, cparams.n_ubatch, n_embd);
        return false;
    }

    // Deterministic non-zero hidden input. Dense GEMM/attention timing is shape
    // dominated, while avoiding an all-zero special case keeps the probe closer
    // to a normal transformer activation.
    std::vector<float> hidden((size_t) n_embd * probe_tokens);
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = (float) ((int) (i % 17) - 8) * 0.001f;
    }

    std::vector<llama_pos> positions(probe_tokens);
    std::vector<int32_t> n_seq_id(probe_tokens, 1);
    std::vector<llama_seq_id> seq_id_data(probe_tokens, 0);
    std::vector<llama_seq_id *> seq_id(probe_tokens);
    std::vector<int8_t> logits(probe_tokens, 0);
    for (uint32_t i = 0; i < probe_tokens; ++i) {
        positions[i] = (llama_pos) i;
        seq_id[i] = &seq_id_data[i];
    }

    llama_batch probe_batch = {
        /*.n_tokens =*/ (int32_t) probe_tokens,
        /*.token    =*/ nullptr,
        /*.embd     =*/ hidden.data(),
        /*.pos      =*/ positions.data(),
        /*.n_seq_id =*/ n_seq_id.data(),
        /*.seq_id   =*/ seq_id.data(),
        /*.logits   =*/ logits.data(),
    };

    llama_batch_allocr probe_balloc(model.hparams.n_pos_per_embd());
    if (!probe_balloc.init(
            probe_batch, model.vocab, memory.get(), (uint32_t) n_embd,
            n_seq_max, false)) {
        LLAMA_LOG_WARN("[WAVE_PROBE] failed to initialize probe batch\n");
        return false;
    }

    llama_memory_context_ptr probe_mctx =
        memory->init_batch(probe_balloc, probe_tokens, false);
    if (!probe_mctx || probe_mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        LLAMA_LOG_WARN("[WAVE_PROBE] failed to initialize scratch memory context\n");
        memory->clear(true);
        return false;
    }

    // init_batch() already split/owns the current ubatch. Pull that exact
    // ubatch from the memory context instead of consuming balloc a second time.
    llama_ubatch probe_ubatch = probe_mctx->get_ubatch();
    if (probe_ubatch.n_tokens != probe_tokens) {
        LLAMA_LOG_WARN(
            "[WAVE_PROBE] unexpected ubatch size=%u expected=%u\n",
            probe_ubatch.n_tokens, probe_tokens);
        memory->clear(true);
        return false;
    }

    llama_hybrid_runtime_stage probe_stage {
        llama_hybrid_runtime_stage_kind::TENSOR,
        tensor_begin,
        tensor_end_probe,
        (int) probe_tokens,
        runtime_plan.tensor_chunk_tokens,
        true,
    };

    ggml_backend_meta_tensor_profile_reset(meta_backend);

    const int32_t n_outputs_saved = n_outputs;
    bool apply_mctx = true;
    ggml_status status = GGML_STATUS_SUCCESS;

    LLAMA_LOG_DEBUG(
        "[WAVE_PROBE_BEGIN] tokens=%u layers=[%d,%d) n_layers=%d XT=%d GA=%d XA=%d R=%.3f\n",
        probe_tokens, tensor_begin, tensor_end_probe, tensor_end_probe - tensor_begin,
        runtime_plan.tensor_chunk_tokens,
        llama_hybrid_runtime_prefill_attn_group_chunks(),
        runtime_plan.tensor_chunk_tokens * llama_hybrid_runtime_prefill_attn_group_chunks(),
        runtime_plan.tensor_pc_ratio);

    llama_hybrid_stage_timing stage_timing {};
    const int64_t wall_begin_us = ggml_time_us();
    llm_graph_result * result = run_hybrid_stage_block(
        probe_ubatch, probe_stage, 0, probe_tokens,
        hidden.data(), nullptr,
        ctx_type_to_graph_type(cparams.ctx_type),
        probe_mctx.get(), sched.get(), gf_res_prev.get(),
        0, 0, 0, llama_hybrid_boundary_action::PASS,
        0, apply_mctx, true, status, &stage_timing);
    const int64_t post_sync_begin_us = ggml_time_us();
    ggml_backend_sched_synchronize(sched.get());
    const int64_t post_sync_us = ggml_time_us() - post_sync_begin_us;
    const int64_t wall_us = ggml_time_us() - wall_begin_us;

    ggml_backend_meta_tensor_profile profile {};
    const bool have_profile =
        ggml_backend_meta_tensor_profile_get(meta_backend, &profile);
    // Do not let probe counters leak into the following real prefill.
    ggml_backend_meta_tensor_profile_reset(meta_backend);

    n_outputs = n_outputs_saved;
    if (auto * kv_mctx = dynamic_cast<llama_kv_cache_context *>(probe_mctx.get())) {
        kv_mctx->clear_stage_range();
    }
    // Probe runs before the first real batch. Clear its scratch KV so the
    // following user prompt starts from exactly the normal empty state.
    memory->clear(true);

    if (result == nullptr || status != GGML_STATUS_SUCCESS || !have_profile) {
        LLAMA_LOG_WARN(
            "[WAVE_PROBE_END] failed status=%d profile=%d wall_ms=%.3f\n",
            (int) status, have_profile ? 1 : 0, wall_us / 1000.0);
        return false;
    }

    const int probe_layer_count = tensor_end_probe - tensor_begin;
    const int attn_group_chunks = llama_hybrid_runtime_prefill_attn_group_chunks();
    const double accounted_probe_us =
        (double) stage_timing.prepare_us +
        (double) stage_timing.compute_range_us +
        (double) stage_timing.sync_us +
        (double) post_sync_us;
    const double unaccounted_probe_us =
        std::max(0.0, (double) wall_us - accounted_probe_us);

    llama_hybrid_wave_calibration calibration {};
    calibration.tokens              = (int) probe_tokens;
    calibration.tensor_layers       = probe_layer_count;
    calibration.tensor_chunk_tokens = runtime_plan.tensor_chunk_tokens;
    calibration.attn_group_chunks   = attn_group_chunks;
    calibration.attn_chunk_tokens   = runtime_plan.tensor_chunk_tokens * attn_group_chunks;
    calibration.tensor_pc_ratio     = runtime_plan.tensor_pc_ratio;

    calibration.wall_ms          = wall_us / 1000.0;
    calibration.prepare_ms       = stage_timing.prepare_us / 1000.0;
    calibration.compute_range_ms = stage_timing.compute_range_us / 1000.0;
    calibration.sync_ms          = stage_timing.sync_us / 1000.0;
    calibration.post_sync_ms     = post_sync_us / 1000.0;

    calibration.wave_layer_start_count = (int) profile.wave_layer_start_count;
    calibration.wave_ii_count          = (int) profile.wave_ii_count;
    calibration.wave_ii_mean_ms        =
        profile.wave_ii_count > 0 ?
            (profile.wave_ii_sum_us / 1000.0) / profile.wave_ii_count : 0.0;
    calibration.wave_ii_median_ms      = profile.wave_ii_median_us / 1000.0;
    calibration.wave_ii_min_ms         = profile.wave_ii_min_us / 1000.0;
    calibration.wave_ii_max_ms         = profile.wave_ii_max_us / 1000.0;
    calibration.wave_fill_ms           = profile.wave_fill_us / 1000.0;
    calibration.wave_drain_ms          = profile.wave_drain_us / 1000.0;
    calibration.wave_span_ms           = profile.wave_span_us / 1000.0;
    calibration.wave_outer_runtime_ms  =
        std::max(0.0, calibration.compute_range_ms - calibration.wave_span_ms);

    calibration.wave_early_ii_count = (int) profile.wave_early_ii_count;
    calibration.wave_early_ii_mean_ms =
        profile.wave_early_ii_count > 0 ?
            (profile.wave_early_ii_sum_us / 1000.0) / profile.wave_early_ii_count : 0.0;
    calibration.wave_early_ii_median_ms =
        profile.wave_early_ii_median_us / 1000.0;
    calibration.wave_late_ii_count = (int) profile.wave_late_ii_count;
    calibration.wave_late_ii_mean_ms =
        profile.wave_late_ii_count > 0 ?
            (profile.wave_late_ii_sum_us / 1000.0) / profile.wave_late_ii_count : 0.0;
    calibration.wave_late_ii_median_ms =
        profile.wave_late_ii_median_us / 1000.0;
    calibration.wave_ii_drift_ratio =
        calibration.wave_early_ii_median_ms > 0.0 ?
            calibration.wave_late_ii_median_ms / calibration.wave_early_ii_median_ms : 0.0;

    calibration.wave_early_layer_count = (int) profile.wave_early_layer_count;
    calibration.wave_early_compute_wall_per_layer_ms =
        profile.wave_early_layer_count > 0 ?
            (profile.wave_early_compute_wall_us / 1000.0) / profile.wave_early_layer_count : 0.0;
    calibration.wave_early_barrier_per_layer_ms =
        profile.wave_early_layer_count > 0 ?
            (profile.wave_early_barrier_us / 1000.0) / profile.wave_early_layer_count : 0.0;
    calibration.wave_late_layer_count = (int) profile.wave_late_layer_count;
    calibration.wave_late_compute_wall_per_layer_ms =
        profile.wave_late_layer_count > 0 ?
            (profile.wave_late_compute_wall_us / 1000.0) / profile.wave_late_layer_count : 0.0;
    calibration.wave_late_barrier_per_layer_ms =
        profile.wave_late_layer_count > 0 ?
            (profile.wave_late_barrier_us / 1000.0) / profile.wave_late_layer_count : 0.0;
    calibration.wave_compute_drift_ratio =
        calibration.wave_early_compute_wall_per_layer_ms > 0.0 ?
            calibration.wave_late_compute_wall_per_layer_ms /
                calibration.wave_early_compute_wall_per_layer_ms : 0.0;
    calibration.wave_barrier_drift_ratio =
        calibration.wave_early_barrier_per_layer_ms > 0.0 ?
            calibration.wave_late_barrier_per_layer_ms /
                calibration.wave_early_barrier_per_layer_ms : 0.0;

    calibration.attn_ms         = profile.attn_us / 1000.0;
    calibration.pc_ffn_ms       = profile.pc_ffn_us / 1000.0;
    calibration.h2d_ms          = profile.h2d_us / 1000.0;
    calibration.phone_ms        = profile.phone_us / 1000.0;
    calibration.d2h_ms          = profile.d2h_us / 1000.0;
    calibration.reduce_ms       = profile.reduce_us / 1000.0;
    calibration.wait_ms         = profile.wait_us / 1000.0;
    calibration.compute_wall_ms = profile.compute_wall_us / 1000.0;
    calibration.reduce_wall_ms  = profile.reduce_wall_us / 1000.0;
    calibration.meta_total_ms   = profile.meta_total_us / 1000.0;
    calibration.other_main_ms   = profile.other_main_us / 1000.0;

    calibration.lane_reuse_wait_ms = profile.lane_reuse_wait_us / 1000.0;
    calibration.barrier_ms         = profile.layer_barrier_wait_us / 1000.0;
    llama_hybrid_runtime_wave_calibration_set(calibration);

    const double operator_compute_ms = calibration.attn_ms + calibration.pc_ffn_ms;
    const double compute_inflation =
        operator_compute_ms > 0.0 ? calibration.compute_wall_ms / operator_compute_ms : 0.0;
    const double meta_inflation =
        calibration.compute_wall_ms > 0.0 ?
            calibration.meta_total_ms / calibration.compute_wall_ms : 0.0;

    LLAMA_LOG_DEBUG(
        "[WAVE_PROBE_SUM] tokens=%u layers=%d XT=%d GA=%d XA=%d R=%.3f "
        "wall_ms=%.3f prepare_ms=%.3f compute_range_ms=%.3f sync_ms=%.3f "
        "post_sync_ms=%.3f unaccounted_ms=%.3f "
        "attn_ms=%.3f pc_ffn_ms=%.3f h2d_ms=%.3f phone_ms=%.3f "
        "d2h_ms=%.3f reduce_ms=%.3f wait_ms=%.3f "
        "compute_wall_ms=%.3f reduce_wall_ms=%.3f meta_total_ms=%.3f other_main_ms=%.3f "
        "lane_reuse_wait_ms=%.3f barrier_ms=%.3f\n",
        probe_tokens, probe_layer_count,
        runtime_plan.tensor_chunk_tokens,
        attn_group_chunks,
        runtime_plan.tensor_chunk_tokens * attn_group_chunks,
        runtime_plan.tensor_pc_ratio,
        calibration.wall_ms,
        calibration.prepare_ms,
        calibration.compute_range_ms,
        calibration.sync_ms,
        calibration.post_sync_ms,
        unaccounted_probe_us / 1000.0,
        calibration.attn_ms,
        calibration.pc_ffn_ms,
        calibration.h2d_ms,
        calibration.phone_ms,
        calibration.d2h_ms,
        calibration.reduce_ms,
        calibration.wait_ms,
        calibration.compute_wall_ms,
        calibration.reduce_wall_ms,
        calibration.meta_total_ms,
        calibration.other_main_ms,
        calibration.lane_reuse_wait_ms,
        calibration.barrier_ms);

    LLAMA_LOG_DEBUG(
        "[WAVE_PROBE_CAL] layers=%d operator_compute_ms=%.3f compute_inflation=%.4f "
        "meta_inflation=%.4f attn_per_layer_ms=%.3f pc_ffn_per_layer_ms=%.3f "
        "meta_per_layer_ms=%.3f\n",
        probe_layer_count,
        operator_compute_ms,
        compute_inflation,
        meta_inflation,
        probe_layer_count > 0 ? calibration.attn_ms / probe_layer_count : 0.0,
        probe_layer_count > 0 ? calibration.pc_ffn_ms / probe_layer_count : 0.0,
        probe_layer_count > 0 ? calibration.meta_total_ms / probe_layer_count : 0.0);

    LLAMA_LOG_DEBUG(
        "[WAVE_PROBE_II] layer_starts=%d ii_count=%d fill_ms=%.3f "
        "ii_mean_ms=%.3f ii_median_ms=%.3f ii_min_ms=%.3f ii_max_ms=%.3f "
        "drain_ms=%.3f span_ms=%.3f outer_runtime_ms=%.3f\n",
        calibration.wave_layer_start_count,
        calibration.wave_ii_count,
        calibration.wave_fill_ms,
        calibration.wave_ii_mean_ms,
        calibration.wave_ii_median_ms,
        calibration.wave_ii_min_ms,
        calibration.wave_ii_max_ms,
        calibration.wave_drain_ms,
        calibration.wave_span_ms,
        calibration.wave_outer_runtime_ms);

    LLAMA_LOG_DEBUG(
        "[WAVE_PROBE_DRIFT] early_ii_count=%d early_ii_mean_ms=%.3f early_ii_median_ms=%.3f "
        "late_ii_count=%d late_ii_mean_ms=%.3f late_ii_median_ms=%.3f ii_drift_ratio=%.4f "
        "early_layers=%d early_compute_per_layer_ms=%.3f early_barrier_per_layer_ms=%.3f "
        "late_layers=%d late_compute_per_layer_ms=%.3f late_barrier_per_layer_ms=%.3f "
        "compute_drift_ratio=%.4f barrier_drift_ratio=%.4f\n",
        calibration.wave_early_ii_count,
        calibration.wave_early_ii_mean_ms,
        calibration.wave_early_ii_median_ms,
        calibration.wave_late_ii_count,
        calibration.wave_late_ii_mean_ms,
        calibration.wave_late_ii_median_ms,
        calibration.wave_ii_drift_ratio,
        calibration.wave_early_layer_count,
        calibration.wave_early_compute_wall_per_layer_ms,
        calibration.wave_early_barrier_per_layer_ms,
        calibration.wave_late_layer_count,
        calibration.wave_late_compute_wall_per_layer_ms,
        calibration.wave_late_barrier_per_layer_ms,
        calibration.wave_compute_drift_ratio,
        calibration.wave_barrier_drift_ratio);

    LLAMA_LOG_DEBUG(
        "[WAVE_PREPARE_MODEL] mode=formal_reuse pending=1 target_tensor_layers=%d\n",
        runtime_plan.tensor_layers);

    return true;
}

llm_graph_result * llama_context::process_ubatch_region_ring(
        const llama_ubatch & ubatch,
        llm_graph_type gtype,
        llama_memory_context_i * mctx,
        ggml_backend_sched_t sched_use,
        llm_graph_result * res_use,
        const std::vector<llama_hybrid_runtime_stage> & stages,
        int ubatch_id,
        bool device_fifo_simulation,
        bool isolate_device_resources,
        bool own_stage_resources,
        bool task_local_graph_outputs,
        bool threaded_execution,
        bool audit_backend_resources,
        bool pc_phone_overlap,
        ggml_status & ret) {
    auto * kv_mctx = dynamic_cast<llama_kv_cache_context *>(mctx);
    if (kv_mctx == nullptr || stages.size() < 6) {
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    // Planner runs ONLY in explicit dry-run mode, before the fixed Ring
    // execution. It reads the measured Hybrid profile, predicts alternative
    // layer/Chunk layouts and never changes stages, KV or scheduler state.
    const char * ring_plan_mode =
        std::getenv("LLAMA_HYBRID_REGION_RING_PLANNER");
    if (pc_phone_overlap && ring_plan_mode != nullptr &&
            std::strcmp(ring_plan_mode, "dry-run") == 0 &&
            ubatch.n_tokens >= 32) {
        const int kv_start =
            ubatch.pos != nullptr && ubatch.n_pos == 1 ?
            std::max(0, (int) ubatch.pos[0]) : 0;
        (void) llama_hybrid_region_ring_planner_dry_run(
            (int) ubatch.n_tokens, kv_start, stages.front().macro_tokens);
    }

    const int32_t saved_outputs = n_outputs;
    const int n_embd = model.hparams.n_embd;
    std::vector<llama_region_ring_region> regions;
    regions.reserve(stages.size());
    for (const auto & stage : stages) {
        const auto device =
            stage.kind == llama_hybrid_runtime_stage_kind::CPU ?
                llama_region_ring_device::CPU :
            stage.kind == llama_hybrid_runtime_stage_kind::GPU ?
                llama_region_ring_device::GPU :
                llama_region_ring_device::PHONE;
        regions.push_back({device, stage.layer_begin, stage.layer_end});
    }
    // Scheduler/Graph Result ownership differs by device in the experimental
    // isolated mode. Only the scheduler state is independent at this stage:
    // ggml_backend_t handles, KV state, n_outputs, and threadpools remain shared.
    // Do not run the three execution paths on separate threads yet.
    std::array<ggml_backend_sched_t, 3> ring_sched = {sched_use, sched_use, sched_use};
    std::array<llm_graph_result *, 3> ring_res = {res_use, res_use, res_use};
    // Shared by Bank resource setup and the per-device GPU worker lambda.
    // This must not be scoped only to isolate_device_resources.
    const bool gpu_weight_streaming =
        isolate_device_resources && pc_phone_overlap &&
        model.hybrid_ring_gpu_weight_streaming;

    // Weight Streaming V1.5 uses the two persistent CUDA banks as a rolling
    // two-Region lookahead window. With the canonical G/C/P Ring, GPU Regions
    // are ordered R0,R1,R2,... regardless of their absolute stage indices.
    // Bank A owns even GPU-region ranks, Bank B owns odd ranks:
    //   INIT: A=R0, B=R1
    //   drain R0 -> A=R2
    //   drain R1 -> B=R3
    // and so on. The old demand-load path remains the correctness fallback.
    std::vector<size_t> gpu_region_order;
    std::vector<int> gpu_region_rank(stages.size(), -1);
    if (gpu_weight_streaming) {
        for (size_t region = 0; region < stages.size(); ++region) {
            if (stages[region].kind != llama_hybrid_runtime_stage_kind::GPU) {
                continue;
            }
            gpu_region_rank[region] = (int) gpu_region_order.size();
            gpu_region_order.push_back(region);
        }
    }
    const int gpu_prefetch_chunk_tokens =
        !gpu_region_order.empty() ?
            stages[gpu_region_order.front()].macro_tokens : 0;
    const int gpu_prefetch_min_tokens =
        gpu_prefetch_chunk_tokens > 0 ?
            2 * gpu_prefetch_chunk_tokens :
            std::numeric_limits<int>::max();
    const bool gpu_weight_prefetch =
        gpu_weight_streaming &&
        gpu_region_order.size() >= 2 &&
        (int) ubatch.n_tokens >= gpu_prefetch_min_tokens;
    std::array<bool, 2> gpu_bank_prefetched = {false, false};

    if (gpu_weight_streaming) {
        for (size_t bank = 1; bank <= 2; ++bank) {
            region_ring_worker_resources[bank].cached_region =
                std::numeric_limits<size_t>::max();
            region_ring_worker_resources[bank].last_use = 0;
        }
        region_ring_gpu_weight_bank_epoch = 0;
        gpu_bank_prefetched.fill(false);
        LLAMA_LOG_ERROR(
            "[REGION_RING_GPU_WEIGHT] phase=BANK_EPOCH_RESET "
            "reason=NEW_UBATCH graph_cache=PRESERVED "
            "region_metadata=CLEARED prefetch_window=%d "
            "prefetch_min_tokens=%d ubatch_tokens=%u status=READY\n",
            gpu_weight_prefetch ? 2 : 0,
            gpu_prefetch_min_tokens,
            ubatch.n_tokens);
    }
    if (isolate_device_resources) {
        // The caller uses the main scheduler to discover and extract logits
        // after this function returns. The last Phone Region must still write
        // the final output into the canonical graph result/scheduler.
        if (!device_fifo_simulation || sched_use != sched.get() ||
            res_use != gf_res_prev.get() ||
            regions.back().device != llama_region_ring_device::PHONE) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_RESOURCE] status=REJECT reason=UNSAFE_FINAL_OUTPUT_OWNER\n");
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (pc_phone_overlap) {
            // This first concurrent mode is supported only for the validated
            // outer topology Meta(CPU,RPC0), CUDA0, CPU. No replacement of
            // GGML model weights and no new Meta instance. CPU Regions use a
            // CPU-only auxiliary Scheduler. GGML requires every Scheduler to
            // end in a CPU backend, so the GPU Scheduler is [CUDA0,CPU], but
            // GPU Region compute is validated to target CUDA0 only before any
            // split is executed. CPU is a scheduler sentinel, not a compute
            // fallback in this validation mode.
            const bool topology_ok = backend_ptrs.size() == 3 &&
                ggml_backend_is_meta(backend_ptrs[0]) &&
                backend_ptrs[1] != nullptr &&
                backend_ptrs[2] == backend_cpu &&
                ggml_backend_dev_type(ggml_backend_get_device(backend_ptrs[1])) ==
                    GGML_BACKEND_DEVICE_TYPE_GPU;
            if (!topology_ok) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_PC_PHONE] event=REJECT reason=OUTER_BACKEND_TOPOLOGY\n");
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }

            if (gpu_weight_prefetch) {
                ggml_backend_dev_t cuda_dev =
                    ggml_backend_get_device(backend_ptrs[1]);
                if (region_ring_cuda_transfer_backend &&
                        ggml_backend_get_device(
                            region_ring_cuda_transfer_backend.get()) != cuda_dev) {
                    region_ring_cuda_transfer_backend.reset();
                }
                if (!region_ring_cuda_transfer_backend) {
                    region_ring_cuda_transfer_backend.reset(
                        ggml_backend_dev_init(cuda_dev, nullptr));
                }
                if (!region_ring_cuda_transfer_backend) {
                    LLAMA_LOG_ERROR(
                        "[REGION_RING_GPU_WEIGHT_TRANSFER] event=INIT "
                        "status=FAIL reason=BACKEND_INIT\n");
                    ret = GGML_STATUS_FAILED;
                    return nullptr;
                }
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT_TRANSFER] event=INIT "
                    "backend=%s device=%s mode=INDEPENDENT_STREAM status=READY\n",
                    ggml_backend_name(region_ring_cuda_transfer_backend.get()),
                    ggml_backend_dev_name(cuda_dev));
            }
        }
        const size_t max_nodes =
            graph_max_nodes(std::min(cparams.n_ctx, cparams.n_ubatch));
        const size_t ring_pc_resource_count =
            gpu_weight_streaming ? 3 : 2;

        for (size_t resource_id = 0;
             resource_id < ring_pc_resource_count; ++resource_id) {
            auto & worker =
                region_ring_worker_resources[resource_id];
            const bool cpu_resource = resource_id == 0;
            const bool gpu_resource = !cpu_resource;

            size_t start = 0;
            size_t count = backend_ptrs.size();
            bool scheduler_offload = cparams.op_offload;
            const char * policy = "CONTEXT_SHARED";
            if (pc_phone_overlap) {
                if (cpu_resource) {
                    start = 2; // validated outer topology: [Meta, CUDA0, CPU]
                    count = 1;
                    scheduler_offload = false;
                    policy = "CPU_ONLY";
                } else {
                    start = 1;
                    count = 2;
                    scheduler_offload =
                        gpu_weight_streaming ?
                            false : cparams.op_offload;
                    policy = gpu_weight_streaming ?
                        "CUDA_WEIGHT_BANK" :
                        "CUDA_ONLY_VALIDATED";
                }
            }

            bool matches = worker.sched &&
                ggml_backend_sched_get_n_backends(
                    worker.sched.get()) == (int) count;
            if (matches) {
                for (size_t i = 0; i < count; ++i) {
                    if (ggml_backend_sched_get_backend(
                            worker.sched.get(), (int) i) !=
                            backend_ptrs[start + i]) {
                        matches = false;
                        break;
                    }
                }
            }
            if (worker.sched && !matches) {
                ggml_backend_sched_synchronize(
                    worker.sched.get());
                worker.res.reset();
                worker.sched.reset();
                worker.cached_region =
                    std::numeric_limits<size_t>::max();
                worker.last_use = 0;
            }
            if (!worker.sched) {
                worker.sched.reset(ggml_backend_sched_new(
                    backend_ptrs.data() + start,
                    backend_buft.data() + start,
                    (int) count, max_nodes, false,
                    scheduler_offload));
                if (!worker.sched) {
                    LLAMA_LOG_ERROR(
                        "[REGION_RING_RESOURCE] resource=%zu "
                        "status=ALLOC_FAILED\n",
                        resource_id);
                    ret = GGML_STATUS_ALLOC_FAILED;
                    return nullptr;
                }
                worker.cached_region =
                    std::numeric_limits<size_t>::max();
                worker.last_use = 0;
            }

            ggml_backend_sched_set_weight_cache(
                worker.sched.get(),
                gpu_resource && gpu_weight_streaming);

            if (!worker.res) {
                worker.res.reset(
                    new llm_graph_result(max_nodes));
            }

            if (cpu_resource) {
                ring_sched[0] = worker.sched.get();
                ring_res[0] = worker.res.get();
            } else if (resource_id == 1) {
                // Bank A is the default pointer for diagnostics. Individual
                // GPU tasks select A/B below.
                ring_sched[1] = worker.sched.get();
                ring_res[1] = worker.res.get();
            }

            if (pc_phone_overlap) {
                const bool cpu_policy_ok =
                    !cpu_resource ||
                    (ggml_backend_sched_get_n_backends(
                         worker.sched.get()) == 1 &&
                     ggml_backend_sched_get_backend(
                         worker.sched.get(), 0) == backend_cpu);
                const bool gpu_policy_ok =
                    !gpu_resource ||
                    (ggml_backend_sched_get_n_backends(
                         worker.sched.get()) == 2 &&
                     ggml_backend_sched_get_backend(
                         worker.sched.get(), 0) ==
                         backend_ptrs[1] &&
                     ggml_backend_sched_get_backend(
                         worker.sched.get(), 1) ==
                         backend_cpu);
                if (!cpu_policy_ok || !gpu_policy_ok) {
                    LLAMA_LOG_ERROR(
                        "[REGION_RING_PC_SCHED] resource=%zu "
                        "policy=%s status=REJECT "
                        "reason=BACKEND_POLICY_MISMATCH\n",
                        resource_id, policy);
                    ret = GGML_STATUS_FAILED;
                    return nullptr;
                }
                LLAMA_LOG_ERROR(
                    "[REGION_RING_PC_SCHED] resource=%zu worker=%s "
                    "policy=%s bank=%s backends=%d "
                    "first=%s last=%s offload=%d "
                    "weight_cache=%d meta_registered=0 status=READY\n",
                    resource_id,
                    cpu_resource ? "CPU" : "GPU",
                    policy,
                    cpu_resource ? "NONE" :
                        (resource_id == 1 ? "A" : "B"),
                    ggml_backend_sched_get_n_backends(
                        worker.sched.get()),
                    ggml_backend_name(
                        ggml_backend_sched_get_backend(
                            worker.sched.get(), 0)),
                    ggml_backend_name(
                        ggml_backend_sched_get_backend(
                            worker.sched.get(),
                            (int) count - 1)),
                    scheduler_offload ? 1 : 0,
                    ggml_backend_sched_get_weight_cache(
                        worker.sched.get()) ? 1 : 0);
            }
        }

        if (gpu_weight_streaming) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_GPU_WEIGHT] phase=BANK_INIT banks=2 "
                "canonical=CPU destination=CUDA "
                "eviction=%s load=%s cache=PERSISTENT "
                "prefetch_min_tokens=%d demand_fallback=%d status=READY\n",
                gpu_weight_prefetch ? "NONE" : "LRU",
                gpu_weight_prefetch ? "LOOKAHEAD_2" : "DEMAND",
                gpu_prefetch_min_tokens,
                gpu_weight_prefetch ? 0 : 1);
        }

        if (ring_sched[0] == ring_sched[1] ||
            ring_sched[0] == ring_sched[2] ||
            ring_sched[1] == ring_sched[2] ||
            ring_res[0] == ring_res[1] ||
            ring_res[0] == ring_res[2] ||
            ring_res[1] == ring_res[2] ||
            (gpu_weight_streaming &&
             (region_ring_worker_resources[2].sched.get() ==
                  ring_sched[0] ||
              region_ring_worker_resources[2].sched.get() ==
                  ring_sched[1] ||
              region_ring_worker_resources[2].sched.get() ==
                  ring_sched[2] ||
              region_ring_worker_resources[2].res.get() ==
                  ring_res[0] ||
              region_ring_worker_resources[2].res.get() ==
                  ring_res[1] ||
              region_ring_worker_resources[2].res.get() ==
                  ring_res[2]))) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_RESOURCE] status=REJECT reason=RESOURCE_ALIAS\n");
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        LLAMA_LOG_ERROR(
            "[REGION_RING_RESOURCE] cpu_sched=%p cpu_res=%p "
            "gpu_sched=%p gpu_res=%p gpu_bank_b_sched=%p "
            "gpu_bank_b_res=%p phone_sched=%p phone_res=%p "
            "backend=SHARED kv=SHARED threads=1 "
            "weight_stream=%d status=ISOLATED_SCHED_ONLY\n",
            (void *) ring_sched[0], (void *) ring_res[0],
            (void *) ring_sched[1], (void *) ring_res[1],
            gpu_weight_streaming ?
                (void *) region_ring_worker_resources[2].sched.get() :
                nullptr,
            gpu_weight_streaming ?
                (void *) region_ring_worker_resources[2].res.get() :
                nullptr,
            (void *) ring_sched[2], (void *) ring_res[2],
            gpu_weight_streaming ? 1 : 0);
    }

    if (own_stage_resources && !isolate_device_resources) {
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (task_local_graph_outputs && !own_stage_resources) {
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    // The worker variant requires all earlier ownership protections.
    // Threaded overlap uses the short PREPARE gate plus per-backend execution
    // gates; non-overlap ownership still uses the legacy broad backend gate.
    if (threaded_execution &&
        (!device_fifo_simulation || !isolate_device_resources ||
         !own_stage_resources || !task_local_graph_outputs)) {
        LLAMA_LOG_ERROR(
            "[REGION_RING_WORKERS] status=REJECT reason=OWNERSHIP_GUARDS_MISSING\n");
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (audit_backend_resources && !threaded_execution) {
        LLAMA_LOG_ERROR(
            "[REGION_RING_BACKEND_OWNERSHIP] status=REJECT reason=REQUIRES_WORKERS\n");
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (pc_phone_overlap &&
        (!threaded_execution || !audit_backend_resources ||
         cparams.cb_eval != nullptr || cparams.pipeline_parallel)) {
        LLAMA_LOG_ERROR(
            "[REGION_RING_PC_PHONE] event=REJECT "
            "reason=UNSAFE_WORKER_EVAL_OR_PIPELINE "
            "threaded=%d audit=%d eval_cb=%d pipeline_parallel=%d\n",
            threaded_execution ? 1 : 0,
            audit_backend_resources ? 1 : 0,
            cparams.cb_eval != nullptr ? 1 : 0,
            cparams.pipeline_parallel ? 1 : 0);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }
    if (task_local_graph_outputs) {
        // CPU auxiliary Scheduler is [CPU]. GPU Scheduler must be
        // [CUDA0,CPU] because GGML requires CPU as the final backend, but GPU
        // Region compute is rejected before execution if any split targets
        // CPU. Graph mutation is protected by PREPARE; CPU/CUDA compute use
        // independent execution gates.
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t backend = backend_ptrs[i];
            ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            const int dev_type = dev ?
                (int) ggml_backend_dev_type(dev) : -1;
            LLAMA_LOG_ERROR(
                "[REGION_RING_BACKEND] index=%zu handle=%p name=%s "
                "device_type=%d schedulers=3 state=SHARED\n",
                i, (void *) backend, ggml_backend_name(backend), dev_type);
        }
        LLAMA_LOG_ERROR(
            "[REGION_RING_BACKEND_SUM] handles=%zu schedulers=3 "
            "backend=%s cpu_threadpool=%s meta_rpc=SHARED "
            "graph_parallelism=%s status=AUDITED\n",
            backend_ptrs.size(),
            pc_phone_overlap ? "PC_CPU_CUDA_AND_META" : "SHARED_GATED",
            pc_phone_overlap ? "ONCE_BEFORE_WORKERS" : "SHARED",
            pc_phone_overlap ? "PC_PHONE_ENABLED" : "DISABLED");
        LLAMA_LOG_ERROR(
            "[REGION_RING_TASK_STATE] outputs=TASK_LOCAL "
            "context_outputs=READ_ONLY kv_views=3 "
            "graph_threads=%s status=READY\n",
            pc_phone_overlap ? "PC_PLUS_PHONE" : "1");
    }
    if (audit_backend_resources) {
        // A Meta backend constructs its own CPU and RPC backend instances.
        // The PC's standalone CPU Backend is a separate handle, even though
        // both run on the same host. No new backend is allocated here.
        for (size_t outer = 0; outer < backend_ptrs.size(); ++outer) {
            ggml_backend_t meta = backend_ptrs[outer];
            if (!ggml_backend_is_meta(meta)) {
                continue;
            }
            for (size_t inner = 0; inner < ggml_backend_meta_n_backends(meta); ++inner) {
                ggml_backend_t backend = ggml_backend_meta_simple_backend(meta, inner);
                bool alias = false;
                for (size_t other = 0; other < backend_ptrs.size(); ++other) {
                    if (backend == backend_ptrs[other]) {
                        alias = true;
                    }
                }
                ggml_backend_dev_t dev = ggml_backend_get_device(backend);
                LLAMA_LOG_ERROR(
                    "[REGION_RING_META_CHILD] outer=%zu inner=%zu handle=%p "
                    "name=%s device_type=%d aliases_outer=%d status=OBSERVED\n",
                    outer, inner, (void *) backend, ggml_backend_name(backend),
                    dev ? (int) ggml_backend_dev_type(dev) : -1, alias ? 1 : 0);
            }
        }
        LLAMA_LOG_ERROR(
            "[REGION_RING_BACKEND_OWNERSHIP] mode=%s "
            "split_inputs=INCLUDED meta_inner=OBSERVED "
            "graph_gate=%s compute_overlap=%s status=READY\n",
            pc_phone_overlap ? "PC_PHONE_SPLIT" : "AUDIT_ONLY",
            pc_phone_overlap ? "PREPARE_PLUS_CPU_CUDA" : "CONTEXT_SHARED",
            pc_phone_overlap ? "MEASURED" : "0");
    }
    const int chunk_tokens = stages.front().macro_tokens;
    std::vector<int> region_chunk_tokens;
    region_chunk_tokens.reserve(stages.size());
    int runtime_xc = 0;
    int runtime_xg = 0;
    int runtime_xp = 0;
    for (const auto & stage : stages) {
        region_chunk_tokens.push_back(stage.macro_tokens);
        int * slot =
            stage.kind == llama_hybrid_runtime_stage_kind::CPU ? &runtime_xc :
            stage.kind == llama_hybrid_runtime_stage_kind::GPU ? &runtime_xg :
                                                                  &runtime_xp;
        if (*slot == 0) {
            *slot = stage.macro_tokens;
        } else if (*slot != stage.macro_tokens) {
            throw std::runtime_error(
                "region ring requires one chunk size per device");
        }
    }
    // Production Region Ring keeps one full-ubatch activation buffer on each
    // boundary. slot_pool_size=0 disables finite producer credits completely.
    const size_t slot_pool_size = 0;
    llama_region_ring_ready ready(
        regions, (int) ubatch.n_tokens,
        region_chunk_tokens, slot_pool_size);
    LLAMA_LOG_ERROR(
        "[REGION_RING_READY_V2] regions=%zu tasks=%zu "
        "XC=%d XG=%d XP=%d dependency=TOKEN_INTERVAL "
        "kv_order=PER_REGION_STRICT activation=FULL_BATCH_PER_EDGE "
        "status=READY\n",
        regions.size(), ready.total_tasks(),
        runtime_xc, runtime_xg, runtime_xp);

    // With no bounded slot pool, each boundary owns storage for the complete
    // ubatch. Producer and consumer tasks address disjoint token intervals
    // directly by task.token_begin.
    const size_t host_tokens_per_edge = slot_pool_size > 0 ?
        slot_pool_size * (size_t) chunk_tokens : (size_t) ubatch.n_tokens;
    std::vector<std::vector<float>> outputs(ready.n_regions() - 1);
    for (auto & hidden : outputs) {
        hidden.resize((size_t) n_embd * host_tokens_per_edge);
    }

    bool apply_mctx = true;
    std::array<std::unique_ptr<llama_kv_cache_context>, 3> worker_kv_views;
    if (own_stage_resources) {
        // Exactly one ubatch assignment against the actual KV cache.
        // All later Region builders receive a private stage-range view;
        // calling apply() on a worker view is forbidden by the KV API.
        if (!kv_mctx->apply()) {
            LLAMA_LOG_ERROR("[REGION_RING_OWNERSHIP] status=FAIL reason=KV_APPLY\n");
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        apply_mctx = false;
        for (size_t dev = 0; dev < worker_kv_views.size(); ++dev) {
            worker_kv_views[dev] = kv_mctx->make_stage_view();
            if (!worker_kv_views[dev]) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_OWNERSHIP] status=FAIL reason=KV_STAGE_VIEW device=%zu\n",
                    dev);
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
        }
        LLAMA_LOG_ERROR(
            "[REGION_RING_OWNERSHIP] activation_buffer=FULL_BATCH_PER_EDGE slot_limit=NONE legacy_slot_pool=%zu "
            "slots_per_edge=%zu kv_views=3 kv_storage=SHARED backend=%s "
            "graph_threads=%s status=READY\n", slot_pool_size, slot_pool_size,
            pc_phone_overlap ? "PREPARE_PLUS_CPU_CUDA_META" : "SHARED_GATED",
            pc_phone_overlap ? "PC_PLUS_PHONE" : "1");
    }
    llm_graph_result * result = nullptr;
    ret = GGML_STATUS_SUCCESS;
    size_t executed = 0;
    size_t next_device = 0;
    const int64_t begin_us = ggml_time_us();

    // GPU-first Ring precomputes the complete token embedding once. This keeps
    // the Q4_K GET_ROWS CPU requirement outside the per-chunk GPU worker and
    // makes stage 0 itself a normal hidden-input CUDA Region.
    std::vector<float> ring_input_hidden;
    bool ring_embedding_prelude = false;
    int64_t ring_embedding_prelude_us = 0;
    const bool gpu_first_ring =
        !stages.empty() &&
        stages.front().kind == llama_hybrid_runtime_stage_kind::GPU &&
        stages.front().layer_begin == 0;
    const bool embedding_prelude_arch =
        model.arch == LLM_ARCH_QWEN2 ||
        model.arch == LLM_ARCH_QWEN3MOE;
    if (gpu_first_ring && embedding_prelude_arch &&
            own_stage_resources && isolate_device_resources &&
            ubatch.token != nullptr) {
        if (!worker_kv_views[0] ||
                !worker_kv_views[0]->set_stage_range(
                    ubatch_id, 0, ubatch.n_tokens)) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_EMBED_PRELUDE] phase=BEGIN status=FAIL "
                "reason=KV_STAGE_RANGE\n");
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        const int64_t prelude_begin_us = ggml_time_us();
        LLAMA_LOG_ERROR(
            "[REGION_RING_EMBED_PRELUDE] phase=BEGIN tokens=%u hidden=%d "
            "backend=CPU_FULL_BATCH status=RUNNING\n",
            ubatch.n_tokens, n_embd);

        ggml_status prelude_status = GGML_STATUS_SUCCESS;
        llm_graph_result * prelude_res = prepare_ubatch(
            ring_res[0], ring_sched[0], ubatch, gtype,
            worker_kv_views[0].get(), prelude_status, false,
            &stages.front(), 0, false, true);
        if (prelude_res == nullptr ||
                prelude_status != GGML_STATUS_SUCCESS) {
            worker_kv_views[0]->clear_stage_range();
            LLAMA_LOG_ERROR(
                "[REGION_RING_EMBED_PRELUDE] phase=PREPARE status=FAIL "
                "reason=GRAPH_PREPARE code=%d\n",
                (int) prelude_status);
            ret = prelude_status == GGML_STATUS_SUCCESS ?
                GGML_STATUS_FAILED : prelude_status;
            return nullptr;
        }

        const int prelude_splits =
            ggml_backend_sched_get_n_splits(ring_sched[0]);
        const bool cpu_only =
            prelude_splits == 1 &&
            ggml_backend_sched_get_n_backends(ring_sched[0]) == 1 &&
            ggml_backend_sched_get_split_backend(ring_sched[0], 0) ==
                backend_cpu;
        if (!cpu_only) {
            worker_kv_views[0]->clear_stage_range();
            LLAMA_LOG_ERROR(
                "[REGION_RING_EMBED_PRELUDE] phase=PREPARE splits=%d "
                "status=FAIL reason=NOT_CPU_ONLY\n",
                prelude_splits);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        const int64_t compute_begin_us = ggml_time_us();
        prelude_status = graph_compute_range(
            ring_sched[0], 0, prelude_splits, ubatch.n_tokens > 1);
        const int64_t compute_end_us = ggml_time_us();
        if (prelude_status != GGML_STATUS_SUCCESS) {
            worker_kv_views[0]->clear_stage_range();
            LLAMA_LOG_ERROR(
                "[REGION_RING_EMBED_PRELUDE] phase=COMPUTE status=FAIL "
                "code=%d\n",
                (int) prelude_status);
            ret = prelude_status;
            return nullptr;
        }

        ring_input_hidden.resize(
            (size_t) ubatch.n_tokens * (size_t) n_embd);
        const int64_t copy_begin_us = ggml_time_us();
        if (!copy_hybrid_stage_output(
                prelude_res, ring_sched[0], ring_input_hidden.data(),
                ubatch.n_tokens, ubatch_id, 0, 0, false)) {
            worker_kv_views[0]->clear_stage_range();
            LLAMA_LOG_ERROR(
                "[REGION_RING_EMBED_PRELUDE] phase=COPY status=FAIL "
                "reason=INVALID_EMBEDDING_OUTPUT\n");
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }
        const int64_t copy_end_us = ggml_time_us();
        worker_kv_views[0]->clear_stage_range();

        ring_embedding_prelude = true;
        ring_embedding_prelude_us =
            ggml_time_us() - prelude_begin_us;
        LLAMA_LOG_ERROR(
            "[REGION_RING_EMBED_PRELUDE] phase=END tokens=%u bytes=%zu "
            "compute_ms=%.3f copy_ms=%.3f wall_ms=%.3f "
            "handoff=HOST_FULL_BATCH next=CUDA status=READY\n",
            ubatch.n_tokens,
            ring_input_hidden.size() * sizeof(float),
            (compute_end_us - compute_begin_us) / 1000.0,
            (copy_end_us - copy_begin_us) / 1000.0,
            ring_embedding_prelude_us / 1000.0);
    }

    if (threaded_execution) {
        if (pc_phone_overlap) {
            // graph_compute_range() used to mutate ALL backend thread configs
            // at every stage. That would race Meta CPU execution after we
            // detach Phone. Configure once before starting any host workers.
            const bool batched = ubatch.n_tokens > 1;
            const int n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
            ggml_threadpool_t tp = batched ? threadpool_batch : threadpool;
            if (backend_cpu != nullptr) {
                auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
                auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *)
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
                if (set_threadpool_fn) {
                    set_threadpool_fn(backend_cpu, tp);
                }
            }
            for (const auto & entry : set_n_threads_fns) {
                entry.second(entry.first, n_threads);
            }
            LLAMA_LOG_ERROR(
                "[REGION_RING_PC_PHONE_THREADS] threads=%d batched=%d "
                "backend_config=ONCE_BEFORE_WORKERS status=READY\n",
                n_threads, batched ? 1 : 0);
        }
        LLAMA_LOG_ERROR(
            "[REGION_RING_WORKERS_PLAN] workers=%d fifo=PER_DEVICE "
            "kv_views=PER_DEVICE activation_buffer=FULL_BATCH_PER_EDGE "
            "slot_limit=NONE legacy_slot_pool=%zu backend=%s "
            "graph_overlap=%s input=%s weight_prefetch=%s status=READY\n",
            gpu_weight_prefetch ? 4 : 3,
            slot_pool_size,
            pc_phone_overlap ? "PREPARE_PLUS_CPU_CUDA_META" : "SHARED_GATED",
            pc_phone_overlap ? "MEASURED" : "0",
            ring_embedding_prelude ?
                "CPU_EMBED_FULL_BATCH_THEN_GPU" :
                "INLINE_STAGE0",
            gpu_weight_prefetch ?
                "CPU_DEDICATED_WORKER" : "DISABLED");
        // The device FIFO and host slots are concurrent. Graph/scheduler
        // mutation is serialized briefly by PREPARE; long CPU and CUDA compute
        // use independent backend locks, while Phone Meta/RPC is detached after
        // its input handoff.
        struct ring_worker_state {
            size_t          completed       = 0;
            int64_t         ready_wait_us   = 0;
            int64_t         backend_wait_us = 0;
            int64_t         prepare_wait_us = 0;
            int64_t         cpu_wait_us     = 0;
            int64_t         cuda_wait_us    = 0;
            int64_t         backend_exec_us = 0;
            ggml_status     status          = GGML_STATUS_SUCCESS;
            llm_graph_result * final_result = nullptr;
            uint64_t        backend_mask    = 0;
            uint64_t        compute_mask    = 0;
            size_t          cross_backend_tasks = 0;
            size_t          unknown_backend_tasks = 0;
            int64_t         weight_stage_us = 0;
            size_t          weight_upload_bytes = 0;
            size_t          weight_cache_hit_bytes = 0;
            size_t          weight_bank_hits = 0;
            size_t          weight_bank_misses = 0;
            size_t          weight_bank_evictions = 0;
            int64_t         weight_prefetch_wait_us = 0;
            size_t          weight_prefetch_hits = 0;
            std::vector<std::pair<int64_t, int64_t>> compute_intervals;
        };
        std::array<ring_worker_state, 3> worker_states;
        const std::array<llama_region_ring_device, 3> devices = {
            llama_region_ring_device::CPU,
            llama_region_ring_device::GPU,
            llama_region_ring_device::PHONE,
        };

        struct gpu_weight_bank_runtime {
            size_t loaded_region = std::numeric_limits<size_t>::max();
            bool loading = false;
        };
        struct gpu_weight_prefetch_job {
            size_t region = 0;
            size_t bank_index = 0;
            size_t source_region = std::numeric_limits<size_t>::max();
            const char * reason = "UNKNOWN";
        };
        struct gpu_weight_prefetch_stats {
            size_t started = 0;
            size_t completed = 0;
            size_t failures = 0;
            int64_t wall_us = 0;
            int64_t prepare_us = 0;
            int64_t h2d_us = 0;
            int64_t stage_cpu_copy_us = 0;
            size_t stage_chunks = 0;
            size_t upload_bytes = 0;
            size_t cache_hit_bytes = 0;
        };

        std::array<gpu_weight_bank_runtime, 2> gpu_bank_runtime;
        std::mutex gpu_prefetch_mutex;
        std::condition_variable gpu_prefetch_cv;
        std::condition_variable gpu_bank_cv;
        std::deque<gpu_weight_prefetch_job> gpu_prefetch_jobs;
        bool gpu_prefetch_stop = false;
        bool gpu_prefetch_failed = false;
        gpu_weight_prefetch_stats gpu_prefetch_stats;
        std::unique_ptr<llama_kv_cache_context> gpu_prefetch_kv_view;

        int gpu_prefetch_tokens_max = 0;
        if (gpu_weight_prefetch) {
            for (size_t region : gpu_region_order) {
                gpu_prefetch_tokens_max =
                    std::max(gpu_prefetch_tokens_max,
                             stages[region].macro_tokens);
            }
            gpu_prefetch_kv_view = kv_mctx->make_stage_view();
            if (!gpu_prefetch_kv_view) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT_PREFETCH_WORKER] "
                    "event=INIT status=FAIL reason=KV_STAGE_VIEW\n");
                ret = GGML_STATUS_FAILED;
                return nullptr;
            }
        }
        std::vector<float> gpu_prefetch_hidden(
            (size_t) std::max(0, gpu_prefetch_tokens_max) *
                (size_t) n_embd,
            0.0f);

        // Only the dedicated host worker calls this function. Graph construction
        // holds the short PREPARE gate; weight H2D then runs on a second CUDA
        // backend instance/stream without CPU or CUDA compute gates.
        auto prefetch_gpu_region =
            [&](size_t region, size_t bank_index,
                const char * reason, size_t source_region) -> bool {
                if (!gpu_weight_prefetch || bank_index >= 2 ||
                        region >= stages.size() ||
                        stages[region].kind !=
                            llama_hybrid_runtime_stage_kind::GPU) {
                    return false;
                }

                auto & bank =
                    region_ring_worker_resources[bank_index + 1];
                const auto & prefetch_stage = stages[region];
                const int prefetch_tokens = std::min(
                    (int) ubatch.n_tokens,
                    prefetch_stage.macro_tokens);
                if (prefetch_tokens <= 0 ||
                        gpu_prefetch_hidden.size() <
                            (size_t) prefetch_tokens *
                                (size_t) n_embd) {
                    return false;
                }

                const std::string source_text =
                    source_region ==
                        std::numeric_limits<size_t>::max() ?
                        "INIT" : std::to_string(source_region);
                const int64_t prefetch_begin_us = ggml_time_us();
                ++gpu_prefetch_stats.started;

                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT_PREFETCH] event=START "
                    "worker=CPU_DEDICATED region=%zu bank=%c "
                    "source_region=%s reason=%s tokens=%d status=RUNNING\n",
                    region, bank_index == 0 ? 'A' : 'B',
                    source_text.c_str(), reason, prefetch_tokens);

                auto fail_prefetch =
                    [&](const char * fail_reason,
                        ggml_status status) -> bool {
                        if (gpu_prefetch_kv_view) {
                            gpu_prefetch_kv_view->clear_stage_range();
                        }
                        if (region_ring_cuda_transfer_backend) {
                            ggml_backend_synchronize(
                                region_ring_cuda_transfer_backend.get());
                        }
                        {
                            std::unique_lock<std::mutex> prepare_gate(
                                region_ring_prepare_mutex);
                            bank.res->reset();
                            ggml_backend_sched_reset(bank.sched.get());
                            ggml_backend_sched_set_weight_cache(
                                bank.sched.get(), true);
                        }
                        bank.cached_region =
                            std::numeric_limits<size_t>::max();
                        bank.last_use = 0;
                        ++gpu_prefetch_stats.failures;
                        gpu_prefetch_stats.wall_us +=
                            ggml_time_us() - prefetch_begin_us;
                        LLAMA_LOG_ERROR(
                            "[REGION_RING_GPU_WEIGHT_PREFETCH] event=FAIL "
                            "worker=CPU_DEDICATED region=%zu bank=%c "
                            "source_region=%s reason=%s status=%d "
                            "action=FAIL_PREFETCH\n",
                            region, bank_index == 0 ? 'A' : 'B',
                            source_text.c_str(), fail_reason,
                            (int) status);
                        return false;
                    };

                if (!gpu_prefetch_kv_view ||
                        !gpu_prefetch_kv_view->set_stage_range(
                            ubatch_id, 0, prefetch_tokens)) {
                    return fail_prefetch(
                        "KV_STAGE_RANGE", GGML_STATUS_FAILED);
                }

                llama_ubatch prefetch_ubatch =
                    llama_hybrid_slice_ubatch(
                        ubatch, 0, prefetch_tokens,
                        gpu_prefetch_hidden.data());

                ggml_status prefetch_status = GGML_STATUS_SUCCESS;
                llm_graph_result * prefetch_result = nullptr;
                const int64_t prepare_begin_us = ggml_time_us();
                {
                    std::unique_lock<std::mutex> prepare_gate(
                        region_ring_prepare_mutex);
                    prefetch_result = prepare_ubatch(
                        bank.res.get(), bank.sched.get(),
                        prefetch_ubatch, gtype,
                        gpu_prefetch_kv_view.get(),
                        prefetch_status, false,
                        &prefetch_stage, 0, true, false);
                }
                const int64_t prepare_us =
                    ggml_time_us() - prepare_begin_us;
                if (prefetch_result == nullptr ||
                        prefetch_status != GGML_STATUS_SUCCESS) {
                    return fail_prefetch(
                        "GRAPH_PREPARE", prefetch_status);
                }

                const int n_splits =
                    ggml_backend_sched_get_n_splits(
                        bank.sched.get());
                if (n_splits <= 0) {
                    return fail_prefetch(
                        "NO_SPLITS", GGML_STATUS_FAILED);
                }
                for (int split = 0; split < n_splits; ++split) {
                    if (ggml_backend_sched_get_split_backend(
                            bank.sched.get(), split) !=
                            backend_ptrs[1]) {
                        return fail_prefetch(
                            "NON_CUDA_SPLIT", GGML_STATUS_FAILED);
                    }
                }

                size_t upload_bytes = 0;
                size_t cache_hit_bytes = 0;
                int64_t stage_cpu_copy_us = 0;
                size_t stage_chunks = 0;
                bool stage_pinned = true;
                size_t stage_slot_size = 0;
                if (!region_ring_cuda_transfer_backend) {
                    return fail_prefetch(
                        "TRANSFER_BACKEND", GGML_STATUS_FAILED);
                }

                const int64_t h2d_begin_us = ggml_time_us();
                // One CPU prefetch worker + one transfer stream means both
                // Banks can share one 2x64 MiB pinned staging ring. Resource
                // #1 (Bank A) owns that ring; Bank B only owns its CUDA cache.
                ggml_backend_sched_t staging_owner =
                    region_ring_worker_resources[1].sched.get();
                for (int split = 0; split < n_splits; ++split) {
                    prefetch_status =
                        ggml_backend_sched_prepare_split_weights_staged_async(
                            bank.sched.get(), split,
                            region_ring_cuda_transfer_backend.get(),
                            staging_owner);
                    upload_bytes +=
                        ggml_backend_sched_get_last_weight_upload_bytes(
                            bank.sched.get());
                    cache_hit_bytes +=
                        ggml_backend_sched_get_last_weight_cache_hit_bytes(
                            bank.sched.get());
                    stage_cpu_copy_us +=
                        ggml_backend_sched_get_last_weight_stage_cpu_copy_us(
                            staging_owner);
                    stage_chunks +=
                        ggml_backend_sched_get_last_weight_stage_chunks(
                            staging_owner);
                    stage_slot_size = std::max(
                        stage_slot_size,
                        ggml_backend_sched_get_weight_stage_slot_size(
                            staging_owner));
                    stage_pinned =
                        stage_pinned &&
                        ggml_backend_sched_weight_stage_is_pinned(
                            staging_owner);

                    if (prefetch_status != GGML_STATUS_SUCCESS) {
                        return fail_prefetch(
                            "WEIGHT_STAGE", prefetch_status);
                    }
                }

                ggml_backend_synchronize(
                    region_ring_cuda_transfer_backend.get());
                const int64_t h2d_us =
                    ggml_time_us() - h2d_begin_us;
                gpu_prefetch_kv_view->clear_stage_range();

                // Publication to the GPU worker happens only after the caller
                // transitions the Bank from LOADING to READY under
                // gpu_prefetch_mutex. These writes are therefore visible when
                // that worker wakes on gpu_bank_cv.
                bank.cached_region = region;
                bank.last_use = 0;

                const int64_t prefetch_us =
                    ggml_time_us() - prefetch_begin_us;
                gpu_prefetch_stats.wall_us += prefetch_us;
                gpu_prefetch_stats.prepare_us += prepare_us;
                gpu_prefetch_stats.h2d_us += h2d_us;
                gpu_prefetch_stats.stage_cpu_copy_us +=
                    stage_cpu_copy_us;
                gpu_prefetch_stats.stage_chunks += stage_chunks;
                gpu_prefetch_stats.upload_bytes += upload_bytes;
                gpu_prefetch_stats.cache_hit_bytes += cache_hit_bytes;
                ++gpu_prefetch_stats.completed;

                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT_PREFETCH] event=DONE "
                    "worker=CPU_DEDICATED region=%zu bank=%c "
                    "source_region=%s reason=%s splits=%d "
                    "upload_mib=%.3f cache_hit_mib=%.3f "
                    "stage=%s slot_mib=%.1f chunks=%zu "
                    "cpu_stage_copy_ms=%.3f h2d_pipeline_ms=%.3f "
                    "prepare_ms=%.3f wall_ms=%.3f status=READY\n",
                    region, bank_index == 0 ? 'A' : 'B',
                    source_text.c_str(), reason, n_splits,
                    upload_bytes / 1048576.0,
                    cache_hit_bytes / 1048576.0,
                    stage_pinned ? "PINNED_DOUBLE" : "DIRECT_PAGEABLE",
                    stage_slot_size / 1048576.0,
                    stage_chunks,
                    stage_cpu_copy_us / 1000.0,
                    h2d_us / 1000.0,
                    prepare_us / 1000.0,
                    prefetch_us / 1000.0);
                return true;
            };

        auto enqueue_gpu_prefetch =
            [&](size_t region, size_t bank_index,
                const char * reason, size_t source_region) -> bool {
                if (!gpu_weight_prefetch ||
                        bank_index >= gpu_bank_runtime.size()) {
                    return false;
                }
                {
                    std::lock_guard<std::mutex> lock(
                        gpu_prefetch_mutex);
                    if (gpu_prefetch_stop || gpu_prefetch_failed) {
                        return false;
                    }
                    auto & bank_state =
                        gpu_bank_runtime[bank_index];
                    bank_state.loading = true;
                    bank_state.loaded_region =
                        std::numeric_limits<size_t>::max();
                    gpu_bank_prefetched[bank_index] = false;
                    gpu_prefetch_jobs.push_back({
                        region, bank_index, source_region, reason});
                }
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT_PREFETCH] event=ENQUEUE "
                    "worker=CPU_DEDICATED region=%zu bank=%c "
                    "source_region=%s reason=%s status=QUEUED\n",
                    region, bank_index == 0 ? 'A' : 'B',
                    source_region ==
                        std::numeric_limits<size_t>::max() ?
                        "INIT" : std::to_string(source_region).c_str(),
                    reason);
                gpu_prefetch_cv.notify_one();
                return true;
            };

        auto run_gpu_prefetch_worker = [&]() {
            LLAMA_LOG_ERROR(
                "[REGION_RING_GPU_WEIGHT_PREFETCH_WORKER] "
                "event=START role=CPU_DEDICATED "
                "banks=2 transfer_stream=INDEPENDENT "
                "cuda_gate=NONE prepare_gate=GRAPH_ONLY status=READY\n");
            try {
                while (true) {
                    gpu_weight_prefetch_job job;
                    {
                        std::unique_lock<std::mutex> lock(
                            gpu_prefetch_mutex);
                        gpu_prefetch_cv.wait(
                            lock, [&]() {
                                return gpu_prefetch_stop ||
                                    !gpu_prefetch_jobs.empty();
                            });
                        if (gpu_prefetch_stop) {
                            gpu_prefetch_jobs.clear();
                            break;
                        }
                        job = gpu_prefetch_jobs.front();
                        gpu_prefetch_jobs.pop_front();
                    }

                    bool ok = false;
                    try {
                        ok = prefetch_gpu_region(
                            job.region, job.bank_index,
                            job.reason, job.source_region);
                    } catch (const std::exception & e) {
                        ++gpu_prefetch_stats.failures;
                        LLAMA_LOG_ERROR(
                            "[REGION_RING_GPU_WEIGHT_PREFETCH_WORKER] "
                            "event=JOB_EXCEPTION region=%zu bank=%c "
                            "reason=%s status=FALLBACK\n",
                            job.region,
                            job.bank_index == 0 ? 'A' : 'B',
                            e.what());
                    } catch (...) {
                        ++gpu_prefetch_stats.failures;
                        LLAMA_LOG_ERROR(
                            "[REGION_RING_GPU_WEIGHT_PREFETCH_WORKER] "
                            "event=JOB_EXCEPTION region=%zu bank=%c "
                            "reason=UNKNOWN status=FALLBACK\n",
                            job.region,
                            job.bank_index == 0 ? 'A' : 'B');
                    }

                    {
                        std::lock_guard<std::mutex> lock(
                            gpu_prefetch_mutex);
                        auto & bank_state =
                            gpu_bank_runtime[job.bank_index];
                        bank_state.loading = false;
                        if (ok) {
                            bank_state.loaded_region = job.region;
                            gpu_bank_prefetched[job.bank_index] = true;
                        } else {
                            bank_state.loaded_region =
                                std::numeric_limits<size_t>::max();
                            gpu_prefetch_failed = true;
                            gpu_bank_prefetched[job.bank_index] = false;
                        }
                    }
                    gpu_bank_cv.notify_all();
                }
            } catch (...) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT_PREFETCH_WORKER] "
                    "event=EXCEPTION reason=UNKNOWN status=STOPPED\n");
                std::lock_guard<std::mutex> lock(
                    gpu_prefetch_mutex);
                gpu_prefetch_failed = true;
                for (auto & bank_state : gpu_bank_runtime) {
                    bank_state.loading = false;
                    bank_state.loaded_region =
                        std::numeric_limits<size_t>::max();
                }
            }
            gpu_bank_cv.notify_all();
            LLAMA_LOG_ERROR(
                "[REGION_RING_GPU_WEIGHT_PREFETCH_WORKER] "
                "event=END status=JOINABLE_DONE\n");
        };

        std::thread gpu_prefetch_worker;
        bool gpu_prefetch_worker_started = false;
        bool gpu_prefetch_worker_joined = false;
        if (gpu_weight_prefetch) {
            const size_t init_source =
                std::numeric_limits<size_t>::max();
            (void) enqueue_gpu_prefetch(
                gpu_region_order[0], 0,
                "INITIAL_WINDOW", init_source);
            (void) enqueue_gpu_prefetch(
                gpu_region_order[1], 1,
                "INITIAL_WINDOW", init_source);
            try {
                gpu_prefetch_worker =
                    std::thread(run_gpu_prefetch_worker);
                gpu_prefetch_worker_started = true;
            } catch (const std::exception & e) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT_PREFETCH_WORKER] "
                    "event=LAUNCH_FAIL reason=%s action=FAIL_PREFETCH\n",
                    e.what());
            } catch (...) {
                LLAMA_LOG_ERROR(
                    "[REGION_RING_GPU_WEIGHT_PREFETCH_WORKER] "
                    "event=LAUNCH_FAIL reason=UNKNOWN "
                    "action=FAIL_PREFETCH\n");
            }

            if (!gpu_prefetch_worker_started) {
                std::lock_guard<std::mutex> lock(
                    gpu_prefetch_mutex);
                gpu_prefetch_stop = true;
                gpu_prefetch_failed = true;
                gpu_prefetch_jobs.clear();
                for (auto & bank_state : gpu_bank_runtime) {
                    bank_state.loading = false;
                    bank_state.loaded_region =
                        std::numeric_limits<size_t>::max();
                }
                gpu_bank_cv.notify_all();
            } else {
                // Initial A/B loads complete before GPU work starts. Rolling
                // loads are requested only by the last chunk of the current
                // GPU Region.
                std::unique_lock<std::mutex> lock(
                    gpu_prefetch_mutex);
                gpu_bank_cv.wait(lock, [&]() {
                    return gpu_prefetch_failed ||
                           (!gpu_bank_runtime[0].loading &&
                            !gpu_bank_runtime[1].loading);
                });
            }

            const bool bank_a_ready =
                !gpu_prefetch_failed &&
                gpu_bank_runtime[0].loaded_region ==
                    gpu_region_order[0];
            const bool bank_b_ready =
                !gpu_prefetch_failed &&
                gpu_bank_runtime[1].loaded_region ==
                    gpu_region_order[1];
            LLAMA_LOG_ERROR(
                "[REGION_RING_GPU_WEIGHT_PREFETCH] event=WINDOW_READY "
                "worker=CPU_DEDICATED region0=%zu bankA=%d "
                "region1=%zu bankB=%d window=2 status=%s\n",
                gpu_region_order[0], bank_a_ready ? 1 : 0,
                gpu_region_order[1], bank_b_ready ? 1 : 0,
                bank_a_ready && bank_b_ready ?
                    "READY" : "PARTIAL_FALLBACK");
        }

        auto run_worker = [&](size_t dev) {
            auto & state = worker_states[dev];
            const auto device = devices[dev];
            const char * name = llama_region_ring_device_name(device);
            bool worker_apply_mctx = false; // canonical KV ubatch was applied once above
            std::optional<llama_region_ring_task> active_task;
            LLAMA_LOG_ERROR(
                "[REGION_RING_WORKER] device=%s event=START "
                "scheduler=%p result=%p kv_view=%p\n",
                name, (void *) ring_sched[dev], (void *) ring_res[dev],
                (void *) worker_kv_views[dev].get());
            try {
                while (true) {
                    const int64_t wait_begin_us = ggml_time_us();
                    const auto next = ready.wait_take(device);
                    state.ready_wait_us += ggml_time_us() - wait_begin_us;
                    if (!next) {
                        break; // all done, or cancelled by another worker
                    }

                    const auto task = *next;
                    active_task = task;
                    const bool first = task.region == 0;
                    const bool last = task.region + 1 == stages.size();
                    const auto & stage = stages[task.region];
                    if (slot_pool_size > 0 &&
                        ((!first && task.input_slot < 0) ||
                         (!last && task.output_slot < 0))) {
                        LLAMA_LOG_ERROR(
                            "[REGION_RING_SLOT] event=REJECT region=%zu chunk=%zu "
                            "input_slot=%d output_slot=%d reason=MISSING_LEASE\n",
                            task.region, task.chunk,
                            task.input_slot, task.output_slot);
                        ready.fail(task);
                        state.status = GGML_STATUS_FAILED;
                        active_task.reset();
                        break;
                    }
                    const size_t input_host_token_begin = first ? 0 :
                        slot_pool_size > 0 ?
                            (size_t) task.input_slot * (size_t) chunk_tokens :
                            (size_t) task.token_begin;
                    const size_t output_host_token_begin = last ? 0 :
                        slot_pool_size > 0 ?
                            (size_t) task.output_slot * (size_t) chunk_tokens :
                            (size_t) task.token_begin;
                    float * input = first ?
                        (ring_embedding_prelude ?
                            ring_input_hidden.data() +
                                (size_t) task.token_begin * n_embd :
                            nullptr) :
                        outputs[task.region - 1].data() +
                            input_host_token_begin * n_embd;
                    float * output = last ? nullptr :
                        outputs[task.region].data() +
                            output_host_token_begin * n_embd;

                    int32_t block_outputs = 0;
                    if (last) {
                        for (int i = 0; i < task.n_tokens; ++i) {
                            block_outputs += ubatch.output[task.token_begin + i] != 0;
                        }
                    }

                    LLAMA_LOG_DEBUG(
                        "[REGION_RING_WORKER] device=%s event=CLAIM "
                        "region=%zu chunk=%zu tokens=%d input_slot=%d output_slot=%d\n",
                        name, task.region, task.chunk, task.n_tokens,
                        task.input_slot, task.output_slot);

                    ggml_backend_sched_t task_sched =
                        ring_sched[dev];
                    llm_graph_result * task_res =
                        ring_res[dev];
                    int gpu_bank = -1;
                    bool gpu_bank_hit = false;
                    bool gpu_prefetch_hit = false;
                    size_t evicted_region =
                        std::numeric_limits<size_t>::max();

                    if (dev == 1 && gpu_weight_streaming) {
                        const int rank =
                            task.region < gpu_region_rank.size() ?
                                gpu_region_rank[task.region] : -1;
                        size_t selected_bank =
                            rank >= 0 ?
                                1 + (size_t) (rank & 1) : 1;
                        int64_t gpu_weight_wait_us = 0;

                        if (gpu_weight_prefetch) {
                            // Strict simple contract:
                            //   GPU never replaces a Bank.
                            //   The last chunk of the previous same-parity
                            //   Region tells the CPU prefetch worker what to
                            //   load. GPU only waits until that load is done.
                            const int64_t wait_begin_us = ggml_time_us();
                            std::unique_lock<std::mutex> lock(
                                gpu_prefetch_mutex);
                            auto & bank_state =
                                gpu_bank_runtime[selected_bank - 1];

                            if (bank_state.loading ||
                                    bank_state.loaded_region != task.region) {
                                LLAMA_LOG_ERROR(
                                    "[REGION_RING_GPU_WEIGHT_PREFETCH] "
                                    "event=WAIT region=%zu chunk=%zu "
                                    "bank=%c status=CPU_LOADING\n",
                                    task.region, task.chunk,
                                    selected_bank == 1 ? 'A' : 'B');
                                gpu_bank_cv.wait(lock, [&]() {
                                    return gpu_prefetch_failed ||
                                        (!bank_state.loading &&
                                         bank_state.loaded_region ==
                                            task.region);
                                });
                                gpu_weight_wait_us =
                                    ggml_time_us() - wait_begin_us;
                                state.weight_prefetch_wait_us +=
                                    gpu_weight_wait_us;
                            }

                            if (gpu_prefetch_failed) {
                                throw std::runtime_error(
                                    "GPU weight prefetch worker failed");
                            }

                            auto & selected =
                                region_ring_worker_resources[selected_bank];
                            if (selected.cached_region != task.region) {
                                throw std::runtime_error(
                                    "GPU weight Bank publication mismatch");
                            }

                            gpu_bank_hit = true;
                            gpu_prefetch_hit =
                                task.chunk == 0 &&
                                gpu_bank_prefetched[selected_bank - 1];
                            if (gpu_prefetch_hit) {
                                ++state.weight_prefetch_hits;
                                gpu_bank_prefetched[selected_bank - 1] = false;
                            }
                            ++state.weight_bank_hits;

                            selected.last_use =
                                ++region_ring_gpu_weight_bank_epoch;
                            task_sched = selected.sched.get();
                            task_res = selected.res.get();
                            gpu_bank = (int) selected_bank - 1;
                        } else {
                            // Tiny batches keep the proven demand-load path.
                            for (size_t bank = 1; bank <= 2; ++bank) {
                                auto & candidate =
                                    region_ring_worker_resources[bank];
                                if (candidate.cached_region == task.region) {
                                    selected_bank = bank;
                                    gpu_bank_hit = true;
                                    break;
                                }
                            }

                            auto & selected =
                                region_ring_worker_resources[selected_bank];
                            if (!gpu_bank_hit) {
                                if (selected.cached_region !=
                                        std::numeric_limits<size_t>::max()) {
                                    evicted_region = selected.cached_region;
                                    ggml_backend_sched_synchronize(
                                        selected.sched.get());
                                    selected.res->reset();
                                    ggml_backend_sched_reset(
                                        selected.sched.get());
                                    ggml_backend_sched_set_weight_cache(
                                        selected.sched.get(), true);
                                    ++state.weight_bank_evictions;
                                }
                                selected.cached_region = task.region;
                                ++state.weight_bank_misses;
                            } else {
                                ++state.weight_bank_hits;
                            }

                            selected.last_use =
                                ++region_ring_gpu_weight_bank_epoch;
                            task_sched = selected.sched.get();
                            task_res = selected.res.get();
                            gpu_bank = (int) selected_bank - 1;
                        }

                        const std::string evicted_region_text =
                            evicted_region ==
                                std::numeric_limits<size_t>::max() ?
                                "NONE" : std::to_string(evicted_region);
                        LLAMA_LOG_ERROR(
                            "[REGION_RING_GPU_WEIGHT] event=BANK_SELECT "
                            "region=%zu chunk=%zu bank=%c region_map=%s "
                            "evicted_region=%s weight_wait_ms=%.3f "
                            "status=READY\n",
                            task.region, task.chunk,
                            gpu_bank == 0 ? 'A' : 'B',
                            gpu_prefetch_hit ? "PREFETCH_HIT" :
                                (gpu_bank_hit ? "HIT" : "MISS"),
                            evicted_region_text.c_str(),
                            gpu_weight_wait_us / 1000.0);
                    }

                    ggml_status task_status = GGML_STATUS_SUCCESS;
                    llm_graph_result * task_result = nullptr;
                    bool cancelled_before_compute = false;
                    const int64_t backend_begin_us = ggml_time_us();
                    {
                        std::unique_lock<std::mutex> broad_backend_lock;
                        std::unique_lock<std::mutex> cpu_gate;
                        std::unique_lock<std::mutex> cuda_gate;
                        std::unique_lock<std::mutex> prepare_gate;

                        if (pc_phone_overlap) {
                            // Phase A starts with PREPARE only. CPU/CUDA locks
                            // are associated but deferred; run_hybrid_stage_block
                            // acquires them only after the graph is split and the
                            // precise prepare-sync sources are known.
                            cpu_gate = std::unique_lock<std::mutex>(
                                region_ring_cpu_execution_mutex,
                                std::defer_lock);
                            if (dev == 1) {
                                cuda_gate = std::unique_lock<std::mutex>(
                                    region_ring_cuda_execution_mutex,
                                    std::defer_lock);
                            }

                            const int64_t wait_begin_us = ggml_time_us();
                            prepare_gate = std::unique_lock<std::mutex>(
                                region_ring_prepare_mutex);
                            const int64_t prepare_wait =
                                ggml_time_us() - wait_begin_us;
                            state.prepare_wait_us += prepare_wait;
                            state.backend_wait_us += prepare_wait;
                        } else {
                            broad_backend_lock = std::unique_lock<std::mutex>(
                                region_ring_backend_execution_mutex);
                            state.backend_wait_us +=
                                ggml_time_us() - backend_begin_us;
                        }

                        if (ready.failed()) {
                            cancelled_before_compute = true;
                        } else {
                            const int64_t compute_begin_us = ggml_time_us();
                            int64_t lane_compute_begin_us = 0;
                            int64_t lane_compute_end_us = 0;
                            llama_hybrid_stage_timing task_timing {};
                            task_result = run_hybrid_stage_block(
                                ubatch, stage, task.token_begin, task.n_tokens,
                                input, output, gtype, worker_kv_views[dev].get(),
                                task_sched, task_res,
                                ubatch_id, task.region, task.chunk,
                                llama_hybrid_boundary_action::PASS,
                                block_outputs, worker_apply_mctx, true,
                                task_status, &task_timing, true,
                                pc_phone_overlap,
                                pc_phone_overlap ? &prepare_gate : nullptr,
                                &lane_compute_begin_us, &lane_compute_end_us,
                                pc_phone_overlap && dev == 1 ?
                                    backend_ptrs[1] : nullptr,
                                pc_phone_overlap ? (int) dev : -1,
                                pc_phone_overlap ? &cpu_gate : nullptr,
                                pc_phone_overlap && dev == 1 ?
                                    &cuda_gate : nullptr,
                                pc_phone_overlap ? &state.prepare_wait_us : nullptr,
                                pc_phone_overlap ? &state.cpu_wait_us : nullptr,
                                pc_phone_overlap ? &state.cuda_wait_us : nullptr,
                                dev == 1 && gpu_weight_streaming &&
                                    gpu_bank_hit);
                            const int64_t compute_end_us = ggml_time_us();
                            state.backend_exec_us +=
                                compute_end_us - compute_begin_us;
                            state.weight_stage_us +=
                                task_timing.weight_stage_us;
                            state.weight_upload_bytes +=
                                task_timing.weight_upload_bytes;
                            state.weight_cache_hit_bytes +=
                                task_timing.weight_cache_hit_bytes;
                            if (dev == 1 && gpu_weight_streaming) {
                                LLAMA_LOG_ERROR(
                                    "[REGION_RING_GPU_WEIGHT] "
                                    "event=TASK_DONE region=%zu chunk=%zu "
                                    "bank=%c weight_cache=%s upload_mib=%.3f "
                                    "cache_hit_mib=%.3f bank_weight_mib=%.3f "
                                    "weight_stage_ms=%.3f prepare_ms=%.3f "
                                    "compute_range_ms=%.3f status=%d\n",
                                    task.region, task.chunk,
                                    gpu_bank == 0 ? 'A' : 'B',
                                    gpu_prefetch_hit ?
                                        "PREFETCH_HIT" :
                                    task_timing.weight_upload_bytes > 0 ?
                                        "MISS_LOAD" :
                                    task_timing.weight_cache_hit_bytes > 0 ?
                                        "HIT" : "NO_WEIGHT_INPUT",
                                    task_timing.weight_upload_bytes /
                                        1048576.0,
                                    task_timing.weight_cache_hit_bytes /
                                        1048576.0,
                                    ggml_backend_sched_get_weight_cache_buffer_size(
                                        task_sched) / 1048576.0,
                                    task_timing.weight_stage_us / 1000.0,
                                    task_timing.prepare_us / 1000.0,
                                    task_timing.compute_range_us / 1000.0,
                                    (int) task_status);
                            }
                            if (pc_phone_overlap) {
                                // prepare/cpu/cuda counters are cumulative.
                                // Reconstruct backend_wait from them at summary
                                // time to avoid double-counting per-call deltas.
                                state.backend_wait_us = 0;
                            }
                            if (pc_phone_overlap && task_result != nullptr &&
                                task_status == GGML_STATUS_SUCCESS &&
                                lane_compute_begin_us > 0 &&
                                lane_compute_end_us >= lane_compute_begin_us) {
                                state.compute_intervals.emplace_back(
                                    lane_compute_begin_us, lane_compute_end_us);
                            }
                            if (task_result != nullptr &&
                                task_status == GGML_STATUS_SUCCESS &&
                                audit_backend_resources) {
                                const int n_splits = ggml_backend_sched_get_n_splits(
                                    task_sched);
                                uint64_t task_mask = 0;
                                uint64_t task_compute_mask = 0;
                                for (int split = 0; split < n_splits; ++split) {
                                    const uint64_t local_mask =
                                        ggml_backend_sched_get_split_resource_mask(
                                            task_sched, split);
                                    uint64_t split_mask = local_mask & (UINT64_C(1) << 63);
                                    // Convert per-Scheduler bits into original
                                    // [Meta,CUDA0,CPU] indices for consistent
                                    // A/B audit. CPU Scheduler is [CPU].
                                    // GPU Scheduler is [CUDA0,CPU] by GGML
                                    // contract, with compute validated CUDA-only.
                                    const int n_local = ggml_backend_sched_get_n_backends(task_sched);
                                    for (int b = 0; b < n_local; ++b) {
                                        if ((local_mask & (UINT64_C(1) << b)) == 0) {
                                            continue;
                                        }
                                        ggml_backend_t handle = ggml_backend_sched_get_backend(task_sched, b);
                                        const auto it = std::find(backend_ptrs.begin(), backend_ptrs.end(), handle);
                                        if (it == backend_ptrs.end() ||
                                            (size_t) (it - backend_ptrs.begin()) >= 63) {
                                            split_mask |= UINT64_C(1) << 63;
                                        } else {
                                            split_mask |= UINT64_C(1) << (size_t) (it - backend_ptrs.begin());
                                        }
                                    }
                                    task_mask |= split_mask;
                                    ggml_backend_t target =
                                        ggml_backend_sched_get_split_backend(
                                            task_sched, split);
                                    uint64_t target_mask = 0;
                                    const auto target_it = std::find(
                                        backend_ptrs.begin(), backend_ptrs.end(), target);
                                    if (target_it == backend_ptrs.end() ||
                                            (size_t) (target_it - backend_ptrs.begin()) >= 63) {
                                        target_mask = UINT64_C(1) << 63;
                                    } else {
                                        target_mask = UINT64_C(1) <<
                                            (size_t) (target_it - backend_ptrs.begin());
                                    }
                                    task_compute_mask |= target_mask;
                                    if (task.chunk == 0 ||
                                            (split_mask >> 63) != 0 ||
                                            (target_mask >> 63) != 0) {
                                        LLAMA_LOG_ERROR(
                                            "[REGION_RING_BACKEND_SPLIT] device=%s "
                                            "region=%zu chunk=%zu split=%d/%d "
                                            "compute=%s resource_mask=0x%016" PRIx64
                                            " compute_mask=0x%016" PRIx64
                                            " unresolved=%d scope=OUTER\n",
                                            name, task.region, task.chunk, split,
                                            n_splits, ggml_backend_name(target),
                                            split_mask, target_mask,
                                            (int) ((split_mask | target_mask) >> 63));
                                    }
                                }
                                state.backend_mask |= task_mask;
                                state.compute_mask |= task_compute_mask;
                                const uint64_t expected_mask = dev == 0 ?
                                    UINT64_C(1) << 2 : dev == 1 ?
                                    UINT64_C(1) << 1 : UINT64_C(1);
                                const bool cross =
                                    (task_compute_mask & ~expected_mask) != 0;
                                const bool unresolved =
                                    ((task_mask | task_compute_mask) >> 63) != 0;
                                state.cross_backend_tasks += cross;
                                state.unknown_backend_tasks += unresolved;
                                LLAMA_LOG_ERROR(
                                    "[REGION_RING_BACKEND_TASK] device=%s "
                                    "region=%zu chunk=%zu n_splits=%d "
                                    "resource_mask=0x%016" PRIx64
                                    " compute_mask=0x%016" PRIx64
                                    " cross_backend=%d unresolved=%d "
                                    "gate=%s scope=OUTER status=OBSERVED\n",
                                    name, task.region, task.chunk, n_splits,
                                    task_mask, task_compute_mask,
                                    cross ? 1 : 0, unresolved ? 1 : 0,
                                    pc_phone_overlap ?
                                        "PREPARE_PLUS_PER_BACKEND" :
                                        "CONTEXT_SHARED");
                            }
                        }
                    }
                    // Do not publish a downstream READY while the producer's
                    // graph or transfer is running. run_hybrid_stage_block()
                    // already synchronizes both intermediate hidden copies
                    // and final outputs before returning.
                    if (cancelled_before_compute || task_result == nullptr ||
                        task_status != GGML_STATUS_SUCCESS ||
                        (last && task_result != res_use)) {
                        state.status = task_status == GGML_STATUS_SUCCESS ?
                            GGML_STATUS_FAILED : task_status;
                        LLAMA_LOG_ERROR(
                            "[REGION_RING_WORKER] device=%s event=FAIL "
                            "region=%zu chunk=%zu status=%d\n",
                            name, task.region, task.chunk, (int) state.status);
                        ready.fail(task);
                        active_task.reset();
                        break;
                    }

                    const bool gpu_region_drained =
                        dev == 1 && gpu_weight_prefetch &&
                        task.chunk + 1 ==
                            ready.n_chunks(task.region);
                    ready.complete(task);
                    ++state.completed;
                    if (last) {
                        state.final_result = task_result;
                    }

                    if (gpu_region_drained) {
                        const int rank =
                            task.region < gpu_region_rank.size() ?
                                gpu_region_rank[task.region] : -1;
                        const size_t next_rank =
                            rank >= 0 ?
                                (size_t) rank + 2 :
                                gpu_region_order.size();
                        if (next_rank < gpu_region_order.size()) {
                            const size_t bank_index =
                                (size_t) rank & 1;
                            (void) enqueue_gpu_prefetch(
                                gpu_region_order[next_rank],
                                bank_index,
                                "REGION_DRAIN",
                                task.region);
                        } else {
                            LLAMA_LOG_ERROR(
                                "[REGION_RING_GPU_WEIGHT_PREFETCH] "
                                "event=WINDOW_DRAIN region=%zu "
                                "bank=%c next=NONE status=DONE\n",
                                task.region,
                                rank >= 0 && (rank & 1) ?
                                    'B' : 'A');
                        }
                    }

                    active_task.reset();
                    LLAMA_LOG_DEBUG(
                        "[REGION_RING_WORKER] device=%s event=DONE "
                        "region=%zu chunk=%zu completed=%zu\n",
                        name, task.region, task.chunk, state.completed);
                }
            } catch (const std::exception & e) {
                state.status = GGML_STATUS_FAILED;
                LLAMA_LOG_ERROR(
                    "[REGION_RING_WORKER] device=%s event=EXCEPTION "
                    "region=%zu chunk=%zu reason=%s\n",
                    name, active_task ? active_task->region : stages.size(),
                    active_task ? active_task->chunk : ready.n_chunks(),
                    e.what());
                ready.cancel();
            } catch (...) {
                state.status = GGML_STATUS_FAILED;
                LLAMA_LOG_ERROR(
                    "[REGION_RING_WORKER] device=%s event=EXCEPTION reason=UNKNOWN\n",
                    name);
                ready.cancel();
            }
            LLAMA_LOG_ERROR(
                "[REGION_RING_WORKER_SUM] device=%s completed=%zu "
                "ready_wait_ms=%.3f backend_wait_ms=%.3f "
                "prepare_wait_ms=%.3f cpu_wait_ms=%.3f cuda_wait_ms=%.3f "
                "backend_exec_ms=%.3f resource_mask=0x%016" PRIx64
                " compute_mask=0x%016" PRIx64
                " cross_backend_tasks=%zu unresolved_tasks=%zu "
                "weight_stage_ms=%.3f weight_upload_mib=%.3f "
                "weight_cache_hit_mib=%.3f "
                "bank_hits=%zu bank_misses=%zu bank_evictions=%zu "
                "prefetch_hits=%zu prefetch_wait_ms=%.3f status=%d\n",
                name, state.completed,
                state.ready_wait_us / 1000.0,
                (pc_phone_overlap ?
                    state.prepare_wait_us + state.cpu_wait_us +
                        state.cuda_wait_us :
                    state.backend_wait_us) / 1000.0,
                state.prepare_wait_us / 1000.0,
                state.cpu_wait_us / 1000.0,
                state.cuda_wait_us / 1000.0,
                state.backend_exec_us / 1000.0, state.backend_mask,
                state.compute_mask, state.cross_backend_tasks,
                state.unknown_backend_tasks,
                state.weight_stage_us / 1000.0,
                state.weight_upload_bytes / 1048576.0,
                state.weight_cache_hit_bytes / 1048576.0,
                state.weight_bank_hits,
                state.weight_bank_misses,
                state.weight_bank_evictions,
                state.weight_prefetch_hits,
                state.weight_prefetch_wait_us / 1000.0,
                (int) state.status);
        };

        std::vector<std::thread> workers;
        bool launched_all = false;
        try {
            workers.reserve(devices.size());
            for (size_t dev = 0; dev < devices.size(); ++dev) {
                workers.emplace_back(run_worker, dev);
            }
            launched_all = true;
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_WORKERS] event=LAUNCH_FAIL reason=%s\n", e.what());
            ready.cancel();
        } catch (...) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_WORKERS] event=LAUNCH_FAIL reason=UNKNOWN\n");
            ready.cancel();
        }
        const size_t created = workers.size();
        size_t joined = 0;
        for (auto & worker : workers) {
            if (worker.joinable()) {
                worker.join();
                ++joined;
            }
        }
        if (gpu_prefetch_worker_started) {
            {
                std::lock_guard<std::mutex> lock(
                    gpu_prefetch_mutex);
                gpu_prefetch_stop = true;
            }
            gpu_prefetch_cv.notify_all();
            gpu_bank_cv.notify_all();
            if (gpu_prefetch_worker.joinable()) {
                gpu_prefetch_worker.join();
                gpu_prefetch_worker_joined = true;
            }
        }
        if (gpu_weight_prefetch) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_GPU_WEIGHT_PREFETCH_SUM] "
                "worker=CPU_DEDICATED started=%zu completed=%zu "
                "failures=%zu upload_mib=%.3f cache_hit_mib=%.3f "
                "stage_cpu_copy_ms=%.3f stage_chunks=%zu "
                "prepare_ms=%.3f h2d_pipeline_ms=%.3f "
                "other_ms=%.3f copy_wall_ms=%.3f gpu_wait_ms=%.3f "
                "thread_started=%d thread_joined=%d status=%s\n",
                gpu_prefetch_stats.started,
                gpu_prefetch_stats.completed,
                gpu_prefetch_stats.failures,
                gpu_prefetch_stats.upload_bytes / 1048576.0,
                gpu_prefetch_stats.cache_hit_bytes / 1048576.0,
                gpu_prefetch_stats.stage_cpu_copy_us / 1000.0,
                gpu_prefetch_stats.stage_chunks,
                gpu_prefetch_stats.prepare_us / 1000.0,
                gpu_prefetch_stats.h2d_us / 1000.0,
                std::max<int64_t>(
                    0,
                    gpu_prefetch_stats.wall_us -
                        gpu_prefetch_stats.prepare_us -
                        gpu_prefetch_stats.h2d_us) / 1000.0,
                gpu_prefetch_stats.wall_us / 1000.0,
                worker_states[1].weight_prefetch_wait_us / 1000.0,
                gpu_prefetch_worker_started ? 1 : 0,
                gpu_prefetch_worker_joined ? 1 : 0,
                gpu_prefetch_stats.failures == 0 ?
                    "PASS" : "FALLBACK");
        }

        for (size_t dev = 0; dev < worker_states.size(); ++dev) {
            executed += worker_states[dev].completed;
            if (worker_states[dev].status != GGML_STATUS_SUCCESS &&
                ret == GGML_STATUS_SUCCESS) {
                ret = worker_states[dev].status;
            }
        }
        result = worker_states[(size_t) llama_region_ring_device::PHONE].final_result;
        if (!launched_all || created != devices.size() ||
            joined != devices.size() || ready.failed() || !ready.all_done() ||
            executed != ready.finished() ||
            result == nullptr || result != res_use) {
            if (ret == GGML_STATUS_SUCCESS) {
                ret = GGML_STATUS_FAILED;
            }
        }
        const int64_t pc_phone_overlap_us = pc_phone_overlap ?
            llama_region_ring_interval_overlap_us(
                worker_states[0].compute_intervals, worker_states[2].compute_intervals) +
            llama_region_ring_interval_overlap_us(
                worker_states[1].compute_intervals, worker_states[2].compute_intervals) : 0;
        LLAMA_LOG_ERROR(
            "[REGION_RING_WORKERS_SUM] created=%zu joined=%zu "
            "prefetch_created=%d prefetch_joined=%d "
            "finished=%zu completed=%zu cpu_tasks=%zu gpu_tasks=%zu "
            "phone_tasks=%zu backend_gate=%s "
            "graph_overlap=%s final_owner=%s status=%s\n",
            created, joined,
            gpu_prefetch_worker_started ? 1 : 0,
            gpu_prefetch_worker_joined ? 1 : 0,
            ready.finished(), executed,
            worker_states[0].completed, worker_states[1].completed,
            worker_states[2].completed,
            pc_phone_overlap ? "PREPARE_PLUS_PER_BACKEND" : "CONTEXT_SHARED",
            pc_phone_overlap ? "MEASURED" : "0",
            result == res_use && result != nullptr ? "MAIN" : "INVALID",
            ret == GGML_STATUS_SUCCESS ? "PASS" : "FAIL");
        if (pc_phone_overlap) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_PC_PHONE_SUM] cpu_intervals=%zu gpu_intervals=%zu "
                "phone_intervals=%zu overlap_ms=%.3f "
                "pc_gate=SPLIT cpu_compute=CPU_GATE cuda_compute=CUDA_GATE "
                "phone_compute=DETACHED phone_prepare=PREPARE_PLUS_CONDITIONAL_CPU "
                "slot_pool=%zu status=%s\n",
                worker_states[0].compute_intervals.size(),
                worker_states[1].compute_intervals.size(),
                worker_states[2].compute_intervals.size(),
                pc_phone_overlap_us / 1000.0, slot_pool_size,
                ret == GGML_STATUS_SUCCESS ? "PASS" : "FAIL");
        }
        if (audit_backend_resources) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_BACKEND_OWNERSHIP_SUM] "
                "cpu_resource_mask=0x%016" PRIx64
                " gpu_resource_mask=0x%016" PRIx64
                " phone_resource_mask=0x%016" PRIx64
                " cpu_compute_mask=0x%016" PRIx64
                " gpu_compute_mask=0x%016" PRIx64
                " phone_compute_mask=0x%016" PRIx64
                " cpu_cross=%zu gpu_cross=%zu phone_cross=%zu "
                "cpu_unresolved=%zu gpu_unresolved=%zu phone_unresolved=%zu "
                "compute_overlap=%s status=%s\n",
                worker_states[0].backend_mask,
                worker_states[1].backend_mask,
                worker_states[2].backend_mask,
                worker_states[0].compute_mask,
                worker_states[1].compute_mask,
                worker_states[2].compute_mask,
                worker_states[0].cross_backend_tasks,
                worker_states[1].cross_backend_tasks,
                worker_states[2].cross_backend_tasks,
                worker_states[0].unknown_backend_tasks,
                worker_states[1].unknown_backend_tasks,
                worker_states[2].unknown_backend_tasks,
                pc_phone_overlap ? "MEASURED" : "0",
                ret == GGML_STATUS_SUCCESS ? "AUDITED" : "INCOMPLETE");
        }
    } else {
    while (!ready.all_done() && ret == GGML_STATUS_SUCCESS) {
        std::optional<llama_region_ring_task> next;
        if (device_fifo_simulation) {
            // Step 1: claim from each device-owned FIFO through the new API,
            // but keep execution on this one thread and reuse the proven
            // GGML execution and Phone l_out handoff paths.
            for (size_t pass = 0; pass < 3; ++pass) {
                const size_t dev = (next_device + pass) % 3;
                next = ready.try_take(static_cast<llama_region_ring_device>(dev));
                if (next) {
                    next_device = (dev + 1) % 3;
                    break;
                }
            }
        } else {
            next = ready.take();
        }
        if (!next) {
            LLAMA_LOG_ERROR("[REGION_RING] event=DEADLOCK finished=%zu\n",
                ready.finished());
            ret = GGML_STATUS_FAILED;
            break;
        }
        const auto task = *next;
        const bool first = task.region == 0;
        const bool last = task.region + 1 == stages.size();
        const auto & stage = stages[task.region];
        if (slot_pool_size > 0 &&
            ((!first && task.input_slot < 0) ||
             (!last && task.output_slot < 0))) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_SLOT] event=REJECT region=%zu chunk=%zu "
                "input_slot=%d output_slot=%d reason=MISSING_LEASE\n",
                task.region, task.chunk, task.input_slot, task.output_slot);
            ready.fail(task);
            ret = GGML_STATUS_FAILED;
            break;
        }
        const size_t input_host_token_begin = first ? 0 :
            slot_pool_size > 0 ?
                (size_t) task.input_slot * (size_t) chunk_tokens :
                (size_t) task.token_begin;
        const size_t output_host_token_begin = last ? 0 :
            slot_pool_size > 0 ?
                (size_t) task.output_slot * (size_t) chunk_tokens :
                (size_t) task.token_begin;
        float * input = first ?
            (ring_embedding_prelude ?
                ring_input_hidden.data() +
                    (size_t) task.token_begin * n_embd :
                nullptr) :
            outputs[task.region - 1].data() + input_host_token_begin * n_embd;
        float * output = last ? nullptr :
            outputs[task.region].data() + output_host_token_begin * n_embd;
        int32_t block_outputs = 0;
        if (last) {
            for (int i = 0; i < task.n_tokens; ++i) {
                block_outputs +=
                    ubatch.output[task.token_begin + i] != 0;
            }
        }
        const int64_t task_begin_us = ggml_time_us();
        LLAMA_LOG_DEBUG(
            "[REGION_RING] event=BEGIN device=%s region=%zu "
            "layers=[%d,%d) chunk=%zu tokens=[%d,%d) "
            "cpu_q=%zu gpu_q=%zu phone_q=%zu\n",
            llama_region_ring_device_name(regions[task.region].device),
            task.region, stage.layer_begin, stage.layer_end,
            task.chunk, task.token_begin,
            task.token_begin + task.n_tokens,
            ready.queue_size(llama_region_ring_device::CPU),
            ready.queue_size(llama_region_ring_device::GPU),
            ready.queue_size(llama_region_ring_device::PHONE));

        auto execute_stage = [&]() -> llm_graph_result * {
            llama_memory_context_i * task_mctx = own_stage_resources ?
                worker_kv_views[(size_t) regions[task.region].device].get() : mctx;
            return run_hybrid_stage_block(
                ubatch, stage, task.token_begin, task.n_tokens,
                input, output, gtype, task_mctx,
                ring_sched[(size_t) regions[task.region].device],
                ring_res[(size_t) regions[task.region].device],
                ubatch_id, task.region, task.chunk,
                llama_hybrid_boundary_action::PASS,
                block_outputs, apply_mctx, true, ret, nullptr,
                task_local_graph_outputs);
        };
        if (own_stage_resources) {
            // The three scheduler instances still share ggml_backend_t.
            // Protect n_outputs / backend runtime / Meta RPC access together.
            // This gate is intentionally broad until the shared backends
            // are replaced with thread-safe, independently owned contexts.
            std::lock_guard<std::mutex> lock(region_ring_backend_execution_mutex);
            result = execute_stage();
        } else {
            result = execute_stage();
        }
        if (result == nullptr || ret != GGML_STATUS_SUCCESS ||
            (isolate_device_resources && last && result != res_use)) {
            if (device_fifo_simulation) {
                ready.fail(task);
                if (ret == GGML_STATUS_SUCCESS) {
                    ret = GGML_STATUS_FAILED;
                }
            }
            break;
        }
        ready.complete(task);
        ++executed;
        LLAMA_LOG_DEBUG(
            "[REGION_RING] event=END device=%s region=%zu chunk=%zu "
            "wall_ms=%.3f finished=%zu\n",
            llama_region_ring_device_name(regions[task.region].device),
            task.region, task.chunk,
            (ggml_time_us() - task_begin_us) / 1000.0,
            ready.finished());
    }
    } // old serial/cooperative execution modes
    kv_mctx->clear_stage_range();
    if (task_local_graph_outputs) {
        const bool unchanged = n_outputs == (uint32_t) saved_outputs;
        LLAMA_LOG_ERROR(
            "[REGION_RING_TASK_STATE_SUM] tasks=%zu "
            "task_outputs=EXPLICIT context_outputs_unchanged=%d "
            "final_graph=MAIN backend=%s status=%s\n",
            executed, unchanged ? 1 : 0,
            pc_phone_overlap ? "PC_GATE_PLUS_META" : "SHARED_GATED",
            unchanged ? "PASS" : "FAIL");
        if (!unchanged) {
            ret = GGML_STATUS_FAILED;
        }
    }
    n_outputs = saved_outputs;
    if (own_stage_resources) {
        const size_t live = ready.live_output_slots();
        LLAMA_LOG_ERROR(
            "[REGION_RING_OWNERSHIP_SUM] activation_buffer=FULL_BATCH_PER_EDGE slot_limit=NONE legacy_slot_pool=%zu "
            "live_slots=%zu finished=%zu status=%s\n",
            slot_pool_size, live, ready.finished(),
            live == 0 && ready.all_done() ? "DRAINED" : "INCOMPLETE");
        if (ret == GGML_STATUS_SUCCESS && live != 0) {
            ret = GGML_STATUS_FAILED;
        }
    }
    LLAMA_LOG_ERROR(
        "[REGION_RING_SUM] tokens=%u XC=%d XG=%d XP=%d regions=%zu "
        "first_region_chunks=%zu tasks=%zu executed=%zu complete=%d "
        "wall_ms=%.3f embedding_prelude_ms=%.3f mode=%s status=%d\n",
        ubatch.n_tokens, runtime_xc, runtime_xg, runtime_xp,
        stages.size(), ready.n_chunks(), ready.total_tasks(),
        executed, ready.all_done() ? 1 : 0,
        (ggml_time_us() - begin_us) / 1000.0,
        ring_embedding_prelude_us / 1000.0,
        pc_phone_overlap ? "DEVICE_FIFO_PC_PHONE_OVERLAP" :
        audit_backend_resources ? "DEVICE_FIFO_BACKEND_AUDIT" :
        threaded_execution ? "DEVICE_FIFO_GATED_WORKERS" :
        task_local_graph_outputs ? "DEVICE_FIFO_TASK_STATE_SIMULATION" :
        own_stage_resources ? "DEVICE_FIFO_OWNED_SIMULATION" :
        isolate_device_resources ? "DEVICE_FIFO_ISOLATED_SIMULATION" :
        device_fifo_simulation ? "DEVICE_FIFO_SIMULATION" : "COOPERATIVE_READY_QUEUE",
        (int) ret);
    return ret == GGML_STATUS_SUCCESS && ready.all_done() ?
        result : nullptr;
}

llm_graph_result * llama_context::process_ubatch_staged(
        const llama_ubatch &                          ubatch,
        llm_graph_type                                gtype,
        llama_memory_context_i *                     mctx,
        ggml_backend_sched_t                         sched_use,
        llm_graph_result *                           res_use,
        const std::vector<llama_hybrid_runtime_stage> & stages,
        int                                           ubatch_id,
        ggml_status &                                 ret) {
    auto * kv_mctx = dynamic_cast<llama_kv_cache_context *>(mctx);
    if (kv_mctx == nullptr || stages.size() < 2) {
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    const int32_t n_outputs_outer = n_outputs;
    const int64_t n_embd = model.hparams.n_embd;
    std::vector<float> hidden[2];
    bool apply_mctx = true;
    llm_graph_result * result = nullptr;

    const auto finish = [&](ggml_status status) {
        kv_mctx->clear_stage_range();
        n_outputs = n_outputs_outer;
        ret = status;
    };

    for (size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
        const auto & stage = stages[stage_index];
        const bool first_stage = stage_index == 0;
        const bool last_stage  = stage_index + 1 == stages.size();
        const bool profile_cpu =
            stage.kind == llama_hybrid_runtime_stage_kind::CPU;
        const bool profile_tensor =
            stage.kind == llama_hybrid_runtime_stage_kind::TENSOR;
        llama_hybrid_stage_timing stage_timing {};
        const int64_t stage_wall_begin_us =
            (profile_cpu || profile_tensor) ? ggml_time_us() : 0;

        if (profile_tensor) {
            const int n_backends =
                ggml_backend_sched_get_n_backends(sched_use);
            for (int i = 0; i < n_backends; ++i) {
                ggml_backend_meta_tensor_profile_reset(
                    ggml_backend_sched_get_backend(sched_use, i));
            }
        }

        std::vector<llama_hybrid_boundary_block> blocks;
        if (first_stage) {
            for (int token_begin = 0; token_begin < (int) ubatch.n_tokens; token_begin += stage.macro_tokens) {
                llama_hybrid_boundary_block block;
                block.output.token_begin = token_begin;
                block.output.n_tokens = std::min(stage.macro_tokens, (int) ubatch.n_tokens - token_begin);
                block.inputs.push_back(block.output);
                blocks.push_back(std::move(block));
            }
        } else {
            blocks = llama_hybrid_plan_boundary(
                ubatch.n_tokens, stages[stage_index - 1].macro_tokens, stage.macro_tokens);
        }

        if (blocks.empty()) {
            finish(GGML_STATUS_FAILED);
            return nullptr;
        }

        if (!last_stage) {
            hidden[stage_index & 1].resize((size_t) n_embd * ubatch.n_tokens);
        }

        for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
            const auto & block = blocks[block_index];
            const uint32_t token_begin = block.output.token_begin;
            const uint32_t block_tokens = block.output.n_tokens;
            float * stage_input = first_stage ? nullptr :
                hidden[(stage_index - 1) & 1].data() + (size_t) token_begin * n_embd;
            float * stage_output = last_stage ? nullptr :
                hidden[stage_index & 1].data() + (size_t) token_begin * n_embd;

            int32_t block_outputs = 0;
            if (last_stage) {
                for (uint32_t i = 0; i < block_tokens; ++i) {
                    block_outputs += ubatch.output[token_begin + i] != 0;
                }
            }

            result = run_hybrid_stage_block(
                ubatch, stage, token_begin, block_tokens, stage_input, stage_output, gtype, mctx, sched_use,
                res_use, ubatch_id, stage_index, block_index, block.action, block_outputs, apply_mctx, true, ret,
                (profile_cpu || profile_tensor) ? &stage_timing : nullptr);
            if (result == nullptr || ret != GGML_STATUS_SUCCESS) {
                finish(ret);
                return nullptr;
            }
        }

        if (profile_cpu) {
            const int64_t stage_total_us =
                ggml_time_us() - stage_wall_begin_us;
            const int64_t accounted_us =
                stage_timing.prepare_us +
                stage_timing.compute_range_us +
                stage_timing.sync_us;
            LLAMA_LOG_DEBUG(
                "[CPU_STAGE_TIMING] ub=%d tokens=%u layers=[%d,%d) "
                "n_layers=%d XC=%d total_ms=%.3f prepare_ms=%.3f "
                "compute_ms=%.3f sync_ms=%.3f unaccounted_ms=%.3f "
                "blocks=%zu mode=SERIAL_ACCUMULATE\n",
                ubatch_id, ubatch.n_tokens,
                stage.layer_begin, stage.layer_end,
                stage.layer_end - stage.layer_begin,
                stage.macro_tokens,
                stage_total_us / 1000.0,
                stage_timing.prepare_us / 1000.0,
                stage_timing.compute_range_us / 1000.0,
                stage_timing.sync_us / 1000.0,
                std::max<int64_t>(0, stage_total_us - accounted_us) / 1000.0,
                blocks.size());
        }

        if (profile_tensor) {
            ggml_backend_meta_tensor_profile profile {};
            const int n_backends =
                ggml_backend_sched_get_n_backends(sched_use);
            for (int i = 0; i < n_backends; ++i) {
                ggml_backend_meta_tensor_profile backend_profile {};
                if (!ggml_backend_meta_tensor_profile_get(
                        ggml_backend_sched_get_backend(sched_use, i),
                        &backend_profile)) {
                    continue;
                }
                profile.attn_us   += backend_profile.attn_us;
                profile.pc_ffn_us += backend_profile.pc_ffn_us;
                profile.h2d_us    += backend_profile.h2d_us;
                profile.phone_us  += backend_profile.phone_us;
                profile.d2h_us    += backend_profile.d2h_us;
                profile.reduce_us += backend_profile.reduce_us;
                profile.wait_us   += backend_profile.wait_us;
                profile.return_transfer_count +=
                    backend_profile.return_transfer_count;
                profile.return_payload_bytes +=
                    backend_profile.return_payload_bytes;
                profile.return_request_us +=
                    backend_profile.return_request_us;
                profile.return_ready_wait_us +=
                    backend_profile.return_ready_wait_us;
                profile.return_recv_payload_us +=
                    backend_profile.return_recv_payload_us;
                profile.return_rpc_total_us +=
                    backend_profile.return_rpc_total_us;
                profile.graph_compute_count +=
                    backend_profile.graph_compute_count;
                profile.graph_rebuild_count +=
                    backend_profile.graph_rebuild_count;
                profile.graph_total_us +=
                    backend_profile.graph_total_us;
                profile.graph_rebuild_us +=
                    backend_profile.graph_rebuild_us;
                profile.graph_execute_us +=
                    backend_profile.graph_execute_us;
                profile.graph_other_us +=
                    backend_profile.graph_other_us;
                profile.lane_reuse_wait_count +=
                    backend_profile.lane_reuse_wait_count;
                profile.lane_reuse_wait_us +=
                    backend_profile.lane_reuse_wait_us;
                profile.lane_reuse_wait_max_us = std::max(
                    profile.lane_reuse_wait_max_us,
                    backend_profile.lane_reuse_wait_max_us);
                for (size_t lane = 0; lane < 2; ++lane) {
                    profile.lane_reuse_wait_count_by_lane[lane] +=
                        backend_profile.lane_reuse_wait_count_by_lane[lane];
                    profile.lane_reuse_wait_us_by_lane[lane] +=
                        backend_profile.lane_reuse_wait_us_by_lane[lane];
                }
                if (backend_profile.layer_barrier_wait_count > 0 &&
                    backend_profile.layer_barrier_wait_max_us >=
                        profile.layer_barrier_wait_max_us) {
                    profile.layer_barrier_wait_max_us =
                        backend_profile.layer_barrier_wait_max_us;
                    profile.layer_barrier_wait_max_layer =
                        backend_profile.layer_barrier_wait_max_layer;
                    profile.layer_barrier_wait_max_pending =
                        backend_profile.layer_barrier_wait_max_pending;
                    profile.layer_barrier_wait_max_last_lane =
                        backend_profile.layer_barrier_wait_max_last_lane;
                }
                profile.layer_barrier_wait_count +=
                    backend_profile.layer_barrier_wait_count;
                profile.layer_barrier_wait_us +=
                    backend_profile.layer_barrier_wait_us;
            }

            const int64_t stage_total_us =
                ggml_time_us() - stage_wall_begin_us;
            const int64_t accounted_us =
                stage_timing.prepare_us +
                stage_timing.compute_range_us +
                stage_timing.sync_us;
            if (std::getenv("LLAMA_HYBRID_PREDICT_VALIDATE") != nullptr &&
                    ubatch.n_tokens > 16) {
                llama_hybrid_tensor_compute_prediction validate_prediction;
                if (llama_hybrid_runtime_predict_tensor_compute(
                        (int) ubatch.n_tokens, validate_prediction)) {
                    const double actual_ms = stage_total_us / 1000.0;
                    const double actual_over_pred =
                        validate_prediction.tensor_total_ms > 0.0 ?
                            actual_ms / validate_prediction.tensor_total_ms :
                            0.0;
                    LLAMA_LOG_ERROR(
                        "[TENSOR_PREDICT_VALIDATE] ub=%d mode=SERIAL_ACCUMULATE "
                        "primary=%s tokens=%d T=%d R=%.3f XT=%d chunks=%d "
                        "actual_ms=%.3f pred_ms=%.3f actual_over_pred=%.3f "
                        "prepare_ms=%.3f compute_ms=%.3f sync_ms=%.3f "
                        "attn_misc_pred_ms=%.3f h2d_sum_pred_ms=%.3f "
                        "pc_ffn_sum_pred_ms=%.3f phone_sum_pred_ms=%.3f "
                        "phone_start_pred_ms=%.3f return_sum_pred_ms=%.3f "
                        "h2d_finish_pred_ms=%.3f "
                        "pc_finish_pred_ms=%.3f phone_finish_pred_ms=%.3f "
                        "return_finish_pred_ms=%.3f tail_pred_ms=%.3f "
                        "pipeline_done_pred_ms=%.3f\n",
                        ubatch_id,
                        validate_prediction.tensor_phone_primary ? "PHONE" : "PC",
                        validate_prediction.tokens,
                        validate_prediction.tensor_layers,
                        validate_prediction.tensor_pc_ratio,
                        validate_prediction.tensor_chunk_tokens,
                        validate_prediction.tensor_chunks,
                        actual_ms,
                        validate_prediction.tensor_total_ms,
                        actual_over_pred,
                        stage_timing.prepare_us / 1000.0,
                        stage_timing.compute_range_us / 1000.0,
                        stage_timing.sync_us / 1000.0,
                        validate_prediction.attn_misc_ms,
                        validate_prediction.pipeline_h2d_sum_ms,
                        validate_prediction.pipeline_pc_ffn_sum_ms,
                        validate_prediction.pipeline_phone_sum_ms,
                        validate_prediction.pipeline_phone_start_ms,
                        validate_prediction.pipeline_d2h_sum_ms,
                        validate_prediction.pipeline_h2d_finish_ms,
                        validate_prediction.pipeline_pc_finish_ms,
                        validate_prediction.pipeline_phone_finish_ms,
                        validate_prediction.pipeline_return_finish_ms,
                        validate_prediction.pipeline_reduce_tail_ms,
                        validate_prediction.pipeline_done_ms);
                }
            }
            LLAMA_LOG_DEBUG(
                "[TENSOR_STAGE_TIMING] ub=%d total=%.3f prepare=%.3f "
                "compute_range=%.3f sync=%.3f unaccounted=%.3f "
                "meta_total=%.3f meta_rebuild=%.3f meta_execute=%.3f "
                "meta_other=%.3f meta_calls=%" PRId64
                " meta_rebuilds=%" PRId64
                " blocks=%zu mode=SERIAL_ACCUMULATE\n",
                ubatch_id,
                stage_total_us / 1000.0,
                stage_timing.prepare_us / 1000.0,
                stage_timing.compute_range_us / 1000.0,
                stage_timing.sync_us / 1000.0,
                std::max<int64_t>(0, stage_total_us - accounted_us) / 1000.0,
                profile.graph_total_us / 1000.0,
                profile.graph_rebuild_us / 1000.0,
                profile.graph_execute_us / 1000.0,
                profile.graph_other_us / 1000.0,
                profile.graph_compute_count,
                profile.graph_rebuild_count,
                blocks.size());

            LLAMA_LOG_DEBUG(
                "[TENSOR_BREAKDOWN] ub=%d total=%.3f attn=%.3f "
                "pc_ffn=%.3f h2d=%.3f phone=%.3f d2h=%.3f "
                "reduce=%.3f wait=%.3f ms mode=SERIAL_ACCUMULATE\n",
                ubatch_id, stage_total_us / 1000.0,
                profile.attn_us / 1000.0,
                profile.pc_ffn_us / 1000.0,
                profile.h2d_us / 1000.0,
                profile.phone_us / 1000.0,
                profile.d2h_us / 1000.0,
                profile.reduce_us / 1000.0,
                profile.wait_us / 1000.0);

            LLAMA_LOG_DEBUG(
                "[PREFILL_RETURN_STALL] lane_reuse_count=%" PRId64
                " lane_reuse_ms=%.3f lane_reuse_max_ms=%.3f "
                "lane0_count=%" PRId64 " lane0_ms=%.3f "
                "lane1_count=%" PRId64 " lane1_ms=%.3f "
                "layer_barrier_count=%" PRId64
                " layer_barrier_ms=%.3f layer_barrier_max_ms=%.3f "
                "max_layer=%" PRId64 " max_pending=%" PRId64
                " max_last_lane=%" PRId64
                " mode=SERIAL_ACCUMULATE\n",
                profile.lane_reuse_wait_count,
                profile.lane_reuse_wait_us / 1000.0,
                profile.lane_reuse_wait_max_us / 1000.0,
                profile.lane_reuse_wait_count_by_lane[0],
                profile.lane_reuse_wait_us_by_lane[0] / 1000.0,
                profile.lane_reuse_wait_count_by_lane[1],
                profile.lane_reuse_wait_us_by_lane[1] / 1000.0,
                profile.layer_barrier_wait_count,
                profile.layer_barrier_wait_us / 1000.0,
                profile.layer_barrier_wait_max_us / 1000.0,
                profile.layer_barrier_wait_max_layer,
                profile.layer_barrier_wait_max_pending,
                profile.layer_barrier_wait_max_last_lane);

            if (profile.return_transfer_count > 0) {
                const int64_t exposed_wait_us =
                    profile.lane_reuse_wait_us +
                    profile.layer_barrier_wait_us;
                const int64_t overlap_est_us =
                    std::max<int64_t>(
                        0, profile.return_rpc_total_us -
                            exposed_wait_us);
                const double overlap_ratio =
                    profile.return_rpc_total_us > 0 ?
                        (double) overlap_est_us /
                            (double) profile.return_rpc_total_us :
                        0.0;
                LLAMA_LOG_DEBUG(
                    "[RETURN_CRITICAL_PATH] ub=%d returns=%" PRId64
                    " payload_mib=%.3f request_ms=%.3f ready_wait_ms=%.3f "
                    "recv_payload_ms=%.3f rpc_total_ms=%.3f "
                    "d2h_accounted_ms=%.3f lane_wait_ms=%.3f "
                    "barrier_wait_ms=%.3f exposed_wait_ms=%.3f "
                    "overlap_est_ms=%.3f overlap_ratio=%.3f "
                    "mode=SERIAL_ACCUMULATE\n",
                    ubatch_id,
                    profile.return_transfer_count,
                    profile.return_payload_bytes / 1048576.0,
                    profile.return_request_us / 1000.0,
                    profile.return_ready_wait_us / 1000.0,
                    profile.return_recv_payload_us / 1000.0,
                    profile.return_rpc_total_us / 1000.0,
                    profile.d2h_us / 1000.0,
                    profile.lane_reuse_wait_us / 1000.0,
                    profile.layer_barrier_wait_us / 1000.0,
                    exposed_wait_us / 1000.0,
                    overlap_est_us / 1000.0,
                    overlap_ratio);
            }
        }
    }

    finish(GGML_STATUS_SUCCESS);
    return result;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch &     ubatch,
                                                 llm_graph_type           gtype,
                                                 llama_memory_context_i * mctx,
                                                 ggml_backend_sched_t     sched_use,
                                                 llm_graph_result *       res_use,
                                                 int                      ubatch_id,
                                                 ggml_status &            ret) {
    GGML_UNUSED(ubatch_id);

    const int64_t prepare_begin_us = ggml_time_us();
    llm_graph_result * res = prepare_ubatch(res_use, sched_use, ubatch, gtype, mctx, ret);
    const int64_t prepare_us = ggml_time_us() - prepare_begin_us;

    if (res == nullptr) {
        return nullptr;
    }

    if (ubatch.n_tokens > 1) {
        LLAMA_LOG_DEBUG(
            "[FORMAL_PREPARE] tokens=%u prepare_ms=%.3f\n",
            ubatch.n_tokens, prepare_us / 1000.0);
        llama_hybrid_log_formal_prepare_prediction(
            ubatch.n_tokens, prepare_us / 1000.0);
    }

    const int n_splits = ggml_backend_sched_get_n_splits(sched_use);

    const bool batched = ubatch.n_tokens > 1;
    const bool decode_token =
        ubatch.n_tokens == 1 &&
        gtype != LLM_GRAPH_TYPE_ENCODER &&
        sched_use == sched.get();

    bool decode_profile_this_token = false;
    if (decode_token) {
        const int64_t sample_index =
            ++decode_runtime_profile_count;
        decode_profile_this_token =
            sample_index <= 8 || (sample_index % 16) == 0;

        if (decode_profile_this_token) {
            decode_runtime_profile = {};
            decode_runtime_profile.pending = true;
            decode_runtime_profile.sample_index = sample_index;
            decode_runtime_profile.prepare_us = prepare_us;
            decode_runtime_profile.n_splits = n_splits;

            // The Meta profile is cumulative. Reset it immediately before the
            // real decode graph so the later synchronize() log describes this
            // exact token only.
            const int n_backends =
                ggml_backend_sched_get_n_backends(sched_use);
            for (int ib = 0; ib < n_backends; ++ib) {
                ggml_backend_meta_tensor_profile_reset(
                    ggml_backend_sched_get_backend(sched_use, ib));
            }
        }
    }

    const int64_t compute_begin_us =
        (batched || decode_profile_this_token) ?
            ggml_time_us() : 0;

    if (decode_profile_this_token) {
        // Keep the scheduler semantics identical to compute_range(): prepare
        // and submit every split in order, but do not add any synchronization.
        // The split timings therefore measure scheduler/dispatch wall time,
        // while the actual completion time is captured later in synchronize().
        ret = GGML_STATUS_SUCCESS;
        for (int split_id = 0;
             split_id < n_splits &&
             ret == GGML_STATUS_SUCCESS;
             ++split_id) {
            ggml_backend_t split_backend =
                ggml_backend_sched_get_split_backend(
                    sched_use, split_id);

            const int64_t split_prepare_begin_us =
                ggml_time_us();
            ret = ggml_backend_sched_prepare_split(
                sched_use, split_id);
            const int64_t split_prepare_us =
                ggml_time_us() - split_prepare_begin_us;

            int64_t split_submit_us = 0;
            if (ret == GGML_STATUS_SUCCESS) {
                const int64_t split_submit_begin_us =
                    ggml_time_us();
                ret = ggml_backend_sched_compute_split(
                    sched_use, split_id);
                split_submit_us =
                    ggml_time_us() - split_submit_begin_us;
            }

            decode_runtime_profile.split_prepare_us +=
                split_prepare_us;
            decode_runtime_profile.split_submit_us +=
                split_submit_us;

            ggml_backend_dev_t split_dev =
                split_backend != nullptr ?
                    ggml_backend_get_device(split_backend) : nullptr;
            LLAMA_LOG_DEBUG(
                "[DECODE_SPLIT_SUBMIT] sample=%" PRId64
                " split=%d/%d backend=%s type=%d "
                "prepare_ms=%.3f submit_ms=%.3f status=%d\n",
                decode_runtime_profile.sample_index,
                split_id,
                n_splits,
                split_backend != nullptr ?
                    ggml_backend_name(split_backend) : "(null)",
                split_dev != nullptr ?
                    (int) ggml_backend_dev_type(split_dev) : -1,
                split_prepare_us / 1000.0,
                split_submit_us / 1000.0,
                (int) ret);
        }
    } else {
        ret = graph_compute_range(
            sched_use, 0, n_splits, batched);
    }

    const int64_t compute_us =
        (batched || decode_profile_this_token) ?
            ggml_time_us() - compute_begin_us : 0;

    if (decode_profile_this_token) {
        LLAMA_LOG_DEBUG(
            "[DECODE_DISPATCH] sample=%" PRId64
            " splits=%d prepare_graph_ms=%.3f "
            "split_prepare_ms=%.3f split_submit_ms=%.3f "
            "dispatch_wall_ms=%.3f\n",
            decode_runtime_profile.sample_index,
            n_splits,
            prepare_us / 1000.0,
            decode_runtime_profile.split_prepare_us / 1000.0,
            decode_runtime_profile.split_submit_us / 1000.0,
            compute_us / 1000.0);
    }

    if (batched) {
        LLAMA_LOG_DEBUG(
            "[FORMAL_UBATCH_TIMING] tokens=%u splits=%d prepare_ms=%.3f "
            "compute_ms=%.3f total_ms=%.3f\n",
            ubatch.n_tokens,
            n_splits,
            prepare_us / 1000.0,
            compute_us / 1000.0,
            (prepare_us + compute_us) / 1000.0);
    }

    if (ret != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute full graph, status: %d\n", __func__, ret);

        return nullptr;
    }

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    // eagle3/DFlash: features as encoder input, and non-draft paths fall back to model's input dim
    const int64_t n_embd  = hparams.n_embd_inp_enc();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max,
                      true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status  status;
    const auto * res =
        process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, sched.get(), gf_res_prev.get(), 0, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:
                return 2;
            case GGML_STATUS_ALLOC_FAILED:
                return -2;
            case GGML_STATUS_FAILED:
                return -3;
            case GGML_STATUS_SUCCESS:
                GGML_ABORT("should not happen");
        }
    }

    auto * t_logits  = res->get_logits();
    auto * t_embd    = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();
    auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // extract nextn embeddings (hidden state before the final output norm)
    if (embd_nextn.data && t_h_nextn && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched.get(), t_h_nextn);
        GGML_ASSERT(backend_h != nullptr);

        const uint32_t n_embd = hparams.n_embd_out();
        GGML_ASSERT(n_tokens*n_embd <= (int64_t) embd_nextn.size);
        ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn.data, 0, n_tokens*n_embd*sizeof(float));
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    // MTP hook batches carry both token (next-token id) and embd (h_nextn row),
    // so accept either present rather than requiring exactly one.
    GGML_ASSERT(batch_inp.token || batch_inp.embd);

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    llama_hybrid_plan runtime_plan;
    const bool has_runtime_plan = llama_hybrid_runtime_plan_get(runtime_plan);
    int tensor_phone_first_layer = -1;
    int tensor_phone_last_layer = -1;
    std::vector<uint8_t> tensor_phone_layer_mask(
        size_t(hparams.n_layer()), uint8_t(0));
    std::vector<uint8_t> tensor_c2p_layer_mask(
        size_t(hparams.n_layer()), uint8_t(0));
    std::vector<uint8_t> tensor_p2c_layer_mask(
        size_t(hparams.n_layer()), uint8_t(0));
    bool directional_region_policy = false;

    // The model policy is the execution source of truth. Region Boundary V1
    // may contain many sparse P2C Tensor layers, so publish an exact layer mask
    // rather than widening them into one [first,last) interval.
    for (int il = 0; il < (int) hparams.n_layer(); ++il) {
        const llama_hybrid_layer_mode mode = model.hybrid_layer_mode(il);
        directional_region_policy =
            directional_region_policy ||
            llama_hybrid_layer_mode_directional_tensor(mode);
        if (mode == llama_hybrid_layer_mode::TENSOR_C2P) {
            tensor_c2p_layer_mask[size_t(il)] = 1;
        }
        if (mode == llama_hybrid_layer_mode::TENSOR_P2C) {
            tensor_p2c_layer_mask[size_t(il)] = 1;
        }
        if (!llama_hybrid_layer_mode_tensor_attention_on_phone(mode)) {
            continue;
        }
        tensor_phone_layer_mask[size_t(il)] = 1;
        if (tensor_phone_first_layer < 0) {
            tensor_phone_first_layer = il;
        }
        tensor_phone_last_layer = il + 1;
    }

    for (ggml_backend_t backend : backend_ptrs) {
        if (directional_region_policy) {
            ggml_backend_meta_set_tensor_phone_primary_layer_mask(
                backend,
                tensor_phone_layer_mask.empty() ? nullptr :
                    tensor_phone_layer_mask.data(),
                tensor_phone_layer_mask.size());
            ggml_backend_meta_set_tensor_c2p_layer_mask(
                backend,
                tensor_c2p_layer_mask.empty() ? nullptr :
                    tensor_c2p_layer_mask.data(),
                tensor_c2p_layer_mask.size());
            ggml_backend_meta_set_tensor_p2c_layer_mask(
                backend,
                tensor_p2c_layer_mask.empty() ? nullptr :
                    tensor_p2c_layer_mask.data(),
                tensor_p2c_layer_mask.size());
        } else {
            ggml_backend_meta_set_tensor_phone_primary_layers(
                backend, tensor_phone_first_layer, tensor_phone_last_layer);
        }
    }
    const bool region_ring_v1 = llama_region_ring_enabled();
    const char * ring_exec_env =
        std::getenv("LLAMA_HYBRID_REGION_RING_EXEC");
    const std::string ring_exec = region_ring_v1 ?
        (ring_exec_env != nullptr ? ring_exec_env : "ready") : "";
    if (!model.hybrid_ring_plan_layout.empty() &&
            (!region_ring_v1 || ring_exec != "fifo-pc-phone")) {
        throw std::runtime_error(
            "model-owned Ring Plan requires the same "
            "LLAMA_HYBRID_REGION_RING_V1=1 and "
            "LLAMA_HYBRID_REGION_RING_EXEC=fifo-pc-phone used during "
            "weight placement");
    }
    if (region_ring_v1 && ring_exec != "ready" &&
        ring_exec != "fifo" && ring_exec != "fifo-isolated" &&
        ring_exec != "fifo-owned" && ring_exec != "fifo-state" &&
        ring_exec != "fifo-workers" && ring_exec != "fifo-backend-audit" &&
        ring_exec != "fifo-pc-phone" &&
        ring_exec != "serial" && ring_exec != "full") {
        throw std::runtime_error(
            "LLAMA_HYBRID_REGION_RING_EXEC must be ready, fifo, fifo-isolated, fifo-owned, fifo-state, fifo-workers, fifo-backend-audit, fifo-pc-phone, serial or full");
    }
    int region_ring_chunk_tokens = 128;
    int region_ring_xc = 0;
    int region_ring_xg = 0;
    int region_ring_xp = 0;
    if (region_ring_v1) {
        if (model.hybrid_ring_plan_cpu_chunk_tokens > 0 ||
                model.hybrid_ring_plan_gpu_chunk_tokens > 0 ||
                model.hybrid_ring_plan_phone_chunk_tokens > 0) {
            if (model.hybrid_ring_plan_cpu_chunk_tokens <= 0 ||
                    model.hybrid_ring_plan_gpu_chunk_tokens <= 0 ||
                    model.hybrid_ring_plan_phone_chunk_tokens <= 0) {
                throw std::runtime_error(
                    "Planner Ring per-device chunks are incomplete");
            }
            region_ring_xc =
                model.hybrid_ring_plan_cpu_chunk_tokens;
            region_ring_xg =
                model.hybrid_ring_plan_gpu_chunk_tokens;
            region_ring_xp =
                model.hybrid_ring_plan_phone_chunk_tokens;
        } else {
            if (model.hybrid_ring_plan_chunk_tokens > 0) {
                region_ring_chunk_tokens =
                    model.hybrid_ring_plan_chunk_tokens;
            } else if (const char * env =
                    std::getenv("LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS")) {
                region_ring_chunk_tokens = std::atoi(env);
            }
            if (region_ring_chunk_tokens <= 0) {
                throw std::runtime_error(
                    "region ring chunk tokens must be positive");
            }
            region_ring_xc = region_ring_chunk_tokens;
            region_ring_xg = region_ring_chunk_tokens;
            region_ring_xp = region_ring_chunk_tokens;
        }
    }
    const std::vector<llama_region_ring_region> ring_runtime_regions =
        region_ring_v1 ?
            (model.hybrid_ring_plan_layout.empty() ?
                llama_region_ring_from_env(model.hparams.n_layer()) :
                llama_region_ring_parse(model.hybrid_ring_plan_layout,
                                        model.hparams.n_layer())) :
            std::vector<llama_region_ring_region>{};
    const std::vector<llama_hybrid_runtime_stage> runtime_stages =
        region_ring_v1 ?
            llama_region_ring_runtime_stages(
                ring_runtime_regions,
                region_ring_xc, region_ring_xg, region_ring_xp) :
        has_runtime_plan ? llama_hybrid_build_runtime_stages(runtime_plan) :
                           std::vector<llama_hybrid_runtime_stage>{};
    if (region_ring_v1 && !model.hybrid_ring_plan_layout.empty()) {
        // Every executable Stage must match the model's actual weight and
        // KV owner; a split between placement and task routing is fatal.
        const bool apply_cgp =
            model.hybrid_ring_plan_mode == "APPLY_CGP";
        const bool apply_wide =
            model.hybrid_ring_plan_mode == "APPLY_WIDE";
        const bool runtime_shape_ok = (apply_cgp || apply_wide) ?
            ((apply_wide || ring_runtime_regions.size() == 9) &&
             llama_region_ring_apply_topology_compatible(
                 ring_runtime_regions, model.hparams.n_layer()) &&
             ring_runtime_regions.back().device ==
                 llama_region_ring_device::PHONE) :
            llama_region_ring_apply_safe_phone_shape(ring_runtime_regions);
        const bool chunk_shape_ok = apply_wide ?
            (region_ring_xc > 0 && region_ring_xg > 0 &&
             region_ring_xp > 0) :
            (region_ring_xc == 64 && region_ring_xg == 64 &&
             region_ring_xp == 64);
        if (!runtime_shape_ok || !chunk_shape_ok ||
                model.pc_layer_backends.size() <
                    (size_t) model.hparams.n_layer()) {
            throw std::runtime_error(
                "Planner Ring runtime shape does not match loaded weights");
        }
        for (const auto & r : ring_runtime_regions) {
            for (int il = r.layer_begin; il < r.layer_end; ++il) {
                const bool phone =
                    r.device == llama_region_ring_device::PHONE;
                const bool gpu =
                    r.device == llama_region_ring_device::GPU;
                const bool owner_ok = phone ?
                    model.hybrid_layer_mode(il) ==
                        llama_hybrid_layer_mode::PHONE_ONLY :
                    model.hybrid_layer_mode(il) ==
                        llama_hybrid_layer_mode::PC_ONLY &&
                    (model.pc_layer_backends[(size_t) il] ==
                        llama_pc_layer_backend::CUDA) == gpu;
                if (!owner_ok) {
                    throw std::runtime_error(
                        "Planner Ring runtime/model backend ownership mismatch");
                }
            }
        }
        const char * plan_mode = model.hybrid_ring_plan_mode.empty() ?
            "APPLY_SAFE" : model.hybrid_ring_plan_mode.c_str();
        LLAMA_LOG_ERROR(
            "[REGION_RING_PLAN_APPLY] mode=%s "
            "phase=RUNTIME_STAGES applied=1 "
            "layout=%s XC=%d XG=%d XP=%d "
            "activation_buffer=FULL_BATCH_PER_EDGE slot_limit=NONE "
            "legacy_slot_pool=%zu status=CONSISTENT\n",
            plan_mode,model.hybrid_ring_plan_layout.c_str(),
            region_ring_xc, region_ring_xg, region_ring_xp,
            LLAMA_REGION_RING_SLOT_POOL_SIZE);
    }
    if (region_ring_v1) {
        LLAMA_LOG_ERROR(
            "[REGION_RING_PLAN] regions=%zu XC=%d XG=%d XP=%d "
            "execution=%s status=READY\n",
            runtime_stages.size(),
            region_ring_xc, region_ring_xg, region_ring_xp,
            ring_exec == "ready" ? "COOPERATIVE_READY_QUEUE" :
            ring_exec == "fifo" ? "DEVICE_FIFO_SIMULATION" :
            ring_exec == "fifo-isolated" ? "DEVICE_FIFO_ISOLATED_SIMULATION" :
            ring_exec == "fifo-owned" ? "DEVICE_FIFO_OWNED_SIMULATION" :
            ring_exec == "fifo-state" ? "DEVICE_FIFO_TASK_STATE_SIMULATION" :
            ring_exec == "fifo-workers" ? "DEVICE_FIFO_GATED_WORKERS" :
            ring_exec == "fifo-backend-audit" ? "DEVICE_FIFO_BACKEND_AUDIT" :
            ring_exec == "fifo-pc-phone" ? "DEVICE_FIFO_PC_PHONE_OVERLAP" :
            ring_exec == "serial" ? "SERIAL_STAGE_REFERENCE" :
                                   "FULL_GRAPH_REFERENCE");
    }
    const char * stage_queue_env = std::getenv("LLAMA_HYBRID_STAGE_QUEUE");
    const char * pc_phone_exec_v1_env =
        std::getenv("GGML_META_PC_PHONE_EXEC_V1");
    const bool pc_phone_exec_v1_requested =
        pc_phone_exec_v1_env != nullptr &&
        std::atoi(pc_phone_exec_v1_env) != 0;
    // PC/Phone executor V1 needs a Tensor-only stage graph for Qwen3-MoE:
    // the per-chunk return-wavefront graph cannot be built from the mixed
    // full GPU+CPU+TENSOR graph.  Make the executor opt-in self-contained
    // instead of requiring a second stage-queue environment switch.
    const bool stage_queue_requested =
        !region_ring_v1 &&
        ((stage_queue_env != nullptr && std::atoi(stage_queue_env) != 0) ||
         pc_phone_exec_v1_requested);
    const bool return_wavefront_requested =
        std::getenv("LLAMA_HYBRID_RETURN_WAVEFRONT") != nullptr ||
        pc_phone_exec_v1_requested;
    // Execution follows the model policy, not merely the planner flag.
    // A PHONE-primary plan may intentionally run with legacy TENSOR_SPLIT
    // semantics for correctness A/B testing.
    // Legacy Phone-primary staged execution assumes one contiguous Tensor
    // interval. Directional Region Boundary V1 uses sparse P2C boundaries and
    // must not enable that interval-wide shortcut; per-layer model ownership
    // plus the Meta masks remain authoritative.
    const bool tensor_phone_primary_exec =
        !directional_region_policy &&
        tensor_phone_first_layer >= 0 &&
        tensor_phone_last_layer > tensor_phone_first_layer;

    const bool gpu_tensor_topology =
        runtime_stages.size() == 2 &&
        runtime_stages[0].kind == llama_hybrid_runtime_stage_kind::GPU &&
        runtime_stages[1].kind == llama_hybrid_runtime_stage_kind::TENSOR;
    const bool valid_stage_macros =
        !runtime_stages.empty() &&
        std::all_of(runtime_stages.begin(), runtime_stages.end(),
                    [](const llama_hybrid_runtime_stage & stage) { return stage.macro_tokens > 0; });
    const bool stage_queue_plan_eligible =
        stage_queue_requested && has_runtime_plan &&
        !tensor_phone_primary_exec &&
        gpu_tensor_topology && model.arch != LLM_ARCH_QWEN2;
    const bool gpu_cpu_tensor_topology =
        runtime_stages.size() == 3 &&
        runtime_stages[0].kind == llama_hybrid_runtime_stage_kind::GPU &&
        runtime_stages[1].kind == llama_hybrid_runtime_stage_kind::CPU &&
        runtime_stages[2].kind == llama_hybrid_runtime_stage_kind::TENSOR &&
        runtime_stages[0].macro_tokens >= runtime_stages[1].macro_tokens &&
        runtime_stages[1].macro_tokens == runtime_stages[2].macro_tokens;
    const bool gpu_tensor_phone_topology =
        runtime_stages.size() == 3 &&
        runtime_stages[0].kind == llama_hybrid_runtime_stage_kind::GPU &&
        runtime_stages[1].kind == llama_hybrid_runtime_stage_kind::TENSOR &&
        runtime_stages[2].kind == llama_hybrid_runtime_stage_kind::PHONE &&
        runtime_stages[2].macro_tokens > 0 &&
        runtime_stages[1].macro_tokens >= runtime_stages[2].macro_tokens;
    const bool gpu_cpu_tensor_phone_topology =
        runtime_stages.size() == 4 &&
        runtime_stages[0].kind == llama_hybrid_runtime_stage_kind::GPU &&
        runtime_stages[1].kind == llama_hybrid_runtime_stage_kind::CPU &&
        runtime_stages[2].kind == llama_hybrid_runtime_stage_kind::TENSOR &&
        runtime_stages[3].kind == llama_hybrid_runtime_stage_kind::PHONE &&
        runtime_stages[0].macro_tokens >= runtime_stages[1].macro_tokens &&
        runtime_stages[1].macro_tokens == runtime_stages[2].macro_tokens &&
        runtime_stages[3].macro_tokens > 0 &&
        runtime_stages[2].macro_tokens >= runtime_stages[3].macro_tokens &&
        runtime_stages[2].macro_tokens % runtime_stages[3].macro_tokens == 0;
    // Qwen2 return-wavefront is currently implemented as a full-graph path.
    // Do not slice a pure GPU->TENSOR prefill into upstream-sized stage graphs:
    // doing so makes each Tensor job only one XT chunk wide (or smaller) and
    // also sets stage_graph=true, which bypasses the wavefront graph builder.
    //
    // Keep the generalized stage queue for the other supported topologies.
    // GPU->TENSOR + return-wavefront intentionally falls back to the normal
    // full graph so the Tensor suffix can contain multiple XT chunks and
    // advance as (layer, chunk) wavefronts.
    const bool return_wavefront_full_graph_override =
        return_wavefront_requested &&
        !tensor_phone_primary_exec &&
        gpu_tensor_topology && model.arch == LLM_ARCH_QWEN2;
    const bool generalized_stage_arch =
        model.arch == LLM_ARCH_QWEN2 || model.arch == LLM_ARCH_QWEN3MOE;
    const bool generalized_stage_queue_plan_eligible =
        stage_queue_requested && has_runtime_plan &&
        !tensor_phone_primary_exec &&
        valid_stage_macros && generalized_stage_arch &&
        !return_wavefront_full_graph_override &&
        (gpu_tensor_topology || gpu_cpu_tensor_topology || gpu_tensor_phone_topology ||
         gpu_cpu_tensor_phone_topology);
    const bool stage_serial_plan_eligible =
        stage_queue_requested && has_runtime_plan &&
        !tensor_phone_primary_exec &&
        !gpu_tensor_topology && runtime_stages.size() >= 2 &&
        valid_stage_macros && generalized_stage_arch;

    // Formal return-wavefront profiling must be self-contained. Historically the
    // online wave probe reset these counters as a side effect before the real
    // prefill; keep the real measurement correct even when no probe is run.
    if (n_tokens_all > 1 && return_wavefront_full_graph_override) {
        for (ggml_backend_t backend : backend_ptrs) {
            if (backend != nullptr) {
                ggml_backend_meta_tensor_profile_reset(backend);
            }
        }
    }

    // The wave probe is diagnostic-only. It is never required for plan
    // selection or for WAVE_REAL collection, because executing it adds a real
    // Tensor forward pass to user-visible prefill latency.
    const char * wave_probe_env = std::getenv("LLAMA_HYBRID_WAVE_PROBE");
    const bool wave_probe_diagnostic_requested =
        wave_probe_env != nullptr && std::atoi(wave_probe_env) != 0;
    if (wave_probe_diagnostic_requested &&
        !hybrid_wave_probe_done &&
        has_runtime_plan &&
        n_tokens_all > (uint32_t) std::max(1, runtime_plan.tensor_chunk_tokens) &&
        return_wavefront_requested &&
        return_wavefront_full_graph_override) {
        LLAMA_LOG_WARN(
            "[WAVE_PROBE] diagnostic mode enabled; probe time is part of prompt latency\n");

        int probe_layers = 16;
        if (const char * layers_env = std::getenv("LLAMA_HYBRID_WAVE_PROBE_LAYERS")) {
            const int requested = std::atoi(layers_env);
            if (requested > 0) {
                probe_layers = requested;
            }
        }
        probe_layers = std::min(probe_layers, runtime_plan.tensor_layers);
        hybrid_wave_probe_done =
            run_hybrid_wave_probe(n_tokens_all, probe_layers, runtime_plan);
    }

    if (stage_queue_requested && has_runtime_plan) {
        if (pc_phone_exec_v1_requested) {
            LLAMA_LOG_INFO(
                "[PC_PHONE_STAGE_AUTO] enabled=1 "
                "stage_queue=%d return_wavefront=%d "
                "reason=PC_PHONE_EXEC_V1\n",
                stage_queue_requested ? 1 : 0,
                return_wavefront_requested ? 1 : 0);
        }
        if (return_wavefront_full_graph_override) {
            const int attn_group_chunks =
                llama_hybrid_runtime_prefill_attn_group_chunks();
            LLAMA_LOG_DEBUG(
                "[HYBRID_PIPE] return-wavefront full-graph override: "
                "GPU->TENSOR stage queue disabled, batch=%u XG=%d XT=%d GA=%d XA=%d\n",
                n_tokens_all, runtime_plan.gpu_chunk_tokens, runtime_plan.tensor_chunk_tokens,
                attn_group_chunks,
                runtime_plan.tensor_chunk_tokens * attn_group_chunks);

            llama_hybrid_full_prefill_prediction full_prediction;
            if (llama_hybrid_runtime_predict_full_prefill(
                    (int) n_tokens_all, full_prediction)) {
                LLAMA_LOG_DEBUG(
                    "[PRED_WAVE_FULL] tokens=%d kv_tokens=%d T=%d XT=%d GA=%d XA=%d R=%.3f "
                    "gpu_ms=%.3f tensor_ms=%.3f total_ms=%.3f\n",
                    full_prediction.tokens,
                    full_prediction.kv_tokens,
                    full_prediction.tensor_layers,
                    full_prediction.tensor_chunk_tokens,
                    full_prediction.attn_group_chunks,
                    full_prediction.attn_chunk_tokens,
                    full_prediction.tensor_pc_ratio,
                    full_prediction.gpu_ms,
                    full_prediction.tensor_ms,
                    full_prediction.total_ms);

                llama_hybrid_wave_calibration calibration;
                llama_hybrid_tensor_compute_prediction tensor_prediction;
                if (llama_hybrid_runtime_wave_calibration_get(calibration) &&
                    llama_hybrid_runtime_predict_tensor_compute(
                        (int) n_tokens_all, tensor_prediction) &&
                    calibration.tokens == (int) n_tokens_all &&
                    calibration.tensor_layers > 0 &&
                    calibration.tensor_chunk_tokens == full_prediction.tensor_chunk_tokens &&
                    calibration.attn_group_chunks == full_prediction.attn_group_chunks &&
                    std::abs(calibration.tensor_pc_ratio - full_prediction.tensor_pc_ratio) < 1e-4f &&
                    tensor_prediction.tensor_layers > 0) {
                    const double cal_layers = (double) calibration.tensor_layers;
                    const double pred_layers = (double) tensor_prediction.tensor_layers;

                    const double cal_attn_per_layer =
                        calibration.attn_ms / cal_layers;
                    const double cal_pc_ffn_per_layer =
                        calibration.pc_ffn_ms / cal_layers;
                    const double cal_compute_wall_per_layer =
                        calibration.compute_wall_ms / cal_layers;
                    const double cal_meta_per_layer =
                        calibration.meta_total_ms / cal_layers;

                    const double pred_attn_per_layer =
                        tensor_prediction.attn_misc_ms / pred_layers;
                    const double pred_pc_ffn_per_layer =
                        tensor_prediction.pc_ffn_ms / pred_layers;
                    const double pred_pc_compute_per_layer =
                        tensor_prediction.pc_compute_ms / pred_layers;
                    const double pred_tensor_per_layer =
                        tensor_prediction.tensor_total_ms / pred_layers;

                    LLAMA_LOG_DEBUG(
                        "[PRED_WAVE_CAL_DIAG] probe_layers=%d "
                        "cal_attn_per_layer_ms=%.3f pred_attn_per_layer_ms=%.3f attn_ratio=%.4f "
                        "cal_pc_ffn_per_layer_ms=%.3f pred_pc_ffn_per_layer_ms=%.3f pc_ffn_ratio=%.4f "
                        "cal_compute_wall_per_layer_ms=%.3f pred_pc_compute_per_layer_ms=%.3f compute_ratio=%.4f "
                        "cal_meta_per_layer_ms=%.3f pred_tensor_per_layer_ms=%.3f meta_ratio=%.4f\n",
                        calibration.tensor_layers,
                        cal_attn_per_layer,
                        pred_attn_per_layer,
                        pred_attn_per_layer > 0.0 ? cal_attn_per_layer / pred_attn_per_layer : 0.0,
                        cal_pc_ffn_per_layer,
                        pred_pc_ffn_per_layer,
                        pred_pc_ffn_per_layer > 0.0 ? cal_pc_ffn_per_layer / pred_pc_ffn_per_layer : 0.0,
                        cal_compute_wall_per_layer,
                        pred_pc_compute_per_layer,
                        pred_pc_compute_per_layer > 0.0 ?
                            cal_compute_wall_per_layer / pred_pc_compute_per_layer : 0.0,
                        cal_meta_per_layer,
                        pred_tensor_per_layer,
                        pred_tensor_per_layer > 0.0 ? cal_meta_per_layer / pred_tensor_per_layer : 0.0);

                    LLAMA_LOG_DEBUG(
                        "[PRED_WAVE_II_PENDING] probe_layers=%d target_layers=%d "
                        "probe_ii_median_ms=%.3f late_ii_median_ms=%.3f "
                        "ii_drift_ratio=%.4f waiting_for_formal_prepare=1\n",
                        calibration.tensor_layers,
                        full_prediction.tensor_layers,
                        calibration.wave_ii_median_ms,
                        calibration.wave_late_ii_median_ms,
                        calibration.wave_ii_drift_ratio);
                }
            }
        }
        for (size_t i = 0; i < runtime_stages.size(); ++i) {
            const auto & stage = runtime_stages[i];
            LLAMA_LOG_INFO("[HYBRID_STAGE] index=%zu kind=%s layers=[%d,%d) macro=%d inner=%d\n",
                            i, llama_hybrid_runtime_stage_name(stage.kind), stage.layer_begin, stage.layer_end,
                            stage.macro_tokens, stage.inner_chunk_tokens);
        }
        if (n_tokens_all > 1) {
            for (size_t i = 1; i < runtime_stages.size(); ++i) {
                const auto & upstream   = runtime_stages[i - 1];
                const auto & downstream = runtime_stages[i];
                const auto blocks = llama_hybrid_plan_boundary(
                    n_tokens_all, upstream.macro_tokens, downstream.macro_tokens);

                LLAMA_LOG_ERROR(
                    "[HYBRID_BOUNDARY] index=%zu %s->%s upstream_macro=%d downstream_macro=%d blocks=%zu\n",
                    i - 1, llama_hybrid_runtime_stage_name(upstream.kind),
                    llama_hybrid_runtime_stage_name(downstream.kind), upstream.macro_tokens,
                    downstream.macro_tokens, blocks.size());

                for (const auto & block : blocks) {
                    std::string inputs;
                    for (const auto & input : block.inputs) {
                        if (!inputs.empty()) {
                            inputs += "+";
                        }
                        inputs += "[" + std::to_string(input.token_begin) + "," +
                            std::to_string(input.token_begin + input.n_tokens) + ")";
                    }
                    LLAMA_LOG_ERROR(
                        "[HYBRID_BLOCK] boundary=%zu action=%s output=[%d,%d) inputs=%s\n",
                        i - 1, llama_hybrid_boundary_action_name(block.action), block.output.token_begin,
                        block.output.token_begin + block.output.n_tokens, inputs.c_str());
                }
            }
        }
    }

    if (stage_queue_requested && !stage_queue_plan_eligible && !generalized_stage_queue_plan_eligible &&
        !stage_serial_plan_eligible) {
        LLAMA_LOG_WARN(
            "[STAGEQ] disabled: runtime stage execution is not supported for this plan or model\n");
    }

    bool stage_queue_runtime_enabled = false;
    const uint32_t generalized_macros = generalized_stage_queue_plan_eligible ?
        (n_tokens_all + runtime_stages.front().macro_tokens - 1) / runtime_stages.front().macro_tokens : 0;
    const bool generalized_stage_queue_runtime_enabled = generalized_macros >= 2;
    const bool stage_serial_runtime_enabled =
        stage_serial_plan_eligible && !generalized_stage_queue_runtime_enabled && n_tokens_all > 1;
    uint32_t runtime_ubatch = cparams.n_ubatch;
    if (stage_queue_plan_eligible) {
        const uint32_t xg = std::min<uint32_t>(cparams.n_ubatch, runtime_plan.gpu_chunk_tokens);
        const uint32_t n_macros = (n_tokens_all + xg - 1) / xg;

        stage_queue_runtime_enabled = n_macros >= 2;
        if (stage_queue_runtime_enabled) {
            runtime_ubatch = xg;
        }
        LLAMA_LOG_DEBUG("[STAGEQ] batch=%u XG=%u macros=%u XT=%d enabled=%d overlap=%d warmup=%d\n",
                        n_tokens_all, xg, n_macros, runtime_plan.tensor_chunk_tokens,
                        (int) stage_queue_runtime_enabled, (int) (n_macros >= 2), (int) cparams.warmup);
    } else if (generalized_stage_queue_runtime_enabled) {
        runtime_ubatch = std::min<uint32_t>(cparams.n_ubatch, runtime_stages.front().macro_tokens);
        LLAMA_LOG_DEBUG(
            "[HYBRID_PIPE] mode=GPU_DOWNSTREAM batch=%u ubatch=%u macros=%u stages=%zu\n",
            n_tokens_all, runtime_ubatch, generalized_macros, runtime_stages.size());
    } else if (stage_serial_runtime_enabled) {
        // Serial staged execution already materializes every stage output into
        // a full hidden buffer. Use the largest stage macro for the memory
        // ubatch so an upstream small macro can be accumulated before a
        // downstream larger macro. Example:
        //
        //   XG=64, XC=256, XT=48
        //   GPU:    64 + 64 + 64 + 64
        //             -> one contiguous 256-token hidden buffer
        //   CPU:   256
        //   TENSOR:256 (internally split by XT)
        //
        // Using the first-stage macro here used to force the entire staged
        // path down to XG, so XC=256 was never realized at runtime.
        uint32_t serial_macro = 1;
        for (const auto & stage : runtime_stages) {
            serial_macro = std::max<uint32_t>(
                serial_macro, (uint32_t) stage.macro_tokens);
        }
        runtime_ubatch =
            std::min<uint32_t>(cparams.n_ubatch, serial_macro);
        LLAMA_LOG_DEBUG(
            "[HYBRID_EXEC] mode=serial batch=%u ubatch=%u "
            "front_macro=%d max_macro=%u stages=%zu\n",
            n_tokens_all, runtime_ubatch,
            runtime_stages.front().macro_tokens,
            serial_macro, runtime_stages.size());
    }

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, runtime_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;
    int64_t n_tokens_prev  = 0;
    int     ubatch_id      = 0;

    llama_prefill_pipe_slot pipe_slots[2];
    pipe_slots[0].sched = sched.get();
    pipe_slots[0].res   = gf_res_prev.get();
    pipe_slots[1].sched = sched_pipe.get();
    pipe_slots[1].res   = gf_res_pipe.get();

    struct llama_hybrid_job {
        llama_ubatch       ubatch;
        size_t             stage_index = 1;
        std::vector<float> hidden;
        int                ubatch_id = -1;
        int32_t            n_outputs = 0;
    };

    llama_hybrid_job hybrid_gpu_pending;
    bool hybrid_gpu_pending_active = false;
    std::deque<llama_hybrid_job> hybrid_cpu_ready_q;
    std::deque<llama_hybrid_job> hybrid_tensor_ready_q;
    std::deque<llama_hybrid_job> hybrid_phone_ready_q;
    size_t hybrid_tensor_ready_bytes = 0;
    size_t hybrid_phone_ready_bytes = 0;
    size_t hybrid_tensor_queue_limit_bytes = 16ull * 1024 * 1024;
    size_t hybrid_phone_queue_limit_bytes = 16ull * 1024 * 1024;
    size_t hybrid_cpu_ready_max = 2;

    if (const char * env = std::getenv("LLAMA_HYBRID_TENSOR_QUEUE_MB")) {
        const long long mb = std::atoll(env);
        if (mb > 0) {
            hybrid_tensor_queue_limit_bytes = (size_t) mb * 1024ull * 1024ull;
        }
    }

    if (const char * env = std::getenv("LLAMA_HYBRID_CPU_READY_MAX")) {
        const long long value = std::atoll(env);
        if (value > 0) {
            hybrid_cpu_ready_max = (size_t) value;
        }
    }

    if (const char * env = std::getenv("LLAMA_HYBRID_PHONE_QUEUE_MB")) {
        const long long mb = std::atoll(env);
        if (mb > 0) {
            hybrid_phone_queue_limit_bytes = (size_t) mb * 1024ull * 1024ull;
        }
    }

    const auto hybrid_has_ready = [&]() {
        return !hybrid_cpu_ready_q.empty() || !hybrid_tensor_ready_q.empty() || !hybrid_phone_ready_q.empty();
    };

    std::deque<llama_prefill_pipe_slot *> handoff_q;
    std::deque<llama_prefill_pipe_slot *> tensor_ready_q;

    llama_prefill_pipe_slot * prev                     = nullptr;
    // Ring's ready/serial/full modes are mutually exclusive with the legacy
    // prefill pipeline. A stale LLAMA_PREFILL_STAGE_PIPELINE environment
    // variable must not silently reroute the Ring full-graph reference.
    bool legacy_pipeline_runtime_enabled =
        !region_ring_v1 && std::getenv("LLAMA_PREFILL_STAGE_PIPELINE") != nullptr;
    bool pipeline_runtime_enabled = stage_queue_runtime_enabled || legacy_pipeline_runtime_enabled;

    const auto stage_queue_handoff = [&](llama_prefill_pipe_slot & slot) {
        LLAMA_LOG_DEBUG("[STAGEQ] ub=%d slot=%d HANDOFF_BEGIN\n", slot.ubatch_id, slot.ubatch_id & 1);
        const ggml_status status = pipe_prepare_post(slot);
        if (status == GGML_STATUS_SUCCESS) {
            LLAMA_LOG_DEBUG("[STAGEQ] ub=%d slot=%d TENSOR_READY\n", slot.ubatch_id, slot.ubatch_id & 1);
        }
        return status;
    };

    const auto stage_queue_run_tensor = [&](llama_prefill_pipe_slot & slot) {
        LLAMA_LOG_DEBUG("[STAGEQ] ub=%d slot=%d TENSOR_BEGIN tokens=%u XT=%d\n", slot.ubatch_id,
                        slot.ubatch_id & 1, slot.n_tokens, runtime_plan.tensor_chunk_tokens);
        ggml_status status = pipe_run_post_prepared(slot);
        if (status == GGML_STATUS_SUCCESS) {
            pipe_sync_post(slot);
            status = pipe_run_tail(slot);
        }
        if (status == GGML_STATUS_SUCCESS) {
            LLAMA_LOG_DEBUG("[STAGEQ] ub=%d slot=%d TENSOR_END\n", slot.ubatch_id, slot.ubatch_id & 1);
        }
        return status;
    };

    const auto stage_queue_drain = [&]() {
        ggml_status status = GGML_STATUS_SUCCESS;
        while (status == GGML_STATUS_SUCCESS && !handoff_q.empty()) {
            llama_prefill_pipe_slot * task = handoff_q.front();
            handoff_q.pop_front();
            status = stage_queue_handoff(*task);
            if (status == GGML_STATUS_SUCCESS) {
                tensor_ready_q.push_back(task);
            }
        }
        while (status == GGML_STATUS_SUCCESS && !tensor_ready_q.empty()) {
            llama_prefill_pipe_slot * task = tensor_ready_q.front();
            tensor_ready_q.pop_front();
            status = stage_queue_run_tensor(*task);
        }
        return status;
    };

    const auto hybrid_enqueue_ready = [&](llama_hybrid_job && job, ggml_status & status) {
        if (job.stage_index >= runtime_stages.size()) {
            LLAMA_LOG_ERROR(
                "[HYBRID_PIPE] ub=%d invalid completed job stage=%zu\n",
                job.ubatch_id, job.stage_index);
            status = GGML_STATUS_FAILED;
            return;
        }

        const auto kind = runtime_stages[job.stage_index].kind;
        switch (kind) {
            case llama_hybrid_runtime_stage_kind::CPU:
                LLAMA_LOG_DEBUG(
                    "[HYBRID_PIPE] ub=%d -> CPU_QUEUE stage=%zu\n",
                    job.ubatch_id, job.stage_index);
                hybrid_cpu_ready_q.push_back(std::move(job));
                break;
            case llama_hybrid_runtime_stage_kind::TENSOR:
                {
                    const size_t tensor_bytes = job.hidden.capacity() * sizeof(float);
                    hybrid_tensor_ready_bytes += tensor_bytes;
                    LLAMA_LOG_DEBUG(
                        "[HYBRID_PIPE] ub=%d -> TENSOR_QUEUE stage=%zu "
                        "job_mib=%.2f queue_mib=%.2f limit_mib=%.2f\n",
                        job.ubatch_id, job.stage_index, tensor_bytes / 1048576.0,
                        hybrid_tensor_ready_bytes / 1048576.0,
                        hybrid_tensor_queue_limit_bytes / 1048576.0);
                    hybrid_tensor_ready_q.push_back(std::move(job));
                }
                break;
            case llama_hybrid_runtime_stage_kind::PHONE:
                {
                    const size_t phone_bytes = job.hidden.capacity() * sizeof(float);
                    hybrid_phone_ready_bytes += phone_bytes;
                    LLAMA_LOG_DEBUG(
                        "[HYBRID_PIPE] ub=%d -> PHONE_QUEUE stage=%zu "
                        "job_mib=%.2f queue_mib=%.2f limit_mib=%.2f\n",
                        job.ubatch_id, job.stage_index, phone_bytes / 1048576.0,
                        hybrid_phone_ready_bytes / 1048576.0,
                        hybrid_phone_queue_limit_bytes / 1048576.0);
                    hybrid_phone_ready_q.push_back(std::move(job));
                }
                break;
            default:
                LLAMA_LOG_ERROR(
                    "[HYBRID_PIPE] ub=%d unsupported next stage=%s stage=%zu\n",
                    job.ubatch_id, llama_hybrid_runtime_stage_name(kind), job.stage_index);
                status = GGML_STATUS_FAILED;
                break;
        }
    };

    const auto hybrid_gpu_harvest = [&](ggml_status & status) {
        GGML_ASSERT(hybrid_gpu_pending_active);

        LLAMA_LOG_DEBUG("[HYBRID_PIPE] ub=%d GPU_WAIT tokens=%u\n",
                        hybrid_gpu_pending.ubatch_id, hybrid_gpu_pending.ubatch.n_tokens);
        if (!copy_hybrid_stage_output(
                gf_res_prev.get(), sched.get(), hybrid_gpu_pending.hidden.data(),
                hybrid_gpu_pending.ubatch.n_tokens, hybrid_gpu_pending.ubatch_id, 0, 0)) {
            status = GGML_STATUS_FAILED;
            return;
        }

        LLAMA_LOG_DEBUG("[HYBRID_PIPE] ub=%d GPU_READY next_stage=%zu\n",
                        hybrid_gpu_pending.ubatch_id, hybrid_gpu_pending.stage_index);
        hybrid_enqueue_ready(std::move(hybrid_gpu_pending), status);
        hybrid_gpu_pending = {};
        hybrid_gpu_pending_active = false;
    };

    const auto hybrid_gpu_submit = [&](
            const llama_ubatch & ubatch, int ubatch_id, int32_t job_outputs, ggml_status & status) {
        GGML_ASSERT(!hybrid_gpu_pending_active);

        hybrid_gpu_pending = {};
        hybrid_gpu_pending.ubatch    = ubatch;
        hybrid_gpu_pending.ubatch_id = ubatch_id;
        hybrid_gpu_pending.n_outputs = job_outputs;
        hybrid_gpu_pending.stage_index = 1;
        hybrid_gpu_pending.hidden.resize((size_t) model.hparams.n_embd * ubatch.n_tokens);

        const int32_t n_outputs_saved = n_outputs;
        bool apply_mctx = true;
        llm_graph_result * result = run_hybrid_stage_block(
            hybrid_gpu_pending.ubatch, runtime_stages[0], 0, ubatch.n_tokens, nullptr, nullptr,
            ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), sched.get(), gf_res_prev.get(), ubatch_id, 0, 0,
            llama_hybrid_boundary_action::PASS, 0, apply_mctx, false, status);
        n_outputs = n_outputs_saved;

        if (result != nullptr && status == GGML_STATUS_SUCCESS) {
            hybrid_gpu_pending_active = true;
            LLAMA_LOG_DEBUG("[HYBRID_PIPE] ub=%d GPU_SUBMIT tokens=%u\n", ubatch_id, ubatch.n_tokens);
        }
    };

    const auto hybrid_run_downstream_stage = [&]
            (llama_hybrid_job & job, ggml_status & status) -> llm_graph_result * {
        GGML_ASSERT(job.stage_index > 0 && job.stage_index < runtime_stages.size());

        const size_t stage_index = job.stage_index;
        const auto & previous_stage = runtime_stages[stage_index - 1];
        const auto & stage = runtime_stages[stage_index];
        const bool last_stage = stage_index + 1 == runtime_stages.size();
        const auto blocks = llama_hybrid_plan_boundary(
            job.ubatch.n_tokens, previous_stage.macro_tokens, stage.macro_tokens);

        if (blocks.empty() || std::any_of(
                blocks.begin(), blocks.end(), [](const llama_hybrid_boundary_block & block) {
                    return block.action == llama_hybrid_boundary_action::ACCUMULATE;
                })) {
            LLAMA_LOG_ERROR(
                "[HYBRID_PIPE] ub=%d stage=%zu kind=%s unsupported boundary plan\n",
                job.ubatch_id, stage_index, llama_hybrid_runtime_stage_name(stage.kind));
            status = GGML_STATUS_FAILED;
            return nullptr;
        }

        const int64_t n_embd = model.hparams.n_embd;
        std::vector<float> next_hidden;
        if (!last_stage) {
            next_hidden.resize((size_t) n_embd * job.ubatch.n_tokens);
        }

        bool apply_mctx = false;
        llm_graph_result * result = nullptr;
        const bool profile_cpu    = stage.kind == llama_hybrid_runtime_stage_kind::CPU;
        const bool profile_tensor = stage.kind == llama_hybrid_runtime_stage_kind::TENSOR;
        const int64_t cpu_start_us    = profile_cpu    ? ggml_time_us() : 0;
        const int64_t tensor_start_us = profile_tensor ? ggml_time_us() : 0;
        llama_hybrid_stage_timing cpu_stage_timing {};
        llama_hybrid_stage_timing tensor_stage_timing {};
        if (profile_tensor) {
            const int n_backends = ggml_backend_sched_get_n_backends(sched_pipe.get());
            for (int i = 0; i < n_backends; ++i) {
                ggml_backend_meta_tensor_profile_reset(ggml_backend_sched_get_backend(sched_pipe.get(), i));
            }
        }
        LLAMA_LOG_DEBUG(
            "[HYBRID_PIPE] ub=%d %s_PHASE_BEGIN stage=%zu blocks=%zu\n",
            job.ubatch_id, llama_hybrid_runtime_stage_name(stage.kind), stage_index, blocks.size());

        for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
            const auto & block = blocks[block_index];
            const uint32_t token_begin = block.output.token_begin;
            const uint32_t block_tokens = block.output.n_tokens;
            float * stage_input = job.hidden.data() + (size_t) token_begin * n_embd;
            float * stage_output = last_stage ? nullptr :
                next_hidden.data() + (size_t) token_begin * n_embd;

            int32_t block_outputs = 0;
            if (last_stage) {
                for (uint32_t i = 0; i < block_tokens; ++i) {
                    block_outputs += job.ubatch.output[token_begin + i] != 0;
                }
            }

            result = run_hybrid_stage_block(
                job.ubatch, stage, token_begin, block_tokens, stage_input, stage_output,
                ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), sched_pipe.get(), gf_res_pipe.get(),
                job.ubatch_id, stage_index, block_index, block.action, block_outputs,
                apply_mctx, true, status,
                profile_tensor ? &tensor_stage_timing :
                profile_cpu ? &cpu_stage_timing : nullptr);
            if (result == nullptr || status != GGML_STATUS_SUCCESS) {
                return nullptr;
            }
        }

        static_cast<llama_kv_cache_context *>(mctx.get())->clear_stage_range();
        job.hidden = std::move(next_hidden);
        job.stage_index++;

        if (profile_cpu) {
            const int64_t cpu_total_us = ggml_time_us() - cpu_start_us;
            const int64_t cpu_accounted_us =
                cpu_stage_timing.prepare_us +
                cpu_stage_timing.compute_range_us +
                cpu_stage_timing.sync_us;
            const int64_t cpu_unaccounted_us =
                std::max<int64_t>(0, cpu_total_us - cpu_accounted_us);
            const int cpu_layers = stage.layer_end - stage.layer_begin;

            LLAMA_LOG_DEBUG(
                "[CPU_STAGE_TIMING] ub=%d tokens=%u layers=[%d,%d) n_layers=%d XC=%d "
                "total_ms=%.3f prepare_ms=%.3f compute_ms=%.3f sync_ms=%.3f "
                "unaccounted_ms=%.3f blocks=%zu\n",
                job.ubatch_id,
                job.ubatch.n_tokens,
                stage.layer_begin,
                stage.layer_end,
                cpu_layers,
                stage.macro_tokens,
                cpu_total_us / 1000.0,
                cpu_stage_timing.prepare_us / 1000.0,
                cpu_stage_timing.compute_range_us / 1000.0,
                cpu_stage_timing.sync_us / 1000.0,
                cpu_unaccounted_us / 1000.0,
                blocks.size());

            llama_hybrid_cpu_compute_prediction cpu_prediction;
            if (llama_hybrid_runtime_predict_cpu_compute(
                    (int) job.ubatch.n_tokens, cpu_prediction)) {
                const double actual_compute_ms =
                    cpu_stage_timing.compute_range_us / 1000.0;
                const double actual_per_layer_ms =
                    cpu_layers > 0 ? actual_compute_ms / cpu_layers : 0.0;
                const double compute_ratio =
                    cpu_prediction.total_ms > 0.0 ?
                        actual_compute_ms / cpu_prediction.total_ms : 0.0;

                const char * profile_mode =
                    cpu_prediction.profile_extrapolated_chunks > 0 ? "EXTRAPOLATE" :
                    cpu_prediction.profile_interpolated_chunks > 0 ? "INTERPOLATE" :
                    "EXACT";

                const char * layer_profile_mode =
                    cpu_prediction.profile_layer_exact ? "EXACT" :
                    cpu_prediction.profile_layer_saturated ? "SATURATED" :
                    "INTERPOLATE";

                LLAMA_LOG_DEBUG(
                    "[PRED_CPU_STAGE] ub=%d tokens=%d kv_tokens=%d layers=%d XC=%d "
                    "pred_total_ms=%.3f pred_per_layer_ms=%.3f "
                    "profile_mode=%s profile_range=[%d,%d] "
                    "exact_chunks=%d interp_chunks=%d extrap_chunks=%d "
                    "layer_profile_mode=%s layer_profile_range=[%d,%d]\n",
                    job.ubatch_id,
                    cpu_prediction.tokens,
                    cpu_prediction.kv_tokens,
                    cpu_prediction.cpu_layers,
                    cpu_prediction.cpu_chunk_tokens,
                    cpu_prediction.total_ms,
                    cpu_prediction.per_layer_ms,
                    profile_mode,
                    cpu_prediction.profile_min_tokens,
                    cpu_prediction.profile_max_tokens,
                    cpu_prediction.profile_exact_chunks,
                    cpu_prediction.profile_interpolated_chunks,
                    cpu_prediction.profile_extrapolated_chunks,
                    layer_profile_mode,
                    cpu_prediction.profile_min_layers,
                    cpu_prediction.profile_max_layers);

                LLAMA_LOG_DEBUG(
                    "[PRED_CPU_ERROR] ub=%d actual_compute_ms=%.3f "
                    "actual_per_layer_ms=%.3f pred_compute_ms=%.3f "
                    "actual_over_pred=%.3f\n",
                    job.ubatch_id,
                    actual_compute_ms,
                    actual_per_layer_ms,
                    cpu_prediction.total_ms,
                    compute_ratio);
            }
        }

        if (profile_tensor) {
            ggml_backend_meta_tensor_profile profile {};
            const int n_backends = ggml_backend_sched_get_n_backends(sched_pipe.get());
            for (int i = 0; i < n_backends; ++i) {
                ggml_backend_meta_tensor_profile backend_profile {};
                if (!ggml_backend_meta_tensor_profile_get(
                        ggml_backend_sched_get_backend(sched_pipe.get(), i), &backend_profile)) {
                    continue;
                }
                profile.attn_us   += backend_profile.attn_us;
                profile.pc_ffn_us += backend_profile.pc_ffn_us;
                profile.h2d_us    += backend_profile.h2d_us;
                profile.phone_us  += backend_profile.phone_us;
                profile.d2h_us    += backend_profile.d2h_us;
                profile.reduce_us += backend_profile.reduce_us;
                profile.wait_us   += backend_profile.wait_us;
                profile.return_transfer_count +=
                    backend_profile.return_transfer_count;
                profile.return_payload_bytes +=
                    backend_profile.return_payload_bytes;
                profile.return_request_us +=
                    backend_profile.return_request_us;
                profile.return_ready_wait_us +=
                    backend_profile.return_ready_wait_us;
                profile.return_recv_payload_us +=
                    backend_profile.return_recv_payload_us;
                profile.return_rpc_total_us +=
                    backend_profile.return_rpc_total_us;
                profile.graph_compute_count += backend_profile.graph_compute_count;
                profile.graph_rebuild_count += backend_profile.graph_rebuild_count;
                profile.graph_total_us += backend_profile.graph_total_us;
                profile.graph_rebuild_us += backend_profile.graph_rebuild_us;
                profile.graph_execute_us += backend_profile.graph_execute_us;
                profile.graph_other_us += backend_profile.graph_other_us;
                profile.lane_reuse_wait_count += backend_profile.lane_reuse_wait_count;
                profile.lane_reuse_wait_us += backend_profile.lane_reuse_wait_us;
                profile.lane_reuse_wait_max_us = std::max(
                    profile.lane_reuse_wait_max_us, backend_profile.lane_reuse_wait_max_us);
                for (size_t lane = 0; lane < 2; ++lane) {
                    profile.lane_reuse_wait_count_by_lane[lane] +=
                        backend_profile.lane_reuse_wait_count_by_lane[lane];
                    profile.lane_reuse_wait_us_by_lane[lane] +=
                        backend_profile.lane_reuse_wait_us_by_lane[lane];
                }
                if (backend_profile.layer_barrier_wait_count > 0 &&
                    backend_profile.layer_barrier_wait_max_us >= profile.layer_barrier_wait_max_us) {
                    profile.layer_barrier_wait_max_us = backend_profile.layer_barrier_wait_max_us;
                    profile.layer_barrier_wait_max_layer = backend_profile.layer_barrier_wait_max_layer;
                    profile.layer_barrier_wait_max_pending = backend_profile.layer_barrier_wait_max_pending;
                    profile.layer_barrier_wait_max_last_lane = backend_profile.layer_barrier_wait_max_last_lane;
                }
                profile.layer_barrier_wait_count += backend_profile.layer_barrier_wait_count;
                profile.layer_barrier_wait_us += backend_profile.layer_barrier_wait_us;
            }
            const int64_t tensor_total_us = ggml_time_us() - tensor_start_us;
            const int64_t tensor_stage_accounted_us =
                tensor_stage_timing.prepare_us +
                tensor_stage_timing.compute_range_us +
                tensor_stage_timing.sync_us;
            const int64_t tensor_stage_unaccounted_us =
                std::max<int64_t>(0, tensor_total_us - tensor_stage_accounted_us);
            if (std::getenv("LLAMA_HYBRID_PREDICT_VALIDATE") != nullptr &&
                    job.ubatch.n_tokens > 16) {
                llama_hybrid_tensor_compute_prediction validate_prediction;
                if (llama_hybrid_runtime_predict_tensor_compute(
                        (int) job.ubatch.n_tokens, validate_prediction)) {
                    const double actual_ms = tensor_total_us / 1000.0;
                    const double actual_over_pred =
                        validate_prediction.tensor_total_ms > 0.0 ?
                            actual_ms / validate_prediction.tensor_total_ms :
                            0.0;
                    LLAMA_LOG_ERROR(
                        "[TENSOR_PREDICT_VALIDATE] ub=%d mode=STAGE_QUEUE "
                        "primary=%s tokens=%d T=%d R=%.3f XT=%d chunks=%d "
                        "actual_ms=%.3f pred_ms=%.3f actual_over_pred=%.3f "
                        "prepare_ms=%.3f compute_ms=%.3f sync_ms=%.3f "
                        "attn_misc_pred_ms=%.3f h2d_sum_pred_ms=%.3f "
                        "pc_ffn_sum_pred_ms=%.3f phone_sum_pred_ms=%.3f "
                        "phone_start_pred_ms=%.3f return_sum_pred_ms=%.3f "
                        "h2d_finish_pred_ms=%.3f "
                        "pc_finish_pred_ms=%.3f phone_finish_pred_ms=%.3f "
                        "return_finish_pred_ms=%.3f tail_pred_ms=%.3f "
                        "pipeline_done_pred_ms=%.3f\n",
                        job.ubatch_id,
                        validate_prediction.tensor_phone_primary ? "PHONE" : "PC",
                        validate_prediction.tokens,
                        validate_prediction.tensor_layers,
                        validate_prediction.tensor_pc_ratio,
                        validate_prediction.tensor_chunk_tokens,
                        validate_prediction.tensor_chunks,
                        actual_ms,
                        validate_prediction.tensor_total_ms,
                        actual_over_pred,
                        tensor_stage_timing.prepare_us / 1000.0,
                        tensor_stage_timing.compute_range_us / 1000.0,
                        tensor_stage_timing.sync_us / 1000.0,
                        validate_prediction.attn_misc_ms,
                        validate_prediction.pipeline_h2d_sum_ms,
                        validate_prediction.pipeline_pc_ffn_sum_ms,
                        validate_prediction.pipeline_phone_sum_ms,
                        validate_prediction.pipeline_phone_start_ms,
                        validate_prediction.pipeline_d2h_sum_ms,
                        validate_prediction.pipeline_h2d_finish_ms,
                        validate_prediction.pipeline_pc_finish_ms,
                        validate_prediction.pipeline_phone_finish_ms,
                        validate_prediction.pipeline_return_finish_ms,
                        validate_prediction.pipeline_reduce_tail_ms,
                        validate_prediction.pipeline_done_ms);
                }
            }
            LLAMA_LOG_DEBUG(
                "[TENSOR_STAGE_TIMING] ub=%d total=%.3f prepare=%.3f compute_range=%.3f "
                "sync=%.3f unaccounted=%.3f meta_total=%.3f meta_rebuild=%.3f "
                "meta_execute=%.3f meta_other=%.3f meta_calls=%" PRId64
                " meta_rebuilds=%" PRId64 " blocks=%zu\n",
                job.ubatch_id,
                tensor_total_us / 1000.0,
                tensor_stage_timing.prepare_us / 1000.0,
                tensor_stage_timing.compute_range_us / 1000.0,
                tensor_stage_timing.sync_us / 1000.0,
                tensor_stage_unaccounted_us / 1000.0,
                profile.graph_total_us / 1000.0,
                profile.graph_rebuild_us / 1000.0,
                profile.graph_execute_us / 1000.0,
                profile.graph_other_us / 1000.0,
                profile.graph_compute_count,
                profile.graph_rebuild_count,
                blocks.size());
            LLAMA_LOG_DEBUG(
                "[TENSOR_BREAKDOWN] ub=%d total=%.3f attn=%.3f pc_ffn=%.3f h2d=%.3f "
                "phone=%.3f d2h=%.3f reduce=%.3f wait=%.3f ms\n",
                job.ubatch_id, tensor_total_us / 1000.0,
                profile.attn_us / 1000.0, profile.pc_ffn_us / 1000.0,
                profile.h2d_us / 1000.0, profile.phone_us / 1000.0,
                profile.d2h_us / 1000.0, profile.reduce_us / 1000.0,
                profile.wait_us / 1000.0);
            llama_hybrid_tensor_compute_prediction tensor_prediction;
            if (llama_hybrid_runtime_predict_tensor_compute(
                    (int) job.ubatch.n_tokens, tensor_prediction)) {
                LLAMA_LOG_DEBUG(
                    "[PRED_TENSOR_COMPUTE] ub=%d tokens=%d kv_tokens=%d T=%d XT=%d GA=%d XA=%d R=%.3f "
                    "attn_misc_ms=%.3f pc_ffn_ms=%.3f pc_compute_ms=%.3f tensor_total_ms=%.3f\n",
                    job.ubatch_id,
                    tensor_prediction.tokens,
                    tensor_prediction.kv_tokens,
                    tensor_prediction.tensor_layers,
                    tensor_prediction.tensor_chunk_tokens,
                    tensor_prediction.attn_group_chunks,
                    tensor_prediction.attn_chunk_tokens,
                    tensor_prediction.tensor_pc_ratio,
                    tensor_prediction.attn_misc_ms,
                    tensor_prediction.pc_ffn_ms,
                    tensor_prediction.pc_compute_ms,
                    tensor_prediction.tensor_total_ms);

                LLAMA_LOG_DEBUG(
                    "[PRED_TENSOR_PIPE_DETAIL] ub=%d chunks=%d "
                    "h2d_sum_ms=%.3f pc_ffn_sum_ms=%.3f phone_sum_ms=%.3f d2h_sum_ms=%.3f "
                    "h2d_finish_ms=%.3f pc_finish_ms=%.3f phone_finish_ms=%.3f "
                    "return_finish_ms=%.3f reduce_tail_ms=%.3f pipeline_done_ms=%.3f "
                    "overlap_saved_ms=%.3f\n",
                    job.ubatch_id,
                    tensor_prediction.tensor_chunks,
                    tensor_prediction.pipeline_h2d_sum_ms,
                    tensor_prediction.pipeline_pc_ffn_sum_ms,
                    tensor_prediction.pipeline_phone_sum_ms,
                    tensor_prediction.pipeline_d2h_sum_ms,
                    tensor_prediction.pipeline_h2d_finish_ms,
                    tensor_prediction.pipeline_pc_finish_ms,
                    tensor_prediction.pipeline_phone_finish_ms,
                    tensor_prediction.pipeline_return_finish_ms,
                    tensor_prediction.pipeline_reduce_tail_ms,
                    tensor_prediction.pipeline_done_ms,
                    tensor_prediction.pipeline_overlap_saved_ms);

                const auto actual_over_pred = [](double actual_ms, double pred_ms) {
                    return pred_ms > 0.0 ? actual_ms / pred_ms : 0.0;
                };
                LLAMA_LOG_DEBUG(
                    "[PRED_TENSOR_ERROR] ub=%d actual_over_pred "
                    "compute=%.3f pc_ffn=%.3f h2d=%.3f phone=%.3f d2h=%.3f\n",
                    job.ubatch_id,
                    actual_over_pred(
                        tensor_stage_timing.compute_range_us / 1000.0,
                        tensor_prediction.tensor_total_ms),
                    actual_over_pred(
                        profile.pc_ffn_us / 1000.0,
                        tensor_prediction.pipeline_pc_ffn_sum_ms),
                    actual_over_pred(
                        profile.h2d_us / 1000.0,
                        tensor_prediction.pipeline_h2d_sum_ms),
                    actual_over_pred(
                        profile.phone_us / 1000.0,
                        tensor_prediction.pipeline_phone_sum_ms),
                    actual_over_pred(
                        profile.d2h_us / 1000.0,
                        tensor_prediction.pipeline_d2h_sum_ms));
            }
            LLAMA_LOG_DEBUG(
                "[PREFILL_RETURN_STALL] lane_reuse_count=%" PRId64 " lane_reuse_ms=%.3f "
                "lane_reuse_max_ms=%.3f lane0_count=%" PRId64 " lane0_ms=%.3f "
                "lane1_count=%" PRId64 " lane1_ms=%.3f layer_barrier_count=%" PRId64 " "
                "layer_barrier_ms=%.3f layer_barrier_max_ms=%.3f max_layer=%" PRId64 " "
                "max_pending=%" PRId64 " max_last_lane=%" PRId64 "\n",
                profile.lane_reuse_wait_count, profile.lane_reuse_wait_us / 1000.0,
                profile.lane_reuse_wait_max_us / 1000.0,
                profile.lane_reuse_wait_count_by_lane[0], profile.lane_reuse_wait_us_by_lane[0] / 1000.0,
                profile.lane_reuse_wait_count_by_lane[1], profile.lane_reuse_wait_us_by_lane[1] / 1000.0,
                profile.layer_barrier_wait_count, profile.layer_barrier_wait_us / 1000.0,
                profile.layer_barrier_wait_max_us / 1000.0, profile.layer_barrier_wait_max_layer,
                profile.layer_barrier_wait_max_pending, profile.layer_barrier_wait_max_last_lane);

            if (profile.return_transfer_count > 0) {
                const int64_t exposed_wait_us =
                    profile.lane_reuse_wait_us +
                    profile.layer_barrier_wait_us;
                const int64_t overlap_est_us =
                    std::max<int64_t>(
                        0, profile.return_rpc_total_us -
                            exposed_wait_us);
                const double overlap_ratio =
                    profile.return_rpc_total_us > 0 ?
                        (double) overlap_est_us /
                            (double) profile.return_rpc_total_us :
                        0.0;
                LLAMA_LOG_DEBUG(
                    "[RETURN_CRITICAL_PATH] ub=%d returns=%" PRId64
                    " payload_mib=%.3f request_ms=%.3f ready_wait_ms=%.3f "
                    "recv_payload_ms=%.3f rpc_total_ms=%.3f "
                    "d2h_accounted_ms=%.3f lane_wait_ms=%.3f "
                    "barrier_wait_ms=%.3f exposed_wait_ms=%.3f "
                    "overlap_est_ms=%.3f overlap_ratio=%.3f\n",
                    job.ubatch_id,
                    profile.return_transfer_count,
                    profile.return_payload_bytes / 1048576.0,
                    profile.return_request_us / 1000.0,
                    profile.return_ready_wait_us / 1000.0,
                    profile.return_recv_payload_us / 1000.0,
                    profile.return_rpc_total_us / 1000.0,
                    profile.d2h_us / 1000.0,
                    profile.lane_reuse_wait_us / 1000.0,
                    profile.layer_barrier_wait_us / 1000.0,
                    exposed_wait_us / 1000.0,
                    overlap_est_us / 1000.0,
                    overlap_ratio);
            }
        }
        LLAMA_LOG_DEBUG(
            "[HYBRID_PIPE] ub=%d %s_PHASE_END stage=%zu\n",
            job.ubatch_id, llama_hybrid_runtime_stage_name(stage.kind), stage_index);
        return result;
    };

    const auto hybrid_run_cpu_one = [&](ggml_status & status) {
        GGML_ASSERT(!hybrid_cpu_ready_q.empty());

        llama_hybrid_job job = std::move(hybrid_cpu_ready_q.front());
        hybrid_cpu_ready_q.pop_front();
        GGML_ASSERT(job.stage_index < runtime_stages.size());
        GGML_ASSERT(runtime_stages[job.stage_index].kind == llama_hybrid_runtime_stage_kind::CPU);

        hybrid_run_downstream_stage(job, status);
        if (status != GGML_STATUS_SUCCESS) {
            return;
        }

        if (job.stage_index < runtime_stages.size()) {
            hybrid_enqueue_ready(std::move(job), status);
        }
    };

    const auto hybrid_run_tensor_one = [&](ggml_status & status) -> llm_graph_result * {
        GGML_ASSERT(!hybrid_tensor_ready_q.empty());

        const size_t tensor_bytes = hybrid_tensor_ready_q.front().hidden.capacity() * sizeof(float);
        GGML_ASSERT(hybrid_tensor_ready_bytes >= tensor_bytes);
        hybrid_tensor_ready_bytes -= tensor_bytes;

        llama_hybrid_job job = std::move(hybrid_tensor_ready_q.front());
        hybrid_tensor_ready_q.pop_front();
        GGML_ASSERT(job.stage_index < runtime_stages.size());
        GGML_ASSERT(runtime_stages[job.stage_index].kind == llama_hybrid_runtime_stage_kind::TENSOR);

        LLAMA_LOG_DEBUG(
            "[HYBRID_PIPE] ub=%d TENSOR_DEQUEUE job_mib=%.2f queue_mib=%.2f\n",
            job.ubatch_id, tensor_bytes / 1048576.0, hybrid_tensor_ready_bytes / 1048576.0);

        llm_graph_result * result = hybrid_run_downstream_stage(job, status);
        if (result == nullptr || status != GGML_STATUS_SUCCESS) {
            return nullptr;
        }

        if (job.stage_index < runtime_stages.size()) {
            hybrid_enqueue_ready(std::move(job), status);
            return nullptr;
        }

        return result;
    };

    const auto hybrid_run_phone_one = [&](ggml_status & status) -> llm_graph_result * {
        GGML_ASSERT(!hybrid_phone_ready_q.empty());

        const size_t phone_bytes = hybrid_phone_ready_q.front().hidden.capacity() * sizeof(float);
        GGML_ASSERT(hybrid_phone_ready_bytes >= phone_bytes);
        hybrid_phone_ready_bytes -= phone_bytes;

        llama_hybrid_job job = std::move(hybrid_phone_ready_q.front());
        hybrid_phone_ready_q.pop_front();
        GGML_ASSERT(job.stage_index < runtime_stages.size());
        GGML_ASSERT(runtime_stages[job.stage_index].kind == llama_hybrid_runtime_stage_kind::PHONE);

        LLAMA_LOG_DEBUG(
            "[HYBRID_PIPE] ub=%d PHONE_DEQUEUE job_mib=%.2f queue_mib=%.2f\n",
            job.ubatch_id, phone_bytes / 1048576.0, hybrid_phone_ready_bytes / 1048576.0);

        llm_graph_result * result = hybrid_run_downstream_stage(job, status);
        if (result == nullptr || status != GGML_STATUS_SUCCESS) {
            return nullptr;
        }

        GGML_ASSERT(job.stage_index == runtime_stages.size());
        return result;
    };

    const auto hybrid_run_ready_one = [&](ggml_status & status) -> llm_graph_result * {
        if (!hybrid_has_ready()) {
            return nullptr;
        }

        const bool phone_pressure = hybrid_phone_ready_bytes >= hybrid_phone_queue_limit_bytes;
        const bool tensor_pressure = hybrid_tensor_ready_bytes >= hybrid_tensor_queue_limit_bytes;

        if (phone_pressure && !hybrid_phone_ready_q.empty()) {
            LLAMA_LOG_DEBUG(
                "[HYBRID_SCHED] choose=PHONE reason=HWM cpu_q=%zu tensor_q=%zu phone_q=%zu phone_mib=%.2f\n",
                hybrid_cpu_ready_q.size(), hybrid_tensor_ready_q.size(), hybrid_phone_ready_q.size(),
                hybrid_phone_ready_bytes / 1048576.0);
            return hybrid_run_phone_one(status);
        }

        if (tensor_pressure && !hybrid_tensor_ready_q.empty()) {
            LLAMA_LOG_DEBUG(
                "[HYBRID_SCHED] choose=TENSOR reason=HWM cpu_q=%zu tensor_q=%zu phone_q=%zu tensor_mib=%.2f\n",
                hybrid_cpu_ready_q.size(), hybrid_tensor_ready_q.size(), hybrid_phone_ready_q.size(),
                hybrid_tensor_ready_bytes / 1048576.0);
            return hybrid_run_tensor_one(status);
        }

        if (!hybrid_cpu_ready_q.empty()) {
            LLAMA_LOG_DEBUG(
                "[HYBRID_SCHED] choose=CPU reason=PRIORITY cpu_q=%zu tensor_q=%zu tensor_mib=%.2f\n",
                hybrid_cpu_ready_q.size(), hybrid_tensor_ready_q.size(),
                hybrid_tensor_ready_bytes / 1048576.0);
            hybrid_run_cpu_one(status);
            return nullptr;
        }

        if (!hybrid_tensor_ready_q.empty()) {
            return hybrid_run_tensor_one(status);
        }

        GGML_ASSERT(!hybrid_phone_ready_q.empty());
        return hybrid_run_phone_one(status);
    };

    do {
        const auto & ubatch = mctx->get_ubatch();
        // 拿出要处理的一个 ubatch
        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }
        // 统计有多少个输出 token ，构建计算图依赖n_outputs
        const int            slot_id   = ubatch_id & 1;
        ggml_backend_sched_t sched_use = pipeline_runtime_enabled ? pipe_slots[slot_id].sched : sched.get();
        llm_graph_result *   res_use   = pipeline_runtime_enabled ? pipe_slots[slot_id].res : gf_res_prev.get();

        if (std::getenv("LLAMA_PIPELINE_DEBUG") != nullptr || std::getenv("GGML_META_PIPELINE_DEBUG") != nullptr) {
            printf("[PIPE_SLOT] ub=%d slot=%d sched=%p res=%p\n", ubatch_id, slot_id, (void *) sched_use,
                   (void *) res_use);
        }

        bool generalized_stage_queue_candidate =
            generalized_stage_queue_runtime_enabled && cparams.causal_attn && ubatch.n_tokens > 1 &&
            ubatch.token != nullptr && ubatch.embd == nullptr && ubatch.n_pos == 1 && !ubatch.equal_seqs() &&
            ubatch.n_seqs_unq == 1 && !cparams.embeddings && !cparams.embeddings_nextn &&
            cparams.cb_eval == nullptr &&
            std::none_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(),
                         [](bool enabled) { return enabled; });

        if (generalized_stage_queue_candidate && n_outputs > 0) {
            const uint32_t final_macro = runtime_stages.back().macro_tokens;
            const uint32_t final_block_begin = (ubatch.n_tokens - 1) / final_macro * final_macro;
            for (uint32_t i = 0; i < final_block_begin; ++i) {
                if (ubatch.output[i] != 0) {
                    generalized_stage_queue_candidate = false;
                    break;
                }
            }
        }

        bool stage_serial_candidate =
            stage_serial_runtime_enabled && cparams.causal_attn && ubatch.n_tokens > 1 && ubatch.token != nullptr &&
            ubatch.embd == nullptr && ubatch.n_pos == 1 && !ubatch.equal_seqs() && ubatch.n_seqs_unq == 1 &&
            !cparams.embeddings && !cparams.embeddings_nextn && cparams.cb_eval == nullptr &&
            std::none_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(),
                         [](bool enabled) { return enabled; });

        if (stage_serial_candidate && n_outputs > 0) {
            const uint32_t final_macro = runtime_stages.back().macro_tokens;
            const uint32_t final_block_begin = (ubatch.n_tokens - 1) / final_macro * final_macro;
            for (uint32_t i = 0; i < final_block_begin; ++i) {
                if (ubatch.output[i] != 0) {
                    stage_serial_candidate = false;
                    break;
                }
            }
        }

        if (stage_serial_runtime_enabled && !stage_serial_candidate) {
            LLAMA_LOG_WARN(
                "[HYBRID_EXEC] ub=%d serial stage path unsupported for this ubatch; using full graph\n",
                ubatch_id);
        }

        if (region_ring_v1 && ring_exec == "full" && ubatch.n_tokens > 1) {
            LLAMA_LOG_ERROR(
                "[REGION_RING_EXEC] mode=full ub=%d tokens=%u "
                "path=DEFAULT_GGML_FULL_GRAPH legacy_pipeline=DISABLED\n",
                ubatch_id, ubatch.n_tokens);
        }

        const bool pipe_candidate =
            pipeline_runtime_enabled && cparams.causal_attn && ubatch.n_tokens > 1 && n_outputs == 0 &&
            !cparams.embeddings && !cparams.embeddings_nextn && cparams.cb_eval == nullptr &&
            std::none_of(cparams.embeddings_layer_inp.begin(), cparams.embeddings_layer_inp.end(),
                         [](bool enabled) { return enabled; });

        ggml_status              status       = GGML_STATUS_SUCCESS;
        const llm_graph_result * res          = nullptr;
        bool                     pipe_success = false;

        if (generalized_stage_queue_runtime_enabled && !generalized_stage_queue_candidate) {
            LLAMA_LOG_WARN(
                "[HYBRID_PIPE] ub=%d generalized stage queue unsupported for this ubatch; using full graph\n",
                ubatch_id);
            const int32_t n_outputs_saved = n_outputs;
            if (hybrid_gpu_pending_active) {
                hybrid_gpu_harvest(status);
            }
            while (status == GGML_STATUS_SUCCESS && hybrid_has_ready()) {
                hybrid_run_ready_one(status);
            }
            n_outputs = n_outputs_saved;
            if (status == GGML_STATUS_SUCCESS) {
                res = process_ubatch(
                    ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), sched.get(), gf_res_prev.get(),
                    ubatch_id, status);
                sched_use = sched.get();
            }
        } else if (generalized_stage_queue_candidate) {
            const int32_t n_outputs_saved = n_outputs;

            if (hybrid_gpu_pending_active) {
                while (status == GGML_STATUS_SUCCESS &&
                       hybrid_cpu_ready_q.size() >= hybrid_cpu_ready_max) {
                    hybrid_run_ready_one(status);
                }
                if (status == GGML_STATUS_SUCCESS) {
                    hybrid_gpu_harvest(status);
                }
            }
            if (status == GGML_STATUS_SUCCESS) {
                hybrid_gpu_submit(ubatch, ubatch_id, n_outputs_saved, status);
            }
            if (status == GGML_STATUS_SUCCESS && hybrid_has_ready()) {
                if (llm_graph_result * ready_res = hybrid_run_ready_one(status)) {
                    res = ready_res;
                }
            }

            if (status == GGML_STATUS_SUCCESS && n_outputs_saved > 0) {
                if (hybrid_gpu_pending_active) {
                    hybrid_gpu_harvest(status);
                }
                while (status == GGML_STATUS_SUCCESS && hybrid_has_ready()) {
                    if (llm_graph_result * ready_res = hybrid_run_ready_one(status)) {
                        res = ready_res;
                    }
                }
                if (status == GGML_STATUS_SUCCESS) {
                    sched_use = sched_pipe.get();
                    res_use   = gf_res_pipe.get();
                }
            } else if (status == GGML_STATUS_SUCCESS) {
                pipe_success = true;
            }
            n_outputs = n_outputs_saved;

            if (status != GGML_STATUS_SUCCESS) {
                synchronize();
                hybrid_gpu_pending_active = false;
                hybrid_cpu_ready_q.clear();
                hybrid_tensor_ready_q.clear();
                hybrid_phone_ready_q.clear();
                hybrid_tensor_ready_bytes = 0;
                hybrid_phone_ready_bytes = 0;
            }
        } else if (region_ring_v1 && ring_exec != "full" &&
                   ubatch.n_tokens > 1 &&
                   cparams.causal_attn && ubatch.token != nullptr &&
                   ubatch.embd == nullptr && ubatch.n_pos == 1 &&
                   !ubatch.equal_seqs() && ubatch.n_seqs_unq == 1 &&
                   !cparams.embeddings && !cparams.embeddings_nextn &&
                   cparams.cb_eval == nullptr) {
            // A/B diagnostics with IDENTICAL model layer ownership:
            // ready  = interleaved Region x Chunk execution;
            // fifo   = single-threaded simulation with per-device claims;
            // fifo-isolated = FIFO + separate CPU/GPU scheduler and graph states;
            // fifo-owned = per-worker KV views + gated backends + 2-slot credits;
            // fifo-state = explicit per-task graph output count; same Backend gate;
            // fifo-workers = three real host threads, one shared Backend graph gate;
            // fifo-backend-audit = workers + actual split/resource/Meta child audit;
            // fifo-pc-phone = PC CPU+CUDA shared execution, Meta/RPC independent;
            // serial = all chunks of one Region before the next Region.
            // Full-graph mode falls through to the existing ggml path.
            LLAMA_LOG_ERROR(
                "[REGION_RING_EXEC] mode=%s ub=%d tokens=%u\n",
                ring_exec.c_str(), ubatch_id, ubatch.n_tokens);
            if (ring_exec == "serial") {
                res = process_ubatch_staged(
                    ubatch, ctx_type_to_graph_type(cparams.ctx_type),
                    mctx.get(), sched.get(), gf_res_prev.get(),
                    runtime_stages, ubatch_id, status);
            } else {
                res = process_ubatch_region_ring(
                    ubatch, ctx_type_to_graph_type(cparams.ctx_type),
                    mctx.get(), sched.get(), gf_res_prev.get(),
                    runtime_stages, ubatch_id,
                    ring_exec == "fifo" || ring_exec == "fifo-isolated" ||
                        ring_exec == "fifo-owned" || ring_exec == "fifo-state" ||
                        ring_exec == "fifo-workers" || ring_exec == "fifo-backend-audit" ||
                        ring_exec == "fifo-pc-phone",
                    ring_exec == "fifo-isolated" || ring_exec == "fifo-owned" ||
                        ring_exec == "fifo-state" || ring_exec == "fifo-workers" ||
                        ring_exec == "fifo-backend-audit" || ring_exec == "fifo-pc-phone",
                    ring_exec == "fifo-owned" || ring_exec == "fifo-state" ||
                        ring_exec == "fifo-workers" || ring_exec == "fifo-backend-audit" ||
                        ring_exec == "fifo-pc-phone",
                    ring_exec == "fifo-state" || ring_exec == "fifo-workers" ||
                        ring_exec == "fifo-backend-audit" || ring_exec == "fifo-pc-phone",
                    ring_exec == "fifo-workers" || ring_exec == "fifo-backend-audit" ||
                        ring_exec == "fifo-pc-phone",
                    ring_exec == "fifo-backend-audit" || ring_exec == "fifo-pc-phone",
                    ring_exec == "fifo-pc-phone", status);
            }
        } else if (stage_serial_candidate) {
            res = process_ubatch_staged(
                ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), sched.get(), gf_res_prev.get(),
                runtime_stages, ubatch_id, status);
        } else if (pipe_candidate) {
            auto & cur = pipe_slots[slot_id];
            GGML_ASSERT(!cur.active);

            const bool pipe_ready = prepare_pipe_slot(
                cur, ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), ubatch_id,
                stage_queue_runtime_enabled, status);

            if (!pipe_ready && status == GGML_STATUS_SUCCESS) {
                pipeline_runtime_enabled = false;
                legacy_pipeline_runtime_enabled = false;

                if (stage_queue_runtime_enabled) {
                    status = stage_queue_drain();
                    stage_queue_runtime_enabled = false;
                } else if (prev != nullptr) {
                    status = pipe_prepare_post(*prev);

                    if (status == GGML_STATUS_SUCCESS) {
                        status = pipe_run_post_prepared(*prev);
                    }

                    if (status == GGML_STATUS_SUCCESS) {
                        pipe_sync_post(*prev);
                    }

                    if (status == GGML_STATUS_SUCCESS) {
                        status = pipe_run_tail(*prev);
                    }

                    if (status == GGML_STATUS_SUCCESS) {
                        prev = nullptr;
                    }
                }

                if (status == GGML_STATUS_SUCCESS) {
                    sched_use = sched.get();
                    res_use   = gf_res_prev.get();

                    res_use->reset();
                    ggml_backend_sched_reset(sched_use);

                    res_use = prepare_ubatch(res_use, sched_use, ubatch, ctx_type_to_graph_type(cparams.ctx_type),
                                             mctx.get(), status, false);

                    if (res_use != nullptr && status == GGML_STATUS_SUCCESS) {
                        const int n_splits = ggml_backend_sched_get_n_splits(sched_use);
                        status = graph_compute_range(sched_use, 0, n_splits, ubatch.n_tokens > 1);

                        if (status == GGML_STATUS_SUCCESS) {
                            res = res_use;
                        }
                    }
                }
            } else if (pipe_ready && stage_queue_runtime_enabled) {
                status = pipe_run_pre(cur);

                if (status == GGML_STATUS_SUCCESS && !handoff_q.empty()) {
                    llama_prefill_pipe_slot * task = handoff_q.front();
                    handoff_q.pop_front();
                    status = stage_queue_handoff(*task);
                    if (status == GGML_STATUS_SUCCESS) {
                        tensor_ready_q.push_back(task);
                    }
                }

                if (status == GGML_STATUS_SUCCESS) {
                    status = pipe_run_gpu(cur);
                    if (status == GGML_STATUS_SUCCESS) {
                        LLAMA_LOG_DEBUG("[STAGEQ] ub=%d slot=%d GPU_SUBMIT tokens=%u\n", cur.ubatch_id,
                                        slot_id, cur.n_tokens);
                        handoff_q.push_back(&cur);
                    }
                }

                if (status == GGML_STATUS_SUCCESS && !tensor_ready_q.empty()) {
                    llama_prefill_pipe_slot * task = tensor_ready_q.front();
                    tensor_ready_q.pop_front();
                    status = stage_queue_run_tensor(*task);
                }

                if (status == GGML_STATUS_SUCCESS) {
                    pipe_success = true;
                }
            } else if (pipe_ready) {
                status = pipe_run_pre(cur);

                if (status == GGML_STATUS_SUCCESS && prev != nullptr) {
                    status = pipe_prepare_post(*prev);
                }

                if (status == GGML_STATUS_SUCCESS) {
                    status = pipe_run_gpu(cur);
                }

                if (status == GGML_STATUS_SUCCESS && prev != nullptr) {
                    status = pipe_run_post_prepared(*prev);
                    if (status == GGML_STATUS_SUCCESS) {
                        pipe_sync_post(*prev);
                    }

                    if (status == GGML_STATUS_SUCCESS) {
                        status = pipe_run_tail(*prev);
                    }
                }

                if (status == GGML_STATUS_SUCCESS) {
                    prev         = &cur;
                    pipe_success = true;
                }
            }

            if (status != GGML_STATUS_SUCCESS) {
                synchronize();
                for (auto & slot : pipe_slots) {
                    slot.active        = false;
                    slot.post_prepared = false;
                }
                prev = nullptr;
                handoff_q.clear();
                tensor_ready_q.clear();
            }
        } else {
            if (stage_queue_runtime_enabled) {
                status = stage_queue_drain();
            } else if (prev != nullptr) {
                status = pipe_prepare_post(*prev);
                if (status == GGML_STATUS_SUCCESS) {
                    status = pipe_run_post_prepared(*prev);
                }
                if (status == GGML_STATUS_SUCCESS) {
                    pipe_sync_post(*prev);
                }
                if (status == GGML_STATUS_SUCCESS) {
                    status = pipe_run_tail(*prev);
                }
                if (status == GGML_STATUS_SUCCESS) {
                    prev = nullptr;
                }
            }

            if (status == GGML_STATUS_SUCCESS) {
                res = process_ubatch(ubatch, ctx_type_to_graph_type(cparams.ctx_type), mctx.get(), sched_use, res_use,
                                     ubatch_id, status);
            }
        }

        if (pipe_success) {
            n_tokens_prev += ubatch.n_tokens;
            ubatch_id++;
            continue;
        }
        // 根据当前ubatch构建计算图并执行计算，返回结果res
        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s,
                               pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                    return -3;
                case GGML_STATUS_SUCCESS:
                    GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits  = res->get_logits();
        auto * t_embd    = cparams.embeddings ? res->get_embd() : nullptr;
        auto * t_h_nextn = cparams.embeddings_nextn ? res->get_h_nextn() : nullptr;
        //
        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched_use, t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev * n_vocab;

            if (n_outputs) {
                GGML_ASSERT(n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs) * n_vocab <= (int64_t) logits.size);

                const int64_t output_submit_begin_us =
                    decode_runtime_profile.pending ?
                        ggml_time_us() : 0;
                const size_t logits_copy_bytes =
                    (size_t) n_outputs * (size_t) n_vocab * sizeof(float);
                ggml_backend_tensor_get_async(
                    backend_res,
                    t_logits,
                    logits_out,
                    0,
                    logits_copy_bytes);

                const bool decode_top5_trace =
                    ubatch.n_tokens == 1 &&
                    (std::getenv("LLAMA_DECODE_TOP5_TRACE") != nullptr ||
                     std::getenv("GGML_META_REGION_C2P_SELF_CHECK") != nullptr);
                if (decode_top5_trace) {
                    // Diagnostics only: make the async host copy visible before
                    // inspecting scores. This intentionally adds a decode fence.
                    ggml_backend_synchronize(backend_res);

                    struct top_entry {
                        float logit;
                        llama_token token;
                    };
                    std::array<top_entry, 5> top;
                    for (auto & entry : top) {
                        entry.logit =
                            -std::numeric_limits<float>::infinity();
                        entry.token = LLAMA_TOKEN_NULL;
                    }

                    const float * row =
                        logits_out +
                        ((size_t) n_outputs - 1) * (size_t) n_vocab;
                    for (llama_token tok = 0;
                            tok < (llama_token) n_vocab; ++tok) {
                        const float value = row[tok];
                        if (!std::isfinite(value) ||
                                value <= top.back().logit) {
                            continue;
                        }
                        size_t pos = top.size() - 1;
                        while (pos > 0 &&
                                value > top[pos - 1].logit) {
                            top[pos] = top[pos - 1];
                            --pos;
                        }
                        top[pos] = { value, tok };
                    }

                    auto escape_piece = [](const std::string & piece) {
                        std::string out;
                        out.reserve(piece.size() + 8);
                        for (unsigned char ch : piece) {
                            switch (ch) {
                                case '\n': out += "\\n"; break;
                                case '\r': out += "\\r"; break;
                                case '\t': out += "\\t"; break;
                                case '\\': out += "\\\\"; break;
                                case '"': out += "\\\""; break;
                                default: out.push_back((char) ch); break;
                            }
                        }
                        return out;
                    };

                    static thread_local uint64_t decode_top5_seq = 0;
                    const uint64_t seq = decode_top5_seq++;
                    printf(
                        "========== DECODE TOP 5 decoded=%" PRIu64
                        " ==========\n",
                        seq);
                    for (size_t rank = 0; rank < top.size(); ++rank) {
                        const llama_token tok = top[rank].token;
                        const std::string piece =
                            tok == LLAMA_TOKEN_NULL ?
                                std::string() :
                                model.vocab.token_to_piece(tok);
                        printf(
                            "[DECODE_TOP5] decoded=%" PRIu64
                            " rank=%zu token=%d logit=%.9f "
                            "piece=\"%s\" raw_argmax=%d\n",
                            seq,
                            rank,
                            (int) tok,
                            top[rank].logit,
                            escape_piece(piece).c_str(),
                            rank == 0 ? 1 : 0);
                    }
                }

                // The server samples the first generated token from the raw
                // logits produced by the prompt graph.  Keep an explicit
                // backend fence on this prefill output copy while diagnosing
                // the rare all-zero first-token logits.  llama_context::
                // synchronize() eventually fences the scheduler too, but a
                // direct fence here makes the output-copy dependency explicit
                // and prevents a stale zero-filled host row from reaching the
                // sampler.
                if (ubatch.n_tokens > 1) {
                    const int64_t logits_sync_begin_us = ggml_time_us();
                    ggml_backend_synchronize(backend_res);
                    const int64_t logits_sync_us =
                        ggml_time_us() - logits_sync_begin_us;

                    const size_t debug_row = (size_t) n_outputs - 1;
                    const size_t debug_count = (size_t) n_vocab;
                    const float * host_row =
                        logits_out + debug_row * debug_count;

                    auto collect_logits_stats = [](const float * data, size_t count,
                                                   size_t & zero, size_t & nonfinite,
                                                   float & min_value, float & max_value) {
                        zero = 0;
                        nonfinite = 0;
                        min_value = std::numeric_limits<float>::infinity();
                        max_value = -std::numeric_limits<float>::infinity();
                        for (size_t i = 0; i < count; ++i) {
                            const float value = data[i];
                            zero += value == 0.0f;
                            if (!std::isfinite(value)) {
                                ++nonfinite;
                                continue;
                            }
                            min_value = std::min(min_value, value);
                            max_value = std::max(max_value, value);
                        }
                        if (!std::isfinite(min_value)) {
                            min_value = 0.0f;
                        }
                        if (!std::isfinite(max_value)) {
                            max_value = 0.0f;
                        }
                    };

                    size_t host_zero = 0;
                    size_t host_nonfinite = 0;
                    float host_min = 0.0f;
                    float host_max = 0.0f;
                    collect_logits_stats(
                        host_row, debug_count,
                        host_zero, host_nonfinite,
                        host_min, host_max);

                    // Read the same output row directly from the source tensor
                    // after the fence.  Comparing source vs host distinguishes
                    // an upstream graph/output failure from a broken or late
                    // backend->host copy.
                    std::vector<float> source_row(debug_count);
                    const size_t debug_offset =
                        debug_row * debug_count * sizeof(float);
                    ggml_backend_tensor_get(
                        t_logits,
                        source_row.data(),
                        debug_offset,
                        debug_count * sizeof(float));

                    size_t source_zero = 0;
                    size_t source_nonfinite = 0;
                    float source_min = 0.0f;
                    float source_max = 0.0f;
                    collect_logits_stats(
                        source_row.data(), debug_count,
                        source_zero, source_nonfinite,
                        source_min, source_max);

                    float max_abs_diff = 0.0f;
                    size_t mismatch_count = 0;
                    for (size_t i = 0; i < debug_count; ++i) {
                        const float a = host_row[i];
                        const float b = source_row[i];
                        if (!std::isfinite(a) || !std::isfinite(b)) {
                            if (!(std::isnan(a) && std::isnan(b)) && a != b) {
                                ++mismatch_count;
                            }
                            continue;
                        }
                        const float diff = std::fabs(a - b);
                        max_abs_diff = std::max(max_abs_diff, diff);
                        mismatch_count += diff != 0.0f;
                    }

                    LLAMA_LOG_INFO(
                        "[PREFILL_LOGITS_COPY_CHECK] tokens=%u outputs=%d row=%zu "
                        "backend=%s count=%zu sync_ms=%.3f "
                        "host_zero=%zu host_nonfinite=%zu host_min=%.9g host_max=%.9g "
                        "source_zero=%zu source_nonfinite=%zu source_min=%.9g source_max=%.9g "
                        "mismatch=%zu max_abs_diff=%.9g\n",
                        ubatch.n_tokens, n_outputs, debug_row,
                        ggml_backend_name(backend_res), debug_count,
                        logits_sync_us / 1000.0,
                        host_zero, host_nonfinite, host_min, host_max,
                        source_zero, source_nonfinite, source_min, source_max,
                        mismatch_count, max_abs_diff);

                    if (host_zero == debug_count ||
                        source_zero == debug_count ||
                        host_nonfinite != 0 ||
                        source_nonfinite != 0 ||
                        mismatch_count != 0) {
                        LLAMA_LOG_ERROR(
                            "[PREFILL_LOGITS_COPY_SUSPECT] host_all_zero=%d source_all_zero=%d "
                            "host_nonfinite=%zu source_nonfinite=%zu mismatch=%zu "
                            "max_abs_diff=%.9g\n",
                            host_zero == debug_count ? 1 : 0,
                            source_zero == debug_count ? 1 : 0,
                            host_nonfinite, source_nonfinite,
                            mismatch_count, max_abs_diff);
                    }
                }

                if (decode_runtime_profile.pending) {
                    decode_runtime_profile.output_submit_us +=
                        ggml_time_us() -
                        output_submit_begin_us;
                    LLAMA_LOG_DEBUG(
                        "[DECODE_OUTPUT_SUBMIT] sample=%" PRId64
                        " backend=%s bytes=%zu submit_ms=%.3f\n",
                        decode_runtime_profile.sample_index,
                        ggml_backend_name(backend_res),
                        (size_t) n_outputs *
                            (size_t) n_vocab *
                            sizeof(float),
                        decode_runtime_profile.output_submit_us /
                            1000.0);
                }
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched_use, t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float *        embd_out   = embd.data + n_outputs_prev * n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT(n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs) * n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0,
                                                          n_outputs * n_embd_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                          (n_embd_out * seq_idx) * sizeof(float),
                                                          n_embd_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(),
                                                          (n_cls_out * seq_idx) * sizeof(float),
                                                          n_cls_out * sizeof(float));
                        }
                    }
                    break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens, sched_use);

        // extract nextn embeddings before
        // only meaningful in LLAMA_POOLING_TYPE_NONE (per-token); other pooling modes are ignored.
        {
            const bool    masked = cparams.embeddings_nextn_masked;
            const int64_t n_rows = masked ? n_outputs : (int64_t) ubatch.n_tokens;
            const int64_t offset = masked ? n_outputs_prev : n_tokens_prev;

            if (embd_nextn.data && t_h_nextn && n_rows > 0 && cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
                ggml_backend_t backend_h = ggml_backend_sched_get_tensor_backend(sched_use, t_h_nextn);
                GGML_ASSERT(backend_h != nullptr);

                const uint32_t n_embd         = hparams.n_embd_out();
                float *        embd_nextn_out = embd_nextn.data + offset * n_embd;

                GGML_ASSERT((offset + n_rows) * n_embd <= (int64_t) embd_nextn.size);
                ggml_backend_tensor_get_async(backend_h, t_h_nextn, embd_nextn_out, 0, n_rows * n_embd * sizeof(float));
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers &&
            (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride            = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched_use);

            copy_tensor_async_floats(res->t_sampled_logits, sampling.logits, stride, sampling.logits_count,
                                     seq_to_output_row, sched_use);
            copy_tensor_async_floats(res->t_sampled_probs, sampling.probs, stride, sampling.probs_count,
                                     seq_to_output_row, sched_use);
            copy_tensor_async_candidates(res->t_candidates, sampling.candidates, stride, sampling.candidates_count,
                                         seq_to_output_row, sched_use);
        }

        n_outputs_prev += n_outputs;
        n_tokens_prev += ubatch.n_tokens;
        ubatch_id++;
    } while (mctx->next());

    if (hybrid_gpu_pending_active || hybrid_has_ready()) {
        ggml_status status = GGML_STATUS_SUCCESS;
        if (hybrid_gpu_pending_active) {
            hybrid_gpu_harvest(status);
        }
        while (status == GGML_STATUS_SUCCESS && hybrid_has_ready()) {
            hybrid_run_ready_one(status);
        }
        if (status != GGML_STATUS_SUCCESS) {
            synchronize();
            switch (status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                    return -3;
                case GGML_STATUS_SUCCESS:
                    GGML_ABORT("should not happen");
            }
        }
    }

    if (stage_queue_runtime_enabled && (!handoff_q.empty() || !tensor_ready_q.empty())) {
        const ggml_status status = stage_queue_drain();
        if (status != GGML_STATUS_SUCCESS) {
            synchronize();
            switch (status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                    return -3;
                case GGML_STATUS_SUCCESS:
                    GGML_ABORT("should not happen");
            }
        }
    } else if (prev != nullptr) {
        ggml_status status = pipe_prepare_post(*prev);
        if (status == GGML_STATUS_SUCCESS) {
            status = pipe_run_post_prepared(*prev);
        }
        if (status == GGML_STATUS_SUCCESS) {
            pipe_sync_post(*prev);
        }
        if (status == GGML_STATUS_SUCCESS) {
            status = pipe_run_tail(*prev);
        }
        if (status == GGML_STATUS_SUCCESS) {
            prev = nullptr;
        } else {
            synchronize();
            switch (status) {
                case GGML_STATUS_ABORTED:
                    return 2;
                case GGML_STATUS_ALLOC_FAILED:
                    return -2;
                case GGML_STATUS_FAILED:
                    return -3;
                case GGML_STATUS_SUCCESS:
                    GGML_ABORT("should not happen");
            }
        }
    }

    if (n_tokens_all > 1 && return_wavefront_full_graph_override) {
        ggml_backend_t meta_backend = nullptr;
        ggml_backend_meta_tensor_profile real_profile {};
        for (ggml_backend_t backend : backend_ptrs) {
            if (backend != nullptr &&
                ggml_backend_meta_tensor_profile_get(backend, &real_profile)) {
                meta_backend = backend;
                break;
            }
        }

        if (meta_backend != nullptr && real_profile.wave_layer_start_count > 0) {
            const double real_ii_mean_ms =
                real_profile.wave_ii_count > 0 ?
                    (real_profile.wave_ii_sum_us / 1000.0) / real_profile.wave_ii_count : 0.0;
            const double real_ii_median_ms = real_profile.wave_ii_median_us / 1000.0;
            const double real_fill_ms      = real_profile.wave_fill_us / 1000.0;
            const double real_drain_ms     = real_profile.wave_drain_us / 1000.0;
            const double real_span_ms      = real_profile.wave_span_us / 1000.0;

            LLAMA_LOG_DEBUG(
                "[WAVE_REAL_II] expected_layers=%d layer_starts=%" PRId64
                " ii_count=%" PRId64 " fill_ms=%.3f ii_mean_ms=%.3f "
                "ii_median_ms=%.3f ii_min_ms=%.3f ii_max_ms=%.3f "
                "drain_ms=%.3f span_ms=%.3f meta_total_ms=%.3f "
                "compute_wall_ms=%.3f reduce_wall_ms=%.3f barrier_ms=%.3f\n",
                runtime_plan.tensor_layers,
                real_profile.wave_layer_start_count,
                real_profile.wave_ii_count,
                real_fill_ms,
                real_ii_mean_ms,
                real_ii_median_ms,
                real_profile.wave_ii_min_us / 1000.0,
                real_profile.wave_ii_max_us / 1000.0,
                real_drain_ms,
                real_span_ms,
                real_profile.meta_total_us / 1000.0,
                real_profile.compute_wall_us / 1000.0,
                real_profile.reduce_wall_us / 1000.0,
                real_profile.layer_barrier_wait_us / 1000.0);

            const double real_early_ii_mean_ms =
                real_profile.wave_early_ii_count > 0 ?
                    (real_profile.wave_early_ii_sum_us / 1000.0) /
                        real_profile.wave_early_ii_count : 0.0;
            const double real_late_ii_mean_ms =
                real_profile.wave_late_ii_count > 0 ?
                    (real_profile.wave_late_ii_sum_us / 1000.0) /
                        real_profile.wave_late_ii_count : 0.0;
            const double real_early_ii_median_ms =
                real_profile.wave_early_ii_median_us / 1000.0;
            const double real_late_ii_median_ms =
                real_profile.wave_late_ii_median_us / 1000.0;
            const double real_early_compute_per_layer_ms =
                real_profile.wave_early_layer_count > 0 ?
                    (real_profile.wave_early_compute_wall_us / 1000.0) /
                        real_profile.wave_early_layer_count : 0.0;
            const double real_late_compute_per_layer_ms =
                real_profile.wave_late_layer_count > 0 ?
                    (real_profile.wave_late_compute_wall_us / 1000.0) /
                        real_profile.wave_late_layer_count : 0.0;
            const double real_early_barrier_per_layer_ms =
                real_profile.wave_early_layer_count > 0 ?
                    (real_profile.wave_early_barrier_us / 1000.0) /
                        real_profile.wave_early_layer_count : 0.0;
            const double real_late_barrier_per_layer_ms =
                real_profile.wave_late_layer_count > 0 ?
                    (real_profile.wave_late_barrier_us / 1000.0) /
                        real_profile.wave_late_layer_count : 0.0;

            LLAMA_LOG_DEBUG(
                "[WAVE_REAL_DRIFT] early_ii_count=%" PRId64
                " early_ii_mean_ms=%.3f early_ii_median_ms=%.3f "
                "late_ii_count=%" PRId64
                " late_ii_mean_ms=%.3f late_ii_median_ms=%.3f ii_drift_ratio=%.4f "
                "early_layers=%" PRId64
                " early_compute_per_layer_ms=%.3f early_barrier_per_layer_ms=%.3f "
                "late_layers=%" PRId64
                " late_compute_per_layer_ms=%.3f late_barrier_per_layer_ms=%.3f "
                "compute_drift_ratio=%.4f barrier_drift_ratio=%.4f\n",
                real_profile.wave_early_ii_count,
                real_early_ii_mean_ms,
                real_early_ii_median_ms,
                real_profile.wave_late_ii_count,
                real_late_ii_mean_ms,
                real_late_ii_median_ms,
                real_early_ii_median_ms > 0.0 ?
                    real_late_ii_median_ms / real_early_ii_median_ms : 0.0,
                real_profile.wave_early_layer_count,
                real_early_compute_per_layer_ms,
                real_early_barrier_per_layer_ms,
                real_profile.wave_late_layer_count,
                real_late_compute_per_layer_ms,
                real_late_barrier_per_layer_ms,
                real_early_compute_per_layer_ms > 0.0 ?
                    real_late_compute_per_layer_ms / real_early_compute_per_layer_ms : 0.0,
                real_early_barrier_per_layer_ms > 0.0 ?
                    real_late_barrier_per_layer_ms / real_early_barrier_per_layer_ms : 0.0);

            llama_hybrid_wave_calibration calibration;
            if (hybrid_wave_probe_done &&
                llama_hybrid_runtime_wave_calibration_get(calibration) &&
                calibration.tokens == (int) n_tokens_all &&
                calibration.tensor_chunk_tokens == runtime_plan.tensor_chunk_tokens &&
                std::abs(calibration.tensor_pc_ratio - runtime_plan.tensor_pc_ratio) < 1e-4f) {
                LLAMA_LOG_DEBUG(
                    "[WAVE_II_COMPARE] probe_layers=%d real_layers=%" PRId64
                    " probe_ii_median_ms=%.3f real_ii_median_ms=%.3f ii_ratio=%.4f "
                    "probe_ii_mean_ms=%.3f real_ii_mean_ms=%.3f mean_ratio=%.4f "
                    "probe_drain_ms=%.3f real_drain_ms=%.3f drain_ratio=%.4f\n",
                    calibration.tensor_layers,
                    real_profile.wave_layer_start_count,
                    calibration.wave_ii_median_ms,
                    real_ii_median_ms,
                    calibration.wave_ii_median_ms > 0.0 ?
                        real_ii_median_ms / calibration.wave_ii_median_ms : 0.0,
                    calibration.wave_ii_mean_ms,
                    real_ii_mean_ms,
                    calibration.wave_ii_mean_ms > 0.0 ?
                        real_ii_mean_ms / calibration.wave_ii_mean_ms : 0.0,
                    calibration.wave_drain_ms,
                    real_drain_ms,
                    calibration.wave_drain_ms > 0.0 ?
                        real_drain_ms / calibration.wave_drain_ms : 0.0);

                LLAMA_LOG_DEBUG(
                    "[WAVE_DRIFT_COMPARE] probe_ii_drift=%.4f real_ii_drift=%.4f "
                    "probe_compute_drift=%.4f real_compute_drift=%.4f "
                    "probe_barrier_drift=%.4f real_barrier_drift=%.4f "
                    "probe_late_ii_ms=%.3f real_late_ii_ms=%.3f late_ii_ratio=%.4f\n",
                    calibration.wave_ii_drift_ratio,
                    real_early_ii_median_ms > 0.0 ?
                        real_late_ii_median_ms / real_early_ii_median_ms : 0.0,
                    calibration.wave_compute_drift_ratio,
                    real_early_compute_per_layer_ms > 0.0 ?
                        real_late_compute_per_layer_ms / real_early_compute_per_layer_ms : 0.0,
                    calibration.wave_barrier_drift_ratio,
                    real_early_barrier_per_layer_ms > 0.0 ?
                        real_late_barrier_per_layer_ms / real_early_barrier_per_layer_ms : 0.0,
                    calibration.wave_late_ii_median_ms,
                    real_late_ii_median_ms,
                    calibration.wave_late_ii_median_ms > 0.0 ?
                        real_late_ii_median_ms / calibration.wave_late_ii_median_ms : 0.0);
            }
        }

        if (meta_backend != nullptr) {
            // Keep later decode/prefill calls from inheriting this prompt's profile.
            ggml_backend_meta_tensor_profile_reset(meta_backend);
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id     = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    if (std::getenv("LLAMA_HYBRID_PREDICT_VALIDATE") != nullptr &&
        n_tokens_all > 16 && has_runtime_plan && runtime_plan.tensor_layers > 0) {
        llama_hybrid_tensor_compute_prediction validate_prediction;
        if (llama_hybrid_runtime_predict_tensor_compute(
                (int) n_tokens_all, validate_prediction)) {
            LLAMA_LOG_ERROR(
                "[TENSOR_PREDICT_VALIDATE] mode=DECODE_SUMMARY "
                "primary=%s tokens=%d T=%d R=%.3f XT=%d chunks=%d "
                "planner_ref_pred_ms=%.3f tensor_pred_ms=%.3f "
                "attn_misc_pred_ms=%.3f h2d_sum_pred_ms=%.3f "
                "pc_ffn_sum_pred_ms=%.3f phone_sum_pred_ms=%.3f "
                "phone_start_pred_ms=%.3f return_sum_pred_ms=%.3f "
                "h2d_finish_pred_ms=%.3f "
                "pc_finish_pred_ms=%.3f phone_finish_pred_ms=%.3f "
                "return_finish_pred_ms=%.3f tail_pred_ms=%.3f "
                "pipeline_done_pred_ms=%.3f overlap_saved_pred_ms=%.3f\n",
                tensor_phone_primary_exec ? "PHONE" : "PC",
                validate_prediction.tokens,
                validate_prediction.tensor_layers,
                validate_prediction.tensor_pc_ratio,
                validate_prediction.tensor_chunk_tokens,
                validate_prediction.tensor_chunks,
                runtime_plan.predicted_ms,
                validate_prediction.tensor_total_ms,
                validate_prediction.attn_misc_ms,
                validate_prediction.pipeline_h2d_sum_ms,
                validate_prediction.pipeline_pc_ffn_sum_ms,
                validate_prediction.pipeline_phone_sum_ms,
                validate_prediction.pipeline_phone_start_ms,
                validate_prediction.pipeline_d2h_sum_ms,
                validate_prediction.pipeline_h2d_finish_ms,
                validate_prediction.pipeline_pc_finish_ms,
                validate_prediction.pipeline_phone_finish_ms,
                validate_prediction.pipeline_return_finish_ms,
                validate_prediction.pipeline_reduce_tail_ms,
                validate_prediction.pipeline_done_ms,
                validate_prediction.pipeline_overlap_saved_ms);
        }
    }

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd     = hparams.n_embd;
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits     = true;
    bool has_embd       = cparams.embeddings;
    bool has_embd_nextn = cparams.embeddings_nextn;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }

    size_t backend_float_count = 0;
    size_t backend_token_count = 0;
    size_t embd_layer_inp_float_count = 0;

    logits.size     = has_logits     ? n_vocab*n_outputs_max     : 0;
    embd.size       = has_embd       ? n_embd_out*n_outputs_max  : 0;
    embd_nextn.size = has_embd_nextn ? n_embd_out*n_outputs_max  : 0;

    if (has_embd_nextn && !cparams.embeddings_nextn_masked) {
        // unmasked: nextn row exists for every token in the batch, not just
        // those flagged via batch.logits[i] -> size by token count instead.
        embd_nextn.size = (size_t) n_embd_out * n_batch;
    }

    for (bool enabled : cparams.embeddings_layer_inp) {
        if (enabled) {
            embd_layer_inp_float_count += (size_t) n_embd * n_batch;
        }
    }

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + embd_nextn.size + embd_layer_inp_float_count + backend_float_count) * sizeof(float) +
        (                                                                         backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
            embd_nextn.data = nullptr;
            for (auto & layer_inp : embd_layer_inp) {
                layer_inp = {nullptr, 0};
            }
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    embd_nextn = has_embd_nextn ? buffer_view<float>{(float *) (base + offset), embd_nextn.size} : buffer_view<float>{nullptr, 0};
    offset += embd_nextn.size * sizeof(float);

    for (uint32_t il = 0; il < embd_layer_inp.size(); ++il) {
        if (cparams.embeddings_layer_inp[il]) {
            embd_layer_inp[il] = buffer_view<float>{(float *) (base + offset), (size_t) n_embd * n_batch};
            offset += embd_layer_inp[il].size * sizeof(float);
        } else {
            embd_layer_inp[il] = buffer_view<float>{nullptr, 0};
        }
    }

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max);

    return n_outputs_max;
}

void llama_context::extract_layer_inputs(const llm_graph_result * res,
                                         size_t                   token_offset,
                                         size_t                   n_tokens,
                                         ggml_backend_sched_t     sched_use) {
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
        }
        if (!embd_layer_inp[il].has_data()) {
            GGML_ABORT("output layer input buffer not allocated");
        }
        ggml_tensor * t = res->get_layer_inp((int) il);
        if (!t) {
            GGML_ABORT("layer input tensor not found");
        }

        const size_t nbytes  = ggml_nbytes(t);
        const size_t nfloats = nbytes / sizeof(float);
        GGML_ASSERT(n_tokens > 0);
        GGML_ASSERT(nfloats % n_tokens == 0);

        const size_t row_floats = nfloats / n_tokens;
        const size_t dst_offset = token_offset * row_floats;
        GGML_ASSERT(dst_offset + nfloats <= embd_layer_inp[il].size);

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched_use, t);
        GGML_ASSERT(backend != nullptr);
        ggml_backend_tensor_get_async(backend, t, embd_layer_inp[il].data + dst_offset, 0, nbytes);
    }
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (embd_nextn.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd_nextn.data[i0*n_embd + k], embd_nextn.data[i1*n_embd + k]);
            }
        }

        if (embd_layer_inp.size() > 0) {
            for (int lid = 0; lid < (int) embd_layer_inp.size(); ++lid) {
                if (embd_layer_inp[lid].size > 0) {
                    for (uint64_t k = 0; k < n_embd; ++k) {
                        std::swap(embd_layer_inp[lid].data[i0*n_embd + k], embd_layer_inp[lid].data[i1*n_embd + k]);
                    }
                }
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    uint64_t base;

    if (model.arch == LLM_ARCH_QWEN3NEXT ||
        model.arch == LLM_ARCH_KIMI_LINEAR ||
        model.arch == LLM_ARCH_QWEN35 ||
        model.arch == LLM_ARCH_QWEN35MOE ||
        model.arch == LLM_ARCH_DEEPSEEK4) {
        base = std::max<uint64_t>((uint64_t) n_tokens * 40, (uint64_t) 32 * model.n_tensors());
    } else {
        base = std::max<uint64_t>(1024, (uint64_t) 8 * model.n_tensors());
    }

    uint64_t res = base;

    const bool decode_chunked_arch =
        model.arch == LLM_ARCH_LLAMA ||
        model.arch == LLM_ARCH_QWEN2 ||
        model.arch == LLM_ARCH_QWEN3 ||
        model.arch == LLM_ARCH_QWEN3MOE;
    if (decode_chunked_arch &&
        model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        const uint32_t default_chunks =
            model.arch == LLM_ARCH_QWEN3MOE ? 1u : 2u;
        const char * value = std::getenv("LLAMA_CHUNKS");
        const int configured_chunks =
            value == nullptr ? (int) default_chunks : std::atoi(value);
        const uint32_t n_chunks = std::min<uint32_t>(
                configured_chunks > 0 ?
                    (uint32_t) configured_chunks : default_chunks,
                model.hparams.n_embd);

        // MoE decode chunking repeats the down-projection/aggregation tail for
        // each output slice, so reserve more headroom than dense FFN chunking.
        const uint64_t nodes_per_chunk =
            model.arch == LLM_ARCH_QWEN3MOE ? 48ull : 24ull;
        res += (uint64_t) model.hparams.n_layer() *
            (nodes_per_chunk*n_chunks + 24ull);
    }

    const int chunk_tokens = llama_hybrid_runtime_prefill_chunk_tokens();

    if (chunk_tokens > 0 && model.hparams.n_layer() > 0) {
        const auto chunks = llama_hybrid_split_by_chunk_size((int) n_tokens, chunk_tokens);
        const uint64_t n_chunks = chunks.size();

        if (n_chunks > 1) {
            const uint64_t tensor_layers = std::count_if(
                    model.hybrid_layer_modes.begin(),
                    model.hybrid_layer_modes.end(),
                    [](llama_hybrid_layer_mode mode) {
                        return llama_hybrid_layer_mode_is_tensor(mode);
                    });
            const uint64_t n_layers = model.hparams.n_layer();
            const uint64_t base_nodes_per_layer = (base + n_layers - 1) / n_layers;

            res += tensor_layers * (n_chunks - 1) * base_nodes_per_layer;
        }
    }

    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }

    return (uint32_t) std::min<uint64_t>(res, std::numeric_limits<uint32_t>::max());
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i]         = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i]   = &seq_ids[i];
        ubatch.output[i]   = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, ctx_type_to_graph_type(cparams.ctx_type), sched.get());

    res->reset();

    LLAMA_LOG_DEBUG("[GRAPH_RESERVE_BEGIN] tokens=%u outputs=%u\n", n_tokens, n_outputs);
    auto * gf = model.build_graph(gparams);
    LLAMA_LOG_DEBUG("[GRAPH_RESERVE_END] tokens=%u nodes=%d\n", n_tokens, ggml_graph_n_nodes(gf));

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(llm_graph_result *             res,
                                             const llama_ubatch &           ubatch,
                                             const llama_memory_context_i * mctx,
                                             llm_graph_type                 gtype,
                                             ggml_backend_sched_t           sched_use,
                                             int32_t                        outputs_override) const {
    return {
        /*.arch        =*/model.arch,
        /*.hparams     =*/model.hparams,
        /*.cparams     =*/cparams,
        /*.ubatch      =*/ubatch,
        /*.gtype       =*/gtype,
        /*.sched       =*/sched_use,
        /*.backend_cpu =*/backend_cpu,
        /*.cvec        =*/cvec.get(),
        /*.loras       =*/loras.get(),
        /*.mctx        =*/mctx,
        /*.cross       =*/&cross,
        /*.samplers    =*/sampling.samplers,
        /*.n_outputs   =*/outputs_override >= 0 ?
                              (uint32_t) outputs_override : n_outputs,
        /*.cb          =*/graph_get_cb(sched_use),
        /*.res         =*/res,
    };
}

ggml_status llama_context::graph_compute(ggml_cgraph * gf, bool batched) {
    int               n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp        = batched ? threadpool_batch : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg               = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

ggml_status llama_context::graph_compute_range(ggml_backend_sched_t sched_use,
                                               int                  first_split,
                                               int                  last_split,
                                               bool                 batched) {
    int               n_threads = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp        = batched ? threadpool_batch : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg               = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    const ggml_status status = ggml_backend_sched_compute_range(sched_use, first_split, last_split);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_compute_range failed with error %d\n", __func__, status);
    }

    return status;
}

llm_graph_cb llama_context::graph_get_cb(ggml_backend_sched_t sched_use) const {
    return [this, sched_use](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        if (il >= 0 && strcmp(name, "prefill_wave_entry") == 0) {
            const ggml_backend_dev_t dev_layer = model.dev_layer(il);

            for (const auto & backend : backends) {
                if (ggml_backend_get_device(backend.get()) == dev_layer &&
                    ggml_backend_supports_op(backend.get(), cur)) {
                    ggml_backend_sched_set_tensor_backend(sched_use, cur, backend.get());
                    break;
                }
            }
        }

        // GPU weight streaming decouples execution from residency:
        // pimpl weight storage is CPU-canonical, while model.dev_layer(il)
        // intentionally reports the CUDA execution owner for KV/fused-op
        // placement. Pin every supported op in those layers to the outer CUDA
        // backend; unsupported ops are caught by the strict-backend guard.
        if (il >= 0 &&
            model.hybrid_ring_gpu_weight_streaming &&
            ggml_backend_sched_get_weight_cache(sched_use) &&
            model.hybrid_layer_mode(il) ==
                llama_hybrid_layer_mode::PC_ONLY &&
            il < (int) model.pc_layer_backends.size() &&
            model.pc_layer_backends[(size_t) il] ==
                llama_pc_layer_backend::CUDA) {
            for (const auto & backend : backends) {
                ggml_backend_dev_t dev =
                    ggml_backend_get_device(backend.get());
                if (dev != nullptr &&
                    ggml_backend_dev_type(dev) ==
                        GGML_BACKEND_DEVICE_TYPE_GPU &&
                    ggml_backend_supports_op(backend.get(), cur)) {
                    ggml_backend_sched_set_tensor_backend(
                        sched_use, cur, backend.get());
                    return;
                }
            }
        }

        // PC_ONLY layers placed on CPU_DIRECT must stay as one CPU region.
        // The generic scheduler deliberately expands higher-priority GPU
        // assignments across unassigned ops while not expanding CPU, which
        // otherwise fragments a direct-CPU layer into hundreds of CPU/CUDA
        // subgraphs during prefill (and dozens during decode). Pin every
        // callback node in a PC_ONLY CPU layer to the raw CPU backend for both
        // prefill and decode. Tensor/Phone layers still use Meta because their
        // dev_layer is the Meta device, not the raw CPU device.
        if (il >= 0 &&
            backend_cpu != nullptr &&
            model.hybrid_layer_mode(il) == llama_hybrid_layer_mode::PC_ONLY &&
            model.dev_layer(il) == ggml_backend_get_device(backend_cpu) &&
            ggml_backend_supports_op(backend_cpu, cur)) {
            ggml_backend_sched_set_tensor_backend(sched_use, cur, backend_cpu);
            return;
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer_all;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched_use, cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            bool need_alloc = false;

            need_alloc = need_alloc || (!mbuf_cur.buf);
            need_alloc = need_alloc || (mbuf_cur.org.size() != mbuf.org.size());
            need_alloc = need_alloc || (mbuf_cur.total_size != mbuf.total_size);

            if (!need_alloc) {
                for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                    auto * org0 = mbuf_cur.org[i];
                    auto * org1 = mbuf.org[i];

                    if (!ggml_are_same_shape(org0, org1)) {
                        need_alloc = true;
                        break;
                    }

                    if (org0->view_src != org1->view_src || org0->view_offs != org1->view_offs) {
                        need_alloc = true;
                        break;
                    }
                }
            }

            if (need_alloc) {
                if (!mbuf_cur.buf || mbuf_cur.total_size != mbuf.total_size) {
                    mbuf_cur = std::move(mbuf);

                    mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                    LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
                } else {
                    //LLAMA_LOG_INFO("%s: reallocating tensors in '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);

                    // save the old buffer and allocate the new tensors in it
                    auto buf = std::move(mbuf_cur.buf);

                    mbuf_cur = std::move(mbuf);

                    ggml_tallocr talloc = ggml_tallocr_new(buf.get());

                    for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                        ggml_backend_view_init(mbuf_cur.org[i]);
                        ggml_tallocr_alloc(&talloc, mbuf_cur.cpy[i]);
                    }

                    mbuf_cur.buf = std::move(buf);
                }
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
        }

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            const int64_t n = rinfo.size/ggml_element_size(rinfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d(mbuf.ctx.get(), rinfo.tensor, n, rinfo.offset));

            ggml_backend_view_init(mbuf.org.back());
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf.org[i]);
            }
        }

        GGML_ASSERT(buf_size == 0);
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(
                res, ubatch, mctx.get(), ctx_type_to_graph_type(cparams.ctx_type), sched.get());

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_rs_seq                    =*/ 0,
        /*.n_outputs_max               =*/ 0,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.ctx_type                    =*/ LLAMA_CONTEXT_TYPE_DEFAULT,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
        /*.ctx_other                   =*/ nullptr,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer(); ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP &&
        model->hparams.n_layer_nextn == 0) {
        LLAMA_LOG_WARN("%s: context type MTP requested but model doesn't contain MTP layers\n", __func__);
        return nullptr;
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

uint32_t llama_n_rs_seq(const llama_context * ctx) {
    return ctx->get_cparams().n_rs_seq;
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

void llama_set_embeddings_nextn(llama_context * ctx, bool value, bool masked) {
    ctx->set_embeddings_nextn(value, masked);
}

void llama_set_embeddings_layer_inp(llama_context * ctx, uint32_t lid, bool value) {
    ctx->set_embeddings_layer_inp(lid, value);
}

void llama_set_nextn_layer_offset(llama_context * ctx, int32_t offset) {
    ctx->set_nextn_layer_offset(offset);
}

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    return ctx->get_memory();
}

float * llama_get_embeddings_nextn(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn();
}

float * llama_get_embeddings_nextn_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_nextn_ith(i);
}

float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}

llama_context * llama_get_ctx_other(struct llama_context * ctx) {
    return ctx->get_cparams().ctx_other;
}
