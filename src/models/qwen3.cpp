#include "llama-hybrid.h"
#include "models.h"

#include <cstdlib>

static int qwen3_ffn_chunk_count() {
    constexpr int default_chunks = 2;
    const char *  value          = std::getenv("LLAMA_CHUNKS");
    if (value == nullptr) {
        return default_chunks;
    }
    const int chunks = std::atoi(value);
    return chunks > 0 ? chunks : default_chunks;
}

static int qwen3_prefill_ffn_chunk_count() {
    const char * value = std::getenv("LLAMA_PREFILL_CHUNKS");
    return value == nullptr ? 0 : std::max(1, std::atoi(value));
}

void llama_model_qwen3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 28:
            type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B;
            break;
        case 36:
            type = hparams.n_embd == 2560 ? LLM_TYPE_4B : LLM_TYPE_8B;
            break;
        case 40:
            type = LLM_TYPE_14B;
            break;
        case 64:
            type = LLM_TYPE_32B;
            break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // output rerank head
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), { n_embd, hparams.n_cls_out }, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), { n_embd_head_k }, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), { n_embd_head_k }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    const bool stage_graph       = params.hybrid_layer_end >= 0;
    const int  layer_begin       = stage_graph ? params.hybrid_layer_begin : 0;
    const int  layer_end         = stage_graph ? params.hybrid_layer_end : n_layer;
    const bool hidden_input      = stage_graph && params.hybrid_hidden_input;
    const bool build_output_head = !stage_graph || params.hybrid_output_head;
    const bool embedding_only    = stage_graph && params.hybrid_embedding_only;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(layer_begin >= 0 && layer_begin < layer_end && layer_end <= n_layer);
    GGML_ASSERT(layer_begin == 0 || hidden_input);
    GGML_ASSERT(!build_output_head || layer_end == n_layer);
    GGML_ASSERT(!hidden_input || (ubatch.token == nullptr && ubatch.embd != nullptr));
    GGML_ASSERT(!embedding_only || (layer_begin == 0 && !hidden_input));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = hidden_input ? build_inp_stage() : build_inp_embd(model.tok_embd);

    // Region Ring GPU-first prelude: build only the token embedding on the
    // CPU-only worker. The real GPU stage is rebuilt with hidden_input=true.
    if (embedding_only) {
        if (!ggml_is_contiguous(inpL)) {
            inpL = ggml_cont(ctx0, inpL);
            cb(inpL, "embedding_prelude_cont", -1);
        }
        res->t_stage_output = inpL;
        ggml_build_forward_expand(gf, inpL);
        return;
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_output_head ? build_inp_out_ids() : nullptr;

    for (int il = layer_begin; il < layer_end; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn, model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s, Qcur, Kcur,
                             Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        const llama_hybrid_layer_mode hybrid_mode = model.hybrid_layer_mode(il);
        const bool directional_stage_prefill =
            stage_graph && n_tokens > 1 &&
            llama_hybrid_layer_mode_directional_tensor(hybrid_mode);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        if (directional_stage_prefill) {
            // Region Ring already sliced this stage into one causal outer
            // chunk. The completed FFN norm is the exact READY payload.
            cur->flags |= GGML_TENSOR_FLAG_META_SEND_READY;
        }
        cb(cur, "ffn_norm", il);

        const int                     n_decode_chunks  = std::min<int64_t>(qwen3_ffn_chunk_count(), n_embd);
        const int planned_chunk_tokens = llama_hybrid_runtime_prefill_chunk_tokens();
        const int n_prefill_chunks = planned_chunk_tokens > 0 ?
            (cur->ne[1] + planned_chunk_tokens - 1) / planned_chunk_tokens :
            std::min<int64_t>(qwen3_prefill_ffn_chunk_count(), cur->ne[1]);
        const bool use_decode_chunked_ffn = n_tokens == 1 && model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
                                            hybrid_mode == llama_hybrid_layer_mode::TENSOR_SPLIT &&
                                            n_decode_chunks >= 1 && loras->empty() && cvec->tensor_for(il) == nullptr;
        const bool use_prefill_chunked_ffn = !directional_stage_prefill &&
                                             n_tokens > 1 && cur->ne[1] > 1 &&
                                             model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
                                             llama_hybrid_layer_mode_uses_split_ffn_chunks(hybrid_mode) &&
                                             n_prefill_chunks >= 1 && loras->empty() && cvec->tensor_for(il) == nullptr;

        if (directional_stage_prefill &&
                std::getenv("GGML_META_PIPELINE_DEBUG") != nullptr) {
            LLAMA_LOG_ERROR(
                "[DIRECTIONAL_DENSE_GRAPH] layer=%d tokens=%" PRId64
                " mode=%d split_mode=%d stage=1 planned_chunk=%d "
                "n_prefill_chunks=%d inner_chunked=%d "
                "ready=ffn_norm partial=directional_ffn_partial\n",
                il, n_tokens, (int) hybrid_mode, (int) model.split_mode(),
                planned_chunk_tokens, n_prefill_chunks,
                use_prefill_chunked_ffn ? 1 : 0);
        }

        if (use_decode_chunked_ffn) {
            ggml_tensor * ffn_hidden = build_ffn(cur, model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                                                 model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s, nullptr,
                                                 nullptr, nullptr, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            std::vector<ggml_tensor *> chunks;
            chunks.reserve(n_decode_chunks);
            for (int i = 0; i < n_decode_chunks; ++i) {
                const int64_t offset = n_embd * i / n_decode_chunks;
                const int64_t end    = n_embd * (i + 1) / n_decode_chunks;
                const int64_t length = end - offset;
                ggml_tensor * down =
                    ggml_view_2d(ctx0, model.layers[il].ffn_down, model.layers[il].ffn_down->ne[0], length,
                                 model.layers[il].ffn_down->nb[1], offset * model.layers[il].ffn_down->nb[1]);
                ggml_tensor *     out        = build_lora_mm(down, ffn_hidden);
                const std::string chunk_name = "ffn_down_chunk_" + std::to_string(i);
                cb(out, chunk_name.c_str(), il);
                if (model.layers[il].ffn_down_s != nullptr) {
                    out = ggml_mul(ctx0, out, model.layers[il].ffn_down_s);
                }
                ggml_tensor * residual =
                    ggml_view_2d(ctx0, ffn_inp, length, ffn_inp->ne[1], ffn_inp->nb[1], offset * ffn_inp->nb[0]);
                chunks.push_back(ggml_add(ctx0, out, residual));
            }
            cur = chunks[0];
            for (int i = 1; i < n_decode_chunks; ++i) {
                cur = ggml_concat(ctx0, cur, chunks[i], 0);
            }
        } else if (use_prefill_chunked_ffn) {
            std::vector<ggml_tensor *> chunks;
            chunks.reserve(n_prefill_chunks);
            const std::vector<int> chunk_sizes = planned_chunk_tokens > 0 ?
                llama_hybrid_split_by_chunk_size((int) cur->ne[1], planned_chunk_tokens) :
                llama_hybrid_split_chunks((int) cur->ne[1], n_prefill_chunks);
            GGML_ASSERT((int) chunk_sizes.size() == n_prefill_chunks);
            int64_t token_begin = 0;
            for (int i = 0; i < n_prefill_chunks; ++i) {
                const int64_t token_count = chunk_sizes[i];
                const int64_t token_end   = token_begin + token_count;
                GGML_ASSERT(token_count > 0);
                ggml_tensor * norm_chunk =
                    ggml_view_2d(ctx0, cur, cur->ne[0], token_count, cur->nb[1], token_begin * cur->nb[1]);
                const std::string norm_name = "prefill_ffn_norm_chunk_" + std::to_string(i);
                if (llama_hybrid_layer_mode_directional_tensor(hybrid_mode)) {
                    norm_chunk->flags |= GGML_TENSOR_FLAG_META_SEND_READY;
                }
                cb(norm_chunk, norm_name.c_str(), il);
                ggml_tensor * down_chunk =
                    build_ffn(norm_chunk, model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
                              model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s, model.layers[il].ffn_down,
                              NULL, model.layers[il].ffn_down_s, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
                const std::string down_name = "prefill_ffn_down_chunk_" + std::to_string(i);
                if (llama_hybrid_layer_mode_directional_tensor(hybrid_mode)) {
                    down_chunk->flags |= GGML_TENSOR_FLAG_META_SEND_PARTIAL;
                }
                cb(down_chunk, down_name.c_str(), il);
                chunks.push_back(down_chunk);
                token_begin = token_end;
            }
            cur = chunks.back();
            for (int i = (int) chunks.size() - 2; i >= 0; --i) {
                cur = ggml_concat(ctx0, chunks[i], cur, 1);
            }
            cur = ggml_add(ctx0, cur, ffn_inp);
        } else {
            cur = build_ffn(cur, model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s, model.layers[il].ffn_gate,
                            NULL, model.layers[il].ffn_gate_s, model.layers[il].ffn_down, NULL,
                            model.layers[il].ffn_down_s, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            if (directional_stage_prefill) {
                // This is the local FFN contribution before residual ADD.
                // Completion of this exact producer means SEND_PARTIAL.
                cur->flags |= GGML_TENSOR_FLAG_META_SEND_PARTIAL;
                cb(cur, "directional_ffn_partial", il);
            }
            cur = ggml_add(ctx0, cur, ffn_inp);
        }
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    if (!build_output_head) {
        res->t_stage_output = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
