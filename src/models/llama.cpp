#include "models.h"

#include <cstdlib>

static int llama_ffn_chunk_count() {
    constexpr int default_chunks = 2;
    const char * value = std::getenv("LLAMA_CHUNKS");
    if (value == nullptr) {
        return default_chunks;
    }

    const int chunks = std::atoi(value);
    return chunks > 0 ? chunks : default_chunks;
}

static int llama_prefill_ffn_chunk_count() {
    const char * value = std::getenv("LLAMA_PREFILL_CHUNKS");
    if (value == nullptr) {
        return 1;
    }

    return std::max(1, std::atoi(value));
}

void llama_model_llama::load_arch_hparams(llama_model_loader & ml) {
    uint32_t n_vocab = 0;
    ml.get_key(LLM_KV_VOCAB_SIZE, n_vocab, false) || ml.get_arr_n(LLM_KV_TOKENIZER_LIST, n_vocab, false);

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    if (hparams.n_expert == 8) {
        switch (hparams.n_layer()) {
            case 32: type = LLM_TYPE_8x7B; break;
            case 56: type = LLM_TYPE_8x22B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    } else {
        switch (hparams.n_layer()) {
            case 16: type = LLM_TYPE_1B; break; // Llama 3.2 1B
            case 22: type = LLM_TYPE_1B; break;
            case 26: type = LLM_TYPE_3B; break;
            case 28: type = LLM_TYPE_3B; break; // Llama 3.2 3B
            case 30: type = LLM_TYPE_256M; break; // smoldocling 256M
            // granite uses a vocab with len 49152
            case 32: type = n_vocab == 49152 ? LLM_TYPE_3B : (n_vocab < 40000 ? LLM_TYPE_7B : LLM_TYPE_8B); break;
            case 36: type = LLM_TYPE_8B; break; // granite
            case 40: type = LLM_TYPE_13B; break;
            case 48: type = LLM_TYPE_34B; break;
            case 60: type = LLM_TYPE_30B; break;
            case 80: type = hparams.n_head() == hparams.n_head_kv() ? LLM_TYPE_65B : LLM_TYPE_70B; break;
            default: type = LLM_TYPE_UNKNOWN;
        }
    }
}

void llama_model_llama::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];   //一个层的张量目录

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // optional bias tensors
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_LONGROPE) {
            layer.rope_long  = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_LONG,  "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
            layer.rope_short = create_tensor(tn(LLM_TENSOR_ROPE_FACTORS_SHORT, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }
        else {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
        }

        if (n_expert == 0) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

            // optional MLP bias
            layer.ffn_gate_b = create_tensor(tn(LLM_TENSOR_FFN_GATE, "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.ffn_up_b   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias", i), {n_ff}, TENSOR_NOT_REQUIRED);
        } else {
            layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff, n_expert}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {  n_ff, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff, n_expert}, 0);

            // For Granite MoE Shared
            if (hparams.n_ff_shexp > 0) {
                layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, hparams.n_ff_shexp}, 0);
                layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {hparams.n_ff_shexp, n_embd}, 0);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_llama::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph<false>>(*this, params);
}

template <bool embed>
llama_model_llama::graph<embed>::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;
    ggml_tensor * inpL_attn_norm = nullptr;
    llm_graph_qkv inpL_qkv = { nullptr, nullptr, nullptr };

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    using inp_attn_type = std::conditional_t<embed, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;

    inp_attn_type * inp_attn = nullptr;
    if constexpr (embed) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
    }
    // 选择不同的模式，embed为true说明是embedding模式，embed为false说明是普通的decoder模式
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();
    // transformer layers
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;
        // 把当前层的输入保存到结果中，方便后续使用
        // inpl是当前的输入hidden states
        ggml_tensor * inpSA = inpL;
        // inpsa把本层输入另存一份，后续会在ffn中使用
        // norm
        if (inpL_attn_norm != nullptr) {
            cur = inpL_attn_norm;
        } else {
            cur = build_norm(inpL,
                    model.layers[il].attn_norm, NULL,
                    LLM_NORM_RMS, il);
        }
        cb(cur, "attn_norm", il);
            //归一化
        // self-attention
        {
            // rope freq factors for llama3; may return nullptr for llama2 and other models
            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

            // compute Q and K and RoPE them
            ggml_tensor * Qcur;
            ggml_tensor * Kcur;
            ggml_tensor * Vcur;
            if (inpL_qkv.q != nullptr) {
                Qcur = inpL_qkv.q;
                Kcur = inpL_qkv.k;
                Vcur = inpL_qkv.v;
                inpL_qkv = { nullptr, nullptr, nullptr };
            } else {
                auto qkv = build_qkv(model.layers[il], cur,
                        n_embd_head, n_head, n_head_kv, il);
                Qcur = qkv.q;
                Kcur = qkv.k;
                Vcur = qkv.v;
            }
                //计算Q、K、V，并进行RoPE处理
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
                    //这里和下面进行位置编码
            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);
                    // 普通的llama不会启用
            if (hparams.use_kq_norm) {
                // Llama4TextL2Norm
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
                cb(Qcur, "Qcur_normed", il);
                cb(Kcur, "Kcur_normed", il);
            }
            //计算self-attention的输出
            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            // 如果是最后一层，并且有输出id，那么就把当前的输出和输入都取出对应的行
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);
        // 残差连接
        // feed-forward network (non-MoE)
        if (model.layers[il].ffn_gate_inp == nullptr) {

            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            const int n_chunks = std::min<int64_t>(llama_ffn_chunk_count(), n_embd);
            const bool use_decode_chunked_ffn =
                !embed &&
                n_tokens == 1 &&
                model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
                n_chunks >= 1 &&
                loras->empty() &&
                cvec->tensor_for(il) == nullptr;
            const int n_prefill_chunks = std::min<int64_t>(llama_prefill_ffn_chunk_count(), n_tokens);
            const bool use_prefill_chunked_ffn =
                !embed &&
                n_tokens > 1 &&
                model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
                n_prefill_chunks > 1 &&
                loras->empty() &&
                cvec->tensor_for(il) == nullptr;

            if (use_decode_chunked_ffn) {
                ggml_tensor * ffn_hidden = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                        nullptr, nullptr, nullptr,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);

                std::vector<ggml_tensor *> hidden_chunks;
                std::vector<ggml_tensor *> norm_pre_chunks;
                hidden_chunks.reserve(n_chunks);
                norm_pre_chunks.reserve(n_chunks);

                ggml_tensor * sum_sq = nullptr;
                ggml_tensor * q_acc  = nullptr;
                ggml_tensor * k_acc  = nullptr;
                ggml_tensor * v_acc  = nullptr;

                const llama_layer * next_layer = il + 1 < n_layer ? &model.layers[il + 1] : nullptr;
                auto can_slice_input = [&](const ggml_tensor * weight) {
                    if (weight == nullptr || weight->ne[0] != n_embd) {
                        return false;
                    }
                    const int64_t block_size = ggml_blck_size(weight->type);
                    for (int i = 0; i <= n_chunks; ++i) {
                        if ((n_embd * i / n_chunks) % block_size != 0) {
                            return false;
                        }
                    }
                    return true;
                };
                const bool use_partial_qkv =
                    next_layer != nullptr &&
                    next_layer->wqkv == nullptr &&
                    can_slice_input(next_layer->wq) &&
                    can_slice_input(next_layer->wk) &&
                    can_slice_input(next_layer->wv);

                for (int i = 0; i < n_chunks; ++i) {
                    const int64_t offset = n_embd * i / n_chunks;
                    const int64_t end    = n_embd * (i + 1) / n_chunks;
                    const int64_t length = end - offset;

                    ggml_tensor * down = ggml_view_2d(ctx0, model.layers[il].ffn_down,
                            model.layers[il].ffn_down->ne[0], length,
                            model.layers[il].ffn_down->nb[1], offset * model.layers[il].ffn_down->nb[1]);
                    ggml_tensor * out = build_lora_mm(down, ffn_hidden);
                    const std::string chunk_name = "ffn_down_chunk_" + std::to_string(i);
                    cb(out, chunk_name.c_str(), il);

                    if (model.layers[il].ffn_down_b != nullptr) {
                        ggml_tensor * bias = ggml_view_1d(ctx0, model.layers[il].ffn_down_b,
                                length, offset * model.layers[il].ffn_down_b->nb[0]);
                        out = ggml_add(ctx0, out, bias);
                    }
                    if (model.layers[il].ffn_down_s != nullptr) {
                        out = ggml_mul(ctx0, out, model.layers[il].ffn_down_s);
                    }

                    ggml_tensor * residual = ggml_view_2d(ctx0, ffn_inp,
                            length, ffn_inp->ne[1], ffn_inp->nb[1], offset * ffn_inp->nb[0]);
                    ggml_tensor * hidden = ggml_add(ctx0, out, residual);
                    cb(hidden, "ffn_out_chunk", il);
                    hidden_chunks.push_back(hidden);

                    ggml_tensor * chunk_sum_sq = ggml_sum_rows(ctx0, ggml_sqr(ctx0, hidden));
                    sum_sq = sum_sq == nullptr ? chunk_sum_sq : ggml_add(ctx0, sum_sq, chunk_sum_sq);

                    if (il + 1 < n_layer) {
                        ggml_tensor * norm_weight = ggml_view_1d(ctx0, model.layers[il + 1].attn_norm,
                                length, offset * model.layers[il + 1].attn_norm->nb[0]);
                        ggml_tensor * norm_pre = ggml_mul(ctx0, hidden, norm_weight);
                        norm_pre_chunks.push_back(norm_pre);

                        if (use_partial_qkv) {
                            auto build_partial = [&](ggml_tensor * weight) {
                                const int64_t bs = ggml_blck_size(weight->type);

                                GGML_ASSERT(offset % bs == 0);
                                GGML_ASSERT(length % bs == 0);

                                const size_t view_offs =
                                    (offset / bs) * weight->nb[0];

                                ggml_tensor * weight_chunk = ggml_view_2d(
                                        ctx0,
                                        weight,
                                        length,
                                        weight->ne[1],
                                        weight->nb[1],
                                        view_offs);

                                return build_lora_mm(weight_chunk, norm_pre);
                            };

                            ggml_tensor * q_part = build_partial(next_layer->wq);
                            ggml_tensor * k_part = build_partial(next_layer->wk);
                            ggml_tensor * v_part = build_partial(next_layer->wv);
                            cb(q_part, "Qcur_partial", il + 1);
                            cb(k_part, "Kcur_partial", il + 1);
                            cb(v_part, "Vcur_partial", il + 1);

                            q_acc = q_acc == nullptr ? q_part : ggml_add(ctx0, q_acc, q_part);
                            k_acc = k_acc == nullptr ? k_part : ggml_add(ctx0, k_acc, k_part);
                            v_acc = v_acc == nullptr ? v_part : ggml_add(ctx0, v_acc, v_part);
                        }
                    }
                }

                cur = hidden_chunks[0];
                for (int i = 1; i < n_chunks; ++i) {
                    cur = ggml_concat(ctx0, cur, hidden_chunks[i], 0);
                }

                if (il + 1 < n_layer) {
                    inpL_attn_norm = norm_pre_chunks[0];
                    for (int i = 1; i < n_chunks; ++i) {
                        inpL_attn_norm = ggml_concat(ctx0, inpL_attn_norm, norm_pre_chunks[i], 0);
                    }
                    ggml_tensor * rms = ggml_scale_bias(ctx0, sum_sq, 1.0f / n_embd, hparams.f_norm_rms_eps);
                    rms = ggml_sqrt(ctx0, rms);
                    inpL_attn_norm = ggml_div(ctx0, inpL_attn_norm, rms);
                    cb(inpL_attn_norm, "next_attn_norm", il);

                    if (use_partial_qkv) {
                        auto finish_projection = [&](ggml_tensor * acc, ggml_tensor * scale, ggml_tensor * bias,
                                                     int64_t n_heads, const char * name) {
                            ggml_tensor * projection = ggml_div(ctx0, acc, rms);
                            if (scale != nullptr) {
                                projection = ggml_mul(ctx0, projection, scale);
                            }
                            if (bias != nullptr) {
                                projection = ggml_add(ctx0, projection, bias);
                            }
                            if (hparams.f_clamp_kqv > 0.0f) {
                                projection = ggml_clamp(ctx0, projection,
                                        -hparams.f_clamp_kqv, hparams.f_clamp_kqv);
                            }
                            projection = ggml_reshape_3d(ctx0, projection,
                                    n_embd_head, n_heads, n_tokens);
                            cb(projection, name, il + 1);
                            return projection;
                        };

                        inpL_qkv = {
                            finish_projection(q_acc, next_layer->wq_s, next_layer->wq_b, n_head,    "Qcur_prepared"),
                            finish_projection(k_acc, next_layer->wk_s, next_layer->wk_b, n_head_kv, "Kcur_prepared"),
                            finish_projection(v_acc, next_layer->wv_s, next_layer->wv_b, n_head_kv, "Vcur_prepared"),
                        };
                    }
                } else {
                    inpL_attn_norm = nullptr;
                    inpL_qkv = { nullptr, nullptr, nullptr };
                }
            } else if (use_prefill_chunked_ffn) {
                std::vector<ggml_tensor *> down_chunks;
                down_chunks.reserve(n_prefill_chunks);

                for (int i = 0; i < n_prefill_chunks; ++i) {
                    const int64_t token_begin = n_tokens * i / n_prefill_chunks;
                    const int64_t token_end   = n_tokens * (i + 1) / n_prefill_chunks;
                    const int64_t token_count = token_end - token_begin;

                    ggml_tensor * ffn_norm_chunk = ggml_view_2d(ctx0, cur,
                            cur->ne[0], token_count, cur->nb[1], token_begin * cur->nb[1]);
                    const std::string norm_name = "prefill_ffn_norm_chunk_" + std::to_string(i);
                    cb(ffn_norm_chunk, norm_name.c_str(), il);

                    ggml_tensor * down_chunk = build_ffn(ffn_norm_chunk,
                            model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                            model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                            model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                            NULL,
                            LLM_FFN_SILU, LLM_FFN_PAR, il);
                    const std::string down_name = "prefill_ffn_down_chunk_" + std::to_string(i);
                    cb(down_chunk, down_name.c_str(), il);
                    down_chunks.push_back(down_chunk);
                }

                cur = down_chunks[0];
                for (int i = 1; i < n_prefill_chunks; ++i) {
                    cur = ggml_concat(ctx0, cur, down_chunks[i], 1);
                }
                cur = ggml_add(ctx0, cur, ffn_inp);
                inpL_attn_norm = nullptr;
                inpL_qkv = { nullptr, nullptr, nullptr };
            } else {
                cur = build_ffn(cur,
                        model.layers[il].ffn_up,   model.layers[il].ffn_up_b,   model.layers[il].ffn_up_s,
                        model.layers[il].ffn_gate, model.layers[il].ffn_gate_b, model.layers[il].ffn_gate_s,
                        model.layers[il].ffn_down, model.layers[il].ffn_down_b, model.layers[il].ffn_down_s,
                        NULL,
                        LLM_FFN_SILU, LLM_FFN_PAR, il);
                cur = ggml_add(ctx0, cur, ffn_inp);
                inpL_attn_norm = nullptr;
                inpL_qkv = { nullptr, nullptr, nullptr };
            }
            cb(cur, "ffn_out", il);
            // 构建普通FFN
        } else {
            // MoE branch
            cur = build_norm(ffn_inp,
                    model.layers[il].ffn_norm, NULL,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cb(cur, "ffn_moe_out", il);
        }
        if (model.layers[il].ffn_gate_inp != nullptr) {
            cur = ggml_add(ctx0, cur, ffn_inp);
            inpL_attn_norm = nullptr;
            inpL_qkv = { nullptr, nullptr, nullptr };
            cb(cur, "ffn_out", il);
        }

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if constexpr (!embed) {
        // lm_head
        cur = build_lora_mm(model.output, cur, model.output_s);

        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }

    ggml_build_forward_expand(gf, cur);
    //创建计算图节点，没有真正的计算，只是把计算图构建出来，后续会在llm_graph_context::compute()中进行计算
}

template struct llama_model_llama::graph<false>;
template struct llama_model_llama::graph<true>;
