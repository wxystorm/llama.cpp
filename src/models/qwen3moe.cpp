#include "llama-hybrid.h"
#include "models.h"

#include <cstdlib>

static int qwen3moe_ffn_chunk_count() {
    constexpr int default_chunks = 1;
    const char * value = std::getenv("LLAMA_CHUNKS");
    if (value == nullptr) {
        return default_chunks;
    }
    const int chunks = std::atoi(value);
    return chunks > 0 ? chunks : default_chunks;
}

void llama_model_qwen3moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_30B_A3B; break;
        case 94: type = LLM_TYPE_235B_A22B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3moe::load_arch_tensors(llama_model_loader &) {
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
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for QWEN3MOE");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for QWEN3MOE");
        }

        // MoE branch
        const int64_t n_ff_exp = hparams.n_ff_exp ? hparams.n_ff_exp : n_ff / n_expert_used;

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3moe::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen3moe::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    const bool stage_graph       = params.hybrid_layer_end >= 0;
    const int  layer_begin       = stage_graph ? params.hybrid_layer_begin : 0;
    const int  layer_end         = stage_graph ? params.hybrid_layer_end : n_layer;
    const bool hidden_input      = stage_graph && params.hybrid_hidden_input;
    const bool build_output_head = !stage_graph || params.hybrid_output_head;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(layer_begin >= 0 && layer_begin < layer_end && layer_end <= n_layer);
    GGML_ASSERT(hidden_input == (layer_begin > 0));
    GGML_ASSERT(!build_output_head || layer_end == n_layer);
    GGML_ASSERT(!hidden_input || (ubatch.token == nullptr && ubatch.embd != nullptr));

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = hidden_input ? build_inp_stage() : build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_output_head ? build_inp_out_ids() : nullptr;

    for (int il = layer_begin; il < layer_end; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        const llama_hybrid_layer_mode hybrid_mode =
            model.hybrid_layer_mode(il);
        const int n_decode_chunks =
            std::min<int64_t>(qwen3moe_ffn_chunk_count(), n_embd);
        const int planned_chunk_tokens =
            llama_hybrid_runtime_prefill_chunk_tokens();
        const bool use_decode_chunked_moe =
            n_tokens == 1 &&
            n_decode_chunks > 1 &&
            model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
            hybrid_mode == llama_hybrid_layer_mode::TENSOR_SPLIT &&
            loras->empty() &&
            cvec->tensor_for(il) == nullptr;
        const bool use_prefill_chunked_moe =
            n_tokens > 1 &&
            cur->ne[1] > 1 &&
            planned_chunk_tokens > 0 &&
            model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
            hybrid_mode == llama_hybrid_layer_mode::TENSOR_SPLIT &&
            loras->empty() &&
            cvec->tensor_for(il) == nullptr;

        if (use_prefill_chunked_moe) {
            const std::vector<int> raw_chunk_sizes =
                llama_hybrid_split_by_chunk_size(
                    (int) cur->ne[1], planned_chunk_tokens);
            const std::vector<int> chunk_sizes =
                llama_hybrid_split_tensor_chunks(
                    (int) cur->ne[1], planned_chunk_tokens);
            GGML_ASSERT(!chunk_sizes.empty());

            if (il == layer_begin) {
                auto chunk_list = [](const std::vector<int> & sizes) {
                    std::string result;
                    for (size_t i = 0; i < sizes.size(); ++i) {
                        if (i > 0) {
                            result += ",";
                        }
                        result += std::to_string(sizes[i]);
                    }
                    return result;
                };

                const bool merged = raw_chunk_sizes != chunk_sizes;
                const int tail = raw_chunk_sizes.empty() ? 0 : raw_chunk_sizes.back();
                LLAMA_LOG_ERROR(
                    "[TENSOR_CHUNK_PLAN] tokens=%" PRId64 " XT=%d raw=%s final=%s "
                    "action=%s tail=%d threshold=%d\n",
                    cur->ne[1], planned_chunk_tokens,
                    chunk_list(raw_chunk_sizes).c_str(),
                    chunk_list(chunk_sizes).c_str(),
                    merged ? "MERGE_TINY_TAIL" : "KEEP",
                    tail, planned_chunk_tokens / 4);
            }

            std::vector<ggml_tensor *> chunks;
            chunks.reserve(chunk_sizes.size());
            int64_t token_begin = 0;

            for (size_t i = 0; i < chunk_sizes.size(); ++i) {
                const int64_t token_count = chunk_sizes[i];
                GGML_ASSERT(token_count > 0);

                ggml_tensor * norm_chunk =
                    ggml_view_2d(
                        ctx0, cur, cur->ne[0], token_count,
                        cur->nb[1], token_begin * cur->nb[1]);
                const std::string norm_name =
                    "prefill_ffn_norm_chunk_" + std::to_string(i);
                cb(norm_chunk, norm_name.c_str(), il);

                ggml_tensor * moe_chunk =
                    build_moe_ffn(
                        norm_chunk,
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
                const std::string down_name =
                    "prefill_ffn_down_chunk_" + std::to_string(i);
                cb(moe_chunk, down_name.c_str(), il);
                chunks.push_back(moe_chunk);
                token_begin += token_count;
            }

            GGML_ASSERT(token_begin == cur->ne[1]);
            cur = chunks.back();
            for (int i = (int) chunks.size() - 2; i >= 0; --i) {
                cur = ggml_concat(ctx0, chunks[(size_t) i], cur, 1);
            }
            cur = ggml_add(ctx0, cur, ffn_inp);
        } else {
            std::vector<ggml_tensor *> decode_down_chunks;
            ggml_tensor * moe_out =
                build_moe_ffn(cur,
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
                        model.layers[il].ffn_down_exps_s,
                        nullptr,
                        use_decode_chunked_moe ? n_decode_chunks : 1,
                        use_decode_chunked_moe ? &decode_down_chunks : nullptr);

            if (use_decode_chunked_moe) {
                GGML_ASSERT(
                    (int) decode_down_chunks.size() == n_decode_chunks);

                std::vector<ggml_tensor *> output_chunks;
                output_chunks.reserve(decode_down_chunks.size());

                int64_t embd_begin = 0;
                for (ggml_tensor * down_chunk : decode_down_chunks) {
                    const int64_t embd_count = down_chunk->ne[0];
                    GGML_ASSERT(embd_count > 0);

                    ggml_tensor * residual_chunk =
                        ggml_view_2d(
                            ctx0,
                            ffn_inp,
                            embd_count,
                            ffn_inp->ne[1],
                            ffn_inp->nb[1],
                            embd_begin * ffn_inp->nb[0]);
                    output_chunks.push_back(
                        ggml_add(ctx0, down_chunk, residual_chunk));
                    embd_begin += embd_count;
                }
                GGML_ASSERT(embd_begin == ffn_inp->ne[0]);

                cur = output_chunks[0];
                for (size_t i = 1; i < output_chunks.size(); ++i) {
                    cur = ggml_concat(
                        ctx0, cur, output_chunks[i], 0);
                }
            } else {
                cb(moe_out, "ffn_moe_out", il);
                cur = ggml_add(ctx0, moe_out, ffn_inp);
            }
        }

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

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
