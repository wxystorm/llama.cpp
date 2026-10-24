#include "llama-hybrid.h"
#include "models.h"

#include <algorithm>
#include <cinttypes>
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

static int qwen3moe_wave_attn_target_tokens() {
    constexpr int default_target_tokens = 128;
    const char * value = std::getenv("LLAMA_HYBRID_ATTN_GROUP_TOKENS");
    if (value == nullptr) {
        return default_target_tokens;
    }
    const int tokens = std::atoi(value);
    return tokens > 0 ? tokens : 0;
}

static int qwen3moe_wave_attn_min_group_tokens(int target_tokens) {
    const char * value =
        std::getenv("LLAMA_HYBRID_ATTN_MIN_GROUP_TOKENS");
    if (value != nullptr) {
        const int tokens = std::atoi(value);
        return std::max(1, tokens);
    }
    return std::max(1, (target_tokens * 3 + 3) / 4);
}

static std::vector<int> qwen3moe_wave_attn_group_counts(
        const std::vector<int> & chunk_sizes,
        int                      target_tokens) {
    if (chunk_sizes.empty()) {
        return {};
    }
    if (target_tokens <= 0 || chunk_sizes.size() == 1) {
        return { (int) chunk_sizes.size() };
    }

    int total_tokens = 0;
    for (const int tokens : chunk_sizes) {
        total_tokens += tokens;
    }

    // Pick the nearest useful number of Attention groups to the requested
    // target size, then place boundaries only between Tensor-return chunks.
    int n_groups = std::max(
        1, (total_tokens + target_tokens / 2) / target_tokens);
    n_groups = std::min(n_groups, (int) chunk_sizes.size());
    if (n_groups <= 1) {
        return { (int) chunk_sizes.size() };
    }

    std::vector<int> result;
    result.reserve((size_t) n_groups);

    size_t chunk_index = 0;
    int remaining_tokens = total_tokens;
    for (int group = 0; group < n_groups; ++group) {
        const int remaining_groups = n_groups - group;
        const int remaining_chunks =
            (int) chunk_sizes.size() - (int) chunk_index;

        if (remaining_groups == 1) {
            result.push_back(remaining_chunks);
            break;
        }

        const int ideal_tokens =
            (remaining_tokens + remaining_groups / 2) / remaining_groups;
        const int max_take =
            remaining_chunks - (remaining_groups - 1);

        int group_chunks = 0;
        int group_tokens = 0;
        while (group_chunks < max_take) {
            const int next_tokens =
                chunk_sizes[chunk_index + (size_t) group_chunks];
            if (group_chunks > 0) {
                const int stop_error =
                    group_tokens > ideal_tokens ?
                        group_tokens - ideal_tokens :
                        ideal_tokens - group_tokens;
                const int take_tokens = group_tokens + next_tokens;
                const int take_error =
                    take_tokens > ideal_tokens ?
                        take_tokens - ideal_tokens :
                        ideal_tokens - take_tokens;
                // On an equal-distance boundary prefer the larger coarse
                // Attention group. Starting the next layer a little later is
                // preferable to reintroducing an XT-sized Attention kernel.
                if (stop_error < take_error) {
                    break;
                }
            }
            group_tokens += next_tokens;
            ++group_chunks;
        }

        if (group_chunks == 0) {
            group_chunks = 1;
            group_tokens = chunk_sizes[chunk_index];
        }

        result.push_back(group_chunks);
        chunk_index += (size_t) group_chunks;
        remaining_tokens -= group_tokens;
    }

    return result;
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

    const int planned_chunk_tokens =
        llama_hybrid_runtime_prefill_chunk_tokens();
    const bool return_wavefront_requested =
        std::getenv("LLAMA_HYBRID_RETURN_WAVEFRONT") != nullptr;

    bool moe_stage_wavefront =
        return_wavefront_requested &&
        stage_graph &&
        n_tokens > 1 &&
        ubatch.n_pos == 1 &&
        ubatch.n_seqs_unq == 1 &&
        (!ubatch.equal_seqs() || ubatch.n_seqs == 1) &&
        model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
        planned_chunk_tokens > 0 &&
        layer_end - layer_begin > 1 &&
        loras->empty();

    for (int il = layer_begin; moe_stage_wavefront && il < layer_end; ++il) {
        moe_stage_wavefront =
            model.hybrid_layer_mode(il) ==
                llama_hybrid_layer_mode::TENSOR_SPLIT &&
            cvec->tensor_for(il) == nullptr;
    }

    std::vector<int> wave_chunk_sizes;
    std::vector<int> wave_attn_group_counts;
    const int wave_attn_target_tokens =
        qwen3moe_wave_attn_target_tokens();
    if (moe_stage_wavefront) {
        wave_chunk_sizes =
            llama_hybrid_runtime_tensor_chunks(
                (int) n_tokens, planned_chunk_tokens);
        wave_attn_group_counts =
            qwen3moe_wave_attn_group_counts(
                wave_chunk_sizes, wave_attn_target_tokens);

        // A single coarse Attention group is equivalent to the old layer
        // barrier but carries extra wavefront bookkeeping, so keep the
        // existing macro-Attention path for small Tensor jobs.
        moe_stage_wavefront =
            wave_chunk_sizes.size() > 1 &&
            wave_attn_group_counts.size() > 1;

        // Do not call a layout "coarse" if any Attention group collapses back
        // to roughly one XT chunk. This is the failure mode of 192 tokens with
        // XT=64,target=128: the only two-group partition is 128+64. The small
        // 64-token Attention can cost more than the return barrier it hides.
        // Requiring >= 75% of the target keeps useful layouts such as
        // 103+153 (target=128) and 128+128, while falling back for 128+64.
        if (moe_stage_wavefront) {
            const int min_group_tokens =
                qwen3moe_wave_attn_min_group_tokens(
                    wave_attn_target_tokens);
            size_t chunk_begin = 0;
            for (const int group_chunks : wave_attn_group_counts) {
                int group_tokens = 0;
                for (int i = 0; i < group_chunks; ++i) {
                    group_tokens +=
                        wave_chunk_sizes[chunk_begin + (size_t) i];
                }
                if (group_tokens < min_group_tokens) {
                    moe_stage_wavefront = false;
                    break;
                }
                chunk_begin += (size_t) group_chunks;
            }
        }
    }

    if (return_wavefront_requested && stage_graph && n_tokens > 1) {
        LLAMA_LOG_DEBUG(
            "[MOE_WAVEFRONT_ELIGIBILITY] enabled=%d tokens=%" PRId64
            " layers=[%d,%d) equal_seqs=%d n_seqs=%u n_seqs_unq=%u "
            "XT=%d target=%d min_group=%d chunks=%zu groups=%zu\n",
            moe_stage_wavefront ? 1 : 0,
            n_tokens, layer_begin, layer_end,
            ubatch.equal_seqs() ? 1 : 0,
            ubatch.n_seqs, ubatch.n_seqs_unq,
            planned_chunk_tokens, wave_attn_target_tokens,
            qwen3moe_wave_attn_min_group_tokens(
                wave_attn_target_tokens),
            wave_chunk_sizes.size(),
            wave_attn_group_counts.size());
    }

    llm_graph_input_attn_kv * inp_attn =
        moe_stage_wavefront ? nullptr : build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_output_head ? build_inp_out_ids() : nullptr;

    const char * region_chunk_cont_env =
        std::getenv("LLAMA_HYBRID_REGION_CHUNK_CONTINUATION");
    const bool region_chunk_cont_requested =
        region_chunk_cont_env != nullptr &&
        std::atoi(region_chunk_cont_env) != 0;

    int region_c2p_layer = -1;
    int region_p2c_layer = -1;
    int region_c2p_count = 0;
    int region_p2c_count = 0;
    bool directional_region_chunk_cont =
        region_chunk_cont_requested &&
        !stage_graph &&
        n_tokens > 1 &&
        ubatch.n_pos == 1 &&
        ubatch.n_seqs_unq == 1 &&
        (!ubatch.equal_seqs() || ubatch.n_seqs == 1) &&
        planned_chunk_tokens > 0 &&
        model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
        loras->empty();

    if (directional_region_chunk_cont) {
        for (int il = 0; il < n_layer; ++il) {
            const llama_hybrid_layer_mode mode =
                model.hybrid_layer_mode(il);
            if (mode == llama_hybrid_layer_mode::TENSOR_C2P) {
                ++region_c2p_count;
                region_c2p_layer = il;
            } else if (mode == llama_hybrid_layer_mode::TENSOR_P2C) {
                ++region_p2c_count;
                region_p2c_layer = il;
            }
        }

        directional_region_chunk_cont =
            region_c2p_count == 1 &&
            region_p2c_count == 1 &&
            region_c2p_layer >= 0 &&
            region_p2c_layer > region_c2p_layer + 1 &&
            region_p2c_layer < n_layer - 1;

        for (int il = region_c2p_layer + 1;
                directional_region_chunk_cont &&
                il < region_p2c_layer;
                ++il) {
            directional_region_chunk_cont =
                model.hybrid_layer_mode(il) ==
                    llama_hybrid_layer_mode::PHONE_ONLY &&
                cvec->tensor_for(il) == nullptr;
        }

        if (directional_region_chunk_cont) {
            directional_region_chunk_cont =
                cvec->tensor_for(region_c2p_layer) == nullptr &&
                cvec->tensor_for(region_p2c_layer) == nullptr &&
                llama_hybrid_runtime_tensor_chunks(
                    (int) n_tokens,
                    planned_chunk_tokens).size() > 1;
        }
    }

    if (region_chunk_cont_requested && n_tokens > 1) {
        LLAMA_LOG_ERROR(
            "[REGION_CONT_FULL_GRAPH_CAP] enabled=%d tokens=%" PRId64
            " c2p=%d p2c=%d c2p_count=%d p2c_count=%d "
            "stage_graph=%d XT=%d status=%s\n",
            directional_region_chunk_cont ? 1 : 0,
            n_tokens,
            region_c2p_layer,
            region_p2c_layer,
            region_c2p_count,
            region_p2c_count,
            stage_graph ? 1 : 0,
            planned_chunk_tokens,
            directional_region_chunk_cont ? "READY" : "DISABLED");
    }

    auto build_directional_region =
        [&](ggml_tensor * region_inp) -> ggml_tensor * {
            GGML_ASSERT(directional_region_chunk_cont);
            const int c2p_layer = region_c2p_layer;
            const int p2c_layer = region_p2c_layer;
            const std::vector<int> chunk_sizes =
                llama_hybrid_runtime_tensor_chunks(
                    (int) n_tokens,
                    planned_chunk_tokens);
            GGML_ASSERT(chunk_sizes.size() > 1);

            std::string layout;
            for (size_t ci = 0; ci < chunk_sizes.size(); ++ci) {
                if (!layout.empty()) {
                    layout += ",";
                }
                layout += std::to_string(chunk_sizes[ci]);
            }
            LLAMA_LOG_ERROR(
                "[REGION_CONT_GRAPH] enabled=1 layers=[%d,%d) "
                "c2p=%d phone=[%d,%d) p2c=%d tokens=%" PRId64
                " XT=%d chunks=%s mode=BOUNDARY_SEED_THEN_CHUNK_MAJOR\n",
                c2p_layer,
                p2c_layer + 1,
                c2p_layer,
                c2p_layer + 1,
                p2c_layer,
                p2c_layer,
                n_tokens,
                planned_chunk_tokens,
                layout.c_str());

            std::vector<int64_t> chunk_begin(
                chunk_sizes.size(), 0);
            int64_t token_cursor = 0;
            for (size_t ci = 0; ci < chunk_sizes.size(); ++ci) {
                chunk_begin[ci] = token_cursor;
                token_cursor += chunk_sizes[ci];
            }
            GGML_ASSERT(token_cursor == n_tokens);

            // Boundary seed: L34 Attention/Router stays coarse on PC, then
            // expose all C2P FFN chunks.  Meta can place the Phone shard on
            // PHONE_C2P_FFN and retire each chunk independently into the
            // stable Phone l_out backing.
            ggml_tensor * c2p_inp = region_inp;
            ggml_tensor * c2p_attn_norm =
                build_norm(
                    c2p_inp,
                    model.layers[c2p_layer].attn_norm,
                    NULL, LLM_NORM_RMS, c2p_layer);
            cb(c2p_attn_norm, "attn_norm", c2p_layer);

            auto [c2p_q, c2p_k, c2p_v] =
                build_qkv(
                    model.layers[c2p_layer],
                    c2p_attn_norm,
                    n_embd_head, n_head, n_head_kv,
                    c2p_layer);
            c2p_q = build_norm(
                c2p_q,
                model.layers[c2p_layer].attn_q_norm,
                NULL, LLM_NORM_RMS, c2p_layer);
            cb(c2p_q, "Qcur_normed", c2p_layer);
            c2p_k = build_norm(
                c2p_k,
                model.layers[c2p_layer].attn_k_norm,
                NULL, LLM_NORM_RMS, c2p_layer);
            cb(c2p_k, "Kcur_normed", c2p_layer);

            c2p_q = ggml_rope_ext(
                ctx0, c2p_q, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale,
                ext_factor, attn_factor,
                beta_fast, beta_slow);
            c2p_k = ggml_rope_ext(
                ctx0, c2p_k, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale,
                ext_factor, attn_factor,
                beta_fast, beta_slow);
            cb(c2p_q, "Qcur", c2p_layer);
            cb(c2p_k, "Kcur", c2p_layer);
            cb(c2p_v, "Vcur", c2p_layer);

            ggml_tensor * c2p_attn =
                build_attn(
                    inp_attn,
                    model.layers[c2p_layer].wo,
                    model.layers[c2p_layer].wo_b,
                    model.layers[c2p_layer].wo_s,
                    c2p_q, c2p_k, c2p_v,
                    nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)),
                    c2p_layer);
            ggml_tensor * c2p_ffn_inp =
                ggml_add(ctx0, c2p_attn, c2p_inp);
            cb(c2p_ffn_inp, "ffn_inp", c2p_layer);

            ggml_tensor * c2p_ffn_norm =
                build_norm(
                    c2p_ffn_inp,
                    model.layers[c2p_layer].ffn_norm,
                    NULL, LLM_NORM_RMS, c2p_layer);
            cb(c2p_ffn_norm, "ffn_norm", c2p_layer);

            // Stable C2P output backing.  This producer intentionally has no
            // dependency on the C2P FFN chunks: Meta disables the DUP before
            // the first continuation consumer and writes each reduced chunk
            // directly into this tensor on Phone.
            ggml_tensor * c2p_l_out =
                ggml_dup(ctx0, c2p_ffn_inp);
            cb(c2p_l_out, "l_out", c2p_layer);

            std::vector<ggml_tensor *> p2c_down_chunks;
            std::vector<ggml_tensor *> p2c_residual_chunks;
            p2c_down_chunks.reserve(chunk_sizes.size());
            p2c_residual_chunks.reserve(chunk_sizes.size());

            // Chunk-major Phone continuation.  For each chunk, L35..L45 are
            // fully Phone-local and L46 Attention/Router stays on Phone before
            // the P2C split FFN boundary.
            for (size_t ci = 0; ci < chunk_sizes.size(); ++ci) {
                const int64_t count = chunk_sizes[ci];

                // Seed exactly one C2P FFN chunk, then immediately continue
                // that same chunk through the Phone-owned region.
                ggml_tensor * c2p_norm_chunk =
                    ggml_view_2d(
                        ctx0,
                        c2p_ffn_norm,
                        c2p_ffn_norm->ne[0],
                        count,
                        c2p_ffn_norm->nb[1],
                        chunk_begin[ci] * c2p_ffn_norm->nb[1]);
                const std::string c2p_norm_name =
                    "prefill_ffn_norm_chunk_" +
                    std::to_string(ci);
                cb(
                    c2p_norm_chunk,
                    c2p_norm_name.c_str(),
                    c2p_layer);

                llm_graph_moe_routing c2p_routing =
                    build_moe_routing(
                        c2p_norm_chunk,
                        model.layers[c2p_layer].ffn_gate_inp,
                        nullptr, nullptr,
                        n_expert, n_expert_used,
                        true,
                        hparams.expert_weights_scale,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        c2p_layer);
                const std::string c2p_topk_name =
                    "c2p_prefill_route_topk_chunk_" +
                    std::to_string(ci);
                const std::string c2p_weights_name =
                    "c2p_prefill_route_weights_chunk_" +
                    std::to_string(ci);
                cb(
                    c2p_routing.selected_experts,
                    c2p_topk_name.c_str(),
                    c2p_layer);
                cb(
                    c2p_routing.weights,
                    c2p_weights_name.c_str(),
                    c2p_layer);

                ggml_tensor * c2p_down =
                    build_moe_ffn(
                        c2p_norm_chunk,
                        model.layers[c2p_layer].ffn_gate_inp,
                        model.layers[c2p_layer].ffn_up_exps,
                        model.layers[c2p_layer].ffn_gate_exps,
                        model.layers[c2p_layer].ffn_down_exps,
                        nullptr,
                        n_expert, n_expert_used,
                        LLM_FFN_SILU, true,
                        hparams.expert_weights_scale,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        c2p_layer,
                        nullptr, nullptr,
                        model.layers[c2p_layer].ffn_up_exps_s,
                        model.layers[c2p_layer].ffn_gate_exps_s,
                        model.layers[c2p_layer].ffn_down_exps_s,
                        c2p_routing.selected_experts,
                        1,
                        nullptr,
                        c2p_routing.weights);
                const std::string c2p_down_name =
                    "prefill_ffn_down_chunk_" +
                    std::to_string(ci);
                cb(
                    c2p_down,
                    c2p_down_name.c_str(),
                    c2p_layer);
                ggml_build_forward_expand(gf, c2p_down);

                ggml_tensor * chunk_hidden =
                    ggml_view_2d(
                        ctx0,
                        c2p_l_out,
                        c2p_l_out->ne[0],
                        count,
                        c2p_l_out->nb[1],
                        chunk_begin[ci] * c2p_l_out->nb[1]);
                const std::string entry_name =
                    "region_cont_entry_chunk_" +
                    std::to_string(ci);
                cb(
                    chunk_hidden,
                    entry_name.c_str(),
                    c2p_layer + 1);

                for (int il = c2p_layer + 1;
                        il <= p2c_layer;
                        ++il) {
                    const llama_hybrid_layer_mode mode =
                        model.hybrid_layer_mode(il);
                    const bool p2c =
                        mode ==
                            llama_hybrid_layer_mode::TENSOR_P2C;
                    GGML_ASSERT(
                        p2c ||
                        mode ==
                            llama_hybrid_layer_mode::PHONE_ONLY);

                    ggml_tensor * attn_norm =
                        build_norm(
                            chunk_hidden,
                            model.layers[il].attn_norm,
                            NULL, LLM_NORM_RMS, il);
                    const std::string attn_norm_name =
                        "region_cont_attn_norm_chunk_" +
                        std::to_string(ci);
                    cb(attn_norm, attn_norm_name.c_str(), il);

                    auto [Qcur, Kcur, Vcur] =
                        build_qkv(
                            model.layers[il],
                            attn_norm,
                            n_embd_head, n_head, n_head_kv,
                            il);
                    Qcur = build_norm(
                        Qcur,
                        model.layers[il].attn_q_norm,
                        NULL, LLM_NORM_RMS, il);
                    cb(Qcur, "Qcur_normed", il);
                    Kcur = build_norm(
                        Kcur,
                        model.layers[il].attn_k_norm,
                        NULL, LLM_NORM_RMS, il);
                    cb(Kcur, "Kcur_normed", il);

                    ggml_tensor * pos_chunk =
                        ggml_view_1d(
                            ctx0,
                            inp_pos,
                            count,
                            chunk_begin[ci] * inp_pos->nb[0]);
                    Qcur = ggml_rope_ext(
                        ctx0, Qcur, pos_chunk, nullptr,
                        n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale,
                        ext_factor, attn_factor,
                        beta_fast, beta_slow);
                    Kcur = ggml_rope_ext(
                        ctx0, Kcur, pos_chunk, nullptr,
                        n_rot, rope_type, n_ctx_orig,
                        freq_base, freq_scale,
                        ext_factor, attn_factor,
                        beta_fast, beta_slow);
                    cb(Qcur, "Qcur", il);
                    cb(Kcur, "Kcur", il);
                    cb(Vcur, "Vcur", il);

                    auto * attn_chunk =
                        build_attn_inp_kv_range(
                            (uint32_t) chunk_begin[ci],
                            (uint32_t) count);
                    ggml_tensor * attn =
                        build_attn(
                            attn_chunk,
                            model.layers[il].wo,
                            model.layers[il].wo_b,
                            model.layers[il].wo_s,
                            Qcur, Kcur, Vcur,
                            nullptr, nullptr, nullptr,
                            1.0f / sqrtf(float(n_embd_head)),
                            il);
                    const std::string attn_name =
                        "prefill_wave_attn_out_group_" +
                        std::to_string(ci) + "_1";
                    cb(attn, attn_name.c_str(), il);

                    ggml_tensor * ffn_inp =
                        ggml_add(ctx0, attn, chunk_hidden);
                    const std::string ffn_inp_name =
                        "prefill_wave_ffn_inp_chunk_" +
                        std::to_string(ci);
                    cb(ffn_inp, ffn_inp_name.c_str(), il);

                    ggml_tensor * ffn_norm =
                        build_norm(
                            ffn_inp,
                            model.layers[il].ffn_norm,
                            NULL, LLM_NORM_RMS, il);
                    const std::string norm_name =
                        "prefill_ffn_norm_chunk_" +
                        std::to_string(ci);
                    cb(ffn_norm, norm_name.c_str(), il);

                    llm_graph_moe_routing routing;
                    if (p2c) {
                        routing =
                            build_moe_routing(
                                ffn_norm,
                                model.layers[il].ffn_gate_inp,
                                nullptr, nullptr,
                                n_expert, n_expert_used,
                                true,
                                hparams.expert_weights_scale,
                                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                                il);
                        const std::string topk_name =
                            "phone_prefill_route_topk_chunk_" +
                            std::to_string(ci);
                        const std::string weights_name =
                            "phone_prefill_route_weights_chunk_" +
                            std::to_string(ci);
                        cb(
                            routing.selected_experts,
                            topk_name.c_str(),
                            il);
                        cb(
                            routing.weights,
                            weights_name.c_str(),
                            il);
                    }

                    ggml_tensor * down =
                        build_moe_ffn(
                            ffn_norm,
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
                            routing.selected_experts,
                            1,
                            nullptr,
                            routing.weights);

                    if (p2c) {
                        const std::string down_name =
                            "prefill_ffn_down_chunk_" +
                            std::to_string(ci);
                        cb(down, down_name.c_str(), il);
                        ggml_build_forward_expand(gf, down);
                        p2c_down_chunks.push_back(down);
                        p2c_residual_chunks.push_back(ffn_inp);
                        break;
                    }

                    const std::string local_down_name =
                        "region_phone_ffn_down_chunk_" +
                        std::to_string(ci);
                    cb(down, local_down_name.c_str(), il);

                    ggml_tensor * l_out =
                        ggml_add(ctx0, down, ffn_inp);
                    const std::string l_out_name =
                        "prefill_wave_l_out_chunk_" +
                        std::to_string(ci);
                    cb(l_out, l_out_name.c_str(), il);
                    ggml_build_forward_expand(gf, l_out);
                    chunk_hidden = l_out;
                }
            }

            GGML_ASSERT(
                p2c_down_chunks.size() ==
                chunk_sizes.size());
            GGML_ASSERT(
                p2c_residual_chunks.size() ==
                chunk_sizes.size());

            // Logical L46 output is again stable destination storage.  P2C
            // Meta disables this producer and commits the reduced chunks into
            // the PC backing; the following PC stage consumes that completed
            // activation.
            ggml_tensor * p2c_down = p2c_down_chunks[0];
            ggml_tensor * p2c_residual =
                p2c_residual_chunks[0];
            for (size_t ci = 1; ci < p2c_down_chunks.size(); ++ci) {
                p2c_down =
                    ggml_concat(
                        ctx0,
                        p2c_down,
                        p2c_down_chunks[ci],
                        1);
                p2c_residual =
                    ggml_concat(
                        ctx0,
                        p2c_residual,
                        p2c_residual_chunks[ci],
                        1);
            }
            cur = ggml_add(ctx0, p2c_down, p2c_residual);
            cb(cur, "l_out", p2c_layer);
            ggml_build_forward_expand(gf, cur);

            return cur;
        };

    if (moe_stage_wavefront) {
        std::vector<int64_t> chunk_begin(
            wave_chunk_sizes.size(), 0);
        std::vector<ggml_tensor *> hidden_chunks;
        std::vector<ggml_tensor *> pending_wave_down_chunks(
            wave_chunk_sizes.size(), nullptr);
        std::vector<ggml_tensor *> pending_wave_residual_chunks(
            wave_chunk_sizes.size(), nullptr);
        hidden_chunks.reserve(wave_chunk_sizes.size());

        int64_t token_begin = 0;
        for (size_t ci = 0; ci < wave_chunk_sizes.size(); ++ci) {
            chunk_begin[ci] = token_begin;
            token_begin += wave_chunk_sizes[ci];
        }
        GGML_ASSERT(token_begin == n_tokens);

        auto list_tokens = [](const std::vector<int> & values) {
            std::string result;
            for (size_t i = 0; i < values.size(); ++i) {
                if (i > 0) {
                    result += ",";
                }
                result += std::to_string(values[i]);
            }
            return result;
        };

        std::vector<int> coarse_group_tokens;
        size_t coarse_chunk_begin = 0;
        for (const int group_chunks : wave_attn_group_counts) {
            int group_tokens = 0;
            for (int i = 0; i < group_chunks; ++i) {
                group_tokens +=
                    wave_chunk_sizes[
                        coarse_chunk_begin + (size_t) i];
            }
            coarse_group_tokens.push_back(group_tokens);
            coarse_chunk_begin += (size_t) group_chunks;
        }
        GGML_ASSERT(
            coarse_chunk_begin == wave_chunk_sizes.size());

        LLAMA_LOG_DEBUG(
            "[MOE_WAVEFRONT_LAYOUT] layers=[%d,%d) tokens=%" PRId64
            " XT=%d attn_target=%d chunks=%s groups=%s "
            "first_attn=%" PRId64 "\n",
            layer_begin, layer_end, n_tokens,
            planned_chunk_tokens, wave_attn_target_tokens,
            list_tokens(wave_chunk_sizes).c_str(),
            list_tokens(coarse_group_tokens).c_str(),
            n_tokens);

        ggml_tensor * wave_entry = ggml_cont(ctx0, inpL);
        cb(wave_entry, "prefill_wave_entry", layer_begin);

        for (size_t ci = 0; ci < wave_chunk_sizes.size(); ++ci) {
            const int64_t count = wave_chunk_sizes[ci];
            hidden_chunks.push_back(
                ggml_view_2d(
                    ctx0, wave_entry, wave_entry->ne[0], count,
                    wave_entry->nb[1],
                    chunk_begin[ci] * wave_entry->nb[1]));
        }

        for (int wave_layer = layer_begin;
             wave_layer < layer_end;
             ++wave_layer) {
            // The first Tensor layer already owns a complete macro from the
            // previous stage. Keep its Attention fully fused. Only later
            // layers trade coarse Attention grouping for cross-layer return
            // overlap.
            const std::vector<int> layer_group_counts =
                wave_layer == layer_begin ?
                    std::vector<int>{ (int) wave_chunk_sizes.size() } :
                    wave_attn_group_counts;

            size_t group_start = 0;
            for (const int group_chunk_count : layer_group_counts) {
                const size_t group_end =
                    group_start + (size_t) group_chunk_count;
                GGML_ASSERT(group_chunk_count > 0);
                GGML_ASSERT(group_end <= wave_chunk_sizes.size());

                const int64_t group_token_begin =
                    chunk_begin[group_start];
                int64_t group_token_count = 0;

                for (size_t ci = group_start;
                     ci < group_end;
                     ++ci) {
                    if (wave_layer > layer_begin) {
                        GGML_ASSERT(
                            pending_wave_down_chunks[ci] != nullptr);
                        GGML_ASSERT(
                            pending_wave_residual_chunks[ci] != nullptr);

                        ggml_tensor * wave_l_out =
                            ggml_add(
                                ctx0,
                                pending_wave_down_chunks[ci],
                                pending_wave_residual_chunks[ci]);
                        const std::string wave_l_out_name =
                            "prefill_wave_l_out_chunk_" +
                            std::to_string(ci);
                        cb(
                            wave_l_out,
                            wave_l_out_name.c_str(),
                            wave_layer - 1);
                        ggml_build_forward_expand(gf, wave_l_out);
                        hidden_chunks[ci] = wave_l_out;
                    }
                    group_token_count += wave_chunk_sizes[ci];
                }

                ggml_tensor * group_inp = nullptr;
                if (wave_layer == layer_begin &&
                    group_start == 0 &&
                    group_end == wave_chunk_sizes.size()) {
                    group_inp = wave_entry;
                } else {
                    group_inp = hidden_chunks[group_start];
                    for (size_t ci = group_start + 1;
                         ci < group_end;
                         ++ci) {
                        group_inp =
                            ggml_concat(
                                ctx0, group_inp,
                                hidden_chunks[ci], 1);
                    }
                }

                ggml_tensor * attn_norm =
                    build_norm(
                        group_inp,
                        model.layers[wave_layer].attn_norm,
                        NULL, LLM_NORM_RMS, wave_layer);
                const std::string attn_norm_name =
                    "prefill_wave_attn_norm_group_" +
                    std::to_string(group_start) + "_" +
                    std::to_string(group_chunk_count);
                cb(
                    attn_norm,
                    attn_norm_name.c_str(),
                    wave_layer);

                auto [Qcur, Kcur, Vcur] =
                    build_qkv(
                        model.layers[wave_layer],
                        attn_norm,
                        n_embd_head, n_head,
                        n_head_kv, wave_layer);

                Qcur = build_norm(
                    Qcur,
                    model.layers[wave_layer].attn_q_norm,
                    NULL, LLM_NORM_RMS, wave_layer);
                cb(Qcur, "Qcur_normed", wave_layer);

                Kcur = build_norm(
                    Kcur,
                    model.layers[wave_layer].attn_k_norm,
                    NULL, LLM_NORM_RMS, wave_layer);
                cb(Kcur, "Kcur_normed", wave_layer);

                ggml_tensor * pos_group =
                    ggml_view_1d(
                        ctx0, inp_pos,
                        group_token_count,
                        group_token_begin * inp_pos->nb[0]);

                Qcur = ggml_rope_ext(
                    ctx0, Qcur, pos_group, nullptr,
                    n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale,
                    ext_factor, attn_factor,
                    beta_fast, beta_slow);

                Kcur = ggml_rope_ext(
                    ctx0, Kcur, pos_group, nullptr,
                    n_rot, rope_type, n_ctx_orig,
                    freq_base, freq_scale,
                    ext_factor, attn_factor,
                    beta_fast, beta_slow);

                cb(Qcur, "Qcur", wave_layer);
                cb(Kcur, "Kcur", wave_layer);
                cb(Vcur, "Vcur", wave_layer);

                auto * attn_group =
                    build_attn_inp_kv_range(
                        (uint32_t) group_token_begin,
                        (uint32_t) group_token_count);
                ggml_tensor * attn =
                    build_attn(
                        attn_group,
                        model.layers[wave_layer].wo,
                        model.layers[wave_layer].wo_b,
                        model.layers[wave_layer].wo_s,
                        Qcur, Kcur, Vcur,
                        nullptr, nullptr, nullptr,
                        1.0f / sqrtf(float(n_embd_head)),
                        wave_layer);
                const std::string attn_name =
                    "prefill_wave_attn_out_group_" +
                    std::to_string(group_start) + "_" +
                    std::to_string(group_chunk_count);
                cb(attn, attn_name.c_str(), wave_layer);

                ggml_tensor * group_ffn_inp =
                    ggml_add(ctx0, attn, group_inp);

                // Keep FFN normalization coarse as well. The old wavefront
                // normalized every XT chunk independently, fragmenting a PC
                // operation that the normal Qwen3-MoE path performs once for
                // the whole macro. Normalize once per coarse Attention group,
                // then expose XT-granular views for Router/Top-K/experts and
                // Meta's existing prefill chunk boundary detection.
                ggml_tensor * group_ffn_norm =
                    build_norm(
                        group_ffn_inp,
                        model.layers[wave_layer].ffn_norm,
                        NULL, LLM_NORM_RMS, wave_layer);
                const std::string group_norm_name =
                    "prefill_wave_ffn_norm_group_" +
                    std::to_string(group_start) + "_" +
                    std::to_string(group_chunk_count);
                cb(
                    group_ffn_norm,
                    group_norm_name.c_str(),
                    wave_layer);

                int64_t group_offset_tokens = 0;
                for (size_t ci = group_start;
                     ci < group_end;
                     ++ci) {
                    const int64_t count =
                        wave_chunk_sizes[ci];
                    ggml_tensor * ffn_inp =
                        ggml_view_2d(
                            ctx0, group_ffn_inp,
                            group_ffn_inp->ne[0], count,
                            group_ffn_inp->nb[1],
                            group_offset_tokens *
                                group_ffn_inp->nb[1]);
                    const std::string ffn_inp_name =
                        "prefill_wave_ffn_inp_chunk_" +
                        std::to_string(ci);
                    cb(
                        ffn_inp,
                        ffn_inp_name.c_str(),
                        wave_layer);

                    ggml_tensor * ffn_norm =
                        ggml_view_2d(
                            ctx0, group_ffn_norm,
                            group_ffn_norm->ne[0], count,
                            group_ffn_norm->nb[1],
                            group_offset_tokens *
                                group_ffn_norm->nb[1]);
                    const std::string norm_name =
                        "prefill_ffn_norm_chunk_" +
                        std::to_string(ci);
                    cb(
                        ffn_norm,
                        norm_name.c_str(),
                        wave_layer);

                    ggml_tensor * down =
                        build_moe_ffn(
                            ffn_norm,
                            model.layers[wave_layer].ffn_gate_inp,
                            model.layers[wave_layer].ffn_up_exps,
                            model.layers[wave_layer].ffn_gate_exps,
                            model.layers[wave_layer].ffn_down_exps,
                            nullptr,
                            n_expert, n_expert_used,
                            LLM_FFN_SILU, true,
                            hparams.expert_weights_scale,
                            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                            wave_layer,
                            nullptr, nullptr,
                            model.layers[wave_layer].ffn_up_exps_s,
                            model.layers[wave_layer].ffn_gate_exps_s,
                            model.layers[wave_layer].ffn_down_exps_s);
                    const std::string down_name =
                        "prefill_ffn_down_chunk_" +
                        std::to_string(ci);
                    cb(
                        down,
                        down_name.c_str(),
                        wave_layer);

                    ggml_build_forward_expand(gf, down);
                    pending_wave_down_chunks[ci] = down;
                    pending_wave_residual_chunks[ci] =
                        ffn_inp;
                    group_offset_tokens += count;
                }
                GGML_ASSERT(
                    group_offset_tokens ==
                    group_token_count);
                group_start = group_end;
            }
            GGML_ASSERT(
                group_start == wave_chunk_sizes.size());
        }

        for (size_t ci = 0;
             ci < wave_chunk_sizes.size();
             ++ci) {
            GGML_ASSERT(
                pending_wave_down_chunks[ci] != nullptr);
            GGML_ASSERT(
                pending_wave_residual_chunks[ci] != nullptr);

            ggml_tensor * wave_l_out =
                ggml_add(
                    ctx0,
                    pending_wave_down_chunks[ci],
                    pending_wave_residual_chunks[ci]);
            const std::string wave_l_out_name =
                "prefill_wave_l_out_chunk_" +
                std::to_string(ci);
            cb(
                wave_l_out,
                wave_l_out_name.c_str(),
                layer_end - 1);
            ggml_build_forward_expand(gf, wave_l_out);
            hidden_chunks[ci] = wave_l_out;
        }

        cur = hidden_chunks[0];
        for (size_t ci = 1;
             ci < hidden_chunks.size();
             ++ci) {
            cur =
                ggml_concat(
                    ctx0, cur,
                    hidden_chunks[ci], 1);
        }
        if (hidden_chunks.size() > 1) {
            cb(cur, "l_out", layer_end - 1);
        }

        if (!build_output_head) {
            res->t_stage_output = cur;
            ggml_build_forward_expand(gf, cur);
            return;
        }

        if (inp_out_ids) {
            cur = ggml_get_rows(
                ctx0, cur, inp_out_ids);
        }

        cur = build_norm(
            cur, model.output_norm, NULL,
            LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);
        res->t_embd = cur;

        cur = build_lora_mm(
            model.output, cur, model.output_s);
        cb(cur, "result_output", -1);
        res->t_logits = cur;

        ggml_build_forward_expand(gf, cur);
        return;
    }

    for (int il = layer_begin; il < layer_end; ++il) {
        if (directional_region_chunk_cont &&
                il == region_c2p_layer) {
            res->t_layer_inp[il] = inpL;
            cur = build_directional_region(inpL);
            inpL = cur;
            il = region_p2c_layer;
            continue;
        }

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
        const llama_hybrid_layer_mode hybrid_mode =
            model.hybrid_layer_mode(il);
        const bool defer_last_output_rows =
            il == n_layer - 1 &&
            inp_out_ids != nullptr &&
            llama_hybrid_layer_mode_tensor_attention_on_phone(hybrid_mode);

        // Normally the last layer is pruned to only the requested output rows
        // before the FFN.  TENSOR_PHONE_PRIMARY needs the same full-token
        // Attention -> residual -> FFN topology as the preceding tensor layers:
        // pruning here gives the two Meta backend copies a different final-layer
        // graph shape and can leave ffn_inp unsynchronized.  Defer the row
        // selection until after the complete last-layer residual is materialized.
        if (il == n_layer - 1 && inp_out_ids && !defer_last_output_rows) {
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

        const int n_decode_chunks =
            std::min<int64_t>(qwen3moe_ffn_chunk_count(), n_embd);
        const bool tensor_split_layer =
            llama_hybrid_layer_mode_is_tensor(hybrid_mode);
        const bool phone_primary_router =
            llama_hybrid_layer_mode_tensor_attention_on_phone(hybrid_mode);
        const bool c2p_router =
            hybrid_mode == llama_hybrid_layer_mode::TENSOR_C2P;
        const bool explicit_router =
            phone_primary_router || c2p_router;
        const bool use_decode_chunked_moe =
            n_tokens == 1 &&
            n_decode_chunks > 1 &&
            model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
            tensor_split_layer &&
            loras->empty() &&
            cvec->tensor_for(il) == nullptr;
        const bool use_prefill_chunked_moe =
            n_tokens > 1 &&
            cur->ne[1] > 1 &&
            planned_chunk_tokens > 0 &&
            model.split_mode() == LLAMA_SPLIT_MODE_TENSOR &&
            tensor_split_layer &&
            loras->empty() &&
            cvec->tensor_for(il) == nullptr;

        if (use_prefill_chunked_moe) {
            const std::vector<int> raw_chunk_sizes =
                llama_hybrid_split_by_chunk_size(
                    (int) cur->ne[1], planned_chunk_tokens);
            const std::vector<int> chunk_sizes =
                llama_hybrid_runtime_tensor_chunks(
                    (int) cur->ne[1], planned_chunk_tokens);
            GGML_ASSERT(!chunk_sizes.empty());

            const bool first_tensor_layer =
                il == layer_begin ||
                model.hybrid_layer_mode(il - 1) != hybrid_mode;
            if (first_tensor_layer) {
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

                const bool optimized = raw_chunk_sizes != chunk_sizes;
                const int tail = raw_chunk_sizes.empty() ? 0 : raw_chunk_sizes.back();
                LLAMA_LOG_ERROR(
                    "[TENSOR_CHUNK_PLAN] tokens=%" PRId64 " XT=%d raw=%s final=%s "
                    "action=%s tail=%d threshold=%d chunks=%zu->%zu\n",
                    cur->ne[1], planned_chunk_tokens,
                    chunk_list(raw_chunk_sizes).c_str(),
                    chunk_list(chunk_sizes).c_str(),
                    optimized ? "OPTIMIZE_LAYOUT" : "KEEP",
                    tail, planned_chunk_tokens / 4,
                    raw_chunk_sizes.size(), chunk_sizes.size());
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

                llm_graph_moe_routing explicit_routing;
                if (explicit_router) {
                    explicit_routing = build_moe_routing(
                        norm_chunk,
                        model.layers[il].ffn_gate_inp,
                        nullptr,
                        nullptr,
                        n_expert, n_expert_used,
                        true,
                        hparams.expert_weights_scale,
                        LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                        il);
                    const std::string route_prefix =
                        phone_primary_router ?
                            "phone_prefill_route_" :
                            "c2p_prefill_route_";
                    const std::string route_topk_name =
                        route_prefix + "topk_chunk_" + std::to_string(i);
                    const std::string route_weights_name =
                        route_prefix + "weights_chunk_" + std::to_string(i);
                    cb(explicit_routing.selected_experts, route_topk_name.c_str(), il);
                    cb(explicit_routing.weights, route_weights_name.c_str(), il);
                }

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
                        model.layers[il].ffn_down_exps_s,
                        explicit_routing.selected_experts,
                        1,
                        nullptr,
                        explicit_routing.weights);
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
            llm_graph_moe_routing explicit_routing;
            if (explicit_router) {
                explicit_routing = build_moe_routing(
                    cur,
                    model.layers[il].ffn_gate_inp,
                    nullptr,
                    nullptr,
                    n_expert, n_expert_used,
                    true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il);
                cb(
                    explicit_routing.selected_experts,
                    phone_primary_router ?
                        "phone_moe_route_topk" :
                        "c2p_moe_route_topk",
                    il);
                cb(
                    explicit_routing.weights,
                    phone_primary_router ?
                        "phone_moe_route_weights" :
                        "c2p_moe_route_weights",
                    il);
            }

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
                        explicit_routing.selected_experts,
                        use_decode_chunked_moe ? n_decode_chunks : 1,
                        use_decode_chunked_moe ? &decode_down_chunks : nullptr,
                        explicit_routing.weights);

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

        if (defer_last_output_rows) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }

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
