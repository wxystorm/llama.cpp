#include "cli-cycle.h"
#include "cli-context.h"

#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// The modern llama-cli is ordinarily a llama-server HTTP client.
// The fixed physical-Cycle KV engine does not support the server's multiple
// slots, KV shifting, prefix snapshots, and request batching yet. When
// explicitly opted in, dispatch directly through the SAME public
// llama_context / llama_decode / llama_get_logits API already verified
// by llama-cycle-standard. No standalone Cycle runner is linked here.
bool cli_cycle_requested() {
    const char * value = std::getenv("LLAMA_HYBRID_CYCLE_STANDARD_DECODE");
    return value && std::strcmp(value, "1") == 0;
}

static bool cli_cycle_eval(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        size_t start, size_t count, bool want_last_logits) {
    if (!ctx || count < 1 || count > 64 ||
        start + count > tokens.size()) {
        return false;
    }
    llama_batch batch = llama_batch_init(int32_t(count), 0, 1);
    batch.n_tokens = int32_t(count);
    for (size_t i = 0; i < count; ++i) {
        batch.token[i] = tokens[start + i];
        batch.pos[i] = llama_pos(start + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        // Only the final position of the final prefill chunk produces
        // logits. No all-token logits or multi-sequence routing.
        batch.logits[i] = want_last_logits && i + 1 == count;
    }
    const int code = llama_decode(ctx, batch);
    llama_batch_free(batch);
    std::fprintf(stderr,
        "[CYCLE_OFFICIAL_CLI] stage=PUBLIC_DECODE "
        "position=%zu tokens=%zu final_logits=%d code=%d status=%s\n",
        start, count, int(want_last_logits), code,
        code == 0 ? "PASS" : "FAIL");
    return code == 0;
}

int cli_cycle_run(common_params & params) {
    if (params.split_mode != LLAMA_SPLIT_MODE_TENSOR ||
        params.hybrid_auto) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL "
            "reason=REQUIRES_SPLIT_MODE_TENSOR_AND_FIXED_CYCLE\n");
        return 2;
    }
    if (params.n_parallel != 1 || params.n_sequences != 1 ||
        params.embedding || !params.server_base.empty() ||
        !params.mmproj.path.empty() || !params.image.empty() ||
        !params.lora_adapters.empty() || !params.control_vectors.empty() ||
        params.sampling.backend_sampling || params.interactive ||
        params.interactive_first) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL "
            "reason=UNSUPPORTED_SERVER_MULTISEQ_MULTIMODAL_ADAPTER_OR_INTERACTIVE "
            "supported=RAW_SINGLE_PROMPT\n");
        return 2;
    }
    if (cli_context::interrupted().load()) {
        return 130;
    }
    if (params.model.path.empty()) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL "
            "reason=REQUIRES_LOCAL_GGUF_MODEL_PATH\n");
        return 2;
    }
    if (params.n_ctx < 0 || params.n_ctx > 512 ||
        params.n_batch < 1 || params.n_ubatch < 1) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL "
            "reason=PHYSICAL_CONTEXT_512_AND_POSITIVE_BATCH_REQUIRED\n");
        return 2;
    }
    if (params.n_predict < -2) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL "
            "reason=UNSUPPORTED_PREDICT_VALUE\n");
        return 2;
    }

    // Ordinary llama-cli defaults to an interactive, server-backed chat.
    // Accept a raw -p prompt or one line of stdin for the initial direct
    // Cycle CLI. Multiple chat turns must wait for real server-slot/KV
    // integration; do not pretend that a new prompt shares a safe context.
    std::string prompt = params.prompt;
    if (prompt.empty()) {
        std::fprintf(stderr, "Cycle prompt> ");
        std::fflush(stderr);
        if (!std::getline(std::cin, prompt) || prompt.empty()) {
            std::fprintf(stderr,
                "[CYCLE_OFFICIAL_CLI] status=FAIL reason=EMPTY_PROMPT\n");
            return 2;
        }
    }

    // Unlike normal server mode, there is no standard KV manager to warm
    // up, shift or restore, and the weights already have a fixed physical
    // ownership. All three features must remain OFF, and the model's output
    // and embedding weights must remain CPU-resolvable as in the native CLI.
    params.fit_params = false;
    params.warmup = false;
    params.ctx_shift = false;
    params.no_extra_bufts = true;
    params.sampling.backend_sampling = false;
    params.n_ctx = params.n_ctx == 0 ? 128 : params.n_ctx;
    params.n_batch = std::min(params.n_batch, 64);
    params.n_ubatch = std::min(params.n_ubatch, params.n_batch);

    llama_backend_init();
    std::fprintf(stderr,
        "[CYCLE_OFFICIAL_CLI] phase=INIT "
        "source=OFFICIAL_LLAMA_CLI mode=DIRECT_PUBLIC_LLAMA_API "
        "server=DISABLED ctx=%d batch=%d ubatch=%d "
        "sampling=COMMON_EXTERNAL status=BEGIN\n",
        params.n_ctx, params.n_batch, params.n_ubatch);
    auto init = common_init_from_params(params);
    if (!init || !init->model() || !init->context()) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL reason=MODEL_OR_CONTEXT_INIT\n");
        return 1;
    }
    llama_context * ctx = init->context();
    const llama_vocab * vocab = llama_model_get_vocab(init->model());
    common_sampler * sampler = init->sampler(0);
    if (!vocab || !sampler) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL reason=VOCAB_OR_SAMPLER\n");
        return 1;
    }

    const auto prompt_tokens = common_tokenize(
        vocab, prompt, /* add_special */ true, params.parse_special);
    if (prompt_tokens.empty() ||
        prompt_tokens.size() > size_t(llama_n_ctx(ctx))) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL reason=PROMPT_EXCEEDS_PHYSICAL_KV "
            "prompt_tokens=%zu capacity=%u\n",
            prompt_tokens.size(), llama_n_ctx(ctx));
        return 2;
    }
    const size_t available =
        size_t(llama_n_ctx(ctx)) - prompt_tokens.size() + 1;
    const size_t predict =
        params.n_predict >= 0 ? size_t(params.n_predict) : available;
    if (predict > available) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL "
            "reason=PREDICTION_EXCEEDS_PHYSICAL_KV "
            "prompt_tokens=%zu predict=%zu capacity=%u\n",
            prompt_tokens.size(), predict, llama_n_ctx(ctx));
        return 2;
    }

    // Preserve the selected official sampler's repetition-history state.
    for (const llama_token tok : prompt_tokens) {
        common_sampler_accept(sampler, tok, /* is_generated */ false);
    }
    for (size_t off = 0; off < prompt_tokens.size();) {
        if (cli_context::interrupted().load()) {
            std::fprintf(stderr,
                "[CYCLE_OFFICIAL_CLI] phase=PREFILL status=INTERRUPTED\n");
            return 130;
        }
        const size_t len = std::min<size_t>(
            64, prompt_tokens.size() - off);
        if (!cli_cycle_eval(ctx, prompt_tokens, off, len,
                            off + len == prompt_tokens.size())) {
            return 1;
        }
        off += len;
    }
    std::fprintf(stderr,
        "[CYCLE_OFFICIAL_CLI] phase=PREFILL "
        "prompt_tokens=%zu kv_prefix=%zu "
        "source=REAL_MODEL status=PASS\n",
        prompt_tokens.size(), prompt_tokens.size());
    if (predict == 0) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] phase=COMPLETE generated=0 status=PASS\n");
        return 0;
    }

    if (params.display_prompt) {
        std::fwrite(prompt.data(), 1, prompt.size(), stdout);
        std::fflush(stdout);
    }
    size_t generated = 0;
    for (size_t i = 0; i < predict; ++i) {
        if (cli_context::interrupted().load()) {
            std::fprintf(stderr,
                "[CYCLE_OFFICIAL_CLI] phase=GENERATION "
                "generated=%zu status=INTERRUPTED\n", generated);
            return 130;
        }
        // This is the standard external sampler (temp/seed/top-k/top-p/
        // penalties), not the hard-coded argmax used by Cycle smoke tests.
        const llama_token token = common_sampler_sample(sampler, ctx, -1);
        common_sampler_accept(sampler, token, /* is_generated */ true);
        if (llama_vocab_is_eog(vocab, token)) {
            std::fprintf(stderr,
                "\n[CYCLE_OFFICIAL_CLI] phase=EOG index=%zu status=PASS\n",
                i);
            break;
        }
        const std::string piece = common_token_to_piece(
            vocab, token, params.special);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
        ++generated;

        // Last generated token does not have to be committed to the KV
        // unless another token is needed. This is the llama.cpp sampling
        // contract, matching a regular prefill+decode generation loop.
        if (i + 1 == predict) break;
        llama_batch step = llama_batch_init(1, 0, 1);
        step.n_tokens = 1;
        step.token[0] = token;
        step.pos[0] = llama_pos(prompt_tokens.size() + i);
        step.n_seq_id[0] = 1;
        step.seq_id[0][0] = 0;
        step.logits[0] = 1;
        const int rc = llama_decode(ctx, step);
        llama_batch_free(step);
        if (rc != 0) {
            std::fprintf(stderr,
                "[CYCLE_OFFICIAL_CLI] stage=INCREMENTAL_DECODE "
                "position=%zu code=%d status=FAIL\n",
                prompt_tokens.size() + i, rc);
            return 1;
        }
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] stage=INCREMENTAL_DECODE "
            "position=%zu tokens=1 status=PASS\n",
            prompt_tokens.size() + i);
    }

    std::fputc('\n', stdout);
    std::fprintf(stderr,
        "[CYCLE_OFFICIAL_CLI] phase=COMPLETE "
        "prompt_tokens=%zu generated=%zu "
        "sampling=COMMON_EXTERNAL status=PASS\n",
        prompt_tokens.size(), generated);
    return 0;
}
