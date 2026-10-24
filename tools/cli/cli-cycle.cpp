#include "cli-cycle.h"
#include "cli-context.h"
#include "cli-ui.h"

#include "chat.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
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

// The CLI uses stderr for operational status and stdout for generated text.
static bool cli_cycle_verbose() {
    const char * value = std::getenv("LLAMA_HYBRID_CYCLE_VERBOSE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// Opt-in exact Token ID trace for comparing Cycle against a CPU reference.
// This does not alter sampling, scheduler activity or model output.
static bool cli_cycle_token_trace() {
    const char * value = std::getenv("LLAMA_HYBRID_CYCLE_TOKEN_TRACE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

static bool cli_cycle_eval(
        llama_context * ctx,
        const std::vector<llama_token> & tokens,
        size_t start, size_t count, bool want_last_logits) {
    if (!ctx || count < 1 || count > 512 ||
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
    if (code != 0 || cli_cycle_verbose()) std::fprintf(stderr,
        "[CYCLE_OFFICIAL_CLI] stage=PUBLIC_DECODE "
        "position=%zu tokens=%zu final_logits=%d code=%d status=%s\n",
        start, count, int(want_last_logits), code,
        code == 0 ? "PASS" : "FAIL");
    return code == 0;
}

// Reuse the official llama.cpp performance structures, formatter, and
// terminal info style. Unlike ordinary llama-cli's server timing JSON,
// Cycle obtains the measurements from its local llama_context directly.
static void cli_cycle_show_timings(
        const common_params & params, llama_context * ctx,
        const common_sampler * sampler) {
    if (!params.show_timings || params.no_perf || !ctx) {
        return;
    }
    const auto perf = llama_perf_context(ctx);
    const double prompt_tps = perf.t_p_eval_ms > 0.0 ?
        1e3 * perf.n_p_eval / perf.t_p_eval_ms : 0.0;
    const bool evaluated_decode = perf.t_eval_ms > 0.0;
    const double decode_tps = evaluated_decode ?
        1e3 * perf.n_eval / perf.t_eval_ms : 0.0;

    // common_perf_print uses LOG_INF. The current CLI filters INFO by
    // default, so allow INFO for this final report only, then restore it.
    // Do not call the upstream formatter without any decode measurement:
    // n_eval is clamped to one and a zero t_eval_ms would print infinity.
    if (perf.t_p_eval_ms > 0.0 && evaluated_decode) {
        const int old_verbosity = common_log_get_verbosity_thold();
        common_log_set_verbosity_thold(
            std::max(old_verbosity, LOG_LEVEL_TRACE));
        common_perf_print(ctx, sampler);
        common_log_flush(common_log_main());
        common_log_set_verbosity_thold(old_verbosity);
        std::fprintf(stderr,
            "[CYCLE_PERF] note=GGML_GRAPH_REUSE_COUNT_EXCLUDES_CYCLE_CACHE "
            "prompt_tokens=%d eval_runs=%d "
            "prompt_ms=%.3f decode_ms=%.3f status=MEASURED\n",
            perf.n_p_eval, perf.n_eval, perf.t_p_eval_ms, perf.t_eval_ms);
    } else {
        std::fprintf(stderr,
            "[CYCLE_PERF] prompt_tokens=%d prompt_ms=%.3f "
            "decode_runs=0 decode_ms=0 "
            "detail=NO_INCREMENTAL_EVALUATION status=MEASURED\n",
            perf.n_p_eval, perf.t_p_eval_ms);
    }

    // Do not change terminal input modes while -p / stdin are being read.
    // Use official UI styling at the end of the single-prompt run.
    ui::init(params);
    if (evaluated_decode) {
        ui::show_info(string_format(
            "\n[ Prompt: %.1f t/s | Generation: %.1f t/s ]",
            prompt_tps, decode_tps));
    } else {
        ui::show_info(string_format(
            "\n[ Prompt: %.1f t/s | Generation: n/a (no decode eval) ]",
            prompt_tps));
    }
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
            "supported=SINGLE_TURN_CHAT_OR_RAW_PROMPT\n");
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
    params.n_batch = std::min(params.n_batch, params.n_ctx);
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

    // AUTO follows the model's GGUF chat template, just like the official
    // completion CLI. Explicit --no-conversation remains byte-for-byte raw
    // input for numerical/parity regression. Neither mode enables a server,
    // multi-turn history, KV shifting or a new executor.
    const bool template_available =
        !params.chat_template.empty() ||
        llama_model_chat_template(init->model(), nullptr) != nullptr;
    const bool template_enabled = params.enable_chat_template &&
        params.conversation_mode != COMMON_CONVERSATION_MODE_DISABLED;
    const bool use_chat_template = template_enabled &&
        (template_available ||
         params.conversation_mode == COMMON_CONVERSATION_MODE_ENABLED);
    if (params.conversation_mode == COMMON_CONVERSATION_MODE_ENABLED &&
        !params.enable_chat_template) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL reason=CHAT_TEMPLATE_DISABLED_BY_INPUT_PREFIX\n");
        return 2;
    }
    if (use_chat_template && !template_available) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL reason=CHAT_TEMPLATE_NOT_AVAILABLE\n");
        return 2;
    }

    std::string model_prompt = prompt;
    if (use_chat_template) {
        try {
            auto templates = common_chat_templates_init(
                init->model(), params.chat_template);
            common_chat_templates_inputs inputs;
            inputs.use_jinja = params.use_jinja;
            inputs.add_generation_prompt = true;
            inputs.enable_thinking = params.enable_reasoning != 0;
            inputs.chat_template_kwargs = params.default_template_kwargs;
            if (!params.system_prompt.empty()) {
                common_chat_msg system_message;
                system_message.role = "system";
                system_message.content = params.system_prompt;
                inputs.messages.push_back(std::move(system_message));
            }
            common_chat_msg user_message;
            user_message.role = "user";
            user_message.content = prompt;
            inputs.messages.push_back(std::move(user_message));
            model_prompt = common_chat_templates_apply(
                templates.get(), inputs).prompt;
            if (model_prompt.empty()) {
                std::fprintf(stderr,
                    "[CYCLE_OFFICIAL_CLI] status=FAIL reason=CHAT_TEMPLATE_EMPTY_PROMPT\n");
                return 2;
            }
        } catch (const std::exception & ex) {
            std::fprintf(stderr,
                "[CYCLE_OFFICIAL_CLI] status=FAIL reason=CHAT_TEMPLATE_APPLY detail=%s\n",
                ex.what());
            return 2;
        }
    }
    // A chat template may contain model-specific structural token spellings.
    // Tokenize those as special IDs; treating them as ordinary text changes
    // the logits at the assistant boundary.
    const auto prompt_tokens = common_tokenize(
        vocab, model_prompt, /* add_special */ true,
        use_chat_template ? true : params.parse_special);
    std::fprintf(stderr,
        "[CYCLE_OFFICIAL_CLI] phase=PROMPT_FORMAT mode=%s user_bytes=%zu "
        "model_bytes=%zu tokens=%zu special=%d status=PASS\n",
        use_chat_template ? "MODEL_CHAT_TEMPLATE" : "RAW_COMPLETION",
        prompt.size(), model_prompt.size(), prompt_tokens.size(),
        int(use_chat_template || params.parse_special));
    if (cli_cycle_verbose() && use_chat_template) {
        std::fprintf(stderr, "[CYCLE_CHAT_TEMPLATE] formatted_prompt_begin\n%s\n"
                             "[CYCLE_CHAT_TEMPLATE] formatted_prompt_end\n",
                     model_prompt.c_str());
    }
    if (cli_cycle_token_trace()) {
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            std::fprintf(stderr,
                "[CYCLE_TOKEN_TRACE] phase=PROMPT index=%zu id=%d\n",
                i, int(prompt_tokens[i]));
        }
    }
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

    if (!llama_cycle_context_set_prompt_phase(ctx, true)) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL "
            "reason=CYCLE_PROMPT_PERF_PHASE_UNAVAILABLE\n");
        return 1;
    }

    // Preserve the selected official sampler's repetition-history state.
    for (const llama_token tok : prompt_tokens) {
        common_sampler_accept(sampler, tok, /* is_generated */ false);
    }
    // Production Prefill is ONE global Layer x Chunk ring. Do not split the
    // Prompt into outer llama_decode() batches: the Cycle executor performs
    // its own 64-token Chunking and keeps the five device queues alive until
    // the complete Prompt drains.
    if (cli_context::interrupted().load()) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] phase=PREFILL status=INTERRUPTED\n");
        return 130;
    }
    if (!cli_cycle_eval(ctx, prompt_tokens, 0, prompt_tokens.size(), true)) {
        return 1;
    }
    std::fprintf(stderr,
        "[CYCLE_OFFICIAL_CLI] phase=PREFILL "
        "prompt_tokens=%zu kv_prefix=%zu "
        "source=REAL_MODEL status=PASS\n",
        prompt_tokens.size(), prompt_tokens.size());
    if (!llama_cycle_context_set_prompt_phase(ctx, false)) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] status=FAIL "
            "reason=CYCLE_DECODE_PERF_PHASE_UNAVAILABLE\n");
        return 1;
    }
    if (predict == 0) {
        std::fprintf(stderr,
            "[CYCLE_OFFICIAL_CLI] phase=COMPLETE generated=0 status=PASS\n");
        cli_cycle_show_timings(params, ctx, sampler);
        return 0;
    }

    if (params.display_prompt) {
        // Display only user-visible input. Never echo the rendered template's
        // assistant prefix or special control tokens into the output stream.
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
        if (cli_cycle_token_trace()) {
            std::fprintf(stderr,
                "[CYCLE_TOKEN_TRACE] phase=GENERATE index=%zu id=%d eog=%d\n",
                i, int(token), int(llama_vocab_is_eog(vocab, token)));
        }
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
        if (cli_cycle_verbose()) std::fprintf(stderr,
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
    cli_cycle_show_timings(params, ctx, sampler);
    return 0;
}
