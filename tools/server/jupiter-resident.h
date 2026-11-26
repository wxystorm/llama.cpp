#pragma once

// Opt-in resident Jupiter experiment server. Deliberately separate from the
// ordinary completion/continuous batching scheduler: no prefix cache, no
// generation or speculative decoding, one full fresh prompt per request.
// Only an authenticated loopback client can call this experimental endpoint.
#include "server-http.h"
#include "common.h"
#include "chat.h"
#include "log.h"
#include "llama.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using jr_json = nlohmann::ordered_json;

inline bool jr_env_enabled(const char * key) {
    const char * s = std::getenv(key);
    return s && std::string(s) == "1";
}

inline void jr_set_env(const char * key, const std::string & value) {
#ifdef _WIN32
    if (_putenv_s(key, value.c_str()) != 0) {
#else
    if ((value.empty() ? unsetenv(key) : setenv(key, value.c_str(), 1)) != 0) {
#endif
        throw std::runtime_error(std::string("cannot set ") + key);
    }
}

inline server_http_res_ptr jr_response(jr_json data, int status = 200) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data = data.dump();
    return res;
}

inline bool jr_id_valid(const std::string & id) {
    return id.size() == 32 && std::all_of(id.begin(), id.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

inline void jr_marker(const char * phase, const std::string & id, const char * status) {
    // Drain the same real runtime logs used by the existing q parser.
    // Markers delimit this request without manufacturing stage samples.
    common_log_flush(common_log_main());
    std::fprintf(stderr, "[JUPITER_RESIDENT] phase=%s id=%s status=%s\n",
                 phase, id.c_str(), status);
    std::fflush(stderr);
}

inline int jupiter_resident_server(common_params & params) {
    if (params.hostname != "127.0.0.1" || params.api_keys.empty() ||
            params.model.path.empty() || !jr_env_enabled("LLAMA_JUPITER_PREFILL")) {
        std::fprintf(stderr, "Resident Jupiter requires a local GGUF, "
                     "--host 127.0.0.1, --api-key, and LLAMA_JUPITER_PREFILL=1\n");
        return 1;
    }
    if (params.embedding || !params.ssl_file_key.empty() || !params.ssl_file_cert.empty()) {
        std::fprintf(stderr, "Resident Jupiter only supports local text Prefill\n");
        return 1;
    }
    const bool native = jr_env_enabled("LLAMA_JUPITER_PHONE_OWNED_NATIVE");
    const bool native_calibration = jr_env_enabled("LLAMA_JUPITER_RESIDENT_NATIVE_CALIBRATION");
    const bool raw_prompt = jr_env_enabled("LLAMA_JUPITER_RESIDENT_RAW_PROMPT");
    const std::string prompt_mode = raw_prompt ? "RAW" : "CHAT";
    if (native_calibration && !native) {
        std::fprintf(stderr, "Native calibration requires Phone-owned model loading\n");
        return 1;
    }
    // Explicit experimental timing-only opt-in. Never applies to native
    // formal Prefill or requests that require real F32 logits.
    const bool legacy_timing_only = !native &&
        jr_env_enabled("LLAMA_JUPITER_RESIDENT_LEGACY_TIMING_ONLY");
    std::fprintf(stderr, "[JUPITER_RESIDENT_POLICY] native=%d legacy_timing_only=%d\n",
                 int(native), int(legacy_timing_only));
    params.n_parallel = 1;
    // Match server_n_outputs_max() for one text sequence, no draft.
    // Do not reserve a vocab row for every token in a calibration UBatch.
    params.n_outputs_max = 1;
    params.kv_unified = false;
    // Stage2/3's legacy CLI initializes the model with its usual startup
    // warmup. Native must never consume a Phone-owned run with fake tokens.
    // This is startup-only and never included in measured Prefill.
    params.warmup = !native;
    params.no_perf = false;
    params.sampling.backend_sampling = false;
    // Never permit a request to select filesystem dump paths or a model.
    jr_set_env("LLAMA_JUPITER_PLAN_FILE", "");
    jr_set_env("LLAMA_JUPITER_SEQUENCE_PLAN", "");
    jr_set_env("LLAMA_JUPITER_LOGITS_DUMP_FILE", "");
    jr_set_env("LLAMA_JUPITER_STAGE_DUMP_DIR", "");
    jr_set_env("LLAMA_JUPITER_STAGE_DUMP_TOKENS", "");
    jr_set_env("LLAMA_JUPITER_GRAPH_PREBUILD", "0");
    jr_set_env("LLAMA_JUPITER_LOCAL_GRAPH_REUSE", "0");
    jr_set_env("LLAMA_JUPITER_ASYNC_C2P", "0");
    jr_set_env("LLAMA_JUPITER_PHONE_OWNED_ASYNC", "0");

    const int resident_window = native ? 2 : 1;
    jr_set_env("LLAMA_JUPITER_WINDOW", std::to_string(resident_window));
    auto loaded = common_init_from_params(params);
    if (!loaded || !loaded->model() || !loaded->context()) {
        return 1;
    }
    auto * model = loaded->model();
    auto * ctx = loaded->context();
    const auto * vocab = llama_model_get_vocab(model);
    const auto templates = common_chat_templates_init(model, "");
    if (!templates) {
        std::fprintf(stderr, "[JUPITER_RESIDENT] chat template initialization failed\n");
        return 1;
    }
    // Use the same capability probe as the regular CLI/server initialization.
    // The existing native-specific implementation is memory-only.
    const auto seq_rm = common_context_can_seq_rm(ctx);
    if (seq_rm == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
        std::fprintf(stderr, "[JUPITER_RESIDENT] KV capability probe failed native=%d\n", int(native));
        return 1;
    }
    std::fprintf(stderr, "[JUPITER_RESIDENT_INIT] native=%d startup_warmup=%d n_outputs_max=%d seq_rm=%d\n",
                 int(native), int(params.warmup), params.n_outputs_max, int(seq_rm));
    if (!llama_jupiter_reset_prefill(ctx)) {
        std::fprintf(stderr, "[JUPITER_RESIDENT] startup reset failed\n");
        return 1;
    }

    server_http_context http;
    if (!http.init(params)) {
        return 1;
    }
    std::mutex request_mutex;
    bool poisoned = false;
    bool stopping = false;
    http.get("/health", [&](const server_http_req &) {
        return jr_response({{"status", "ok"}, {"resident", true},
                            {"native", native}, {"legacy_timing_only", legacy_timing_only},
                            {"native_calibration", native_calibration},
                            {"prompt_mode", prompt_mode},
                            {"ctx", llama_n_ctx(ctx)},
                            {"batch", llama_n_batch(ctx)}, {"ubatch", llama_n_ubatch(ctx)}});
    });
    http.post("/jupiter/prefill", [&](const server_http_req & req) -> server_http_res_ptr {
        std::lock_guard<std::mutex> lock(request_mutex);
        if (poisoned || stopping) {
            return jr_response({{"error", "resident context is unavailable; restart required"}}, 503);
        }
        std::string id;
        bool began = false;
        bool decoding = false;
        try {
            const auto body = jr_json::parse(req.body);
            id = body.at("id").get<std::string>();
            if (!jr_id_valid(id)) {
                throw std::invalid_argument("invalid request id");
            }
            auto prompt = body.at("prompt").get<std::string>();
            if (params.escape) {
                string_process_escapes(prompt);
            }
            if (prompt.empty()) {
                throw std::invalid_argument("empty Prompt");
            }
            if (body.value("prompt_mode", std::string("CHAT")) != prompt_mode) {
                throw std::invalid_argument("resident prompt mode mismatch");
            }
            // RAW matches llama-tokenize without --chat-user and the
            // historical 24B completion benchmark. In particular, never
            // insert Mistral's implicit system or assistant prefix.
            std::string formatted = prompt;
            if (!raw_prompt) {
                common_chat_templates_inputs inputs {};
                inputs.use_jinja = true;
                inputs.add_generation_prompt = true;
                inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
                inputs.enable_thinking = common_chat_templates_support_enable_thinking(templates.get());
                common_chat_msg user {};
                user.role = "user";
                user.content = prompt;
                inputs.messages.push_back(std::move(user));
                formatted = common_chat_templates_apply(templates.get(), inputs).prompt;
            }
            auto tokens = common_tokenize(vocab, formatted, llama_vocab_get_add_bos(vocab), true);
            const size_t source_tokens = tokens.size();
            auto cuts = body.value("chunks", std::vector<uint32_t>{});
            const auto expected = body.value("expected_tokens", uint32_t(0));
            const auto flags = body.value("flags", jr_json::object());
            if (body.value("calibration", false) != native_calibration) {
                throw std::invalid_argument("resident calibration namespace mismatch");
            }
            const auto raw_prefix_tokens = body.value("raw_prefix_tokens", uint32_t(0));
            // Only synthetic offline calibration may take an exact prefix
            // of a longer raw token stream. Formal prompts are never cropped.
            if (raw_prefix_tokens != 0) {
                if (!raw_prompt || !native_calibration ||
                        !flags.value("PROFILE_STAGE", false) || cuts.empty() ||
                        raw_prefix_tokens != expected || raw_prefix_tokens < 2 ||
                        raw_prefix_tokens > llama_n_ctx(ctx) ||
                        raw_prefix_tokens > llama_n_batch(ctx) ||
                        raw_prefix_tokens > tokens.size()) {
                    throw std::invalid_argument("invalid synthetic raw calibration prefix");
                }
                tokens.resize(raw_prefix_tokens);
            }
            if (native_calibration && cuts.empty() && !flags.value("PROFILE_STAGE", false)) {
                // Tokenizer probe only: still a real native, exact ordered
                // Plan derived from the server's own templated token count.
                const uint32_t width = llama_n_ubatch(ctx);
                if (!width || tokens.size() > uint64_t(width)*16) {
                    throw std::invalid_argument("native calibration probe exceeds 16 Chunk slots");
                }
                for (uint32_t remaining = uint32_t(tokens.size()); remaining;) {
                    const uint32_t next = std::min(remaining, width);
                    cuts.push_back(next);
                    remaining -= next;
                }
            }
            if (tokens.size() < 2 || tokens.size() > llama_n_ctx(ctx) ||
                    tokens.size() > llama_n_batch(ctx) ||
                    (expected && expected != tokens.size()) ||
                    (native && (cuts.empty() || cuts.size() > 16))) {
                throw std::invalid_argument("invalid or mismatched full Prompt shape");
            }
            uint64_t sum = 0;
            std::string sequence;
            for (auto width : cuts) {
                if (!width || width > llama_n_ubatch(ctx)) {
                    throw std::invalid_argument("Chunk exceeds resident UBatch");
                }
                sum += width;
                if (!sequence.empty()) { sequence += ","; }
                sequence += std::to_string(width);
            }
            if (!cuts.empty() && sum != tokens.size()) {
                throw std::invalid_argument("Chunk sum differs from actual templated Prompt");
            }
            const bool autonomous = body.value("autonomous", false);
            if (autonomous && !native) {
                throw std::invalid_argument("Autonomous requires a native resident model");
            }
            if (native_calibration && (!autonomous || body.value("return_logits", false))) {
                throw std::invalid_argument("native calibration requires autonomous input and no logits export");
            }
            const int window = body.value("window", 1);
            if (window != resident_window) {
                throw std::invalid_argument("window differs from immutable scheduler capacity");
            }

            jr_marker("BEGIN", id, "RUN");
            began = true;
            std::fprintf(stderr, "[JUPITER_RESIDENT_INPUT] mode=%s source_tokens=%zu tokens=%zu"
                         " raw_prefix_tokens=%u\n", prompt_mode.c_str(),
                         source_tokens, tokens.size(), raw_prefix_tokens);
            if (native_calibration) {
                std::fprintf(stderr, "[JUPITER_NATIVE_CALIBRATION] mode=PHONE_AUTONOMOUS profile=%d\n",
                             int(flags.value("PROFILE_STAGE", false)));
            }
            // These flags are read at decode/graph execution time. Geometry,
            // placement, session count and model remain immutable at startup.
            jr_set_env("LLAMA_JUPITER_SEQUENCE_PLAN", sequence);
            jr_set_env("LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS", autonomous ? "1" : "0");
            for (const auto * key : {"TRACE", "PROFILE_STAGE", "Q_SWEEP", "TIMELINE",
                                     "SCHED_ALLOC_TIMING", "LOGITS_HASH"}) {
                const std::string name = std::string("LLAMA_JUPITER_") + key;
                jr_set_env(name.c_str(), flags.value(key, false) ? "1" : "0");
            }

            // Explicit fresh sequence ID 0, full prompt in one decode batch.
            // Only the final token requests logits; no generated-token decode.
            llama_batch batch = llama_batch_get_one(tokens.data(), int32_t(tokens.size()));
            std::vector<int8_t> output(tokens.size(), 0);
            output.back() = 1;
            batch.logits = output.data();
            llama_set_embeddings(ctx, false);
            llama_perf_context_reset(ctx);
            const auto t0 = ggml_time_us();
            decoding = true;
            const int rc = llama_decode(ctx, batch);
            llama_synchronize(ctx);
            if (rc != 0) {
                throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
            }
            const float * logits = llama_get_logits_ith(ctx, int32_t(tokens.size() - 1));
            const double ms = (ggml_time_us() - t0) / 1000.0;
            const auto n_vocab = llama_vocab_n_tokens(vocab);
            if (!std::isfinite(ms) || ms <= 0.0) {
                throw std::runtime_error("INVALID_PREFILL_TIMING ms=" + std::to_string(ms));
            }
            if (n_vocab <= 0) {
                throw std::runtime_error("INVALID_VOCAB_SIZE n_vocab=" + std::to_string(n_vocab));
            }
            if (!logits) {
                throw std::runtime_error("MISSING_FINAL_LOGITS tokens=" + std::to_string(tokens.size()));
            }
            size_t nan_count = 0, inf_count = 0, zero_count = 0;
            int64_t first_bad = -1;
            for (int64_t i = 0; i < n_vocab; ++i) {
                const float v = logits[i];
                if (std::isnan(v)) {
                    ++nan_count;
                } else if (std::isinf(v)) {
                    ++inf_count;
                } else if (v == 0.0f) {
                    ++zero_count;
                }
                if (!std::isfinite(v) && first_bad < 0) {
                    first_bad = i;
                }
            }
            const bool return_logits = body.value("return_logits", false);
            const bool timing_only = legacy_timing_only && !return_logits;
            const bool logits_finite = first_bad < 0;
            std::fprintf(stderr, "[JUPITER_RESIDENT_LOGITS] native=%d tokens=%zu vocab=%lld"
                         " finite=%zu nan=%zu inf=%zu zero=%zu first_bad=%lld"
                         " prefill_ms=%.3f validation=%s status=%s\n",
                         int(native), tokens.size(), (long long)n_vocab,
                         size_t(n_vocab) - nan_count - inf_count, nan_count,
                         inf_count, zero_count, (long long)first_bad, ms,
                         timing_only ? "TIMING_ONLY" : "FINITE_ONLY",
                         logits_finite ? "PASS" : (timing_only ? "UNVERIFIED" : "FAIL"));
            if (native && zero_count == size_t(n_vocab)) {
                throw std::runtime_error("ALL_ZERO_NATIVE_LOGITS");
            }
            if (!logits_finite && !timing_only) {
                throw std::runtime_error("NONFINITE_FINAL_LOGITS nan=" + std::to_string(nan_count) +
                    " inf=" + std::to_string(inf_count) + " first_bad=" + std::to_string(first_bad));
            }
            // PASS is execution/reset success only. Do not invent a top1
            // from NaNs or serialize invalid floats as legitimate logits.
            jr_json result = {{"id", id}, {"status", "PASS"}, {"tokens", tokens.size()},
                              {"prefill_ms", ms}, {"metric", "DECODE_TO_LOGITS"},
                              {"prompt_mode", prompt_mode}, {"source_tokens", source_tokens},
                              {"raw_prefix_tokens", raw_prefix_tokens},
                              {"logits_validation", timing_only ? "TIMING_ONLY" : "FINITE_ONLY"},
                              {"logits_finite", logits_finite}, {"logits_nan", nan_count},
                              {"logits_inf", inf_count}, {"logits_zero", zero_count},
                              {"top1", nullptr}};
            if (logits_finite) {
                result["top1"] = std::max_element(logits, logits + n_vocab) - logits;
            } else {
                std::fprintf(stderr, "[JUPITER_RESIDENT_TIMING_ONLY] id=%s nan=%zu inf=%zu"
                             " accuracy=UNVERIFIED\n", id.c_str(), nan_count, inf_count);
            }
            if (return_logits) {
                result["logits"] = std::vector<float>(logits, logits + n_vocab);
            }
            // Measure only fresh Prefill through synchronized logits. Return
            // reset and HTTP/JSON overhead separately, outside that metric.
            const auto reset_start = ggml_time_us();
            if (!llama_jupiter_reset_prefill(ctx)) {
                throw std::runtime_error("PC/Phone fresh-request reset failed");
            }
            decoding = false;
            result["reset_ms"] = (ggml_time_us() - reset_start) / 1000.0;
            common_log_flush(common_log_main());
            std::fprintf(stderr, "[JUPITER_RESIDENT_RESET] id=%s status=PASS\n", id.c_str());
            std::fprintf(stderr, "prompt eval time = %.3f ms / %zu tokens\n", ms, tokens.size());
            jr_marker("END", id, "PASS");
            return jr_response(std::move(result));
        } catch (const std::exception & e) {
            if (decoding) {
                // Never accept another prompt after a partially failed device
                // run. Attempt cleanup, but require a server restart.
                poisoned = true;
                llama_jupiter_reset_prefill(ctx);
            }
            if (began) {
                std::fprintf(stderr, "[JUPITER_RESIDENT] error=%s\n", e.what());
                jr_marker("END", id, "FAIL");
            }
            return jr_response({{"id", id}, {"error", e.what()}}, poisoned ? 500 : 400);
        }
    });
    http.post("/jupiter/shutdown", [&](const server_http_req &) {
        std::lock_guard<std::mutex> lock(request_mutex);
        stopping = true;
        http.stop();
        return jr_response({{"status", "stopping"}});
    });
    if (!http.start()) { return 1; }
    http.is_ready.store(true);
    common_log_flush(common_log_main());
    std::fprintf(stderr, "[JUPITER_RESIDENT] phase=READY port=%d native=%d ctx=%u batch=%u ubatch=%u\n",
                 http.port, int(native), llama_n_ctx(ctx), llama_n_batch(ctx), llama_n_ubatch(ctx));
    std::fflush(stderr);
    if (http.thread.joinable()) {
        http.thread.join();
    }
    return 0;
}
} // namespace
