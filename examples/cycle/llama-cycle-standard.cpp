// Standard llama.cpp C API -> opt-in REAL 48-layer Cycle backend.
// This executable does NOT call llama_hybrid_cycle_session directly.
// It exercises llama_init_from_model(), llama_decode(),
// llama_get_logits_ith(), and llama_free() with one sequence.
#include "llama.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

static bool cycle_set_standard_switch() {
#ifdef _WIN32
    return _putenv_s("LLAMA_HYBRID_CYCLE_STANDARD_DECODE", "1") == 0;
#else
    return setenv("LLAMA_HYBRID_CYCLE_STANDARD_DECODE", "1", 1) == 0;
#endif
}

static bool print_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> piece(256);
    int32_t n = llama_token_to_piece(
        vocab, token, piece.data(), int32_t(piece.size()), 0, true);
    if (n < 0) {
        piece.resize(size_t(-n));
        n = llama_token_to_piece(
            vocab, token, piece.data(), int32_t(piece.size()), 0, true);
    }
    if (n < 0) return false;
    std::fwrite(piece.data(), 1, size_t(n), stdout);
    std::fflush(stdout);
    return true;
}

static bool decode_batch(llama_context * ctx,
                         const llama_token * tokens,
                         int count, int64_t base_position) {
    llama_batch b = llama_batch_init(count, 0, 1);
    b.n_tokens = count;
    for (int i = 0; i < count; ++i) {
        b.token[i] = tokens[i];
        b.pos[i] = llama_pos(base_position + i);
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = i == count - 1;
    }
    const int res = llama_decode(ctx, b);
    llama_batch_free(b);
    std::printf(
        "[CYCLE_STANDARD_APP] stage=LLAMA_DECODE "
        "start=%lld tokens=%d public_api=1 status=%s code=%d\n",
        (long long) base_position, count,
        res == 0 ? "PASS" : "FAIL", res);
    return res == 0;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr,
            "Usage: llama-cycle-standard <Qwen3-MoE.gguf> "
            "<Phone-RPC-address:port> <prompt> [new_tokens=4]\n");
        return 2;
    }
    int n_generate = 4;
    if (argc == 5) {
        char * end = nullptr;
        const long parsed = std::strtol(argv[4], &end, 10);
        if (!end || *end != '\0' ||
            parsed < 1 || parsed > 64) {
            std::fprintf(stderr,
                "[CYCLE_STANDARD_APP] status=FAIL reason=GENERATION_COUNT\n");
            return 2;
        }
        n_generate = int(parsed);
    }
    if (!cycle_set_standard_switch()) {
        std::fprintf(stderr,
            "[CYCLE_STANDARD_APP] status=FAIL reason=ENABLE_SWITCH\n");
        return 2;
    }

    llama_backend_init();
    const auto cpu_dev = ggml_backend_dev_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU);
    const auto rpc = ggml_backend_rpc_add_server(argv[2]);
    if (!cpu_dev || !rpc || ggml_backend_reg_dev_count(rpc) != 1) {
        std::fprintf(stderr,
            "[CYCLE_STANDARD_APP] status=FAIL reason=DEVICE_REGISTRATION\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(rpc, 0);
    if (!phone_dev ||
        ggml_backend_dev_type(phone_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr,
            "[CYCLE_STANDARD_APP] status=FAIL reason=PHONE_NOT_GPU\n");
        return 1;
    }
    ggml_backend_dev_t devices[] = {cpu_dev,phone_dev,nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.devices = devices;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.hybrid_auto = false;
    mp.use_extra_bufts = false;
    int rc = 1;
    {
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1],mp),&llama_model_free);
        if (!model) {
            std::fprintf(stderr,
                "[CYCLE_STANDARD_APP] status=FAIL reason=MODEL_LOAD\n");
            return 1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        const std::string prompt(argv[3]);
        std::vector<llama_token> ids(128);
        const int n = llama_tokenize(
            vocab, prompt.c_str(), int32_t(prompt.size()),
            ids.data(), int32_t(ids.size()), true, true);
        if (n < 1 || n > 64 || n + n_generate - 1 > 128) {
            std::fprintf(stderr,
                "[CYCLE_STANDARD_APP] status=FAIL reason=PROMPT_OR_KV_CAP "
                "prompt_tokens=%d generation=%d\n",n,n_generate);
            return 1;
        }
        ids.resize(size_t(n));

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_seq_max = 1;
        cp.n_batch = 64;
        cp.n_ubatch = 64;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(),cp),&llama_free);
        if (!ctx) {
            std::fprintf(stderr,
                "[CYCLE_STANDARD_APP] status=FAIL reason=CONTEXT_INIT\n");
            return 1;
        }
        std::printf(
            "[CYCLE_STANDARD_APP] phase=INIT context_api=PUBLIC_LLAMA "
            "kv_capacity=%u status=PASS\n",llama_n_ctx(ctx.get()));
        bool ok = llama_cycle_context_set_prompt_phase(ctx.get(),true);
        if (!ok) {
            std::fprintf(stderr,
                "[CYCLE_STANDARD_APP] phase=PERF_HINT "
                "reason=SET_PROMPT_PHASE status=FAIL\n");
        }
        if (ok) {
            ok = decode_batch(
                ctx.get(),ids.data(),int(ids.size()),0);
        }
        if (ok) {
            ok = llama_cycle_context_set_prompt_phase(ctx.get(),false);
        }
        if (ok) {
            std::printf("[CYCLE_STANDARD_APP_TEXT] ");
            int emitted = 0;
            for (int i = 0; i < n_generate && ok; ++i) {
                float * logits = llama_get_logits_ith(ctx.get(),-1);
                if (!logits) {
                    ok = false;
                    break;
                }
                const int vocab_count = llama_vocab_n_tokens(vocab);
                int top = 0;
                for (int v = 1; v < vocab_count; ++v) {
                    if (logits[v] > logits[top]) top = v;
                }
                const llama_token token = llama_token(top);
                if (llama_vocab_is_eog(vocab,token)) break;
                ok = print_piece(vocab,token);
                if (!ok) break;
                ++emitted;
                if (i + 1 == n_generate) break;
                ok = decode_batch(
                    ctx.get(),&token,1,n + i);
            }
            const auto perf = llama_perf_context(ctx.get());
            const bool perf_ok =
                perf.t_p_eval_ms > 0.0 && perf.n_p_eval == n &&
                (emitted <= 1 || (perf.t_eval_ms > 0.0 &&
                                  perf.n_eval >= emitted - 1));
            std::printf(
                "\n[CYCLE_STANDARD_PERF] prompt_ms=%.3f "
                "eval_ms=%.3f prompt_tokens=%d eval_runs=%d "
                "status=%s\n",
                perf.t_p_eval_ms, perf.t_eval_ms,
                perf.n_p_eval, perf.n_eval,
                perf_ok ? "PASS" : "FAIL");
            ok = ok && perf_ok;
            std::printf(
                "\n[CYCLE_STANDARD_APP] phase=COMPLETE "
                "requested=%d generated=%d "
                "execution=PUBLIC_LLAMA_CONTEXT "
                "status=%s\n",
                n_generate,emitted,ok ? "PASS" : "FAIL");
            if (ok) rc = 0;
        }
    }
    return rc;
}
