// Native Cycle generator: one tokenized prompt, one persistent physical KV
// session, then autoregressive single-token Decode over the SAME five FIFO
// queues and PC/Phone RPC pair. No CPU oracle or test-only synthetic input.
#include "llama.h"
#include "llama-model.h"
#include "llama-hybrid-cycle-forward.h"
#include "llama-hybrid-cycle-model-io.h"
#include "llama-hybrid-cycle-runtime.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

static bool cycle_print_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> piece(256);
    int32_t bytes = llama_token_to_piece(
        vocab, token, piece.data(), int32_t(piece.size()), 0, true);
    if (bytes < 0) {
        piece.resize(size_t(-bytes));
        bytes = llama_token_to_piece(
            vocab, token, piece.data(), int32_t(piece.size()), 0, true);
    }
    if (bytes < 0) return false;
    std::fwrite(piece.data(), 1, size_t(bytes), stdout);
    std::fflush(stdout);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) {
        std::fprintf(stderr,
            "Usage: llama-cycle-cli <qwen3-30b-a3b.gguf> "
            "<phone-rpc-ip:port> <prompt> [generation_tokens=8]\n"
            "Requires LLAMA_HYBRID_CYCLE_PC_LAYERS=4 and "
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS=2.\n"
            "Current physical KV limit = 128 tokens; "
            "prompt <=64 tokens; single sequence, greedy sampling.\n");
        return 2;
    }
    int max_generate = 8;
    if (argc == 5) {
        char * end = nullptr;
        const long value = std::strtol(argv[4], &end, 10);
        if (!end || *end != '\0' || value < 1 || value > 128) {
            std::fprintf(stderr,
                "[CYCLE_NATIVE] status=FAIL reason=BAD_GENERATION_COUNT\n");
            return 2;
        }
        max_generate = int(value);
    }
    if (!llama_hybrid_cycle_requested()) {
        std::fprintf(stderr,
            "[CYCLE_NATIVE] status=FAIL reason=PHYSICAL_CYCLE_NOT_REQUESTED\n");
        return 2;
    }
    llama_backend_init();
    const auto cpu_dev = ggml_backend_dev_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU);
    const auto rpc_reg = ggml_backend_rpc_add_server(argv[2]);
    if (!cpu_dev || !rpc_reg || ggml_backend_reg_dev_count(rpc_reg) != 1) {
        std::fprintf(stderr,
            "[CYCLE_NATIVE] status=FAIL reason=BACKEND_REGISTRATION\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(rpc_reg, 0);
    if (!phone_dev ||
        ggml_backend_dev_type(phone_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr,
            "[CYCLE_NATIVE] status=FAIL reason=PHONE_GPU_MISSING\n");
        return 1;
    }
    const auto cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    const auto phone = ggml_backend_dev_init(phone_dev, nullptr);
    if (!cpu || !phone) {
        if (phone) ggml_backend_free(phone);
        if (cpu) ggml_backend_free(cpu);
        return 1;
    }

    int rc = 1;
    {
        ggml_backend_dev_t devices[] = {cpu_dev,phone_dev,nullptr};
        llama_model_params params = llama_model_default_params();
        params.devices = devices;
        params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
        params.hybrid_auto = false;
        params.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], params),
            &llama_model_free);
        if (!model || model->arch != LLM_ARCH_QWEN3MOE ||
            llama_model_n_layer(model.get()) != 48) {
            std::fprintf(stderr,
                "[CYCLE_NATIVE] status=FAIL reason=EXPECTED_QWEN3_MOE_48\n");
        } else {
            const llama_vocab * vocab = llama_model_get_vocab(model.get());
            std::string prompt(argv[3]);
            std::vector<llama_token> token_ids(128);
            const int32_t count = llama_tokenize(
                vocab, prompt.c_str(), int32_t(prompt.size()),
                token_ids.data(), int32_t(token_ids.size()),
                true, true);
            if (count < 1 || count > 64) {
                std::fprintf(stderr,
                    "[CYCLE_NATIVE] status=FAIL reason=PROMPT_TOKENS "
                    "n_tokens=%d allowed=1..64\n",count);
            } else if (count + max_generate - 1 > 128) {
                std::fprintf(stderr,
                    "[CYCLE_NATIVE] status=FAIL reason=PHYSICAL_KV_LIMIT "
                    "prompt=%d generation=%d capacity=128\n",
                    count,max_generate);
            } else {
                token_ids.resize(size_t(count));
                std::string reason;
                llama_hybrid_cycle_session session;
                bool ok = session.prepare(
                    *model, cpu, phone, 128, reason, 16, 2);
                if (!ok) {
                    std::fprintf(stderr,
                        "[CYCLE_NATIVE] phase=PREPARE status=FAIL reason=%s\n",
                        reason.c_str());
                }
                std::vector<float> embeddings, hidden, logits;
                if (ok) {
                    ok = llama_hybrid_cycle_embed_tokens(
                        *model,cpu,token_ids,embeddings,reason);
                }
                llama_hybrid_cycle_forward_metrics metrics;
                if (ok) {
                    ok = session.evaluate(
                        embeddings, count, hidden, reason, &metrics);
                }
                if (ok) {
                    ok = llama_hybrid_cycle_project_last_token(
                        *model,cpu,hidden,count,logits,reason);
                }
                if (!ok) {
                    std::fprintf(stderr,
                        "[CYCLE_NATIVE] phase=PREFILL status=FAIL reason=%s\n",
                        reason.c_str());
                } else {
                    std::printf(
                        "[CYCLE_NATIVE] phase=PREFILL prompt_tokens=%d "
                        "kv_committed=%lld chunks=%d "
                        "real_c2p=%d real_phone_full=%d real_p2c=%d "
                        "logits=%zu status=PASS\n",
                        count, (long long)session.committed_tokens(),
                        metrics.n_chunks, metrics.completed_c2p,
                        metrics.completed_phone_full, metrics.completed_p2c,
                        logits.size());
                    std::printf("[CYCLE_NATIVE_TEXT] ");
                    int emitted = 0;
                    for (int i = 0; i < max_generate && ok; ++i) {
                        const auto best = std::max_element(
                            logits.begin(),logits.end());
                        if (best == logits.end()) {
                            reason = "EMPTY_LOGITS";
                            ok = false;
                            break;
                        }
                        const auto token =
                            llama_token(best - logits.begin());
                        if (llama_vocab_is_eog(vocab, token)) {
                            std::printf(
                                "\n[CYCLE_NATIVE_DECODE] index=%d "
                                "token=%d eog=1 kv_committed=%lld "
                                "status=STOP\n",
                                i,int(token),
                                (long long)session.committed_tokens());
                            break;
                        }
                        ok = cycle_print_piece(vocab, token);
                        if (!ok) {
                            reason = "TOKEN_DETOKENIZE";
                            break;
                        }
                        ++emitted;
                        if (i + 1 == max_generate) break;
                        // Real one-token incremental Decode: the next
                        // absolute KV position follows the committed prompt
                        // and previous generated tokens on ALL 48 layers.
                        ok = llama_hybrid_cycle_embed_tokens(
                            *model,cpu,{token},embeddings,reason) &&
                            session.evaluate(
                                embeddings,1,hidden,reason,&metrics) &&
                            llama_hybrid_cycle_project_last_token(
                                *model,cpu,hidden,1,logits,reason);
                        std::printf(
                            "\n[CYCLE_NATIVE_DECODE] index=%d "
                            "next_kv_position=%lld "
                            "physical_layers=48 status=%s\n",
                            i+1, (long long)session.committed_tokens(),
                            ok ? "PASS" : "FAIL");
                    }
                    std::printf(
                        "\n[CYCLE_NATIVE] phase=COMPLETE "
                        "generated=%d kv_committed=%lld "
                        "max_kv=128 model=QWEN3_MOE "
                        "sampling=GREEDY status=%s\n",
                        emitted, (long long)session.committed_tokens(),
                        ok ? "PASS" : "FAIL");
                    if (ok) rc = 0;
                }
                if (!ok) {
                    std::fprintf(stderr,
                        "[CYCLE_NATIVE] status=FAIL reason=%s\n",
                        reason.c_str());
                }
                // Session's physical KV and in-flight graph objects are
                // released BEFORE the model and backend handles.
            }
        }
    }
    ggml_backend_free(phone);
    ggml_backend_free(cpu);
    return rc;
}
