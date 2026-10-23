// Standalone REAL physical-Cycle prefill. Not a test harness or CPU oracle.
// Runs ordinary tokenization -> model embedding -> one 48-layer five-FIFO
// prefill -> output RMSNorm/lm_head -> next-token greedy logits.
// Persistent KV autoregressive decode will be a separate session API.
#include "llama.h"
#include "llama-model.h"
#include "llama-hybrid-cycle-forward.h"
#include "llama-hybrid-cycle-model-io.h"
#include "llama-hybrid-cycle-runtime.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr,
            "Usage: llama-cycle-prefill <qwen3-30b-a3b.gguf> "
            "<phone-rpc-address:port> <prompt-text>\n"
            "Requires LLAMA_HYBRID_CYCLE_PC_LAYERS=4 "
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS=2.\n"
            "Current real physical Cycle prefill: 1..64 tokens, "
            "first next-token logits only.\n");
        return 2;
    }
    if (!llama_hybrid_cycle_requested()) {
        std::fprintf(stderr,
            "[CYCLE_NATIVE] status=FAIL reason=CYCLE_LAYOUT_NOT_REQUESTED\n");
        return 2;
    }
    llama_backend_init();
    int rc = 1;
    const auto cpu_dev = ggml_backend_dev_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU);
    const auto rpc_reg = ggml_backend_rpc_add_server(argv[2]);
    if (!cpu_dev || !rpc_reg ||
        ggml_backend_reg_dev_count(rpc_reg) != 1) {
        std::fprintf(stderr,
            "[CYCLE_NATIVE] status=FAIL reason=BACKEND_REGISTRATION\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(rpc_reg, 0);
    if (!phone_dev ||
        ggml_backend_dev_type(phone_dev) !=
            GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr,
            "[CYCLE_NATIVE] status=FAIL reason=PHONE_GPU_NOT_FOUND\n");
        return 1;
    }

    ggml_backend_dev_t devices[] = {cpu_dev, phone_dev, nullptr};
    llama_model_params model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    model_params.hybrid_auto = false;
    model_params.use_extra_bufts = false;
    const auto cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    const auto phone = ggml_backend_dev_init(phone_dev, nullptr);
    if (!cpu || !phone) {
        std::fprintf(stderr,
            "[CYCLE_NATIVE] status=FAIL reason=BACKEND_INIT\n");
        if (phone) ggml_backend_free(phone);
        if (cpu) ggml_backend_free(cpu);
        return 1;
    }
    {
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1],model_params),
            &llama_model_free);
        if (!model) {
            std::fprintf(stderr,
                "[CYCLE_NATIVE] status=FAIL reason=MODEL_LOAD\n");
        } else if (model->arch != LLM_ARCH_QWEN3MOE ||
                   llama_model_n_layer(model.get()) != 48) {
            std::fprintf(stderr,
                "[CYCLE_NATIVE] status=FAIL reason=EXPECTED_48_LAYER_QWEN3MOE\n");
        } else {
            const auto * vocab = llama_model_get_vocab(model.get());
            const std::string prompt(argv[3]);
            std::vector<llama_token> ids(128);
            int32_t n = llama_tokenize(
                vocab,prompt.c_str(),int32_t(prompt.size()),ids.data(),
                int32_t(ids.size()),true,true);
            if (n > 0 && n <= 64) {
                ids.resize(size_t(n));
                std::printf(
                    "[CYCLE_NATIVE] prompt_tokens=%d model=QWEN3_30B_A3B "
                    "embedding=PHYSICAL_CPU forward=GLOBAL_48_LAYER_FIFO "
                    "logits=PHYSICAL_CPU_HEAD status=BEGIN\n", n);
                std::string error;
                std::vector<float> embeddings,hidden,logits;
                llama_hybrid_cycle_forward_metrics metrics;
                const bool embedded =
                    llama_hybrid_cycle_embed_tokens(
                        *model,cpu,ids,embeddings,error);
                if (!embedded) {
                    std::fprintf(stderr,
                        "[CYCLE_NATIVE] stage=EMBED status=FAIL reason=%s\n",
                        error.c_str());
                }
                const bool forwarded = embedded &&
                    llama_hybrid_cycle_forward_prefill(
                        *model,cpu,phone,embeddings,n,hidden,error,
                        &metrics);
                if (embedded && !forwarded) {
                    std::fprintf(stderr,
                        "[CYCLE_NATIVE] stage=PHYSICAL_FORWARD "
                        "status=FAIL reason=%s\n",error.c_str());
                }
                const bool projected = forwarded &&
                    llama_hybrid_cycle_project_last_token(
                        *model,cpu,hidden,n,logits,error);
                if (forwarded && !projected) {
                    std::fprintf(stderr,
                        "[CYCLE_NATIVE] stage=OUTPUT_HEAD "
                        "status=FAIL reason=%s\n",error.c_str());
                }
                if (projected) {
                    const auto best = std::max_element(
                        logits.begin(),logits.end());
                    const llama_token token =
                        llama_token(best-logits.begin());
                    std::vector<char> piece(256);
                    int32_t bytes = llama_token_to_piece(
                        vocab,token,piece.data(),int32_t(piece.size()),0,true);
                    if (bytes < 0) {
                        piece.resize(size_t(-bytes));
                        bytes = llama_token_to_piece(
                            vocab,token,piece.data(),
                            int32_t(piece.size()),0,true);
                    }
                    const std::string next_piece = bytes >= 0 ?
                        std::string(piece.data(),size_t(bytes)) :
                        std::string();
                    std::printf(
                        "[CYCLE_NATIVE] stage=PREFILL_TO_LOGITS "
                        "prompt_tokens=%d physical_layers=48 "
                        "chunks=%d completed_c2p=%d "
                        "completed_phone_full=%d completed_p2c=%d "
                        "kv=COMPLETE next_token_id=%d "
                        "top_logit=%.9g logits_size=%zu "
                        "autoregressive_decode=NOT_YET_ENABLED "
                        "status=PASS\n",
                        n, metrics.n_chunks, metrics.completed_c2p,
                        metrics.completed_phone_full,
                        metrics.completed_p2c,
                        int(token), double(*best), logits.size());
                    std::printf("[CYCLE_NATIVE_NEXT_TOKEN] %s\n",
                                next_piece.c_str());
                    rc = 0;
                }
            } else {
                std::fprintf(stderr,
                    "[CYCLE_NATIVE] status=FAIL "
                    "reason=PROMPT_TOKEN_COUNT_UNSUPPORTED count=%d "
                    "supported=1..64\n", n);
            }
        }
    } // weights before all backend handles
    ggml_backend_free(phone);
    ggml_backend_free(cpu);
    return rc;
}
