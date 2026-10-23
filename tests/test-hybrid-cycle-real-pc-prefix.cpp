// Opt-in physical Qwen3-MoE PC FULL_LAYER compute smoke test.
//
// This is REAL Attention + RoPE + persistent K/V + Router + Gate/Up/SwiGLU
// + Down + residual ADD on CPU model weights for L0..L3. It does not fake
// missing Phone RPC stages or claim a complete 48-layer forward pass.
//
// Usage:
//   test-hybrid-cycle-real-pc-prefix <model.gguf> <phone:rpc-port> [tokens=16]
// Environment:
//   LLAMA_HYBRID_CYCLE_PC_LAYERS=4
//   LLAMA_HYBRID_CYCLE_PHONE_LAYERS=2
//
// Real PC chunks are executed in token order (sequential prefix smoke);
// the one-global-FIFO full forward still requires real Phone stage binding.
#include "llama.h"
#include "llama-model.h"
#include "llama-hybrid-cycle-pc-full.h"

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4 || !llama_hybrid_cycle_requested()) {
        std::fprintf(stderr,
            "Usage: test-hybrid-cycle-real-pc-prefix <model.gguf> "
            "<phone:rpc-port> [tokens=16]\n"
            "Set LLAMA_HYBRID_CYCLE_PC_LAYERS=4 and "
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS=2.\n");
        return 2;
    }
    char * end = nullptr;
    const long parsed = argc == 4 ? std::strtol(argv[3], &end, 10) : 16;
    if (parsed < 1 || parsed > 64 ||
        (argc == 4 && (!end || *end != '\0'))) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_PREFIX] status=FAIL reason=BAD_TOKEN_COUNT\n");
        return 2;
    }
    const int n_tokens = int(parsed);
    const int chunk_tokens = 16;
    llama_backend_init();
    const auto cpu_dev = ggml_backend_dev_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU);
    const auto rpc_reg = ggml_backend_rpc_add_server(argv[2]);
    if (!cpu_dev || !rpc_reg ||
        ggml_backend_reg_dev_count(rpc_reg) != 1) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_PREFIX] status=FAIL reason=BACKEND_REGISTRATION\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(rpc_reg, 0);
    if (!phone_dev || ggml_backend_dev_type(phone_dev) !=
                          GGML_BACKEND_DEVICE_TYPE_GPU) {
        return 1;
    }

    ggml_backend_dev_t devices[] = {cpu_dev, phone_dev, nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.devices = devices;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.hybrid_auto = false;
    mp.use_extra_bufts = false;

    const auto cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    const auto phone = ggml_backend_dev_init(phone_dev, nullptr);
    if (!cpu || !phone) {
        if (phone) ggml_backend_free(phone);
        if (cpu) ggml_backend_free(cpu);
        return 1;
    }

    bool ok = false;
    {
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), &llama_model_free);
        if (model) {
            llama_hybrid_cycle_layout physical;
            const int n_layers = llama_model_n_layer(model.get());
            if (n_layers == 48 &&
                llama_hybrid_cycle_layout_from_env(n_layers, physical) &&
                physical.cycle_size == 8 &&
                physical.x == 4 && physical.y == 2) {
                llama_hybrid_cycle_physical_resource_pool resources;
                llama_hybrid_cycle_resource_config config;
                config.kv_tokens = 64;
                config.max_inflight = 2;
                config.cuda_weight_budget_bytes = 0;
                std::string reason;
                if (resources.prepare(
                        *model, physical, cpu, phone, nullptr,
                        config, reason)) {
                    llama_hybrid_cycle_execution_session_config sc;
                    sc.layout = resources.execution_layout();
                    sc.physical_layer_begin = 0;
                    sc.physical_layer_count = physical.x;
                    sc.tokens = n_tokens;
                    sc.chunk_tokens = chunk_tokens;
                    llama_hybrid_cycle_execution_session<> session(sc);

                    const int64_t hidden = model->hparams.n_embd;
                    const size_t elem_count = size_t(n_tokens) *
                                              size_t(hidden);
                    std::vector<float> synthetic_input(elem_count);
                    for (size_t i = 0; i < elem_count; ++i) {
                        synthetic_input[i] =
                            0.02f + 0.01f *
                            std::sin(float((i % 127) + 1) * 0.03f);
                    }
                    ok = true;
                    for (int l = 0; l < physical.x; ++l) {
                        if (!session.borrow_kv(
                                l, resources.layer(l)->kv.get())) {
                            ok = false;
                            break;
                        }
                    }
                    llama_hybrid_cycle_stage_dispatcher stages(
                        resources.execution_layout());
                    if (ok && !llama_hybrid_cycle_bind_pc_full_stages(
                            stages, *model, resources, cpu, session,
                            synthetic_input, hidden, reason)) {
                        std::fprintf(stderr,
                            "[CYCLE_REAL_PC_PREFIX] status=FAIL "
                            "reason=BIND_PC_FULL details=%s\n",
                            reason.c_str());
                        ok = false;
                    }
                    // A direct smoke test deliberately calls ONLY the
                    // physically bound PC stages. Do not run the complete
                    // FIFO while Phone Tensor/MAIN handlers are unbound.
                    for (int l = 0; l < physical.x && ok; ++l) {
                        for (int c = 0; c < session.chunks_per_layer(); ++c) {
                            llama_hybrid_cycle_task task {};
                            task.layer = l;
                            task.chunk = c;
                            task.stage =
                                llama_hybrid_cycle_stage::FULL_LAYER;
                            task.owner =
                                llama_hybrid_cycle_owner::PC_CPU;
                            task.queue =
                                llama_hybrid_cycle_queue::PC_CPU;
                            if (!stages.execute(task)) {
                                std::fprintf(stderr,
                                    "[CYCLE_REAL_PC_PREFIX] layer=%d "
                                    "chunk=%d status=FAIL reason=EXECUTE\n",
                                    l, c);
                                ok = false;
                                break;
                            }
                            const auto * layer = resources.layer(l);
                            if (!layer || !layer->kv ||
                                layer->kv->committed_tokens !=
                                    session.at(l, c).start +
                                    session.at(l, c).tokens) {
                                ok = false;
                                break;
                            }
                            std::printf(
                                "[CYCLE_REAL_PC_PREFIX_LAYER] "
                                "layer=%d chunk=%d pos=[%" PRId64
                                ",%" PRId64 "] backend=CPU "
                                "kv_committed=%" PRId64 " status=%s\n",
                                l, c, session.at(l, c).start,
                                session.at(l, c).start +
                                    session.at(l, c).tokens - 1,
                                layer->kv->committed_tokens,
                                ok ? "OK" : "FAIL");
                        }
                    }
                    if (ok) {
                        ok = true;
                        double sum = 0.0;
                        double max_abs = 0.0;
                        for (int c = 0; c < session.chunks_per_layer();
                             ++c) {
                            const auto & state =
                                session.at(physical.x - 1, c);
                            if (!state.host_output_published.load(
                                    std::memory_order_acquire) ||
                                state.host_output.size() !=
                                    size_t(state.tokens) * size_t(hidden)) {
                                ok = false;
                                break;
                            }
                            for (float v : state.host_output) {
                                if (!std::isfinite(v)) {
                                    ok = false;
                                    break;
                                }
                                sum += v;
                                max_abs = std::max(
                                    max_abs, std::fabs(double(v)));
                            }
                            if (!ok) break;
                        }
                        std::printf(
                            "[CYCLE_REAL_PC_PREFIX] layers=4 tokens=%d "
                            "chunks=%d source=SYNTHETIC_HIDDEN "
                            "real_attention_ffn=1 full_48_forward=NOT_RUN "
                            "logits=NOT_RUN sum=%.9g max_abs=%.9g "
                            "status=%s\n",
                            n_tokens, session.chunks_per_layer(),
                            sum, max_abs, ok ? "PASS" : "FAIL");
                    }
                } else {
                    std::fprintf(stderr,
                        "[CYCLE_REAL_PC_PREFIX] status=FAIL "
                        "reason=PHYSICAL_RESOURCES detail=%s\n",
                        reason.c_str());
                }
            } else {
                std::fprintf(stderr,
                    "[CYCLE_REAL_PC_PREFIX] status=FAIL "
                    "reason=EXPECTED_48_LAYER_X4_Y2\n");
            }
        } else {
            std::fprintf(stderr,
                "[CYCLE_REAL_PC_PREFIX] status=FAIL reason=MODEL_LOAD\n");
        }
    }
    ggml_backend_free(phone);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
