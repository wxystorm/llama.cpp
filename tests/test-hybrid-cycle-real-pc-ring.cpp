// Opt-in mixed Cycle probe: real PC Full/C2P and optionally real Phone C2P.
//
// All 48 layers share ONE five-worker FIFO. The 24 PC Full layers execute
// real Qwen3-MoE Attention/RoPE/Router/FFN on their physical CPU weights,
// using physical persistent KV and actual Session host handoffs.
// C2P PC Attention and its FFN_SOURCE shard ALSO execute real CPU weights.
// LLAMA_HYBRID_CYCLE_REAL_C2P_PHONE=1 enables legacy V5 REAL C2P GPU.
// LLAMA_HYBRID_CYCLE_REAL_PHONE_FULL=1 additionally executes all 12
// Phone Full layers using direct GPU l_out handoffs (requires REAL C2P).
// LLAMA_HYBRID_CYCLE_REAL_P2C=1 additionally runs all six P2C stages
// with real Phone Attention/FFN, independent P2C READY/PARTIAL transport,
// PC destination FFN, exclusive PC Reduce, and real reverse ACK.
// Without LLAMA_HYBRID_CYCLE_REAL_REFERENCE=1, all 48 layers use a SYNTHETIC
// hidden input and no numerical oracle. With the flag, first run an ordinary
// CPU-only llama_decode on real tokens, capture embedding + all six P2C l_out,
// RELEASE that model, then run all 48 real physical layers and enforce the
// same strict full/chunk parity bound as V5. Logits remain NOT RUN.
// Without these flags, preserve the established regression paths.
// This is NOT a 48-layer Transformer forward or a numeric parity test.
//
// Usage:
//   test-hybrid-cycle-real-pc-ring <model.gguf> <phone:rpc-port> [tokens=16]
// Requires LLAMA_HYBRID_CYCLE_PC_LAYERS=4 and
//          LLAMA_HYBRID_CYCLE_PHONE_LAYERS=2.
// 16/32/64 tokens; fixed 16 tokens per Chunk and max_inflight <=2.
#include "llama.h"
#include "llama-model.h"
#include "llama-hybrid-cycle-pc-full.h"
#include "llama-hybrid-cycle-c2p-pc.h"
#include "llama-hybrid-cycle-phone-full.h"
#include "llama-hybrid-cycle-p2c-real.h"
#include "llama-hybrid-cycle-phone-regions.h"
#include "llama-hybrid-cycle-wire.h"
#include "inference-cycle-reduce.h"
#include "hybrid-cycle-full-reference.h"
#include "hybrid-cycle-same-input.h"

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Match V5's low-memory data-plane settings without overriding user values.
// The slots/payload are initialized before opening Phone inference transport.
static bool cycle_pc_ring_env_default(
        const char * name, const char * value) {
    if (std::getenv(name)) return true;
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 0) == 0;
#endif
}

static bool cycle_pc_ring_finite(const std::vector<float> & values,
                                size_t expected) {
    if (values.size() != expected) return false;
    for (float value : values) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

static const llama_hybrid_cycle_task_interval * cycle_pc_ring_find(
        const llama_hybrid_cycle_chunk_timeline & timeline, int layer,
        int chunk, llama_hybrid_cycle_stage stage) {
    for (const auto & events : timeline.by_queue) {
        for (const auto & event : events) {
            if (event.task.layer == layer && event.task.chunk == chunk &&
                event.task.stage == stage) {
                return &event;
            }
        }
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4 || !llama_hybrid_cycle_requested()) {
        std::fprintf(stderr,
            "Usage: test-hybrid-cycle-real-pc-ring <model.gguf> "
            "<phone:rpc-port> [tokens=16]\n"
            "Set LLAMA_HYBRID_CYCLE_PC_LAYERS=4 and "
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS=2.\n");
        return 2;
    }
    char * end = nullptr;
    const long parsed = argc == 4 ? std::strtol(argv[3], &end, 10) : 16;
    if (parsed <= 0 || parsed > 64 ||
        (argc == 4 && (!end || *end != '\0'))) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=BAD_TOKEN_COUNT\n");
        return 2;
    }
    const int n_tokens = int(parsed);
    constexpr int chunk_tokens = 16;
    const int n_chunks = (n_tokens + chunk_tokens - 1) / chunk_tokens;
    const int max_inflight = std::min(2, n_chunks);
    const char * real_phone_env = std::getenv(
        "LLAMA_HYBRID_CYCLE_REAL_C2P_PHONE");
    if (real_phone_env && std::strcmp(real_phone_env, "0") != 0 &&
        std::strcmp(real_phone_env, "1") != 0) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL "
            "reason=REAL_C2P_PHONE_REQUIRES_0_OR_1\n");
        return 2;
    }
    const bool real_c2p_phone =
        real_phone_env && std::strcmp(real_phone_env, "1") == 0;
    const char * real_full_env = std::getenv(
        "LLAMA_HYBRID_CYCLE_REAL_PHONE_FULL");
    if (real_full_env && std::strcmp(real_full_env, "0") != 0 &&
        std::strcmp(real_full_env, "1") != 0) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL "
            "reason=REAL_PHONE_FULL_REQUIRES_0_OR_1\n");
        return 2;
    }
    const bool real_phone_full =
        real_full_env && std::strcmp(real_full_env, "1") == 0;
    if (real_phone_full && !real_c2p_phone) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL "
            "reason=REAL_PHONE_FULL_REQUIRES_REAL_C2P\n");
        return 2;
    }
    const char * real_p2c_env = std::getenv(
        "LLAMA_HYBRID_CYCLE_REAL_P2C");
    if (real_p2c_env && std::strcmp(real_p2c_env, "0") != 0 &&
        std::strcmp(real_p2c_env, "1") != 0) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=BAD_REAL_P2C_FLAG\n");
        return 2;
    }
    const bool real_p2c = real_p2c_env &&
        std::strcmp(real_p2c_env, "1") == 0;
    if (real_p2c && !real_phone_full) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL "
            "reason=REAL_P2C_REQUIRES_REAL_PHONE_FULL\n");
        return 2;
    }
    const char * reference_env = std::getenv(
        "LLAMA_HYBRID_CYCLE_REAL_REFERENCE");
    if (reference_env && std::strcmp(reference_env, "0") != 0 &&
        std::strcmp(reference_env, "1") != 0) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=BAD_REFERENCE_FLAG\n");
        return 2;
    }
    const bool use_cpu_reference = reference_env &&
        std::strcmp(reference_env, "1") == 0;
    if (use_cpu_reference && !real_p2c) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL "
            "reason=REAL_REFERENCE_REQUIRES_REAL_48_LAYERS\n");
        return 2;
    }
    const char * trace_env = std::getenv("LLAMA_HYBRID_CYCLE_TRACE_PARITY");
    if (trace_env && std::strcmp(trace_env, "0") != 0 &&
        std::strcmp(trace_env, "1") != 0) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=BAD_TRACE_PARITY_FLAG\n");
        return 2;
    }
    const bool trace_parity = trace_env &&
        std::strcmp(trace_env, "1") == 0;
    if (trace_parity && !use_cpu_reference) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL "
            "reason=TRACE_PARITY_REQUIRES_CPU_REFERENCE\n");
        return 2;
    }
    const char * stage_env = std::getenv("LLAMA_HYBRID_CYCLE_TRACE_STAGES");
    if (stage_env && std::strcmp(stage_env, "0") != 0 &&
        std::strcmp(stage_env, "1") != 0) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=BAD_TRACE_STAGES_FLAG\n");
        return 2;
    }
    const bool trace_stages = stage_env && std::strcmp(stage_env, "1") == 0;
    if (trace_stages && !trace_parity) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL "
            "reason=TRACE_STAGES_REQUIRES_TRACE_PARITY\n");
        return 2;
    }
    const char * ab_env = std::getenv(
        "LLAMA_HYBRID_CYCLE_AB_SAME_INPUT");
    if (ab_env && std::strcmp(ab_env, "0") != 0 &&
        std::strcmp(ab_env, "1") != 0) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=BAD_AB_SAME_INPUT_FLAG\n");
        return 2;
    }
    const bool same_input_ab = ab_env && std::strcmp(ab_env, "1") == 0;
    if (same_input_ab && (!trace_stages || n_tokens != 16)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL "
            "reason=AB_SAME_INPUT_REQUIRES_STAGE_TRACE_AND_16_TOKENS\n");
        return 2;
    }

    llama_backend_init();
    const auto cpu_dev = ggml_backend_dev_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=CPU_NOT_REGISTERED\n");
        return 1;
    }
    // The independent CPU-only model/context is destroyed INSIDE run()
    // before any PC/Phone shard model or Phone transport is opened.
    // Keep only the captured ~7x(token*hidden) F32 host arrays.
    cycle_full_reference::baseline reference;
    if (use_cpu_reference &&
        !reference.run(argv[1], cpu_dev, n_tokens, trace_parity,
                       trace_stages)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=CPU_ORACLE\n");
        return 1;
    }
    const auto rpc_reg = ggml_backend_rpc_add_server(argv[2]);
    if (!cpu_dev || !rpc_reg ||
        ggml_backend_reg_dev_count(rpc_reg) != 1) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=BACKEND_REGISTRATION\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(rpc_reg, 0);
    if (!phone_dev ||
        ggml_backend_dev_type(phone_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr,
            "[CYCLE_REAL_PC_RING] status=FAIL reason=PHONE_GPU_MISSING\n");
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
        if (!model) {
            std::fprintf(stderr,
                "[CYCLE_REAL_PC_RING] status=FAIL reason=MODEL_LOAD\n");
        } else {
            llama_hybrid_cycle_layout physical;
            if (llama_model_n_layer(model.get()) != 48 ||
                !llama_hybrid_cycle_layout_from_env(48, physical) ||
                physical.x != 4 || physical.y != 2 ||
                physical.cycle_size != 8 || physical.cycles != 6) {
                std::fprintf(stderr,
                    "[CYCLE_REAL_PC_RING] status=FAIL "
                    "reason=EXPECTED_48_LAYER_X4_Y2\n");
            } else {
                llama_hybrid_cycle_physical_resource_pool resources;
                llama_hybrid_cycle_resource_config config;
                config.kv_tokens = 64;
                config.max_inflight = max_inflight;
                config.cuda_weight_budget_bytes = 0; // Explicit CPU fallback.
                std::string reason;
                if (!resources.prepare(
                        *model, physical, cpu, phone, nullptr,
                        config, reason)) {
                    std::fprintf(stderr,
                        "[CYCLE_REAL_PC_RING] status=FAIL "
                        "reason=PHYSICAL_RESOURCES detail=%s\n",
                        reason.c_str());
                } else {
                    const auto & layout = resources.execution_layout();
                    llama_hybrid_cycle_phone_regions phone_regions;
                    ok = phone_regions.prepare(
                            layout, n_tokens, chunk_tokens, max_inflight,
                            reason) &&
                         phone_regions.validate_resources(resources, reason) &&
                         phone_regions.region_count() == 6;
                    if (!ok) {
                        std::fprintf(stderr,
                            "[CYCLE_REAL_PC_RING] status=FAIL "
                            "reason=PHONE_REGION detail=%s\n",
                            reason.c_str());
                    }
                    if (ok && real_c2p_phone) {
                        // V5 uses two 1-MiB TX/RX slots for 16-token READY.
                        // This probe has one global transport, not six sockets.
                        auto * dev = ggml_backend_get_device(phone);
                        auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
                        const auto open = reg ?
                            reinterpret_cast<
                                ggml_backend_rpc_inference_transport_open_t>(
                                ggml_backend_reg_get_proc_address(
                                    reg,
                                    GGML_BACKEND_RPC_INFERENCE_TRANSPORT_OPEN_PROC)) :
                            nullptr;
                        ok = cycle_pc_ring_env_default(
                                 "GGML_INFERENCE_TRANSPORT_SLOTS", "2") &&
                             cycle_pc_ring_env_default(
                                 "GGML_INFERENCE_TRANSPORT_MAX_PAYLOAD_MIB", "1") &&
                             ggml_backend_is_rpc(phone) &&
                             open && open(phone);
                        std::printf(
                            "[CYCLE_REAL_C2P_PHONE] phase=TRANSPORT_OPEN "
                            "regions=6 slots=ENV_OR_DEFAULT "
                            "mode=REAL_RPC_DATA_PLANE status=%s\n",
                            ok ? "PASS" : "FAIL");
                    }
                    if (ok) {
                        llama_hybrid_cycle_execution_session_config sc;
                        sc.layout = layout;
                        sc.physical_layer_count = 48;
                        sc.tokens = n_tokens;
                        sc.chunk_tokens = chunk_tokens;
                        llama_hybrid_cycle_execution_session<> session(sc);
                        for (int layer = 0; layer < 48; ++layer) {
                            const auto * r = resources.layer(layer);
                            if (!r || !r->kv ||
                                !session.borrow_kv(layer, r->kv.get())) {
                                ok = false;
                                break;
                            }
                        }
                        const int64_t hidden = model->hparams.n_embd;
                        const size_t total = size_t(n_tokens) * size_t(hidden);
                        std::vector<float> initial_hidden(total);
                        for (size_t i = 0; i < total; ++i) {
                            initial_hidden[i] = 0.02f + 0.01f *
                                std::sin(float((i % 127) + 1) * 0.03f);
                        }
                        if (use_cpu_reference) {
                            // Same absolute token positions and F32 embedding
                            // as the independent standard llama_decode oracle.
                            // No synthetic activation can enter the real test.
                            const bool input_ok =
                                reference.n_tokens == n_tokens &&
                                reference.hidden == hidden &&
                                reference.embedding.size() == total;
                            if (input_ok) {
                                initial_hidden = reference.embedding;
                            } else {
                                ok = false;
                            }
                            std::printf(
                                "[CYCLE_FULL_REFERENCE_INPUT] source=CPU_TOKEN_EMBD "
                                "tokens=%d elements=%zu status=%s\n",
                                n_tokens, total, input_ok ? "PASS" : "FAIL");
                        }

                        // Diagnostic mode: compare every immutable layer
                        // output against the ordinary llama_decode CPU oracle.
                        // Normally DISABLED: GPU readback here would distort
                        // latency and must not become part of the real pipeline.
                        std::atomic<int> trace_seen {0};
                        std::atomic<int> trace_failed {0};
                        std::atomic<int> trace_first_bad {48};
                        const auto trace_host = [&](int layer, int chunk,
                                int64_t position, int64_t tokens,
                                const std::vector<float> & actual,
                                const char * backend) {
                            if (!trace_parity) return;
                            const bool pass =
                                cycle_full_reference::compare_layer_chunk(
                                    actual,
                                    reference.after_layer[size_t(layer)],
                                    layer, chunk, position, tokens,
                                    hidden, backend);
                            trace_seen.fetch_add(1, std::memory_order_relaxed);
                            if (!pass) {
                                trace_failed.fetch_add(
                                    1, std::memory_order_relaxed);
                                // Many FIFO workers may finish out-of-order.
                                // Report the FIRST PHYSICAL layer with a
                                // strict mismatch, not the first log timestamp.
                                int first = trace_first_bad.load(
                                    std::memory_order_relaxed);
                                while (layer < first &&
                                       !trace_first_bad.compare_exchange_weak(
                                            first, layer,
                                            std::memory_order_relaxed)) {}
                            }
                        };
                        const auto trace_gpu = [&](int layer, int chunk,
                                int64_t position, int64_t tokens,
                                const ggml_tensor * tensor,
                                const char * backend) {
                            if (!trace_parity) return;
                            const size_t expected =
                                size_t(tokens) * size_t(hidden);
                            std::vector<float> host;
                            if (tensor && tensor->buffer &&
                                tensor->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(tensor) &&
                                ggml_nelements(tensor) == int64_t(expected)) {
                                // Device-complete, immutable output BEFORE
                                // downstream slot retirement/reuse. This is
                                // an explicit diagnostic transfer ONLY.
                                host.resize(expected);
                                ggml_backend_tensor_get(
                                    tensor, host.data(), 0,
                                    expected * sizeof(float));
                            }
                            trace_host(layer, chunk, position,
                                       tokens, host, backend);
                        };
                        const auto trace_stage_host = [&](
                                int l, int c, int64_t start, int64_t ntok,
                                const std::vector<float> & values,
                                const char * backend, const char * point) {
                            if (!trace_stages ||
                                !cycle_full_reference::stage_focus(l)) return;
                            const std::vector<float> * ref = nullptr;
                            if (std::strcmp(point, "ATTN_RESIDUAL") == 0) {
                                ref = &reference.after_attention[size_t(l)];
                            } else if (std::strcmp(point, "FFN_NORM") == 0) {
                                ref = &reference.after_ffn_norm[size_t(l)];
                            } else if (std::strcmp(point, "FFN_OUTPUT") == 0) {
                                ref = &reference.after_ffn[size_t(l)];
                            }
                            if (!ref || ref->empty()) {
                                std::printf(
                                    "[CYCLE_STAGE_PARITY] layer=%d chunk=%d "
                                    "backend=%s point=%s status=NOT_CAPTURED\n",
                                    l, c, backend, point);
                                return;
                            }
                            cycle_full_reference::compare_layer_chunk(
                                values, *ref, l, c, start, ntok,
                                hidden, backend, point);
                        };
                        const auto trace_stage_gpu = [&](
                                int l, int c, int64_t start, int64_t ntok,
                                const ggml_tensor * tensor,
                                const char * backend, const char * point) {
                            if (!trace_stages ||
                                !cycle_full_reference::stage_focus(l)) return;
                            std::vector<float> values;
                            const size_t elements =
                                size_t(ntok) * size_t(hidden);
                            if (tensor && tensor->buffer &&
                                tensor->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(tensor) &&
                                ggml_nelements(tensor) == int64_t(elements)) {
                                values.resize(elements);
                                // Phone FFN/Reduce or Attention has completed
                                // and the GPU slot is still owned. Diagnostic
                                // copies are never part of normal FIFO flow.
                                ggml_backend_tensor_get(
                                    tensor, values.data(), 0,
                                    elements * sizeof(float));
                            }
                            trace_stage_host(l, c, start, ntok, values,
                                             backend, point);
                        };

                        std::function<void(
                            int, int, int64_t, int64_t,
                            const std::vector<float> &)> pc_trace;
                        if (trace_parity) {
                            pc_trace = [&](int l, int c, int64_t start,
                                           int64_t tokens,
                                           const std::vector<float> & v) {
                                trace_host(
                                    l, c, start, tokens, v, "PC_CPU_FULL");
                            };
                        }

                        // Mock Phone device activations live until all FIFO
                        // workers join. Exactly one producing stage writes
                        // each (physical layer, Chunk); dependencies in the
                        // Ring publish before any successor reads.
                        std::vector<std::vector<float>> mock_phone_output(
                            size_t(48 * n_chunks));
                        const auto mock_at = [&](int layer, int chunk)
                                -> std::vector<float> & {
                            return mock_phone_output[
                                size_t(layer * n_chunks + chunk)];
                        };
                        std::array<std::atomic<int>, 6> p2c_completed {};
                        for (auto & count : p2c_completed) count.store(0);
                        // Each period owns its FFN buffers. Idle slots can
                        // cross worker queues only under the period mutex.
                        llama_hybrid_cycle_phone_region_slot_banks<
                            llama_hybrid_cycle_phone_ffn_slot,
                            llama_hybrid_cycle_p2c_ffn_slot> phone_slot_banks;
                        if (real_c2p_phone) phone_slot_banks.prepare(phone_regions);
                        std::atomic<int> mock_stages {0};
                        std::atomic<int> real_c2p_attn {0};
                        std::atomic<int> real_c2p_ffn {0};
                        std::atomic<int> phone_c2p_ffn {0};
                        std::atomic<int> phone_c2p_ready {0};
                        std::atomic<int> phone_c2p_partial {0};
                        std::atomic<int> phone_c2p_reduce {0};
                        std::atomic<int> c2p_slot_reuse {0};
                        std::atomic<int> real_phone_full_calls {0};
                        std::atomic<int> c2p_device_handoffs {0};
                        std::atomic<int> phone_full_device_handoffs {0};
                        std::atomic<int> phone_full_p2c_readbacks {0};
                        std::atomic<int> phone_full_slot_reuse {0};
                        std::atomic<int> real_p2c_attn {0};
                        std::atomic<int> real_p2c_ready_rx {0};
                        std::atomic<int> real_p2c_phone_ffn {0};
                        std::atomic<int> real_p2c_pc_ffn {0};
                        std::atomic<int> real_p2c_partial_rx {0};
                        std::atomic<int> real_p2c_reduce {0};
                        std::atomic<int> real_p2c_partial_ack {0};
                        std::atomic<int> p2c_device_handoffs {0};
                        std::atomic<int> p2c_slot_reuse {0};
                        llama_hybrid_cycle_runtime_config rc;
                        rc.layout = layout;
                        rc.n_chunks = n_chunks;
                        rc.max_inflight = max_inflight;
                        rc.credit_until_handoff = true;
                        rc.pc_cpu_reduce_priority = true;
                        rc.ordered_c2p_reduce = true;
                        rc.phone_main_priority = true;
                        llama_hybrid_cycle_runtime runtime(rc);
                        llama_hybrid_cycle_stage_dispatcher stages(layout);
                        if (ok && !llama_hybrid_cycle_bind_pc_full_stages(
                                stages, *model, resources, cpu, session,
                                initial_hidden, hidden, reason, pc_trace)) {
                            std::fprintf(stderr,
                                "[CYCLE_REAL_PC_RING] status=FAIL "
                                "reason=BIND_REAL_PC detail=%s\n",
                                reason.c_str());
                            ok = false;
                        }

                        if (ok && !phone_regions.bind_phone_stages(
                                stages,
                                [&](llama_hybrid_cycle_phone_region_progress & region,
                                    const llama_hybrid_cycle_task & task) -> bool {
                                    const int l = task.layer;
                                    const int c = task.chunk;
                                    if (l < region.location.c2p_layer ||
                                        l > region.location.p2c_layer ||
                                        c < 0 || c >= n_chunks ||
                                        region.location.cycle_index != l / 8) {
                                        return false;
                                    }
                                    const auto kind = layout.kinds[size_t(l)];
                                    auto & state = session.at(l, c);
                                    const size_t expected =
                                        size_t(state.tokens) * size_t(hidden);
                                    auto & target = mock_at(l, c);
                                    bool valid = true;
                                    // Real C2P PC source uses physical weights and persistent
                                    // KV. The GPU and both Wire stages remain MOCK.
                                    // No fake RPC graph or ACK is submitted.
                                    if (kind ==
                                        llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                                        if (task.stage ==
                                            llama_hybrid_cycle_stage::ATTN_PRE) {
                                            std::vector<float> host_input;
                                            valid = session.copy_predecessor_host_input(
                                                l, c, hidden, host_input);
                                            const auto * r = resources.layer(l);
                                            valid = valid && r &&
                                                llama_hybrid_cycle_execute_c2p_pc_attention(
                                                    *model, *r, cpu, l,
                                                    state.start, state.tokens,
                                                    host_input, state.ready);
                                            if (valid && trace_stages) {
                                                trace_stage_host(
                                                    l, c, state.start, state.tokens,
                                                    state.ready.residual,
                                                    "C2P_PC_ATTN", "ATTN_RESIDUAL");
                                                trace_stage_host(
                                                    l, c, state.start, state.tokens,
                                                    state.ready.norm,
                                                    "C2P_PC_ATTN", "FFN_NORM");
                                            }
                                            if (valid && real_c2p_phone) {
                                                auto * bank = phone_slot_banks.by_index(
                                                    region.location.cycle_index);
                                                valid = bank != nullptr;
                                                std::unique_ptr<
                                                    llama_hybrid_cycle_phone_ffn_slot> slot;
                                                if (valid) {
                                                    std::lock_guard<std::mutex> lock(
                                                        bank->c2p_mutex);
                                                    for (auto it = bank->c2p_idle.begin();
                                                         it != bank->c2p_idle.end(); ++it) {
                                                        if (*it &&
                                                            (*it)->prepared_tokens ==
                                                                state.tokens &&
                                                            !(*it)->is_full_layer) {
                                                            slot = std::move(*it);
                                                            bank->c2p_idle.erase(it);
                                                            ++c2p_slot_reuse;
                                                            break;
                                                        }
                                                    }
                                                }
                                                if (valid && !slot) {
                                                    const auto & lw = model->layers[size_t(l)];
                                                    const llama_hybrid_cycle_ffn_shard_weights
                                                        shard {
                                                            llama_hybrid_cycle_resolve_physical_weight(
                                                                lw.ffn_up_exps, phone, 1),
                                                            llama_hybrid_cycle_resolve_physical_weight(
                                                                lw.ffn_gate_exps, phone, 1),
                                                            llama_hybrid_cycle_resolve_physical_weight(
                                                                lw.ffn_down_exps, phone, 1),
                                                        };
                                                    slot = std::make_unique<
                                                        llama_hybrid_cycle_phone_ffn_slot>();
                                                    // Legacy V5: FFN and Reduce graphs are
                                                    // submitted by their real FIFO stages.
                                                    valid = shard.available() &&
                                                        slot->prepare(
                                                            phone, shard, hidden,
                                                            state.tokens,
                                                            model->hparams.n_expert_used,
                                                            model->hparams.n_expert);
                                                    if (!valid) slot.reset();
                                                }
                                                if (valid) {
                                                    state.phone_output = std::move(slot);
                                                }
                                                if (!valid) {
                                                    std::fprintf(stderr,
                                                        "[CYCLE_REAL_C2P_PHONE] "
                                                        "layer=%d chunk=%d stage=PREPARE "
                                                        "status=FAIL\n", l, c);
                                                }
                                            }
                                            if (valid) ++real_c2p_attn;
                                        } else if (task.stage ==
                                                   llama_hybrid_cycle_stage::SEND_ACTIVATION &&
                                                   real_c2p_phone) {
                                            const auto & wire = state.wire;
                                            const auto * slot = state.phone_output.get();
                                            valid = slot &&
                                                llama_hybrid_cycle_send_c2p_frame(
                                                    phone, wire,
                                                    INFERENCE_CYCLE_READY,
                                                    state.ready, state.partial,
                                                    uint32_t(state.tokens),
                                                    uint32_t(hidden),
                                                    uint32_t(model->hparams.n_expert_used)) &&
                                                ggml_backend_rpc_cycle_apply_ready(
                                                    phone, wire.layer, wire.chunk,
                                                    wire.seq, wire.graph_id,
                                                    slot->residual, slot->inputs.norm,
                                                    slot->inputs.topk_ids,
                                                    slot->inputs.topk_weights) &&
                                                ggml_backend_rpc_cycle_wait_ack(
                                                    phone, INFERENCE_CYCLE_READY,
                                                    wire.layer, wire.chunk,
                                                    wire.seq, wire.graph_id, 30000);
                                            if (valid) ++phone_c2p_ready;
                                        } else if (task.stage ==
                                                   llama_hybrid_cycle_stage::FFN_SOURCE) {
                                            const auto * r = resources.layer(l);
                                            valid = r &&
                                                llama_hybrid_cycle_execute_c2p_pc_ffn_source(
                                                    *model, *r, cpu, l, state.tokens,
                                                    state.ready, state.partial);
                                            if (valid) ++real_c2p_ffn;
                                        } else if (task.stage ==
                                                   llama_hybrid_cycle_stage::FFN_DESTINATION &&
                                                   real_c2p_phone) {
                                            valid = state.phone_output &&
                                                state.phone_output->execute_ffn();
                                            if (valid) ++phone_c2p_ffn;
                                        } else if (task.stage ==
                                                   llama_hybrid_cycle_stage::SEND_PARTIAL &&
                                                   real_c2p_phone) {
                                            const auto & wire = state.wire;
                                            valid = state.phone_output &&
                                                llama_hybrid_cycle_send_c2p_frame(
                                                    phone, wire,
                                                    INFERENCE_CYCLE_PARTIAL,
                                                    state.ready, state.partial,
                                                    uint32_t(state.tokens),
                                                    uint32_t(hidden),
                                                    uint32_t(model->hparams.n_expert_used)) &&
                                                ggml_backend_rpc_cycle_apply_partial(
                                                    phone, wire.layer, wire.chunk,
                                                    wire.seq, wire.graph_id,
                                                    state.phone_output->pc_partial);
                                            if (valid) ++phone_c2p_partial;
                                        } else if (task.stage ==
                                                   llama_hybrid_cycle_stage::REDUCE) {
                                            valid = cycle_pc_ring_finite(
                                                state.ready.residual, expected) &&
                                                cycle_pc_ring_finite(
                                                    state.partial, expected);
                                            if (valid && real_c2p_phone) {
                                                auto * slot = state.phone_output.get();
                                                const auto & wire = state.wire;
                                                valid = slot && slot->output &&
                                                    slot->output->type == GGML_TYPE_F32 &&
                                                    ggml_is_contiguous(slot->output) &&
                                                    ggml_nelements(slot->output) ==
                                                        int64_t(expected) &&
                                                    slot->execute_reduce(phone) &&
                                                    ggml_backend_rpc_cycle_finish_reduce(
                                                        phone, wire.layer, wire.chunk,
                                                        wire.seq, wire.graph_id) &&
                                                    ggml_backend_rpc_cycle_wait_ack(
                                                        phone, INFERENCE_CYCLE_PARTIAL,
                                                        wire.layer, wire.chunk,
                                                        wire.seq, wire.graph_id, 30000);
                                                if (valid && trace_parity) {
                                                    trace_gpu(
                                                        l, c, state.start,
                                                        state.tokens,
                                                        slot->output,
                                                        "C2P_PHONE_GPU");
                                                }
                                                if (valid && trace_stages &&
                                                    cycle_full_reference::stage_focus(l)) {
                                                    // Full FFN is the PC shard +
                                                    // PHONE_GPU shard before ADD
                                                    // residual. Read only the
                                                    // Phone shard for diagnosis.
                                                    std::vector<float> full_ffn;
                                                    const auto * phone_shard =
                                                        slot->inputs.ffn_output;
                                                    if (phone_shard && phone_shard->buffer &&
                                                        phone_shard->type == GGML_TYPE_F32 &&
                                                        ggml_is_contiguous(phone_shard) &&
                                                        ggml_nelements(phone_shard) ==
                                                            int64_t(expected)) {
                                                        full_ffn.resize(expected);
                                                        ggml_backend_tensor_get(
                                                            phone_shard, full_ffn.data(),
                                                            0, expected * sizeof(float));
                                                        for (size_t i = 0; i < expected; ++i) {
                                                            full_ffn[i] += state.partial[i];
                                                        }
                                                    }
                                                    trace_stage_host(
                                                        l, c, state.start,
                                                        state.tokens, full_ffn,
                                                        "C2P_PC_PLUS_PHONE", "FFN_OUTPUT");
                                                }
                                                if (valid && !real_phone_full) {
                                                    // Legacy diagnostic Host bridge
                                                    // to the old MOCK L5/L13/... layer.
                                                    target.resize(expected);
                                                    ggml_backend_tensor_get(
                                                        slot->output, target.data(), 0,
                                                        target.size() * sizeof(float));
                                                    valid = cycle_pc_ring_finite(
                                                        target, expected);
                                                    if (valid) {
                                                        auto * bank =
                                                            phone_slot_banks.by_index(
                                                                region.location.cycle_index);
                                                        valid = bank != nullptr;
                                                        if (valid) {
                                                            // The Host readback has consumed
                                                            // the output; no GPU successor.
                                                            std::lock_guard<std::mutex> lock(
                                                                bank->c2p_mutex);
                                                            bank->c2p_idle.push_back(
                                                                std::move(state.phone_output));
                                                        }
                                                    }
                                                }
                                                // With REAL Phone Full, L4's GPU slot
                                                // remains owned by THIS chunk until
                                                // L5's Attention finishes consuming it.
                                                // Never recycle on C2P Reduce ACK.
                                                if (valid) {
                                                    ++phone_c2p_reduce;
                                                    std::printf(
                                                        "[CYCLE_REAL_C2P_PHONE] layer=%d "
                                                        "chunk=%d seq=%llu "
                                                        "compute=REAL_GPU reduce=REAL_GPU "
                                                        "handoff=%s status=PASS\n",
                                                        l, c,
                                                        (unsigned long long)wire.seq,
                                                        real_phone_full ?
                                                            "GPU_DEVICE_OWNED_UNTIL_L5" :
                                                            "DIAGNOSTIC_READBACK");
                                                }
                                            } else if (valid) {
                                                // Original mock result: keep the
                                                // already-passing legacy test.
                                                target.resize(expected);
                                                for (size_t i = 0; i < expected; ++i) {
                                                    target[i] =
                                                        state.ready.residual[i] +
                                                        state.partial[i] + 0.001f;
                                                }
                                                valid = cycle_pc_ring_finite(
                                                    target, expected);
                                            }
                                        }
                                    } else if (kind ==
                                               llama_hybrid_cycle_layer_kind::PHONE_GPU) {
                                        if (task.stage !=
                                            llama_hybrid_cycle_stage::FULL_LAYER) {
                                            return false;
                                        }
                                        if (real_phone_full) {
                                            const auto * predecessor =
                                                session.phone_predecessor_output(l, c);
                                            const auto * r = resources.layer(l);
                                            auto * bank = phone_slot_banks.by_index(
                                                region.location.cycle_index);
                                            valid = predecessor && r && bank;
                                            std::unique_ptr<
                                                llama_hybrid_cycle_phone_ffn_slot> full_slot;
                                            if (valid) {
                                                // The sole PHONE_GPU FIFO worker owns
                                                // every idle Full slot in this region.
                                                auto & idle =
                                                    bank->full_idle_by_layer[size_t(l)];
                                                for (auto it = idle.begin();
                                                     it != idle.end(); ++it) {
                                                    if (*it && (*it)->is_full_layer &&
                                                        (*it)->prepared_tokens ==
                                                            state.tokens &&
                                                        !(*it)->fused_full_graph) {
                                                        full_slot = std::move(*it);
                                                        idle.erase(it);
                                                        ++phone_full_slot_reuse;
                                                        break;
                                                    }
                                                }
                                                if (!full_slot) {
                                                    full_slot = std::make_unique<
                                                        llama_hybrid_cycle_phone_ffn_slot>();
                                                }
                                                std::function<void(
                                                    const ggml_tensor *,
                                                    const ggml_tensor *,
                                                    const ggml_tensor *)> stage_probe;
                                                if (trace_stages &&
                                                    cycle_full_reference::stage_focus(l)) {
                                                    stage_probe = [&](const ggml_tensor * residual,
                                                                      const ggml_tensor * norm,
                                                                      const ggml_tensor * ffn) {
                                                        trace_stage_gpu(
                                                            l, c, state.start,
                                                            state.tokens, residual,
                                                            "PHONE_GPU_FULL", "ATTN_RESIDUAL");
                                                        trace_stage_gpu(
                                                            l, c, state.start,
                                                            state.tokens, norm,
                                                            "PHONE_GPU_FULL", "FFN_NORM");
                                                        trace_stage_gpu(
                                                            l, c, state.start,
                                                            state.tokens, ffn,
                                                            "PHONE_GPU_FULL", "FFN_OUTPUT");
                                                    };
                                                }
                                                valid = llama_hybrid_cycle_execute_phone_full(
                                                    *model, *r, phone, l,
                                                    state.start, state.tokens,
                                                    predecessor, state.ready,
                                                    *full_slot, stage_probe);
                                            }
                                            if (valid) {
                                                state.phone_output = std::move(full_slot);
                                                if (trace_parity) {
                                                    trace_gpu(
                                                        l, c, state.start,
                                                        state.tokens,
                                                        state.phone_output->output,
                                                        "PHONE_GPU_FULL");
                                                }
                                                // The previous physical GPU l_out was
                                                // read by this Attention and the backend
                                                // synchronized. This is the ONLY safe
                                                // ownership transfer point for the slot.
                                                auto previous =
                                                    session.retire_phone_output_after_handoff(
                                                        l - 1, l, c);
                                                valid = previous != nullptr;
                                                if (valid) {
                                                    if (l - 1 ==
                                                        region.location.c2p_layer) {
                                                        std::lock_guard<std::mutex> lock(
                                                            bank->c2p_mutex);
                                                        bank->c2p_idle.push_back(
                                                            std::move(previous));
                                                        ++c2p_device_handoffs;
                                                    } else {
                                                        bank->full_idle_by_layer[
                                                            size_t(l - 1)].push_back(
                                                                std::move(previous));
                                                        ++phone_full_device_handoffs;
                                                    }
                                                }
                                            }
                                            if (valid) ++real_phone_full_calls;
                                        } else {
                                            const auto & prior = mock_at(l - 1, c);
                                            valid = cycle_pc_ring_finite(
                                                prior, expected);
                                            if (valid) {
                                                target = prior;
                                                for (float & v : target) v += 0.001f;
                                            }
                                        }
                                    } else if (kind ==
                                               llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
                                        if (real_p2c) {
                                            const auto & wire = state.wire;
                                            const int64_t topk =
                                                model->hparams.n_expert_used;
                                            if (task.stage ==
                                                llama_hybrid_cycle_stage::ATTN_PRE) {
                                                const auto * predecessor =
                                                    session.phone_predecessor_output(l, c);
                                                const auto * resource =
                                                    resources.layer(l);
                                                auto * bank =
                                                    phone_slot_banks.by_index(
                                                        region.location.cycle_index);
                                                llama_hybrid_cycle_attention_rpc_ops rpc {};
                                                rpc.is_rpc = ggml_backend_is_rpc(phone);
                                                rpc.send_ready =
                                                    &ggml_backend_rpc_cycle_send_ready;
                                                valid = predecessor && resource && bank &&
                                                    llama_hybrid_cycle_execute_p2c_phone_attention(
                                                        *model, *resource, phone, l,
                                                        state.start, state.tokens,
                                                        predecessor, wire, state.ready, rpc);
                                                if (valid && trace_stages) {
                                                    trace_stage_gpu(
                                                        l, c, state.start,
                                                        state.tokens,
                                                        state.ready.phone_graph.residual,
                                                        "P2C_PHONE_ATTN", "ATTN_RESIDUAL");
                                                    trace_stage_gpu(
                                                        l, c, state.start,
                                                        state.tokens,
                                                        state.ready.phone_graph.ffn_norm,
                                                        "P2C_PHONE_ATTN", "FFN_NORM");
                                                }
                                                if (valid) {
                                                    // Phone Attention has consumed L6
                                                    // on the same GPU, and the READY TX
                                                    // source is now independently staged.
                                                    auto old_slot =
                                                        session.retire_phone_output_after_handoff(
                                                            l - 1, l, c);
                                                    valid = old_slot != nullptr;
                                                    if (valid) {
                                                        bank->full_idle_by_layer[
                                                            size_t(l - 1)].push_back(
                                                                std::move(old_slot));
                                                        ++p2c_device_handoffs;
                                                        ++real_p2c_attn;
                                                    }
                                                }
                                            } else if (task.stage ==
                                                       llama_hybrid_cycle_stage::SEND_ACTIVATION) {
                                                valid = llama_hybrid_cycle_receive_wire(
                                                    phone, wire, INFERENCE_CYCLE_READY,
                                                    uint32_t(state.tokens),
                                                    uint32_t(hidden), uint32_t(topk),
                                                    state.recv_ready);
                                                if (valid) {
                                                    const auto & v = state.recv_ready.ready;
                                                    valid = v.tokens == uint32_t(state.tokens) &&
                                                        v.hidden == uint32_t(hidden) &&
                                                        v.topk == uint32_t(topk) &&
                                                        cycle_pc_ring_finite(
                                                            v.residual, expected) &&
                                                        cycle_pc_ring_finite(
                                                            v.ffn_norm, expected) &&
                                                        v.topk_ids.size() ==
                                                            size_t(state.tokens * topk) &&
                                                        cycle_pc_ring_finite(
                                                            v.topk_weights,
                                                            size_t(state.tokens * topk));
                                                }
                                                if (valid) {
                                                    // Send READY ACK only after the
                                                    // matching host RX has decoded it.
                                                    valid = ggml_backend_rpc_cycle_send_ack(
                                                        phone, wire.layer, wire.chunk,
                                                        wire.seq, wire.graph_id,
                                                        INFERENCE_CYCLE_READY, true);
                                                }
                                                if (valid) ++real_p2c_ready_rx;
                                            } else if (task.stage ==
                                                       llama_hybrid_cycle_stage::FFN_SOURCE) {
                                                auto * bank =
                                                    phone_slot_banks.by_index(
                                                        region.location.cycle_index);
                                                valid = bank &&
                                                    state.ready.phone_graph.ffn_norm &&
                                                    state.ready.phone_graph.topk_ids &&
                                                    state.ready.phone_graph.topk_weights;
                                                std::unique_ptr<
                                                    llama_hybrid_cycle_p2c_ffn_slot> slot;
                                                if (valid) {
                                                    // Only PHONE_GPU worker touches
                                                    // the P2C idle bank for this region.
                                                    for (auto it = bank->p2c_idle.begin();
                                                         it != bank->p2c_idle.end(); ++it) {
                                                        if (*it &&
                                                            (*it)->prepared_tokens ==
                                                                state.tokens) {
                                                            slot = std::move(*it);
                                                            bank->p2c_idle.erase(it);
                                                            ++p2c_slot_reuse;
                                                            break;
                                                        }
                                                    }
                                                }
                                                if (valid && !slot) {
                                                    const auto & lw = model->layers[size_t(l)];
                                                    const llama_hybrid_cycle_ffn_shard_weights
                                                        shard {
                                                            llama_hybrid_cycle_resolve_physical_weight(
                                                                lw.ffn_up_exps, phone, 1),
                                                            llama_hybrid_cycle_resolve_physical_weight(
                                                                lw.ffn_gate_exps, phone, 1),
                                                            llama_hybrid_cycle_resolve_physical_weight(
                                                                lw.ffn_down_exps, phone, 1),
                                                        };
                                                    slot = std::make_unique<
                                                        llama_hybrid_cycle_p2c_ffn_slot>();
                                                    slot->rpc_ops.send_partial =
                                                        &ggml_backend_rpc_cycle_send_partial;
                                                    valid = shard.available() &&
                                                        slot->prepare(
                                                            phone, shard, hidden,
                                                            state.tokens, topk,
                                                            model->hparams.n_expert);
                                                }
                                                if (valid) {
                                                    state.p2c_main_ffn = std::move(slot);
                                                    valid = state.p2c_main_ffn->execute_and_send(
                                                        phone, state.ready.phone_graph,
                                                        wire, nullptr);
                                                }
                                                if (!valid) {
                                                    // Failed GPU work may not free its
                                                    // borrowed READY Graph/Scratch early.
                                                    ggml_backend_synchronize(phone);
                                                }
                                                state.ready.release_phone_graph();
                                                if (valid) {
                                                    // send_partial RPC stages its own
                                                    // immutable Phone TX buffer first.
                                                    bank->p2c_idle.push_back(
                                                        std::move(state.p2c_main_ffn));
                                                    ++real_p2c_phone_ffn;
                                                } else {
                                                    state.p2c_main_ffn.reset();
                                                }
                                            } else if (task.stage ==
                                                       llama_hybrid_cycle_stage::FFN_DESTINATION) {
                                                const auto * resource =
                                                    resources.layer(l);
                                                const auto & v =
                                                    state.recv_ready.ready;
                                                valid = resource &&
                                                    llama_hybrid_cycle_execute_p2c_pc_ffn(
                                                        *model, *resource, cpu, l,
                                                        state.tokens, v.ffn_norm,
                                                        v.topk_ids, v.topk_weights,
                                                        state.partial);
                                                if (valid) ++real_p2c_pc_ffn;
                                            } else if (task.stage ==
                                                       llama_hybrid_cycle_stage::SEND_PARTIAL) {
                                                valid = llama_hybrid_cycle_receive_wire(
                                                    phone, wire, INFERENCE_CYCLE_PARTIAL,
                                                    uint32_t(state.tokens),
                                                    uint32_t(hidden), 0,
                                                    state.recv_partial);
                                                if (valid) {
                                                    const auto & v =
                                                        state.recv_partial.partial;
                                                    valid = v.tokens == uint32_t(state.tokens) &&
                                                        v.hidden == uint32_t(hidden) &&
                                                        cycle_pc_ring_finite(
                                                            v.partial, expected);
                                                }
                                                if (valid) ++real_p2c_partial_rx;
                                            } else if (task.stage ==
                                                       llama_hybrid_cycle_stage::REDUCE) {
                                                // One PC Reduce is the host output
                                                // owner. It waits for both partials.
                                                state.host_output.resize(expected);
                                                valid = inference_cycle_reduce_p2c_pc_cpu(
                                                    state.recv_ready.ready,
                                                    state.partial.data(),
                                                    state.partial.size(),
                                                    state.recv_partial.partial,
                                                    state.host_output.data(),
                                                    state.host_output.size()) &&
                                                    cycle_pc_ring_finite(
                                                        state.host_output, expected);
                                                if (valid && trace_stages &&
                                                    cycle_full_reference::stage_focus(l)) {
                                                    // Numerical probe of
                                                    // complete two-sided FFN
                                                    // before residual ADD;
                                                    // PC is the sole Reduce.
                                                    std::vector<float> full_ffn(expected);
                                                    for (size_t i = 0; i < expected; ++i) {
                                                        full_ffn[i] =
                                                            state.partial[i] +
                                                            state.recv_partial.partial.partial[i];
                                                    }
                                                    trace_stage_host(
                                                        l, c, state.start,
                                                        state.tokens, full_ffn,
                                                        "P2C_PC_REDUCE", "FFN_OUTPUT");
                                                }
                                                if (valid && trace_parity) {
                                                    trace_host(
                                                        l, c, state.start,
                                                        state.tokens,
                                                        state.host_output,
                                                        "P2C_PC_REDUCE");
                                                }
                                                if (valid) ++real_p2c_reduce;
                                                if (valid) {
                                                    // Remote FFN TX buffer is ACKed
                                                    // only AFTER PC has read PARTIAL.
                                                    valid = ggml_backend_rpc_cycle_send_ack(
                                                        phone, wire.layer, wire.chunk,
                                                        wire.seq, wire.graph_id,
                                                        INFERENCE_CYCLE_PARTIAL, true);
                                                    if (valid) ++real_p2c_partial_ack;
                                                }
                                                if (valid) {
                                                    valid = session.publish_host_output(
                                                        l, c, hidden);
                                                }
                                                if (valid) {
                                                    ++p2c_completed[size_t(
                                                        region.location.cycle_index)];
                                                    std::printf(
                                                        "[CYCLE_REAL_P2C] layer=%d "
                                                        "chunk=%d seq=%llu "
                                                        "attn=PHONE_GPU "
                                                        "ffn=PHONE_GPU_PLUS_PC_CPU "
                                                        "reduce=PC_CPU "
                                                        "host_output=PUBLISHED "
                                                        "status=PASS\n",
                                                        l, c,
                                                        (unsigned long long)wire.seq);
                                                }
                                            } else {
                                                valid = false;
                                            }
                                            if (!valid) {
                                                std::fprintf(stderr,
                                                    "[CYCLE_REAL_P2C] layer=%d "
                                                    "chunk=%d stage=%s status=FAIL\n",
                                                    l, c,
                                                    llama_hybrid_cycle_stage_name(
                                                        task.stage));
                                            }
                                        } else if (task.stage ==
                                                   llama_hybrid_cycle_stage::ATTN_PRE) {
                                            // Preserve old MOCK P2C and the one
                                            // final diagnostic Phone readback.
                                            if (real_phone_full) {
                                                const auto * producer =
                                                    session.phone_predecessor_output(l, c);
                                                valid = producer &&
                                                    producer->type == GGML_TYPE_F32 &&
                                                    ggml_is_contiguous(producer) &&
                                                    ggml_nelements(producer) ==
                                                        int64_t(expected);
                                                if (valid) {
                                                    target.resize(expected);
                                                    ggml_backend_tensor_get(
                                                        producer, target.data(), 0,
                                                        target.size() * sizeof(float));
                                                    valid = cycle_pc_ring_finite(
                                                        target, expected);
                                                }
                                                if (valid) {
                                                    auto * bank =
                                                        phone_slot_banks.by_index(
                                                            region.location.cycle_index);
                                                    auto old_slot =
                                                        session.retire_phone_output_after_handoff(
                                                            l - 1, l, c);
                                                    valid = bank && old_slot;
                                                    if (valid) {
                                                        bank->full_idle_by_layer[
                                                            size_t(l - 1)].push_back(
                                                                std::move(old_slot));
                                                        ++phone_full_p2c_readbacks;
                                                    }
                                                }
                                            } else {
                                                const auto & prior =
                                                    mock_at(l - 1, c);
                                                valid = cycle_pc_ring_finite(
                                                    prior, expected);
                                                if (valid) target = prior;
                                            }
                                        } else if (task.stage ==
                                                   llama_hybrid_cycle_stage::REDUCE) {
                                            valid = cycle_pc_ring_finite(
                                                target, expected);
                                            if (valid) {
                                                state.host_output = target;
                                                for (float & v : state.host_output) {
                                                    v += 0.001f;
                                                }
                                                valid = session.publish_host_output(
                                                    l, c, hidden);
                                            }
                                            if (valid) {
                                                ++p2c_completed[size_t(
                                                    region.location.cycle_index)];
                                            }
                                        }
                                    } else {
                                        return false;
                                    }
                                    if ((task.stage ==
                                             llama_hybrid_cycle_stage::SEND_ACTIVATION ||
                                         task.stage ==
                                             llama_hybrid_cycle_stage::SEND_PARTIAL) &&
                                        task.queue !=
                                            (kind ==
                                             llama_hybrid_cycle_layer_kind::TENSOR_C2P ?
                                             llama_hybrid_cycle_queue::PC_TO_PHONE :
                                             llama_hybrid_cycle_queue::PHONE_TO_PC)) {
                                        valid = false;
                                    }
                                    if (valid) ++mock_stages;
                                    return valid;
                                }, reason)) {
                            std::fprintf(stderr,
                                "[CYCLE_REAL_PC_RING] status=FAIL "
                                "reason=BIND_MOCK_PHONE detail=%s\n",
                                reason.c_str());
                            ok = false;
                        }

                        llama_hybrid_cycle_chunk_timeline timeline;
                        if (ok && !stages.validate(reason)) {
                            std::fprintf(stderr,
                                "[CYCLE_REAL_PC_RING] status=FAIL "
                                "reason=DISPATCHER detail=%s\n",
                                reason.c_str());
                            ok = false;
                        }
                        if (ok) {
                            std::printf(
                                "[CYCLE_REAL_PC_RING] phase=FORWARD "
                                "layers=48 chunks=%d max_inflight=%d "
                                "pc_full=REAL_CPU c2p_pc=REAL_CPU "
                                "c2p_phone=%s phone_full=%s p2c=%s "
                                "wire=%s source=%s status=BEGIN\n",
                                n_chunks, max_inflight,
                                real_c2p_phone ? "REAL_GPU_ONLY" : "MOCK_ONLY",
                                real_phone_full ? "REAL_GPU_DEVICE_CHAIN" : "MOCK_ONLY",
                                real_p2c ? "REAL_GPU_CPU_SPLIT" : "MOCK_ONLY",
                                real_p2c ? "REAL_BIDIRECTIONAL_RPC" :
                                    (real_c2p_phone ? "REAL_C2P_ONLY" : "MOCK_ONLY"),
                                use_cpu_reference ?
                                    "REAL_TOKEN_EMBEDDING" : "SYNTHETIC_HIDDEN");
                            ok = runtime.run(stages, nullptr, &timeline);
                            if (real_c2p_phone &&
                                !ggml_backend_rpc_cycle_wait_tx_idle(phone, 30000)) {
                                std::fprintf(stderr,
                                    "[CYCLE_REAL_C2P_PHONE] phase=TX_DRAIN "
                                    "status=FAIL\n");
                                ok = false;
                            }
                            if (!ok && real_c2p_phone) {
                                // On failed RPC submission, do not release a
                                // borrowed Graph allocation with work in flight.
                                ggml_backend_synchronize(phone);
                            }
                        }

                        int pc_layers = 0;
                        if (ok) {
                            for (int l = 0; l < 48 && ok; ++l) {
                                if (layout.kinds[size_t(l)] !=
                                    llama_hybrid_cycle_layer_kind::PC_CPU) {
                                    continue;
                                }
                                ++pc_layers;
                                const auto * r = resources.layer(l);
                                ok = r && r->kv &&
                                     r->kv->committed_tokens == n_tokens;
                                for (int c = 0; c < n_chunks && ok; ++c) {
                                    const auto & s = session.at(l, c);
                                    ok = s.host_output_published.load(
                                             std::memory_order_acquire) &&
                                         cycle_pc_ring_finite(s.host_output,
                                             size_t(s.tokens) * size_t(hidden));
                                }
                                std::printf(
                                    "[CYCLE_REAL_PC_RING_KV] layer=%d "
                                    "committed=%lld expected=%d status=%s\n",
                                    l,
                                    (long long)(r && r->kv ?
                                        r->kv->committed_tokens : -1),
                                    n_tokens, ok ? "PASS" : "FAIL");
                            }
                        }
                        int c2p_pc_layers = 0;
                        if (ok) {
                            for (int l = 4; l < 48 && ok; l += 8) {
                                const auto * r = resources.layer(l);
                                ok = r && r->kv &&
                                     r->kv->committed_tokens == n_tokens;
                                for (int c = 0; c < n_chunks && ok; ++c) {
                                    const auto & s = session.at(l, c);
                                    ok = cycle_pc_ring_finite(
                                        s.partial,
                                        size_t(s.tokens) * size_t(hidden));
                                }
                                if (ok) ++c2p_pc_layers;
                                std::printf(
                                    "[CYCLE_REAL_C2P_PC_KV] layer=%d "
                                    "committed=%lld expected=%d status=%s\n",
                                    l,
                                    (long long)(r && r->kv ?
                                        r->kv->committed_tokens : -1),
                                    n_tokens, ok ? "PASS" : "FAIL");
                            }
                        }
                        // Check physical Phone KV and Device Output
                        // ownership after the full FIFO has joined. Every
                        // C2P producer must have been consumed by L5, each
                        // L5 producer by L6, and every L6 producer by the
                        // explicit P2C diagnostic bridge. No active graph
                        // slot may remain attached to a completed Chunk.
                        int phone_full_kv_layers = 0;
                        if (ok && real_phone_full) {
                            for (int cycle = 0; cycle < 6 && ok; ++cycle) {
                                const int c2p = cycle * 8 + 4;
                                 // Do not check only the last Chunk: a missing
                                // earlier ownership ACK could silently leak
                                // a GPU slot despite a finite output.
                                for (int c = 0; c < n_chunks && ok; ++c) {
                                    ok = session.at(c2p, c).phone_output == nullptr;
                                }
                                for (int l = c2p + 1; l <= c2p + 2 && ok; ++l) {
                                    const auto * r = resources.layer(l);
                                    ok = r && r->kv &&
                                         r->kv->committed_tokens == n_tokens;
                                    for (int c = 0; c < n_chunks && ok; ++c) {
                                        const auto & st = session.at(l, c);
                                        ok = st.phone_output == nullptr &&
                                             st.ready.phone_compute_buffer == nullptr;
                                    }
                                    if (ok) ++phone_full_kv_layers;
                                    std::printf(
                                        "[CYCLE_REAL_PHONE_FULL_KV] "
                                        "layer=%d committed=%lld expected=%d "
                                        "output_slot=DRAINED status=%s\n",
                                        l,
                                        (long long)(r && r->kv ?
                                            r->kv->committed_tokens : -1),
                                        n_tokens, ok ? "PASS" : "FAIL");
                                }
                            }
                        }
                        // The six physical P2C KV prefixes and published
                        // host outputs are checked after all workers join.
                        // There must be no retained READY Graph or active FFN.
                        int p2c_kv_layers = 0;
                        if (ok && real_p2c) {
                            for (int l = 7; l < 48 && ok; l += 8) {
                                const auto * r = resources.layer(l);
                                ok = r && r->kv &&
                                     r->kv->committed_tokens == n_tokens;
                                for (int c = 0; c < n_chunks && ok; ++c) {
                                    const auto & st = session.at(l, c);
                                    ok = st.ready.phone_compute_buffer == nullptr &&
                                         st.p2c_main_ffn == nullptr &&
                                         st.host_output_published.load(
                                             std::memory_order_acquire) &&
                                         cycle_pc_ring_finite(
                                             st.host_output,
                                             size_t(st.tokens) * size_t(hidden));
                                }
                                if (ok) ++p2c_kv_layers;
                                std::printf(
                                    "[CYCLE_REAL_P2C_KV] layer=%d "
                                    "committed=%lld expected=%d "
                                    "graph_slot=DRAINED host_output=PUBLISHED "
                                    "status=%s\n",
                                    l,
                                    (long long)(r && r->kv ?
                                        r->kv->committed_tokens : -1),
                                    n_tokens, ok ? "PASS" : "FAIL");
                            }
                        }
                        if (ok) {
                            ok = pc_layers == 24 &&
                                 c2p_pc_layers == 6 &&
                                 (!real_p2c || p2c_kv_layers == 6) &&
                                 (!real_phone_full ||
                                  phone_full_kv_layers == 12) &&
                                 real_c2p_attn.load() == 6 * n_chunks &&
                                 real_c2p_ffn.load() == 6 * n_chunks &&
                                 mock_stages.load() == 6 * n_chunks * 14 &&
                                 (!real_c2p_phone ||
                                  (phone_c2p_ready.load() == 6 * n_chunks &&
                                   phone_c2p_ffn.load() == 6 * n_chunks &&
                                   phone_c2p_partial.load() == 6 * n_chunks &&
                                   phone_c2p_reduce.load() == 6 * n_chunks)) &&
                                 (!real_phone_full ||
                                  (real_phone_full_calls.load() == 12 * n_chunks &&
                                   c2p_device_handoffs.load() == 6 * n_chunks &&
                                   phone_full_device_handoffs.load() == 6 * n_chunks &&
                                   (real_p2c ?
                                        (phone_full_p2c_readbacks.load() == 0 &&
                                         p2c_device_handoffs.load() == 6 * n_chunks) :
                                        phone_full_p2c_readbacks.load() == 6 * n_chunks))) &&
                                 (!real_p2c ||
                                  (real_p2c_attn.load() == 6 * n_chunks &&
                                   real_p2c_ready_rx.load() == 6 * n_chunks &&
                                   real_p2c_phone_ffn.load() == 6 * n_chunks &&
                                   real_p2c_pc_ffn.load() == 6 * n_chunks &&
                                   real_p2c_partial_rx.load() == 6 * n_chunks &&
                                   real_p2c_reduce.load() == 6 * n_chunks &&
                                   real_p2c_partial_ack.load() == 6 * n_chunks));
                            for (int cycle = 0; cycle < 6 && ok; ++cycle) {
                                ok = p2c_completed[size_t(cycle)].load() ==
                                    n_chunks;
                                for (int c = 0; c < n_chunks && ok; ++c) {
                                    const auto & s = session.at(8 * cycle + 7, c);
                                    ok = s.host_output_published.load(
                                             std::memory_order_acquire) &&
                                         cycle_pc_ring_finite(s.host_output,
                                             size_t(s.tokens) * size_t(hidden));
                                }
                            }
                        }

                        // The next PC Chunk may begin as soon as this Chunk's
                        // P2C output is published. Compare timestamps on
                        // the SAME host steady_clock after all workers join;
                        // report early overlap without requiring it (CPU
                        // queue scheduling can legitimately delay execution).
                        int earlier_than_previous_last = 0;
                        if (ok) {
                            for (int cycle = 1; cycle < 6 && ok; ++cycle) {
                                const int base = cycle * 8;
                                const auto * previous_last = cycle_pc_ring_find(
                                    timeline, base - 1, n_chunks - 1,
                                    llama_hybrid_cycle_stage::REDUCE);
                                for (int c = 0; c < n_chunks && ok; ++c) {
                                    const auto * previous = cycle_pc_ring_find(
                                        timeline, base - 1, c,
                                        llama_hybrid_cycle_stage::REDUCE);
                                    const auto * next = cycle_pc_ring_find(
                                        timeline, base, c,
                                        llama_hybrid_cycle_stage::FULL_LAYER);
                                    ok = previous && next &&
                                         previous->success && next->success &&
                                         next->start >= previous->finish;
                                    if (ok && c == 0 && previous_last &&
                                        next->start < previous_last->finish) {
                                        ++earlier_than_previous_last;
                                    }
                                }
                                std::printf(
                                    "[CYCLE_REAL_PC_RING_HANDOFF] "
                                    "p2c=%d pc_next=%d causal_order=%s\n",
                                    base - 1, base, ok ? "PASS" : "FAIL");
                            }
                        }

                        double sum = 0.0;
                        double max_abs = 0.0;
                        if (ok) {
                            for (int c = 0; c < n_chunks; ++c) {
                                const auto & out = session.at(47, c);
                                for (float v : out.host_output) {
                                    sum += v;
                                    max_abs = std::max(max_abs,
                                        std::fabs(double(v)));
                                }
                            }
                            ok = std::isfinite(sum) && std::isfinite(max_abs) &&
                                 max_abs > 0.0;
                        }
                        if (trace_parity && ok) {
                            const int samples = trace_seen.load(
                                std::memory_order_relaxed);
                            const int failed = trace_failed.load(
                                std::memory_order_relaxed);
                            const int first_bad = trace_first_bad.load(
                                std::memory_order_relaxed);
                            std::printf(
                                "[CYCLE_LAYER_PARITY_SUM] "
                                "layers=48 chunks=%d "
                                "compared=%d expected=%d "
                                "strict_failed=%d first_failed_layer=%d "
                                "gpu_readback=DIAGNOSTIC_ONLY "
                                "status=%s\n",
                                n_chunks, samples, 48 * n_chunks,
                                failed, first_bad == 48 ? -1 : first_bad,
                                samples == 48 * n_chunks ? "COMPLETE" : "FAIL");
                            if (samples != 48 * n_chunks) {
                                ok = false;
                            }
                        }

                        // Diagnostic A/B replays independent physical Phone
                        // Full layers (5,46) using the oracle's EXACT L4/L45
                        // input. This excludes all upstream layer drift.
                        // Run ONLY after all global FIFO workers/transport
                        // have joined; no physical KV/FFN slot is borrowed.
                        // Numerical mismatch is informational here. Normal
                        // six-period strict parity remains the sole pass
                        // criterion, but execution failure is fatal.
                        if (ok && same_input_ab) {
                            const auto ab = cycle_same_input::run(
                                *model, phone, reference, n_tokens);
                            if (!ab.execution_ok) ok = false;
                        }

                        // Six independent physical boundaries. The CPU
                        // reference is a NORMAL llama_decode full graph,
                        // not the same custom Attention/FFN graph run on CPU.
                        // This optional check adds NO GPU readbacks: all six
                        // P2C outputs are already immutable PC-host tensors.
                        bool parity_ok = true;
                        int parity_checked = 0;
                        if (ok && use_cpu_reference) {
                            for (int period = 0; period < 6; ++period) {
                                const int layer = 8 * period + 7;
                                std::vector<float> actual(total);
                                for (int c = 0; c < n_chunks; ++c) {
                                    const auto & st = session.at(layer, c);
                                    const size_t offset =
                                        size_t(st.start) * size_t(hidden);
                                    std::copy(
                                        st.host_output.begin(),
                                        st.host_output.end(),
                                        actual.begin() + offset);
                                }
                                const bool passed =
                                    cycle_full_reference::compare_period(
                                        actual,
                                        reference.after_period[size_t(period)],
                                        layer, n_tokens, chunk_tokens, hidden);
                                parity_ok = parity_ok && passed;
                                ++parity_checked;
                            }
                            std::printf(
                                "[CYCLE_FULL_REFERENCE_SUM] periods=%d "
                                "expected=6 all_strict=%s "
                                "cpu_model=RELEASED_BEFORE_HYBRID "
                                "status=%s\n",
                                parity_checked,
                                parity_ok ? "PASS" : "FAIL",
                                parity_ok && parity_checked == 6 ?
                                    "PASS" : "FAIL");
                            ok = ok && parity_ok && parity_checked == 6;
                        }
                        std::printf(
                            "[CYCLE_REAL_PC_RING] layers=48 "
                            "pc_full_layers=%d pc_full=REAL_CPU "
                            "c2p_pc_layers=%d c2p_pc=REAL_CPU "
                            "c2p_phone=%s phone_rpc_compute=%s "
                            "phone_full=%s p2c=%s "
                            "tokens=%d chunks=%d max_inflight=%d "
                            "real_c2p_attn_calls=%d real_c2p_ffn_calls=%d "
                            "real_c2p_ready_acks=%d real_c2p_phone_ffn=%d "
                            "real_c2p_partial_applies=%d real_c2p_reduces=%d "
                            "real_phone_full_calls=%d "
                            "c2p_device_handoffs=%d phone_full_device_handoffs=%d "
                            "phone_full_p2c_readbacks=%d p2c_device_handoffs=%d "
                            "real_p2c_attn_calls=%d real_p2c_ready_rx=%d "
                            "real_p2c_phone_ffn=%d real_p2c_pc_ffn=%d "
                            "real_p2c_partial_rx=%d real_p2c_reduce=%d "
                            "real_p2c_partial_acks=%d "
                            "c2p_slot_reuse=%d phone_full_slot_reuse=%d "
                            "p2c_slot_reuse=%d "
                            "region_stage_calls=%d expected_region_stage_calls=%d "
                            "cross_period_early=%d "
                            "full_48_forward=%s source=%s strict_parity=%s "
                            "logits=NOT_RUN sum=%.9g max_abs=%.9g status=%s\n",
                            pc_layers, c2p_pc_layers,
                            real_c2p_phone ? (real_phone_full ?
                                "REAL_GPU_DEVICE_CHAIN" :
                                "REAL_GPU_WITH_HOST_BRIDGE") : "MOCK_ONLY",
                            real_p2c ? "C2P_AND_P2C" :
                                (real_c2p_phone ? "C2P_ONLY" : "NOT_RUN"),
                            real_phone_full ? "REAL_GPU_DEVICE_CHAIN" : "MOCK_ONLY",
                            real_p2c ? "REAL_GPU_CPU_SPLIT" : "MOCK_ONLY",
                            n_tokens, n_chunks, max_inflight,
                            real_c2p_attn.load(), real_c2p_ffn.load(),
                            phone_c2p_ready.load(), phone_c2p_ffn.load(),
                            phone_c2p_partial.load(), phone_c2p_reduce.load(),
                            real_phone_full_calls.load(),
                            c2p_device_handoffs.load(),
                            phone_full_device_handoffs.load(),
                            phone_full_p2c_readbacks.load(),
                            p2c_device_handoffs.load(),
                            real_p2c_attn.load(), real_p2c_ready_rx.load(),
                            real_p2c_phone_ffn.load(), real_p2c_pc_ffn.load(),
                            real_p2c_partial_rx.load(), real_p2c_reduce.load(),
                            real_p2c_partial_ack.load(),
                            c2p_slot_reuse.load(), phone_full_slot_reuse.load(),
                            p2c_slot_reuse.load(),
                            mock_stages.load(), 6 * n_chunks * 14,
                            earlier_than_previous_last,
                            real_p2c ?
                                (use_cpu_reference ? "REAL_TOKEN_EMBEDDING" :
                                    "REAL_SYNTHETIC_INPUT_UNVALIDATED") :
                                "NOT_RUN",
                            use_cpu_reference ?
                                "CPU_REFERENCE_EMBEDDING" : "SYNTHETIC_HIDDEN",
                            !use_cpu_reference ? "NOT_RUN" :
                                parity_checked != 6 ? "NOT_CHECKED" :
                                    parity_ok ? "PASS" : "FAIL",
                            sum, max_abs, ok ? "PASS" : "FAIL");
                    }
                }
                // All workers are joined and Session has been destroyed.
                // Release the GPU/RPC KV resources before the model/backend.
                resources.clear();
            }
        }
    }
    ggml_backend_free(phone);
    ggml_backend_free(cpu);
    return ok ? 0 : 1;
}
