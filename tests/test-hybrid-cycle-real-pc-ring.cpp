// Opt-in mixed Cycle probe: real PC Full/C2P and optionally real Phone C2P.
//
// All 48 layers share ONE five-worker FIFO. The 24 PC Full layers execute
// real Qwen3-MoE Attention/RoPE/Router/FFN on their physical CPU weights,
// using physical persistent KV and actual Session host handoffs.
// C2P PC Attention and its FFN_SOURCE shard ALSO execute real CPU weights.
// LLAMA_HYBRID_CYCLE_REAL_C2P_PHONE=1 enables legacy V5 REAL C2P GPU.
// LLAMA_HYBRID_CYCLE_REAL_PHONE_FULL=1 additionally executes all 12
// Phone Full layers using direct GPU l_out handoffs (requires REAL C2P).
// In that mode the ONLY diagnostic GPU readback occurs at the still-MOCK
// P2C Attention boundary. No period barrier or Meta re-execution.
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
#include "llama-hybrid-cycle-phone-regions.h"
#include "llama-hybrid-cycle-wire.h"

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

    llama_backend_init();
    const auto cpu_dev = ggml_backend_dev_by_type(
        GGML_BACKEND_DEVICE_TYPE_CPU);
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
                                initial_hidden, hidden, reason)) {
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
                                                valid = llama_hybrid_cycle_execute_phone_full(
                                                    *model, *r, phone, l,
                                                    state.start, state.tokens,
                                                    predecessor, state.ready,
                                                    *full_slot);
                                            }
                                            if (valid) {
                                                state.phone_output = std::move(full_slot);
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
                                        if (task.stage ==
                                            llama_hybrid_cycle_stage::ATTN_PRE) {
                                            if (real_phone_full) {
                                                // P2C Attention is still MOCK.
                                                // Read back ONLY the final Phone Full
                                                // l_out once at this diagnostic
                                                // boundary; earlier C2P/L5 handoffs
                                                // stayed entirely on Phone GPU.
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
                                                    auto retired =
                                                        session.retire_phone_output_after_handoff(
                                                            l - 1, l, c);
                                                    valid = bank && retired;
                                                    if (valid) {
                                                        bank->full_idle_by_layer[
                                                            size_t(l - 1)].push_back(
                                                                std::move(retired));
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
                                                auto & output = session.at(l, c);
                                                output.host_output = target;
                                                for (float & v : output.host_output) {
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
                                "c2p_phone=%s phone_full=%s p2c=MOCK_ONLY "
                                "wire=%s source=SYNTHETIC_HIDDEN status=BEGIN\n",
                                n_chunks, max_inflight,
                                real_c2p_phone ? "REAL_GPU_ONLY" : "MOCK_ONLY",
                                real_phone_full ? "REAL_GPU_DEVICE_CHAIN" : "MOCK_ONLY",
                                real_c2p_phone ? "REAL_C2P_ONLY" : "MOCK_ONLY");
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
                        if (ok) {
                            ok = pc_layers == 24 &&
                                 c2p_pc_layers == 6 &&
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
                                   phone_full_p2c_readbacks.load() == 6 * n_chunks));
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
                        std::printf(
                            "[CYCLE_REAL_PC_RING] layers=48 "
                            "pc_full_layers=%d pc_full=REAL_CPU "
                            "c2p_pc_layers=%d c2p_pc=REAL_CPU "
                            "c2p_phone=%s phone_rpc_compute=%s "
                            "phone_full=%s p2c=MOCK_ONLY "
                            "tokens=%d chunks=%d max_inflight=%d "
                            "real_c2p_attn_calls=%d real_c2p_ffn_calls=%d "
                            "real_c2p_ready_acks=%d real_c2p_phone_ffn=%d "
                            "real_c2p_partial_applies=%d real_c2p_reduces=%d "
                            "real_phone_full_calls=%d "
                            "c2p_device_handoffs=%d phone_full_device_handoffs=%d "
                            "phone_full_p2c_readbacks=%d "
                            "c2p_slot_reuse=%d phone_full_slot_reuse=%d "
                            "region_stage_calls=%d expected_region_stage_calls=%d "
                            "cross_period_early=%d "
                            "full_48_forward=NOT_RUN logits=NOT_RUN "
                            "sum=%.9g max_abs=%.9g status=%s\n",
                            pc_layers, c2p_pc_layers,
                            real_c2p_phone ? (real_phone_full ?
                                "REAL_GPU_DEVICE_CHAIN" :
                                "REAL_GPU_WITH_HOST_BRIDGE") : "MOCK_ONLY",
                            real_c2p_phone ? "C2P_ONLY" : "NOT_RUN",
                            real_phone_full ? "REAL_GPU_DEVICE_CHAIN" : "MOCK_ONLY",
                            n_tokens, n_chunks, max_inflight,
                            real_c2p_attn.load(), real_c2p_ffn.load(),
                            phone_c2p_ready.load(), phone_c2p_ffn.load(),
                            phone_c2p_partial.load(), phone_c2p_reduce.load(),
                            real_phone_full_calls.load(),
                            c2p_device_handoffs.load(),
                            phone_full_device_handoffs.load(),
                            phone_full_p2c_readbacks.load(),
                            c2p_slot_reuse.load(), phone_full_slot_reuse.load(),
                            mock_stages.load(), 6 * n_chunks * 14,
                            earlier_than_previous_last,
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
