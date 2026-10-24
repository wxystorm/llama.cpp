#include "llama-hybrid-cycle-forward.h"
#include "llama-hybrid-cycle-log.h"

#include "llama-model.h"
#include "llama-hybrid-cycle-pc-full.h"
#include "llama-hybrid-cycle-c2p-pc.h"
#include "llama-hybrid-cycle-phone-full.h"
#include "llama-hybrid-cycle-p2c-real.h"
#include "llama-hybrid-cycle-phone-regions.h"
#include "llama-hybrid-cycle-wire.h"

#include "../ggml/include/ggml-rpc.h"
#include "../ggml/src/ggml-rpc/inference-cycle-reduce.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

static bool cycle_core_finite(const std::vector<float> & v, size_t expect) {
    if (v.size() != expect) return false;
    for (float x : v) if (!std::isfinite(x)) return false;
    return true;
}

// Keep the validated standalone stage body but give the core its own
// private helper name; the test program does not participate in this API.
#define cycle_pc_ring_finite cycle_core_finite

static bool cycle_core_env_default(const char * key, const char * value) {
    if (std::getenv(key)) return true;
#ifdef _WIN32
    return _putenv_s(key, value) == 0;
#else
    return setenv(key, value, 0) == 0;
#endif
}

static bool cycle_core_evaluate_with_kv(
        const llama_model & loaded_model,
        ggml_backend_t pc_cpu,
        ggml_backend_t phone_rpc_gpu,
        llama_hybrid_cycle_physical_resource_pool & resources,
        int64_t position_base,
        uint64_t wire_epoch,
        const std::vector<float> & token_embeddings,
        int n_tokens,
        std::vector<float> & final_hidden,
        std::string & reason,
        llama_hybrid_cycle_forward_metrics * metrics,
        int chunk_tokens,
        int max_inflight) {
    reason.clear();
    final_hidden.clear();
    if (metrics) *metrics = {};
    const llama_model * model = &loaded_model;
    const ggml_backend_t cpu = pc_cpu;
    const ggml_backend_t phone = phone_rpc_gpu;
    const int64_t hidden = model->hparams.n_embd;
    if (!cpu || !phone || !ggml_backend_is_rpc(phone) ||
        model->arch != LLM_ARCH_QWEN3MOE ||
        llama_model_n_layer(model) != 48 ||
        hidden <= 0 || n_tokens < 1 || n_tokens > 512 ||
        position_base < 0 || position_base > 512 - n_tokens ||
        chunk_tokens < 1 || chunk_tokens > 64 ||
        max_inflight < 1 || max_inflight > 2 ||
        n_tokens > std::numeric_limits<int>::max() - chunk_tokens + 1 ||
        token_embeddings.size() != size_t(n_tokens) * size_t(hidden)) {
        reason = "UNSUPPORTED_PREFILL_SHAPE_MODEL_OR_BACKENDS";
        return false;
    }
    for (float x : token_embeddings) {
        if (!std::isfinite(x)) {
            reason = "NONFINITE_TOKEN_EMBEDDING";
            return false;
        }
    }
    const auto & loaded_layout = resources.execution_layout();
    if (!loaded_layout.enabled() ||
        loaded_layout.kinds.size() != 48 ||
        loaded_layout.x != 4 || loaded_layout.y != 2 ||
        loaded_layout.cycle_size != 8 || loaded_layout.cycles != 6 ||
        resources.prepared_layers() != 48) {
        reason = "PHYSICAL_48_LAYOUT_NOT_PREPARED";
        return false;
    }
    // A new invocation can only append to an EXACT physical KV prefix.
    // No stale P2C/Phone handoff or previous graph may be recomputed.
    for (int l = 0; l < 48; ++l) {
        const auto * r = resources.layer(l);
        if (!r || !r->kv ||
            r->kv->max_tokens < position_base + n_tokens ||
            r->kv->committed_tokens != position_base) {
            reason = "PHYSICAL_KV_PREFIX_NOT_PREPARED";
            return false;
        }
    }
    const int n_chunks = (n_tokens + chunk_tokens - 1) / chunk_tokens;
    if (max_inflight > n_chunks) max_inflight = n_chunks;
    bool ok = true;
    {
        const auto & layout = resources.execution_layout();
        llama_hybrid_cycle_phone_regions phone_regions;
        ok = phone_regions.prepare(layout, n_tokens, chunk_tokens,
                                   max_inflight, reason) &&
             phone_regions.validate_resources(resources, reason) &&
             phone_regions.region_count() == 6;
        if (!ok && reason.empty()) reason = "REAL_PHONE_REGION_NOT_PREPARED";
        auto * dev = ggml_backend_get_device(phone);
        auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        const auto open = reg ? reinterpret_cast<
            ggml_backend_rpc_inference_transport_open_t>(
                ggml_backend_reg_get_proc_address(
                    reg, GGML_BACKEND_RPC_INFERENCE_TRANSPORT_OPEN_PROC)) :
            nullptr;
        if (ok) {
            ok = cycle_core_env_default("GGML_INFERENCE_TRANSPORT_SLOTS","2") &&
                 cycle_core_env_default(
                     "GGML_INFERENCE_TRANSPORT_MAX_PAYLOAD_MIB","2") &&
                 open && open(phone);
            if (!ok) reason = "REAL_RPC_TRANSPORT_OPEN";
        }
        if (ok) {
            llama_hybrid_cycle_execution_session_config sc;
            sc.layout = layout;
            sc.physical_layer_count = 48;
            sc.tokens = n_tokens;
            sc.chunk_tokens = chunk_tokens;
            sc.token_position_base = position_base;
            sc.wire_epoch = wire_epoch;
            llama_hybrid_cycle_execution_session<> session(sc);
            for (int layer = 0; layer < 48; ++layer) {
                const auto * r = resources.layer(layer);
                if (!r || !r->kv ||
                    !session.borrow_kv(layer,r->kv.get())) {
                    reason = "REAL_PHYSICAL_KV_BORROW_FAILED";
                    ok = false;
                    break;
                }
            }
            // This core has no mock, CPU oracle, or diagnostic GPU readbacks.
            // The stage binding is lifted from the successful 48-layer real
            // test; the real options are fixed TRUE here.
            const bool real_c2p_phone = true;
            const bool real_phone_full = true;
            const bool real_p2c = true;
            const bool trace_parity = false;
            const bool trace_stages = false;
            const auto trace_gpu = [](int,int,int64_t,int64_t,
                                      const ggml_tensor *,const char *) {};
            const auto trace_host = [](int,int,int64_t,int64_t,
                                       const std::vector<float> &,
                                       const char *) {};
            const auto trace_stage_host = [](int,int,int64_t,int64_t,
                const std::vector<float> &,const char *,const char *) {};
            const auto trace_stage_gpu = [](int,int,int64_t,int64_t,
                const ggml_tensor *,const char *,const char *) {};
            std::function<void(int,int,int64_t,int64_t,
                               const std::vector<float> &)> pc_trace;
            const std::vector<float> & initial_hidden = token_embeddings;
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
                                                    false) {
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
                                                    if (llama_hybrid_cycle_verbose()) std::printf(
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
                                                    false) {
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
                                                    false) {
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
                                                    if (llama_hybrid_cycle_verbose()) std::printf(
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

            const bool collect_queue_profile = llama_hybrid_cycle_queue_profile_enabled();
            llama_hybrid_cycle_chunk_timeline timeline;
            llama_hybrid_cycle_fifo_profile queue_profile;
            if (ok && !stages.validate(reason)) ok = false;
            if (ok) {
                if (llama_hybrid_cycle_verbose()) std::printf(
                    "[CYCLE_CORE_FORWARD] layers=48 tokens=%d chunks=%d "
                    "window=%d input=REAL_TOKEN_EMBEDDING "
                    "phone=REAL_GPU rpc=REAL_BIDIRECTIONAL "
                    "pc_reduce=EXCLUSIVE status=BEGIN\n",
                    n_tokens, n_chunks, max_inflight);
                const auto fifo_start = std::chrono::steady_clock::now();
                ok = runtime.run(stages,
                    collect_queue_profile ? &queue_profile : nullptr,
                    collect_queue_profile ? &timeline : nullptr);
                if (collect_queue_profile) {
                    const auto fifo_done = std::chrono::steady_clock::now();
                    std::printf(
                        "[CYCLE_CORE_FIFO_WALL] tokens=%d chunks=%d "
                        "window=%d wall_ms=%.3f scope=PC_STEADY "
                        "status=%s\n",
                        n_tokens, n_chunks, max_inflight,
                        std::chrono::duration<double, std::milli>(
                            fifo_done - fifo_start).count(),
                        ok ? "PASS" : "FAIL");
                    llama_hybrid_cycle_print_stage_summary(timeline);
                    for (size_t q = 0; q < queue_profile.queues.size(); ++q) {
                        const auto & v = queue_profile.queues[q];
                        std::printf("[CYCLE_CORE_QUEUE] queue=%s tasks=%llu "
                                    "execute_ms=%.3f max_execute_ms=%.3f "
                                    "queue_wait_ms=%.3f max_queue_wait_ms=%.3f "
                                    "idle_wait_ms=%.3f idle_wakes=%llu "
                                    "scope=HOST_CALLBACK status=MEASURED\n",
                                    llama_hybrid_cycle_queue_name(
                                        static_cast<llama_hybrid_cycle_queue>(q)),
                                    (unsigned long long) v.tasks, v.execute_ms,
                                    v.max_execute_ms, v.queue_wait_ms,
                                    v.max_queue_wait_ms, v.idle_wait_ms,
                                    (unsigned long long) v.idle_wakes);
                    }
                }
            }
            if (ok) {
                const auto drain_begin = std::chrono::steady_clock::now();
                ok = ggml_backend_rpc_cycle_wait_tx_idle(phone, 30000);
                if (collect_queue_profile) {
                    const double drain_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - drain_begin).count();
                    std::printf(
                        "[CYCLE_CORE_RPC_DRAIN] tokens=%d wait_ms=%.3f "
                        "scope=HOST_AFTER_FIFO status=%s\n",
                        n_tokens, drain_ms, ok ? "PASS" : "FAIL");
                }
                if (!ok) reason = "REAL_RPC_TX_NOT_DRAINED";
            }
            if (!ok) ggml_backend_synchronize(phone);
            if (ok && !session.kv_complete(position_base + n_tokens)) {
                reason = "REAL_PHYSICAL_KV_INCOMPLETE";
                ok = false;
            }
            const bool expected_stages =
                real_c2p_attn.load() == 6*n_chunks &&
                real_c2p_ffn.load() == 6*n_chunks &&
                phone_c2p_ready.load() == 6*n_chunks &&
                phone_c2p_ffn.load() == 6*n_chunks &&
                phone_c2p_partial.load() == 6*n_chunks &&
                phone_c2p_reduce.load() == 6*n_chunks &&
                real_phone_full_calls.load() == 12*n_chunks &&
                c2p_device_handoffs.load() == 6*n_chunks &&
                phone_full_device_handoffs.load() == 6*n_chunks &&
                phone_full_p2c_readbacks.load() == 0 &&
                real_p2c_attn.load() == 6*n_chunks &&
                real_p2c_ready_rx.load() == 6*n_chunks &&
                real_p2c_phone_ffn.load() == 6*n_chunks &&
                real_p2c_pc_ffn.load() == 6*n_chunks &&
                real_p2c_partial_rx.load() == 6*n_chunks &&
                real_p2c_reduce.load() == 6*n_chunks &&
                real_p2c_partial_ack.load() == 6*n_chunks &&
                p2c_device_handoffs.load() == 6*n_chunks;
            if (ok && !expected_stages) {
                reason = "REAL_PHYSICAL_STAGES_INCOMPLETE";
                ok = false;
            }
            if (ok) {
                final_hidden.assign(
                    size_t(n_tokens) * size_t(hidden), 0.0f);
                for (int c = 0; c < n_chunks; ++c) {
                    const auto & output = session.at(47,c);
                    const size_t expected =
                        size_t(output.tokens) * size_t(hidden);
                    if (!output.host_output_published.load(
                            std::memory_order_acquire) ||
                        !cycle_core_finite(output.host_output,expected)) {
                        reason = "REAL_P2C_LAST_OUTPUT_NOT_PUBLISHED";
                        ok = false;
                        break;
                    }
                    std::copy(
                        output.host_output.begin(),output.host_output.end(),
                        final_hidden.begin() +
                            size_t(output.start - position_base)*size_t(hidden));
                }
            }
            if (ok) {
                // No C2P or Full output can still be borrowed after the
                // GPU-local successor consumed it. P2C READY and FFN scratch
                // were released before publishing the final host result.
                for (int l = 0; l < 48 && ok; ++l) {
                    for (int c = 0; c < n_chunks && ok; ++c) {
                        const auto & st = session.at(l,c);
                        if (st.phone_output ||
                            st.ready.phone_compute_buffer ||
                            st.p2c_main_ffn) {
                            reason = "REAL_DEVICE_SLOT_NOT_RETIRED";
                            ok = false;
                        }
                    }
                }
            }
            if (metrics) {
                metrics->n_tokens = n_tokens;
                metrics->n_chunks = n_chunks;
                metrics->completed_c2p = phone_c2p_reduce.load();
                metrics->completed_phone_full = real_phone_full_calls.load();
                metrics->completed_p2c = real_p2c_reduce.load();
                metrics->p2c_ready_received = real_p2c_ready_rx.load();
                metrics->p2c_partial_received = real_p2c_partial_rx.load();
                metrics->p2c_reduce_ack = real_p2c_partial_ack.load();
                metrics->all_kv_complete = session.kv_complete(position_base + n_tokens);
                metrics->all_host_outputs_published = !final_hidden.empty();
                metrics->all_phone_slots_retired = ok;
            }
        } // all Stage handlers, Session & Phone slots destroyed here.
    }
    // The owning session retains physical K/V after the per-append FIFO
    // workers and their temporary Graph Slots have been destroyed.
    if (!ok) {
        final_hidden.clear();
        if (reason.empty()) reason = "REAL_FIFO_OR_DEPENDENCY_FAILED";
    }
    if (!ok || llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_CORE_FORWARD] tokens=%d chunks=%d hidden=%zu "
        "physical_layers=48 full_forward=%s "
        "numeric_reference=NOT_RUN logits=NOT_RUN status=%s "
        "reason=%s\n",
        n_tokens, n_chunks, final_hidden.size(),
        ok ? "REAL_PHYSICAL" : "FAILED",
        ok ? "PASS" : "FAIL",
        ok ? "NONE" : reason.c_str());
    return ok;
}

#undef cycle_pc_ring_finite

bool llama_hybrid_cycle_forward_prefill(
        const llama_model & model, ggml_backend_t pc_cpu,
        ggml_backend_t phone_rpc_gpu,
        const std::vector<float> & token_embeddings, int n_tokens,
        std::vector<float> & final_hidden, std::string & reason,
        llama_hybrid_cycle_forward_metrics * metrics,
        int chunk_tokens, int max_inflight) {
    llama_hybrid_cycle_session session;
    if (!session.prepare(model, pc_cpu, phone_rpc_gpu, n_tokens,
                         reason, chunk_tokens, max_inflight)) {
        final_hidden.clear();
        return false;
    }
    return session.evaluate(token_embeddings, n_tokens,
                            final_hidden, reason, metrics);
}

llama_hybrid_cycle_session::llama_hybrid_cycle_session() = default;
llama_hybrid_cycle_session::~llama_hybrid_cycle_session() {
    reset();
}

bool llama_hybrid_cycle_session::prepare(
        const llama_model & model, ggml_backend_t cpu,
        ggml_backend_t phone, int64_t max_total_tokens,
        std::string & reason, int chunk_tokens, int max_inflight,
        ggml_backend_t pc_cuda, size_t cuda_weight_budget_bytes) {
    reset();
    reason.clear();
    if (!cpu || !phone ||
        model.arch != LLM_ARCH_QWEN3MOE ||
        llama_model_n_layer(&model) != 48 ||
        !ggml_backend_is_rpc(phone) ||
        max_total_tokens < 1 || max_total_tokens > 512 ||
        chunk_tokens < 1 || chunk_tokens > 64 ||
        max_inflight < 1 || max_inflight > 2 ||
        (cuda_weight_budget_bytes != 0 && !pc_cuda)) {
        reason = "INVALID_CYCLE_SESSION_CONFIG";
        return false;
    }
    llama_hybrid_cycle_layout layout;
    if (!llama_hybrid_cycle_layout_from_env(48, layout) ||
        layout.x != 4 || layout.y != 2 ||
        layout.cycle_size != 8 || layout.cycles != 6) {
        reason = "EXPECTED_REAL_48_LAYER_X4_Y2_LAYOUT";
        return false;
    }
    auto pool = std::make_unique<llama_hybrid_cycle_physical_resource_pool>();
    llama_hybrid_cycle_resource_config config;
    config.kv_tokens = max_total_tokens;
    config.max_inflight = max_inflight;
    config.cuda_weight_budget_bytes = cuda_weight_budget_bytes;
    // Production Cycle uses a monotonic physical KV prefix. The Attention
    // cache key contains (query_tokens, kv_tokens), so a Prompt/Decode session
    // does not revisit old shapes. Keep the cache OWNER object for existing
    // resource contracts, but retain zero stale graphs; Scratch and FFN slots
    // are the reusable GPU memory mechanisms.
    config.attention_graph_cache = true;
    config.attention_graph_cache_max_graphs = 0;
    if (!pool->prepare(model, layout, cpu, phone, pc_cuda,
                       config, reason)) return false;
    model_ = &model;
    cpu_ = cpu;
    phone_ = phone;
    cuda_ = pc_cuda;
    cuda_weight_budget_bytes_ = cuda_weight_budget_bytes;
    capacity_ = max_total_tokens;
    chunk_tokens_ = chunk_tokens;
    max_inflight_ = max_inflight;
    resources_ = std::move(pool);
    prefix_ = 0;
    wire_epoch_ = 0;
    poisoned_ = false;
    if (llama_hybrid_cycle_verbose()) std::printf(
        "[CYCLE_SESSION] phase=PREPARE physical_kv_capacity=%lld "
        "layers=48 window=%d attention_graph_cache_slots=%zu "
        "scratch_pool=1 phone_gpu=REAL cuda_layers=%d "
        "cuda_budget_mib=%.2f status=PASS\n",
        (long long) capacity_, max_inflight_,
        config.attention_graph_cache_max_graphs,
        resources_->cuda_mirror_layers(),
        double(cuda_weight_budget_bytes_) / (1024.0 * 1024.0));
    return true;
}

bool llama_hybrid_cycle_session::evaluate(
        const std::vector<float> & embeddings, int tokens,
        std::vector<float> & hidden, std::string & reason,
        llama_hybrid_cycle_forward_metrics * metrics) {
    hidden.clear();
    reason.clear();
    if (!usable() || tokens < 1 || tokens > 512 ||
        int64_t(tokens) > capacity_ - prefix_) {
        reason = "SESSION_NOT_READY_OR_KV_CAPACITY";
        return false;
    }
    const bool ok = cycle_core_evaluate_with_kv(
        *model_, cpu_, phone_, *resources_,
        prefix_, wire_epoch_, embeddings, tokens,
        hidden, reason, metrics, chunk_tokens_, max_inflight_);
    if (ok) {
        prefix_ += tokens;
        ++wire_epoch_;
        if (llama_hybrid_cycle_verbose()) std::printf(
            "[CYCLE_SESSION] phase=COMMIT "
            "absolute_kv_prefix=%lld epoch=%llu "
            "appended_tokens=%d status=PASS\n",
            (long long) prefix_,
            (unsigned long long) wire_epoch_, tokens);
    } else {
        // The failed graph may have committed only some physical K/V.
        // Poison the session: never mix those prefixes on a retry.
        poisoned_ = true;
        hidden.clear();
        std::fprintf(stderr,
            "[CYCLE_SESSION] phase=COMMIT status=FAIL "
            "session=POISONED reason=%s\n", reason.c_str());
    }
    return ok;
}

bool llama_hybrid_cycle_session::usable() const {
    return model_ && cpu_ && phone_ && resources_ && !poisoned_ &&
           prefix_ >= 0 && prefix_ <= capacity_;
}

int64_t llama_hybrid_cycle_session::committed_tokens() const {
    return prefix_;
}

void llama_hybrid_cycle_session::memory_allocations(
        std::map<ggml_backend_buffer_type_t, size_t> & kv,
        std::map<ggml_backend_buffer_type_t, size_t> & scratch,
        std::map<ggml_backend_buffer_type_t, size_t> & cuda_mirrors) const {
    kv.clear();
    scratch.clear();
    cuda_mirrors.clear();
    if (!resources_) return;

    // Only count physical buffers owned by this Session. Buffer metadata
    // is local even for RPC allocations; no Phone GPU compute is issued.
    const auto account = [](
            std::map<ggml_backend_buffer_type_t, size_t> & target,
            ggml_backend_buffer_t buffer) {
        if (!buffer) return;
        target[ggml_backend_buffer_get_type(buffer)] +=
            ggml_backend_buffer_get_size(buffer);
    };

    for (int i = 0; i < resources_->prepared_layers(); ++i) {
        const auto * layer = resources_->layer(i);
        if (!layer) continue;
        if (layer->kv) {
            account(kv, layer->kv->buffer);
        }
        if (layer->cuda_weights) {
            account(cuda_mirrors, layer->cuda_weights->buffer);
        }
        if (layer->scratch_pool) {
            // Synchronous forward normally releases all leases to idle;
            // include active leases for accurate mid-session inspection.
            for (const auto & entry : layer->scratch_pool->idle) {
                account(scratch, entry.buffer);
            }
            for (const auto & entry : layer->scratch_pool->active) {
                account(scratch, entry.first);
            }
        }
    }
}

void llama_hybrid_cycle_session::reset() {
    // evaluate() is synchronous: no Stage worker can outlive its call.
    if (resources_) resources_->clear();
    resources_.reset();
    model_ = nullptr;
    cpu_ = nullptr;
    phone_ = nullptr;
    cuda_ = nullptr;
    cuda_weight_budget_bytes_ = 0;
    prefix_ = 0;
    capacity_ = 0;
    wire_epoch_ = 0;
    chunk_tokens_ = 64;
    max_inflight_ = 2;
    poisoned_ = false;
}
