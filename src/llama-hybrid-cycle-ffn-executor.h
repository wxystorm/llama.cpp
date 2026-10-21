#pragma once

#include "llama-hybrid-cycle-graph.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h" // ggml_backend_buffer_copy_tensor: no host fallback

#include <cstddef>
#include <cstdint>

// Executes ONE pre-built physical MoE FFN shard on exactly ONE backend.
// The full Qwen3-MoE builder constructs the same FFN graph with precomputed
// norm/Top-K inputs. No Meta backend, Meta AllReduce, Router, or residual ADD.
//
// The GGML graph/context, backend, and static model weight slices are BORROWED.
// The temporary compute allocation is owned by this executor and MUST be
// destroyed before the owning GGML graph context/model weights. Neither a
// second backend nor any transport is silently invoked.
//
// This primitive is synchronous and correctness-oriented. The region FIFO
// scheduler will run one instance per backend on independent compute workers.
class llama_hybrid_cycle_ffn_executor {
public:
    llama_hybrid_cycle_ffn_executor() = default;
    llama_hybrid_cycle_ffn_executor(const llama_hybrid_cycle_ffn_executor &) = delete;
    llama_hybrid_cycle_ffn_executor & operator=(const llama_hybrid_cycle_ffn_executor &) = delete;

    ~llama_hybrid_cycle_ffn_executor() { reset(); }

    bool prepare(
            ggml_backend_t backend,
            ggml_context * compute_ctx,
            ggml_cgraph * graph,
            const llama_hybrid_cycle_ffn_graph & ffn,
            const llama_hybrid_cycle_ffn_shard_weights & weights,
            int64_t hidden,
            int64_t tokens,
            int64_t topk,
            int64_t experts) {
        reset();

        if (!backend || !compute_ctx || !graph ||
            !llama_hybrid_cycle_ffn_graph_valid(ffn, hidden, tokens, topk) ||
            !llama_hybrid_cycle_ffn_shard_weights_valid(weights, hidden, experts)) {
            return false;
        }

        const auto backend_buft = ggml_backend_get_default_buffer_type(backend);
        if (!backend_buft ||
            ggml_backend_buffer_get_type(weights.up->buffer) != backend_buft ||
            ggml_backend_buffer_get_type(weights.gate->buffer) != backend_buft ||
            ggml_backend_buffer_get_type(weights.down->buffer) != backend_buft) {
            return false;
        }

        // Never attach a Meta backend to the local FFN graph. In particular,
        // a Meta graph would recombine the PC/Phone partials too early.
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) ==
            GGML_BACKEND_DEVICE_TYPE_META) {
            return false;
        }

        // Only independently allocated graph compute tensors live in
        // compute_ctx. Weight shards are borrowed from the model's static
        // Meta allocation and must not be reallocated or copied here.
        ggml_backend_buffer_t compute_buffer =
            ggml_backend_alloc_ctx_tensors_from_buft(compute_ctx, backend_buft);
        if (!compute_buffer) {
            return false;
        }

        backend_ = backend;
        graph_ = graph;
        ffn_ = ffn;
        compute_buffer_ = compute_buffer;
        return true;
    }

    bool execute(
            const float * norm, size_t norm_count,
            const int32_t * ids, size_t ids_count,
            const float * routing, size_t routing_count,
            float * output, size_t output_count,
            const llama_hybrid_cycle_ready_graph * local_phone_ready = nullptr) {
        // The Phone source may transfer its output directly from the
        // remote GPU through the independent Cycle data-plane. In that case
        // no ordinary RPC tensor_get is allowed: the caller will use the
        // received PARTIAL frame on the PC, not a duplicate host readback.
        const bool read_output = output != nullptr;
        const bool phone_local = local_phone_ready != nullptr;
        // Phone GPU consumes Attention/Router tensors through device-local
        // copies. Do not gate this compute on a redundant PC READY readback:
        // the sender can run its FFN as soon as Attention has completed.
        // Host pointers/counts are required only by the PC FFN path.
        if (!backend_ || !graph_ || !compute_buffer_ ||
            (!phone_local &&
                (!norm || !ids || !routing ||
                 norm_count != size_t(ggml_nelements(ffn_.norm)) ||
                 ids_count != size_t(ggml_nelements(ffn_.topk_ids)) ||
                 routing_count != size_t(ggml_nelements(ffn_.topk_weights)))) ||
            (read_output
                ? output_count != size_t(ggml_nelements(ffn_.ffn_output))
                : output_count != 0)) {
            return false;
        }

        // In P2C the Phone's Attention/Router outputs already reside on
        // Phone GPU. Use only SAME-RPC-SERVER device-local buffer copies:
        // never fall back to ggml_backend_tensor_copy(), which silently
        // performs Phone->PC get + PC->Phone set when cpy_tensor fails.
        // PC FFN continues to consume the received host READY arrays.
        if (local_phone_ready != nullptr) {
            const auto copy_local = [](const ggml_tensor * src, ggml_tensor * dst) {
                if (src == nullptr || dst == nullptr ||
                    src->buffer == nullptr || dst->buffer == nullptr ||
                    src->type != dst->type) {
                    return false;
                }
                // ggml_are_same_layout is private to ggml-impl.h and is
                // not declared for this header. Preserve its exact tensor
                // type/dimension/stride contract without importing it.
                for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                    if (src->ne[i] != dst->ne[i] ||
                        src->nb[i] != dst->nb[i]) {
                        return false;
                    }
                }
                // Direct same-device copy ONLY; never use a host fallback.
                return ggml_backend_buffer_copy_tensor(src, dst);
            };
            if (!copy_local(local_phone_ready->ffn_norm, ffn_.norm) ||
                !copy_local(local_phone_ready->topk_ids, ffn_.topk_ids) ||
                !copy_local(local_phone_ready->topk_weights, ffn_.topk_weights)) {
                return false;
            }
        } else {
            ggml_backend_tensor_set(ffn_.norm, norm, 0, ggml_nbytes(ffn_.norm));
            ggml_backend_tensor_set(ffn_.topk_ids, ids, 0, ggml_nbytes(ffn_.topk_ids));
            ggml_backend_tensor_set(ffn_.topk_weights, routing, 0, ggml_nbytes(ffn_.topk_weights));
        }

        if (ggml_backend_graph_compute(backend_, graph_) != GGML_STATUS_SUCCESS) {
            return false;
        }
        ggml_backend_synchronize(backend_);

        // The caller owns the output buffer. On the real P2C Phone backend,
        // this is the source FFN partial for Send #2, not a layer output.
        if (read_output) {
            ggml_backend_tensor_get(
                ffn_.ffn_output, output, 0, ggml_nbytes(ffn_.ffn_output));
        }
        return true;
    }

    void reset() {
        if (compute_buffer_) {
            ggml_backend_buffer_free(compute_buffer_);
            compute_buffer_ = nullptr;
        }
        backend_ = nullptr;
        graph_ = nullptr;
        ffn_ = {};
    }

private:
    ggml_backend_t backend_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_backend_buffer_t compute_buffer_ = nullptr;
    llama_hybrid_cycle_ffn_graph ffn_;
};
