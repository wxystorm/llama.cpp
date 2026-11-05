#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Experimental, strictly serial *full-region* HTP execution. The RPC-visible
// backend remains OpenCL; only explicitly selected whole Phone Regions pass
// through this local mirror. In particular, neither individual operators nor
// activation channels are split across GPU and NPU.
//
// The first implementation prioritizes correctness: it keeps private HTP KV
// and weight buffers, copies each dynamic input from OpenCL at the boundary,
// and copies the complete l_out back to OpenCL for the unchanged P2C sender.
// It never silently computes a selected NPU Region on OpenCL.
//
// NOTE: A backend's native quantized weight format is not necessarily the
// GGUF format. HTP quantized weights MUST be allocated in its repack buft.
namespace ggml_phone_npu_serial {

class full_region_executor {
public:
    explicit full_region_executor(ggml_backend_t backend) : backend_(backend) {
        if (backend_ == nullptr) {
            return;
        }
        const auto dev = ggml_backend_get_device(backend_);
        const auto reg = ggml_backend_dev_backend_reg(dev);
        if (reg != nullptr) {
            using extra_bufts_fn = ggml_backend_buffer_type_t * (*)(
                ggml_backend_dev_t);
            const auto extra = reinterpret_cast<extra_bufts_fn>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts"));
            if (extra != nullptr) {
                ggml_backend_buffer_type_t * types = extra(dev);
                if (types != nullptr) {
                    repack_buft_ = types[0];
                }
            }
        }
    }

    struct weight_preload_stats {
        size_t new_weights = 0;
        size_t reused_weights = 0;
        size_t new_bytes = 0;
        double sync_ms = 0.0;
        double copy_ms = 0.0;
    };

    // Called during Phone AUTO task registration, before any selected NPU
    // Chunk can execute. The reconstructed RPC Graph owns valid OpenCL
    // weight sources at this point. Do not preload activations or KV: their
    // values and lifetime depend on the actual Chunk and input stream.
    //
    // The same original OpenCL buffer/address is shared by Chunk UIDs, so
    // persistence keys guarantee one allocation/repack/copy per weight,
    // rather than repeating work for each 128-token Chunk.
    // The caller holds graph_compute_mutex, as for run()/reset_kv().
    bool preload_weights(const ggml_cgraph * source,
                         ggml_backend_t source_backend,
                         std::string & reason,
                         weight_preload_stats & stats) {
        reason.clear();
        stats = {};
        if (backend_ == nullptr || source == nullptr ||
                source->n_nodes <= 0 || source_backend == nullptr) {
            reason = "HTP_PRELOAD_INVALID_SOURCE";
            return false;
        }
        std::unordered_set<const ggml_tensor *> visited;
        std::unordered_set<std::string> weight_keys;
        std::vector<const ggml_tensor *> missing;
        std::function<void(const ggml_tensor *)> scan =
            [&](const ggml_tensor * t) {
                if (t == nullptr || !visited.insert(t).second) {
                    return;
                }
                if (t->op == GGML_OP_NONE && t->view_src == nullptr &&
                        is_weight(t) && !is_kv(t)) {
                    const std::string key = storage_key(t, false);
                    if (weight_keys.insert(key).second) {
                        if (persistent_.find(key) == persistent_.end()) {
                            missing.push_back(t);
                        } else {
                            ++stats.reused_weights;
                        }
                    }
                }
                if (t->view_src != nullptr) {
                    scan(t->view_src);
                }
                for (const ggml_tensor * src : t->src) {
                    scan(src);
                }
            };
        for (int i = 0; i < source->n_nodes; ++i) {
            scan(source->nodes[i]);
        }
        if (missing.empty() && stats.reused_weights == 0) {
            reason = "HTP_PRELOAD_NO_WEIGHTS";
            return false;
        }
        if (missing.empty()) {
            return true;
        }

        // RPC uploads to the OpenCL weight buffers can be asynchronous.
        // Fence them before the first cross-backend copy, but only when
        // a weight genuinely needs to be loaded.
        const int64_t sync_begin_us = ggml_time_us();
        ggml_backend_synchronize(source_backend);
        stats.sync_ms = (ggml_time_us() - sync_begin_us) / 1000.0;

        const int64_t copy_begin_us = ggml_time_us();
        for (const ggml_tensor * t : missing) {
            if (persistent_copy(t, false, reason) == nullptr) {
                stats.copy_ms = (ggml_time_us() - copy_begin_us) / 1000.0;
                return false;
            }
            ++stats.new_weights;
            stats.new_bytes += ggml_nbytes(t);
        }
        stats.copy_ms = (ggml_time_us() - copy_begin_us) / 1000.0;
        return true;
    }

    // Caller holds the server's single Phone graph-compute mutex. Only one
    // full-region execution is in progress at any moment, on either device.
    bool run(ggml_cgraph * source, ggml_tensor * output,
             std::string & reason, double & stage_ms,
             double & compute_ms, double & output_ms) {
        reason.clear();
        stage_ms = compute_ms = output_ms = 0;
        if (backend_ == nullptr || source == nullptr || output == nullptr) {
            reason = "MISSING_HTP_BACKEND_OR_GRAPH";
            return false;
        }
        if (source->n_nodes <= 0 || source->visited_hash_set.size == 0) {
            reason = "EMPTY_GRAPH";
            return false;
        }
        const int64_t start_us = ggml_time_us();
        const size_t graph_size = source->size;
        const size_t n_tensors = source->visited_hash_set.size + 64;
        const size_t mem_bytes =
            ggml_tensor_overhead() * n_tensors * 2 +
            ggml_graph_overhead_custom(graph_size, false) + 4096;
        ggml_init_params params = {mem_bytes, nullptr, true};
        ggml_context_ptr ctx { ggml_init(params) };
        if (!ctx) {
            reason = "HTP_GRAPH_CONTEXT_ALLOC";
            return false;
        }
        std::unordered_map<const ggml_tensor *, ggml_tensor *> mapped;
        std::vector<std::pair<const ggml_tensor *, ggml_tensor *>> inputs;
        bool valid = true;
        std::function<ggml_tensor *(const ggml_tensor *)> mirror =
            [&](const ggml_tensor * t) -> ggml_tensor * {
            if (t == nullptr || !valid) {
                return nullptr;
            }
            const auto found = mapped.find(t);
            if (found != mapped.end()) {
                return found->second;
            }
            // Weight/KV allocation is persistent across different Chunk UIDs;
            // intermediate activations and per-Chunk inputs are not.
            if (t->op == GGML_OP_NONE && t->view_src == nullptr &&
                    (is_weight(t) || is_kv(t))) {
                ggml_tensor * cached = persistent_copy(t, is_kv(t), reason);
                if (cached == nullptr) {
                    valid = false;
                    return nullptr;
                }
                mapped.emplace(t, cached);
                return cached;
            }
            ggml_tensor * copy = ggml_dup_tensor(ctx.get(), t);
            if (copy == nullptr) {
                reason = "HTP_DUP_TENSOR";
                valid = false;
                return nullptr;
            }
            mapped.emplace(t, copy);
            copy->op = t->op;
            copy->flags = t->flags;
            std::memcpy(copy->nb, t->nb, sizeof(copy->nb));
            std::memcpy(copy->op_params, t->op_params,
                        sizeof(copy->op_params));
            ggml_set_name(copy, t->name);
            if (t->view_src != nullptr) {
                copy->view_src = mirror(t->view_src);
                copy->view_offs = t->view_offs;
                if (copy->view_src == nullptr) {
                    valid = false;
                    return nullptr;
                }
            }
            for (int i = 0; i < GGML_MAX_SRC; ++i) {
                if (t->src[i] == nullptr) {
                    continue;
                }
                copy->src[i] = mirror(t->src[i]);
                if (copy->src[i] == nullptr) {
                    valid = false;
                    return nullptr;
                }
            }
            if (t->op == GGML_OP_NONE && t->view_src == nullptr) {
                inputs.emplace_back(t, copy);
            }
            return copy;
        };
        ggml_cgraph * graph =
            ggml_new_graph_custom(ctx.get(), graph_size, false);
        if (graph == nullptr) {
            reason = "HTP_GRAPH_ALLOC";
            return false;
        }
        for (int i = 0; i < source->n_nodes && valid; ++i) {
            ggml_tensor * cloned = mirror(source->nodes[i]);
            if (cloned == nullptr) {
                break;
            }
            ggml_build_forward_expand(graph, cloned);
        }
        const auto out = mapped.find(output);
        if (!valid || out == mapped.end() || graph->n_nodes == 0) {
            if (reason.empty()) {
                reason = "HTP_GRAPH_MIRROR_INCOMPLETE";
            }
            return false;
        }
        ggml_backend_buffer_ptr temporary {
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend_)
        };
        if (!temporary) {
            reason = "HTP_STAGE_BUFFER_ALLOC";
            return false;
        }
        for (const auto & item : inputs) {
            if (item.first->data == nullptr || item.second->buffer == nullptr ||
                    !ggml_is_contiguous(item.first)) {
                reason = std::string("HTP_INPUT_INVALID:") + item.first->name;
                return false;
            }
            ggml_backend_tensor_copy(item.first, item.second);
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            const ggml_tensor * op = graph->nodes[i];
            if (!ggml_backend_supports_op(backend_, op)) {
                reason = std::string("HTP_OP_UNSUPPORTED:") + op->name +
                         ":" + ggml_op_desc(op);
                return false;
            }
        }
        const int64_t compute_begin_us = ggml_time_us();
        stage_ms = (compute_begin_us - start_us) / 1000.0;
        if (ggml_backend_graph_compute(backend_, graph) !=
                GGML_STATUS_SUCCESS) {
            reason = "HTP_GRAPH_COMPUTE_FAILED";
            return false;
        }
        ggml_backend_synchronize(backend_);
        const int64_t compute_end_us = ggml_time_us();
        compute_ms = (compute_end_us - compute_begin_us) / 1000.0;
        // ggml_set_rows(ctx, base_cache, updates, indices) creates a VIEW
        // of the base cache; legacy op source order is updates=src[0],
        // indices=src[1], base_cache=src[2]. The mirrored SET_ROWS must
        // therefore alias the HTP-owned persistent cache allocation, not
        // copy the output into the indices tensor after execution.
        for (int i = 0; i < source->n_nodes; ++i) {
            const ggml_tensor * op = source->nodes[i];
            if (op->op != GGML_OP_SET_ROWS || op->src[2] == nullptr) {
                continue;
            }
            const auto result = mapped.find(op);
            const auto base = mapped.find(op->src[2]);
            if (result == mapped.end() || base == mapped.end() ||
                    result->second->buffer != base->second->buffer ||
                    result->second->data != base->second->data) {
                reason = "HTP_SET_ROWS_KV_ALIAS_MISMATCH";
                return false;
            }
        }
        // Keep immutable OpenCL output allocation / P2C contract unchanged.
        if (!ggml_are_same_layout(out->second, output)) {
            reason = "HTP_OUTPUT_LAYOUT_MISMATCH";
            return false;
        }
        ggml_backend_tensor_copy(out->second, output);
        output_ms = (ggml_time_us() - compute_end_us) / 1000.0;
        return true;
    }

    // Copy NPU-owned KV to the canonical RPC-visible OpenCL buffer exactly
    // once at the end of prefill, before legacy decode may use those layers.
    bool synchronize_kv(std::string & reason, size_t & n_bytes) {
        reason.clear();
        n_bytes = 0;
        if (backend_ == nullptr) {
            return true;
        }
        ggml_backend_synchronize(backend_);
        for (auto & [key, entry] : persistent_) {
            if (!entry->kv) {
                continue;
            }
            if (entry->tensor == nullptr || entry->original == nullptr ||
                    entry->original->buffer == nullptr ||
                    !ggml_are_same_layout(entry->tensor, entry->original)) {
                reason = "HTP_KV_SNAPSHOT_INVALID";
                return false;
            }
            ggml_backend_tensor_copy(entry->tensor, entry->original);
            n_bytes += ggml_nbytes(entry->tensor);
        }
        return true;
    }

    void reset_kv() {
        for (auto it = persistent_.begin(); it != persistent_.end();) {
            it = it->second->kv ? persistent_.erase(it) : ++it;
        }
    }

    bool has_backend() const { return backend_ != nullptr; }

private:
    struct persistent_tensor {
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        ggml_tensor * tensor = nullptr;
        const ggml_tensor * original = nullptr;
        bool kv = false;
    };

    static bool is_weight(const ggml_tensor * t) {
        if (t == nullptr) { return false; }
        return std::strstr(t->name, ".weight") != nullptr ||
               std::strstr(t->name, ".bias") != nullptr ||
               ggml_is_quantized(t->type);
    }

    static bool is_kv(const ggml_tensor * t) {
        if (t == nullptr) { return false; }
        return std::strstr(t->name, "cache_k_l") != nullptr ||
               std::strstr(t->name, "cache_v_l") != nullptr;
    }

    static std::string storage_key(const ggml_tensor * t, bool kv) {
        const size_t buffer = reinterpret_cast<size_t>(t->buffer);
        const size_t data = reinterpret_cast<size_t>(t->data);
        return std::string(kv ? "KV:" : "W:") +
               std::to_string(buffer) + ":" + std::to_string(data) +
               ":" + std::to_string(static_cast<int>(t->type)) +
               ":" + std::to_string(t->ne[0]) +
               ":" + std::to_string(t->ne[1]) +
               ":" + std::to_string(t->ne[2]) +
               ":" + std::to_string(t->ne[3]);
    }

    ggml_tensor * persistent_copy(const ggml_tensor * src, bool kv,
                                  std::string & reason) {
        if (src->buffer == nullptr || src->data == nullptr ||
                !ggml_is_contiguous(src)) {
            reason = std::string("HTP_PERSISTENT_SOURCE_INVALID:") + src->name;
            return nullptr;
        }
        const std::string key = storage_key(src, kv);
        const auto found = persistent_.find(key);
        if (found != persistent_.end()) {
            found->second->original = src;
            return found->second->tensor;
        }
        auto entry = std::make_unique<persistent_tensor>();
        ggml_init_params params = {
            ggml_tensor_overhead() * 4, nullptr, true
        };
        entry->ctx.reset(ggml_init(params));
        if (!entry->ctx) {
            reason = "HTP_PERSISTENT_CTX_ALLOC";
            return nullptr;
        }
        entry->tensor = ggml_dup_tensor(entry->ctx.get(), src);
        if (!entry->tensor) {
            reason = "HTP_PERSISTENT_DUP";
            return nullptr;
        }
        std::memcpy(entry->tensor->nb, src->nb,
                    sizeof(entry->tensor->nb));
        ggml_set_name(entry->tensor, src->name);
        const auto dev = ggml_backend_get_device(backend_);
        ggml_backend_buffer_type_t buft =
            ggml_backend_dev_buffer_type(dev);
        if (!kv && ggml_is_quantized(src->type)) {
            if (repack_buft_ == nullptr) {
                reason = "HTP_QUANT_REPACK_UNAVAILABLE";
                return nullptr;
            }
            buft = repack_buft_;
        }
        entry->buffer.reset(
            ggml_backend_alloc_ctx_tensors_from_buft(
                entry->ctx.get(), buft));
        if (!entry->buffer) {
            reason = "HTP_PERSISTENT_BUFFER_ALLOC";
            return nullptr;
        }
        if (!kv) {
            ggml_backend_buffer_set_usage(
                entry->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_backend_tensor_copy(src, entry->tensor);
        entry->original = src;
        entry->kv = kv;
        ggml_tensor * result = entry->tensor;
        persistent_.emplace(key, std::move(entry));
        return result;
    }

    ggml_backend_t backend_ = nullptr;  // non-owning; RPC server owns it
    ggml_backend_buffer_type_t repack_buft_ = nullptr;
    std::unordered_map<std::string, std::unique_ptr<persistent_tensor>>
        persistent_;
};

}  // namespace ggml_phone_npu_serial
