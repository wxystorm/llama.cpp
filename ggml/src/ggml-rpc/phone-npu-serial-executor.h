#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"
#include "phone-qwen3-local-builder.h"
#include "phone-qwen3-local-parity.h"
#include "phone-npu-weight-banks.h"
#include "phone-npu-serial-policy.h"
#include "phone-plan-region-map.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <cinttypes>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <map>
#include <tuple>
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

    ~full_region_executor() {
        // rpc_server destroys this executor before its private HTP backend.
        // Drop any backend opnodes that still reference registered Graphs.
        if (backend_ != nullptr) {
            const auto reg = ggml_backend_dev_backend_reg(
                ggml_backend_get_device(backend_));
            if (reg != nullptr) {
                using clear_fn = void (*)(ggml_backend_t);
                const auto clear = reinterpret_cast<clear_fn>(
                    ggml_backend_reg_get_proc_address(
                        reg, "ggml_backend_hexagon_clear_fixed_graph_cache"));
                if (clear != nullptr) { clear(backend_); }
            }
        }
    }

    struct weight_preload_stats {
        size_t new_weights = 0;
        size_t reused_weights = 0;
        size_t model_hits = 0;
        size_t new_bytes = 0;
        double sync_ms = 0.0;
        double copy_ms = 0.0;
    };

    // Opt-in timing for HTP preparation in run(). graph_mirror_ms excludes
    // persistent_bind_ms, so all phase durations can be summed.
    struct stage_profile_stats {
        double context_create_ms = 0.0;
        double graph_mirror_ms = 0.0;
        double persistent_bind_ms = 0.0;
        double weight_bind_ms = 0.0;
        double kv_bind_ms = 0.0;
        double temporary_alloc_ms = 0.0;
        double input_copy_ms = 0.0;
        double op_check_ms = 0.0;
        double other_ms = 0.0;
        size_t graph_nodes = 0;
        size_t input_tensors = 0;
        size_t kv_new = 0;
        size_t kv_reused = 0;
    };

    struct model_load_stats {
        size_t new_weights = 0;
        size_t new_bytes = 0;
        double file_read_ms = 0.0;
        double htp_repack_ms = 0.0;
    };

    // Allocate either fully resident Stage weights or exactly two rolling
    // HTP-repack arenas. A Region may contain consecutive Qwen3 layers.
    // Bank A holds even ranks; Bank B holds odd ranks. KV allocations are
    // outside both Banks. No third Bank exists even when many NPU layers are
    // assigned. Physical allocation (not device_get_memory(), which reports
    // zero on Hexagon) validates the user's optional budget.
    bool configure_banked_model(
            const ggml_rpc_local_tensor_source & local,
            const std::vector<ggml_phone_plan::stage_region_range> & ordered,
            size_t budget_bytes, std::string & reason,
            std::array<size_t, 2> & capacities,
            bool fully_resident) {
        capacities = {};
        reason.clear();
        if (banked_mode_ || resident_mode_ || !model_weights_.empty() ||
                repack_buft_ == nullptr || !local.get_info || !local.read ||
                ordered.empty() || (!fully_resident && ordered.size() < 2) ||
                budget_bytes == 0) {
            reason = "NPU_BANK_CONFIG_OR_LOCAL_GGUF_UNAVAILABLE";
            return false;
        }
        const auto dev_reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(backend_));
        if (!fully_resident && (dev_reg == nullptr ||
                ggml_backend_reg_get_proc_address(dev_reg,
                    "ggml_backend_hexagon_evict_fixed_graph_uid") == nullptr)) {
            reason = "NPU_BANK_REQUIRES_SELECTIVE_GRAPH_EVICTION";
            return false;
        }
        const size_t alignment = ggml_backend_buft_get_alignment(repack_buft_);
        if (!alignment || alignment > 65536) {
            reason = "NPU_BANK_BAD_ALIGNMENT";
            return false;
        }
        static const char * const weights[] = {
            "attn_norm.weight", "attn_q.weight", "attn_k.weight",
            "attn_v.weight", "attn_output.weight", "ffn_norm.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "attn_q_norm.weight", "attn_k_norm.weight"
        };
        std::vector<banked_layer_spec> layouts;
        std::vector<weight_bank_region> policy_regions;
        layouts.reserve(ordered.size());
        policy_regions.reserve(ordered.size());
        for (size_t rank = 0; rank < ordered.size(); ++rank) {
            if (rank > 0 &&
                    (ordered[rank - 1].region >= ordered[rank].region ||
                     ordered[rank - 1].layer_end > ordered[rank].layer_begin)) {
                reason = "NPU_BANK_REGION_LAYER_ORDER";
                return false;
            }
            banked_layer_spec spec {};
            spec.region = ordered[rank].region;
            spec.layer = ordered[rank].layer_begin;
            spec.layer_end = ordered[rank].layer_end;
            ggml_init_params params {
                ggml_tensor_overhead() *
                    (size_t(32) * (spec.layer_end - spec.layer) + 8) +
                    4096, nullptr, true
            };
            ggml_context_ptr ctx { ggml_init(params) };
            if (!ctx) {
                reason = "NPU_BANK_METADATA_CONTEXT";
                return false;
            }
            if (spec.layer >= spec.layer_end ||
                    spec.layer_end > 256 ||
                    spec.layer_end - spec.layer > 32) {
                reason = "NPU_STAGE_RANGE_INVALID";
                return false;
            }
            for (uint32_t il = spec.layer; il < spec.layer_end; ++il) {
            for (const char * suffix : weights) {
                banked_tensor_spec tensor {};
                char name[GGML_MAX_NAME] = {};
                const int written = std::snprintf(name, sizeof(name),
                    "blk.%u.%s", il, suffix);
                if (written <= 0 ||
                        (size_t) written >= sizeof(name)) {
                    reason = "NPU_BANK_WEIGHT_NAME";
                    return false;
                }
                tensor.name = name;
                if (!local.get_info(local.user_data, name, &tensor.info) ||
                        tensor.info.type >= GGML_TYPE_COUNT ||
                        tensor.info.n_dims < 1 ||
                        tensor.info.n_dims > GGML_MAX_DIMS ||
                        tensor.info.nbytes == 0 ||
                        tensor.info.nbytes > SIZE_MAX) {
                    reason = "NPU_BANK_GGUF_TENSOR_MISSING:" + tensor.name;
                    return false;
                }
                int64_t shape[GGML_MAX_DIMS] = {1, 1, 1, 1};
                for (uint32_t d = 0; d < tensor.info.n_dims; ++d) {
                    if (tensor.info.ne[d] == 0 ||
                            tensor.info.ne[d] > uint64_t(INT64_MAX)) {
                        reason = "NPU_BANK_GGUF_TENSOR_SHAPE:" + tensor.name;
                        return false;
                    }
                    shape[d] = static_cast<int64_t>(tensor.info.ne[d]);
                }
                ggml_tensor * t = ggml_new_tensor(
                    ctx.get(), static_cast<ggml_type>(tensor.info.type),
                    GGML_MAX_DIMS, shape);
                if (!t || ggml_nbytes(t) !=
                        static_cast<size_t>(tensor.info.nbytes)) {
                    reason = "NPU_BANK_GGUF_LAYOUT:" + tensor.name;
                    return false;
                }
                const size_t need =
                    ggml_backend_buft_get_alloc_size(repack_buft_, t);
                if (need == 0 || need > SIZE_MAX - alignment ||
                        spec.bytes > SIZE_MAX - alignment ||
                        spec.bytes + alignment - 1 < spec.bytes) {
                    reason = "NPU_BANK_TENSOR_ALLOCATION_OVERFLOW";
                    return false;
                }
                const size_t aligned =
                    ((spec.bytes + alignment - 1) / alignment) * alignment;
                if (aligned > SIZE_MAX - need) {
                    reason = "NPU_BANK_REGION_ALLOCATION_OVERFLOW";
                    return false;
                }
                tensor.offset = aligned;
                tensor.alloc_bytes = need;
                spec.bytes = aligned + need;
                spec.tensors.push_back(std::move(tensor));
            }
            }
            capacities[rank % 2] =
                std::max(capacities[rank % 2], spec.bytes);
            policy_regions.push_back({
                spec.region, static_cast<int>(spec.layer),
                static_cast<int>(spec.layer_end), spec.bytes
            });
            region_rank_[spec.region] = rank;
            layouts.push_back(std::move(spec));
        }
        size_t resident_bytes = 0;
        for (const auto & stage : layouts) {
            if (stage.bytes > SIZE_MAX - resident_bytes) {
                reason = "NPU_RESIDENT_SIZE_OVERFLOW";
                region_rank_.clear();
                return false;
            }
            resident_bytes += stage.bytes;
            if (stage.bytes > ggml_backend_buft_get_max_size(repack_buft_)) {
                reason = "NPU_STAGE_SINGLE_ALLOC_EXCEEDED";
                region_rank_.clear();
                return false;
            }
        }
        if (!fully_resident && (capacities[0] > budget_bytes ||
                capacities[1] > budget_bytes - capacities[0] ||
                capacities[0] > ggml_backend_buft_get_max_size(repack_buft_) ||
                capacities[1] > ggml_backend_buft_get_max_size(repack_buft_))) {
            reason = "NPU_BANK_BUDGET_OR_SINGLE_ALLOC_EXCEEDED";
            region_rank_.clear();
            return false;
        }
        if (fully_resident && resident_bytes > budget_bytes) {
            reason = "NPU_RESIDENT_BUDGET_EXCEEDED";
            region_rank_.clear();
            return false;
        }
        // The rolling Bank lifetime protocol does not apply to resident
        // weights. All selected Stage buffers are allocated once and remain.
        if (!fully_resident && !weight_banks_.configure(
                policy_regions, 0, 0, budget_bytes, reason)) {
            region_rank_.clear();
            return false;
        }
        banked_layers_ = std::move(layouts);
        bank_local_source_ = local;
        if (fully_resident) {
            resident_weight_arenas_.resize(banked_layers_.size());
            for (size_t rank = 0; rank < banked_layers_.size(); ++rank) {
                auto & arena = resident_weight_arenas_[rank];
                arena.reset(ggml_backend_buft_alloc_buffer(
                    repack_buft_, banked_layers_[rank].bytes));
                if (!arena) {
                    reason = "NPU_RESIDENT_PHYSICAL_ALLOC_FAILED";
                    model_weights_.clear();
                    resident_weight_arenas_.clear();
                    banked_layers_.clear();
                    region_rank_.clear();
                    return false;
                }
                ggml_backend_buffer_set_usage(
                    arena.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
            resident_mode_ = true;
            for (size_t rank = 0; rank < banked_layers_.size(); ++rank) {
                model_load_stats stats {};
                if (!load_banked_rank(rank, reason, stats)) {
                    model_weights_.clear();
                    resident_weight_arenas_.clear();
                    banked_layers_.clear();
                    region_rank_.clear();
                    resident_mode_ = false;
                    return false;
                }
            }
            capacities[0] = resident_bytes;
            capacities[1] = 0;
            return true;
        }
        for (size_t bank = 0; bank < 2; ++bank) {
            bank_weight_arenas_[bank].reset(ggml_backend_buft_alloc_buffer(
                repack_buft_, capacities[bank]));
            if (!bank_weight_arenas_[bank]) {
                reason = "NPU_BANK_PHYSICAL_ALLOC_OR_MAP_FAILED";
                bank_weight_arenas_ = {};
                weight_banks_.reset();
                banked_layers_.clear();
                region_rank_.clear();
                return false;
            }
            ggml_backend_buffer_set_usage(
                bank_weight_arenas_[bank].get(),
                GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        bank_local_source_ = local;
        banked_mode_ = true;
        for (size_t rank = 0; rank < 2; ++rank) {
            model_load_stats stats {};
            if (!load_banked_rank(rank, reason, stats) ||
                    !weight_banks_.prime_complete(rank)) {
                if (reason.empty()) {
                    reason = "NPU_BANK_INITIAL_PRIME_FAILED";
                }
                model_weights_.clear();
                bank_weight_arenas_ = {};
                banked_layers_.clear();
                region_rank_.clear();
                weight_banks_.reset();
                banked_mode_ = false;
                return false;
            }
        }
        reason.clear();
        return true;
    }

    bool configure_banked_model(
            const ggml_rpc_local_tensor_source & local,
            const std::vector<std::pair<uint32_t, uint32_t>> & ordered,
            size_t budget_bytes, std::string & reason,
            std::array<size_t, 2> & capacities) {
        std::vector<ggml_phone_plan::stage_region_range> stages;
        stages.reserve(ordered.size());
        for (const auto & item : ordered) {
            stages.push_back({item.first, item.second, item.second + 1});
        }
        return configure_banked_model(
            local, stages, budget_bytes, reason, capacities, false);
    }

    // Host-only GGUF prefetch; no HTP mutex, no Bank mutation.
    // Bounded because staging a complete large layer can exhaust phone RAM.
    bool stage_banked_rank(size_t rank,
                           std::vector<std::vector<uint8_t>> & raw,
                           double & read_ms, std::string & reason) const {
        read_ms = 0;
        raw.clear();
        reason.clear();
        if (!banked_mode_ || rank >= banked_layers_.size() ||
                !bank_local_source_.read) {
            reason = "NPU_BANK_HOST_STAGE_INVALID_RANK";
            return false;
        }
        size_t cap_mib = 2048;
        if (const char * budget =
                std::getenv("GGML_RPC_PHONE_NPU_HOST_STAGE_MIB")) {
            char * end = nullptr;
            const unsigned long long n = std::strtoull(budget, &end, 10);
            if (end == budget || *end != '\0' || n == 0 ||
                    n > SIZE_MAX / (1024 * 1024)) {
                reason = "NPU_BANK_HOST_STAGE_INVALID_BUDGET";
                return false;
            }
            cap_mib = static_cast<size_t>(n);
        }
        size_t total = 0;
        for (const auto & t : banked_layers_[rank].tensors) {
            const size_t bytes = static_cast<size_t>(t.info.nbytes);
            if (bytes > cap_mib * 1024 * 1024 - total) {
                reason = "NPU_BANK_HOST_STAGE_BUDGET_EXCEEDED";
                return false;
            }
            total += bytes;
        }
        const int64_t t0 = ggml_time_us();
        try {
            raw.reserve(banked_layers_[rank].tensors.size());
            for (const auto & tensor : banked_layers_[rank].tensors) {
                auto & data = raw.emplace_back();
                data.resize(static_cast<size_t>(tensor.info.nbytes));
                if (!bank_local_source_.read(
                        bank_local_source_.user_data, tensor.name.c_str(),
                        0, data.data(), data.size())) {
                    raw.clear();
                    reason = "NPU_BANK_HOST_STAGE_GGUF_READ:" + tensor.name;
                    return false;
                }
            }
        } catch (const std::bad_alloc &) {
            raw.clear();
            reason = "NPU_BANK_HOST_STAGE_OOM";
            return false;
        }
        read_ms = (ggml_time_us() - t0) / 1000.0;
        return true;
    }

    // Actual backend buffer sizes, not raw GGUF file sizes or Hexagon's
    // unavailable free-memory query. Call under the HTP mutex.
    struct banked_memory_snapshot {
        size_t a_bytes = 0;
        size_t b_bytes = 0;
        size_t kv_bytes = 0;
        size_t graph_bytes = 0;
        size_t weight_tensor_count = 0;
        size_t compiled_chunk_graphs = 0;
    };
    banked_memory_snapshot banked_memory_for_run(uint64_t run_id) const {
        banked_memory_snapshot result {};
        if (!has_any_weight_plan()) { return result; }
        if (resident_mode_) {
            for (const auto & arena : resident_weight_arenas_) {
                if (arena) {
                    result.a_bytes += ggml_backend_buffer_get_size(arena.get());
                }
            }
        } else {
            result.a_bytes = ggml_backend_buffer_get_size(
                bank_weight_arenas_[0].get());
            result.b_bytes = ggml_backend_buffer_get_size(
                bank_weight_arenas_[1].get());
        }
        result.weight_tensor_count = model_weights_.size();
        for (const auto & item : owned_kv_) {
            if (item.first.first == run_id && item.second->buffer) {
                result.kv_bytes += ggml_backend_buffer_get_size(
                    item.second->buffer.get());
            }
        }
        for (const auto & item : owned_graphs_) {
            if (std::get<0>(item.first) == run_id &&
                    item.second->buffer) {
                result.graph_bytes += ggml_backend_buffer_get_size(
                    item.second->buffer.get());
                ++result.compiled_chunk_graphs;
            }
        }
        return result;
    }

    bool has_weight_banks() const { return banked_mode_; }
    bool has_resident_weights() const { return resident_mode_; }
    bool has_any_weight_plan() const { return banked_mode_ || resident_mode_; }
    int rank_for_region(uint32_t region, uint32_t layer) const {
        if (!has_any_weight_plan()) { return -1; }
        const auto it = region_rank_.find(region);
        if (it == region_rank_.end() ||
                layer < banked_layers_[it->second].layer ||
                layer >= banked_layers_[it->second].layer_end) {
            return -1;
        }
        return static_cast<int>(it->second);
    }
    bool rank_banked_ready(uint32_t region) const {
        const auto it = region_rank_.find(region);
        if (!has_any_weight_plan() || it == region_rank_.end()) {
            return false;
        }
        if (resident_mode_) { return true; }
        const size_t rank = it->second;
        const auto & bank = weight_banks_.bank(rank % 2);
        return bank.rank == rank &&
               bank.state == two_weight_banks::phase::READY;
    }
    bool initial_banked_region(uint32_t region) const {
        const auto it = region_rank_.find(region);
        return banked_mode_ && it != region_rank_.end() && it->second < 2;
    }

    // Called once after the Phone's optional local GGUF loader has parsed
    // model metadata, *before* any RPC client connects. This is the exact
    // model-load stage: all selected immutable weight tensors live in HTP0
    // for the server lifetime, independently of per-Prompt Graph UIDs.
    //
    // No source OpenCL buffer or Graph is needed. A missing/incompatible
    // local GGUF tensor fails only this bootstrap layer; the existing
    // on-demand/async preload remains the cache-miss fallback.
    bool load_model_layer(const ggml_rpc_local_tensor_source & local,
                          uint32_t layer,
                          std::string & reason,
                          model_load_stats & stats) {
        reason.clear();
        stats = {};
        if (has_any_weight_plan()) {
            reason = "HTP_MODEL_LOAD_PROHIBITED_WHILE_PLAN_WEIGHTS_ACTIVE";
            return false;
        }
        if (backend_ == nullptr || local.get_info == nullptr ||
                local.read == nullptr) {
            reason = "HTP_MODEL_LOAD_NO_LOCAL_GGUF";
            return false;
        }

        // Dense Qwen3 full-layer weight contract. Do not guess missing
        // optional tensors or silently build an incomplete HTP layer.
        static const char * suffixes[] = {
            "attn_norm.weight", "attn_q.weight", "attn_k.weight",
            "attn_v.weight", "attn_output.weight", "ffn_norm.weight",
            "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
            "attn_q_norm.weight", "attn_k_norm.weight"
        };
        std::unordered_map<
            std::string, std::unique_ptr<persistent_tensor>> pending;
        for (const char * suffix : suffixes) {
            char name[GGML_MAX_NAME] = {};
            const int written = std::snprintf(
                name, sizeof(name), "blk.%u.%s", layer, suffix);
            if (written <= 0 ||
                    static_cast<size_t>(written) >= sizeof(name)) {
                reason = "HTP_MODEL_LOAD_NAME_TOO_LONG";
                return false;
            }

            ggml_rpc_local_tensor_info info {};
            if (!local.get_info(local.user_data, name, &info)) {
                reason = std::string("HTP_MODEL_LOAD_TENSOR_NOT_FOUND:") +
                    name;
                return false;
            }
            if (info.type >= GGML_TYPE_COUNT ||
                    info.n_dims < 1 || info.n_dims > GGML_MAX_DIMS ||
                    info.nbytes == 0 || info.nbytes > SIZE_MAX) {
                reason = std::string("HTP_MODEL_LOAD_INVALID_INFO:") +
                    name;
                return false;
            }

            int64_t shape[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (uint32_t d = 0; d < info.n_dims; ++d) {
                if (info.ne[d] == 0 ||
                        info.ne[d] >
                            static_cast<uint64_t>(INT64_MAX)) {
                    reason = std::string("HTP_MODEL_LOAD_BAD_SHAPE:") +
                        name;
                    return false;
                }
                shape[d] = static_cast<int64_t>(info.ne[d]);
            }
            auto entry = std::make_unique<persistent_tensor>();
            ggml_init_params params = {
                ggml_tensor_overhead() * 4, nullptr, true
            };
            entry->ctx.reset(ggml_init(params));
            if (!entry->ctx) {
                reason = "HTP_MODEL_LOAD_CONTEXT_ALLOC";
                return false;
            }
            entry->tensor = ggml_new_tensor(
                entry->ctx.get(),
                static_cast<ggml_type>(info.type),
                GGML_MAX_DIMS, shape);
            if (!entry->tensor ||
                    ggml_nbytes(entry->tensor) !=
                        static_cast<size_t>(info.nbytes)) {
                reason = std::string("HTP_MODEL_LOAD_LAYOUT_MISMATCH:") +
                    name;
                return false;
            }
            ggml_set_name(entry->tensor, name);
            const std::string key = model_weight_key(entry->tensor);
            if (model_weights_.find(key) != model_weights_.end()) {
                continue;
            }

            const auto dev = ggml_backend_get_device(backend_);
            ggml_backend_buffer_type_t buft =
                ggml_backend_dev_buffer_type(dev);
            if (ggml_is_quantized(entry->tensor->type)) {
                if (repack_buft_ == nullptr) {
                    reason = "HTP_MODEL_LOAD_REPACK_UNAVAILABLE";
                    return false;
                }
                buft = repack_buft_;
            }
            entry->buffer.reset(
                ggml_backend_alloc_ctx_tensors_from_buft(
                    entry->ctx.get(), buft));
            if (!entry->buffer) {
                reason = std::string("HTP_MODEL_LOAD_BUFFER_ALLOC:") +
                    name;
                return false;
            }
            ggml_backend_buffer_set_usage(
                entry->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            // HTP repack works on a complete GGUF-layout quantized tensor.
            // Hold just one weight's raw bytes at a time to bound startup
            // RAM. The local loader callback already protects GGUF reads.
            std::vector<uint8_t> raw;
            try {
                raw.resize(static_cast<size_t>(info.nbytes));
            } catch (const std::bad_alloc &) {
                reason = "HTP_MODEL_LOAD_HOST_OOM";
                return false;
            }
            const int64_t read_begin = ggml_time_us();
            if (!local.read(local.user_data, name, 0,
                            raw.data(), raw.size())) {
                reason = std::string("HTP_MODEL_LOAD_READ_FAILED:") +
                    name;
                return false;
            }
            stats.file_read_ms += (ggml_time_us() - read_begin) / 1000.0;

            const int64_t repack_begin = ggml_time_us();
            ggml_backend_tensor_set(
                entry->tensor, raw.data(), 0, raw.size());
            stats.htp_repack_ms +=
                (ggml_time_us() - repack_begin) / 1000.0;
            entry->original = nullptr;
            entry->kv = false;
            pending.emplace(key, std::move(entry));
            ++stats.new_weights;
            stats.new_bytes += raw.size();
        }

        // Atomic layer publication: no partial cache hit if one tensor
        // could not be loaded.
        for (auto & item : pending) {
            model_weights_.emplace(
                item.first, std::move(item.second));
        }
        return true;
    }

    size_t model_resident_weights() const {
        return model_weights_.size();
    }

    struct local_graph_shadow_stats {
        uint32_t chunks = 0;
        size_t total_nodes = 0;
        size_t kv_writes = 0;
        size_t flash_nodes = 0;
        size_t rope_nodes = 0;
    };

    // Phase 2: build every Qwen3 Chunk graph from Phone-local immutable
    // weights, without reading or mirroring the PC's RPC graph. No backend
    // memory is allocated and nothing is compiled/executed in this check.
    // A later cutover will bind fixed activation buffers and shared KV to
    // these same graph shapes before precompiling on HTP.
    bool validate_owned_local_graphs(
            const ggml_phone_owned::region_plan & plan,
            std::string & reason,
            local_graph_shadow_stats & stats) const {
        stats = {};
        const char * why = "NONE";
        if (!ggml_phone_owned::validate(plan, why)) {
            reason = why;
            return false;
        }
        // Ranks >= 2 do not have weights resident yet: the Bank Mover
        // installs them only after the previous same-parity Region drains.
        // Shadow-validation may build GGML *metadata* graphs from the
        // already-checked GGUF layouts, but must not load another HTP Bank
        // or precompile against addresses that will be repacked later.
        const int bank_rank = banked_mode_ ?
            rank_for_region(plan.region, plan.layer) : -1;
        const bool deferred_metadata = use_deferred_weight_metadata(
            banked_mode_, bank_rank,
            bank_rank >= 0 && rank_banked_ready(plan.region));
        ggml_context_ptr weight_meta_ctx;
        if (deferred_metadata) {
            ggml_init_params params = {
                ggml_tensor_overhead() * 32 + 4096, nullptr, true
            };
            weight_meta_ctx.reset(ggml_init(params));
            if (!weight_meta_ctx) {
                reason = "PHONE_LOCAL_DEFERRED_GGUF_CONTEXT_ALLOC";
                return false;
            }
        }
        std::string deferred_lookup_error;
        ggml_phone_qwen3::layer_weights weights {};
        const auto find_weight = [&](const char * suffix) -> ggml_tensor * {
            char name[GGML_MAX_NAME] = {};
            std::snprintf(name, sizeof(name), "blk.%u.%s", plan.layer, suffix);
            if (deferred_metadata) {
                const auto & specs =
                    banked_layers_[static_cast<size_t>(bank_rank)].tensors;
                const banked_tensor_spec * found = nullptr;
                for (const auto & spec : specs) {
                    if (spec.name != name) { continue; }
                    if (found != nullptr) {
                        deferred_lookup_error =
                            std::string("PHONE_LOCAL_DEFERRED_DUPLICATE_GGUF_WEIGHT:") + name;
                        return nullptr;
                    }
                    found = &spec;
                }
                if (found == nullptr) {
                    deferred_lookup_error =
                        std::string("PHONE_LOCAL_DEFERRED_MISSING_GGUF_WEIGHT:") + name;
                    return nullptr;
                }
                const auto & info = found->info;
                if (info.type >= GGML_TYPE_COUNT ||
                        info.n_dims == 0 || info.n_dims > GGML_MAX_DIMS ||
                        info.nbytes == 0) {
                    deferred_lookup_error =
                        std::string("PHONE_LOCAL_DEFERRED_INVALID_GGUF_LAYOUT:") + name;
                    return nullptr;
                }
                int64_t shape[GGML_MAX_DIMS] = {1, 1, 1, 1};
                for (uint32_t d = 0; d < info.n_dims; ++d) {
                    if (info.ne[d] == 0 ||
                            info.ne[d] > uint64_t(INT64_MAX)) {
                        deferred_lookup_error =
                            std::string("PHONE_LOCAL_DEFERRED_INVALID_GGUF_SHAPE:") + name;
                        return nullptr;
                    }
                    shape[d] = static_cast<int64_t>(info.ne[d]);
                }
                ggml_tensor * tensor = ggml_new_tensor(
                    weight_meta_ctx.get(), static_cast<ggml_type>(info.type),
                    GGML_MAX_DIMS, shape);
                if (tensor == nullptr ||
                        ggml_nbytes(tensor) != static_cast<size_t>(info.nbytes)) {
                    deferred_lookup_error =
                        std::string("PHONE_LOCAL_DEFERRED_GGUF_SIZE_MISMATCH:") + name;
                    return nullptr;
                }
                ggml_set_name(tensor, name);
                return tensor;
            }
            ggml_tensor * selected = nullptr;
            for (const auto & entry : model_weights_) {
                ggml_tensor * tensor = entry.second->tensor;
                if (tensor == nullptr ||
                        std::strcmp(tensor->name, name) != 0) {
                    continue;
                }
                if (selected != nullptr) { return nullptr; }
                selected = tensor;
            }
            return selected;
        };
        weights.attn_norm = find_weight("attn_norm.weight");
        weights.wq = find_weight("attn_q.weight");
        weights.wk = find_weight("attn_k.weight");
        weights.wv = find_weight("attn_v.weight");
        weights.wo = find_weight("attn_output.weight");
        weights.ffn_norm = find_weight("ffn_norm.weight");
        weights.ffn_gate = find_weight("ffn_gate.weight");
        weights.ffn_up = find_weight("ffn_up.weight");
        weights.ffn_down = find_weight("ffn_down.weight");
        weights.q_norm = find_weight("attn_q_norm.weight");
        weights.k_norm = find_weight("attn_k_norm.weight");
        if (!ggml_phone_qwen3::validate_weights(plan, weights, why)) {
            reason = !deferred_lookup_error.empty() ? deferred_lookup_error :
                std::string("PHONE_LOCAL_MODEL_") + why;
            return false;
        }
        const size_t meta_bytes = ggml_tensor_overhead() * 1024 +
            ggml_graph_overhead_custom(512, false) + 4096;
        for (uint32_t chunk = 0;
                chunk < ggml_phone_owned::chunk_count(plan); ++chunk) {
            ggml_init_params params = {meta_bytes, nullptr, true};
            ggml_context_ptr ctx { ggml_init(params) };
            if (!ctx) {
                reason = "PHONE_LOCAL_CONTEXT_ALLOC";
                return false;
            }
            ggml_phone_qwen3::graph_result built {};
            if (!ggml_phone_qwen3::build_graph(
                    ctx.get(), plan, chunk, weights, built, why)) {
                reason = std::string("PHONE_LOCAL_BUILD:") + why;
                return false;
            }
            ggml_phone_qwen3::side_inputs input {};
            if (!ggml_phone_qwen3::make_side_inputs(
                    plan, chunk, input, why)) {
                reason = std::string("PHONE_LOCAL_SIDE_INPUTS:") + why;
                return false;
            }
            size_t set_rows = 0;
            size_t flash = 0;
            size_t rope = 0;
            size_t glu = 0;
            for (int i = 0; i < ggml_graph_n_nodes(built.graph); ++i) {
                const auto * op = ggml_graph_node(built.graph, i);
                set_rows += op->op == GGML_OP_SET_ROWS ? 1 : 0;
                flash += op->op == GGML_OP_FLASH_ATTN_EXT ? 1 : 0;
                rope += op->op == GGML_OP_ROPE ? 1 : 0;
                glu += op->op == GGML_OP_GLU ? 1 : 0;
            }
            if (set_rows != 2 || flash != 1 || rope != 2 || glu != 1 ||
                    input.positions.size() != built.tokens ||
                    input.k_indices.size() != built.tokens ||
                    input.v_indices.size() != built.tokens ||
                    input.causal_mask_f16.size() !=
                        uint64_t(plan.n_kv) * built.tokens) {
                reason = "PHONE_LOCAL_OPERATOR_OR_MASK_PARITY";
                return false;
            }
            ++stats.chunks;
            stats.total_nodes += ggml_graph_n_nodes(built.graph);
            stats.kv_writes += set_rows;
            stats.flash_nodes += flash;
            stats.rope_nodes += rope;
        }
        reason.clear();
        return true;
    }

    struct local_htp_prepare_stats {
        uint32_t chunks = 0;
        size_t graph_nodes = 0;
        size_t kv_bytes = 0;
        size_t graph_storage_bytes = 0;
        double precompile_ms = 0.0;
    };

    // KV is allocated once for ALL selected NPU Regions, including those
    // whose weight Bank is not ready yet. It survives Region graph retirement.
    bool reserve_owned_htp_kv(const ggml_phone_owned::region_plan & plan,
                              std::string & reason) {
        const char * why = "NONE";
        if (!ggml_phone_owned::validate(plan, why)) {
            reason = why;
            return false;
        }
        const auto key = std::make_pair(plan.run_id, plan.region);
        const auto existing = owned_kv_.find(key);
        if (existing != owned_kv_.end()) {
            if (!ggml_phone_owned::equal(existing->second->plan, plan)) {
                reason = "PHONE_OWNED_HTP_PLAN_CONFLICT";
                return false;
            }
            reason.clear();
            return true;
        }
        auto kv = std::make_unique<owned_layer_kv>();
        kv->plan = plan;
        const size_t depth = plan.layer_end - plan.layer;
        const size_t kv_ctx_bytes =
            ggml_tensor_overhead() * (8 + 2 * depth) + 4096;
        ggml_init_params kv_params = {kv_ctx_bytes, nullptr, true};
        kv->ctx.reset(ggml_init(kv_params));
        if (!kv->ctx) {
            reason = "PHONE_OWNED_HTP_KV_CONTEXT";
            return false;
        }
        const int64_t width = int64_t(plan.n_heads_kv) * plan.head_dim;
        for (uint32_t il = plan.layer; il < plan.layer_end; ++il) {
            auto * k = ggml_new_tensor_2d(
                kv->ctx.get(), GGML_TYPE_F16, width, plan.kv_capacity);
            auto * v = ggml_new_tensor_2d(
                kv->ctx.get(), GGML_TYPE_F16, width, plan.kv_capacity);
            if (!k || !v) {
                reason = "PHONE_OWNED_HTP_KV_METADATA";
                return false;
            }
            char name[GGML_MAX_NAME] = {};
            std::snprintf(name, sizeof(name), "cache_k_l%u", il);
            ggml_set_name(k, name);
            std::snprintf(name, sizeof(name), "cache_v_l%u", il);
            ggml_set_name(v, name);
            kv->layer_kv.emplace_back(k, v);
        }
        kv->k = kv->layer_kv.front().first;
        kv->v = kv->layer_kv.front().second;
        kv->canonical_k_layers.resize(depth, nullptr);
        kv->canonical_v_layers.resize(depth, nullptr);
        kv->buffer.reset(ggml_backend_alloc_ctx_tensors(
            kv->ctx.get(), backend_));
        if (!kv->buffer) {
            reason = "PHONE_OWNED_HTP_KV_ALLOC";
            return false;
        }
        ggml_backend_buffer_clear(kv->buffer.get(), 0);
        owned_kv_.emplace(key, std::move(kv));
        reason.clear();
        return true;
    }

    // Opt-in precompile stage. The Phone allocates KV once per assigned Layer,
    // fixes every Chunk's input/side-input/output Tensor storage, then invokes
    // the same HTP fixed-graph precompiler used by PC-built Graphs. No PC
    // Graph is consulted. This prepares graphs only: input PUSH and P2C still
    // follow the legacy path until a later cutover.
    //
    // Caller holds phone_npu_backend_mutex, and may not call this while a
    // different RUN is executing. Stream reset clears these plans first.
    bool prepare_owned_htp_graphs(
            const ggml_phone_owned::region_plan & plan,
            std::string & reason,
            local_htp_prepare_stats & stats) {
        stats = {};
        const char * why = "NONE";
        if (!ggml_phone_owned::validate(plan, why)) {
            reason = why;
            return false;
        }
        if (!fixed_graph_cache_supported()) {
            reason = "PHONE_OWNED_HTP_FIXED_CACHE_UNAVAILABLE";
            return false;
        }
        const auto key = std::make_pair(plan.run_id, plan.region);
        auto existing = owned_kv_.find(key);
        if (existing != owned_kv_.end()) {
            if (!ggml_phone_owned::equal(existing->second->plan, plan)) {
                reason = "PHONE_OWNED_HTP_PLAN_CONFLICT";
                return false;
            }
            size_t compiled = 0;
            for (uint32_t chunk = 0;
                    chunk < ggml_phone_owned::chunk_count(plan); ++chunk) {
                compiled += owned_graphs_.count(
                    std::make_tuple(plan.run_id, plan.region, chunk));
            }
            if (compiled == ggml_phone_owned::chunk_count(plan)) {
                stats.chunks = static_cast<uint32_t>(compiled);
                return true;
            }
            if (compiled != 0) {
                reason = "PHONE_OWNED_HTP_PARTIAL_REGISTRATION";
                return false;
            }
            // Deferred Bank: its KV already exists, but no Graph is built
            // until the previous rank-k Region drains and rank-k+2 refills.
        }

        std::vector<ggml_phone_qwen3::layer_weights> all_weights;
        all_weights.reserve(plan.layer_end - plan.layer);
        for (uint32_t il = plan.layer; il < plan.layer_end; ++il) {
            const auto load = [&](const char * suffix) -> ggml_tensor * {
                char name[GGML_MAX_NAME] = {};
                std::snprintf(name, sizeof(name), "blk.%u.%s", il, suffix);
                for (const auto & entry : model_weights_) {
                    ggml_tensor * t = entry.second->tensor;
                    if (t && std::strcmp(t->name, name) == 0 &&
                            t->buffer && t->data) {
                        return t;
                    }
                }
                return nullptr;
            };
            ggml_phone_qwen3::layer_weights w {};
            w.attn_norm = load("attn_norm.weight");
            w.wq = load("attn_q.weight");
            w.wk = load("attn_k.weight");
            w.wv = load("attn_v.weight");
            w.wo = load("attn_output.weight");
            w.ffn_norm = load("ffn_norm.weight");
            w.ffn_gate = load("ffn_gate.weight");
            w.ffn_up = load("ffn_up.weight");
            w.ffn_down = load("ffn_down.weight");
            w.q_norm = load("attn_q_norm.weight");
            w.k_norm = load("attn_k_norm.weight");
            ggml_phone_owned::region_plan one = plan;
            one.layer = il;
            if (!ggml_phone_qwen3::validate_weights(one, w, why)) {
                reason = std::string("PHONE_OWNED_HTP_WEIGHTS:") + why;
                return false;
            }
            all_weights.push_back(w);
        }

        if (!reserve_owned_htp_kv(plan, reason)) {
            return false;
        }
        auto * kv = owned_kv_.at(key).get();
        stats.kv_bytes = ggml_backend_buffer_get_size(kv->buffer.get());

        const auto reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(backend_));
        using compile_fn = bool (*)(ggml_backend_t, ggml_cgraph *);
        const auto compile = reg == nullptr ? nullptr :
            reinterpret_cast<compile_fn>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_hexagon_precompile_fixed_graph"));
        if (compile == nullptr) {
            reason = "PHONE_OWNED_HTP_COMPILE_PROC";
            return false;
        }
        // Holding temporary Graphs through the entire loop ensures a
        // successful HTP compilation never sees prematurely released memory.
        std::vector<std::unique_ptr<owned_chunk_graph>> pending;
        pending.reserve(ggml_phone_owned::chunk_count(plan));
        const size_t depth = plan.layer_end - plan.layer;
        const size_t graph_ctx_bytes =
            ggml_tensor_overhead() * (1024 * depth + 32) +
            ggml_graph_overhead_custom(512 * depth, false) + 4096;
        for (uint32_t chunk = 0;
                chunk < ggml_phone_owned::chunk_count(plan); ++chunk) {
            auto item = std::make_unique<owned_chunk_graph>();
            ggml_init_params params = {graph_ctx_bytes, nullptr, true};
            item->ctx.reset(ggml_init(params));
            if (!item->ctx) {
                reason = "PHONE_OWNED_HTP_GRAPH_CONTEXT";
                break;
            }
            ggml_phone_qwen3::multi_graph_result built_stage {};
            if (!ggml_phone_qwen3::build_graph_layers(
                    item->ctx.get(), plan, chunk, all_weights,
                    built_stage, why, kv->layer_kv)) {
                reason = std::string("PHONE_OWNED_HTP_BUILD:") + why;
                break;
            }
            item->all_layer_graphs = std::move(built_stage.layers);
            item->built = item->all_layer_graphs.front();
            item->built.graph = built_stage.graph;
            item->built.input = built_stage.input;
            item->built.output = built_stage.output;
            item->buffer.reset(ggml_backend_alloc_ctx_tensors(
                item->ctx.get(), backend_));
            if (!item->buffer) {
                reason = "PHONE_OWNED_HTP_GRAPH_ALLOC";
                break;
            }
            bool side_ready = true;
            for (size_t il = 0; il < item->all_layer_graphs.size(); ++il) {
                ggml_phone_owned::region_plan layer_plan = plan;
                layer_plan.layer += (uint32_t) il;
                ggml_phone_qwen3::side_inputs side {};
                if (!ggml_phone_qwen3::make_side_inputs(
                        layer_plan, chunk, side, why)) {
                    reason = std::string("PHONE_OWNED_HTP_INPUTS:") + why;
                    side_ready = false;
                    break;
                }
                const auto & g = item->all_layer_graphs[il];
                ggml_backend_tensor_set(
                    g.positions, side.positions.data(), 0,
                    side.positions.size() * sizeof(int32_t));
                ggml_backend_tensor_set(
                    g.k_indices, side.k_indices.data(), 0,
                    side.k_indices.size() * sizeof(int64_t));
                ggml_backend_tensor_set(
                    g.v_indices, side.v_indices.data(), 0,
                    side.v_indices.size() * sizeof(int64_t));
                ggml_backend_tensor_set(
                    g.mask, side.causal_mask_f16.data(), 0,
                    side.causal_mask_f16.size() * sizeof(uint16_t));
            }
            if (!side_ready) { break; }
            // Ensure fixed-position/mask bytes are visible to any backend
            // compile-time specialization before registering this graph.
            ggml_backend_synchronize(backend_);
            bool supported = true;
            for (int i = 0; i < ggml_graph_n_nodes(item->built.graph); ++i) {
                if (!ggml_backend_supports_op(
                        backend_, ggml_graph_node(item->built.graph, i))) {
                    supported = false;
                    reason = std::string("PHONE_OWNED_HTP_UNSUPPORTED_OP:") +
                        ggml_graph_node(item->built.graph, i)->name;
                    break;
                }
            }
            if (!supported) { break; }
            if (next_owned_uid_ == 0 ||
                    next_owned_uid_ >= (UINT64_C(1) << 62)) {
                reason = "PHONE_OWNED_HTP_GRAPH_UID_EXHAUSTED";
                break;
            }
            item->built.graph->uid = (UINT64_C(3) << 62) |
                next_owned_uid_++;
            if (banked_mode_) {
                const int rank = rank_for_region(
                    plan.region, plan.layer);
                if (rank < 0 || !rank_banked_ready(plan.region)) {
                    reason = "NPU_BANK_PRECOMPILE_BEFORE_WEIGHT_READY";
                    break;
                }
                item->bank_generation =
                    weight_banks_.bank(size_t(rank) % 2).generation;
            }
            const int64_t t0 = ggml_time_us();
            const bool compiled = compile(backend_, item->built.graph);
            stats.precompile_ms += (ggml_time_us() - t0) / 1000.0;
            if (!compiled) {
                reason = "PHONE_OWNED_HTP_COMPILE_FAILED";
                break;
            }
            ++stats.chunks;
            stats.graph_nodes += ggml_graph_n_nodes(item->built.graph);
            stats.graph_storage_bytes +=
                ggml_backend_buffer_get_size(item->buffer.get());
            pending.push_back(std::move(item));
        }
        if (stats.chunks != ggml_phone_owned::chunk_count(plan)) {
            // The optional precompiler caches Tensor addresses. Invalidate
            // cached plans before releasing any partially built objects.
            using clear_fn = void (*)(ggml_backend_t);
            auto clear = reinterpret_cast<clear_fn>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_hexagon_clear_fixed_graph_cache"));
            if (clear) { clear(backend_); }
            if (reason.empty()) { reason = "PHONE_OWNED_HTP_PARTIAL"; }
            return false;
        }
        for (uint32_t chunk = 0; chunk < pending.size(); ++chunk) {
            owned_graphs_.emplace(
                std::make_tuple(plan.run_id, plan.region, chunk),
                std::move(pending[chunk]));
        }
        reason.clear();
        return true;
    }


    // Called AFTER the corresponding output frame is safely published to
    // P2C. The caller holds HTP mutex, and all Graph compute is serialized.
    // The old Bank cannot be reused while any Chunk is in flight.
    bool finish_banked_chunk(uint64_t run_id, uint32_t region,
                             uint32_t chunk,
                             std::optional<weight_bank_refill> & refill,
                             std::string & reason) {
        refill.reset();
        reason.clear();
        if (!banked_mode_) { return true; }
        const auto it = owned_kv_.find({run_id, region});
        if (it == owned_kv_.end()) {
            reason = "NPU_BANK_KV_REGION_NOT_REGISTERED";
            return false;
        }
        const int rank = rank_for_region(region, it->second->plan.layer);
        if (rank < 0) {
            reason = "NPU_BANK_REGION_LAYER_MISMATCH";
            return false;
        }
        const size_t r = static_cast<size_t>(rank);
        const auto & b = weight_banks_.bank(r % 2);
        if (b.state != two_weight_banks::phase::COMPUTING ||
                b.rank != r || b.next_chunk != chunk ||
                it->second->next_chunk != chunk + 1) {
            reason = "NPU_BANK_CHUNK_NOT_COMPUTED_OR_UNPUBLISHED";
            return false;
        }
        const bool last =
            chunk + 1 == ggml_phone_owned::chunk_count(it->second->plan);
        refill = weight_banks_.finish_chunk(r, chunk, last);
        if (weight_banks_.bank(r % 2).next_chunk != chunk + 1) {
            reason = "NPU_BANK_FINISH_STATE_INVALID";
            return false;
        }
        return true;
    }

    // Serialized HTP critical section, called by the independent Bank Mover
    // after rank k has fully drained and published P2C. Only cache entries
    // for k are removed; Bank B's graph metadata and KV remain valid.
    bool refill_banked_rank(
            uint64_t run_id, const weight_bank_refill & job,
            std::string & reason, model_load_stats & load_stats,
            local_htp_prepare_stats & compile_stats,
            const std::vector<std::vector<uint8_t>> * staged = nullptr) {
        reason.clear();
        load_stats = {};
        compile_stats = {};
        if (!banked_mode_ ||
                job.from_rank >= banked_layers_.size() ||
                job.to_rank >= banked_layers_.size() ||
                job.bank >= 2 ||
                job.bank != job.from_rank % 2 ||
                job.bank != job.to_rank % 2) {
            reason = "NPU_BANK_INVALID_REFILL_JOB";
            return false;
        }
        const auto old_region = banked_layers_[job.from_rank].region;
        const auto next_region = banked_layers_[job.to_rank].region;
        const auto old_it = owned_kv_.find({run_id, old_region});
        const auto next_it = owned_kv_.find({run_id, next_region});
        if (old_it == owned_kv_.end() || next_it == owned_kv_.end() ||
                old_it->second->next_chunk !=
                    ggml_phone_owned::chunk_count(old_it->second->plan) ||
                old_it->second->plan.layer !=
                    banked_layers_[job.from_rank].layer ||
                next_it->second->plan.layer !=
                    banked_layers_[job.to_rank].layer ||
                next_it->second->next_chunk != 0) {
            reason = "NPU_BANK_REFILL_PLAN_OR_KV_NOT_READY";
            return false;
        }
        ggml_backend_synchronize(backend_);
        const auto reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(backend_));
        using evict_fn = bool (*)(ggml_backend_t, uint64_t);
        const auto evict = reg == nullptr ? nullptr :
            reinterpret_cast<evict_fn>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_hexagon_evict_fixed_graph_uid"));
        if (!evict) {
            reason = "NPU_BANK_NO_TARGETED_GRAPH_EVICTION";
            return false;
        }
        size_t retired = 0;
        for (auto it = owned_graphs_.begin();
                it != owned_graphs_.end();) {
            if (std::get<0>(it->first) != run_id ||
                    std::get<1>(it->first) != old_region) {
                ++it;
                continue;
            }
            const auto * g = it->second->built.graph;
            if (!g || !evict(backend_, g->uid)) {
                reason = "NPU_BANK_GRAPH_RETIRE_FAILED";
                return false;
            }
            it = owned_graphs_.erase(it);
            ++retired;
        }
        if (retired !=
                ggml_phone_owned::chunk_count(old_it->second->plan) ||
                !weight_banks_.graphs_retired(job.from_rank) ||
                !weight_banks_.begin_refill(job)) {
            reason = "NPU_BANK_DRAIN_OR_REFILL_TRANSITION";
            return false;
        }
        // Invalidate Tensor metadata only AFTER no compiled Graph references
        // its addresses. The physical A/B mapping is not freed or remapped.
        for (const auto & key : banked_layers_[job.from_rank].keys) {
            if (model_weights_.erase(key) != 1) {
                reason = "NPU_BANK_OLD_WEIGHT_NOT_RESIDENT";
                return false;
            }
        }
        banked_layers_[job.from_rank].keys.clear();
        if (!load_banked_rank(job.to_rank, reason, load_stats, staged)) {
            return false;
        }
        if (!weight_banks_.refill_complete(job.to_rank, true)) {
            reason = "NPU_BANK_REFILL_READY_TRANSITION_FAILED";
            return false;
        }
        // All precompiled Graphs point into the freshly repacked Bank, with
        // this generation's UID and the unchanged per-layer KV storage.
        if (!prepare_owned_htp_graphs(
                next_it->second->plan, reason, compile_stats)) {
            return false;
        }
        return true;
    }

    // New RUN reset: after all prior HTP users have left, restore the first
    // two resident Regions (A=rank0, B=rank1) for repeat experiments. The
    // fixed arenas are not freed. Must follow reset_kv() under HTP lock.
    bool reset_banked_initial_weights(std::string & reason,
                                      model_load_stats & stats) {
        reason.clear();
        stats = {};
        if (!banked_mode_) { return true; }
        if (banked_layers_.size() < 2 ||
                !owned_graphs_.empty() || !owned_kv_.empty()) {
            reason = "NPU_BANK_RESET_REQUIRES_NO_RUNNING_GRAPH_OR_KV";
            return false;
        }
        if (rank_banked_ready(banked_layers_[0].region) &&
                rank_banked_ready(banked_layers_[1].region)) {
            return true;
        }
        ggml_backend_synchronize(backend_);
        model_weights_.clear();
        for (auto & item : banked_layers_) { item.keys.clear(); }
        std::vector<weight_bank_region> regions;
        for (const auto & item : banked_layers_) {
            regions.push_back({item.region, static_cast<int>(item.layer),
                static_cast<int>(item.layer + 1), item.bytes});
        }
        weight_banks_.reset();
        if (!weight_banks_.configure(regions, 0, 0,
                std::numeric_limits<size_t>::max(), reason)) {
            return false;
        }
        for (size_t rank = 0; rank < 2; ++rank) {
            model_load_stats delta {};
            if (!load_banked_rank(rank, reason, delta) ||
                    !weight_banks_.prime_complete(rank)) {
                if (reason.empty()) {
                    reason = "NPU_BANK_REPRIME_FAILED";
                }
                return false;
            }
            stats.new_weights += delta.new_weights;
            stats.new_bytes += delta.new_bytes;
            stats.file_read_ms += delta.file_read_ms;
            stats.htp_repack_ms += delta.htp_repack_ms;
        }
        return true;
    }

    // Bind canonical Decode storage once, independently of any PC Graph.
    // Caller holds HTP mutex before the Graph mutex.
    bool bind_owned_decode_kv_storage(
            uint64_t run_id, uint32_t region, uint32_t layer,
            ggml_context_ptr canonical_ctx,
            ggml_tensor * canonical_k, ggml_tensor * canonical_v,
            bool identity_slots, std::string & reason) {
        reason.clear();
        const auto it = owned_kv_.find({run_id, region});
        if (it == owned_kv_.end() || !canonical_ctx || !identity_slots) {
            reason = "PHONE_LOCAL_KV_GRAPHLESS_PLAN_OR_SLOTS";
            return false;
        }
        auto & state = *it->second;
        const auto & p = state.plan;
        const uint64_t width = uint64_t(p.n_heads_kv) * p.head_dim;
        const auto valid = [&](const ggml_tensor * t) {
            return t && t->buffer && t->data &&
                t->type == GGML_TYPE_F16 && t->op == GGML_OP_NONE &&
                ggml_is_contiguous(t) && t->ne[0] == int64_t(width) &&
                t->ne[1] >= p.total_tokens && t->ne[1] <= p.kv_capacity &&
                t->ne[2] == 1 && t->ne[3] == 1 &&
                ggml_nbytes(t) >= width * p.total_tokens * sizeof(uint16_t) &&
                !ggml_backend_buffer_is_host(t->buffer);
        };
        if (layer < p.layer || layer >= p.layer_end ||
                state.next_chunk != 0 ||
                !valid(canonical_k) || !valid(canonical_v) ||
                (canonical_k->buffer == canonical_v->buffer &&
                 canonical_k->data == canonical_v->data)) {
            reason = "PHONE_LOCAL_KV_GRAPHLESS_DESTINATION_CONTRACT";
            return false;
        }
        const size_t index = layer - p.layer;
        if (index >= state.canonical_k_layers.size() ||
                state.canonical_k_layers[index] ||
                state.canonical_v_layers[index]) {
            reason = "PHONE_LOCAL_KV_DUPLICATE_LAYER_BINDING";
            return false;
        }
        state.canonical_k_layers[index] = canonical_k;
        state.canonical_v_layers[index] = canonical_v;
        if (index == 0) {
            state.canonical_ctx = std::move(canonical_ctx);
            state.canonical_k = canonical_k;
            state.canonical_v = canonical_v;
        } else {
            state.canonical_extra_contexts.push_back(
                std::move(canonical_ctx));
        }
        state.graphless_identity_slots = true;
        for (size_t i = 0; i < state.canonical_k_layers.size(); ++i) {
            if (!state.canonical_k_layers[i] ||
                    !state.canonical_v_layers[i]) {
                state.graphless_identity_slots = false;
                break;
            }
        }
        return true;
    }

    bool all_owned_decode_kv_bound(
            uint64_t run_id, uint32_t region) const {
        const auto it = owned_kv_.find({run_id, region});
        return it != owned_kv_.end() &&
            it->second->graphless_identity_slots;
    }

    // The PC-built Graph remains registered as a metadata carrier in the
    // first local-only phase. Extract ONLY its canonical OpenCL KV leaves,
    // so Phone-owned HTP KV can be copied to the original Decode cache when
    // the input-stream RUN ends. No PC-built Graph executes in this mode.
    //
    // Caller holds HTP before graph_compute_mutex; both KV pointers refer
    // to immutable per-RUN graph metadata while their GPU buffers remain
    // owned by the RPC server until RUN teardown.
    bool bind_owned_decode_kv(uint64_t run_id, uint32_t region,
                              uint32_t chunk, ggml_cgraph * source,
                              std::string & reason) {
        reason.clear();
        auto it = owned_kv_.find(std::make_pair(run_id, region));
        auto chunk_it = owned_graphs_.find(
            std::make_tuple(run_id, region, chunk));
        if (it == owned_kv_.end() ||
                chunk_it == owned_graphs_.end() || source == nullptr) {
            reason = "PHONE_LOCAL_KV_PLAN_OR_GRAPH_MISSING";
            return false;
        }
        auto & state = *it->second;
        const auto & p = state.plan;
        const std::string kn = "cache_k_l" + std::to_string(p.layer);
        const std::string vn = "cache_v_l" + std::to_string(p.layer);
        const auto matches_name = [](const ggml_tensor * t,
                                     const std::string & name) {
            if (!t) { return false; }
            const std::string n(t->name);
            size_t pos = n.find(name);
            while (pos != std::string::npos) {
                const bool left = pos == 0 || n[pos - 1] == '#';
                const size_t after = pos + name.size();
                const bool right = after == n.size() || n[after] == '#';
                if (left && right) { return true; }
                pos = n.find(name, pos + 1);
            }
            return false;
        };
        const ggml_tensor * canonical_k = nullptr;
        const ggml_tensor * canonical_v = nullptr;
        bool ambiguous = false;
        std::unordered_set<const ggml_tensor *> visited;
        std::function<void(const ggml_tensor *)> scan =
            [&](const ggml_tensor * t) {
                if (!t || !visited.insert(t).second) { return; }
                if (t->op == GGML_OP_NONE && t->view_src == nullptr) {
                    const auto assign = [&](const ggml_tensor * & dst) {
                        if (dst && (dst->buffer != t->buffer ||
                                    dst->data != t->data ||
                                    !ggml_are_same_layout(dst, t))) {
                            ambiguous = true;
                        } else {
                            dst = t;
                        }
                    };
                    if (matches_name(t, kn)) { assign(canonical_k); }
                    if (matches_name(t, vn)) { assign(canonical_v); }
                }
                if (t->view_src) { scan(t->view_src); }
                for (const ggml_tensor * src : t->src) { scan(src); }
            };
        for (int i = 0; i < ggml_graph_n_nodes(source); ++i) {
            scan(ggml_graph_node(source, i));
        }
        const uint64_t width =
            uint64_t(p.n_heads_kv) * p.head_dim;
        const uint64_t total_bytes =
            width * p.total_tokens * sizeof(uint16_t);
        const auto valid = [&](const ggml_tensor * t) {
            return t && t->buffer && t->data &&
                t->type == GGML_TYPE_F16 &&
                t->ne[0] == static_cast<int64_t>(width) &&
                t->ne[1] >= static_cast<int64_t>(p.total_tokens) &&
                t->ne[2] == 1 && t->ne[3] == 1 &&
                ggml_is_contiguous(t) &&
                ggml_nbytes(t) >= total_bytes &&
                ggml_backend_buffer_is_host(t->buffer) == false;
        };
        if (ambiguous || !valid(canonical_k) || !valid(canonical_v) ||
                (canonical_k->buffer == canonical_v->buffer &&
                 canonical_k->data == canonical_v->data)) {
            reason = ambiguous ?
                "PHONE_LOCAL_KV_AMBIGUOUS_CANONICAL" :
                "PHONE_LOCAL_KV_CANONICAL_LAYOUT_MISMATCH";
            return false;
        }
        // Extract the exact slot indices from the reference PC graph.
        // The local-only builder assumes KV slot i == absolute position i,
        // but a KV allocator may reuse a nonzero slot even when RoPE
        // positions start at zero. We must validate real PC KV indices
        // per Chunk BEFORE running the local HTP Graph or copying KV back.
        const ggml_tensor * indices_k = nullptr;
        const ggml_tensor * indices_v = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(source); ++i) {
            const ggml_tensor * op = ggml_graph_node(source, i);
            if (!op || op->op != GGML_OP_SET_ROWS ||
                    !op->src[1] || !op->src[2]) {
                continue;
            }
            const ggml_tensor * base = op->src[2];
            // Fresh single-stream Qwen3 cpy_k/cpy_v writes directly
            // to the backing KV leaf; reject additional transforms.
            const ggml_tensor ** dst = nullptr;
            if (base == canonical_k) { dst = &indices_k; }
            if (base == canonical_v) { dst = &indices_v; }
            if (!dst) { continue; }
            if (*dst != nullptr && *dst != op->src[1]) {
                reason = "PHONE_LOCAL_KV_AMBIGUOUS_SLOT_INDICES";
                return false;
            }
            *dst = op->src[1];
        }
        if (chunk >= ggml_phone_owned::chunk_count(p)) {
            reason = "PHONE_LOCAL_KV_CHUNK_OUT_OF_RANGE";
            return false;
        }
        const uint32_t expected_tokens =
            std::min<uint32_t>(
                p.chunk_tokens, p.total_tokens - chunk * p.chunk_tokens);
        const auto valid_indices = [&](const ggml_tensor * t) {
            return t && t->buffer && t->data &&
                t->type == GGML_TYPE_I64 &&
                ggml_is_contiguous(t) &&
                ggml_nelements(t) == expected_tokens;
        };
        if (chunk >= ggml_phone_owned::chunk_count(p) ||
                !valid_indices(indices_k) ||
                !valid_indices(indices_v)) {
            reason = "PHONE_LOCAL_KV_SLOT_INDICES_UNAVAILABLE";
            return false;
        }
        const auto same_binding = [](const ggml_tensor * a,
                                      const ggml_tensor * b) {
            return !a || (a->buffer == b->buffer && a->data == b->data &&
                          ggml_are_same_layout(a, b));
        };
        if (!same_binding(state.canonical_k, canonical_k) ||
                !same_binding(state.canonical_v, canonical_v)) {
            reason = "PHONE_LOCAL_KV_CANONICAL_BINDING_CHANGED";
            return false;
        }
        // Rebind to the most recently registered Chunk's equivalent
        // metadata; old RPC graph metadata may be recycled earlier.
        state.canonical_k = canonical_k;
        state.canonical_v = canonical_v;
        chunk_it->second->canonical_k_indices = indices_k;
        chunk_it->second->canonical_v_indices = indices_v;
        return true;
    }

    // Copy every fully-computed Phone-owned local KV prefix to the
    // authoritative RPC-visible OpenCL cache before decode begins.
    // This is the ONLY KV handoff required for the fresh, 0-based,
    // single-sequence prefill plan validated by region_plan v2.
    bool synchronize_owned_decode_kv(uint64_t run_id,
                                     std::string & reason,
                                     size_t & copied_bytes,
                                     size_t & regions) {
        reason.clear();
        copied_bytes = 0;
        regions = 0;
        std::vector<uint16_t> scratch;
        ggml_backend_synchronize(backend_);
        for (const auto & [key, item] : owned_kv_) {
            if (key.first != run_id) { continue; }
            const auto & state = *item;
            const auto & p = state.plan;
            if (state.next_chunk != ggml_phone_owned::chunk_count(p) ||
                    !state.graphless_identity_slots ||
                    state.layer_kv.size() !=
                        size_t(p.layer_end - p.layer) ||
                    state.canonical_k_layers.size() !=
                        state.layer_kv.size() ||
                    state.canonical_v_layers.size() !=
                        state.layer_kv.size()) {
                reason = "PHONE_LOCAL_KV_INCOMPLETE_PREFILL_OR_BINDING";
                return false;
            }
            const uint64_t width =
                uint64_t(p.n_heads_kv) * p.head_dim;
            const uint64_t bytes =
                width * p.total_tokens * sizeof(uint16_t);
            if (bytes > ggml_phone_owned::k_max_activation_bytes ||
                    bytes == 0 || bytes > SIZE_MAX) {
                reason = "PHONE_LOCAL_KV_SYNC_SIZE_MISMATCH";
                return false;
            }
            scratch.resize(static_cast<size_t>(bytes / sizeof(uint16_t)));
            for (size_t i = 0; i < state.layer_kv.size(); ++i) {
                const auto & pair = state.layer_kv[i];
                const auto * canonical_k = state.canonical_k_layers[i];
                const auto * canonical_v = state.canonical_v_layers[i];
                if (!canonical_k || !canonical_v ||
                        !pair.first || !pair.second ||
                        ggml_nbytes(pair.first) < bytes ||
                        ggml_nbytes(pair.second) < bytes ||
                        ggml_nbytes(canonical_k) < bytes ||
                        ggml_nbytes(canonical_v) < bytes) {
                    reason = "PHONE_LOCAL_KV_SYNC_LAYER_SHAPE";
                    return false;
                }
                for (const auto & entry : {
                        std::make_pair(pair.first, canonical_k),
                        std::make_pair(pair.second, canonical_v) }) {
                    ggml_backend_tensor_get(
                        entry.first, scratch.data(), 0, size_t(bytes));
                    ggml_backend_tensor_set(
                        const_cast<ggml_tensor *>(entry.second),
                        scratch.data(), 0, size_t(bytes));
                    copied_bytes += size_t(bytes);
                }
            }
            ++regions;
        }
        if (regions == 0) {
            reason = "PHONE_LOCAL_KV_NO_REGIONS";
            return false;
        }
        return true;
    }

    struct local_only_profile {
        double input_read_ms = 0.0;
        double input_upload_ms = 0.0;
        double htp_compute_ms = 0.0;
        double output_copy_ms = 0.0;
        uint32_t tokens = 0;
        uint32_t layer = 0;
    };

    // Real Phone-local-only execution. The old PC-built Graph supplies
    // the native OpenCL input/output allocation contract, but NO legacy
    // HTP Graph is computed, compiled or copied. The Phone-owned HTP
    // Region KV is the sole prefill cache and is synchronized at RUN END.
    bool run_owned_only_chunk(uint64_t run_id, uint32_t region,
                              uint32_t chunk,
                              const ggml_tensor * input,
                              ggml_tensor * output,
                              std::string & reason,
                              local_only_profile & profile,
                              std::vector<float> * direct_p2c_payload = nullptr,
                              const float * direct_input = nullptr,
                              size_t direct_input_floats = 0) {
        reason.clear();
        profile = {};
        const auto key = std::make_pair(run_id, region);
        const auto kit = owned_kv_.find(key);
        const auto git = owned_graphs_.find(
            std::make_tuple(run_id, region, chunk));
        if (kit == owned_kv_.end() || git == owned_graphs_.end()) {
            reason = "PHONE_LOCAL_ONLY_GRAPH_NOT_PREPARED";
            return false;
        }
        auto & state = *kit->second;
        const auto & p = state.plan;
        const auto & g = git->second->built;
        if (state.next_chunk != chunk ||
                chunk >= ggml_phone_owned::chunk_count(p) ||
                g.token_begin != uint64_t(chunk) * p.chunk_tokens ||
                g.tokens == 0 ||
                g.tokens != std::min<uint32_t>(
                    p.chunk_tokens, p.total_tokens - g.token_begin) ||
                g.cache_k != state.k || g.cache_v != state.v ||
                !state.canonical_k || !state.canonical_v) {
            reason = "PHONE_LOCAL_ONLY_KV_FIFO_OR_BINDING_MISMATCH";
            return false;
        }
        // Fail closed unless the PC's actual KV cache slot mapping is
        // exactly the fresh monotonic mapping described by Plan v2.
        // The indices are populated dynamically and are not trustworthy
        // until the OpenCL input graph has been prepared and synchronized.
        const auto & chunk_state = *git->second;
        if (state.graphless_identity_slots) {
            // The PC allocator validated every physical slot for this
            // immutable fresh UBatch when it bound Decode K/V storage.
            if (!state.canonical_ctx) {
                reason = "PHONE_LOCAL_ONLY_GRAPHLESS_KV_CONTEXT_MISSING";
                return false;
            }
        } else {
            // Preserve the old per-Chunk I64 index check when PC Graphs
            // are still registered as metadata carriers.
            if (!chunk_state.canonical_k_indices ||
                    !chunk_state.canonical_v_indices) {
                reason = "PHONE_LOCAL_ONLY_KV_INDICES_NOT_BOUND";
                return false;
            }
            std::vector<int64_t> slot_indices(g.tokens);
            for (const ggml_tensor * src : {
                    chunk_state.canonical_k_indices,
                    chunk_state.canonical_v_indices }) {
                ggml_backend_tensor_get(src, slot_indices.data(), 0,
                                        slot_indices.size() * sizeof(int64_t));
                if (!ggml_phone_owned::identity_kv_slots(
                        slot_indices.data(), slot_indices.size(),
                        g.token_begin)) {
                    reason = "PHONE_LOCAL_ONLY_NONCONTIGUOUS_PC_KV_SLOTS";
                    return false;
                }
            }
        }
        const size_t bytes =
            size_t(g.tokens) * p.n_embd * sizeof(float);
        const bool direct = direct_input != nullptr;
        if ((direct && direct_input_floats != bytes / sizeof(float)) ||
                (!direct && direct_input_floats != 0) ||
                !g.input || !g.output ||
                !g.input->buffer || !g.output->buffer ||
                g.input->type != GGML_TYPE_F32 ||
                g.output->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(g.input) ||
                !ggml_is_contiguous(g.output) ||
                ggml_nbytes(g.input) != bytes ||
                ggml_nbytes(g.output) != bytes ||
                (!direct && (!input || !input->buffer || !input->data ||
                    input->type != GGML_TYPE_F32 ||
                    !ggml_is_contiguous(input) ||
                    ggml_nbytes(input) != bytes ||
                    !ggml_are_same_layout(g.input, input))) ||
                (direct_p2c_payload == nullptr &&
                 (!output || !output->buffer || !output->data ||
                  output->type != GGML_TYPE_F32 ||
                  !ggml_is_contiguous(output) ||
                  ggml_nbytes(output) != bytes ||
                  !ggml_are_same_layout(g.output, output)))) {
            reason = "PHONE_LOCAL_ONLY_ACTIVATION_CONTRACT";
            return false;
        }
        // The verified PUSH buffer already contains the exact precompute
        // activation. Avoid a host->OpenCL->host roundtrip in direct mode.
        std::vector<float> activation;
        const float * source = direct_input;
        if (!direct) {
            activation.resize(bytes / sizeof(float));
            const int64_t read_begin = ggml_time_us();
            // Always snapshot before Graph compute: the old path can alias
            // an output allocation with the original input.
            ggml_backend_tensor_get(input, activation.data(), 0, bytes);
            profile.input_read_ms = (ggml_time_us() - read_begin) / 1000.0;
            source = activation.data();
        }
        for (size_t i = 0; i < bytes / sizeof(float); ++i) {
            if (!std::isfinite(source[i])) {
                reason = "PHONE_LOCAL_ONLY_NONFINITE_INPUT";
                return false;
            }
        }

        // The Graph was compiled against THIS Bank generation, not merely
        // a compatible weight address or model-shape. A stale graph cannot
        // be run after another layer replaces its Bank.
        if (banked_mode_) {
            const int rank = rank_for_region(region, p.layer);
            if (rank < 0 || !weight_banks_.begin_chunk(
                    static_cast<size_t>(rank), chunk,
                    chunk_state.bank_generation)) {
                reason = "NPU_BANK_STALE_GRAPH_OR_NOT_READY";
                return false;
            }
        }
        const int64_t upload_begin = ggml_time_us();
        ggml_backend_tensor_set(g.input, source, 0, bytes);
        ggml_backend_synchronize(backend_);
        profile.input_upload_ms = (ggml_time_us() - upload_begin) / 1000.0;
        const int64_t compute_begin = ggml_time_us();
        const ggml_status code =
            ggml_backend_graph_compute(backend_, g.graph);
        ggml_backend_synchronize(backend_);
        profile.htp_compute_ms =
            (ggml_time_us() - compute_begin) / 1000.0;
        if (code != GGML_STATUS_SUCCESS) {
            reason = "PHONE_LOCAL_ONLY_HTP_COMPUTE_FAILED";
            return false;
        }
        if (direct_p2c_payload != nullptr) {
            // The HTP output is the authoritative payload. Read it once
            // into owned host storage and hand it directly to the P2C Arena:
            // no PC Graph output lookup or HTP->OpenCL->Host bounce.
            const int64_t read_begin = ggml_time_us();
            direct_p2c_payload->resize(bytes / sizeof(float));
            ggml_backend_tensor_get(
                g.output, direct_p2c_payload->data(), 0, bytes);
            profile.output_copy_ms =
                (ggml_time_us() - read_begin) / 1000.0;
            for (float value : *direct_p2c_payload) {
                if (!std::isfinite(value)) {
                    direct_p2c_payload->clear();
                    reason = "PHONE_LOCAL_DIRECT_P2C_NONFINITE_OUTPUT";
                    return false;
                }
            }
        } else {
            double copy_ms = 0.0;
            if (!publish_owned_htp_chunk(
                    run_id, region, chunk, output, reason, copy_ms)) {
                return false;
            }
            profile.output_copy_ms = copy_ms;
        }
        profile.tokens = g.tokens;
        profile.layer = p.layer;
        ++state.next_chunk;
        return true;
    }

    struct local_exec_profile {
        double input_read_ms = 0.0;
        double input_upload_ms = 0.0;
        double compute_ms = 0.0;
        double compare_ms = 0.0;
        ggml_phone_qwen3::parity_stats parity {};
        ggml_phone_qwen3::feature_error_stats feature {};
        uint32_t tokens = 0;
        uint32_t layer = 0;
    };

    // REAL HTP execution of the Phone-constructed graph. The activation
    // comes from an immutable snapshot captured from the actual baseline
    // HTP-bound input BEFORE baseline Compute can overwrite its source.
    // The caller still holds the HTP mutex.
    // This path has its OWN KV and DOES NOT change the canonical OpenCL
    // output or the legacy HTP KV: the old path remains the P2C producer.
    // A/B measurements cover the same input activation and the complete
    // per-region FIFO sequence (including dependencies on earlier KV rows).
    bool run_owned_htp_chunk(uint64_t run_id, uint32_t region, uint32_t chunk,
                             const float * reference_input_snapshot,
                             size_t reference_input_count,
                             const ggml_tensor * reference_output,
                             std::string & reason,
                             local_exec_profile & profile) {
        reason.clear();
        profile = {};
        const auto key = std::make_pair(run_id, region);
        const auto kit = owned_kv_.find(key);
        const auto git = owned_graphs_.find(
            std::make_tuple(run_id, region, chunk));
        if (kit == owned_kv_.end() || git == owned_graphs_.end()) {
            reason = "PHONE_LOCAL_EXEC_GRAPH_NOT_PREPARED";
            return false;
        }
        auto & state = *kit->second;
        auto & item = *git->second;
        const auto & p = state.plan;
        const auto & g = item.built;
        if (state.next_chunk != chunk ||
                chunk >= ggml_phone_owned::chunk_count(p) ||
                g.token_begin != uint64_t(chunk) * p.chunk_tokens ||
                g.tokens != std::min<uint32_t>(
                    p.chunk_tokens, p.total_tokens - g.token_begin) ||
                g.cache_k != state.k || g.cache_v != state.v) {
            reason = "PHONE_LOCAL_EXEC_KV_FIFO_OR_BINDING_MISMATCH";
            return false;
        }
        const size_t bytes = uint64_t(g.tokens) * p.n_embd * sizeof(float);
        if (!reference_input_snapshot ||
                reference_input_count != bytes / sizeof(float) ||
                !reference_output || !g.input || !g.output ||
                g.input->buffer == nullptr || g.output->buffer == nullptr ||
                reference_output->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(reference_output) ||
                !ggml_is_contiguous(g.input) ||
                !ggml_is_contiguous(g.output) ||
                ggml_nbytes(g.input) != bytes ||
                ggml_nbytes(g.output) != bytes ||
                ggml_nbytes(reference_output) != bytes) {
            reason = "PHONE_LOCAL_EXEC_ACTIVATION_CONTRACT";
            return false;
        }
        const size_t count = bytes / sizeof(float);
        std::vector<float> ref(count);
        std::vector<float> got(count);

        // Input snapshot read time is accounted at baseline input-copy time,
        // before baseline compute. Do not read the OpenCL source here.
        const int64_t upload_begin = ggml_time_us();
        ggml_backend_tensor_set(g.input, reference_input_snapshot, 0, bytes);
        ggml_backend_synchronize(backend_);
        profile.input_upload_ms = (ggml_time_us() - upload_begin) / 1000.0;

        const int64_t compute_begin = ggml_time_us();
        const ggml_status code = ggml_backend_graph_compute(backend_, g.graph);
        ggml_backend_synchronize(backend_);
        profile.compute_ms = (ggml_time_us() - compute_begin) / 1000.0;
        if (code != GGML_STATUS_SUCCESS) {
            reason = "PHONE_LOCAL_EXEC_HTP_COMPUTE_FAILED";
            return false;
        }

        const int64_t compare_begin = ggml_time_us();
        ggml_backend_tensor_get(reference_output, ref.data(), 0, bytes);
        ggml_backend_tensor_get(g.output, got.data(), 0, bytes);
        const bool finite = ggml_phone_qwen3::compare_outputs(
            ref.data(), got.data(), count, profile.parity);
        profile.compare_ms = (ggml_time_us() - compare_begin) / 1000.0;
        profile.tokens = g.tokens;
        profile.layer = p.layer;
        if (!finite) {
            reason = "PHONE_LOCAL_EXEC_NONFINITE_OUTPUT";
            return false;
        }
        if (!ggml_phone_qwen3::compare_feature_errors(
                ref.data(), got.data(), count, p.n_embd,
                profile.parity, profile.feature)) {
            reason = "PHONE_LOCAL_EXEC_FEATURE_ATTRIBUTION_FAILED";
            return false;
        }
        // Only commit this state after a completed HTP run and comparison.
        // It is local-only and never aliases the PC/legacy decode KV.
        ++state.next_chunk;
        return true;
    }

    // Experimental P2C source cutover, gated by exact A/B parity in the
    // RPC compute loop. The baseline graph still executes to preserve its
    // canonical KV for subsequent decode. Only the OpenCL output buffer
    // (which the existing phone_auto_wait_capture publishes) is replaced.
    // Caller holds phone_npu_backend_mutex and graph_compute_mutex.
    bool publish_owned_htp_chunk(uint64_t run_id,
                                 uint32_t region, uint32_t chunk,
                                 ggml_tensor * destination,
                                 std::string & reason,
                                 double & copy_ms) {
        reason.clear();
        copy_ms = 0.0;
        const auto it = owned_graphs_.find(
            std::make_tuple(run_id, region, chunk));
        if (it == owned_graphs_.end() || !it->second ||
                it->second->built.graph == nullptr) {
            reason = "PHONE_LOCAL_P2C_GRAPH_NOT_PREPARED";
            return false;
        }
        const auto & local = it->second->built;
        if (local.output == nullptr || destination == nullptr ||
                local.output->buffer == nullptr ||
                destination->buffer == nullptr ||
                local.output->data == nullptr ||
                destination->data == nullptr ||
                local.output->type != GGML_TYPE_F32 ||
                destination->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(local.output) ||
                !ggml_is_contiguous(destination) ||
                !ggml_are_same_layout(local.output, destination) ||
                ggml_nbytes(local.output) == 0 ||
                ggml_nbytes(local.output) > 
                    ggml_phone_owned::k_max_activation_bytes) {
            reason = "PHONE_LOCAL_P2C_OUTPUT_CONTRACT";
            return false;
        }
        const int64_t t0 = ggml_time_us();
        ggml_backend_synchronize(backend_);
        // The canonical OpenCL Output buffer is private to this Chunk, and
        // is still the object consumed by the proven P2C capture path.
        ggml_backend_tensor_copy(local.output, destination);
        copy_ms = (ggml_time_us() - t0) / 1000.0;
        return true;
    }

    struct local_stage_parity {
        const char * stage = "";
        const char * state = "NOT_CHECKED";
        size_t count = 0;
        ggml_phone_qwen3::parity_stats parity {};
    };

    // Diagnostic only: compare actual HTP buffers from the PC-derived fixed
    // graph with the locally built HTP graph. All graph computation has
    // finished, and the caller still holds phone_npu_backend_mutex.
    // Unlike comparing P2C, these read intermediate Qwen3 operator outputs
    // and identify the FIRST point where local computation diverges.
    std::vector<local_stage_parity> compare_owned_htp_stages(
            uint64_t source_graph_uid, uint64_t run_id,
            uint32_t region, uint32_t chunk, uint32_t layer) const {
        static const std::array<const char *, 12> stages = {{
            "inp_stage", "attn_norm", "Qcur", "Kcur", "Vcur",
            "attn_out", "ffn_inp", "ffn_norm",
            "ffn_up", "ffn_gate", "ffn_swiglu", "ffn_down"
        }};
        std::vector<local_stage_parity> records;
        records.reserve(stages.size());
        const auto reference = registered_graphs_.find(source_graph_uid);
        const auto local = owned_graphs_.find(
            std::make_tuple(run_id, region, chunk));
        if (reference == registered_graphs_.end() ||
                local == owned_graphs_.end() ||
                reference->second->graph == nullptr ||
                local->second->built.graph == nullptr) {
            records.push_back({
                "all", "REFERENCE_FIXED_GRAPH_NOT_AVAILABLE", 0, {}
            });
            return records;
        }
        const auto find_named = [&](ggml_cgraph * graph,
                                    const char * stage,
                                    bool & ambiguous) -> const ggml_tensor * {
            ambiguous = false;
            const std::string tag = std::string(stage) + "-" +
                std::to_string(layer);
            const ggml_tensor * found = nullptr;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const ggml_tensor * t = ggml_graph_node(graph, i);
                if (t == nullptr) { continue; }
                const std::string name(t->name);
                const auto pos = name.rfind(tag);
                if (pos == std::string::npos ||
                        (pos != 0 && name[pos - 1] != '#') ||
                        (pos + tag.size() != name.size() &&
                         name[pos + tag.size()] != '#')) {
                    continue;
                }
                if (found != nullptr && found != t) {
                    ambiguous = true;
                    return nullptr;
                }
                found = t;
            }
            return found;
        };
        for (const char * stage : stages) {
            local_stage_parity row {};
            row.stage = stage;
            bool ambiguous_ref = false;
            bool ambiguous_local = false;
            const ggml_tensor * a = nullptr;
            const ggml_tensor * b = nullptr;
            if (std::strcmp(stage, "inp_stage") == 0) {
                // The stage activation is a LEAF, not a graph node.
                // Compare the baseline's HTP-bound input against the exact
                // HTP buffer the Phone-local Graph consumed. If this row
                // already differs, downstream attention is not the cause.
                for (const auto & in : reference->second->inputs) {
                    if (in.source == nullptr || in.target == nullptr) {
                        continue;
                    }
                    const std::string input_name(in.source->name);
                    const std::string suffix("#inp_stage");
                    const bool name_match =
                        input_name == "inp_stage" ||
                        input_name.find("#inp_stage#") !=
                            std::string::npos ||
                        (input_name.size() >= suffix.size() &&
                         input_name.compare(input_name.size() - suffix.size(),
                                            suffix.size(), suffix) == 0);
                    if (!name_match) { continue; }
                    if (a != nullptr && a != in.target) {
                        ambiguous_ref = true;
                        break;
                    }
                    a = in.target;
                }
                b = local->second->built.input;
            } else {
                a = find_named(reference->second->graph,
                               stage, ambiguous_ref);
                b = find_named(local->second->built.graph,
                               stage, ambiguous_local);
            }
            if (ambiguous_ref || ambiguous_local) {
                row.state = "AMBIGUOUS_STAGE_LABEL";
            } else if (!a || !b) {
                row.state = !a ? "MISSING_REFERENCE_STAGE" :
                    "MISSING_LOCAL_STAGE";
            } else if (a->type != GGML_TYPE_F32 ||
                    b->type != GGML_TYPE_F32 ||
                    !ggml_are_same_shape(a, b) ||
                    !ggml_is_contiguous(a) ||
                    !ggml_is_contiguous(b) ||
                    a->buffer == nullptr || b->buffer == nullptr ||
                    ggml_nbytes(a) == 0 ||
                    ggml_nbytes(a) > 128ULL * 1024 * 1024) {
                row.state = "UNSUPPORTED_STAGE_LAYOUT";
            } else {
                const size_t bytes = ggml_nbytes(a);
                const size_t count = bytes / sizeof(float);
                std::vector<float> ref_values(count);
                std::vector<float> local_values(count);
                ggml_backend_tensor_get(a, ref_values.data(), 0, bytes);
                ggml_backend_tensor_get(b, local_values.data(), 0, bytes);
                row.count = count;
                if (!ggml_phone_qwen3::compare_outputs(
                        ref_values.data(), local_values.data(),
                        count, row.parity)) {
                    row.state = "NONFINITE_STAGE_DATA";
                } else if (row.parity.reference_rms == 0.0 &&
                           row.parity.candidate_rms == 0.0 &&
                           std::strcmp(stage, "inp_stage") != 0) {
                    // Fused HTP kernels may never materialize some
                    // intermediates even when their metadata is present.
                    // Two untouched zero buffers are not parity proof.
                    row.state = "ZERO_INTERMEDIATE_UNVERIFIED";
                } else {
                    row.state = "MEASURED";
                }
            }
            records.push_back(std::move(row));
        }
        return records;
    }

    // Called after an input RUN has fully ended while HTP is serialized.
    // Shadow graphs never contribute to decode KV, so release their scratch
    // and KV immediately instead of reserving Phone HTP RAM until next RUN.
    void release_owned_htp_graphs() {
        if (owned_graphs_.empty() && owned_kv_.empty()) { return; }
        const auto reg = backend_ != nullptr ?
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_)) :
            nullptr;
        if (reg != nullptr) {
            using clear_fn = void (*)(ggml_backend_t);
            auto clear = reinterpret_cast<clear_fn>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_hexagon_clear_fixed_graph_cache"));
            if (clear) { clear(backend_); }
        }
        owned_graphs_.clear();
        owned_kv_.clear();
    }

    // Capture only immutable weight-source metadata while the RPC graph
    // registry is locked. The async worker must not retain graph/tensor
    // pointers: a subsequent GRAPH_REGISTER may reallocate graph storage.
    // Actual OpenCL buffers are kept alive separately through preload.
    static bool capture_weight_sources(
            const ggml_cgraph * source,
            std::vector<ggml_tensor> & snapshots,
            std::string & reason) {
        reason.clear();
        snapshots.clear();
        if (source == nullptr || source->n_nodes <= 0) {
            reason = "HTP_PRELOAD_INVALID_GRAPH";
            return false;
        }
        std::unordered_set<const ggml_tensor *> visited;
        std::unordered_set<std::string> keys;
        bool valid = true;
        std::function<void(const ggml_tensor *)> scan =
            [&](const ggml_tensor * t) {
                if (t == nullptr || !valid ||
                        !visited.insert(t).second) {
                    return;
                }
                if (t->op == GGML_OP_NONE && t->view_src == nullptr &&
                        is_weight(t) && !is_kv(t)) {
                    if (t->buffer == nullptr || t->data == nullptr ||
                            !ggml_is_contiguous(t)) {
                        reason = std::string("HTP_PRELOAD_INVALID_WEIGHT:") +
                                 t->name;
                        valid = false;
                        return;
                    }
                    if (keys.insert(storage_key(t, false)).second) {
                        // ggml_tensor is a plain C struct. Keep buffer/data,
                        // name, layout and OpenCL extra; never dereference
                        // src/view pointers in the detached snapshot.
                        snapshots.push_back(*t);
                    }
                }
                if (t->view_src != nullptr) {
                    scan(t->view_src);
                }
                for (const ggml_tensor * src : t->src) {
                    scan(src);
                }
            };
        for (int i = 0; i < source->n_nodes && valid; ++i) {
            scan(source->nodes[i]);
        }
        if (!valid) {
            snapshots.clear();
            return false;
        }
        if (snapshots.empty()) {
            reason = "HTP_PRELOAD_NO_WEIGHTS";
            return false;
        }
        return true;
    }

    // Does not dereference the original RPC graph: suitable for the dedicated
    // asynchronous worker. Caller serializes HTP access with run()/KV_SYNC.
    bool preload_weights(const std::vector<ggml_tensor> & snapshots,
                         ggml_backend_t source_backend,
                         std::string & reason,
                         weight_preload_stats & stats) {
        reason.clear();
        stats = {};
        if (backend_ == nullptr || source_backend == nullptr ||
                snapshots.empty()) {
            reason = "HTP_PRELOAD_INVALID_SOURCE";
            return false;
        }
        std::vector<const ggml_tensor *> missing;
        for (const ggml_tensor & snapshot : snapshots) {
            const std::string key = storage_key(&snapshot, false);
            if (persistent_.find(key) != persistent_.end()) {
                ++stats.reused_weights;
            } else if (has_model_weight(&snapshot)) {
                ++stats.reused_weights;
                ++stats.model_hits;
            } else {
                missing.push_back(&snapshot);
            }
        }
        if (missing.empty()) {
            return true;
        }

        // Wait for any earlier GPU weight uploads before copying to HTP.
        const int64_t sync_begin_us = ggml_time_us();
        ggml_backend_synchronize(source_backend);
        stats.sync_ms = (ggml_time_us() - sync_begin_us) / 1000.0;

        const int64_t copy_begin_us = ggml_time_us();
        for (const ggml_tensor * src : missing) {
            if (persistent_copy(src, false, reason) == nullptr) {
                stats.copy_ms = (ggml_time_us() - copy_begin_us) / 1000.0;
                return false;
            }
            ++stats.new_weights;
            stats.new_bytes += ggml_nbytes(src);
        }
        stats.copy_ms = (ggml_time_us() - copy_begin_us) / 1000.0;
        return true;
    }

    // Legacy synchronous test remains available for A/B comparisons.
    bool preload_weights(const ggml_cgraph * source,
                         ggml_backend_t source_backend,
                         std::string & reason,
                         weight_preload_stats & stats) {
        std::vector<ggml_tensor> snapshots;
        if (!capture_weight_sources(source, snapshots, reason)) {
            stats = {};
            return false;
        }
        return preload_weights(snapshots, source_backend, reason, stats);
    }

    // A separate HTP graph is registered for each PC Chunk UID. Its topology,
    // KV views, op_params and input destinations remain fixed for the RUN.
    // Different graphs share a preallocated HTP activation arena because
    // HTP execution is serialized (no compute/prepare overlap on that arena).
    //
    // Caller locks HTP before graph_compute_mutex. Only use registered source
    // graphs for which model weights have already been made resident.
    bool register_chunk(uint64_t uid, const ggml_cgraph * source,
                        const ggml_tensor * output,
                        std::string & reason, double & prepare_ms,
                        size_t & arena_bytes, bool & arena_reused) {
        reason.clear();
        prepare_ms = 0.0;
        arena_bytes = 0;
        arena_reused = false;
        if (uid == 0 || source == nullptr || source->n_nodes <= 0 ||
                output == nullptr || backend_ == nullptr ||
                registered_graphs_.size() >= 256) {
            reason = "HTP_FIXED_REGISTER_INVALID";
            return false;
        }
        if (registered_graphs_.count(uid)) {
            reason = "HTP_FIXED_DUPLICATE_UID";
            return false;
        }
        const int64_t begin_us = ggml_time_us();
        // Never snapshot a weight that the PC has not finished uploading.
        // The existing async preload/legacy execution remains the fallback.
        std::unordered_set<const ggml_tensor *> visited;
        bool weights_ready = true;
        std::function<void(const ggml_tensor *)> check =
            [&](const ggml_tensor * t) {
            if (t == nullptr || !visited.insert(t).second) { return; }
            if (t->op == GGML_OP_NONE && t->view_src == nullptr &&
                    is_weight(t) && !is_kv(t) &&
                    !has_model_weight(t) &&
                    persistent_.count(storage_key(t, false)) == 0) {
                weights_ready = false;
            }
            if (t->view_src != nullptr) { check(t->view_src); }
            for (const ggml_tensor * src : t->src) { check(src); }
        };
        for (int i = 0; i < source->n_nodes; ++i) { check(source->nodes[i]); }
        if (!weights_ready) {
            reason = "HTP_FIXED_WEIGHTS_NOT_RESIDENT";
            return false;
        }

        auto prepared = std::make_unique<registered_graph>();
        prepared->source = source;
        prepared->source_output = output;
        const size_t n_tensors = source->visited_hash_set.size + 64;
        const size_t mem_bytes =
            ggml_tensor_overhead() * n_tensors * 2 +
            ggml_graph_overhead_custom(source->size, false) + 4096;
        ggml_init_params params = {mem_bytes, nullptr, true};
        prepared->ctx.reset(ggml_init(params));
        if (!prepared->ctx) {
            reason = "HTP_FIXED_CONTEXT_ALLOC";
            return false;
        }

        std::unordered_map<const ggml_tensor *, ggml_tensor *> mapped;
        bool valid = true;
        std::function<ggml_tensor *(const ggml_tensor *)> mirror =
            [&](const ggml_tensor * t) -> ggml_tensor * {
            if (t == nullptr || !valid) { return nullptr; }
            const auto found = mapped.find(t);
            if (found != mapped.end()) { return found->second; }
            const bool kv = is_kv(t);
            if (t->op == GGML_OP_NONE && t->view_src == nullptr &&
                    (is_weight(t) || kv)) {
                // KV storage is allocated during registration, but its initial
                // OpenCL contents are copied at first compute. PC may still
                // reset/populate canonical KV after TASK_REGISTER.
                ggml_tensor * cached = persistent_copy(
                    t, kv, reason, nullptr, !kv);
                if (cached == nullptr) { valid = false; return nullptr; }
                mapped.emplace(t, cached);
                if (kv) {
                    prepared->kv_sources.emplace_back(storage_key(t, true), t);
                }
                return cached;
            }
            ggml_tensor * copy = ggml_dup_tensor(prepared->ctx.get(), t);
            if (copy == nullptr) {
                reason = "HTP_FIXED_DUP_TENSOR";
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
                if (t->src[i] == nullptr) { continue; }
                copy->src[i] = mirror(t->src[i]);
                if (copy->src[i] == nullptr) {
                    valid = false;
                    return nullptr;
                }
            }
            if (t->op == GGML_OP_NONE && t->view_src == nullptr) {
                prepared->inputs.push_back({
                    t, copy, t->buffer, t->data
                });
            }
            return copy;
        };
        ggml_cgraph * graph = ggml_new_graph_custom(
            prepared->ctx.get(), source->size, false);
        if (graph == nullptr) {
            reason = "HTP_FIXED_GRAPH_ALLOC";
            return false;
        }
        prepared->graph = graph;
        for (int i = 0; i < source->n_nodes && valid; ++i) {
            ggml_tensor * cloned = mirror(source->nodes[i]);
            if (cloned == nullptr) { break; }
            ggml_build_forward_expand(graph, cloned);
        }
        auto out = mapped.find(output);
        if (!valid || out == mapped.end() || graph->n_nodes == 0) {
            if (reason.empty()) { reason = "HTP_FIXED_MIRROR_INCOMPLETE"; }
            return false;
        }
        prepared->output = out->second;
        if (!ggml_are_same_layout(prepared->output, output)) {
            reason = "HTP_FIXED_OUTPUT_LAYOUT";
            return false;
        }
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (!ggml_backend_supports_op(backend_, graph->nodes[i])) {
                reason = std::string("HTP_FIXED_UNSUPPORTED_OP:") +
                    graph->nodes[i]->name;
                return false;
            }
        }
        // A SET_ROWS result must remain a VIEW of the same persistent KV.
        for (int i = 0; i < source->n_nodes; ++i) {
            const ggml_tensor * t = source->nodes[i];
            if (t->op != GGML_OP_SET_ROWS || t->src[2] == nullptr) { continue; }
            auto result = mapped.find(t);
            auto base = mapped.find(t->src[2]);
            if (result == mapped.end() || base == mapped.end() ||
                    result->second->view_src == nullptr) {
                reason = "HTP_FIXED_SET_ROWS_VIEW";
                return false;
            }
        }

        // Plan fixed offsets in context allocation order. Views get their
        // canonical parent's address, not independent arena allocations.
        const auto buft = ggml_backend_get_default_buffer_type(backend_);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            reason = "HTP_FIXED_BAD_ALIGNMENT";
            return false;
        }
        size_t offset = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(prepared->ctx.get());
             t != nullptr;
             t = ggml_get_next_tensor(prepared->ctx.get(), t)) {
            if (t->view_src != nullptr) {
                prepared->views.push_back(t);
            } else if (t->data == nullptr) {
                const size_t bytes = ggml_backend_buft_get_alloc_size(buft, t);
                if (offset > SIZE_MAX - (alignment - 1) ||
                        bytes > SIZE_MAX - offset - (alignment - 1)) {
                    reason = "HTP_FIXED_SIZE_OVERFLOW";
                    return false;
                }
                offset = (offset + alignment - 1) & ~(alignment - 1);
                prepared->roots.push_back({ t, offset });
                offset += bytes;
            }
        }
        if (offset == 0 || offset > SIZE_MAX - alignment) {
            reason = "HTP_FIXED_NO_ACTIVATION_STORAGE";
            return false;
        }
        const size_t needed = offset + alignment;
        registered_arena * chosen = nullptr;
        for (auto & candidate : arenas_) {
            if (candidate->capacity >= needed &&
                    (chosen == nullptr ||
                     candidate->capacity < chosen->capacity)) {
                chosen = candidate.get();
            }
        }
        if (chosen == nullptr) {
            // Power-of-two size class preserves addresses of previously
            // registered graphs: a larger tail never relocates an active plan.
            size_t capacity = 4096;
            while (capacity < needed && capacity <= SIZE_MAX / 2) {
                capacity *= 2;
            }
            if (capacity < needed ||
                    capacity > ggml_backend_buft_get_max_size(buft)) {
                reason = "HTP_FIXED_ARENA_SIZE_LIMIT";
                return false;
            }
            auto next = std::make_unique<registered_arena>();
            next->buffer.reset(ggml_backend_buft_alloc_buffer(buft, capacity));
            if (!next->buffer) {
                reason = "HTP_FIXED_ARENA_ALLOC_FAILED";
                return false;
            }
            next->capacity = capacity;
            chosen = next.get();
            arenas_.push_back(std::move(next));
        } else {
            arena_reused = true;
        }
        prepared->arena = chosen;
        arena_bytes = chosen->capacity;
        const size_t base = reinterpret_cast<size_t>(
            ggml_backend_buffer_get_base(chosen->buffer.get()));
        if (base == 0) {
            reason = "HTP_FIXED_ARENA_NO_BASE";
            return false;
        }
        const size_t padding = (alignment - (base & (alignment - 1))) &
            (alignment - 1);
        for (const auto & binding : prepared->roots) {
            ggml_tensor * t = binding.first;
            if (padding > chosen->capacity ||
                    binding.second > chosen->capacity - padding ||
                    ggml_backend_buft_get_alloc_size(buft, t) >
                        chosen->capacity - padding - binding.second) {
                reason = "HTP_FIXED_ARENA_BOUNDS";
                return false;
            }
            if (ggml_backend_tensor_alloc(chosen->buffer.get(), t,
                    reinterpret_cast<void *>(
                        base + padding + binding.second)) !=
                    GGML_STATUS_SUCCESS) {
                reason = "HTP_FIXED_TENSOR_BIND";
                return false;
            }
        }
        // Registration is allowed to encounter a VIEW before its parent in
        // metadata order. Resolve views recursively in dependency order.
        std::unordered_set<ggml_tensor *> visiting;
        std::function<bool(ggml_tensor *)> init_view =
            [&](ggml_tensor * t) {
            if (t->buffer != nullptr && t->data != nullptr) { return true; }
            if (t->view_src == nullptr || !visiting.insert(t).second) {
                return false;
            }
            bool ok = init_view(t->view_src) &&
                ggml_backend_view_init(t) == GGML_STATUS_SUCCESS;
            visiting.erase(t);
            return ok;
        };
        for (ggml_tensor * t : prepared->views) {
            if (!init_view(t)) {
                reason = "HTP_FIXED_VIEW_BIND";
                return false;
            }
        }
        for (const auto & item : prepared->inputs) {
            if (item.source->buffer == nullptr ||
                    item.source->data == nullptr ||
                    item.target->buffer == nullptr ||
                    !ggml_is_contiguous(item.source)) {
                reason = std::string("HTP_FIXED_INPUT_INVALID:") +
                    item.source->name;
                return false;
            }
        }
        for (int i = 0; i < source->n_nodes; ++i) {
            const ggml_tensor * op = source->nodes[i];
            if (op->op != GGML_OP_SET_ROWS || op->src[2] == nullptr) { continue; }
            const auto result = mapped.find(op);
            const auto kv = mapped.find(op->src[2]);
            if (result == mapped.end() || kv == mapped.end() ||
                    result->second->data != kv->second->data ||
                    result->second->buffer != kv->second->buffer) {
                reason = "HTP_FIXED_KV_ALIAS";
                return false;
            }
        }
        // High bit reserves a dedicated backend graph-cache namespace for
        // persistent fixed Chunk graphs. Never reuse within server lifetime.
        if (next_fixed_uid_ == 0 ||
                next_fixed_uid_ >= (UINT64_C(1) << 63)) {
            reason = "HTP_FIXED_UID_EXHAUSTED";
            return false;
        }
        graph->uid = (UINT64_C(1) << 63) | next_fixed_uid_++;
        // Compile fused HTP opnodes and fixed kernel parameters now, not
        // during the first Chunk compute. Every referenced Tensor address is
        // already bound to either persistent KV/weights or the arena pool.
        const auto reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(backend_));
        using compile_fn = bool (*)(ggml_backend_t, ggml_cgraph *);
        auto compile = reg != nullptr ? reinterpret_cast<compile_fn>(
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_hexagon_precompile_fixed_graph")) :
            nullptr;
        if (compile == nullptr || !compile(backend_, graph)) {
            reason = "HTP_FIXED_PLAN_PRECOMPILE";
            return false;
        }
        registered_graphs_.emplace(uid, std::move(prepared));
        prepare_ms = (ggml_time_us() - begin_us) / 1000.0;
        return true;
    }

    bool has_registered_chunk(uint64_t uid) const {
        return registered_graphs_.count(uid) != 0;
    }

    bool run_registered_chunk(uint64_t uid, ggml_cgraph * source,
                              ggml_tensor * output, std::string & reason,
                              double & stage_ms, double & compute_ms,
                              double & output_ms,
                              stage_profile_stats * profile = nullptr,
                              const ggml_tensor * ab_source = nullptr,
                              std::vector<float> * ab_input_snapshot = nullptr,
                              double * ab_capture_ms = nullptr) {
        reason.clear();
        stage_ms = compute_ms = output_ms = 0.0;
        if (profile != nullptr) { *profile = {}; }
        if ((ab_source == nullptr) != (ab_input_snapshot == nullptr) ||
                (ab_capture_ms != nullptr && ab_input_snapshot == nullptr)) {
            reason = "HTP_FIXED_AB_INCOMPLETE_SNAPSHOT_CONTRACT";
            return false;
        }
        if (ab_input_snapshot != nullptr) {
            ab_input_snapshot->clear();
        }
        if (ab_capture_ms != nullptr) { *ab_capture_ms = 0.0; }
        const auto it = registered_graphs_.find(uid);
        if (it == registered_graphs_.end()) {
            reason = "HTP_FIXED_UID_NOT_FOUND";
            return false;
        }
        registered_graph & item = *it->second;
        if (item.source != source || item.source_output != output) {
            reason = "HTP_FIXED_SOURCE_CHANGED";
            return false;
        }
        const int64_t start_us = ggml_time_us();
        // Keep canonical OpenCL KV writeback target fresh and defer the first
        // KV upload until the producer's runtime OpenCL writes are complete.
        for (const auto & binding : item.kv_sources) {
            const auto found = persistent_.find(binding.first);
            if (found == persistent_.end() ||
                    found->second->tensor == nullptr) {
                reason = "HTP_FIXED_KV_GONE";
                return false;
            }
            found->second->original = binding.second;
            if (deferred_kv_.count(binding.first)) {
                const int64_t t0 = profile != nullptr ? ggml_time_us() : 0;
                ggml_backend_tensor_copy(binding.second,
                                         found->second->tensor);
                if (profile != nullptr) {
                    profile->kv_bind_ms +=
                        (ggml_time_us() - t0) / 1000.0;
                    ++profile->kv_new;
                }
                deferred_kv_.erase(binding.first);
            } else if (profile != nullptr) {
                ++profile->kv_reused;
            }
        }
        if (profile != nullptr) {
            profile->persistent_bind_ms = profile->kv_bind_ms;
            profile->graph_nodes = static_cast<size_t>(item.graph->n_nodes);
            profile->input_tensors = item.inputs.size();
        }
        const int64_t copy_begin = ggml_time_us();
        const ggml_tensor * htp_snapshot_target = nullptr;
        for (const auto & in : item.inputs) {
            if (in.source->data != in.source_data ||
                    in.source->buffer != in.source_buffer ||
                    !ggml_are_same_layout(in.source, in.target)) {
                reason = "HTP_FIXED_INPUT_BINDING_DRIFT";
                return false;
            }
            ggml_backend_tensor_copy(in.source, in.target);
            if (ab_input_snapshot != nullptr && in.source == ab_source) {
                if (htp_snapshot_target != nullptr) {
                    reason = "HTP_FIXED_AB_MULTIPLE_STAGE_INPUTS";
                    return false;
                }
                htp_snapshot_target = in.target;
            }
        }
        if (ab_input_snapshot != nullptr) {
            // Read exactly what the baseline HTP Graph will consume.
            // Tensor inputs can be aliased/reused by the legacy Graph,
            // so this must happen BEFORE the first compute.
            if (htp_snapshot_target == nullptr ||
                    ab_source->type != GGML_TYPE_F32 ||
                    htp_snapshot_target->type != GGML_TYPE_F32 ||
                    !ggml_is_contiguous(htp_snapshot_target) ||
                    !ggml_is_contiguous(ab_source) ||
                    !ggml_are_same_layout(ab_source,
                                          htp_snapshot_target) ||
                    ggml_nbytes(htp_snapshot_target) == 0 ||
                    ggml_nbytes(htp_snapshot_target) >
                        ggml_phone_owned::k_max_activation_bytes) {
                reason = "HTP_FIXED_AB_STAGE_INPUT_NOT_FOUND_OR_INVALID";
                return false;
            }
            const size_t bytes = ggml_nbytes(htp_snapshot_target);
            ab_input_snapshot->resize(bytes / sizeof(float));
            const int64_t capture_begin_us = ggml_time_us();
            ggml_backend_synchronize(backend_);
            ggml_backend_tensor_get(htp_snapshot_target,
                                    ab_input_snapshot->data(), 0, bytes);
            if (ab_capture_ms != nullptr) {
                *ab_capture_ms =
                    (ggml_time_us() - capture_begin_us) / 1000.0;
            }
            for (float x : *ab_input_snapshot) {
                if (!std::isfinite(x)) {
                    reason = "HTP_FIXED_AB_NONFINITE_BASELINE_INPUT";
                    return false;
                }
            }
        }
        const int64_t compute_begin = ggml_time_us();
        stage_ms = (compute_begin - start_us) / 1000.0;
        if (profile != nullptr) {
            profile->input_copy_ms =
                (compute_begin - copy_begin) / 1000.0;
            profile->other_ms = std::max(0.0, stage_ms -
                profile->input_copy_ms - profile->persistent_bind_ms);
        }
        if (ggml_backend_graph_compute(backend_, item.graph) !=
                GGML_STATUS_SUCCESS) {
            reason = "HTP_FIXED_GRAPH_COMPUTE";
            return false;
        }
        ggml_backend_synchronize(backend_);
        const int64_t compute_end = ggml_time_us();
        compute_ms = (compute_end - compute_begin) / 1000.0;
        ggml_backend_tensor_copy(item.output, output);
        output_ms = (ggml_time_us() - compute_end) / 1000.0;
        return true;
    }

    // Invalidate backend plans BEFORE releasing their referenced tensors/KV.
    // The HTP backend proc is optional for older libraries, so fixed-graph
    // registration is enabled only when that invalidation hook exists.
    bool fixed_graph_cache_supported() const {
        if (backend_ == nullptr) { return false; }
        const auto reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(backend_));
        return reg != nullptr &&
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_hexagon_clear_fixed_graph_cache") != nullptr &&
            ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_hexagon_precompile_fixed_graph") != nullptr;
    }

    // Caller holds the server's single Phone graph-compute mutex. Only one
    // full-region execution is in progress at any moment, on either device.
    bool run(ggml_cgraph * source, ggml_tensor * output,
             std::string & reason, double & stage_ms,
             double & compute_ms, double & output_ms,
             stage_profile_stats * stage_profile = nullptr) {
        reason.clear();
        stage_ms = compute_ms = output_ms = 0;
        if (stage_profile != nullptr) { *stage_profile = {}; }
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
        if (stage_profile != nullptr) {
            stage_profile->context_create_ms =
                (ggml_time_us() - start_us) / 1000.0;
        }
        const int64_t mirror_begin_us =
            stage_profile != nullptr ? ggml_time_us() : 0;
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
                const bool kv = is_kv(t);
                const int64_t bind_begin_us =
                    stage_profile != nullptr ? ggml_time_us() : 0;
                bool new_persistent = false;
                ggml_tensor * cached = persistent_copy(
                    t, kv, reason,
                    stage_profile != nullptr ? &new_persistent : nullptr);
                if (stage_profile != nullptr) {
                    const double bind_ms =
                        (ggml_time_us() - bind_begin_us) / 1000.0;
                    stage_profile->persistent_bind_ms += bind_ms;
                    if (kv) {
                        stage_profile->kv_bind_ms += bind_ms;
                        if (new_persistent) {
                            ++stage_profile->kv_new;
                        } else if (cached != nullptr) {
                            ++stage_profile->kv_reused;
                        }
                    } else {
                        stage_profile->weight_bind_ms += bind_ms;
                    }
                }
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
        if (stage_profile != nullptr) {
            const double mirror_wall_ms =
                (ggml_time_us() - mirror_begin_us) / 1000.0;
            stage_profile->graph_mirror_ms = std::max(
                0.0, mirror_wall_ms - stage_profile->persistent_bind_ms);
            stage_profile->graph_nodes = static_cast<size_t>(graph->n_nodes);
            stage_profile->input_tensors = inputs.size();
        }
        const int64_t alloc_begin_us =
            stage_profile != nullptr ? ggml_time_us() : 0;
        ggml_backend_buffer_ptr temporary {
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend_)
        };
        if (!temporary) {
            reason = "HTP_STAGE_BUFFER_ALLOC";
            return false;
        }
        if (stage_profile != nullptr) {
            stage_profile->temporary_alloc_ms =
                (ggml_time_us() - alloc_begin_us) / 1000.0;
        }
        const int64_t input_begin_us =
            stage_profile != nullptr ? ggml_time_us() : 0;
        for (const auto & item : inputs) {
            if (item.first->data == nullptr || item.second->buffer == nullptr ||
                    !ggml_is_contiguous(item.first)) {
                reason = std::string("HTP_INPUT_INVALID:") + item.first->name;
                return false;
            }
            ggml_backend_tensor_copy(item.first, item.second);
        }
        if (stage_profile != nullptr) {
            stage_profile->input_copy_ms =
                (ggml_time_us() - input_begin_us) / 1000.0;
        }
        const int64_t check_begin_us =
            stage_profile != nullptr ? ggml_time_us() : 0;
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
        if (stage_profile != nullptr) {
            stage_profile->op_check_ms =
                (compute_begin_us - check_begin_us) / 1000.0;
            const double accounted_ms =
                stage_profile->context_create_ms +
                stage_profile->graph_mirror_ms +
                stage_profile->persistent_bind_ms +
                stage_profile->temporary_alloc_ms +
                stage_profile->input_copy_ms +
                stage_profile->op_check_ms;
            stage_profile->other_ms = std::max(0.0, stage_ms - accounted_ms);
        }
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
            if (!entry->kv || deferred_kv_.count(key) != 0) {
                // Fixed Chunk registration allocates KV storage early.
                // A never-executed graph must not write uninitialized HTP
                // KV bytes back to the canonical OpenCL decode cache.
                continue;
            }
            if (entry->tensor == nullptr || entry->original == nullptr ||
                    entry->original->buffer == nullptr ||
                    !ggml_are_same_layout(entry->tensor, entry->original)) {
                reason = "HTP_KV_SNAPSHOT_INVALID";
                return false;
            }
            // The RPC graph owns this mutable OpenCL KV tensor. We keep
            // only a const view for identity/layout checks, but this is the
            // explicit end-of-prefill writeback destination.
            ggml_backend_tensor_copy(entry->tensor,
                                     const_cast<ggml_tensor *>(entry->original));
            n_bytes += ggml_nbytes(entry->tensor);
        }
        return true;
    }

    void reset_kv() {
        // This is called at a new RUN boundary while HTP and Graph locks are
        // held. Cached Hexagon nodes point to the per-Chunk tensor metadata.
        const auto reg = backend_ != nullptr ?
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_)) :
            nullptr;
        if (reg != nullptr) {
            using clear_fn = void (*)(ggml_backend_t);
            auto clear = reinterpret_cast<clear_fn>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_hexagon_clear_fixed_graph_cache"));
            if (clear != nullptr) { clear(backend_); }
        }
        registered_graphs_.clear();
        owned_graphs_.clear();
        owned_kv_.clear();
        deferred_kv_.clear();
        for (auto it = persistent_.begin(); it != persistent_.end();) {
            it = it->second->kv ? persistent_.erase(it) : ++it;
        }
        // Keep standalone scratch arenas resident across RUNs; only graph
        // metadata and per-Prompt KV are invalidated. Their addresses stay
        // stable until server teardown.
    }

    bool has_backend() const { return backend_ != nullptr; }

private:
    struct banked_tensor_spec {
        std::string name;
        ggml_rpc_local_tensor_info info {};
        size_t offset = 0;
        size_t alloc_bytes = 0;
    };
    struct banked_layer_spec {
        uint32_t region = 0;
        uint32_t layer = 0;
        uint32_t layer_end = 0;
        size_t bytes = 0;
        std::vector<banked_tensor_spec> tensors;
        std::vector<std::string> keys;
    };

    struct persistent_tensor {
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        ggml_tensor * tensor = nullptr;
        const ggml_tensor * original = nullptr;
        bool kv = false;
    };

    // The HTP repack Tensor descriptors are recreated at every refill.
    // Address offsets into the two backing arenas never change. Only the
    // bytes and metadata for the current Region are resident per Bank.
    bool load_banked_rank(size_t rank, std::string & reason,
                          model_load_stats & stats,
                          const std::vector<std::vector<uint8_t>> * staged = nullptr) {
        reason.clear();
        stats = {};
        if (!has_any_weight_plan() || rank >= banked_layers_.size() ||
                !bank_local_source_.get_info || !bank_local_source_.read) {
            reason = "NPU_BANK_LOAD_INVALID_RANK_OR_SOURCE";
            return false;
        }
        auto & layer = banked_layers_[rank];
        if (staged != nullptr && staged->size() != layer.tensors.size()) {
            reason = "NPU_BANK_STAGE_COUNT_MISMATCH";
            return false;
        }
        ggml_backend_buffer_t bank = resident_mode_ ?
            resident_weight_arenas_[rank].get() :
            bank_weight_arenas_[rank % 2].get();
        if (!bank) {
            reason = "NPU_BANK_ARENA_NOT_MAPPED";
            return false;
        }
        // Do not publish a partially repacked Region into model_weights_.
        std::unordered_map<std::string, std::unique_ptr<persistent_tensor>>
            pending;
        std::vector<std::string> keys;
        keys.reserve(layer.tensors.size());
        for (size_t tensor_index = 0;
                tensor_index < layer.tensors.size(); ++tensor_index) {
            const auto & spec = layer.tensors[tensor_index];
            ggml_init_params params {
                ggml_tensor_overhead() * 4 + 4096, nullptr, true
            };
            auto entry = std::make_unique<persistent_tensor>();
            entry->ctx.reset(ggml_init(params));
            if (!entry->ctx) {
                reason = "NPU_BANK_LOAD_CONTEXT";
                return false;
            }
            int64_t shape[GGML_MAX_DIMS] = {1, 1, 1, 1};
            for (uint32_t d = 0; d < spec.info.n_dims; ++d) {
                shape[d] = static_cast<int64_t>(spec.info.ne[d]);
            }
            entry->tensor = ggml_new_tensor(
                entry->ctx.get(), static_cast<ggml_type>(spec.info.type),
                GGML_MAX_DIMS, shape);
            if (!entry->tensor ||
                    ggml_nbytes(entry->tensor) !=
                        static_cast<size_t>(spec.info.nbytes)) {
                reason = "NPU_BANK_LOAD_LAYOUT_DRIFT:" + spec.name;
                return false;
            }
            ggml_set_name(entry->tensor, spec.name.c_str());
            const size_t actual_size =
                ggml_backend_buft_get_alloc_size(repack_buft_, entry->tensor);
            const size_t bank_size = ggml_backend_buffer_get_size(bank);
            if (actual_size != spec.alloc_bytes ||
                    spec.offset > bank_size ||
                    actual_size > bank_size - spec.offset) {
                reason = "NPU_BANK_LOAD_SIZE_DRIFT:" + spec.name;
                return false;
            }
            void * const at =
                static_cast<uint8_t *>(ggml_backend_buffer_get_base(bank)) +
                spec.offset;
            if (ggml_backend_tensor_alloc(
                    bank, entry->tensor, at) != GGML_STATUS_SUCCESS) {
                reason = "NPU_BANK_LOAD_TENSOR_BIND:" + spec.name;
                return false;
            }
            const std::string key = model_weight_key(entry->tensor);
            if (model_weights_.count(key) || pending.count(key)) {
                reason = "NPU_BANK_LOAD_DUPLICATE_WEIGHT:" + spec.name;
                return false;
            }
            std::vector<uint8_t> direct;
            const std::vector<uint8_t> * raw = nullptr;
            if (staged != nullptr) {
                raw = &(*staged)[tensor_index];
                if (raw->size() != static_cast<size_t>(spec.info.nbytes)) {
                    reason = "NPU_BANK_STAGE_LAYOUT_MISMATCH:" + spec.name;
                    return false;
                }
            } else {
                try {
                    direct.resize(static_cast<size_t>(spec.info.nbytes));
                } catch (const std::bad_alloc &) {
                    reason = "NPU_BANK_LOAD_HOST_STAGING_OOM";
                    return false;
                }
                const int64_t read_us = ggml_time_us();
                if (!bank_local_source_.read(bank_local_source_.user_data,
                        spec.name.c_str(), 0, direct.data(), direct.size())) {
                    reason = "NPU_BANK_LOAD_GGUF_READ:" + spec.name;
                    return false;
                }
                stats.file_read_ms +=
                    (ggml_time_us() - read_us) / 1000.0;
                raw = &direct;
            }
            const int64_t repack_us = ggml_time_us();
            ggml_backend_tensor_set(
                entry->tensor, raw->data(), 0, raw->size());
            stats.htp_repack_ms +=
                (ggml_time_us() - repack_us) / 1000.0;
            entry->kv = false;
            entry->original = nullptr;
            pending.emplace(key, std::move(entry));
            keys.push_back(key);
            ++stats.new_weights;
            stats.new_bytes += raw->size();
        }
        ggml_backend_synchronize(backend_);
        for (auto & p : pending) {
            model_weights_.emplace(p.first, std::move(p.second));
        }
        layer.keys = std::move(keys);
        return true;
    }

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

    static std::string model_weight_key(const ggml_tensor * t) {
        // Meta RPC may namespace a weight leaf as
        // "Meta(CPU,RPC0)#blk.19.attn_q.weight", whereas the local GGUF
        // loader returns "blk.19.attn_q.weight". Match the canonical
        // layer-scoped suffix, but still require equal type and layout.
        const std::string name(t->name);
        const size_t at = name.find("blk.");
        std::string key = "MODEL:";
        key += at == std::string::npos ? name : name.substr(at);
        key += ":" + std::to_string(static_cast<int>(t->type));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            key += ":" + std::to_string(t->ne[d]);
        }
        return key;
    }

    bool has_model_weight(const ggml_tensor * t) const {
        if (!is_weight(t) || is_kv(t)) { return false; }
        const auto it = model_weights_.find(model_weight_key(t));
        // GGUF weight metadata must have the same canonical layout as
        // the first RPC Graph's original tensor, or fall back to copying.
        return it != model_weights_.end() &&
            ggml_are_same_layout(it->second->tensor, t);
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
                                  std::string & reason,
                                  bool * created = nullptr,
                                  bool initialize_from_source = true) {
        if (created != nullptr) { *created = false; }
        if (src->buffer == nullptr || src->data == nullptr ||
                !ggml_is_contiguous(src)) {
            reason = std::string("HTP_PERSISTENT_SOURCE_INVALID:") + src->name;
            return nullptr;
        }
        const std::string key = storage_key(src, kv);
        const auto found = persistent_.find(key);
        if (found != persistent_.end()) {
            if (kv) {
                found->second->original = src;
            }
            return found->second->tensor;
        }
        if (!kv && has_model_weight(src)) {
            return model_weights_.at(model_weight_key(src))->tensor;
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
        if (initialize_from_source) {
            ggml_backend_tensor_copy(src, entry->tensor);
        } else if (kv) {
            deferred_kv_.insert(key);
        }
        // Weight preload may use a detached, short-lived metadata snapshot.
        // Only KV needs the canonical OpenCL destination for final sync.
        entry->original = kv ? src : nullptr;
        entry->kv = kv;
        ggml_tensor * result = entry->tensor;
        persistent_.emplace(key, std::move(entry));
        if (created != nullptr) { *created = true; }
        return result;
    }

    struct registered_input {
        const ggml_tensor * source = nullptr;
        ggml_tensor * target = nullptr;
        ggml_backend_buffer_t source_buffer = nullptr;
        const void * source_data = nullptr;
    };
    struct registered_arena {
        ggml_backend_buffer_ptr buffer;
        size_t capacity = 0;
    };
    struct registered_graph {
        ggml_context_ptr ctx;
        ggml_cgraph * graph = nullptr;
        const ggml_cgraph * source = nullptr;
        const ggml_tensor * source_output = nullptr;
        ggml_tensor * output = nullptr;
        registered_arena * arena = nullptr;
        std::vector<registered_input> inputs;
        std::vector<std::pair<std::string, const ggml_tensor *>> kv_sources;
        std::vector<std::pair<ggml_tensor *, size_t>> roots;
        std::vector<ggml_tensor *> views;
    };
    // Arenas are declared BEFORE graph entries so graphs are destroyed first
    // in the C++ destruction sequence; HTP weights/KV outlive the entries.
    std::vector<std::unique_ptr<registered_arena>> arenas_;
    std::unordered_map<uint64_t, std::unique_ptr<registered_graph>>
        registered_graphs_;
    std::unordered_set<std::string> deferred_kv_;
    uint64_t next_fixed_uid_ = 1;

    ggml_backend_t backend_ = nullptr;  // non-owning; RPC server owns it
    ggml_backend_buffer_type_t repack_buft_ = nullptr;
    std::unordered_map<std::string, std::unique_ptr<persistent_tensor>>
        persistent_;
    // Dedicated model-lifetime cache, never reset by a new Prompt.
    std::unordered_map<std::string, std::unique_ptr<persistent_tensor>>
        model_weights_;
    // Fixed HTP repack storage, in physical A/B arenas. Each arena remains
    // mapped for the life of the server, while its quantized Tensor metadata
    // and Graph UID are recreated after the old Region drains.
    two_weight_banks weight_banks_;
    ggml_rpc_local_tensor_source bank_local_source_ {};
    std::array<ggml_backend_buffer_ptr, 2> bank_weight_arenas_ {};
    std::vector<ggml_backend_buffer_ptr> resident_weight_arenas_;
    std::vector<banked_layer_spec> banked_layers_;
    std::unordered_map<uint32_t, size_t> region_rank_;
    bool banked_mode_ = false;
    bool resident_mode_ = false;

    struct owned_layer_kv {
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        ggml_phone_owned::region_plan plan {};
        uint32_t next_chunk = 0;
        ggml_tensor * k = nullptr;
        ggml_tensor * v = nullptr;
        std::vector<std::pair<ggml_tensor *, ggml_tensor *>> layer_kv;
        std::vector<const ggml_tensor *> canonical_k_layers;
        std::vector<const ggml_tensor *> canonical_v_layers;
        std::vector<ggml_context_ptr> canonical_extra_contexts;
        // Graphless mode owns the imported canonical OpenCL K/V metadata;
        // PC still owns the underlying physical buffers until RUN END.
        ggml_context_ptr canonical_ctx;
        bool graphless_identity_slots = false;
        // Legacy mode uses non-owning RPC Graph leaf pointers instead.
        const ggml_tensor * canonical_k = nullptr;
        const ggml_tensor * canonical_v = nullptr;
    };
    struct owned_chunk_graph {
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        ggml_phone_qwen3::graph_result built {};
        std::vector<ggml_phone_qwen3::graph_result> all_layer_graphs;
        uint64_t bank_generation = 0;
        const ggml_tensor * canonical_k_indices = nullptr;
        const ggml_tensor * canonical_v_indices = nullptr;
    };
    // Declare after model_weights_, and graph after KV, so destruction
    // order is graph -> KV -> model weights.
    std::map<std::pair<uint64_t, uint32_t>,
             std::unique_ptr<owned_layer_kv>> owned_kv_;
    std::map<std::tuple<uint64_t, uint32_t, uint32_t>,
             std::unique_ptr<owned_chunk_graph>> owned_graphs_;
    uint64_t next_owned_uid_ = 1;
};

}  // namespace ggml_phone_npu_serial
