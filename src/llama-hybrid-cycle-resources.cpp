#include "llama-hybrid-cycle-resources.h"
#include "llama-hybrid-cycle-log.h"

#include "llama-model.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>

namespace {

bool cycle_cuda_mirror_estimate(
        const llama_layer & lw, ggml_backend_t cpu,
        ggml_backend_t cuda, size_t & bytes) {
    bytes = 0;
    if (!cpu || !cuda) return false;
    const auto buft = ggml_backend_get_default_buffer_type(cuda);
    if (!buft) return false;
    const ggml_tensor * logical[] = {
        lw.attn_norm, lw.wq, lw.wk, lw.wv, lw.wo,
        lw.attn_q_norm, lw.attn_k_norm, lw.ffn_norm,
        lw.ffn_gate_inp, lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps
    };
    const size_t alignment =
        std::max<size_t>(1, ggml_backend_buft_get_alignment(buft));
    for (const ggml_tensor * tensor : logical) {
        const auto * physical = llama_hybrid_cycle_resolve_physical_weight(
            tensor, cpu, 0);
        if (!physical || !physical->buffer ||
            !ggml_is_contiguous(physical)) {
            return false;
        }
        size_t n = ggml_backend_buft_get_alloc_size(buft, physical);
        if (!n) return false;
        if (n > std::numeric_limits<size_t>::max() - (alignment - 1)) {
            return false;
        }
        n = ((n + alignment - 1) / alignment) * alignment;
        if (bytes > std::numeric_limits<size_t>::max() - n) {
            return false;
        }
        bytes += n;
    }
    // Reserve for the allocator alignment, metadata and temporary
    // GPU workspace; this is a conservative planning bound, not actual
    // ggml_backend_buffer_get_size() reported after allocation.
    const size_t minimum_headroom = 8ULL * 1024 * 1024;
    const size_t fractional_headroom = bytes / 20;
    const size_t extra = std::max(minimum_headroom, fractional_headroom);
    if (bytes > std::numeric_limits<size_t>::max() - extra) return false;
    bytes += extra;
    return true;
}

} // namespace

bool llama_hybrid_cycle_select_cuda_full_layers(
        const llama_hybrid_cycle_layout & layout,
        const std::vector<size_t> & estimates, size_t budget,
        std::vector<uint8_t> & selected, size_t & selected_bytes,
        std::string & reason) {
    selected.clear();
    selected_bytes = 0;
    reason.clear();
    if (!layout.enabled() || !llama_hybrid_cycle_cuda_layers_valid(layout) ||
        estimates.size() != layout.kinds.size()) {
        reason = "INVALID_LAYOUT_OR_ESTIMATE_VECTOR";
        return false;
    }
    selected.assign(layout.kinds.size(), 0);
    const bool explicit_selection = !layout.pc_cuda_layers.empty();

    // An explicit bitmap is a hard placement contract. Preserve its exact
    // semantics: every requested PC Full layer must have a valid estimate
    // and the whole selection must fit the CUDA mirror budget.
    if (explicit_selection) {
        for (size_t i = 0; i < layout.kinds.size(); ++i) {
            if (layout.kinds[i] != llama_hybrid_cycle_layer_kind::PC_CPU ||
                layout.pc_cuda_layers[i] == 0) {
                continue;
            }
            const size_t bytes = estimates[i];
            if (!bytes) {
                reason = "CUDA_ESTIMATE_UNAVAILABLE layer=" +
                         std::to_string(i);
                selected.clear();
                selected_bytes = 0;
                return false;
            }
            if (bytes > budget - selected_bytes) {
                reason = "CUDA_MIRROR_BUDGET_EXCEEDED layer=" +
                         std::to_string(i);
                selected.clear();
                selected_bytes = 0;
                return false;
            }
            selected[i] = 1;
            selected_bytes += bytes;
        }
        return true;
    }

    // Automatic placement must not spend the entire budget on the first
    // physical period. Build the PC Full candidate list in model order,
    // determine how many mirrors the budget can support, then choose that
    // many layers from evenly spaced strata across the whole 48-layer
    // pipeline. With the X4/Y2 production layout this gives, for example:
    //   3 mirrors -> physical layers 8,24,40
    //   6 mirrors -> one PC Full layer in each of the six periods.
    //
    // Estimates are almost equal for Qwen3-MoE, but the cheapest-first pass
    // below is only used to derive an upper bound on mirror COUNT. If the
    // evenly distributed set is more expensive, reduce the count until the
    // actual selected bytes fit. This never exceeds the requested budget.
    std::vector<size_t> candidates;
    std::vector<size_t> costs;
    candidates.reserve(layout.kinds.size());
    costs.reserve(layout.kinds.size());
    for (size_t i = 0; i < layout.kinds.size(); ++i) {
        if (layout.kinds[i] != llama_hybrid_cycle_layer_kind::PC_CPU ||
            estimates[i] == 0) {
            continue;
        }
        candidates.push_back(i);
        costs.push_back(estimates[i]);
    }
    if (candidates.empty() || budget == 0) {
        return true;
    }

    std::sort(costs.begin(), costs.end());
    size_t affordable = 0;
    size_t cheapest_bytes = 0;
    for (const size_t bytes : costs) {
        if (bytes > budget - cheapest_bytes) break;
        cheapest_bytes += bytes;
        ++affordable;
    }

    for (size_t count = affordable; count > 0; --count) {
        std::vector<size_t> trial;
        trial.reserve(count);
        size_t trial_bytes = 0;
        bool fits = true;
        const size_t n = candidates.size();
        for (size_t k = 0; k < count; ++k) {
            const size_t candidate_index =
                ((2 * k + 1) * n) / (2 * count);
            const size_t layer =
                candidates[std::min(candidate_index, n - 1)];
            const size_t bytes = estimates[layer];
            if (bytes > budget - trial_bytes) {
                fits = false;
                break;
            }
            trial.push_back(layer);
            trial_bytes += bytes;
        }
        if (!fits) continue;
        for (const size_t layer : trial) {
            selected[layer] = 1;
        }
        selected_bytes = trial_bytes;
        return true;
    }
    return true;
}

llama_hybrid_cycle_physical_resource_pool::
    ~llama_hybrid_cycle_physical_resource_pool() {
    clear();
}

void llama_hybrid_cycle_physical_resource_pool::clear() {
    std::lock_guard<std::mutex> lock(cuda_residency_mutex_);
    // Keep KV/model/Backend storage alive while cached graph metadata,
    // Scratch pools and any resident CUDA weight mirrors are released.
    for (auto & resource : layers_) {
        if (!resource) continue;
        resource->graph_cache.reset();
        resource->scratch_pool.reset();
        resource->kv.reset();
        resource->cuda_weights.reset();
        resource->cuda_residency_stamp = 0;
    }
    layers_.clear();
    layout_ = {};
    model_ = nullptr;
    cpu_ = nullptr;
    cuda_ = nullptr;
    cuda_residency_clock_ = 0;
    cuda_residency_loads_ = 0;
    cuda_residency_evictions_ = 0;
    cuda_mirror_bytes_planned_ = 0;
    cuda_mirror_layers_ = 0;
}

bool llama_hybrid_cycle_physical_resource_pool::prepare(
        const llama_model & model, const llama_hybrid_cycle_layout & layout,
        ggml_backend_t cpu, ggml_backend_t phone, ggml_backend_t cuda,
        const llama_hybrid_cycle_resource_config & config,
        std::string & reason) {
    clear();
    reason.clear();
    const auto fail = [&](const std::string & why) {
        clear();
        reason = why;
        return false;
    };

    if (!cpu || !phone || config.kv_tokens <= 0 ||
        config.kv_tokens > 512 || config.max_inflight <= 0 ||
        !llama_hybrid_cycle_runtime_validate_model(model, layout, reason)) {
        if (reason.empty()) reason = "INVALID_BACKENDS_OR_CONTEXT";
        return false;
    }
    if (!cuda) {
        return fail("PC_FULL_CUDA_BACKEND_REQUIRED");
    }

    auto * cuda_dev = ggml_backend_get_device(cuda);
    if (!cuda_dev || ggml_backend_dev_type(cuda_dev) !=
                        GGML_BACKEND_DEVICE_TYPE_GPU) {
        return fail("CUDA_BACKEND_DEVICE_INVALID");
    }

    size_t free_gpu = 0, total_gpu = 0;
    ggml_backend_dev_memory(cuda_dev, &free_gpu, &total_gpu);
    const size_t usable_gpu =
        free_gpu > config.cuda_safety_reserve_bytes ?
        free_gpu - config.cuda_safety_reserve_bytes : 0;
    const size_t requested_budget = config.cuda_weight_budget_bytes ?
        config.cuda_weight_budget_bytes : free_gpu;
    const size_t budget = std::min(requested_budget, usable_gpu);

    // Every ordinary PC Full layer is a CUDA task. The budget sizes only the
    // number of simultaneously resident full-weight mirrors; it no longer
    // selects CPU fallback layers.
    std::vector<size_t> estimated(layout.kinds.size(), 0);
    size_t pc_full_layers = 0;
    size_t max_layer_estimate = 0;
    for (size_t i = 0; i < layout.kinds.size(); ++i) {
        if (layout.kinds[i] != llama_hybrid_cycle_layer_kind::PC_CPU) continue;
        ++pc_full_layers;
        if (!cycle_cuda_mirror_estimate(
                model.layers[i], cpu, cuda, estimated[i])) {
            return fail("CUDA_WEIGHT_ESTIMATE_FAILED layer=" +
                        std::to_string(i));
        }
        max_layer_estimate = std::max(max_layer_estimate, estimated[i]);
    }
    if (pc_full_layers == 0 || max_layer_estimate == 0) {
        return fail("PC_FULL_CUDA_RESIDENCY_ESTIMATE_EMPTY");
    }

    const size_t resident_capacity = std::min(
        pc_full_layers, budget / max_layer_estimate);
    if (resident_capacity == 0) {
        return fail(
            "PC_FULL_CUDA_RESIDENCY_BUDGET_TOO_SMALL required_one_mib=" +
            std::to_string(max_layer_estimate / (1024 * 1024)) +
            " budget_mib=" + std::to_string(budget / (1024 * 1024)));
    }

    layout_ = layout;
    layout_.pc_cuda_layers.assign(layout_.kinds.size(), 0);
    for (size_t i = 0; i < layout_.kinds.size(); ++i) {
        if (layout_.kinds[i] == llama_hybrid_cycle_layer_kind::PC_CPU) {
            layout_.pc_cuda_layers[i] = 1;
        }
    }

    model_ = &model;
    cpu_ = cpu;
    cuda_ = cuda;
    cuda_mirror_layers_ = int(resident_capacity);
    cuda_mirror_bytes_planned_ = resident_capacity * max_layer_estimate;

    std::printf(
        "[CYCLE_CUDA_RESIDENCY_PLAN] policy=GPU_FULL_LRU "
        "pc_full_layers=%zu resident_slots=%zu pc_cpu_fallback=0 "
        "slot_estimate_mib=%.2f planned_mib=%.2f budget_mib=%.2f "
        "cuda_free_mib=%.2f status=READY\n",
        pc_full_layers, resident_capacity,
        double(max_layer_estimate) / (1024.0 * 1024.0),
        double(cuda_mirror_bytes_planned_) / (1024.0 * 1024.0),
        double(budget) / (1024.0 * 1024.0),
        double(free_gpu) / (1024.0 * 1024.0));

    const size_t n_layers = layout_.kinds.size();
    layers_.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        auto layer = std::make_unique<llama_hybrid_cycle_physical_layer_resources>();
        layer->physical_layer = int(i);
        layer->kind = layout_.kinds[i];
        const bool pc_full =
            layer->kind == llama_hybrid_cycle_layer_kind::PC_CPU;
        const bool phone_layer =
            layer->kind == llama_hybrid_cycle_layer_kind::PHONE_GPU ||
            layer->kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C;
        const bool cuda_tail =
            layer->kind == llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL;
        // Tensor C2P remains CPU by design. Ordinary PC Full is always CUDA;
        // its weights are populated lazily by the residency cache.
        layer->attn_backend = phone_layer ? phone :
            (pc_full || cuda_tail ? cuda : cpu);
        if (!layer->attn_backend) {
            return fail("PHYSICAL_BACKEND_MISSING layer=" + std::to_string(i));
        }
        // Insert before any allocations so clear() always handles partial
        // construction in reverse dependency order.
        layers_.push_back(std::move(layer));
        auto & resource = *layers_.back();
        resource.kv = std::make_unique<llama_hybrid_cycle_physical_kv_cache>();
        if (!resource.kv->init(
                resource.attn_backend, model.layers[i], config.kv_tokens,
                phone_layer ? 1 : 0, nullptr)) {
            return fail("PHYSICAL_KV_ALLOC_FAILED layer=" +
                        std::to_string(i));
        }
        if (config.attention_graph_cache) {
            resource.graph_cache =
                std::make_unique<cycle_real_attention_graph_cache>();
            resource.graph_cache->max_graphs =
                config.attention_graph_cache_max_graphs;
        }
        if (config.attention_scratch_pool) {
            resource.scratch_pool =
                std::make_unique<cycle_real_attention_scratch_pool>();
            resource.scratch_pool->configure(
                resource.attn_backend, config.max_inflight);
        }
    }
    std::printf(
        "[CYCLE_FULL_RESOURCE_PLAN] layers=%zu kv_tokens=%" PRId64
        " max_inflight=%d pc_full_backend=CUDA pc_cpu_fallback=0"
        " cuda_resident_slots=%d cuda_residency_planned_mib=%.2f"
        " cuda_cap_mib=%.2f cuda_free_mib=%.2f"
        " graph_cache=%d scratch_pool=%d status=READY_ONLY\n",
        n_layers, config.kv_tokens, config.max_inflight,
        cuda_mirror_layers_,
        double(cuda_mirror_bytes_planned_) / (1024.0 * 1024.0),
        double(budget) / (1024.0 * 1024.0),
        double(free_gpu) / (1024.0 * 1024.0),
        config.attention_graph_cache ? 1 : 0,
        config.attention_scratch_pool ? 1 : 0);
    return true;
}

const llama_hybrid_cycle_layout &
llama_hybrid_cycle_physical_resource_pool::execution_layout() const {
    return layout_;
}

const llama_hybrid_cycle_physical_layer_resources *
llama_hybrid_cycle_physical_resource_pool::layer(int physical_layer) const {
    if (physical_layer < 0 ||
        static_cast<size_t>(physical_layer) >= layers_.size()) return nullptr;
    return layers_[static_cast<size_t>(physical_layer)].get();
}

bool llama_hybrid_cycle_physical_resource_pool::ensure_pc_full_cuda_resident(
        int physical_layer, std::string & reason) {
    std::lock_guard<std::mutex> lock(cuda_residency_mutex_);
    reason.clear();
    if (!model_ || !cpu_ || !cuda_ || cuda_mirror_layers_ <= 0 ||
        physical_layer < 0 ||
        static_cast<size_t>(physical_layer) >= layers_.size()) {
        reason = "CUDA_RESIDENCY_NOT_PREPARED";
        return false;
    }
    auto & target = layers_[static_cast<size_t>(physical_layer)];
    if (!target || target->kind != llama_hybrid_cycle_layer_kind::PC_CPU ||
        target->attn_backend != cuda_) {
        reason = "CUDA_RESIDENCY_TARGET_NOT_PC_FULL";
        return false;
    }

    const uint64_t stamp = ++cuda_residency_clock_;
    if (target->cuda_weights) {
        target->cuda_residency_stamp = stamp;
        return true;
    }

    int resident = 0;
    int victim_layer = -1;
    uint64_t victim_stamp = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto & item = layers_[i];
        if (!item || !item->cuda_weights) continue;
        ++resident;
        if (item->cuda_residency_stamp < victim_stamp) {
            victim_stamp = item->cuda_residency_stamp;
            victim_layer = int(i);
        }
    }

    std::unique_ptr<llama_hybrid_cycle_cuda_full_weights> slot;
    if (resident >= cuda_mirror_layers_) {
        if (victim_layer < 0) {
            reason = "CUDA_RESIDENCY_EVICTION_TARGET_MISSING";
            return false;
        }
        auto & victim = layers_[static_cast<size_t>(victim_layer)];
        slot = std::move(victim->cuda_weights);
        victim->cuda_residency_stamp = 0;
        ++cuda_residency_evictions_;
        if (llama_hybrid_cycle_verbose()) {
            std::printf(
                "[CYCLE_CUDA_RESIDENCY] event=EVICT layer=%d for_layer=%d "
                "resident_slots=%d status=OK\n",
                victim_layer, physical_layer, cuda_mirror_layers_);
        }
    } else {
        slot = std::make_unique<llama_hybrid_cycle_cuda_full_weights>();
    }

    if (!slot || !slot->prepare(
            cuda_, cpu_, model_->layers[static_cast<size_t>(physical_layer)],
            physical_layer)) {
        reason = "CUDA_RESIDENCY_LOAD_FAILED layer=" +
                 std::to_string(physical_layer);
        return false;
    }
    target->cuda_weights = std::move(slot);
    target->cuda_residency_stamp = stamp;
    ++cuda_residency_loads_;
    if (llama_hybrid_cycle_verbose()) {
        std::printf(
            "[CYCLE_CUDA_RESIDENCY] event=LOAD layer=%d load_index=%llu "
            "evictions=%llu resident_slots=%d status=OK\n",
            physical_layer,
            (unsigned long long) cuda_residency_loads_,
            (unsigned long long) cuda_residency_evictions_,
            cuda_mirror_layers_);
    }
    return true;
}

int llama_hybrid_cycle_physical_resource_pool::cuda_resident_layers() const {
    std::lock_guard<std::mutex> lock(cuda_residency_mutex_);
    int count = 0;
    for (const auto & item : layers_) {
        if (item && item->cuda_weights) ++count;
    }
    return count;
}

uint64_t llama_hybrid_cycle_physical_resource_pool::cuda_residency_loads() const {
    std::lock_guard<std::mutex> lock(cuda_residency_mutex_);
    return cuda_residency_loads_;
}

uint64_t llama_hybrid_cycle_physical_resource_pool::cuda_residency_evictions() const {
    std::lock_guard<std::mutex> lock(cuda_residency_mutex_);
    return cuda_residency_evictions_;
}

size_t llama_hybrid_cycle_physical_resource_pool::
    cuda_mirror_bytes_planned() const {
    return cuda_mirror_bytes_planned_;
}

int llama_hybrid_cycle_physical_resource_pool::cuda_mirror_layers() const {
    return cuda_mirror_layers_;
}

int llama_hybrid_cycle_physical_resource_pool::prepared_layers() const {
    return int(layers_.size());
}
