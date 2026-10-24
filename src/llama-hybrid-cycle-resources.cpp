#include "llama-hybrid-cycle-resources.h"

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
    // Keep KV/model/Backend storage alive while the cached graph metadata
    // and Scratch pools release their borrowed tensor references.
    for (auto & resource : layers_) {
        if (!resource) continue;
        resource->graph_cache.reset();
        resource->scratch_pool.reset();
        resource->kv.reset();
        resource->cuda_weights.reset();
    }
    layers_.clear();
    layout_ = {};
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
    if (config.cuda_weight_budget_bytes && !cuda) {
        reason = "CUDA_BUDGET_WITHOUT_BACKEND";
        return false;
    }

    size_t budget = config.cuda_weight_budget_bytes;
    size_t free_gpu = 0, total_gpu = 0;
    if (cuda && budget) {
        auto * dev = ggml_backend_get_device(cuda);
        if (!dev || ggml_backend_dev_type(dev) !=
                        GGML_BACKEND_DEVICE_TYPE_GPU) {
            return fail("CUDA_BACKEND_DEVICE_INVALID");
        }
        ggml_backend_dev_memory(dev, &free_gpu, &total_gpu);
        budget = free_gpu > config.cuda_safety_reserve_bytes ?
            std::min(budget, free_gpu - config.cuda_safety_reserve_bytes) : 0;
    }
    std::vector<size_t> estimated(layout.kinds.size(), 0);
    if (budget || !layout.pc_cuda_layers.empty()) {
        for (size_t i = 0; i < layout.kinds.size(); ++i) {
            if (layout.kinds[i] != llama_hybrid_cycle_layer_kind::PC_CPU ||
                (!layout.pc_cuda_layers.empty() &&
                 layout.pc_cuda_layers[i] == 0)) continue;
            if (!cycle_cuda_mirror_estimate(
                    model.layers[i], cpu, cuda, estimated[i])) {
                return fail("CUDA_WEIGHT_ESTIMATE_FAILED layer=" +
                            std::to_string(i));
            }
        }
    }
    std::vector<uint8_t> selected;
    size_t selected_bytes = 0;
    if (!llama_hybrid_cycle_select_cuda_full_layers(
            layout, estimated, budget, selected, selected_bytes, reason)) {
        return fail(reason);
    }
    std::string selected_layers;
    for (size_t i = 0; i < selected.size(); ++i) {
        if (!selected[i]) continue;
        if (!selected_layers.empty()) selected_layers += ",";
        selected_layers += std::to_string(i);
    }
    std::printf(
        "[CYCLE_CUDA_SELECTION] policy=%s selected=%zu layers=%s "
        "planned_mib=%.2f budget_mib=%.2f status=READY\n",
        layout.pc_cuda_layers.empty() ? "BALANCED_STRATA" : "EXPLICIT",
        size_t(std::count(selected.begin(), selected.end(), uint8_t(1))),
        selected_layers.empty() ? "none" : selected_layers.c_str(),
        double(selected_bytes) / (1024.0 * 1024.0),
        double(budget) / (1024.0 * 1024.0));
    layout_ = layout;
    // The explicit bitmap OVERRIDES legacy pc_cuda_full=true. Unmirrored
    // PC layers always route to PC_CPU, including on full 48-layer runs.
    layout_.pc_cuda_layers = std::move(selected);
    cuda_mirror_bytes_planned_ = selected_bytes;
    const size_t n_layers = layout_.kinds.size();
    layers_.reserve(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        auto layer = std::make_unique<llama_hybrid_cycle_physical_layer_resources>();
        layer->physical_layer = int(i);
        layer->kind = layout_.kinds[i];
        const bool mirrored = llama_hybrid_cycle_full_layer_on_cuda(
            layout_, int(i));
        const bool phone_layer = layer->kind == llama_hybrid_cycle_layer_kind::PHONE_GPU ||
                                  layer->kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C;
        const bool cuda_tail = layer->kind == llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL;
        layer->attn_backend = phone_layer ? phone :
            (mirrored || cuda_tail ? cuda : cpu);
        if (!layer->attn_backend) {
            return fail("PHYSICAL_BACKEND_MISSING layer=" + std::to_string(i));
        }
        // Insert before any allocations so clear() always handles partial
        // construction in reverse dependency order.
        layers_.push_back(std::move(layer));
        auto & resource = *layers_.back();
        if (mirrored && resource.kind == llama_hybrid_cycle_layer_kind::PC_CPU) {
            resource.cuda_weights =
                std::make_unique<llama_hybrid_cycle_cuda_full_weights>();
            if (!resource.cuda_weights->prepare(
                    cuda, cpu, model.layers[i], int(i))) {
                return fail("CUDA_WEIGHT_MIRROR_ALLOCATION_FAILED layer=" +
                            std::to_string(i));
            }
            ++cuda_mirror_layers_;
        }
        resource.kv = std::make_unique<llama_hybrid_cycle_physical_kv_cache>();
        if (!resource.kv->init(
                resource.attn_backend, model.layers[i], config.kv_tokens,
                phone_layer ? 1 : 0, resource.cuda_weights.get())) {
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
        " max_inflight=%d cuda_mirror_layers=%d pc_cpu_fallback=%zu"
        " cuda_estimated_mib=%.2f cuda_cap_mib=%.2f"
        " cuda_free_mib=%.2f graph_cache=%d scratch_pool=%d"
        " status=READY_ONLY\n",
        n_layers, config.kv_tokens, config.max_inflight,
        cuda_mirror_layers_,
        size_t(std::count(layout_.kinds.begin(), layout_.kinds.end(),
              llama_hybrid_cycle_layer_kind::PC_CPU)) -
              size_t(cuda_mirror_layers_),
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
