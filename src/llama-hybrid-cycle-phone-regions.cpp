#include "llama-hybrid-cycle-phone-regions.h"

#include "llama-hybrid-cycle-resources.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <utility>

void llama_hybrid_cycle_phone_regions::clear() {
    regions_.clear();
    layout_ = {};
    n_chunks_ = 0;
    tokens_ = 0;
}

bool llama_hybrid_cycle_phone_regions::prepare(
        const llama_hybrid_cycle_layout & physical_layout,
        int64_t tokens, int chunk_tokens, int max_inflight,
        std::string & reason) {
    clear();
    reason.clear();
    if (!physical_layout.enabled() ||
        !llama_hybrid_cycle_cuda_layers_valid(physical_layout) ||
        physical_layout.cycle_size <= 0 || physical_layout.cycles <= 0 ||
        physical_layout.x <= 0 || physical_layout.y <= 0 ||
        int64_t(physical_layout.x) + physical_layout.y + 2 !=
            physical_layout.cycle_size ||
        int64_t(physical_layout.cycles) * physical_layout.cycle_size !=
            int64_t(physical_layout.kinds.size()) ||
        physical_layout.tail_begin != int(physical_layout.kinds.size())) {
        reason = "INVALID_FULL_PHYSICAL_PERIOD_LAYOUT";
        return false;
    }
    // 'tokens' is the complete Prefill invocation (up to the physical
    // 512-token context). Cycle owns the only batching layer; each internal
    // Chunk is at most 64 tokens so READY fits the 2-MiB data-plane payload.
    if (tokens <= 0 || tokens > 512 ||
        chunk_tokens <= 0 || chunk_tokens > 64 ||
        max_inflight < 0 || max_inflight > tokens ||
        1 + (tokens - 1) / chunk_tokens > 999) {
        reason = "INVALID_PHONE_TOKENS_OR_ADMISSION";
        return false;
    }
    const int chunks = int(1 + (tokens - 1) / chunk_tokens);
    if (max_inflight != 0 && max_inflight > chunks) {
        reason = "MAX_INFLIGHT_EXCEEDS_CHUNKS";
        return false;
    }
    llama_hybrid_cycle_runtime_config rc;
    rc.layout = physical_layout;
    rc.n_chunks = chunks;
    rc.max_inflight = max_inflight;
    llama_hybrid_cycle_runtime runtime(std::move(rc));
    std::vector<llama_hybrid_cycle_phone_region_progress> pending;
    pending.reserve(size_t(physical_layout.cycles));
    for (int cycle = 0; cycle < physical_layout.cycles; ++cycle) {
        llama_hybrid_cycle_runtime_region placement;
        if (!runtime.region_for_layer(cycle * physical_layout.cycle_size,
                                      placement) ||
            placement.cycle_index != cycle ||
            placement.end != (cycle + 1) * physical_layout.cycle_size ||
            placement.phone_end - placement.phone_begin != physical_layout.y) {
            reason = "PHYSICAL_REGION_ROLE_MISMATCH cycle=" +
                     std::to_string(cycle);
            return false;
        }
        llama_hybrid_cycle_phone_region_progress state;
        state.location = placement;
        state.n_chunks = chunks;
        state.max_inflight = max_inflight;
        state.c2p_ready_chunks.assign(size_t(chunks), uint8_t(0));
        pending.emplace_back(std::move(state));
    }
    regions_ = std::move(pending);
    layout_ = physical_layout;
    n_chunks_ = chunks;
    tokens_ = tokens;
    return true;
}

int llama_hybrid_cycle_phone_regions::region_count() const {
    return int(regions_.size());
}
int llama_hybrid_cycle_phone_regions::n_chunks() const {
    return n_chunks_;
}
const llama_hybrid_cycle_layout & llama_hybrid_cycle_phone_regions::layout() const {
    return layout_;
}

llama_hybrid_cycle_phone_region_progress *
llama_hybrid_cycle_phone_regions::by_index(int index) {
    return index >= 0 && size_t(index) < regions_.size() ?
        &regions_[size_t(index)] : nullptr;
}
const llama_hybrid_cycle_phone_region_progress *
llama_hybrid_cycle_phone_regions::by_index(int index) const {
    return index >= 0 && size_t(index) < regions_.size() ?
        &regions_[size_t(index)] : nullptr;
}
llama_hybrid_cycle_phone_region_progress *
llama_hybrid_cycle_phone_regions::for_layer(int physical_layer) {
    if (physical_layer < 0 || layout_.cycle_size <= 0 ||
        physical_layer >= layout_.tail_begin) return nullptr;
    return by_index(physical_layer / layout_.cycle_size);
}
const llama_hybrid_cycle_phone_region_progress *
llama_hybrid_cycle_phone_regions::for_layer(int physical_layer) const {
    if (physical_layer < 0 || layout_.cycle_size <= 0 ||
        physical_layer >= layout_.tail_begin) return nullptr;
    return by_index(physical_layer / layout_.cycle_size);
}

bool llama_hybrid_cycle_phone_regions::validate_resources(
        const llama_hybrid_cycle_physical_resource_pool & resources,
        std::string & reason) const {
    reason.clear();
    const auto & actual = resources.execution_layout();
    if (regions_.empty() ||
        actual.kinds != layout_.kinds ||
        actual.x != layout_.x || actual.y != layout_.y ||
        actual.cycle_size != layout_.cycle_size ||
        actual.pc_cuda_full != layout_.pc_cuda_full ||
        actual.pc_cuda_layers != layout_.pc_cuda_layers ||
        resources.prepared_layers() != int(layout_.kinds.size())) {
        reason = "PHONE_REGION_RESOURCE_LAYOUT_MISMATCH";
        return false;
    }
    std::set<ggml_backend_buffer_t> kv_allocations;
    std::set<const void *> graph_owners;
    std::set<const void *> scratch_owners;
    ggml_backend_t first_phone_backend = nullptr;
    for (const auto & region : regions_) {
        const auto * phone_start =
            resources.layer(region.location.phone_begin);
        if (!phone_start || !phone_start->attn_backend) {
            reason = "PHONE_REGION_BACKEND_MISSING cycle=" +
                     std::to_string(region.location.cycle_index);
            return false;
        }
        if (first_phone_backend == nullptr) {
            first_phone_backend = phone_start->attn_backend;
        } else if (phone_start->attn_backend != first_phone_backend) {
            reason = "PHONE_REGION_DIFFERENT_RPC_BACKEND cycle=" +
                     std::to_string(region.location.cycle_index);
            return false;
        }
        for (int layer = region.location.c2p_layer;
             layer <= region.location.p2c_layer; ++layer) {
            const auto * resource = resources.layer(layer);
            if (!resource || resource->physical_layer != layer ||
                resource->kind != layout_.kinds[size_t(layer)] ||
                !resource->attn_backend || !resource->kv ||
                !resource->kv->buffer ||
                !resource->kv->keys || !resource->kv->values ||
                resource->kv->max_tokens < tokens_ ||
                !resource->graph_cache || !resource->scratch_pool) {
                reason = "PHONE_REGION_PHYSICAL_RESOURCE_MISSING layer=" +
                         std::to_string(layer);
                return false;
            }
            const bool phone_attention =
                layer >= region.location.phone_begin;
            if ((phone_attention &&
                 resource->attn_backend != first_phone_backend) ||
                (!phone_attention &&
                 resource->attn_backend == first_phone_backend)) {
                reason = "PHONE_REGION_ATTENTION_BACKEND_MISMATCH layer=" +
                         std::to_string(layer);
                return false;
            }
            if (!kv_allocations.insert(resource->kv->buffer).second ||
                !graph_owners.insert(resource->graph_cache.get()).second ||
                !scratch_owners.insert(resource->scratch_pool.get()).second) {
                reason = "PHONE_REGION_CROSS_LAYER_RESOURCE_ALIAS layer=" +
                         std::to_string(layer);
                return false;
            }
        }
    }
    return true;
}

bool llama_hybrid_cycle_phone_regions::bind_phone_stages(
        llama_hybrid_cycle_stage_dispatcher & stages,
        real_stage_handler execute, std::string & reason) {
    reason.clear();
    if (!execute || regions_.empty() || !stages.matches(layout_)) {
        reason = "PHONE_REGION_INVALID_DISPATCHER_OR_HANDLER";
        return false;
    }
    // StageDispatcher handlers are immutable once the FIFO starts.
    // Capture this registry by pointer; its lifetime must extend past the
    // five worker joins. Never capture a temporary local period state.
    const auto common = std::make_shared<real_stage_handler>(std::move(execute));
    for (size_t i = 0; i < regions_.size(); ++i) {
        const auto & r = regions_[i].location;
        const auto bind = [&](int physical_layer, llama_hybrid_cycle_stage stage) {
            return stages.bind(physical_layer, stage,
                [this, i, common](const llama_hybrid_cycle_task & task) {
                    const auto & region = regions_[i].location;
                    if (task.layer < region.c2p_layer ||
                        task.layer > region.p2c_layer ||
                        task.chunk < 0 || task.chunk >= n_chunks_) {
                        return false;
                    }
                    return (*common)(regions_[i], task);
                });
        };
        for (int l = r.c2p_layer; l <= r.p2c_layer; ++l) {
            const auto kind = layout_.kinds[size_t(l)];
            if (kind == llama_hybrid_cycle_layer_kind::PHONE_GPU) {
                if (!bind(l, llama_hybrid_cycle_stage::FULL_LAYER)) {
                    reason = "PHONE_FULL_STAGE_ALREADY_BOUND layer=" +
                             std::to_string(l);
                    return false;
                }
            } else if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P ||
                       kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
                for (const auto stage : {
                        llama_hybrid_cycle_stage::ATTN_PRE,
                        llama_hybrid_cycle_stage::SEND_ACTIVATION,
                        llama_hybrid_cycle_stage::FFN_SOURCE,
                        llama_hybrid_cycle_stage::FFN_DESTINATION,
                        llama_hybrid_cycle_stage::SEND_PARTIAL,
                        llama_hybrid_cycle_stage::REDUCE}) {
                    if (!bind(l, stage)) {
                        reason = "PHONE_TENSOR_STAGE_ALREADY_BOUND layer=" +
                                 std::to_string(l);
                        return false;
                    }
                }
            } else {
                reason = "PHONE_REGION_UNEXPECTED_LAYER_KIND layer=" +
                         std::to_string(l);
                return false;
            }
        }
    }
    return true;
}
