#include "llama-region-cuda-mirror.h"

#include "llama-hybrid-cycle-attention.h"
#include "llama-hybrid-cycle-log.h"
#include "llama-hybrid-cycle-phone-executor.h"
#include "llama-model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

bool region_cuda_mirror_estimate(
        const llama_layer & layer,
        ggml_backend_t cpu,
        ggml_backend_t cuda,
        size_t & bytes) {
    bytes = 0;
    if (!cpu || !cuda) {
        return false;
    }
    const auto buft =
        ggml_backend_get_default_buffer_type(cuda);
    if (!buft) {
        return false;
    }

    const ggml_tensor * logical[] = {
        layer.attn_norm,
        layer.wq,
        layer.wk,
        layer.wv,
        layer.wo,
        layer.attn_q_norm,
        layer.attn_k_norm,
        layer.ffn_norm,
        layer.ffn_gate_inp,
        layer.ffn_up_exps,
        layer.ffn_gate_exps,
        layer.ffn_down_exps,
    };

    const size_t alignment =
        std::max<size_t>(
            1,
            ggml_backend_buft_get_alignment(buft));
    for (const ggml_tensor * tensor : logical) {
        const ggml_tensor * physical =
            llama_hybrid_cycle_resolve_physical_weight(
                tensor, cpu, 0);
        if (!physical || !physical->buffer ||
                !ggml_is_contiguous(physical)) {
            return false;
        }
        size_t n =
            ggml_backend_buft_get_alloc_size(
                buft, physical);
        if (n == 0 ||
                n > std::numeric_limits<size_t>::max() -
                    (alignment - 1)) {
            return false;
        }
        n =
            ((n + alignment - 1) / alignment) *
            alignment;
        if (bytes >
                std::numeric_limits<size_t>::max() - n) {
            return false;
        }
        bytes += n;
    }

    const size_t minimum_headroom =
        8ULL * 1024 * 1024;
    const size_t fractional_headroom =
        bytes / 20;
    const size_t extra =
        std::max(
            minimum_headroom,
            fractional_headroom);
    if (bytes >
            std::numeric_limits<size_t>::max() - extra) {
        return false;
    }
    bytes += extra;
    return true;
}

size_t env_mib(
        const char * name,
        size_t fallback_mib) {
    const char * raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return fallback_mib;
    }
    char * tail = nullptr;
    const unsigned long long value =
        std::strtoull(raw, &tail, 10);
    if (raw[0] == '-' || tail == raw ||
            (tail && *tail != '\0') ||
            value >
                std::numeric_limits<size_t>::max() /
                    (1024ULL * 1024ULL)) {
        return fallback_mib;
    }
    return static_cast<size_t>(value);
}

} // namespace

llama_region_cuda_mirror::~llama_region_cuda_mirror() {
    clear();
}

void llama_region_cuda_mirror::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto & resource : layers_) {
        if (!resource) {
            continue;
        }
        resource->scratch.reset();
        resource->kv.reset();
        resource->weights.reset();
        resource->stamp = 0;
    }
    layers_.clear();
    model_ = nullptr;
    cpu_ = nullptr;
    cuda_ = nullptr;
    clock_ = 0;
    loads_ = 0;
    evictions_ = 0;
    resident_capacity_ = 0;
    planned_bytes_ = 0;
    max_tokens_ = 0;
}

bool llama_region_cuda_mirror::prepare(
        const llama_model & model,
        ggml_backend_t cpu,
        ggml_backend_t cuda,
        int64_t max_tokens,
        size_t requested_budget_bytes,
        std::string & reason) {
    clear();
    reason.clear();

    auto fail = [&](const std::string & why) {
        clear();
        reason = why;
        return false;
    };

    if (!cpu || !cuda ||
            model.arch != LLM_ARCH_QWEN3MOE ||
            max_tokens <= 0 || max_tokens > 512) {
        return fail("INVALID_MODEL_BACKEND_OR_CONTEXT");
    }

    auto * cpu_dev = ggml_backend_get_device(cpu);
    auto * cuda_dev = ggml_backend_get_device(cuda);
    if (!cpu_dev || !cuda_dev ||
            ggml_backend_dev_type(cpu_dev) !=
                GGML_BACKEND_DEVICE_TYPE_CPU ||
            ggml_backend_dev_type(cuda_dev) !=
                GGML_BACKEND_DEVICE_TYPE_GPU) {
        return fail("CPU_OR_CUDA_DEVICE_INVALID");
    }

    const int n_layer = model.hparams.n_layer();
    if (n_layer <= 0 ||
            static_cast<size_t>(n_layer) >
                model.layers.size()) {
        return fail("INVALID_LAYER_COUNT");
    }

    std::vector<size_t> estimates(
        static_cast<size_t>(n_layer), 0);
    size_t pc_full_layers = 0;
    size_t max_layer_estimate = 0;
    for (int il = 0; il < n_layer; ++il) {
        if (model.hybrid_layer_mode(il) !=
                llama_hybrid_layer_mode::PC_ONLY) {
            continue;
        }
        ++pc_full_layers;
        if (!region_cuda_mirror_estimate(
                model.layers[static_cast<size_t>(il)],
                cpu, cuda,
                estimates[static_cast<size_t>(il)])) {
            return fail(
                "CUDA_WEIGHT_ESTIMATE_FAILED layer=" +
                std::to_string(il));
        }
        max_layer_estimate =
            std::max(
                max_layer_estimate,
                estimates[static_cast<size_t>(il)]);
    }
    if (pc_full_layers == 0 ||
            max_layer_estimate == 0) {
        return fail("NO_PC_FULL_LAYER");
    }

    size_t free_gpu = 0;
    size_t total_gpu = 0;
    ggml_backend_dev_memory(
        cuda_dev, &free_gpu, &total_gpu);

    const size_t safety_mib =
        env_mib(
            "LLAMA_HYBRID_REGION_CUDA_SAFETY_MIB",
            512);
    const size_t safety_bytes =
        safety_mib * 1024ULL * 1024ULL;
    const size_t usable_gpu =
        free_gpu > safety_bytes ?
            free_gpu - safety_bytes : 0;
    const size_t requested =
        requested_budget_bytes != 0 ?
            requested_budget_bytes : usable_gpu;
    const size_t budget =
        std::min(requested, usable_gpu);
    const size_t resident_capacity =
        std::min(
            pc_full_layers,
            budget / max_layer_estimate);
    if (resident_capacity == 0) {
        return fail(
            "CUDA_RESIDENCY_BUDGET_TOO_SMALL required_one_mib=" +
            std::to_string(
                max_layer_estimate /
                (1024ULL * 1024ULL)) +
            " budget_mib=" +
            std::to_string(
                budget /
                (1024ULL * 1024ULL)));
    }

    model_ = &model;
    cpu_ = cpu;
    cuda_ = cuda;
    max_tokens_ = max_tokens;
    resident_capacity_ =
        static_cast<int>(resident_capacity);
    planned_bytes_ =
        resident_capacity * max_layer_estimate;

    layers_.resize(
        static_cast<size_t>(n_layer));
    for (int il = 0; il < n_layer; ++il) {
        if (model.hybrid_layer_mode(il) !=
                llama_hybrid_layer_mode::PC_ONLY) {
            continue;
        }
        auto resource =
            std::make_unique<layer_resource>();
        resource->layer = il;
        resource->kv =
            std::make_unique<
                llama_hybrid_cycle_physical_kv_cache>();
        if (!resource->kv->init(
                cuda,
                model.layers[
                    static_cast<size_t>(il)],
                max_tokens,
                0,
                nullptr)) {
            return fail(
                "CUDA_KV_ALLOC_FAILED layer=" +
                std::to_string(il));
        }

        // Scratch reuse is safe across LRU weight replacement because it is
        // weight-independent. Attention graph metadata is intentionally not
        // cached yet: cached graphs borrow mirror tensor pointers and would
        // need a residency-generation key before surviving an eviction.
        resource->scratch =
            std::make_unique<
                cycle_real_attention_scratch_pool>();
        resource->scratch->configure(cuda, 2);

        layers_[static_cast<size_t>(il)] =
            std::move(resource);
    }

    std::printf(
        "[REGION_CUDA_MIRROR_PLAN] policy=GPU_FULL_LRU "
        "pc_full_layers=%zu resident_slots=%zu "
        "pc_cpu_fallback=0 slot_estimate_mib=%.2f "
        "planned_mib=%.2f budget_mib=%.2f "
        "cuda_free_mib=%.2f safety_mib=%zu "
        "kv_tokens=%lld graph_cache=0 scratch_pool=1 "
        "status=READY\n",
        pc_full_layers,
        resident_capacity,
        double(max_layer_estimate) /
            (1024.0 * 1024.0),
        double(planned_bytes_) /
            (1024.0 * 1024.0),
        double(budget) /
            (1024.0 * 1024.0),
        double(free_gpu) /
            (1024.0 * 1024.0),
        safety_mib,
        (long long) max_tokens);
    return true;
}

bool llama_region_cuda_mirror::ensure_resident(
        int layer,
        std::string & reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    reason.clear();

    if (!model_ || !cpu_ || !cuda_ ||
            resident_capacity_ <= 0 ||
            layer < 0 ||
            static_cast<size_t>(layer) >=
                layers_.size()) {
        reason = "CUDA_RESIDENCY_NOT_PREPARED";
        return false;
    }

    auto & target =
        layers_[static_cast<size_t>(layer)];
    if (!target ||
            model_->hybrid_layer_mode(layer) !=
                llama_hybrid_layer_mode::PC_ONLY) {
        reason =
            "CUDA_RESIDENCY_TARGET_NOT_PC_FULL";
        return false;
    }

    const uint64_t stamp = ++clock_;
    if (target->weights) {
        target->stamp = stamp;
        return true;
    }

    int resident = 0;
    int victim_layer = -1;
    uint64_t victim_stamp =
        std::numeric_limits<uint64_t>::max();
    for (size_t i = 0;
            i < layers_.size(); ++i) {
        const auto & item = layers_[i];
        if (!item || !item->weights) {
            continue;
        }
        ++resident;
        if (item->stamp < victim_stamp) {
            victim_stamp = item->stamp;
            victim_layer =
                static_cast<int>(i);
        }
    }

    std::unique_ptr<
        llama_hybrid_cycle_cuda_full_weights>
        slot;
    if (resident >= resident_capacity_) {
        if (victim_layer < 0) {
            reason =
                "CUDA_RESIDENCY_EVICTION_TARGET_MISSING";
            return false;
        }
        auto & victim =
            layers_[
                static_cast<size_t>(victim_layer)];
        slot = std::move(victim->weights);
        victim->stamp = 0;
        ++evictions_;
        if (llama_hybrid_cycle_verbose()) {
            std::printf(
                "[REGION_CUDA_MIRROR] event=EVICT "
                "layer=%d for_layer=%d resident_slots=%d "
                "status=OK\n",
                victim_layer,
                layer,
                resident_capacity_);
        }
    } else {
        slot =
            std::make_unique<
                llama_hybrid_cycle_cuda_full_weights>();
    }

    if (!slot ||
            !slot->prepare(
                cuda_,
                cpu_,
                model_->layers[
                    static_cast<size_t>(layer)],
                layer)) {
        reason =
            "CUDA_RESIDENCY_LOAD_FAILED layer=" +
            std::to_string(layer);
        return false;
    }

    target->weights = std::move(slot);
    target->stamp = stamp;
    ++loads_;
    if (llama_hybrid_cycle_verbose()) {
        std::printf(
            "[REGION_CUDA_MIRROR] event=LOAD "
            "layer=%d load_index=%llu evictions=%llu "
            "resident_slots=%d status=OK\n",
            layer,
            (unsigned long long) loads_,
            (unsigned long long) evictions_,
            resident_capacity_);
    }
    return true;
}

bool llama_region_cuda_mirror::execute_layer(
        int layer,
        int64_t token_start,
        int64_t tokens,
        const std::vector<float> & input,
        std::vector<float> & output,
        llama_region_cuda_mirror_timing * timing,
        const ggml_tensor * device_input,
        llama_hybrid_cycle_cuda_activation_slot * device_output,
        std::string & reason) {
    output.clear();
    reason.clear();

    if (!model_ || !cuda_ ||
            layer < 0 ||
            static_cast<size_t>(layer) >=
                layers_.size() ||
            tokens <= 0 || tokens > 128) {
        reason = "INVALID_PC_FULL_EXECUTE";
        return false;
    }
    if (!ensure_resident(layer, reason)) {
        return false;
    }

    auto & resource =
        layers_[static_cast<size_t>(layer)];
    if (!resource || !resource->weights ||
            !resource->kv ||
            !resource->scratch) {
        reason = "PC_FULL_RESOURCE_MISSING";
        return false;
    }

    const int64_t hidden =
        model_->hparams.n_embd;
    const int64_t topk =
        model_->hparams.n_expert_used;
    const int64_t experts =
        model_->hparams.n_expert;
    if (hidden <= 0 ||
            topk <= 0 ||
            experts < topk ||
            token_start < 0 ||
            token_start + tokens >
                resource->kv->max_tokens ||
            resource->kv->committed_tokens !=
                token_start) {
        reason =
            "PC_FULL_KV_OR_SHAPE_MISMATCH layer=" +
            std::to_string(layer);
        return false;
    }

    const size_t elements =
        static_cast<size_t>(hidden) *
        static_cast<size_t>(tokens);
    if (device_input == nullptr &&
            input.size() != elements) {
        reason = "PC_FULL_INPUT_SHAPE";
        return false;
    }

    const auto & lw =
        model_->layers[
            static_cast<size_t>(layer)];
    const auto shard =
        resource->weights->full_ffn_shard();
    if (!shard.available()) {
        reason = "PC_FULL_FFN_MIRROR_MISSING";
        return false;
    }

    llama_hybrid_cycle_attention_ready ready;
    double attn_kernel_ms = 0.0;
    const auto attn_begin =
        std::chrono::steady_clock::now();
    const bool attention_ok =
        llama_hybrid_cycle_execute_attention_pre(
            cuda_,
            lw,
            model_->hparams,
            input,
            hidden,
            tokens,
            topk,
            experts,
            ready,
            attn_kernel_ms,
            nullptr,
            resource->kv.get(),
            token_start,
            0,
            device_input,
            true,
            nullptr,
            resource->weights.get(),
            resource->scratch.get(),
            nullptr);
    if (timing) {
        timing->attention_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                attn_begin).count();
    }
    if (!attention_ok) {
        reason =
            "PC_FULL_ATTENTION_FAILED layer=" +
            std::to_string(layer);
        return false;
    }

    const auto ffn_begin =
        std::chrono::steady_clock::now();
    llama_hybrid_cycle_phone_ffn_slot local_ffn;
    bool ok = local_ffn.prepare(
        cuda_,
        shard,
        hidden,
        tokens,
        topk,
        experts,
        true,
        true,
        false);
    if (ok) {
        ok =
            local_ffn.execute_full_local(ready);
        ggml_backend_synchronize(cuda_);
    }

    if (ok && local_ffn.output &&
            local_ffn.output->type ==
                GGML_TYPE_F32 &&
            ggml_is_contiguous(
                local_ffn.output) &&
            ggml_nelements(
                local_ffn.output) ==
                hidden * tokens) {
        if (device_output) {
            const auto copy_begin =
                std::chrono::steady_clock::now();
            ok =
                device_output->prepare(
                    cuda_, hidden, tokens) &&
                device_output->store(
                    local_ffn.output);
            if (timing) {
                timing->output_device_copy_ms =
                    std::chrono::duration<
                        double, std::milli>(
                        std::chrono::steady_clock::now() -
                        copy_begin).count();
            }
        } else {
            const auto read_begin =
                std::chrono::steady_clock::now();
            output.resize(elements);
            ggml_backend_tensor_get(
                local_ffn.output,
                output.data(),
                0,
                output.size() * sizeof(float));
            if (timing) {
                timing->output_readback_ms =
                    std::chrono::duration<
                        double, std::milli>(
                        std::chrono::steady_clock::now() -
                        read_begin).count();
            }
        }
    } else {
        ok = false;
    }

    ready.release_phone_graph();
    if (timing) {
        timing->ffn_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() -
                ffn_begin).count();
    }

    if (!ok) {
        output.clear();
        reason =
            "PC_FULL_FFN_OR_OUTPUT_FAILED layer=" +
            std::to_string(layer);
        return false;
    }

    if (!device_output) {
        for (float value : output) {
            if (!std::isfinite(value)) {
                output.clear();
                reason =
                    "PC_FULL_NONFINITE layer=" +
                    std::to_string(layer);
                return false;
            }
        }
    }
    return true;
}

bool llama_region_cuda_mirror::execute_block(
        int layer_begin,
        int layer_end,
        int64_t token_start,
        int64_t tokens,
        const std::vector<float> & input,
        std::vector<float> & output,
        std::string & reason,
        int chunk_tokens) {
    output.clear();
    reason.clear();

    if (!model_ || layer_begin < 0 ||
            layer_end <= layer_begin ||
            layer_end >
                model_->hparams.n_layer() ||
            token_start < 0 ||
            tokens <= 0 ||
            token_start + tokens > max_tokens_ ||
            chunk_tokens <= 0 ||
            chunk_tokens > 128) {
        reason = "PC_BLOCK_INVALID_RANGE";
        return false;
    }

    const int64_t hidden =
        model_->hparams.n_embd;
    if (hidden <= 0 ||
            input.size() !=
                static_cast<size_t>(hidden) *
                static_cast<size_t>(tokens)) {
        reason = "PC_BLOCK_INPUT_SHAPE";
        return false;
    }

    for (int il = layer_begin;
            il < layer_end; ++il) {
        if (model_->hybrid_layer_mode(il) !=
                llama_hybrid_layer_mode::PC_ONLY) {
            reason =
                "PC_BLOCK_NON_PC_LAYER layer=" +
                std::to_string(il);
            return false;
        }
    }

    output.resize(input.size());
    for (int64_t chunk_begin = 0;
            chunk_begin < tokens;
            chunk_begin += chunk_tokens) {
        const int64_t n =
            std::min<int64_t>(
                chunk_tokens,
                tokens - chunk_begin);
        const size_t offset =
            static_cast<size_t>(chunk_begin) *
            static_cast<size_t>(hidden);
        const size_t count =
            static_cast<size_t>(n) *
            static_cast<size_t>(hidden);

        std::vector<float> host_input(
            input.begin() +
                static_cast<std::ptrdiff_t>(offset),
            input.begin() +
                static_cast<std::ptrdiff_t>(
                    offset + count));
        std::vector<float> host_output;

        llama_hybrid_cycle_cuda_activation_slot
            slot_a;
        llama_hybrid_cycle_cuda_activation_slot
            slot_b;
        const ggml_tensor * device_input = nullptr;

        for (int il = layer_begin;
                il < layer_end; ++il) {
            const bool last =
                il + 1 == layer_end;
            llama_hybrid_cycle_cuda_activation_slot *
                device_output = nullptr;
            if (!last) {
                const int ordinal =
                    il - layer_begin;
                device_output =
                    (ordinal & 1) ?
                        &slot_b : &slot_a;
            }

            llama_region_cuda_mirror_timing timing {};
            if (!execute_layer(
                    il,
                    token_start + chunk_begin,
                    n,
                    host_input,
                    host_output,
                    &timing,
                    device_input,
                    device_output,
                    reason)) {
                return false;
            }

            if (llama_hybrid_cycle_verbose()) {
                std::printf(
                    "[REGION_CUDA_FULL] layer=%d "
                    "chunk_start=%lld tokens=%lld "
                    "attn_ms=%.3f ffn_ms=%.3f "
                    "readback_ms=%.3f device_copy_ms=%.3f "
                    "status=OK\n",
                    il,
                    (long long) (
                        token_start + chunk_begin),
                    (long long) n,
                    timing.attention_ms,
                    timing.ffn_ms,
                    timing.output_readback_ms,
                    timing.output_device_copy_ms);
            }

            if (!last) {
                device_input =
                    device_output->tensor;
                host_input.clear();
            }
        }

        if (host_output.size() != count) {
            reason = "PC_BLOCK_OUTPUT_SHAPE";
            return false;
        }
        std::copy(
            host_output.begin(),
            host_output.end(),
            output.begin() +
                static_cast<std::ptrdiff_t>(offset));
    }

    return true;
}

int llama_region_cuda_mirror::resident_capacity() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return resident_capacity_;
}

int llama_region_cuda_mirror::resident_layers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int result = 0;
    for (const auto & resource : layers_) {
        if (resource && resource->weights) {
            ++result;
        }
    }
    return result;
}

uint64_t llama_region_cuda_mirror::residency_loads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loads_;
}

uint64_t llama_region_cuda_mirror::residency_evictions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return evictions_;
}

size_t llama_region_cuda_mirror::planned_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return planned_bytes_;
}
