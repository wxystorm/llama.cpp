#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

// Cache ONLY a physical backend scratch buffer, NOT ggml_context,
// ggml_cgraph, positions, causal mask, READY activations or KV. Cached
// Attention graph metadata is independently rebound to each idle lease.
//
// One pool per physical layer, accessed by its single FIFO worker. A
// lease is recycled only after the FFN consumer has completed and fenced.
// By default the pool is per-Cycle; opt-in Phone Session ownership retains
// idle allocations across Warmup/Measure without pinning them to Graphs.
// Every lease must be returned before the pool or backend is destroyed.
struct cycle_real_attention_scratch_pool {
    struct slot {
        ggml_backend_buffer_t buffer = nullptr;
        size_t bytes = 0;
    };

    ggml_backend_buffer_type_t buft = nullptr;
    size_t max_slots = 0;
    std::vector<slot> idle;
    std::map<ggml_backend_buffer_t, size_t> active;
    uint64_t allocations = 0;
    uint64_t hits = 0;
    uint64_t evictions = 0;
    uint64_t fallbacks = 0;
    uint64_t recycled = 0;
    uint64_t bind_failures = 0;
    size_t peak_slots = 0;
    double size_query_ms = 0.0;
    double buffer_alloc_ms = 0.0;
    double graph_bind_ms = 0.0;
    double buffer_free_ms = 0.0;

    cycle_real_attention_scratch_pool() = default;
    cycle_real_attention_scratch_pool(const cycle_real_attention_scratch_pool &) = delete;
    cycle_real_attention_scratch_pool & operator=(
        const cycle_real_attention_scratch_pool &) = delete;
    ~cycle_real_attention_scratch_pool() { clear(); }

    void configure(ggml_backend_t backend, int inflight) {
        clear();
        buft = ggml_backend_get_default_buffer_type(backend);
        max_slots = size_t(inflight);
    }

    size_t slots() const { return idle.size() + active.size(); }

    // Equivalent to ggml-alloc.c:alloc_tensor_range(), but reuse the
    // returned RPC buffer instead of issuing a new ALLOC_BUFFER.
    static bool bind_graph(ggml_context * ctx, ggml_backend_buffer_t buffer) {
        ggml_tallocr tallocr = ggml_tallocr_new(buffer);
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
             t = ggml_get_next_tensor(ctx, t)) {
            ggml_status status = GGML_STATUS_SUCCESS;
            if (t->data == nullptr) {
                if (t->view_src == nullptr) {
                    status = ggml_tallocr_alloc(&tallocr, t);
                } else if (t->buffer == nullptr) {
                    status = ggml_backend_view_init(t);
                }
            } else if (t->view_src != nullptr && t->buffer == nullptr) {
                status = ggml_backend_view_init(t);
            }
            if (status != GGML_STATUS_SUCCESS) return false;
        }
        return true;
    }

    // pooled=false means the old allocator owns the resulting buffer.
    // This allows a safe fallback if two slots are simultaneously active
    // or a graph exceeds the RPC backend's single-buffer limit.
    bool acquire(ggml_context * ctx, ggml_backend_buffer_t & out, bool & pooled) {
        out = nullptr;
        pooled = false;
        if (!ctx || !buft || max_slots == 0) return false;

        const auto query_begin = std::chrono::steady_clock::now();
        const size_t needed =
            ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
        size_query_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - query_begin).count();
        if (needed == 0) return false;
        const size_t max_alloc = ggml_backend_buft_get_max_size(buft);
        if (needed > max_alloc ||
            (idle.empty() && slots() >= max_slots)) {
            ++fallbacks;
            out = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            return out != nullptr;
        }

        // Best-fit. KV prefix (16/32/48/64) can increase scratch usage;
        // replace too-small *idle* slots but never evict a live lease.
        size_t best = idle.size();
        for (size_t i = 0; i < idle.size(); ++i) {
            if (idle[i].bytes >= needed &&
                (best == idle.size() ||
                 idle[i].bytes < idle[best].bytes)) {
                best = i;
            }
        }

        ggml_backend_buffer_t buffer = nullptr;
        if (best != idle.size()) {
            buffer = idle[best].buffer;
            idle.erase(idle.begin() + best);
            ++hits;
        } else {
            if (!idle.empty()) {
                const auto begin = std::chrono::steady_clock::now();
                ggml_backend_buffer_free(idle.back().buffer);
                buffer_free_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - begin).count();
                idle.pop_back();
                ++evictions;
            }

            // Small headroom avoids repeatedly reallocating as KV grows,
            // without exceeding the backend's maximum allocation size.
            const size_t extra = std::max<size_t>(size_t(1) << 20, needed / 8);
            const size_t reserve =
                needed + std::min(extra, max_alloc - needed);
            const auto begin = std::chrono::steady_clock::now();
            buffer = ggml_backend_buft_alloc_buffer(buft, reserve);
            buffer_alloc_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            if (!buffer) return false;
            ++allocations;
        }

        const auto bind_begin = std::chrono::steady_clock::now();
        const bool ok = bind_graph(ctx, buffer);
        graph_bind_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - bind_begin).count();
        if (!ok) {
            ++bind_failures;
            ggml_backend_buffer_free(buffer);
            return false;
        }

        active.emplace(buffer, ggml_backend_buffer_get_size(buffer));
        peak_slots = std::max(peak_slots, slots());
        out = buffer;
        pooled = true;
        return true;
    }

    void recycle(ggml_backend_buffer_t buffer) {
        if (!buffer) return;
        const auto it = active.find(buffer);
        if (it == active.end()) {
            // Never hand an untracked or potentially active allocation to
            // a new Chunk. Fail closed rather than reuse an unknown buffer.
            std::fprintf(stderr,
                "[CYCLE_ATTN_SCRATCH] status=FAIL reason=UNTRACKED_BUFFER\n");
            ggml_backend_buffer_free(buffer);
            return;
        }
        const size_t bytes = it->second;
        active.erase(it);
        idle.push_back({buffer, bytes});
        ++recycled;
    }

    void clear() {
        for (const auto & entry : idle) {
            const auto begin = std::chrono::steady_clock::now();
            ggml_backend_buffer_free(entry.buffer);
            buffer_free_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
        }
        idle.clear();
        if (!active.empty()) {
            // An active lease belongs to a cycle_real_phone_ready. The caller
            // must destroy/release READY before this pool/backend disappears.
            std::fprintf(stderr,
                "[CYCLE_ATTN_SCRATCH] status=FAIL "
                "reason=ACTIVE_AT_POOL_TEARDOWN count=%zu\n",
                active.size());
        }
        buft = nullptr;
    }
};
