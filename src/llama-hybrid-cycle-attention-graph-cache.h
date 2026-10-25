#pragma once

#include "llama-hybrid-cycle-graph.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

// This cache owns ONLY host-side GGML Attention graph metadata. It does not
// own any GPU scratch memory, computed hidden states or KV bytes. The physical
// KV tensor object and backend/model weights are borrowed from the same model
// session for the lifetime of each graph.
//
// The sole PHONE_GPU FIFO worker checks out the graph and releases it only
// after the FFN consumer has completed. Graphs can be rebound to a separately
// leased scratch buffer, even when successive KV prefixes require different
// graph shapes. The RPC graph still uses its existing non-cacheable compute
// path: this is NOT a server-side Graph Replay optimization.
struct cycle_real_attention_graph_cache {
    struct slot {
        ggml_context * ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * positions = nullptr;
        ggml_tensor * causal_mask = nullptr;
        ggml_tensor * router_probs = nullptr;
        llama_hybrid_cycle_ready_graph ready {};
        const void * kv_identity = nullptr;
        bool busy = false;

        slot() = default;
        slot(const slot &) = delete;
        slot & operator=(const slot &) = delete;
        ~slot() {
            if (ctx) ggml_free(ctx);
        }

        // Graph structure and source links stay intact. Clear ONLY runtime
        // allocations on tensors created inside this GGML context, not the
        // external persistent KV or physical weight tensors. Views will be
        // rebound by ggml_backend_view_init() when a scratch lease is acquired.
        void unbind_compute_tensors() {
            if (!ctx) return;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
                 t = ggml_get_next_tensor(ctx, t)) {
                t->buffer = nullptr;
                t->data = nullptr;
            }
        }
    };

    using key = std::pair<int64_t, int64_t>; // query_tokens, kv_tokens
    std::map<key, std::unique_ptr<slot>> entries;
    size_t max_graphs = 16;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t prepares = 0;
    uint64_t busy_misses = 0;
    uint64_t capacity_misses = 0;
    uint64_t returned = 0;
    uint64_t aborted = 0;

    cycle_real_attention_graph_cache() = default;
    cycle_real_attention_graph_cache(const cycle_real_attention_graph_cache &) = delete;
    cycle_real_attention_graph_cache & operator=(
        const cycle_real_attention_graph_cache &) = delete;
    ~cycle_real_attention_graph_cache() { clear(); }

    slot * checkout(int64_t query_tokens, int64_t kv_tokens,
                    const void * kv_identity) {
        const auto it = entries.find({query_tokens, kv_tokens});
        if (it == entries.end()) {
            ++misses;
            return nullptr;
        }
        slot * result = it->second.get();
        if (result->kv_identity != kv_identity || result->busy) {
            ++busy_misses;
            return nullptr;
        }
        result->busy = true;
        ++hits;
        return result;
    }

    // Only insert a SUCCESSFULLY executed graph whose READY outputs remain
    // leased to the current FFN consumer. The caller transfers ctx ownership.
    slot * retain(int64_t query_tokens, int64_t kv_tokens,
                  const void * kv_identity, ggml_context * ctx,
                  ggml_cgraph * graph, ggml_tensor * input,
                  ggml_tensor * positions, ggml_tensor * causal_mask,
                  ggml_tensor * router_probs,
                  const llama_hybrid_cycle_ready_graph & ready) {
        const key k{query_tokens, kv_tokens};
        if (!ctx || !graph || !input || !positions || !causal_mask ||
            !router_probs || !kv_identity || entries.count(k) ||
            entries.size() >= max_graphs) {
            if (entries.size() >= max_graphs) ++capacity_misses;
            return nullptr;
        }
        auto owned = std::make_unique<slot>();
        owned->ctx = ctx;
        owned->graph = graph;
        owned->input = input;
        owned->positions = positions;
        owned->causal_mask = causal_mask;
        owned->router_probs = router_probs;
        owned->ready = ready;
        owned->kv_identity = kv_identity;
        owned->busy = true;
        slot * result = owned.get();
        entries.emplace(k, std::move(owned));
        ++prepares;
        return result;
    }

    void recycle(slot * s, bool success) {
        if (!s) return;
        s->unbind_compute_tensors();
        s->busy = false;
        if (success) ++returned; else ++aborted;
    }

    bool is_idle() const {
        for (const auto & kv : entries) {
            if (kv.second->busy) return false;
        }
        return true;
    }

    void clear() {
        entries.clear();
    }
};
