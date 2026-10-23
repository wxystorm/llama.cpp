#pragma once

#include "llama-hybrid-cycle-runtime.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

class llama_hybrid_cycle_physical_resource_pool;

// This is the single-worker PHONE_GPU command/ACK ledger for ONE physical
// region. No counter is shared between L4..7 and L12..15 (or later regions).
// The ready bitmap can retain early C2P completions until that region's
// Phone MAIN worker is allowed to arm the next chunk. The normal Ring owns
// interlayer dependencies and credits; these are RPC registration cursors.
struct llama_hybrid_cycle_phone_region_progress {
    llama_hybrid_cycle_runtime_region location;
    int n_chunks = 0;
    int max_inflight = 0;

    uint64_t ordered_full_handoffs = 0;
    uint64_t ordered_p2c_handoffs = 0;
    uint64_t next_attention_prepared = 0;
    uint64_t next_attention_completed = 0;
    uint64_t full_pairs_armed = 0;
    uint64_t next_full_completed = 0;

    std::vector<uint8_t> c2p_ready_chunks;
    int next_arm_chunk = 0;
    int64_t first_phone_registered_tokens = 0;
    uint64_t region_armed = 0;
    uint64_t first_phone_completed = 0;
    uint64_t last_phone_completed = 0;
    int64_t p2c_registered_tokens = 0;
    uint64_t p2c_ready_completed = 0;
    uint64_t p2c_ffn_completed = 0;
    int64_t next_attention_registered_tokens = 0;
};

// A real Phone region is a contiguous:
//    PC Full* -> C2P -> Phone Full* -> P2C
// All identifiers are ABSOLUTE model layers. A 48-layer X=4/Y=2 model has
// six independent ledgers but shares the same FIVE backend worker queues.
//
// prepare() merely validates the physical topology and allocates host-side
// control ledgers; it neither opens RPC connections nor builds a GGML graph.
// initialize before worker start, keep this object fixed in memory while
// StageDispatcher has callbacks referring to it, destroy after worker join.
class llama_hybrid_cycle_phone_regions {
public:
    using real_stage_handler = std::function<bool(
        llama_hybrid_cycle_phone_region_progress &,
        const llama_hybrid_cycle_task &)>;

    LLAMA_API bool prepare(
        const llama_hybrid_cycle_layout & physical_layout,
        int64_t tokens, int chunk_tokens, int max_inflight,
        std::string & reason);

    LLAMA_API void clear();

    LLAMA_API int region_count() const;
    LLAMA_API int n_chunks() const;
    LLAMA_API const llama_hybrid_cycle_layout & layout() const;

    LLAMA_API llama_hybrid_cycle_phone_region_progress * by_index(int index);
    LLAMA_API const llama_hybrid_cycle_phone_region_progress * by_index(int index) const;
    LLAMA_API llama_hybrid_cycle_phone_region_progress * for_layer(int physical_layer);
    LLAMA_API const llama_hybrid_cycle_phone_region_progress * for_layer(int physical_layer) const;

    // Validate the REAL 48-layer KV/Graph/Scratch owners once, before
    // registering any remote Graph. Ownership itself stays in ResourcePool.
    LLAMA_API bool validate_resources(
        const llama_hybrid_cycle_physical_resource_pool & resources,
        std::string & reason) const;

    // Bind only C2P, Phone Full and P2C tasks. PC Full handlers are bound
    // independently (e.g. by llama_hybrid_cycle_bind_pc_full_stages).
    // Callbacks receive their own cycle ledger; never infer L5/L6/L7 from
    // a local hardcoded index. The handler must ACK its real dependencies
    // before returning true; this wrapper does NOT invent an RPC success.
    LLAMA_API bool bind_phone_stages(
        llama_hybrid_cycle_stage_dispatcher & stages,
        real_stage_handler execute, std::string & reason);

private:
    llama_hybrid_cycle_layout layout_;
    int n_chunks_ = 0;
    int64_t tokens_ = 0;
    std::vector<llama_hybrid_cycle_phone_region_progress> regions_;
};

// Banks own idle physical Phone FFN/Reduce/P2C slots for ONE region each.
// Actual Slot constructors (RPC hooks) are supplied by the caller's types.
// Both the original V5 test and a full 48-layer runner can instantiate these
// with their own RPC-bound slot types. Banks own no KV/Attention Graph/Scratch;
// those remain with ResourcePool (or the established V5 persistent session).
//
// All slots MUST be returned after downstream consumption/fence before
// destroying the bank. No bank is shared between different physical cycles.
template<class PhoneSlot, class P2cSlot>
class llama_hybrid_cycle_phone_region_slot_banks {
public:
    using phone_ptr = std::unique_ptr<PhoneSlot>;
    using p2c_ptr = std::unique_ptr<P2cSlot>;

    struct bank {
        std::mutex c2p_mutex;
        std::vector<phone_ptr> c2p_idle;
        size_t c2p_live_slots = 0;
        // Indexed by ABSOLUTE physical model layer, not local L5/L6.
        std::vector<std::vector<phone_ptr>> full_idle_by_layer;
        std::vector<p2c_ptr> p2c_idle;
        int c2p_layer = -1;
        int p2c_layer = -1;
    };

    void prepare(const llama_hybrid_cycle_phone_regions & regions) {
        banks_.clear();
        const auto count = static_cast<size_t>(regions.region_count());
        banks_.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto * source = regions.by_index(int(i));
            if (!source) throw std::invalid_argument("phone bank region missing");
            auto entry = std::make_unique<bank>();
            entry->c2p_layer = source->location.c2p_layer;
            entry->p2c_layer = source->location.p2c_layer;
            entry->full_idle_by_layer.resize(regions.layout().kinds.size());
            banks_.push_back(std::move(entry));
        }
    }

    bank * by_index(int index) {
        if (index < 0 || static_cast<size_t>(index) >= banks_.size()) return nullptr;
        return banks_[static_cast<size_t>(index)].get();
    }

    size_t size() const { return banks_.size(); }

private:
    std::vector<std::unique_ptr<bank>> banks_;
};
