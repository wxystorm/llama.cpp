#pragma once

// PC GPU rolling two-Bank semantics, adapted as a fail-closed Phone HTP
// *weight* bank policy. No HTP graph runs concurrently with another Graph.
// This file deliberately owns no backend allocation: the RPC/executor layer
// must provide stable physical A/B weight buffers, graph retirement and load
// completion before publishing READY. KV is never evicted with a weight bank.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace ggml_phone_npu_serial {

struct weight_bank_region {
    uint32_t region = 0;       // canonical Phone Region index
    int layer_begin = 0;
    int layer_end = 0;
    size_t weights_bytes = 0;  // HTP-repacked size, not GGUF raw size
};

struct weight_bank_refill {
    size_t bank = 0;
    size_t from_rank = 0;
    size_t to_rank = 0;
    uint32_t to_region = 0;
    size_t required_bytes = 0;
};

class two_weight_banks {
public:
    enum class phase : uint8_t { EMPTY, PRIMING, READY, COMPUTING,
                                 DRAINED, REFILLING, FAILED };

    struct bank_state {
        phase state = phase::EMPTY;
        size_t rank = 0;
        size_t capacity = 0;
        uint64_t generation = 0;
        bool compiled_graphs_retired = false;
        uint32_t next_chunk = 0;
    };

    static size_t bank_for_rank(size_t rank) { return rank % 2; }

    bool configure(const std::vector<weight_bank_region> & ordered,
                   size_t kv_resident_bytes, size_t scratch_bytes,
                   size_t npu_budget_bytes, std::string & reason) {
        reason.clear();
        reset();
        if (ordered.size() < 2 || npu_budget_bytes == 0) {
            reason = "HTP_TWO_BANK_REQUIRES_TWO_REGIONS_AND_BUDGET";
            return false;
        }
        for (size_t rank = 0; rank < ordered.size(); ++rank) {
            const auto & r = ordered[rank];
            if (r.layer_end <= r.layer_begin || r.weights_bytes == 0 ||
                    (rank && ordered[rank - 1].region >= r.region)) {
                reason = "HTP_TWO_BANK_REGION_ORDER_OR_SIZE";
                return false;
            }
            caps_[bank_for_rank(rank)] = std::max(
                caps_[bank_for_rank(rank)], r.weights_bytes);
        }
        // Exact same parity/max-window accounting as PC GPU: two physical
        // banks, not a full-model weight cache. HTP KV and scratch are charged
        // once and remain resident for the complete RUN.
        size_t required = kv_resident_bytes;
        for (size_t x : {scratch_bytes, caps_[0], caps_[1]}) {
            if (required > std::numeric_limits<size_t>::max() - x) {
                reason = "HTP_TWO_BANK_MEMORY_OVERFLOW";
                return false;
            }
            required += x;
        }
        if (required > npu_budget_bytes) {
            reason = "HTP_TWO_BANK_MEMORY_EXCEEDED";
            return false;
        }
        regions_ = ordered;
        kv_bytes_ = kv_resident_bytes;
        scratch_bytes_ = scratch_bytes;
        required_bytes_ = required;
        for (size_t bank = 0; bank < 2; ++bank) {
            states_[bank].state = phase::PRIMING;
            states_[bank].rank = bank;
            states_[bank].capacity = caps_[bank];
        }
        return true;
    }

    // The first two Regions must be uploaded before prefill dispatch.
    bool prime_complete(size_t rank) {
        if (rank >= 2 || rank >= regions_.size()) { return false; }
        auto & b = states_[rank];
        if (b.rank != rank || b.state != phase::PRIMING) { return false; }
        b.state = phase::READY;
        ++b.generation;
        return true;
    }

    bool begin_chunk(size_t rank, uint32_t chunk,
                     uint64_t graph_bank_generation) {
        if (rank >= regions_.size() || active_rank_.has_value()) {
            return false; // one HTP compute lane, never dual Graph compute
        }
        auto & b = states_[bank_for_rank(rank)];
        // A precompiled Graph bound to an older Bank generation must never
        // be executed after the physical storage has been recycled.
        if (b.state != phase::READY || b.rank != rank ||
                chunk != b.next_chunk ||
                graph_bank_generation != b.generation) { return false; }
        b.state = phase::COMPUTING;
        active_rank_ = rank;
        return true;
    }

    // Report completion *after* output/P2C capture and KV visibility.
    // A refill request is available only after the Region's LAST Chunk.
    std::optional<weight_bank_refill> finish_chunk(
            size_t rank, uint32_t chunk, bool last_chunk) {
        if (rank >= regions_.size()) { return std::nullopt; }
        auto & b = states_[bank_for_rank(rank)];
        if (b.state != phase::COMPUTING || b.rank != rank ||
                b.next_chunk != chunk || active_rank_ != rank) {
            return std::nullopt;
        }
        active_rank_.reset();
        ++b.next_chunk;
        b.state = last_chunk ? phase::DRAINED : phase::READY;
        if (!last_chunk || rank + 2 >= regions_.size()) {
            return std::nullopt;
        }
        const size_t target = rank + 2;
        return weight_bank_refill{
            bank_for_rank(rank), rank, target, regions_[target].region,
            regions_[target].weights_bytes
        };
    }

    // Fixed-graph cache may retain old weight *addresses*. The caller must
    // retire all Graphs referring to the drained rank before reusing A or B.
    bool graphs_retired(size_t rank) {
        if (rank >= regions_.size()) { return false; }
        auto & b = states_[bank_for_rank(rank)];
        if (b.rank != rank || b.state != phase::DRAINED) { return false; }
        b.compiled_graphs_retired = true;
        return true;
    }

    bool begin_refill(const weight_bank_refill & job) {
        if (job.bank >= 2 || job.to_rank != job.from_rank + 2 ||
                job.to_rank >= regions_.size() ||
                job.bank != bank_for_rank(job.from_rank) ||
                job.bank != bank_for_rank(job.to_rank) ||
                regions_[job.to_rank].region != job.to_region ||
                regions_[job.to_rank].weights_bytes != job.required_bytes) {
            return false;
        }
        auto & b = states_[job.bank];
        if (b.rank != job.from_rank || b.state != phase::DRAINED ||
                !b.compiled_graphs_retired ||
                job.required_bytes > b.capacity) { return false; }
        b.rank = job.to_rank;
        b.state = phase::REFILLING;
        b.compiled_graphs_retired = false;
        b.next_chunk = 0;
        return true;
    }

    bool refill_complete(size_t target_rank, bool success) {
        if (target_rank >= regions_.size()) { return false; }
        auto & b = states_[bank_for_rank(target_rank)];
        if (b.rank != target_rank || b.state != phase::REFILLING) {
            return false;
        }
        b.state = success ? phase::READY : phase::FAILED;
        if (success) { ++b.generation; }
        return success;
    }

    void reset() {
        regions_.clear();
        states_ = {};
        caps_ = {};
        kv_bytes_ = 0;
        scratch_bytes_ = 0;
        required_bytes_ = 0;
        active_rank_.reset();
    }
    size_t total_required_bytes() const { return required_bytes_; }
    size_t kv_resident_bytes() const { return kv_bytes_; }
    size_t scratch_bytes() const { return scratch_bytes_; }
    size_t bank_capacity(size_t bank) const {
        return bank < 2 ? caps_[bank] : 0;
    }
    const bank_state & bank(size_t index) const {
        return states_[index]; // caller checks index < 2
    }
    size_t regions() const { return regions_.size(); }

private:
    std::vector<weight_bank_region> regions_;
    std::array<bank_state, 2> states_ {};
    std::array<size_t, 2> caps_ {};
    size_t kv_bytes_ = 0;
    size_t scratch_bytes_ = 0;
    size_t required_bytes_ = 0;
    std::optional<size_t> active_rank_;
};

} // namespace ggml_phone_npu_serial
