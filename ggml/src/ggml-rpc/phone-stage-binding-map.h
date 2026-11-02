#pragma once

#include "phone-stage-binding-pool.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ggml_phone_stage {

struct stage_binding_entry {
    uint32_t id = 0;
    size_t source_offset = 0;
    size_t bytes = 0;
    uint32_t view_source_id = 0;
    size_t view_offset = 0;
    bool protected_storage = false;
};

struct stage_slot_address {
    size_t base = 0;
    size_t capacity = 0;
};

// Diagnostic-only address projection. It preserves the original gallocr
// offsets, including aliases to stage inputs and observable outputs.
class stage_binding_map {
public:
    explicit stage_binding_map(size_t source_capacity)
        : source_capacity_(source_capacity) {}

    status add(const stage_binding_entry & entry) {
        if (!entry.id || !entry.bytes ||
            entry.source_offset > source_capacity_ ||
            entry.bytes > source_capacity_ - entry.source_offset ||
            entry.view_source_id == entry.id) {
            return status::invalid;
        }
        for (const auto & existing : entries_) {
            if (existing.id == entry.id) {
                return status::invalid;
            }
        }
        entries_.push_back(entry);
        required_bytes_ = std::max(required_bytes_,
                                   entry.source_offset + entry.bytes);
        view_count_ += entry.view_source_id != 0;
        protected_count_ += entry.protected_storage;
        return status::ok;
    }

    status validate(uint32_t * failed_id = nullptr) const {
        if (failed_id != nullptr) {
            *failed_id = 0;
        }
        const auto reject = [&](uint32_t id) {
            if (failed_id != nullptr) {
                *failed_id = id;
            }
            return status::invalid;
        };
        if (entries_.empty()) {
            return reject(0);
        }
        std::unordered_map<uint32_t, const stage_binding_entry *> by_id;
        by_id.reserve(entries_.size());
        for (const auto & entry : entries_) {
            if (!by_id.emplace(entry.id, &entry).second) {
                return reject(entry.id);
            }
        }
        for (const auto & entry : entries_) {
            if (!entry.view_source_id) {
                if (entry.view_offset) {
                    return reject(entry.id);
                }
                continue;
            }
            const auto it = by_id.find(entry.view_source_id);
            if (it == by_id.end()) {
                return reject(entry.id);
            }
            const auto & parent = *it->second;
            if (entry.view_offset > parent.bytes ||
                entry.bytes > parent.bytes - entry.view_offset) {
                return reject(entry.id);
            }
            size_t expected_offset;
            if (!checked_add(parent.source_offset, entry.view_offset,
                             expected_offset) ||
                expected_offset != entry.source_offset) {
                return reject(entry.id);
            }
            // Reject malformed cycles even if their offsets happen to match.
            uint32_t next_id = entry.view_source_id;
            for (size_t depth = 0; next_id; ++depth) {
                if (depth >= entries_.size() || next_id == entry.id) {
                    return reject(entry.id);
                }
                const auto next = by_id.find(next_id);
                if (next == by_id.end()) {
                    return reject(entry.id);
                }
                next_id = next->second->view_source_id;
            }
        }
        return status::ok;
    }

    status resolve_slots(
            const std::array<stage_slot_address, k_slot_count> & slots,
            std::array<std::unordered_map<uint32_t, size_t>, k_slot_count> & bound) const {
        if (validate() != status::ok) {
            return status::invalid;
        }
        std::array<size_t, k_slot_count> ends {};
        for (size_t slot = 0; slot < k_slot_count; ++slot) {
            const auto & address = slots[slot];
            if (!address.base || !address.capacity) {
                return status::invalid;
            }
            if (required_bytes_ > address.capacity) {
                return status::out_of_memory;
            }
            if (!checked_add(address.base, address.capacity, ends[slot])) {
                return status::overflow;
            }
        }
        if (may_overlap(slots[0].base, slots[0].capacity,
                        slots[1].base, slots[1].capacity)) {
            return status::invalid;
        }

        std::array<std::unordered_map<uint32_t, size_t>, k_slot_count> pending;
        for (size_t slot = 0; slot < k_slot_count; ++slot) {
            pending[slot].reserve(entries_.size());
            for (const auto & entry : entries_) {
                size_t candidate;
                size_t end;
                if (!checked_add(slots[slot].base, entry.source_offset,
                                 candidate) ||
                    !checked_add(candidate, entry.bytes, end) ||
                    end > ends[slot]) {
                    return status::overflow;
                }
                pending[slot].emplace(entry.id, candidate);
            }
        }
        bound.swap(pending);
        return status::ok;
    }

    uint64_t layout_signature() const {
        uint64_t hash = UINT64_C(14695981039346656037);
        const auto mix = [&](uint64_t value) {
            hash ^= value;
            hash *= UINT64_C(1099511628211);
        };
        mix(entries_.size());
        for (const auto & entry : entries_) {
            mix(entry.id);
            mix(entry.source_offset);
            mix(entry.bytes);
            mix(entry.view_source_id);
            mix(entry.view_offset);
            mix(entry.protected_storage ? 1 : 0);
        }
        return hash ? hash : 1;
    }

    const std::vector<stage_binding_entry> & entries() const { return entries_; }
    size_t required_bytes() const { return required_bytes_; }
    size_t view_count() const { return view_count_; }
    size_t protected_count() const { return protected_count_; }

private:
    size_t source_capacity_ = 0;
    size_t required_bytes_ = 0;
    size_t view_count_ = 0;
    size_t protected_count_ = 0;
    std::vector<stage_binding_entry> entries_;
};

} // namespace ggml_phone_stage
