// Phone-side logical activation binding and two-slot lifetime policy.
// No PC pointer, RPC handle or OpenCL object is stored here.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ggml_phone_stage {

constexpr size_t k_slot_count = 2;
enum class status { ok, invalid, overflow, out_of_memory, busy, stale };
enum class state  { free, computing, output_pending };

struct activation {
    uint32_t id;
    size_t bytes;
    size_t alignment;
    uint32_t first_node;
    uint32_t last_node;
};
struct view {
    uint32_t id;
    uint32_t root_id;
    size_t root_offset;
    size_t bytes;
};
struct allocation {
    uint32_t id;
    size_t offset;
    size_t bytes;
    uint32_t last_node;
};

inline bool checked_add(size_t a, size_t b, size_t & dst) {
    if (b > std::numeric_limits<size_t>::max() - a) { return false; }
    dst = a + b;
    return true;
}
inline bool checked_align(size_t value, size_t align, size_t & dst) {
    if (!align || (align & (align - 1))) { return false; }
    size_t upper;
    if (!checked_add(value, align - 1, upper)) { return false; }
    dst = upper & ~(align - 1);
    return true;
}

// An immutable layout. Only *private, transient* compute roots are included;
// weights, KV, graph inputs, outputs, persistent SET_ROWS and all their aliases
// MUST be excluded by the registering runtime before calling build().
struct binding_layout {
    std::vector<allocation> roots;
    std::vector<view> views;
    size_t required_bytes = 0;

    status build(std::vector<activation> tensors,
                 const std::vector<view> & aliases) {
        roots.clear();
        views.clear();
        required_bytes = 0;
        std::sort(tensors.begin(), tensors.end(),
            [](const activation & a, const activation & b) {
                return a.first_node != b.first_node ?
                    a.first_node < b.first_node : a.id < b.id;
            });
        std::unordered_set<uint32_t> ids;
        for (const activation & t : tensors) {
            if (!t.bytes || t.first_node > t.last_node ||
                !t.alignment || (t.alignment & (t.alignment - 1)) ||
                !ids.insert(t.id).second) {
                return status::invalid;
            }
            size_t offset = 0;
            for (;;) {
                if (!checked_align(offset, t.alignment, offset)) {
                    return status::overflow;
                }
                size_t end;
                if (!checked_add(offset, t.bytes, end)) {
                    return status::overflow;
                }
                size_t next_offset = 0;
                bool overlap = false;
                for (const allocation & other : roots) {
                    if (other.last_node < t.first_node) { continue; }
                    size_t other_end;
                    if (!checked_add(other.offset, other.bytes, other_end)) {
                        return status::overflow;
                    }
                    if (offset < other_end && other.offset < end) {
                        overlap = true;
                        next_offset = std::max(next_offset, other_end);
                    }
                }
                if (!overlap) {
                    roots.push_back({t.id, offset, t.bytes, t.last_node});
                    required_bytes = std::max(required_bytes, end);
                    break;
                }
                if (next_offset <= offset) { return status::invalid; }
                offset = next_offset;
            }
        }
        for (const view & v : aliases) {
            if (!v.bytes || !ids.insert(v.id).second) {
                return status::invalid;
            }
            const auto it = std::find_if(roots.begin(), roots.end(),
                [&](const allocation & a) { return a.id == v.root_id; });
            if (it == roots.end() || v.root_offset > it->bytes ||
                v.bytes > it->bytes - v.root_offset) {
                return status::invalid;
            }
            views.push_back(v);
        }
        return status::ok;
    }

    // All bounds and integer overflow checks precede the one final commit.
    // This must be followed by OpenCL extra + VIEW initialization; updating
    // ggml_tensor::data alone is not enough.
    status resolve(size_t slot_base, size_t slot_size,
                   std::unordered_map<uint32_t, size_t> & bound) const {
        if (required_bytes > slot_size) { return status::out_of_memory; }
        size_t slot_end;
        if (!checked_add(slot_base, slot_size, slot_end)) {
            return status::overflow;
        }
        std::unordered_map<uint32_t, size_t> pending;
        for (const allocation & a : roots) {
            size_t end, address;
            if (!checked_add(a.offset, a.bytes, end) ||
                end > slot_size || !checked_add(slot_base, a.offset, address)) {
                return status::overflow;
            }
            pending.emplace(a.id, address);
        }
        for (const view & v : views) {
            const auto it = pending.find(v.root_id);
            size_t address;
            if (it == pending.end() ||
                !checked_add(it->second, v.root_offset, address)) {
                return status::overflow;
            }
            pending.emplace(v.id, address);
        }
        bound.swap(pending);
        return status::ok;
    }
};

struct template_plan {
    uint32_t layer_begin = 0;
    uint32_t layer_end = 0;
    uint32_t tokens = 0;
    uint32_t output_kind = 0;
    uint64_t topology_signature = 0;
    binding_layout scratch;
};

// Slots are not graphs. A slot is released only after both GPU completion
// and staging of every externally observable output. The epoch invalidates
// old leases when the model, cache or arena allocation changes.
class two_slot_pool {
public:
    status reserve(const std::vector<template_plan> & plans,
                   size_t alignment, size_t headroom = 0) {
        if (plans.empty() || !alignment ||
            (alignment & (alignment - 1))) { return status::invalid; }
        for (const auto & slot : slots_) {
            if (slot.phase != state::free) { return status::busy; }
        }
        size_t maximum = 0;
        for (const auto & p : plans) {
            if (p.layer_begin >= p.layer_end ||
                !p.tokens || !p.topology_signature) {
                return status::invalid;
            }
            maximum = std::max(maximum, p.scratch.required_bytes);
        }
        size_t needed;
        if (!checked_add(maximum, headroom, needed) ||
            !checked_align(needed, alignment, needed)) {
            return status::overflow;
        }
        capacity_ = needed;
        ++epoch_;
        slots_ = {};
        return status::ok;
    }
    size_t bytes_per_slot() const { return capacity_; }
    uint64_t epoch() const { return epoch_; }

    status acquire(size_t index, uint64_t seq, uint64_t epoch) {
        if (index >= k_slot_count || !seq) { return status::invalid; }
        if (epoch != epoch_) { return status::stale; }
        slot & s = slots_[index];
        if (s.phase != state::free) { return status::busy; }
        s.phase = state::computing;
        s.seq = seq;
        return status::ok;
    }
    status compute_complete(size_t index, uint64_t seq) {
        if (index >= k_slot_count || !seq) { return status::invalid; }
        slot & s = slots_[index];
        if (s.phase != state::computing || s.seq != seq) {
            return status::stale;
        }
        s.phase = state::output_pending;
        return status::ok;
    }
    status output_published(size_t index, uint64_t seq) {
        if (index >= k_slot_count || !seq) { return status::invalid; }
        slot & s = slots_[index];
        if (s.phase != state::output_pending || s.seq != seq) {
            return status::stale;
        }
        s = {};
        return status::ok;
    }
    bool busy(size_t index) const {
        return index >= k_slot_count || slots_[index].phase != state::free;
    }
private:
    struct slot {
        state phase = state::free;
        uint64_t seq = 0;
    };
    std::array<slot, k_slot_count> slots_ {};
    size_t capacity_ = 0;
    uint64_t epoch_ = 0;
};
} // namespace ggml_phone_stage
