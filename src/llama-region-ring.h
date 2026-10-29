#pragma once

// Experimental CUDA -> CPU -> Phone region ring.  This header is intentionally
// independent of GGML so its task dependencies can be tested in isolation.
#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

enum class llama_region_ring_device : uint8_t {
    CPU = 0,
    GPU = 1,
    PHONE = 2,
};

// Logical Region stage kind. C2P/P2C are boundary layers, not additional
// physical executors: both run through the existing Meta/Phone lane while
// reusing the directional Tensor implementation already present in Meta.
enum class llama_region_ring_stage_kind : uint8_t {
    CPU = 0,
    GPU = 1,
    PHONE = 2,
    C2P = 3,
    P2C = 4,
};

inline constexpr std::array<llama_region_ring_device, 3>
LLAMA_REGION_RING_CYCLE_ORDER = {
    llama_region_ring_device::GPU,
    llama_region_ring_device::CPU,
    llama_region_ring_device::PHONE,
};

// Region Ring production runtime uses one full-ubatch activation buffer per
// boundary and therefore has no producer-credit/slot-count backpressure.
// A value of 0 selects that unbounded mode. Explicit non-zero capacities are
// retained only for deterministic bounded-slot scheduler regression tests.
inline constexpr size_t LLAMA_REGION_RING_SLOT_POOL_SIZE = 0;

inline llama_region_ring_device llama_region_ring_stage_device(
        llama_region_ring_stage_kind kind) {
    switch (kind) {
        case llama_region_ring_stage_kind::CPU:
            return llama_region_ring_device::CPU;
        case llama_region_ring_stage_kind::GPU:
            return llama_region_ring_device::GPU;
        case llama_region_ring_stage_kind::PHONE:
        case llama_region_ring_stage_kind::C2P:
        case llama_region_ring_stage_kind::P2C:
            return llama_region_ring_device::PHONE;
    }
    throw std::runtime_error("invalid region ring stage kind");
}

inline llama_region_ring_stage_kind llama_region_ring_device_stage(
        llama_region_ring_device device) {
    switch (device) {
        case llama_region_ring_device::CPU:
            return llama_region_ring_stage_kind::CPU;
        case llama_region_ring_device::GPU:
            return llama_region_ring_stage_kind::GPU;
        case llama_region_ring_device::PHONE:
            return llama_region_ring_stage_kind::PHONE;
    }
    throw std::runtime_error("invalid region ring device");
}

inline const char * llama_region_ring_stage_name(
        llama_region_ring_stage_kind kind) {
    switch (kind) {
        case llama_region_ring_stage_kind::CPU:   return "CPU";
        case llama_region_ring_stage_kind::GPU:   return "GPU";
        case llama_region_ring_stage_kind::PHONE: return "PHONE";
        case llama_region_ring_stage_kind::C2P:   return "C2P";
        case llama_region_ring_stage_kind::P2C:   return "P2C";
    }
    return "UNKNOWN";
}

inline bool llama_region_ring_stage_is_directional(
        llama_region_ring_stage_kind kind) {
    return kind == llama_region_ring_stage_kind::C2P ||
           kind == llama_region_ring_stage_kind::P2C;
}

struct llama_region_ring_region {
    llama_region_ring_device device;
    int layer_begin;
    int layer_end; // exclusive
    llama_region_ring_stage_kind kind;

    llama_region_ring_region(
            llama_region_ring_device device_,
            int layer_begin_,
            int layer_end_) :
        device(device_),
        layer_begin(layer_begin_),
        layer_end(layer_end_),
        kind(llama_region_ring_device_stage(device_)) {
    }

    llama_region_ring_region(
            llama_region_ring_stage_kind kind_,
            int layer_begin_,
            int layer_end_) :
        device(llama_region_ring_stage_device(kind_)),
        layer_begin(layer_begin_),
        layer_end(layer_end_),
        kind(kind_) {
    }
};

// GPU weight streaming uses two fixed physical Banks. Keep Bank
// parity and rank+2 lookahead in one shared policy so runtime, planner and
// regression tests cannot silently drift.
inline size_t llama_region_ring_gpu_weight_bank_for_rank(size_t rank) {
    return rank & 1u;
}

inline std::optional<size_t> llama_region_ring_gpu_weight_next_rank(
        size_t rank, size_t gpu_region_count) {
    if (rank > std::numeric_limits<size_t>::max() - 2) {
        return std::nullopt;
    }
    const size_t next = rank + 2;
    return next < gpu_region_count ?
        std::optional<size_t>(next) : std::nullopt;
}

// Exact intersection of disjoint PC execution intervals with disjoint
// Phone-compute intervals. Both vectors are written by their owner threads
// and inspected only AFTER all worker threads have joined.
inline int64_t llama_region_ring_interval_overlap_us(
        const std::vector<std::pair<int64_t, int64_t>> & pc,
        const std::vector<std::pair<int64_t, int64_t>> & phone) {
    int64_t overlap = 0;
    for (const auto & a : pc) {
        for (const auto & b : phone) {
            const int64_t lo = std::max(a.first, b.first);
            const int64_t hi = std::min(a.second, b.second);
            if (hi > lo) {
                overlap += hi - lo;
            }
        }
    }
    return overlap;
}

inline const char * llama_region_ring_device_name(llama_region_ring_device device) {
    switch (device) {
        case llama_region_ring_device::CPU:   return "CPU";
        case llama_region_ring_device::GPU:   return "GPU";
        case llama_region_ring_device::PHONE: return "PHONE";
    }
    return "UNKNOWN";
}

inline bool llama_region_ring_enabled() {
    const char * value = std::getenv("LLAMA_HYBRID_REGION_RING_V1");
    return value != nullptr && std::atoi(value) != 0;
}

inline bool llama_region_ring_directional_topology_compatible(
        const std::vector<llama_region_ring_region> & regions,
        int n_layer) {
    if (regions.size() < 6 || n_layer <= 0 ||
            regions.front().layer_begin != 0 ||
            regions.front().kind != llama_region_ring_stage_kind::GPU ||
            regions.back().layer_end != n_layer ||
            regions.back().kind != llama_region_ring_stage_kind::PHONE) {
        return false;
    }

    int next_layer = 0;
    int phone_regions = 0;
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto & r = regions[i];
        if (r.layer_begin != next_layer ||
                r.layer_end <= r.layer_begin ||
                r.layer_end > n_layer) {
            return false;
        }
        if (llama_region_ring_stage_is_directional(r.kind) &&
                r.layer_end - r.layer_begin != 1) {
            return false;
        }
        if (r.kind == llama_region_ring_stage_kind::PHONE) {
            ++phone_regions;
        }

        if (i + 1 < regions.size()) {
            const auto next = regions[i + 1].kind;
            bool ok = false;
            switch (r.kind) {
                case llama_region_ring_stage_kind::GPU:
                    ok = next == llama_region_ring_stage_kind::CPU;
                    break;
                case llama_region_ring_stage_kind::CPU:
                    ok = next == llama_region_ring_stage_kind::PHONE ||
                         next == llama_region_ring_stage_kind::C2P;
                    break;
                case llama_region_ring_stage_kind::C2P:
                    ok = next == llama_region_ring_stage_kind::PHONE;
                    break;
                case llama_region_ring_stage_kind::PHONE:
                    ok = next == llama_region_ring_stage_kind::GPU ||
                         next == llama_region_ring_stage_kind::P2C;
                    break;
                case llama_region_ring_stage_kind::P2C:
                    ok = next == llama_region_ring_stage_kind::GPU;
                    break;
            }
            if (!ok) {
                return false;
            }
        }
        next_layer = r.layer_end;
    }

    return phone_regions >= 2;
}

inline bool llama_region_ring_has_directional_stages(
        const std::vector<llama_region_ring_region> & regions) {
    return std::any_of(
        regions.begin(), regions.end(),
        [](const llama_region_ring_region & r) {
            return llama_region_ring_stage_is_directional(r.kind);
        });
}

inline std::vector<llama_region_ring_region> llama_region_ring_parse(
        const std::string & spec, int n_layer, bool allow_pc_tail = false) {
    // Legacy default remains byte-for-byte compatible with the final Ring
    // baseline. Directional handoffs are opt-in through the layout spelling,
    // e.g. G4,C6,C2P1,P4,P2C1,...
    const std::string layout = spec.empty() ?
        "G4,C7,P5,G4,C7,P5,G3,C8,P5" : spec;
    if (n_layer <= 0) {
        throw std::runtime_error("region ring requires a positive layer count");
    }

    std::vector<llama_region_ring_region> regions;
    size_t pos = 0;
    int layer = 0;
    while (pos < layout.size()) {
        while (pos < layout.size() &&
               std::isspace((unsigned char) layout[pos])) {
            ++pos;
        }
        if (pos == layout.size()) {
            break;
        }

        llama_region_ring_stage_kind kind;
        if (layout.compare(pos, 3, "C2P") == 0) {
            kind = llama_region_ring_stage_kind::C2P;
            pos += 3;
        } else if (layout.compare(pos, 3, "P2C") == 0) {
            kind = llama_region_ring_stage_kind::P2C;
            pos += 3;
        } else {
            const char letter = layout[pos++];
            if (letter == 'C') {
                kind = llama_region_ring_stage_kind::CPU;
            } else if (letter == 'G') {
                kind = llama_region_ring_stage_kind::GPU;
            } else if (letter == 'P') {
                kind = llama_region_ring_stage_kind::PHONE;
            } else {
                throw std::runtime_error(
                    "ring stages must be G, C, P, C2P or P2C");
            }
        }

        const size_t num_begin = pos;
        int count = 0;
        while (pos < layout.size() &&
               std::isdigit((unsigned char) layout[pos])) {
            count = count * 10 + (layout[pos++] - '0');
            if (count > n_layer) {
                throw std::runtime_error("region ring layer count overflow");
            }
        }
        if (num_begin == pos || count == 0 || layer + count > n_layer) {
            throw std::runtime_error("invalid region ring layer count");
        }
        if (llama_region_ring_stage_is_directional(kind) && count != 1) {
            throw std::runtime_error(
                "directional C2P/P2C stages must contain exactly one layer");
        }

        regions.emplace_back(kind, layer, layer + count);
        layer += count;

        while (pos < layout.size() &&
               std::isspace((unsigned char) layout[pos])) {
            ++pos;
        }
        if (pos == layout.size()) {
            break;
        }
        if (layout[pos] != ',') {
            throw std::runtime_error(
                "region ring stages must be comma-separated");
        }
        ++pos;
        while (pos < layout.size() &&
               std::isspace((unsigned char) layout[pos])) {
            ++pos;
        }
        if (pos == layout.size()) {
            throw std::runtime_error("trailing comma in region ring");
        }
    }

    if (layer != n_layer) {
        throw std::runtime_error(
            "region ring layout must cover every layer exactly once");
    }

    const bool directional =
        llama_region_ring_has_directional_stages(regions);
    if (directional) {
        if (allow_pc_tail ||
                !llama_region_ring_directional_topology_compatible(
                    regions, n_layer)) {
            throw std::runtime_error(
                "directional region topology must follow "
                "GPU -> CPU -> [C2P] -> Phone -> [P2C] -> GPU, "
                "end on Phone, and cover every layer");
        }
        return regions;
    }

    for (size_t i = 0; i < regions.size(); ++i) {
        const auto expected =
            LLAMA_REGION_RING_CYCLE_ORDER[i % 3];
        if (regions[i].device != expected) {
            throw std::runtime_error(
                "region ring must repeat GPU -> CPU -> Phone");
        }
    }

    const size_t full_cycles = regions.size() / 3;
    const size_t pc_tail_stages = regions.size() % 3;
    if (full_cycles < 2 ||
            (!allow_pc_tail && pc_tail_stages != 0) ||
            (allow_pc_tail && full_cycles > 4)) {
        throw std::runtime_error(
            allow_pc_tail ?
                "region topology requires 2-4 GPU/CPU/Phone cycles and "
                "an optional CPU/GPU tail covering every layer" :
                "region ring requires at least two GPU/CPU/Phone cycles "
                "covering every layer exactly once");
    }
    return regions;
}

// Enforce the general Plan contract before allocating any weights. The
// legacy parser intentionally remains strict unless allow_pc_tail is set.
inline bool llama_region_ring_apply_topology_compatible(
        const std::vector<llama_region_ring_region> & regions,
        int n_layer) {
    if (regions.size() < 6 || regions.size() > 14 ||
            n_layer <= 0 || regions.front().layer_begin != 0 ||
            regions.back().layer_end != n_layer) {
        return false;
    }
    const size_t cycles = regions.size() / 3;
    if (cycles < 2 || cycles > 4) {
        return false;
    }
    int next_layer = 0;
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto & r = regions[i];
        const auto expected = LLAMA_REGION_RING_CYCLE_ORDER[i % 3];
        if (r.device != expected || r.layer_begin != next_layer ||
                r.layer_end <= r.layer_begin || r.layer_end > n_layer) {
            return false;
        }
        if (expected == llama_region_ring_device::PHONE &&
                (r.layer_end - r.layer_begin < 2 ||
                 r.layer_end - r.layer_begin > 8)) {
            return false;
        }
        next_layer = r.layer_end;
    }
    return next_layer == n_layer;
}

// Canonical Ring layout spelling, shared by the model loader, runtime
// scheduler and Planner. The selected Plan is stored on llama_model; this
// function is not a global mutable environment override.
inline std::string llama_region_ring_layout_string(
        const std::vector<llama_region_ring_region> & regions) {
    std::string out;
    for (const auto & r : regions) {
        if (!out.empty()) { out += ","; }
        switch (r.kind) {
            case llama_region_ring_stage_kind::CPU:   out += "C"; break;
            case llama_region_ring_stage_kind::GPU:   out += "G"; break;
            case llama_region_ring_stage_kind::PHONE: out += "P"; break;
            case llama_region_ring_stage_kind::C2P:   out += "C2P"; break;
            case llama_region_ring_stage_kind::P2C:   out += "P2C"; break;
        }
        out += std::to_string(r.layer_end - r.layer_begin);
    }
    return out;
}

// Phase-1 APPLY supports changing ONLY the CPU/CUDA boundary inside
// existing PC windows. Phone regions, their layers/KV, cycle count,
// total PC/CUDA weights and last-Phone logits owner must not change.
// No PC tail, nonuniform Chunk or new Phone graph shape is permitted.
inline bool llama_region_ring_apply_safe_compatible(
        const std::vector<llama_region_ring_region> & baseline,
        const std::vector<llama_region_ring_region> & proposed) {
    if (baseline.size() < 6 || baseline.size() != proposed.size() ||
            baseline.size() % 3 != 0 ||
            baseline.front().layer_begin != 0 ||
            proposed.front().layer_begin != 0) {
        return false;
    }
    int baseline_gpu_layers = 0;
    int proposed_gpu_layers = 0;
    for (size_t i = 0; i < baseline.size(); i += 3) {
        const auto & bg = baseline[i];
        const auto & bc = baseline[i + 1];
        const auto & bp = baseline[i + 2];
        const auto & g = proposed[i];
        const auto & c = proposed[i + 1];
        const auto & p = proposed[i + 2];
        if (bg.device != llama_region_ring_device::GPU ||
            bc.device != llama_region_ring_device::CPU ||
            bp.device != llama_region_ring_device::PHONE ||
            g.device != bg.device || c.device != bc.device ||
            p.device != bp.device ||
            bg.layer_end != bc.layer_begin ||
            bc.layer_end != bp.layer_begin ||
            g.layer_begin != bg.layer_begin ||
            g.layer_begin >= g.layer_end ||
            g.layer_end != c.layer_begin ||
            c.layer_begin >= c.layer_end ||
            c.layer_end != bp.layer_begin ||
            p.layer_begin != bp.layer_begin ||
            p.layer_end != bp.layer_end ||
            p.layer_begin >= p.layer_end ||
            (i > 0 && baseline[i - 1].layer_end != bg.layer_begin) ||
            (i > 0 && proposed[i - 1].layer_end != g.layer_begin)) {
            return false;
        }
        baseline_gpu_layers += bg.layer_end - bg.layer_begin;
        proposed_gpu_layers += g.layer_end - g.layer_begin;
    }
    return baseline_gpu_layers == proposed_gpu_layers &&
           baseline.back().layer_end == proposed.back().layer_end;
}

// Phone READY/Fusion/ownership on the current 48-layer Qwen3 MoE has
// only been exercised with these three exact five-layer Phone regions.
inline bool llama_region_ring_apply_safe_phone_shape(
        const std::vector<llama_region_ring_region> & regions) {
    if (regions.size() != 9 || regions.back().layer_end != 48) {
        return false;
    }
    const int begins[3] = {11, 27, 43};
    const int ends[3]   = {16, 32, 48};
    for (int i = 0; i < 3; ++i) {
        const auto & p = regions[3 * i + 2];
        if (p.device != llama_region_ring_device::PHONE ||
            p.layer_begin != begins[i] || p.layer_end != ends[i]) {
            return false;
        }
    }
    return true;
}

inline std::vector<llama_region_ring_region> llama_region_ring_from_env(
        int n_layer) {
    const char * spec = std::getenv("LLAMA_HYBRID_REGION_RING_LAYOUT");
    return llama_region_ring_parse(spec != nullptr ? spec : "", n_layer);
}

inline std::array<int, 3> llama_region_ring_layer_counts(
        const std::vector<llama_region_ring_region> & regions) {
    std::array<int, 3> counts = {};
    for (const auto & region : regions) {
        // Directional boundary layers are Tensor layers physically present on
        // both sides; do not misreport them as ordinary Phone-full layers.
        if (llama_region_ring_stage_is_directional(region.kind)) {
            continue;
        }
        counts[(size_t) region.device] +=
            region.layer_end - region.layer_begin;
    }
    return counts;
}

inline int llama_region_ring_directional_layer_count(
        const std::vector<llama_region_ring_region> & regions) {
    int count = 0;
    for (const auto & region : regions) {
        if (llama_region_ring_stage_is_directional(region.kind)) {
            count += region.layer_end - region.layer_begin;
        }
    }
    return count;
}

struct llama_region_ring_task {
    size_t region = 0;
    size_t chunk = 0;
    int token_begin = 0;
    int n_tokens = 0;
    // Dynamic activation-slot leases. They are assigned only when the task is
    // claimed; READY queue entries are not tied to chunk % capacity.
    int input_slot = -1;
    int output_slot = -1;
};

// Cooperative, dependency-driven per-device FIFOs.  A region's (L,C) task
// becomes runnable only after its upstream region has completed that chunk AND
// the same region has completed chunk C-1 (KV-cache commit ordering).  Completion
// publishes downstream readiness; device execution is supplied by the caller.
class llama_region_ring_ready {
public:
    llama_region_ring_ready(
            const std::vector<llama_region_ring_region> & regions,
            int n_tokens, int chunk_tokens,
            size_t slot_pool_size = 0,
            bool gpu_two_bank_lifetime = false) :
        llama_region_ring_ready(
            regions, n_tokens,
            std::vector<int>(regions.size(), chunk_tokens),
            slot_pool_size, gpu_two_bank_lifetime) {
    }

    llama_region_ring_ready(
            const std::vector<llama_region_ring_region> & regions,
            int n_tokens,
            const std::vector<int> & region_chunk_tokens,
            size_t slot_pool_size = 0,
            bool gpu_two_bank_lifetime = false) :
        regions_(regions),
        region_chunk_tokens_(region_chunk_tokens),
        slot_pool_size_(slot_pool_size),
        n_tokens_(n_tokens),
        gpu_two_bank_lifetime_(gpu_two_bank_lifetime) {
        if (regions_.empty() || n_tokens_ <= 0 ||
                region_chunk_tokens_.size() != regions_.size() ||
                std::any_of(
                    region_chunk_tokens_.begin(), region_chunk_tokens_.end(),
                    [](int x) { return x <= 0; })) {
            throw std::runtime_error(
                "invalid region ring scheduler dimensions");
        }

        const bool uniform_chunks =
            std::all_of(
                region_chunk_tokens_.begin() + 1,
                region_chunk_tokens_.end(),
                [&](int x) {
                    return x == region_chunk_tokens_.front();
                });
        if (slot_pool_size_ > 0 && !uniform_chunks) {
            throw std::runtime_error(
                "bounded legacy slots require uniform region chunks");
        }

        chunk_begins_.resize(regions_.size());
        chunk_sizes_.resize(regions_.size());
        input_pending_.resize(regions_.size());
        state_.resize(regions_.size());

        for (size_t r = 0; r < regions_.size(); ++r) {
            const int x = region_chunk_tokens_[r];
            for (int begin = 0; begin < n_tokens_; begin += x) {
                chunk_begins_[r].push_back(begin);
                chunk_sizes_[r].push_back(
                    std::min(x, n_tokens_ - begin));
            }
            total_tasks_ += chunk_sizes_[r].size();
            input_pending_[r].assign(chunk_sizes_[r].size(), 0);
            state_[r].assign(chunk_sizes_[r].size(), 0);
        }

        // READY_V2 dependencies are token intervals, not equal chunk ids.
        // Each consumer records the number of overlapping producer intervals
        // that must complete before it can run.
        for (size_t r = 1; r < regions_.size(); ++r) {
            for (size_t dst = 0; dst < chunk_sizes_[r].size(); ++dst) {
                const int db = chunk_begins_[r][dst];
                const int de = db + chunk_sizes_[r][dst];
                int pending = 0;
                for (size_t src = 0;
                     src < chunk_sizes_[r - 1].size(); ++src) {
                    const int sb = chunk_begins_[r - 1][src];
                    const int se = sb + chunk_sizes_[r - 1][src];
                    if (std::min(se, de) > std::max(sb, db)) {
                        ++pending;
                    }
                }
                if (pending <= 0) {
                    throw std::runtime_error(
                        "region ring interval dependency gap");
                }
                input_pending_[r][dst] = pending;
            }
        }

        if (slot_pool_size_ > 0 && regions_.size() > 1) {
            output_slots_.assign(
                regions_.size() - 1,
                std::vector<output_slot>(slot_pool_size_));
        }

        offer_locked(0, 0);
    }

    bool has_ready() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            return false;
        }
        for (size_t dev = 0; dev < queues_.size(); ++dev) {
            if (has_claimable_locked(dev)) {
                return true;
            }
        }
        return false;
    }

    bool all_done() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return all_done_locked();
    }

    bool failed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return failed_;
    }

    // Compatibility for existing uniform-chunk tests/callers.
    size_t n_chunks() const {
        return chunk_sizes_.empty() ? 0 : chunk_sizes_.front().size();
    }

    size_t n_chunks(size_t region) const {
        if (region >= chunk_sizes_.size()) {
            throw std::runtime_error(
                "invalid region ring region index");
        }
        return chunk_sizes_[region].size();
    }

    size_t total_tasks() const {
        return total_tasks_;
    }

    int chunk_tokens(size_t region) const {
        if (region >= region_chunk_tokens_.size()) {
            throw std::runtime_error(
                "invalid region ring region index");
        }
        return region_chunk_tokens_[region];
    }

    size_t n_regions() const { return regions_.size(); }
    size_t slot_pool_size() const { return slot_pool_size_; }
    size_t output_window() const { return slot_pool_size_; }

    size_t live_output_slots() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (const auto & edge : output_slots_) {
            for (const auto & slot : edge) {
                count += slot.state != 0;
            }
        }
        return count;
    }

    size_t finished() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return finished_;
    }

    bool region_done(size_t region) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (region >= state_.size()) {
            throw std::runtime_error(
                "invalid region ring region index");
        }
        return std::all_of(
            state_[region].begin(), state_[region].end(),
            [](uint8_t state) { return state == 3; });
    }

    size_t queue_size(
            llama_region_ring_device device) const {
        const size_t dev = device_index(device);
        std::lock_guard<std::mutex> lock(mutex_);
        return queues_[dev].size();
    }

    std::optional<llama_region_ring_task> take() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            return std::nullopt;
        }
        for (size_t pass = 0; pass < queues_.size(); ++pass) {
            const size_t dev =
                (next_device_ + pass) % queues_.size();
            if (auto task = take_device_locked(dev)) {
                next_device_ = (dev + 1) % queues_.size();
                return task;
            }
        }
        return std::nullopt;
    }

    std::optional<llama_region_ring_task> try_take(
            llama_region_ring_device device) {
        const size_t dev = device_index(device);
        std::lock_guard<std::mutex> lock(mutex_);
        return failed_ ?
            std::nullopt : take_device_locked(dev);
    }

    std::optional<llama_region_ring_task> wait_take(
            llama_region_ring_device device) {
        const size_t dev = device_index(device);
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            return failed_ || all_done_locked() ||
                   has_claimable_locked(dev);
        });
        if (failed_ || all_done_locked()) {
            return std::nullopt;
        }
        return take_device_locked(dev);
    }

    void complete(const llama_region_ring_task & task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!valid_running_locked(task)) {
                throw std::runtime_error(
                    "invalid region ring completion");
            }

            if (slot_pool_size_ > 0) {
                // Legacy bounded-slot mode remains uniform-chunk only.
                if (task.region > 0) {
                    if (task.input_slot < 0 ||
                            (size_t) task.input_slot >=
                                output_slots_[task.region - 1].size()) {
                        throw std::runtime_error(
                            "region ring task lost its input slot lease");
                    }
                    auto & input =
                        output_slots_[task.region - 1]
                                     [(size_t) task.input_slot];
                    if (input.state != 3 ||
                            input.chunk != task.chunk) {
                        throw std::runtime_error(
                            "region ring input slot released before consumption");
                    }
                    input = {};
                }
                if (task.region + 1 < regions_.size()) {
                    if (task.output_slot < 0 ||
                            (size_t) task.output_slot >=
                                output_slots_[task.region].size()) {
                        throw std::runtime_error(
                            "region ring task lost its output slot lease");
                    }
                    auto & output =
                        output_slots_[task.region]
                                     [(size_t) task.output_slot];
                    if (output.state != 1 ||
                            output.chunk != task.chunk) {
                        throw std::runtime_error(
                            "region ring output slot was overwritten");
                    }
                    output.state = 2;
                }
            }

            state_[task.region][task.chunk] = 3;
            ++finished_;

            if (!failed_) {
                if (task.region + 1 < regions_.size()) {
                    const size_t next_region = task.region + 1;
                    const int sb = task.token_begin;
                    const int se = sb + task.n_tokens;
                    for (size_t dst = 0;
                         dst < chunk_sizes_[next_region].size();
                         ++dst) {
                        const int db =
                            chunk_begins_[next_region][dst];
                        const int de =
                            db + chunk_sizes_[next_region][dst];
                        if (std::min(se, de) <=
                                std::max(sb, db)) {
                            continue;
                        }
                        int & pending =
                            input_pending_[next_region][dst];
                        if (pending <= 0) {
                            throw std::runtime_error(
                                "region ring interval dependency underflow");
                        }
                        --pending;
                        if (pending == 0) {
                            offer_locked(next_region, dst);
                        }
                    }
                }

                // Strict per-Region KV commit order is independent of the
                // upstream interval geometry.
                offer_locked(task.region, task.chunk + 1);
            }
        }
        cv_.notify_all();
    }

    void fail(const llama_region_ring_task & task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!valid_running_locked(task)) {
                throw std::runtime_error(
                    "invalid region ring failure");
            }
            state_[task.region][task.chunk] = 4;
            failed_ = true;
        }
        cv_.notify_all();
    }

    void cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
        }
        cv_.notify_all();
    }

private:
    static size_t device_index(
            llama_region_ring_device device) {
        const size_t dev = (size_t) device;
        if (dev >= 3) {
            throw std::runtime_error(
                "invalid region ring device");
        }
        return dev;
    }

    bool all_done_locked() const {
        return finished_ == total_tasks_;
    }

    bool valid_running_locked(
            const llama_region_ring_task & task) const {
        return task.region < regions_.size() &&
               task.chunk < state_[task.region].size() &&
               state_[task.region][task.chunk] == 2;
    }

    struct output_slot {
        size_t  chunk = 0;
        uint8_t state = 0;
    };

    int find_free_slot_locked(size_t edge) const {
        if (edge >= output_slots_.size()) {
            return -1;
        }
        const auto & slots = output_slots_[edge];
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].state == 0) {
                return (int) i;
            }
        }
        return -1;
    }

    int find_ready_slot_locked(
            size_t edge, size_t chunk) const {
        if (edge >= output_slots_.size()) {
            return -1;
        }
        const auto & slots = output_slots_[edge];
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].state == 2 &&
                    slots[i].chunk == chunk) {
                return (int) i;
            }
        }
        return -1;
    }

    bool claimable_locked(
            const llama_region_ring_task & task) const {
        // Adjacent streamed GPU Regions live in different Banks and may
        // interleave as soon as their token-interval dependencies become
        // READY. Only same-Bank reuse needs an extra lifetime constraint:
        //
        //   rank0 -> A, rank1 -> B, rank2 -> A, rank3 -> B, ...
        //
        // Therefore rank k>=2 waits only for rank k-2 to drain. The previous
        // "all earlier GPU Regions must finish" rule serialized the Ring.
        if (gpu_two_bank_lifetime_ &&
                regions_[task.region].device ==
                    llama_region_ring_device::GPU) {
            size_t gpu_rank = 0;
            for (size_t r = 0; r < task.region; ++r) {
                gpu_rank +=
                    regions_[r].device ==
                    llama_region_ring_device::GPU;
            }

            if (gpu_rank >= 2) {
                const size_t predecessor_rank = gpu_rank - 2;
                size_t seen_rank = 0;
                size_t predecessor_region = regions_.size();
                for (size_t r = 0; r < task.region; ++r) {
                    if (regions_[r].device !=
                            llama_region_ring_device::GPU) {
                        continue;
                    }
                    if (seen_rank == predecessor_rank) {
                        predecessor_region = r;
                        break;
                    }
                    ++seen_rank;
                }

                if (predecessor_region >= regions_.size() ||
                        std::any_of(
                            state_[predecessor_region].begin(),
                            state_[predecessor_region].end(),
                            [](uint8_t state) { return state != 3; })) {
                    return false;
                }
            }
        }

        if (slot_pool_size_ == 0) {
            return true;
        }
        if (task.region + 1 < regions_.size() &&
                find_free_slot_locked(task.region) < 0) {
            return false;
        }
        if (task.region > 0 &&
                find_ready_slot_locked(
                    task.region - 1, task.chunk) < 0) {
            return false;
        }
        return true;
    }

    bool has_claimable_locked(size_t dev) const {
        const auto & queue = queues_[dev];
        return std::any_of(
            queue.begin(), queue.end(),
            [this](const auto & task) {
                return claimable_locked(task);
            });
    }

    std::optional<llama_region_ring_task>
    take_device_locked(size_t dev) {
        auto & queue = queues_[dev];
        const auto it = std::find_if(
            queue.begin(), queue.end(),
            [this](const auto & task) {
                return claimable_locked(task);
            });
        if (it == queue.end()) {
            return std::nullopt;
        }

        auto task = *it;
        queue.erase(it);
        if (state_[task.region][task.chunk] != 1 ||
                (size_t) regions_[task.region].device != dev) {
            throw std::runtime_error(
                "region ring task was claimed twice or by wrong device");
        }

        if (slot_pool_size_ > 0) {
            if (task.region + 1 < regions_.size()) {
                task.output_slot =
                    find_free_slot_locked(task.region);
                if (task.output_slot < 0) {
                    throw std::runtime_error(
                        "region ring output slot disappeared while claiming");
                }
                auto & output =
                    output_slots_[task.region]
                                 [(size_t) task.output_slot];
                output = {task.chunk, 1};
            }
            if (task.region > 0) {
                task.input_slot =
                    find_ready_slot_locked(
                        task.region - 1, task.chunk);
                if (task.input_slot < 0) {
                    throw std::runtime_error(
                        "region ring input slot disappeared while claiming");
                }
                auto & input =
                    output_slots_[task.region - 1]
                                 [(size_t) task.input_slot];
                input.state = 3;
            }
        }

        state_[task.region][task.chunk] = 2;
        return task;
    }

    void offer_locked(size_t region, size_t chunk) {
        if (region >= regions_.size() ||
                chunk >= chunk_sizes_[region].size() ||
                state_[region][chunk] != 0 ||
                input_pending_[region][chunk] != 0 ||
                (chunk > 0 &&
                 state_[region][chunk - 1] != 3)) {
            return;
        }

        state_[region][chunk] = 1;
        queues_[(size_t) regions_[region].device].push_back({
            region,
            chunk,
            chunk_begins_[region][chunk],
            chunk_sizes_[region][chunk]
        });
    }

    const std::vector<llama_region_ring_region> regions_;
    const std::vector<int> region_chunk_tokens_;
    const size_t slot_pool_size_ = 0;
    const int n_tokens_ = 0;
    const bool gpu_two_bank_lifetime_ = false;
    std::vector<std::vector<output_slot>> output_slots_;
    std::vector<std::vector<int>> chunk_begins_;
    std::vector<std::vector<int>> chunk_sizes_;
    std::vector<std::vector<int>> input_pending_;
    std::vector<std::vector<uint8_t>> state_;
    std::array<std::deque<llama_region_ring_task>, 3> queues_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t total_tasks_ = 0;
    size_t finished_ = 0;
    size_t next_device_ = 0;
    bool failed_ = false;
};
