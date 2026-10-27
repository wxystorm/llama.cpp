#pragma once

// Experimental CPU -> CUDA -> Phone region ring.  This header is intentionally
// independent of GGML so its task dependencies can be tested in isolation.
#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <cstdlib>
#include <deque>
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

struct llama_region_ring_region {
    llama_region_ring_device device;
    int layer_begin;
    int layer_end; // exclusive
};

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

inline std::vector<llama_region_ring_region> llama_region_ring_parse(
        const std::string & spec, int n_layer, bool allow_pc_tail = false) {
    // L0..L47: CPU7 / GPU4 / Phone5 / CPU7 / GPU4 / Phone5 /
    //          CPU8 / GPU3 / Phone5.  Counts: CPU22, GPU11, Phone15.
    const std::string layout = spec.empty() ?
        "C7,G4,P5,C7,G4,P5,C8,G3,P5" : spec;
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
        const char letter = layout[pos++];
        llama_region_ring_device device;
        if (letter == 'C') {
            device = llama_region_ring_device::CPU;
        } else if (letter == 'G') {
            device = llama_region_ring_device::GPU;
        } else if (letter == 'P') {
            device = llama_region_ring_device::PHONE;
        } else {
            throw std::runtime_error("ring stages must be C, G or P");
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
        const auto expected =
            static_cast<llama_region_ring_device>(regions.size() % 3);
        if (device != expected) {
            throw std::runtime_error(
                "region ring must repeat CPU -> GPU -> Phone");
        }
        regions.push_back({device, layer, layer + count});
        layer += count;
        while (pos < layout.size() &&
               std::isspace((unsigned char) layout[pos])) {
            ++pos;
        }
        // Reaching the end after a valid region is normal. Only treat the
        // end as a trailing comma if we actually consumed a separator.
        if (pos == layout.size()) {
            break;
        }
        if (layout[pos] != ',') {
            throw std::runtime_error("region ring stages must be comma-separated");
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

    const size_t full_cycles = regions.size() / 3;
    const size_t pc_tail_stages = regions.size() % 3;
    if (full_cycles < 2 || layer != n_layer ||
            (!allow_pc_tail && pc_tail_stages != 0) ||
            (allow_pc_tail && full_cycles > 4)) {
        throw std::runtime_error(
            allow_pc_tail ?
                "region topology requires 2-4 CPU/GPU/Phone cycles and "
                "an optional CPU/GPU tail covering every layer" :
                "region ring requires at least two CPU/GPU/Phone cycles "
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
        const auto expected = static_cast<llama_region_ring_device>(i % 3);
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
        const char letter =
            r.device == llama_region_ring_device::CPU ? 'C' :
            r.device == llama_region_ring_device::GPU ? 'G' : 'P';
        out += letter;
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
        const auto & bc = baseline[i];
        const auto & bg = baseline[i + 1];
        const auto & bp = baseline[i + 2];
        const auto & c = proposed[i];
        const auto & g = proposed[i + 1];
        const auto & p = proposed[i + 2];
        if (bc.device != llama_region_ring_device::CPU ||
            bg.device != llama_region_ring_device::GPU ||
            bp.device != llama_region_ring_device::PHONE ||
            c.device != bc.device || g.device != bg.device ||
            p.device != bp.device ||
            bc.layer_end != bg.layer_begin ||
            bg.layer_end != bp.layer_begin ||
            c.layer_begin != bc.layer_begin ||
            c.layer_begin >= c.layer_end ||
            c.layer_end != g.layer_begin ||
            g.layer_begin >= g.layer_end ||
            g.layer_end != bp.layer_begin ||
            p.layer_begin != bp.layer_begin ||
            p.layer_end != bp.layer_end ||
            p.layer_begin >= p.layer_end ||
            (i > 0 && baseline[i - 1].layer_end != bc.layer_begin) ||
            (i > 0 && proposed[i - 1].layer_end != c.layer_begin)) {
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
        counts[(size_t) region.device] +=
            region.layer_end - region.layer_begin;
    }
    return counts;
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
            int n_tokens, int chunk_tokens, size_t slot_pool_size = 0) :
        regions_(regions), slot_pool_size_(slot_pool_size) {
        if (regions_.empty() || n_tokens <= 0 || chunk_tokens <= 0) {
            throw std::runtime_error("invalid region ring scheduler dimensions");
        }
        for (int begin = 0; begin < n_tokens; begin += chunk_tokens) {
            chunk_begins_.push_back(begin);
            chunk_sizes_.push_back(std::min(chunk_tokens, n_tokens - begin));
        }
        const size_t n_chunks = chunk_sizes_.size();
        if (slot_pool_size_ > 0 && regions_.size() > 1) {
            output_slots_.assign(regions_.size() - 1, std::vector<output_slot>(slot_pool_size_));
        }
        input_ready_.assign(regions_.size(), std::vector<uint8_t>(n_chunks, 0));
        state_.assign(regions_.size(), std::vector<uint8_t>(n_chunks, 0));
        for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
            input_ready_[0][chunk] = 1;
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

    size_t n_chunks() const { return chunk_sizes_.size(); }
    size_t n_regions() const { return regions_.size(); }
    size_t slot_pool_size() const { return slot_pool_size_; }
    // Compatibility alias for older tests/callers; runtime semantics are a
    // dynamic slot pool, not chunk-to-slot modulo mapping.
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

    size_t queue_size(llama_region_ring_device device) const {
        const size_t dev = device_index(device);
        std::lock_guard<std::mutex> lock(mutex_);
        return queues_[dev].size();
    }

    // Preserve the original cooperative round-robin entry point for "ready".
    std::optional<llama_region_ring_task> take() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            return std::nullopt;
        }
        for (size_t pass = 0; pass < queues_.size(); ++pass) {
            const size_t dev = (next_device_ + pass) % queues_.size();
            if (auto task = take_device_locked(dev)) {
                next_device_ = (dev + 1) % queues_.size();
                return task;
            }
        }
        return std::nullopt;
    }

    // New device-owned FIFO claim. This does not block or rotate other queues.
    std::optional<llama_region_ring_task> try_take(llama_region_ring_device device) {
        const size_t dev = device_index(device);
        std::lock_guard<std::mutex> lock(mutex_);
        return failed_ ? std::nullopt : take_device_locked(dev);
    }

    // Block a device-specific host worker until its next task owns the needed
    // input and output credits, or all work completes / a failure cancels it.
    std::optional<llama_region_ring_task> wait_take(llama_region_ring_device device) {
        const size_t dev = device_index(device);
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] {
            return failed_ || all_done_locked() || has_claimable_locked(dev);
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
                throw std::runtime_error("invalid region ring completion");
            }
            if (slot_pool_size_ > 0) {
                // Release the exact input lease carried by this task only
                // after the consumer has finished all required reads.
                if (task.region > 0) {
                    if (task.input_slot < 0 ||
                        (size_t) task.input_slot >= output_slots_[task.region - 1].size()) {
                        throw std::runtime_error("region ring task lost its input slot lease");
                    }
                    auto & input =
                        output_slots_[task.region - 1][(size_t) task.input_slot];
                    if (input.state != 3 || input.chunk != task.chunk) {
                        throw std::runtime_error("region ring input slot released before consumption");
                    }
                    input = {};
                }
                if (task.region + 1 < regions_.size()) {
                    if (task.output_slot < 0 ||
                        (size_t) task.output_slot >= output_slots_[task.region].size()) {
                        throw std::runtime_error("region ring task lost its output slot lease");
                    }
                    auto & output =
                        output_slots_[task.region][(size_t) task.output_slot];
                    if (output.state != 1 || output.chunk != task.chunk) {
                        throw std::runtime_error("region ring output slot was overwritten");
                    }
                    output.state = 2;
                }
            }
            state_[task.region][task.chunk] = 3;
            ++finished_;

            // If another worker failed, drain in-flight work without publishing
            // any new tasks. Existing READY entries become unclaimable.
            if (!failed_) {
                if (task.region + 1 < regions_.size()) {
                    input_ready_[task.region + 1][task.chunk] = 1;
                    offer_locked(task.region + 1, task.chunk);
                }
                offer_locked(task.region, task.chunk + 1);
            }
        }
        cv_.notify_all();
    }

    void fail(const llama_region_ring_task & task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!valid_running_locked(task)) {
                throw std::runtime_error("invalid region ring failure");
            }
            state_[task.region][task.chunk] = 4;
            failed_ = true;
        }
        cv_.notify_all();
    }

    // Used when an executor fails outside a claimed task.
    void cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
        }
        cv_.notify_all();
    }

private:
    static size_t device_index(llama_region_ring_device device) {
        const size_t dev = (size_t) device;
        if (dev >= 3) {
            throw std::runtime_error("invalid region ring device");
        }
        return dev;
    }

    bool all_done_locked() const {
        return finished_ == regions_.size() * chunk_sizes_.size();
    }

    bool valid_running_locked(const llama_region_ring_task & task) const {
        return task.region < regions_.size() &&
               task.chunk < chunk_sizes_.size() &&
               state_[task.region][task.chunk] == 2;
    }

    struct output_slot {
        size_t  chunk = 0;
        uint8_t state = 0; // 0=FREE 1=WRITING 2=READY 3=READING
    };

    // Activation buffers are a dynamic per-edge pool. A producer may use ANY
    // free slot; a consumer looks up the READY slot that owns its chunk.
    // This removes the artificial chunk % capacity binding while preserving
    // bounded memory and explicit FREE/WRITING/READY/READING ownership.
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

    int find_ready_slot_locked(size_t edge, size_t chunk) const {
        if (edge >= output_slots_.size()) {
            return -1;
        }
        const auto & slots = output_slots_[edge];
        for (size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].state == 2 && slots[i].chunk == chunk) {
                return (int) i;
            }
        }
        return -1;
    }

    bool claimable_locked(const llama_region_ring_task & task) const {
        if (slot_pool_size_ == 0) {
            return true;
        }
        if (task.region + 1 < regions_.size() &&
            find_free_slot_locked(task.region) < 0) {
            return false;
        }
        if (task.region > 0 &&
            find_ready_slot_locked(task.region - 1, task.chunk) < 0) {
            return false;
        }
        return true;
    }

    bool has_claimable_locked(size_t dev) const {
        const auto & queue = queues_[dev];
        return std::any_of(queue.begin(), queue.end(), [this](const auto & task) {
            return claimable_locked(task);
        });
    }

    std::optional<llama_region_ring_task> take_device_locked(size_t dev) {
        auto & queue = queues_[dev];
        // Skip credit-blocked tasks; do not block unrelated READY regions.
        // Exact per-region KV dependencies still order consecutive chunks.
        const auto it = std::find_if(queue.begin(), queue.end(),
            [this](const auto & task) { return claimable_locked(task); });
        if (it == queue.end()) {
            return std::nullopt;
        }
        auto task = *it;
        queue.erase(it);
        if (state_[task.region][task.chunk] != 1 ||
            (size_t) regions_[task.region].device != dev) {
            throw std::runtime_error("region ring task was claimed twice or by wrong device");
        }
        if (slot_pool_size_ > 0) {
            if (task.region + 1 < regions_.size()) {
                task.output_slot = find_free_slot_locked(task.region);
                if (task.output_slot < 0) {
                    throw std::runtime_error("region ring output slot disappeared while claiming");
                }
                auto & output =
                    output_slots_[task.region][(size_t) task.output_slot];
                output = {task.chunk, 1};
            }
            if (task.region > 0) {
                task.input_slot =
                    find_ready_slot_locked(task.region - 1, task.chunk);
                if (task.input_slot < 0) {
                    throw std::runtime_error("region ring input slot disappeared while claiming");
                }
                auto & input =
                    output_slots_[task.region - 1][(size_t) task.input_slot];
                input.state = 3;
            }
        }
        state_[task.region][task.chunk] = 2;
        return task;
    }

    // Called only while holding mutex_, except from the constructor.
    void offer_locked(size_t region, size_t chunk) {
        if (region >= regions_.size() ||
            chunk >= chunk_sizes_.size() ||
            state_[region][chunk] != 0 ||
            !input_ready_[region][chunk] ||
            (chunk > 0 && state_[region][chunk - 1] != 3)) {
            return;
        }
        state_[region][chunk] = 1;
        queues_[(size_t) regions_[region].device].push_back({
            region, chunk, chunk_begins_[chunk], chunk_sizes_[chunk]});
    }

    const std::vector<llama_region_ring_region> regions_;
    const size_t slot_pool_size_ = 0; // 0: legacy full-ubatch output storage
    std::vector<std::vector<output_slot>> output_slots_;
    std::vector<int> chunk_begins_;
    std::vector<int> chunk_sizes_;
    std::vector<std::vector<uint8_t>> input_ready_;
    // 0=not queued, 1=ready, 2=running, 3=complete, 4=failed
    std::vector<std::vector<uint8_t>> state_;
    std::array<std::deque<llama_region_ring_task>, 3> queues_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t finished_ = 0;
    size_t next_device_ = 0;
    bool failed_ = false;
};
