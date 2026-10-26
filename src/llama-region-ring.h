#pragma once

// Experimental CPU -> CUDA -> Phone region ring.  This header is intentionally
// independent of GGML so its task dependencies can be tested in isolation.
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
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
        const std::string & spec, int n_layer) {
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
        if (pos < layout.size() && layout[pos++] != ',') {
            throw std::runtime_error("region ring stages must be comma-separated");
        }
        if (pos == layout.size()) {
            throw std::runtime_error("trailing comma in region ring");
        }
    }

    if (regions.size() < 6 || regions.size() % 3 != 0 || layer != n_layer) {
        throw std::runtime_error(
            "region ring requires at least two CPU/GPU/Phone cycles "
            "covering every layer exactly once");
    }
    return regions;
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
};

// Cooperative, dependency-driven per-device FIFOs.  A region's (L,C) task
// becomes runnable only after its upstream region has completed that chunk AND
// the same region has completed chunk C-1 (KV-cache commit ordering).  Completion
// publishes downstream readiness; device execution is supplied by the caller.
class llama_region_ring_ready {
public:
    llama_region_ring_ready(
            const std::vector<llama_region_ring_region> & regions,
            int n_tokens, int chunk_tokens) :
        regions_(regions) {
        if (regions_.empty() || n_tokens <= 0 || chunk_tokens <= 0) {
            throw std::runtime_error("invalid region ring scheduler dimensions");
        }
        for (int begin = 0; begin < n_tokens; begin += chunk_tokens) {
            chunk_begins_.push_back(begin);
            chunk_sizes_.push_back(std::min(chunk_tokens, n_tokens - begin));
        }
        const size_t n_chunks = chunk_sizes_.size();
        input_ready_.assign(regions_.size(), std::vector<uint8_t>(n_chunks, 0));
        state_.assign(regions_.size(), std::vector<uint8_t>(n_chunks, 0));
        for (size_t chunk = 0; chunk < n_chunks; ++chunk) {
            input_ready_[0][chunk] = 1;
        }
        offer(0, 0);
    }

    bool has_ready() const {
        for (const auto & queue : queues_) {
            if (!queue.empty()) {
                return true;
            }
        }
        return false;
    }

    bool all_done() const {
        return finished_ == regions_.size() * chunk_sizes_.size();
    }

    size_t n_chunks() const { return chunk_sizes_.size(); }
    size_t n_regions() const { return regions_.size(); }
    size_t finished() const { return finished_; }

    size_t queue_size(llama_region_ring_device device) const {
        return queues_[(size_t) device].size();
    }

    std::optional<llama_region_ring_task> take() {
        for (size_t pass = 0; pass < 3; ++pass) {
            const size_t dev = (next_device_ + pass) % 3;
            auto & queue = queues_[dev];
            if (queue.empty()) {
                continue;
            }
            auto task = queue.front();
            queue.pop_front();
            if (state_[task.region][task.chunk] != 1) {
                throw std::runtime_error("region ring task was claimed twice");
            }
            state_[task.region][task.chunk] = 2;
            next_device_ = (dev + 1) % 3;
            return task;
        }
        return std::nullopt;
    }

    void complete(const llama_region_ring_task & task) {
        if (task.region >= regions_.size() ||
            task.chunk >= chunk_sizes_.size() ||
            state_[task.region][task.chunk] != 2) {
            throw std::runtime_error("invalid region ring completion");
        }
        state_[task.region][task.chunk] = 3;
        ++finished_;

        // Publish the current chunk to its next region. The downstream task
        // will be queued only after its own previous chunk has committed KV.
        if (task.region + 1 < regions_.size()) {
            input_ready_[task.region + 1][task.chunk] = 1;
            offer(task.region + 1, task.chunk);
        }
        offer(task.region, task.chunk + 1);
    }

private:
    void offer(size_t region, size_t chunk) {
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
    std::vector<int> chunk_begins_;
    std::vector<int> chunk_sizes_;
    std::vector<std::vector<uint8_t>> input_ready_;
    // 0=not queued, 1=ready, 2=running, 3=complete
    std::vector<std::vector<uint8_t>> state_;
    std::array<std::deque<llama_region_ring_task>, 3> queues_;
    size_t finished_ = 0;
    size_t next_device_ = 0;
};
