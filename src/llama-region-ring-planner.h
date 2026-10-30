#pragma once

// Read-only Region Ring Planner: search G/C/P cycles and independent Chunk
// sizes, then simulate token-interval dependencies. It never modifies the
// live Ring scheduler, buffers, KV cache, or Phone RPC state.
#include "llama-region-ring.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llama_ring_planner {

using device = llama_region_ring_device;

struct candidate {
    int cycles = 0;
    int c = 0;
    int g = 0;
    int p = 0;
    int tail_c = 0;
    int tail_g = 0;
    int xc = 64;
    int xg = 64;
    int xp = 64;
    // Exact ordered full-stage layout for non-uniform cycles. Empty retains
    // the historical uniform (G,C,P)*cycles + optional tail generator.
    std::vector<llama_region_ring_region> concrete_regions;
    // Complete first-Cycle-seeded prefix trace (not a runtime measurement).
    // Used to explain how each subsequent Cycle compensates previous load.
    int prefix_cycles = 0;
    std::array<double, 4> prefix_error {};
    std::array<std::array<double, 3>, 4> prefix_service {};
    double balance = 0.0;
    double coarse_ms = 0.0;
    double wall_ms = 0.0;
    std::array<double, 3> busy_ms {};
    // Non-zero only when the legacy CPU/CUDA serialization model is enabled.
    double pc_gate_busy_ms = 0.0;
    // Lower-bound PC compute span under the selected runtime semantics:
    // sum(CPU,GPU) when gated, max(CPU,GPU) when independent.
    double pc_compute_bound_ms = 0.0;
    // Total per-token service demand after simulation (or coarse estimation).
    std::array<double, 3> service_ms_per_token {};
    // Local one-cycle terms for the G->C->P ring balance equations:
    //   A = C*fC(XC)/XC
    //   B = G*fG(XG)/XG
    //   D = P*fP(XP)/XP
    //   e1 = |A-(D+TP2C)| / max(A,D+TP2C)
    //   e2 = |A-B|         / max(A,B)
    //   e3 = |D-(B+TC2P)| / max(D,B+TC2P)
    // balance is the minimax objective max(e1,e2,e3).
    std::array<double, 3> ring_stage_ms_per_token {};
    double ring_p2c_ms_per_token = 0.0;
    double ring_c2p_ms_per_token = 0.0;
    std::array<double, 3> ring_error {};
    double ring_error_max = 0.0;
    double ring_error_mean = 0.0;
    double phone_link_ms_per_token = 0.0;
    double pc_local_link_ms_per_token = 0.0;
    std::array<double, 3> transfer_busy_ms {};
    // Shared host-side graph/split preparation lane. This is service time,
    // not lock wait; queueing is reconstructed by simulate().
    double prepare_busy_ms = 0.0;
    double weight_prepare_busy_ms = 0.0;
    // Single CPU-owned CUDA transfer lane used by rolling two-Bank weight
    // streaming. This is one load per logical GPU Region, not per Chunk.
    double weight_h2d_busy_ms = 0.0;
    // The first two GPU Regions are load-time resident in Bank A/B and do not
    // contribute to prompt latency. Only rolling rank>=2 replacements are
    // modeled here; their H2D is exposed only when it misses the natural-ready
    // point of the target GPU Region.
    double weight_h2d_startup_ms = 0.0;
    double weight_h2d_exposed_ms = 0.0;
    double weight_h2d_hidden_ms = 0.0;
    int tasks = 0;
    int edges = 0;
    int max_live_slots = 0;
};

struct model {
    int layers = 48;
    int tokens = 358;
    int kv_start = 0;
    int hidden = 2048;
    int slot_pool = (int) LLAMA_REGION_RING_SLOT_POOL_SIZE;
    int max_tail = 12;
    bool shared_phone_link = true; // Conservative effective Wi-Fi contention.
    // Current Region Ring runtime has independent CPU and CUDA compute lanes.
    // Keep this switch only so tests can compare the historical gated model.
    bool pc_compute_gate = false;
    // Runtime contract: logical CUDA layer count may exceed VRAM because
    // weights are rolled through two persistent Banks. Adjacent GPU Regions
    // may interleave; only same-Bank rank k-2 lifetime is serialized.
    bool gpu_weight_streaming = false;
    double phone_register_ms = 0.0; // Legacy MoE P5 fallback only.
    // Dense: measured/proxied RPC graph bytes, not a hard-coded P5 graph.
    // embedded_directional means C2P -> Phone Full -> P2C runs inside Meta.
    // The callback accounts for one fused graph when that feature is enabled,
    // or for the actual per-layer Meta SG submissions when it is disabled.
    std::function<bool(
        llama_region_ring_stage_kind, int, int, bool, double &)>
        phone_registration_cost;
    // Return false when no measured/interpolated profile supports the shape.
    std::function<bool(device, int, int, int, double &)> compute_ms;
    // Optional host-side scheduler prepare cost for the same shape. Missing
    // callbacks preserve the historical zero-prepare simulator behaviour.
    std::function<bool(device, int, int, int, double &)> prepare_ms;
    // One directional Tensor layer already includes its internal route,
    // split-FFN, return and reduce traffic. Keeping this callback separate
    // prevents C2P/P2C from being mis-scored as ordinary Phone Full work.
    std::function<bool(
        llama_region_ring_stage_kind, int, int, double &)> directional_ms;
    std::function<bool(
        llama_region_ring_stage_kind, int, int, double &)>
        directional_prepare_ms;
    std::function<bool(device, device, size_t, double &)> transfer_ms;
    // Exact weight bytes for one contiguous logical GPU Region and the
    // measured/proxied host->CUDA transfer time for those bytes.
    std::function<bool(int, int, size_t &)> gpu_region_weight_bytes;
    std::function<bool(size_t, double &)> gpu_weight_h2d_ms;
    std::function<bool(const candidate &)> memory_feasible;
    // Unlike memory_feasible(candidate), this callback accepts per-cycle
    // C/G boundaries. It is used ONLY by the real APPLY-safe search.
    std::function<bool(const std::vector<llama_region_ring_region> &)>
        region_memory_feasible;
};

inline int chunk_for(const candidate & c, device d) {
    return d == device::CPU ? c.xc : d == device::GPU ? c.xg : c.xp;
}

inline bool phone_registration_cost(
        const model & m, llama_region_ring_stage_kind kind,
        int layers, int tokens, bool embedded_directional,
        double & cost) {
    if (kind != llama_region_ring_stage_kind::PHONE &&
            !llama_region_ring_stage_is_directional(kind)) {
        cost = 0.0;
        return true;
    }
    if (m.phone_registration_cost) {
        if (!m.phone_registration_cost(
                kind, layers, tokens, embedded_directional, cost) ||
                !std::isfinite(cost) || cost < 0.0) {
            return false;
        }
        return true;
    }
    // Preserve the historical MoE/fixture model if no Dense probe exists.
    cost = m.phone_register_ms * (double) layers / 5.0;
    return std::isfinite(cost) && cost >= 0.0;
}

inline device region_input_owner(
        const llama_region_ring_region & region) {
    switch (region.kind) {
        case llama_region_ring_stage_kind::C2P:
            return device::CPU;
        case llama_region_ring_stage_kind::P2C:
            return device::PHONE;
        default:
            return region.device;
    }
}

inline device region_output_owner(
        const llama_region_ring_region & region) {
    switch (region.kind) {
        case llama_region_ring_stage_kind::C2P:
            return device::PHONE;
        case llama_region_ring_stage_kind::P2C:
            return device::CPU;
        default:
            return region.device;
    }
}

inline device region_queue_device(
        const llama_region_ring_region & region) {
    // Queue on the Attention/MAIN owner. The conservative simulator below
    // still reserves both CPU and Phone compute resources for Tensor work.
    return region.kind == llama_region_ring_stage_kind::C2P ?
        device::CPU :
        region.kind == llama_region_ring_stage_kind::P2C ?
            device::PHONE : region.device;
}

inline uint8_t region_resource_mask(
        const llama_region_ring_region & region) {
    if (llama_region_ring_stage_is_directional(region.kind)) {
        return (uint8_t) (
            (1u << (unsigned) device::CPU) |
            (1u << (unsigned) device::PHONE));
    }
    return (uint8_t) (1u << (unsigned) region.device);
}

inline int chunk_for_region(
        const candidate & c,
        const llama_region_ring_region & region) {
    // Current Region Ring executes directional Meta stages with XP.
    return llama_region_ring_stage_is_directional(region.kind) ?
        c.xp : chunk_for(c, region.device);
}

inline bool make_regions(const candidate & c, int total_layers,
                         std::vector<llama_region_ring_region> & regions) {
    regions.clear();
    if (!c.concrete_regions.empty()) {
        // The independent Stage widths in each Cycle are authoritative.
        // Reject invalid ownership/coverage instead of reconstructing a
        // repeated (G,C,P) geometry from the first Cycle.
        if (llama_region_ring_has_directional_stages(c.concrete_regions) ||
                !llama_region_ring_apply_topology_compatible(
                    c.concrete_regions, total_layers)) {
            return false;
        }
        regions = c.concrete_regions;
        return true;
    }
    if (c.cycles < 1 || c.c < 1 || c.g < 1 || c.p < 1 ||
        c.tail_c < 0 || c.tail_g < 0 ||
        c.cycles * (c.c + c.g + c.p) + c.tail_c + c.tail_g != total_layers) {
        return false;
    }
    int layer = 0;
    auto append = [&](device d, int n) {
        if (n > 0) {
            regions.push_back({d, layer, layer + n});
            layer += n;
        }
    };
    if (c.tail_c > 0 && c.tail_g <= 0) {
        return false;
    }
    for (int i = 0; i < c.cycles; ++i) {
        append(device::GPU, c.g);
        append(device::CPU, c.c);
        append(device::PHONE, c.p);
    }
    append(device::GPU, c.tail_g);
    append(device::CPU, c.tail_c);
    return layer == total_layers;
}

inline bool gpu_weight_bank_peaks(
        const model & m,
        const std::vector<llama_region_ring_region> & regions,
        std::array<size_t, 2> & peaks,
        size_t * total_bytes = nullptr) {
    peaks = {};
    if (total_bytes) {
        *total_bytes = 0;
    }
    if (!m.gpu_weight_streaming) {
        return true;
    }
    if (!m.gpu_region_weight_bytes) {
        return false;
    }

    size_t gpu_ordinal = 0;
    for (const auto & region : regions) {
        if (region.device != device::GPU) {
            continue;
        }
        size_t bytes = 0;
        if (!m.gpu_region_weight_bytes(
                region.layer_begin, region.layer_end, bytes) ||
                bytes == 0) {
            return false;
        }
        const size_t bank =
            llama_region_ring_gpu_weight_bank_for_rank(gpu_ordinal);
        peaks[bank] = std::max(peaks[bank], bytes);
        if (total_bytes) {
            if (bytes > std::numeric_limits<size_t>::max() - *total_bytes) {
                return false;
            }
            *total_bytes += bytes;
        }
        ++gpu_ordinal;
    }
    return true;
}

inline bool gpu_weight_h2d_total(
        const model & m,
        const std::vector<llama_region_ring_region> & regions,
        double & total_ms) {
    total_ms = 0.0;
    if (!m.gpu_weight_streaming) {
        return true;
    }
    if (!m.gpu_region_weight_bytes || !m.gpu_weight_h2d_ms) {
        return false;
    }
    size_t gpu_rank = 0;
    for (const auto & region : regions) {
        if (region.device != device::GPU) {
            continue;
        }
        // rank0/rank1 are primed during load/warmup and already resident when
        // prompt timing begins. Weight Streaming accounts only rolling
        // replacements from rank2 onward.
        if (gpu_rank++ < 2) {
            continue;
        }
        size_t bytes = 0;
        double ms = 0.0;
        if (!m.gpu_region_weight_bytes(
                region.layer_begin, region.layer_end, bytes) ||
                !m.gpu_weight_h2d_ms(bytes, ms) ||
                !std::isfinite(ms) || ms < 0.0) {
            return false;
        }
        total_ms += ms;
    }
    return std::isfinite(total_ms);
}

struct task {
    int region = 0;
    int begin = 0;
    int end = 0;
    int previous = -1;
    int next = -1;
    int inputs = 0;
    int outputs = 0;
    int queue_device = 0;
    uint8_t resource_mask = 0;
    // Number of downstream consumers that still retain this producer slot.
    int remaining_outputs = 0;
    double prepare_duration = 0.0;
    double duration = 0.0;
    bool queued = false;
    bool started = false;
    bool done = false;
};
struct edge {
    int producer = -1;
    int consumer = -1;
    int lane = 0; // 0=PC-local, 1=PC->Phone, 2=Phone->PC (when independent)
    double duration = 0.0;
    bool started = false;
};
enum class event_kind : uint8_t {
    TASK = 0,
    EDGE = 1,
    WEIGHT = 2,
};

struct event {
    double at = 0.0;
    int id = -1;
    event_kind kind = event_kind::TASK;
    bool operator<(const event & rhs) const {
        if (at != rhs.at) { return at > rhs.at; }
        if (kind != rhs.kind) {
            return (int) kind > (int) rhs.kind;
        }
        return id > rhs.id;
    }
};

// Simulate the current runtime contract: per-device FIFO queues preserve
// strict per-region KV order while boundary activations live in full-ubatch
// storage with no producer-credit limit. A non-zero slot_pool is retained
// only for bounded-slot regression tests. Boundary transfers are explicit
// link events.
inline bool simulate(const model & m,
                     const std::vector<llama_region_ring_region> & regions,
                     candidate & c, std::string * reason = nullptr) {
    auto reject = [&](const char * why) {
        if (reason) { *reason = why; }
        return false;
    };
    if (regions.empty() || m.tokens <= 0 || m.hidden <= 0 ||
        m.slot_pool < 0 || !m.compute_ms || !m.transfer_ms) {
        return reject("INVALID_MODEL");
    }
    if (m.gpu_weight_streaming &&
            (!m.gpu_region_weight_bytes || !m.gpu_weight_h2d_ms)) {
        return reject("WEIGHT_PROFILE_MISSING");
    }

    std::vector<task> tasks;
    std::vector<std::vector<int>> by_region(regions.size());
    for (size_t r = 0; r < regions.size(); ++r) {
        const auto & region = regions[r];
        const auto d = region.device;
        const bool directional =
            llama_region_ring_stage_is_directional(region.kind);
        const int x = chunk_for_region(c, region);
        if (x <= 0) { return reject("INVALID_CHUNK"); }
        if (directional &&
                (!m.directional_ms ||
                 !m.directional_prepare_ms)) {
            return reject("DIRECTIONAL_PROFILE_MISSING");
        }

        int previous = -1;
        for (int begin = 0; begin < m.tokens; begin += x) {
            const int end = std::min(m.tokens, begin + x);
            const int kv = m.kv_start + end;
            const int layers =
                region.layer_end - region.layer_begin;
            double ms = 0.0;
            double prepare_ms = 0.0;
            const bool compute_ok = directional ?
                m.directional_ms(
                    region.kind, end - begin, kv, ms) :
                m.compute_ms(
                    d, layers, end - begin, kv, ms);
            if (!compute_ok || !std::isfinite(ms) || ms < 0.0) {
                return reject(
                    directional ?
                        "DIRECTIONAL_PROFILE_MISSING" :
                        "COMPUTE_PROFILE_MISSING");
            }

            if (directional) {
                if (!m.directional_prepare_ms(
                        region.kind, end - begin, kv,
                        prepare_ms) ||
                        !std::isfinite(prepare_ms) ||
                        prepare_ms < 0.0) {
                    return reject(
                        "DIRECTIONAL_PREPARE_PROFILE_MISSING");
                }
            } else if (m.prepare_ms &&
                    (!m.prepare_ms(
                        d, layers, end - begin, kv, prepare_ms) ||
                     !std::isfinite(prepare_ms) ||
                     prepare_ms < 0.0)) {
                return reject("PREPARE_PROFILE_MISSING");
            }

            // Directional stages submit Phone-side work, but only a
            // C2P/Phone/P2C island uses the Dense Meta fusion graph.
            const bool embedded_directional =
                region.kind == llama_region_ring_stage_kind::PHONE &&
                r > 0 && r + 1 < regions.size() &&
                regions[r - 1].kind ==
                    llama_region_ring_stage_kind::C2P &&
                regions[r + 1].kind ==
                    llama_region_ring_stage_kind::P2C;
            double registration_ms = 0.0;
            if (!phone_registration_cost(
                    m, region.kind, layers, end - begin,
                    embedded_directional, registration_ms)) {
                return reject("PHONE_REGISTRATION_PROFILE_MISSING");
            }
            ms += registration_ms;

            task t;
            t.region = (int) r;
            t.begin = begin;
            t.end = end;
            t.previous = previous;
            t.queue_device = (int) region_queue_device(region);
            t.resource_mask = region_resource_mask(region);
            t.prepare_duration = prepare_ms;
            t.duration = ms;
            const int id = (int) tasks.size();
            tasks.push_back(t);
            if (previous >= 0) {
                tasks[(size_t) previous].next = id;
            }
            by_region[r].push_back(id);
            previous = id;
        }
    }

    std::vector<edge> edges;
    for (size_t r = 0; r + 1 < regions.size(); ++r) {
        const auto a = region_output_owner(regions[r]);
        const auto b = region_input_owner(regions[r + 1]);
        const int lane = a == device::PHONE ?
            (m.shared_phone_link ? 1 : 2) :
            b == device::PHONE ? 1 : 0;
        for (int src : by_region[r]) {
            for (int dst : by_region[r + 1]) {
                const int overlap = std::min(tasks[(size_t) src].end,
                                             tasks[(size_t) dst].end) -
                                    std::max(tasks[(size_t) src].begin,
                                             tasks[(size_t) dst].begin);
                if (overlap <= 0) { continue; }
                const size_t bytes =
                    (size_t) overlap * (size_t) m.hidden * sizeof(float);
                double ms = 0.0;
                // Same-owner adjacency is a device-local dependency. The
                // directional callback has already charged internal network
                // traffic, so never add another fake boundary transfer.
                if (a != b &&
                        (!m.transfer_ms(a, b, bytes, ms) ||
                         !std::isfinite(ms) || ms < 0.0)) {
                    return reject("TRANSFER_PROFILE_MISSING");
                }
                edges.push_back({src, dst, lane, ms});
                ++tasks[(size_t) src].outputs;
                ++tasks[(size_t) dst].inputs;
            }
        }
    }
    for (auto & t : tasks) {
        t.remaining_outputs = t.outputs;
    }

    std::vector<int> gpu_regions;
    std::vector<int> gpu_rank(regions.size(), -1);
    std::vector<double> gpu_weight_load_ms(regions.size(), 0.0);
    std::vector<double> gpu_weight_prepare_ms(regions.size(), 0.0);
    std::vector<bool> gpu_weight_ready(
        regions.size(), !m.gpu_weight_streaming);
    std::vector<bool> gpu_weight_enqueued(regions.size(), false);
    std::vector<double> gpu_weight_release_ms(
        regions.size(), -1.0);
    std::vector<double> gpu_weight_start_ms(
        regions.size(), -1.0);
    std::vector<double> gpu_weight_ready_ms(
        regions.size(), -1.0);
    std::vector<double> gpu_natural_ready_ms(
        regions.size(), -1.0);
    for (size_t r = 0; r < regions.size(); ++r) {
        if (regions[r].device != device::GPU) {
            continue;
        }
        gpu_rank[r] = (int) gpu_regions.size();
        gpu_regions.push_back((int) r);
        if (m.gpu_weight_streaming) {
            size_t bytes = 0;
            double ms = 0.0;
            if (!m.gpu_region_weight_bytes(
                    regions[r].layer_begin, regions[r].layer_end, bytes) ||
                    !m.gpu_weight_h2d_ms(bytes, ms) ||
                    !std::isfinite(ms) || ms < 0.0) {
                return reject("WEIGHT_PROFILE_MISSING");
            }
            gpu_weight_load_ms[r] = ms;
            if (m.prepare_ms) {
                const int layers =
                    regions[r].layer_end - regions[r].layer_begin;
                const int refill_tokens =
                    std::min(m.tokens, std::max(1, c.xg));
                const int refill_kv =
                    m.kv_start + refill_tokens;
                double prepare_ms = 0.0;
                if (!m.prepare_ms(
                        device::GPU, layers, refill_tokens,
                        refill_kv, prepare_ms) ||
                    !std::isfinite(prepare_ms) ||
                    prepare_ms < 0.0) {
                    return reject("WEIGHT_PREPARE_PROFILE_MISSING");
                }
                gpu_weight_prepare_ms[r] = prepare_ms;
            }
        }
    }

    std::priority_queue<event> events;
    std::array<std::deque<int>, 3> ready_queues;
    std::array<bool, 3> device_busy {}, link_busy {};
    bool pc_compute_gate_busy = false;
    int next_pc_device = 0;
    std::vector<int> live_slots(regions.size(), 0);
    std::vector<bool> region_done(regions.size(), false);
    std::array<double, 3> busy_ms {}, transfer_busy_ms {};
    std::deque<int> weight_queue;
    bool weight_busy = false;
    // A single shared PREPARE lane approximates region graph/split setup and
    // the single Bank mover's Region binding. The lane is
    // independent from CPU/CUDA/Phone compute.
    double prepare_available_ms = 0.0;
    double prepare_busy_ms = 0.0;
    double weight_prepare_busy_ms = 0.0;
    double weight_h2d_busy_ms = 0.0;
    double weight_h2d_startup_ms = 0.0;

    auto task_dependencies_ready = [&](int id) -> bool {
        if (id < 0 || id >= (int) tasks.size()) {
            return false;
        }
        const auto & t = tasks[(size_t) id];
        return !t.started && !t.done && t.inputs == 0 &&
               (t.previous < 0 || tasks[(size_t) t.previous].done);
    };
    auto task_ready = [&](int id) -> bool {
        return task_dependencies_ready(id) &&
               !tasks[(size_t) id].queued;
    };
    auto offer = [&](int id) {
        if (!task_ready(id)) {
            return;
        }
        auto & t = tasks[(size_t) id];
        t.queued = true;
        ready_queues[(size_t) t.queue_device].push_back(id);
    };
    auto claimable = [&](int id) -> bool {
        const auto & t = tasks[(size_t) id];
        const int r = t.region;

        if (regions[(size_t) r].device == device::GPU &&
                m.gpu_weight_streaming) {
            const int rank = gpu_rank[(size_t) r];
            if (rank < 0) {
                return false;
            }
            if (rank >= 2) {
                const int predecessor =
                    gpu_regions[(size_t) rank - 2];
                if (!region_done[(size_t) predecessor]) {
                    return false;
                }
            }
            if (!gpu_weight_ready[(size_t) r]) {
                return false;
            }
        }

        for (int dev = 0; dev < 3; ++dev) {
            if ((t.resource_mask & (1u << (unsigned) dev)) != 0 &&
                    device_busy[(size_t) dev]) {
                return false;
            }
        }
        if (m.pc_compute_gate &&
                (t.resource_mask &
                 ((1u << (unsigned) device::CPU) |
                  (1u << (unsigned) device::GPU))) != 0 &&
                pc_compute_gate_busy) {
            return false;
        }

        if (m.slot_pool == 0) {
            return true;
        }
        return !(r + 1 < (int) regions.size() &&
                 live_slots[(size_t) r] >= m.slot_pool);
    };

    if (by_region.empty() || by_region[0].empty()) {
        return reject("NO_INITIAL_TASK");
    }
    offer(by_region[0].front());

    // Load-time residency contract: rank0(A) and rank1(B) are already
    // prepared and resident before prompt timing begins. The simulator starts
    // at t=0 with both Banks ready; only rank2+ rolling replacements consume
    // PREPARE/H2D service during the prompt.
    double now = 0.0;
    if (m.gpu_weight_streaming) {
        const size_t initial =
            std::min<size_t>(2, gpu_regions.size());
        for (size_t i = 0; i < initial; ++i) {
            const int r = gpu_regions[i];
            gpu_weight_release_ms[(size_t) r] = 0.0;
            gpu_weight_start_ms[(size_t) r] = 0.0;
            gpu_weight_ready_ms[(size_t) r] = 0.0;
            gpu_weight_ready[(size_t) r] = true;
            gpu_weight_enqueued[(size_t) r] = true;
        }
    }

    int completed = 0;
    int max_live = 0;
    const int n_tasks = (int) tasks.size();

    while (completed < n_tasks) {
        bool scheduled = false;

        if (m.gpu_weight_streaming &&
                !weight_busy && !weight_queue.empty()) {
            const int r = weight_queue.front();
            weight_queue.pop_front();
            weight_busy = true;

            const double weight_prepare =
                gpu_weight_prepare_ms[(size_t) r];
            const double prepare_start =
                std::max(now, prepare_available_ms);
            const double prepare_end =
                prepare_start + weight_prepare;
            prepare_available_ms = prepare_end;
            prepare_busy_ms += weight_prepare;
            weight_prepare_busy_ms += weight_prepare;

            // H2D uses its own transfer backend/stream. The single dedicated
            // single Bank mover serializes refill jobs end-to-end.
            gpu_weight_start_ms[(size_t) r] = prepare_end;
            const double ms = gpu_weight_load_ms[(size_t) r];
            weight_h2d_busy_ms += ms;
            events.push({
                prepare_end + ms, r, event_kind::WEIGHT
            });
            scheduled = true;
        }

        // Producer completion makes its output transferable immediately.
        // Phone directions share one conservative Wi-Fi lane.
        for (size_t e = 0; e < edges.size(); ++e) {
            auto & it = edges[e];
            if (it.started || !tasks[(size_t) it.producer].done ||
                link_busy[(size_t) it.lane]) {
                continue;
            }
            it.started = true;
            link_busy[(size_t) it.lane] = true;
            transfer_busy_ms[(size_t) it.lane] += it.duration;
            events.push({now + it.duration, (int) e, event_kind::EDGE});
            scheduled = true;
        }

        // Record the first instant a GPU Region would be runnable if its
        // weights were already present. This is the profile-derived hiding
        // window endpoint used to decide whether a larger G is actually worth
        // its extra H2D traffic.
        if (m.gpu_weight_streaming &&
                !device_busy[(size_t) device::GPU]) {
            for (int r : gpu_regions) {
                if (gpu_natural_ready_ms[(size_t) r] >= 0.0 ||
                        by_region[(size_t) r].empty()) {
                    continue;
                }
                const int rank = gpu_rank[(size_t) r];
                bool bank_reusable = rank >= 0;
                if (bank_reusable && rank >= 2) {
                    const int predecessor =
                        gpu_regions[(size_t) rank - 2];
                    bank_reusable =
                        region_done[(size_t) predecessor];
                }
                const int first = by_region[(size_t) r].front();
                if (bank_reusable &&
                        task_dependencies_ready(first)) {
                    gpu_natural_ready_ms[(size_t) r] = now;
                }
            }
        }

        // Match llama_region_ring_ready::take_device_locked(): each device
        // owns a FIFO, but may skip an older READY task whose output edge has
        // no free slot. CPU and CUDA are independent unless the historical
        // gated model is explicitly requested.
        for (int turn = 0; turn < 3; ++turn) {
            const int dev = m.pc_compute_gate && turn < 2 ?
                (turn == 0 ? next_pc_device : 1 - next_pc_device) : turn;
            auto & q = ready_queues[(size_t) dev];
            auto it = std::find_if(q.begin(), q.end(),
                [&](int id) { return claimable(id); });
            if (it == q.end()) {
                continue;
            }

            const int selected = *it;
            q.erase(it);
            auto & t = tasks[(size_t) selected];
            t.queued = false;
            t.started = true;
            for (int resource = 0; resource < 3; ++resource) {
                if ((t.resource_mask &
                        (1u << (unsigned) resource)) != 0) {
                    device_busy[(size_t) resource] = true;
                }
            }
            const bool uses_pc_compute =
                (t.resource_mask &
                 ((1u << (unsigned) device::CPU) |
                  (1u << (unsigned) device::GPU))) != 0;
            if (m.pc_compute_gate && uses_pc_compute) {
                pc_compute_gate_busy = true;
                if (dev == (int) device::CPU ||
                        dev == (int) device::GPU) {
                    next_pc_device = 1 - dev;
                }
            }

            if (m.slot_pool > 0 &&
                    t.region + 1 < (int) regions.size()) {
                ++live_slots[(size_t) t.region];
                max_live = std::max(max_live,
                                    live_slots[(size_t) t.region]);
                if (live_slots[(size_t) t.region] > m.slot_pool) {
                    return reject("SLOT_POOL_OVERFLOW");
                }
            }

            const double prepare_start =
                std::max(now, prepare_available_ms);
            const double prepare_end =
                prepare_start + t.prepare_duration;
            prepare_available_ms = prepare_end;
            prepare_busy_ms += t.prepare_duration;

            // The worker has already claimed this task, so it cannot claim
            // another while queued on PREPARE. CPU/CUDA/Phone compute lanes
            // remain independent once preparation completes.
            for (int resource = 0; resource < 3; ++resource) {
                if ((t.resource_mask &
                        (1u << (unsigned) resource)) != 0) {
                    busy_ms[(size_t) resource] += t.duration;
                }
            }
            events.push({
                prepare_end + t.duration,
                selected,
                event_kind::TASK
            });
            scheduled = true;
        }

        if (events.empty()) {
            return reject(scheduled ? "INTERNAL_EVENT_ERROR" :
                                      "SLOT_OR_KV_DEADLOCK");
        }

        now = events.top().at;
        if (!std::isfinite(now)) {
            return reject("TIME_NONFINITE");
        }

        // Process every completion at this timestamp before claiming more work.
        while (!events.empty() && events.top().at <= now) {
            const event current = events.top();
            events.pop();

            if (current.kind == event_kind::EDGE) {
                const auto & e = edges[(size_t) current.id];
                link_busy[(size_t) e.lane] = false;
                auto & consumer = tasks[(size_t) e.consumer];
                if (--consumer.inputs < 0) {
                    return reject("EDGE_COUNT_UNDERFLOW");
                }
                offer(e.consumer);
                continue;
            }

            if (current.kind == event_kind::WEIGHT) {
                const int r = current.id;
                if (r < 0 || r >= (int) regions.size() ||
                        regions[(size_t) r].device != device::GPU) {
                    return reject("WEIGHT_EVENT_INVALID");
                }
                gpu_weight_ready[(size_t) r] = true;
                gpu_weight_ready_ms[(size_t) r] = now;
                weight_busy = false;
                continue;
            }

            auto & t = tasks[(size_t) current.id];
            if (t.done) {
                return reject("DUPLICATE_TASK_DONE");
            }
            t.done = true;
            for (int resource = 0; resource < 3; ++resource) {
                if ((t.resource_mask &
                        (1u << (unsigned) resource)) != 0) {
                    device_busy[(size_t) resource] = false;
                }
            }
            if (m.pc_compute_gate &&
                    (t.resource_mask &
                     ((1u << (unsigned) device::CPU) |
                      (1u << (unsigned) device::GPU))) != 0) {
                pc_compute_gate_busy = false;
            }
            ++completed;

            if (t.next < 0) {
                region_done[(size_t) t.region] = true;
                if (m.gpu_weight_streaming &&
                        regions[(size_t) t.region].device == device::GPU) {
                    const int rank = gpu_rank[(size_t) t.region];
                    if (rank < 0) {
                        return reject("GPU_RANK_MISSING");
                    }
                    const auto next_rank =
                        llama_region_ring_gpu_weight_next_rank(
                            (size_t) rank, gpu_regions.size());
                    if (next_rank.has_value()) {
                        const int next_region =
                            gpu_regions[*next_rank];
                        if (!gpu_weight_enqueued[
                                (size_t) next_region]) {
                            gpu_weight_enqueued[
                                (size_t) next_region] = true;
                            gpu_weight_release_ms[
                                (size_t) next_region] = now;
                            weight_queue.push_back(next_region);
                        }
                    }
                }
            }

            // Bounded-slot accounting remains only for regression tests.
            // Production Region Ring uses slot_pool=0 and full per-boundary
            // activation storage, so producer completion never consumes a
            // finite output credit.
            if (m.slot_pool > 0) {
                for (const auto & e : edges) {
                    if (e.consumer != current.id) {
                        continue;
                    }
                    auto & producer = tasks[(size_t) e.producer];
                    if (--producer.remaining_outputs < 0) {
                        return reject("CONSUMER_COUNT_UNDERFLOW");
                    }
                    if (producer.remaining_outputs == 0) {
                        const int r = producer.region;
                        if (r + 1 >= (int) regions.size() ||
                            --live_slots[(size_t) r] < 0) {
                            return reject("SLOT_RELEASE_UNDERFLOW");
                        }
                    }
                }
            }

            offer(t.next);
        }
    }

    if (m.slot_pool > 0 &&
            std::any_of(live_slots.begin(), live_slots.end(),
                [](int x) { return x != 0; })) {
        return reject("LIVE_SLOTS_NOT_DRAINED");
    }
    if (m.pc_compute_gate &&
        busy_ms[(int) device::CPU] + busy_ms[(int) device::GPU] >
            now + std::max(1e-6, 1e-9 * now)) {
        return reject("PC_GATE_OVERCOMMITTED");
    }

    c.wall_ms = now;
    c.pc_gate_busy_ms = m.pc_compute_gate ?
        busy_ms[(int) device::CPU] + busy_ms[(int) device::GPU] : 0.0;
    c.pc_compute_bound_ms = m.pc_compute_gate ?
        busy_ms[(int) device::CPU] + busy_ms[(int) device::GPU] :
        std::max(busy_ms[(int) device::CPU],
                 busy_ms[(int) device::GPU]);
    c.busy_ms = busy_ms;
    c.transfer_busy_ms = transfer_busy_ms;
    c.prepare_busy_ms = prepare_busy_ms;
    c.weight_prepare_busy_ms = weight_prepare_busy_ms;
    c.weight_h2d_busy_ms = weight_h2d_busy_ms;
    c.weight_h2d_startup_ms = weight_h2d_startup_ms;
    c.weight_h2d_exposed_ms = weight_h2d_startup_ms;
    c.weight_h2d_hidden_ms = 0.0;
    if (m.gpu_weight_streaming) {
        const size_t initial =
            std::min<size_t>(2, gpu_regions.size());
        for (size_t rank = initial;
                rank < gpu_regions.size(); ++rank) {
            const int r = gpu_regions[rank];
            const double start =
                gpu_weight_start_ms[(size_t) r];
            const double done =
                gpu_weight_ready_ms[(size_t) r];
            const double natural =
                gpu_natural_ready_ms[(size_t) r];
            const double duration =
                gpu_weight_load_ms[(size_t) r];
            if (start < 0.0 || done < 0.0 ||
                    natural < 0.0 || duration < 0.0) {
                return reject("WEIGHT_TIMELINE_INCOMPLETE");
            }

            // Actual transfer overlap, bounded by the transfer duration.
            c.weight_h2d_hidden_ms += std::min(
                duration,
                std::max(0.0, natural - start));

            // GPU stall due to weight availability. This may include queueing
            // behind an earlier Bank upload on the single H2D lane.
            c.weight_h2d_exposed_ms +=
                std::max(0.0, done - natural);
        }
    }
    for (size_t i = 0; i < c.service_ms_per_token.size(); ++i) {
        c.service_ms_per_token[i] =
            busy_ms[i] / std::max(1, m.tokens);
    }
    c.pc_local_link_ms_per_token =
        transfer_busy_ms[0] / std::max(1, m.tokens);
    c.phone_link_ms_per_token =
        (transfer_busy_ms[1] + transfer_busy_ms[2]) /
        std::max(1, m.tokens);
    c.tasks = n_tasks;
    c.edges = (int) edges.size();
    c.max_live_slots = max_live;
    return true;
}


struct boundary_anchor {
    int layer = -1;
    llama_region_ring_stage_kind first =
        llama_region_ring_stage_kind::C2P;
    int left_begin = -1;
    int left_end = -1;
    int right_begin = -1;
    int right_end = -1;
};

struct boundary_choice {
    std::vector<llama_region_ring_region> regions;
    candidate estimate;
    std::vector<int> widths;
    int tensor_layers = 0;
};

struct boundary_result {
    bool valid = false;
    bool baseline_valid = false;
    boundary_choice baseline;
    boundary_choice selected;
    std::vector<boundary_choice> top;
    size_t boundaries = 0;
    size_t enumerated = 0;
    size_t scored = 0;
    size_t rejected_shape = 0;
    size_t rejected_memory = 0;
    size_t rejected_profile = 0;
};

inline std::vector<boundary_anchor> boundary_anchors(
        const std::vector<llama_region_ring_region> & baseline) {
    std::vector<boundary_anchor> result;
    for (size_t i = 0; i + 1 < baseline.size(); ++i) {
        const auto & left = baseline[i];
        const auto & right = baseline[i + 1];
        if (llama_region_ring_stage_is_directional(left.kind) ||
                llama_region_ring_stage_is_directional(right.kind)) {
            return {};
        }

        llama_region_ring_stage_kind first;
        if (left.kind == llama_region_ring_stage_kind::CPU &&
                right.kind == llama_region_ring_stage_kind::PHONE) {
            first = llama_region_ring_stage_kind::C2P;
        } else if (left.kind == llama_region_ring_stage_kind::PHONE &&
                right.kind == llama_region_ring_stage_kind::GPU) {
            first = llama_region_ring_stage_kind::P2C;
        } else {
            continue;
        }

        result.push_back({
            left.layer_end,
            first,
            left.layer_begin,
            left.layer_end,
            right.layer_begin,
            right.layer_end,
        });
    }
    return result;
}

inline std::vector<int> boundary_legal_widths(
        const boundary_anchor & b) {
    std::vector<int> widths {0}; // NO_TENSOR is always a candidate.
    const int left_layers = b.left_end - b.left_begin;
    const int right_layers = b.right_end - b.right_begin;
    for (int k = 0;
            left_layers - (k + 1) >= 1 &&
            right_layers - k >= 1;
            ++k) {
        widths.push_back(1 + 2 * k);
    }
    return widths;
}

inline bool build_boundary_regions(
        const std::vector<llama_region_ring_region> & baseline,
        const std::vector<boundary_anchor> & anchors,
        const std::vector<int> & widths,
        int n_layer,
        std::vector<llama_region_ring_region> & regions) {
    regions.clear();
    if (n_layer <= 0 || baseline.empty() ||
            anchors.size() != widths.size()) {
        return false;
    }

    std::vector<llama_region_ring_stage_kind> kinds(
        (size_t) n_layer, llama_region_ring_stage_kind::CPU);
    std::vector<uint8_t> overwritten((size_t) n_layer, 0);

    int next = 0;
    for (const auto & r : baseline) {
        if (llama_region_ring_stage_is_directional(r.kind) ||
                r.layer_begin != next ||
                r.layer_end <= r.layer_begin ||
                r.layer_end > n_layer) {
            return false;
        }
        for (int il = r.layer_begin; il < r.layer_end; ++il) {
            kinds[(size_t) il] = r.kind;
        }
        next = r.layer_end;
    }
    if (next != n_layer) {
        return false;
    }

    for (size_t bi = 0; bi < anchors.size(); ++bi) {
        const int width = widths[bi];
        if (width == 0) {
            continue;
        }
        if (width < 0 || (width & 1) == 0) {
            return false;
        }
        const int k = (width - 1) / 2;
        const auto & b = anchors[bi];
        const int begin = b.layer - (k + 1);
        const int end = b.layer + k;
        if (begin < b.left_begin || end > b.right_end ||
                begin < 0 || end > n_layer ||
                end - begin != width) {
            return false;
        }

        auto kind = b.first;
        for (int il = begin; il < end; ++il) {
            if (overwritten[(size_t) il] != 0) {
                return false;
            }
            overwritten[(size_t) il] = 1;
            kinds[(size_t) il] = kind;
            kind = kind == llama_region_ring_stage_kind::C2P ?
                llama_region_ring_stage_kind::P2C :
                llama_region_ring_stage_kind::C2P;
        }
    }

    int begin = 0;
    while (begin < n_layer) {
        const auto kind = kinds[(size_t) begin];
        int end = begin + 1;
        // Directional layers intentionally remain one layer each; Full
        // regions may be compressed back into one contiguous stage.
        if (!llama_region_ring_stage_is_directional(kind)) {
            while (end < n_layer &&
                    kinds[(size_t) end] == kind) {
                ++end;
            }
        }
        regions.emplace_back(kind, begin, end);
        begin = end;
    }

    return llama_region_ring_directional_topology_compatible(
        regions, n_layer);
}

inline int boundary_tensor_layers(const std::vector<int> & widths) {
    int total = 0;
    for (int width : widths) {
        total += std::max(0, width);
    }
    return total;
}

inline boundary_result refine_optional_boundaries(
        const model & m,
        const std::vector<llama_region_ring_region> & baseline_regions,
        const candidate & baseline_candidate,
        double improve_epsilon_ms = 0.05) {
    boundary_result result;
    const auto anchors = boundary_anchors(baseline_regions);
    result.boundaries = anchors.size();

    auto score = [&](const std::vector<int> & widths,
                     boundary_choice & choice) -> bool {
        ++result.enumerated;
        choice = {};
        choice.widths = widths;
        choice.tensor_layers = boundary_tensor_layers(widths);
        if (!build_boundary_regions(
                baseline_regions, anchors, widths, m.layers,
                choice.regions)) {
            ++result.rejected_shape;
            return false;
        }
        if (m.region_memory_feasible &&
                !m.region_memory_feasible(choice.regions)) {
            ++result.rejected_memory;
            return false;
        }
        choice.estimate = baseline_candidate;
        std::string reason;
        if (!simulate(m, choice.regions, choice.estimate, &reason)) {
            ++result.rejected_profile;
            return false;
        }
        ++result.scored;
        return true;
    };

    std::vector<int> current_widths(anchors.size(), 0);
    if (!score(current_widths, result.baseline)) {
        return result;
    }
    result.baseline_valid = true;
    result.selected = result.baseline;

    std::vector<boundary_choice> seen;
    seen.push_back(result.baseline);
    auto record = [&](const boundary_choice & choice) {
        const std::string layout =
            llama_region_ring_layout_string(choice.regions);
        const auto it = std::find_if(
            seen.begin(), seen.end(),
            [&](const boundary_choice & x) {
                return llama_region_ring_layout_string(x.regions) ==
                    layout;
            });
        if (it == seen.end()) {
            seen.push_back(choice);
        }
    };

    // Coordinate search keeps this second-stage planner deliberately local.
    // Each update must improve the already selected Cycle plan, and a second
    // pass captures interactions between neighboring entry/exit boundaries.
    for (int pass = 0; pass < 2; ++pass) {
        bool changed = false;
        for (size_t bi = 0; bi < anchors.size(); ++bi) {
            boundary_choice local_best = result.selected;
            std::vector<int> local_widths = current_widths;

            for (int width : boundary_legal_widths(anchors[bi])) {
                if (width == current_widths[bi]) {
                    continue;
                }
                auto probe_widths = current_widths;
                probe_widths[bi] = width;
                boundary_choice probe;
                if (!score(probe_widths, probe)) {
                    continue;
                }
                record(probe);

                const bool faster =
                    probe.estimate.wall_ms <
                        local_best.estimate.wall_ms -
                        improve_epsilon_ms;
                const bool tie_fewer_tensor =
                    std::fabs(
                        probe.estimate.wall_ms -
                        local_best.estimate.wall_ms) <=
                        improve_epsilon_ms &&
                    probe.tensor_layers <
                        local_best.tensor_layers;
                if (faster || tie_fewer_tensor) {
                    local_best = probe;
                    local_widths = probe_widths;
                }
            }

            if (local_widths != current_widths) {
                current_widths = local_widths;
                result.selected = local_best;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }

    std::sort(
        seen.begin(), seen.end(),
        [](const boundary_choice & a, const boundary_choice & b) {
            if (std::fabs(
                    a.estimate.wall_ms -
                    b.estimate.wall_ms) > 1e-9) {
                return a.estimate.wall_ms < b.estimate.wall_ms;
            }
            return a.tensor_layers < b.tensor_layers;
        });
    if (seen.size() > 10) {
        seen.resize(10);
    }
    result.top = std::move(seen);
    result.valid = result.baseline_valid;
    return result;
}

inline double relative_ring_mismatch(double lhs, double rhs) {
    const double denom = std::max(lhs, rhs);
    if (!(denom > 0.0) || !std::isfinite(denom)) {
        return std::numeric_limits<double>::infinity();
    }
    return std::abs(lhs - rhs) / denom;
}

// Minimax load imbalance of the first N cycles. The accumulated service
// vector contains real per-Chunk CPU/GPU/Phone profiled times divided by XC,
// XG/XP, including the Phone RPC graph cost. Link times are per-token
// G->C, C->P, P->G measurements. Multiplication by N preserves the existing
// cyclic E1 equation for the first cycle. The exact final boundary count,
// KV dependencies, PREPARE and H2D belong to the full Prompt simulator.
inline double prefix_cycle_max_error(
        const std::array<double, 3> & work,
        const std::array<double, 3> & links,
        int n_cycles) {
    if (n_cycles <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    const double scale = (double) n_cycles;
    const double A = work[(size_t) device::CPU];
    const double B = work[(size_t) device::GPU];
    const double D = work[(size_t) device::PHONE];
    const double e1 = relative_ring_mismatch(
        B, D + scale * links[0]); // P -> G
    const double e2 = relative_ring_mismatch(
        A, B + scale * links[1]); // G -> C
    const double e3 = relative_ring_mismatch(
        D, A + scale * links[2]); // C -> P
    return std::max({e1, e2, e3});
}

// Evaluate the same local ring equations for a concrete runtime layout whose
// three cycles need not have identical C/G widths (the fixed baseline ends in
// G3,C8,P5 rather than G4,C7,P5). A/B/D are the mean per-cycle station
// service rates; aggregate service demand is kept separately.
// Cost the *actual intersecting token intervals* at a device boundary.
// This matches simulate()'s per-overlap EDGE events, including fixed
// transfer/setup costs when two independently chosen Chunk grids misalign.
// The producer's full Chunk must finish before any intersecting edge starts.
// The full Prompt simulator still owns the final ordering/overlap decision.
inline bool region_ring_handoff_ms_per_token(
        const model & m, device source, device target,
        int source_chunk, int target_chunk, double & per_token_ms) {
    per_token_ms = 0.0;
    if (m.tokens <= 0 || m.hidden <= 0 ||
        source_chunk <= 0 || target_chunk <= 0 ||
        !m.transfer_ms) {
        return false;
    }
    double total_ms = 0.0;
    int source_begin = 0;
    int target_begin = 0;
    while (source_begin < m.tokens && target_begin < m.tokens) {
        const int source_end =
            source_begin + std::min(source_chunk, m.tokens - source_begin);
        const int target_end =
            target_begin + std::min(target_chunk, m.tokens - target_begin);
        const int overlap =
            std::min(source_end, target_end) -
            std::max(source_begin, target_begin);
        if (overlap > 0) {
            double edge_ms = 0.0;
            const size_t bytes = (size_t) overlap *
                (size_t) m.hidden * sizeof(float);
            if (!m.transfer_ms(source, target, bytes, edge_ms) ||
                !std::isfinite(edge_ms) || edge_ms < 0.0) {
                return false;
            }
            total_ms += edge_ms;
        }
        if (source_end <= target_end) {
            source_begin = source_end;
        }
        if (target_end <= source_end) {
            target_begin = target_end;
        }
    }
    per_token_ms = total_ms / m.tokens;
    return std::isfinite(per_token_ms) && per_token_ms >= 0.0;
}

inline bool estimate_region_ring_balance(
        const model & m,
        const std::vector<llama_region_ring_region> & regions,
        candidate & c) {
    if (!m.compute_ms || !m.transfer_ms || m.hidden <= 0 ||
        regions.empty() || c.xc <= 0 || c.xg <= 0 || c.xp <= 0) {
        return false;
    }

    const int kv = m.kv_start + m.tokens / 2;
    std::array<double, 3> stage_sum {};
    std::array<int, 3> stage_count {};
    for (size_t ri = 0; ri < regions.size(); ++ri) {
        const auto & region = regions[ri];
        const int d = (int) region.device;
        if (d < 0 || d >= 3) {
            return false;
        }
        const int x = chunk_for(c, region.device);
        const int layers = region.layer_end - region.layer_begin;
        double ms = 0.0;
        if (layers <= 0 ||
            !m.compute_ms(region.device, layers, x, kv, ms) ||
            !std::isfinite(ms) || ms < 0.0) {
            return false;
        }
        const bool embedded_directional =
            region.kind == llama_region_ring_stage_kind::PHONE &&
            ri > 0 && ri + 1 < regions.size() &&
            regions[ri - 1].kind ==
                llama_region_ring_stage_kind::C2P &&
            regions[ri + 1].kind ==
                llama_region_ring_stage_kind::P2C;
        double registration_ms = 0.0;
        if (!phone_registration_cost(
                m, region.kind, layers, x, embedded_directional,
                registration_ms)) {
            return false;
        }
        ms += registration_ms;
        stage_sum[(size_t) d] += ms / x;
        ++stage_count[(size_t) d];
    }
    if (stage_count[0] <= 0 || stage_count[1] <= 0 ||
        stage_count[2] <= 0) {
        return false;
    }

    const double A = stage_sum[0] / stage_count[0];
    const double B = stage_sum[1] / stage_count[1];
    const double D = stage_sum[2] / stage_count[2];

    double TP2G = 0.0;
    double TC2P = 0.0;
    double TG2C = 0.0;
    // E uses the same interval-overlap transfers as the full DAG simulator.
    // Non-multiple XC/XG/XP can create several boundary edges per Chunk.
    if (!region_ring_handoff_ms_per_token(
            m, device::PHONE, device::GPU, c.xp, c.xg, TP2G) ||
        !region_ring_handoff_ms_per_token(
            m, device::CPU, device::PHONE, c.xc, c.xp, TC2P) ||
        !region_ring_handoff_ms_per_token(
            m, device::GPU, device::CPU, c.xg, c.xc, TG2C)) {
        return false;
    }

    const double e1 = relative_ring_mismatch(B, D + TP2G);
    const double e2 = relative_ring_mismatch(A, B + TG2C);
    const double e3 = relative_ring_mismatch(D, A + TC2P);
    if (!std::isfinite(e1) || !std::isfinite(e2) ||
        !std::isfinite(e3)) {
        return false;
    }

    c.ring_stage_ms_per_token = {A, B, D};
    c.ring_p2c_ms_per_token = TP2G;
    c.ring_c2p_ms_per_token = TC2P;
    c.ring_error = {e1, e2, e3};
    c.ring_error_max = std::max({e1, e2, e3});
    c.ring_error_mean = (e1 + e2 + e3) / 3.0;
    c.balance = c.ring_error_max;
    if (!gpu_weight_h2d_total(m, regions, c.weight_h2d_busy_ms)) {
        return false;
    }
    return true;
}

inline bool estimate_balance(const model & m, candidate & c) {
    if (!m.compute_ms || !m.transfer_ms || m.hidden <= 0 ||
        c.c <= 0 || c.g <= 0 || c.p <= 0 ||
        c.xc <= 0 || c.xg <= 0 || c.xp <= 0) {
        return false;
    }

    const int kv = m.kv_start + m.tokens / 2;
    double tc = 0.0, tg = 0.0, tp = 0.0;
    if (!m.compute_ms(device::CPU, c.c, c.xc, kv, tc) ||
        !m.compute_ms(device::GPU, c.g, c.xg, kv, tg) ||
        !m.compute_ms(device::PHONE, c.p, c.xp, kv, tp)) {
        return false;
    }
    double registration_ms = 0.0;
    if (!phone_registration_cost(
            m, llama_region_ring_stage_kind::PHONE,
            c.p, c.xp, false, registration_ms)) {
        return false;
    }
    tp += registration_ms;

    // Same E equations as before; the link terms now account for the
    // producer/consumer interval intersections, including non-multiple X.
    double TP2G = 0.0;
    double TC2P = 0.0;
    double TLOCAL = 0.0;
    if (!region_ring_handoff_ms_per_token(
            m, device::PHONE, device::GPU, c.xp, c.xg, TP2G) ||
        !region_ring_handoff_ms_per_token(
            m, device::CPU, device::PHONE, c.xc, c.xp, TC2P) ||
        !region_ring_handoff_ms_per_token(
            m, device::GPU, device::CPU, c.xg, c.xc, TLOCAL)) {
        return false;
    }

    const double A = tc / c.xc;
    const double B = tg / c.xg;
    const double D = tp / c.xp;
    if (!(A > 0.0) || !(B > 0.0) || !(D > 0.0) ||
        TP2G < 0.0 || TC2P < 0.0 || TLOCAL < 0.0 ||
        !std::isfinite(A) || !std::isfinite(B) || !std::isfinite(D) ||
        !std::isfinite(TP2G) || !std::isfinite(TC2P) ||
        !std::isfinite(TLOCAL)) {
        return false;
    }

    const double e1 = relative_ring_mismatch(B, D + TP2G);
    const double e2 = relative_ring_mismatch(A, B + TLOCAL);
    const double e3 = relative_ring_mismatch(D, A + TC2P);
    if (!std::isfinite(e1) || !std::isfinite(e2) ||
        !std::isfinite(e3)) {
        return false;
    }

    c.ring_stage_ms_per_token = {A, B, D};
    c.ring_p2c_ms_per_token = TP2G;
    c.ring_c2p_ms_per_token = TC2P;
    c.ring_error = {e1, e2, e3};
    c.ring_error_max = std::max({e1, e2, e3});
    c.ring_error_mean = (e1 + e2 + e3) / 3.0;
    // Keep the existing field as the public coarse-balance key.
    c.balance = c.ring_error_max;

    const double cpu_layer_per_token = A / std::max(c.c, 1);
    const double gpu_layer_per_token = B / std::max(c.g, 1);
    const double cpu_total =
        c.cycles * A + c.tail_c * cpu_layer_per_token;
    const double gpu_total =
        c.cycles * B + c.tail_g * gpu_layer_per_token;
    const double phone_total = c.cycles * D;

    const bool has_tail = c.tail_c + c.tail_g > 0;
    const int phone_to_pc_edges =
        std::max(0, c.cycles - 1) + (has_tail ? 1 : 0);
    const int pc_to_phone_edges = c.cycles;
    const int pc_local_edges =
        c.cycles + (c.tail_c > 0 && c.tail_g > 0 ? 1 : 0);
    const double phone_link_total =
        pc_to_phone_edges * TC2P + phone_to_pc_edges * TP2G;
    const double local_link_total = pc_local_edges * TLOCAL;

    c.service_ms_per_token = {cpu_total, gpu_total, phone_total};
    c.phone_link_ms_per_token = phone_link_total;
    c.pc_local_link_ms_per_token = local_link_total;

    std::vector<llama_region_ring_region> coarse_regions;
    if (!make_regions(c, m.layers, coarse_regions) ||
            !gpu_weight_h2d_total(
                m, coarse_regions, c.weight_h2d_busy_ms)) {
        return false;
    }

    if (m.pc_compute_gate) {
        // Historical compatibility path only. The current runtime does not
        // serialize CPU and CUDA compute, but retain a meaningful lower bound
        // for gated-regression tests.
        const double pc_total = cpu_total + gpu_total;
        c.coarse_ms = std::max(
            m.tokens *
                std::max({pc_total, phone_total,
                          phone_link_total, local_link_total}),
            c.weight_h2d_busy_ms);
    } else {
        // Coarse time is only a throughput lower bound. Candidate ordering for
        // large searches is driven first by the user's minimax ring mismatch
        // E=max(e1,e2,e3); exact FIFO simulation remains the final objective.
        c.coarse_ms = std::max(
            m.tokens *
                std::max({cpu_total, gpu_total, phone_total,
                          phone_link_total, local_link_total}),
            c.weight_h2d_busy_ms);
    }

    return std::isfinite(c.coarse_ms) &&
           std::isfinite(c.balance);
}

struct result {
    bool valid = false;
    candidate selected;
    candidate baseline;
    bool baseline_valid = false;
    size_t enumerated = 0;
    size_t profiled = 0;
    size_t simulated = 0;
    size_t rejected_memory = 0;
    size_t rejected_profile = 0;
    size_t rejected_deadlock = 0;
    size_t shortlisted = 0;
    size_t e_window_shortlisted = 0;
    size_t nonuniform_profiled = 0;
    size_t nonuniform_simulated = 0;
    size_t first_cycle_profiled = 0;
    size_t first_cycle_selected = 0;
    size_t prefix_expanded = 0;
    size_t prefix_completed = 0;
    double first_cycle_e_min = std::numeric_limits<double>::infinity();
    double coarse_e_min = std::numeric_limits<double>::infinity();
    double coarse_e_limit = std::numeric_limits<double>::infinity();
    double shortlist_e_limit = std::numeric_limits<double>::infinity();
    std::vector<candidate> top;
};

inline result search(
        const model & m,
        int max_simulations = 256,
        double e_margin = 0.08,
        int min_simulations = 32) {
    result out;
    if (m.layers < 6 || m.tokens <= 0 || max_simulations <= 0 ||
        min_simulations < 0 || !std::isfinite(e_margin) ||
        e_margin < 0.0) {
        return out;
    }

    const std::array<int,4> x_cpu {16,32,64,128};
    const std::array<int,4> x_gpu {16,32,64,128};
    const std::array<int,4> x_phone {16,32,64,128};
    std::vector<candidate> coarse;

    for (int n = 2; n <= 4; ++n) {
        const int max_region_width = m.layers / n;
        for (int p = 2; p <= std::min(8, max_region_width - 2); ++p) {
            for (int c = 1; c <= max_region_width - p - 1; ++c) {
                for (int g = 1; g <= max_region_width - p - c; ++g) {
                    const int tail = m.layers - n * (c + g + p);
                    if (tail < 0 || tail > m.max_tail) {
                        continue;
                    }
                    for (int tg = 0; tg <= tail; ++tg) {
                        const int tc = tail - tg;
                        if (tc > 0 && tg == 0) {
                            continue;
                        }
                        for (int xc : x_cpu) for (int xg : x_gpu) for (int xp : x_phone) {
                            candidate plan;
                            plan.cycles = n;
                            plan.c = c;
                            plan.g = g;
                            plan.p = p;
                            plan.tail_c = tc;
                            plan.tail_g = tg;
                            plan.xc = xc;
                            plan.xg = xg;
                            plan.xp = xp;
                            ++out.enumerated;
                            if (m.memory_feasible && !m.memory_feasible(plan)) {
                                ++out.rejected_memory;
                                continue;
                            }
                            if (!estimate_balance(m, plan)) {
                                ++out.rejected_profile;
                                continue;
                            }
                            ++out.profiled;
                            coarse.push_back(std::move(plan));
                        }
                    }
                }
            }
        }
    }

    if (coarse.empty()) {
        return out;
    }

    std::sort(coarse.begin(), coarse.end(),
        [](const candidate & a, const candidate & b) {
            if (std::abs(a.ring_error_max - b.ring_error_max) > 1e-12) {
                return a.ring_error_max < b.ring_error_max;
            }
            if (std::abs(a.coarse_ms - b.coarse_ms) > 1e-9) {
                return a.coarse_ms < b.coarse_ms;
            }
            return a.ring_error_mean < b.ring_error_mean;
        });

    out.coarse_e_min = coarse.front().ring_error_max;
    out.coarse_e_limit = std::min(1.0, out.coarse_e_min + e_margin);

    std::vector<candidate> shortlist;
    shortlist.reserve((size_t) max_simulations);
    const auto exact_key = [](const candidate & c) {
        return std::make_tuple(c.cycles, c.c, c.g, c.p,
                               c.tail_c, c.tail_g, c.xc, c.xg, c.xp);
    };
    const auto family_key = [](const candidate & c) {
        return std::make_tuple(c.cycles, c.c, c.g, c.p,
                               c.tail_c, c.tail_g);
    };

    std::vector<std::tuple<int,int,int,int,int,int>> families;
    for (const auto & plan : coarse) {
        if (plan.ring_error_max > out.coarse_e_limit + 1e-12) break;
        const auto family = family_key(plan);
        if (std::find(families.begin(), families.end(), family) != families.end()) continue;
        families.push_back(family);
        shortlist.push_back(plan);
        if ((int) shortlist.size() >= max_simulations / 2) break;
    }

    for (const auto & plan : coarse) {
        if ((int) shortlist.size() >= max_simulations) break;
        if (plan.ring_error_max > out.coarse_e_limit + 1e-12) break;
        if (std::any_of(shortlist.begin(), shortlist.end(),
                [&](const candidate & x) { return exact_key(x) == exact_key(plan); })) continue;
        shortlist.push_back(plan);
    }

    const int floor_count = std::min(max_simulations, std::max(0, min_simulations));
    if ((int) shortlist.size() < floor_count) {
        for (const auto & plan : coarse) {
            if ((int) shortlist.size() >= floor_count) break;
            if (std::any_of(shortlist.begin(), shortlist.end(),
                    [&](const candidate & x) { return exact_key(x) == exact_key(plan); })) continue;
            shortlist.push_back(plan);
        }
    }

    out.shortlisted = shortlist.size();
    for (auto plan : shortlist) {
        std::vector<llama_region_ring_region> regions;
        if (!make_regions(plan, m.layers, regions)) continue;
        std::string reason;
        if (!simulate(m, regions, plan, &reason)) {
            ++out.rejected_deadlock;
            continue;
        }
        ++out.simulated;
        out.top.push_back(std::move(plan));
    }

    std::sort(out.top.begin(), out.top.end(),
        [](const candidate & a, const candidate & b) {
            if (std::abs(a.wall_ms - b.wall_ms) > 0.05) return a.wall_ms < b.wall_ms;
            if (std::abs(a.ring_error_max - b.ring_error_max) > 1e-9) {
                return a.ring_error_max < b.ring_error_max;
            }
            return a.ring_error_mean < b.ring_error_mean;
        });
    if (!out.top.empty()) {
        out.selected = out.top.front();
        out.valid = true;
        if (out.top.size() > 10) out.top.resize(10);
    }
    return out;
}

// Runtime-applicable wide search. Unlike WIDE_PREVIEW, every candidate
// ends on Phone and has no tail, so the selected topology can be loaded and
// executed by the current canonical final-output path. C/G/P and XC/XG/XP
// remain free planner variables.
inline result search_runtime_applicable(
        const model & m,
        const std::vector<llama_region_ring_region> & baseline_regions,
        int baseline_chunk,
        int max_simulations = 256,
        double e_margin = 0.08,
        int min_simulations = 32) {
    result out;
    if (m.layers < 6 || m.tokens <= 0 || m.hidden <= 0 ||
            !m.compute_ms || !m.transfer_ms ||
            baseline_chunk <= 0 || max_simulations <= 0 ||
            min_simulations < 0 || !std::isfinite(e_margin) ||
            e_margin < 0.0 || baseline_regions.empty() ||
            baseline_regions.back().device != device::PHONE) {
        return out;
    }

    out.baseline.xc = baseline_chunk;
    out.baseline.xg = baseline_chunk;
    out.baseline.xp = baseline_chunk;
    out.baseline_valid =
        estimate_region_ring_balance(
            m, baseline_regions, out.baseline) &&
        simulate(m, baseline_regions, out.baseline);

    const std::array<int,4> x_cpu {16,32,64,128};
    const std::array<int,4> x_gpu {16,32,64,128};
    const std::array<int,4> x_phone {16,32,64,128};
    std::vector<candidate> coarse;

    for (int cycles = 2; cycles <= 4; ++cycles) {
        const int width = m.layers / cycles;
        const int remainder = m.layers % cycles;
        if (width < 4) {
            continue;
        }
        for (int p = 2; p <= std::min(8, width - 2); ++p) {
            for (int g = 1; g <= width - p - 1; ++g) {
                const int c = width - p - g;
                if (c < 1) {
                    continue;
                }
                // A non-dividing model still has 2/3/4 Cycle candidates.
                // Put the extra layers on G, C, or P in the early Cycles;
                // full boundary refinement may subsequently move *every*
                // boundary to any legal position, so these are only seeds.
                const int variations = remainder == 0 ? 1 : 3;
                for (int extra_owner = 0;
                        extra_owner < variations; ++extra_owner) {
                    for (int xc : x_cpu) {
                        for (int xg : x_gpu) {
                            for (int xp : x_phone) {
                                candidate plan;
                                plan.cycles = cycles;
                                plan.c = c;
                                plan.g = g;
                                plan.p = p;
                                plan.tail_c = 0;
                                plan.tail_g = 0;
                                plan.xc = xc;
                                plan.xg = xg;
                                plan.xp = xp;
                                ++out.enumerated;

                                std::vector<llama_region_ring_region> regions;
                                if (remainder == 0) {
                                    if (!make_regions(
                                            plan, m.layers, regions)) {
                                        continue;
                                    }
                                } else {
                                    int layer = 0;
                                    for (int n = 0; n < cycles; ++n) {
                                        const int extra =
                                            n < remainder ? 1 : 0;
                                        const int gg = g +
                                            (extra_owner == 0 ? extra : 0);
                                        const int cc = c +
                                            (extra_owner == 1 ? extra : 0);
                                        const int pp = p +
                                            (extra_owner == 2 ? extra : 0);
                                        regions.emplace_back(
                                            device::GPU, layer, layer + gg);
                                        layer += gg;
                                        regions.emplace_back(
                                            device::CPU, layer, layer + cc);
                                        layer += cc;
                                        regions.emplace_back(
                                            device::PHONE, layer, layer + pp);
                                        layer += pp;
                                    }
                                    plan.concrete_regions = regions;
                                }
                                if (!llama_region_ring_apply_topology_compatible(
                                        regions, m.layers) ||
                                        regions.back().device !=
                                            device::PHONE) {
                                    continue;
                                }
                                if ((m.memory_feasible &&
                                        !m.memory_feasible(plan)) ||
                                        (m.region_memory_feasible &&
                                         !m.region_memory_feasible(regions))) {
                                    ++out.rejected_memory;
                                    continue;
                                }
                                const bool profiled = remainder == 0 ?
                                    estimate_balance(m, plan) :
                                    estimate_region_ring_balance(
                                        m, regions, plan);
                                if (!profiled) {
                                    ++out.rejected_profile;
                                    continue;
                                }
                                if (remainder != 0) {
                                    const auto & a =
                                        plan.ring_stage_ms_per_token;
                                    plan.coarse_ms = std::max(
                                        m.tokens * std::max({a[0],a[1],a[2]}),
                                        plan.weight_h2d_busy_ms);
                                }
                                ++out.profiled;
                                coarse.push_back(std::move(plan));
                            }
                        }
                    }
                }
            }
        }
    }

    // Uniform control candidates may all violate memory, yet a
    // prefix-compensated nonuniform layout can still fit. Continue into
    // first-Cycle E search instead of returning an empty result.
    if (!coarse.empty()) {
    // The minimax ring error is the first-stage physical filter.  Do not
    // let a lower-bound wall estimate or topology-family quota override it:
    // every profile-feasible candidate inside the E window is cheap enough to
    // run through the full prompt event simulator.
    out.coarse_e_min = std::numeric_limits<double>::infinity();
    for (const auto & plan : coarse) {
        out.coarse_e_min =
            std::min(out.coarse_e_min, plan.ring_error_max);
    }
    out.coarse_e_limit =
        std::min(1.0, out.coarse_e_min + e_margin);

    std::sort(
        coarse.begin(), coarse.end(),
        [](const candidate & a, const candidate & b) {
            if (std::abs(a.ring_error_max -
                         b.ring_error_max) > 1e-12) {
                return a.ring_error_max < b.ring_error_max;
            }
            if (std::abs(a.coarse_ms - b.coarse_ms) > 1e-9) {
                return a.coarse_ms < b.coarse_ms;
            }
            return a.ring_error_mean < b.ring_error_mean;
        });

    std::vector<candidate> shortlist;
    shortlist.reserve(coarse.size());
    for (const auto & plan : coarse) {
        if (plan.ring_error_max >
                out.coarse_e_limit + 1e-12) {
            break;
        }
        shortlist.push_back(plan);
    }

    // E remains the primary physical filter, but a very sharp minimum can
    // otherwise leave the full-prompt simulator with almost nothing to rank.
    // If the natural E window is smaller than min_simulations, extend the
    // already E-sorted prefix until the requested floor is reached.  This is
    // not a topology quota and coarse_ms is only a tie-breaker within equal E.
    out.e_window_shortlisted = shortlist.size();
    const size_t floor_count = std::min(
        coarse.size(), (size_t) std::max(0, min_simulations));
    for (size_t i = shortlist.size(); i < floor_count; ++i) {
        shortlist.push_back(coarse[i]);
    }

    out.shortlisted = shortlist.size();
    out.shortlist_e_limit = shortlist.empty() ?
        out.coarse_e_limit : shortlist.back().ring_error_max;
    for (auto plan : shortlist) {
        std::vector<llama_region_ring_region> regions;
        if (!make_regions(plan, m.layers, regions) ||
                regions.empty() ||
                regions.back().device != device::PHONE) {
            continue;
        }
        std::string reason;
        if (!simulate(m, regions, plan, &reason)) {
            ++out.rejected_deadlock;
            continue;
        }
        ++out.simulated;
        out.top.push_back(std::move(plan));
    }

    } // Optional uniform control candidates

    // Two-level nonuniform construction:
    //  1) Optimize the first Cycle with the unchanged minimax E equation;
    //     retain a handful of low-E geometries per Cycle count.
    //  2) For EACH first-Cycle seed, extend the next Cycle according to the
    //     cumulative CPU/GPU/Phone load of the preceding Cycles.
    //  3) Score only *complete* layouts with the original memory guards,
    //     exact full-Prompt DAG simulator and subsequent optional Tensor pass.
    // The old equal-width candidates above remain explicit A/B baselines.
    //
    // A one-Cycle E estimate does not determine the final wall time:
    // independent per-device Chunk FIFOs, stage PREPARE, network contention
    // and GPU two-Bank rolling loads are all charged in simulate().
    const auto wall_less = [](const candidate & a, const candidate & b) {
        if (std::abs(a.wall_ms - b.wall_ms) > 0.05) {
            return a.wall_ms < b.wall_ms;
        }
        if (std::abs(a.ring_error_max - b.ring_error_max) > 1e-9) {
            return a.ring_error_max < b.ring_error_max;
        }
        return a.ring_error_mean < b.ring_error_mean;
    };
    const auto key_for = [](const candidate & v,
                            const std::vector<llama_region_ring_region> & rs) {
        return llama_region_ring_layout_string(rs) + "/" +
            std::to_string(v.xc) + "/" + std::to_string(v.xg) + "/" +
            std::to_string(v.xp);
    };
    std::unordered_set<std::string> scored;
    for (const auto & v : out.top) {
        std::vector<llama_region_ring_region> rs;
        if (make_regions(v, m.layers, rs)) {
            scored.insert(key_for(v, rs));
        }
    }

    const std::array<int, 4> chunks {16, 32, 64, 128};
    const double invalid = std::numeric_limits<double>::infinity();
    // Cache every measurable Dense/MoE full-stage service time once. The
    // Hybrid Profile callback depends on (device, layers, X, KV), not on
    // the logical start layer, so this also keeps prefix expansion cheap.
    std::array<std::array<std::vector<double>, 4>, 3> service;
    const int kv = m.kv_start + m.tokens / 2;
    for (int dev = 0; dev < 3; ++dev) {
        for (size_t xi = 0; xi < chunks.size(); ++xi) {
            service[(size_t) dev][xi].assign(
                (size_t) m.layers + 1, invalid);
            const int x = chunks[xi];
            for (int layers = 1; layers <= m.layers; ++layers) {
                if (dev == (int) device::PHONE && layers > 8) {
                    break;
                }
                double duration_ms = 0.0;
                if (!m.compute_ms((device) dev, layers, x, kv,
                                 duration_ms) ||
                        !std::isfinite(duration_ms) ||
                        duration_ms < 0.0) {
                    continue;
                }
                if (dev == (int) device::PHONE) {
                    double reg_ms = 0.0;
                    if (!phone_registration_cost(
                            m, llama_region_ring_stage_kind::PHONE,
                            layers, x, false, reg_ms)) {
                        continue;
                    }
                    duration_ms += reg_ms;
                }
                service[(size_t) dev][xi][(size_t) layers] =
                    duration_ms / x;
            }
        }
    }

    // Links are P->G, G->C, C->P in cyclic E order. Each is measured at
    // its *source* Chunk size and charged once per conceptual Cycle.
    std::array<double, 4> pg {}, gc {}, cp {};
    for (size_t xi = 0; xi < chunks.size(); ++xi) {
        const int x = chunks[xi];
        const size_t bytes =
            (size_t) m.hidden * (size_t) x * sizeof(float);
        double t = 0.0;
        pg[xi] = m.transfer_ms(
            device::PHONE, device::GPU, bytes, t) &&
            std::isfinite(t) && t >= 0.0 ? t / x : invalid;
        t = 0.0;
        gc[xi] = m.transfer_ms(
            device::GPU, device::CPU, bytes, t) &&
            std::isfinite(t) && t >= 0.0 ? t / x : invalid;
        t = 0.0;
        cp[xi] = m.transfer_ms(
            device::CPU, device::PHONE, bytes, t) &&
            std::isfinite(t) && t >= 0.0 ? t / x : invalid;
    }

    struct first_cycle_seed {
        int cycles = 0;
        int g = 0;
        int c = 0;
        int p = 0;
        int width = 0;
        int width_slack = 0;
        size_t ci = 0;
        size_t gi = 0;
        size_t pi = 0;
        double first_e = 0.0;
        std::array<double, 3> load {};
        std::array<double, 3> links {};
    };
    struct prefix_state {
        std::vector<llama_region_ring_region> regions;
        std::array<double, 3> load {};
        std::array<double, 4> prefix_error {};
        std::array<std::array<double, 3>, 4> prefix_service {};
        double prefix_e_sum = 0.0;
        double width_variance = 0.0;
        double rank = 0.0;
        int min_width = 0;
        int max_width = 0;
    };
    // First Cycle is close to L/N. Subsequent Cycles may differ, but the
    // maximum spread across Cycles is about 25% of average length. The
    // squared-length term below is a *soft* preference inside that range.
    constexpr size_t k_first_seeds_per_n = 12;
    constexpr size_t k_seed_geometry_diversity = 8;
    constexpr size_t k_prefix_beam_per_seed = 8;
    constexpr double k_history_weight = 0.20;
    constexpr double k_width_weight = 0.20;

    for (int n_cycles = 2; n_cycles <= 4; ++n_cycles) {
        const int avg_low = m.layers / n_cycles;
        const int avg_high =
            (m.layers + n_cycles - 1) / n_cycles;
        if (avg_low < 4) {
            continue;
        }
        const int spread = std::max(2, (avg_high + 3) / 4);
        const int width_low = std::max(4, avg_low - spread);
        const int width_high =
            std::min(avg_high + spread,
                     m.layers - (n_cycles - 1) * 4);
        std::vector<first_cycle_seed> first_seeds;
        for (int width = width_low; width <= width_high; ++width) {
            const int remaining = m.layers - width;
            // Can the remaining Cycles have similar lengths to this one?
            const int min_other = std::max(
                width_low, width - spread);
            const int max_other = std::min(
                width_high, width + spread);
            if (remaining < (n_cycles - 1) * min_other ||
                    remaining > (n_cycles - 1) * max_other) {
                continue;
            }
            for (int p = 2; p <= std::min(8, width - 2); ++p) {
                for (int g = 1; g < width - p; ++g) {
                    const int c = width - p - g;
                    for (size_t ci = 0; ci < chunks.size(); ++ci) {
                        for (size_t gi = 0; gi < chunks.size(); ++gi) {
                            for (size_t pi = 0; pi < chunks.size(); ++pi) {
                                first_cycle_seed seed;
                                seed.cycles = n_cycles;
                                seed.g = g;
                                seed.c = c;
                                seed.p = p;
                                seed.width = width;
                                seed.width_slack = spread;
                                seed.ci = ci;
                                seed.gi = gi;
                                seed.pi = pi;
                                seed.load = {
                                    service[(size_t) device::CPU][ci][(size_t) c],
                                    service[(size_t) device::GPU][gi][(size_t) g],
                                    service[(size_t) device::PHONE][pi][(size_t) p]
                                };
                                seed.links = {pg[pi], gc[gi], cp[ci]};
                                if (!std::all_of(
                                        seed.load.begin(), seed.load.end(),
                                        [](double v) {
                                            return std::isfinite(v);
                                        }) ||
                                        !std::all_of(
                                            seed.links.begin(), seed.links.end(),
                                            [](double v) {
                                                return std::isfinite(v) &&
                                                       v >= 0.0;
                                            })) {
                                    continue;
                                }
                                seed.first_e = prefix_cycle_max_error(
                                    seed.load, seed.links, 1);
                                if (!std::isfinite(seed.first_e)) {
                                    continue;
                                }
                                ++out.first_cycle_profiled;
                                first_seeds.push_back(seed);
                            }
                        }
                    }
                }
            }
        }
        if (first_seeds.empty()) {
            continue;
        }
        std::sort(first_seeds.begin(), first_seeds.end(),
            [](const first_cycle_seed & a,
               const first_cycle_seed & b) {
                if (std::abs(a.first_e - b.first_e) > 1e-12) {
                    return a.first_e < b.first_e;
                }
                return a.width < b.width;
            });
        out.first_cycle_e_min = std::min(
            out.first_cycle_e_min, first_seeds.front().first_e);
        const double seed_e_limit =
            std::min(1.0, first_seeds.front().first_e + e_margin);

        // Keep several low-E first-Cycle *geometries* before filling
        // remaining slots with alternative Chunk triples. This prevents
        // the beam from collapsing to one shape with 12 X variations.
        std::vector<size_t> chosen;
        std::unordered_set<std::string> chosen_shapes;
        std::unordered_set<size_t> chosen_indices;
        for (size_t i = 0; i < first_seeds.size(); ++i) {
            if (first_seeds[i].first_e > seed_e_limit + 1e-12 ||
                    chosen.size() >= k_seed_geometry_diversity) {
                break;
            }
            const auto & v = first_seeds[i];
            const std::string shape =
                std::to_string(v.g) + "/" + std::to_string(v.c) +
                "/" + std::to_string(v.p);
            if (chosen_shapes.insert(shape).second) {
                chosen.push_back(i);
                chosen_indices.insert(i);
            }
        }
        for (size_t i = 0; i < first_seeds.size() &&
                chosen.size() < k_first_seeds_per_n; ++i) {
            if (first_seeds[i].first_e > seed_e_limit + 1e-12) {
                break;
            }
            if (chosen_indices.insert(i).second) {
                chosen.push_back(i);
            }
        }
        // A very sharp minimum should not reduce the search to one seed.
        const size_t floor = std::min(
            k_seed_geometry_diversity, first_seeds.size());
        for (size_t i = 0; i < first_seeds.size() &&
                chosen.size() < floor; ++i) {
            if (chosen_indices.insert(i).second) {
                chosen.push_back(i);
            }
        }

        out.first_cycle_selected += chosen.size();
        for (size_t si : chosen) {
            const auto & seed = first_seeds[si];
            prefix_state initial;
            initial.regions.emplace_back(
                device::GPU, 0, seed.g);
            initial.regions.emplace_back(
                device::CPU, seed.g, seed.g + seed.c);
            initial.regions.emplace_back(
                device::PHONE, seed.g + seed.c, seed.width);
            initial.load = seed.load;
            initial.prefix_error[0] = seed.first_e;
            initial.prefix_service[0] = seed.load;
            initial.prefix_e_sum = seed.first_e;
            initial.rank = seed.first_e;
            initial.min_width = seed.width;
            initial.max_width = seed.width;
            std::vector<prefix_state> beam { std::move(initial) };

            for (int completed = 1;
                    completed < n_cycles && !beam.empty(); ++completed) {
                std::vector<prefix_state> expanded;
                for (const auto & state : beam) {
                    const int begin = state.regions.back().layer_end;
                    const int remain = m.layers - begin;
                    const int later = n_cycles - completed - 1;
                    const int next_min_width = std::max({
                        width_low,
                        state.max_width - seed.width_slack
                    });
                    const int next_max_width = std::min({
                        width_high,
                        state.min_width + seed.width_slack
                    });
                    for (int width = next_min_width;
                            width <= next_max_width; ++width) {
                        const int rest = remain - width;
                        if (rest < later * width_low ||
                                rest > later * width_high ||
                                (later == 0 && rest != 0)) {
                            continue;
                        }
                        const int new_min =
                            std::min(state.min_width, width);
                        const int new_max =
                            std::max(state.max_width, width);
                        if (new_max - new_min > seed.width_slack) {
                            continue;
                        }
                        const int later_min = std::max(
                            width_low,
                            new_max - seed.width_slack);
                        const int later_max = std::min(
                            width_high,
                            new_min + seed.width_slack);
                        if (rest < later * later_min ||
                                rest > later * later_max) {
                            continue;
                        }

                        for (int p = 2;
                                p <= std::min(8, width - 2); ++p) {
                            for (int g = 1; g < width - p; ++g) {
                                const int c = width - p - g;
                                const std::array<double, 3> extra {{
                                    service[(size_t) device::CPU][seed.ci][(size_t) c],
                                    service[(size_t) device::GPU][seed.gi][(size_t) g],
                                    service[(size_t) device::PHONE][seed.pi][(size_t) p]
                                }};
                                if (!std::isfinite(extra[0]) ||
                                        !std::isfinite(extra[1]) ||
                                        !std::isfinite(extra[2])) {
                                    continue;
                                }
                                ++out.prefix_expanded;
                                prefix_state next = state;
                                for (size_t i = 0; i < 3; ++i) {
                                    next.load[i] += extra[i];
                                }
                                const double prefix_e =
                                    prefix_cycle_max_error(
                                        next.load, seed.links,
                                        completed + 1);
                                if (!std::isfinite(prefix_e)) {
                                    continue;
                                }
                                next.prefix_e_sum += prefix_e;
                                next.prefix_error[(size_t) completed] =
                                    prefix_e;
                                next.prefix_service[(size_t) completed] =
                                    next.load;
                                const double relative_width =
                                    (double) (width - seed.width) /
                                    seed.width;
                                next.width_variance +=
                                    relative_width * relative_width;
                                next.rank = prefix_e +
                                    k_history_weight *
                                        next.prefix_e_sum / (completed + 1) +
                                    k_width_weight *
                                        next.width_variance / (completed + 1);
                                next.min_width = new_min;
                                next.max_width = new_max;
                                next.regions.emplace_back(
                                    device::GPU, begin, begin + g);
                                next.regions.emplace_back(
                                    device::CPU,
                                    begin + g, begin + g + c);
                                next.regions.emplace_back(
                                    device::PHONE,
                                    begin + g + c, begin + width);
                                expanded.push_back(std::move(next));
                            }
                        }
                    }
                }
                std::sort(expanded.begin(), expanded.end(),
                    [](const prefix_state & a,
                       const prefix_state & b) {
                        if (std::abs(a.rank - b.rank) > 1e-12) {
                            return a.rank < b.rank;
                        }
                        return a.prefix_e_sum < b.prefix_e_sum;
                    });
                if (expanded.size() > k_prefix_beam_per_seed) {
                    expanded.resize(k_prefix_beam_per_seed);
                }
                beam.swap(expanded);
            }

            for (const auto & state : beam) {
                if (state.regions.size() !=
                        (size_t) (3 * n_cycles) ||
                        state.regions.back().layer_end != m.layers ||
                        !llama_region_ring_apply_topology_compatible(
                            state.regions, m.layers)) {
                    continue;
                }
                ++out.prefix_completed;
                candidate proposal;
                proposal.cycles = n_cycles;
                proposal.g = seed.g;
                proposal.c = seed.c;
                proposal.p = seed.p;
                proposal.xc = chunks[seed.ci];
                proposal.xg = chunks[seed.gi];
                proposal.xp = chunks[seed.pi];
                proposal.concrete_regions = state.regions;
                proposal.prefix_cycles = n_cycles;
                proposal.prefix_error = state.prefix_error;
                proposal.prefix_service = state.prefix_service;
                if (!scored.insert(
                        key_for(proposal, state.regions)).second) {
                    continue;
                }
                ++out.enumerated;
                if ((m.memory_feasible &&
                        !m.memory_feasible(proposal)) ||
                        (m.region_memory_feasible &&
                         !m.region_memory_feasible(state.regions))) {
                    ++out.rejected_memory;
                    continue;
                }
                if (!estimate_region_ring_balance(
                        m, state.regions, proposal)) {
                    ++out.rejected_profile;
                    continue;
                }
                ++out.profiled;
                ++out.nonuniform_profiled;
                // Do not reject a compensated layout just because its
                // final E is a little worse than a uniform seed's E.
                // The complete DAG simulator is the final authority.
                std::string why;
                if (!simulate(
                        m, state.regions, proposal, &why)) {
                    ++out.rejected_deadlock;
                    continue;
                }
                ++out.simulated;
                ++out.nonuniform_simulated;
                ++out.shortlisted;
                out.top.push_back(std::move(proposal));
            }
        }
    }

    std::sort(out.top.begin(), out.top.end(), wall_less);
    if (!out.top.empty()) {
        out.selected = out.top.front();
        out.valid = true;
        if (out.top.size() > 10) {
            out.top.resize(10);
        }
    }
    return out;
}

// Phase-1 supported search: keep the model's exact Phone regions and
// original CUDA weight count, adjust only CUDA/CPU boundaries inside the
// three 11-layer PC windows. No changes to Chunk, KV, Phone graph UID
// location, final logits ownership, or live output-slot protocol.
struct supported_choice {
    std::vector<llama_region_ring_region> regions;
    candidate estimate;
    int gpu_shift = 0;
};

struct supported_result {
    bool valid = false;
    bool baseline_valid = false;
    supported_choice selected;
    supported_choice baseline;
    size_t enumerated = 0;
    size_t simulated = 0;
    size_t rejected_memory = 0;
    size_t rejected_profile = 0;
    std::vector<supported_choice> top;
};

inline supported_result search_supported_fixed_phone(
        const model & m,
        const std::vector<llama_region_ring_region> & baseline,
        int chunk_tokens) {
    supported_result result;
    if (!llama_region_ring_apply_safe_phone_shape(baseline) ||
            m.layers != 48 || m.tokens <= 0 || chunk_tokens != 64 ||
            m.slot_pool != (int) LLAMA_REGION_RING_SLOT_POOL_SIZE) {
        return result;
    }
    result.baseline.regions = baseline;
    result.baseline.estimate.xc = chunk_tokens;
    result.baseline.estimate.xg = chunk_tokens;
    result.baseline.estimate.xp = chunk_tokens;
    if (!simulate(m,baseline,result.baseline.estimate)) {
        return result;
    }
    result.baseline_valid = true;

    int pc_starts[3] = {};
    int pc_widths[3] = {};
    int baseline_g[3] = {};
    int gpu_total = 0;
    for (int i = 0; i < 3; ++i) {
        const int j = i * 3;
        pc_starts[i] = baseline[j].layer_begin;
        pc_widths[i] = baseline[j + 2].layer_begin - pc_starts[i];
        baseline_g[i] = baseline[j].layer_end -
                        baseline[j].layer_begin;
        gpu_total += baseline_g[i];
    }

    for (int g0 = 1; g0 < pc_widths[0]; ++g0) {
        for (int g1 = 1; g1 < pc_widths[1]; ++g1) {
            const int g2 = gpu_total - g0 - g1;
            if (g2 <= 0 || g2 >= pc_widths[2]) {
                continue;
            }
            const int gs[3] = {g0,g1,g2};
            supported_choice choice;
            choice.regions.reserve(baseline.size());
            for (int i = 0; i < 3; ++i) {
                const int begin = pc_starts[i];
                const int middle = begin + gs[i];
                const int end = begin + pc_widths[i];
                choice.regions.push_back({device::GPU, begin,middle});
                choice.regions.push_back({device::CPU,middle,end});
                choice.regions.push_back(baseline[i*3+2]);
                choice.gpu_shift += std::abs(gs[i]-baseline_g[i]);
            }
            ++result.enumerated;
            if (!llama_region_ring_apply_safe_compatible(
                    baseline,choice.regions)) {
                ++result.rejected_profile;
                continue;
            }
            if (m.region_memory_feasible &&
                    !m.region_memory_feasible(choice.regions)) {
                ++result.rejected_memory;
                continue;
            }
            choice.estimate.xc = chunk_tokens;
            choice.estimate.xg = chunk_tokens;
            choice.estimate.xp = chunk_tokens;
            if (!simulate(m,choice.regions,choice.estimate)) {
                ++result.rejected_profile;
                continue;
            }
            ++result.simulated;
            result.top.push_back(std::move(choice));
        }
    }
    std::sort(result.top.begin(),result.top.end(),
        [](const supported_choice &a,const supported_choice &b) {
            if (std::abs(a.estimate.wall_ms-b.estimate.wall_ms) > 0.05) {
                return a.estimate.wall_ms < b.estimate.wall_ms;
            }
            return a.gpu_shift < b.gpu_shift;
        });
    if (!result.top.empty()) {
        result.selected=result.top.front();
        result.valid=true;
        if (result.top.size()>5) result.top.resize(5);
    }
    return result;
}

// Executable free C/G/P search for the current three-cycle Ring. Each
// candidate first receives the handwritten minimax ring score
// E=max(e1,e2,e3), then all 70 geometries are still run through the exact
// FIFO/KV/dynamic-slot event simulation. This keeps Chunk64, G->C->P ordering
// and final Phone ownership fixed, but does NOT preserve P5 or the old
// CUDA/Phone totals.
// For a 48-layer model, C+G+P=16 and the current Phone contract allows P2..P8.
inline result search_three_cycle_cgp(
        const model & m,
        const std::vector<llama_region_ring_region> & baseline,
        int chunk_tokens) {
    result out;
    if (m.layers < 9 || m.layers % 3 != 0 || m.tokens <= 0 ||
            chunk_tokens != 64 ||
            m.slot_pool != (int) LLAMA_REGION_RING_SLOT_POOL_SIZE ||
            !m.memory_feasible) {
        return out;
    }

    out.baseline.xc = chunk_tokens;
    out.baseline.xg = chunk_tokens;
    out.baseline.xp = chunk_tokens;
    // Only 70 C/G/P geometries exist here, so compute the minimax equation
    // score for every candidate but do not prune any of them. The fixed
    // baseline is slightly non-uniform (C7/G4/P5, C7/G4/P5, C8/G3/P5), so
    // evaluate its concrete three-cycle layout instead of inventing one C/G.
    // Exact FIFO simulation remains exhaustive in APPLY_CGP.
    if (!estimate_region_ring_balance(m, baseline, out.baseline) ||
        !simulate(m, baseline, out.baseline)) {
        return out;
    }
    out.baseline_valid = true;

    const int cycle_width = m.layers / 3;
    const int max_phone = std::min(8, cycle_width - 2);
    for (int p = 2; p <= max_phone; ++p) {
        for (int g = 1; g < cycle_width - p; ++g) {
            const int c = cycle_width - g - p;
            if (c < 1) {
                continue;
            }

            candidate choice;
            choice.cycles = 3;
            choice.c = c;
            choice.g = g;
            choice.p = p;
            choice.tail_c = 0;
            choice.tail_g = 0;
            choice.xc = chunk_tokens;
            choice.xg = chunk_tokens;
            choice.xp = chunk_tokens;
            ++out.enumerated;

            if (!m.memory_feasible(choice)) {
                ++out.rejected_memory;
                continue;
            }

            std::vector<llama_region_ring_region> regions;
            if (!make_regions(choice, m.layers, regions) ||
                    regions.size() != 9 ||
                    !llama_region_ring_apply_topology_compatible(
                        regions, m.layers)) {
                ++out.rejected_profile;
                continue;
            }

            if (!estimate_balance(m, choice)) {
                ++out.rejected_profile;
                continue;
            }
            ++out.profiled;

            std::string reason;
            if (!simulate(m, regions, choice, &reason)) {
                if (reason == "COMPUTE_PROFILE_MISSING" ||
                        reason == "TRANSFER_PROFILE_MISSING" ||
                        reason == "INVALID_MODEL" ||
                        reason == "INVALID_CHUNK") {
                    ++out.rejected_profile;
                } else {
                    ++out.rejected_deadlock;
                }
                continue;
            }
            ++out.simulated;
            out.top.push_back(std::move(choice));
        }
    }

    std::sort(out.top.begin(), out.top.end(),
        [](const candidate & a, const candidate & b) {
            if (std::abs(a.wall_ms - b.wall_ms) > 0.05) {
                return a.wall_ms < b.wall_ms;
            }
            if (std::abs(a.ring_error_max - b.ring_error_max) > 1e-9) {
                return a.ring_error_max < b.ring_error_max;
            }
            if (a.p != b.p) {
                return a.p < b.p;
            }
            return a.g < b.g;
        });
    if (!out.top.empty()) {
        out.selected = out.top.front();
        out.valid = true;
        if (out.top.size() > 5) {
            out.top.resize(5);
        }
    }
    return out;
}

    
// Executable general-topology search. Unlike the research-only search(),
// every candidate uses the same fixed 64-token chunk and shared dynamic
// activation-slot pool as the runtime. Different CPU/CUDA/Phone counts and a PC tail are legal.
// Coarse filtering keeps measured profiling work bounded; selected candidates
// must also survive the exact FIFO/KV/slot event simulation.
inline result search_supported_topology(
        const model & m, int chunk_tokens, int max_simulations = 192) {
    result out;
    if (m.layers < 6 || m.tokens <= 0 || chunk_tokens != 64 ||
            m.slot_pool != (int) LLAMA_REGION_RING_SLOT_POOL_SIZE ||
            max_simulations <= 0) {
        return out;
    }
    std::vector<candidate> coarse;
    for (int cycles = 2; cycles <= 4; ++cycles) {
        for (int c = 2; c <= 12; ++c) {
            for (int g = 1; g <= m.layers - c - 2; ++g) {
                for (int p = 2; p <= 8; ++p) {
                    const int tail = m.layers - cycles * (c + g + p);
                    if (tail < 0 || tail > m.max_tail) {
                        continue;
                    }
                    for (int tail_g = 0; tail_g <= tail; ++tail_g) {
                        candidate choice;
                        choice.cycles = cycles;
                        choice.c = c;
                        choice.g = g;
                        choice.p = p;
                        choice.tail_c = tail - tail_g;
                        choice.tail_g = tail_g;
                        choice.xc = choice.xg = choice.xp = chunk_tokens;
                        std::vector<llama_region_ring_region> regions;
                        if (!make_regions(choice, m.layers, regions) ||
                                !llama_region_ring_apply_topology_compatible(
                                    regions, m.layers)) {
                            continue;
                        }
                        ++out.enumerated;
                        if ((m.memory_feasible && !m.memory_feasible(choice)) ||
                                (m.region_memory_feasible &&
                                 !m.region_memory_feasible(regions))) {
                            ++out.rejected_memory;
                            continue;
                        }
                        if (!estimate_balance(m, choice)) {
                            ++out.rejected_profile;
                            continue;
                        }
                        ++out.profiled;
                        coarse.push_back(std::move(choice));
                    }
                }
            }
        }
    }
    std::sort(coarse.begin(), coarse.end(), [](const candidate & a,
                                               const candidate & b) {
        if (std::abs(a.ring_error_max-b.ring_error_max) > 1e-12) {
            return a.ring_error_max < b.ring_error_max;
        }
        if (a.coarse_ms != b.coarse_ms) {
            return a.coarse_ms < b.coarse_ms;
        }
        return a.ring_error_mean < b.ring_error_mean;
    });
    // Sample different cycle/Phone geometries first, then use the remaining
    // budget for the best-scored layouts. This avoids concentrating all tests
    // in one family of equivalent layer totals.
    std::vector<candidate> shortlist;
    std::vector<std::pair<int,int>> families;
    for (const auto & choice : coarse) {
        const std::pair<int,int> key {choice.cycles, choice.p};
        if (std::find(families.begin(),families.end(),key) != families.end()) {
            continue;
        }
        families.push_back(key);
        shortlist.push_back(choice);
        if ((int) shortlist.size() >= max_simulations / 3) {
            break;
        }
    }
    for (const auto & choice : coarse) {
        if ((int) shortlist.size() >= max_simulations) {
            break;
        }
        const auto same = [&](const candidate & a) {
            return std::make_tuple(a.cycles,a.c,a.g,a.p,a.tail_c,a.tail_g) ==
                   std::make_tuple(choice.cycles,choice.c,choice.g,choice.p,
                                   choice.tail_c,choice.tail_g);
        };
        if (std::any_of(shortlist.begin(),shortlist.end(),same)) {
            continue;
        }
        shortlist.push_back(choice);
    }
    for (auto choice : shortlist) {
        std::vector<llama_region_ring_region> regions;
        if (!make_regions(choice, m.layers, regions)) {
            continue;
        }
        if (!simulate(m, regions, choice)) {
            ++out.rejected_deadlock;
            continue;
        }
        ++out.simulated;
        out.top.push_back(choice);
    }
    std::sort(out.top.begin(),out.top.end(),
              [](const candidate & a,const candidate & b) {
        if (std::abs(a.wall_ms-b.wall_ms) > 0.05) {
            return a.wall_ms < b.wall_ms;
        }
        return a.ring_error_max < b.ring_error_max;
    });
    if (!out.top.empty()) {
        out.selected = out.top.front();
        out.valid = true;
        if (out.top.size() > 5) {
            out.top.resize(5);
        }
    }
    return out;
}

} // namespace llama_ring_planner
