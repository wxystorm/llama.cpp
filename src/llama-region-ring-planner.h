#pragma once

// Read-only Region Ring Planner: search C/G/P cycles and independent Chunk
// sizes, then simulate token-interval dependencies. It never modifies the
// live Ring scheduler, buffers, KV cache, or Phone RPC state.
#include "llama-region-ring.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <string>
#include <tuple>
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
    double balance = 0.0;
    double coarse_ms = 0.0;
    double wall_ms = 0.0;
    std::array<double, 3> busy_ms {};
    double pc_gate_busy_ms = 0.0;
    std::array<double, 3> transfer_busy_ms {};
    int tasks = 0;
    int edges = 0;
    int max_live_slots = 0;
};

struct model {
    int layers = 48;
    int tokens = 358;
    int kv_start = 0;
    int hidden = 2048;
    int window = 2;
    int max_tail = 12;
    bool shared_phone_link = true; // Conservative effective Wi-Fi contention.
    // Runtime region_ring_backend_execution_mutex serializes PC CPU/CUDA
    // graph execution. Phone computation can overlap after READY releases it.
    bool pc_compute_gate = true;
    double phone_register_ms = 0.0; // 5-layer reference; conservative per Phone task.
    // Return false when no measured/interpolated profile supports the shape.
    std::function<bool(device, int, int, int, double &)> compute_ms;
    std::function<bool(device, device, size_t, double &)> transfer_ms;
    std::function<bool(const candidate &)> memory_feasible;
    // Unlike memory_feasible(candidate), this callback accepts per-cycle
    // C/G boundaries. It is used ONLY by the real APPLY-safe search.
    std::function<bool(const std::vector<llama_region_ring_region> &)>
        region_memory_feasible;
};

inline int chunk_for(const candidate & c, device d) {
    return d == device::CPU ? c.xc : d == device::GPU ? c.xg : c.xp;
}

inline bool make_regions(const candidate & c, int total_layers,
                         std::vector<llama_region_ring_region> & regions) {
    regions.clear();
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
    for (int i = 0; i < c.cycles; ++i) {
        append(device::CPU, c.c);
        append(device::GPU, c.g);
        append(device::PHONE, c.p);
    }
    append(device::CPU, c.tail_c);
    append(device::GPU, c.tail_g);
    return layer == total_layers;
}

struct task {
    int region = 0;
    int begin = 0;
    int end = 0;
    int previous = -1;
    int inputs = 0;
    int outputs = 0;
    int remaining_outputs = 0;
    double duration = 0.0;
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
struct event {
    double at = 0.0;
    int id = -1;
    bool is_edge = false;
    bool operator<(const event & rhs) const {
        if (at != rhs.at) { return at > rhs.at; }
        if (is_edge != rhs.is_edge) { return !is_edge; }
        return id > rhs.id;
    }
};

// Simulate staged input copies, so a producer Slot is released after its
// last output transfer, not after a larger downstream Chunk finishes. The
// future runtime must provide those staging buffers before APPLY.
inline bool simulate(const model & m,
                     const std::vector<llama_region_ring_region> & regions,
                     candidate & c, std::string * reason = nullptr) {
    auto reject = [&](const char * why) {
        if (reason) { *reason = why; }
        return false;
    };
    if (regions.empty() || m.tokens <= 0 || m.hidden <= 0 ||
        m.window <= 0 || !m.compute_ms || !m.transfer_ms) {
        return reject("INVALID_MODEL");
    }
    std::vector<task> tasks;
    std::vector<std::vector<int>> by_region(regions.size());
    for (size_t r = 0; r < regions.size(); ++r) {
        const auto d = regions[r].device;
        const int x = chunk_for(c, d);
        if (x <= 0) { return reject("INVALID_CHUNK"); }
        int previous = -1;
        for (int begin = 0; begin < m.tokens; begin += x) {
            const int end = std::min(m.tokens, begin + x);
            const int kv = m.kv_start + end;
            double ms = 0.0;
            if (!m.compute_ms(d, regions[r].layer_end - regions[r].layer_begin,
                              end - begin, kv, ms) ||
                !std::isfinite(ms) || ms < 0.0) {
                return reject("COMPUTE_PROFILE_MISSING");
            }
            if (d == device::PHONE) {
                // The registered graph scales with the number of Phone
                // layers; never apply the same 314-node/P5 registration
                // estimate unchanged to arbitrary P2/P8 candidates.
                ms += m.phone_register_ms *
                      (double) (regions[r].layer_end - regions[r].layer_begin) / 5.0;
            }
            const int id = (int) tasks.size();
            tasks.push_back({(int) r, begin, end, previous, 0, 0, 0, ms});
            by_region[r].push_back(id);
            previous = id;
        }
    }
    std::vector<edge> edges;
    for (size_t r = 0; r + 1 < regions.size(); ++r) {
        const auto a = regions[r].device;
        const auto b = regions[r + 1].device;
        const int lane = a == device::PHONE ? (m.shared_phone_link ? 1 : 2) :
                         b == device::PHONE ? 1 : 0;
        for (int src : by_region[r]) {
            for (int dst : by_region[r + 1]) {
                const int overlap = std::min(tasks[src].end, tasks[dst].end) -
                                    std::max(tasks[src].begin, tasks[dst].begin);
                if (overlap <= 0) { continue; }
                const size_t bytes = (size_t) overlap * (size_t) m.hidden * sizeof(float);
                double ms = 0.0;
                if (!m.transfer_ms(a, b, bytes, ms) ||
                    !std::isfinite(ms) || ms < 0.0) {
                    return reject("TRANSFER_PROFILE_MISSING");
                }
                edges.push_back({src, dst, lane, ms});
                ++tasks[src].outputs;
                ++tasks[dst].inputs;
            }
        }
    }
    for (auto & t : tasks) { t.remaining_outputs = t.outputs; }
    std::priority_queue<event> events;
    std::array<bool, 3> device_busy {}, link_busy {};
    bool pc_compute_gate_busy = false;
    int next_pc_device = 0;  // Round-robin tie break across PC CPU/CUDA.
    std::vector<int> live_slots(regions.size(), 0);
    std::array<double, 3> busy_ms {}, transfer_busy_ms {};
    double now = 0.0;
    int completed = 0;
    int max_live = 0;
    const int n_tasks = (int) tasks.size();
    while (completed < n_tasks) {
        bool scheduled = false;
        // Prefer ready transfers: they release producer output-slot credits.
        for (size_t e = 0; e < edges.size(); ++e) {
            auto & it = edges[e];
            if (it.started || !tasks[it.producer].done || link_busy[it.lane]) {
                continue;
            }
            it.started = true;
            link_busy[it.lane] = true;
            transfer_busy_ms[it.lane] += it.duration;
            events.push({now + it.duration, (int) e, true});
            scheduled = true;
        }
        for (int turn = 0; turn < 3; ++turn) {
            const int dev = m.pc_compute_gate && turn < 2 ?
                (turn == 0 ? next_pc_device : 1 - next_pc_device) : turn;
            if (device_busy[dev] ||
                (m.pc_compute_gate && dev != (int) device::PHONE &&
                 pc_compute_gate_busy)) {
                continue;
            }
            // Downstream-first scheduling helps clear Window credits without
            // overtaking the previous KV interval in a single Region.
            for (int r = (int) regions.size() - 1; r >= 0; --r) {
                if ((int) regions[r].device != dev) { continue; }
                int selected = -1;
                for (int id : by_region[r]) {
                    const auto & t = tasks[id];
                    if (t.started || t.inputs != 0 ||
                        (t.previous >= 0 && !tasks[t.previous].done) ||
                        (r + 1 < (int) regions.size() &&
                         live_slots[r] >= m.window)) {
                        continue;
                    }
                    selected = id;
                    break;
                }
                if (selected < 0) { continue; }
                auto & t = tasks[selected];
                t.started = true;
                device_busy[dev] = true;
                if (m.pc_compute_gate && dev != (int) device::PHONE) {
                    pc_compute_gate_busy = true;
                    next_pc_device = 1 - dev;
                }
                if (r + 1 < (int) regions.size()) {
                    ++live_slots[r];
                    for (int s : live_slots) { max_live = std::max(max_live, s); }
                }
                busy_ms[dev] += t.duration;
                events.push({now + t.duration, selected, false});
                scheduled = true;
                break;
            }
        }
        if (events.empty()) {
            return reject(scheduled ? "INTERNAL_EVENT_ERROR" : "SLOT_OR_KV_DEADLOCK");
        }
        now = events.top().at;
        if (!std::isfinite(now)) { return reject("TIME_NONFINITE"); }
        // Handle all completions at this simulated time before scheduling.
        while (!events.empty() && events.top().at <= now) {
            const event current = events.top();
            events.pop();
            if (current.is_edge) {
                const auto & e = edges[current.id];
                link_busy[e.lane] = false;
                if (--tasks[e.consumer].inputs < 0 ||
                    --tasks[e.producer].remaining_outputs < 0) {
                    return reject("EDGE_COUNT_UNDERFLOW");
                }
                if (tasks[e.producer].remaining_outputs == 0) {
                    --live_slots[tasks[e.producer].region];
                }
            } else {
                auto & t = tasks[current.id];
                if (t.done) { return reject("DUPLICATE_TASK_DONE"); }
                t.done = true;
                const int finished_device = (int) regions[t.region].device;
                device_busy[finished_device] = false;
                if (m.pc_compute_gate &&
                    finished_device != (int) device::PHONE) {
                    pc_compute_gate_busy = false;
                }
                ++completed;
            }
        }
    }
    if (std::any_of(live_slots.begin(), live_slots.end(), [](int x){return x != 0;})) {
        return reject("LIVE_SLOTS_NOT_DRAINED");
    }
    if (m.pc_compute_gate &&
        busy_ms[(int) device::CPU] + busy_ms[(int) device::GPU] >
        now + std::max(1e-6, 1e-9 * now)) {
        return reject("PC_GATE_OVERCOMMITTED");
    }
    c.wall_ms = now;
    c.pc_gate_busy_ms = busy_ms[(int) device::CPU] +
                        busy_ms[(int) device::GPU];
    c.busy_ms = busy_ms;
    c.transfer_busy_ms = transfer_busy_ms;
    c.tasks = n_tasks;
    c.edges = (int) edges.size();
    c.max_live_slots = max_live;
    return true;
}

inline bool estimate_balance(const model & m, candidate & c) {
    if (!m.compute_ms || !m.transfer_ms || m.hidden <= 0) { return false; }
    const int kv = m.kv_start + m.tokens / 2;
    double tc=0, tg=0, tp=0, cp=0, pc=0, local=0;
    if (!m.compute_ms(device::CPU,c.c,c.xc,kv,tc) ||
        !m.compute_ms(device::GPU,c.g,c.xg,kv,tg) ||
        !m.compute_ms(device::PHONE,c.p,c.xp,kv,tp)) { return false; }
    constexpr int ref_tokens = 64;
    const size_t bytes = (size_t) m.hidden * ref_tokens * sizeof(float);
    if (!m.transfer_ms(device::GPU,device::PHONE,bytes,cp) ||
        !m.transfer_ms(device::PHONE,device::CPU,bytes,pc) ||
        !m.transfer_ms(device::CPU,device::GPU,bytes,local)) { return false; }
    const double s_pc = tc/c.xc + tg/c.xg + (cp+local)/ref_tokens;
    const double s_phone = tp/c.xp + pc/ref_tokens;
    if (!(s_pc > 0.0) || !(s_phone > 0.0)) { return false; }
    c.balance = std::abs(s_pc-s_phone) / std::max(s_pc,s_phone);
    const double cpu_total = c.cycles*tc/c.xc + c.tail_c*(tc/c.xc/std::max(c.c,1));
    const double gpu_total = c.cycles*tg/c.xg + c.tail_g*(tg/c.xg/std::max(c.g,1));
    const double phone_total = c.cycles*tp/c.xp;
    const double net_total = c.cycles*(cp+pc)/ref_tokens;
    const double pc_gate_total = m.pc_compute_gate ?
        cpu_total + gpu_total : std::max(cpu_total,gpu_total);
    c.coarse_ms = m.tokens*std::max({pc_gate_total,phone_total,net_total}) +
                  m.tokens*0.03*c.balance*std::max(s_pc,s_phone);
    return std::isfinite(c.coarse_ms);
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
    std::vector<candidate> top;
};

inline result search(const model & m, int max_simulations = 192) {
    result out;
    if (m.layers < 6 || m.tokens <= 0 || max_simulations <= 0) { return out; }
    const std::array<int,3> x_pc {32,64,128};
    const std::array<int,4> x_phone {16,32,64,128};
    std::vector<candidate> coarse;
    for (int n = 2; n <= 4; ++n) {
        for (int c = 2; c <= 12; ++c) {
            for (int g = 1; g <= 8; ++g) {
                for (int p = 2; p <= 8; ++p) {
                    const int tail = m.layers - n*(c+g+p);
                    if (tail < 0 || tail > m.max_tail) { continue; }
                    for (int tg = 0; tg <= tail; ++tg) {
                        const int tc = tail-tg;
                        for (int xc : x_pc) for (int xg : x_pc) for (int xp : x_phone) {
                            candidate plan;
                            plan.cycles=n; plan.c=c; plan.g=g; plan.p=p;
                            plan.tail_c=tc; plan.tail_g=tg;
                            plan.xc=xc; plan.xg=xg; plan.xp=xp;
                            ++out.enumerated;
                            if (m.memory_feasible && !m.memory_feasible(plan)) {
                                ++out.rejected_memory;
                                continue;
                            }
                            if (!estimate_balance(m,plan)) {
                                ++out.rejected_profile;
                                continue;
                            }
                            ++out.profiled;
                            coarse.push_back(plan);
                        }
                    }
                }
            }
        }
    }
    std::sort(coarse.begin(),coarse.end(),[](const candidate &a,const candidate &b){
        if (a.coarse_ms != b.coarse_ms) return a.coarse_ms < b.coarse_ms;
        return a.balance < b.balance;
    });
    std::vector<candidate> shortlist;
    std::vector<std::tuple<int,int,int,int>> families;
    for (const auto & plan : coarse) {
        const auto family=std::make_tuple(plan.cycles,plan.c,plan.g,plan.p);
        if (std::find(families.begin(),families.end(),family)!=families.end()) continue;
        families.push_back(family);
        shortlist.push_back(plan);
        if ((int) shortlist.size() >= max_simulations / 2) break;
    }
    for (const auto & plan : coarse) {
        if ((int) shortlist.size() >= max_simulations) break;
        auto key=[](const candidate &c){return std::make_tuple(c.cycles,c.c,c.g,c.p,c.tail_c,c.tail_g,c.xc,c.xg,c.xp);};
        if (std::any_of(shortlist.begin(),shortlist.end(),[&](const candidate &x){return key(x)==key(plan);})) continue;
        shortlist.push_back(plan);
    }
    for (auto plan : shortlist) {
        std::vector<llama_region_ring_region> regions;
        if (!make_regions(plan,m.layers,regions)) continue;
        std::string reason;
        if (!simulate(m,regions,plan,&reason)) {
            ++out.rejected_deadlock;
            continue;
        }
        ++out.simulated;
        out.top.push_back(plan);
    }
    std::sort(out.top.begin(),out.top.end(),[](const candidate&a,const candidate&b){
        if(a.wall_ms!=b.wall_ms) return a.wall_ms<b.wall_ms;
        return a.balance<b.balance;
    });
    if (!out.top.empty()) {
        out.selected=out.top.front();
        out.valid=true;
        if(out.top.size()>5) out.top.resize(5);
    }
    return out;
}

// Phase-1 supported search: keep the model's exact Phone regions and
// original CUDA weight count, adjust only CPU/CUDA boundaries inside the
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
            m.window != 2 || !m.pc_compute_gate) {
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
        baseline_g[i] = baseline[j + 1].layer_end -
                        baseline[j + 1].layer_begin;
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
                const int middle = begin + pc_widths[i] - gs[i];
                const int end = begin + pc_widths[i];
                choice.regions.push_back({device::CPU, begin,middle});
                choice.regions.push_back({device::GPU,middle,end});
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

    
// Executable general-topology search. Unlike the research-only search(),
// every candidate uses the same fixed 64-token chunk and 2-slot protocol
// as the runtime. Different CPU/CUDA/Phone counts and a PC tail are legal.
// Coarse filtering keeps measured profiling work bounded; selected candidates
// must also survive the exact FIFO/KV/slot event simulation.
inline result search_supported_topology(
        const model & m, int chunk_tokens, int max_simulations = 192) {
    result out;
    if (m.layers < 6 || m.tokens <= 0 || chunk_tokens != 64 ||
            m.window != 2 || !m.pc_compute_gate || max_simulations <= 0) {
        return out;
    }
    std::vector<candidate> coarse;
    for (int cycles = 2; cycles <= 4; ++cycles) {
        for (int c = 2; c <= 12; ++c) {
            for (int g = 1; g <= 8; ++g) {
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
        if (a.coarse_ms != b.coarse_ms) {
            return a.coarse_ms < b.coarse_ms;
        }
        return a.balance < b.balance;
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
        return a.balance < b.balance;
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
