#pragma once

// Opt-in CUDA event timeline. Measures stream intervals without injecting
// per-graph synchronization. Collects events at the existing CUDA synchronize
// boundary so PROFILE does not accidentally serialize the Jupiter pipeline.
#include "../ggml-jupiter-fig3.h"
#include <cmath>
#include <map>
#include <mutex>
#include <vector>

namespace jupiter_fig3_cuda {
struct ticket {
    cudaEvent_t start = nullptr;
    int device = -1;
    const char * kind = nullptr;
    const char * label = nullptr;
    uint64_t bytes = 0;
};
struct pending { ticket t; cudaEvent_t end = nullptr; };
struct state {
    cudaEvent_t anchor = nullptr;
    int64_t anchor_pc_ns = 0;
    std::vector<pending> spans;
};
inline std::mutex & lock() { static std::mutex m; return m; }
inline std::map<int, state> & states() { static std::map<int, state> m; return m; }

inline ticket begin(int device, cudaStream_t stream, const char * kind,
                    const char * label, uint64_t bytes = 0) {
    if (!jupiter_fig3::pc_enabled()) return {};
    // Device already selected by ggml-cuda caller.
    std::lock_guard<std::mutex> guard(lock());
    state & d = states()[device];
    if (!d.anchor) {
        cudaEvent_t anchor = nullptr;
        if (cudaEventCreate(&anchor) != cudaSuccess) return {};
        if (cudaEventRecord(anchor, stream) != cudaSuccess ||
            cudaEventSynchronize(anchor) != cudaSuccess) {
            cudaEventDestroy(anchor);
            return {};
        }
        d.anchor = anchor;
        d.anchor_pc_ns = jupiter_fig3::ns();
        jupiter_fig3::pc_row("pc_cuda_events.csv",
                            "category,start_ns,end_ns,source,device,bytes");
    }
    cudaEvent_t first = nullptr;
    if (cudaEventCreate(&first) != cudaSuccess) return {};
    if (cudaEventRecord(first, stream) != cudaSuccess) {
        cudaEventDestroy(first);
        return {};
    }
    return {first, device, kind, label, bytes};
}
inline void end(ticket t, cudaStream_t stream) {
    if (!t.start) return;
    cudaEvent_t last = nullptr;
    if (cudaEventCreate(&last) != cudaSuccess) {
        cudaEventDestroy(t.start);
        return;
    }
    if (cudaEventRecord(last, stream) != cudaSuccess) {
        cudaEventDestroy(t.start);
        cudaEventDestroy(last);
        return;
    }
    std::lock_guard<std::mutex> guard(lock());
    states()[t.device].spans.push_back({t, last});
}
inline void flush(int device) {
    if (!jupiter_fig3::pc_enabled()) return;
    std::lock_guard<std::mutex> guard(lock());
    auto it = states().find(device);
    if (it == states().end() || !it->second.anchor) return;
    auto & d = it->second;
    std::vector<pending> deferred;
    for (auto & item : d.spans) {
        if (cudaEventQuery(item.end) != cudaSuccess) {
            deferred.push_back(item);
            continue;
        }
        float a = 0.f, b = 0.f;
        if (cudaEventElapsedTime(&a,d.anchor,item.t.start) == cudaSuccess &&
            cudaEventElapsedTime(&b,d.anchor,item.end) == cudaSuccess && b > a) {
            const auto start = d.anchor_pc_ns + static_cast<int64_t>(
                std::llround(static_cast<double>(a)*1000000.0));
            const auto stop = d.anchor_pc_ns + static_cast<int64_t>(
                std::llround(static_cast<double>(b)*1000000.0));
            jupiter_fig3::device("pc_cuda_events.csv", item.t.kind,
                                 start, stop, item.t.label,
                                 item.t.device, item.t.bytes);
        }
        cudaEventDestroy(item.t.start);
        cudaEventDestroy(item.end);
    }
    d.spans.swap(deferred);
}
} // namespace jupiter_fig3_cuda
