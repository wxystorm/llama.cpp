#pragma once

// Read real, isolated G/C/N/A Stage measurements collected by
// run_v2_stage_matrix.py. This is not the compact-GGUF runtime proxy.
// Only the orchestrator's PASS CSV is accepted. HTP always requires
// CPU numerical parity PASS; missing shapes never silently become NPU
// compute estimates. The CSV itself is model-fingerprint checked by the
// Python coordinator before this header is invoked.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace llama_ring_native_stage {

struct point {
    char backend = '?';
    int layers = 0;
    int chunk = 0;
    int kv = 0;
    double compute_ms = 0.0;
    double prepare_ms = 0.0;
    size_t weight_bytes = 0;
    size_t kv_bytes = 0;
    size_t workspace_bytes = 0;
    size_t raw_weight_bytes = 0;
    double refill_ms = 0.0;
};

enum class phase { COMPUTE, PREPARE };

class profile {
    std::vector<point> points_;

    static std::vector<std::string> split(const std::string & line) {
        std::vector<std::string> result;
        std::stringstream input(line);
        std::string col;
        while (std::getline(input, col, ',')) {
            result.push_back(std::move(col));
        }
        return result;
    }

    static bool interpolation(
            const std::vector<std::pair<int, double>> & samples,
            int query, double & result) {
        if (query < 1 || samples.empty()) { return false; }
        std::map<int, double> by_x;
        for (auto item : samples) {
            if (item.first < 1 || !std::isfinite(item.second) ||
                    item.second < 0) {
                return false;
            }
            by_x[item.first] = item.second;
        }
        auto hi = by_x.lower_bound(query);
        if (hi == by_x.end()) { return false; } // no unsupported large X/KV
        if (hi == by_x.begin() || hi->first == query) {
            result = hi->second; // conservative cost at smaller X/KV
            return true;
        }
        auto lo = std::prev(hi);
        const double ratio = (double) (query - lo->first) /
                             (hi->first - lo->first);
        result = lo->second + ratio * (hi->second - lo->second);
        return std::isfinite(result) && result >= 0;
    }

public:
    bool load(const std::string & path, std::string & reason) {
        points_.clear();
        std::ifstream file(path);
        if (!file) { reason = "NO_NATIVE_STAGE_CSV"; return false; }
        std::string header;
        if (!std::getline(file, header)) {
            reason = "EMPTY_NATIVE_STAGE_CSV";
            return false;
        }
        if (!header.empty() && header.back() == '\r') { header.pop_back(); }
        if (header.size() >= 3 &&
                (unsigned char) header[0] == 0xEF &&
                (unsigned char) header[1] == 0xBB &&
                (unsigned char) header[2] == 0xBF) {
            header.erase(0, 3);
        }
        const auto cols = split(header);
        std::map<std::string, size_t> offsets;
        for (size_t i = 0; i < cols.size(); ++i) {
            offsets.emplace(cols[i], i);
        }
        for (const auto & required : {
                "backend", "layers", "chunk", "kv_tokens",
                "compute_median_ms", "prepare_median_ms", "weight_bytes",
                "kv_bytes", "graph_workspace_bytes", "parity", "source"}) {
            if (offsets.count(required) == 0) {
                reason = std::string("MISSING_COLUMN_") + required;
                return false;
            }
        }
        std::set<std::tuple<char, int, int, int>> seen;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') { line.pop_back(); }
            if (line.empty()) { continue; }
            const auto fields = split(line);
            if (fields.size() != cols.size()) {
                reason = "BAD_NATIVE_STAGE_CSV_ROW";
                points_.clear();
                return false;
            }
            const auto get = [&](const std::string & key) ->
                    const std::string & {
                return fields[offsets.at(key)];
            };
            point p;
            try {
                const std::string & backend = get("backend");
                if (backend.size() != 1 ||
                        std::string("GCNA").find(backend[0]) ==
                            std::string::npos) {
                    reason = "UNKNOWN_NATIVE_BACKEND";
                    return false;
                }
                p.backend = backend[0];
                p.layers = std::stoi(get("layers"));
                p.chunk = std::stoi(get("chunk"));
                p.kv = std::stoi(get("kv_tokens"));
                p.compute_ms = std::stod(get("compute_median_ms"));
                p.prepare_ms = std::stod(get("prepare_median_ms"));
                p.weight_bytes = (size_t) std::stoull(get("weight_bytes"));
                p.kv_bytes = (size_t) std::stoull(get("kv_bytes"));
                p.workspace_bytes =
                    (size_t) std::stoull(get("graph_workspace_bytes"));
                if (offsets.count("raw_weight_bytes")) {
                    p.raw_weight_bytes =
                        (size_t) std::stoull(get("raw_weight_bytes"));
                }
                if (offsets.count("same_bank_refill_median_ms")) {
                    p.refill_ms =
                        std::stod(get("same_bank_refill_median_ms"));
                }
                if (get("source") != "NATIVE_SINGLE_BACKEND_GGUF_PREFIX" ||
                        (p.backend == 'N' && get("parity") != "PASS") ||
                        get("parity") == "FAIL" || p.layers <= 0 ||
                        p.chunk <= 0 || p.kv < p.chunk ||
                        !std::isfinite(p.compute_ms) ||
                        p.compute_ms <= 0 ||
                        !std::isfinite(p.prepare_ms) ||
                        p.prepare_ms < 0 ||
                        !std::isfinite(p.refill_ms) ||
                        p.refill_ms < 0 || p.weight_bytes == 0 ||
                        !seen.insert({
                            p.backend,p.layers,p.chunk,p.kv}).second) {
                    reason = "UNSAFE_OR_DUPLICATE_NATIVE_STAGE";
                    points_.clear();
                    return false;
                }
            } catch (const std::exception &) {
                reason = "BAD_NATIVE_STAGE_NUMERIC_VALUE";
                points_.clear();
                return false;
            }
            points_.push_back(p);
        }
        if (points_.empty()) {
            reason = "NO_PASS_NATIVE_STAGE_POINTS";
            return false;
        }
        reason.clear();
        return true;
    }

    size_t size() const { return points_.size(); }

    bool has_shape(char backend, int layers) const {
        for (const auto & p : points_) {
            if (p.backend == backend && p.layers == layers) { return true; }
        }
        return false;
    }

    int max_profiled_depth(char backend) const {
        int n = 0;
        for (const auto & p : points_) {
            if (p.backend == backend) {
                n = std::max(n, p.layers);
            }
        }
        return n;
    }

    size_t max_profiled_weight_bytes(char backend) const {
        size_t bytes = 0;
        for (const auto & p : points_) {
            if (p.backend == backend) {
                bytes = std::max(bytes, p.weight_bytes);
            }
        }
        return bytes;
    }

    // A sampled Stage depth is always used as-is. For NPU regions deeper
    // than the sampled matrix, the Planner needs a service-cost model to
    // consider a single true N3/N4/... Region. This approximation never
    // changes its physical layer range or bypasses the Phone allocator.
    bool estimate_measured(char backend, int layers, int chunk, int kv,
                           phase kind, double & ms) const {
        std::map<int, std::vector<std::pair<int, double>>> by_chunk;
        for (const auto & p : points_) {
            if (p.backend != backend || p.layers != layers) { continue; }
            by_chunk[p.chunk].push_back({
                p.kv, kind == phase::COMPUTE ?
                      p.compute_ms : p.prepare_ms});
        }
        std::vector<std::pair<int, double>> at_kv;
        for (const auto & entry : by_chunk) {
            double cost = 0;
            if (interpolation(entry.second, kv, cost)) {
                at_kv.push_back({entry.first, cost});
            }
        }
        return interpolation(at_kv, chunk, ms);
    }

    bool estimate(char backend, int layers, int chunk, int kv,
                  phase kind, double & ms) const {
        if (layers <= 0) { return false; }
        if (estimate_measured(backend, layers, chunk, kv, kind, ms)) {
            return true;
        }
        // Only the HTP backend supports deep contiguous Stage estimates.
        // No interpolation across unsupported Chunk/KV windows is allowed.
        if (backend != 'N' || layers > 32) { return false; }
        std::map<int, double> depth_values;
        for (const auto & p : points_) {
            if (p.backend != 'N' || depth_values.count(p.layers)) {
                continue;
            }
            double value = 0.0;
            if (estimate_measured(
                    'N', p.layers, chunk, kv, kind, value)) {
                depth_values[p.layers] = value;
            }
        }
        if (depth_values.empty()) { return false; }
        const auto upper = depth_values.lower_bound(layers);
        if (upper == depth_values.begin()) {
            ms = upper->second * (double) layers / upper->first;
        } else if (upper == depth_values.end()) {
            const auto last = std::prev(depth_values.end());
            ms = last->second * (double) layers / last->first;
        } else {
            const auto lower = std::prev(upper);
            const double alpha = (double) (layers - lower->first) /
                (upper->first - lower->first);
            ms = lower->second +
                alpha * (upper->second - lower->second);
        }
        return std::isfinite(ms) && ms >= 0.0;
    }

    // PC CPU/CUDA search may ask for more layers than the sampled
    // 2/4/6 prefixes. Use an explicit linear-depth estimate from measured
    // native blocks; do not apply this approximation to NPU parity-gated
    // regions or resident Adreno capacity checks.
    bool estimate_pc_depth(char backend, int layers, int chunk, int kv,
                           phase kind, double & ms) const {
        if ((backend != 'C' && backend != 'G') || layers <= 0) {
            return false;
        }
        if (estimate(backend, layers, chunk, kv, kind, ms)) {
            return true;
        }
        std::map<int, double> depths;
        for (const auto & row : points_) {
            if (row.backend != backend || depths.count(row.layers)) {
                continue;
            }
            double value = 0;
            if (estimate(backend, row.layers, chunk, kv, kind, value)) {
                depths[row.layers] = value;
            }
        }
        if (depths.empty()) { return false; }
        const auto upper = depths.lower_bound(layers);
        if (upper == depths.begin()) {
            // One smaller stage cannot cost less than the measured
            // smallest-depth block; this is a conservative proxy.
            ms = upper->second;
            return true;
        }
        if (upper == depths.end()) {
            const auto last = std::prev(depths.end());
            ms = last->second * (double) layers / last->first;
            return std::isfinite(ms) && ms >= 0;
        }
        const auto lower = std::prev(upper);
        const double alpha = (double) (layers - lower->first) /
            (upper->first - lower->first);
        ms = lower->second +
            alpha * (upper->second - lower->second);
        return std::isfinite(ms) && ms >= 0;
    }

    // Cheap original E/Beam search proxy only. Logical P is one native N1
    // followed by a resident Adreno suffix, NOT the old arbitrary PHONE
    // backend. When the Adreno suffix is deeper than measured shapes,
    // extrapolate its service demand as an explicit cheap-search heuristic.
    // Exact final DAG uses measured N/A shapes and fails closed on misses.
    bool estimate_phone_coarse(int logical_layers, int chunk, int kv,
                               phase kind, double & ms) const {
        if (logical_layers < 1) { return false; }
        double npu = 0.0;
        if (!estimate('N', 1, chunk, kv, kind, npu)) {
            // With NPU0 the phone can run an Adreno-only Stage. The
            // physical DAG still uses exact A samples for each Region.
            return estimate('A', logical_layers, chunk, kv, kind, ms);
        }
        if (logical_layers == 1) {
            ms = npu;
            return true;
        }
        const int adreno_layers = logical_layers - 1;
        double a = 0.0;
        if (estimate('A', adreno_layers, chunk, kv, kind, a)) {
            ms = npu + a;
            return true;
        }
        int largest = 0;
        double measured = 0.0;
        for (const auto & p : points_) {
            if (p.backend != 'A' || p.layers <= largest) {
                continue;
            }
            double value = 0.0;
            if (estimate('A', p.layers, chunk, kv, kind, value)) {
                largest = p.layers;
                measured = value;
            }
        }
        if (largest == 0) { return false; }
        a = measured * (double) adreno_layers / largest;
        ms = npu + a;
        return std::isfinite(ms) && ms >= 0.0;
    }

    // Padded HTP repack allocations and Graph scratch are estimated from
    // the deepest measured prefix. They are only admission estimates:
    // actual resident arenas, individual HTP allocations and shared RAM
    // are verified again by the Phone when the exact Plan is installed.
    bool estimate_npu_allocation(int layers, bool raw, bool workspace,
                                 size_t & bytes) const {
        if (layers <= 0 || layers > 32) { return false; }
        int deepest = 0;
        size_t measured = 0;
        for (const auto & p : points_) {
            if (p.backend != 'N') { continue; }
            const size_t n = workspace ? p.workspace_bytes :
                (raw ? p.raw_weight_bytes : p.weight_bytes);
            if (n == 0) { continue; }
            if (p.layers > deepest) {
                deepest = p.layers;
                measured = n;
            } else if (p.layers == deepest) {
                measured = std::max(measured, n);
            }
        }
        if (deepest == 0) { return false; }
        const long double scaling = workspace ? 1.10L : 1.05L;
        const long double estimated =
            std::ceil(((long double) measured * layers / deepest) *
                      scaling);
        if (!std::isfinite(estimated) ||
                estimated <= 0 ||
                estimated > (long double)
                    std::numeric_limits<size_t>::max()) {
            return false;
        }
        bytes = (size_t) estimated;
        return true;
    }

    bool weight_bytes(char backend, int layers, bool raw,
                      size_t & bytes) const {
        bytes = 0;
        for (const auto & p : points_) {
            if (p.backend != backend || p.layers != layers) { continue; }
            bytes = std::max(bytes, raw ?
                p.raw_weight_bytes : p.weight_bytes);
        }
        if (bytes > 0) { return true; }
        return backend == 'N' &&
            estimate_npu_allocation(layers, raw, false, bytes);
    }

    bool workspace_bytes(char backend, int layers,
                         size_t & bytes) const {
        bytes = 0;
        for (const auto & p : points_) {
            if (p.backend == backend && p.layers == layers) {
                bytes = std::max(bytes, p.workspace_bytes);
            }
        }
        if (bytes > 0) { return true; }
        return backend == 'N' &&
            estimate_npu_allocation(layers, false, true, bytes);
    }

    // One conservative HTP Bank refill event, not a measured overlap
    // percentage. Reject extrapolation beyond measured HTP weight sizes.
    bool npu_refill_ms(size_t bytes, double & cost) const {
        std::map<size_t, double> by_weight;
        for (const auto & p : points_) {
            if (p.backend == 'N' && p.raw_weight_bytes > 0 &&
                    p.refill_ms > 0) {
                by_weight[p.raw_weight_bytes] =
                    std::max(by_weight[p.raw_weight_bytes], p.refill_ms);
            }
        }
        const auto it = by_weight.lower_bound(bytes);
        if (bytes == 0 || it == by_weight.end()) { return false; }
        cost = it->second;
        return true;
    }
};

} // namespace llama_ring_native_stage
