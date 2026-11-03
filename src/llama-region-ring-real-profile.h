#pragma once

// Real-path Stage service measurements for Region Ring Planner.
//
// The CSV is generated from genuine CPU/CUDA Chunk traces and Phone-owned
// Graph execution/registration logs; it is never a synthetic GGML graph.
// Profile selection is opt-in and missing shapes fall back to legacy probes.
// No inference, RPC or file I/O is performed during candidate search.

#include "llama-region-ring-planner.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <iterator>
#include <set>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace llama_ring_planner {

enum class real_stage_phase { COMPUTE, PREPARE, SETUP };

struct real_stage_point {
    device dev = device::CPU;
    int layers = 0;
    int tokens = 0;
    int kv_tokens = 0;
    double setup_ms = 0.0;
    double upload_ms = 0.0;
    double prepare_ms = 0.0;
    double compute_ms = 0.0;
    bool terminal = false;
    int samples = 0;
};

class real_stage_profile {
public:
    bool load(const std::string & path, std::string & error) {
        points_.clear();
        std::ifstream in(path);
        if (!in) {
            error = "CANNOT_OPEN_PROFILE";
            return false;
        }
        std::string line;
        bool header_seen = false;
        int line_number = 0;
        while (std::getline(in, line)) {
            ++line_number;
            if (line.empty() || line[0] == '#') {
                continue;
            }
            if (!header_seen) {
                if (line.size() >= 3 &&
                        (unsigned char) line[0] == 0xEF &&
                        (unsigned char) line[1] == 0xBB &&
                        (unsigned char) line[2] == 0xBF) {
                    line.erase(0, 3);
                }
                if (line !=
                    "device,layers,tokens,kv_tokens,setup_ms,upload_ms,"
                    "prepare_ms,compute_ms,terminal,samples") {
                    error = "SCHEMA_MISMATCH";
                    return false;
                }
                header_seen = true;
                continue;
            }
            std::vector<std::string> cols;
            std::stringstream stream(line);
            std::string col;
            while (std::getline(stream, col, ',')) {
                cols.push_back(col);
            }
            if (cols.size() != 10) {
                error = "INVALID_COLUMN_COUNT_AT_" + std::to_string(line_number);
                return false;
            }
            real_stage_point p;
            try {
                if (cols[0] == "CPU") {
                    p.dev = device::CPU;
                } else if (cols[0] == "GPU") {
                    p.dev = device::GPU;
                } else if (cols[0] == "PHONE") {
                    p.dev = device::PHONE;
                } else {
                    throw std::invalid_argument("device");
                }
                const auto integer = [&](size_t c) {
                    size_t pos = 0;
                    const int n = std::stoi(cols[c], &pos);
                    if (pos != cols[c].size()) {
                        throw std::invalid_argument("integer");
                    }
                    return n;
                };
                const auto scalar = [&](size_t c) {
                    size_t pos = 0;
                    const double n = std::stod(cols[c], &pos);
                    if (pos != cols[c].size() || !std::isfinite(n)) {
                        throw std::invalid_argument("float");
                    }
                    return n;
                };
                p.layers = integer(1);
                p.tokens = integer(2);
                p.kv_tokens = integer(3);
                p.setup_ms = scalar(4);
                p.upload_ms = scalar(5);
                p.prepare_ms = scalar(6);
                p.compute_ms = scalar(7);
                const int terminal = integer(8);
                p.terminal = terminal == 1;
                p.samples = integer(9);
                if (p.layers <= 0 || p.tokens <= 0 ||
                        p.kv_tokens < p.tokens || p.samples <= 0 ||
                        terminal < 0 || terminal > 1 ||
                        (p.terminal && p.dev != device::PHONE) ||
                        p.setup_ms < 0.0 || p.upload_ms < 0.0 ||
                        p.prepare_ms < 0.0 || p.compute_ms < 0.0) {
                    throw std::invalid_argument("range");
                }
            } catch (const std::exception &) {
                error = "INVALID_ROW_AT_" + std::to_string(line_number);
                return false;
            }
            points_.push_back(p);
        }
        if (!header_seen || points_.empty()) {
            error = "EMPTY_REAL_STAGE_PROFILE";
            return false;
        }
        error.clear();
        return true;
    }

    size_t size() const { return points_.size(); }

    bool estimate(device d, int layers, int tokens, int kv_tokens,
                  bool terminal, real_stage_phase phase, double & result) const {
        if (layers <= 0 || tokens <= 0 || kv_tokens < tokens) {
            return false;
        }
        if (estimate_exact_class(
                d, layers, tokens, kv_tokens, terminal, phase, result)) {
            return true;
        }
        // One-layer L39 may be absent from the calibration file. In that
        // case a measured nonterminal Phone layer is preferable to
        // silently fabricating a zero-cost terminal computation.
        if (terminal && d == device::PHONE) {
            return estimate_exact_class(
                d, layers, tokens, kv_tokens, false, phase, result);
        }
        return false;
    }

private:
    std::vector<real_stage_point> points_;

    static double metric(const real_stage_point & p, real_stage_phase phase) {
        switch (phase) {
            case real_stage_phase::COMPUTE: return p.compute_ms;
            case real_stage_phase::PREPARE: return p.upload_ms + p.prepare_ms;
            case real_stage_phase::SETUP: return p.setup_ms;
        }
        return 0.0;
    }

    static double lerp(double x, double x0, double y0,
                       double x1, double y1) {
        return x1 > x0 ?
            std::max(0.0, y0 + (y1 - y0) * (x - x0) / (x1 - x0)) : y0;
    }

    bool estimate_exact_class(
            device d, int layers, int tokens, int kv_tokens, bool terminal,
            real_stage_phase phase, double & result) const {
        std::set<int> depths;
        for (const auto & p : points_) {
            if (p.dev == d && p.terminal == terminal) {
                depths.insert(p.layers);
            }
        }
        if (depths.empty()) {
            return false;
        }

        // At a fixed layer depth and Token count, interpolate over actual
        // KV positions first (the per-chunk Attention context changes).
        const auto at_depth = [&](int depth, double & v) -> bool {
            std::set<int> xs;
            for (const auto & p : points_) {
                if (p.dev == d && p.terminal == terminal &&
                        p.layers == depth) {
                    xs.insert(p.tokens);
                }
            }
            if (xs.empty() || tokens > *xs.rbegin() * 2 ||
                    (long long) tokens * 2 < *xs.begin()) {
                return false;
            }
            const auto at_tokens = [&](int x, double & m) -> bool {
                const real_stage_point * lo = nullptr;
                const real_stage_point * hi = nullptr;
                for (const auto & p : points_) {
                    if (p.dev != d || p.terminal != terminal ||
                            p.layers != depth || p.tokens != x) {
                        continue;
                    }
                    if (p.kv_tokens <= kv_tokens &&
                            (lo == nullptr || p.kv_tokens > lo->kv_tokens)) {
                        lo = &p;
                    }
                    if (p.kv_tokens >= kv_tokens &&
                            (hi == nullptr || p.kv_tokens < hi->kv_tokens)) {
                        hi = &p;
                    }
                }
                if (lo == nullptr) { lo = hi; }
                if (hi == nullptr) { hi = lo; }
                if (lo == nullptr) { return false; }
                m = lerp((double) kv_tokens,
                         (double) lo->kv_tokens, metric(*lo, phase),
                         (double) hi->kv_tokens, metric(*hi, phase));
                return true;
            };
            const auto upper = xs.lower_bound(tokens);
            int left = upper == xs.end() ? *xs.rbegin() :
                upper == xs.begin() ? *xs.begin() : *std::prev(upper);
            int right = upper == xs.end() ? *xs.rbegin() : *upper;
            if (left == right && xs.size() >= 2) {
                if (upper == xs.end()) {
                    left = *std::prev(xs.end(), 2);
                } else if (upper == xs.begin() && tokens < *upper) {
                    right = *std::next(xs.begin());
                }
            }
            double a = 0.0, b = 0.0;
            if (!at_tokens(left, a) || !at_tokens(right, b)) {
                return false;
            }
            v = lerp((double) tokens, left, a, right, b);
            if (left == right && tokens != left) {
                // A single token-size sample cannot identify fixed overhead.
                // Keep it only near the measured point, otherwise use the
                // existing independent Profile as the safer fallback.
                if ((long long) tokens * 4 < 3LL * left ||
                        (long long) tokens * 4 > 5LL * left) {
                    return false;
                }
            }
            return std::isfinite(v) && v >= 0.0;
        };

        const auto upper = depths.lower_bound(layers);
        int left = upper == depths.end() ? *depths.rbegin() :
            upper == depths.begin() ? *depths.begin() : *std::prev(upper);
        int right = upper == depths.end() ? *depths.rbegin() : *upper;
        if (left == right && depths.size() >= 2) {
            if (upper == depths.end()) {
                left = *std::prev(depths.end(), 2);
            } else if (upper == depths.begin() && layers < *upper) {
                right = *std::next(depths.begin());
            }
        }
        if (layers > *depths.rbegin() * 2 ||
                (long long) layers * 2 < *depths.begin()) {
            return false;
        }
        double a = 0.0, b = 0.0;
        if (!at_depth(left, a) || !at_depth(right, b)) {
            return false;
        }
        result = lerp((double) layers, left, a, right, b);
        if (left == right && layers != left) {
            // A single depth anchor is not enough to extrapolate arbitrary
            // Stage widths; preserve the prior model for these candidates.
            return false;
        }
        return std::isfinite(result) && result >= 0.0;
    }
};

} // namespace llama_ring_planner
