#pragma once

// A deterministic minimax partition of the Phone suffix into contiguous
// FastRPC HTP sessions. Physical GGML backend allocations (not raw GGUF
// nbytes) and per-layer KV estimates are charged before any weights load.
// Keep this free of GGML dependencies so it can be tested independently.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace jupiter_htp {

struct placement {
    // Local exclusive end indices in the Phone suffix, one for each session.
    std::vector<size_t> ends;
    // Weight + KV charges for each session, including output on the final one.
    std::vector<uint64_t> used_bytes;
    double max_utilization = 0.0;
};

inline placement partition_contiguous(
    const std::vector<uint64_t> & layers,
    uint64_t output_bytes,
    const std::vector<uint64_t> & capacities) {
    if (layers.empty() || capacities.empty()) {
        throw std::invalid_argument("HTP partition needs Phone layers and sessions");
    }
    const size_t n = layers.size();
    const size_t s = std::min(capacities.size(), n);
    for (size_t k = 0; k < s; ++k) {
        if (capacities[k] == 0) {
            throw std::invalid_argument("HTP session capacity must be positive");
        }
    }
    std::vector<uint64_t> prefix(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        if (layers[i] == 0 || prefix[i] > UINT64_MAX - layers[i]) {
            throw std::invalid_argument("invalid HTP per-layer memory charge");
        }
        prefix[i + 1] = prefix[i] + layers[i];
    }

    // dp[k][j]: best maximum session utilization for the first j layers,
    // using exactly k nonempty contiguous sessions.
    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> dp(s + 1, std::vector<double>(n + 1, infinity));
    std::vector<std::vector<size_t>> previous(s + 1, std::vector<size_t>(n + 1, 0));
    dp[0][0] = 0.0;

    for (size_t k = 1; k <= s; ++k) {
        const size_t min_end = k;
        const size_t max_end = n - (s - k);
        for (size_t end = min_end; end <= max_end; ++end) {
            for (size_t start = k - 1; start < end; ++start) {
                if (dp[k - 1][start] == infinity) {
                    continue;
                }
                const uint64_t weight = prefix[end] - prefix[start];
                const uint64_t head = k == s ? output_bytes : 0;
                if (weight > UINT64_MAX - head ||
                    weight + head > capacities[k - 1]) {
                    continue;
                }
                const uint64_t charge = weight + head;
                const double use = static_cast<double>(charge) / capacities[k - 1];
                const double worst = std::max(dp[k - 1][start], use);
                if (worst < dp[k][end]) {
                    dp[k][end] = worst;
                    previous[k][end] = start;
                }
            }
        }
    }

    if (dp[s][n] == infinity) {
        throw std::runtime_error(
            "HTP_SESSION_MEMORY_INFEASIBLE: no contiguous per-session "
            "weight+KV partition fits the configured memory budgets");
    }

    placement result;
    result.ends.resize(s);
    result.used_bytes.resize(s);
    result.max_utilization = dp[s][n];
    size_t end = n;
    for (size_t k = s; k > 0; --k) {
        const size_t start = previous[k][end];
        result.ends[k - 1] = end;
        result.used_bytes[k - 1] =
            prefix[end] - prefix[start] + (k == s ? output_bytes : 0);
        end = start;
    }
    if (end != 0 || result.ends.back() != n) {
        throw std::runtime_error("HTP partition reconstruction failed");
    }
    return result;
}

} // namespace jupiter_htp
