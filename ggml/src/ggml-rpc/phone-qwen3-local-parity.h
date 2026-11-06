#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

// Pure host-side diagnostics. No acceptance threshold is imposed until the
// real HTP/PC-built output distribution has been measured.
namespace ggml_phone_qwen3 {

struct parity_stats {
    size_t count = 0;
    size_t nonfinite = 0;
    size_t max_index = 0;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rmse = 0.0;
    double cosine = 0.0;
    double reference_rms = 0.0;
    double candidate_rms = 0.0;
};

// Returns false if either input is malformed or a non-finite value is found.
// Keep finite-sample metrics even on failure, but NEVER label the result
// as parity PASS merely because the graph compute returned success.
inline bool compare_outputs(const float * reference,
                            const float * candidate,
                            size_t count,
                            parity_stats & result) {
    result = {};
    if (reference == nullptr || candidate == nullptr || count == 0) {
        return false;
    }
    result.count = count;
    long double sum_abs = 0.0L;
    long double sum_sq = 0.0L;
    long double sum_ref_sq = 0.0L;
    long double sum_cand_sq = 0.0L;
    long double dot = 0.0L;
    for (size_t i = 0; i < count; ++i) {
        const double a = reference[i];
        const double b = candidate[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            ++result.nonfinite;
            continue;
        }
        const double diff = std::abs(a - b);
        if (diff > result.max_abs) {
            result.max_abs = diff;
            result.max_index = i;
        }
        sum_abs += diff;
        sum_sq += diff * diff;
        sum_ref_sq += a * a;
        sum_cand_sq += b * b;
        dot += a * b;
    }
    if (result.nonfinite != 0) {
        return false;
    }
    const long double n = static_cast<long double>(count);
    result.mean_abs = static_cast<double>(sum_abs / n);
    result.rmse = std::sqrt(static_cast<double>(sum_sq / n));
    result.reference_rms = std::sqrt(static_cast<double>(sum_ref_sq / n));
    result.candidate_rms = std::sqrt(static_cast<double>(sum_cand_sq / n));
    if (sum_ref_sq == 0.0L || sum_cand_sq == 0.0L) {
        result.cosine = sum_ref_sq == sum_cand_sq ? 1.0 : 0.0;
    } else {
        result.cosine = static_cast<double>(
            dot / std::sqrt(sum_ref_sq * sum_cand_sq));
    }
    return true;
}

} // namespace ggml_phone_qwen3
