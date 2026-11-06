#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

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


struct feature_error_stats {
    size_t feature = 0;
    size_t worst_token = 0;
    size_t dominant_feature = 0;
    double reference_at_worst = 0.0;
    double candidate_at_worst = 0.0;
    double feature_rmse = 0.0;
    double other_rmse = 0.0;
    double other_cosine = 0.0;
    double feature_mse_share = 0.0;
    double dominant_feature_mse_share = 0.0;
    double max_abs_excluding_feature = 0.0;
};

// This makes an outlier hidden channel measurable rather than treating the
// largest absolute element as representative of the full layer.
// The layout is GGML F32 [n_embd, n_tokens], so i % n_embd is its feature ID.
// The selected feature is the one at the global maximum error. Independently
// identify the feature carrying the largest TOTAL squared error to distinguish
// a single spike from a persistent channel-wide mismatch.
inline bool compare_feature_errors(const float * reference,
                                   const float * candidate,
                                   size_t count, size_t n_embd,
                                   const parity_stats & global,
                                   feature_error_stats & out) {
    out = {};
    if (!reference || !candidate || n_embd <= 1 ||
            count <= n_embd || count % n_embd != 0 ||
            global.count != count || global.nonfinite != 0 ||
            global.max_index >= count) {
        return false;
    }
    out.feature = global.max_index % n_embd;
    out.worst_token = global.max_index / n_embd;
    out.reference_at_worst = reference[global.max_index];
    out.candidate_at_worst = candidate[global.max_index];
    long double focused_error_sq = 0.0L;
    long double remaining_error_sq = 0.0L;
    long double remaining_ref_sq = 0.0L;
    long double remaining_cand_sq = 0.0L;
    long double remaining_dot = 0.0L;
    std::vector<long double> by_feature(n_embd, 0.0L);
    for (size_t i = 0; i < count; ++i) {
        const double a = reference[i];
        const double b = candidate[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            return false;
        }
        const long double delta = static_cast<long double>(a) - b;
        const long double err_sq = delta * delta;
        const size_t feature = i % n_embd;
        by_feature[feature] += err_sq;
        if (feature == out.feature) {
            focused_error_sq += err_sq;
        } else {
            remaining_error_sq += err_sq;
            remaining_ref_sq += static_cast<long double>(a) * a;
            remaining_cand_sq += static_cast<long double>(b) * b;
            remaining_dot += static_cast<long double>(a) * b;
            out.max_abs_excluding_feature =
                std::max(out.max_abs_excluding_feature,
                         static_cast<double>(std::abs(delta)));
        }
    }
    const long double total_error_sq =
        focused_error_sq + remaining_error_sq;
    const size_t tokens = count / n_embd;
    out.feature_rmse = std::sqrt(static_cast<double>(
        focused_error_sq / tokens));
    out.other_rmse = std::sqrt(static_cast<double>(
        remaining_error_sq / (count - tokens)));
    out.feature_mse_share = total_error_sq == 0.0L ? 0.0 :
        static_cast<double>(focused_error_sq / total_error_sq);
    const auto dominant = std::max_element(
        by_feature.begin(), by_feature.end());
    out.dominant_feature = static_cast<size_t>(
        dominant - by_feature.begin());
    out.dominant_feature_mse_share = total_error_sq == 0.0L ? 0.0 :
        static_cast<double>(*dominant / total_error_sq);
    if (remaining_ref_sq == 0.0L || remaining_cand_sq == 0.0L) {
        out.other_cosine = remaining_ref_sq == remaining_cand_sq ?
            1.0 : 0.0;
    } else {
        out.other_cosine = static_cast<double>(remaining_dot /
            std::sqrt(remaining_ref_sq * remaining_cand_sq));
    }
    return true;
}

} // namespace ggml_phone_qwen3
