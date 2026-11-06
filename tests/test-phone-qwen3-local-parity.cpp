#include "../ggml/src/ggml-rpc/phone-qwen3-local-parity.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "CHECK failed: %s at line %d\n", #x, __LINE__); \
    std::abort(); \
} } while (false)

int main() {
    using ggml_phone_qwen3::compare_outputs;
    using ggml_phone_qwen3::parity_stats;
    parity_stats s {};
    float reference[] = {1.0f, -2.0f, 3.0f, 4.0f};
    float equal[]     = {1.0f, -2.0f, 3.0f, 4.0f};
    CHECK(compare_outputs(reference, equal, 4, s));
    CHECK(s.count == 4 && s.nonfinite == 0 && s.max_abs == 0);
    CHECK(std::abs(s.cosine - 1.0) < 1.e-12);
    CHECK(s.rmse == 0.0);

    float changed[]   = {1.0f, -1.0f, 3.0f, 4.0f};
    CHECK(compare_outputs(reference, changed, 4, s));
    CHECK(s.max_abs == 1.0 && s.max_index == 1);
    CHECK(std::abs(s.rmse - 0.5) < 1.e-12);
    CHECK(std::abs(s.mean_abs - 0.25) < 1.e-12);

    CHECK(!compare_outputs(nullptr, changed, 4, s));
    CHECK(!compare_outputs(reference, changed, 0, s));

    changed[2] = std::numeric_limits<float>::quiet_NaN();
    CHECK(!compare_outputs(reference, changed, 4, s));
    CHECK(s.nonfinite == 1);

    float zeros[] = {0.f,0.f,0.f,0.f};
    CHECK(compare_outputs(zeros, zeros, 4, s));
    CHECK(s.cosine == 1.0);
    CHECK(compare_outputs(zeros, equal, 4, s));
    CHECK(s.cosine == 0.0);
    // The maximum absolute error is at feature 2 in four consecutive
    // tokens. Attribute its 16/17 share of squared error and compute a
    // feature-excluded RMSE/cosine over the other 7 channels.
    float feature_reference[32];
    float feature_candidate[32];
    for (int i = 0; i < 32; ++i) {
        feature_reference[i] = 1.0f;
        feature_candidate[i] = 1.0f;
    }
    for (int token = 0; token < 4; ++token) {
        feature_candidate[token * 8 + 2] = 3.0f;
    }
    feature_candidate[1 * 8 + 7] = 2.0f;
    CHECK(compare_outputs(feature_reference, feature_candidate, 32, s));
    CHECK(s.max_index == 2 && s.max_abs == 2);
    ggml_phone_qwen3::feature_error_stats feature {};
    CHECK(ggml_phone_qwen3::compare_feature_errors(
        feature_reference, feature_candidate, 32, 8, s, feature));
    CHECK(feature.feature == 2 && feature.worst_token == 0);
    CHECK(feature.dominant_feature == 2);
    CHECK(feature.reference_at_worst == 1.0);
    CHECK(feature.candidate_at_worst == 3.0);
    CHECK(std::abs(feature.feature_mse_share - 16.0/17.0) < 1e-12);
    CHECK(std::abs(feature.dominant_feature_mse_share - 16.0/17.0) < 1e-12);
    CHECK(std::abs(feature.feature_rmse - 2.0) < 1e-12);
    CHECK(std::abs(feature.other_rmse - std::sqrt(1.0 / 28.0)) < 1e-12);
    CHECK(feature.max_abs_excluding_feature == 1.0);
    CHECK(feature.other_cosine > 0.98);
    CHECK(!ggml_phone_qwen3::compare_feature_errors(
        feature_reference, feature_candidate, 32, 1, s, feature));
    CHECK(!ggml_phone_qwen3::compare_feature_errors(
        feature_reference, feature_candidate, 31, 8, s, feature));

    std::puts("test-phone-qwen3-local-parity PASS");
    return 0;
}
