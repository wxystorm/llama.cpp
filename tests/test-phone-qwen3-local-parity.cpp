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
    std::puts("test-phone-qwen3-local-parity PASS");
    return 0;
}
