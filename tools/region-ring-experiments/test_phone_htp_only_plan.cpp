// Standalone protocol regression test:
// c++ -std=c++17 -O2 -Wall -Wextra tools/region-ring-experiments/test_phone_htp_only_plan.cpp -o test_phone_htp_only_plan
// ./test_phone_htp_only_plan
#include "../../ggml/src/ggml-rpc/phone-owned-execution-plan.h"

#include <cassert>
#include <cstdint>

static ggml_phone_owned::region_plan make_plan(uint32_t layers) {
    using namespace ggml_phone_owned;
    region_plan p {};
    p.version = k_htp_only_plan_version;
    p.flags = flags_for_layers(layers);
    p.run_id = 1;
    p.p2c_run_id = 2;
    p.region = 1;
    p.layer = 8;
    p.total_tokens = 1855;
    p.chunk_tokens = 128;
    p.n_embd = 5120;
    p.kv_capacity = 2048;
    p.n_heads = 64;
    p.n_heads_kv = 8;
    p.head_dim = 128;
    p.n_kv = 2048;
    p.rope_type = k_rope_neox;
    p.n_ctx_orig = 32768;
    p.kv_type = k_f16;
    p.rms_norm_eps = 1e-6f;
    p.rope_freq_base = 1000000.f;
    p.rope_freq_scale = 1.f;
    p.rope_ext_factor = 0.f;
    p.rope_attn_factor = 1.f;
    p.rope_beta_fast = 32.f;
    p.rope_beta_slow = 1.f;
    return p;
}

int main() {
    using namespace ggml_phone_owned;
    for (uint32_t layers : {1u, 2u, 4u, 8u, 16u, 32u}) {
        const auto p = make_plan(layers);
        const char * why = nullptr;
        assert(validate(p, why));
        assert(layer_count(p) == layers);
        for (uint32_t i = 0; i < layers; ++i) {
            const auto child = single_layer_plan(p, i);
            assert(child.version == k_plan_version);
            assert(child.layer == p.layer + i);
            assert(layer_count(child) == 1);
            assert(validate(child, why));
        }
    }

    for (uint32_t layers : {0u, 65u}) {
        const auto p = make_plan(layers);
        const char * why = nullptr;
        assert(!validate(p, why));
    }

    // The v3 multi-layer protocol must remain backward compatible.
    auto legacy = make_plan(4);
    legacy.version = k_multi_plan_version;
    const char * why = nullptr;
    assert(validate(legacy, why));
    legacy.version = k_plan_version;
    legacy.flags = k_flag_flash_attn;
    assert(validate(legacy, why));
    legacy.flags = flags_for_layers(1);
    assert(!validate(legacy, why));
    return 0;
}
