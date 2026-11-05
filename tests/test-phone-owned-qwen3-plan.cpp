#include "../ggml/src/ggml-rpc/phone-owned-qwen3-plan.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ggml_phone_owned_qwen3;

static plan sample(const std::vector<uint32_t> & sizes) {
    plan p;
    p.run_id = 11798519;
    p.n_ctx = 2048;
    p.n_tokens = 0;
    for (uint32_t n : sizes) { p.n_tokens += n; }
    p.n_embd = 5120;
    p.n_layer = 40;
    p.n_head = 40;
    p.n_head_kv = 8;
    p.head_dim = 128;
    p.n_ff = 17408;
    p.rms_eps = 1e-6f;
    p.rope_base = 1000000.0f;
    p.rope_scale = 1.0f;
    for (uint32_t t = 0; t < p.n_tokens; ++t) {
        p.positions.push_back(int32_t(t));
    }
    const uint32_t regions[] = {1, 3, 5, 7};
    const uint32_t layers[] = {9, 19, 29, 39};
    for (int r = 0; r < 4; ++r) {
        uint32_t begin = 0;
        for (uint32_t c = 0; c < sizes.size(); ++c) {
            p.chunks.push_back({regions[r], layers[r], c, begin, sizes[c]});
            begin += sizes[c];
        }
    }
    return p;
}

static void roundtrip(const plan & original) {
    std::string why;
    std::vector<uint8_t> wire;
    assert(encode(original, wire, why));
    assert(wire.size() == 80 + 4 * original.n_tokens +
                          20 * original.chunks.size());
    plan decoded;
    assert(decode(wire.data(), wire.size(), decoded, why));
    assert(decoded.run_id == original.run_id);
    assert(decoded.n_tokens == original.n_tokens);
    assert(decoded.chunks.size() == original.chunks.size());
    assert(decoded.positions == original.positions);
    for (size_t i = 0; i < decoded.chunks.size(); ++i) {
        const chunk & a = original.chunks[i];
        const chunk & b = decoded.chunks[i];
        assert(a.region == b.region && a.layer == b.layer);
        assert(a.index == b.index && a.token_begin == b.token_begin);
        assert(a.n_tokens == b.n_tokens);
    }
    for (size_t i = 0; i < wire.size(); ++i) {
        // Fail closed when the packet is truncated, including header-only.
        if (i < 80 || i >= wire.size() - 10) {
            plan bad;
            assert(!decode(wire.data(), i, bad, why));
        }
    }
    plan extra;
    std::vector<uint8_t> oversized = wire;
    oversized.push_back(0);
    assert(!decode(oversized.data(), oversized.size(), extra, why));
    oversized = wire;
    oversized[0] = 0;
    assert(!decode(oversized.data(), oversized.size(), extra, why));
    oversized = wire;
    oversized[4] = 0xff;
    assert(!decode(oversized.data(), oversized.size(), extra, why));
}

int main() {
    // Different Planner-selected widths, including a non-power-of-two tail.
    roundtrip(sample({16, 32, 64, 128, 118}));
    roundtrip(sample({128, 128, 128, 128}));
    roundtrip(sample({64}));
    roundtrip(sample({1, 2, 3, 4, 6}));

    std::string why;
    auto p = sample({64, 96, 128, 70});
    assert(validate(p, why));
    p.chunks[3].n_tokens = 69;  // gap, not full coverage
    assert(!validate(p, why));
    p = sample({64, 96, 128, 70});
    p.chunks[2].token_begin = 161;  // uninitialized hole
    assert(!validate(p, why));
    p = sample({64, 96, 128, 70});
    p.chunks[1].layer = 20;  // wrong layer for region
    assert(!validate(p, why));
    p = sample({64, 96, 128, 70});
    p.chunks[1].index = 0;  // duplicate sequence number
    assert(!validate(p, why));
    p = sample({64, 96, 128, 70});
    p.positions[200] += 1;  // cannot infer missing KV map
    assert(!validate(p, why));
    p = sample({64, 96, 128, 70});
    p.positions.back() = p.n_ctx;
    assert(!validate(p, why));
    p = sample({64, 96, 128, 70});
    p.n_head_kv = 3;  // invalid GQA grouping
    assert(!validate(p, why));
    p = sample({64, 96, 128, 70});
    p.rms_eps = 0.0f;
    assert(!validate(p, why));
    p = sample({64, 96, 128, 70});
    p.chunks[8].region = 1;  // region order regression
    assert(!validate(p, why));

    std::puts("test-phone-owned-qwen3-plan PASS");
    return 0;
}
