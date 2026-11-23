#include "../ggml/src/ggml-rpc/phone-owned-execution-plan.h"
#include "../ggml/src/ggml-rpc/phone-owned-cache-policy.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

using namespace ggml_phone_owned;

#define CHECK(expr) do { if (!(expr)) { \
    std::fprintf(stderr, "check failed: %s at line %d\n", #expr, __LINE__); \
    std::abort(); \
} } while (false)

static region_plan make_plan() {
    region_plan p {};
    p.run_id = 42;
    p.p2c_run_id = 91;
    p.region = 1;
    p.layer = 9;
    p.total_tokens = 512;
    p.chunk_tokens = 128;
    p.n_embd = 5120;
    p.position_base = 0;
    p.kv_capacity = 2048;
    p.n_heads = 40;
    p.n_heads_kv = 8;
    p.head_dim = 128;
    p.n_kv = 512;
    p.rope_type = k_rope_neox;
    p.n_ctx_orig = 32768;
    p.kv_type = k_f16;
    p.rms_norm_eps = 1.0e-6f;
    p.rope_freq_base = 1000000.0f;
    p.rope_freq_scale = 1.0f;
    p.rope_ext_factor = -1.0f;
    p.rope_attn_factor = 1.0f;
    p.rope_beta_fast = 32.0f;
    p.rope_beta_slow = 1.0f;
    return p;
}

int main() {
    const char * reason = nullptr;
    const region_plan a = make_plan();
    CHECK(validate(a, reason));
    // Qwen3-32B: 5120 hidden features, 64 query heads, 128 head dim.
    auto wide = a;
    wide.region = 15;
    wide.layer = 63;
    wide.n_heads = 64;
    CHECK(validate(wide, reason));
    CHECK(owned_task_uid(15, 0) != owned_task_uid(1, 0));
    CHECK(chunk_count(a) == 4);
    // Versioned contiguous NPU Region: one external Region/Chunk can
    // dispatch several ordered, separately cached whole-layer graphs.
    auto chain = a;
    chain.version = k_multi_plan_version;
    chain.flags = flags_for_layers(3);
    chain.region = 5;
    chain.layer = 23;
    CHECK(validate(chain, reason));
    CHECK(layer_count(chain) == 3);
    // Reuse a precompiled Graph across requests only with identical
    // execution geometry; a new transport run ID alone must not invalidate it.
    auto new_request = chain;
    new_request.run_id = 2000;
    new_request.p2c_run_id = 2001;
    CHECK(!equal(chain, new_request));
    CHECK(same_execution_template(chain, new_request));
    // Different Prompt shape uses another fixed Graph template; the cache
    // must keep multiple layouts only within its measured HTP byte budget.
    const uint64_t mib = UINT64_C(1024) * 1024;
    auto short_plan = chain;
    short_plan.total_tokens = 175;
    short_plan.chunk_tokens = 88;
    auto alt_plan = short_plan;
    alt_plan.total_tokens = 176;
    CHECK(!same_execution_template(short_plan, alt_plan));
    CHECK(estimate_resident_bytes(short_plan, 700*mib, alt_plan) >= 700*mib);
    CHECK(!must_evict_generation(0, 2, 0, 700*mib, 2048*mib));
    CHECK(!must_evict_generation(1, 2, 700*mib, 900*mib, 2048*mib));
    CHECK(must_evict_generation(2, 2, 1400*mib, 100*mib, 2048*mib));
    CHECK(must_evict_generation(1, 2, 1400*mib, 700*mib, 2048*mib));
    CHECK(must_evict_generation(1, 2, 1500*mib, UINT64_MAX, 2048*mib));
    const auto larger_plan = short_plan;
    auto long_plan = larger_plan;
    long_plan.total_tokens = 512;
    long_plan.chunk_tokens = 128;
    CHECK(estimate_resident_bytes(short_plan, 700*mib, long_plan) >
          estimate_resident_bytes(short_plan, 700*mib, alt_plan));
    CHECK(estimate_resident_bytes(short_plan, 0, long_plan) == UINT64_MAX);

    new_request.total_tokens = 256;
    CHECK(!same_execution_template(chain, new_request));
    new_request = chain;
    new_request.run_id = 2000;
    new_request.chunk_tokens = 64;
    CHECK(!same_execution_template(chain, new_request));
    new_request = chain;
    new_request.run_id = 2000;
    new_request.kv_capacity = 4096;
    CHECK(!same_execution_template(chain, new_request));
    new_request = chain;
    new_request.run_id = 2000;
    new_request.rope_freq_scale = 0.5f;
    CHECK(!same_execution_template(chain, new_request));

    // Jupiter's native suffix can now be 1..8 layers with a bounded,
    // context-dependent Chunk/KV geometry; the original 8-layer/512 pilot
    // remains valid while longer prompts require appropriately sized Chunks.
    for (uint32_t layers = 1; layers <= 8; ++layers) {
        for (uint32_t ctx : {512u, 1024u, 2048u, 4096u}) {
            auto native = a;
            native.version = layers == 1 ? k_plan_version : k_multi_plan_version;
            native.flags = layers == 1 ? k_flag_flash_attn : flags_for_layers(layers);
            native.layer = 40 - layers;
            native.region = 0;
            native.total_tokens = ctx;
            native.chunk_tokens = ctx == 4096 ? 256 : 128;
            native.kv_capacity = ctx;
            native.n_kv = ctx;
            CHECK(validate(native, reason));
            CHECK(layer_count(native) == layers);
            CHECK(chunk_count(native) <= 16);
            for (uint32_t i = 0; i < layers; ++i) {
                const auto child = single_layer_plan(native, i);
                CHECK(child.layer == 40 - layers + i);
                CHECK(validate(child, reason));
            }
        }
    }

    CHECK(chunk_count(chain) == 4);
    CHECK(internal_layer_region(chain, 0) != chain.region);
    CHECK(internal_layer_region(chain, 0) !=
          internal_layer_region(chain, 1));
    for (uint32_t off = 0; off < 3; ++off) {
        const auto child = single_layer_plan(chain, off);
        CHECK(child.version == k_plan_version);
        CHECK(child.flags == k_flag_flash_attn);
        CHECK(child.layer == 23 + off);
        CHECK(child.region == internal_layer_region(chain, off));
        CHECK(child.region < k_max_regions);
        CHECK(layer_count(child) == 1);
        CHECK(validate(child, reason));
        CHECK(chunk_count(child) == chunk_count(chain));
    }
    auto separate_chain = chain;
    separate_chain.region = 6;
    CHECK(internal_layer_region(separate_chain, 0) !=
          internal_layer_region(chain, 0));
    auto invalid_chain = chain;
    invalid_chain.flags = flags_for_layers(1);
    CHECK(!validate(invalid_chain, reason));
    invalid_chain.flags = flags_for_layers(65);
    CHECK(!validate(invalid_chain, reason));
    invalid_chain = chain;
    invalid_chain.layer = 254;
    CHECK(!validate(invalid_chain, reason));
    invalid_chain = chain;
    invalid_chain.flags = flags_for_layers(3) | 2;
    CHECK(!validate(invalid_chain, reason));
    const auto legacy_child = single_layer_plan(a, 0);
    CHECK(legacy_child.region == a.region);
    CHECK(legacy_child.layer == a.layer);

    CHECK(owned_task_uid(1, 0) != 0);
    CHECK(owned_task_uid(1, 0) != owned_task_uid(1, 1));
    CHECK(owned_task_uid(1, 1) != owned_task_uid(3, 1));
    CHECK(valid_direct_host_bytes(128ULL*5120*4, 128ULL*5120*4));
    CHECK(!valid_direct_host_bytes(128ULL*5120*4, 128ULL*5120*4 - 4));
    CHECK(!valid_direct_host_bytes(0, 0));
    CHECK(!valid_direct_host_bytes(7, 7));
    CHECK(!valid_direct_host_bytes(k_max_activation_bytes + 4,
                                   k_max_activation_bytes + 4));
    CHECK(matches_chunk(a, 0, 0, 128, 91, 128ULL*5120*4, 128ULL*5120*4));
    CHECK(matches_chunk(a, 3, 384, 128, 91, 128ULL*5120*4, 128ULL*5120*4));
    CHECK(!matches_chunk(a, 4, 512, 128, 91, 128ULL*5120*4, 128ULL*5120*4));
    CHECK(!matches_chunk(a, 1, 128, 128, 92, 128ULL*5120*4, 128ULL*5120*4));
    CHECK(!matches_chunk(a, 1, 128, 128, 91, 128ULL*5120*4 - 4, 128ULL*5120*4));
    // Local-only execution may synchronize KV to the main decode cache
    // only if the physical PC cache slots are 0..N-1 in every Chunk.
    const int64_t first[] = { 0, 1, 2, 3 };
    const int64_t next[]  = { 128, 129, 130, 131 };
    const int64_t reused[] = { 128, 129, 7, 131 };
    CHECK(identity_kv_slots(first, 4, 0));
    CHECK(identity_kv_slots(next, 4, 128));
    CHECK(!identity_kv_slots(next, 4, 127));
    CHECK(!identity_kv_slots(reused, 4, 128));
    CHECK(!identity_kv_slots(nullptr, 4, 0));
    CHECK(!identity_kv_slots(first, 0, 0));
    CHECK(!identity_kv_slots(first, 4, UINT64_MAX));
    registry plans;
    CHECK(plans.add(a, reason));
    CHECK(plans.add(a, reason)); // idempotent re-registration
    CHECK(plans.size() == 1);
    auto b = a;
    b.layer = 19;
    CHECK(!plans.add(b, reason));
    CHECK(std::strcmp(reason, "CONFLICTING_REGION_PLAN") == 0);
    b = a;
    b.region = 3;
    b.layer = 19;
    CHECK(plans.add(b, reason));
    CHECK(plans.size() == 2);
    CHECK(plans.count_run(42) == 2);
    CHECK(plans.count_run(41) == 0);

    auto invalid = a;
    invalid.version = 3;
    CHECK(!validate(invalid, reason));
    // V3 represents one physical P3 Region with three independent HTP
    // layer graphs and KV stores. Internal IDs cannot collide with
    // physical [0,64) Region IDs or one another.
    auto three = a;
    three.version = k_multi_plan_version;
    three.flags = flags_for_layers(3);
    three.region = 5;
    three.layer = 23;
    CHECK(validate(three, reason));
    CHECK(layer_count(three) == 3);
    const auto c0 = single_layer_plan(three, 0);
    const auto c1 = single_layer_plan(three, 1);
    const auto c2 = single_layer_plan(three, 2);
    CHECK(validate(c0, reason));
    CHECK(validate(c1, reason));
    CHECK(validate(c2, reason));
    CHECK(c0.version == k_plan_version);
    CHECK(c0.layer == 23 && c1.layer == 24 && c2.layer == 25);
    CHECK(c0.region >= k_wire_region_limit);
    CHECK(c0.region != c1.region && c1.region != c2.region);
    CHECK(internal_layer_region(three, 1) !=
          internal_layer_region(a, 0));
    CHECK(internal_layer_region(a, 0) == a.region);
    CHECK(matches_chunk(three, 3, 384, 128, 91,
                        128ULL*5120*4, 128ULL*5120*4));
    auto bad_multi = three;
    bad_multi.flags = flags_for_layers(1);
    CHECK(!validate(bad_multi, reason));
    bad_multi.flags = flags_for_layers(65);
    CHECK(!validate(bad_multi, reason));
    bad_multi = three;
    bad_multi.layer = 255;
    CHECK(!validate(bad_multi, reason));
    bad_multi = three;
    bad_multi.flags |= 2;
    CHECK(!validate(bad_multi, reason));

    invalid = a;
    invalid.flags = 0;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.position_base = 2000;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.n_embd = 0;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.n_heads = 63; // not divisible by 8 KV heads
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.n_heads = 1024; // 1024*128 exceeds output width budget
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.chunk_tokens = 0;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.n_kv = 256;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.rope_freq_base = 0;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.chunk_tokens = 100;
    CHECK(validate(invalid, reason));
    CHECK(chunk_count(invalid) == 6);
    CHECK(matches_chunk(invalid, 5, 500, 12, 91, 12ULL*5120*4, 12ULL*5120*4));
    CHECK(!matches_chunk(invalid, 5, 500, 100, 91, 100ULL*5120*4, 100ULL*5120*4));

    plans.erase_run(41);
    CHECK(plans.size() == 2);
    CHECK(plans.count_run(42) == 2);
    plans.erase_run(42);
    CHECK(plans.size() == 0);
    CHECK(plans.count_run(42) == 0);
    return 0;
}
