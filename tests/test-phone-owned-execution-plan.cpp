#include "../ggml/src/ggml-rpc/phone-owned-execution-plan.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

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
