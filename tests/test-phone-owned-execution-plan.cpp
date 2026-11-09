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
    p.layer_end = 10;
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
    CHECK(chunk_count(a) == 4);
    auto n3 = a;
    n3.layer_end = 12;
    CHECK(validate(n3, reason));
    CHECK(!equal(a, n3));
    n3.layer_end = n3.layer;
    CHECK(!validate(n3, reason));
    n3.layer_end = 42;
    CHECK(!validate(n3, reason));
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
    b.layer_end = 20;
    CHECK(plans.add(b, reason));
    CHECK(plans.size() == 2);
    CHECK(plans.count_run(42) == 2);
    CHECK(plans.count_run(41) == 0);

    auto invalid = a;
    invalid.version = 2;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.flags = 0;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.position_base = 2000;
    CHECK(!validate(invalid, reason));
    invalid = a;
    invalid.n_embd = 4096;
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
