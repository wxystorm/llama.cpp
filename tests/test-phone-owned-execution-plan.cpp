#include "../ggml/src/ggml-rpc/phone-owned-execution-plan.h"

#include <cassert>
#include <cstring>

using namespace ggml_phone_owned;

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
    return p;
}

int main() {
    const char * reason = nullptr;
    const region_plan a = make_plan();
    assert(validate(a, reason));
    assert(chunk_count(a) == 4);
    assert(matches_chunk(a, 0, 0, 128, 91, 128ULL*5120*4, 128ULL*5120*4));
    assert(matches_chunk(a, 3, 384, 128, 91, 128ULL*5120*4, 128ULL*5120*4));
    assert(!matches_chunk(a, 4, 512, 128, 91, 128ULL*5120*4, 128ULL*5120*4));
    assert(!matches_chunk(a, 1, 128, 128, 92, 128ULL*5120*4, 128ULL*5120*4));
    assert(!matches_chunk(a, 1, 128, 128, 91, 128ULL*5120*4 - 4, 128ULL*5120*4));
    registry plans;
    assert(plans.add(a, reason));
    assert(plans.add(a, reason)); // idempotent re-registration
    assert(plans.size() == 1);
    auto b = a;
    b.layer = 19;
    assert(!plans.add(b, reason));
    assert(std::strcmp(reason, "CONFLICTING_REGION_PLAN") == 0);
    b = a;
    b.region = 3;
    b.layer = 19;
    assert(plans.add(b, reason));
    assert(plans.size() == 2);

    auto invalid = a;
    invalid.version = 2;
    assert(!validate(invalid, reason));
    invalid = a;
    invalid.flags = 1;
    assert(!validate(invalid, reason));
    invalid = a;
    invalid.position_base = 2000;
    assert(!validate(invalid, reason));
    invalid = a;
    invalid.n_embd = 4096;
    assert(!validate(invalid, reason));
    invalid = a;
    invalid.chunk_tokens = 0;
    assert(!validate(invalid, reason));
    invalid = a;
    invalid.chunk_tokens = 100;
    assert(validate(invalid, reason));
    assert(chunk_count(invalid) == 6);
    assert(matches_chunk(invalid, 5, 500, 12, 91, 12ULL*5120*4, 12ULL*5120*4));
    assert(!matches_chunk(invalid, 5, 500, 100, 91, 100ULL*5120*4, 100ULL*5120*4));

    plans.erase_run(41);
    assert(plans.size() == 2);
    plans.erase_run(42);
    assert(plans.size() == 0);
    return 0;
}
