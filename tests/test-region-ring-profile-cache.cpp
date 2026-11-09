// Device-free regression for model-bound persisted Hybrid Profile snapshots.
#include "llama-region-ring-profile-cache.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#define CHECK(x) do { if (!(x)) { \
    std::fprintf(stderr, "PROFILE_CACHE_TEST_FAIL line=%d expr=%s\\n", \
                 __LINE__, #x); \
    return 1; \
} } while (0)

int main() {
    using namespace llama_ring_profile_cache;
    const std::string path = "region-ring-profile-cache-test.rrhp";
    llama_hybrid_profile profile;
    profile.n_layer = 40;
    profile.n_embd = 5120;
    profile.n_ff = 17408;
    profile.n_ctx_train = 32768;
    profile.probe_tokens = 32;
    profile.reference_tokens = 128;
    profile.profile_block_layers = 4;
    profile.cpu_layer_blocks.push_back({64, 2, 12.0, 11.0, 0.3, 0.0});
    profile.gpu_layer_blocks.push_back({64, 2, 3.2, 3.1, 0.1, 0.0});
    profile.phone_layer_blocks.push_back({64, 2, 20.5, 20.0, 0.2, 19.5});
    profile.pc_to_phone.push_back({1024, 0.5, 0.6});
    profile.phone_to_pc.push_back({1024, 0.7, 0.9});
    profile.gpu_weight_h2d.push_back({1024 * 1024, 1.9, 2.0});
    profile.layer_weight_bytes.assign(40, 256 * 1024 * 1024);
    profile.gpu_free_mem = 4ULL * 1024 * 1024 * 1024;
    profile.phone_free_mem = 6ULL * 1024 * 1024 * 1024;
    CHECK(valid(profile));
    CHECK(save(path, profile));

    llama_hybrid_profile recovered;
    CHECK(load(path, recovered));
    CHECK(recovered.n_layer == 40);
    CHECK(recovered.n_embd == profile.n_embd);
    CHECK(recovered.gpu_free_mem == profile.gpu_free_mem);
    CHECK(recovered.layer_weight_bytes.size() == 40);
    CHECK(recovered.layer_weight_bytes[39] == profile.layer_weight_bytes[39]);
    CHECK(recovered.cpu_layer_blocks.size() == 1);
    CHECK(recovered.gpu_layer_blocks.size() == 1);
    CHECK(recovered.phone_layer_blocks.size() == 1);
    CHECK(std::fabs(recovered.phone_layer_blocks[0].phone_server_ms - 19.5) < 1e-9);
    CHECK(recovered.gpu_weight_h2d.size() == 1);
    CHECK(std::fabs(recovered.gpu_weight_h2d[0].ms - 1.9) < 1e-9);

    // A corrupted header must never silently reactivate stale measurements.
    {
        std::fstream corrupt(path, std::ios::in | std::ios::out |
                                      std::ios::binary);
        CHECK(bool(corrupt));
        corrupt.put('X');
    }
    CHECK(!load(path, recovered));
    std::remove(path.c_str());

    profile.phone_layer_blocks.clear();
    CHECK(!save(path, profile));
    CHECK(!load(path, recovered));
    std::printf("REGION_RING_PROFILE_CACHE_TEST status=PASS\\n");
    return 0;
}
