#include "llama-hybrid.h"

#include <cassert>
#include <vector>

int main() {
    assert((llama_hybrid_split_chunks(18, 4) == std::vector<int>{ 5, 5, 4, 4 }));
    assert(llama_hybrid_split_chunks(3, 4).empty());

    llama_hybrid_profile profile;
    profile.probe_tokens           = 16;
    profile.probe_chunk_min_tokens = 4;
    assert((llama_hybrid_probe_chunk_tokens(profile) == std::vector<int>{ 4, 8, 16 }));

    const int64_t pc_size    = llama_hybrid_ffn_shard_size(11008, GGML_TYPE_Q4_0, 0.5f, 0);
    const int64_t phone_size = llama_hybrid_ffn_shard_size(11008, GGML_TYPE_Q4_0, 0.5f, 1);
    assert(pc_size == 5504);
    assert(pc_size + phone_size == 11008);

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(cpu != nullptr);

    llama_hybrid_ffn_desc desc;
    desc.n_embd    = 128;
    desc.n_ff      = 512;
    desc.gate_type = GGML_TYPE_F32;
    desc.up_type   = GGML_TYPE_F32;
    desc.down_type = GGML_TYPE_F32;

    profile.probe_tokens           = 4;
    profile.probe_chunk_min_tokens = 2;
    assert(llama_hybrid_profile_ffn(profile, desc, cpu, cpu));
    assert(profile.cpu_ffn.size() == 8);
    assert(profile.phone_ffn.size() == 8);

    ggml_backend_free(cpu);

    return 0;
}
