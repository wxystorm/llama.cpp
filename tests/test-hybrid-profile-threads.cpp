#include "common.h"
#include "llama.h"

#include <cstdio>

#define REQUIRE(x) do { if (!(x)) { std::fprintf(stderr, "HYBRID_PROFILE_THREADS_TEST_FAIL line=%d expr=%s\n", __LINE__, #x); return 1; } } while (0)

int main() {
    // Unspecified direct API calls retain the CPU backend default.
    REQUIRE(llama_model_default_params().hybrid_profile_n_threads_batch == 0);

    common_params params;
    params.cpuparams.n_threads = 8;
    params.cpuparams_batch.n_threads = -1;
    auto m = common_model_params_to_llama(params);
    auto c = common_context_params_to_llama(params);
    REQUIRE(m.hybrid_profile_n_threads_batch == 8);
    REQUIRE(m.hybrid_profile_n_threads_batch == c.n_threads_batch);

    // Explicit batch threads override generation threads.
    params.cpuparams_batch.n_threads = 6;
    m = common_model_params_to_llama(params);
    c = common_context_params_to_llama(params);
    REQUIRE(m.hybrid_profile_n_threads_batch == 6);
    REQUIRE(m.hybrid_profile_n_threads_batch == c.n_threads_batch);

    params.cpuparams.n_threads = -1;
    params.cpuparams_batch.n_threads = -1;
    m = common_model_params_to_llama(params);
    REQUIRE(m.hybrid_profile_n_threads_batch == 0);

    return 0;
}
