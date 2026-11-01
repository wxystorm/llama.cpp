#include "common.h"
#include "llama.h"
#include "llama-hybrid-profile-cpu-threadpool.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

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

    // No GGUF, CUDA, or Phone server: exercise the same scoped profile pool
    // through two real CPU graph computes. Its identity must stay unchanged.
    ggml_backend_load_all();
    ggml_backend_dev_t device =
        ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    REQUIRE(device != nullptr);
    ggml_backend_t cpu = ggml_backend_dev_init(device, nullptr);
    REQUIRE(cpu != nullptr);

    {
        llama_hybrid_profile_cpu_threadpool pool;
        REQUIRE(!pool.attach(nullptr, 8));
        REQUIRE(!pool.attach(cpu, 0));
        REQUIRE(pool.attach(cpu, 8));
        REQUIRE(pool.attached());
        REQUIRE(!pool.attach(cpu, 8));
        const ggml_threadpool_t original_pool = pool.get();

        ggml_init_params init = { 1024 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(init);
        REQUIRE(ctx != nullptr);

        ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
        ggml_tensor * sum = ggml_add(ctx, a, b);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, sum);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        REQUIRE(buffer != nullptr);

        for (int run = 0; run < 2; ++run) {
            const float lhs[8] = {1, 2, 3, 4, 5, 6, 7, 8};
            const float rhs[8] = {10, 10, 10, 10, 10, 10, 10, 10};
            float result[8] = {};
            ggml_backend_tensor_set(a, lhs, 0, sizeof(lhs));
            ggml_backend_tensor_set(b, rhs, 0, sizeof(rhs));
            REQUIRE(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(sum, result, 0, sizeof(result));
            REQUIRE(result[0] == 11.0f);
            REQUIRE(result[7] == 18.0f);
            REQUIRE(pool.get() == original_pool);
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        pool.reset();
        REQUIRE(!pool.attached());
    }

    ggml_backend_free(cpu);
    return 0;
}
