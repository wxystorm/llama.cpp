// Hardware-free regression for physical K/V allocation lengths. The full
// PC/Phone FIFO integration test covers causal Attention with a live model.
// Here we verify that the old 128 cap has actually been lifted through the
// physical Tensor allocation layer, without any implicit truncation.
#include "llama-hybrid-cycle-physical.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdint>

int main() {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        std::fprintf(stderr,
            "[CYCLE_KV_CAPACITY_TEST] status=FAIL reason=CPU_INIT\n");
        return 1;
    }

    ggml_init_params params {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(cpu);
        return 1;
    }

    // The physical Cache uses the two Qwen3-MoE weight dimensions, but no
    // actual weight bytes or model file are needed for this shape test.
    ggml_tensor * k_norm = ggml_new_tensor_1d(
        ctx, GGML_TYPE_F32, 128);
    ggml_tensor * wk = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, 2048, 512);
    ggml_backend_buffer_t weights = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(cpu));
    if (!k_norm || !wk || !weights) {
        if (weights) ggml_backend_buffer_free(weights);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        std::fprintf(stderr,
            "[CYCLE_KV_CAPACITY_TEST] status=FAIL reason=WEIGHT_BUFFER\n");
        return 1;
    }

    llama_layer layer {};
    layer.attn_k_norm = k_norm;
    layer.wk = wk;

    bool passed = true;
    const int64_t capacities[] = {128, 256, 512};
    for (const int64_t cap : capacities) {
        llama_hybrid_cycle_physical_kv_cache kv;
        const bool ok = kv.init(cpu, layer, cap, 0) &&
            kv.buffer != nullptr &&
            kv.keys != nullptr && kv.values != nullptr &&
            kv.keys->type == GGML_TYPE_F16 &&
            kv.values->type == GGML_TYPE_F16 &&
            kv.keys->ne[0] == 128 && kv.keys->ne[1] == 4 &&
            kv.keys->ne[2] == cap &&
            kv.values->ne[0] == 128 && kv.values->ne[1] == 4 &&
            kv.values->ne[2] == cap &&
            kv.max_tokens == cap &&
            kv.committed_tokens == 0;
        std::printf(
            "[CYCLE_KV_CAPACITY_TEST] tokens=%lld "
            "alloc=%s shape=%s status=%s\n",
            (long long) cap,
            kv.buffer ? "YES" : "NO",
            ok ? "EXPECTED" : "INVALID",
            ok ? "PASS" : "FAIL");
        passed = ok && passed;
    }

    llama_hybrid_cycle_physical_kv_cache rejected;
    const bool rejects_oversize = !rejected.init(cpu, layer, 513, 0);
    std::printf(
        "[CYCLE_KV_CAPACITY_TEST] tokens=513 expected=REJECT status=%s\n",
        rejects_oversize ? "PASS" : "FAIL");
    passed = passed && rejects_oversize;
    ggml_backend_buffer_free(weights);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    std::printf("[CYCLE_KV_CAPACITY_TEST] scope=CPU_PHYSICAL_ALLOC "
                "status=%s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
