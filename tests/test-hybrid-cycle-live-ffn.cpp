// Optional live backend probe. It is NOT a llama_context decode test:
// it exercises two physical Qwen3-MoE FFN weight shards and device compute
// using PC CPU + an actual phone RPC device, then checks their single-owner
// PC CPU sum against a full-FFN reference. No old Meta AllReduce is invoked.
//
// Run manually with an RPC server exposing ONLY the phone GPU:
//   test-hybrid-cycle-live-ffn.exe <phone-ip:port> [remote-device-index]
//
// The READY data below is deterministic host-side test data. Actual Phone
// Attention/Router production and READY/PARTIAL data-plane wiring are next.
#include "llama-hybrid-cycle-ffn-executor.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <string>
#include <vector>

static ggml_backend_meta_split_state split_weights(
        const ggml_tensor * tensor, void *) {
    ggml_backend_meta_split_state s {};
    s.axis = std::strncmp(tensor->name, "down.", 5) == 0 ?
        GGML_BACKEND_SPLIT_AXIS_0 : GGML_BACKEND_SPLIT_AXIS_1;
    s.n_segments = 1;
    s.nr[0] = 1;
    const int64_t full = tensor->ne[s.axis];
    s.ne[0] = full / 2;
    s.ne[1] = full - s.ne[0];
    return s;
}

static bool execute_ffn(
        ggml_backend_t backend,
        const llama_hybrid_cycle_ffn_shard_weights & shard,
        const std::vector<float> & norm,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        int64_t hidden,
        int64_t tokens,
        int64_t topk,
        int64_t experts,
        std::vector<float> & partial) {
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    auto input = llama_hybrid_cycle_make_ffn_graph_inputs(ctx, hidden, tokens, topk);
    auto * normalized = ggml_reshape_3d(ctx, input.norm, hidden, 1, tokens);
    auto * gate = ggml_mul_mat_id(ctx, shard.gate, normalized, input.topk_ids);
    auto * up = ggml_mul_mat_id(ctx, shard.up, normalized, input.topk_ids);
    auto * activation = ggml_swiglu_split(ctx, gate, up);
    auto * projected = ggml_mul_mat_id(ctx, shard.down, activation, input.topk_ids);
    auto * weighted = ggml_mul(ctx, projected, input.topk_weights);

    // Sum the top-k expert outputs without introducing any residual ADD.
    ggml_tensor * summed = nullptr;
    for (int64_t k = 0; k < topk; ++k) {
        auto * v = ggml_view_2d(ctx, weighted, hidden, tokens, weighted->nb[2],
                               size_t(k) * weighted->nb[1]);
        summed = summed ? ggml_add(ctx, summed, v) : v;
    }
    input.ffn_output = ggml_cont(ctx, summed);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, input.ffn_output);

    bool ok = false;
    {
        llama_hybrid_cycle_ffn_executor executor;
        if (executor.prepare(backend, ctx, graph, input, shard,
                             hidden, tokens, topk, experts)) {
            partial.resize(size_t(hidden * tokens));
            ok = executor.execute(norm.data(), norm.size(),
                                  ids.data(), ids.size(),
                                  routing.data(), routing.size(),
                                  partial.data(), partial.size());
        }
    } // Free executor allocations before destroying graph/context.
    ggml_free(ctx);
    return ok;
}

static float reference_ffn(
        const std::vector<float> & up,
        const std::vector<float> & gate,
        const std::vector<float> & down,
        const std::vector<float> & norm,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        int64_t hidden, int64_t channels, int64_t topk,
        int64_t token, int64_t output) {
    float value = 0.0f;
    for (int64_t k = 0; k < topk; ++k) {
        const int64_t expert = ids[size_t(token * topk + k)];
        const float route = routing[size_t(token * topk + k)];
        for (int64_t f = 0; f < channels; ++f) {
            float g = 0.0f, u = 0.0f;
            for (int64_t i = 0; i < hidden; ++i) {
                const size_t loc = size_t(expert * channels * hidden + f * hidden + i);
                g += gate[loc] * norm[size_t(token * hidden + i)];
                u += up[loc] * norm[size_t(token * hidden + i)];
            }
            const float silu = g / (1.0f + std::exp(-g));
            value += route * silu * u *
                down[size_t(expert * hidden * channels + output * channels + f)];
        }
    }
    return value;
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr,
            "Usage: test-hybrid-cycle-live-ffn <phone-ip:rpc-port> [phone-gpu-device-index]\n"
            "Run a compatible Phone OpenCL-only RPC server first. Not a full P2C decode.\n");
        return 2;
    }
    const char * endpoint = argv[1];
    char * end = nullptr;
    const long index = argc == 3 ? std::strtol(argv[2], &end, 10) : 0;
    if (index < 0 || index > 15 || (argc == 3 && (!end || *end != '\0'))) {
        std::fprintf(stderr, "Bad RPC device index\n");
        return 2;
    }

    ggml_backend_load_all();
    const auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const auto remote_reg = ggml_backend_rpc_add_server(endpoint);
    if (!cpu_dev || !remote_reg ||
        size_t(index) >= ggml_backend_reg_dev_count(remote_reg)) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=RPC_UNAVAILABLE\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(remote_reg, size_t(index));
    if (!phone_dev || ggml_backend_dev_type(phone_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=WRONG_DEVICE\n");
        return 1;
    }
    auto * cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    auto * phone = ggml_backend_dev_init(phone_dev, nullptr);
    if (!cpu || !phone || !ggml_backend_is_rpc(phone)) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=BACKEND_INIT\n");
        return 1;
    }
    std::printf("[CYCLE_LIVE_FFN] endpoint=%s device=%ld cpu=%s phone=%s\n",
        endpoint, index, ggml_backend_name(cpu), ggml_backend_name(phone));

    const ggml_backend_dev_t devices[] = {cpu_dev, phone_dev};
    const auto meta = ggml_backend_meta_device(
        const_cast<ggml_backend_dev_t *>(devices), 2, split_weights, nullptr);
    if (!meta) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=META_DEVICE\n");
        return 1;
    }

    constexpr int64_t hidden = 64;
    constexpr int64_t channels = 128;
    constexpr int64_t experts = 4;
    constexpr int64_t tokens = 4;
    constexpr int64_t topk = 2;
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * model_ctx = ggml_init(params);
    if (!model_ctx) {
        return 1;
    }

    auto * up = ggml_new_tensor_3d(model_ctx, GGML_TYPE_F32, hidden, channels, experts);
    auto * gate = ggml_new_tensor_3d(model_ctx, GGML_TYPE_F32, hidden, channels, experts);
    auto * down = ggml_new_tensor_3d(model_ctx, GGML_TYPE_F32, channels, hidden, experts);
    ggml_set_name(up, "up.exps");
    ggml_set_name(gate, "gate.exps");
    ggml_set_name(down, "down.exps");

    auto * weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(
        model_ctx, ggml_backend_dev_buffer_type(meta));
    if (!weight_buf) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=MODEL_WEIGHT_ALLOC\n");
        return 1;
    }

    std::vector<float> up_host(size_t(hidden * channels * experts));
    std::vector<float> gate_host(up_host.size());
    std::vector<float> down_host(size_t(hidden * channels * experts));
    for (size_t i = 0; i < up_host.size(); ++i) {
        up_host[i] = (int(i % 29) - 14) * 0.005f;
        gate_host[i] = (int(i % 31) - 15) * 0.006f;
        down_host[i] = (int(i % 37) - 18) * 0.007f;
    }
    ggml_backend_tensor_set(up, up_host.data(), 0, up_host.size() * sizeof(float));
    ggml_backend_tensor_set(gate, gate_host.data(), 0, gate_host.size() * sizeof(float));
    ggml_backend_tensor_set(down, down_host.data(), 0, down_host.size() * sizeof(float));

    if (!llama_hybrid_cycle_ffn_partition_valid(up, gate, down, hidden, experts)) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=SHARD_PARTITION\n");
        return 1;
    }
    const auto w_pc = llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 0);
    const auto w_phone = llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 1);

    std::vector<float> norm(size_t(hidden * tokens));
    for (size_t i = 0; i < norm.size(); ++i) {
        norm[i] = (int(i % 13) - 6) * 0.045f;
    }
    const std::vector<int32_t> ids {0, 2, 1, 3, 2, 0, 3, 1};
    const std::vector<float> routing {0.75f, 0.25f, 0.3f, 0.7f, 0.55f, 0.45f, 0.2f, 0.8f};
    std::vector<float> residual(norm.size(), 0.125f);

    const auto start = std::chrono::steady_clock::now();
    // Each worker binds a single device and holds only its own weight slice.
    auto phone_task = std::async(std::launch::async, [&]() {
        std::vector<float> result;
        if (!execute_ffn(phone, w_phone, norm, ids, routing,
                         hidden, tokens, topk, experts, result)) {
            result.clear();
        }
        return result;
    });
    std::vector<float> pc_part;
    const bool pc_ok = execute_ffn(cpu, w_pc, norm, ids, routing,
                                   hidden, tokens, topk, experts, pc_part);
    const auto phone_part = phone_task.get();
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    if (!pc_ok || pc_part.size() != norm.size() || phone_part.size() != norm.size()) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=PHYSICAL_GRAPH_COMPUTE\n");
        return 1;
    }

    float max_abs = 0.0f, max_ref = 0.0f;
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < hidden; ++h) {
            const size_t pos = size_t(t * hidden + h);
            const float ref = residual[pos] +
                reference_ffn(up_host, gate_host, down_host, norm, ids, routing,
                              hidden, channels, topk, t, h);
            const float actual = residual[pos] + (pc_part[pos] + phone_part[pos]);
            max_ref = std::max(max_ref, std::fabs(ref));
            max_abs = std::max(max_abs, std::fabs(ref - actual));
        }
    }
    const float tolerance = 0.0001f + 0.002f * max_ref;
    const bool pass = max_abs <= tolerance;
    std::printf("[CYCLE_LIVE_FFN_SUM] device=CPU+PHONE_RPC split_pc=%lld split_phone=%lld "
                "tokens=%lld topk=%lld wall_ms=%.3f max_abs=%.8f tolerance=%.8f "
                "data_plane=RPC_BACKEND_ONLY status=%s\n",
        (long long)w_pc.up->ne[1], (long long)w_phone.up->ne[1],
        (long long)tokens, (long long)topk, wall_ms, max_abs, tolerance,
        pass ? "PASS" : "FAIL");

    // Async work must finish before freeing static weights and devices.
    ggml_backend_buffer_free(weight_buf);
    ggml_free(model_ctx);
    ggml_backend_free(phone);
    ggml_backend_free(cpu);
    return pass ? 0 : 1;
}
