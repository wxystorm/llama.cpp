#include "llama-hybrid-cycle-graph.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

// Pure CPU stand-in for PC CPU + Phone GPU. The actual Meta *weight*
// partitioner is used, but neither shard executes a Meta graph/AllReduce.
// Both CPU instances execute independent standard GGML FFN graphs.
static ggml_backend_meta_split_state split_ffn_weights(
        const ggml_tensor * weight, void *) {
    ggml_backend_meta_split_state ss {};
    const bool is_down = std::strncmp(weight->name, "down", 4) == 0;
    ss.axis = is_down ? GGML_BACKEND_SPLIT_AXIS_0 :
                        GGML_BACKEND_SPLIT_AXIS_1;
    ss.n_segments = 1;
    ss.nr[0] = 1;
    const int64_t full = weight->ne[ss.axis];
    ss.ne[0] = full / 2;
    ss.ne[1] = full - ss.ne[0];
    return ss;
}

static std::vector<float> run_ffn_shard(
        ggml_backend_t cpu, const llama_hybrid_cycle_ffn_shard_weights & w,
        const std::vector<float> & input,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        int64_t hidden, int64_t tokens, int64_t topk) {
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    assert(ctx);

    const auto v = llama_hybrid_cycle_make_ffn_graph_inputs(
        ctx, hidden, tokens, topk);
    auto * normalized = ggml_reshape_3d(ctx, v.norm, hidden, 1, tokens);
    auto * gate = ggml_mul_mat_id(ctx, w.gate, normalized, v.topk_ids);
    auto * up = ggml_mul_mat_id(ctx, w.up, normalized, v.topk_ids);
    auto * gated = ggml_swiglu_split(ctx, gate, up);
    auto * down = ggml_mul_mat_id(ctx, w.down, gated, v.topk_ids);
    auto * weighted = ggml_mul(ctx, down, v.topk_weights);

    // Qwen3-MoE aggregates the topk expert rows into [hidden,tokens].
    ggml_tensor * out = nullptr;
    for (int64_t k = 0; k < topk; ++k) {
        auto * view = ggml_view_2d(
            ctx, weighted, hidden, tokens, weighted->nb[2],
            size_t(k) * weighted->nb[1]);
        out = out ? ggml_add(ctx, out, view) : view;
    }
    out = ggml_cont(ctx, out);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, out);

    auto * tmp = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(cpu));
    assert(tmp);
    ggml_backend_tensor_set(v.norm, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(v.topk_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(v.topk_weights, routing.data(), 0, routing.size() * sizeof(float));
    assert(ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> values(size_t(hidden * tokens));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));

    ggml_backend_buffer_free(tmp);
    ggml_free(ctx);
    return values;
}

static void test_quantized_q4k_shard_shapes(ggml_backend_dev_t meta) {
    // Model Q4_K_M stores intermediate-neuron shards quantized. Each DOWN
    // fragment must preserve a full Q4_K block along dimension 0.
    ggml_init_params params {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    assert(ctx);

    constexpr int64_t hidden = 256;
    constexpr int64_t channels = 512;
    constexpr int64_t experts = 4;
    auto * up = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, hidden, channels, experts);
    auto * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, hidden, channels, experts);
    auto * down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_K, channels, hidden, experts);
    ggml_set_name(up, "up.quant");
    ggml_set_name(gate, "gate.quant");
    ggml_set_name(down, "down.quant");

    auto * buf = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_dev_buffer_type(meta));
    assert(buf);
    const auto pc = llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 0);
    const auto phone = llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 1);
    assert(llama_hybrid_cycle_ffn_shard_weights_valid(pc, hidden, experts));
    assert(llama_hybrid_cycle_ffn_shard_weights_valid(phone, hidden, experts));
    assert(pc.up->ne[1] + phone.up->ne[1] == channels);
    assert(pc.down->ne[0] + phone.down->ne[0] == channels);
    const int64_t qblock = ggml_blck_size(GGML_TYPE_Q4_K);
    assert(pc.down->ne[0] % qblock == 0);
    assert(phone.down->ne[0] % qblock == 0);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    assert(device);
    ggml_backend_dev_t simulated[2] = {device, device};
    ggml_backend_dev_t meta = ggml_backend_meta_device(
        simulated, 2, split_ffn_weights, nullptr);
    assert(meta);
    test_quantized_q4k_shard_shapes(meta);

    constexpr int64_t hidden = 8;
    constexpr int64_t channels = 16;
    constexpr int64_t experts = 4;
    constexpr int64_t tokens = 3;
    constexpr int64_t topk = 2;

    ggml_init_params params {};
    params.mem_size = 1024 * 1024;
    params.no_alloc = true;
    ggml_context * model = ggml_init(params);
    assert(model);
    auto * up = ggml_new_tensor_3d(model, GGML_TYPE_F32, hidden, channels, experts);
    auto * gate = ggml_new_tensor_3d(model, GGML_TYPE_F32, hidden, channels, experts);
    auto * down = ggml_new_tensor_3d(model, GGML_TYPE_F32, channels, hidden, experts);
    ggml_set_name(up, "up.exps");
    ggml_set_name(gate, "gate.exps");
    ggml_set_name(down, "down.exps");

    auto * weights = ggml_backend_alloc_ctx_tensors_from_buft(
        model, ggml_backend_dev_buffer_type(meta));
    assert(weights);

    std::vector<float> up_data(size_t(hidden * channels * experts));
    std::vector<float> gate_data(up_data.size());
    std::vector<float> down_data(size_t(channels * hidden * experts));
    for (size_t i = 0; i < up_data.size(); ++i) {
        up_data[i] = (int(i % 17) - 8) * 0.016f;
        gate_data[i] = (int(i % 19) - 9) * 0.018f;
        down_data[i] = (int(i % 23) - 11) * 0.021f;
    }
    ggml_backend_tensor_set(up, up_data.data(), 0, up_data.size() * sizeof(float));
    ggml_backend_tensor_set(gate, gate_data.data(), 0, gate_data.size() * sizeof(float));
    ggml_backend_tensor_set(down, down_data.data(), 0, down_data.size() * sizeof(float));

    const auto pc = llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 0);
    const auto phone = llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 1);
    assert(llama_hybrid_cycle_ffn_shard_weights_valid(pc, hidden, experts));
    assert(llama_hybrid_cycle_ffn_shard_weights_valid(phone, hidden, experts));
    assert(pc.up->ne[1] + phone.up->ne[1] == channels);
    assert(pc.down->ne[0] + phone.down->ne[0] == channels);
    assert(pc.up != phone.up && pc.down != phone.down);
    assert(!llama_hybrid_cycle_get_physical_ffn_weights(up, gate, down, 2).available());

    std::vector<float> input(size_t(hidden * tokens));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 0.1f * float(1 + i % 7);
    }
    const std::vector<int32_t> ids {0, 2, 1, 3, 2, 0};
    const std::vector<float> routing {0.65f, 0.35f, 0.4f, 0.6f, 0.7f, 0.3f};
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    assert(backend);
    const auto y_pc = run_ffn_shard(
        backend, pc, input, ids, routing, hidden, tokens, topk);
    const auto y_phone = run_ffn_shard(
        backend, phone, input, ids, routing, hidden, tokens, topk);

    float worst_abs = 0.0f;
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t h = 0; h < hidden; ++h) {
            float full = 0.0f;
            for (int64_t k = 0; k < topk; ++k) {
                const int32_t e = ids[size_t(t * topk + k)];
                for (int64_t f = 0; f < channels; ++f) {
                    float gate_dot = 0.0f;
                    float up_dot = 0.0f;
                    for (int64_t j = 0; j < hidden; ++j) {
                        const size_t row = size_t(e * channels * hidden + f * hidden + j);
                        gate_dot += gate_data[row] * input[size_t(t * hidden + j)];
                        up_dot += up_data[row] * input[size_t(t * hidden + j)];
                    }
                    const float silu = gate_dot / (1.0f + std::exp(-gate_dot));
                    full += routing[size_t(t * topk + k)] * down_data[
                        size_t(e * hidden * channels + h * channels + f)] * silu * up_dot;
                }
            }
            const size_t pos = size_t(t * hidden + h);
            worst_abs = std::max(worst_abs, std::fabs(
                y_pc[pos] + y_phone[pos] - full));
        }
    }
    std::printf("[CYCLE_FFN_SHARD_SUM] pc=%lld phone=%lld total=%lld "
                "worst_abs=%.9f\n",
                (long long)pc.up->ne[1],
                (long long)phone.up->ne[1],
                (long long)channels, worst_abs);
    assert(worst_abs < 1.0e-4f);

    ggml_backend_free(backend);
    ggml_backend_buffer_free(weights);
    ggml_free(model);
    return 0;
}
