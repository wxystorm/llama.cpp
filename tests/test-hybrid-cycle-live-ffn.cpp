// Optional live backend probe. It is NOT a llama_context decode test:
// it exercises two physical Qwen3-MoE FFN weight shards and device compute
// using PC CPU + an actual phone RPC device, then checks their single-owner
// PC CPU sum against a full-FFN reference. No old Meta AllReduce is invoked.
//
// Run manually with an RPC server exposing ONLY the phone GPU:
//   test-hybrid-cycle-live-ffn.exe <phone-ip:port> [remote-device-index] [q4_0|q4_k|f32]
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

// GGML OpenCL MoE MUL_MAT_ID kernels are principally designed for
// quantized expert weights. The former F32-only live test could trigger an
// unsupported OpenCL path while the RPC device stub reported all ops as
// supported. Pack the generated weights just as a real quantized model does.
// Dequantize into ref for an apples-to-apples full-FFN numeric comparison.
// The Meta buffer's generic set_tensor path scatters one MoE expert at a
// time. That is correct for raw buffers, but an OpenCL Q4_0/Q4_K weight set
// converts the *entire* physical tensor to a device-specific quantized layout.
// Re-entering set_tensor for the next expert would interpret its converted
// tensor->extra as an ordinary OpenCL buffer descriptor (CL_INVALID_MEM_OBJECT).
//
// Keep Meta responsible for allocating the actual physical weight shards;
// gather their packed bytes on the host and upload each complete shard ONCE.
// This probe never invokes the old Meta graph executor or its AllReduce.
static bool upload_cycle_test_weight(
        ggml_tensor * tensor, std::vector<float> & ref) {
    if (!tensor || int64_t(ref.size()) != ggml_nelements(tensor) ||
        tensor->ne[2] <= 0 || tensor->ne[3] != 1) {
        return false;
    }

    auto * pc = ggml_backend_meta_get_static_weight_shard(tensor, 0);
    auto * phone = ggml_backend_meta_get_static_weight_shard(tensor, 1);
    if (!pc || !phone || !pc->buffer || !phone->buffer ||
        pc->type != tensor->type || phone->type != tensor->type ||
        pc->ne[2] != tensor->ne[2] || phone->ne[2] != tensor->ne[2] ||
        pc->ne[3] != 1 || phone->ne[3] != 1 ||
        !ggml_is_contiguous(pc) || !ggml_is_contiguous(phone)) {
        return false;
    }

    // Quantize first, then use the dequantized values as the numerical
    // reference. Physical shard extraction below operates on the exact
    // packed bytes, so it cannot introduce another quantization difference.
    std::vector<uint8_t> packed(ggml_nbytes(tensor));
    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(packed.data(), ref.data(), packed.size());
    } else {
        const auto * traits = ggml_get_type_traits(tensor->type);
        if (!traits || !traits->to_float || !ggml_is_quantized(tensor->type)) {
            return false;
        }
        const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
        const int64_t rows = ggml_nrows(tensor);
        const size_t wrote = ggml_quantize_chunk(tensor->type, ref.data(),
            packed.data(), 0, rows, tensor->ne[0], nullptr);
        if (wrote != packed.size() ||
            packed.size() != row_bytes * size_t(rows)) {
            return false;
        }
        for (int64_t row = 0; row < rows; ++row) {
            traits->to_float(
                packed.data() + size_t(row) * row_bytes,
                ref.data() + size_t(row) * size_t(tensor->ne[0]),
                tensor->ne[0]);
        }
    }

    std::vector<uint8_t> pc_bytes(ggml_nbytes(pc));
    std::vector<uint8_t> phone_bytes(ggml_nbytes(phone));
    const bool split_axis_0 = std::strncmp(tensor->name, "down.", 5) == 0;
    const size_t experts = size_t(tensor->ne[2]);
    if (split_axis_0) {
        // down: [full_channels, hidden, experts]; split the quantized blocks
        // of each row along dimension 0. Each physical shard remains aligned
        // to its quantization block size (Q4_0:32, Q4_K:256).
        if (pc->ne[0] + phone->ne[0] != tensor->ne[0] ||
            pc->ne[1] != tensor->ne[1] ||
            phone->ne[1] != tensor->ne[1]) {
            return false;
        }
        const size_t rows = size_t(tensor->ne[1]) * experts;
        const size_t full_row = ggml_row_size(tensor->type, tensor->ne[0]);
        const size_t pc_row = ggml_row_size(pc->type, pc->ne[0]);
        const size_t phone_row = ggml_row_size(phone->type, phone->ne[0]);
        if (pc_row + phone_row != full_row ||
            pc_bytes.size() != rows * pc_row ||
            phone_bytes.size() != rows * phone_row) {
            return false;
        }
        for (size_t row = 0; row < rows; ++row) {
            const uint8_t * src = packed.data() + row * full_row;
            std::memcpy(pc_bytes.data() + row * pc_row, src, pc_row);
            std::memcpy(phone_bytes.data() + row * phone_row,
                        src + pc_row, phone_row);
        }
    } else {
        // gate/up: [hidden, full_channels, experts]; rows belonging to
        // adjacent intermediate channels are contiguous within each expert.
        if (pc->ne[0] != tensor->ne[0] ||
            phone->ne[0] != tensor->ne[0] ||
            pc->ne[1] + phone->ne[1] != tensor->ne[1]) {
            return false;
        }
        const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
        const size_t pc_expert_bytes = size_t(pc->ne[1]) * row_bytes;
        const size_t phone_expert_bytes = size_t(phone->ne[1]) * row_bytes;
        const size_t full_expert_bytes = size_t(tensor->ne[1]) * row_bytes;
        if (pc_expert_bytes + phone_expert_bytes != full_expert_bytes ||
            pc_bytes.size() != experts * pc_expert_bytes ||
            phone_bytes.size() != experts * phone_expert_bytes) {
            return false;
        }
        for (size_t e = 0; e < experts; ++e) {
            const uint8_t * src = packed.data() + e * full_expert_bytes;
            std::memcpy(pc_bytes.data() + e * pc_expert_bytes,
                        src, pc_expert_bytes);
            std::memcpy(phone_bytes.data() + e * phone_expert_bytes,
                        src + pc_expert_bytes, phone_expert_bytes);
        }
    }

    // No Meta scatter / sub-tensor partial writes reach the OpenCL backend.
    ggml_backend_tensor_set(pc, pc_bytes.data(), 0, pc_bytes.size());
    ggml_backend_tensor_set(phone, phone_bytes.data(), 0, phone_bytes.size());
    std::printf("[CYCLE_LIVE_FFN_WEIGHT] tensor=%s type=%s "
                "pc_bytes=%zu phone_bytes=%zu upload=FULL_PHYSICAL_SHARD\n",
                tensor->name, ggml_type_name(tensor->type),
                pc_bytes.size(), phone_bytes.size());
    return true;
}

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
    if (argc < 2 || argc > 4) {
        std::fprintf(stderr,
            "Usage: test-hybrid-cycle-live-ffn <phone-ip:rpc-port> [phone-gpu-device-index] [q4_0|q4_k|f32]\n"
            "Default q4_0 exercises OpenCL's quantized MoE kernel. Not a full P2C decode.\n");
        return 2;
    }
    const char * endpoint = argv[1];
    char * end = nullptr;
    const long index = argc >= 3 ? std::strtol(argv[2], &end, 10) : 0;
    if (index < 0 || index > 15 || (argc >= 3 && (!end || *end != '\0'))) {
        std::fprintf(stderr, "Bad RPC device index\n");
        return 2;
    }

    const char * mode = argc == 4 ? argv[3] : "q4_0";
    const ggml_type expert_type =
        std::strcmp(mode, "q4_0") == 0 ? GGML_TYPE_Q4_0 :
        std::strcmp(mode, "q4_k") == 0 ? GGML_TYPE_Q4_K :
        std::strcmp(mode, "f32") == 0 ? GGML_TYPE_F32 :
        GGML_TYPE_COUNT;
    if (expert_type == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "Unrecognized expert weight type: %s\n", mode);
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

    ggml_backend_dev_t devices[] = {cpu_dev, phone_dev};
    const auto meta = ggml_backend_meta_device(
        devices, 2, split_weights, nullptr);
    if (!meta) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=META_DEVICE\n");
        return 1;
    }

    const int64_t hidden   = expert_type == GGML_TYPE_Q4_K ? 256 : 64;
    const int64_t channels = expert_type == GGML_TYPE_Q4_K ? 512 : 128;
    constexpr int64_t experts = 4;
    const int64_t tokens = expert_type == GGML_TYPE_Q4_K ? 1 : 4;
    constexpr int64_t topk = 2;
    std::printf("[CYCLE_LIVE_FFN_CONFIG] weight_type=%s hidden=%lld channels=%lld tokens=%lld\n",
        ggml_type_name(expert_type), (long long)hidden,
        (long long)channels, (long long)tokens);
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * model_ctx = ggml_init(params);
    if (!model_ctx) {
        return 1;
    }

    auto * up = ggml_new_tensor_3d(model_ctx, expert_type, hidden, channels, experts);
    auto * gate = ggml_new_tensor_3d(model_ctx, expert_type, hidden, channels, experts);
    auto * down = ggml_new_tensor_3d(model_ctx, expert_type, channels, hidden, experts);
    // Q4_K MUL_MAT_ID has no generic OpenCL fallback. The Adreno MoE
    // support and upload paths currently recognize expert weights by the
    // joint presence of "ffn" and "exps" in the physical tensor name
    // (plus ne[1] % 32 == 0). Keep the Q4_0 control names unchanged.
    // Retain the "down." prefix because split_weights() uses it to select
    // dimension 0 for the down projection.
    const bool adreno_moe_name = expert_type == GGML_TYPE_Q4_K;
    ggml_set_name(up, adreno_moe_name ? "up.ffn.exps" : "up.exps");
    ggml_set_name(gate, adreno_moe_name ? "gate.ffn.exps" : "gate.exps");
    ggml_set_name(down, adreno_moe_name ? "down.ffn.exps" : "down.exps");

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
    if (!upload_cycle_test_weight(up, up_host) ||
        !upload_cycle_test_weight(gate, gate_host) ||
        !upload_cycle_test_weight(down, down_host)) {
        std::fprintf(stderr, "[CYCLE_LIVE_FFN] status=FAIL reason=WEIGHT_QUANTIZE_OR_UPLOAD\n");
        return 1;
    }

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
    const std::vector<int32_t> ids = tokens == 1 ?
        std::vector<int32_t>{0, 2} :
        std::vector<int32_t>{0, 2, 1, 3, 2, 0, 3, 1};
    const std::vector<float> routing = tokens == 1 ?
        std::vector<float>{0.75f, 0.25f} :
        std::vector<float>{0.75f, 0.25f, 0.3f, 0.7f, 0.55f, 0.45f, 0.2f, 0.8f};
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
