#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

static int64_t now_us() {
    using namespace std::chrono;
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static ggml_context * make_ctx(size_t bytes = 32 * 1024 * 1024) {
    ggml_init_params params = {
        /* .mem_size   = */ bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_init(params);
}

static bool fill_f32(ggml_tensor * t, float scale = 0.01f, int seed = 0) {
    if (t == nullptr || t->type != GGML_TYPE_F32) {
        return false;
    }
    std::vector<float> data((size_t) ggml_nelements(t));
    for (size_t i = 0; i < data.size(); ++i) {
        const int v = (int) ((i + 17u * (unsigned) seed) % 23u) - 11;
        data[i] = scale * (float) v;
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    return true;
}

static bool fill_f16(ggml_tensor * t, float scale = 0.01f, int seed = 0) {
    if (t == nullptr || t->type != GGML_TYPE_F16) {
        return false;
    }
    std::vector<ggml_fp16_t> data((size_t) ggml_nelements(t));
    for (size_t i = 0; i < data.size(); ++i) {
        const int v = (int) ((i + 13u * (unsigned) seed) % 19u) - 9;
        data[i] = ggml_fp32_to_fp16(scale * (float) v);
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
    return true;
}

static bool fill_i32_routes(ggml_tensor * t, int n_expert_used) {
    if (t == nullptr || t->type != GGML_TYPE_I32 || t->ne[0] != n_expert_used) {
        return false;
    }
    std::vector<int32_t> data((size_t) ggml_nelements(t));
    for (int64_t tok = 0; tok < t->ne[1]; ++tok) {
        for (int e = 0; e < n_expert_used; ++e) {
            data[(size_t) tok * n_expert_used + e] = (int32_t) e;
        }
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int32_t));
    return true;
}

static bool fill_route_weights(ggml_tensor * t, int n_expert_used) {
    if (t == nullptr || t->type != GGML_TYPE_F32) {
        return false;
    }
    std::vector<float> data((size_t) ggml_nelements(t), 1.0f / (float) n_expert_used);
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
    return true;
}

static bool fill_q4_0_weight(ggml_tensor * t, float scale, int seed) {
    if (t == nullptr || t->type != GGML_TYPE_Q4_0) {
        return false;
    }

    const int64_t k = t->ne[0];
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, k);
    const size_t total_bytes = ggml_nbytes(t);
    if (row_bytes == 0 || total_bytes % row_bytes != 0) {
        return false;
    }

    std::vector<float> row((size_t) k);
    for (int64_t i = 0; i < k; ++i) {
        const int v = (int) ((i + 7 * seed) % 17) - 8;
        row[(size_t) i] = scale * (float) v;
    }

    std::vector<uint8_t> qrow(row_bytes);
    const size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q4_0, row.data(), qrow.data(), 0, 1, k, nullptr);
    if (written != row_bytes) {
        fprintf(stderr, "q4_0 row quantization mismatch: got=%zu expected=%zu\n",
                written, row_bytes);
        return false;
    }

    std::vector<uint8_t> all(total_bytes);
    const size_t rows = total_bytes / row_bytes;
    for (size_t r = 0; r < rows; ++r) {
        memcpy(all.data() + r * row_bytes, qrow.data(), row_bytes);
    }

    ggml_backend_tensor_set(t, all.data(), 0, all.size());
    return true;
}

static bool check_graph_support(const char * tag, ggml_backend_t backend, ggml_cgraph * gf) {
    bool ok = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        ggml_tensor * node = ggml_graph_node(gf, i);
        if (!ggml_backend_supports_op(backend, node)) {
            fprintf(stderr,
                    "[PIPE_SUPPORT_FAIL] stage=%s index=%d op=%s name=%s\n",
                    tag, i, ggml_op_desc(node), node->name);
            ok = false;
        }
    }
    return ok;
}

struct gpu_stage {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * route_ids = nullptr;
    ggml_tensor * route_weights = nullptr;

    std::array<ggml_tensor *, 2> hidden_out = {nullptr, nullptr};
    std::array<ggml_cgraph *, 2> front_graph = {nullptr, nullptr};

    std::array<ggml_tensor *, 2> return_in = {nullptr, nullptr};
    std::array<ggml_tensor *, 2> residual = {nullptr, nullptr};
    std::array<ggml_tensor *, 2> reduce_out = {nullptr, nullptr};
    std::array<ggml_cgraph *, 2> reduce_graph = {nullptr, nullptr};
};

struct npu_stage {
    ggml_context * ctx_weights = nullptr;
    ggml_context * ctx_act = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_backend_buffer_t act_buf = nullptr;

    ggml_tensor * w_up = nullptr;
    ggml_tensor * w_gate = nullptr;
    ggml_tensor * w_down = nullptr;

    std::array<ggml_tensor *, 2> hidden_in = {nullptr, nullptr};
    std::array<ggml_tensor *, 2> route_ids = {nullptr, nullptr};
    std::array<ggml_tensor *, 2> route_weights = {nullptr, nullptr};
    std::array<ggml_tensor *, 2> out = {nullptr, nullptr};
    std::array<ggml_cgraph *, 2> graph = {nullptr, nullptr};
};

static bool build_gpu_stage(
        gpu_stage & s,
        ggml_backend_t gpu,
        int n_embd,
        int n_tokens,
        int kv) {
    s.ctx = make_ctx();
    if (s.ctx == nullptr) {
        return false;
    }

    constexpr int head_dim = 128;
    constexpr int n_kv_head = 4;
    constexpr int q_per_kv = 8;
    constexpr int n_q_head = n_kv_head * q_per_kv;

    s.q = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F32,
                             head_dim, n_tokens, n_q_head, 1);
    s.k = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16,
                             head_dim, kv, n_kv_head, 1);
    s.v = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16,
                             head_dim, kv, n_kv_head, 1);
    s.mask = ggml_new_tensor_4d(s.ctx, GGML_TYPE_F16,
                                kv, n_tokens, 1, 1);

    s.route_ids = ggml_new_tensor_2d(s.ctx, GGML_TYPE_I32, 8, n_tokens);
    s.route_weights = ggml_new_tensor_3d(s.ctx, GGML_TYPE_F32, 1, 8, n_tokens);

    ggml_set_name(s.q, "pipe_q");
    ggml_set_name(s.k, "pipe_k");
    ggml_set_name(s.v, "pipe_v");
    ggml_set_name(s.mask, "pipe_mask");
    ggml_set_name(s.route_ids, "pipe_route_ids");
    ggml_set_name(s.route_weights, "pipe_route_weights");

    for (int slot = 0; slot < 2; ++slot) {
        ggml_tensor * fa = ggml_flash_attn_ext(
            s.ctx, s.q, s.k, s.v, s.mask,
            1.0f / sqrtf((float) head_dim), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);

        ggml_tensor * flat = ggml_reshape_2d(
            s.ctx, fa, head_dim * n_q_head, n_tokens);

        GGML_ASSERT(flat->ne[0] >= n_embd);
        ggml_tensor * half = ggml_view_2d(
            s.ctx, flat, n_embd, n_tokens, flat->nb[1], 0);

        s.hidden_out[slot] = ggml_cont(s.ctx, half);
        ggml_format_name(s.hidden_out[slot], "pipe_gpu_hidden_%d", slot);

        s.front_graph[slot] = ggml_new_graph_custom(s.ctx, 1024, false);
        ggml_build_forward_expand(s.front_graph[slot], s.hidden_out[slot]);

        s.return_in[slot] = ggml_new_tensor_2d(
            s.ctx, GGML_TYPE_F32, n_embd, n_tokens);
        s.residual[slot] = ggml_new_tensor_2d(
            s.ctx, GGML_TYPE_F32, n_embd, n_tokens);
        s.reduce_out[slot] = ggml_add(
            s.ctx, s.return_in[slot], s.residual[slot]);

        ggml_format_name(s.return_in[slot], "pipe_gpu_return_%d", slot);
        ggml_format_name(s.residual[slot], "pipe_gpu_residual_%d", slot);
        ggml_format_name(s.reduce_out[slot], "pipe_gpu_reduce_%d", slot);

        s.reduce_graph[slot] = ggml_new_graph_custom(s.ctx, 128, false);
        ggml_build_forward_expand(s.reduce_graph[slot], s.reduce_out[slot]);
    }

    s.buf = ggml_backend_alloc_ctx_tensors(s.ctx, gpu);
    if (s.buf == nullptr) {
        return false;
    }

    if (!fill_f32(s.q, 0.005f, 1) ||
        !fill_f16(s.k, 0.005f, 2) ||
        !fill_f16(s.v, 0.005f, 3)) {
        return false;
    }

    std::vector<ggml_fp16_t> mask_data((size_t) ggml_nelements(s.mask),
                                       ggml_fp32_to_fp16(0.0f));
    ggml_backend_tensor_set(
        s.mask, mask_data.data(), 0,
        mask_data.size() * sizeof(ggml_fp16_t));

    if (!fill_i32_routes(s.route_ids, 8) ||
        !fill_route_weights(s.route_weights, 8)) {
        return false;
    }

    for (int slot = 0; slot < 2; ++slot) {
        if (!fill_f32(s.residual[slot], 0.001f, 11 + slot)) {
            return false;
        }
    }

    return true;
}

static ggml_backend_buffer_type_t get_npu_repack_buft(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_dev_get_extra_bufts");
    if (fn == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_type_t * bufts = fn(dev);
    if (bufts == nullptr) {
        return nullptr;
    }
    return bufts[0];
}

static bool build_npu_stage(
        npu_stage & s,
        ggml_backend_t npu,
        ggml_backend_dev_t npu_dev,
        int n_embd,
        int n_ff,
        int n_expert,
        int n_expert_used,
        int n_tokens) {
    s.ctx_weights = make_ctx(4 * 1024 * 1024);
    s.ctx_act = make_ctx();
    if (s.ctx_weights == nullptr || s.ctx_act == nullptr) {
        return false;
    }

    s.w_up = ggml_new_tensor_3d(
        s.ctx_weights, GGML_TYPE_Q4_0,
        n_embd, n_ff, n_expert);
    s.w_gate = ggml_new_tensor_3d(
        s.ctx_weights, GGML_TYPE_Q4_0,
        n_embd, n_ff, n_expert);
    s.w_down = ggml_new_tensor_3d(
        s.ctx_weights, GGML_TYPE_Q4_0,
        n_ff, n_embd, n_expert);

    ggml_set_name(s.w_up, "pipe_npu_up");
    ggml_set_name(s.w_gate, "pipe_npu_gate");
    ggml_set_name(s.w_down, "pipe_npu_down");

    for (int slot = 0; slot < 2; ++slot) {
        s.hidden_in[slot] = ggml_new_tensor_2d(
            s.ctx_act, GGML_TYPE_F32, n_embd, n_tokens);
        s.route_ids[slot] = ggml_new_tensor_2d(
            s.ctx_act, GGML_TYPE_I32, n_expert_used, n_tokens);
        s.route_weights[slot] = ggml_new_tensor_3d(
            s.ctx_act, GGML_TYPE_F32, 1, n_expert_used, n_tokens);

        ggml_format_name(s.hidden_in[slot], "pipe_npu_hidden_%d", slot);
        ggml_format_name(s.route_ids[slot], "pipe_npu_ids_%d", slot);
        ggml_format_name(s.route_weights[slot], "pipe_npu_weights_%d", slot);

        ggml_tensor * cur = ggml_reshape_3d(
            s.ctx_act, s.hidden_in[slot], n_embd, 1, n_tokens);

        ggml_tensor * up = ggml_mul_mat_id(
            s.ctx_act, s.w_up, cur, s.route_ids[slot]);
        ggml_tensor * gate = ggml_mul_mat_id(
            s.ctx_act, s.w_gate, cur, s.route_ids[slot]);
        ggml_format_name(up, "pipe_npu_up_out_%d", slot);
        ggml_format_name(gate, "pipe_npu_gate_out_%d", slot);

        ggml_tensor * act = ggml_swiglu_split(s.ctx_act, gate, up);
        ggml_format_name(act, "pipe_npu_swiglu_%d", slot);

        ggml_tensor * down = ggml_mul_mat_id(
            s.ctx_act, s.w_down, act, s.route_ids[slot]);
        ggml_format_name(down, "pipe_npu_down_out_%d", slot);

        ggml_tensor * weighted = ggml_mul(
            s.ctx_act, down, s.route_weights[slot]);
        ggml_format_name(weighted, "pipe_npu_weighted_%d", slot);

        std::vector<ggml_tensor *> experts;
        experts.reserve((size_t) n_expert_used);
        for (int e = 0; e < n_expert_used; ++e) {
            ggml_tensor * view = ggml_view_2d(
                s.ctx_act,
                weighted,
                n_embd,
                n_tokens,
                weighted->nb[2],
                (size_t) e * weighted->nb[1]);
            experts.push_back(view);
        }

        ggml_tensor * out = experts[0];
        for (int e = 1; e < n_expert_used; ++e) {
            out = ggml_add(s.ctx_act, out, experts[(size_t) e]);
        }
        s.out[slot] = out;
        ggml_format_name(s.out[slot], "pipe_npu_out_%d", slot);

        s.graph[slot] = ggml_new_graph_custom(s.ctx_act, 2048, false);
        ggml_build_forward_expand(s.graph[slot], s.out[slot]);
    }

    ggml_backend_buffer_type_t repack_buft =
        get_npu_repack_buft(npu_dev);
    if (repack_buft == nullptr) {
        fprintf(stderr,
                "[PIPE_ERROR] no HTP repack buffer type; "
                "run with GGML_HEXAGON_HOSTBUF=1\n");
        return false;
    }

    s.weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(
        s.ctx_weights, repack_buft);
    if (s.weight_buf == nullptr) {
        fprintf(stderr, "[PIPE_ERROR] failed to allocate HTP repack weights\n");
        return false;
    }
    ggml_backend_buffer_set_usage(
        s.weight_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    s.act_buf = ggml_backend_alloc_ctx_tensors(s.ctx_act, npu);
    if (s.act_buf == nullptr) {
        fprintf(stderr, "[PIPE_ERROR] failed to allocate HTP activation buffer\n");
        return false;
    }

    printf("[PIPE_BUFFER] npu_weight_buffer=%s host=%d "
           "npu_act_buffer=%s host=%d\n",
           ggml_backend_buffer_name(s.weight_buf),
           ggml_backend_buffer_is_host(s.weight_buf) ? 1 : 0,
           ggml_backend_buffer_name(s.act_buf),
           ggml_backend_buffer_is_host(s.act_buf) ? 1 : 0);

    printf("[PIPE_INIT] quantizing/uploading q4_0 expert weights...\n");
    if (!fill_q4_0_weight(s.w_up, 0.0020f, 1) ||
        !fill_q4_0_weight(s.w_gate, 0.0015f, 2) ||
        !fill_q4_0_weight(s.w_down, 0.0010f, 3)) {
        fprintf(stderr, "[PIPE_ERROR] failed to initialize q4_0 weights\n");
        return false;
    }

    return true;
}

struct stage_times {
    int64_t front_us = 0;
    int64_t g2n_us = 0;
    int64_t npu_us = 0;
    int64_t n2g_us = 0;
    int64_t reduce_us = 0;
};

static bool copy_gpu_to_npu(
        gpu_stage & gpu_s,
        npu_stage & npu_s,
        int slot,
        int64_t & elapsed_us) {
    const int64_t t0 = now_us();
    ggml_backend_tensor_copy(
        gpu_s.hidden_out[slot], npu_s.hidden_in[slot]);
    ggml_backend_tensor_copy(
        gpu_s.route_ids, npu_s.route_ids[slot]);
    ggml_backend_tensor_copy(
        gpu_s.route_weights, npu_s.route_weights[slot]);
    elapsed_us += now_us() - t0;
    return true;
}

static bool copy_npu_to_gpu(
        npu_stage & npu_s,
        gpu_stage & gpu_s,
        int slot,
        int64_t & elapsed_us) {
    const int64_t t0 = now_us();
    ggml_backend_tensor_copy(
        npu_s.out[slot], gpu_s.return_in[slot]);
    elapsed_us += now_us() - t0;
    return true;
}

static bool run_front(
        ggml_backend_t gpu,
        gpu_stage & s,
        int slot,
        int64_t & elapsed_us) {
    const int64_t t0 = now_us();
    const ggml_status st =
        ggml_backend_graph_compute(gpu, s.front_graph[slot]);
    elapsed_us += now_us() - t0;
    return st == GGML_STATUS_SUCCESS;
}

static bool run_reduce(
        ggml_backend_t gpu,
        gpu_stage & s,
        int slot,
        int64_t & elapsed_us) {
    const int64_t t0 = now_us();
    const ggml_status st =
        ggml_backend_graph_compute(gpu, s.reduce_graph[slot]);
    elapsed_us += now_us() - t0;
    return st == GGML_STATUS_SUCCESS;
}

static bool run_npu(
        ggml_backend_t npu,
        npu_stage & s,
        int slot,
        int64_t & elapsed_us) {
    const int64_t t0 = now_us();
    const ggml_status st =
        ggml_backend_graph_compute(npu, s.graph[slot]);
    elapsed_us += now_us() - t0;
    return st == GGML_STATUS_SUCCESS;
}

static bool warmup(
        ggml_backend_t gpu,
        ggml_backend_t npu,
        gpu_stage & gpu_s,
        npu_stage & npu_s) {
    stage_times ignored;
    for (int slot = 0; slot < 2; ++slot) {
        if (!run_front(gpu, gpu_s, slot, ignored.front_us)) return false;
        copy_gpu_to_npu(gpu_s, npu_s, slot, ignored.g2n_us);
        if (!run_npu(npu, npu_s, slot, ignored.npu_us)) return false;
        copy_npu_to_gpu(npu_s, gpu_s, slot, ignored.n2g_us);
        if (!run_reduce(gpu, gpu_s, slot, ignored.reduce_us)) return false;
    }
    return true;
}

static bool run_serial(
        int chunks,
        ggml_backend_t gpu,
        ggml_backend_t npu,
        gpu_stage & gpu_s,
        npu_stage & npu_s,
        stage_times & times,
        int64_t & wall_us) {
    const int64_t wall0 = now_us();
    for (int i = 0; i < chunks; ++i) {
        const int slot = i & 1;
        if (!run_front(gpu, gpu_s, slot, times.front_us)) return false;
        copy_gpu_to_npu(gpu_s, npu_s, slot, times.g2n_us);
        if (!run_npu(npu, npu_s, slot, times.npu_us)) return false;
        copy_npu_to_gpu(npu_s, gpu_s, slot, times.n2g_us);
        if (!run_reduce(gpu, gpu_s, slot, times.reduce_us)) return false;
    }
    wall_us = now_us() - wall0;
    return true;
}

enum class slot_state {
    FREE,
    NPU_READY,
    NPU_RUNNING,
    NPU_DONE,
};

struct pipe_slot {
    slot_state state = slot_state::FREE;
    int seq = -1;
};

static bool run_pipeline(
        int chunks,
        ggml_backend_t gpu,
        ggml_backend_t npu,
        gpu_stage & gpu_s,
        npu_stage & npu_s,
        stage_times & times,
        int64_t & wall_us) {
    std::array<pipe_slot, 2> slots;
    std::mutex mu;
    std::condition_variable cv;
    bool failed = false;
    int64_t npu_time_us = 0;

    std::thread npu_worker([&]() {
        for (int seq = 0; seq < chunks; ++seq) {
            const int slot = seq & 1;

            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&]() {
                    return failed ||
                           (slots[slot].state == slot_state::NPU_READY &&
                            slots[slot].seq == seq);
                });
                if (failed) {
                    return;
                }
                slots[slot].state = slot_state::NPU_RUNNING;
            }

            int64_t one_us = 0;
            if (!run_npu(npu, npu_s, slot, one_us)) {
                std::lock_guard<std::mutex> lock(mu);
                failed = true;
                cv.notify_all();
                return;
            }
            npu_time_us += one_us;

            {
                std::lock_guard<std::mutex> lock(mu);
                slots[slot].state = slot_state::NPU_DONE;
            }
            cv.notify_all();
        }
    });

    const int64_t wall0 = now_us();
    int produced = 0;
    int completed = 0;

    while (completed < chunks && !failed) {
        bool progress = false;

        for (int slot = 0; slot < 2; ++slot) {
            bool done = false;
            {
                std::lock_guard<std::mutex> lock(mu);
                done = slots[slot].state == slot_state::NPU_DONE;
            }
            if (!done) {
                continue;
            }

            copy_npu_to_gpu(npu_s, gpu_s, slot, times.n2g_us);
            if (!run_reduce(gpu, gpu_s, slot, times.reduce_us)) {
                std::lock_guard<std::mutex> lock(mu);
                failed = true;
                cv.notify_all();
                break;
            }

            {
                std::lock_guard<std::mutex> lock(mu);
                slots[slot].state = slot_state::FREE;
                slots[slot].seq = -1;
            }
            ++completed;
            progress = true;
            cv.notify_all();
        }

        if (failed) {
            break;
        }

        if (produced < chunks) {
            const int slot = produced & 1;
            bool is_free = false;
            {
                std::lock_guard<std::mutex> lock(mu);
                is_free = slots[slot].state == slot_state::FREE;
            }

            if (is_free) {
                if (!run_front(gpu, gpu_s, slot, times.front_us)) {
                    std::lock_guard<std::mutex> lock(mu);
                    failed = true;
                    cv.notify_all();
                    break;
                }

                copy_gpu_to_npu(gpu_s, npu_s, slot, times.g2n_us);

                {
                    std::lock_guard<std::mutex> lock(mu);
                    slots[slot].seq = produced;
                    slots[slot].state = slot_state::NPU_READY;
                }
                ++produced;
                progress = true;
                cv.notify_all();
            }
        }

        if (!progress && !failed) {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&]() {
                if (failed) return true;
                for (const auto & slot : slots) {
                    if (slot.state == slot_state::NPU_DONE) {
                        return true;
                    }
                }
                if (produced < chunks) {
                    const int slot = produced & 1;
                    if (slots[slot].state == slot_state::FREE) {
                        return true;
                    }
                }
                return false;
            });
        }
    }

    wall_us = now_us() - wall0;

    {
        std::lock_guard<std::mutex> lock(mu);
        if (failed) {
            cv.notify_all();
        }
    }
    npu_worker.join();
    times.npu_us = npu_time_us;

    return !failed && completed == chunks;
}

static std::vector<float> read_f32(ggml_tensor * t) {
    std::vector<float> data((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(
        t, data.data(), 0, data.size() * sizeof(float));
    return data;
}

static double max_abs_diff(
        const std::vector<float> & a,
        const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(
            max_diff,
            std::abs((double) a[i] - (double) b[i]));
    }
    return max_diff;
}

static void print_times(
        const char * tag,
        int chunks,
        const stage_times & t,
        int64_t wall_us) {
    const double wall_ms = wall_us / 1000.0;
    const double per_chunk_ms = wall_ms / chunks;
    printf("[%s] chunks=%d wall_ms=%.3f per_chunk_ms=%.3f "
           "front_total_ms=%.3f g2n_total_ms=%.3f "
           "npu_total_ms=%.3f n2g_total_ms=%.3f "
           "reduce_total_ms=%.3f\n",
           tag, chunks, wall_ms, per_chunk_ms,
           t.front_us / 1000.0,
           t.g2n_us / 1000.0,
           t.npu_us / 1000.0,
           t.n2g_us / 1000.0,
           t.reduce_us / 1000.0);
}

int main(int argc, char ** argv) {
    const char * gpu_name = "GPUOpenCL";
    const char * npu_name = "HTP0";
    int chunks = 8;
    int kv = 512;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            gpu_name = argv[++i];
        } else if (strcmp(argv[i], "--npu") == 0 && i + 1 < argc) {
            npu_name = argv[++i];
        } else if (strcmp(argv[i], "--chunks") == 0 && i + 1 < argc) {
            chunks = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--kv") == 0 && i + 1 < argc) {
            kv = atoi(argv[++i]);
        } else {
            fprintf(stderr,
                    "Usage: %s [--gpu GPUOpenCL] [--npu HTP0] "
                    "[--chunks N] [--kv 128|256|384|512]\n",
                    argv[0]);
            return 2;
        }
    }

    if (chunks < 2 || kv <= 0) {
        fprintf(stderr, "chunks must be >= 2 and kv must be positive\n");
        return 2;
    }

    const char * hostbuf = getenv("GGML_HEXAGON_HOSTBUF");
    if (hostbuf == nullptr || atoi(hostbuf) != 1) {
        fprintf(stderr,
                "[PIPE_ERROR] GGML_HEXAGON_HOSTBUF=1 is required "
                "for non-blocking host-visible activation slots\n");
        return 2;
    }

    constexpr int n_embd = 2048;
    constexpr int n_ff = 768;
    constexpr int n_expert = 128;
    constexpr int n_expert_used = 8;
    constexpr int n_tokens = 128;

    ggml_backend_load_all();

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_name(gpu_name);
    ggml_backend_dev_t npu_dev = ggml_backend_dev_by_name(npu_name);
    if (gpu_dev == nullptr || npu_dev == nullptr) {
        fprintf(stderr,
                "[PIPE_ERROR] missing devices gpu=%s npu=%s\n",
                gpu_name, npu_name);
        return 1;
    }

    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    ggml_backend_t npu = ggml_backend_dev_init(npu_dev, nullptr);
    if (gpu == nullptr || npu == nullptr) {
        fprintf(stderr, "[PIPE_ERROR] backend initialization failed\n");
        return 1;
    }

    printf("[PIPE_CONFIG] gpu=%s npu=%s chunks=%d kv=%d "
           "tokens=%d hidden=%d ff=%d experts=%d topk=%d hostbuf=1\n",
           gpu_name, npu_name, chunks, kv,
           n_tokens, n_embd, n_ff, n_expert, n_expert_used);

    gpu_stage gpu_s;
    npu_stage npu_s;

    bool ok =
        build_gpu_stage(gpu_s, gpu, n_embd, n_tokens, kv) &&
        build_npu_stage(
            npu_s, npu, npu_dev,
            n_embd, n_ff, n_expert, n_expert_used, n_tokens);

    if (!ok) {
        fprintf(stderr, "[PIPE_ERROR] failed to construct stages\n");
        return 1;
    }

    printf("[PIPE_BUFFER] gpu_buffer=%s host=%d "
           "hidden_bytes=%zu route_ids_bytes=%zu route_weights_bytes=%zu "
           "return_bytes=%zu\n",
           ggml_backend_buffer_name(gpu_s.buf),
           ggml_backend_buffer_is_host(gpu_s.buf) ? 1 : 0,
           ggml_nbytes(gpu_s.hidden_out[0]),
           ggml_nbytes(gpu_s.route_ids),
           ggml_nbytes(gpu_s.route_weights),
           ggml_nbytes(npu_s.out[0]));

    for (int slot = 0; slot < 2; ++slot) {
        if (!check_graph_support("GPU_FRONT", gpu, gpu_s.front_graph[slot]) ||
            !check_graph_support("GPU_REDUCE", gpu, gpu_s.reduce_graph[slot]) ||
            !check_graph_support("NPU_FFN", npu, npu_s.graph[slot])) {
            fprintf(stderr, "[PIPE_ERROR] unsupported graph node\n");
            return 1;
        }
    }

    printf("[PIPE_WARMUP] begin\n");
    if (!warmup(gpu, npu, gpu_s, npu_s)) {
        fprintf(stderr, "[PIPE_ERROR] warmup failed\n");
        return 1;
    }
    printf("[PIPE_WARMUP] done\n");

    stage_times serial_times;
    int64_t serial_wall_us = 0;
    if (!run_serial(
            chunks, gpu, npu, gpu_s, npu_s,
            serial_times, serial_wall_us)) {
        fprintf(stderr, "[PIPE_ERROR] serial baseline failed\n");
        return 1;
    }

    std::vector<float> serial_ref =
        read_f32(gpu_s.reduce_out[(chunks - 1) & 1]);

    stage_times pipeline_times;
    int64_t pipeline_wall_us = 0;
    if (!run_pipeline(
            chunks, gpu, npu, gpu_s, npu_s,
            pipeline_times, pipeline_wall_us)) {
        fprintf(stderr, "[PIPE_ERROR] pipeline failed\n");
        return 1;
    }

    std::vector<float> pipe_out0 = read_f32(gpu_s.reduce_out[0]);
    std::vector<float> pipe_out1 = read_f32(gpu_s.reduce_out[1]);

    const double diff0 = max_abs_diff(serial_ref, pipe_out0);
    const double diff1 = max_abs_diff(serial_ref, pipe_out1);
    const double speedup =
        pipeline_wall_us > 0 ?
        (double) serial_wall_us / (double) pipeline_wall_us : 0.0;

    print_times("PIPE_SERIAL", chunks, serial_times, serial_wall_us);
    print_times("PIPE_2SLOT", chunks, pipeline_times, pipeline_wall_us);

    printf("[PIPE_RESULT] speedup=%.4f "
           "serial_per_chunk_ms=%.3f pipeline_per_chunk_ms=%.3f "
           "diff_slot0=%.9g diff_slot1=%.9g\n",
           speedup,
           serial_wall_us / 1000.0 / chunks,
           pipeline_wall_us / 1000.0 / chunks,
           diff0, diff1);

    const bool numeric_ok =
        std::isfinite(diff0) && std::isfinite(diff1) &&
        diff0 <= 1e-5 && diff1 <= 1e-5;

    printf("[PIPE_DONE] ok=%d numeric_ok=%d\n",
           numeric_ok ? 1 : 0,
           numeric_ok ? 1 : 0);

    ggml_quantize_free();

    if (gpu_s.buf) ggml_backend_buffer_free(gpu_s.buf);
    if (npu_s.act_buf) ggml_backend_buffer_free(npu_s.act_buf);
    if (npu_s.weight_buf) ggml_backend_buffer_free(npu_s.weight_buf);

    if (gpu_s.ctx) ggml_free(gpu_s.ctx);
    if (npu_s.ctx_act) ggml_free(npu_s.ctx_act);
    if (npu_s.ctx_weights) ggml_free(npu_s.ctx_weights);

    ggml_backend_free(gpu);
    ggml_backend_free(npu);

    return numeric_ok ? 0 : 1;
}
