#include "llama-hybrid.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"
#include "llama-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <utility>
#include <vector>

static constexpr int LLAMA_HYBRID_RPC_WARMUP_RUNS  = 2;
static constexpr int LLAMA_HYBRID_RPC_MEASURE_RUNS = 5;

static constexpr std::array<int, 3>   LLAMA_HYBRID_CHUNK_CANDIDATES = { 1, 2, 4 };
static constexpr std::array<float, 4> LLAMA_HYBRID_FFN_RATIOS       = { 0.25f, 0.50f, 0.75f, 1.00f };

std::vector<int> llama_hybrid_split_chunks(int tokens, int n_chunks) {
    if (tokens <= 0 || n_chunks <= 0 || n_chunks > tokens) {
        return {};
    }

    const int base      = tokens / n_chunks;
    const int remainder = tokens % n_chunks;

    std::vector<int> result(n_chunks, base);
    for (int i = 0; i < remainder; ++i) {
        ++result[i];
    }
    return result;
}

static std::vector<std::vector<int>> llama_hybrid_probe_chunk_layouts(const llama_hybrid_profile & profile) {
    std::vector<std::vector<int>> result;

    for (const int n_chunks : LLAMA_HYBRID_CHUNK_CANDIDATES) {
        std::vector<int> chunks = llama_hybrid_split_chunks(profile.probe_tokens, n_chunks);
        if (chunks.empty() || *std::min_element(chunks.begin(), chunks.end()) < profile.probe_chunk_min_tokens) {
            continue;
        }
        result.push_back(std::move(chunks));
    }
    return result;
}

std::vector<int> llama_hybrid_probe_chunk_tokens(const llama_hybrid_profile & profile) {
    const auto layouts = llama_hybrid_probe_chunk_layouts(profile);

    std::vector<int> result;
    for (const auto & chunks : layouts) {
        result.insert(result.end(), chunks.begin(), chunks.end());
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

static std::vector<std::pair<int, int>> llama_hybrid_probe_dual_chunks(const llama_hybrid_profile & profile) {
    const auto                    layouts = llama_hybrid_probe_chunk_layouts(profile);
    std::set<std::pair<int, int>> pairs;

    for (const auto & chunks : layouts) {
        for (size_t i = 0; i + 1 < chunks.size(); i += 2) {
            pairs.emplace(chunks[i], chunks[i + 1]);
        }
    }
    return { pairs.begin(), pairs.end() };
}

static double llama_hybrid_rpc_median(std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void * llama_hybrid_rpc_get_proc_address(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    return reg == nullptr ? nullptr : ggml_backend_reg_get_proc_address(reg, name);
}

static void llama_hybrid_transfer_print(const char * label, const std::vector<llama_hybrid_transfer_point> & points) {
    for (const auto & point : points) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE] %s bytes=%zu ms=%.3f\n", label, point.bytes, point.ms);
    }
}

void llama_hybrid_profile_print(const llama_hybrid_profile & profile) {
    LLAMA_LOG_INFO("[HYBRID_PROFILE] probe_tokens=%d probe_chunk_min_tokens=%d\n", profile.probe_tokens,
                   profile.probe_chunk_min_tokens);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] n_layer=%d n_embd=%d n_ff=%d\n", profile.n_layer, profile.n_embd, profile.n_ff);
    for (const auto & point : profile.cpu_ffn) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=CPU kind=ffn tokens=%d ratio=%.2f ms=%.3f\n", point.tokens,
                       point.ratio, point.ms);
    }
    for (const auto & point : profile.phone_ffn) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=PHONE kind=ffn tokens=%d ratio=%.2f ms=%.3f\n", point.tokens,
                       point.ratio, point.ms);
    }
    LLAMA_LOG_INFO("[HYBRID_PROFILE] cpu attn_ms=%.3f full_layer_ms=%.3f\n", profile.cpu_attn_ms,
                   profile.cpu_full_layer_ms);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] phone attn_ms=%.3f full_layer_ms=%.3f\n", profile.phone_attn_ms,
                   profile.phone_full_layer_ms);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] gpu full_layer_ms=%.3f\n", profile.gpu_full_layer_ms);

    llama_hybrid_transfer_print("pc_to_phone", profile.pc_to_phone);
    llama_hybrid_transfer_print("phone_to_pc", profile.phone_to_pc);

    for (const auto & point : profile.dual_phone_to_pc) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE] dual_phone_to_pc bytes0=%zu bytes1=%zu wall_ms=%.3f\n", point.bytes0,
                       point.bytes1, point.wall_ms);
    }

    LLAMA_LOG_INFO("[HYBRID_PROFILE] reduce_ms=%.3f rpc_submit_ms=%.3f\n", profile.reduce_ms, profile.rpc_submit_ms);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] pc_free_mem=%zu phone_free_mem=%zu gpu_free_mem=%zu\n", profile.pc_free_mem,
                   profile.phone_free_mem, profile.gpu_free_mem);
    LLAMA_LOG_INFO(
        "[HYBRID_PROFILE] attn_weight_bytes_per_layer=%zu ffn_weight_bytes_per_layer=%zu "
        "total_weight_bytes_per_layer=%zu non_layer_weight_bytes=%zu\n",
        profile.attn_weight_bytes_per_layer, profile.ffn_weight_bytes_per_layer, profile.total_weight_bytes_per_layer,
        profile.non_layer_weight_bytes);
}

void llama_hybrid_plan_print(const llama_hybrid_plan & plan) {
    LLAMA_LOG_INFO(
        "[HYBRID_PLAN] tensor_layers=%d phone_layers=%d pc_layers=%d tensor_pc_ratio=%.3f gpu_pc_layers=%d "
        "tensor_chunks_per_ubatch=%d predicted_ms=%.3f\n",
        plan.tensor_layers, plan.phone_layers, plan.pc_layers, plan.tensor_pc_ratio, plan.gpu_pc_layers,
        plan.tensor_chunks_per_ubatch, plan.predicted_ms);
    LLAMA_LOG_INFO("[HYBRID_PLAN] pc_memory=%zu phone_memory=%zu gpu_memory=%zu\n", plan.pc_memory, plan.phone_memory,
                   plan.gpu_memory);
}

bool llama_hybrid_profile_rpc(llama_hybrid_profile & profile, ggml_backend_t pc_backend, ggml_backend_t phone_backend) {
    if (pc_backend == nullptr || phone_backend == nullptr) {
        LLAMA_LOG_ERROR("%s: backend is null\n", __func__);
        return false;
    }
    if (profile.n_embd <= 0 || profile.probe_tokens <= 0 || profile.probe_chunk_min_tokens <= 0) {
        LLAMA_LOG_ERROR("%s: invalid probe shape: n_embd=%d probe_tokens=%d probe_chunk_min_tokens=%d\n", __func__,
                        profile.n_embd, profile.probe_tokens, profile.probe_chunk_min_tokens);
        return false;
    }

    const std::vector<int> chunk_tokens = llama_hybrid_probe_chunk_tokens(profile);
    if (chunk_tokens.empty()) {
        LLAMA_LOG_ERROR("%s: no valid chunk candidates\n", __func__);
        return false;
    }

    if ((size_t) profile.n_embd > std::numeric_limits<size_t>::max() / sizeof(float) / (size_t) chunk_tokens.back()) {
        LLAMA_LOG_ERROR("%s: probe tensor size overflow\n", __func__);
        return false;
    }

    const std::vector<std::pair<int, int>> dual_chunks = llama_hybrid_probe_dual_chunks(profile);

    const size_t           n_tensors  = chunk_tokens.size() + 2 * dual_chunks.size();
    const size_t           ctx_size   = n_tensors * ggml_tensor_overhead();
    const ggml_init_params ctx_params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };

    ggml_context_ptr pc_ctx(ggml_init(ctx_params));
    ggml_context_ptr phone_ctx(ggml_init(ctx_params));
    if (!pc_ctx || !phone_ctx) {
        LLAMA_LOG_ERROR("%s: failed to create tensor contexts\n", __func__);
        return false;
    }

    std::vector<ggml_tensor *> pc_tensors;
    std::vector<ggml_tensor *> phone_tensors;
    pc_tensors.reserve(chunk_tokens.size());
    phone_tensors.reserve(chunk_tokens.size());

    for (const int tokens : chunk_tokens) {
        pc_tensors.push_back(ggml_new_tensor_2d(pc_ctx.get(), GGML_TYPE_F32, profile.n_embd, tokens));
        phone_tensors.push_back(ggml_new_tensor_2d(phone_ctx.get(), GGML_TYPE_F32, profile.n_embd, tokens));
    }

    std::vector<std::array<ggml_tensor *, 2>> dual_pc_tensors(dual_chunks.size());
    std::vector<std::array<ggml_tensor *, 2>> dual_phone_tensors(dual_chunks.size());
    for (size_t i = 0; i < dual_chunks.size(); ++i) {
        const std::array<int, 2> tokens = { dual_chunks[i].first, dual_chunks[i].second };
        for (size_t lane = 0; lane < 2; ++lane) {
            dual_pc_tensors[i][lane] = ggml_new_tensor_2d(pc_ctx.get(), GGML_TYPE_F32, profile.n_embd, tokens[lane]);
            dual_phone_tensors[i][lane] =
                ggml_new_tensor_2d(phone_ctx.get(), GGML_TYPE_F32, profile.n_embd, tokens[lane]);
        }
    }

    ggml_backend_buffer_ptr pc_buffer(ggml_backend_alloc_ctx_tensors(pc_ctx.get(), pc_backend));
    ggml_backend_buffer_ptr phone_buffer(ggml_backend_alloc_ctx_tensors(phone_ctx.get(), phone_backend));
    if (!pc_buffer || !phone_buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate probe tensors\n", __func__);
        return false;
    }

    profile.pc_to_phone.clear();
    profile.phone_to_pc.clear();
    profile.dual_phone_to_pc.clear();
    profile.pc_to_phone.reserve(chunk_tokens.size());
    profile.phone_to_pc.reserve(chunk_tokens.size());

    const size_t         max_bytes = (size_t) profile.n_embd * (size_t) profile.probe_tokens * sizeof(float);
    std::vector<uint8_t> source_data(max_bytes);

    LLAMA_LOG_INFO("[HYBRID_PROFILE_RPC] pc=%s phone=%s\n", ggml_backend_name(pc_backend),
                   ggml_backend_name(phone_backend));
    LLAMA_LOG_INFO("[HYBRID_PROFILE_RPC] warmup begin sizes=%zu warmup=%d measure=%d\n", chunk_tokens.size(),
                   LLAMA_HYBRID_RPC_WARMUP_RUNS, LLAMA_HYBRID_RPC_MEASURE_RUNS);

    for (size_t i = 0; i < chunk_tokens.size(); ++i) {
        ggml_tensor * pc_tensor    = pc_tensors[i];
        ggml_tensor * phone_tensor = phone_tensors[i];
        const size_t  bytes        = ggml_nbytes(pc_tensor);

        std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> pc_to_phone_samples;
        std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> phone_to_pc_samples;

        for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
            std::fill_n(source_data.begin(), bytes, (uint8_t) (run + 1));
            ggml_backend_tensor_set(pc_tensor, source_data.data(), 0, bytes);
            const int64_t begin_us = ggml_time_us();
            ggml_backend_tensor_copy(pc_tensor, phone_tensor);
            const int64_t end_us = ggml_time_us();

            if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
                pc_to_phone_samples[run - LLAMA_HYBRID_RPC_WARMUP_RUNS] = (end_us - begin_us) / 1000.0;
            }
        }

        for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
            std::fill_n(source_data.begin(), bytes, (uint8_t) (run + 17));
            ggml_backend_tensor_set(phone_tensor, source_data.data(), 0, bytes);
            const int64_t begin_us = ggml_time_us();
            ggml_backend_tensor_copy(phone_tensor, pc_tensor);
            const int64_t end_us = ggml_time_us();

            if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
                phone_to_pc_samples[run - LLAMA_HYBRID_RPC_WARMUP_RUNS] = (end_us - begin_us) / 1000.0;
            }
        }

        profile.pc_to_phone.push_back({ bytes, llama_hybrid_rpc_median(pc_to_phone_samples) });
        profile.phone_to_pc.push_back({ bytes, llama_hybrid_rpc_median(phone_to_pc_samples) });
    }

    if (!dual_chunks.empty()) {
        const auto snapshot_arm = reinterpret_cast<ggml_backend_rpc_snapshot_arm_t>(
            llama_hybrid_rpc_get_proc_address(phone_backend, GGML_BACKEND_RPC_SNAPSHOT_ARM_PROC));
        const auto set_snapshot_read = reinterpret_cast<ggml_backend_rpc_set_snapshot_read_t>(
            llama_hybrid_rpc_get_proc_address(phone_backend, GGML_BACKEND_RPC_SET_SNAPSHOT_READ_PROC));
        if (snapshot_arm == nullptr || set_snapshot_read == nullptr) {
            LLAMA_LOG_ERROR("%s: phone backend does not support RPC snapshots\n", __func__);
            return false;
        }

        static std::atomic<uint64_t> next_snapshot_seq{ uint64_t(1) << 63 };
        for (size_t pair_index = 0; pair_index < dual_chunks.size(); ++pair_index) {
            const std::array<size_t, 2> dual_bytes = {
                ggml_nbytes(dual_phone_tensors[pair_index][0]),
                ggml_nbytes(dual_phone_tensors[pair_index][1]),
            };
            std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> dual_samples;

            LLAMA_LOG_INFO("[HYBRID_PROFILE_RPC] dual warmup begin bytes0=%zu bytes1=%zu\n", dual_bytes[0],
                           dual_bytes[1]);

            for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
                std::array<uint64_t, 2> snapshot_seq;
                for (size_t lane = 0; lane < 2; ++lane) {
                    std::fill_n(source_data.begin(), dual_bytes[lane], (uint8_t) (run + 33 + lane));
                    ggml_backend_tensor_set(dual_phone_tensors[pair_index][lane], source_data.data(), 0,
                                            dual_bytes[lane]);
                    snapshot_seq[lane] = next_snapshot_seq.fetch_add(1, std::memory_order_relaxed);
                    if (!snapshot_arm(phone_backend, dual_phone_tensors[pair_index][lane], 0, dual_bytes[lane],
                                      (uint32_t) lane, snapshot_seq[lane])) {
                        LLAMA_LOG_ERROR("%s: failed to arm snapshot lane %zu\n", __func__, lane);
                        return false;
                    }
                }

                std::mutex              start_mutex;
                std::condition_variable start_cv;
                int                     ready = 0;
                bool                    start = false;
                std::array<int64_t, 2>  lane_begin_us{ 0, 0 };
                std::array<int64_t, 2>  lane_end_us{ 0, 0 };

                std::array<std::thread, 2> workers;
                for (size_t lane = 0; lane < workers.size(); ++lane) {
                    workers[lane] = std::thread([&, lane]() {
                        {
                            std::unique_lock<std::mutex> lock(start_mutex);
                            ++ready;
                            start_cv.notify_all();
                            start_cv.wait(lock, [&]() { return start; });
                        }

                        lane_begin_us[lane] = ggml_time_us();
                        set_snapshot_read(true, (uint32_t) lane, snapshot_seq[lane]);
                        ggml_backend_tensor_copy_async(phone_backend, pc_backend, dual_phone_tensors[pair_index][lane],
                                                       dual_pc_tensors[pair_index][lane]);
                        set_snapshot_read(false, 0, 0);
                        lane_end_us[lane] = ggml_time_us();
                    });
                }

                {
                    std::unique_lock<std::mutex> lock(start_mutex);
                    start_cv.wait(lock, [&]() { return ready == (int) workers.size(); });
                    start = true;
                }
                start_cv.notify_all();

                for (auto & worker : workers) {
                    worker.join();
                }

                if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
                    const int64_t begin_us                           = std::min(lane_begin_us[0], lane_begin_us[1]);
                    const int64_t end_us                             = std::max(lane_end_us[0], lane_end_us[1]);
                    dual_samples[run - LLAMA_HYBRID_RPC_WARMUP_RUNS] = (end_us - begin_us) / 1000.0;
                }
            }

            profile.dual_phone_to_pc.push_back({
                dual_bytes[0],
                dual_bytes[1],
                llama_hybrid_rpc_median(dual_samples),
            });
        }
    }

    llama_hybrid_transfer_print("pc_to_phone", profile.pc_to_phone);
    llama_hybrid_transfer_print("phone_to_pc", profile.phone_to_pc);
    for (const auto & point : profile.dual_phone_to_pc) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE] dual_phone_to_pc bytes0=%zu bytes1=%zu wall_ms=%.3f\n", point.bytes0,
                       point.bytes1, point.wall_ms);
    }
    return true;
}

int64_t llama_hybrid_ffn_shard_size(int64_t n_ff, ggml_type down_type, float pc_ratio, int backend_index) {
    if (n_ff <= 0 || down_type < 0 || down_type >= GGML_TYPE_COUNT || pc_ratio < 0.0f || pc_ratio > 1.0f ||
        (backend_index != 0 && backend_index != 1)) {
        return 0;
    }
    if (pc_ratio == 0.0f) {
        return backend_index == 0 ? 0 : n_ff;
    }
    if (pc_ratio == 1.0f) {
        return backend_index == 0 ? n_ff : 0;
    }

    const int64_t granularity = std::lcm<int64_t>(ggml_blck_size(down_type), 128);
    int64_t       pc_size     = (int64_t) (n_ff * (double) pc_ratio);
    pc_size -= pc_size % granularity;
    return backend_index == 0 ? pc_size : n_ff - pc_size;
}

static bool llama_hybrid_profile_ffn_point(const llama_hybrid_ffn_desc & desc,
                                           ggml_backend_t                backend,
                                           int                           backend_index,
                                           int                           tokens,
                                           float                         backend_ratio,
                                           double &                      result_ms) {
    const float   pc_ratio   = backend_index == 0 ? backend_ratio : 1.0f - backend_ratio;
    const int64_t n_ff_shard = llama_hybrid_ffn_shard_size(desc.n_ff, desc.down_type, pc_ratio, backend_index);
    if (n_ff_shard <= 0) {
        LLAMA_LOG_ERROR("%s: empty FFN shard for backend=%d ratio=%.2f\n", __func__, backend_index, backend_ratio);
        return false;
    }

    static constexpr size_t graph_size = 16;
    const ggml_init_params  params     = {
        /*.mem_size   =*/32 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: failed to create FFN context\n", __func__);
        return false;
    }

    ggml_tensor * input  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, desc.n_embd, tokens);
    ggml_tensor * w_gate = ggml_new_tensor_2d(ctx.get(), desc.gate_type, desc.n_embd, n_ff_shard);
    ggml_tensor * w_up   = ggml_new_tensor_2d(ctx.get(), desc.up_type, desc.n_embd, n_ff_shard);
    ggml_tensor * w_down = ggml_new_tensor_2d(ctx.get(), desc.down_type, n_ff_shard, desc.n_embd);

    ggml_tensor * gate   = ggml_mul_mat(ctx.get(), w_gate, input);
    gate                 = ggml_silu(ctx.get(), gate);
    ggml_tensor * up     = ggml_mul_mat(ctx.get(), w_up, input);
    ggml_tensor * hidden = ggml_mul(ctx.get(), gate, up);
    ggml_tensor * output = ggml_mul_mat(ctx.get(), w_down, hidden);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);

    if (!ggml_backend_supports_op(backend, gate) || !ggml_backend_supports_op(backend, up) ||
        !ggml_backend_supports_op(backend, hidden) || !ggml_backend_supports_op(backend, output)) {
        LLAMA_LOG_ERROR("%s: backend %s does not support synthetic FFN graph\n", __func__, ggml_backend_name(backend));
        return false;
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate synthetic FFN tensors on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> samples;
    for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
        const int64_t     begin_us = ggml_time_us();
        const ggml_status status   = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const int64_t end_us = ggml_time_us();
        if (status != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: FFN graph failed on %s with status %d\n", __func__, ggml_backend_name(backend),
                            (int) status);
            return false;
        }
        if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
            samples[run - LLAMA_HYBRID_RPC_WARMUP_RUNS] = (end_us - begin_us) / 1000.0;
        }
    }

    result_ms = llama_hybrid_rpc_median(samples);
    return true;
}

bool llama_hybrid_profile_ffn(llama_hybrid_profile &        profile,
                              const llama_hybrid_ffn_desc & desc,
                              ggml_backend_t                cpu_backend,
                              ggml_backend_t                phone_backend) {
    if (cpu_backend == nullptr || phone_backend == nullptr || desc.n_embd <= 0 || desc.n_ff <= 0 ||
        desc.gate_type < 0 || desc.gate_type >= GGML_TYPE_COUNT || desc.up_type < 0 ||
        desc.up_type >= GGML_TYPE_COUNT || desc.down_type < 0 || desc.down_type >= GGML_TYPE_COUNT) {
        LLAMA_LOG_ERROR("%s: invalid FFN profile arguments\n", __func__);
        return false;
    }

    const std::vector<int> chunk_tokens = llama_hybrid_probe_chunk_tokens(profile);
    if (chunk_tokens.empty()) {
        LLAMA_LOG_ERROR("%s: no valid chunk candidates\n", __func__);
        return false;
    }
    if (profile.n_embd != 0 && profile.n_embd != desc.n_embd) {
        LLAMA_LOG_ERROR("%s: profile n_embd=%d does not match FFN n_embd=%" PRId64 "\n", __func__, profile.n_embd,
                        desc.n_embd);
        return false;
    }

    profile.n_embd = (int) desc.n_embd;
    profile.n_ff   = (int) desc.n_ff;
    profile.cpu_ffn.clear();
    profile.phone_ffn.clear();

    for (const int tokens : chunk_tokens) {
        for (const float ratio : LLAMA_HYBRID_FFN_RATIOS) {
            double cpu_ms;
            if (!llama_hybrid_profile_ffn_point(desc, cpu_backend, 0, tokens, ratio, cpu_ms)) {
                return false;
            }
            profile.cpu_ffn.push_back({ tokens, ratio, cpu_ms });
            LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=ffn tokens=%d ratio=%.2f ms=%.3f\n",
                           ggml_backend_name(cpu_backend), tokens, ratio, cpu_ms);

            double phone_ms;
            if (!llama_hybrid_profile_ffn_point(desc, phone_backend, 1, tokens, ratio, phone_ms)) {
                return false;
            }
            profile.phone_ffn.push_back({ tokens, ratio, phone_ms });
            LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=ffn tokens=%d ratio=%.2f ms=%.3f\n",
                           ggml_backend_name(phone_backend), tokens, ratio, phone_ms);
        }
    }
    return true;
}
