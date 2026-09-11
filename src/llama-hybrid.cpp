#include "llama-hybrid.h"

#include "../ggml/src/ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"
#include "llama-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <thread>
#include <utility>
#include <vector>

static constexpr int LLAMA_HYBRID_RPC_WARMUP_RUNS  = 2;
static constexpr int LLAMA_HYBRID_RPC_MEASURE_RUNS = 5;

static constexpr std::array<int, 3>   LLAMA_HYBRID_CHUNK_CANDIDATES       = { 1, 2, 4 };
static constexpr std::array<float, 4> LLAMA_HYBRID_FFN_RATIOS             = { 0.25f, 0.50f, 0.75f, 1.00f };
static constexpr std::array<int, 4>   LLAMA_HYBRID_PREFILL_KV_ANCHORS     = { 16, 256, 1024, 4096 };
static constexpr std::array<int, 5>   LLAMA_HYBRID_PHONE_BLOCK_CANDIDATES = { 1, 2, 4, 8, 16 };
static constexpr std::array<float, 3> LLAMA_HYBRID_PLAN_RATIOS            = { 0.25f, 0.50f, 0.75f };

static constexpr double LLAMA_HYBRID_PC_MEMORY_FRACTION    = 0.90;
static constexpr double LLAMA_HYBRID_PHONE_MEMORY_FRACTION = 0.90;
static constexpr double LLAMA_HYBRID_GPU_MEMORY_FRACTION   = 0.85;

static std::mutex                       g_llama_hybrid_runtime_plan_mutex;
static std::optional<llama_hybrid_plan> g_llama_hybrid_runtime_plan;

static bool llama_hybrid_runtime_plan_valid(const llama_hybrid_plan & plan) {
    const bool valid_chunks =
        plan.tensor_chunks_per_ubatch == 1 || plan.tensor_chunks_per_ubatch == 2 || plan.tensor_chunks_per_ubatch == 4;
    return plan.tensor_layers >= 0 && plan.phone_layers >= 0 && plan.pc_layers >= 0 &&
           plan.tensor_layers + plan.phone_layers + plan.pc_layers > 0 && plan.tensor_pc_ratio > 0.0f &&
           plan.tensor_pc_ratio < 1.0f && plan.gpu_pc_layers >= 0 && plan.gpu_pc_layers <= plan.pc_layers &&
           valid_chunks;
}

static void llama_hybrid_runtime_set_dual_return(bool enabled) {
#ifdef _WIN32
    _putenv_s("GGML_META_PREFILL_DUAL_RETURN", enabled ? "1" : "");
#else
    if (enabled) {
        setenv("GGML_META_PREFILL_DUAL_RETURN", "1", 1);
    } else {
        unsetenv("GGML_META_PREFILL_DUAL_RETURN");
    }
#endif
}

bool llama_hybrid_runtime_plan_set(const llama_hybrid_plan & plan) {
    if (!llama_hybrid_runtime_plan_valid(plan)) {
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
        g_llama_hybrid_runtime_plan = plan;
        llama_hybrid_runtime_set_dual_return(plan.tensor_layers > 0 && plan.tensor_chunks_per_ubatch > 1);
    }

    LLAMA_LOG_INFO("[HYBRID_RUNTIME] plan published T=%d P=%d C=%d R=%.3f G=%d K=%d\n", plan.tensor_layers,
                   plan.phone_layers, plan.pc_layers, plan.tensor_pc_ratio, plan.gpu_pc_layers,
                   plan.tensor_chunks_per_ubatch);
    return true;
}

bool llama_hybrid_runtime_plan_get(llama_hybrid_plan & plan) {
    std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
    if (!g_llama_hybrid_runtime_plan.has_value()) {
        return false;
    }
    plan = *g_llama_hybrid_runtime_plan;
    return true;
}

void llama_hybrid_runtime_plan_clear() {
    std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
    g_llama_hybrid_runtime_plan.reset();
    llama_hybrid_runtime_set_dual_return(false);
}

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

static void llama_hybrid_update_weight_bytes(llama_hybrid_profile & profile) {
    profile.total_weight_bytes_per_layer = profile.attn_weight_bytes_per_layer + profile.ffn_weight_bytes_per_layer;
}

static void * llama_hybrid_rpc_get_proc_address(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) {
        return nullptr;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
    return reg == nullptr ? nullptr : ggml_backend_reg_get_proc_address(reg, name);
}

static bool llama_hybrid_profile_rpc_fence(ggml_backend_t backend, double & result_ms) {
    const auto rpc_fence = reinterpret_cast<ggml_backend_rpc_fence_t>(
        llama_hybrid_rpc_get_proc_address(backend, GGML_BACKEND_RPC_FENCE_PROC));
    if (rpc_fence == nullptr) {
        LLAMA_LOG_ERROR("%s: backend %s does not support RPC fence\n", __func__, ggml_backend_name(backend));
        return false;
    }

    std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> samples;
    for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
        const int64_t begin_us = ggml_time_us();
        rpc_fence(backend);
        const int64_t end_us = ggml_time_us();

        if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
            samples[run - LLAMA_HYBRID_RPC_WARMUP_RUNS] = (end_us - begin_us) / 1000.0;
        }
    }

    result_ms = llama_hybrid_rpc_median(samples);
    return true;
}

struct llama_hybrid_graph_timing {
    double wall_ms        = 0.0;
    double compute_est_ms = 0.0;
};

static bool llama_hybrid_profile_graph_timing(ggml_backend_t              backend,
                                              ggml_cgraph *               graph,
                                              llama_hybrid_graph_timing & result) {
    if (backend == nullptr || graph == nullptr) {
        return false;
    }

    const auto rpc_fence = reinterpret_cast<ggml_backend_rpc_fence_t>(
        llama_hybrid_rpc_get_proc_address(backend, GGML_BACKEND_RPC_FENCE_PROC));

    std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> wall_samples;
    std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> compute_samples;

    for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
        double fence_before_ms = 0.0;
        double fence_after_ms  = 0.0;

        if (rpc_fence != nullptr) {
            const int64_t begin_us = ggml_time_us();
            rpc_fence(backend);
            const int64_t end_us = ggml_time_us();
            fence_before_ms      = (end_us - begin_us) / 1000.0;
        }

        const int64_t     begin_us = ggml_time_us();
        const ggml_status status   = ggml_backend_graph_compute_async(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            return false;
        }

        if (rpc_fence != nullptr) {
            rpc_fence(backend);
        } else {
            ggml_backend_synchronize(backend);
        }
        const int64_t end_us = ggml_time_us();

        if (rpc_fence != nullptr) {
            const int64_t begin_us = ggml_time_us();
            rpc_fence(backend);
            const int64_t end_us = ggml_time_us();
            fence_after_ms       = (end_us - begin_us) / 1000.0;
        }

        if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
            const int    sample     = run - LLAMA_HYBRID_RPC_WARMUP_RUNS;
            const double wall_ms    = (end_us - begin_us) / 1000.0;
            double       compute_ms = wall_ms;

            if (rpc_fence != nullptr) {
                compute_ms -= 0.5 * (fence_before_ms + fence_after_ms);
            }

            wall_samples[sample]    = wall_ms;
            compute_samples[sample] = compute_ms;
        }
    }

    result.wall_ms        = llama_hybrid_rpc_median(wall_samples);
    result.compute_est_ms = std::max(0.0, llama_hybrid_rpc_median(compute_samples));
    return true;
}

static bool llama_hybrid_read_backend_memory(ggml_backend_t backend, size_t & free_mem, size_t & total_mem) {
    free_mem  = 0;
    total_mem = 0;
    if (backend == nullptr) {
        return false;
    }

    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    if (device == nullptr) {
        return false;
    }

    ggml_backend_dev_memory(device, &free_mem, &total_mem);
    return true;
}

bool llama_hybrid_profile_memory(llama_hybrid_profile & profile,
                                 ggml_backend_t         pc_backend,
                                 ggml_backend_t         phone_backend,
                                 ggml_backend_t         gpu_backend) {
    if (pc_backend == nullptr || phone_backend == nullptr) {
        return false;
    }

    if (!llama_hybrid_read_backend_memory(pc_backend, profile.pc_free_mem, profile.pc_total_mem)) {
        LLAMA_LOG_ERROR("%s: failed to query PC memory\n", __func__);
        return false;
    }
    if (!llama_hybrid_read_backend_memory(phone_backend, profile.phone_free_mem, profile.phone_total_mem)) {
        LLAMA_LOG_ERROR("%s: failed to query Phone memory\n", __func__);
        return false;
    }

    profile.gpu_free_mem  = 0;
    profile.gpu_total_mem = 0;
    if (gpu_backend != nullptr &&
        !llama_hybrid_read_backend_memory(gpu_backend, profile.gpu_free_mem, profile.gpu_total_mem)) {
        LLAMA_LOG_ERROR("%s: failed to query GPU memory\n", __func__);
        return false;
    }

    constexpr double mib = 1024.0 * 1024.0;
    LLAMA_LOG_INFO("[HYBRID_PROFILE_MEMORY] backend=%s free=%zu total=%zu free_mib=%.1f total_mib=%.1f\n",
                   ggml_backend_name(pc_backend), profile.pc_free_mem, profile.pc_total_mem, profile.pc_free_mem / mib,
                   profile.pc_total_mem / mib);
    LLAMA_LOG_INFO("[HYBRID_PROFILE_MEMORY] backend=%s free=%zu total=%zu free_mib=%.1f total_mib=%.1f\n",
                   ggml_backend_name(phone_backend), profile.phone_free_mem, profile.phone_total_mem,
                   profile.phone_free_mem / mib, profile.phone_total_mem / mib);
    if (gpu_backend != nullptr) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE_MEMORY] backend=%s free=%zu total=%zu free_mib=%.1f total_mib=%.1f\n",
                       ggml_backend_name(gpu_backend), profile.gpu_free_mem, profile.gpu_total_mem,
                       profile.gpu_free_mem / mib, profile.gpu_total_mem / mib);
    }

    return true;
}

static void llama_hybrid_transfer_print(const char * label, const std::vector<llama_hybrid_transfer_point> & points) {
    for (const auto & point : points) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE] %s bytes=%zu ms=%.3f confirmed_wall_ms=%.3f\n", label, point.bytes, point.ms,
                       point.confirmed_wall_ms);
    }
}

void llama_hybrid_profile_print(const llama_hybrid_profile & profile) {
    LLAMA_LOG_INFO("[HYBRID_PROFILE] probe_tokens=%d probe_chunk_min_tokens=%d\n", profile.probe_tokens,
                   profile.probe_chunk_min_tokens);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] n_layer=%d n_embd=%d n_ff=%d\n", profile.n_layer, profile.n_embd, profile.n_ff);
    for (const auto & point : profile.cpu_ffn) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=CPU kind=ffn tokens=%d local_ratio=%.2f ms=%.3f\n",
                       point.tokens, point.local_ratio, point.ms);
    }
    for (const auto & point : profile.phone_ffn) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=PHONE kind=ffn tokens=%d local_ratio=%.2f ms=%.3f\n",
                       point.tokens, point.local_ratio, point.ms);
    }
    for (const auto & point : profile.cpu_attn) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=CPU kind=attention tokens=%d kv_tokens=%d ms=%.3f\n",
                       point.tokens, point.kv_tokens, point.ms);
    }
    for (const auto & point : profile.phone_attn) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=PHONE kind=attention tokens=%d kv_tokens=%d ms=%.3f\n",
                       point.tokens, point.kv_tokens, point.ms);
    }
    for (const auto & point : profile.gpu_attn) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=GPU kind=attention tokens=%d kv_tokens=%d ms=%.3f\n",
                       point.tokens, point.kv_tokens, point.ms);
    }
    for (const auto & point : profile.phone_blocks) {
        const double per_layer_est_ms = point.layers > 0 ? point.compute_est_ms / point.layers : 0.0;
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=PHONE kind=phone_block layers=%d tokens=%d kv_tokens=%d "
            "wall_ms=%.3f compute_est_ms=%.3f per_layer_est_ms=%.3f\n",
            point.layers, point.tokens, point.kv_tokens, point.wall_ms, point.compute_est_ms, per_layer_est_ms);
    }
    LLAMA_LOG_INFO("[HYBRID_PROFILE] cpu attn_ms=%.3f full_layer_ms=%.3f\n", profile.cpu_attn_ms,
                   profile.cpu_full_layer_ms);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] phone attn_ms=%.3f full_layer_ms=%.3f full_layer_compute_est_ms=%.3f\n",
                   profile.phone_attn_ms, profile.phone_full_layer_ms, profile.phone_full_layer_compute_est_ms);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] gpu full_layer_ms=%.3f\n", profile.gpu_full_layer_ms);

    llama_hybrid_transfer_print("pc_to_phone", profile.pc_to_phone);
    llama_hybrid_transfer_print("snapshot_phone_to_pc", profile.snapshot_phone_to_pc);
    llama_hybrid_transfer_print("phone_to_pc", profile.phone_to_pc);

    for (const auto & point : profile.dual_phone_to_pc) {
        LLAMA_LOG_INFO("[HYBRID_PROFILE] dual_phone_to_pc bytes0=%zu bytes1=%zu wall_ms=%.3f\n", point.bytes0,
                       point.bytes1, point.wall_ms);
    }

    LLAMA_LOG_INFO("[HYBRID_PROFILE] reduce_ms=%.3f rpc_submit_ms=%.3f rpc_fence_ms=%.3f\n", profile.reduce_ms,
                   profile.rpc_submit_ms, profile.rpc_fence_ms);
    LLAMA_LOG_INFO(
        "[HYBRID_PROFILE] pc_memory free=%zu total=%zu phone_memory free=%zu total=%zu "
        "gpu_memory free=%zu total=%zu\n",
        profile.pc_free_mem, profile.pc_total_mem, profile.phone_free_mem, profile.phone_total_mem,
        profile.gpu_free_mem, profile.gpu_total_mem);
    LLAMA_LOG_INFO(
        "[HYBRID_PROFILE] attn_weight_bytes_per_layer=%zu ffn_weight_bytes_per_layer=%zu "
        "total_weight_bytes_per_layer=%zu\n",
        profile.attn_weight_bytes_per_layer, profile.ffn_weight_bytes_per_layer, profile.total_weight_bytes_per_layer);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] model_weight_bytes=%zu non_layer_weight_bytes=%zu\n", profile.model_weight_bytes,
                   profile.non_layer_weight_bytes);

    if (!profile.layer_weight_bytes.empty()) {
        const auto     mm = std::minmax_element(profile.layer_weight_bytes.begin(), profile.layer_weight_bytes.end());
        const uint64_t layer_sum =
            std::accumulate(profile.layer_weight_bytes.begin(), profile.layer_weight_bytes.end(), uint64_t(0));
        const double layer_avg = (double) layer_sum / profile.layer_weight_bytes.size();
        LLAMA_LOG_INFO("[HYBRID_PROFILE] layer_weight_bytes count=%zu min=%zu max=%zu avg=%.1f\n",
                       profile.layer_weight_bytes.size(), *mm.first, *mm.second, layer_avg);
    }
}

void llama_hybrid_plan_print(const llama_hybrid_plan & plan) {
    LLAMA_LOG_INFO(
        "[HYBRID_PLAN] tensor_layers=%d phone_layers=%d pc_layers=%d tensor_pc_ratio=%.3f gpu_pc_layers=%d "
        "tensor_chunks_per_ubatch=%d predicted_ms=%.3f\n",
        plan.tensor_layers, plan.phone_layers, plan.pc_layers, plan.tensor_pc_ratio, plan.gpu_pc_layers,
        plan.tensor_chunks_per_ubatch, plan.predicted_ms);
    LLAMA_LOG_INFO("[HYBRID_PLAN] pc_memory=%zu phone_memory=%zu gpu_memory=%zu\n", plan.pc_memory, plan.phone_memory,
                   plan.gpu_memory);
    LLAMA_LOG_INFO("[HYBRID_PLAN_COST] tensor=%.3f phone=%.3f pc_cpu=%.3f pc_gpu=%.3f handoff=%.3f total=%.3f\n",
                   plan.predicted_tensor_ms, plan.predicted_phone_ms, plan.predicted_pc_cpu_ms,
                   plan.predicted_pc_gpu_ms, plan.predicted_handoff_ms, plan.predicted_ms);
}

static size_t llama_hybrid_effective_budget(size_t requested, size_t free_mem, double default_fraction) {
    if (requested > 0) {
        return free_mem > 0 ? std::min(requested, free_mem) : requested;
    }
    if (free_mem == 0) {
        return 0;
    }
    return (size_t) ((long double) free_mem * default_fraction);
}

static bool llama_hybrid_add_bytes(size_t & dst, size_t value) {
    if (value > std::numeric_limits<size_t>::max() - dst) {
        return false;
    }
    dst += value;
    return true;
}

static size_t llama_hybrid_ratio_bytes(size_t bytes, float ratio) {
    if (ratio <= 0.0f) {
        return 0;
    }
    if (ratio >= 1.0f) {
        return bytes;
    }
    const long double value = (long double) bytes * (long double) ratio;
    return std::min(bytes, (size_t) std::ceil(value));
}

static bool llama_hybrid_estimate_plan_memory(const llama_hybrid_profile &     profile,
                                              const llama_hybrid_constraints & constraints,
                                              llama_hybrid_plan &              plan) {
    const int   n_layer       = profile.n_layer;
    const int   tensor_layers = plan.tensor_layers;
    const int   phone_layers  = plan.phone_layers;
    const int   pc_layers     = plan.pc_layers;
    const int   gpu_layers    = plan.gpu_pc_layers;
    const float pc_ratio      = plan.tensor_pc_ratio;

    if (tensor_layers < 0 || phone_layers < 0 || pc_layers < 0 || tensor_layers + phone_layers + pc_layers != n_layer ||
        gpu_layers < 0 || gpu_layers > pc_layers || pc_ratio <= 0.0f || pc_ratio >= 1.0f) {
        return false;
    }

    const size_t n = (size_t) n_layer;
    if (profile.layer_weight_bytes.size() != n || profile.layer_attn_forced_bytes.size() != n ||
        profile.layer_ffn_split_bytes.size() != n || profile.layer_mirrored_bytes.size() != n) {
        return false;
    }

    size_t pc_memory    = 0;
    size_t phone_memory = 0;
    size_t gpu_memory   = 0;
    if (!llama_hybrid_add_bytes(pc_memory, profile.non_layer_weight_bytes)) {
        return false;
    }

    int ctx = constraints.target_ctx;
    if (ctx <= 0) {
        ctx = profile.n_ctx_train;
    }
    if (ctx <= 0 || profile.kv_bytes_per_token_per_layer > std::numeric_limits<size_t>::max() / (size_t) ctx) {
        return false;
    }
    const size_t kv_per_layer = profile.kv_bytes_per_token_per_layer * (size_t) ctx;

    const int phone_begin = tensor_layers;
    const int pc_begin    = tensor_layers + phone_layers;
    const int gpu_end     = pc_begin + gpu_layers;

    for (int il = 0; il < n_layer; ++il) {
        const size_t i        = (size_t) il;
        const size_t total    = profile.layer_weight_bytes[i];
        const size_t attn     = profile.layer_attn_forced_bytes[i];
        const size_t ffn      = profile.layer_ffn_split_bytes[i];
        const size_t mirrored = profile.layer_mirrored_bytes[i];

        if (il < phone_begin) {
            const size_t pc_ffn    = llama_hybrid_ratio_bytes(ffn, pc_ratio);
            const size_t phone_ffn = ffn - pc_ffn;
            if (!llama_hybrid_add_bytes(pc_memory, attn) || !llama_hybrid_add_bytes(pc_memory, pc_ffn) ||
                !llama_hybrid_add_bytes(pc_memory, mirrored) || !llama_hybrid_add_bytes(pc_memory, kv_per_layer) ||
                !llama_hybrid_add_bytes(phone_memory, phone_ffn) || !llama_hybrid_add_bytes(phone_memory, mirrored)) {
                return false;
            }
        } else if (il < pc_begin) {
            if (!llama_hybrid_add_bytes(phone_memory, attn) || !llama_hybrid_add_bytes(phone_memory, ffn) ||
                !llama_hybrid_add_bytes(phone_memory, mirrored) ||
                !llama_hybrid_add_bytes(phone_memory, kv_per_layer) || !llama_hybrid_add_bytes(pc_memory, mirrored)) {
                return false;
            }
        } else if (il < gpu_end) {
            if (!llama_hybrid_add_bytes(gpu_memory, total) || !llama_hybrid_add_bytes(gpu_memory, kv_per_layer)) {
                return false;
            }
        } else {
            if (!llama_hybrid_add_bytes(pc_memory, total) || !llama_hybrid_add_bytes(pc_memory, kv_per_layer)) {
                return false;
            }
        }
    }

    plan.pc_memory    = pc_memory;
    plan.phone_memory = phone_memory;
    plan.gpu_memory   = gpu_memory;
    return true;
}

static bool llama_hybrid_attn_cost(const std::vector<llama_hybrid_attn_compute_point> & points,
                                   int                                                  kv_tokens,
                                   double &                                             result_ms) {
    if (points.empty()) {
        return false;
    }

    const llama_hybrid_attn_compute_point * lower = nullptr;
    const llama_hybrid_attn_compute_point * upper = nullptr;
    for (const auto & point : points) {
        if (point.kv_tokens == kv_tokens) {
            result_ms = point.ms;
            return true;
        }
        if (point.kv_tokens < kv_tokens && (lower == nullptr || point.kv_tokens > lower->kv_tokens)) {
            lower = &point;
        }
        if (point.kv_tokens > kv_tokens && (upper == nullptr || point.kv_tokens < upper->kv_tokens)) {
            upper = &point;
        }
    }

    if (lower == nullptr) {
        result_ms = upper->ms;
        return true;
    }
    if (upper == nullptr) {
        result_ms = lower->ms;
        return true;
    }

    const double t = (double) (kv_tokens - lower->kv_tokens) / (double) (upper->kv_tokens - lower->kv_tokens);
    result_ms      = lower->ms + t * (upper->ms - lower->ms);
    return true;
}

static bool llama_hybrid_ffn_cost(const std::vector<llama_hybrid_ffn_compute_point> & points,
                                  int                                                 tokens,
                                  float                                               ratio,
                                  double &                                            result_ms) {
    for (const auto & point : points) {
        if (point.tokens == tokens && std::fabs(point.local_ratio - ratio) < 1e-4f) {
            result_ms = point.ms;
            return true;
        }
    }
    return false;
}

static bool llama_hybrid_transfer_cost(const std::vector<llama_hybrid_transfer_point> & points,
                                       size_t                                           bytes,
                                       double &                                         result_ms) {
    if (bytes == 0) {
        result_ms = 0.0;
        return true;
    }
    if (points.empty()) {
        return false;
    }

    const llama_hybrid_transfer_point * lower = nullptr;
    const llama_hybrid_transfer_point * upper = nullptr;
    for (const auto & point : points) {
        if (point.bytes == bytes) {
            result_ms = point.ms;
            return true;
        }
        if (point.bytes < bytes && (lower == nullptr || point.bytes > lower->bytes)) {
            lower = &point;
        }
        if (point.bytes > bytes && (upper == nullptr || point.bytes < upper->bytes)) {
            upper = &point;
        }
    }

    if (lower == nullptr) {
        result_ms = upper->ms;
        return true;
    }
    if (upper == nullptr) {
        result_ms = lower->ms;
        return true;
    }

    const double t = (double) (bytes - lower->bytes) / (double) (upper->bytes - lower->bytes);
    result_ms      = lower->ms + t * (upper->ms - lower->ms);
    return true;
}

static bool llama_hybrid_dual_transfer_cost(const llama_hybrid_profile & profile,
                                            size_t                       bytes0,
                                            size_t                       bytes1,
                                            double &                     result_ms) {
    for (const auto & point : profile.dual_phone_to_pc) {
        if ((point.bytes0 == bytes0 && point.bytes1 == bytes1) || (point.bytes0 == bytes1 && point.bytes1 == bytes0)) {
            result_ms = point.wall_ms;
            return true;
        }
    }
    return false;
}

static bool llama_hybrid_phone_block_cost(const llama_hybrid_profile & profile, int layers, double & result_ms) {
    if (layers == 0) {
        result_ms = 0.0;
        return true;
    }
    if (profile.phone_blocks.empty()) {
        return false;
    }

    const llama_hybrid_phone_block_point * lower = nullptr;
    const llama_hybrid_phone_block_point * upper = nullptr;
    for (const auto & point : profile.phone_blocks) {
        if (point.layers == layers) {
            result_ms = point.compute_est_ms;
            return true;
        }
        if (point.layers < layers && (lower == nullptr || point.layers > lower->layers)) {
            lower = &point;
        }
        if (point.layers > layers && (upper == nullptr || point.layers < upper->layers)) {
            upper = &point;
        }
    }

    if (lower == nullptr) {
        result_ms = upper->compute_est_ms;
        return true;
    }
    if (upper == nullptr) {
        const double per_layer = lower->compute_est_ms / lower->layers;
        result_ms              = lower->compute_est_ms + (layers - lower->layers) * per_layer;
        return true;
    }

    const double t = (double) (layers - lower->layers) / (double) (upper->layers - lower->layers);
    result_ms      = lower->compute_est_ms + t * (upper->compute_est_ms - lower->compute_est_ms);
    return true;
}

static bool llama_hybrid_tensor_ffn_cost(const llama_hybrid_profile & profile,
                                         float                        pc_ratio,
                                         int                          n_chunks,
                                         double &                     result_ms) {
    const std::vector<int> chunks = llama_hybrid_split_chunks(profile.probe_tokens, n_chunks);
    if (chunks.empty()) {
        return false;
    }

    std::vector<double> cpu_ready(chunks.size(), 0.0);
    std::vector<double> phone_ready(chunks.size(), 0.0);
    std::vector<double> return_single(chunks.size(), 0.0);
    std::vector<size_t> return_bytes(chunks.size(), 0);

    double h2d_available   = 0.0;
    double cpu_available   = 0.0;
    double phone_available = 0.0;

    for (size_t i = 0; i < chunks.size(); ++i) {
        const int    tokens        = chunks[i];
        const size_t bytes         = (size_t) profile.n_embd * (size_t) tokens * sizeof(float);
        double       h2d_ms        = 0.0;
        double       cpu_ms        = 0.0;
        double       phone_ms      = 0.0;
        double       d2h_single_ms = 0.0;

        if (!llama_hybrid_transfer_cost(profile.pc_to_phone, bytes, h2d_ms) ||
            !llama_hybrid_transfer_cost(profile.snapshot_phone_to_pc, bytes, d2h_single_ms) ||
            !llama_hybrid_ffn_cost(profile.cpu_ffn, tokens, pc_ratio, cpu_ms) ||
            !llama_hybrid_ffn_cost(profile.phone_ffn, tokens, 1.0f - pc_ratio, phone_ms)) {
            return false;
        }

        h2d_available += h2d_ms;
        cpu_available += cpu_ms;
        cpu_ready[i] = cpu_available;

        phone_available = std::max(phone_available, h2d_available);
        phone_available += phone_ms;
        phone_ready[i]   = phone_available;
        return_single[i] = d2h_single_ms;
        return_bytes[i]  = bytes;
    }

    double return_wave_available = 0.0;
    double done_ms               = 0.0;
    for (size_t i = 0; i < chunks.size(); i += 2) {
        if (i + 1 >= chunks.size()) {
            const double start    = std::max(return_wave_available, phone_ready[i]);
            const double end      = start + return_single[i];
            done_ms               = std::max(done_ms, std::max(cpu_ready[i], end) + profile.reduce_ms);
            return_wave_available = end;
            continue;
        }

        double dual_ms = 0.0;
        if (!llama_hybrid_dual_transfer_cost(profile, return_bytes[i], return_bytes[i + 1], dual_ms)) {
            dual_ms = std::max(return_single[i], return_single[i + 1]);
        }

        const double first_start  = std::max(return_wave_available, phone_ready[i]);
        const double second_start = std::max(return_wave_available, phone_ready[i + 1]);
        const double delta        = std::max(0.0, second_start - first_start);
        const double normal_end   = std::max(first_start + return_single[i], second_start + return_single[i + 1]);
        double       pair_end     = normal_end;

        if (delta < return_single[i] && return_single[i] > 0.0) {
            const double overlap_fraction = std::clamp((return_single[i] - delta) / return_single[i], 0.0, 1.0);
            const double dual_penalty     = std::max(0.0, dual_ms - std::max(return_single[i], return_single[i + 1]));
            pair_end += dual_penalty * overlap_fraction;
        }

        done_ms = std::max(done_ms, std::max(std::max(cpu_ready[i], cpu_ready[i + 1]), pair_end) + profile.reduce_ms);
        return_wave_available = pair_end;
    }

    result_ms = done_ms;
    return true;
}

static double llama_hybrid_tensor_misc_cost(const llama_hybrid_profile & profile) {
    double full_ffn_ms = 0.0;
    if (!llama_hybrid_ffn_cost(profile.cpu_ffn, profile.probe_tokens, 1.0f, full_ffn_ms)) {
        return 0.0;
    }
    return std::max(0.0, profile.cpu_full_layer_ms - profile.cpu_attn_ms - full_ffn_ms);
}

std::vector<llama_hybrid_plan> llama_hybrid_enumerate_feasible_plans(const llama_hybrid_profile &     profile,
                                                                     const llama_hybrid_constraints & constraints) {
    std::vector<llama_hybrid_plan> result;
    if (profile.n_layer <= 0) {
        return result;
    }

    const size_t pc_budget    = llama_hybrid_effective_budget(constraints.pc_memory_budget, profile.pc_free_mem,
                                                              LLAMA_HYBRID_PC_MEMORY_FRACTION);
    const size_t phone_budget = llama_hybrid_effective_budget(constraints.phone_memory_budget, profile.phone_free_mem,
                                                              LLAMA_HYBRID_PHONE_MEMORY_FRACTION);
    const size_t gpu_budget   = llama_hybrid_effective_budget(constraints.gpu_memory_budget, profile.gpu_free_mem,
                                                              LLAMA_HYBRID_GPU_MEMORY_FRACTION);

    std::vector<float> ratios;
    if (constraints.fixed_tensor_pc_ratio) {
        const float ratio = *constraints.fixed_tensor_pc_ratio;
        if (ratio <= 0.0f || ratio >= 1.0f) {
            return result;
        }
        ratios.push_back(ratio);
    } else {
        ratios.assign(LLAMA_HYBRID_PLAN_RATIOS.begin(), LLAMA_HYBRID_PLAN_RATIOS.end());
    }

    std::vector<int> chunk_candidates;
    if (constraints.fixed_tensor_chunks_per_ubatch) {
        chunk_candidates.push_back(*constraints.fixed_tensor_chunks_per_ubatch);
    } else {
        chunk_candidates.assign(LLAMA_HYBRID_CHUNK_CANDIDATES.begin(), LLAMA_HYBRID_CHUNK_CANDIDATES.end());
    }

    size_t    total_candidates = 0;
    size_t    reject_pc        = 0;
    size_t    reject_phone     = 0;
    size_t    reject_gpu       = 0;
    const int n_layer          = profile.n_layer;

    for (int tensor_layers = 0; tensor_layers <= n_layer; ++tensor_layers) {
        if (constraints.fixed_tensor_layers && tensor_layers != *constraints.fixed_tensor_layers) {
            continue;
        }

        for (int phone_layers = 0; phone_layers <= n_layer - tensor_layers; ++phone_layers) {
            if (constraints.fixed_phone_layers && phone_layers != *constraints.fixed_phone_layers) {
                continue;
            }

            const int pc_layers = n_layer - tensor_layers - phone_layers;
            if (constraints.fixed_pc_layers && pc_layers != *constraints.fixed_pc_layers) {
                continue;
            }

            for (const float ratio : ratios) {
                if (tensor_layers == 0 && !constraints.fixed_tensor_pc_ratio && std::fabs(ratio - 0.50f) >= 1e-4f) {
                    continue;
                }

                const int gpu_max = gpu_budget > 0 ? pc_layers : 0;
                for (int gpu_layers = 0; gpu_layers <= gpu_max; ++gpu_layers) {
                    if (constraints.fixed_gpu_pc_layers && gpu_layers != *constraints.fixed_gpu_pc_layers) {
                        continue;
                    }

                    llama_hybrid_plan base;
                    base.tensor_layers   = tensor_layers;
                    base.phone_layers    = phone_layers;
                    base.pc_layers       = pc_layers;
                    base.tensor_pc_ratio = ratio;
                    base.gpu_pc_layers   = gpu_layers;
                    if (!llama_hybrid_estimate_plan_memory(profile, constraints, base)) {
                        continue;
                    }

                    for (const int chunks_per_ubatch : chunk_candidates) {
                        if (tensor_layers == 0 && !constraints.fixed_tensor_chunks_per_ubatch &&
                            chunks_per_ubatch != 1) {
                            continue;
                        }

                        const std::vector<int> chunks =
                            llama_hybrid_split_chunks(profile.probe_tokens, chunks_per_ubatch);
                        if (chunks.empty() ||
                            *std::min_element(chunks.begin(), chunks.end()) < profile.probe_chunk_min_tokens) {
                            continue;
                        }

                        ++total_candidates;
                        if (pc_budget > 0 && base.pc_memory > pc_budget) {
                            ++reject_pc;
                            continue;
                        }
                        if (phone_budget > 0 && base.phone_memory > phone_budget) {
                            ++reject_phone;
                            continue;
                        }
                        if (base.gpu_pc_layers > 0) {
                            size_t gpu_required = base.gpu_memory;
                            if (!llama_hybrid_add_bytes(gpu_required, constraints.gpu_runtime_reserve_bytes) ||
                                gpu_budget == 0 || gpu_required > gpu_budget) {
                                ++reject_gpu;
                                continue;
                            }
                        }

                        llama_hybrid_plan plan        = base;
                        plan.tensor_chunks_per_ubatch = chunks_per_ubatch;
                        plan.predicted_ms             = 0.0;
                        result.push_back(plan);
                    }
                }
            }
        }
    }

    LLAMA_LOG_INFO(
        "[HYBRID_PLAN_ENUM] ctx=%d pc_budget=%zu phone_budget=%zu gpu_budget=%zu gpu_runtime_reserve=%zu "
        "total=%zu feasible=%zu reject_pc=%zu reject_phone=%zu reject_gpu=%zu\n",
        constraints.target_ctx > 0 ? constraints.target_ctx : profile.n_ctx_train, pc_budget, phone_budget, gpu_budget,
        constraints.gpu_runtime_reserve_bytes, total_candidates, result.size(), reject_pc, reject_phone, reject_gpu);
    return result;
}

bool llama_hybrid_score_plan(const llama_hybrid_profile &     profile,
                             const llama_hybrid_constraints & constraints,
                             llama_hybrid_plan &              plan) {
    if (plan.tensor_layers < 0 || plan.phone_layers < 0 || plan.pc_layers < 0 ||
        plan.tensor_layers + plan.phone_layers + plan.pc_layers != profile.n_layer || plan.gpu_pc_layers < 0 ||
        plan.gpu_pc_layers > plan.pc_layers) {
        return false;
    }

    int kv_tokens = constraints.score_kv_tokens;
    if (kv_tokens <= 0) {
        kv_tokens = constraints.target_ctx > 0 ? constraints.target_ctx : profile.n_ctx_train;
    }
    kv_tokens = std::max(kv_tokens, profile.probe_tokens);
    if (profile.n_ctx_train > 0) {
        kv_tokens = std::min(kv_tokens, profile.n_ctx_train);
    }

    double cpu_attn_base   = 0.0;
    double cpu_attn_kv     = 0.0;
    double phone_attn_base = 0.0;
    double phone_attn_kv   = 0.0;
    if (!llama_hybrid_attn_cost(profile.cpu_attn, profile.probe_tokens, cpu_attn_base) ||
        !llama_hybrid_attn_cost(profile.cpu_attn, kv_tokens, cpu_attn_kv) ||
        !llama_hybrid_attn_cost(profile.phone_attn, profile.probe_tokens, phone_attn_base) ||
        !llama_hybrid_attn_cost(profile.phone_attn, kv_tokens, phone_attn_kv)) {
        return false;
    }

    plan.predicted_tensor_ms = 0.0;
    if (plan.tensor_layers > 0) {
        double tensor_ffn_ms = 0.0;
        if (!llama_hybrid_tensor_ffn_cost(profile, plan.tensor_pc_ratio, plan.tensor_chunks_per_ubatch,
                                          tensor_ffn_ms)) {
            return false;
        }
        const double tensor_layer_ms = cpu_attn_kv + llama_hybrid_tensor_misc_cost(profile) + tensor_ffn_ms;
        plan.predicted_tensor_ms     = plan.tensor_layers * tensor_layer_ms;
    }

    plan.predicted_phone_ms = 0.0;
    if (plan.phone_layers > 0) {
        double base_block_ms = 0.0;
        if (!llama_hybrid_phone_block_cost(profile, plan.phone_layers, base_block_ms)) {
            return false;
        }
        const double attn_delta = std::max(0.0, phone_attn_kv - phone_attn_base);
        plan.predicted_phone_ms = base_block_ms + plan.phone_layers * attn_delta;
    }

    const double cpu_layer_ms = std::max(0.0, profile.cpu_full_layer_ms + cpu_attn_kv - cpu_attn_base);
    plan.predicted_pc_cpu_ms  = (plan.pc_layers - plan.gpu_pc_layers) * cpu_layer_ms;

    plan.predicted_pc_gpu_ms = 0.0;
    if (plan.gpu_pc_layers > 0) {
        double gpu_attn_base = 0.0;
        double gpu_attn_kv   = 0.0;
        if (!llama_hybrid_attn_cost(profile.gpu_attn, profile.probe_tokens, gpu_attn_base) ||
            !llama_hybrid_attn_cost(profile.gpu_attn, kv_tokens, gpu_attn_kv)) {
            return false;
        }
        const double gpu_layer_ms = std::max(0.0, profile.gpu_full_layer_ms + gpu_attn_kv - gpu_attn_base);
        plan.predicted_pc_gpu_ms  = plan.gpu_pc_layers * gpu_layer_ms;
    }

    plan.predicted_handoff_ms = 0.0;
    if (plan.phone_layers > 0) {
        if (profile.n_embd <= 0 || profile.probe_tokens <= 0 ||
            (size_t) profile.n_embd >
                std::numeric_limits<size_t>::max() / sizeof(float) / (size_t) profile.probe_tokens) {
            return false;
        }

        const size_t bytes    = (size_t) profile.n_embd * (size_t) profile.probe_tokens * sizeof(float);
        double       enter_ms = 0.0;
        double       exit_ms  = 0.0;
        if (!llama_hybrid_transfer_cost(profile.pc_to_phone, bytes, enter_ms) ||
            !llama_hybrid_transfer_cost(profile.phone_to_pc, bytes, exit_ms)) {
            return false;
        }
        plan.predicted_handoff_ms = enter_ms + exit_ms;
    }

    plan.predicted_ms = plan.predicted_tensor_ms + plan.predicted_phone_ms + plan.predicted_pc_cpu_ms +
                        plan.predicted_pc_gpu_ms + plan.predicted_handoff_ms;
    return std::isfinite(plan.predicted_ms);
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

    const auto rpc_fence = reinterpret_cast<ggml_backend_rpc_fence_t>(
        llama_hybrid_rpc_get_proc_address(phone_backend, GGML_BACKEND_RPC_FENCE_PROC));
    if (rpc_fence == nullptr) {
        LLAMA_LOG_ERROR("%s: phone backend does not support RPC fence\n", __func__);
        return false;
    }
    if (profile.rpc_fence_ms <= 0.0 && !llama_hybrid_profile_rpc_fence(phone_backend, profile.rpc_fence_ms)) {
        return false;
    }
    LLAMA_LOG_INFO("[HYBRID_PROFILE_RPC] completion fence baseline=%.3f ms\n", profile.rpc_fence_ms);

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
    profile.snapshot_phone_to_pc.clear();
    profile.phone_to_pc.clear();
    profile.dual_phone_to_pc.clear();
    profile.pc_to_phone.reserve(chunk_tokens.size());
    profile.snapshot_phone_to_pc.reserve(chunk_tokens.size());
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

        std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> pc_to_phone_est_samples;
        std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> pc_to_phone_wall_samples;
        std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> phone_to_pc_samples;

        for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
            std::fill_n(source_data.begin(), bytes, (uint8_t) (run + 1));
            ggml_backend_tensor_set(pc_tensor, source_data.data(), 0, bytes);

            const int64_t fence0_begin_us = ggml_time_us();
            rpc_fence(phone_backend);
            const int64_t fence0_end_us = ggml_time_us();

            const int64_t begin_us = ggml_time_us();
            ggml_backend_tensor_copy(pc_tensor, phone_tensor);
            rpc_fence(phone_backend);
            const int64_t end_us = ggml_time_us();

            const int64_t fence1_begin_us = ggml_time_us();
            rpc_fence(phone_backend);
            const int64_t fence1_end_us = ggml_time_us();

            if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
                const int    sample          = run - LLAMA_HYBRID_RPC_WARMUP_RUNS;
                const double wall_ms         = (end_us - begin_us) / 1000.0;
                const double fence_before_ms = (fence0_end_us - fence0_begin_us) / 1000.0;
                const double fence_after_ms  = (fence1_end_us - fence1_begin_us) / 1000.0;
                const double local_fence_ms  = 0.5 * (fence_before_ms + fence_after_ms);
                const double transfer_est_ms = wall_ms - local_fence_ms;

                pc_to_phone_wall_samples[sample] = wall_ms;
                pc_to_phone_est_samples[sample]  = transfer_est_ms;
            }
        }

        for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
            std::fill_n(source_data.begin(), bytes, (uint8_t) (run + 17));
            ggml_backend_tensor_set(phone_tensor, source_data.data(), 0, bytes);
            rpc_fence(phone_backend);
            const int64_t begin_us = ggml_time_us();
            ggml_backend_tensor_copy(phone_tensor, pc_tensor);
            const int64_t end_us = ggml_time_us();

            if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
                phone_to_pc_samples[run - LLAMA_HYBRID_RPC_WARMUP_RUNS] = (end_us - begin_us) / 1000.0;
            }
        }

        const double pc_to_phone_ms      = std::max(0.0, llama_hybrid_rpc_median(pc_to_phone_est_samples));
        const double pc_to_phone_wall_ms = llama_hybrid_rpc_median(pc_to_phone_wall_samples);
        const double phone_to_pc_ms      = llama_hybrid_rpc_median(phone_to_pc_samples);

        profile.pc_to_phone.push_back({ bytes, pc_to_phone_ms, pc_to_phone_wall_ms });
        profile.phone_to_pc.push_back({ bytes, phone_to_pc_ms, phone_to_pc_ms });
    }

    const auto snapshot_arm = reinterpret_cast<ggml_backend_rpc_snapshot_arm_t>(
        llama_hybrid_rpc_get_proc_address(phone_backend, GGML_BACKEND_RPC_SNAPSHOT_ARM_PROC));
    const auto set_snapshot_read = reinterpret_cast<ggml_backend_rpc_set_snapshot_read_t>(
        llama_hybrid_rpc_get_proc_address(phone_backend, GGML_BACKEND_RPC_SET_SNAPSHOT_READ_PROC));
    if (snapshot_arm == nullptr || set_snapshot_read == nullptr) {
        LLAMA_LOG_ERROR("%s: phone backend does not support RPC snapshots\n", __func__);
        return false;
    }

    static std::atomic<uint64_t> next_snapshot_seq{ uint64_t(1) << 63 };
    for (size_t i = 0; i < chunk_tokens.size(); ++i) {
        ggml_tensor *                                     pc_tensor    = pc_tensors[i];
        ggml_tensor *                                     phone_tensor = phone_tensors[i];
        const size_t                                      bytes        = ggml_nbytes(phone_tensor);
        std::array<double, LLAMA_HYBRID_RPC_MEASURE_RUNS> samples;

        for (int run = 0; run < LLAMA_HYBRID_RPC_WARMUP_RUNS + LLAMA_HYBRID_RPC_MEASURE_RUNS; ++run) {
            std::fill_n(source_data.begin(), bytes, (uint8_t) (run + 81));
            ggml_backend_tensor_set(phone_tensor, source_data.data(), 0, bytes);
            rpc_fence(phone_backend);

            const uint64_t seq = next_snapshot_seq.fetch_add(1, std::memory_order_relaxed);
            if (!snapshot_arm(phone_backend, phone_tensor, 0, bytes, 0, seq)) {
                LLAMA_LOG_ERROR("%s: failed to arm single snapshot\n", __func__);
                return false;
            }

            const int64_t begin_us = ggml_time_us();
            set_snapshot_read(true, 0, seq);
            ggml_backend_tensor_copy_async(phone_backend, pc_backend, phone_tensor, pc_tensor);
            set_snapshot_read(false, 0, 0);
            const int64_t end_us = ggml_time_us();

            if (run >= LLAMA_HYBRID_RPC_WARMUP_RUNS) {
                samples[run - LLAMA_HYBRID_RPC_WARMUP_RUNS] = (end_us - begin_us) / 1000.0;
            }
        }

        const double ms = llama_hybrid_rpc_median(samples);
        profile.snapshot_phone_to_pc.push_back({ bytes, ms, ms });
    }

    if (!dual_chunks.empty()) {
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
                }

                rpc_fence(phone_backend);

                for (size_t lane = 0; lane < 2; ++lane) {
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
    llama_hybrid_transfer_print("snapshot_phone_to_pc", profile.snapshot_phone_to_pc);
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
                                           float                         local_ratio,
                                           double &                      result_ms) {
    const float   pc_ratio   = backend_index == 0 ? local_ratio : 1.0f - local_ratio;
    const int64_t n_ff_shard = llama_hybrid_ffn_shard_size(desc.n_ff, desc.down_type, pc_ratio, backend_index);
    if (n_ff_shard <= 0) {
        LLAMA_LOG_ERROR("%s: empty FFN shard for backend=%d local_ratio=%.2f\n", __func__, backend_index, local_ratio);
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

    ggml_cgraph *                graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_probe_uid{ uint64_t(1) << 62 };
    graph->uid = next_probe_uid.fetch_add(1, std::memory_order_relaxed);
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

    llama_hybrid_graph_timing timing;
    if (!llama_hybrid_profile_graph_timing(backend, graph, timing)) {
        LLAMA_LOG_ERROR("%s: FFN graph failed on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }

    const bool is_rpc = llama_hybrid_rpc_get_proc_address(backend, GGML_BACKEND_RPC_FENCE_PROC) != nullptr;
    result_ms         = is_rpc ? timing.compute_est_ms : timing.wall_ms;
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

    profile.n_embd                     = (int) desc.n_embd;
    profile.n_ff                       = (int) desc.n_ff;
    profile.ffn_weight_bytes_per_layer = desc.weight_bytes;
    llama_hybrid_update_weight_bytes(profile);
    profile.cpu_ffn.clear();
    profile.phone_ffn.clear();

    if (profile.rpc_fence_ms <= 0.0) {
        if (!llama_hybrid_profile_rpc_fence(phone_backend, profile.rpc_fence_ms)) {
            return false;
        }
    }
    LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=rpc_fence ms=%.3f\n", ggml_backend_name(phone_backend),
                   profile.rpc_fence_ms);

    for (const int tokens : chunk_tokens) {
        for (const float ratio : LLAMA_HYBRID_FFN_RATIOS) {
            double cpu_ms;
            if (!llama_hybrid_profile_ffn_point(desc, cpu_backend, 0, tokens, ratio, cpu_ms)) {
                return false;
            }
            profile.cpu_ffn.push_back({ tokens, ratio, cpu_ms });
            LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=ffn tokens=%d local_ratio=%.2f ms=%.3f\n",
                           ggml_backend_name(cpu_backend), tokens, ratio, cpu_ms);

            double phone_ms;
            if (!llama_hybrid_profile_ffn_point(desc, phone_backend, 1, tokens, ratio, phone_ms)) {
                return false;
            }
            profile.phone_ffn.push_back({ tokens, ratio, phone_ms });
            LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=ffn tokens=%d local_ratio=%.2f ms=%.3f\n",
                           ggml_backend_name(phone_backend), tokens, ratio, phone_ms);
        }
    }
    return true;
}

static bool llama_hybrid_profile_attn_point(const llama_hybrid_attn_desc & desc,
                                            ggml_backend_t                 backend,
                                            int                            tokens,
                                            int                            kv_tokens,
                                            double &                       result_ms) {
    if (backend == nullptr || tokens <= 0 || kv_tokens <= 0) {
        return false;
    }
    if (kv_tokens != tokens) {
        LLAMA_LOG_ERROR("%s: full attention probe requires kv_tokens == tokens\n", __func__);
        return false;
    }
    if (desc.n_embd <= 0 || desc.n_head <= 0 || desc.n_head_kv <= 0 || desc.q_dim % desc.n_head != 0 ||
        desc.k_dim % desc.n_head_kv != 0 || desc.v_dim % desc.n_head_kv != 0 ||
        desc.rope_type == LLAMA_ROPE_TYPE_NONE || desc.n_rot <= 0 || desc.n_ctx_orig <= 0 || desc.freq_base <= 0.0f ||
        desc.freq_scale <= 0.0f) {
        return false;
    }

    const int64_t head_dim_q = desc.q_dim / desc.n_head;
    const int64_t head_dim_k = desc.k_dim / desc.n_head_kv;
    const int64_t head_dim_v = desc.v_dim / desc.n_head_kv;
    if (head_dim_q != head_dim_k || desc.o_in_dim != head_dim_v * desc.n_head) {
        return false;
    }

    static constexpr size_t graph_size = 64;
    const ggml_init_params  params     = {
        /*.mem_size   =*/128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * input       = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, desc.n_embd, tokens);
    ggml_tensor * attn_norm_w = ggml_new_tensor_1d(ctx.get(), desc.attn_norm_type, desc.n_embd);
    ggml_tensor * wq          = ggml_new_tensor_2d(ctx.get(), desc.q_type, desc.n_embd, desc.q_dim);
    ggml_tensor * wk          = ggml_new_tensor_2d(ctx.get(), desc.k_type, desc.n_embd, desc.k_dim);
    ggml_tensor * wv          = ggml_new_tensor_2d(ctx.get(), desc.v_type, desc.n_embd, desc.v_dim);
    ggml_tensor * wo          = ggml_new_tensor_2d(ctx.get(), desc.o_type, desc.o_in_dim, desc.n_embd);

    ggml_tensor * norm = ggml_rms_norm(ctx.get(), input, desc.rms_eps);
    norm               = ggml_mul(ctx.get(), norm, attn_norm_w);

    ggml_tensor * q = ggml_mul_mat(ctx.get(), wq, norm);
    ggml_tensor * k = ggml_mul_mat(ctx.get(), wk, norm);
    ggml_tensor * v = ggml_mul_mat(ctx.get(), wv, norm);

    q = ggml_reshape_3d(ctx.get(), q, head_dim_q, desc.n_head, tokens);
    k = ggml_reshape_3d(ctx.get(), k, head_dim_k, desc.n_head_kv, tokens);
    v = ggml_reshape_3d(ctx.get(), v, head_dim_v, desc.n_head_kv, tokens);

    if (desc.has_q_norm) {
        ggml_tensor * q_norm_w = ggml_new_tensor_1d(ctx.get(), desc.q_norm_type, head_dim_q);
        q                      = ggml_rms_norm(ctx.get(), q, desc.rms_eps);
        q                      = ggml_mul(ctx.get(), q, q_norm_w);
    }
    if (desc.has_k_norm) {
        ggml_tensor * k_norm_w = ggml_new_tensor_1d(ctx.get(), desc.k_norm_type, head_dim_k);
        k                      = ggml_rms_norm(ctx.get(), k, desc.rms_eps);
        k                      = ggml_mul(ctx.get(), k, k_norm_w);
    }

    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    q = ggml_rope_ext(ctx.get(), q, pos, nullptr, desc.n_rot, (int) desc.rope_type, desc.n_ctx_orig, desc.freq_base,
                      desc.freq_scale, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx.get(), k, pos, nullptr, desc.n_rot, (int) desc.rope_type, desc.n_ctx_orig, desc.freq_base,
                      desc.freq_scale, 0.0f, 1.0f, 32.0f, 1.0f);

    q = ggml_permute(ctx.get(), q, 0, 2, 1, 3);
    k = ggml_permute(ctx.get(), k, 0, 2, 1, 3);
    v = ggml_permute(ctx.get(), v, 0, 2, 1, 3);
    k = ggml_cast(ctx.get(), k, GGML_TYPE_F16);
    v = ggml_cast(ctx.get(), v, GGML_TYPE_F16);

    ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, kv_tokens, tokens);

    const float   kq_scale = 1.0f / std::sqrt((float) head_dim_q);
    ggml_tensor * attn     = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(ctx.get(), attn, desc.o_in_dim, tokens);

    ggml_tensor * output = ggml_mul_mat(ctx.get(), wo, attn);
    output               = ggml_add(ctx.get(), output, input);

    ggml_cgraph *                graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_attn_probe_uid{ uint64_t(1) << 61 };
    graph->uid = next_attn_probe_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, output);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!ggml_backend_supports_op(backend, node)) {
            LLAMA_LOG_ERROR("%s: backend=%s does not support attention op=%s node=%s\n", __func__,
                            ggml_backend_name(backend), ggml_op_name(node->op), node->name);
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate attention probe on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    std::vector<int32_t> pos_data(tokens);
    for (int i = 0; i < tokens; ++i) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(pos, pos_data.data(), 0, ggml_nbytes(pos));

    std::vector<ggml_fp16_t> mask_data((size_t) kv_tokens * (size_t) tokens);
    const float              neg_inf = -std::numeric_limits<float>::infinity();
    for (int iq = 0; iq < tokens; ++iq) {
        for (int ik = 0; ik < kv_tokens; ++ik) {
            const float value                                         = ik <= iq ? 0.0f : neg_inf;
            mask_data[(size_t) iq * (size_t) kv_tokens + (size_t) ik] = ggml_fp32_to_fp16(value);
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    llama_hybrid_graph_timing timing;
    if (!llama_hybrid_profile_graph_timing(backend, graph, timing)) {
        LLAMA_LOG_ERROR("%s: attention graph failed on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }

    const bool is_rpc = llama_hybrid_rpc_get_proc_address(backend, GGML_BACKEND_RPC_FENCE_PROC) != nullptr;
    result_ms         = is_rpc ? timing.compute_est_ms : timing.wall_ms;
    return true;
}

static bool llama_hybrid_profile_attn_kv_kernel_point(const llama_hybrid_attn_desc & desc,
                                                      ggml_backend_t                 backend,
                                                      int                            tokens,
                                                      int                            kv_tokens,
                                                      double &                       result_ms) {
    if (backend == nullptr || tokens <= 0 || kv_tokens < tokens || desc.n_head <= 0 || desc.n_head_kv <= 0 ||
        desc.q_dim % desc.n_head != 0 || desc.k_dim % desc.n_head_kv != 0 || desc.v_dim % desc.n_head_kv != 0) {
        return false;
    }

    const int64_t head_dim_q = desc.q_dim / desc.n_head;
    const int64_t head_dim_k = desc.k_dim / desc.n_head_kv;
    const int64_t head_dim_v = desc.v_dim / desc.n_head_kv;
    if (head_dim_q != head_dim_k) {
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
        return false;
    }

    ggml_tensor * q    = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim_q, tokens, desc.n_head);
    ggml_tensor * k    = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, head_dim_k, kv_tokens, desc.n_head_kv);
    ggml_tensor * v    = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, head_dim_v, kv_tokens, desc.n_head_kv);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, kv_tokens, tokens);

    const float   kq_scale = 1.0f / std::sqrt((float) head_dim_q);
    ggml_tensor * attn     = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

    ggml_cgraph *                graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_attn_kv_probe_uid{ (uint64_t(1) << 60) | (uint64_t(1) << 59) };
    graph->uid = next_attn_kv_probe_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, attn);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!ggml_backend_supports_op(backend, node)) {
            LLAMA_LOG_ERROR("%s: backend=%s does not support attention KV op=%s node=%s\n", __func__,
                            ggml_backend_name(backend), ggml_op_name(node->op), node->name);
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate attention KV probe on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    const int                past_tokens = kv_tokens - tokens;
    std::vector<ggml_fp16_t> mask_data((size_t) kv_tokens * (size_t) tokens);
    const float              neg_inf = -std::numeric_limits<float>::infinity();
    for (int iq = 0; iq < tokens; ++iq) {
        for (int ik = 0; ik < kv_tokens; ++ik) {
            const float value                                         = ik <= past_tokens + iq ? 0.0f : neg_inf;
            mask_data[(size_t) iq * (size_t) kv_tokens + (size_t) ik] = ggml_fp32_to_fp16(value);
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    llama_hybrid_graph_timing timing;
    if (!llama_hybrid_profile_graph_timing(backend, graph, timing)) {
        LLAMA_LOG_ERROR("%s: attention KV graph failed on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }

    const bool is_rpc = llama_hybrid_rpc_get_proc_address(backend, GGML_BACKEND_RPC_FENCE_PROC) != nullptr;
    result_ms         = is_rpc ? timing.compute_est_ms : timing.wall_ms;
    return true;
}

bool llama_hybrid_profile_attention(llama_hybrid_profile &         profile,
                                    const llama_hybrid_attn_desc & desc,
                                    ggml_backend_t                 cpu_backend,
                                    ggml_backend_t                 phone_backend,
                                    ggml_backend_t                 gpu_backend) {
    if (cpu_backend == nullptr || phone_backend == nullptr) {
        LLAMA_LOG_ERROR("%s: CPU/Phone backend missing\n", __func__);
        return false;
    }
    if (desc.n_embd <= 0 || desc.k_dim <= 0 || desc.v_dim <= 0 || desc.n_head <= 0 || desc.n_head_kv <= 0 ||
        desc.n_ctx_orig <= 0 || desc.attn_norm_type >= GGML_TYPE_COUNT || desc.q_type >= GGML_TYPE_COUNT ||
        desc.k_type >= GGML_TYPE_COUNT || desc.v_type >= GGML_TYPE_COUNT || desc.o_type >= GGML_TYPE_COUNT ||
        (desc.has_q_norm && desc.q_norm_type >= GGML_TYPE_COUNT) ||
        (desc.has_k_norm && desc.k_norm_type >= GGML_TYPE_COUNT)) {
        LLAMA_LOG_ERROR("%s: invalid attention descriptor\n", __func__);
        return false;
    }
    if (profile.n_embd != 0 && profile.n_embd != desc.n_embd) {
        LLAMA_LOG_ERROR("%s: profile n_embd=%d attention n_embd=%" PRId64 "\n", __func__, profile.n_embd, desc.n_embd);
        return false;
    }

    const int tokens = profile.probe_tokens;
    if (tokens <= 0) {
        LLAMA_LOG_ERROR("%s: invalid probe token count\n", __func__);
        return false;
    }

    profile.cpu_attn.clear();
    profile.phone_attn.clear();
    profile.gpu_attn.clear();
    profile.attn_weight_bytes_per_layer  = desc.weight_bytes;
    profile.kv_bytes_per_token_per_layer = ((size_t) desc.k_dim + (size_t) desc.v_dim) * sizeof(ggml_fp16_t);
    profile.n_ctx_train                  = desc.n_ctx_orig;
    llama_hybrid_update_weight_bytes(profile);

    profile.cpu_attn_ms   = 0.0;
    profile.phone_attn_ms = 0.0;

    const auto profile_backend = [&](ggml_backend_t backend, std::vector<llama_hybrid_attn_compute_point> & points,
                                     double * legacy_ms) {
        double full_base_ms = 0.0;
        if (!llama_hybrid_profile_attn_point(desc, backend, tokens, tokens, full_base_ms)) {
            return false;
        }
        if (legacy_ms != nullptr) {
            *legacy_ms = full_base_ms;
        }

        double flash_base_ms = 0.0;
        if (!llama_hybrid_profile_attn_kv_kernel_point(desc, backend, tokens, tokens, flash_base_ms)) {
            return false;
        }

        for (const int kv_tokens : LLAMA_HYBRID_PREFILL_KV_ANCHORS) {
            if (kv_tokens < tokens || kv_tokens > desc.n_ctx_orig) {
                continue;
            }

            double flash_ms = flash_base_ms;
            if (kv_tokens != tokens &&
                !llama_hybrid_profile_attn_kv_kernel_point(desc, backend, tokens, kv_tokens, flash_ms)) {
                return false;
            }

            const double attn_ms = full_base_ms + flash_ms - flash_base_ms;
            points.push_back({ tokens, kv_tokens, attn_ms });
            LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=attention_flash tokens=%d kv_tokens=%d ms=%.3f\n",
                           ggml_backend_name(backend), tokens, kv_tokens, flash_ms);
            LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=attention tokens=%d kv_tokens=%d ms=%.3f\n",
                           ggml_backend_name(backend), tokens, kv_tokens, attn_ms);
        }

        return true;
    };

    if (!profile_backend(cpu_backend, profile.cpu_attn, &profile.cpu_attn_ms) ||
        !profile_backend(phone_backend, profile.phone_attn, &profile.phone_attn_ms) ||
        (gpu_backend != nullptr && !profile_backend(gpu_backend, profile.gpu_attn, nullptr))) {
        return false;
    }

    if (profile.cpu_attn.empty()) {
        LLAMA_LOG_ERROR("%s: no valid KV anchors for probe tokens=%d context=%" PRId64 "\n", __func__, tokens,
                        desc.n_ctx_orig);
        return false;
    }

    return true;
}

struct llama_hybrid_probe_layer_weights {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq        = nullptr;
    ggml_tensor * wk        = nullptr;
    ggml_tensor * wv        = nullptr;
    ggml_tensor * wo        = nullptr;
    ggml_tensor * q_norm    = nullptr;
    ggml_tensor * k_norm    = nullptr;
    ggml_tensor * ffn_norm  = nullptr;
    ggml_tensor * gate      = nullptr;
    ggml_tensor * up        = nullptr;
    ggml_tensor * down      = nullptr;
};

static llama_hybrid_probe_layer_weights llama_hybrid_make_probe_layer_weights(ggml_context *                 ctx,
                                                                              const llama_hybrid_attn_desc & attn,
                                                                              const llama_hybrid_ffn_desc &  ffn) {
    llama_hybrid_probe_layer_weights weights;

    weights.attn_norm = ggml_new_tensor_1d(ctx, attn.attn_norm_type, attn.n_embd);
    weights.wq        = ggml_new_tensor_2d(ctx, attn.q_type, attn.n_embd, attn.q_dim);
    weights.wk        = ggml_new_tensor_2d(ctx, attn.k_type, attn.n_embd, attn.k_dim);
    weights.wv        = ggml_new_tensor_2d(ctx, attn.v_type, attn.n_embd, attn.v_dim);
    weights.wo        = ggml_new_tensor_2d(ctx, attn.o_type, attn.o_in_dim, attn.n_embd);

    if (attn.has_q_norm) {
        const int64_t head_dim = attn.q_dim / attn.n_head;
        weights.q_norm         = ggml_new_tensor_1d(ctx, attn.q_norm_type, head_dim);
    }
    if (attn.has_k_norm) {
        const int64_t head_dim = attn.k_dim / attn.n_head_kv;
        weights.k_norm         = ggml_new_tensor_1d(ctx, attn.k_norm_type, head_dim);
    }

    weights.ffn_norm = ggml_new_tensor_1d(ctx, ffn.ffn_norm_type, ffn.n_embd);
    weights.gate     = ggml_new_tensor_2d(ctx, ffn.gate_type, ffn.n_embd, ffn.n_ff);
    weights.up       = ggml_new_tensor_2d(ctx, ffn.up_type, ffn.n_embd, ffn.n_ff);
    weights.down     = ggml_new_tensor_2d(ctx, ffn.down_type, ffn.n_ff, ffn.n_embd);

    return weights;
}

static ggml_tensor * llama_hybrid_build_probe_layer(ggml_context *                           ctx,
                                                    ggml_tensor *                            input,
                                                    ggml_tensor *                            pos,
                                                    ggml_tensor *                            mask,
                                                    const llama_hybrid_attn_desc &           attn_desc,
                                                    const llama_hybrid_ffn_desc &            ffn_desc,
                                                    const llama_hybrid_probe_layer_weights & weights,
                                                    int                                      tokens) {
    (void) ffn_desc;

    const int64_t head_dim_q = attn_desc.q_dim / attn_desc.n_head;
    const int64_t head_dim_k = attn_desc.k_dim / attn_desc.n_head_kv;
    const int64_t head_dim_v = attn_desc.v_dim / attn_desc.n_head_kv;

    ggml_tensor * norm = ggml_rms_norm(ctx, input, attn_desc.rms_eps);
    norm               = ggml_mul(ctx, norm, weights.attn_norm);

    ggml_tensor * q = ggml_mul_mat(ctx, weights.wq, norm);
    ggml_tensor * k = ggml_mul_mat(ctx, weights.wk, norm);
    ggml_tensor * v = ggml_mul_mat(ctx, weights.wv, norm);

    q = ggml_reshape_3d(ctx, q, head_dim_q, attn_desc.n_head, tokens);
    k = ggml_reshape_3d(ctx, k, head_dim_k, attn_desc.n_head_kv, tokens);
    v = ggml_reshape_3d(ctx, v, head_dim_v, attn_desc.n_head_kv, tokens);

    if (weights.q_norm != nullptr) {
        q = ggml_rms_norm(ctx, q, attn_desc.rms_eps);
        q = ggml_mul(ctx, q, weights.q_norm);
    }
    if (weights.k_norm != nullptr) {
        k = ggml_rms_norm(ctx, k, attn_desc.rms_eps);
        k = ggml_mul(ctx, k, weights.k_norm);
    }

    q = ggml_rope_ext(ctx, q, pos, nullptr, attn_desc.n_rot, (int) attn_desc.rope_type, attn_desc.n_ctx_orig,
                      attn_desc.freq_base, attn_desc.freq_scale, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, pos, nullptr, attn_desc.n_rot, (int) attn_desc.rope_type, attn_desc.n_ctx_orig,
                      attn_desc.freq_base, attn_desc.freq_scale, 0.0f, 1.0f, 32.0f, 1.0f);

    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
    v = ggml_cast(ctx, v, GGML_TYPE_F16);

    const float   kq_scale = 1.0f / std::sqrt((float) head_dim_q);
    ggml_tensor * attn     = ggml_flash_attn_ext(ctx, q, k, v, mask, kq_scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
    attn = ggml_reshape_2d(ctx, attn, attn_desc.o_in_dim, tokens);

    ggml_tensor * attn_out = ggml_mul_mat(ctx, weights.wo, attn);
    ggml_tensor * ffn_inp  = ggml_add(ctx, attn_out, input);

    ggml_tensor * ffn_norm = ggml_rms_norm(ctx, ffn_inp, attn_desc.rms_eps);
    ffn_norm               = ggml_mul(ctx, ffn_norm, weights.ffn_norm);

    ggml_tensor * gate    = ggml_silu(ctx, ggml_mul_mat(ctx, weights.gate, ffn_norm));
    ggml_tensor * up      = ggml_mul_mat(ctx, weights.up, ffn_norm);
    ggml_tensor * hidden  = ggml_mul(ctx, gate, up);
    ggml_tensor * ffn_out = ggml_mul_mat(ctx, weights.down, hidden);

    return ggml_add(ctx, ffn_out, ffn_inp);
}

static bool llama_hybrid_profile_full_layer_point(const llama_hybrid_attn_desc & attn_desc,
                                                  const llama_hybrid_ffn_desc &  ffn_desc,
                                                  ggml_backend_t                 backend,
                                                  int                            tokens,
                                                  int                            kv_tokens,
                                                  llama_hybrid_graph_timing &    result) {
    if (backend == nullptr || tokens <= 0 || kv_tokens <= 0) {
        return false;
    }
    if (kv_tokens != tokens) {
        LLAMA_LOG_ERROR("%s: current full-layer probe requires kv_tokens == tokens\n", __func__);
        return false;
    }
    if (attn_desc.n_embd <= 0 || ffn_desc.n_embd <= 0 || ffn_desc.n_ff <= 0 || attn_desc.n_embd != ffn_desc.n_embd) {
        LLAMA_LOG_ERROR("%s: invalid/mismatched layer descriptor\n", __func__);
        return false;
    }
    if (ffn_desc.ffn_norm_type < 0 || ffn_desc.ffn_norm_type >= GGML_TYPE_COUNT || ffn_desc.gate_type < 0 ||
        ffn_desc.gate_type >= GGML_TYPE_COUNT || ffn_desc.up_type < 0 || ffn_desc.up_type >= GGML_TYPE_COUNT ||
        ffn_desc.down_type < 0 || ffn_desc.down_type >= GGML_TYPE_COUNT || attn_desc.attn_norm_type < 0 ||
        attn_desc.attn_norm_type >= GGML_TYPE_COUNT || attn_desc.q_type < 0 || attn_desc.q_type >= GGML_TYPE_COUNT ||
        attn_desc.k_type < 0 || attn_desc.k_type >= GGML_TYPE_COUNT || attn_desc.v_type < 0 ||
        attn_desc.v_type >= GGML_TYPE_COUNT || attn_desc.o_type < 0 || attn_desc.o_type >= GGML_TYPE_COUNT ||
        (attn_desc.has_q_norm && (attn_desc.q_norm_type < 0 || attn_desc.q_norm_type >= GGML_TYPE_COUNT)) ||
        (attn_desc.has_k_norm && (attn_desc.k_norm_type < 0 || attn_desc.k_norm_type >= GGML_TYPE_COUNT))) {
        LLAMA_LOG_ERROR("%s: invalid layer tensor types\n", __func__);
        return false;
    }
    if (attn_desc.n_head <= 0 || attn_desc.n_head_kv <= 0 || attn_desc.q_dim % attn_desc.n_head != 0 ||
        attn_desc.k_dim % attn_desc.n_head_kv != 0 || attn_desc.v_dim % attn_desc.n_head_kv != 0 ||
        attn_desc.rope_type == LLAMA_ROPE_TYPE_NONE || attn_desc.n_rot <= 0 || attn_desc.n_ctx_orig <= 0 ||
        attn_desc.freq_base <= 0.0f || attn_desc.freq_scale <= 0.0f) {
        return false;
    }

    const int64_t head_dim_q = attn_desc.q_dim / attn_desc.n_head;
    const int64_t head_dim_k = attn_desc.k_dim / attn_desc.n_head_kv;
    const int64_t head_dim_v = attn_desc.v_dim / attn_desc.n_head_kv;
    if (head_dim_q != head_dim_k || attn_desc.o_in_dim != head_dim_v * attn_desc.n_head) {
        return false;
    }

    static constexpr size_t graph_size = 128;
    const ggml_init_params  params     = {
        /*.mem_size   =*/256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: failed to create full-layer context\n", __func__);
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, attn_desc.n_embd, tokens);
    ggml_tensor * pos   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, kv_tokens, tokens);

    const llama_hybrid_probe_layer_weights weights =
        llama_hybrid_make_probe_layer_weights(ctx.get(), attn_desc, ffn_desc);
    ggml_tensor * output =
        llama_hybrid_build_probe_layer(ctx.get(), input, pos, mask, attn_desc, ffn_desc, weights, tokens);

    ggml_cgraph *                graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_full_layer_probe_uid{ uint64_t(1) << 60 };
    graph->uid = next_full_layer_probe_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, output);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!ggml_backend_supports_op(backend, node)) {
            LLAMA_LOG_ERROR("%s: backend=%s does not support full-layer op=%s node=%s\n", __func__,
                            ggml_backend_name(backend), ggml_op_name(node->op), node->name);
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate full-layer probe on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    std::vector<int32_t> pos_data(tokens);
    for (int i = 0; i < tokens; ++i) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(pos, pos_data.data(), 0, ggml_nbytes(pos));

    std::vector<ggml_fp16_t> mask_data((size_t) kv_tokens * (size_t) tokens);
    const float              neg_inf = -std::numeric_limits<float>::infinity();
    for (int iq = 0; iq < tokens; ++iq) {
        for (int ik = 0; ik < kv_tokens; ++ik) {
            const float value                                         = ik <= iq ? 0.0f : neg_inf;
            mask_data[(size_t) iq * (size_t) kv_tokens + (size_t) ik] = ggml_fp32_to_fp16(value);
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    if (!llama_hybrid_profile_graph_timing(backend, graph, result)) {
        LLAMA_LOG_ERROR("%s: full-layer graph failed on %s\n", __func__, ggml_backend_name(backend));
        return false;
    }

    return true;
}

static bool llama_hybrid_profile_phone_block_point(const llama_hybrid_attn_desc & attn_desc,
                                                   const llama_hybrid_ffn_desc &  ffn_desc,
                                                   ggml_backend_t                 phone_backend,
                                                   int                            n_layers,
                                                   int                            tokens,
                                                   llama_hybrid_graph_timing &    timing) {
    if (phone_backend == nullptr || n_layers <= 0 || tokens <= 0 || attn_desc.n_embd <= 0 || ffn_desc.n_embd <= 0 ||
        ffn_desc.n_ff <= 0 || attn_desc.n_embd != ffn_desc.n_embd || attn_desc.n_head <= 0 ||
        attn_desc.n_head_kv <= 0 || attn_desc.q_dim % attn_desc.n_head != 0 ||
        attn_desc.k_dim % attn_desc.n_head_kv != 0 || attn_desc.v_dim % attn_desc.n_head_kv != 0) {
        return false;
    }

    const int64_t head_dim_q = attn_desc.q_dim / attn_desc.n_head;
    const int64_t head_dim_k = attn_desc.k_dim / attn_desc.n_head_kv;
    const int64_t head_dim_v = attn_desc.v_dim / attn_desc.n_head_kv;
    if (head_dim_q != head_dim_k || attn_desc.o_in_dim != head_dim_v * attn_desc.n_head) {
        return false;
    }

    const size_t           graph_size    = (size_t) n_layers * 32 + 32;
    const size_t           tensor_budget = 192 + (size_t) n_layers * 96;
    const ggml_init_params params        = {
        /*.mem_size   =*/tensor_budget * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, attn_desc.n_embd, tokens);
    ggml_tensor * pos   = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, tokens, tokens);

    const llama_hybrid_probe_layer_weights weights =
        llama_hybrid_make_probe_layer_weights(ctx.get(), attn_desc, ffn_desc);
    ggml_tensor * cur = input;
    for (int il = 0; il < n_layers; ++il) {
        cur = llama_hybrid_build_probe_layer(ctx.get(), cur, pos, mask, attn_desc, ffn_desc, weights, tokens);
    }

    ggml_cgraph *                graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_phone_block_uid{ uint64_t(1) << 59 };
    graph->uid = next_phone_block_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, cur);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!ggml_backend_supports_op(phone_backend, node)) {
            LLAMA_LOG_ERROR("%s: backend=%s layers=%d unsupported op=%s node=%s\n", __func__,
                            ggml_backend_name(phone_backend), n_layers, ggml_op_name(node->op), node->name);
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), phone_backend));
    if (!buffer) {
        LLAMA_LOG_ERROR("%s: allocation failed layers=%d backend=%s\n", __func__, n_layers,
                        ggml_backend_name(phone_backend));
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    std::vector<int32_t> pos_data(tokens);
    for (int i = 0; i < tokens; ++i) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(pos, pos_data.data(), 0, ggml_nbytes(pos));

    std::vector<ggml_fp16_t> mask_data((size_t) tokens * (size_t) tokens);
    const float              neg_inf = -std::numeric_limits<float>::infinity();
    for (int iq = 0; iq < tokens; ++iq) {
        for (int ik = 0; ik < tokens; ++ik) {
            const float value                                      = ik <= iq ? 0.0f : neg_inf;
            mask_data[(size_t) iq * (size_t) tokens + (size_t) ik] = ggml_fp32_to_fp16(value);
        }
    }
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));

    if (!llama_hybrid_profile_graph_timing(phone_backend, graph, timing)) {
        LLAMA_LOG_ERROR("%s: phone block execution failed layers=%d\n", __func__, n_layers);
        return false;
    }

    return true;
}

bool llama_hybrid_profile_full_layer(llama_hybrid_profile &         profile,
                                     const llama_hybrid_attn_desc & attn_desc,
                                     const llama_hybrid_ffn_desc &  ffn_desc,
                                     ggml_backend_t                 cpu_backend,
                                     ggml_backend_t                 phone_backend,
                                     ggml_backend_t                 gpu_backend) {
    if (cpu_backend == nullptr || phone_backend == nullptr) {
        LLAMA_LOG_ERROR("%s: CPU/Phone backend missing\n", __func__);
        return false;
    }
    if (attn_desc.n_embd != ffn_desc.n_embd) {
        LLAMA_LOG_ERROR("%s: ATTN/FFN n_embd mismatch: %" PRId64 " vs %" PRId64 "\n", __func__, attn_desc.n_embd,
                        ffn_desc.n_embd);
        return false;
    }

    const int tokens    = profile.probe_tokens;
    const int kv_tokens = tokens;
    if (tokens <= 0) {
        return false;
    }

    if (profile.rpc_fence_ms <= 0.0 && !llama_hybrid_profile_rpc_fence(phone_backend, profile.rpc_fence_ms)) {
        return false;
    }

    profile.attn_weight_bytes_per_layer = attn_desc.weight_bytes;
    profile.ffn_weight_bytes_per_layer  = ffn_desc.weight_bytes;
    llama_hybrid_update_weight_bytes(profile);

    llama_hybrid_graph_timing cpu_timing;
    if (!llama_hybrid_profile_full_layer_point(attn_desc, ffn_desc, cpu_backend, tokens, kv_tokens, cpu_timing)) {
        return false;
    }
    profile.cpu_full_layer_ms = cpu_timing.wall_ms;
    LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=full_layer tokens=%d kv_tokens=%d ms=%.3f\n",
                   ggml_backend_name(cpu_backend), tokens, kv_tokens, profile.cpu_full_layer_ms);

    llama_hybrid_graph_timing phone_timing;
    if (!llama_hybrid_profile_full_layer_point(attn_desc, ffn_desc, phone_backend, tokens, kv_tokens, phone_timing)) {
        return false;
    }
    profile.phone_full_layer_ms             = phone_timing.wall_ms;
    profile.phone_full_layer_compute_est_ms = phone_timing.compute_est_ms;
    LLAMA_LOG_INFO(
        "[HYBRID_PROFILE_COMPUTE] backend=%s kind=full_layer tokens=%d kv_tokens=%d wall_ms=%.3f "
        "compute_est_ms=%.3f\n",
        ggml_backend_name(phone_backend), tokens, kv_tokens, profile.phone_full_layer_ms,
        profile.phone_full_layer_compute_est_ms);

    profile.gpu_full_layer_ms = 0.0;
    if (gpu_backend != nullptr) {
        llama_hybrid_graph_timing gpu_timing;
        if (!llama_hybrid_profile_full_layer_point(attn_desc, ffn_desc, gpu_backend, tokens, kv_tokens, gpu_timing)) {
            return false;
        }
        profile.gpu_full_layer_ms = gpu_timing.wall_ms;
        LLAMA_LOG_INFO("[HYBRID_PROFILE_COMPUTE] backend=%s kind=full_layer tokens=%d kv_tokens=%d ms=%.3f\n",
                       ggml_backend_name(gpu_backend), tokens, kv_tokens, profile.gpu_full_layer_ms);
    }

    return true;
}

bool llama_hybrid_profile_phone_blocks(llama_hybrid_profile &         profile,
                                       const llama_hybrid_attn_desc & attn_desc,
                                       const llama_hybrid_ffn_desc &  ffn_desc,
                                       ggml_backend_t                 phone_backend) {
    if (phone_backend == nullptr || attn_desc.n_embd != ffn_desc.n_embd) {
        return false;
    }

    const int tokens = profile.probe_tokens;
    if (tokens <= 0) {
        return false;
    }

    profile.phone_blocks.clear();
    for (const int n_layers : LLAMA_HYBRID_PHONE_BLOCK_CANDIDATES) {
        if (profile.n_layer > 0 && n_layers > profile.n_layer) {
            continue;
        }

        llama_hybrid_graph_timing timing;
        if (!llama_hybrid_profile_phone_block_point(attn_desc, ffn_desc, phone_backend, n_layers, tokens, timing)) {
            return false;
        }

        profile.phone_blocks.push_back({ n_layers, tokens, tokens, timing.wall_ms, timing.compute_est_ms });
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=%s kind=phone_block layers=%d tokens=%d kv_tokens=%d "
            "wall_ms=%.3f compute_est_ms=%.3f per_layer_est_ms=%.3f\n",
            ggml_backend_name(phone_backend), n_layers, tokens, tokens, timing.wall_ms, timing.compute_est_ms,
            timing.compute_est_ms / n_layers);
    }

    return true;
}
