#include "llama-hybrid.h"

#include "../ggml/src/ggml-impl.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-rpc.h"
#include "llama-impl.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <limits>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static constexpr int LLAMA_HYBRID_RPC_WARMUP_RUNS      = 2;
static constexpr int LLAMA_HYBRID_RPC_MEASURE_RUNS     = 5;
static constexpr int LLAMA_HYBRID_COMPUTE_WARMUP_RUNS  = 1;
static constexpr int LLAMA_HYBRID_COMPUTE_MEASURE_RUNS = 3;

static constexpr int                  LLAMA_HYBRID_PROFILE_TOKENS          = 16;
static constexpr int                  LLAMA_HYBRID_REFERENCE_TOKENS        = 128;
static constexpr int                  LLAMA_HYBRID_PROFILE_BLOCK_LAYERS    = 5;
static constexpr std::array<int, 9>   LLAMA_HYBRID_CHUNK_TOKEN_CANDIDATES = { 4, 8, 12, 16, 20, 24, 32, 48, 64 };
static constexpr std::array<int, 8>   LLAMA_HYBRID_REGION_CHUNK_CANDIDATES = { 8, 16, 32, 64, 96, 128, 192, 256 };
static constexpr std::array<int, 9>   LLAMA_HYBRID_CPU_LAYER_TOKENS        = { 1, 8, 16, 32, 64, 96, 128, 192, 256 };
static constexpr std::array<int, 5>   LLAMA_HYBRID_PHONE_LAYER_TOKENS      = { 1, 8, 16, 32, 64 };
static constexpr std::array<int, 8>   LLAMA_HYBRID_GPU_LAYER_TOKENS        = { 1, 8, 16, 32, 64, 128, 192, 256 };
static constexpr std::array<int, 4>   LLAMA_HYBRID_DECODE_CHUNK_CANDIDATES = { 1, 2, 3, 4 };
static constexpr int                  LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS      = 4;
static constexpr std::array<int, 4>   LLAMA_HYBRID_MOE_CPU_BLOCK_TOKENS      = { 1, 64, 128, 256 };
static constexpr std::array<float, 8> LLAMA_HYBRID_FFN_RATIO_PROBES        = { 0.05f, 0.20f, 0.35f, 0.50f, 0.65f, 0.80f, 0.95f, 1.00f };
static constexpr std::array<int, 4>   LLAMA_HYBRID_PREFILL_KV_ANCHORS     = { 128, 256, 1024, 4096 };
static constexpr std::array<int, 2>   LLAMA_HYBRID_PHONE_BLOCK_CANDIDATES = { 1, LLAMA_HYBRID_PROFILE_BLOCK_LAYERS };

static constexpr double LLAMA_HYBRID_PC_MEMORY_FRACTION    = 0.70;
static constexpr double LLAMA_HYBRID_PHONE_MEMORY_FRACTION = 0.90;
static constexpr double LLAMA_HYBRID_GPU_MEMORY_FRACTION   = 0.90;

// HYBRID_AUTO uses a two-level search.  The cheap pass ranks complete
// (topology, XT, R) candidates, while the expensive pass expands stage
// macro sizes (XG/XC/XP) and runs the full pipeline simulator.
static constexpr size_t LLAMA_HYBRID_COARSE_FAMILY_TOP_K = 24;
static constexpr size_t LLAMA_HYBRID_COARSE_GLOBAL_TOP_K = 64;
static constexpr size_t LLAMA_HYBRID_COARSE_MARGIN_POOL  = 256;
static constexpr double LLAMA_HYBRID_COARSE_MARGIN       = 1.20;

static std::mutex                              g_llama_hybrid_runtime_plan_mutex;
static std::optional<llama_hybrid_plan>        g_llama_hybrid_runtime_plan;
static std::optional<llama_hybrid_profile>     g_llama_hybrid_runtime_profile;
static std::optional<llama_hybrid_constraints> g_llama_hybrid_runtime_constraints;
static std::optional<llama_hybrid_wave_calibration> g_llama_hybrid_runtime_wave_calibration;

static bool llama_hybrid_runtime_plan_valid(const llama_hybrid_plan & plan) {
    return plan.tensor_layers >= 0 && plan.phone_layers >= 0 && plan.pc_layers >= 0 &&
           plan.tensor_layers + plan.phone_layers + plan.pc_layers > 0 && plan.tensor_pc_ratio > 0.0f &&
           plan.tensor_pc_ratio < 1.0f && plan.gpu_pc_layers >= 0 && plan.gpu_pc_layers <= plan.pc_layers &&
           plan.gpu_chunk_tokens > 0 && plan.cpu_chunk_tokens > 0 && plan.tensor_chunk_tokens > 0 &&
           plan.phone_chunk_tokens > 0;
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
        g_llama_hybrid_runtime_wave_calibration.reset();
        llama_hybrid_runtime_set_dual_return(plan.tensor_layers > 0);
    }

    LLAMA_LOG_INFO(
        "[HYBRID_RUNTIME] plan published T=%d P=%d C=%d R=%.3f G=%d XG=%d XC=%d XT=%d XP=%d\n",
        plan.tensor_layers, plan.phone_layers, plan.pc_layers, plan.tensor_pc_ratio, plan.gpu_pc_layers,
        plan.gpu_chunk_tokens, plan.cpu_chunk_tokens, plan.tensor_chunk_tokens, plan.phone_chunk_tokens);
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

void llama_hybrid_runtime_wave_calibration_set(
        const llama_hybrid_wave_calibration & calibration) {
    std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
    g_llama_hybrid_runtime_wave_calibration = calibration;
}

bool llama_hybrid_runtime_wave_calibration_get(
        llama_hybrid_wave_calibration & calibration) {
    std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
    if (!g_llama_hybrid_runtime_wave_calibration.has_value()) {
        return false;
    }
    calibration = *g_llama_hybrid_runtime_wave_calibration;
    return true;
}

void llama_hybrid_runtime_plan_clear() {
    std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
    g_llama_hybrid_runtime_plan.reset();
    g_llama_hybrid_runtime_profile.reset();
    g_llama_hybrid_runtime_constraints.reset();
    g_llama_hybrid_runtime_wave_calibration.reset();
    llama_hybrid_runtime_set_dual_return(false);
}

int llama_hybrid_runtime_prefill_chunk_tokens() {
    std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
    return g_llama_hybrid_runtime_plan.has_value() ? g_llama_hybrid_runtime_plan->tensor_chunk_tokens : 0;
}

int llama_hybrid_runtime_prefill_attn_group_chunks() {
    const char * value = std::getenv("LLAMA_HYBRID_ATTN_GROUP_CHUNKS");
    if (value == nullptr) {
        return 1;
    }
    const int group = std::atoi(value);
    return group == 2 || group == 4 ? group : 1;
}

bool llama_hybrid_runtime_prefill_dag_enabled() {
    const char * value = std::getenv("LLAMA_HYBRID_PREFILL_DAG");
    return value != nullptr && std::atoi(value) != 0;
}

int llama_hybrid_runtime_prefill_dag_max_ahead() {
    const char * value = std::getenv("LLAMA_HYBRID_DAG_MAX_AHEAD");
    if (value == nullptr) {
        return 2;
    }
    return std::max(1, std::atoi(value));
}

bool llama_hybrid_runtime_prefill_dag_eligible(int layer_begin, int layer_end) {
    if (!llama_hybrid_runtime_prefill_dag_enabled()) {
        return false;
    }

    std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
    if (!g_llama_hybrid_runtime_plan.has_value()) {
        return false;
    }

    const llama_hybrid_plan & plan = *g_llama_hybrid_runtime_plan;
    const int tensor_begin = plan.pc_layers;
    const int tensor_end   = tensor_begin + plan.tensor_layers;

    return plan.gpu_pc_layers > 0 &&
           plan.pc_layers == plan.gpu_pc_layers &&
           plan.tensor_layers > 0 &&
           plan.phone_layers == 0 &&
           layer_begin == tensor_begin &&
           layer_end == tensor_end;
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

std::vector<int> llama_hybrid_split_by_chunk_size(int tokens, int chunk_tokens) {
    if (tokens <= 0 || chunk_tokens <= 0) {
        return {};
    }

    std::vector<int> result;
    result.reserve((tokens + chunk_tokens - 1) / chunk_tokens);
    while (tokens > 0) {
        const int current = std::min(tokens, chunk_tokens);
        result.push_back(current);
        tokens -= current;
    }
    return result;
}

std::vector<llama_hybrid_boundary_block> llama_hybrid_plan_boundary(
        int tokens, int upstream_chunk_tokens, int downstream_chunk_tokens) {
    if (tokens <= 0 || upstream_chunk_tokens <= 0 || downstream_chunk_tokens <= 0) {
        return {};
    }

    std::vector<llama_hybrid_boundary_block> blocks;
    for (int output_begin = 0; output_begin < tokens; output_begin += downstream_chunk_tokens) {
        llama_hybrid_boundary_block block;
        block.output.token_begin = output_begin;
        block.output.n_tokens    = std::min(downstream_chunk_tokens, tokens - output_begin);

        const int output_end = output_begin + block.output.n_tokens;
        int input_begin = output_begin - output_begin % upstream_chunk_tokens;
        while (input_begin < output_end) {
            const int input_end    = std::min(input_begin + upstream_chunk_tokens, tokens);
            const int overlap_begin = std::max(output_begin, input_begin);
            const int overlap_end   = std::min(output_end, input_end);
            if (overlap_begin < overlap_end) {
                block.inputs.push_back({ overlap_begin, overlap_end - overlap_begin });
            }
            input_begin = input_end;
        }

        GGML_ASSERT(!block.inputs.empty());
        if (block.inputs.size() > 1) {
            block.action = llama_hybrid_boundary_action::ACCUMULATE;
        } else {
            const int upstream_begin = output_begin - output_begin % upstream_chunk_tokens;
            const int upstream_size  = std::min(upstream_chunk_tokens, tokens - upstream_begin);
            block.action = output_begin == upstream_begin && block.output.n_tokens == upstream_size ?
                llama_hybrid_boundary_action::PASS : llama_hybrid_boundary_action::SPLIT;
        }

        blocks.push_back(std::move(block));
    }
    return blocks;
}

static std::vector<std::vector<int>> llama_hybrid_probe_chunk_layouts(const llama_hybrid_profile & profile) {
    std::vector<std::vector<int>> result;

    const int reference_tokens = profile.reference_tokens > 0 ? profile.reference_tokens : LLAMA_HYBRID_REFERENCE_TOKENS;
    for (const int chunk_tokens : LLAMA_HYBRID_CHUNK_TOKEN_CANDIDATES) {
        std::vector<int> chunks = llama_hybrid_split_by_chunk_size(reference_tokens, chunk_tokens);
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

static std::vector<int> llama_hybrid_probe_transfer_tokens(const llama_hybrid_profile & profile) {
    std::vector<int> result = llama_hybrid_probe_chunk_tokens(profile);
    const int reference_tokens = profile.reference_tokens > 0 ? profile.reference_tokens : LLAMA_HYBRID_REFERENCE_TOKENS;
    if (profile.probe_tokens > 0) {
        result.push_back(profile.probe_tokens);
    }
    if (reference_tokens > 0) {
        result.push_back(reference_tokens);
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

template<size_t N>
static double llama_hybrid_rpc_median(std::array<double, N> samples) {
    static_assert(N > 0, "median needs at least one sample");
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

    std::array<double, LLAMA_HYBRID_COMPUTE_MEASURE_RUNS> wall_samples;
    std::array<double, LLAMA_HYBRID_COMPUTE_MEASURE_RUNS> compute_samples;

    for (int run = 0; run < LLAMA_HYBRID_COMPUTE_WARMUP_RUNS + LLAMA_HYBRID_COMPUTE_MEASURE_RUNS; ++run) {
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

        if (run >= LLAMA_HYBRID_COMPUTE_WARMUP_RUNS) {
            const int    sample     = run - LLAMA_HYBRID_COMPUTE_WARMUP_RUNS;
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
    LLAMA_LOG_ERROR("[HYBRID_PROFILE_MEMORY] backend=%s free=%zu total=%zu free_mib=%.1f total_mib=%.1f\n",
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
    LLAMA_LOG_INFO(
        "[HYBRID_PROFILE] probe_tokens=%d reference_tokens=%d probe_chunk_min_tokens=%d block_layers=%d\n",
        profile.probe_tokens, profile.reference_tokens, profile.probe_chunk_min_tokens, profile.profile_block_layers);
    LLAMA_LOG_INFO("[HYBRID_PROFILE] n_layer=%d n_embd=%d n_ff=%d ffn_granularity=%" PRId64 "\n",
                   profile.n_layer, profile.n_embd, profile.n_ff, profile.ffn_shard_granularity);
    for (const auto & point : profile.cpu_ffn) {
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=CPU kind=ffn tokens=%d local_ratio=%.5f local_ff=%" PRId64
            " ms=%.3f runtime_bytes=%zu\n",
            point.tokens, point.local_ratio, point.local_ff, point.ms, point.runtime_bytes);
    }
    for (const auto & point : profile.phone_ffn) {
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=PHONE kind=ffn tokens=%d local_ratio=%.5f local_ff=%" PRId64
            " ms=%.3f runtime_bytes=%zu\n",
            point.tokens, point.local_ratio, point.local_ff, point.ms, point.runtime_bytes);
    }
    for (const auto & point : profile.cpu_attn) {
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=CPU kind=attention tokens=%d kv_tokens=%d ms=%.3f "
            "runtime_bytes=%zu\n",
            point.tokens, point.kv_tokens, point.ms, point.runtime_bytes);
    }
    for (const auto & point : profile.phone_attn) {
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=PHONE kind=attention tokens=%d kv_tokens=%d ms=%.3f "
            "runtime_bytes=%zu\n",
            point.tokens, point.kv_tokens, point.ms, point.runtime_bytes);
    }
    for (const auto & point : profile.gpu_attn) {
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=GPU kind=attention tokens=%d kv_tokens=%d ms=%.3f "
            "runtime_bytes=%zu\n",
            point.tokens, point.kv_tokens, point.ms, point.runtime_bytes);
    }
    const auto print_layer_blocks = [](const char * backend,
                                      const std::vector<llama_hybrid_layer_compute_point> & points) {
        for (const auto & point : points) {
            const double wall_per_layer = point.layers > 0 ? point.wall_ms / point.layers : 0.0;
            const double compute_per_layer = point.layers > 0 ? point.compute_est_ms / point.layers : 0.0;
            LLAMA_LOG_INFO(
                "[HYBRID_PROFILE_COMPUTE] backend=%s kind=layer_block tokens=%d layers=%d "
                "wall_ms=%.3f compute_est_ms=%.3f wall_per_layer_ms=%.3f compute_per_layer_ms=%.3f\n",
                backend, point.tokens, point.layers, point.wall_ms, point.compute_est_ms,
                wall_per_layer, compute_per_layer);
        }
    };
    print_layer_blocks("CPU", profile.cpu_layer_blocks);
    print_layer_blocks("PHONE", profile.phone_layer_blocks);
    print_layer_blocks("GPU", profile.gpu_layer_blocks);
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

    llama_hybrid_transfer_print("gpu_to_pc", profile.gpu_to_pc);
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
    const double additive_ms = plan.predicted_tensor_ms + plan.predicted_phone_ms +
        plan.predicted_pc_cpu_ms + plan.predicted_pc_gpu_ms + plan.predicted_handoff_ms;
    LLAMA_LOG_INFO(
        "[HYBRID_PLAN] tensor_layers=%d phone_layers=%d pc_layers=%d tensor_pc_ratio=%.3f gpu_pc_layers=%d "
        "gpu_chunk_tokens=%d cpu_chunk_tokens=%d tensor_chunk_tokens=%d phone_chunk_tokens=%d predicted_ms=%.3f\n",
        plan.tensor_layers, plan.phone_layers, plan.pc_layers, plan.tensor_pc_ratio, plan.gpu_pc_layers,
        plan.gpu_chunk_tokens, plan.cpu_chunk_tokens, plan.tensor_chunk_tokens, plan.phone_chunk_tokens,
        plan.predicted_ms);
    LLAMA_LOG_ERROR("[HYBRID_PLAN_MEMORY] pc_required=%zu phone_required=%zu gpu_required=%zu\n", plan.pc_memory,
                   plan.phone_memory, plan.gpu_memory);
    LLAMA_LOG_INFO(
        "[HYBRID_PLAN_COST] tensor=%.3f phone=%.3f pc_cpu=%.3f pc_gpu=%.3f handoff=%.3f "
        "additive=%.3f makespan=%.3f\n",
                   plan.predicted_tensor_ms, plan.predicted_phone_ms, plan.predicted_pc_cpu_ms,
                   plan.predicted_pc_gpu_ms, plan.predicted_handoff_ms, additive_ms, plan.predicted_ms);
    LLAMA_LOG_INFO(
        "[HYBRID_PLAN_PIPE] gpu_busy=%.3f downstream=%.3f gpu_wait=%.3f tensor_peak_mib=%.2f "
        "phone_peak_mib=%.2f\n",
        plan.predicted_gpu_busy_ms, plan.predicted_downstream_ms, plan.predicted_gpu_wait_ms,
        plan.predicted_tensor_peak_bytes / 1048576.0, plan.predicted_phone_peak_bytes / 1048576.0);
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

static bool llama_hybrid_ffn_runtime_bytes_at_ratio(
    const std::vector<llama_hybrid_ffn_compute_point> & points,
    int                                                   tokens,
    float                                                 ratio,
    size_t &                                              result) {
    const llama_hybrid_ffn_compute_point * exact = nullptr;
    const llama_hybrid_ffn_compute_point * lower = nullptr;
    const llama_hybrid_ffn_compute_point * upper = nullptr;

    for (const auto & point : points) {
        if (std::fabs(point.local_ratio - ratio) >= 1e-4f) {
            continue;
        }
        if (point.tokens == tokens) {
            exact = &point;
            break;
        }
        if (point.tokens < tokens && (lower == nullptr || point.tokens > lower->tokens)) {
            lower = &point;
        }
        if (point.tokens > tokens && (upper == nullptr || point.tokens < upper->tokens)) {
            upper = &point;
        }
    }

    if (exact != nullptr) {
        result = exact->runtime_bytes;
        return true;
    }
    if (lower == nullptr && upper == nullptr) {
        return false;
    }
    if (lower == nullptr) {
        result = upper->runtime_bytes;
        return true;
    }
    if (upper == nullptr) {
        const long double scaled = (long double) lower->runtime_bytes * tokens / std::max(1, lower->tokens);
        if (scaled > std::numeric_limits<size_t>::max()) {
            return false;
        }
        result = (size_t) std::ceil(scaled);
        return true;
    }

    const long double t = (long double) (tokens - lower->tokens) / (upper->tokens - lower->tokens);
    const long double interpolated =
        (long double) lower->runtime_bytes +
        t * ((long double) upper->runtime_bytes - (long double) lower->runtime_bytes);
    result = (size_t) std::ceil(std::max<long double>(0.0L, interpolated));
    return true;
}

static bool llama_hybrid_ffn_runtime_bytes(const std::vector<llama_hybrid_ffn_compute_point> & points,
                                           int                                                 tokens,
                                           float                                               ratio,
                                           size_t &                                            result) {
    if (points.empty() || tokens <= 0 || ratio <= 0.0f || ratio > 1.0f) {
        return false;
    }

    std::vector<float> ratios;
    ratios.reserve(points.size());
    for (const auto & point : points) {
        if (point.local_ratio > 0.0f && point.local_ratio <= 1.0f) {
            ratios.push_back(point.local_ratio);
        }
    }
    std::sort(ratios.begin(), ratios.end());
    ratios.erase(std::unique(ratios.begin(), ratios.end(), [](float a, float b) {
        return std::fabs(a - b) < 1e-4f;
    }), ratios.end());
    if (ratios.empty()) {
        return false;
    }

    float lower_ratio = -1.0f;
    float upper_ratio = -1.0f;
    for (const float r : ratios) {
        if (std::fabs(r - ratio) < 1e-4f) {
            return llama_hybrid_ffn_runtime_bytes_at_ratio(points, tokens, r, result);
        }
        if (r < ratio) {
            lower_ratio = r;
        } else if (r > ratio && upper_ratio < 0.0f) {
            upper_ratio = r;
        }
    }

    if (lower_ratio < 0.0f) {
        return llama_hybrid_ffn_runtime_bytes_at_ratio(points, tokens, ratios.front(), result);
    }
    if (upper_ratio < 0.0f) {
        return llama_hybrid_ffn_runtime_bytes_at_ratio(points, tokens, ratios.back(), result);
    }

    size_t lower_value = 0;
    size_t upper_value = 0;
    if (!llama_hybrid_ffn_runtime_bytes_at_ratio(points, tokens, lower_ratio, lower_value) ||
        !llama_hybrid_ffn_runtime_bytes_at_ratio(points, tokens, upper_ratio, upper_value)) {
        return false;
    }

    const long double t = (long double) (ratio - lower_ratio) / (upper_ratio - lower_ratio);
    const long double value = (long double) lower_value + t * ((long double) upper_value - lower_value);
    if (value > std::numeric_limits<size_t>::max()) {
        return false;
    }
    result = (size_t) std::ceil(std::max<long double>(0.0L, value));
    return true;
}

static bool llama_hybrid_attn_runtime_bytes_at_query(
    const std::vector<llama_hybrid_attn_compute_point> & points,
    int                                                   query_tokens,
    int                                                   kv_tokens,
    size_t &                                              result) {
    const llama_hybrid_attn_compute_point * exact = nullptr;
    const llama_hybrid_attn_compute_point * lower = nullptr;
    const llama_hybrid_attn_compute_point * upper = nullptr;
    for (const auto & point : points) {
        if (point.tokens != query_tokens) {
            continue;
        }
        if (point.kv_tokens == kv_tokens) {
            exact = &point;
            break;
        }
        if (point.kv_tokens < kv_tokens && (lower == nullptr || point.kv_tokens > lower->kv_tokens)) {
            lower = &point;
        }
        if (point.kv_tokens > kv_tokens && (upper == nullptr || point.kv_tokens < upper->kv_tokens)) {
            upper = &point;
        }
    }

    if (exact != nullptr) {
        result = exact->runtime_bytes;
        return true;
    }
    if (lower == nullptr && upper == nullptr) {
        return false;
    }
    if (lower == nullptr) {
        result = upper->runtime_bytes;
        return true;
    }
    if (upper == nullptr) {
        // Conservatively scale the last measured runtime with KV length.
        const long double value = (long double) lower->runtime_bytes * kv_tokens /
                                  std::max(1, lower->kv_tokens);
        if (value > std::numeric_limits<size_t>::max()) {
            return false;
        }
        result = (size_t) std::ceil(value);
        return true;
    }

    const long double t = (long double) (kv_tokens - lower->kv_tokens) /
                          (upper->kv_tokens - lower->kv_tokens);
    const long double value = (long double) lower->runtime_bytes +
                              t * ((long double) upper->runtime_bytes - lower->runtime_bytes);
    if (value > std::numeric_limits<size_t>::max()) {
        return false;
    }
    result = (size_t) std::ceil(std::max<long double>(0.0L, value));
    return true;
}

static bool llama_hybrid_attn_runtime_bytes(const std::vector<llama_hybrid_attn_compute_point> & points,
                                            int                                                  query_tokens,
                                            int                                                  kv_tokens,
                                            size_t &                                             result) {
    if (points.empty() || query_tokens <= 0 || kv_tokens <= 0) {
        return false;
    }

    std::vector<int> queries;
    for (const auto & point : points) {
        if (point.tokens > 0) {
            queries.push_back(point.tokens);
        }
    }
    std::sort(queries.begin(), queries.end());
    queries.erase(std::unique(queries.begin(), queries.end()), queries.end());
    if (queries.empty()) {
        return false;
    }

    int lower_query = -1;
    int upper_query = -1;
    for (const int q : queries) {
        if (q == query_tokens) {
            return llama_hybrid_attn_runtime_bytes_at_query(points, q, std::max(kv_tokens, q), result);
        }
        if (q < query_tokens) {
            lower_query = q;
        } else if (q > query_tokens && upper_query < 0) {
            upper_query = q;
        }
    }

    if (lower_query < 0) {
        return llama_hybrid_attn_runtime_bytes_at_query(points, queries.front(),
                                                        std::max(kv_tokens, queries.front()), result);
    }
    if (upper_query < 0) {
        size_t base = 0;
        if (!llama_hybrid_attn_runtime_bytes_at_query(points, lower_query,
                                                       std::max(kv_tokens, lower_query), base)) {
            return false;
        }
        const long double scaled = (long double) base * query_tokens / lower_query;
        if (scaled > std::numeric_limits<size_t>::max()) {
            return false;
        }
        result = (size_t) std::ceil(scaled);
        return true;
    }

    size_t lower_value = 0;
    size_t upper_value = 0;
    if (!llama_hybrid_attn_runtime_bytes_at_query(points, lower_query, std::max(kv_tokens, lower_query), lower_value) ||
        !llama_hybrid_attn_runtime_bytes_at_query(points, upper_query, std::max(kv_tokens, upper_query), upper_value)) {
        return false;
    }
    const long double t = (long double) (query_tokens - lower_query) / (upper_query - lower_query);
    const long double value = (long double) lower_value + t * ((long double) upper_value - lower_value);
    if (value > std::numeric_limits<size_t>::max()) {
        return false;
    }
    result = (size_t) std::ceil(std::max<long double>(0.0L, value));
    return true;
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

    const int gpu_end    = gpu_layers;
    const int cpu_end    = pc_layers;
    const int tensor_end = pc_layers + tensor_layers;

    for (int il = 0; il < n_layer; ++il) {
        const size_t i        = (size_t) il;
        const size_t total    = profile.layer_weight_bytes[i];
        const size_t attn     = profile.layer_attn_forced_bytes[i];
        const size_t ffn      = profile.layer_ffn_split_bytes[i];
        const size_t mirrored = profile.layer_mirrored_bytes[i];

        if (il < gpu_end) {
            if (!llama_hybrid_add_bytes(gpu_memory, total) || !llama_hybrid_add_bytes(gpu_memory, kv_per_layer)) {
                return false;
            }
        } else if (il < cpu_end) {
            if (!llama_hybrid_add_bytes(pc_memory, total) || !llama_hybrid_add_bytes(pc_memory, kv_per_layer)) {
                return false;
            }
        } else if (il < tensor_end) {
            const size_t pc_ffn    = llama_hybrid_ratio_bytes(ffn, pc_ratio);
            const size_t phone_ffn = ffn - pc_ffn;
            if (!llama_hybrid_add_bytes(pc_memory, attn) || !llama_hybrid_add_bytes(pc_memory, pc_ffn) ||
                !llama_hybrid_add_bytes(pc_memory, mirrored) || !llama_hybrid_add_bytes(pc_memory, kv_per_layer) ||
                !llama_hybrid_add_bytes(phone_memory, phone_ffn) || !llama_hybrid_add_bytes(phone_memory, mirrored)) {
                return false;
            }
        } else {
            if (!llama_hybrid_add_bytes(phone_memory, attn) || !llama_hybrid_add_bytes(phone_memory, ffn) ||
                !llama_hybrid_add_bytes(phone_memory, mirrored) ||
                !llama_hybrid_add_bytes(phone_memory, kv_per_layer) || !llama_hybrid_add_bytes(pc_memory, mirrored)) {
                return false;
            }
        }
    }

    if (gpu_memory > 0 && !llama_hybrid_add_bytes(gpu_memory, constraints.gpu_runtime_reserve_bytes)) {
        return false;
    }

    const int cpu_pc_layers = pc_layers - gpu_layers;
    if (tensor_layers > 0 || cpu_pc_layers > 0 || phone_layers > 0) {
        const int work_tokens = constraints.target_ubatch_tokens > 0 ?
            constraints.target_ubatch_tokens : profile.probe_tokens;
        if (work_tokens <= 0) {
            return false;
        }

        int attn_kv_tokens = constraints.score_kv_tokens;
        if (attn_kv_tokens <= 0) {
            attn_kv_tokens = constraints.target_ctx > 0 ? constraints.target_ctx : profile.n_ctx_train;
        }
        attn_kv_tokens = std::max(attn_kv_tokens, work_tokens);

        size_t pc_runtime_peak    = 0;
        size_t phone_runtime_peak = 0;

        if (cpu_pc_layers > 0) {
            const int cpu_tokens = std::min(work_tokens, plan.cpu_chunk_tokens);
            if (cpu_tokens <= 0) {
                return false;
            }

            size_t pc_cpu_attn_runtime = 0;
            size_t pc_full_ffn_runtime = 0;
            if (!llama_hybrid_attn_runtime_bytes(
                    profile.cpu_attn, cpu_tokens, attn_kv_tokens,
                    pc_cpu_attn_runtime) ||
                !llama_hybrid_ffn_runtime_bytes(
                    profile.cpu_ffn, cpu_tokens, 1.0f,
                    pc_full_ffn_runtime)) {
                return false;
            }
            pc_runtime_peak = std::max(
                pc_runtime_peak,
                std::max(pc_cpu_attn_runtime, pc_full_ffn_runtime));
        }

        if (tensor_layers > 0) {
            int tensor_macro_tokens = work_tokens;
            if (cpu_pc_layers > 0) {
                tensor_macro_tokens =
                    std::min(work_tokens, plan.cpu_chunk_tokens);
            } else if (gpu_layers > 0) {
                tensor_macro_tokens =
                    std::min(work_tokens, plan.gpu_chunk_tokens);
            }
            if (tensor_macro_tokens <= 0) {
                return false;
            }

            size_t pc_tensor_attn_runtime = 0;
            if (!llama_hybrid_attn_runtime_bytes(
                    profile.cpu_attn, tensor_macro_tokens,
                    attn_kv_tokens, pc_tensor_attn_runtime)) {
                return false;
            }

            const int chunk_tokens =
                std::min(tensor_macro_tokens, plan.tensor_chunk_tokens);
            if (chunk_tokens <= 0) {
                return false;
            }
            const int chunks =
                (tensor_macro_tokens + chunk_tokens - 1) / chunk_tokens;

            size_t pc_ffn_runtime    = 0;
            size_t phone_ffn_runtime = 0;
            if (!llama_hybrid_ffn_runtime_bytes(
                    profile.cpu_ffn, chunk_tokens, pc_ratio,
                    pc_ffn_runtime) ||
                !llama_hybrid_ffn_runtime_bytes(
                    profile.phone_ffn, chunk_tokens,
                    1.0f - pc_ratio, phone_ffn_runtime)) {
                return false;
            }

            const size_t live_chunks =
                (size_t) std::min(chunks, 2);
            if (profile.n_embd <= 0 || chunk_tokens <= 0 ||
                (size_t) profile.n_embd >
                    std::numeric_limits<size_t>::max() /
                    (size_t) chunk_tokens / sizeof(float) /
                    live_chunks) {
                return false;
            }
            const size_t transfer_runtime =
                (size_t) profile.n_embd *
                (size_t) chunk_tokens *
                sizeof(float) * live_chunks;

            size_t pc_ffn_stage = 0;
            size_t phone_ffn_stage = 0;
            if (!llama_hybrid_add_bytes(
                    pc_ffn_stage, pc_ffn_runtime) ||
                !llama_hybrid_add_bytes(
                    pc_ffn_stage, transfer_runtime) ||
                !llama_hybrid_add_bytes(
                    phone_ffn_stage, phone_ffn_runtime) ||
                !llama_hybrid_add_bytes(
                    phone_ffn_stage, transfer_runtime)) {
                return false;
            }

            const size_t tensor_pc_peak =
                std::max(pc_tensor_attn_runtime, pc_ffn_stage);
            pc_runtime_peak =
                std::max(pc_runtime_peak, tensor_pc_peak);
            phone_runtime_peak =
                std::max(phone_runtime_peak, phone_ffn_stage);
        }

        if (phone_layers > 0) {
            const int phone_tokens =
                std::min(work_tokens, plan.phone_chunk_tokens);
            if (phone_tokens <= 0) {
                return false;
            }

            size_t phone_attn_runtime = 0;
            size_t phone_full_ffn_runtime = 0;
            if (!llama_hybrid_attn_runtime_bytes(
                    profile.phone_attn, phone_tokens,
                    attn_kv_tokens, phone_attn_runtime) ||
                !llama_hybrid_ffn_runtime_bytes(
                    profile.phone_ffn, phone_tokens, 1.0f,
                    phone_full_ffn_runtime)) {
                return false;
            }
            const size_t phone_stage_peak =
                std::max(phone_attn_runtime, phone_full_ffn_runtime);
            phone_runtime_peak =
                std::max(phone_runtime_peak, phone_stage_peak);
        }

        if (!llama_hybrid_add_bytes(pc_memory, pc_runtime_peak) ||
            !llama_hybrid_add_bytes(phone_memory, phone_runtime_peak)) {
            return false;
        }
    }

    plan.pc_memory    = pc_memory;
    plan.phone_memory = phone_memory;
    plan.gpu_memory   = gpu_memory;
    return true;
}

static bool llama_hybrid_attn_cost_at_query(const std::vector<llama_hybrid_attn_compute_point> & points,
                                            int                                                  query_tokens,
                                            int                                                  kv_tokens,
                                            double &                                             result_ms) {
    const llama_hybrid_attn_compute_point * exact = nullptr;
    const llama_hybrid_attn_compute_point * lower = nullptr;
    const llama_hybrid_attn_compute_point * upper = nullptr;

    for (const auto & point : points) {
        if (point.tokens != query_tokens) {
            continue;
        }
        if (point.kv_tokens == kv_tokens) {
            exact = &point;
            break;
        }
        if (point.kv_tokens < kv_tokens && (lower == nullptr || point.kv_tokens > lower->kv_tokens)) {
            lower = &point;
        }
        if (point.kv_tokens > kv_tokens && (upper == nullptr || point.kv_tokens < upper->kv_tokens)) {
            upper = &point;
        }
    }

    if (exact != nullptr) {
        result_ms = exact->ms;
        return true;
    }
    if (lower == nullptr && upper == nullptr) {
        return false;
    }
    if (lower == nullptr) {
        result_ms = upper->ms;
        return true;
    }
    if (upper == nullptr) {
        const llama_hybrid_attn_compute_point * prev = nullptr;
        for (const auto & point : points) {
            if (point.tokens == query_tokens && point.kv_tokens < lower->kv_tokens &&
                (prev == nullptr || point.kv_tokens > prev->kv_tokens)) {
                prev = &point;
            }
        }
        if (prev == nullptr || lower->kv_tokens == prev->kv_tokens) {
            result_ms = lower->ms;
            return true;
        }
        const double slope = std::max(0.0, (lower->ms - prev->ms) /
                                               (double) (lower->kv_tokens - prev->kv_tokens));
        result_ms = lower->ms + slope * (kv_tokens - lower->kv_tokens);
        return true;
    }

    const double t = (double) (kv_tokens - lower->kv_tokens) / (upper->kv_tokens - lower->kv_tokens);
    result_ms      = lower->ms + t * (upper->ms - lower->ms);
    return true;
}

static bool llama_hybrid_attn_cost(const std::vector<llama_hybrid_attn_compute_point> & points,
                                   int                                                  query_tokens,
                                   int                                                  kv_tokens,
                                   double &                                             result_ms) {
    if (points.empty() || query_tokens <= 0 || kv_tokens <= 0) {
        return false;
    }

    std::vector<int> queries;
    queries.reserve(points.size());
    for (const auto & point : points) {
        if (point.tokens > 0) {
            queries.push_back(point.tokens);
        }
    }
    std::sort(queries.begin(), queries.end());
    queries.erase(std::unique(queries.begin(), queries.end()), queries.end());
    if (queries.empty()) {
        return false;
    }

    int lower_query = -1;
    int upper_query = -1;
    for (const int q : queries) {
        if (q == query_tokens) {
            return llama_hybrid_attn_cost_at_query(points, q, std::max(kv_tokens, q), result_ms);
        }
        if (q < query_tokens) {
            lower_query = q;
        } else if (q > query_tokens && upper_query < 0) {
            upper_query = q;
        }
    }

    if (lower_query < 0) {
        return llama_hybrid_attn_cost_at_query(points, queries.front(), std::max(kv_tokens, queries.front()), result_ms);
    }
    if (upper_query < 0) {
        // Extrapolate query-token cost from the last two measured query sizes. This is safer
        // than assuming perfect proportionality while still allowing a larger reference chunk.
        if (queries.size() < 2) {
            return llama_hybrid_attn_cost_at_query(points, lower_query, std::max(kv_tokens, lower_query), result_ms);
        }
        const int prev_query = queries[queries.size() - 2];
        double prev_ms = 0.0;
        double lower_ms = 0.0;
        if (!llama_hybrid_attn_cost_at_query(points, prev_query, std::max(kv_tokens, prev_query), prev_ms) ||
            !llama_hybrid_attn_cost_at_query(points, lower_query, std::max(kv_tokens, lower_query), lower_ms)) {
            return false;
        }
        const double slope = std::max(0.0, (lower_ms - prev_ms) / (double) (lower_query - prev_query));
        result_ms = lower_ms + slope * (query_tokens - lower_query);
        return true;
    }

    double lower_ms = 0.0;
    double upper_ms = 0.0;
    if (!llama_hybrid_attn_cost_at_query(points, lower_query, std::max(kv_tokens, lower_query), lower_ms) ||
        !llama_hybrid_attn_cost_at_query(points, upper_query, std::max(kv_tokens, upper_query), upper_ms)) {
        return false;
    }
    const double t = (double) (query_tokens - lower_query) / (upper_query - lower_query);
    result_ms      = lower_ms + t * (upper_ms - lower_ms);
    return true;
}

static bool llama_hybrid_ffn_cost_at_ratio(const std::vector<llama_hybrid_ffn_compute_point> & points,
                                           int                                                 tokens,
                                           float                                               ratio,
                                           double &                                            result_ms) {
    const llama_hybrid_ffn_compute_point * exact = nullptr;
    const llama_hybrid_ffn_compute_point * lower = nullptr;
    const llama_hybrid_ffn_compute_point * upper = nullptr;

    for (const auto & point : points) {
        if (std::fabs(point.local_ratio - ratio) >= 1e-4f) {
            continue;
        }
        if (point.tokens == tokens) {
            exact = &point;
            break;
        }
        if (point.tokens < tokens && (lower == nullptr || point.tokens > lower->tokens)) {
            lower = &point;
        }
        if (point.tokens > tokens && (upper == nullptr || point.tokens < upper->tokens)) {
            upper = &point;
        }
    }

    if (exact != nullptr) {
        result_ms = exact->ms;
        return true;
    }
    if (lower == nullptr && upper == nullptr) {
        return false;
    }
    if (lower == nullptr) {
        result_ms = upper->ms;
        return true;
    }
    if (upper == nullptr) {
        result_ms = lower->tokens > 0 ? lower->ms * (double) tokens / lower->tokens : lower->ms;
        return true;
    }

    const double t = (double) (tokens - lower->tokens) / (double) (upper->tokens - lower->tokens);
    result_ms      = lower->ms + t * (upper->ms - lower->ms);
    return true;
}

static bool llama_hybrid_ffn_cost(const std::vector<llama_hybrid_ffn_compute_point> & points,
                                  int                                                 tokens,
                                  float                                               ratio,
                                  double &                                            result_ms) {
    if (points.empty() || tokens <= 0 || ratio <= 0.0f || ratio > 1.0f) {
        return false;
    }

    std::vector<float> ratios;
    ratios.reserve(points.size());
    for (const auto & point : points) {
        if (point.local_ratio > 0.0f && point.local_ratio <= 1.0f) {
            ratios.push_back(point.local_ratio);
        }
    }
    std::sort(ratios.begin(), ratios.end());
    ratios.erase(std::unique(ratios.begin(), ratios.end(), [](float a, float b) {
        return std::fabs(a - b) < 1e-4f;
    }), ratios.end());
    if (ratios.empty()) {
        return false;
    }

    float lower_ratio = -1.0f;
    float upper_ratio = -1.0f;
    for (const float r : ratios) {
        if (std::fabs(r - ratio) < 1e-4f) {
            return llama_hybrid_ffn_cost_at_ratio(points, tokens, r, result_ms);
        }
        if (r < ratio) {
            lower_ratio = r;
        } else if (r > ratio && upper_ratio < 0.0f) {
            upper_ratio = r;
        }
    }

    if (lower_ratio < 0.0f) {
        return llama_hybrid_ffn_cost_at_ratio(points, tokens, ratios.front(), result_ms);
    }
    if (upper_ratio < 0.0f) {
        return llama_hybrid_ffn_cost_at_ratio(points, tokens, ratios.back(), result_ms);
    }

    double lower_ms = 0.0;
    double upper_ms = 0.0;
    if (!llama_hybrid_ffn_cost_at_ratio(points, tokens, lower_ratio, lower_ms) ||
        !llama_hybrid_ffn_cost_at_ratio(points, tokens, upper_ratio, upper_ms)) {
        return false;
    }

    const double t = (double) (ratio - lower_ratio) / (double) (upper_ratio - lower_ratio);
    result_ms      = lower_ms + t * (upper_ms - lower_ms);
    return true;
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
        const llama_hybrid_transfer_point * prev = nullptr;
        for (const auto & point : points) {
            if (point.bytes < lower->bytes && (prev == nullptr || point.bytes > prev->bytes)) {
                prev = &point;
            }
        }
        if (prev == nullptr || lower->bytes == prev->bytes) {
            result_ms = lower->ms;
            return true;
        }
        const double slope = std::max(0.0, (lower->ms - prev->ms) / (double) (lower->bytes - prev->bytes));
        result_ms = lower->ms + slope * (double) (bytes - lower->bytes);
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

static bool llama_hybrid_layer_block_cost_at_depth(
        const std::vector<llama_hybrid_layer_compute_point> & points,
        int tokens,
        int profile_layers,
        bool use_compute_est,
        double & per_layer_ms) {
    if (points.empty() || tokens <= 0 || profile_layers <= 0) {
        return false;
    }

    const llama_hybrid_layer_compute_point * exact = nullptr;
    const llama_hybrid_layer_compute_point * lower = nullptr;
    const llama_hybrid_layer_compute_point * upper = nullptr;
    for (const auto & point : points) {
        if (point.layers != profile_layers) {
            continue;
        }
        if (point.tokens == tokens) {
            exact = &point;
            break;
        }
        if (point.tokens < tokens &&
            (lower == nullptr || point.tokens > lower->tokens)) {
            lower = &point;
        }
        if (point.tokens > tokens &&
            (upper == nullptr || point.tokens < upper->tokens)) {
            upper = &point;
        }
    }

    const auto value = [&](const llama_hybrid_layer_compute_point * point) {
        const double total =
            use_compute_est ? point->compute_est_ms : point->wall_ms;
        return total / point->layers;
    };

    if (exact != nullptr) {
        per_layer_ms = value(exact);
        return true;
    }
    if (lower == nullptr && upper == nullptr) {
        return false;
    }
    if (lower == nullptr) {
        per_layer_ms = value(upper);
        return true;
    }
    if (upper == nullptr) {
        const llama_hybrid_layer_compute_point * prev = nullptr;
        for (const auto & point : points) {
            if (point.layers == profile_layers &&
                point.tokens < lower->tokens &&
                (prev == nullptr || point.tokens > prev->tokens)) {
                prev = &point;
            }
        }
        if (prev == nullptr || lower->tokens == prev->tokens) {
            per_layer_ms = value(lower);
            return true;
        }
        const double lower_ms = value(lower);
        const double prev_ms  = value(prev);
        const double slope = std::max(
            0.0,
            (lower_ms - prev_ms) /
                (double) (lower->tokens - prev->tokens));
        per_layer_ms =
            lower_ms + slope * (tokens - lower->tokens);
        return true;
    }

    const double lower_ms = value(lower);
    const double upper_ms = value(upper);
    const double t =
        (double) (tokens - lower->tokens) /
        (double) (upper->tokens - lower->tokens);
    per_layer_ms = lower_ms + t * (upper_ms - lower_ms);
    return true;
}

static bool llama_hybrid_layer_block_cost(
        const std::vector<llama_hybrid_layer_compute_point> & points,
        int tokens,
        bool use_compute_est,
        double & per_layer_ms) {
    int min_layers = std::numeric_limits<int>::max();
    for (const auto & point : points) {
        if (point.layers > 0) {
            min_layers = std::min(min_layers, point.layers);
        }
    }
    if (min_layers == std::numeric_limits<int>::max()) {
        return false;
    }
    return llama_hybrid_layer_block_cost_at_depth(
        points, tokens, min_layers, use_compute_est, per_layer_ms);
}

static bool llama_hybrid_layer_block_total_cost(
        const std::vector<llama_hybrid_layer_compute_point> & points,
        int tokens,
        int target_layers,
        bool use_compute_est,
        double & total_ms) {
    total_ms = 0.0;
    if (target_layers == 0) {
        return true;
    }
    if (points.empty() || tokens <= 0 || target_layers < 0) {
        return false;
    }

    std::vector<std::pair<int, double>> depth_costs;
    std::set<int> seen_layers;
    for (const auto & point : points) {
        if (point.layers <= 0 || !seen_layers.insert(point.layers).second) {
            continue;
        }
        double per_layer_ms = 0.0;
        if (llama_hybrid_layer_block_cost_at_depth(
                points, tokens, point.layers, use_compute_est,
                per_layer_ms)) {
            depth_costs.emplace_back(point.layers, per_layer_ms);
        }
    }
    if (depth_costs.empty()) {
        return false;
    }
    std::sort(depth_costs.begin(), depth_costs.end());

    double per_layer_ms = depth_costs.front().second;
    if (target_layers <= depth_costs.front().first) {
        per_layer_ms = depth_costs.front().second;
    } else if (target_layers >= depth_costs.back().first) {
        // A measured multi-layer block represents the sustained regime. Once
        // the requested stage is deeper than the largest probe, clamp to that
        // sustained per-layer rate rather than extrapolating a 1-layer rate.
        per_layer_ms = depth_costs.back().second;
    } else {
        for (size_t i = 0; i + 1 < depth_costs.size(); ++i) {
            if (target_layers < depth_costs[i].first ||
                target_layers > depth_costs[i + 1].first) {
                continue;
            }
            const double t =
                (double) (target_layers - depth_costs[i].first) /
                (double) (depth_costs[i + 1].first -
                          depth_costs[i].first);
            per_layer_ms =
                depth_costs[i].second +
                t * (depth_costs[i + 1].second -
                     depth_costs[i].second);
            break;
        }
    }

    total_ms = target_layers * std::max(0.0, per_layer_ms);
    return std::isfinite(total_ms);
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

struct llama_hybrid_tensor_ffn_detail {
    int    chunks            = 0;
    double h2d_sum_ms        = 0.0;
    double pc_ffn_sum_ms     = 0.0;
    double phone_sum_ms      = 0.0;
    double d2h_sum_ms        = 0.0;
    double h2d_finish_ms     = 0.0;
    double pc_finish_ms      = 0.0;
    double phone_finish_ms   = 0.0;
    double return_finish_ms  = 0.0;
    double reduce_tail_ms    = 0.0;
    double done_ms           = 0.0;
    double overlap_saved_ms  = 0.0;
};

static bool llama_hybrid_tensor_ffn_cost(const llama_hybrid_profile & profile,
                                         float                        pc_ratio,
                                         int                          total_tokens,
                                         int                          chunk_tokens,
                                         double &                     result_ms,
                                         llama_hybrid_tensor_ffn_detail * detail = nullptr) {
    const std::vector<int> chunks = llama_hybrid_split_by_chunk_size(total_tokens, chunk_tokens);
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

    double h2d_sum_ms    = 0.0;
    double cpu_sum_ms    = 0.0;
    double phone_sum_ms  = 0.0;
    double d2h_sum_ms    = 0.0;

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

        h2d_sum_ms   += h2d_ms;
        cpu_sum_ms   += cpu_ms;
        phone_sum_ms += phone_ms;
        d2h_sum_ms   += d2h_single_ms;

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

    if (detail != nullptr) {
        detail->chunks            = (int) chunks.size();
        detail->h2d_sum_ms        = h2d_sum_ms;
        detail->pc_ffn_sum_ms     = cpu_sum_ms;
        detail->phone_sum_ms      = phone_sum_ms;
        detail->d2h_sum_ms        = d2h_sum_ms;
        detail->h2d_finish_ms     = h2d_available;
        detail->pc_finish_ms      = cpu_available;
        detail->phone_finish_ms   = phone_available;
        detail->return_finish_ms  = return_wave_available;
        detail->reduce_tail_ms    = profile.reduce_ms;
        detail->done_ms           = done_ms;

        const double serial_ms =
            h2d_sum_ms + cpu_sum_ms + phone_sum_ms + d2h_sum_ms + profile.reduce_ms;
        detail->overlap_saved_ms = std::max(0.0, serial_ms - done_ms);
    }

    return true;
}

static double llama_hybrid_tensor_misc_cost(const llama_hybrid_profile & profile,
                                             int                          tokens,
                                             double                       cpu_layer_base_ms,
                                             double                       cpu_attn_base_ms) {
    double full_ffn_ms = 0.0;
    if (!llama_hybrid_ffn_cost(profile.cpu_ffn, tokens, 1.0f, full_ffn_ms)) {
        return 0.0;
    }
    return std::max(0.0, cpu_layer_base_ms - cpu_attn_base_ms - full_ffn_ms);
}

bool llama_hybrid_runtime_predict_tensor_compute(
        int tokens, llama_hybrid_tensor_compute_prediction & prediction) {
    prediction = {};
    if (tokens <= 0) {
        return false;
    }

    llama_hybrid_plan plan;
    llama_hybrid_profile profile;
    llama_hybrid_constraints constraints;
    {
        std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
        if (!g_llama_hybrid_runtime_plan.has_value() ||
            !g_llama_hybrid_runtime_profile.has_value() ||
            !g_llama_hybrid_runtime_constraints.has_value()) {
            return false;
        }
        plan        = *g_llama_hybrid_runtime_plan;
        profile     = *g_llama_hybrid_runtime_profile;
        constraints = *g_llama_hybrid_runtime_constraints;
    }

    if (plan.tensor_layers <= 0 || plan.tensor_chunk_tokens <= 0) {
        return false;
    }

    int kv_tokens = constraints.score_kv_tokens;
    if (kv_tokens <= 0) {
        kv_tokens = constraints.target_ctx > 0 ? constraints.target_ctx : profile.n_ctx_train;
    }
    kv_tokens = std::max(kv_tokens, tokens);
    if (profile.n_ctx_train > 0) {
        kv_tokens = std::min(kv_tokens, profile.n_ctx_train);
    }

    const std::vector<int> chunks =
        llama_hybrid_split_by_chunk_size(tokens, plan.tensor_chunk_tokens);
    if (chunks.empty()) {
        return false;
    }

    int attn_group_chunks = llama_hybrid_runtime_prefill_attn_group_chunks();
    double per_layer_attn_misc = 0.0;
    double per_layer_pc_ffn    = 0.0;

    if (profile.is_moe) {
        // Qwen3-MoE stage execution computes Attention and FFN norm once
        // for the whole stage macro, then runs Router/Top-K/experts per XT.
        // Keep the predictor aligned with that graph shape.
        double macro_attn_ms = 0.0;
        double macro_layer_base = 0.0;
        double macro_attn_base = 0.0;
        if (!llama_hybrid_attn_cost(
                profile.cpu_attn, tokens, kv_tokens, macro_attn_ms) ||
            !llama_hybrid_layer_block_cost(
                profile.cpu_layer_blocks, tokens, false,
                macro_layer_base) ||
            !llama_hybrid_attn_cost(
                profile.cpu_attn, tokens, tokens,
                macro_attn_base)) {
            return false;
        }

        per_layer_attn_misc =
            macro_attn_ms +
            llama_hybrid_tensor_misc_cost(
                profile, tokens,
                macro_layer_base, macro_attn_base);

        for (const int chunk_tokens : chunks) {
            double chunk_pc_ffn = 0.0;
            if (!llama_hybrid_ffn_cost(
                    profile.cpu_ffn, chunk_tokens,
                    plan.tensor_pc_ratio,
                    chunk_pc_ffn)) {
                return false;
            }
            per_layer_pc_ffn += chunk_pc_ffn;
        }

        // Attention is not XT-grouped in the Qwen3-MoE staged graph.
        attn_group_chunks = (int) chunks.size();
    } else {
        // Dense Qwen2 keeps its existing attention-group / XT-granular
        // accounting.
        for (size_t group_begin = 0; group_begin < chunks.size();
             group_begin += (size_t) attn_group_chunks) {
            const size_t group_end = std::min(
                chunks.size(), group_begin + (size_t) attn_group_chunks);
            int group_tokens = 0;
            for (size_t ci = group_begin; ci < group_end; ++ci) {
                group_tokens += chunks[ci];
            }

            double group_attn_ms = 0.0;
            if (!llama_hybrid_attn_cost(
                    profile.cpu_attn, group_tokens, kv_tokens,
                    group_attn_ms)) {
                return false;
            }
            per_layer_attn_misc += group_attn_ms;

            for (size_t ci = group_begin; ci < group_end; ++ci) {
                const int chunk_tokens = chunks[ci];
                double cpu_layer_base = 0.0;
                double cpu_attn_base  = 0.0;
                double chunk_pc_ffn   = 0.0;
                if (!llama_hybrid_layer_block_cost(
                        profile.cpu_layer_blocks,
                        chunk_tokens, false,
                        cpu_layer_base) ||
                    !llama_hybrid_attn_cost(
                        profile.cpu_attn,
                        chunk_tokens, chunk_tokens,
                        cpu_attn_base) ||
                    !llama_hybrid_ffn_cost(
                        profile.cpu_ffn, chunk_tokens,
                        plan.tensor_pc_ratio,
                        chunk_pc_ffn)) {
                    return false;
                }

                per_layer_attn_misc +=
                    llama_hybrid_tensor_misc_cost(
                        profile, chunk_tokens,
                        cpu_layer_base, cpu_attn_base);
                per_layer_pc_ffn += chunk_pc_ffn;
            }
        }
    }

    double per_layer_tensor_pipeline = 0.0;
    llama_hybrid_tensor_ffn_detail pipeline_detail;
    if (!llama_hybrid_tensor_ffn_cost(
            profile, plan.tensor_pc_ratio, tokens, plan.tensor_chunk_tokens,
            per_layer_tensor_pipeline, &pipeline_detail)) {
        return false;
    }

    prediction.tokens             = tokens;
    prediction.kv_tokens          = kv_tokens;
    prediction.tensor_layers       = plan.tensor_layers;
    prediction.tensor_chunk_tokens  = plan.tensor_chunk_tokens;
    prediction.attn_group_chunks    = attn_group_chunks;
    prediction.attn_chunk_tokens    = profile.is_moe ?
        tokens : plan.tensor_chunk_tokens * attn_group_chunks;
    prediction.tensor_pc_ratio      = plan.tensor_pc_ratio;
    prediction.attn_misc_ms       = plan.tensor_layers * per_layer_attn_misc;
    prediction.pc_ffn_ms          = plan.tensor_layers * per_layer_pc_ffn;
    prediction.pc_compute_ms      = prediction.attn_misc_ms + prediction.pc_ffn_ms;
    prediction.tensor_total_ms    =
        plan.tensor_layers * (per_layer_attn_misc + per_layer_tensor_pipeline);

    const double tensor_layers = (double) plan.tensor_layers;
    prediction.tensor_chunks             = pipeline_detail.chunks;
    prediction.pipeline_h2d_sum_ms       = tensor_layers * pipeline_detail.h2d_sum_ms;
    prediction.pipeline_pc_ffn_sum_ms    = tensor_layers * pipeline_detail.pc_ffn_sum_ms;
    prediction.pipeline_phone_sum_ms     = tensor_layers * pipeline_detail.phone_sum_ms;
    prediction.pipeline_d2h_sum_ms       = tensor_layers * pipeline_detail.d2h_sum_ms;
    prediction.pipeline_h2d_finish_ms    = tensor_layers * pipeline_detail.h2d_finish_ms;
    prediction.pipeline_pc_finish_ms     = tensor_layers * pipeline_detail.pc_finish_ms;
    prediction.pipeline_phone_finish_ms  = tensor_layers * pipeline_detail.phone_finish_ms;
    prediction.pipeline_return_finish_ms = tensor_layers * pipeline_detail.return_finish_ms;
    prediction.pipeline_reduce_tail_ms   = tensor_layers * pipeline_detail.reduce_tail_ms;
    prediction.pipeline_done_ms          = tensor_layers * pipeline_detail.done_ms;
    prediction.pipeline_overlap_saved_ms = tensor_layers * pipeline_detail.overlap_saved_ms;

    return std::isfinite(prediction.attn_misc_ms) &&
           std::isfinite(prediction.pc_ffn_ms) &&
           std::isfinite(prediction.pc_compute_ms) &&
           std::isfinite(prediction.tensor_total_ms);
}

static bool llama_hybrid_layer_region_cost(
        const llama_hybrid_profile &                            profile,
        const std::vector<llama_hybrid_layer_compute_point> & layer_points,
        const std::vector<llama_hybrid_attn_compute_point> &  attn_points,
        int                                                    layers,
        int                                                    total_tokens,
        int                                                    chunk_tokens,
        int                                                    kv_tokens,
        bool                                                   use_compute_est,
        double &                                               result_ms);

bool llama_hybrid_runtime_predict_full_prefill(
        int tokens, llama_hybrid_full_prefill_prediction & prediction) {
    prediction = {};
    if (tokens <= 0) {
        return false;
    }

    llama_hybrid_tensor_compute_prediction tensor_prediction;
    if (!llama_hybrid_runtime_predict_tensor_compute(tokens, tensor_prediction)) {
        return false;
    }

    llama_hybrid_plan plan;
    llama_hybrid_profile profile;
    {
        std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
        if (!g_llama_hybrid_runtime_plan.has_value() ||
            !g_llama_hybrid_runtime_profile.has_value()) {
            return false;
        }
        plan    = *g_llama_hybrid_runtime_plan;
        profile = *g_llama_hybrid_runtime_profile;
    }

    // This diagnostic intentionally matches the current full-graph override
    // topology only: a GPU prefix followed by a Tensor-split suffix.
    if (plan.gpu_pc_layers <= 0 ||
        plan.tensor_layers <= 0 ||
        plan.phone_layers != 0 ||
        plan.pc_layers != plan.gpu_pc_layers) {
        return false;
    }

    double gpu_compute_ms = 0.0;
    if (!llama_hybrid_layer_region_cost(
            profile, profile.gpu_layer_blocks, profile.gpu_attn,
            plan.gpu_pc_layers, tokens, tokens,
            tensor_prediction.kv_tokens, false, gpu_compute_ms)) {
        return false;
    }

    const size_t boundary_bytes =
        (size_t) profile.n_embd * (size_t) tokens * sizeof(float);
    double gpu_to_pc_ms = 0.0;
    if (!llama_hybrid_transfer_cost(
            profile.gpu_to_pc, boundary_bytes, gpu_to_pc_ms)) {
        return false;
    }

    prediction.tokens              = tokens;
    prediction.kv_tokens           = tensor_prediction.kv_tokens;
    prediction.tensor_layers       = tensor_prediction.tensor_layers;
    prediction.tensor_chunk_tokens = tensor_prediction.tensor_chunk_tokens;
    prediction.attn_group_chunks   = tensor_prediction.attn_group_chunks;
    prediction.attn_chunk_tokens   = tensor_prediction.attn_chunk_tokens;
    prediction.tensor_pc_ratio     = tensor_prediction.tensor_pc_ratio;
    prediction.gpu_ms              = gpu_compute_ms + gpu_to_pc_ms;
    prediction.tensor_ms           = tensor_prediction.tensor_total_ms;
    prediction.total_ms            = prediction.gpu_ms + prediction.tensor_ms;

    return std::isfinite(prediction.gpu_ms) &&
           std::isfinite(prediction.tensor_ms) &&
           std::isfinite(prediction.total_ms);
}

static float llama_hybrid_align_pc_ratio(const llama_hybrid_profile & profile, float ratio) {
    if (profile.n_ff <= 0 || profile.ffn_shard_granularity <= 0) {
        return std::clamp(ratio, 0.01f, 0.99f);
    }

    const int64_t granularity = profile.ffn_shard_granularity;
    const int64_t max_width   = std::max<int64_t>(granularity,
        ((int64_t) profile.n_ff / granularity - 1) * granularity);
    int64_t width             = (int64_t) std::floor((double) profile.n_ff * ratio);
    width -= width % granularity;
    width = std::clamp<int64_t>(width, granularity, max_width);
    return (float) ((double) width / (double) profile.n_ff);
}

static bool llama_hybrid_tensor_balance_diff(const llama_hybrid_profile & profile,
                                             int                          chunk_tokens,
                                             float                        pc_ratio,
                                             double &                     diff_ms) {
    if (profile.n_embd <= 0 || chunk_tokens <= 0) {
        return false;
    }

    double pc_ms    = 0.0;
    double phone_ms = 0.0;
    double return_ms = 0.0;
    const size_t bytes = (size_t) profile.n_embd * (size_t) chunk_tokens * sizeof(float);
    if (!llama_hybrid_ffn_cost(profile.cpu_ffn, chunk_tokens, pc_ratio, pc_ms) ||
        !llama_hybrid_ffn_cost(profile.phone_ffn, chunk_tokens, 1.0f - pc_ratio, phone_ms) ||
        !llama_hybrid_transfer_cost(profile.snapshot_phone_to_pc, bytes, return_ms)) {
        return false;
    }

    // PC->Phone is intentionally not included in the steady-state balance equation: the
    // transfer worker can prefetch the next chunk while the phone computes the current one.
    // The full tensor pipeline simulator still accounts for the actual H2D timeline.
    diff_ms = pc_ms - (phone_ms + return_ms);
    return true;
}

static std::vector<float> llama_hybrid_plan_ratio_candidates(const llama_hybrid_profile & profile, int chunk_tokens) {
    std::vector<float> anchors;
    for (const auto & point : profile.cpu_ffn) {
        if (point.local_ratio > 0.0f && point.local_ratio < 1.0f) {
            anchors.push_back(point.local_ratio);
        }
    }
    std::sort(anchors.begin(), anchors.end());
    anchors.erase(std::unique(anchors.begin(), anchors.end(), [](float a, float b) {
        return std::fabs(a - b) < 1e-4f;
    }), anchors.end());
    if (anchors.empty() || chunk_tokens <= 0) {
        return {};
    }

    float center = anchors.front();
    double best_abs_diff = std::numeric_limits<double>::infinity();
    bool found_crossing = false;

    for (size_t i = 0; i < anchors.size(); ++i) {
        double diff = 0.0;
        if (llama_hybrid_tensor_balance_diff(profile, chunk_tokens, anchors[i], diff) &&
            std::fabs(diff) < best_abs_diff) {
            best_abs_diff = std::fabs(diff);
            center = anchors[i];
        }
    }

    for (size_t i = 0; i + 1 < anchors.size(); ++i) {
        const float q0 = anchors[i];
        const float q1 = anchors[i + 1];
        double d0 = 0.0;
        double d1 = 0.0;
        if (!llama_hybrid_tensor_balance_diff(profile, chunk_tokens, q0, d0) ||
            !llama_hybrid_tensor_balance_diff(profile, chunk_tokens, q1, d1)) {
            continue;
        }
        if (d0 == 0.0) {
            center = q0;
            found_crossing = true;
            break;
        }
        if (d1 == 0.0) {
            center = q1;
            found_crossing = true;
            break;
        }
        if ((d0 < 0.0) == (d1 < 0.0)) {
            continue;
        }

        const double t = std::clamp(-d0 / (d1 - d0), 0.0, 1.0);
        center = llama_hybrid_align_pc_ratio(profile, (float) (q0 + t * (q1 - q0)));
        found_crossing = true;
        break;
    }

    std::vector<float> result;
    const float ratio_step = profile.n_ff > 0 && profile.ffn_shard_granularity > 0 ?
        (float) ((double) profile.ffn_shard_granularity / (double) profile.n_ff) : 0.0f;
    const auto add = [&](float q) {
        q = llama_hybrid_align_pc_ratio(profile, q);
        if (q > 0.0f && q < 1.0f) {
            result.push_back(q);
        }
    };

    add(center);
    if (ratio_step > 0.0f) {
        add(center - ratio_step);
        add(center + ratio_step);
    }

    auto upper = std::lower_bound(anchors.begin(), anchors.end(), center);
    if (upper != anchors.end()) {
        add(*upper);
    }
    if (upper != anchors.begin()) {
        add(*std::prev(upper));
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end(), [](float a, float b) {
        return std::fabs(a - b) < 1e-4f;
    }), result.end());

    if (!found_crossing && result.size() > 5) {
        std::stable_sort(result.begin(), result.end(), [center](float a, float b) {
            return std::fabs(a - center) < std::fabs(b - center);
        });
        result.resize(5);
        std::sort(result.begin(), result.end());
    }
    return result;
}



struct llama_hybrid_decode_tensor_cost {
    double total_ms     = 0.0;
    double base_ms      = 0.0;
    double pc_ffn_ms    = 0.0;
    double phone_ffn_ms = 0.0;
    double h2d_ms       = 0.0;
    double d2h_ms       = 0.0;
    double reduce_ms    = 0.0;
};

static bool llama_hybrid_decode_region_cost(
        const std::vector<llama_hybrid_layer_compute_point> & layer_points,
        const std::vector<llama_hybrid_attn_compute_point> & attn_points,
        int kv_tokens,
        int target_layers,
        bool use_compute_est,
        double & result_ms) {
    result_ms = 0.0;
    if (target_layers == 0) {
        return true;
    }
    if (target_layers < 0) {
        return false;
    }

    double layer_total = 0.0;
    double attn_base   = 0.0;
    double attn_kv     = 0.0;
    if (!llama_hybrid_layer_block_total_cost(
            layer_points, 1, target_layers,
            use_compute_est, layer_total) ||
        !llama_hybrid_attn_cost(
            attn_points, 1, 1, attn_base) ||
        !llama_hybrid_attn_cost(
            attn_points, 1, kv_tokens, attn_kv)) {
        return false;
    }

    // layer_total is measured/calibrated at token=1, kv=1.  Decode keeps one
    // query token but attends over the target KV context, so add only the
    // per-layer attention delta while preserving the measured depth effect
    // for the rest of the CPU layer.
    result_ms = std::max(
        0.0,
        layer_total +
        target_layers * (attn_kv - attn_base));
    return std::isfinite(result_ms);
}

static bool llama_hybrid_decode_layer_cost(
        const std::vector<llama_hybrid_layer_compute_point> & layer_points,
        const std::vector<llama_hybrid_attn_compute_point> & attn_points,
        int kv_tokens,
        bool use_compute_est,
        double & result_ms) {
    return llama_hybrid_decode_region_cost(
        layer_points, attn_points,
        kv_tokens, 1, use_compute_est, result_ms);
}

static bool llama_hybrid_decode_tensor_layer_cost(
        const llama_hybrid_profile & profile,
        int kv_tokens,
        float pc_ratio,
        llama_hybrid_decode_tensor_cost & cost) {
    cost = {};
    if (profile.n_embd <= 0 || pc_ratio <= 0.0f || pc_ratio >= 1.0f) {
        return false;
    }

    double cpu_layer_base = 0.0;
    double cpu_attn_base  = 0.0;
    double cpu_attn_kv    = 0.0;
    double cpu_ffn_full   = 0.0;
    if (!llama_hybrid_layer_block_cost(
            profile.cpu_layer_blocks, 1, false, cpu_layer_base) ||
        !llama_hybrid_attn_cost(profile.cpu_attn, 1, 1, cpu_attn_base) ||
        !llama_hybrid_attn_cost(profile.cpu_attn, 1, kv_tokens, cpu_attn_kv) ||
        !llama_hybrid_ffn_cost(profile.cpu_ffn, 1, 1.0f, cpu_ffn_full) ||
        !llama_hybrid_ffn_cost(profile.cpu_ffn, 1, pc_ratio, cost.pc_ffn_ms) ||
        !llama_hybrid_ffn_cost(profile.phone_ffn, 1, 1.0f - pc_ratio, cost.phone_ffn_ms)) {
        return false;
    }

    const size_t hidden_bytes =
        (size_t) profile.n_embd * sizeof(float);
    if (!llama_hybrid_transfer_cost(
            profile.pc_to_phone, hidden_bytes, cost.h2d_ms) ||
        !llama_hybrid_transfer_cost(
            profile.phone_to_pc, hidden_bytes, cost.d2h_ms)) {
        return false;
    }

    // Reuse the measured full CPU layer for attention/norm/router-independent
    // work, replacing only the full local FFN branch with the split branches.
    cost.base_ms = std::max(
        0.0,
        cpu_layer_base - cpu_ffn_full + (cpu_attn_kv - cpu_attn_base));

    // Decode has no cross-layer stage pipeline today. Within a tensor layer the
    // local PC branch can overlap the H2D + phone branch; the returned partial
    // result and reduction are on the critical path afterwards.
    cost.reduce_ms = std::max(0.0, profile.reduce_ms);
    const double branch_ms = std::max(
        cost.pc_ffn_ms,
        cost.h2d_ms + cost.phone_ffn_ms);
    cost.total_ms =
        cost.base_ms + branch_ms + cost.d2h_ms + cost.reduce_ms;
    return std::isfinite(cost.total_ms);
}

static std::vector<float> llama_hybrid_decode_ratio_candidates(
        const llama_hybrid_profile & profile) {
    std::vector<float> result;
    for (const auto & point : profile.cpu_ffn) {
        if (point.tokens != 1 ||
            point.local_ratio <= 0.0f ||
            point.local_ratio >= 1.0f) {
            continue;
        }
        result.push_back(
            llama_hybrid_align_pc_ratio(profile, point.local_ratio));
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end(), [](float a, float b) {
        return std::fabs(a - b) < 1e-4f;
    }), result.end());
    return result;
}

void llama_hybrid_decode_plan_print(
        const llama_hybrid_decode_plan & plan) {
    const double tps =
        plan.predicted_ms > 0.0 ? 1000.0 / plan.predicted_ms : 0.0;
    LLAMA_LOG_ERROR(
        "[DECODE_PLAN] T=%d P=%d C=%d G=%d R=%.3f LLAMA_CHUNKS=%d "
        "kv_tokens=%d predicted_ms=%.3f predicted_tps=%.3f "
        "gpu_ms=%.3f cpu_ms=%.3f tensor_ms=%.3f phone_ms=%.3f "
        "boundary_ms=%.3f pc_mib=%.1f phone_mib=%.1f gpu_mib=%.1f "
        "mode=PREDICT_ONLY applied=0\n",
        plan.tensor_layers,
        plan.phone_layers,
        plan.cpu_layers,
        plan.gpu_layers,
        plan.tensor_pc_ratio,
        plan.llama_chunks,
        plan.kv_tokens,
        plan.predicted_ms,
        tps,
        plan.predicted_gpu_ms,
        plan.predicted_cpu_ms,
        plan.predicted_tensor_ms,
        plan.predicted_phone_ms,
        plan.predicted_boundary_ms,
        plan.pc_memory / 1048576.0,
        plan.phone_memory / 1048576.0,
        plan.gpu_memory / 1048576.0);
}

bool llama_hybrid_predict_decode_plan(
        const llama_hybrid_profile & profile,
        const llama_hybrid_constraints & constraints,
        llama_hybrid_decode_plan & best_plan) {
    best_plan = {};
    if (profile.n_layer <= 0 || profile.n_embd <= 0) {
        return false;
    }

    int kv_tokens = constraints.score_kv_tokens;
    if (kv_tokens <= 0) {
        kv_tokens = constraints.target_ctx > 0 ?
            constraints.target_ctx : profile.n_ctx_train;
    }
    kv_tokens = std::max(1, kv_tokens);
    if (profile.n_ctx_train > 0) {
        kv_tokens = std::min(kv_tokens, profile.n_ctx_train);
    }

    double gpu_layer_ms = 0.0;
    double cpu_layer_ms = 0.0;
    double phone_layer_ms = 0.0;
    if (!llama_hybrid_decode_layer_cost(
            profile.cpu_layer_blocks, profile.cpu_attn,
            kv_tokens, false, cpu_layer_ms) ||
        !llama_hybrid_decode_layer_cost(
            profile.phone_layer_blocks, profile.phone_attn,
            kv_tokens, true, phone_layer_ms)) {
        LLAMA_LOG_ERROR(
            "[DECODE_PLAN] unavailable reason=missing_cpu_or_phone_decode_profile kv_tokens=%d\n",
            kv_tokens);
        return false;
    }

    const bool have_gpu =
        !profile.gpu_layer_blocks.empty() && !profile.gpu_attn.empty() &&
        llama_hybrid_decode_layer_cost(
            profile.gpu_layer_blocks, profile.gpu_attn,
            kv_tokens, false, gpu_layer_ms);

    std::vector<double> cpu_decode_region_ms(
        (size_t) profile.n_layer + 1, 0.0);
    for (int layers = 1; layers <= profile.n_layer; ++layers) {
        if (!llama_hybrid_decode_region_cost(
                profile.cpu_layer_blocks,
                profile.cpu_attn,
                kv_tokens,
                layers,
                false,
                cpu_decode_region_ms[(size_t) layers])) {
            LLAMA_LOG_ERROR(
                "[DECODE_PLAN] unavailable reason=missing_cpu_depth_profile "
                "layers=%d kv_tokens=%d\n",
                layers, kv_tokens);
            return false;
        }
    }

    if (profile.is_moe &&
        profile.n_layer >= LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS) {
        double token1_one_layer_ms = 0.0;
        double token1_sustained_per_layer_ms = 0.0;
        const bool have_one_layer =
            llama_hybrid_layer_block_cost_at_depth(
                profile.cpu_layer_blocks,
                1,
                1,
                false,
                token1_one_layer_ms);
        const bool have_sustained =
            llama_hybrid_layer_block_cost_at_depth(
                profile.cpu_layer_blocks,
                1,
                LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS,
                false,
                token1_sustained_per_layer_ms);
        if (have_one_layer && have_sustained) {
            const int deep_layers = profile.n_layer;
            LLAMA_LOG_ERROR(
                "[DECODE_CPU_DEPTH] tokens=1 kv_tokens=%d "
                "probe_layers=%d one_layer_base_ms=%.3f "
                "sustained_per_layer_base_ms=%.3f sustained_over_one=%.3f "
                "C1_ms=%.3f C%d_ms=%.3f C%d_ms=%.3f\n",
                kv_tokens,
                LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS,
                token1_one_layer_ms,
                token1_sustained_per_layer_ms,
                token1_one_layer_ms > 0.0 ?
                    token1_sustained_per_layer_ms / token1_one_layer_ms : 0.0,
                cpu_decode_region_ms[1],
                LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS,
                cpu_decode_region_ms[
                    (size_t) LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS],
                deep_layers,
                cpu_decode_region_ms[(size_t) deep_layers]);
        }
    }

    const auto ratios = llama_hybrid_decode_ratio_candidates(profile);
    if (ratios.empty()) {
        LLAMA_LOG_ERROR(
            "[DECODE_PLAN] unavailable reason=missing_token1_tensor_ratio_profile\n");
        return false;
    }

    bool ratio_found = false;
    float best_ratio = 0.5f;
    llama_hybrid_decode_tensor_cost best_tensor_cost;
    for (const float ratio : ratios) {
        llama_hybrid_decode_tensor_cost tensor_cost;
        if (!llama_hybrid_decode_tensor_layer_cost(
                profile, kv_tokens, ratio, tensor_cost)) {
            continue;
        }
        LLAMA_LOG_ERROR(
            "[DECODE_RATIO] R=%.5f layer_ms=%.3f base_ms=%.3f "
            "pc_ffn_ms=%.3f phone_ffn_ms=%.3f h2d_ms=%.3f "
            "d2h_ms=%.3f reduce_ms=%.3f\n",
            ratio,
            tensor_cost.total_ms,
            tensor_cost.base_ms,
            tensor_cost.pc_ffn_ms,
            tensor_cost.phone_ffn_ms,
            tensor_cost.h2d_ms,
            tensor_cost.d2h_ms,
            tensor_cost.reduce_ms);
        if (!ratio_found || tensor_cost.total_ms < best_tensor_cost.total_ms) {
            ratio_found = true;
            best_ratio = ratio;
            best_tensor_cost = tensor_cost;
        }
    }
    if (!ratio_found) {
        return false;
    }

    LLAMA_LOG_ERROR(
        "[DECODE_RATIO_BEST] R=%.5f tensor_layer_ms=%.3f kv_tokens=%d\n",
        best_ratio, best_tensor_cost.total_ms, kv_tokens);

    const size_t hidden_bytes =
        (size_t) profile.n_embd * sizeof(float);
    double gpu_to_pc_ms = 0.0;
    double pc_to_phone_ms = 0.0;
    double phone_to_pc_ms = 0.0;
    if (!llama_hybrid_transfer_cost(
            profile.pc_to_phone, hidden_bytes, pc_to_phone_ms) ||
        !llama_hybrid_transfer_cost(
            profile.phone_to_pc, hidden_bytes, phone_to_pc_ms)) {
        return false;
    }
    if (have_gpu &&
        !llama_hybrid_transfer_cost(
            profile.gpu_to_pc, hidden_bytes, gpu_to_pc_ms)) {
        return false;
    }

    llama_hybrid_constraints decode_constraints = constraints;
    decode_constraints.target_ubatch_tokens = 1;
    decode_constraints.max_tensor_chunks = 1;

    const size_t pc_budget = llama_hybrid_effective_budget(
        decode_constraints.pc_memory_budget,
        profile.pc_free_mem,
        LLAMA_HYBRID_PC_MEMORY_FRACTION);
    const size_t phone_budget = llama_hybrid_effective_budget(
        decode_constraints.phone_memory_budget,
        profile.phone_free_mem,
        LLAMA_HYBRID_PHONE_MEMORY_FRACTION);
    const size_t gpu_budget = llama_hybrid_effective_budget(
        decode_constraints.gpu_memory_budget,
        profile.gpu_free_mem,
        LLAMA_HYBRID_GPU_MEMORY_FRACTION);

    struct ranked_decode_plan {
        llama_hybrid_decode_plan plan;
    };
    std::vector<ranked_decode_plan> top;
    bool found = false;

    for (int gpu_layers = 0; gpu_layers <= profile.n_layer; ++gpu_layers) {
        if (gpu_layers > 0 && !have_gpu) {
            break;
        }
        const int after_gpu = profile.n_layer - gpu_layers;
        for (int cpu_layers = 0; cpu_layers <= after_gpu; ++cpu_layers) {
            const int after_cpu = after_gpu - cpu_layers;
            for (int tensor_layers = 0; tensor_layers <= after_cpu; ++tensor_layers) {
                const int phone_layers = after_cpu - tensor_layers;

                llama_hybrid_plan memory_plan;
                memory_plan.tensor_layers = tensor_layers;
                memory_plan.phone_layers = phone_layers;
                memory_plan.pc_layers = gpu_layers + cpu_layers;
                memory_plan.gpu_pc_layers = gpu_layers;
                memory_plan.tensor_pc_ratio =
                    tensor_layers > 0 ? best_ratio : 0.5f;
                memory_plan.gpu_chunk_tokens = 1;
                memory_plan.cpu_chunk_tokens = 1;
                memory_plan.tensor_chunk_tokens = 1;
                memory_plan.phone_chunk_tokens = 1;
                if (!llama_hybrid_estimate_plan_memory(
                        profile, decode_constraints, memory_plan)) {
                    continue;
                }
                if (pc_budget > 0 && memory_plan.pc_memory > pc_budget) {
                    continue;
                }
                if (phone_budget > 0 && memory_plan.phone_memory > phone_budget) {
                    continue;
                }
                if (gpu_layers > 0 &&
                    (gpu_budget == 0 || memory_plan.gpu_memory > gpu_budget)) {
                    continue;
                }

                llama_hybrid_decode_plan plan;
                plan.tensor_layers = tensor_layers;
                plan.phone_layers  = phone_layers;
                plan.cpu_layers    = cpu_layers;
                plan.gpu_layers    = gpu_layers;
                plan.tensor_pc_ratio = tensor_layers > 0 ? best_ratio : 0.5f;
                // Current Qwen3-MoE measurements show LLAMA_CHUNKS=1 is the
                // fastest correct decode setting on the target platform. Keep
                // it fixed while the planner focuses on layer placement.
                plan.llama_chunks = 1;
                plan.kv_tokens = kv_tokens;

                plan.predicted_gpu_ms =
                    gpu_layers * gpu_layer_ms;
                plan.predicted_cpu_ms =
                    cpu_decode_region_ms[(size_t) cpu_layers];
                plan.predicted_tensor_ms =
                    tensor_layers * best_tensor_cost.total_ms;
                plan.predicted_phone_ms =
                    phone_layers * phone_layer_ms;

                if (gpu_layers > 0 && gpu_layers < profile.n_layer) {
                    plan.predicted_boundary_ms += gpu_to_pc_ms;
                }
                if (phone_layers > 0) {
                    // Hidden state enters the phone-only suffix once and the
                    // final result returns to the PC/output side once.
                    plan.predicted_boundary_ms +=
                        pc_to_phone_ms + phone_to_pc_ms;
                }

                plan.predicted_ms =
                    plan.predicted_gpu_ms +
                    plan.predicted_cpu_ms +
                    plan.predicted_tensor_ms +
                    plan.predicted_phone_ms +
                    plan.predicted_boundary_ms;
                plan.pc_memory    = memory_plan.pc_memory;
                plan.phone_memory = memory_plan.phone_memory;
                plan.gpu_memory   = memory_plan.gpu_memory;

                if (!std::isfinite(plan.predicted_ms)) {
                    continue;
                }

                if (!found || plan.predicted_ms < best_plan.predicted_ms) {
                    best_plan = plan;
                    found = true;
                }

                top.push_back({plan});
            }
        }
    }

    if (!found) {
        LLAMA_LOG_ERROR(
            "[DECODE_PLAN] unavailable reason=no_memory_feasible_layout\n");
        return false;
    }

    std::sort(top.begin(), top.end(), [](const auto & a, const auto & b) {
        if (a.plan.predicted_ms != b.plan.predicted_ms) {
            return a.plan.predicted_ms < b.plan.predicted_ms;
        }
        return a.plan.gpu_memory < b.plan.gpu_memory;
    });
    if (top.size() > 5) {
        top.resize(5);
    }
    for (size_t i = 0; i < top.size(); ++i) {
        const auto & p = top[i].plan;
        LLAMA_LOG_ERROR(
            "[DECODE_TOP] rank=%zu T=%d P=%d C=%d G=%d R=%.3f "
            "predicted_ms=%.3f predicted_tps=%.3f\n",
            i + 1,
            p.tensor_layers, p.phone_layers, p.cpu_layers, p.gpu_layers,
            p.tensor_pc_ratio,
            p.predicted_ms,
            p.predicted_ms > 0.0 ? 1000.0 / p.predicted_ms : 0.0);
    }

    llama_hybrid_decode_plan_print(best_plan);
    if (profile.is_moe) {
        LLAMA_LOG_ERROR(
            "[DECODE_CHUNK] selected=1 mode=FIXED "
            "reason=measured_fastest_correct_qwen3moe\n");
    } else {
        LLAMA_LOG_ERROR(
            "[DECODE_CHUNK] selected=1 mode=BASELINE "
            "reason=decode_chunk_search_not_enabled\n");
    }
    return true;
}

struct llama_hybrid_lp_model {
    int layers = 0;

    double cpu_cost    = 0.0;
    double tensor_cost = 0.0;
    double phone_cost  = 0.0;

    double pc_cpu_bytes    = 0.0;
    double pc_tensor_bytes = 0.0;
    double pc_phone_bytes  = 0.0;

    double phone_cpu_bytes    = 0.0;
    double phone_tensor_bytes = 0.0;
    double phone_phone_bytes  = 0.0;

    double pc_capacity_bytes    = std::numeric_limits<double>::infinity();
    double phone_capacity_bytes = std::numeric_limits<double>::infinity();
};

static bool llama_hybrid_coarse_lp_model(const llama_hybrid_profile &     profile,
                                         const llama_hybrid_constraints & constraints,
                                         int                              gpu_layers,
                                         int                              tensor_chunk_tokens,
                                         float                            pc_ratio,
                                         llama_hybrid_lp_model &          model) {
    if (profile.n_layer <= 0 || gpu_layers < 0 || gpu_layers > profile.n_layer ||
        tensor_chunk_tokens <= 0 || pc_ratio <= 0.0f || pc_ratio >= 1.0f) {
        return false;
    }

    model = {};
    model.layers = profile.n_layer - gpu_layers;
    if (model.layers < 0) {
        return false;
    }

    const int work_tokens = constraints.target_ubatch_tokens > 0 ?
        constraints.target_ubatch_tokens :
        (profile.reference_tokens > 0 ? profile.reference_tokens : LLAMA_HYBRID_REFERENCE_TOKENS);
    const int probe_tokens = std::max(1, profile.probe_tokens);
    if (work_tokens <= 0) {
        return false;
    }

    const double token_scale = (double) work_tokens / probe_tokens;

    // CPU-only stages are allowed to use XC up to 256. Use the measured
    // large-token layer profile here as well as in the expensive scorer;
    // otherwise the coarse pass can prune plans using a probe-token linear
    // extrapolation before the accurate scorer ever sees them.
    model.cpu_cost = 1e12;
    {
        int coarse_kv_tokens = constraints.score_kv_tokens;
        if (coarse_kv_tokens <= 0) {
            coarse_kv_tokens = constraints.target_ctx > 0 ?
                constraints.target_ctx : profile.n_ctx_train;
        }
        coarse_kv_tokens = std::max(coarse_kv_tokens, work_tokens);
        if (profile.n_ctx_train > 0) {
            coarse_kv_tokens = std::min(coarse_kv_tokens, profile.n_ctx_train);
        }

        double cpu_layer_total = 0.0;
        double cpu_attn_base  = 0.0;
        double cpu_attn_kv    = 0.0;
        const int coarse_cpu_depth = std::max(1, model.layers);
        if (llama_hybrid_layer_block_total_cost(
                profile.cpu_layer_blocks, work_tokens,
                coarse_cpu_depth, false, cpu_layer_total) &&
            llama_hybrid_attn_cost(
                profile.cpu_attn, work_tokens, work_tokens, cpu_attn_base) &&
            llama_hybrid_attn_cost(
                profile.cpu_attn, work_tokens, coarse_kv_tokens, cpu_attn_kv)) {
            model.cpu_cost = std::max(
                0.0,
                cpu_layer_total / coarse_cpu_depth +
                cpu_attn_kv - cpu_attn_base);
        } else if (profile.cpu_full_layer_ms > 0.0) {
            model.cpu_cost = profile.cpu_full_layer_ms * token_scale;
        }
    }

    const double phone_base = profile.phone_full_layer_compute_est_ms > 0.0 ?
        profile.phone_full_layer_compute_est_ms : profile.phone_full_layer_ms;
    model.phone_cost = phone_base > 0.0 ? phone_base * token_scale : 1e12;

    double pc_ffn_ms = 0.0;
    double phone_ffn_ms = 0.0;
    double return_ms = 0.0;
    const int inner_tokens = std::min(work_tokens, tensor_chunk_tokens);
    const size_t bytes = (size_t) profile.n_embd * (size_t) inner_tokens * sizeof(float);
    if (!llama_hybrid_ffn_cost(profile.cpu_ffn, inner_tokens, pc_ratio, pc_ffn_ms) ||
        !llama_hybrid_ffn_cost(profile.phone_ffn, inner_tokens, 1.0f - pc_ratio, phone_ffn_ms) ||
        !llama_hybrid_transfer_cost(profile.snapshot_phone_to_pc, bytes, return_ms)) {
        return false;
    }
    const int inner_chunks = (work_tokens + inner_tokens - 1) / inner_tokens;
    const double tensor_attn = std::max(0.0, profile.cpu_attn_ms) * token_scale;
    model.tensor_cost = tensor_attn + inner_chunks * std::max(pc_ffn_ms, phone_ffn_ms + return_ms);

    if (profile.layer_weight_bytes.size() != (size_t) profile.n_layer ||
        profile.layer_attn_forced_bytes.size() != (size_t) profile.n_layer ||
        profile.layer_ffn_split_bytes.size() != (size_t) profile.n_layer ||
        profile.layer_mirrored_bytes.size() != (size_t) profile.n_layer) {
        return false;
    }

    int ctx = constraints.target_ctx > 0 ? constraints.target_ctx : profile.n_ctx_train;
    if (ctx <= 0) {
        return false;
    }
    const long double kv_value = (long double) profile.kv_bytes_per_token_per_layer * ctx;
    if (kv_value > std::numeric_limits<size_t>::max()) {
        return false;
    }
    const double kv_per_layer = (double) (size_t) kv_value;

    long double total_sum = 0.0;
    long double attn_sum = 0.0;
    long double ffn_sum = 0.0;
    long double mirrored_sum = 0.0;
    int count = 0;
    for (int il = gpu_layers; il < profile.n_layer; ++il) {
        const size_t i = (size_t) il;
        total_sum += profile.layer_weight_bytes[i];
        attn_sum += profile.layer_attn_forced_bytes[i];
        ffn_sum += profile.layer_ffn_split_bytes[i];
        mirrored_sum += profile.layer_mirrored_bytes[i];
        ++count;
    }
    if (count <= 0) {
        return true;
    }

    const double total_avg = (double) (total_sum / count);
    const double attn_avg = (double) (attn_sum / count);
    const double ffn_avg = (double) (ffn_sum / count);
    const double mirrored_avg = (double) (mirrored_sum / count);

    model.pc_cpu_bytes    = total_avg + kv_per_layer;
    model.pc_tensor_bytes = attn_avg + ffn_avg * pc_ratio + mirrored_avg + kv_per_layer;
    model.pc_phone_bytes  = mirrored_avg;

    model.phone_cpu_bytes    = 0.0;
    model.phone_tensor_bytes = ffn_avg * (1.0 - pc_ratio) + mirrored_avg;
    model.phone_phone_bytes  = attn_avg + ffn_avg + mirrored_avg + kv_per_layer;

    const size_t pc_budget = llama_hybrid_effective_budget(
        constraints.pc_memory_budget, profile.pc_free_mem, LLAMA_HYBRID_PC_MEMORY_FRACTION);
    const size_t phone_budget = llama_hybrid_effective_budget(
        constraints.phone_memory_budget, profile.phone_free_mem, LLAMA_HYBRID_PHONE_MEMORY_FRACTION);

    if (pc_budget > 0) {
        const double base = (double) profile.non_layer_weight_bytes;
        model.pc_capacity_bytes = std::max(0.0, (double) pc_budget - base);
    }
    if (phone_budget > 0) {
        model.phone_capacity_bytes = (double) phone_budget;
    }
    return true;
}


struct llama_hybrid_coarse_candidate {
    llama_hybrid_plan plan;
    double            coarse_ms = std::numeric_limits<double>::infinity();
    int               family    = 0;
};

static int llama_hybrid_topology_family(const llama_hybrid_plan & plan) {
    const int cpu_layers = plan.pc_layers - plan.gpu_pc_layers;
    return (plan.gpu_pc_layers > 0 ? 8 : 0) |
           (cpu_layers > 0 ? 4 : 0) |
           (plan.tensor_layers > 0 ? 2 : 0) |
           (plan.phone_layers > 0 ? 1 : 0);
}

static bool llama_hybrid_coarse_memory_possible(
        const llama_hybrid_profile &     profile,
        const llama_hybrid_constraints & constraints,
        const llama_hybrid_plan &        base_plan) {
    llama_hybrid_plan probe = base_plan;

    // Use the smallest stage macros as a lower-bound activation footprint.
    // If even this version does not fit, no later XG/XC/XP expansion can make
    // the topology feasible. Fixed chunk constraints must still be honored.
    probe.gpu_chunk_tokens =
        constraints.fixed_gpu_chunk_tokens.value_or(4);
    probe.cpu_chunk_tokens =
        constraints.fixed_cpu_chunk_tokens.value_or(4);
    probe.phone_chunk_tokens =
        constraints.fixed_phone_chunk_tokens.value_or(4);

    probe.predicted_tensor_peak_bytes = 0;
    probe.predicted_phone_peak_bytes  = 0;

    if (!llama_hybrid_estimate_plan_memory(profile, constraints, probe)) {
        return false;
    }

    const size_t pc_budget = llama_hybrid_effective_budget(
        constraints.pc_memory_budget, profile.pc_free_mem,
        LLAMA_HYBRID_PC_MEMORY_FRACTION);
    const size_t phone_budget = llama_hybrid_effective_budget(
        constraints.phone_memory_budget, profile.phone_free_mem,
        LLAMA_HYBRID_PHONE_MEMORY_FRACTION);
    const size_t gpu_budget = llama_hybrid_effective_budget(
        constraints.gpu_memory_budget, profile.gpu_free_mem,
        LLAMA_HYBRID_GPU_MEMORY_FRACTION);

    if (pc_budget > 0 && probe.pc_memory > pc_budget) {
        return false;
    }
    if (phone_budget > 0 && probe.phone_memory > phone_budget) {
        return false;
    }
    if (probe.gpu_pc_layers > 0 &&
        (gpu_budget == 0 || probe.gpu_memory > gpu_budget)) {
        return false;
    }
    return true;
}

static bool llama_hybrid_coarse_plan_score(
        const llama_hybrid_lp_model & model,
        const llama_hybrid_plan &    plan,
        double                       gpu_lane_ms,
        double                       phone_boundary_ms,
        double &                     result_ms) {
    const int cpu_layers = plan.pc_layers - plan.gpu_pc_layers;
    if (cpu_layers < 0 || plan.tensor_layers < 0 || plan.phone_layers < 0) {
        return false;
    }

    double downstream_ms =
        cpu_layers * model.cpu_cost +
        plan.tensor_layers * model.tensor_cost +
        plan.phone_layers * model.phone_cost;

    if (plan.phone_layers > 0) {
        if (!std::isfinite(phone_boundary_ms)) {
            return false;
        }
        downstream_ms += phone_boundary_ms;
    }

    if (plan.gpu_pc_layers > 0 && downstream_ms > 0.0) {
        result_ms = std::max(gpu_lane_ms, downstream_ms);
    } else {
        result_ms = gpu_lane_ms + downstream_ms;
    }
    return std::isfinite(result_ms);
}

static bool llama_hybrid_coarse_candidate_less(
        const llama_hybrid_coarse_candidate & a,
        const llama_hybrid_coarse_candidate & b) {
    if (a.coarse_ms != b.coarse_ms) {
        return a.coarse_ms < b.coarse_ms;
    }
    if (a.plan.gpu_pc_layers != b.plan.gpu_pc_layers) {
        return a.plan.gpu_pc_layers > b.plan.gpu_pc_layers;
    }
    if (a.plan.tensor_layers != b.plan.tensor_layers) {
        return a.plan.tensor_layers > b.plan.tensor_layers;
    }
    return a.plan.tensor_chunk_tokens > b.plan.tensor_chunk_tokens;
}

static void llama_hybrid_coarse_insert_top(
        std::vector<llama_hybrid_coarse_candidate> & bucket,
        const llama_hybrid_coarse_candidate &        candidate,
        size_t                                       limit) {
    if (limit == 0) {
        return;
    }
    if (bucket.size() == limit &&
        !llama_hybrid_coarse_candidate_less(candidate, bucket.back())) {
        return;
    }

    const auto it = std::lower_bound(
        bucket.begin(), bucket.end(), candidate,
        [](const auto & a, const auto & b) {
            return llama_hybrid_coarse_candidate_less(a, b);
        });
    bucket.insert(it, candidate);
    if (bucket.size() > limit) {
        bucket.pop_back();
    }
}

std::vector<llama_hybrid_plan> llama_hybrid_enumerate_feasible_plans(const llama_hybrid_profile &     profile,
                                                                     const llama_hybrid_constraints & constraints) {
    std::vector<llama_hybrid_plan> result;
    if (profile.n_layer <= 0) {
        return result;
    }

    const size_t pc_budget = llama_hybrid_effective_budget(
        constraints.pc_memory_budget, profile.pc_free_mem, LLAMA_HYBRID_PC_MEMORY_FRACTION);
    const size_t phone_budget = llama_hybrid_effective_budget(
        constraints.phone_memory_budget, profile.phone_free_mem, LLAMA_HYBRID_PHONE_MEMORY_FRACTION);
    const size_t gpu_budget = llama_hybrid_effective_budget(
        constraints.gpu_memory_budget, profile.gpu_free_mem, LLAMA_HYBRID_GPU_MEMORY_FRACTION);

    std::vector<int> chunk_token_candidates;
    if (constraints.fixed_tensor_chunk_tokens) {
        chunk_token_candidates.push_back(*constraints.fixed_tensor_chunk_tokens);
    } else {
        chunk_token_candidates.assign(LLAMA_HYBRID_CHUNK_TOKEN_CANDIDATES.begin(),
                                      LLAMA_HYBRID_CHUNK_TOKEN_CANDIDATES.end());
    }

    std::vector<int> gpu_candidates;
    if (constraints.fixed_gpu_pc_layers) {
        gpu_candidates.push_back(*constraints.fixed_gpu_pc_layers);
    } else if (gpu_budget == 0) {
        gpu_candidates.push_back(0);
    } else {
        for (int g = 0; g <= profile.n_layer; ++g) {
            gpu_candidates.push_back(g);
        }
    }

    const bool topology_fixed = constraints.fixed_tensor_layers.has_value() ||
                                constraints.fixed_phone_layers.has_value() ||
                                constraints.fixed_pc_layers.has_value();

    const int score_tokens = constraints.target_ubatch_tokens > 0 ?
        constraints.target_ubatch_tokens :
        (profile.reference_tokens > 0 ? profile.reference_tokens : LLAMA_HYBRID_REFERENCE_TOKENS);

    std::array<std::vector<llama_hybrid_coarse_candidate>, 16> family_top;
    std::vector<llama_hybrid_coarse_candidate> global_top;
    std::vector<llama_hybrid_coarse_candidate> margin_pool;

    double best_coarse = std::numeric_limits<double>::infinity();
    size_t total_candidates      = 0;
    size_t coarse_scoreable       = 0;
    size_t reject_chunks          = 0;
    size_t reject_coarse_memory   = 0;
    size_t ratio_evals            = 0;

    for (const int chunk_tokens : chunk_token_candidates) {
        if (chunk_tokens <= 0 || chunk_tokens > score_tokens) {
            continue;
        }
        const std::vector<int> chunks = llama_hybrid_split_by_chunk_size(score_tokens, chunk_tokens);
        if (chunks.empty() ||
            *std::min_element(chunks.begin(), chunks.end()) < profile.probe_chunk_min_tokens) {
            continue;
        }
        const bool tensor_chunk_valid = constraints.max_tensor_chunks <= 0 ||
            chunks.size() <= (size_t) constraints.max_tensor_chunks;
        if (!tensor_chunk_valid) {
            ++reject_chunks;
        }

        std::vector<float> ratios;
        if (constraints.fixed_tensor_layers && *constraints.fixed_tensor_layers == 0) {
            ratios.push_back(0.50f);
        } else if (constraints.fixed_tensor_pc_ratio) {
            const float ratio = *constraints.fixed_tensor_pc_ratio;
            if (ratio <= 0.0f || ratio >= 1.0f) {
                continue;
            }
            ratios.push_back(llama_hybrid_align_pc_ratio(profile, ratio));
        } else {
            ratios = llama_hybrid_plan_ratio_candidates(profile, chunk_tokens);
            if (ratios.empty()) {
                continue;
            }
        }
        ratio_evals += ratios.size();

        for (const int gpu_layers : gpu_candidates) {
            if (gpu_layers < 0 || gpu_layers > profile.n_layer) {
                continue;
            }
            const int remaining = profile.n_layer - gpu_layers;

            for (size_t ir = 0; ir < ratios.size(); ++ir) {
                const float ratio = ratios[ir];

                llama_hybrid_lp_model coarse_model;
                if (!llama_hybrid_coarse_lp_model(
                        profile, constraints, gpu_layers, chunk_tokens, ratio, coarse_model)) {
                    continue;
                }

                double gpu_lane_ms = 0.0;
                if (gpu_layers > 0) {
                    if (profile.gpu_full_layer_ms <= 0.0 || profile.probe_tokens <= 0) {
                        continue;
                    }
                    const double token_scale =
                        (double) score_tokens / (double) profile.probe_tokens;
                    gpu_lane_ms =
                        gpu_layers * profile.gpu_full_layer_ms * token_scale;

                    if (remaining > 0) {
                        const size_t bytes =
                            (size_t) profile.n_embd * (size_t) score_tokens * sizeof(float);
                        double transfer_ms = 0.0;
                        if (!llama_hybrid_transfer_cost(
                                profile.gpu_to_pc, bytes, transfer_ms)) {
                            continue;
                        }
                        gpu_lane_ms += transfer_ms;
                    }
                }

                double phone_boundary_ms = 0.0;
                {
                    const size_t bytes =
                        (size_t) profile.n_embd * (size_t) score_tokens * sizeof(float);
                    double enter_ms = 0.0;
                    double exit_ms  = 0.0;
                    if (!llama_hybrid_transfer_cost(
                            profile.pc_to_phone, bytes, enter_ms) ||
                        !llama_hybrid_transfer_cost(
                            profile.phone_to_pc, bytes, exit_ms)) {
                        phone_boundary_ms =
                            std::numeric_limits<double>::infinity();
                    } else {
                        phone_boundary_ms = enter_ms + exit_ms;
                    }
                }

                for (int cpu_layers = 0; cpu_layers <= remaining; ++cpu_layers) {
                    for (int tensor_layers = 0;
                         tensor_layers <= remaining - cpu_layers;
                         ++tensor_layers) {
                        const int phone_layers =
                            remaining - cpu_layers - tensor_layers;
                        const int pc_layers = gpu_layers + cpu_layers;

                        if (topology_fixed) {
                            if (constraints.fixed_tensor_layers &&
                                tensor_layers != *constraints.fixed_tensor_layers) {
                                continue;
                            }
                            if (constraints.fixed_phone_layers &&
                                phone_layers != *constraints.fixed_phone_layers) {
                                continue;
                            }
                            if (constraints.fixed_pc_layers &&
                                pc_layers != *constraints.fixed_pc_layers) {
                                continue;
                            }
                        }

                        if (tensor_layers > 0 && !tensor_chunk_valid) {
                            continue;
                        }

                        // Tensor-free layouts are independent of XT/R.  Keep one
                        // representative unless the user explicitly fixed them.
                        if (tensor_layers == 0) {
                            if (!constraints.fixed_tensor_chunk_tokens &&
                                chunk_tokens != chunk_token_candidates.front()) {
                                continue;
                            }
                            if (!constraints.fixed_tensor_pc_ratio && ir != 0) {
                                continue;
                            }
                        }

                        ++total_candidates;
                        llama_hybrid_plan plan;
                        plan.tensor_layers       = tensor_layers;
                        plan.phone_layers        = phone_layers;
                        plan.pc_layers           = pc_layers;
                        plan.tensor_pc_ratio     = tensor_layers > 0 ? ratio : 0.50f;
                        plan.gpu_pc_layers       = gpu_layers;
                        plan.tensor_chunk_tokens = chunk_tokens;

                        if (!llama_hybrid_coarse_memory_possible(
                                profile, constraints, plan)) {
                            ++reject_coarse_memory;
                            continue;
                        }

                        double coarse_ms = 0.0;
                        if (!llama_hybrid_coarse_plan_score(
                                coarse_model, plan, gpu_lane_ms,
                                phone_boundary_ms, coarse_ms)) {
                            continue;
                        }
                        ++coarse_scoreable;
                        best_coarse = std::min(best_coarse, coarse_ms);

                        llama_hybrid_coarse_candidate candidate;
                        candidate.plan      = plan;
                        candidate.coarse_ms = coarse_ms;
                        candidate.family    = llama_hybrid_topology_family(plan);

                        llama_hybrid_coarse_insert_top(
                            family_top[(size_t) candidate.family], candidate,
                            LLAMA_HYBRID_COARSE_FAMILY_TOP_K);
                        llama_hybrid_coarse_insert_top(
                            global_top, candidate,
                            LLAMA_HYBRID_COARSE_GLOBAL_TOP_K);
                        llama_hybrid_coarse_insert_top(
                            margin_pool, candidate,
                            LLAMA_HYBRID_COARSE_MARGIN_POOL);
                    }
                }
            }
        }
    }

    const auto same_base_candidate = [](const llama_hybrid_plan & a, const llama_hybrid_plan & b) {
        return a.tensor_layers == b.tensor_layers &&
               a.phone_layers == b.phone_layers &&
               a.pc_layers == b.pc_layers &&
               a.gpu_pc_layers == b.gpu_pc_layers &&
               a.tensor_chunk_tokens == b.tensor_chunk_tokens &&
               std::fabs(a.tensor_pc_ratio - b.tensor_pc_ratio) < 1e-6f;
    };

    const auto keep_unique = [&](const llama_hybrid_coarse_candidate & candidate) {
        const bool duplicate = std::any_of(result.begin(), result.end(), [&](const llama_hybrid_plan & plan) {
            return same_base_candidate(plan, candidate.plan);
        });
        if (!duplicate) {
            result.push_back(candidate.plan);
        }
    };

    for (const auto & bucket : family_top) {
        for (const auto & candidate : bucket) {
            keep_unique(candidate);
        }
    }
    for (const auto & candidate : global_top) {
        keep_unique(candidate);
    }
    if (std::isfinite(best_coarse)) {
        const double margin_limit = best_coarse * LLAMA_HYBRID_COARSE_MARGIN;
        for (const auto & candidate : margin_pool) {
            if (candidate.coarse_ms <= margin_limit) {
                keep_unique(candidate);
            }
        }
    }

    LLAMA_LOG_ERROR(
        "[HYBRID_PLAN_ENUM] ctx=%d ubatch=%d reference_tokens=%d max_tensor_chunks=%d pc_budget=%zu "
        "phone_budget=%zu gpu_budget=%zu gpu_runtime_reserve=%zu total=%zu coarse_scoreable=%zu kept=%zu "
        "reject_chunks=%zu reject_coarse_memory=%zu ratio_evals=%zu best_coarse_ms=%.3f "
        "family_top_k=%zu global_top_k=%zu margin_pool=%zu margin=%.2f\n",
        constraints.target_ctx > 0 ? constraints.target_ctx : profile.n_ctx_train,
        constraints.target_ubatch_tokens > 0 ? constraints.target_ubatch_tokens : profile.probe_tokens,
        profile.reference_tokens > 0 ? profile.reference_tokens : LLAMA_HYBRID_REFERENCE_TOKENS,
        constraints.max_tensor_chunks, pc_budget, phone_budget, gpu_budget, constraints.gpu_runtime_reserve_bytes,
        total_candidates, coarse_scoreable, result.size(), reject_chunks,
        reject_coarse_memory, ratio_evals,
        std::isfinite(best_coarse) ? best_coarse : 0.0,
        LLAMA_HYBRID_COARSE_FAMILY_TOP_K, LLAMA_HYBRID_COARSE_GLOBAL_TOP_K,
        LLAMA_HYBRID_COARSE_MARGIN_POOL, LLAMA_HYBRID_COARSE_MARGIN);

    return result;
}

static std::vector<int> llama_hybrid_plan_chunk_candidates(
        const llama_hybrid_profile & profile,
        int                          total_tokens,
        const std::optional<int> &   fixed) {
    std::vector<int> candidates;
    if (fixed) {
        candidates.push_back(*fixed);
    } else {
        candidates.assign(LLAMA_HYBRID_REGION_CHUNK_CANDIDATES.begin(),
                          LLAMA_HYBRID_REGION_CHUNK_CANDIDATES.end());
    }

    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](int chunk_tokens) {
        if (chunk_tokens <= 0 || chunk_tokens > total_tokens) {
            return true;
        }
        const std::vector<int> chunks = llama_hybrid_split_by_chunk_size(total_tokens, chunk_tokens);
        return chunks.empty() ||
               *std::min_element(chunks.begin(), chunks.end()) < profile.probe_chunk_min_tokens;
    }), candidates.end());
    return candidates;
}

static bool llama_hybrid_layer_region_cost(
        const llama_hybrid_profile &                            profile,
        const std::vector<llama_hybrid_layer_compute_point> & layer_points,
        const std::vector<llama_hybrid_attn_compute_point> &  attn_points,
        int                                                    layers,
        int                                                    total_tokens,
        int                                                    chunk_tokens,
        int                                                    kv_tokens,
        bool                                                   use_compute_est,
        double &                                               result_ms) {
    result_ms = 0.0;
    if (layers == 0) {
        return true;
    }

    const std::vector<int> chunks = llama_hybrid_split_by_chunk_size(total_tokens, chunk_tokens);
    if (chunks.empty()) {
        return false;
    }

    for (const int tokens : chunks) {
        double layer_block_total = 0.0;
        double attn_base         = 0.0;
        double attn_kv           = 0.0;
        if (!llama_hybrid_layer_block_total_cost(
                layer_points, tokens, layers, use_compute_est,
                layer_block_total) ||
            !llama_hybrid_attn_cost(
                attn_points, tokens, tokens, attn_base) ||
            !llama_hybrid_attn_cost(
                attn_points, tokens, kv_tokens, attn_kv)) {
            return false;
        }
        result_ms += std::max(
            0.0,
            layer_block_total +
            layers * (attn_kv - attn_base));
    }
    return std::isfinite(result_ms);
}

bool llama_hybrid_runtime_predict_cpu_compute(
        int tokens, llama_hybrid_cpu_compute_prediction & prediction) {
    prediction = {};
    if (tokens <= 0) {
        return false;
    }

    llama_hybrid_plan plan;
    llama_hybrid_profile profile;
    llama_hybrid_constraints constraints;
    {
        std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
        if (!g_llama_hybrid_runtime_plan.has_value() ||
            !g_llama_hybrid_runtime_profile.has_value() ||
            !g_llama_hybrid_runtime_constraints.has_value()) {
            return false;
        }
        plan        = *g_llama_hybrid_runtime_plan;
        profile     = *g_llama_hybrid_runtime_profile;
        constraints = *g_llama_hybrid_runtime_constraints;
    }

    const int cpu_layers = plan.pc_layers - plan.gpu_pc_layers;
    if (cpu_layers <= 0 || plan.cpu_chunk_tokens <= 0 ||
        profile.cpu_layer_blocks.empty()) {
        return false;
    }

    int kv_tokens = constraints.score_kv_tokens;
    if (kv_tokens <= 0) {
        kv_tokens = constraints.target_ctx > 0 ?
            constraints.target_ctx : profile.n_ctx_train;
    }
    kv_tokens = std::max(kv_tokens, tokens);
    if (profile.n_ctx_train > 0) {
        kv_tokens = std::min(kv_tokens, profile.n_ctx_train);
    }

    double total_ms = 0.0;
    if (!llama_hybrid_layer_region_cost(
            profile, profile.cpu_layer_blocks, profile.cpu_attn,
            cpu_layers, tokens, plan.cpu_chunk_tokens, kv_tokens,
            false, total_ms)) {
        return false;
    }

    int profile_min_tokens = std::numeric_limits<int>::max();
    int profile_max_tokens = 0;
    int profile_min_layers = std::numeric_limits<int>::max();
    int profile_max_layers = 0;
    std::set<int> profile_tokens;
    std::set<int> profile_layers;
    for (const auto & point : profile.cpu_layer_blocks) {
        if (point.layers <= 0 || point.tokens <= 0) {
            continue;
        }
        profile_tokens.insert(point.tokens);
        profile_layers.insert(point.layers);
        profile_min_tokens = std::min(profile_min_tokens, point.tokens);
        profile_max_tokens = std::max(profile_max_tokens, point.tokens);
        profile_min_layers = std::min(profile_min_layers, point.layers);
        profile_max_layers = std::max(profile_max_layers, point.layers);
    }
    if (profile_tokens.empty() || profile_layers.empty()) {
        return false;
    }

    const auto chunks =
        llama_hybrid_split_by_chunk_size(tokens, plan.cpu_chunk_tokens);
    if (chunks.empty()) {
        return false;
    }

    int exact_chunks = 0;
    int interpolated_chunks = 0;
    int extrapolated_chunks = 0;
    for (const int chunk_tokens : chunks) {
        if (profile_tokens.count(chunk_tokens) != 0) {
            ++exact_chunks;
        } else if (chunk_tokens < profile_min_tokens ||
                   chunk_tokens > profile_max_tokens) {
            ++extrapolated_chunks;
        } else {
            ++interpolated_chunks;
        }
    }

    prediction.tokens                      = tokens;
    prediction.kv_tokens                   = kv_tokens;
    prediction.cpu_layers                  = cpu_layers;
    prediction.cpu_chunk_tokens            = plan.cpu_chunk_tokens;
    prediction.profile_min_tokens           = profile_min_tokens;
    prediction.profile_max_tokens           = profile_max_tokens;
    prediction.profile_exact_chunks         = exact_chunks;
    prediction.profile_interpolated_chunks  = interpolated_chunks;
    prediction.profile_extrapolated_chunks  = extrapolated_chunks;
    prediction.profile_min_layers           = profile_min_layers;
    prediction.profile_max_layers           = profile_max_layers;
    prediction.profile_layer_exact          =
        profile_layers.count(cpu_layers) != 0;
    prediction.profile_layer_saturated      =
        cpu_layers > profile_max_layers;
    prediction.total_ms                     = total_ms;
    prediction.per_layer_ms                 =
        cpu_layers > 0 ? total_ms / cpu_layers : 0.0;

    return std::isfinite(prediction.total_ms) &&
           std::isfinite(prediction.per_layer_ms);
}

enum class llama_hybrid_sim_stage_kind {
    GPU,
    CPU,
    TENSOR,
    PHONE,
};

struct llama_hybrid_sim_stage {
    llama_hybrid_sim_stage_kind kind;
    int                         layers;
    int                         macro_tokens;
};

struct llama_hybrid_sim_job {
    int    id          = -1;
    int    tokens      = 0;
    size_t stage_index = 1;
    size_t bytes       = 0;
};

struct llama_hybrid_sim_result {
    double makespan_ms   = 0.0;
    double gpu_busy_ms   = 0.0;
    double downstream_ms = 0.0;
    double gpu_wait_ms   = 0.0;

    size_t tensor_peak_bytes = 0;
    size_t phone_peak_bytes  = 0;

    std::string schedule;
};

static std::vector<llama_hybrid_sim_stage> llama_hybrid_sim_stages(const llama_hybrid_plan & plan) {
    std::vector<llama_hybrid_sim_stage> stages;
    const int cpu_layers = plan.pc_layers - plan.gpu_pc_layers;

    if (plan.gpu_pc_layers <= 0 || plan.tensor_layers <= 0) {
        return stages;
    }

    int upstream_macro = plan.gpu_chunk_tokens;
    stages.push_back({ llama_hybrid_sim_stage_kind::GPU, plan.gpu_pc_layers, upstream_macro });
    if (cpu_layers > 0) {
        upstream_macro = plan.cpu_chunk_tokens;
        stages.push_back({ llama_hybrid_sim_stage_kind::CPU, cpu_layers, upstream_macro });
    }
    stages.push_back({ llama_hybrid_sim_stage_kind::TENSOR, plan.tensor_layers, upstream_macro });
    if (plan.phone_layers > 0) {
        stages.push_back({ llama_hybrid_sim_stage_kind::PHONE, plan.phone_layers, plan.phone_chunk_tokens });
    }
    return stages;
}

static bool llama_hybrid_sim_stage_cost(
        const llama_hybrid_profile &                profile,
        const llama_hybrid_plan &                   plan,
        const std::vector<llama_hybrid_sim_stage> & stages,
        size_t                                      stage_index,
        int                                         job_tokens,
        int                                         kv_tokens,
        double &                                    result_ms) {
    if (stage_index >= stages.size() || job_tokens <= 0) {
        return false;
    }

    const auto & stage = stages[stage_index];
    std::vector<llama_hybrid_boundary_block> blocks;
    if (stage_index == 0) {
        llama_hybrid_boundary_block block;
        block.output = { 0, job_tokens };
        blocks.push_back(std::move(block));
    } else {
        blocks = llama_hybrid_plan_boundary(
            job_tokens, stages[stage_index - 1].macro_tokens, stage.macro_tokens);
        if (blocks.empty() || std::any_of(
                blocks.begin(), blocks.end(), [](const llama_hybrid_boundary_block & block) {
                    return block.action == llama_hybrid_boundary_action::ACCUMULATE;
                })) {
            return false;
        }
    }

    result_ms = 0.0;
    for (const auto & block : blocks) {
        const int tokens = block.output.n_tokens;
        double block_ms = 0.0;

        switch (stage.kind) {
            case llama_hybrid_sim_stage_kind::GPU:
                {
                    if (!llama_hybrid_layer_region_cost(
                            profile, profile.gpu_layer_blocks, profile.gpu_attn, stage.layers,
                            tokens, tokens, kv_tokens, false, block_ms)) {
                        return false;
                    }
                    const size_t bytes = (size_t) profile.n_embd * tokens * sizeof(float);
                    double transfer_ms = 0.0;
                    if (!llama_hybrid_transfer_cost(profile.gpu_to_pc, bytes, transfer_ms)) {
                        return false;
                    }
                    block_ms += transfer_ms;
                }
                break;
            case llama_hybrid_sim_stage_kind::CPU:
                if (!llama_hybrid_layer_region_cost(
                        profile, profile.cpu_layer_blocks, profile.cpu_attn, stage.layers,
                        tokens, tokens, kv_tokens, false, block_ms)) {
                    return false;
                }
                break;
            case llama_hybrid_sim_stage_kind::TENSOR:
                {
                    double tensor_ffn_ms = 0.0;
                    double cpu_layer_base = 0.0;
                    double cpu_attn_base = 0.0;
                    double cpu_attn_kv = 0.0;
                    if (!llama_hybrid_tensor_ffn_cost(
                            profile, plan.tensor_pc_ratio, tokens, plan.tensor_chunk_tokens, tensor_ffn_ms) ||
                        !llama_hybrid_layer_block_cost(
                            profile.cpu_layer_blocks, tokens, false, cpu_layer_base) ||
                        !llama_hybrid_attn_cost(profile.cpu_attn, tokens, tokens, cpu_attn_base) ||
                        !llama_hybrid_attn_cost(profile.cpu_attn, tokens, kv_tokens, cpu_attn_kv)) {
                        return false;
                    }
                    block_ms = stage.layers * (
                        cpu_attn_kv + llama_hybrid_tensor_misc_cost(
                            profile, tokens, cpu_layer_base, cpu_attn_base) + tensor_ffn_ms);
                }
                break;
            case llama_hybrid_sim_stage_kind::PHONE:
                {
                    if (!llama_hybrid_layer_region_cost(
                            profile, profile.phone_layer_blocks, profile.phone_attn, stage.layers,
                            tokens, tokens, kv_tokens, true, block_ms)) {
                        return false;
                    }
                    const size_t bytes = (size_t) profile.n_embd * tokens * sizeof(float);
                    double enter_ms = 0.0;
                    double exit_ms = 0.0;
                    if (!llama_hybrid_transfer_cost(profile.pc_to_phone, bytes, enter_ms) ||
                        !llama_hybrid_transfer_cost(profile.phone_to_pc, bytes, exit_ms)) {
                        return false;
                    }
                    block_ms += enter_ms + exit_ms;
                }
                break;
        }

        result_ms += block_ms;
    }
    return std::isfinite(result_ms);
}

static size_t llama_hybrid_sim_queue_limit(const char * name) {
    size_t result = 16ull * 1024 * 1024;
    if (const char * env = std::getenv(name)) {
        const long long mb = std::atoll(env);
        if (mb > 0) {
            result = (size_t) mb * 1024ull * 1024ull;
        }
    }
    return result;
}

static bool llama_hybrid_simulate_prefill(
        const llama_hybrid_profile &     profile,
        const llama_hybrid_constraints & constraints,
        const llama_hybrid_plan &        plan,
        int                              tokens,
        bool                             capture_schedule,
        llama_hybrid_sim_result &        result) {
    const auto stages = llama_hybrid_sim_stages(plan);
    if (stages.size() < 2 || stages.front().kind != llama_hybrid_sim_stage_kind::GPU ||
        (stages.back().kind != llama_hybrid_sim_stage_kind::TENSOR &&
         stages.back().kind != llama_hybrid_sim_stage_kind::PHONE) ||
        tokens <= 0 || profile.n_embd <= 0) {
        return false;
    }

    int kv_tokens = constraints.score_kv_tokens;
    if (kv_tokens <= 0) {
        kv_tokens = constraints.target_ctx > 0 ? constraints.target_ctx : profile.n_ctx_train;
    }
    kv_tokens = std::max(kv_tokens, tokens);
    if (profile.n_ctx_train > 0) {
        kv_tokens = std::min(kv_tokens, profile.n_ctx_train);
    }

    const auto macro_jobs = llama_hybrid_split_by_chunk_size(tokens, plan.gpu_chunk_tokens);
    if (macro_jobs.empty()) {
        return false;
    }

    const size_t tensor_limit = llama_hybrid_sim_queue_limit("LLAMA_HYBRID_TENSOR_QUEUE_MB");
    const size_t phone_limit = llama_hybrid_sim_queue_limit("LLAMA_HYBRID_PHONE_QUEUE_MB");
    size_t cpu_ready_max = 2;
    if (const char * env = std::getenv("LLAMA_HYBRID_CPU_READY_MAX")) {
        const long long value = std::atoll(env);
        if (value > 0) {
            cpu_ready_max = (size_t) value;
        }
    }

    double host_ms = 0.0;
    double gpu_finish_ms = 0.0;
    bool gpu_pending = false;
    llama_hybrid_sim_job gpu_job;
    std::deque<llama_hybrid_sim_job> cpu_q;
    std::deque<llama_hybrid_sim_job> tensor_q;
    std::deque<llama_hybrid_sim_job> phone_q;
    size_t tensor_bytes = 0;
    size_t phone_bytes = 0;
    result = {};

    const auto append_event = [&](char kind, int id) {
        if (!capture_schedule) {
            return;
        }
        if (!result.schedule.empty()) {
            result.schedule += ',';
        }
        result.schedule += kind;
        result.schedule += std::to_string(id);
    };

    const auto enqueue_ready = [&](llama_hybrid_sim_job && job) {
        if (job.stage_index >= stages.size()) {
            return false;
        }
        switch (stages[job.stage_index].kind) {
            case llama_hybrid_sim_stage_kind::CPU:
                cpu_q.push_back(std::move(job));
                break;
            case llama_hybrid_sim_stage_kind::TENSOR:
                tensor_bytes += job.bytes;
                result.tensor_peak_bytes = std::max(result.tensor_peak_bytes, tensor_bytes);
                tensor_q.push_back(std::move(job));
                break;
            case llama_hybrid_sim_stage_kind::PHONE:
                phone_bytes += job.bytes;
                result.phone_peak_bytes = std::max(result.phone_peak_bytes, phone_bytes);
                phone_q.push_back(std::move(job));
                break;
            case llama_hybrid_sim_stage_kind::GPU:
                return false;
        }
        return true;
    };

    const auto has_ready = [&]() {
        return !cpu_q.empty() || !tensor_q.empty() || !phone_q.empty();
    };

    const auto run_stage = [&](llama_hybrid_sim_job job, llama_hybrid_sim_stage_kind kind) {
        double stage_ms = 0.0;
        if (!llama_hybrid_sim_stage_cost(profile, plan, stages, job.stage_index, job.tokens, kv_tokens, stage_ms)) {
            return false;
        }
        host_ms += stage_ms;
        result.downstream_ms += stage_ms;
        append_event(kind == llama_hybrid_sim_stage_kind::CPU ? 'C' :
                     kind == llama_hybrid_sim_stage_kind::TENSOR ? 'T' : 'P', job.id);
        job.stage_index++;
        return job.stage_index == stages.size() || enqueue_ready(std::move(job));
    };

    const auto run_ready_one = [&]() {
        if (!has_ready()) {
            return true;
        }

        if (phone_bytes >= phone_limit && !phone_q.empty()) {
            auto job = std::move(phone_q.front());
            phone_q.pop_front();
            phone_bytes -= job.bytes;
            return run_stage(std::move(job), llama_hybrid_sim_stage_kind::PHONE);
        }
        if (tensor_bytes >= tensor_limit && !tensor_q.empty()) {
            auto job = std::move(tensor_q.front());
            tensor_q.pop_front();
            tensor_bytes -= job.bytes;
            return run_stage(std::move(job), llama_hybrid_sim_stage_kind::TENSOR);
        }
        if (!cpu_q.empty()) {
            auto job = std::move(cpu_q.front());
            cpu_q.pop_front();
            return run_stage(std::move(job), llama_hybrid_sim_stage_kind::CPU);
        }
        if (!tensor_q.empty()) {
            auto job = std::move(tensor_q.front());
            tensor_q.pop_front();
            tensor_bytes -= job.bytes;
            return run_stage(std::move(job), llama_hybrid_sim_stage_kind::TENSOR);
        }

        auto job = std::move(phone_q.front());
        phone_q.pop_front();
        phone_bytes -= job.bytes;
        return run_stage(std::move(job), llama_hybrid_sim_stage_kind::PHONE);
    };

    const auto harvest_gpu = [&]() {
        if (!gpu_pending) {
            return false;
        }
        result.gpu_wait_ms += std::max(0.0, gpu_finish_ms - host_ms);
        host_ms = std::max(host_ms, gpu_finish_ms);
        gpu_pending = false;
        return enqueue_ready(std::move(gpu_job));
    };

    for (size_t i = 0; i < macro_jobs.size(); ++i) {
        if (gpu_pending) {
            while (cpu_q.size() >= cpu_ready_max) {
                if (!run_ready_one()) {
                    return false;
                }
            }
            if (!harvest_gpu()) {
                return false;
            }
        }

        const int job_tokens = macro_jobs[i];
        double gpu_ms = 0.0;
        if (!llama_hybrid_sim_stage_cost(profile, plan, stages, 0, job_tokens, kv_tokens, gpu_ms)) {
            return false;
        }
        gpu_job = {
            (int) i,
            job_tokens,
            1,
            (size_t) profile.n_embd * (size_t) job_tokens * sizeof(float),
        };
        gpu_finish_ms = std::max(host_ms, gpu_finish_ms) + gpu_ms;
        result.gpu_busy_ms += gpu_ms;
        gpu_pending = true;
        append_event('G', (int) i);

        if (has_ready() && !run_ready_one()) {
            return false;
        }

        if (i + 1 == macro_jobs.size()) {
            if (!harvest_gpu()) {
                return false;
            }
            while (has_ready()) {
                if (!run_ready_one()) {
                    return false;
                }
            }
        }
    }

    result.makespan_ms = host_ms;
    return std::isfinite(result.makespan_ms);
}

bool llama_hybrid_score_plan(const llama_hybrid_profile &     profile,
                             const llama_hybrid_constraints & constraints,
                             llama_hybrid_plan &              plan) {
    if (plan.tensor_layers < 0 || plan.phone_layers < 0 || plan.pc_layers < 0 ||
        plan.tensor_layers + plan.phone_layers + plan.pc_layers != profile.n_layer || plan.gpu_pc_layers < 0 ||
        plan.gpu_pc_layers > plan.pc_layers) {
        return false;
    }

    const int reference_tokens =
        profile.reference_tokens > 0 ? profile.reference_tokens : LLAMA_HYBRID_REFERENCE_TOKENS;
    const int work_tokens =
        constraints.target_ubatch_tokens > 0 ? constraints.target_ubatch_tokens : reference_tokens;
    const bool debug_search = std::getenv("LLAMA_HYBRID_PLAN_DEBUG") != nullptr;
    if (work_tokens <= 0) {
        return false;
    }

    int kv_tokens = constraints.score_kv_tokens;
    if (kv_tokens <= 0) {
        kv_tokens = constraints.target_ctx > 0 ? constraints.target_ctx : profile.n_ctx_train;
    }
    kv_tokens = std::max(kv_tokens, work_tokens);
    if (profile.n_ctx_train > 0) {
        kv_tokens = std::min(kv_tokens, profile.n_ctx_train);
    }

    const int cpu_layers = plan.pc_layers - plan.gpu_pc_layers;

    struct llama_hybrid_stage_choice {
        int    chunk_tokens = 4;
        double compute_ms   = 0.0;
        double handoff_ms   = 0.0;
    };

    std::vector<llama_hybrid_stage_choice> cpu_choices;
    if (cpu_layers > 0) {
        const auto chunks = llama_hybrid_plan_chunk_candidates(
            profile, work_tokens, constraints.fixed_cpu_chunk_tokens);
        for (const int chunk_tokens : chunks) {
            double cost = 0.0;
            if (!llama_hybrid_layer_region_cost(
                    profile, profile.cpu_layer_blocks, profile.cpu_attn, cpu_layers,
                    work_tokens, chunk_tokens, kv_tokens, false, cost)) {
                continue;
            }
            cpu_choices.push_back({ chunk_tokens, cost, 0.0 });
        }
        if (cpu_choices.empty()) {
            return false;
        }
    } else {
        cpu_choices.push_back({ plan.cpu_chunk_tokens, 0.0, 0.0 });
    }

    std::vector<llama_hybrid_stage_choice> phone_choices;
    if (plan.phone_layers > 0) {
        const auto chunks = llama_hybrid_plan_chunk_candidates(
            profile, work_tokens, constraints.fixed_phone_chunk_tokens);
        for (const int chunk_tokens : chunks) {
            double compute_ms = 0.0;
            if (!llama_hybrid_layer_region_cost(
                    profile, profile.phone_layer_blocks, profile.phone_attn,
                    plan.phone_layers, work_tokens, chunk_tokens, kv_tokens, true, compute_ms)) {
                continue;
            }

            double handoff_ms = 0.0;
            const auto parts = llama_hybrid_split_by_chunk_size(work_tokens, chunk_tokens);
            bool ok = true;
            for (const int tokens : parts) {
                const size_t bytes = (size_t) profile.n_embd * (size_t) tokens * sizeof(float);
                double enter_ms = 0.0;
                double exit_ms  = 0.0;
                if (!llama_hybrid_transfer_cost(profile.pc_to_phone, bytes, enter_ms) ||
                    !llama_hybrid_transfer_cost(profile.phone_to_pc, bytes, exit_ms)) {
                    ok = false;
                    break;
                }
                handoff_ms += enter_ms + exit_ms;
            }
            if (ok) {
                phone_choices.push_back({ chunk_tokens, compute_ms, handoff_ms });
            }
        }
        if (phone_choices.empty()) {
            return false;
        }
    } else {
        phone_choices.push_back({ plan.phone_chunk_tokens, 0.0, 0.0 });
    }

    std::vector<llama_hybrid_stage_choice> gpu_choices;
    if (plan.gpu_pc_layers > 0) {
        const auto chunks = llama_hybrid_plan_chunk_candidates(
            profile, work_tokens, constraints.fixed_gpu_chunk_tokens);
        for (const int chunk_tokens : chunks) {
            double compute_ms = 0.0;
            if (!llama_hybrid_layer_region_cost(
                    profile, profile.gpu_layer_blocks, profile.gpu_attn,
                    plan.gpu_pc_layers, work_tokens, chunk_tokens, kv_tokens, false, compute_ms)) {
                continue;
            }

            double transfer_ms = 0.0;
            const auto parts = llama_hybrid_split_by_chunk_size(work_tokens, chunk_tokens);
            bool ok = true;
            for (const int tokens : parts) {
                const size_t bytes = (size_t) profile.n_embd * (size_t) tokens * sizeof(float);
                double chunk_transfer_ms = 0.0;
                if (!llama_hybrid_transfer_cost(profile.gpu_to_pc, bytes, chunk_transfer_ms)) {
                    ok = false;
                    break;
                }
                transfer_ms += chunk_transfer_ms;
            }
            if (ok) {
                gpu_choices.push_back({ chunk_tokens, compute_ms + transfer_ms, 0.0 });
            }
        }
        if (gpu_choices.empty()) {
            return false;
        }
    } else {
        gpu_choices.push_back({ plan.gpu_chunk_tokens, 0.0, 0.0 });
    }

    const auto score_tensor = [&](llama_hybrid_plan & candidate) {
        candidate.predicted_tensor_ms = 0.0;
        if (candidate.tensor_layers <= 0) {
            return true;
        }

        int macro_chunk_tokens = work_tokens;
        if (cpu_layers > 0) {
            macro_chunk_tokens = candidate.cpu_chunk_tokens;
        } else if (candidate.gpu_pc_layers > 0) {
            macro_chunk_tokens = candidate.gpu_chunk_tokens;
        }

        const auto macro_chunks =
            llama_hybrid_split_by_chunk_size(work_tokens, macro_chunk_tokens);
        if (macro_chunks.empty()) {
            return false;
        }

        double tensor_layer_ms = 0.0;
        for (const int macro_tokens : macro_chunks) {
            double tensor_ffn_ms = 0.0;
            if (!llama_hybrid_tensor_ffn_cost(
                    profile, candidate.tensor_pc_ratio, macro_tokens,
                    candidate.tensor_chunk_tokens, tensor_ffn_ms)) {
                return false;
            }

            double cpu_layer_base = 0.0;
            double cpu_attn_base  = 0.0;
            double cpu_attn_kv    = 0.0;
            if (!llama_hybrid_layer_block_cost(
                    profile.cpu_layer_blocks, macro_tokens, false, cpu_layer_base) ||
                !llama_hybrid_attn_cost(
                    profile.cpu_attn, macro_tokens, macro_tokens, cpu_attn_base) ||
                !llama_hybrid_attn_cost(
                    profile.cpu_attn, macro_tokens, kv_tokens, cpu_attn_kv)) {
                return false;
            }

            tensor_layer_ms += cpu_attn_kv +
                llama_hybrid_tensor_misc_cost(
                    profile, macro_tokens, cpu_layer_base, cpu_attn_base) +
                tensor_ffn_ms;
        }

        candidate.predicted_tensor_ms = candidate.tensor_layers * tensor_layer_ms;
        return true;
    };

    const auto finalize_score = [&](llama_hybrid_plan & candidate) {
        candidate.predicted_ms =
            candidate.predicted_tensor_ms +
            candidate.predicted_phone_ms +
            candidate.predicted_pc_cpu_ms +
            candidate.predicted_pc_gpu_ms +
            candidate.predicted_handoff_ms;

        candidate.predicted_gpu_busy_ms = 0.0;
        candidate.predicted_downstream_ms = 0.0;
        candidate.predicted_gpu_wait_ms = 0.0;
        candidate.predicted_tensor_peak_bytes = 0;
        candidate.predicted_phone_peak_bytes = 0;

        if (!llama_hybrid_sim_stages(candidate).empty()) {
            llama_hybrid_sim_result sim;
            if (!llama_hybrid_simulate_prefill(
                    profile, constraints, candidate, work_tokens, false, sim)) {
                return false;
            }
            candidate.predicted_ms = sim.makespan_ms;
            candidate.predicted_gpu_busy_ms = sim.gpu_busy_ms;
            candidate.predicted_downstream_ms = sim.downstream_ms;
            candidate.predicted_gpu_wait_ms = sim.gpu_wait_ms;
            candidate.predicted_tensor_peak_bytes = sim.tensor_peak_bytes;
            candidate.predicted_phone_peak_bytes = sim.phone_peak_bytes;
        }
        return std::isfinite(candidate.predicted_ms);
    };

    const size_t pc_budget = llama_hybrid_effective_budget(
        constraints.pc_memory_budget, profile.pc_free_mem, LLAMA_HYBRID_PC_MEMORY_FRACTION);
    const size_t phone_budget = llama_hybrid_effective_budget(
        constraints.phone_memory_budget, profile.phone_free_mem, LLAMA_HYBRID_PHONE_MEMORY_FRACTION);
    const size_t gpu_budget = llama_hybrid_effective_budget(
        constraints.gpu_memory_budget, profile.gpu_free_mem, LLAMA_HYBRID_GPU_MEMORY_FRACTION);

    const auto memory_feasible = [&](llama_hybrid_plan & candidate) {
        if (!llama_hybrid_estimate_plan_memory(profile, constraints, candidate)) {
            return false;
        }
        if (pc_budget > 0 && candidate.pc_memory > pc_budget) {
            return false;
        }
        if (phone_budget > 0 && candidate.phone_memory > phone_budget) {
            return false;
        }
        if (candidate.gpu_pc_layers > 0 &&
            (gpu_budget == 0 || candidate.gpu_memory > gpu_budget)) {
            return false;
        }
        return true;
    };

    bool found = false;
    llama_hybrid_plan best_plan;

    for (const auto & gpu : gpu_choices) {
        for (const auto & cpu : cpu_choices) {
            for (const auto & phone : phone_choices) {
                llama_hybrid_plan candidate = plan;
                candidate.gpu_chunk_tokens   = gpu.chunk_tokens;
                candidate.cpu_chunk_tokens   = cpu.chunk_tokens;
                candidate.phone_chunk_tokens = phone.chunk_tokens;

                candidate.predicted_pc_gpu_ms  = gpu.compute_ms;
                candidate.predicted_pc_cpu_ms  = cpu.compute_ms;
                candidate.predicted_phone_ms   = phone.compute_ms;
                candidate.predicted_handoff_ms = phone.handoff_ms;

                if (!score_tensor(candidate) ||
                    !finalize_score(candidate) ||
                    !memory_feasible(candidate)) {
                    continue;
                }

                if (debug_search) {
                    LLAMA_LOG_INFO(
                        "[HYBRID_STAGE_SEARCH] T=%d P=%d C=%d G=%d R=%.3f XT=%d "
                        "XG=%d XC=%d XP=%d gpu=%.3f cpu=%.3f tensor=%.3f phone=%.3f "
                        "handoff=%.3f sim=%.3f gpu_wait=%.3f pc_mib=%.2f phone_mib=%.2f gpu_mib=%.2f\n",
                        candidate.tensor_layers, candidate.phone_layers, candidate.pc_layers,
                        candidate.gpu_pc_layers, candidate.tensor_pc_ratio,
                        candidate.tensor_chunk_tokens, candidate.gpu_chunk_tokens,
                        candidate.cpu_chunk_tokens, candidate.phone_chunk_tokens,
                        candidate.predicted_pc_gpu_ms, candidate.predicted_pc_cpu_ms,
                        candidate.predicted_tensor_ms, candidate.predicted_phone_ms,
                        candidate.predicted_handoff_ms, candidate.predicted_ms,
                        candidate.predicted_gpu_wait_ms,
                        candidate.pc_memory / 1048576.0,
                        candidate.phone_memory / 1048576.0,
                        candidate.gpu_memory / 1048576.0);
                }

                const bool better =
                    !found ||
                    candidate.predicted_ms < best_plan.predicted_ms ||
                    (candidate.predicted_ms == best_plan.predicted_ms &&
                     candidate.gpu_chunk_tokens > best_plan.gpu_chunk_tokens) ||
                    (candidate.predicted_ms == best_plan.predicted_ms &&
                     candidate.gpu_chunk_tokens == best_plan.gpu_chunk_tokens &&
                     candidate.cpu_chunk_tokens > best_plan.cpu_chunk_tokens) ||
                    (candidate.predicted_ms == best_plan.predicted_ms &&
                     candidate.gpu_chunk_tokens == best_plan.gpu_chunk_tokens &&
                     candidate.cpu_chunk_tokens == best_plan.cpu_chunk_tokens &&
                     candidate.phone_chunk_tokens > best_plan.phone_chunk_tokens);

                if (better) {
                    best_plan = std::move(candidate);
                    found = true;
                }
            }
        }
    }

    if (!found) {
        return false;
    }

    plan = std::move(best_plan);
    return true;
}

bool llama_hybrid_profile_gpu_transfer(llama_hybrid_profile & profile,
                                               ggml_backend_t         gpu_backend,
                                               ggml_backend_t         pc_backend) {
    profile.gpu_to_pc.clear();
    if (gpu_backend == nullptr) {
        return true;
    }
    if (pc_backend == nullptr || profile.n_embd <= 0) {
        return false;
    }

    const std::vector<int> token_sizes = llama_hybrid_probe_transfer_tokens(profile);
    if (token_sizes.empty()) {
        return false;
    }

    const size_t ctx_size = token_sizes.size() * ggml_tensor_overhead();
    const ggml_init_params params = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr gpu_ctx(ggml_init(params));
    ggml_context_ptr pc_ctx(ggml_init(params));
    if (!gpu_ctx || !pc_ctx) {
        return false;
    }

    std::vector<ggml_tensor *> gpu_tensors;
    std::vector<ggml_tensor *> pc_tensors;
    gpu_tensors.reserve(token_sizes.size());
    pc_tensors.reserve(token_sizes.size());
    for (const int tokens : token_sizes) {
        gpu_tensors.push_back(ggml_new_tensor_2d(gpu_ctx.get(), GGML_TYPE_F32, profile.n_embd, tokens));
        pc_tensors.push_back(ggml_new_tensor_2d(pc_ctx.get(), GGML_TYPE_F32, profile.n_embd, tokens));
    }

    ggml_backend_buffer_ptr gpu_buffer(ggml_backend_alloc_ctx_tensors(gpu_ctx.get(), gpu_backend));
    ggml_backend_buffer_ptr pc_buffer(ggml_backend_alloc_ctx_tensors(pc_ctx.get(), pc_backend));
    if (!gpu_buffer || !pc_buffer) {
        LLAMA_LOG_ERROR("%s: failed to allocate GPU/PC transfer probe buffers\n", __func__);
        return false;
    }

    const size_t max_bytes = ggml_nbytes(gpu_tensors.back());
    std::vector<uint8_t> source_data(max_bytes, 0x5a);

    for (size_t i = 0; i < token_sizes.size(); ++i) {
        ggml_tensor * src = gpu_tensors[i];
        ggml_tensor * dst = pc_tensors[i];
        const size_t bytes = ggml_nbytes(src);
        std::array<double, LLAMA_HYBRID_COMPUTE_MEASURE_RUNS> samples;

        for (int run = 0; run < LLAMA_HYBRID_COMPUTE_WARMUP_RUNS + LLAMA_HYBRID_COMPUTE_MEASURE_RUNS; ++run) {
            ggml_backend_tensor_set(src, source_data.data(), 0, bytes);
            ggml_backend_synchronize(gpu_backend);

            const int64_t begin_us = ggml_time_us();
            ggml_backend_tensor_copy(src, dst);
            ggml_backend_synchronize(gpu_backend);
            ggml_backend_synchronize(pc_backend);
            const int64_t end_us = ggml_time_us();

            if (run >= LLAMA_HYBRID_COMPUTE_WARMUP_RUNS) {
                samples[run - LLAMA_HYBRID_COMPUTE_WARMUP_RUNS] = (end_us - begin_us) / 1000.0;
            }
        }

        const double ms = llama_hybrid_rpc_median(samples);
        profile.gpu_to_pc.push_back({ bytes, ms, ms });
        LLAMA_LOG_INFO("[HYBRID_PROFILE_TRANSFER] kind=gpu_to_pc tokens=%d bytes=%zu ms=%.3f\n",
                       token_sizes[i], bytes, ms);
    }

    return true;
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

    const std::vector<int> chunk_tokens = llama_hybrid_probe_transfer_tokens(profile);
    if (chunk_tokens.empty()) {
        LLAMA_LOG_ERROR("%s: no valid transfer probe sizes\n", __func__);
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

    const size_t         max_bytes = (size_t) profile.n_embd * (size_t) chunk_tokens.back() * sizeof(float);
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
                                           double &                      result_ms,
                                           size_t &                      runtime_bytes) {
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

    const size_t total_buffer = ggml_backend_buffer_get_size(buffer.get());
    size_t       weight_bytes = 0;
    if (!llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(w_gate)) ||
        !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(w_up)) ||
        !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(w_down))) {
        return false;
    }
    runtime_bytes = total_buffer > weight_bytes ? total_buffer - weight_bytes : 0;

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

    std::vector<int> chunk_tokens = llama_hybrid_probe_chunk_tokens(profile);
    chunk_tokens.push_back(1);
    std::sort(chunk_tokens.begin(), chunk_tokens.end());
    chunk_tokens.erase(std::unique(chunk_tokens.begin(), chunk_tokens.end()), chunk_tokens.end());
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
    profile.ffn_shard_granularity      = std::lcm<int64_t>(ggml_blck_size(desc.down_type), 128);
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
        for (const float requested_local_ratio : LLAMA_HYBRID_FFN_RATIO_PROBES) {
            const int64_t cpu_local_ff = llama_hybrid_ffn_shard_size(desc.n_ff, desc.down_type,
                                                                      requested_local_ratio, 0);
            if (cpu_local_ff <= 0) {
                continue;
            }
            const float cpu_ratio_eff = (float) ((double) cpu_local_ff / (double) desc.n_ff);

            double cpu_ms;
            size_t cpu_runtime_bytes;
            if (!llama_hybrid_profile_ffn_point(desc, cpu_backend, 0, tokens, requested_local_ratio,
                                                cpu_ms, cpu_runtime_bytes)) {
                return false;
            }
            profile.cpu_ffn.push_back({ tokens, cpu_ratio_eff, cpu_local_ff, cpu_ms, cpu_runtime_bytes });
            LLAMA_LOG_INFO(
                "[HYBRID_PROFILE_COMPUTE] backend=%s kind=ffn tokens=%d local_ratio=%.5f local_ff=%" PRId64
                " ms=%.3f runtime_bytes=%zu\n",
                ggml_backend_name(cpu_backend), tokens, cpu_ratio_eff, cpu_local_ff, cpu_ms, cpu_runtime_bytes);

            const float pc_ratio_for_phone = 1.0f - requested_local_ratio;
            const int64_t phone_local_ff = llama_hybrid_ffn_shard_size(desc.n_ff, desc.down_type,
                                                                        pc_ratio_for_phone, 1);
            if (phone_local_ff <= 0) {
                continue;
            }
            const float phone_ratio_eff = (float) ((double) phone_local_ff / (double) desc.n_ff);

            double phone_ms;
            size_t phone_runtime_bytes;
            if (!llama_hybrid_profile_ffn_point(desc, phone_backend, 1, tokens, requested_local_ratio,
                                                phone_ms, phone_runtime_bytes)) {
                return false;
            }
            profile.phone_ffn.push_back({ tokens, phone_ratio_eff, phone_local_ff, phone_ms, phone_runtime_bytes });
            LLAMA_LOG_INFO(
                "[HYBRID_PROFILE_COMPUTE] backend=%s kind=ffn tokens=%d local_ratio=%.5f local_ff=%" PRId64
                " ms=%.3f runtime_bytes=%zu\n",
                ggml_backend_name(phone_backend), tokens, phone_ratio_eff, phone_local_ff, phone_ms,
                phone_runtime_bytes);
        }
    }
    return true;
}


static bool llama_hybrid_profile_moe_ffn_point(
        const llama_hybrid_moe_desc & desc,
        ggml_backend_t                backend,
        int                           backend_index,
        int                           tokens,
        float                         local_ratio,
        double &                      result_ms,
        size_t &                      runtime_bytes) {
    if (backend == nullptr || tokens <= 0 || desc.n_embd <= 0 || desc.n_ff_exp <= 0 ||
        desc.n_expert <= 0 || desc.n_expert_used <= 0 || desc.n_expert_used > desc.n_expert) {
        return false;
    }

    const float pc_ratio = backend_index == 0 ? local_ratio : 1.0f - local_ratio;
    const int64_t n_ff_shard = llama_hybrid_ffn_shard_size(
        desc.n_ff_exp, desc.down_exps_type, pc_ratio, backend_index);
    if (n_ff_shard <= 0) {
        LLAMA_LOG_ERROR(
            "%s: empty MoE FFN shard backend=%d local_ratio=%.3f\n",
            __func__, backend_index, local_ratio);
        return false;
    }

    static constexpr size_t graph_size = 128;
    const ggml_init_params params = {
        /*.mem_size   =*/256 * ggml_tensor_overhead() +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    const int64_t k = desc.n_expert_used;
    ggml_tensor * input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, desc.n_embd, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_I32, k, tokens);
    ggml_tensor * mix_weights = ggml_new_tensor_3d(
        ctx.get(), GGML_TYPE_F32, 1, k, tokens);

    ggml_tensor * w_gate = ggml_new_tensor_3d(
        ctx.get(), desc.gate_exps_type, desc.n_embd, n_ff_shard, desc.n_expert);
    ggml_tensor * w_up = ggml_new_tensor_3d(
        ctx.get(), desc.up_exps_type, desc.n_embd, n_ff_shard, desc.n_expert);
    ggml_tensor * w_down = ggml_new_tensor_3d(
        ctx.get(), desc.down_exps_type, n_ff_shard, desc.n_embd, desc.n_expert);

    ggml_tensor * input3 = ggml_reshape_3d(
        ctx.get(), input, desc.n_embd, 1, tokens);
    ggml_tensor * gate = ggml_mul_mat_id(ctx.get(), w_gate, input3, ids);
    gate = ggml_silu(ctx.get(), gate);
    ggml_tensor * up = ggml_mul_mat_id(ctx.get(), w_up, input3, ids);
    ggml_tensor * hidden = ggml_mul(ctx.get(), gate, up);
    ggml_tensor * experts = ggml_mul_mat_id(ctx.get(), w_down, hidden, ids);
    experts = ggml_mul(ctx.get(), experts, mix_weights);

    std::vector<ggml_tensor *> expert_views;
    expert_views.reserve((size_t) k);
    for (int64_t ie = 0; ie < k; ++ie) {
        expert_views.push_back(ggml_view_2d(
            ctx.get(), experts, desc.n_embd, tokens,
            experts->nb[2], ie * experts->nb[1]));
    }

    ggml_tensor * output = expert_views.front();
    for (size_t ie = 1; ie < expert_views.size(); ++ie) {
        output = ggml_add(ctx.get(), output, expert_views[ie]);
    }
    if (k == 1) {
        output = ggml_cont(ctx.get(), output);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_moe_probe_uid{
        (uint64_t(1) << 62) | (uint64_t(1) << 58)
    };
    graph->uid = next_moe_probe_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, output);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!ggml_backend_supports_op(backend, node)) {
            LLAMA_LOG_ERROR(
                "%s: backend=%s unsupported MoE expert op=%s node=%s\n",
                __func__, ggml_backend_name(backend),
                ggml_op_name(node->op), node->name);
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        LLAMA_LOG_ERROR(
            "%s: failed to allocate MoE expert probe on %s\n",
            __func__, ggml_backend_name(backend));
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    std::vector<int32_t> ids_data((size_t) k * (size_t) tokens);
    for (int t = 0; t < tokens; ++t) {
        for (int64_t ie = 0; ie < k; ++ie) {
            ids_data[(size_t) t * (size_t) k + (size_t) ie] =
                (int32_t) (((int64_t) t * k + ie) % desc.n_expert);
        }
    }
    ggml_backend_tensor_set(
        ids, ids_data.data(), 0, ids_data.size() * sizeof(ids_data[0]));

    std::vector<float> mix_data((size_t) k * (size_t) tokens, 1.0f / (float) k);
    ggml_backend_tensor_set(
        mix_weights, mix_data.data(), 0, mix_data.size() * sizeof(mix_data[0]));

    const size_t total_buffer = ggml_backend_buffer_get_size(buffer.get());
    size_t weight_bytes = 0;
    if (!llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(w_gate)) ||
        !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(w_up)) ||
        !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(w_down))) {
        return false;
    }
    runtime_bytes = total_buffer > weight_bytes ? total_buffer - weight_bytes : 0;

    llama_hybrid_graph_timing timing;
    if (!llama_hybrid_profile_graph_timing(backend, graph, timing)) {
        LLAMA_LOG_ERROR(
            "%s: MoE expert graph failed on %s\n",
            __func__, ggml_backend_name(backend));
        return false;
    }

    const bool is_rpc =
        llama_hybrid_rpc_get_proc_address(
            backend, GGML_BACKEND_RPC_FENCE_PROC) != nullptr;
    result_ms = is_rpc ? timing.compute_est_ms : timing.wall_ms;
    return true;
}

static bool llama_hybrid_profile_moe_router_point(
        const llama_hybrid_attn_desc & attn_desc,
        const llama_hybrid_moe_desc &  moe_desc,
        ggml_backend_t                 backend,
        int                            tokens,
        double &                       result_ms);

bool llama_hybrid_profile_moe_ffn(
        llama_hybrid_profile &         profile,
        const llama_hybrid_moe_desc &  desc,
        const llama_hybrid_attn_desc & attn_desc,
        ggml_backend_t                 cpu_backend,
        ggml_backend_t                 phone_backend) {
    if (cpu_backend == nullptr || phone_backend == nullptr ||
        desc.n_embd <= 0 || desc.n_ff_exp <= 0 ||
        desc.n_expert <= 0 || desc.n_expert_used <= 0 ||
        desc.gate_exps_type < 0 || desc.gate_exps_type >= GGML_TYPE_COUNT ||
        desc.up_exps_type < 0 || desc.up_exps_type >= GGML_TYPE_COUNT ||
        desc.down_exps_type < 0 || desc.down_exps_type >= GGML_TYPE_COUNT) {
        LLAMA_LOG_ERROR("%s: invalid MoE profile arguments\n", __func__);
        return false;
    }

    std::vector<int> chunk_tokens =
        llama_hybrid_probe_chunk_tokens(profile);
    chunk_tokens.push_back(1);
    std::sort(chunk_tokens.begin(), chunk_tokens.end());
    chunk_tokens.erase(std::unique(chunk_tokens.begin(), chunk_tokens.end()), chunk_tokens.end());
    if (chunk_tokens.empty()) {
        return false;
    }

    profile.n_embd = (int) desc.n_embd;
    profile.n_ff = (int) desc.n_ff_exp;
    profile.ffn_shard_granularity =
        std::lcm<int64_t>(ggml_blck_size(desc.down_exps_type), 128);
    profile.ffn_weight_bytes_per_layer = desc.weight_bytes;
    llama_hybrid_update_weight_bytes(profile);
    profile.cpu_ffn.clear();
    profile.phone_ffn.clear();

    if (profile.rpc_fence_ms <= 0.0 &&
        !llama_hybrid_profile_rpc_fence(
            phone_backend, profile.rpc_fence_ms)) {
        return false;
    }

    for (const int tokens : chunk_tokens) {
        std::set<int64_t> seen_cpu_ff;
        std::set<int64_t> seen_phone_ff;

        double cpu_router_ms = 0.0;
        double phone_router_ms = 0.0;
        if (!llama_hybrid_profile_moe_router_point(
                attn_desc, desc, cpu_backend, tokens, cpu_router_ms) ||
            !llama_hybrid_profile_moe_router_point(
                attn_desc, desc, phone_backend, tokens, phone_router_ms)) {
            return false;
        }

        for (const float requested_local_ratio :
             LLAMA_HYBRID_FFN_RATIO_PROBES) {
            const int64_t cpu_local_ff = llama_hybrid_ffn_shard_size(
                desc.n_ff_exp, desc.down_exps_type,
                requested_local_ratio, 0);
            if (cpu_local_ff > 0 && seen_cpu_ff.insert(cpu_local_ff).second) {
                const float cpu_ratio_eff =
                    (float) ((double) cpu_local_ff / (double) desc.n_ff_exp);
                double cpu_ms = 0.0;
                size_t cpu_runtime_bytes = 0;
                if (!llama_hybrid_profile_moe_ffn_point(
                        desc, cpu_backend, 0, tokens,
                        requested_local_ratio,
                        cpu_ms, cpu_runtime_bytes)) {
                    return false;
                }
                const double cpu_branch_ms = cpu_router_ms + cpu_ms;
                profile.cpu_ffn.push_back({
                    tokens, cpu_ratio_eff, cpu_local_ff,
                    cpu_branch_ms, cpu_runtime_bytes
                });
                LLAMA_LOG_INFO(
                    "[HYBRID_PROFILE_COMPUTE] backend=%s kind=moe_branch "
                    "tokens=%d topk=%" PRId64 " local_ratio=%.5f "
                    "local_ff=%" PRId64 " router_ms=%.3f expert_ms=%.3f "
                    "total_ms=%.3f runtime_bytes=%zu\n",
                    ggml_backend_name(cpu_backend), tokens,
                    desc.n_expert_used, cpu_ratio_eff,
                    cpu_local_ff, cpu_router_ms, cpu_ms,
                    cpu_branch_ms, cpu_runtime_bytes);
            }

            const float pc_ratio_for_phone =
                1.0f - requested_local_ratio;
            const int64_t phone_local_ff =
                llama_hybrid_ffn_shard_size(
                    desc.n_ff_exp, desc.down_exps_type,
                    pc_ratio_for_phone, 1);
            if (phone_local_ff > 0 && seen_phone_ff.insert(phone_local_ff).second) {
                const float phone_ratio_eff =
                    (float) ((double) phone_local_ff /
                             (double) desc.n_ff_exp);
                double phone_ms = 0.0;
                size_t phone_runtime_bytes = 0;
                if (!llama_hybrid_profile_moe_ffn_point(
                        desc, phone_backend, 1, tokens,
                        requested_local_ratio,
                        phone_ms, phone_runtime_bytes)) {
                    return false;
                }
                const double phone_branch_ms = phone_router_ms + phone_ms;
                profile.phone_ffn.push_back({
                    tokens, phone_ratio_eff, phone_local_ff,
                    phone_branch_ms, phone_runtime_bytes
                });
                LLAMA_LOG_INFO(
                    "[HYBRID_PROFILE_COMPUTE] backend=%s kind=moe_branch "
                    "tokens=%d topk=%" PRId64 " local_ratio=%.5f "
                    "local_ff=%" PRId64 " router_ms=%.3f expert_ms=%.3f "
                    "total_ms=%.3f runtime_bytes=%zu\n",
                    ggml_backend_name(phone_backend), tokens,
                    desc.n_expert_used, phone_ratio_eff,
                    phone_local_ff, phone_router_ms, phone_ms,
                    phone_branch_ms, phone_runtime_bytes);
            }
        }
    }

    return !profile.cpu_ffn.empty() && !profile.phone_ffn.empty();
}

static bool llama_hybrid_profile_attn_point(const llama_hybrid_attn_desc & desc,
                                            ggml_backend_t                 backend,
                                            int                            tokens,
                                            int                            kv_tokens,
                                            double &                       result_ms,
                                            size_t &                       runtime_bytes) {
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

    ggml_tensor * q_norm_w = nullptr;
    ggml_tensor * k_norm_w = nullptr;
    if (desc.has_q_norm) {
        q_norm_w = ggml_new_tensor_1d(ctx.get(), desc.q_norm_type, head_dim_q);
        q        = ggml_rms_norm(ctx.get(), q, desc.rms_eps);
        q        = ggml_mul(ctx.get(), q, q_norm_w);
    }
    if (desc.has_k_norm) {
        k_norm_w = ggml_new_tensor_1d(ctx.get(), desc.k_norm_type, head_dim_k);
        k        = ggml_rms_norm(ctx.get(), k, desc.rms_eps);
        k        = ggml_mul(ctx.get(), k, k_norm_w);
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

    const size_t total_buffer = ggml_backend_buffer_get_size(buffer.get());
    size_t       weight_bytes = 0;
    if (!llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(attn_norm_w)) ||
        !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(wq)) ||
        !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(wk)) ||
        !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(wv)) ||
        !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(wo)) ||
        (q_norm_w != nullptr && !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(q_norm_w))) ||
        (k_norm_w != nullptr && !llama_hybrid_add_bytes(weight_bytes, ggml_nbytes(k_norm_w)))) {
        return false;
    }
    runtime_bytes = total_buffer > weight_bytes ? total_buffer - weight_bytes : 0;

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
                                                      double &                       result_ms,
                                                      size_t &                       runtime_bytes) {
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

    const size_t total_buffer = ggml_backend_buffer_get_size(buffer.get());
    size_t       persistent_kv = 0;
    if (!llama_hybrid_add_bytes(persistent_kv, ggml_nbytes(k)) ||
        !llama_hybrid_add_bytes(persistent_kv, ggml_nbytes(v))) {
        return false;
    }
    runtime_bytes = total_buffer > persistent_kv ? total_buffer - persistent_kv : 0;

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

    if (profile.probe_tokens <= 0) {
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
                                     double * legacy_ms, const auto & token_candidates) {
        for (const int tokens : token_candidates) {
            if (tokens <= 0 || tokens > desc.n_ctx_orig) {
                continue;
            }

            double full_base_ms      = 0.0;
            size_t full_base_runtime = 0;
            if (!llama_hybrid_profile_attn_point(desc, backend, tokens, tokens, full_base_ms, full_base_runtime)) {
                return false;
            }
            if (legacy_ms != nullptr && tokens == profile.probe_tokens) {
                *legacy_ms = full_base_ms;
            }

            points.push_back({ tokens, tokens, full_base_ms, full_base_runtime });
            LLAMA_LOG_INFO(
                "[HYBRID_PROFILE_COMPUTE] backend=%s kind=attention tokens=%d kv_tokens=%d ms=%.3f "
                "runtime_bytes=%zu\n",
                ggml_backend_name(backend), tokens, tokens, full_base_ms, full_base_runtime);

            double flash_base_ms      = 0.0;
            size_t flash_base_runtime = 0;
            if (!llama_hybrid_profile_attn_kv_kernel_point(desc, backend, tokens, tokens, flash_base_ms,
                                                           flash_base_runtime)) {
                return false;
            }

            for (const int kv_tokens : LLAMA_HYBRID_PREFILL_KV_ANCHORS) {
                if (kv_tokens <= tokens || kv_tokens > desc.n_ctx_orig) {
                    continue;
                }

                double flash_ms      = 0.0;
                size_t flash_runtime = 0;
                if (!llama_hybrid_profile_attn_kv_kernel_point(desc, backend, tokens, kv_tokens, flash_ms,
                                                               flash_runtime)) {
                    return false;
                }

                const double attn_ms = full_base_ms + flash_ms - flash_base_ms;
                size_t       attn_runtime = full_base_runtime;
                if (flash_runtime > flash_base_runtime &&
                    !llama_hybrid_add_bytes(attn_runtime, flash_runtime - flash_base_runtime)) {
                    return false;
                }
                points.push_back({ tokens, kv_tokens, attn_ms, attn_runtime });
                LLAMA_LOG_INFO(
                    "[HYBRID_PROFILE_COMPUTE] backend=%s kind=attention tokens=%d kv_tokens=%d ms=%.3f "
                    "runtime_bytes=%zu\n",
                    ggml_backend_name(backend), tokens, kv_tokens, attn_ms, attn_runtime);
            }
        }
        return true;
    };

    if (!profile_backend(cpu_backend, profile.cpu_attn, &profile.cpu_attn_ms,
                         LLAMA_HYBRID_CPU_LAYER_TOKENS) ||
        !profile_backend(phone_backend, profile.phone_attn, &profile.phone_attn_ms,
                         LLAMA_HYBRID_PHONE_LAYER_TOKENS) ||
        (gpu_backend != nullptr &&
         !profile_backend(gpu_backend, profile.gpu_attn, nullptr, LLAMA_HYBRID_GPU_LAYER_TOKENS))) {
        return false;
    }

    if (profile.cpu_attn.empty()) {
        LLAMA_LOG_ERROR("%s: no valid attention profile points for context=%" PRId64 "\n", __func__,
                        desc.n_ctx_orig);
        return false;
    }

    return true;
}


static bool llama_hybrid_profile_moe_router_point(
        const llama_hybrid_attn_desc & attn_desc,
        const llama_hybrid_moe_desc &  moe_desc,
        ggml_backend_t                 backend,
        int                            tokens,
        double &                       result_ms) {
    if (backend == nullptr || tokens <= 0 ||
        moe_desc.n_embd <= 0 || moe_desc.n_expert <= 0 ||
        moe_desc.n_expert_used <= 0 ||
        moe_desc.n_expert_used > moe_desc.n_expert) {
        return false;
    }

    static constexpr size_t graph_size = 64;
    const ggml_init_params params = {
        /*.mem_size   =*/128 * ggml_tensor_overhead() +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, moe_desc.n_embd, tokens);
    ggml_tensor * router_w = ggml_new_tensor_2d(
        ctx.get(), moe_desc.router_type,
        moe_desc.n_embd, moe_desc.n_expert);

    // Qwen3-MoE computes FFN norm once for the whole stage macro, then
    // Router/Top-K per XT chunk.  Keep the per-chunk profile aligned with
    // that execution and do not charge RMSNorm here.
    GGML_UNUSED(attn_desc);
    ggml_tensor * logits = ggml_mul_mat(
        ctx.get(), router_w, input);
    ggml_tensor * probs = ggml_soft_max(
        ctx.get(), logits);
    ggml_tensor * selected = ggml_argsort_top_k(
        ctx.get(), probs, (int) moe_desc.n_expert_used);

    probs = ggml_reshape_3d(
        ctx.get(), probs, 1, moe_desc.n_expert, tokens);
    ggml_tensor * weights = ggml_get_rows(
        ctx.get(), probs, selected);
    weights = ggml_reshape_2d(
        ctx.get(), weights, moe_desc.n_expert_used, tokens);
    ggml_tensor * weights_sum = ggml_sum_rows(
        ctx.get(), weights);
    weights_sum = ggml_clamp(
        ctx.get(), weights_sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx.get(), weights, weights_sum);
    weights = ggml_reshape_3d(
        ctx.get(), weights, 1, moe_desc.n_expert_used, tokens);

    ggml_cgraph * graph =
        ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_moe_router_uid{
        (uint64_t(1) << 62) | (uint64_t(1) << 57)
    };
    graph->uid =
        next_moe_router_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, weights);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!ggml_backend_supports_op(backend, node)) {
            LLAMA_LOG_ERROR(
                "%s: backend=%s unsupported MoE router op=%s node=%s\n",
                __func__, ggml_backend_name(backend),
                ggml_op_name(node->op), node->name);
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        LLAMA_LOG_ERROR(
            "%s: failed to allocate MoE router probe on %s\n",
            __func__, ggml_backend_name(backend));
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    llama_hybrid_graph_timing timing;
    if (!llama_hybrid_profile_graph_timing(
            backend, graph, timing)) {
        return false;
    }

    const bool is_rpc =
        llama_hybrid_rpc_get_proc_address(
            backend, GGML_BACKEND_RPC_FENCE_PROC) != nullptr;
    result_ms =
        is_rpc ? timing.compute_est_ms : timing.wall_ms;
    return true;
}

static bool llama_hybrid_profile_moe_misc_point(
        const llama_hybrid_attn_desc & attn_desc,
        const llama_hybrid_moe_desc &  moe_desc,
        ggml_backend_t                 backend,
        int                            tokens,
        double &                       result_ms) {
    if (backend == nullptr || tokens <= 0 ||
        moe_desc.n_embd <= 0 ||
        moe_desc.ffn_norm_type < 0 ||
        moe_desc.ffn_norm_type >= GGML_TYPE_COUNT) {
        return false;
    }

    static constexpr size_t graph_size = 16;
    const ggml_init_params params = {
        /*.mem_size   =*/32 * ggml_tensor_overhead() +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    ggml_tensor * input = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, moe_desc.n_embd, tokens);
    ggml_tensor * norm_w = ggml_new_tensor_1d(
        ctx.get(), moe_desc.ffn_norm_type, moe_desc.n_embd);

    ggml_tensor * norm = ggml_rms_norm(
        ctx.get(), input, attn_desc.rms_eps);
    norm = ggml_mul(ctx.get(), norm, norm_w);
    ggml_tensor * output = ggml_add(ctx.get(), norm, input);

    ggml_cgraph * graph =
        ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_moe_misc_uid{
        (uint64_t(1) << 62) | (uint64_t(1) << 56)
    };
    graph->uid =
        next_moe_misc_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, output);

    for (int i = 0; i < graph->n_nodes; ++i) {
        if (!ggml_backend_supports_op(backend, graph->nodes[i])) {
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    llama_hybrid_graph_timing timing;
    if (!llama_hybrid_profile_graph_timing(
            backend, graph, timing)) {
        return false;
    }

    const bool is_rpc =
        llama_hybrid_rpc_get_proc_address(
            backend, GGML_BACKEND_RPC_FENCE_PROC) != nullptr;
    result_ms =
        is_rpc ? timing.compute_est_ms : timing.wall_ms;
    return true;
}

static bool llama_hybrid_profile_moe_branch_block_point(
        const llama_hybrid_attn_desc & attn_desc,
        const llama_hybrid_moe_desc &  moe_desc,
        ggml_backend_t                 backend,
        int                            n_layers,
        int                            tokens,
        llama_hybrid_graph_timing &    timing) {
    if (backend == nullptr || n_layers <= 0 || tokens <= 0 ||
        moe_desc.n_embd <= 0 || moe_desc.n_ff_exp <= 0 ||
        moe_desc.n_expert <= 0 || moe_desc.n_expert_used <= 0 ||
        moe_desc.n_expert_used > moe_desc.n_expert ||
        moe_desc.ffn_norm_type < 0 ||
        moe_desc.ffn_norm_type >= GGML_TYPE_COUNT ||
        moe_desc.router_type < 0 ||
        moe_desc.router_type >= GGML_TYPE_COUNT ||
        moe_desc.gate_exps_type < 0 ||
        moe_desc.gate_exps_type >= GGML_TYPE_COUNT ||
        moe_desc.up_exps_type < 0 ||
        moe_desc.up_exps_type >= GGML_TYPE_COUNT ||
        moe_desc.down_exps_type < 0 ||
        moe_desc.down_exps_type >= GGML_TYPE_COUNT) {
        return false;
    }

    const size_t graph_size =
        (size_t) n_layers * 64 + 32;
    const size_t tensor_budget =
        192 + (size_t) n_layers * 96;
    const ggml_init_params params = {
        /*.mem_size   =*/tensor_budget * ggml_tensor_overhead() +
                         ggml_graph_overhead_custom(graph_size, false),
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return false;
    }

    const int64_t k = moe_desc.n_expert_used;
    ggml_tensor * cur = ggml_new_tensor_2d(
        ctx.get(), GGML_TYPE_F32, moe_desc.n_embd, tokens);

    for (int il = 0; il < n_layers; ++il) {
        // Distinct synthetic weights per layer are essential here: the point
        // of this probe is to expose sustained weight-streaming behaviour that
        // a one-layer cache-hot microbenchmark cannot see.
        ggml_tensor * norm_w = ggml_new_tensor_1d(
            ctx.get(), moe_desc.ffn_norm_type, moe_desc.n_embd);
        ggml_tensor * router_w = ggml_new_tensor_2d(
            ctx.get(), moe_desc.router_type,
            moe_desc.n_embd, moe_desc.n_expert);
        ggml_tensor * w_gate = ggml_new_tensor_3d(
            ctx.get(), moe_desc.gate_exps_type,
            moe_desc.n_embd, moe_desc.n_ff_exp,
            moe_desc.n_expert);
        ggml_tensor * w_up = ggml_new_tensor_3d(
            ctx.get(), moe_desc.up_exps_type,
            moe_desc.n_embd, moe_desc.n_ff_exp,
            moe_desc.n_expert);
        ggml_tensor * w_down = ggml_new_tensor_3d(
            ctx.get(), moe_desc.down_exps_type,
            moe_desc.n_ff_exp, moe_desc.n_embd,
            moe_desc.n_expert);

        ggml_tensor * norm =
            ggml_rms_norm(ctx.get(), cur, attn_desc.rms_eps);
        norm = ggml_mul(ctx.get(), norm, norm_w);

        ggml_tensor * logits =
            ggml_mul_mat(ctx.get(), router_w, norm);
        ggml_tensor * probs =
            ggml_soft_max(ctx.get(), logits);
        ggml_tensor * selected =
            ggml_argsort_top_k(ctx.get(), probs, (int) k);

        probs = ggml_reshape_3d(
            ctx.get(), probs, 1, moe_desc.n_expert, tokens);
        ggml_tensor * mix_weights =
            ggml_get_rows(ctx.get(), probs, selected);
        mix_weights = ggml_reshape_2d(
            ctx.get(), mix_weights, k, tokens);
        ggml_tensor * weights_sum =
            ggml_sum_rows(ctx.get(), mix_weights);
        weights_sum = ggml_clamp(
            ctx.get(), weights_sum, 6.103515625e-5f, INFINITY);
        mix_weights =
            ggml_div(ctx.get(), mix_weights, weights_sum);
        mix_weights = ggml_reshape_3d(
            ctx.get(), mix_weights, 1, k, tokens);

        ggml_tensor * input3 = ggml_reshape_3d(
            ctx.get(), norm, moe_desc.n_embd, 1, tokens);
        ggml_tensor * gate =
            ggml_mul_mat_id(ctx.get(), w_gate, input3, selected);
        gate = ggml_silu(ctx.get(), gate);
        ggml_tensor * up =
            ggml_mul_mat_id(ctx.get(), w_up, input3, selected);
        ggml_tensor * hidden =
            ggml_mul(ctx.get(), gate, up);
        ggml_tensor * experts =
            ggml_mul_mat_id(ctx.get(), w_down, hidden, selected);
        experts = ggml_mul(ctx.get(), experts, mix_weights);

        std::vector<ggml_tensor *> expert_views;
        expert_views.reserve((size_t) k);
        for (int64_t ie = 0; ie < k; ++ie) {
            expert_views.push_back(ggml_view_2d(
                ctx.get(), experts, moe_desc.n_embd, tokens,
                experts->nb[2], ie * experts->nb[1]));
        }

        ggml_tensor * branch = expert_views.front();
        for (size_t ie = 1; ie < expert_views.size(); ++ie) {
            branch = ggml_add(ctx.get(), branch, expert_views[ie]);
        }
        if (k == 1) {
            branch = ggml_cont(ctx.get(), branch);
        }
        cur = ggml_add(ctx.get(), branch, cur);
    }

    ggml_cgraph * graph =
        ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_moe_block_uid{
        (uint64_t(1) << 62) | (uint64_t(1) << 55)
    };
    graph->uid =
        next_moe_block_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, cur);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!ggml_backend_supports_op(backend, node)) {
            LLAMA_LOG_ERROR(
                "%s: backend=%s layers=%d tokens=%d unsupported "
                "op=%s node=%s\n",
                __func__, ggml_backend_name(backend),
                n_layers, tokens,
                ggml_op_name(node->op), node->name);
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        LLAMA_LOG_ERROR(
            "%s: allocation failed backend=%s layers=%d tokens=%d\n",
            __func__, ggml_backend_name(backend),
            n_layers, tokens);
        return false;
    }
    ggml_backend_buffer_clear(buffer.get(), 0);

    if (!llama_hybrid_profile_graph_timing(
            backend, graph, timing)) {
        LLAMA_LOG_ERROR(
            "%s: execution failed backend=%s layers=%d tokens=%d\n",
            __func__, ggml_backend_name(backend),
            n_layers, tokens);
        return false;
    }
    return true;
}

bool llama_hybrid_profile_moe_full_layer(
        llama_hybrid_profile &         profile,
        const llama_hybrid_attn_desc & attn_desc,
        const llama_hybrid_moe_desc &  moe_desc,
        ggml_backend_t                 cpu_backend,
        ggml_backend_t                 phone_backend,
        ggml_backend_t                 gpu_backend) {
    if (cpu_backend == nullptr || phone_backend == nullptr ||
        attn_desc.n_embd != moe_desc.n_embd) {
        return false;
    }

    profile.cpu_layer_blocks.clear();
    profile.phone_layer_blocks.clear();
    profile.gpu_layer_blocks.clear();
    profile.phone_blocks.clear();

    profile.cpu_full_layer_ms = 0.0;
    profile.phone_full_layer_ms = 0.0;
    profile.phone_full_layer_compute_est_ms = 0.0;
    profile.gpu_full_layer_ms = 0.0;

    const auto profile_one = [&](ggml_backend_t backend,
                                 const std::vector<llama_hybrid_attn_compute_point> & attn_points,
                                 const std::vector<llama_hybrid_ffn_compute_point> * expert_points,
                                 int backend_index,
                                 int tokens,
                                 double & total_ms) {
        double attn_ms = 0.0;
        double branch_ms = 0.0;
        double misc_ms = 0.0;

        if (!llama_hybrid_attn_cost(
                attn_points, tokens, tokens, attn_ms) ||
            !llama_hybrid_profile_moe_misc_point(
                attn_desc, moe_desc, backend, tokens, misc_ms)) {
            return false;
        }

        if (expert_points != nullptr) {
            // cpu_ffn/phone_ffn are MoE branch points here:
            // Router + Top-K + expert compute + combine.
            if (!llama_hybrid_ffn_cost(
                    *expert_points, tokens, 1.0f, branch_ms)) {
                return false;
            }
        } else {
            double router_ms = 0.0;
            double expert_ms = 0.0;
            size_t expert_runtime_bytes = 0;
            if (!llama_hybrid_profile_moe_router_point(
                    attn_desc, moe_desc, backend, tokens, router_ms) ||
                !llama_hybrid_profile_moe_ffn_point(
                    moe_desc, backend, backend_index, tokens, 1.0f,
                    expert_ms, expert_runtime_bytes)) {
                return false;
            }
            branch_ms = router_ms + expert_ms;
        }

        total_ms = attn_ms + misc_ms + branch_ms;
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=%s kind=moe_layer "
            "tokens=%d attn_ms=%.3f misc_ms=%.3f branch_ms=%.3f "
            "total_ms=%.3f\n",
            ggml_backend_name(backend), tokens,
            attn_ms, misc_ms, branch_ms, total_ms);
        return true;
    };

    // CPU-only regions are searched up to XC=256. Probe those token sizes
    // directly instead of extrapolating the 32->64 slope into the large-token
    // regime. Passing nullptr for expert_points measures Router + Top-K +
    // full local expert compute on the CPU backend at the requested token size.
    for (const int tokens : LLAMA_HYBRID_CPU_LAYER_TOKENS) {
        if (tokens <= 0 || tokens > attn_desc.n_ctx_orig) {
            continue;
        }

        double cpu_ms = 0.0;
        if (!profile_one(
                cpu_backend, profile.cpu_attn, nullptr, 0,
                tokens, cpu_ms)) {
            return false;
        }
        profile.cpu_layer_blocks.push_back({
            tokens, 1, cpu_ms, cpu_ms
        });
        if (tokens == profile.probe_tokens) {
            profile.cpu_full_layer_ms = cpu_ms;
        }
    }

    // A one-layer MoE microbenchmark can be substantially cache-hot compared
    // with a real C=20..40 CPU stage. Add a small number of multi-layer block
    // probes to calibrate the sustained regime without turning startup into a
    // long stress test. Token=1 is the decode anchor; the larger anchors
    // calibrate sustained prefill CPU stages.
    for (const int tokens : LLAMA_HYBRID_MOE_CPU_BLOCK_TOKENS) {
        if (tokens <= 0 || tokens > attn_desc.n_ctx_orig ||
            LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS > profile.n_layer) {
            continue;
        }

        llama_hybrid_graph_timing branch_timing;
        if (!llama_hybrid_profile_moe_branch_block_point(
                attn_desc, moe_desc, cpu_backend,
                LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS,
                tokens, branch_timing)) {
            return false;
        }

        double attn_ms = 0.0;
        if (!llama_hybrid_attn_cost(
                profile.cpu_attn, tokens, tokens, attn_ms)) {
            return false;
        }

        const double total_ms =
            branch_timing.wall_ms +
            LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS * attn_ms;
        profile.cpu_layer_blocks.push_back({
            tokens,
            LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS,
            total_ms,
            total_ms
        });
        LLAMA_LOG_ERROR(
            "[HYBRID_PROFILE_CPU_DEPTH] kind=moe_block layers=%d "
            "tokens=%d branch_ms=%.3f attn_ms_per_layer=%.3f "
            "total_ms=%.3f per_layer_ms=%.3f\n",
            LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS,
            tokens,
            branch_timing.wall_ms,
            attn_ms,
            total_ms,
            total_ms / LLAMA_HYBRID_MOE_CPU_BLOCK_LAYERS);
    }

    // Keep phone full-layer probes at the existing small-token anchors; large
    // CPU anchors are specifically for the PC CPU-only planner stage.
    for (const int tokens : LLAMA_HYBRID_PHONE_LAYER_TOKENS) {
        if (tokens <= 0 || tokens > attn_desc.n_ctx_orig) {
            continue;
        }

        double phone_ms = 0.0;
        if (!profile_one(
                phone_backend, profile.phone_attn, &profile.phone_ffn, 1,
                tokens, phone_ms)) {
            return false;
        }
        profile.phone_layer_blocks.push_back({
            tokens, 1, phone_ms, phone_ms
        });
        if (tokens == profile.probe_tokens) {
            profile.phone_full_layer_ms = phone_ms;
            profile.phone_full_layer_compute_est_ms = phone_ms;
        }
    }

    if (gpu_backend != nullptr) {
        for (const int tokens : LLAMA_HYBRID_GPU_LAYER_TOKENS) {
            if (tokens <= 0 || tokens > attn_desc.n_ctx_orig) {
                continue;
            }
            double gpu_ms = 0.0;
            if (!profile_one(
                    gpu_backend, profile.gpu_attn, nullptr, 0,
                    tokens, gpu_ms)) {
                return false;
            }
            profile.gpu_layer_blocks.push_back({
                tokens, 1, gpu_ms, gpu_ms
            });
            if (tokens == profile.probe_tokens) {
                profile.gpu_full_layer_ms = gpu_ms;
            }
        }
    }

    if (profile.cpu_full_layer_ms <= 0.0 &&
        !profile.cpu_layer_blocks.empty()) {
        profile.cpu_full_layer_ms =
            profile.cpu_layer_blocks.front().wall_ms;
    }
    if (profile.phone_full_layer_compute_est_ms <= 0.0 &&
        !profile.phone_layer_blocks.empty()) {
        profile.phone_full_layer_ms =
            profile.phone_layer_blocks.front().wall_ms;
        profile.phone_full_layer_compute_est_ms =
            profile.phone_layer_blocks.front().compute_est_ms;
    }
    if (gpu_backend != nullptr &&
        profile.gpu_full_layer_ms <= 0.0 &&
        !profile.gpu_layer_blocks.empty()) {
        profile.gpu_full_layer_ms =
            profile.gpu_layer_blocks.front().wall_ms;
    }

    return !profile.cpu_layer_blocks.empty() &&
           !profile.phone_layer_blocks.empty() &&
           (gpu_backend == nullptr || !profile.gpu_layer_blocks.empty());
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

static bool llama_hybrid_profile_layer_block_point(const llama_hybrid_attn_desc & attn_desc,
                                                   const llama_hybrid_ffn_desc &  ffn_desc,
                                                   ggml_backend_t                 backend,
                                                   int                            n_layers,
                                                   int                            tokens,
                                                   llama_hybrid_graph_timing &    timing) {
    if (backend == nullptr || n_layers <= 0 || tokens <= 0 || attn_desc.n_embd <= 0 || ffn_desc.n_embd <= 0 ||
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

    ggml_tensor * cur = input;
    for (int il = 0; il < n_layers; ++il) {
        // Use distinct synthetic weights per layer. Reusing one weight set would make later
        // layers unrealistically cache-hot and underestimate real multi-layer execution time.
        const llama_hybrid_probe_layer_weights weights =
            llama_hybrid_make_probe_layer_weights(ctx.get(), attn_desc, ffn_desc);
        cur = llama_hybrid_build_probe_layer(ctx.get(), cur, pos, mask, attn_desc, ffn_desc, weights, tokens);
    }

    ggml_cgraph *                graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    static std::atomic<uint64_t> next_layer_block_uid{ uint64_t(1) << 59 };
    graph->uid = next_layer_block_uid.fetch_add(1, std::memory_order_relaxed);
    ggml_build_forward_expand(graph, cur);

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (!ggml_backend_supports_op(backend, node)) {
            LLAMA_LOG_ERROR("%s: backend=%s layers=%d unsupported op=%s node=%s\n", __func__,
                            ggml_backend_name(backend), n_layers, ggml_op_name(node->op), node->name);
            return false;
        }
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    if (!buffer) {
        LLAMA_LOG_ERROR("%s: allocation failed layers=%d backend=%s\n", __func__, n_layers,
                        ggml_backend_name(backend));
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

    if (!llama_hybrid_profile_graph_timing(backend, graph, timing)) {
        LLAMA_LOG_ERROR("%s: layer block execution failed layers=%d\n", __func__, n_layers);
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

    const int tokens = profile.probe_tokens;
    if (tokens <= 0) {
        return false;
    }

    if (profile.rpc_fence_ms <= 0.0 && !llama_hybrid_profile_rpc_fence(phone_backend, profile.rpc_fence_ms)) {
        return false;
    }

    profile.attn_weight_bytes_per_layer = attn_desc.weight_bytes;
    profile.ffn_weight_bytes_per_layer  = ffn_desc.weight_bytes;
    llama_hybrid_update_weight_bytes(profile);

    profile.cpu_layer_blocks.clear();
    profile.phone_layer_blocks.clear();
    profile.gpu_layer_blocks.clear();

    const int block_layers = std::max(1, std::min(profile.profile_block_layers > 0 ? profile.profile_block_layers :
                                                  LLAMA_HYBRID_PROFILE_BLOCK_LAYERS,
                                                  profile.n_layer > 0 ? profile.n_layer : LLAMA_HYBRID_PROFILE_BLOCK_LAYERS));

    profile.cpu_full_layer_ms              = 0.0;
    profile.phone_full_layer_ms            = 0.0;
    profile.phone_full_layer_compute_est_ms = 0.0;
    profile.gpu_full_layer_ms              = 0.0;

    for (const int tokens : LLAMA_HYBRID_CPU_LAYER_TOKENS) {
        if (tokens <= 0 || tokens > attn_desc.n_ctx_orig) {
            continue;
        }

        llama_hybrid_graph_timing cpu_timing;
        if (!llama_hybrid_profile_layer_block_point(attn_desc, ffn_desc, cpu_backend, block_layers, tokens,
                                                    cpu_timing)) {
            return false;
        }
        profile.cpu_layer_blocks.push_back({ tokens, block_layers, cpu_timing.wall_ms, cpu_timing.compute_est_ms });
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=%s kind=layer_block layers=%d tokens=%d total_ms=%.3f "
            "per_layer_ms=%.3f\n",
            ggml_backend_name(cpu_backend), block_layers, tokens, cpu_timing.wall_ms,
            cpu_timing.wall_ms / block_layers);
    }

    for (const int tokens : LLAMA_HYBRID_PHONE_LAYER_TOKENS) {
        if (tokens <= 0 || tokens > attn_desc.n_ctx_orig) {
            continue;
        }

        llama_hybrid_graph_timing phone_timing;
        if (!llama_hybrid_profile_layer_block_point(attn_desc, ffn_desc, phone_backend, block_layers, tokens,
                                                    phone_timing)) {
            return false;
        }
        profile.phone_layer_blocks.push_back({ tokens, block_layers, phone_timing.wall_ms,
                                               phone_timing.compute_est_ms });
        LLAMA_LOG_INFO(
            "[HYBRID_PROFILE_COMPUTE] backend=%s kind=layer_block layers=%d tokens=%d wall_ms=%.3f "
            "compute_est_ms=%.3f per_layer_est_ms=%.3f\n",
            ggml_backend_name(phone_backend), block_layers, tokens, phone_timing.wall_ms,
            phone_timing.compute_est_ms, phone_timing.compute_est_ms / block_layers);
    }

    if (gpu_backend != nullptr) {
        for (const int tokens : LLAMA_HYBRID_GPU_LAYER_TOKENS) {
            if (tokens <= 0 || tokens > attn_desc.n_ctx_orig) {
                continue;
            }

            llama_hybrid_graph_timing gpu_timing;
            if (!llama_hybrid_profile_layer_block_point(attn_desc, ffn_desc, gpu_backend, block_layers, tokens,
                                                        gpu_timing)) {
                return false;
            }
            profile.gpu_layer_blocks.push_back({ tokens, block_layers, gpu_timing.wall_ms, gpu_timing.compute_est_ms });
            LLAMA_LOG_INFO(
                "[HYBRID_PROFILE_COMPUTE] backend=%s kind=layer_block layers=%d tokens=%d total_ms=%.3f "
                "per_layer_ms=%.3f\n",
                ggml_backend_name(gpu_backend), block_layers, tokens, gpu_timing.wall_ms,
                gpu_timing.wall_ms / block_layers);
        }
    }

    auto find_probe_block = [&](const std::vector<llama_hybrid_layer_compute_point> & points,
                                bool use_compute_est, double & result_ms) -> bool {
        if (points.empty()) {
            return false;
        }
        const llama_hybrid_layer_compute_point * best = nullptr;
        for (const auto & point : points) {
            if (best == nullptr || std::abs(point.tokens - profile.probe_tokens) <
                                   std::abs(best->tokens - profile.probe_tokens)) {
                best = &point;
            }
        }
        if (best == nullptr || best->layers <= 0) {
            return false;
        }
        const double total = use_compute_est ? best->compute_est_ms : best->wall_ms;
        result_ms = total / best->layers;
        return true;
    };

    if (!find_probe_block(profile.cpu_layer_blocks, false, profile.cpu_full_layer_ms) ||
        !find_probe_block(profile.phone_layer_blocks, false, profile.phone_full_layer_ms) ||
        !find_probe_block(profile.phone_layer_blocks, true, profile.phone_full_layer_compute_est_ms)) {
        return false;
    }
    if (gpu_backend != nullptr && !find_probe_block(profile.gpu_layer_blocks, false, profile.gpu_full_layer_ms)) {
        return false;
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
        if (!llama_hybrid_profile_layer_block_point(attn_desc, ffn_desc, phone_backend, n_layers, tokens, timing)) {
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

bool llama_hybrid_autoplan(llama_model_loader & ml, const llama_model_params & params, llama_hybrid_plan & best_plan) {
    llama_hybrid_runtime_plan_clear();

    ggml_backend_ptr cpu;
    ggml_backend_ptr phone;
    ggml_backend_ptr gpu;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev      = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg      = ggml_backend_dev_backend_reg(dev);
        const std::string  reg_name = reg != nullptr ? ggml_backend_reg_name(reg) : "";

        if (!phone && reg_name == "RPC") {
            phone.reset(ggml_backend_dev_init(dev, nullptr));
        } else if (!cpu && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu.reset(ggml_backend_dev_init(dev, nullptr));
        } else if (!gpu && reg_name == "CUDA" && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu.reset(ggml_backend_dev_init(dev, nullptr));
        }
    }

    if (!cpu || !phone) {
        LLAMA_LOG_ERROR("%s: automatic planning requires CPU and RPC backends\n", __func__);
        return false;
    }

    const bool is_qwen3_moe = ml.get_arch() == LLM_ARCH_QWEN3MOE;
    if (ml.get_arch() != LLM_ARCH_LLAMA &&
        ml.get_arch() != LLM_ARCH_QWEN2 &&
        ml.get_arch() != LLM_ARCH_QWEN3 &&
        !is_qwen3_moe) {
        LLAMA_LOG_ERROR(
            "%s: hybrid auto does not support architecture %s\n",
            __func__, ml.get_arch_name().c_str());
        return false;
    }

    int n_layer = 0;
    for (int il = 0; il < 1024; ++il) {
        bool ok = false;
        if (is_qwen3_moe) {
            llama_hybrid_moe_desc desc;
            ok = ml.get_hybrid_moe_desc(il, desc);
        } else {
            llama_hybrid_ffn_desc desc;
            ok = ml.get_hybrid_ffn_desc(il, desc);
        }
        if (!ok) {
            break;
        }
        ++n_layer;
    }
    if (n_layer <= 0) {
        LLAMA_LOG_ERROR(
            "%s: model has no supported %s layers\n",
            __func__, is_qwen3_moe ? "MoE" : "dense FFN");
        return false;
    }

    const int probe_layer = n_layer / 2;
    llama_hybrid_ffn_desc  ffn_desc;
    llama_hybrid_moe_desc  moe_desc;
    llama_hybrid_attn_desc attn_desc;

    const bool desc_ok = is_qwen3_moe ?
        ml.get_hybrid_moe_desc(probe_layer, moe_desc) :
        ml.get_hybrid_ffn_desc(probe_layer, ffn_desc);
    if (!desc_ok ||
        !ml.get_hybrid_attn_desc(probe_layer, attn_desc)) {
        LLAMA_LOG_ERROR(
            "%s: failed to describe probe layer %d\n",
            __func__, probe_layer);
        return false;
    }

    llama_hybrid_profile profile;
    profile.is_moe = is_qwen3_moe;
    profile.n_layer = n_layer;
    profile.n_embd = (int) (is_qwen3_moe ?
        moe_desc.n_embd : ffn_desc.n_embd);
    profile.n_ff = (int) (is_qwen3_moe ?
        moe_desc.n_ff_exp : ffn_desc.n_ff);
    profile.probe_tokens = LLAMA_HYBRID_PROFILE_TOKENS;
    profile.reference_tokens = LLAMA_HYBRID_REFERENCE_TOKENS;
    profile.probe_chunk_min_tokens = 4;
    profile.profile_block_layers = LLAMA_HYBRID_PROFILE_BLOCK_LAYERS;

    if (!ml.get_hybrid_weight_bytes(
            n_layer,
            profile.model_weight_bytes,
            profile.non_layer_weight_bytes,
            profile.layer_weight_bytes,
            profile.layer_attn_forced_bytes,
            profile.layer_ffn_split_bytes,
            profile.layer_mirrored_bytes) ||
        !llama_hybrid_profile_memory(
            profile, cpu.get(), phone.get(), gpu.get()) ||
        !llama_hybrid_profile_gpu_transfer(
            profile, gpu.get(), cpu.get()) ||
        !llama_hybrid_profile_rpc(
            profile, cpu.get(), phone.get()) ||
        !llama_hybrid_profile_attention(
            profile, attn_desc, cpu.get(), phone.get(), gpu.get())) {
        LLAMA_LOG_ERROR("%s: common profiling failed\n", __func__);
        return false;
    }

    if (is_qwen3_moe) {
        LLAMA_LOG_INFO(
            "[HYBRID_MOE] arch=qwen3moe layers=%d n_embd=%" PRId64
            " n_ff_exp=%" PRId64 " experts=%" PRId64
            " topk=%" PRId64 "\n",
            n_layer, moe_desc.n_embd, moe_desc.n_ff_exp,
            moe_desc.n_expert, moe_desc.n_expert_used);

        if (!llama_hybrid_profile_moe_ffn(
                profile, moe_desc, attn_desc, cpu.get(), phone.get()) ||
            !llama_hybrid_profile_moe_full_layer(
                profile, attn_desc, moe_desc,
                cpu.get(), phone.get(), gpu.get())) {
            LLAMA_LOG_ERROR("%s: MoE profiling failed\n", __func__);
            return false;
        }
    } else {
        if (!llama_hybrid_profile_ffn(
                profile, ffn_desc, cpu.get(), phone.get()) ||
            !llama_hybrid_profile_full_layer(
                profile, attn_desc, ffn_desc,
                cpu.get(), phone.get(), gpu.get())) {
            LLAMA_LOG_ERROR("%s: dense profiling failed\n", __func__);
            return false;
        }
    }

    llama_hybrid_profile_print(profile);

    llama_hybrid_constraints constraints;
    constraints.target_ctx           = params.hybrid_target_ctx;
    constraints.score_kv_tokens      = params.hybrid_target_ctx;
    constraints.target_ubatch_tokens = params.hybrid_target_ubatch_tokens;
    constraints.fixed_tensor_layers = 37;
    // HYBRID_AUTO intentionally leaves topology, ratio and chunk choices unfixed.
    // Fixed fields remain available in llama_hybrid_constraints for targeted
    // diagnostics, but normal planning compares all supported alternatives.
    const size_t pc_budget = llama_hybrid_effective_budget(
        constraints.pc_memory_budget, profile.pc_free_mem, LLAMA_HYBRID_PC_MEMORY_FRACTION);
    const size_t phone_budget = llama_hybrid_effective_budget(
        constraints.phone_memory_budget, profile.phone_free_mem, LLAMA_HYBRID_PHONE_MEMORY_FRACTION);
    const size_t gpu_budget = llama_hybrid_effective_budget(
        constraints.gpu_memory_budget, profile.gpu_free_mem, LLAMA_HYBRID_GPU_MEMORY_FRACTION);

    const std::vector<llama_hybrid_plan> candidates = llama_hybrid_enumerate_feasible_plans(profile, constraints);
    bool   found                  = false;
    size_t scored                 = 0;
    size_t memory_feasible        = 0;
    size_t memory_estimate_failed = 0;
    size_t reject_pc              = 0;
    size_t reject_phone           = 0;
    size_t reject_gpu             = 0;

    for (auto plan : candidates) {
        if (!llama_hybrid_score_plan(profile, constraints, plan)) {
            continue;
        }
        ++scored;

        // XG/XC/XP are final only after score_plan(). Estimate the runtime peak
        // with those selected chunks, then apply the single memory feasibility filter.
        if (!llama_hybrid_estimate_plan_memory(profile, constraints, plan)) {
            ++memory_estimate_failed;
            continue;
        }
        if (pc_budget > 0 && plan.pc_memory > pc_budget) {
            ++reject_pc;
            continue;
        }
        if (phone_budget > 0 && plan.phone_memory > phone_budget) {
            ++reject_phone;
            continue;
        }
        if (plan.gpu_pc_layers > 0 && (gpu_budget == 0 || plan.gpu_memory > gpu_budget)) {
            ++reject_gpu;
            continue;
        }
        ++memory_feasible;

        if (!found || plan.predicted_ms < best_plan.predicted_ms ||
            (plan.predicted_ms == best_plan.predicted_ms && plan.gpu_memory < best_plan.gpu_memory)) {
            best_plan = plan;
            found     = true;
        }
    }

    LLAMA_LOG_ERROR(
        "[HYBRID_PLAN_MEMORY_FILTER] scored=%zu feasible=%zu estimate_failed=%zu "
        "reject_pc=%zu reject_phone=%zu reject_gpu=%zu pc_budget=%zu phone_budget=%zu gpu_budget=%zu\n",
        scored, memory_feasible, memory_estimate_failed, reject_pc, reject_phone, reject_gpu,
        pc_budget, phone_budget, gpu_budget);

    if (!found) {
        LLAMA_LOG_ERROR("%s: no scoreable hybrid plans\n", __func__);
    } else {
        {
            std::lock_guard<std::mutex> lock(g_llama_hybrid_runtime_plan_mutex);
            g_llama_hybrid_runtime_profile = profile;
            g_llama_hybrid_runtime_constraints = constraints;
        }
    }

    if (found) {
        llama_hybrid_decode_plan decode_plan;
        if (!llama_hybrid_predict_decode_plan(
                profile, constraints, decode_plan)) {
            LLAMA_LOG_ERROR(
                "[DECODE_PLAN] prediction_failed applied=0\n");
        }
    }

    if (found && !llama_hybrid_sim_stages(best_plan).empty()) {
        const int reference_tokens = profile.reference_tokens > 0 ?
            profile.reference_tokens : LLAMA_HYBRID_REFERENCE_TOKENS;
        const int work_tokens = constraints.target_ubatch_tokens > 0 ?
            constraints.target_ubatch_tokens : reference_tokens;
        llama_hybrid_sim_result sim;
        if (llama_hybrid_simulate_prefill(profile, constraints, best_plan, work_tokens, true, sim)) {
            LLAMA_LOG_ERROR(
                "[PRED_PIPE_LEGACY] plan=T%d,P%d,C%d,G%d,XG%d,XC%d,XT%d,XP%d sim_ms=%.3f "
                "gpu_busy_ms=%.3f downstream_ms=%.3f gpu_wait_ms=%.3f tensor_peak_mib=%.2f "
                "phone_peak_mib=%.2f schedule=%s\n",
                best_plan.tensor_layers, best_plan.phone_layers, best_plan.pc_layers,
                best_plan.gpu_pc_layers, best_plan.gpu_chunk_tokens, best_plan.cpu_chunk_tokens,
                best_plan.tensor_chunk_tokens, best_plan.phone_chunk_tokens, sim.makespan_ms,
                sim.gpu_busy_ms, sim.downstream_ms, sim.gpu_wait_ms,
                sim.tensor_peak_bytes / 1048576.0, sim.phone_peak_bytes / 1048576.0,
                sim.schedule.c_str());
        }
    }
    return found;
}
/*constraints.fixed_tensor_layers      = 37;
    constraints.fixed_phone_layers       = 0;
    constraints.fixed_pc_layers          = 27;
    constraints.fixed_tensor_pc_ratio     = 0.713f;
    constraints.fixed_gpu_pc_layers      = 14;
    constraints.fixed_gpu_chunk_tokens    = 192;
    constraints.fixed_cpu_chunk_tokens    = 192;
    constraints.fixed_tensor_chunk_tokens = 48;
    constraints.fixed_phone_chunk_tokens  = 4;
    */
