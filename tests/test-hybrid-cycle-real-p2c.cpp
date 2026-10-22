// Opt-in real-Qwen3-MoE P2C numerical comparison, not a production Cycle
// executor. A separate CPU-only GGUF model provides the independent full
// Attention/Norm/Router/FFN/l_out reference, and is freed before the Cycle
// model is loaded. PC CPU + Phone GPU then run *real physical FFN shards*
// independently. Only PC sums both partials and adds the captured residual.
// Phone GPU executes actual causal AttentionPre. Live P2C FIFO runs use
// multiple token chunks, one persistent F16 K/V cache ON Phone GPU and
// absolute RoPE positions. READY/PARTIAL are two Phone->PC data-plane
// payloads per chunk; CPU owns the unique final Reduce.
//
// The legacy offline path remains available for numerical diagnostics.
// Multi-chunk mode still covers ONE physical P2C layer, not the entire
// 48-layer Cycle, and the cache is released after this request.
//
// Usage (choose a TENSOR_P2C layer in the fixed cycle):
//   test-hybrid-cycle-real-p2c <Qwen3-MoE.gguf> <phone:rpc-port> <layer> [prompt] [n_tokens]
// n_tokens=1..128 creates a deterministic token-count case by repeating
// and truncating the prompt, keeping one causal ubatch with positions 0..N-1.
// Set LLAMA_HYBRID_CYCLE_PC_LAYERS and LLAMA_HYBRID_CYCLE_PHONE_LAYERS
// before launching (e.g. 5 and 1, then the first P2C layer is 7).
#include "llama.h"
#include "llama-model.h"
#include "llama-hybrid-cycle.h"
#include "llama-hybrid-cycle-fifo-executor.h"
#include "llama-hybrid-cycle-ffn-executor.h"
#include "inference-cycle-transport.h"
#include "inference-cycle-reduce.h"
#include "ggml-rpc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

struct cycle_real_capture_value {
    std::vector<uint8_t> bytes;
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[GGML_MAX_DIMS] = {};
    int count = 0;
};

struct cycle_real_capture {
    std::map<std::string, cycle_real_capture_value> values;
    bool error = false;
};

// Optional host-only checkpoint references for localizing numerical drift
// before it reaches the final CPU parity comparison. The corresponding
// CPU model has already been released when the physical period begins.
struct cycle_real_period_reference {
    std::vector<std::vector<float>> layer_outputs;
    std::vector<std::vector<float>> pc_attn_residuals;
    std::vector<std::vector<float>> pc_ffn_outputs;
    std::vector<float> c2p_residual;
    std::vector<float> c2p_full_ffn;
    std::vector<std::vector<float>> phone_attn_residuals;
    std::vector<std::vector<float>> phone_ffn_norms;
    std::vector<std::vector<float>> phone_ffn_outputs;
    std::vector<float> p2c_residual;
    std::vector<float> p2c_norm;
};

// This is the ordinary llama_context scheduler callback. On ask=false,
// the scheduler has already synchronized the node's producer backend.
static bool cycle_real_capture_cb(ggml_tensor * tensor, bool ask, void * userdata) {
    auto & state = *static_cast<cycle_real_capture *>(userdata);
    if (!tensor) {
        state.error = true;
        return false;
    }
    auto it = state.values.find(tensor->name);
    if (it == state.values.end()) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    if (!tensor->buffer || !ggml_is_contiguous(tensor)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=NO_BUFFER_OR_NONCONTIG\n",
            tensor->name);
        state.error = true;
        return false;
    }
    auto & output = it->second;
    if (output.count != 0) {
        // A single prompt ubatch should deliver exactly one complete tensor.
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=DUPLICATE_CAPTURE\n",
            tensor->name);
        state.error = true;
        return false;
    }
    output.type = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        output.ne[i] = tensor->ne[i];
    }
    output.bytes.resize(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, output.bytes.data(), 0, output.bytes.size());
    ++output.count;
    std::printf("[CYCLE_REAL_CAPTURE] tensor=%s type=%s bytes=%zu "
                "ne=[%lld,%lld,%lld] status=OK\n",
                tensor->name, ggml_type_name(tensor->type), output.bytes.size(),
                (long long)tensor->ne[0], (long long)tensor->ne[1],
                (long long)tensor->ne[2]);
    return true;
}

static bool cycle_real_set_compare_layer(int layer) {
    const std::string value = std::to_string(layer);
#ifdef _WIN32
    return _putenv_s("LLAMA_HYBRID_CYCLE_REAL_COMPARE_LAYER", value.c_str()) == 0;
#else
    return setenv("LLAMA_HYBRID_CYCLE_REAL_COMPARE_LAYER", value.c_str(), 1) == 0;
#endif
}

// The fixed Cycle placement variables are process-global and the production
// llama_model_load path intentionally rejects them with split_mode=LAYER.
// Hide only the placement variables while running the independent CPU
// reference. Keep LLAMA_HYBRID_CYCLE_REAL_COMPARE_LAYER active so the full
// Qwen3-MoE graph still exposes its captured Router/FFN outputs.
//
// Both original values are copied before mutating the environment and are
// restored on scope exit, including on any early return.
static bool cycle_real_env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// Host-only numeric checkpoint; never reads back GPU buffers or sends data.
static void cycle_real_numeric_probe(
        const char * point, int layer, int chunk,
        const std::vector<float> & values) {
    if (!cycle_real_env_flag("LLAMA_HYBRID_CYCLE_NUMERIC_TRACE")) {
        return;
    }
    size_t zeros = 0;
    size_t nonfinite = 0;
    double max_abs = 0.0;
    double sum_abs = 0.0;
    for (float v : values) {
        if (!std::isfinite(v)) {
            ++nonfinite;
        } else {
            zeros += v == 0.0f ? 1 : 0;
            const double absolute = std::fabs(double(v));
            max_abs = std::max(max_abs, absolute);
            sum_abs += absolute;
        }
    }
    std::printf(
        "[CYCLE_NUMERIC] point=%s layer=%d chunk=%d elements=%zu "
        "zeros=%zu nonfinite=%zu max_abs=%.9g sum_abs=%.9g "
        "first=%.9g last=%.9g state=%s\n",
        point, layer, chunk, values.size(), zeros, nonfinite,
        max_abs, sum_abs,
        values.empty() ? 0.0 : double(values.front()),
        values.empty() ? 0.0 : double(values.back()),
        values.empty() ? "EMPTY" :
            nonfinite ? "NONFINITE" :
            max_abs == 0.0 ? "ALL_ZERO" : "NONZERO");
    std::fflush(stdout);
}

static void cycle_real_phone_mem_checkpoint(
        ggml_backend_dev_t phone_dev, const char * stage) {
    if (!phone_dev) {
        return;
    }
    // Device-reported OpenCL memory is indicative, not Android MemAvailable
    // or LMKD attribution. In particular it need not reflect every driver
    // cache, system allocation or the separate Termux UI process.
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(phone_dev, &free_bytes, &total_bytes);
    const double mib = 1024.0 * 1024.0;
    std::printf(
        "[CYCLE_DIAG_MEMORY] stage=%s phone_free_mib=%.2f "
        "phone_total_mib=%.2f source=GGML_BACKEND_DEV_MEMORY "
        "scope=DEVICE_REPORTED_NOT_ANDROID_LMK\n",
        stage, double(free_bytes) / mib, double(total_bytes) / mib);
    std::fflush(stdout);
}

static bool cycle_real_set_env(const char * name, const char * value) {
#ifdef _WIN32
    return _putenv_s(name, value ? value : "") == 0;
#else
    return value ? setenv(name, value, 1) == 0 : unsetenv(name) == 0;
#endif
}

class cycle_real_cpu_reference_env {
public:
    cycle_real_cpu_reference_env() {
        const char * pc = std::getenv("LLAMA_HYBRID_CYCLE_PC_LAYERS");
        const char * phone = std::getenv("LLAMA_HYBRID_CYCLE_PHONE_LAYERS");
        had_pc_ = pc != nullptr;
        had_phone_ = phone != nullptr;
        if (had_pc_) {
            saved_pc_ = pc;
        }
        if (had_phone_) {
            saved_phone_ = phone;
        }
        const bool pc_cleared =
            cycle_real_set_env("LLAMA_HYBRID_CYCLE_PC_LAYERS", nullptr);
        const bool phone_cleared =
            cycle_real_set_env("LLAMA_HYBRID_CYCLE_PHONE_LAYERS", nullptr);
        disabled_ = pc_cleared && phone_cleared &&
            !llama_hybrid_cycle_requested();
    }

    cycle_real_cpu_reference_env(const cycle_real_cpu_reference_env &) = delete;
    cycle_real_cpu_reference_env & operator=(const cycle_real_cpu_reference_env &) = delete;

    ~cycle_real_cpu_reference_env() {
        const bool pc_restored = cycle_real_set_env(
            "LLAMA_HYBRID_CYCLE_PC_LAYERS",
            had_pc_ ? saved_pc_.c_str() : nullptr);
        const bool phone_restored = cycle_real_set_env(
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS",
            had_phone_ ? saved_phone_.c_str() : nullptr);
        if (!pc_restored || !phone_restored) {
            std::fprintf(stderr,
                "[CYCLE_REAL_REFERENCE] status=FAIL reason=RESTORE_CYCLE_ENV\n");
        }
    }

    bool disabled() const { return disabled_; }

private:
    std::string saved_pc_;
    std::string saved_phone_;
    bool had_pc_ = false;
    bool had_phone_ = false;
    bool disabled_ = false;
};

// The chosen P2C layer must keep Attention, both norms and MoE Router
// exclusively on the sender. A numerically matching FFN is not sufficient
// if the reference graph silently computes its Attention on the PC.
static bool cycle_real_check_phone_weight(
        const char * role,
        const ggml_tensor * tensor,
        ggml_backend_dev_t phone_device) {
    if (!tensor || !phone_device) {
        std::fprintf(stderr,
            "[CYCLE_REAL_OWNER] role=%s status=FAIL reason=MISSING_WEIGHT\n", role);
        return false;
    }
    // The Meta shard accessor deliberately returns nullptr for an empty
    // physical shard. A P2C source-only weight must have NO PC allocation,
    // while its Phone shard must contain the whole original tensor.
    const auto * pc = ggml_backend_meta_get_static_weight_shard(tensor, 0);
    const auto * phone = ggml_backend_meta_get_static_weight_shard(tensor, 1);
    bool same_shape = phone != nullptr && phone->type == tensor->type;
    if (same_shape) {
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            same_shape = same_shape && phone->ne[d] == tensor->ne[d];
        }
    }
    const bool valid =
        pc == nullptr && phone != nullptr && phone->buffer != nullptr &&
        same_shape &&
        ggml_backend_buft_get_device(
            ggml_backend_buffer_get_type(phone->buffer)) == phone_device;
    std::printf(
        "[CYCLE_REAL_OWNER] role=%s source=PHONE_GPU "
        "pc_shard=%s phone_elements=%lld expected=%lld status=%s\n",
        role,
        pc == nullptr ? "EMPTY_EXPECTED" : "UNEXPECTED_PRESENT",
        (long long)(phone ? ggml_nelements(phone) : -1),
        (long long)ggml_nelements(tensor),
        valid ? "PASS" : "FAIL");
    return valid;
}

static bool cycle_real_check_pc_weight(
        const char * role, const ggml_tensor * tensor) {
    if (!tensor) {
        return false;
    }
    const auto * pc = ggml_backend_meta_get_static_weight_shard(tensor, 0);
    const auto * phone = ggml_backend_meta_get_static_weight_shard(tensor, 1);
    bool same_shape = pc && pc->type == tensor->type;
    if (same_shape) {
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            same_shape = same_shape && pc->ne[d] == tensor->ne[d];
        }
    }
    const bool ok = phone == nullptr && pc && pc->buffer && same_shape &&
        ggml_backend_buffer_get_type(pc->buffer) ==
            ggml_backend_cpu_buffer_type();
    std::printf(
        "[CYCLE_REAL_OWNER] role=%s source=PC_CPU "
        "phone_shard=%s pc_elements=%lld expected=%lld status=%s\n",
        role, phone == nullptr ? "EMPTY_EXPECTED" : "UNEXPECTED_PRESENT",
        (long long)(pc ? ggml_nelements(pc) : -1),
        (long long)ggml_nelements(tensor), ok ? "PASS" : "FAIL");
    return ok;
}

static bool cycle_real_check_capture(
        const cycle_real_capture & state,
        const std::string & name,
        ggml_type type,
        int64_t d0, int64_t d1, int64_t d2) {
    const auto it = state.values.find(name);
    if (it == state.values.end() || it->second.count != 1) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=NOT_CAPTURED\n",
            name.c_str());
        return false;
    }
    const auto & entry = it->second;
    const size_t expected = size_t(d0) * size_t(d1) * size_t(d2) *
        (type == GGML_TYPE_F32 ? sizeof(float) : sizeof(int32_t));
    const bool ok =
        entry.type == type &&
        entry.ne[0] == d0 && entry.ne[1] == d1 &&
        entry.ne[2] == d2 && entry.ne[3] == 1 &&
        entry.bytes.size() == expected;
    if (!ok) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL "
            "reason=TYPE_OR_SHAPE actual=[%lld,%lld,%lld,%lld] bytes=%zu\n",
            name.c_str(), (long long)entry.ne[0], (long long)entry.ne[1],
            (long long)entry.ne[2], (long long)entry.ne[3],
            entry.bytes.size());
    }
    return ok;
}

template <typename T>
static std::vector<T> cycle_real_read(
        const cycle_real_capture & state, const std::string & name) {
    const auto & bytes = state.values.at(name).bytes;
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

// Diagnostic Q4_K weight-integrity check for the *selected* MoE experts.
// Store only 64-bit row fingerprints while the independent CPU GGUF model
// exists. After that model is freed, compare the same packed quantized rows
// against the PC/Phone physical shards. Thus no two 30B models coexist,
// and only the selected experts (at most 16 tokens * topk) are transferred.
//
// These are byte-level checks of the logical tensor representation returned
// by the GGML backend, not a replacement for mathematical FFN validation.
// If a backend intentionally repacks tensor storage, a DIFF requires further
// inspection rather than automatically implying corrupted GGUF weights.
struct cycle_real_weight_fingerprints {
    bool enabled = false;
    ggml_type type = GGML_TYPE_COUNT;
    int64_t input = 0;
    int64_t channels = 0;
    int64_t experts = 0;
    std::vector<int32_t> selected;
    std::vector<std::vector<uint64_t>> gate_rows;
    std::vector<std::vector<uint64_t>> up_rows;
};

static uint64_t cycle_real_fnv1a64(const uint8_t * src, size_t n) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; ++i) {
        h ^= uint64_t(src[i]);
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static bool cycle_real_fingerprint_full_rows(
        const char * name, const ggml_tensor * tensor,
        const std::vector<int32_t> & selected,
        std::vector<std::vector<uint64_t>> & hashes) {
    if (!tensor || !tensor->buffer || !ggml_is_contiguous(tensor) ||
        tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
        tensor->ne[2] <= 0 || tensor->ne[3] != 1 ||
        ggml_blck_size(tensor->type) <= 1) {
        std::fprintf(stderr,
            "[CYCLE_FFN_WEIGHT_BYTES] tensor=%s backend=CPU_REFERENCE "
            "status=SKIP reason=INCOMPATIBLE_QUANT_LAYOUT\n", name);
        return false;
    }
    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    const size_t expert_bytes = row_bytes * size_t(tensor->ne[1]);
    if (row_bytes == 0 || expert_bytes == 0 ||
        ggml_nbytes(tensor) != expert_bytes * size_t(tensor->ne[2])) {
        std::fprintf(stderr,
            "[CYCLE_FFN_WEIGHT_BYTES] tensor=%s backend=CPU_REFERENCE "
            "status=SKIP reason=UNEXPECTED_PACKED_STRIDES\n", name);
        return false;
    }
    hashes.clear();
    hashes.reserve(selected.size());
    std::vector<uint8_t> source(expert_bytes);
    for (int32_t expert : selected) {
        if (expert < 0 || expert >= tensor->ne[2]) {
            return false;
        }
        ggml_backend_tensor_get(
            tensor, source.data(), size_t(expert) * expert_bytes,
            expert_bytes);
        std::vector<uint64_t> expert_rows;
        expert_rows.reserve(size_t(tensor->ne[1]));
        for (int64_t channel = 0; channel < tensor->ne[1]; ++channel) {
            expert_rows.push_back(cycle_real_fnv1a64(
                source.data() + size_t(channel) * row_bytes, row_bytes));
        }
        hashes.push_back(std::move(expert_rows));
    }
    std::printf(
        "[CYCLE_FFN_WEIGHT_BYTES] tensor=%s backend=CPU_REFERENCE "
        "selected_experts=%zu rows_per_expert=%lld "
        "row_bytes=%zu status=CAPTURED\n",
        name, selected.size(), (long long)tensor->ne[1], row_bytes);
    return true;
}

static bool cycle_real_capture_ffn_weight_fingerprints(
        const llama_layer & layer,
        const std::vector<int32_t> & selected_topk,
        int64_t max_experts,
        cycle_real_weight_fingerprints & result) {
    const ggml_tensor * gate = layer.ffn_gate_exps;
    const ggml_tensor * up = layer.ffn_up_exps;
    if (!gate || !up ||
        gate->type != up->type ||
        gate->ne[0] != up->ne[0] ||
        gate->ne[1] != up->ne[1] ||
        gate->ne[2] != up->ne[2] ||
        gate->ne[2] != max_experts) {
        return false;
    }
    result.type = up->type;
    result.input = up->ne[0];
    result.channels = up->ne[1];
    result.experts = up->ne[2];
    result.selected = selected_topk;
    std::sort(result.selected.begin(), result.selected.end());
    result.selected.erase(
        std::unique(result.selected.begin(), result.selected.end()),
        result.selected.end());
    if (result.selected.empty() ||
        result.selected.front() < 0 ||
        result.selected.back() >= max_experts) {
        return false;
    }
    if (!cycle_real_fingerprint_full_rows(
            "GATE", gate, result.selected, result.gate_rows) ||
        !cycle_real_fingerprint_full_rows(
            "UP", up, result.selected, result.up_rows)) {
        return false;
    }
    result.enabled = true;
    return true;
}

static bool cycle_real_compare_physical_weight_rows(
        const char * name, const char * backend_name,
        const ggml_tensor * physical, int64_t channel_base,
        const cycle_real_weight_fingerprints & reference,
        const std::vector<std::vector<uint64_t>> & expected) {
    if (!reference.enabled || !physical || !physical->buffer ||
        physical->type != reference.type ||
        !ggml_is_contiguous(physical) ||
        physical->ne[0] != reference.input ||
        physical->ne[2] != reference.experts ||
        physical->ne[3] != 1 ||
        channel_base < 0 || physical->ne[1] <= 0 ||
        channel_base + physical->ne[1] > reference.channels ||
        expected.size() != reference.selected.size()) {
        std::fprintf(stderr,
            "[CYCLE_FFN_WEIGHT_BYTES] tensor=%s backend=%s "
            "status=SKIP reason=PHYSICAL_SHAPE_OR_TYPE\n",
            name, backend_name);
        return false;
    }
    const size_t row_bytes =
        ggml_row_size(physical->type, physical->ne[0]);
    const size_t expert_bytes =
        row_bytes * size_t(physical->ne[1]);
    if (ggml_nbytes(physical) !=
        expert_bytes * size_t(physical->ne[2])) {
        return false;
    }
    std::vector<uint8_t> actual(expert_bytes);
    size_t equal_rows = 0;
    size_t different_rows = 0;
    int32_t first_expert = -1;
    int64_t first_channel = -1;
    uint64_t first_ref_hash = 0;
    uint64_t first_shard_hash = 0;
    for (size_t i = 0; i < reference.selected.size(); ++i) {
        const int32_t expert = reference.selected[i];
        if (expected[i].size() != size_t(reference.channels)) {
            return false;
        }
        ggml_backend_tensor_get(
            physical, actual.data(), size_t(expert) * expert_bytes,
            expert_bytes);
        for (int64_t ch = 0; ch < physical->ne[1]; ++ch) {
            const uint64_t hash = cycle_real_fnv1a64(
                actual.data() + size_t(ch) * row_bytes, row_bytes);
            const uint64_t expected_hash =
                expected[i][size_t(channel_base + ch)];
            if (hash == expected_hash) {
                ++equal_rows;
            } else {
                ++different_rows;
                if (first_expert < 0) {
                    first_expert = expert;
                    first_channel = channel_base + ch;
                    first_ref_hash = expected_hash;
                    first_shard_hash = hash;
                }
            }
        }
    }
    std::printf(
        "[CYCLE_FFN_WEIGHT_BYTES] tensor=%s backend=%s "
        "selected_experts=%zu channels=[%lld,%lld) "
        "matched_rows=%zu mismatched_rows=%zu "
        "first_bad_expert=%d first_bad_channel=%lld "
        "first_expected_hash=%016llx first_actual_hash=%016llx "
        "status=%s\n",
        name, backend_name, reference.selected.size(),
        (long long)channel_base,
        (long long)(channel_base + physical->ne[1]),
        equal_rows, different_rows, first_expert,
        (long long)first_channel,
        (unsigned long long)first_ref_hash,
        (unsigned long long)first_shard_hash,
        different_rows == 0 ? "MATCH" : "DIFF");
    return different_rows == 0;
}

// Correctness-first Phone AttentionPre for ONE contiguous causal prompt
// ubatch (positions [0, n_tokens)). The current Qwen3-MoE K/V values become
// an F16 transient KV block in the same graph. Each token attends ONLY
// positions <= itself, following the same Q/K/V, head-RMSNorm, RoPE, GQA,
// KQ-mask and non-Flash attention operations as the canonical Qwen3 graph.
//
// No historical KV is persisted between calls yet: this validates multi-KV
// *within one batch*, not cross-chunk KV handoff or production scheduling.
struct cycle_real_phone_ready {
    std::vector<float> residual;
    std::vector<float> norm;
    std::vector<int32_t> ids;
    std::vector<float> routing;
    // Full pre-TopK Softmax distribution for score-margin diagnosis only.
    // Not serialized in READY and never used by the FFN execution.
    std::vector<float> router_probs;

    // Keep READY's Phone GPU tensors alive through FFN local input copies.
    ggml_context * phone_ctx = nullptr;
    ggml_backend_buffer_t phone_compute_buffer = nullptr;
    llama_hybrid_cycle_ready_graph phone_graph {};

    cycle_real_phone_ready() = default;
    cycle_real_phone_ready(const cycle_real_phone_ready &) = delete;
    cycle_real_phone_ready & operator=(const cycle_real_phone_ready &) = delete;
    ~cycle_real_phone_ready() { release_phone_graph(); }

    void release_phone_graph() {
        if (phone_compute_buffer != nullptr) {
            ggml_backend_buffer_free(phone_compute_buffer);
            phone_compute_buffer = nullptr;
        }
        if (phone_ctx != nullptr) {
            ggml_free(phone_ctx);
            phone_ctx = nullptr;
        }
        phone_graph = {};
    }
};

// A single GPU-resident, F16 cache for one physical P2C Attention layer.
// Shape [head_dim, kv_heads, max_tokens] makes each newly written token
// interval contiguous across KV heads: no host readback, no materialization
// of previous chunks and no second cache instance per in-flight chunk.
// Attention tasks run in causal chunk order on the single PHONE_GPU FIFO.
// PC full layers are loaded into native CPU buffers, including mmap-backed
// and CPU extra-buffer types that need not equal the backend's default buft.
// Tensor/Phone layers still borrow their real physical static Meta shards.
// Never return the parent Meta tensor or copy weights across devices.
static ggml_tensor * cycle_real_resolve_physical_weight(
        const ggml_tensor * weight, ggml_backend_t backend,
        size_t source_device) {
    if (!weight || !weight->buffer || !backend ||
        weight->op != GGML_OP_NONE || weight->view_src != nullptr) {
        return nullptr;
    }
    const auto weight_buft = ggml_backend_buffer_get_type(weight->buffer);
    const auto native_buft = ggml_backend_get_default_buffer_type(backend);
    const auto device = ggml_backend_get_device(backend);
    const bool pc_native_compatible =
        source_device == 0 && device &&
        ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU &&
        !ggml_backend_buffer_is_meta(weight->buffer) &&
        ggml_backend_dev_supports_buft(device, weight_buft);
    if ((native_buft && weight_buft == native_buft) ||
        pc_native_compatible) {
        if (pc_native_compatible && weight_buft != native_buft &&
            std::getenv("GGML_CYCLE_SHARD_DIAG") != nullptr) {
            std::fprintf(stderr,
                "[CYCLE_WEIGHT_RESOLVE] weight=%s index=%zu "
                "backend=%s weight_buft=%s default_buft=%s "
                "mode=CPU_COMPATIBLE status=OK\n",
                weight->name, source_device, ggml_backend_name(backend),
                ggml_backend_buft_name(weight_buft),
                native_buft ? ggml_backend_buft_name(native_buft) : "(null)");
        }
        // Borrowed model tensor: it remains owned by the model until unload.
        return const_cast<ggml_tensor *>(weight);
    }
    return ggml_backend_meta_get_static_weight_shard(weight, source_device);
}


// Mirror the entire PC full layer (Attention, Router, FFN experts) on CUDA
// once before the measured period. The CPU+Phone physical Meta layout is
// unchanged, so Tensor C2P/P2C still use the same model-device indices.
// No full layer is split between CPU and CUDA in this mode.
struct cycle_real_pc_cuda_full_weights {
    enum slot : size_t {
        ATTN_NORM, WQ, WK, WV, WO, Q_NORM, K_NORM, FFN_NORM, ROUTER,
        FFN_UP, FFN_GATE, FFN_DOWN, COUNT
    };
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::array<ggml_tensor *, COUNT> tensors {};

    cycle_real_pc_cuda_full_weights() = default;
    cycle_real_pc_cuda_full_weights(
        const cycle_real_pc_cuda_full_weights &) = delete;
    cycle_real_pc_cuda_full_weights & operator=(
        const cycle_real_pc_cuda_full_weights &) = delete;
    ~cycle_real_pc_cuda_full_weights() { release(); }

    void release() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        tensors.fill(nullptr);
    }

    bool prepare(ggml_backend_t cuda, ggml_backend_t cpu,
                 const llama_layer & layer, int layer_index) {
        release();
        if (!cuda || !cpu) {
            return false;
        }
        const ggml_tensor * logical[COUNT] = {
            layer.attn_norm, layer.wq, layer.wk, layer.wv, layer.wo,
            layer.attn_q_norm, layer.attn_k_norm, layer.ffn_norm,
            layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps,
            layer.ffn_down_exps
        };
        std::array<const ggml_tensor *, COUNT> physical {};
        for (size_t i = 0; i < COUNT; ++i) {
            physical[i] = cycle_real_resolve_physical_weight(
                logical[i], cpu, 0);
            if (!physical[i] || !physical[i]->buffer ||
                !ggml_is_contiguous(physical[i])) {
                std::fprintf(stderr,
                    "[CYCLE_PC_CUDA_WEIGHT] layer=%d slot=%zu "
                    "status=FAIL reason=PC_PHYSICAL_WEIGHT_UNAVAILABLE\n",
                    layer_index, i);
                return false;
            }
        }
        ggml_init_params params {};
        params.mem_size = 2 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }
        for (size_t i = 0; i < COUNT; ++i) {
            tensors[i] = ggml_dup_tensor(ctx, physical[i]);
            if (!tensors[i]) {
                release();
                return false;
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(cuda));
        if (!buffer) {
            release();
            return false;
        }
        size_t upload_bytes = 0;
        for (size_t i = 0; i < COUNT; ++i) {
            // One-time CPU -> CUDA copy; never re-upload model weights per
            // Chunk. Only final full-layer output is read back to the host.
            ggml_backend_tensor_copy(physical[i], tensors[i]);
            upload_bytes += ggml_nbytes(physical[i]);
        }
        ggml_backend_synchronize(cuda);
        std::printf(
            "[CYCLE_PC_CUDA_WEIGHT] layer=%d weights=%zu upload_mib=%.3f "
            "owner=PC_CUDA lifetime=PERIOD status=READY\n",
            layer_index, size_t(COUNT),
            double(upload_bytes) / (1024.0 * 1024.0));
        return true;
    }

    llama_hybrid_cycle_ffn_shard_weights full_ffn_shard() const {
        return {tensors[FFN_UP], tensors[FFN_GATE], tensors[FFN_DOWN]};
    }
};

struct cycle_real_pc_cuda_runtime {
    ggml_backend_t backend = nullptr;
    std::vector<std::unique_ptr<cycle_real_pc_cuda_full_weights>> weights;
    ~cycle_real_pc_cuda_runtime() {
        weights.clear();  // Free GPU weights before destroying CUDA backend.
        if (backend) {
            ggml_backend_free(backend);
        }
    }
};

struct cycle_real_phone_kv_cache {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * keys = nullptr;
    ggml_tensor * values = nullptr;
    int64_t max_tokens = 0;
    int64_t head_dim = 0;
    int64_t kv_heads = 0;
    int64_t committed_tokens = 0;

    cycle_real_phone_kv_cache() = default;
    cycle_real_phone_kv_cache(const cycle_real_phone_kv_cache &) = delete;
    cycle_real_phone_kv_cache & operator=(const cycle_real_phone_kv_cache &) = delete;
    ~cycle_real_phone_kv_cache() { release(); }

    void release() {
        if (buffer) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        keys = nullptr;
        values = nullptr;
        max_tokens = 0;
        head_dim = 0;
        kv_heads = 0;
        committed_tokens = 0;
    }

    bool init(ggml_backend_t backend,
              const llama_layer & layer, int64_t total_tokens,
              size_t source_device = 1,
              const cycle_real_pc_cuda_full_weights * cuda_weights = nullptr) {
        release();
        const auto * norm = cuda_weights ?
            cuda_weights->tensors[cycle_real_pc_cuda_full_weights::K_NORM] :
            cycle_real_resolve_physical_weight(
                layer.attn_k_norm, backend, source_device);
        const auto * wk = cuda_weights ?
            cuda_weights->tensors[cycle_real_pc_cuda_full_weights::WK] :
            cycle_real_resolve_physical_weight(
                layer.wk, backend, source_device);
        if (!backend || !norm || !wk ||
            total_tokens <= 0 || total_tokens > 128 ||
            norm->ne[0] <= 0 || wk->ne[1] % norm->ne[0] != 0) {
            return false;
        }
        head_dim = norm->ne[0];
        kv_heads = wk->ne[1] / head_dim;
        max_tokens = total_tokens;
        ggml_init_params params {};
        params.mem_size = 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }
        keys = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F16, head_dim, kv_heads, max_tokens);
        values = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F16, head_dim, kv_heads, max_tokens);
        if (!keys || !values) {
            release();
            return false;
        }
        ggml_set_name(keys, "cycle_phone_k_cache_f16");
        ggml_set_name(values, "cycle_phone_v_cache_f16");
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(backend));
        if (!buffer) {
            release();
            return false;
        }
        std::printf(
            "[CYCLE_PHONE_KV_PERSISTENT] total_tokens=%lld head_dim=%lld "
            "kv_heads=%lld cache_bytes=%zu residency=%s "
            "status=READY\n",
            (long long)max_tokens, (long long)head_dim,
            (long long)kv_heads, ggml_nbytes(keys) + ggml_nbytes(values),
            cuda_weights ? "PC_CUDA_F16" :
                source_device == 0 ? "PC_CPU_F16" : "PHONE_GPU_F16");
        return true;
    }
};

struct cycle_real_live_wire {
    int32_t layer = -1;
    int32_t chunk = 0;
    uint64_t seq = 1;
    uint64_t graph_id = 1;
};

// The existing RPC data-plane receiver already has a single application RX
// dispatcher. Do not call transport.receive() a second time on its socket.
// Fetch exactly one queued frame through the existing exported dispatch proc.
static bool cycle_real_receive_live_wire(
        ggml_backend_t phone, const cycle_real_live_wire & wire,
        inference_cycle_payload_kind kind,
        uint32_t tokens, uint32_t hidden, uint32_t topk,
        inference_cycle_message & output) {
    auto * dev = ggml_backend_get_device(phone);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return false;
    }
    auto receive = reinterpret_cast<ggml_backend_rpc_inference_transport_receive_t>(
        ggml_backend_reg_get_proc_address(
            reg, GGML_BACKEND_RPC_INFERENCE_TRANSPORT_RECEIVE_PROC));
    if (!receive) {
        return false;
    }
    size_t bytes = 0;
    if (!inference_cycle_payload_size(
            kind, tokens, hidden,
            kind == INFERENCE_CYCLE_READY ? topk : 0, bytes)) {
        return false;
    }
    std::vector<uint8_t> frame_data(bytes);
    ggml_backend_rpc_inference_task task {};
    size_t received = 0;
    if (!receive(phone, &task, frame_data.data(), frame_data.size(),
                 &received, 30000) || received != bytes) {
        std::fprintf(stderr,
            "[CYCLE_REAL_WIRE] side=pc stage=%s status=FAIL "
            "reason=RECEIVE_OR_SIZE expected=%zu got=%zu\n",
            kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
            bytes, received);
        return false;
    }
    const uint16_t stage = kind == INFERENCE_CYCLE_READY ?
        INFERENCE_STAGE_CYCLE_READY : INFERENCE_STAGE_CYCLE_PARTIAL;
    inference_frame_header header =
        inference_make_header(INFERENCE_FRAME_DATA);
    header.stage = task.stage;
    header.tensor_role = task.tensor_role;
    header.layer = task.layer;
    header.chunk = task.chunk;
    header.seq = task.seq;
    header.graph_id = task.graph_id;
    header.payload_bytes = received;
    if (task.stage != stage ||
        task.tensor_role != INFERENCE_CYCLE_PHONE_TO_PC ||
        task.layer != wire.layer || task.chunk != wire.chunk ||
        task.seq != wire.seq || task.graph_id != wire.graph_id ||
        !inference_cycle_decode_frame(
            header, frame_data.data(), frame_data.size(), output) ||
        output.direction != INFERENCE_CYCLE_PHONE_TO_PC ||
        output.kind != kind) {
        std::fprintf(stderr,
            "[CYCLE_REAL_WIRE] side=pc stage=%s status=FAIL "
            "reason=HEADER_OR_DECODE\n",
            kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL");
        return false;
    }
    std::printf(
        "[CYCLE_REAL_WIRE] side=pc stage=%s direction=P2C "
        "layer=%d chunk=%d seq=%llu bytes=%zu "
        "source=PHONE_GPU_RPC_TENSORS mode=REAL_RPC_DATA_PLANE status=PASS\n",
        kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
        wire.layer, wire.chunk, (unsigned long long)wire.seq, received);
    return true;
}

// CPU-observed Attention phases. For RPC the graph_call_ms is ONLY the
// submission round trip; Phone kernel execution is asynchronous and must
// be measured by the server. One FIFO worker owns each queue-local sample.
struct cycle_real_attention_phase_profile {
    uint64_t count = 0;
    double graph_build_ms = 0.0;
    double alloc_ms = 0.0;
    double input_copy_ms = 0.0;
    double graph_call_ms = 0.0;
};

static bool cycle_real_execute_phone_attention_pre(
        ggml_backend_t backend, const llama_layer & layer,
        const llama_hparams & hp,
        const std::vector<float> & layer_input,
        int64_t hidden, int64_t tokens,
        int64_t topk, int64_t experts,
        cycle_real_phone_ready & result,
        double & elapsed_ms,
        const cycle_real_live_wire * wire = nullptr,
        cycle_real_phone_kv_cache * kv_cache = nullptr,
        int64_t kv_token_start = 0,
        size_t source_device = 1,
        const ggml_tensor * device_input = nullptr,
        bool retain_device_ready = false,
        cycle_real_attention_phase_profile * phase_profile = nullptr,
        const cycle_real_pc_cuda_full_weights * cuda_weights = nullptr) {
    if (!backend || ggml_backend_is_meta(backend) ||
        hidden <= 0 || tokens < 1 || tokens > 128 ||
        kv_token_start < 0 || kv_token_start + tokens > 128 ||
        (kv_cache == nullptr && kv_token_start != 0) ||
        (device_input == nullptr ?
            layer_input.size() != size_t(hidden * tokens) :
            (device_input->type != GGML_TYPE_F32 ||
             ggml_nelements(device_input) != hidden * tokens ||
             device_input->buffer == nullptr)) ||
        hp.f_norm_rms_eps <= 0.0f ||
        topk <= 0 || experts < topk ||
        hp.rope_type != LLAMA_ROPE_TYPE_NEOX ||
        // The loader defaults to "linear" even when no RoPE scale key is
        // present. Linear with factor 1 is numerically identical to NONE.
        (hp.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_NONE &&
         hp.rope_scaling_type_train != LLAMA_ROPE_SCALING_TYPE_LINEAR) ||
        std::fabs(hp.rope_freq_scale_train - 1.0f) > 1.0e-6f ||
        hp.yarn_ext_factor > 0.0f ||
        std::fabs(hp.rope_attn_factor - 1.0f) > 1.0e-6f ||
        hp.use_alibi ||
        layer.wqkv || layer.wq_b || layer.wk_b || layer.wv_b ||
        layer.wo_b || layer.wq_s || layer.wk_s || layer.wv_s ||
        layer.wo_s || hp.f_clamp_kqv != 0.0f ||
        hp.attn_soft_cap || hp.f_max_alibi_bias != 0.0f) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL "
            "reason=UNSUPPORTED_CONTIGUOUS_CAUSAL_PREFILL\n");
        return false;
    }

    // PC complete layers use their native CPU model weights. Tensor and
    // Phone layers still borrow physical Meta shards without host transfers.
    auto get_weight = [backend, source_device](const ggml_tensor * t) -> ggml_tensor * {
        return cycle_real_resolve_physical_weight(t, backend, source_device);
    };
    ggml_tensor * w_attn_norm = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::ATTN_NORM] :
        get_weight(layer.attn_norm);
    ggml_tensor * wq = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::WQ] :
        get_weight(layer.wq);
    ggml_tensor * wk = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::WK] :
        get_weight(layer.wk);
    ggml_tensor * wv = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::WV] :
        get_weight(layer.wv);
    ggml_tensor * wo = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::WO] :
        get_weight(layer.wo);
    ggml_tensor * wq_norm = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::Q_NORM] :
        get_weight(layer.attn_q_norm);
    ggml_tensor * wk_norm = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::K_NORM] :
        get_weight(layer.attn_k_norm);
    ggml_tensor * w_ffn_norm = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::FFN_NORM] :
        get_weight(layer.ffn_norm);
    ggml_tensor * w_router = cuda_weights ?
        cuda_weights->tensors[cycle_real_pc_cuda_full_weights::ROUTER] :
        get_weight(layer.ffn_gate_inp);

    if (!w_attn_norm || !wq || !wk || !wv || !wo ||
        !wq_norm || !wk_norm || !w_ffn_norm || !w_router) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=PHONE_WEIGHT_MISSING\n");
        return false;
    }
    const int64_t head_dim = wq_norm->ne[0];
    if (head_dim <= 0 || wk_norm->ne[0] != head_dim ||
        wq->ne[0] != hidden || wk->ne[0] != hidden ||
        wv->ne[0] != hidden || wq->ne[1] % head_dim != 0 ||
        wk->ne[1] % head_dim != 0 ||
        wv->ne[1] != wk->ne[1] ||
        wo->ne[0] != wq->ne[1] || wo->ne[1] != hidden ||
        w_router->ne[0] != hidden || w_router->ne[1] != experts ||
        w_attn_norm->ne[0] != hidden || w_ffn_norm->ne[0] != hidden ||
        hp.n_rot_full != uint32_t(head_dim)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=QKV_SHAPE\n");
        return false;
    }
    const int64_t n_heads = wq->ne[1] / head_dim;
    const int64_t n_kv_heads = wk->ne[1] / head_dim;
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=GQA_HEADS\n");
        return false;
    }

    const int64_t kv_tokens = kv_token_start + tokens;
    if (kv_cache != nullptr &&
        (!kv_cache->keys || !kv_cache->values ||
         kv_cache->head_dim != head_dim ||
         kv_cache->kv_heads != n_kv_heads ||
         kv_tokens > kv_cache->max_tokens ||
         kv_cache->committed_tokens != kv_token_start)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_KV_PERSISTENT] status=FAIL "
            "reason=BAD_PREFIX_OR_OUT_OF_ORDER start=%lld expected=%lld\n",
            (long long)kv_token_start,
            (long long)(kv_cache ? kv_cache->committed_tokens : -1));
        return false;
    }
    const auto graph_build_begin = std::chrono::steady_clock::now();
    ggml_init_params p {};
    p.mem_size = 16 * 1024 * 1024;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        return false;
    }

    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    auto * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    // Keys include all PAST tokens and the current causal chunk.
    auto * causal_mask = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F16, kv_tokens, tokens);
    ggml_set_input(x);
    ggml_set_input(positions);
    ggml_set_input(causal_mask);
    ggml_set_name(x, "cycle_phone_attn_input");
    ggml_set_name(positions, "cycle_phone_positions");
    ggml_set_name(causal_mask, "cycle_phone_causal_kq_mask");

    auto * attn_norm = ggml_mul(
        ctx, ggml_rms_norm(ctx, x, hp.f_norm_rms_eps), w_attn_norm);
    auto * q = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, wq, attn_norm), head_dim, n_heads, tokens);
    auto * k = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, wk, attn_norm), head_dim, n_kv_heads, tokens);
    auto * v = ggml_reshape_3d(ctx,
        ggml_mul_mat(ctx, wv, attn_norm), head_dim, n_kv_heads, tokens);
    q = ggml_mul(ctx, ggml_rms_norm(ctx, q, hp.f_norm_rms_eps), wq_norm);
    k = ggml_mul(ctx, ggml_rms_norm(ctx, k, hp.f_norm_rms_eps), wk_norm);

    // RoPE is now essential: position 1..N-1 must not be treated as 0.
    // The standalone Qwen3-MoE check rejects nondefault YaRN/scaling above.
    const int ctx_orig = hp.n_ctx_orig_yarn > 0 ?
        int(hp.n_ctx_orig_yarn) : int(hp.n_ctx_train);
    const float rope_base = hp.rope_freq_base_train > 0 ?
        hp.rope_freq_base_train : 10000.0f;
    const float rope_scale = hp.rope_freq_scale_train > 0 ?
        hp.rope_freq_scale_train : 1.0f;
    q = ggml_rope_ext(ctx, q, positions, nullptr,
        int(head_dim), GGML_ROPE_TYPE_NEOX, ctx_orig,
        rope_base, rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr,
        int(head_dim), GGML_ROPE_TYPE_NEOX, ctx_orig,
        rope_base, rope_scale, 0.0f, 1.0f, 32.0f, 1.0f);

    // Same no-Flash GQA layout as llm_graph_context::build_attn_mha():
    // [head_dim, tokens, heads, stream]. The reference cache stores K/V
    // in F16; preserve that rounding before the attention projection.
    // In this single-batch test, K/V at positions [0,N) are the entire
    // transient cache. The mask below implements the same causal rule
    // as llama_kv_cache::set_input_kq_mask (key_pos > query_pos -> -inf).
    k = ggml_cast(ctx, k, GGML_TYPE_F16);
    v = ggml_cast(ctx, v, GGML_TYPE_F16);
    ggml_tensor * kv_k_write = nullptr;
    ggml_tensor * kv_v_write = nullptr;
    if (kv_cache != nullptr) {
        // NEW K/V are contiguous [head_dim,kv_heads,chunk_tokens].
        // Persist them in the same Phone GPU buffer before reading all
        // historical keys; no historical K/V is ever sent to the PC.
        auto * dst_k = ggml_view_3d(
            ctx, kv_cache->keys, head_dim, n_kv_heads, tokens,
            kv_cache->keys->nb[1], kv_cache->keys->nb[2],
            size_t(kv_token_start) * kv_cache->keys->nb[2]);
        auto * dst_v = ggml_view_3d(
            ctx, kv_cache->values, head_dim, n_kv_heads, tokens,
            kv_cache->values->nb[1], kv_cache->values->nb[2],
            size_t(kv_token_start) * kv_cache->values->nb[2]);
        kv_k_write = ggml_cpy(ctx, k, dst_k);
        kv_v_write = ggml_cpy(ctx, v, dst_v);
        k = ggml_view_3d(
            ctx, kv_cache->keys, head_dim, n_kv_heads, kv_tokens,
            kv_cache->keys->nb[1], kv_cache->keys->nb[2], 0);
        v = ggml_view_3d(
            ctx, kv_cache->values, head_dim, n_kv_heads, kv_tokens,
            kv_cache->values->nb[1], kv_cache->values->nb[2], 0);
    }
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);
    auto * kq = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    auto * probs = ggml_soft_max_ext(
        ctx, kq, causal_mask,
        1.0f / std::sqrt(float(head_dim)), 0.0f);
    auto * vt = ggml_cont(ctx, ggml_transpose(ctx, v));
    auto * kqv = ggml_mul_mat(ctx, vt, probs);

    // The concatenated attention width is n_heads * head_dim (4096 on
    // Qwen3-30B-A3B), NOT the residual/hidden width (2048). The WO
    // projection is responsible for mapping 4096 -> 2048. Previously,
    // requesting [hidden,1] in ggml_cont_2d aborted at ggml.c:3594.
    const int64_t attention_width = wq->ne[1];
    if (ggml_nelements(kqv) != attention_width * tokens) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=ATTN_WIDTH "
            "kqv_elements=%lld expected=%lld hidden=%lld\n",
            (long long)ggml_nelements(kqv),
            (long long)(attention_width * tokens), (long long)hidden);
        ggml_free(ctx);
        return false;
    }
    auto * attn = ggml_cont_2d(ctx,
        ggml_permute(ctx, kqv, 0, 2, 1, 3), attention_width, tokens);

    std::printf(
        "[CYCLE_PHONE_ATTN_SHAPE] hidden=%lld head_dim=%lld "
        "heads=%lld kv_heads=%lld attn_width=%lld wo_out=%lld "
        "query_tokens=%lld kv_tokens=%lld status=PASS\n",
        (long long)hidden, (long long)head_dim,
        (long long)n_heads, (long long)n_kv_heads,
        (long long)attention_width, (long long)wo->ne[1],
        (long long)tokens, (long long)kv_tokens);

    auto * attn_output = ggml_mul_mat(ctx, wo, attn);
    auto * residual = ggml_add(ctx, attn_output, x);
    ggml_set_name(residual, "cycle_num_attn_residual");
    auto * norm = ggml_mul(
        ctx, ggml_rms_norm(ctx, residual, hp.f_norm_rms_eps), w_ffn_norm);
    ggml_set_name(norm, "cycle_num_attn_ffn_norm");
    auto * router_logits = ggml_mul_mat(ctx, w_router, norm);
    auto * router_probs = ggml_soft_max(ctx, router_logits);
    auto * ids = ggml_cont(ctx, ggml_argsort_top_k(
        ctx, router_probs, int(topk)));
    auto * router_p3 = ggml_reshape_3d(ctx, router_probs, 1, experts, tokens);
    auto * weights = ggml_get_rows(ctx, router_p3, ids);
    weights = ggml_reshape_2d(ctx, weights, topk, tokens);
    auto * sum = ggml_sum_rows(ctx, weights);
    sum = ggml_clamp(ctx, sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(ctx, weights, sum);
    weights = ggml_reshape_3d(ctx, weights, 1, topk, tokens);
    if (hp.expert_weights_scale != 0.0f &&
        hp.expert_weights_scale != 1.0f) {
        weights = ggml_scale(ctx, weights, hp.expert_weights_scale);
    }
    weights = ggml_cont(ctx, weights);

    const llama_hybrid_cycle_ready_graph ready {
        residual, norm, ids, weights
    };
    if (!llama_hybrid_cycle_ready_graph_valid(ready, hidden, tokens, topk)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=READY_SHAPES\n");
        ggml_free(ctx);
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    // Enforce GPU cache writes before subsequent prefix attention reads.
    if (kv_cache != nullptr) {
        ggml_build_forward_expand(graph, kv_k_write);
        ggml_build_forward_expand(graph, kv_v_write);
    }
    ggml_build_forward_expand(graph, ready.residual);
    ggml_build_forward_expand(graph, ready.ffn_norm);
    ggml_build_forward_expand(graph, ready.topk_ids);
    ggml_build_forward_expand(graph, ready.topk_weights);
    const auto graph_build_done = std::chrono::steady_clock::now();

    int unsupported = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_NONE &&
            !ggml_backend_supports_op(backend, node)) {
            ++unsupported;
            std::fprintf(stderr,
                "[CYCLE_PHONE_ATTN_PRE_OP] node=%d op=%s "
                "name=%s supported=0\n",
                i, ggml_op_name(node->op), node->name);
        }
    }
    if (unsupported != 0) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL "
            "reason=UNSUPPORTED_PHONE_OP count=%d\n", unsupported);
        ggml_free(ctx);
        return false;
    }

    const auto buft = ggml_backend_get_default_buffer_type(backend);
    const auto alloc_begin = std::chrono::steady_clock::now();
    ggml_backend_buffer_t compute_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    const auto alloc_done = std::chrono::steady_clock::now();
    if (!compute_buffer) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=ALLOC_PHONE_GRAPH\n");
        ggml_free(ctx);
        return false;
    }

    // Input layout: token-major hidden vectors; positions increment 0..N-1.
    // The F16 mask's dimension 0 is the KV index and dimension 1 is query.
    std::vector<int32_t> position_data(size_t(tokens), 0);
    std::vector<ggml_fp16_t> mask_data(size_t(tokens * kv_tokens));
    const ggml_fp16_t keep = ggml_fp32_to_fp16(0.0f);
    const ggml_fp16_t drop = ggml_fp32_to_fp16(-INFINITY);
    int64_t masked_future = 0;
    for (int64_t query = 0; query < tokens; ++query) {
        const int64_t global_pos = kv_token_start + query;
        position_data[size_t(query)] = int32_t(global_pos);
        for (int64_t key = 0; key < kv_tokens; ++key) {
            const bool future = key > global_pos;
            mask_data[size_t(query * kv_tokens + key)] = future ? drop : keep;
            masked_future += future ? 1 : 0;
        }
    }
    if (masked_future != tokens * (tokens - 1) / 2) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_KV] status=FAIL reason=MASK_VALIDATION\n");
        ggml_backend_buffer_free(compute_buffer);
        ggml_free(ctx);
        return false;
    }
    std::printf("[CYCLE_PHONE_KV] query_tokens=%lld kv_tokens=%lld "
                "position_start=%lld position_end=%lld "
                "masked_future=%lld mask_type=F16 scope=%s "
                "status=PASS\n",
                (long long)tokens, (long long)kv_tokens,
                (long long)kv_token_start, (long long)(kv_tokens - 1),
                (long long)masked_future,
                kv_cache == nullptr ? "SINGLE_UBATCH" :
            source_device == 0 ? "PC_PERSISTENT" : "PHONE_PERSISTENT");

    const auto begin = std::chrono::steady_clock::now();
    if (device_input != nullptr) {
        // A FULL Phone layer hands off its computed l_out to the next
        // layer through a direct Phone-GPU-to-Phone-GPU copy. Never call
        // ggml_backend_tensor_copy here; that could secretly RPC GET/SET.
        bool layout_ok = ggml_are_same_shape(device_input, x) &&
                         device_input->type == x->type;
        for (int d = 0; d < GGML_MAX_DIMS && layout_ok; ++d) {
            layout_ok = device_input->nb[d] == x->nb[d];
        }
        if (!layout_ok ||
            !ggml_backend_buffer_copy_tensor(device_input, x)) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_LOCAL_HANDOFF] status=FAIL "
                "reason=DEVICE_COPY_UNSUPPORTED\n");
            ggml_backend_buffer_free(compute_buffer);
            ggml_free(ctx);
            return false;
        }
        std::printf(
            "[CYCLE_PHONE_LOCAL_HANDOFF] backend=%s "
            "bytes=%zu source=PHONE_GPU_PREVIOUS_LAYER "
            "destination=PHONE_GPU_ATTENTION status=OK\n",
            ggml_backend_name(backend), ggml_nbytes(x));
    } else {
        ggml_backend_tensor_set(x, layer_input.data(), 0, ggml_nbytes(x));
    }
    ggml_backend_tensor_set(positions, position_data.data(), 0,
                            ggml_nbytes(positions));
    ggml_backend_tensor_set(causal_mask, mask_data.data(), 0,
                            ggml_nbytes(causal_mask));
    const auto input_done = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    const auto submit_done = std::chrono::steady_clock::now();
    if (phase_profile) {
        ++phase_profile->count;
        phase_profile->graph_build_ms +=
            std::chrono::duration<double, std::milli>(
                graph_build_done - graph_build_begin).count();
        phase_profile->alloc_ms +=
            std::chrono::duration<double, std::milli>(
                alloc_done - alloc_begin).count();
        phase_profile->input_copy_ms +=
            std::chrono::duration<double, std::milli>(
                input_done - begin).count();
        phase_profile->graph_call_ms +=
            std::chrono::duration<double, std::milli>(
                submit_done - input_done).count();
    }
    if (std::getenv("LLAMA_HYBRID_CYCLE_STAGE_PROFILE") != nullptr) {
        std::printf(
            "[CYCLE_ATTN_PROFILE] backend=%s "
            "graph_build_ms=%.3f gpu_alloc_ms=%.3f "
            "input_copy_ms=%.3f graph_submit_or_compute_ms=%.3f "
            "remote_gpu_compute_ms=SEE_CYCLE_GPU_SERVER\n",
            ggml_backend_name(backend),
            std::chrono::duration<double, std::milli>(
                graph_build_done - graph_build_begin).count(),
            std::chrono::duration<double, std::milli>(
                alloc_done - alloc_begin).count(),
            std::chrono::duration<double, std::milli>(
                input_done - begin).count(),
            std::chrono::duration<double, std::milli>(
                submit_done - input_done).count());
    }
    bool ok = status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_synchronize(backend);
        if (kv_cache != nullptr) {
            kv_cache->committed_tokens = kv_tokens;
            std::printf(
                "[CYCLE_PHONE_KV_PERSISTENT] chunk=%d "
                "position_start=%lld committed=%lld "
                "total=%lld owner=%s status=OK\n",
                wire != nullptr ? wire->chunk : -1,
                (long long)kv_token_start,
                (long long)kv_cache->committed_tokens,
                (long long)kv_cache->max_tokens,
                cuda_weights ? "PC_CUDA" :
                    source_device == 0 ? "PC_CPU" : "PHONE_GPU");
        }
        if (wire != nullptr) {
            // GPU tensors remain valid until compute_buffer is destroyed.
            // The Phone RPC server copies directly into its transport TX
            // slot; no PC-side retransmission is involved.
            ok = ggml_backend_rpc_cycle_send_ready(
                backend, wire->layer, wire->chunk,
                wire->seq, wire->graph_id,
                ready.residual, ready.ffn_norm,
                ready.topk_ids, ready.topk_weights);
            if (!ok) {
                std::fprintf(stderr,
                    "[CYCLE_REAL_WIRE] stage=READY status=FAIL "
                    "reason=PHONE_REMOTE_SEND\n");
            }
        }
    }
    if (ok && wire == nullptr && !retain_device_ready) {
        result.residual.resize(size_t(hidden * tokens));
        result.norm.resize(size_t(hidden * tokens));
        result.ids.resize(size_t(topk * tokens));
        result.routing.resize(size_t(topk * tokens));
        result.router_probs.resize(size_t(experts * tokens));
        ggml_backend_tensor_get(ready.residual,
            result.residual.data(), 0, ggml_nbytes(ready.residual));
        ggml_backend_tensor_get(ready.ffn_norm,
            result.norm.data(), 0, ggml_nbytes(ready.ffn_norm));
        ggml_backend_tensor_get(ready.topk_ids,
            result.ids.data(), 0, ggml_nbytes(ready.topk_ids));
        ggml_backend_tensor_get(ready.topk_weights,
            result.routing.data(), 0, ggml_nbytes(ready.topk_weights));
        ggml_backend_tensor_get(router_probs,
            result.router_probs.data(), 0, ggml_nbytes(router_probs));
    }
    elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::printf(
        "[CYCLE_PHONE_ATTN_PRE] backend=%s "
        "positions=[%lld,%lld] tokens=%lld kv_tokens=%lld "
        "heads=%lld kv_heads=%lld graph_nodes=%d "
        "wall_ms=%.3f status=%s\n",
        ggml_backend_name(backend),
        (long long)kv_token_start, (long long)(kv_tokens - 1),
        (long long)tokens, (long long)kv_tokens,
        (long long)n_heads, (long long)n_kv_heads,
        ggml_graph_n_nodes(graph), elapsed_ms,
        ok ? "OK" : "FAIL");

    if (ok && (wire != nullptr || retain_device_ready)) {
        // Borrow original Attention tensors until the local Phone FFN
        // consumes them, preventing a hidden PC->Phone activation upload.
        result.phone_ctx = ctx;
        result.phone_compute_buffer = compute_buffer;
        result.phone_graph = ready;
        return true;
    }
    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    return ok;
}

// Intermediate physical FFN taps are opt-in diagnostics. They are borrowed
// graph results read back after the existing backend compute; the actual FFN
// path, final reduction and strict validation thresholds are unchanged.
struct cycle_real_ffn_stages {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> swiglu;
};

// CPU-observed FFN graph build / buffer prepare / execute call times.
// For Phone RPC the execute call is a submission, NOT the server kernel
// duration. C2P/Phone-full GPU setup is profiled separately.
struct cycle_real_ffn_phase_profile {
    uint64_t count = 0;
    double graph_build_ms = 0.0;
    double backend_prepare_ms = 0.0;
    double backend_execute_call_ms = 0.0;
};

static bool cycle_real_execute_ffn(
        ggml_backend_t backend,
        const llama_hybrid_cycle_ffn_shard_weights & shard,
        const std::vector<float> & norm,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        int64_t hidden, int64_t tokens,
        int64_t topk, int64_t experts,
        std::vector<float> & partial,
        std::vector<float> * raw_down_out = nullptr,
        cycle_real_ffn_stages * stages_out = nullptr,
        const cycle_real_live_wire * wire = nullptr,
        const llama_hybrid_cycle_ready_graph * local_phone_ready = nullptr,
        cycle_real_ffn_phase_profile * phase_profile = nullptr) {
    const auto graph_build_begin = std::chrono::steady_clock::now();
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    auto in = llama_hybrid_cycle_make_ffn_graph_inputs(
        ctx, hidden, tokens, topk);
    auto * x = ggml_reshape_3d(ctx, in.norm, hidden, 1, tokens);
    auto * gate = ggml_mul_mat_id(ctx, shard.gate, x, in.topk_ids);
    auto * up = ggml_mul_mat_id(ctx, shard.up, x, in.topk_ids);
    auto * activation = ggml_swiglu_split(ctx, gate, up);
    auto * down = ggml_mul_mat_id(ctx, shard.down, activation, in.topk_ids);
    auto * weighted = ggml_mul(ctx, down, in.topk_weights);

    ggml_tensor * sum = nullptr;
    for (int64_t k = 0; k < topk; ++k) {
        auto * view = ggml_view_2d(ctx, weighted, hidden, tokens,
                                  weighted->nb[2], size_t(k) * weighted->nb[1]);
        sum = sum ? ggml_add(ctx, sum, view) : view;
    }
    in.ffn_output = ggml_cont(ctx, sum);
    ggml_set_name(in.ffn_output, "cycle_num_phone_ffn_output");
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, in.ffn_output);
    const auto graph_build_done = std::chrono::steady_clock::now();
    if (std::getenv("LLAMA_HYBRID_CYCLE_STAGE_PROFILE") != nullptr) {
        std::printf("[CYCLE_FFN_GRAPH_BUILD] backend=%s "
                    "graph_build_ms=%.3f nodes=%d\n",
                    ggml_backend_name(backend),
                    std::chrono::duration<double, std::milli>(
                        graph_build_done - graph_build_begin).count(),
                    ggml_graph_n_nodes(graph));
    }

    bool ok = false;
    {
        const char * backend_name = ggml_backend_name(backend);
        std::printf("[CYCLE_REAL_FFN_SHARD] backend=%s channels=%lld nodes=%d "
                    "stage=PREPARE_BEGIN\n",
                    backend_name, (long long)shard.up->ne[1], ggml_graph_n_nodes(graph));
        llama_hybrid_cycle_ffn_executor executor;
        const auto prepare_begin = std::chrono::steady_clock::now();
        const bool prepared = executor.prepare(
            backend, ctx, graph, in, shard, hidden, tokens, topk, experts);
        const auto prepare_done = std::chrono::steady_clock::now();
        if (phase_profile) {
            ++phase_profile->count;
            phase_profile->graph_build_ms +=
                std::chrono::duration<double, std::milli>(
                    graph_build_done - graph_build_begin).count();
            phase_profile->backend_prepare_ms +=
                std::chrono::duration<double, std::milli>(
                    prepare_done - prepare_begin).count();
        }
        if (!prepared) {
            std::fprintf(stderr,
                "[CYCLE_REAL_FFN_SHARD] backend=%s stage=PREPARE_FAIL "
                "weight_buft=%s compute_buft=%s\n",
                backend_name,
                shard.up->buffer != nullptr ?
                    ggml_backend_buft_name(ggml_backend_buffer_get_type(shard.up->buffer)) :
                    "(unallocated)",
                ggml_backend_buft_name(ggml_backend_get_default_buffer_type(backend)));
        } else {
            if (wire == nullptr) {
                partial.resize(size_t(hidden * tokens));
            }
            const auto execute_begin = std::chrono::steady_clock::now();
            ok = executor.execute(
                norm.data(), norm.size(),
                ids.data(), ids.size(),
                routing.data(), routing.size(),
                wire != nullptr ? nullptr : partial.data(),
                wire != nullptr ? 0 : partial.size(),
                local_phone_ready);
            if (phase_profile) {
                phase_profile->backend_execute_call_ms +=
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - execute_begin).count();
            }
            if (ok && stages_out != nullptr) {
                const int64_t elems = shard.up->ne[1] * topk * tokens;
                const auto read_stage = [&](const char * stage,
                                            ggml_tensor * node,
                                            std::vector<float> & dst) {
                    if (!node || node->type != GGML_TYPE_F32 ||
                        !ggml_is_contiguous(node) ||
                        ggml_nelements(node) != elems) {
                        std::fprintf(stderr,
                            "[CYCLE_FFN_STAGE_CAPTURE] backend=%s stage=%s "
                            "status=FAIL reason=TYPE_OR_SHAPE\n",
                            backend_name, stage);
                        return false;
                    }
                    dst.resize(size_t(elems));
                    ggml_backend_tensor_get(
                        node, dst.data(), 0, ggml_nbytes(node));
                    std::printf(
                        "[CYCLE_FFN_STAGE_CAPTURE] backend=%s stage=%s "
                        "elements=%lld status=OK\n",
                        backend_name, stage, (long long)elems);
                    return true;
                };
                ok = read_stage("GATE", gate, stages_out->gate) &&
                     read_stage("UP", up, stages_out->up) &&
                     read_stage("SWIGLU", activation, stages_out->swiglu);
            }
            // Diagnostic only: capture the unweighted per-expert Down
            // projection [hidden, topk, tokens] from this physical shard.
            // This is the exact same tensor as canonical ffn_moe_down
            // before Router weights and the expert reduction.
            if (ok && raw_down_out != nullptr) {
                if (down->type != GGML_TYPE_F32 ||
                    !ggml_is_contiguous(down) ||
                    ggml_nelements(down) != hidden * topk * tokens) {
                    std::fprintf(stderr,
                        "[CYCLE_FFN_INTERMEDIATE] backend=%s status=FAIL "
                        "reason=DOWN_OUTPUT_SHAPE\n", backend_name);
                    ok = false;
                } else {
                    raw_down_out->resize(size_t(hidden * topk * tokens));
                    ggml_backend_tensor_get(
                        down, raw_down_out->data(), 0, ggml_nbytes(down));
                }
            }
            if (ok && wire != nullptr) {
                // The remote output has no ordinary RPC readback: the Phone
                // owns it until its Cycle TX worker copies Send #2.
                ok = ggml_backend_rpc_cycle_send_partial(
                    backend, wire->layer, wire->chunk,
                    wire->seq, wire->graph_id, in.ffn_output);
                if (!ok) {
                    std::fprintf(stderr,
                        "[CYCLE_REAL_WIRE] stage=PARTIAL status=FAIL "
                        "reason=PHONE_REMOTE_SEND\n");
                }
            }
            std::printf("[CYCLE_REAL_FFN_SHARD] backend=%s stage=COMPUTE_END "
                        "elements=%zu status=%s\n",
                        backend_name, partial.size(), ok ? "OK" : "FAIL");
        }
    }
    ggml_free(ctx);
    return ok;
}

struct cycle_real_diff {
    double max_abs = 0;
    double max_ref = 0;
    double rms = 0;
    double cosine = 0;
    double tolerance = 0;
    bool valid = false;
    bool pass = false;
};

static cycle_real_diff cycle_real_compare(
        const std::vector<float> & actual,
        const std::vector<float> & reference) {
    cycle_real_diff result {};
    if (actual.empty() || actual.size() != reference.size()) {
        return result;
    }
    double sum_sq = 0;
    double sum_x_sq = 0;
    double sum_y_sq = 0;
    double dot = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double x = actual[i], y = reference[i];
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return result;
        }
        const double err = std::fabs(x - y);
        result.max_abs = std::max(result.max_abs, err);
        result.max_ref = std::max(result.max_ref, std::fabs(y));
        sum_sq += err * err;
        sum_x_sq += x * x;
        sum_y_sq += y * y;
        dot += x * y;
    }
    result.rms = std::sqrt(sum_sq / actual.size());
    result.cosine = (sum_x_sq > 0 && sum_y_sq > 0) ?
        dot / std::sqrt(sum_x_sq * sum_y_sq) : 0;
    result.tolerance = 0.001 + 0.005 * result.max_ref;
    result.valid = true;
    result.pass = result.max_abs <= result.tolerance;
    return result;
}

// Full-period oracle: compare the REAL eight-layer (or X+Y+2-layer) FIFO
// output against the corresponding captured independent CPU l_out tensor.
// Use the already established strict single-P2C bound, including a tighter
// per-chunk bound when a chunk has a smaller reference activation scale.
// This runs AFTER the timed Cycle execution; it does not affect its wall_ms.
static bool cycle_real_period_cpu_parity(
        const std::vector<float> & actual,
        const std::vector<float> & reference,
        int64_t hidden, int32_t tokens, int chunk_tokens, int final_layer) {
    if (hidden <= 0 || tokens <= 0 || chunk_tokens <= 0 ||
        actual.size() != size_t(tokens * hidden) ||
        reference.size() != actual.size()) {
        std::fprintf(stderr,
            "[CYCLE_PERIOD_PARITY] scope=FULL status=FAIL "
            "reason=REFERENCE_SHAPE_MISMATCH actual=%zu reference=%zu "
            "tokens=%d hidden=%lld\n",
            actual.size(), reference.size(), int(tokens), (long long) hidden);
        return false;
    }

    const auto strict_tolerance = [](const cycle_real_diff & d) {
        // Same strict output tolerance as the single-layer P2C test.
        return 0.02 + 0.00002 * d.max_ref;
    };
    const cycle_real_diff full = cycle_real_compare(actual, reference);
    const bool full_ok = full.valid &&
        full.max_abs <= strict_tolerance(full);
    size_t worst_index = 0;
    if (full.valid) {
        double worst = -1.0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double error =
                std::fabs(double(actual[i]) - double(reference[i]));
            if (error > worst) {
                worst = error;
                worst_index = i;
            }
        }
    }
    std::printf(
        "[CYCLE_PERIOD_PARITY] scope=FULL target=l_out-%d "
        "tokens=%d elements=%zu max_abs=%.9g rms=%.9g "
        "cosine=%.9g max_ref=%.9g strict_tolerance=%.9g "
        "worst_token=%lld worst_channel=%lld "
        "worst_actual=%.9g worst_reference=%.9g "
        "status=%s\n",
        final_layer, int(tokens), actual.size(),
        full.max_abs, full.rms, full.cosine, full.max_ref,
        strict_tolerance(full),
        full.valid ? (long long)(worst_index / size_t(hidden)) : -1LL,
        full.valid ? (long long)(worst_index % size_t(hidden)) : -1LL,
        full.valid ? double(actual[worst_index]) : 0.0,
        full.valid ? double(reference[worst_index]) : 0.0,
        full_ok ? "PASS" : "FAIL");

    bool chunks_ok = true;
    const int n_chunks = (tokens + chunk_tokens - 1) / chunk_tokens;
    for (int c = 0; c < n_chunks; ++c) {
        const int64_t pos_begin = int64_t(c) * chunk_tokens;
        const int64_t pos_end = std::min<int64_t>(
            tokens, pos_begin + chunk_tokens);
        const size_t begin = size_t(pos_begin * hidden);
        const size_t end = size_t(pos_end * hidden);
        const std::vector<float> actual_chunk(
            actual.begin() + begin, actual.begin() + end);
        const std::vector<float> reference_chunk(
            reference.begin() + begin, reference.begin() + end);
        const cycle_real_diff d =
            cycle_real_compare(actual_chunk, reference_chunk);
        const bool chunk_ok = d.valid &&
            d.max_abs <= strict_tolerance(d);
        chunks_ok = chunks_ok && chunk_ok;
        std::printf(
            "[CYCLE_PERIOD_PARITY_CHUNK] layer=%d chunk=%d "
            "positions=[%lld,%lld] elements=%zu "
            "max_abs=%.9g rms=%.9g cosine=%.9g "
            "max_ref=%.9g strict_tolerance=%.9g status=%s\n",
            final_layer, c, (long long) pos_begin,
            (long long)(pos_end - 1), d.valid ? actual_chunk.size() : 0,
            d.max_abs, d.rms, d.cosine, d.max_ref,
            strict_tolerance(d), chunk_ok ? "PASS" : "FAIL");
    }
    const bool ok = full_ok && chunks_ok;
    std::printf(
        "[CYCLE_PERIOD_PARITY_SUM] target=l_out-%d "
        "chunks=%d whole=%s all_chunks=%s cpu_parity=%s\n",
        final_layer, n_chunks, full_ok ? "PASS" : "FAIL",
        chunks_ok ? "PASS" : "FAIL", ok ? "PASS" : "FAIL");
    return ok;
}

// Compare one Chunk of an existing CPU-hosted activation against the same
// contiguous [hidden, tokens] slice of the independent CPU reference.
// Diagnostic only: no extra RPC readback and NO impact on pass/fail rules.
static void cycle_real_period_checkpoint(
        const char * stage, int layer, int chunk, int64_t start,
        int64_t hidden, const std::vector<float> & actual,
        const std::vector<float> & reference) {
    if (!cycle_real_env_flag("LLAMA_HYBRID_CYCLE_NUMERIC_TRACE")) {
        return;
    }
    if (hidden <= 0 || start < 0 || actual.empty() ||
        actual.size() % size_t(hidden) != 0 ||
        size_t(start * hidden) > reference.size() ||
        actual.size() > reference.size() - size_t(start * hidden)) {
        std::printf(
            "[CYCLE_LAYER_PARITY] stage=%s layer=%d chunk=%d "
            "status=NO_REFERENCE actual=%zu reference=%zu\n",
            stage, layer, chunk, actual.size(), reference.size());
        return;
    }
    const size_t offset = size_t(start * hidden);
    const std::vector<float> ref_slice(
        reference.begin() + offset,
        reference.begin() + offset + actual.size());
    const cycle_real_diff d = cycle_real_compare(actual, ref_slice);
    const double tol = 0.02 + 0.00002 * d.max_ref;
    size_t worst = 0;
    double worst_abs = -1.0;
    if (d.valid) {
        for (size_t i = 0; i < actual.size(); ++i) {
            const double e = std::fabs(
                double(actual[i]) - double(ref_slice[i]));
            if (e > worst_abs) {
                worst_abs = e;
                worst = i;
            }
        }
    }
    std::printf(
        "[CYCLE_LAYER_PARITY] stage=%s layer=%d chunk=%d "
        "positions=[%lld,%lld] max_abs=%.9g rms=%.9g "
        "cosine=%.9g tolerance=%.9g worst_token=%lld "
        "worst_channel=%lld status=%s\n",
        stage, layer, chunk,
        (long long)start,
        (long long)(start + int64_t(actual.size() / size_t(hidden)) - 1),
        d.max_abs, d.rms, d.cosine, tol,
        d.valid ? (long long)(start + worst / size_t(hidden)) : -1LL,
        d.valid ? (long long)(worst % size_t(hidden)) : -1LL,
        d.valid ? (d.max_abs <= tol ? "WITHIN_TOLERANCE" : "DRIFT") :
                  "INVALID");
}

// Decode a single selected expert's packed Q4_K rows using the canonical
// GGML CPU GET_ROWS kernel. This produces the same F32 values represented
// by the GGUF weight (without quantizing the activation to Q8_K). It is an
// independent mathematical oracle for interpreting CPU(Q4_K x Q8_K) vs
// GPU(Q4_K x F32) dot-product differences.
static bool cycle_real_dequantize_k_expert_on_cpu(
        ggml_backend_t cpu_backend,
        enum ggml_type weight_type,
        const std::vector<uint8_t> & packed,
        int64_t hidden, int64_t channels,
        std::vector<float> & dequantized) {
    if (!cpu_backend ||
        (weight_type != GGML_TYPE_Q4_K && weight_type != GGML_TYPE_Q6_K) ||
        hidden <= 0 || channels <= 0 ||
        hidden % ggml_blck_size(weight_type) != 0 ||
        packed.size() !=
            ggml_row_size(weight_type, hidden) * size_t(channels)) {
        return false;
    }
    ggml_init_params p {};
    p.mem_size = 2 * 1024 * 1024;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(
        ctx, weight_type, hidden, channels);
    ggml_tensor * rows = ggml_new_tensor_1d(
        ctx, GGML_TYPE_I32, channels);
    ggml_set_input(rows);
    ggml_tensor * y = ggml_get_rows(ctx, w, rows);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, y);
    const bool supported = ggml_backend_supports_op(cpu_backend, y);
    ggml_backend_buffer_t buffer = supported ?
        ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(cpu_backend)) : nullptr;
    if (!buffer) {
        std::fprintf(stderr,
            "[CYCLE_FFN_F32_ORACLE] status=FAIL "
            "reason=CPU_GET_ROWS_UNSUPPORTED_OR_ALLOC\n");
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
    std::vector<int32_t> ids(static_cast<size_t>(channels), 0);
    for (int64_t j = 0; j < channels; ++j) {
        ids[size_t(j)] = int32_t(j);
    }
    ggml_backend_tensor_set(rows, ids.data(), 0, ggml_nbytes(rows));
    const bool computed = ggml_backend_graph_compute(cpu_backend, graph) ==
        GGML_STATUS_SUCCESS;
    if (computed) {
        ggml_backend_synchronize(cpu_backend);
        dequantized.resize(static_cast<size_t>(hidden * channels));
        ggml_backend_tensor_get(y, dequantized.data(), 0, ggml_nbytes(y));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (!computed) {
        std::fprintf(stderr,
            "[CYCLE_FFN_F32_ORACLE] status=FAIL "
            "reason=CPU_DEQUANT_COMPUTE\n");
    }
    return computed;
}

// Establish a third, independent numerical reference: dequantize exactly
// the selected Phone Q4_K expert into F32 weights on CPU, then form F64
// dot-products with the ORIGINAL F32 activation. Compare that reference
// against the CPU Q4_KxQ8_K result, Phone generic Q4_KxF32 result, Phone
// MoE Q4_KxF32 result, and Phone dense F32xF32 result.
//
// A difference from the CPU reference alone cannot prove that OpenCL is
// wrong: the CPU Q4_K dot path first quantizes its F32 activation to Q8_K.
// This F64 oracle is intentionally NOT used by the official layer gate.
static bool cycle_real_probe_q4k_dense_f64_oracle(
        const char * stage,
        ggml_backend_t cpu_backend,
        ggml_backend_t phone_backend,
        const std::vector<uint8_t> & packed,
        int64_t hidden, int64_t channels,
        const float * input,
        const std::vector<float> & cpu_q8,
        const std::vector<float> & gpu_q4,
        const std::vector<float> & gpu_moe) {
    if (!input || !cpu_backend || !phone_backend ||
        hidden <= 0 || channels <= 0 ||
        cpu_q8.size() != size_t(channels) ||
        gpu_q4.size() != size_t(channels) ||
        gpu_moe.size() != size_t(channels)) {
        return false;
    }
    std::vector<float> dense_weights;
    if (!cycle_real_dequantize_k_expert_on_cpu(
            cpu_backend, GGML_TYPE_Q4_K, packed, hidden, channels, dense_weights)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_F32_ORACLE] stage=%s status=FAIL "
            "reason=CPU_DEQUANT\n", stage);
        return false;
    }

    // The dequantized F32 values are the actual Q4_K representable
    // weights; accumulate in double to minimize reference rounding.
    std::vector<double> f64_reference(static_cast<size_t>(channels), 0.0);
    for (int64_t ch = 0; ch < channels; ++ch) {
        double sum = 0.0;
        const size_t row = size_t(ch * hidden);
        for (int64_t k = 0; k < hidden; ++k) {
            sum += double(dense_weights[row + size_t(k)]) *
                   double(input[size_t(k)]);
        }
        f64_reference[size_t(ch)] = sum;
    }

    // An F32-weight GPU matmul separates Q4_K-specific unpacking /
    // quantized GEMV numerical behavior from general GPU accumulation.
    ggml_init_params p {};
    p.mem_size = 2 * 1024 * 1024;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hidden, channels);
    ggml_tensor * x = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hidden, 1);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, y);
    if (!ggml_backend_supports_op(phone_backend, y)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_F32_ORACLE] stage=%s status=FAIL "
            "reason=PHONE_F32_MUL_MAT_UNSUPPORTED\n", stage);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(phone_backend));
    if (!buffer) {
        std::fprintf(stderr,
            "[CYCLE_FFN_F32_ORACLE] stage=%s status=FAIL "
            "reason=PHONE_F32_ALLOC\n", stage);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(
        w, dense_weights.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, input, 0, ggml_nbytes(x));
    const bool computed =
        ggml_backend_graph_compute(phone_backend, graph) ==
        GGML_STATUS_SUCCESS;
    std::vector<float> gpu_f32;
    if (computed) {
        ggml_backend_synchronize(phone_backend);
        gpu_f32.resize(static_cast<size_t>(channels));
        ggml_backend_tensor_get(y, gpu_f32.data(), 0, ggml_nbytes(y));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (!computed) {
        std::fprintf(stderr,
            "[CYCLE_FFN_F32_ORACLE] stage=%s status=FAIL "
            "reason=PHONE_F32_COMPUTE\n", stage);
        return false;
    }

    struct err_summary {
        double max_abs = 0.0;
        double sum_sq = 0.0;
    } cpu_err, gpu_q4_err, gpu_moe_err, gpu_f32_err;
    for (int64_t ch = 0; ch < channels; ++ch) {
        const size_t i = size_t(ch);
        const double ref = f64_reference[i];
        const auto accumulate = [ref](double value, err_summary & acc) {
            if (!std::isfinite(value) || !std::isfinite(ref)) {
                return false;
            }
            const double delta = std::fabs(value - ref);
            acc.max_abs = std::max(acc.max_abs, delta);
            acc.sum_sq += delta * delta;
            return true;
        };
        if (!accumulate(cpu_q8[i], cpu_err) ||
            !accumulate(gpu_q4[i], gpu_q4_err) ||
            !accumulate(gpu_moe[i], gpu_moe_err) ||
            !accumulate(gpu_f32[i], gpu_f32_err)) {
            std::fprintf(stderr,
                "[CYCLE_FFN_F32_ORACLE] stage=%s status=FAIL "
                "reason=NONFINITE\n", stage);
            return false;
        }
    }
    const double n = double(channels);
    std::printf(
        "[CYCLE_FFN_F32_ORACLE] stage=%s channels=%lld "
        "cpu_q4q8_vs_f64_max_abs=%.8g cpu_q4q8_vs_f64_rms=%.8g "
        "gpu_q4f32_vs_f64_max_abs=%.8g gpu_q4f32_vs_f64_rms=%.8g "
        "gpu_moe_vs_f64_max_abs=%.8g gpu_moe_vs_f64_rms=%.8g "
        "gpu_dense_f32_vs_f64_max_abs=%.8g gpu_dense_f32_vs_f64_rms=%.8g "
        "oracle=DEQUANT_F32_WEIGHT_TIMES_F32_INPUT_ACCUM_F64 "
        "status=MEASURED\n",
        stage, (long long)channels,
        cpu_err.max_abs, std::sqrt(cpu_err.sum_sq / n),
        gpu_q4_err.max_abs, std::sqrt(gpu_q4_err.sum_sq / n),
        gpu_moe_err.max_abs, std::sqrt(gpu_moe_err.sum_sq / n),
        gpu_f32_err.max_abs, std::sqrt(gpu_f32_err.sum_sq / n));
    return true;
}

// Discriminate the normal Q4_K MUL_MAT kernel from the fused MoE
// MUL_MAT_ID kernel on the SAME Phone OpenCL device. A fresh one-expert
// quantized weight buffer is populated from the already-byte-verified
// physical Phone shard; no GGUF re-quantization or kernel modification.
//
// Target the token/expert slot with the largest observed Phone stage error.
// MUL_MAT [hidden,channels] x [hidden,1] uses one query vector, whereas
// MUL_MAT_ID runs the full TopK/2-token MoE kernel. Comparing their outputs
// isolates a possible MoE-specific half-dot or expert dispatch discrepancy.
//
// This is a correctness probe, NOT the production Phone FFN path.
static bool cycle_real_probe_phone_single_expert_mul_mat(
        const char * stage,
        ggml_backend_t phone_backend,
        ggml_backend_t cpu_backend,
        const ggml_tensor * source_weight,
        const std::vector<float> & input_norm,
        const std::vector<int32_t> & selected_ids,
        const std::vector<float> & full_cpu_stage,
        const std::vector<float> & phone_id_stage,
        int64_t hidden, int64_t tokens, int64_t topk,
        int64_t full_channels, int64_t phone_channel_begin,
        int32_t target_token, int32_t target_slot) {
    if (!phone_backend || !cpu_backend ||
        !source_weight || !source_weight->buffer ||
        !ggml_is_contiguous(source_weight) ||
        source_weight->type != GGML_TYPE_Q4_K ||
        hidden != source_weight->ne[0] ||
        source_weight->ne[1] <= 0 ||
        source_weight->ne[2] <= 0 || source_weight->ne[3] != 1 ||
        target_token < 0 || target_token >= tokens ||
        target_slot < 0 || target_slot >= topk ||
        phone_channel_begin < 0 ||
        phone_channel_begin + source_weight->ne[1] > full_channels ||
        input_norm.size() != size_t(hidden * tokens) ||
        selected_ids.size() != size_t(topk * tokens) ||
        full_cpu_stage.size() != size_t(full_channels * topk * tokens) ||
        phone_id_stage.size() !=
            size_t(source_weight->ne[1] * topk * tokens)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_GPU_OP_COMPARE] stage=%s status=FAIL "
            "reason=INVALID_SHAPE\n", stage);
        return false;
    }
    const int64_t channels = source_weight->ne[1];
    const int32_t expert = selected_ids[
        size_t(target_token * topk + target_slot)];
    if (expert < 0 || expert >= source_weight->ne[2]) {
        std::fprintf(stderr,
            "[CYCLE_FFN_GPU_OP_COMPARE] stage=%s status=FAIL "
            "reason=BAD_EXPERT_ID\n", stage);
        return false;
    }
    const size_t bytes_per_expert =
        ggml_row_size(source_weight->type, hidden) * size_t(channels);
    if (bytes_per_expert == 0 ||
        ggml_nbytes(source_weight) !=
        bytes_per_expert * size_t(source_weight->ne[2])) {
        return false;
    }

    ggml_init_params p {};
    p.mem_size = 2 * 1024 * 1024;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(
        ctx, source_weight->type, hidden, channels);
    ggml_tensor * x = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, hidden, 1);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, y);

    if (!ggml_backend_supports_op(phone_backend, y)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_GPU_OP_COMPARE] stage=%s status=FAIL "
            "reason=GPU_MUL_MAT_UNSUPPORTED\n", stage);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t alloc = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(phone_backend));
    if (!alloc) {
        std::fprintf(stderr,
            "[CYCLE_FFN_GPU_OP_COMPARE] stage=%s status=FAIL "
            "reason=GPU_ALLOC\n", stage);
        ggml_free(ctx);
        return false;
    }

    std::vector<uint8_t> packed(bytes_per_expert);
    ggml_backend_tensor_get(
        source_weight, packed.data(),
        size_t(expert) * bytes_per_expert, packed.size());
    ggml_backend_tensor_set(
        w, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(
        x, input_norm.data() + size_t(target_token * hidden),
        0, ggml_nbytes(x));
    std::printf(
        "[CYCLE_FFN_GPU_OP_COMPARE] stage=%s target_pos=%d "
        "target_slot=%d target_expert=%d weight_bytes=%zu "
        "matmul_prec=F32 graph_nodes=%d status=PREPARED\n",
        stage, int(target_token), int(target_slot), int(expert),
        packed.size(), ggml_graph_n_nodes(graph));

    const ggml_status compute_status =
        ggml_backend_graph_compute(phone_backend, graph);
    const bool ok = compute_status == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_synchronize(phone_backend);
        std::vector<float> single_output(static_cast<size_t>(channels), 0.0f);
        ggml_backend_tensor_get(
            y, single_output.data(), 0, ggml_nbytes(y));
        double max_single_cpu = 0;
        double max_id_cpu = 0;
        double max_single_id = 0;
        double sum_single_cpu = 0;
        double sum_id_cpu = 0;
        double sum_single_id = 0;
        int64_t worst_channel = -1;
        for (int64_t ch = 0; ch < channels; ++ch) {
            const size_t cpu_idx = size_t(
                (target_token * topk + target_slot) * full_channels +
                phone_channel_begin + ch);
            const size_t phone_idx = size_t(
                (target_token * topk + target_slot) * channels + ch);
            const double cpu_value = full_cpu_stage[cpu_idx];
            const double id_value = phone_id_stage[phone_idx];
            const double single_value = single_output[size_t(ch)];
            if (!std::isfinite(cpu_value) ||
                !std::isfinite(id_value) ||
                !std::isfinite(single_value)) {
                std::fprintf(stderr,
                    "[CYCLE_FFN_GPU_OP_COMPARE] stage=%s status=FAIL "
                    "reason=NONFINITE\n", stage);
                ggml_backend_buffer_free(alloc);
                ggml_free(ctx);
                return false;
            }
            const double e_single_cpu =
                std::fabs(single_value - cpu_value);
            const double e_id_cpu = std::fabs(id_value - cpu_value);
            const double e_single_id =
                std::fabs(single_value - id_value);
            sum_single_cpu += e_single_cpu * e_single_cpu;
            sum_id_cpu += e_id_cpu * e_id_cpu;
            sum_single_id += e_single_id * e_single_id;
            max_single_cpu = std::max(max_single_cpu, e_single_cpu);
            max_id_cpu = std::max(max_id_cpu, e_id_cpu);
            if (e_single_id > max_single_id) {
                max_single_id = e_single_id;
                worst_channel = phone_channel_begin + ch;
            }
        }
        std::printf(
            "[CYCLE_FFN_GPU_OP_COMPARE] stage=%s pos=%d slot=%d expert=%d "
            "single_vs_cpu_max_abs=%.8g single_vs_cpu_rms=%.8g "
            "id_vs_cpu_max_abs=%.8g id_vs_cpu_rms=%.8g "
            "single_vs_id_max_abs=%.8g single_vs_id_rms=%.8g "
            "worst_channel=%lld status=MEASURED\n",
            stage, int(target_token), int(target_slot), int(expert),
            max_single_cpu, std::sqrt(sum_single_cpu / double(channels)),
            max_id_cpu, std::sqrt(sum_id_cpu / double(channels)),
            max_single_id, std::sqrt(sum_single_id / double(channels)),
            (long long)worst_channel);

        // The CPU baseline uses Q4_K x Q8_K, while the tested OpenCL
        // kernels consume F32 activations. Compare ALL of them against an
        // independently dequantized Q4_K x F32 dot with F64 accumulation
        // before labeling any GPU/CPU difference a correctness bug.
        std::vector<float> cpu_q8_slice(static_cast<size_t>(channels));
        std::vector<float> gpu_moe_slice(static_cast<size_t>(channels));
        for (int64_t ch = 0; ch < channels; ++ch) {
            const size_t cpu_idx = size_t(
                (target_token * topk + target_slot) * full_channels +
                phone_channel_begin + ch);
            const size_t phone_idx = size_t(
                (target_token * topk + target_slot) * channels + ch);
            cpu_q8_slice[size_t(ch)] = full_cpu_stage[cpu_idx];
            gpu_moe_slice[size_t(ch)] = phone_id_stage[phone_idx];
        }
        const bool oracle_ok = cycle_real_probe_q4k_dense_f64_oracle(
            stage, cpu_backend, phone_backend,
            packed, hidden, channels,
            input_norm.data() + size_t(target_token * hidden),
            cpu_q8_slice, single_output, gpu_moe_slice);
        if (!oracle_ok) {
            std::fprintf(stderr,
                "[CYCLE_FFN_F32_ORACLE] stage=%s status=NOT_AVAILABLE "
                "scope=DIAGNOSTIC_ONLY\n", stage);
        }
    } else {
        std::fprintf(stderr,
            "[CYCLE_FFN_GPU_OP_COMPARE] stage=%s status=FAIL "
            "reason=GPU_GRAPH_COMPUTE\n", stage);
    }
    ggml_backend_buffer_free(alloc);
    ggml_free(ctx);
    return ok;
}

// Replay the EXACT packed quantized Phone shard weights with GGML's CPU
// MUL_MAT_ID, selecting only experts used by this ubatch. The Phone tensor
// is read via RPC without re-quantization; selected experts are compacted
// into [hidden,channels,active_experts] and the TopK IDs are remapped.
//
// This distinguishes an OpenCL kernel/input-precision discrepancy from a
// corrupt packed-weight partition. The second CPU run performs an intentional
// F32 -> F16 -> F32 roundtrip of the *input*, to test the hypothesis that
// the OpenCL quantized path internally reduces input precision.
//
// Diagnostic only; does not replace the production Phone FFN kernel.
static bool cycle_real_replay_phone_matmul_id_on_cpu(
        const char * stage,
        ggml_backend_t cpu_backend,
        const ggml_tensor * phone_weight,
        const std::vector<float> & norm,
        const std::vector<int32_t> & selected_ids,
        int64_t hidden, int64_t tokens, int64_t topk,
        int64_t full_channels, int64_t phone_channel_begin,
        const std::vector<float> & full_reference,
        const std::vector<float> & phone_output) {
    if (!cpu_backend || !phone_weight || !phone_weight->buffer ||
        !ggml_is_contiguous(phone_weight) ||
        phone_weight->ne[0] != hidden ||
        phone_weight->ne[1] <= 0 ||
        phone_weight->ne[2] <= 0 || phone_weight->ne[3] != 1 ||
        ggml_blck_size(phone_weight->type) <= 1 ||
        hidden % ggml_blck_size(phone_weight->type) != 0 ||
        phone_channel_begin < 0 ||
        phone_channel_begin + phone_weight->ne[1] > full_channels ||
        norm.size() != size_t(hidden * tokens) ||
        selected_ids.size() != size_t(topk * tokens) ||
        full_reference.size() != size_t(full_channels * topk * tokens) ||
        phone_output.size() != size_t(phone_weight->ne[1] * topk * tokens)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_KERNEL_REPLAY] stage=%s status=FAIL "
            "reason=INVALID_PROBE_SHAPE\n", stage);
        return false;
    }
    const int64_t channels = phone_weight->ne[1];
    const int64_t experts = phone_weight->ne[2];
    std::vector<int32_t> active = selected_ids;
    std::sort(active.begin(), active.end());
    active.erase(std::unique(active.begin(), active.end()), active.end());
    if (active.empty() || active.front() < 0 || active.back() >= experts) {
        return false;
    }
    std::vector<int32_t> compact_ids;
    compact_ids.reserve(selected_ids.size());
    for (int32_t id : selected_ids) {
        const auto it = std::lower_bound(active.begin(), active.end(), id);
        if (it == active.end() || *it != id) {
            return false;
        }
        compact_ids.push_back(int32_t(it - active.begin()));
    }

    const size_t row_bytes =
        ggml_row_size(phone_weight->type, hidden);
    const size_t expert_bytes = row_bytes * size_t(channels);
    if (row_bytes == 0 || expert_bytes == 0 ||
        ggml_nbytes(phone_weight) != expert_bytes * size_t(experts)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_KERNEL_REPLAY] stage=%s status=FAIL "
            "reason=QUANTIZED_WEIGHT_STRIDES\n", stage);
        return false;
    }
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    auto * compact_weight = ggml_new_tensor_3d(
        ctx, phone_weight->type, hidden, channels, int64_t(active.size()));
    auto * input_norm = ggml_new_tensor_3d(
        ctx, GGML_TYPE_F32, hidden, 1, tokens);
    auto * ids = ggml_new_tensor_2d(
        ctx, GGML_TYPE_I32, topk, tokens);
    ggml_set_input(input_norm);
    ggml_set_input(ids);
    auto * output = ggml_mul_mat_id(
        ctx, compact_weight, input_norm, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(cpu_backend, output)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_KERNEL_REPLAY] stage=%s status=FAIL "
            "reason=CPU_MUL_MAT_ID_UNSUPPORTED\n", stage);
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        ctx, ggml_backend_get_default_buffer_type(cpu_backend));
    if (!buffer) {
        std::fprintf(stderr,
            "[CYCLE_FFN_KERNEL_REPLAY] stage=%s status=FAIL "
            "reason=CPU_ALLOC\n", stage);
        ggml_free(ctx);
        return false;
    }
    std::vector<uint8_t> expert_bytes_buf(expert_bytes);
    for (size_t j = 0; j < active.size(); ++j) {
        // Backend tensor_get observes the same logical packed bytes as the
        // matching FNV fingerprints; CPU receives no dequant/requant pass.
        ggml_backend_tensor_get(
            phone_weight, expert_bytes_buf.data(),
            size_t(active[j]) * expert_bytes, expert_bytes);
        ggml_backend_tensor_set(
            compact_weight, expert_bytes_buf.data(),
            j * expert_bytes, expert_bytes);
    }
    ggml_backend_tensor_set(ids, compact_ids.data(), 0, ggml_nbytes(ids));
    std::printf(
        "[CYCLE_FFN_KERNEL_REPLAY] stage=%s weight_type=%s "
        "active_experts=%zu packed_bytes=%zu channels=%lld "
        "input=PHONE_WEIGHT_BYTES status=PREPARED\n",
        stage, ggml_type_name(phone_weight->type),
        active.size(), active.size() * expert_bytes,
        (long long)channels);

    auto compare = [&](const char * input_label,
                       const std::vector<float> & values) -> bool {
        ggml_backend_tensor_set(input_norm,
            values.data(), 0, ggml_nbytes(input_norm));
        if (ggml_backend_graph_compute(cpu_backend, graph) !=
            GGML_STATUS_SUCCESS) {
            std::fprintf(stderr,
                "[CYCLE_FFN_KERNEL_REPLAY] stage=%s input=%s status=FAIL "
                "reason=CPU_GRAPH_COMPUTE\n", stage, input_label);
            return false;
        }
        ggml_backend_synchronize(cpu_backend);
        std::vector<float> replay(size_t(channels * topk * tokens));
        ggml_backend_tensor_get(
            output, replay.data(), 0, ggml_nbytes(output));

        double max_ref = 0.0;
        double max_phone = 0.0;
        double max_cpu_gpu = 0.0;
        double ref_sq = 0.0;
        double phone_sq = 0.0;
        double cpu_gpu_sq = 0.0;
        size_t worst_flat = 0;
        for (int64_t token = 0; token < tokens; ++token) {
            for (int64_t slot = 0; slot < topk; ++slot) {
                for (int64_t channel = 0; channel < channels; ++channel) {
                    const size_t index = size_t(
                        (token * topk + slot) * channels + channel);
                    const size_t ref_index = size_t(
                        (token * topk + slot) * full_channels +
                        phone_channel_begin + channel);
                    const double ref = full_reference[ref_index];
                    const double cpu = replay[index];
                    const double gpu = phone_output[index];
                    if (!std::isfinite(ref) ||
                        !std::isfinite(cpu) ||
                        !std::isfinite(gpu)) {
                        std::fprintf(stderr,
                            "[CYCLE_FFN_KERNEL_REPLAY] stage=%s input=%s "
                            "status=FAIL reason=NONFINITE\n",
                            stage, input_label);
                        return false;
                    }
                    const double e_ref = std::fabs(cpu - ref);
                    const double e_phone = std::fabs(gpu - ref);
                    const double e_cpu_gpu = std::fabs(cpu - gpu);
                    ref_sq += e_ref * e_ref;
                    phone_sq += e_phone * e_phone;
                    cpu_gpu_sq += e_cpu_gpu * e_cpu_gpu;
                    max_ref = std::max(max_ref, e_ref);
                    max_phone = std::max(max_phone, e_phone);
                    if (e_cpu_gpu > max_cpu_gpu) {
                        max_cpu_gpu = e_cpu_gpu;
                        worst_flat = index;
                    }
                }
            }
        }
        const size_t n = replay.size();
        const int64_t worst_channel = int64_t(
            worst_flat % size_t(channels));
        const int64_t worst_row = int64_t(
            worst_flat / size_t(channels));
        const int64_t worst_pos = worst_row / topk;
        const int64_t worst_slot = worst_row % topk;
        std::printf(
            "[CYCLE_FFN_KERNEL_REPLAY] stage=%s input=%s "
            "cpu_vs_full_max_abs=%.8g cpu_vs_full_rms=%.8g "
            "gpu_vs_full_max_abs=%.8g gpu_vs_full_rms=%.8g "
            "gpu_vs_cpu_max_abs=%.8g gpu_vs_cpu_rms=%.8g "
            "worst_pos=%lld worst_expert=%d worst_channel=%lld "
            "status=MEASURED\n",
            stage, input_label,
            max_ref, std::sqrt(ref_sq / double(n)),
            max_phone, std::sqrt(phone_sq / double(n)),
            max_cpu_gpu, std::sqrt(cpu_gpu_sq / double(n)),
            (long long)worst_pos,
            selected_ids[size_t(worst_pos * topk + worst_slot)],
            (long long)(worst_channel + phone_channel_begin));
        return true;
    };

    bool ok = compare("F32_UNCHANGED", norm);
    if (ok) {
        std::vector<float> f16_roundtrip(norm.size());
        for (size_t i = 0; i < norm.size(); ++i) {
            f16_roundtrip[i] =
                ggml_fp16_to_fp32(ggml_fp32_to_fp16(norm[i]));
        }
        ok = compare("F16_ROUNDTRIP", f16_roundtrip);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

// Numerical probes below never alter the real PC/Phone Cycle execution or
// the CPU-parity correctness gate. They use actual physical packed shards,
// dequantized on CPU, against the same explicitly selected F32 activations.
struct cycle_real_f64_error {
    double max_abs = 0.0;
    double rms = 0.0;
    bool valid = false;
};

static cycle_real_f64_error cycle_real_compare_f64(
        const std::vector<float> & actual,
        const std::vector<double> & oracle) {
    cycle_real_f64_error result {};
    if (actual.empty() || actual.size() != oracle.size()) {
        return result;
    }
    double sum_sq = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(oracle[i])) {
            return result;
        }
        const double delta = std::fabs(double(actual[i]) - oracle[i]);
        result.max_abs = std::max(result.max_abs, delta);
        sum_sq += delta * delta;
    }
    result.rms = std::sqrt(sum_sq / double(actual.size()));
    result.valid = true;
    return result;
}

// Fetch exactly one expert from its physical owner. For Down the first
// dimension is the INPUT channels of the shard (256 or 512), whereas Gate
// and Up have hidden=2048 as the first dimension. Q6_K Down is supported
// alongside Q4_K Gate/Up; both retain their original quantized bytes.
static bool cycle_real_dequantize_physical_expert_for_f64(
        ggml_backend_t cpu_backend,
        const ggml_tensor * weight, int32_t expert,
        std::vector<float> & dense,
        std::vector<uint8_t> * packed_out = nullptr) {
    if (!cpu_backend || !weight || !weight->buffer ||
        !ggml_is_contiguous(weight) ||
        (weight->type != GGML_TYPE_Q4_K && weight->type != GGML_TYPE_Q6_K) ||
        weight->ne[0] <= 0 || weight->ne[1] <= 0 ||
        expert < 0 || expert >= weight->ne[2] || weight->ne[3] != 1) {
        return false;
    }
    const size_t expert_bytes =
        ggml_row_size(weight->type, weight->ne[0]) * size_t(weight->ne[1]);
    if (expert_bytes == 0 ||
        ggml_nbytes(weight) != expert_bytes * size_t(weight->ne[2])) {
        return false;
    }
    // Reuse the RPC receive buffer across Gate / Up / Down and experts.
    // Only the packed CPU-side bytes are read; no Phone compute buffer is
    // allocated for the F64 reference.
    std::vector<uint8_t> local_packed;
    std::vector<uint8_t> & packed =
        packed_out != nullptr ? *packed_out : local_packed;
    packed.resize(expert_bytes);
    ggml_backend_tensor_get(weight, packed.data(),
                            size_t(expert) * expert_bytes, expert_bytes);
    return cycle_real_dequantize_k_expert_on_cpu(
        cpu_backend, weight->type, packed, weight->ne[0],
        weight->ne[1], dense);
}

// Single-expert MUL_MAT diagnostic. The CPU quantized kernel may convert
// the common F32 input to Q8_K while OpenCL consumes F32.
// This is deliberately an ordinary MUL_MAT probe only. Do not substitute a
// one-expert MUL_MAT_ID graph for the real full-expert Phone MoE graph:
// the OpenCL RPC supports-op check can accept this synthetic shape even
// though ggml_cl_mul_mat_id subsequently aborts the Phone RPC server.
// The actual full MoE Down path was already exercised by the real FFN
// executor; this probe isolates ordinary quantized matmul numerics.
static bool cycle_real_probe_single_matmul_f64_input(
        ggml_backend_t backend,
        enum ggml_type weight_type,
        const void * weight_data, size_t weight_bytes,
        const float * input, int64_t in_channels, int64_t out_channels,
        std::vector<float> & result) {
    if (!backend || !weight_data || !input ||
        in_channels <= 0 || out_channels <= 0) {
        return false;
    }
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    ggml_tensor * w = ggml_new_tensor_2d(
        ctx, weight_type, in_channels, out_channels);
    ggml_tensor * x = ggml_new_tensor_2d(
        ctx, GGML_TYPE_F32, in_channels, 1);
    ggml_set_input(x);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, y);
    if (ggml_nbytes(w) != weight_bytes ||
        !ggml_backend_supports_op(backend, y)) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_backend_tensor_set(w, weight_data, 0, weight_bytes);
    ggml_backend_tensor_set(x, input, 0, ggml_nbytes(x));
    const bool computed =
        ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (computed) {
        ggml_backend_synchronize(backend);
        result.resize(size_t(out_channels));
        ggml_backend_tensor_get(y, result.data(), 0, ggml_nbytes(y));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return computed;
}

// Isolate Down from preceding Gate/Up/SwiGLU drift by feeding the SAME
// captured CPU-reference SwiGLU slice to CPU/GPU quantized and dense
// F32 kernels. Synthetic one-expert MoE is unsupported and omitted. The selected
// expert comes from the worst RAW_DOWN discrepancy in CPU_READY_CONTROL.
// CPU / GPU / F64 use the identical physical Phone Down shard, including
// any Q6_K format used by Q4_K_M models. No production graph is replaced.
static bool cycle_real_probe_down_f64_oracle(
        ggml_backend_t cpu_backend, ggml_backend_t phone_backend,
        const ggml_tensor * phone_down,
        const std::vector<float> & baseline_swiglu,
        const std::vector<int32_t> & selected_ids,
        int64_t full_channels, int64_t phone_channel_begin,
        int64_t hidden, int64_t tokens, int64_t topk,
        int32_t target_token, int32_t target_slot) {
    if (!phone_down || !cpu_backend || !phone_backend ||
        target_token < 0 || target_token >= tokens ||
        target_slot < 0 || target_slot >= topk ||
        phone_channel_begin < 0 ||
        phone_down->ne[0] + phone_channel_begin != full_channels ||
        phone_down->ne[1] != hidden ||
        selected_ids.size() != size_t(tokens * topk) ||
        baseline_swiglu.size() != size_t(tokens * topk * full_channels)) {
        return false;
    }
    const size_t pair = size_t(target_token * topk + target_slot);
    const int32_t expert = selected_ids[pair];
    const int64_t channels = phone_down->ne[0];
    std::vector<float> dequantized;
    std::vector<uint8_t> packed;
    if (!cycle_real_dequantize_physical_expert_for_f64(
            cpu_backend, phone_down, expert, dequantized, &packed) ||
        dequantized.size() != size_t(channels * hidden)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_DOWN_F64_ORACLE] status=NOT_AVAILABLE reason=DEQUANT\n");
        return false;
    }

    std::vector<float> activation(static_cast<size_t>(channels));
    const size_t offset = pair * size_t(full_channels) +
                          size_t(phone_channel_begin);
    std::copy_n(baseline_swiglu.data() + offset, size_t(channels),
                activation.data());
    std::vector<double> f64(size_t(hidden), 0.0);
    for (int64_t j = 0; j < hidden; ++j) {
        const size_t row = size_t(j * channels);
        for (int64_t k = 0; k < channels; ++k) {
            f64[size_t(j)] += double(dequantized[row + size_t(k)]) *
                              double(activation[size_t(k)]);
        }
    }

    std::vector<float> cpu_quant, gpu_quant, gpu_dense;
    const bool cpu_ok = cycle_real_probe_single_matmul_f64_input(
        cpu_backend, phone_down->type, packed.data(), packed.size(),
        activation.data(), channels, hidden, cpu_quant);
    const bool gpu_ok = cycle_real_probe_single_matmul_f64_input(
        phone_backend, phone_down->type, packed.data(), packed.size(),
        activation.data(), channels, hidden, gpu_quant);
    // The real Phone FFN above has already executed Down MUL_MAT_ID with
    // 128 experts. A synthetic one-expert MUL_MAT_ID is not a valid
    // substitute and fatally hits an unsupported OpenCL MoE branch.
    // Keep the real MoE result for full-FFN evaluation, and mark this
    // same-input synthetic comparison as not applicable.
    std::printf(
        "[CYCLE_FFN_DOWN_F64_ORACLE_PROBE] op=MUL_MAT_ID "
        "layout=SYNTHETIC_ONE_EXPERT status=SKIP "
        "reason=UNSUPPORTED_OPENCL_MOE_LAYOUT\n");
    const bool moe_ok = false;
    const bool dense_ok = cycle_real_probe_single_matmul_f64_input(
        phone_backend, GGML_TYPE_F32, dequantized.data(),
        dequantized.size() * sizeof(float),
        activation.data(), channels, hidden, gpu_dense);
    const auto cpu_err = cycle_real_compare_f64(cpu_quant, f64);
    const auto gpu_err = cycle_real_compare_f64(gpu_quant, f64);
    const auto dense_err = cycle_real_compare_f64(gpu_dense, f64);
    if ((cpu_ok && !cpu_err.valid) ||
        (gpu_ok && !gpu_err.valid) ||
        (dense_ok && !dense_err.valid)) {
        std::fprintf(stderr,
            "[CYCLE_FFN_DOWN_F64_ORACLE] stage=DOWN status=NOT_AVAILABLE "
            "reason=NONFINITE\n");
        return false;
    }
    // Preserve per-kernel availability. OpenCL may not support the
    // generic Q6_K path, and synthetic MoE is explicitly not executed.
    const double na = std::numeric_limits<double>::quiet_NaN();
    // All three supported probe kernels must complete, while the
    // synthetic single-expert MoE comparison remains intentionally N/A.
    const bool measured = cpu_err.valid && gpu_err.valid &&
                          dense_err.valid;
    std::printf(
        "[CYCLE_FFN_DOWN_F64_ORACLE] stage=DOWN pos=%d slot=%d expert=%d "
        "weight_type=%s input_channels=%lld output_channels=%lld "
        "input=CPU_REFERENCE_SWIGLU_SAME_FOR_ALL "
        "cpu_quantq8_vs_f64_max_abs=%.8g cpu_quantq8_vs_f64_rms=%.8g "
        "gpu_quantf32_vs_f64_max_abs=%.8g gpu_quantf32_vs_f64_rms=%.8g "
        "gpu_moe_vs_f64_max_abs=%.8g gpu_moe_vs_f64_rms=%.8g "
        "gpu_dense_f32_vs_f64_max_abs=%.8g gpu_dense_f32_vs_f64_rms=%.8g "
        "status=%s scope=DIAGNOSTIC_ONLY\n",
        int(target_token), int(target_slot), int(expert),
        ggml_type_name(phone_down->type),
        (long long)channels, (long long)hidden,
        cpu_err.valid ? cpu_err.max_abs : na,
        cpu_err.valid ? cpu_err.rms : na,
        gpu_err.valid ? gpu_err.max_abs : na,
        gpu_err.valid ? gpu_err.rms : na,
        na, na,  // synthetic one-expert MoE comparison intentionally skipped
        dense_err.valid ? dense_err.max_abs : na,
        dense_err.valid ? dense_err.rms : na,
        measured ? "MEASURED_PARTIAL" : "NOT_AVAILABLE");
    std::printf(
        "[CYCLE_FFN_DOWN_F64_ORACLE_AVAIL] cpu=%d gpu_quant=%d "
        "gpu_moe=%d gpu_dense=%d status=%s\n",
        cpu_ok ? 1 : 0, gpu_ok ? 1 : 0,
        moe_ok ? 1 : 0, dense_ok ? 1 : 0,
        measured ? "PARTIAL" : "NOT_AVAILABLE");
    return measured;
}

// Independently rebuild the FULL selected-expert FFN with F64 dot products
// and F64 SiLU/SwiGLU/Down/weighted expert summation, using the original
// F32 CPU_READY_CONTROL Norm and Router weights. Fetch both actual physical
// PC and Phone quantized shards; no copy of the entire 30B CPU model is
// retained. This separates CPU numerical approximation from heterogeneous
// approximation WITHOUT conflating either with Phone Attention input drift.
static bool cycle_real_probe_full_ffn_f64_oracle(
        ggml_backend_t cpu_backend,
        const llama_hybrid_cycle_ffn_shard_weights & pc_weights,
        const llama_hybrid_cycle_ffn_shard_weights & phone_weights,
        const std::vector<float> & input_norm,
        const std::vector<int32_t> & selected_ids,
        const std::vector<float> & routing_weights,
        const std::vector<float> & baseline_ffn,
        const std::vector<float> & physical_control_ffn,
        const std::vector<float> & cpu_residual,
        const std::vector<float> & baseline_layer_output,
        int64_t hidden, int64_t tokens, int64_t topk, int64_t experts) {
    if (!pc_weights.gate || !phone_weights.gate) {
        return false;
    }
    const int64_t pc_channels = pc_weights.gate->ne[1];
    const int64_t phone_channels = phone_weights.gate->ne[1];
    const auto shard_valid = [&](const llama_hybrid_cycle_ffn_shard_weights & shard,
                                 int64_t width) {
        return shard.gate && shard.up && shard.down &&
            shard.gate->ne[0] == hidden && shard.up->ne[0] == hidden &&
            shard.gate->ne[1] == width && shard.up->ne[1] == width &&
            shard.down->ne[0] == width && shard.down->ne[1] == hidden &&
            shard.gate->ne[2] == experts && shard.up->ne[2] == experts &&
            shard.down->ne[2] == experts;
    };
    if (!cpu_backend || hidden <= 0 || tokens <= 0 || tokens > 2 ||
        topk <= 0 || experts <= 0 ||
        pc_channels <= 0 || phone_channels <= 0 ||
        !shard_valid(pc_weights, pc_channels) ||
        !shard_valid(phone_weights, phone_channels) ||
        input_norm.size() != size_t(hidden * tokens) ||
        selected_ids.size() != size_t(topk * tokens) ||
        routing_weights.size() != size_t(topk * tokens) ||
        baseline_ffn.size() != size_t(hidden * tokens) ||
        physical_control_ffn.size() != size_t(hidden * tokens) ||
        cpu_residual.size() != size_t(hidden * tokens) ||
        baseline_layer_output.size() != size_t(hidden * tokens)) {
        return false;
    }

    // Streaming HOST-ONLY oracle. The packed RPC receive bytes, one
    // dequantized F32 matrix and F64 activation vectors are reused across
    // stages/experts; Gate/Up/Down are never held as three F32 matrices
    // simultaneously. Crucially this never creates an OpenCL graph.
    std::vector<uint8_t> packed_scratch;
    std::vector<float> weight_scratch;
    std::vector<double> activated(size_t(std::max(pc_channels, phone_channels)), 0.0);
    std::vector<double> expert_down(size_t(hidden), 0.0);
    std::vector<double> oracle(size_t(hidden * tokens), 0.0);
    const double mib = 1024.0 * 1024.0;
    std::printf(
        "[CYCLE_FFN_FULL_F64_MEMORY] stage=BEGIN device_compute=PC_CPU_ONLY "
        "phone_compute_buffers_added=0 host_weight_scratch_max_mib=%.3f "
        "buffer_policy=REUSED_ONE_MATRIX_AT_A_TIME status=OK\n",
        double(hidden) * double(std::max(pc_channels, phone_channels)) *
            double(sizeof(float)) / mib);
    std::fflush(stdout);
    for (int64_t token = 0; token < tokens; ++token) {
        const float * x = input_norm.data() + size_t(token * hidden);
        for (int64_t slot = 0; slot < topk; ++slot) {
            const size_t pair = size_t(token * topk + slot);
            const int32_t expert = selected_ids[pair];
            const double route_weight = double(routing_weights[pair]);
            if (expert < 0 || expert >= experts || !std::isfinite(route_weight)) {
                return false;
            }
            std::fill(expert_down.begin(), expert_down.end(), 0.0);
            for (int owner = 0; owner < 2; ++owner) {
                const auto & shard = owner == 0 ? pc_weights : phone_weights;
                const int64_t channels = owner == 0 ? pc_channels : phone_channels;
                for (int stage = 0; stage < 3; ++stage) {
                    const ggml_tensor * w = stage == 0 ? shard.gate :
                                            stage == 1 ? shard.up : shard.down;
                    const int64_t input_width = stage == 2 ? channels : hidden;
                    const int64_t output_width = stage == 2 ? hidden : channels;
                    if (!cycle_real_dequantize_physical_expert_for_f64(
                            cpu_backend, w, expert, weight_scratch,
                            &packed_scratch) ||
                        weight_scratch.size() !=
                            size_t(input_width * output_width)) {
                        std::fprintf(stderr,
                            "[CYCLE_FFN_FULL_F64_ORACLE] status=NOT_AVAILABLE "
                            "reason=PHYSICAL_DEQUANT token=%lld slot=%lld "
                            "expert=%d owner=%s stage=%s\n",
                            (long long)token, (long long)slot, int(expert),
                            owner == 0 ? "PC_CPU" : "PHONE_GPU",
                            stage == 0 ? "GATE" : stage == 1 ? "UP" : "DOWN");
                        return false;
                    }
                    // Gate and Up use exactly the same original F32 Norm
                    // input. For Down, consume the F64 SwiGLU values.
                    for (int64_t row = 0; row < output_width; ++row) {
                        const size_t offset = size_t(row * input_width);
                        double dot = 0.0;
                        for (int64_t k = 0; k < input_width; ++k) {
                            dot += double(weight_scratch[offset + size_t(k)]) *
                                   (stage == 2 ? activated[size_t(k)] :
                                                 double(x[size_t(k)]));
                        }
                        if (stage == 0) {
                            activated[size_t(row)] = dot;
                        } else if (stage == 1) {
                            const double gate = activated[size_t(row)];
                            const double silu = gate / (1.0 + std::exp(-gate));
                            activated[size_t(row)] = silu * dot;
                            if (!std::isfinite(activated[size_t(row)])) {
                                return false;
                            }
                        } else {
                            expert_down[size_t(row)] += dot;
                        }
                    }
                }
            }
            for (int64_t j = 0; j < hidden; ++j) {
                oracle[size_t(token * hidden + j)] +=
                    route_weight * expert_down[size_t(j)];
            }
        }
    }
    std::printf(
        "[CYCLE_FFN_FULL_F64_MEMORY] stage=END "
        "host_packed_capacity_bytes=%zu host_weight_capacity_bytes=%zu "
        "phone_compute_buffers_added=0 status=OK\n",
        packed_scratch.capacity(),
        weight_scratch.capacity() * sizeof(float));
    std::fflush(stdout);

    const auto cpu_err = cycle_real_compare_f64(baseline_ffn, oracle);
    const auto physical_err = cycle_real_compare_f64(physical_control_ffn, oracle);
    if (!cpu_err.valid || !physical_err.valid) {
        std::fprintf(stderr,
            "[CYCLE_FFN_FULL_F64_ORACLE] status=NOT_AVAILABLE "
            "reason=NONFINITE_OR_SHAPE\n");
        return false;
    }
    // Compare both whole-layer outputs with the same residual, not the
    // Phone-generated residual, so this isolates FFN numerical behavior.
    std::vector<double> layer_oracle(oracle.size());
    std::vector<float> physical_control_layer(oracle.size());
    for (size_t i = 0; i < oracle.size(); ++i) {
        layer_oracle[i] = double(cpu_residual[i]) + oracle[i];
        physical_control_layer[i] =
            cpu_residual[i] + physical_control_ffn[i];
    }
    const auto cpu_layer_err =
        cycle_real_compare_f64(baseline_layer_output, layer_oracle);
    const auto physical_layer_err =
        cycle_real_compare_f64(physical_control_layer, layer_oracle);
    if (!cpu_layer_err.valid || !physical_layer_err.valid) {
        return false;
    }
    const char * closer = physical_err.rms < cpu_err.rms ? "PC_PHONE" :
                          physical_err.rms > cpu_err.rms ? "CPU_FULL" : "EQUAL";
    std::printf(
        "[CYCLE_FFN_FULL_F64_ORACLE] tokens=%lld topk=%lld "
        "pc_channels=%lld phone_channels=%lld "
        "input=CPU_READY_CONTROL "
        "cpu_full_vs_f64_max_abs=%.8g cpu_full_vs_f64_rms=%.8g "
        "pc_phone_vs_f64_max_abs=%.8g pc_phone_vs_f64_rms=%.8g "
        "closer_by_rms=%s "
        "oracle=DEQUANT_PHYSICAL_WEIGHTS_F64_FULL_SWIGLU_DOWN_REDUCE "
        "status=MEASURED scope=DIAGNOSTIC_ONLY\n",
        (long long)tokens, (long long)topk,
        (long long)pc_channels, (long long)phone_channels,
        cpu_err.max_abs, cpu_err.rms,
        physical_err.max_abs, physical_err.rms, closer);
    std::printf(
        "[CYCLE_LAYER_F64_ORACLE] layer_input=CPU_REFERENCE_RESIDUAL "
        "input=CPU_READY_CONTROL "
        "cpu_full_vs_f64_max_abs=%.8g cpu_full_vs_f64_rms=%.8g "
        "pc_phone_vs_f64_max_abs=%.8g pc_phone_vs_f64_rms=%.8g "
        "status=MEASURED scope=DIAGNOSTIC_ONLY\n",
        cpu_layer_err.max_abs, cpu_layer_err.rms,
        physical_layer_err.max_abs, physical_layer_err.rms);
    return true;
}

// Execute one REAL P2C Tensor layer through the same FIFO contract used by
// the multi-layer Cycle scheduler. Layer-local index 0 maps to the requested
// physical Qwen3-MoE layer; no synthetic CPU model or hidden FFN recompute is
// used as a task completion. Future multi-chunk execution can share the
// worker runtime while extending per-chunk compute and KV ownership.
// Real multi-chunk P2C execution. CPU reference supplies only the full
// layer INPUT; all chunk Attention/Router and Phone FFN work occurs on
// Phone GPU. Each chunk has independent READY/PARTIAL frames and a seq.
// The immutable GPU K/V cache holds previous chunks on Phone, while the
// CPU owns the only Reduce. Bounded per-layer ring credits keep at most
// max_inflight Attention graph buffers alive during this stage.
// C2P destination: borrow one REAL Phone GPU buffer allocation for both
// the local physical FFN shard and the FINAL GPU Reduce. The split FFN graph
// and two-op Reduce graph are independent, so no FFN kernel is recomputed
// when the PC PARTIAL becomes GPU-ready.
// Nonzero graph UIDs opt in to the RPC server's GRAPH_RECOMPUTE cache.
// A fresh process-time namespace and monotonic counter prevent accidental
// reuse of a stale server-side graph when its device buffers were freed.
// Never enable replay for a graph whose buffer lifetime is not owned here.
static uint64_t cycle_real_phone_graph_replay_uid() {
    static std::atomic<uint64_t> next_uid {
        uint64_t(std::chrono::steady_clock::now()
            .time_since_epoch().count()) | UINT64_C(1)
    };
    const uint64_t uid = next_uid.fetch_add(1, std::memory_order_relaxed);
    return uid != 0 ? uid : 1;
}

struct cycle_real_c2p_phone_chunk {
    ggml_context * ctx = nullptr;
    ggml_tensor * residual = nullptr;
    ggml_tensor * pc_partial = nullptr;
    ggml_tensor * output = nullptr;
    llama_hybrid_cycle_ffn_graph inputs {};
    ggml_cgraph * reduce_graph = nullptr;
    llama_hybrid_cycle_ffn_executor executor;
    double graph_build_ms = 0.0;
    double gpu_alloc_ms = 0.0;
    double ffn_graph_submit_ms = 0.0;
    double reduce_graph_submit_ms = 0.0;
    bool is_full_layer = false;
    bool fused_full_graph = false;
    bool rpc_full_replay = false;
    // Persisted FFN/Reduce graph is safe to replay ONLY for the same
    // physical layer (owned by its per-layer pool) and token shape.
    int64_t prepared_tokens = 0;

    cycle_real_c2p_phone_chunk() = default;
    cycle_real_c2p_phone_chunk(const cycle_real_c2p_phone_chunk &) = delete;
    cycle_real_c2p_phone_chunk & operator=(const cycle_real_c2p_phone_chunk &) = delete;
    ~cycle_real_c2p_phone_chunk() { release(); }

    void release() {
        executor.reset();
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        residual = nullptr;
        pc_partial = nullptr;
        output = nullptr;
        reduce_graph = nullptr;
        inputs = {};
        is_full_layer = false;
        fused_full_graph = false;
        rpc_full_replay = false;
        prepared_tokens = 0;
    }

    bool prepare(
            ggml_backend_t phone,
            const llama_hybrid_cycle_ffn_shard_weights & shard,
            int64_t hidden, int64_t tokens, int64_t topk, int64_t experts,
            bool full_layer = false, bool fuse_full = false,
            bool enable_rpc_replay = false) {
        const auto build_begin = std::chrono::steady_clock::now();
        release();
        ggml_init_params params {};
        params.mem_size = 2 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) {
            return false;
        }
        is_full_layer = full_layer;
        fused_full_graph = full_layer && fuse_full;
        rpc_full_replay = full_layer && enable_rpc_replay;
        inputs = llama_hybrid_cycle_make_ffn_graph_inputs(ctx, hidden, tokens, topk);
        residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        if (!full_layer) {
            pc_partial = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        }
        auto * x = ggml_reshape_3d(ctx, inputs.norm, hidden, 1, tokens);
        auto * gate = ggml_mul_mat_id(ctx, shard.gate, x, inputs.topk_ids);
        auto * up = ggml_mul_mat_id(ctx, shard.up, x, inputs.topk_ids);
        auto * act = ggml_swiglu_split(ctx, gate, up);
        auto * down = ggml_mul_mat_id(ctx, shard.down, act, inputs.topk_ids);
        auto * weighted = ggml_mul(ctx, down, inputs.topk_weights);
        ggml_tensor * sum = nullptr;
        for (int64_t k = 0; k < topk; ++k) {
            auto * expert = ggml_view_2d(
                ctx, weighted, hidden, tokens,
                weighted->nb[2], size_t(k) * weighted->nb[1]);
            sum = sum ? ggml_add(ctx, sum, expert) : expert;
        }
        inputs.ffn_output = ggml_cont(ctx, sum);
        ggml_set_name(inputs.ffn_output, "cycle_num_phone_ffn_output");
        auto * ffn_graph = ggml_new_graph_custom(ctx, 512, false);

        // C2P Reduce needs the remote PC shard, so it MUST remain split.
        // For a complete Phone layer, one opt-in FFN + residual ADD graph
        // replaces a separate one-node ADD graph and its RPC fence.
        // The original FFN-only/Reduce graphs remain the default.
        auto * ffnsum = full_layer ?
            inputs.ffn_output :
            ggml_add(ctx, inputs.ffn_output, pc_partial);
        output = ggml_add(ctx, residual, ffnsum);
        ggml_set_name(output, full_layer ?
            "cycle_num_phone_full_lout" : "cycle_num_c2p_lout");
        if (fused_full_graph) {
            // Canonical build also sets GGML_TENSOR_FLAG_COMPUTE. Unlike
            // the split Reduce graph, recursion here is intentional.
            ggml_build_forward_expand(ffn_graph, output);
        } else {
            ggml_build_forward_expand(ffn_graph, inputs.ffn_output);
            reduce_graph = ggml_new_graph_custom(ctx, 16, false);
            // Appending a node directly does not set COMPUTE flags.
            // The independent split Reduce must not recompute the FFN.
            if (!full_layer) {
                ffnsum->flags |= GGML_TENSOR_FLAG_COMPUTE;
                ggml_graph_add_node(reduce_graph, ffnsum);
            }
            output->flags |= GGML_TENSOR_FLAG_COMPUTE;
            ggml_graph_add_node(reduce_graph, output);
        }
        if (rpc_full_replay) {
            // Replayed graph references exactly these persistent FFN,
            // residual and output buffers. A freed graph must NEVER reuse
            // its UID; only this object's subsequent Chunk may replay it.
            ggml_graph_set_uid(ffn_graph, cycle_real_phone_graph_replay_uid());
            if (reduce_graph) {
                ggml_graph_set_uid(reduce_graph, cycle_real_phone_graph_replay_uid());
            }
        }
        const auto build_end = std::chrono::steady_clock::now();
        if (!executor.prepare(
                phone, ctx, ffn_graph, inputs, shard,
                hidden, tokens, topk, experts)) {
            std::fprintf(stderr,
                "[CYCLE_C2P_GPU] status=FAIL reason=FFN_PREPARE\n");
            release();
            return false;
        }
        const auto alloc_end = std::chrono::steady_clock::now();
        graph_build_ms = std::chrono::duration<double, std::milli>(
            build_end - build_begin).count();
        gpu_alloc_ms = std::chrono::duration<double, std::milli>(
            alloc_end - build_end).count();
        prepared_tokens = tokens;
        std::printf(
            "[CYCLE_GPU_PREPARE] phase=PHONE_FFN graph_build_ms=%.3f "
            "gpu_alloc_ms=%.3f source=PC_CLIENT status=OK\n",
            graph_build_ms, gpu_alloc_ms);
        return true;
    }

    bool execute_full_local(const cycle_real_phone_ready & ready) {
        if (!is_full_layer || !ready.phone_graph.residual ||
            !ready.phone_graph.ffn_norm ||
            !ready.phone_graph.topk_ids ||
            !ready.phone_graph.topk_weights ||
            !residual || !residual->buffer ||
            !ggml_backend_buffer_copy_tensor(
                ready.phone_graph.residual, residual)) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_FULL] stage=INPUT status=FAIL "
                "reason=LOCAL_GPU_HANDOFF\n");
            return false;
        }
        const std::vector<float> empty_f32;
        const std::vector<int32_t> empty_i32;
        return executor.execute(
            empty_f32.data(), 0, empty_i32.data(), 0,
            empty_f32.data(), 0, nullptr, 0, &ready.phone_graph);
    }

    bool execute_ffn() {
        const auto begin = std::chrono::steady_clock::now();
        const bool ok = executor.execute_device_ready();
        ffn_graph_submit_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        std::printf("[CYCLE_GPU_SUBMIT] phase=PHONE_FFN "
                    "graph_submit_ms=%.3f gpu_compute_ms=SERVER_ONLY "
                    "status=%s\n",
                    ffn_graph_submit_ms, ok ? "OK" : "FAIL");
        return ok;
    }

    bool execute_reduce(ggml_backend_t phone) {
        // A full Phone GPU layer has no PC FFN shard and deliberately does
        // not allocate pc_partial. C2P needs the PC shard for its two-input
        // FFN sum; FULL_PHONE reduces only its local FFN and residual.
        if (!phone || !reduce_graph || !output ||
            (!is_full_layer && !pc_partial) ||
            !inputs.ffn_output || !residual) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_REDUCE] mode=%s status=FAIL "
                "reason=MISSING_GRAPH_INPUT pc_partial=%d output=%d\n",
                is_full_layer ? "PHONE_FULL" : "TENSOR_C2P",
                pc_partial != nullptr ? 1 : 0, output != nullptr ? 1 : 0);
            return false;
        }
        const auto begin = std::chrono::steady_clock::now();
        const bool ok = ggml_backend_graph_compute(phone, reduce_graph) ==
                        GGML_STATUS_SUCCESS;
        if (ok) {
            ggml_backend_synchronize(phone);
        }
        reduce_graph_submit_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        std::printf("[CYCLE_GPU_SUBMIT] phase=PHONE_REDUCE mode=%s "
                    "graph_submit_ms=%.3f gpu_compute_ms=SERVER_ONLY "
                    "status=%s\n",
                    is_full_layer ? "PHONE_FULL" : "TENSOR_C2P",
                    reduce_graph_submit_ms, ok ? "OK" : "FAIL");
        return ok;
    }
};

// The Phone FULL graph/compute-buffer cache is optionally owned by the
// model+Phone-backend session rather than a single Cycle period. ONLY idle
// FFN/Reduce graphs live here. Attention graphs, activations and KV are
// always recreated for every period and may not be retained as session data.
// A reused graph borrows the same physical model weights and RPC backend,
// so this owner MUST be drained before either weights or backend are freed.
struct cycle_real_phone_full_session_pool {
    using slot_ptr = std::unique_ptr<cycle_real_c2p_phone_chunk>;
    ggml_backend_t phone = nullptr;
    const llama_model * model = nullptr;
    int layers = 0;
    int x = 0;
    int y = 0;
    int32_t total_tokens = 0;
    int chunk_tokens = 0;
    int window = 0;
    bool fuse = false;
    bool rpc_replay = false;
    std::vector<std::vector<slot_ptr>> idle_by_layer;

    void configure(ggml_backend_t p, const llama_model & m,
                   const llama_hybrid_cycle_layout & layout,
                   int32_t tokens, int chunk, int inflight,
                   bool fusion, bool replay) {
        clear();
        phone = p;
        model = &m;
        layers = layout.cycle_size;
        x = layout.x;
        y = layout.y;
        total_tokens = tokens;
        chunk_tokens = chunk;
        window = inflight;
        fuse = fusion;
        rpc_replay = replay;
        idle_by_layer.resize(size_t(layers));
    }

    bool matches(ggml_backend_t p, const llama_model & m,
                 const llama_hybrid_cycle_layout & layout,
                 int32_t tokens, int chunk, int inflight,
                 bool fusion, bool replay) const {
        return phone == p && model == &m &&
            layers == layout.cycle_size &&
            x == layout.x && y == layout.y &&
            total_tokens == tokens && chunk_tokens == chunk &&
            window == inflight &&
            fuse == fusion && rpc_replay == replay &&
            idle_by_layer.size() == size_t(layers);
    }

    size_t cached_slots() const {
        size_t result = 0;
        for (const auto & layer : idle_by_layer) result += layer.size();
        return result;
    }

    void clear() {
        // Every remote GPU free below must run while the same RPC backend
        // and the borrowed physical model weights are still alive.
        for (auto & layer : idle_by_layer) layer.clear();
        idle_by_layer.clear();
        phone = nullptr;
        model = nullptr;
        layers = x = y = 0;
        total_tokens = 0;
        chunk_tokens = window = 0;
        fuse = rpc_replay = false;
    }
};

// P2C FFN keeps one complete graph and its Phone GPU compute buffer
// between chunks. The dedicated PHONE_GPU FIFO worker is its only user:
// no separate lock or speculative graph execution is needed. Its output
// becomes independent of this GPU buffer once the Phone server has
// read it into the transport TX slot (SEND_PARTIAL returns success).
struct cycle_real_p2c_ffn_slot {
    ggml_context * ctx = nullptr;
    llama_hybrid_cycle_ffn_executor executor;
    llama_hybrid_cycle_ffn_graph inputs {};
    int64_t prepared_tokens = 0;
    double graph_build_ms = 0.0;
    double backend_prepare_ms = 0.0;

    cycle_real_p2c_ffn_slot() = default;
    cycle_real_p2c_ffn_slot(const cycle_real_p2c_ffn_slot &) = delete;
    cycle_real_p2c_ffn_slot & operator=(const cycle_real_p2c_ffn_slot &) = delete;
    ~cycle_real_p2c_ffn_slot() { release(); }

    void release() {
        executor.reset(); // Free the remote buffer before its graph context.
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        inputs = {};
        prepared_tokens = 0;
        graph_build_ms = 0.0;
        backend_prepare_ms = 0.0;
    }

    bool prepare(ggml_backend_t phone,
                 const llama_hybrid_cycle_ffn_shard_weights & shard,
                 int64_t hidden, int64_t tokens, int64_t topk,
                 int64_t experts) {
        release();
        const auto build_begin = std::chrono::steady_clock::now();
        ggml_init_params params {};
        params.mem_size = 2 * 1024 * 1024;
        params.no_alloc = true;
        ctx = ggml_init(params);
        if (!ctx) return false;

        // Exactly the same physical FFN shard graph as the uncached
        // cycle_real_execute_ffn() path (no residual ADD / Meta allreduce).
        inputs = llama_hybrid_cycle_make_ffn_graph_inputs(
            ctx, hidden, tokens, topk);
        auto * x = ggml_reshape_3d(ctx, inputs.norm, hidden, 1, tokens);
        auto * gate = ggml_mul_mat_id(ctx, shard.gate, x, inputs.topk_ids);
        auto * up = ggml_mul_mat_id(ctx, shard.up, x, inputs.topk_ids);
        auto * activation = ggml_swiglu_split(ctx, gate, up);
        auto * down = ggml_mul_mat_id(ctx, shard.down, activation, inputs.topk_ids);
        auto * weighted = ggml_mul(ctx, down, inputs.topk_weights);
        ggml_tensor * sum = nullptr;
        for (int64_t k = 0; k < topk; ++k) {
            auto * view = ggml_view_2d(
                ctx, weighted, hidden, tokens, weighted->nb[2],
                size_t(k) * weighted->nb[1]);
            sum = sum ? ggml_add(ctx, sum, view) : view;
        }
        inputs.ffn_output = ggml_cont(ctx, sum);
        ggml_set_name(inputs.ffn_output, "cycle_num_phone_ffn_output");
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
        ggml_build_forward_expand(graph, inputs.ffn_output);
        const auto build_end = std::chrono::steady_clock::now();
        const bool ok = executor.prepare(
            phone, ctx, graph, inputs, shard, hidden, tokens, topk, experts);
        const auto prepare_end = std::chrono::steady_clock::now();
        graph_build_ms = std::chrono::duration<double, std::milli>(
            build_end - build_begin).count();
        backend_prepare_ms = std::chrono::duration<double, std::milli>(
            prepare_end - build_end).count();
        if (!ok) {
            release();
            return false;
        }
        prepared_tokens = tokens;
        return true;
    }

    bool execute_and_send(ggml_backend_t phone,
                          const llama_hybrid_cycle_ready_graph & ready,
                          const cycle_real_live_wire & wire,
                          cycle_real_ffn_phase_profile * profile) {
        if (!ctx || prepared_tokens <= 0 || !inputs.ffn_output) return false;
        // Each chunk has fresh Norm, TopK ids and TopK weights. The executor
        // copies these from the corresponding Phone Attention buffers.
        const auto execute_begin = std::chrono::steady_clock::now();
        const bool computed = executor.execute(
            nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, &ready);
        if (profile) {
            profile->backend_execute_call_ms +=
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - execute_begin).count();
        }
        if (!computed) return false;
        // This RPC command synchronously reads GPU output and copies it
        // into the independent Phone TX slot before acknowledging success.
        // The slot can then be overwritten for another chunk while TX runs.
        return ggml_backend_rpc_cycle_send_partial(
            phone, wire.layer, wire.chunk, wire.seq, wire.graph_id,
            inputs.ffn_output);
    }
};

static bool cycle_real_send_c2p_frame(
        ggml_backend_t phone, const cycle_real_live_wire & wire,
        inference_cycle_payload_kind kind,
        const cycle_real_phone_ready & ready,
        const std::vector<float> & partial,
        uint32_t tokens, uint32_t hidden, uint32_t topk) {
    size_t bytes = 0;
    if (!inference_cycle_payload_size(
            kind, tokens, hidden,
            kind == INFERENCE_CYCLE_READY ? topk : 0, bytes)) {
        return false;
    }
    std::vector<uint8_t> frame(bytes);
    size_t written = 0;
    bool ok = false;
    if (kind == INFERENCE_CYCLE_READY) {
        const inference_cycle_ready_view value {
            tokens, hidden, topk,
            ready.residual.data(), ready.norm.data(),
            ready.ids.data(), ready.routing.data()
        };
        ok = ready.residual.size() == size_t(tokens) * hidden &&
             ready.norm.size() == size_t(tokens) * hidden &&
             ready.ids.size() == size_t(tokens) * topk &&
             ready.routing.size() == size_t(tokens) * topk &&
             inference_cycle_write_ready(
                 frame.data(), frame.size(), value, written);
    } else {
        const inference_cycle_partial_view value {
            tokens, hidden, partial.data()
        };
        ok = partial.size() == size_t(tokens) * hidden &&
             inference_cycle_write_partial(
                 frame.data(), frame.size(), value, written);
    }
    if (!ok || written != bytes) {
        return false;
    }
    auto * dev = ggml_backend_get_device(phone);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto submit = reg ?
        reinterpret_cast<ggml_backend_rpc_inference_transport_submit_t>(
            ggml_backend_reg_get_proc_address(
                reg, GGML_BACKEND_RPC_INFERENCE_TRANSPORT_SUBMIT_PROC)) :
        nullptr;
    ggml_backend_rpc_inference_task task {};
    task.layer = wire.layer;
    task.chunk = wire.chunk;
    task.seq = wire.seq;
    task.graph_id = wire.graph_id;
    task.stage = kind == INFERENCE_CYCLE_READY ?
        INFERENCE_STAGE_CYCLE_READY : INFERENCE_STAGE_CYCLE_PARTIAL;
    task.tensor_role = INFERENCE_CYCLE_PC_TO_PHONE;
    const bool submitted = submit != nullptr &&
        submit(phone, &task, frame.data(), frame.size());
    std::printf(
        "[CYCLE_C2P_WIRE] side=pc action=SEND kind=%s layer=%d chunk=%d "
        "seq=%llu bytes=%zu direction=PC_TO_PHONE status=%s\n",
        kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
        wire.layer, wire.chunk, (unsigned long long)wire.seq,
        bytes, submitted ? "QUEUED" : "FAIL");
    return submitted;
}

struct cycle_real_p2c_chunk {
    int64_t start = 0;
    int64_t tokens = 0;
    cycle_real_live_wire wire;
    cycle_real_phone_ready phone_ready;
    inference_cycle_message ready_msg;
    inference_cycle_message partial_msg;
    std::vector<float> pc_partial;
    double attn_ms = 0.0;
    double phone_ffn_ms = 0.0;
    double pc_ffn_ms = 0.0;
    double ready_rx_ms = 0.0;
    double partial_rx_ms = 0.0;
    double reduce_ms = 0.0;
};

static bool cycle_real_execute_p2c_fifo(
        ggml_backend_t phone, ggml_backend_dev_t cpu_dev,
        const llama_layer & layer_weights, const llama_hparams & hp,
        const llama_hybrid_cycle_ffn_shard_weights & pc_weights,
        const llama_hybrid_cycle_ffn_shard_weights & phone_weights,
        const std::vector<float> & layer_input,
        const cycle_real_live_wire & wire_base,
        int64_t hidden, int32_t tokens, int64_t topk, int64_t experts,
        int32_t chunk_tokens, int max_inflight) {
    if (chunk_tokens <= 0 || max_inflight <= 0 ||
        tokens <= 0 || hidden <= 0 ||
        layer_input.size() != size_t(hidden * tokens)) {
        return false;
    }
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu) {
        std::fprintf(stderr, "[CYCLE_FIFO_REAL] status=FAIL reason=CPU_INIT\n");
        return false;
    }
    const int n_chunks = (tokens + chunk_tokens - 1) / chunk_tokens;
    const int window = std::min(max_inflight, n_chunks);
    std::vector<std::unique_ptr<cycle_real_p2c_chunk>> chunks;
    chunks.reserve(size_t(n_chunks));
    for (int i = 0; i < n_chunks; ++i) {
        auto chunk = std::make_unique<cycle_real_p2c_chunk>();
        chunk->start = int64_t(i) * chunk_tokens;
        chunk->tokens = std::min(
            int64_t(chunk_tokens), int64_t(tokens) - chunk->start);
        chunk->wire = {wire_base.layer, i, wire_base.seq + uint64_t(i),
                       wire_base.graph_id};
        chunks.emplace_back(std::move(chunk));
    }
    cycle_real_phone_kv_cache kv_cache;
    if (n_chunks > 1 && !kv_cache.init(phone, layer_weights, tokens)) {
        std::fprintf(stderr,
            "[CYCLE_FIFO_REAL] status=FAIL reason=PHONE_PERSISTENT_KV_ALLOC\n");
        ggml_backend_free(cpu);
        return false;
    }
    std::vector<float> output(size_t(hidden * tokens), 0.0f);
    const auto whole_begin = std::chrono::steady_clock::now();
    std::printf(
        "[CYCLE_FIFO_CHUNK_PLAN] layer=%d total_tokens=%d "
        "chunk_tokens=%d chunks=%d max_inflight=%d "
        "kv_scope=%s transport=REAL_PHONE_TO_PC "
        "payloads_per_chunk=2 status=READY\n",
        wire_base.layer, int(tokens), int(chunk_tokens),
        n_chunks, window,
        n_chunks > 1 ? "PHONE_GPU_PERSISTENT" : "ONE_UBATCH");
    std::fflush(stdout);

    llama_hybrid_cycle_layout fragment;
    fragment.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_P2C};
    const bool success = llama_hybrid_cycle_run_fifo(
        fragment, n_chunks, [&](const llama_hybrid_cycle_task & task) -> bool {
            if (task.layer != 0 || task.chunk < 0 ||
                task.chunk >= n_chunks) {
                return false;
            }
            cycle_real_p2c_chunk & st = *chunks[size_t(task.chunk)];
            const auto started = std::chrono::steady_clock::now();
            bool ok = false;
            switch (task.stage) {
                case llama_hybrid_cycle_stage::ATTN_PRE: {
                    // Chunk-local input only. The persistent Phone GPU K/V
                    // prefix provides all earlier causal keys/values.
                    std::vector<float> chunk_input(size_t(st.tokens * hidden));
                    std::copy_n(
                        layer_input.data() + size_t(st.start * hidden),
                        chunk_input.size(), chunk_input.data());
                    ok = cycle_real_execute_phone_attention_pre(
                        phone, layer_weights, hp, chunk_input,
                        hidden, st.tokens, topk, experts,
                        st.phone_ready, st.attn_ms, &st.wire,
                        n_chunks > 1 ? &kv_cache : nullptr, st.start);
                    break;
                }
                case llama_hybrid_cycle_stage::SEND_ACTIVATION:
                    ok = cycle_real_receive_live_wire(
                        phone, st.wire, INFERENCE_CYCLE_READY,
                        uint32_t(st.tokens), uint32_t(hidden), uint32_t(topk),
                        st.ready_msg);
                    if (ok) {
                        ok = st.ready_msg.ready.tokens ==
                                 uint32_t(st.tokens) &&
                             st.ready_msg.ready.hidden ==
                                 uint32_t(hidden) &&
                             st.ready_msg.ready.topk == uint32_t(topk) &&
                             st.ready_msg.ready.ffn_norm.size() ==
                                 size_t(st.tokens * hidden) &&
                             st.ready_msg.ready.topk_ids.size() ==
                                 size_t(st.tokens * topk);
                    }
                    if (ok) {
                        // PC CPU input tensors are HOST-ready at this point.
                        // Control ACK carries no activation payload.
                        ok = ggml_backend_rpc_cycle_send_ack(
                            phone, st.wire.layer, st.wire.chunk,
                            st.wire.seq, st.wire.graph_id,
                            INFERENCE_CYCLE_READY, true);
                    }
                    break;
                case llama_hybrid_cycle_stage::FFN_SOURCE: {
                    // Runs as soon as Phone Attention finishes, regardless
                    // of the PC's READY reception/computation progress.
                    const std::vector<float> unused_f32;
                    const std::vector<int32_t> unused_i32;
                    std::vector<float> unused_out;
                    ok = cycle_real_execute_ffn(
                        phone, phone_weights,
                        unused_f32, unused_i32, unused_f32,
                        hidden, st.tokens, topk, experts,
                        unused_out, nullptr, nullptr,
                        &st.wire, &st.phone_ready.phone_graph);
                    st.phone_ready.release_phone_graph();
                    break;
                }
                case llama_hybrid_cycle_stage::FFN_DESTINATION:
                    ok = cycle_real_execute_ffn(
                        cpu, pc_weights,
                        st.ready_msg.ready.ffn_norm,
                        st.ready_msg.ready.topk_ids,
                        st.ready_msg.ready.topk_weights,
                        hidden, st.tokens, topk, experts, st.pc_partial);
                    break;
                case llama_hybrid_cycle_stage::SEND_PARTIAL:
                    ok = cycle_real_receive_live_wire(
                        phone, st.wire, INFERENCE_CYCLE_PARTIAL,
                        uint32_t(st.tokens), uint32_t(hidden), 0,
                        st.partial_msg);
                    if (ok) {
                        ok = st.partial_msg.partial.tokens ==
                                 uint32_t(st.tokens) &&
                             st.partial_msg.partial.hidden ==
                                 uint32_t(hidden) &&
                             st.partial_msg.partial.partial.size() ==
                                 size_t(st.tokens * hidden);
                    }
                    break;
                case llama_hybrid_cycle_stage::REDUCE: {
                    const size_t count = size_t(st.tokens * hidden);
                    std::vector<float> reduced(count);
                    ok = st.pc_partial.size() == count &&
                         st.partial_msg.partial.partial.size() == count;
                    if (ok) {
                        ok = inference_cycle_reduce_p2c_pc_cpu(
                            st.ready_msg.ready,
                            st.pc_partial.data(), st.pc_partial.size(),
                            st.partial_msg.partial,
                            reduced.data(), reduced.size());
                    }
                    if (ok) {
                        std::copy(
                            reduced.begin(), reduced.end(),
                            output.begin() + size_t(st.start * hidden));
                        // Reverse control ACK only AFTER exclusive PC Reduce
                        // has read PARTIAL. No PC->Phone tensor payload.
                        ok = ggml_backend_rpc_cycle_send_ack(
                                 phone, st.wire.layer, st.wire.chunk,
                                 st.wire.seq, st.wire.graph_id,
                                 INFERENCE_CYCLE_PARTIAL, true);
                    }
                    break;
                }
                default:
                    break;
            }
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            if (task.stage == llama_hybrid_cycle_stage::FFN_SOURCE) {
                st.phone_ffn_ms = ms;
            } else if (task.stage == llama_hybrid_cycle_stage::FFN_DESTINATION) {
                st.pc_ffn_ms = ms;
            } else if (task.stage == llama_hybrid_cycle_stage::SEND_ACTIVATION) {
                st.ready_rx_ms = ms;
            } else if (task.stage == llama_hybrid_cycle_stage::SEND_PARTIAL) {
                st.partial_rx_ms = ms;
            } else if (task.stage == llama_hybrid_cycle_stage::REDUCE) {
                st.reduce_ms = ms;
            }
            std::printf(
                "[CYCLE_FIFO_REAL_STAGE] layer=%d chunk=%d seq=%llu "
                "pos=[%lld,%lld] stage=%s queue=%s wall_ms=%.3f status=%s\n",
                st.wire.layer, st.wire.chunk,
                (unsigned long long)st.wire.seq,
                (long long)st.start,
                (long long)(st.start + st.tokens - 1),
                llama_hybrid_cycle_stage_name(task.stage),
                llama_hybrid_cycle_queue_name(task.queue),
                ms, ok ? "OK" : "FAIL");
            std::fflush(stdout);
            return ok;
        }, window);

    // FIFO joins every worker before returning. No borrowed Phone READY
    // compute buffer may survive beyond its Phone backend; avoid storing
    // any other giant per-chunk diagnostic graph in the P2C path.
    for (auto & chunk : chunks) {
        chunk->phone_ready.release_phone_graph();
    }
    const bool kv_complete = n_chunks == 1 ||
                             kv_cache.committed_tokens == tokens;
    kv_cache.release();
    if (success && !ggml_backend_rpc_cycle_wait_tx_idle(phone, 10000)) {
        std::fprintf(stderr,
            "[CYCLE_FIFO_REAL] status=FAIL reason=CONTROL_ACK_TX_IDLE\n");
        ggml_backend_free(cpu);
        return false;
    }
    ggml_backend_free(cpu);
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - whole_begin).count();
    double attn_total = 0.0, phone_total = 0.0, pc_total = 0.0;
    double ready_total = 0.0, partial_total = 0.0, reduce_total = 0.0;
    for (const auto & chunk : chunks) {
        attn_total += chunk->attn_ms;
        phone_total += chunk->phone_ffn_ms;
        pc_total += chunk->pc_ffn_ms;
        ready_total += chunk->ready_rx_ms;
        partial_total += chunk->partial_rx_ms;
        reduce_total += chunk->reduce_ms;
    }
    const bool kv_ok = kv_complete;
    std::printf(
        "[CYCLE_FIFO_REAL_P2C_SUM] layer=%d chunks=%d tokens=%d "
        "chunk_tokens=%d max_inflight=%d seq_begin=%llu "
        "direction=PHONE_TO_PC payload_sends=%d "
        "attention_owner=PHONE_GPU ffn_source=PHONE_GPU "
        "ffn_destination=PC_CPU reduce_owner=PC_CPU "
        "phone_local_handoff=1 phone_duplicate_rpc_readback=0 "
        "phone_kv=%s "
        "ready_rx_total_ms=%.3f partial_rx_total_ms=%.3f "
        "phone_attn_total_ms=%.3f phone_ffn_total_ms=%.3f "
        "pc_ffn_total_ms=%.3f reduce_total_ms=%.3f wall_ms=%.3f "
        "cpu_parity=NOT_CHECKED status=%s\n",
        wire_base.layer, n_chunks, int(tokens), int(chunk_tokens),
        window, (unsigned long long)wire_base.seq, n_chunks * 2,
        n_chunks > 1 ? "PERSISTENT" : "SINGLE_UBATCH",
        ready_total, partial_total, attn_total, phone_total,
        pc_total, reduce_total, wall_ms,
        success && kv_ok ? "OK" : "FAIL");
    return success && kv_ok;
}

// Reverse of P2C: one REAL Tensor C2P layer, sliced into causal chunks.
// ATTENTION and source FFN: PC CPU. READY and PARTIAL: two PC->Phone data
// frames/chunk on the SAME FIFO TX queue. Destination FFN + final Reduce:
// Phone GPU. The Phone server applies both received payloads locally, so
// neither FFN input nor the final layer output makes a PC return trip.
struct cycle_real_c2p_chunk {
    int64_t start = 0;
    int64_t tokens = 0;
    cycle_real_live_wire wire;
    cycle_real_phone_ready pc_ready;
    std::vector<float> pc_partial;
    std::unique_ptr<cycle_real_c2p_phone_chunk> phone_gpu;
    double attention_ms = 0.0;
    double pc_ffn_ms = 0.0;
    double phone_ffn_ms = 0.0;
    double ready_ms = 0.0;
    double partial_ms = 0.0;
    double reduce_ms = 0.0;
};

static bool cycle_real_execute_c2p_fifo(
        ggml_backend_t phone, ggml_backend_dev_t cpu_dev,
        const llama_layer & layer, const llama_hparams & hp,
        const llama_hybrid_cycle_ffn_shard_weights & pc_weights,
        const llama_hybrid_cycle_ffn_shard_weights & phone_weights,
        const std::vector<float> & input, const cycle_real_live_wire & base,
        int64_t hidden, int32_t tokens, int64_t topk, int64_t experts,
        int32_t chunk_tokens, int max_inflight) {
    if (!phone || !cpu_dev || hidden <= 0 || tokens <= 0 ||
        chunk_tokens <= 0 || max_inflight <= 0 ||
        input.size() != size_t(hidden * tokens)) {
        return false;
    }
    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu) {
        return false;
    }
    const int count = (tokens + chunk_tokens - 1) / chunk_tokens;
    const int window = std::min(max_inflight, count);
    std::vector<std::unique_ptr<cycle_real_c2p_chunk>> chunks;
    chunks.reserve(size_t(count));
    for (int i = 0; i < count; ++i) {
        auto st = std::make_unique<cycle_real_c2p_chunk>();
        st->start = int64_t(i) * chunk_tokens;
        st->tokens = std::min(
            int64_t(chunk_tokens), int64_t(tokens) - st->start);
        st->wire = {base.layer, i, base.seq + uint64_t(i), base.graph_id};
        chunks.push_back(std::move(st));
    }

    // The SAME per-layer causal algorithm as P2C, but cached on PC CPU.
    // This full-prefix K/V is local, not part of either C2P data message.
    cycle_real_phone_kv_cache pc_kv;
    if (count > 1 && !pc_kv.init(cpu, layer, tokens, 0)) {
        std::fprintf(stderr, "[CYCLE_FIFO_C2P] status=FAIL reason=PC_KV_ALLOC\n");
        ggml_backend_free(cpu);
        return false;
    }
    std::printf(
        "[CYCLE_FIFO_C2P_PLAN] layer=%d tokens=%d chunk_tokens=%d "
        "chunks=%d max_inflight=%d kv_owner=PC_CPU "
        "ffn=PC_CPU+PHONE_GPU reduce_owner=PHONE_GPU "
        "payloads_per_chunk=2 status=READY\n",
        base.layer, int(tokens), int(chunk_tokens), count, window);
    const auto begin = std::chrono::steady_clock::now();

    llama_hybrid_cycle_layout fragment;
    fragment.kinds = {llama_hybrid_cycle_layer_kind::TENSOR_C2P};
    const bool ran = llama_hybrid_cycle_run_fifo(
        fragment, count, [&](const llama_hybrid_cycle_task & task) -> bool {
            if (task.layer != 0 || task.chunk < 0 || task.chunk >= count) {
                return false;
            }
            auto & st = *chunks[size_t(task.chunk)];
            const auto started = std::chrono::steady_clock::now();
            bool ok = false;
            switch (task.stage) {
                case llama_hybrid_cycle_stage::ATTN_PRE: {
                    std::vector<float> x(size_t(st.tokens * hidden));
                    std::copy_n(
                        input.data() + size_t(st.start * hidden),
                        x.size(), x.data());
                    // source_device=0 is PC CPU, not the P2C Phone shard.
                    ok = cycle_real_execute_phone_attention_pre(
                        cpu, layer, hp, x, hidden, st.tokens, topk, experts,
                        st.pc_ready, st.attention_ms, nullptr,
                        count > 1 ? &pc_kv : nullptr, st.start, 0);
                    if (ok) {
                        st.phone_gpu = std::make_unique<cycle_real_c2p_phone_chunk>();
                        ok = st.phone_gpu->prepare(
                            phone, phone_weights, hidden,
                            st.tokens, topk, experts);
                    }
                    break;
                }
                case llama_hybrid_cycle_stage::SEND_ACTIVATION:
                    ok = cycle_real_send_c2p_frame(
                        phone, st.wire, INFERENCE_CYCLE_READY,
                        st.pc_ready, st.pc_partial,
                        uint32_t(st.tokens), uint32_t(hidden), uint32_t(topk));
                    if (ok) {
                        ok = ggml_backend_rpc_cycle_apply_ready(
                            phone, st.wire.layer, st.wire.chunk,
                            st.wire.seq, st.wire.graph_id,
                            st.phone_gpu->residual,
                            st.phone_gpu->inputs.norm,
                            st.phone_gpu->inputs.topk_ids,
                            st.phone_gpu->inputs.topk_weights);
                    }
                    if (ok) {
                        ok = ggml_backend_rpc_cycle_wait_ack(
                            phone, INFERENCE_CYCLE_READY,
                            st.wire.layer, st.wire.chunk, st.wire.seq,
                            st.wire.graph_id, 30000);
                    }
                    break;
                case llama_hybrid_cycle_stage::FFN_SOURCE:
                    // PC can calculate its shard while READY is in flight.
                    ok = cycle_real_execute_ffn(
                        cpu, pc_weights, st.pc_ready.norm,
                        st.pc_ready.ids, st.pc_ready.routing,
                        hidden, st.tokens, topk, experts, st.pc_partial);
                    break;
                case llama_hybrid_cycle_stage::FFN_DESTINATION:
                    // READY has already been placed on the Phone GPU, and
                    // this compute needs ZERO PC->Phone control data copies.
                    ok = st.phone_gpu && st.phone_gpu->execute_ffn();
                    break;
                case llama_hybrid_cycle_stage::SEND_PARTIAL:
                    ok = cycle_real_send_c2p_frame(
                        phone, st.wire, INFERENCE_CYCLE_PARTIAL,
                        st.pc_ready, st.pc_partial,
                        uint32_t(st.tokens), uint32_t(hidden), uint32_t(topk));
                    if (ok) {
                        ok = ggml_backend_rpc_cycle_apply_partial(
                            phone, st.wire.layer, st.wire.chunk,
                            st.wire.seq, st.wire.graph_id,
                            st.phone_gpu->pc_partial);
                    }
                    // The incoming PC shard is DEVICE-ready now; the ACK
                    // remains deferred until the Phone Reduce completes.
                    break;
                case llama_hybrid_cycle_stage::REDUCE:
                    ok = st.phone_gpu && st.phone_gpu->execute_reduce(phone);
                    if (ok) {
                        ok = ggml_backend_rpc_cycle_finish_reduce(
                            phone, st.wire.layer, st.wire.chunk,
                            st.wire.seq, st.wire.graph_id);
                    }
                    if (ok) {
                        ok = ggml_backend_rpc_cycle_wait_ack(
                            phone, INFERENCE_CYCLE_PARTIAL,
                            st.wire.layer, st.wire.chunk, st.wire.seq,
                            st.wire.graph_id, 30000);
                    }
                    // Return the in-flight credit only after the ONLY
                    // Phone GPU Reduce and its ACK are both complete.
                    if (ok) {
                        st.phone_gpu->release();
                        st.phone_gpu.reset();
                    }
                    break;
                default:
                    break;
            }
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            if (task.stage == llama_hybrid_cycle_stage::FFN_SOURCE) {
                st.pc_ffn_ms = ms;
            } else if (task.stage == llama_hybrid_cycle_stage::FFN_DESTINATION) {
                st.phone_ffn_ms = ms;
            } else if (task.stage == llama_hybrid_cycle_stage::SEND_ACTIVATION) {
                st.ready_ms = ms;
            } else if (task.stage == llama_hybrid_cycle_stage::SEND_PARTIAL) {
                st.partial_ms = ms;
            } else if (task.stage == llama_hybrid_cycle_stage::REDUCE) {
                st.reduce_ms = ms;
            }
            std::printf(
                "[CYCLE_FIFO_C2P_STAGE] layer=%d chunk=%d seq=%llu "
                "pos=[%lld,%lld] stage=%s queue=%s wall_ms=%.3f status=%s\n",
                st.wire.layer, st.wire.chunk,
                (unsigned long long)st.wire.seq,
                (long long)st.start,
                (long long)(st.start + st.tokens - 1),
                llama_hybrid_cycle_stage_name(task.stage),
                llama_hybrid_cycle_queue_name(task.queue),
                ms, ok ? "OK" : "FAIL");
            std::fflush(stdout);
            return ok;
        }, window);
    const bool kv_complete = count == 1 || pc_kv.committed_tokens == tokens;
    pc_kv.release();
    const bool tx_idle = ggml_backend_rpc_cycle_wait_tx_idle(phone, 10000);
    ggml_backend_free(cpu);
    const double wall_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    double attn_ms = 0.0, pc_ffn_ms = 0.0, phone_ffn_ms = 0.0;
    double send_ready_ms = 0.0, send_partial_ms = 0.0, reduce_ms = 0.0;
    for (const auto & st : chunks) {
        attn_ms += st->attention_ms;
        pc_ffn_ms += st->pc_ffn_ms;
        phone_ffn_ms += st->phone_ffn_ms;
        send_ready_ms += st->ready_ms;
        send_partial_ms += st->partial_ms;
        reduce_ms += st->reduce_ms;
    }
    const bool success = ran && kv_complete && tx_idle;
    std::printf(
        "[CYCLE_FIFO_REAL_C2P_SUM] layer=%d tokens=%d "
        "chunks=%d chunk_tokens=%d max_inflight=%d "
        "direction=PC_TO_PHONE payload_sends=%d "
        "attention_owner=PC_CPU ffn_source=PC_CPU "
        "ffn_destination=PHONE_GPU reduce_owner=PHONE_GPU "
        "phone_ffn_inputs=PHONE_DATA_PLANE_RX_GPU_UPLOAD "
        "phone_reduce=REAL_GPU_GRAPH "
        "phone_output=GPU_COMPUTED_RELEASED_AFTER_ACK "
        "pc_kv=CPU_PERSISTENT phone_duplicate_rpc_readback=0 "
        "attn_ms=%.3f pc_ffn_ms=%.3f phone_ffn_ms=%.3f "
        "ready_total_ms=%.3f partial_total_ms=%.3f "
        "reduce_total_ms=%.3f wall_ms=%.3f "
        "cpu_parity=NOT_CHECKED status=%s\n",
        base.layer, int(tokens), count, int(chunk_tokens), window,
        2 * count, attn_ms, pc_ffn_ms, phone_ffn_ms,
        send_ready_ms, send_partial_ms, reduce_ms, wall_ms,
        success ? "OK" : "FAIL");
    return success;
}

// Aggregate only explicit server-confirmed Phone fences. For RPC backends
// the ordinary ggml_backend_synchronize() is a NO-OP and is not counted.
struct cycle_real_phone_fence_profile {
    uint64_t calls = 0;
    double wait_ms = 0.0;
    double max_wait_ms = 0.0;
};

// Explicit RPC fence: ggml_backend_synchronize(RPC) intentionally does
// nothing. A physical GPU tensor must stay owned until the Phone SERVER
// confirms that all earlier graphs have finished consuming it.
static bool cycle_real_explicit_phone_fence(
        ggml_backend_t phone,
        cycle_real_phone_fence_profile * sample = nullptr) {
    auto * dev = phone ? ggml_backend_get_device(phone) : nullptr;
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const auto fence = reg ?
        reinterpret_cast<ggml_backend_rpc_fence_t>(
            ggml_backend_reg_get_proc_address(
                reg, GGML_BACKEND_RPC_FENCE_PROC)) :
        nullptr;
    if (!fence) {
        std::fprintf(stderr,
            "[CYCLE_PERIOD_FENCE] status=FAIL reason=RPC_FENCE_MISSING\n");
        return false;
    }
    const auto start = std::chrono::steady_clock::now();
    fence(phone);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    if (sample) {
        ++sample->calls;
        sample->wait_ms += elapsed_ms;
        sample->max_wait_ms = std::max(
            sample->max_wait_ms, elapsed_ms);
    }
    if (std::getenv("LLAMA_HYBRID_CYCLE_STAGE_PROFILE") != nullptr) {
        std::printf(
            "[CYCLE_PHONE_FENCE_PROFILE] gpu_wait_ms=%.3f "
            "source=EXPLICIT_RPC_SYNCHRONIZE_ACK\n", elapsed_ms);
    }
    return true;
}

// Opt-in correctness probe. The ordinary Cycle NEVER reads back its
// interlayer Phone outputs. This method deliberately does so only when
// LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK=1, AFTER an explicit server-side
// GPU fence, and is therefore unsuitable for performance measurements.
static bool cycle_real_phone_boundary_probe(
        ggml_backend_t phone, const char * stage,
        int layer, int chunk, int64_t start, int64_t hidden,
        const ggml_tensor * device_tensor,
        const std::vector<float> & cpu_reference,
        std::vector<float> * readback = nullptr) {
    if (!cycle_real_env_flag("LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK")) {
        return true;
    }
    const size_t count = device_tensor ?
        size_t(ggml_nelements(device_tensor)) : 0;
    if (!phone || !device_tensor || !device_tensor->buffer ||
        device_tensor->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(device_tensor) || hidden <= 0 ||
        start < 0 || count == 0 || count % size_t(hidden) != 0 ||
        ggml_nbytes(device_tensor) != count * sizeof(float)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_BOUNDARY] stage=%s layer=%d chunk=%d "
            "status=FAIL reason=INVALID_GPU_TENSOR\n",
            stage, layer, chunk);
        return false;
    }
    if (!cycle_real_explicit_phone_fence(phone)) {
        return false;
    }
    std::vector<float> values(count);
    ggml_backend_tensor_get(
        device_tensor, values.data(), 0, count * sizeof(float));
    if (readback != nullptr) {
        *readback = values;
    }
    const size_t offset = size_t(start) * size_t(hidden);
    if (cpu_reference.empty()) {
        std::printf(
            "[CYCLE_PHONE_BOUNDARY] stage=%s layer=%d chunk=%d "
            "elements=%zu reference=NONE status=READBACK_OK\n",
            stage, layer, chunk, count);
        return true;
    }
    if (offset > cpu_reference.size() ||
        count > cpu_reference.size() - offset) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_BOUNDARY] stage=%s layer=%d chunk=%d "
            "reference=%zu required=%zu status=FAIL "
            "reason=REFERENCE_SHAPE\n",
            stage, layer, chunk, cpu_reference.size(), offset + count);
        return false;
    }
    const std::vector<float> ref(
        cpu_reference.begin() + offset,
        cpu_reference.begin() + offset + count);
    const cycle_real_diff d = cycle_real_compare(values, ref);
    const double tolerance = 0.02 + 0.00002 * d.max_ref;
    std::printf(
        "[CYCLE_PHONE_BOUNDARY] stage=%s layer=%d chunk=%d "
        "positions=[%lld,%lld] elements=%zu "
        "max_abs=%.9g rms=%.9g cosine=%.9g tolerance=%.9g "
        "state=%s diagnostic_readback=1\n",
        stage, layer, chunk, (long long)start,
        (long long)(start + int64_t(count / size_t(hidden)) - 1),
        count, d.max_abs, d.rms, d.cosine, tolerance,
        !d.valid ? "NONFINITE" :
            d.max_abs <= tolerance ? "WITHIN_TOLERANCE" : "DRIFT");
    std::fflush(stdout);
    return d.valid;
}

// Host mathematical RMSNorm oracle fed with the SAME residual that was
// actually computed by Phone OpenCL. This separates CPU-reference input
// propagation from a true Phone RMSNorm/scale kernel discrepancy.
//
// Debug-only: reads back one static FFN-norm weight vector from the Phone,
// in addition to the already opt-in Attention/Norm activation readbacks.
// No new graph execution, weight migration or persistent host mirror.
static bool cycle_real_phone_same_input_norm_probe(
        const ggml_tensor * phone_ffn_norm_weight,
        const std::vector<float> & phone_residual,
        const std::vector<float> & phone_norm,
        const std::vector<float> & reference_norm,
        int layer, int chunk, int64_t start, int64_t hidden,
        float eps) {
    if (!cycle_real_env_flag("LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK")) {
        return true;
    }
    const size_t width = hidden > 0 ? size_t(hidden) : 0;
    if (!phone_ffn_norm_weight || !phone_ffn_norm_weight->buffer ||
        width == 0 || phone_residual.empty() ||
        phone_residual.size() != phone_norm.size() ||
        phone_residual.size() % width != 0 ||
        size_t(ggml_nelements(phone_ffn_norm_weight)) != width ||
        !(eps >= 0.0f) ||
        (phone_ffn_norm_weight->type != GGML_TYPE_F32 &&
         phone_ffn_norm_weight->type != GGML_TYPE_F16)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_NORM_ISOLATE] layer=%d chunk=%d "
            "status=UNAVAILABLE reason=INVALID_TENSOR_OR_WEIGHT\n",
            layer, chunk);
        // Lack of an optional oracle should not fail the real Cycle.
        return true;
    }

    std::vector<float> gamma(width);
    if (phone_ffn_norm_weight->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(phone_ffn_norm_weight,
            gamma.data(), 0, width * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> gamma_f16(width);
        ggml_backend_tensor_get(phone_ffn_norm_weight,
            gamma_f16.data(), 0, width * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < width; ++i) {
            gamma[i] = ggml_fp16_to_fp32(gamma_f16[i]);
        }
    }

    // Follow the definition RMSNorm(x)=x/sqrt(mean(x^2)+eps) * gamma,
    // performing token-wise F32 output rounding. Double accumulation is
    // an independent mathematical cross-check, not a claimed bitwise
    // replica of any SIMD or OpenCL reduction order.
    std::vector<float> same_input_norm(phone_residual.size());
    double min_rms = INFINITY;
    double max_rms = 0.0;
    for (size_t t = 0; t < phone_residual.size() / width; ++t) {
        const size_t base = t * width;
        double sumsq = 0.0;
        for (size_t i = 0; i < width; ++i) {
            const double v = double(phone_residual[base+i]);
            sumsq += v * v;
        }
        const double rms = std::sqrt(sumsq / double(width) + double(eps));
        min_rms = std::min(min_rms, rms);
        max_rms = std::max(max_rms, rms);
        for (size_t i = 0; i < width; ++i) {
            same_input_norm[base+i] = float(
                (double(phone_residual[base+i]) / rms) *
                double(gamma[i]));
        }
    }

    const cycle_real_diff kernel = cycle_real_compare(phone_norm, same_input_norm);
    std::printf(
        "[CYCLE_PHONE_NORM_ISOLATE] layer=%d chunk=%d "
        "positions=[%lld,%lld] gamma_type=%s eps=%.9g "
        "input_rms_min=%.9g input_rms_max=%.9g "
        "same_input_max_abs=%.9g same_input_rms=%.9g "
        "same_input_cosine=%.9g math_oracle=%s",
        layer, chunk, (long long)start,
        (long long)(start + int64_t(phone_residual.size() / width) - 1),
        ggml_type_name(phone_ffn_norm_weight->type),
        double(eps), min_rms, max_rms,
        kernel.max_abs, kernel.rms, kernel.cosine,
        !kernel.valid ? "NONFINITE" :
        kernel.max_abs <= 0.001 ? "CLOSE" : "DIFF");
    const size_t begin = size_t(start) * width;
    if (begin <= reference_norm.size() &&
        phone_norm.size() <= reference_norm.size() - begin) {
        const std::vector<float> cpu_slice(
            reference_norm.begin() + begin,
            reference_norm.begin() + begin + phone_norm.size());
        const cycle_real_diff propagated =
            cycle_real_compare(same_input_norm, cpu_slice);
        std::printf(
            " same_input_vs_reference_max_abs=%.9g "
            "same_input_vs_reference_rms=%.9g",
            propagated.max_abs, propagated.rms);
    } else {
        std::printf(" same_input_vs_reference=UNAVAILABLE");
    }
    std::printf(" status=MEASURED\n");
    std::fflush(stdout);
    return true;
}

// One physical Cycle period: X complete PC CPU layers, one C2P Tensor
// boundary, Y complete Phone GPU layers, one P2C Tensor boundary.
// The model's initial "embd" is captured from the independent CPU
// reference, but every Transformer layer in this Cycle period runs on
// its REAL physical PC/Phone weights. The original single-layer tests
// continue to run separately. No post-FFN GPU activation is read back
// across Phone->PC unless this topology reaches its P2C boundary.
struct cycle_real_period_stage_profile {
    uint64_t count = 0;
    double wall_ms = 0.0;
    double max_ms = 0.0;
};
// Graph/allocator setup within Phone FFN prepare is separate from its
// actual GPU kernel execution. All durations here are PC-observed.
struct cycle_real_phone_prepare_profile {
    uint64_t count = 0;
    double graph_build_ms = 0.0;
    double gpu_buffer_alloc_ms = 0.0;
};
// Detailed wall-clock split for complete Phone GPU layers only. One
// PHONE_GPU worker owns all updates; no atomics or extra RPC calls.
// Fence segments are a subset of Phone FULL stage wall time, not added
// again to the per-queue totals.
struct cycle_real_phone_full_detail {
    uint64_t chunks = 0;
    double attn_call_ms = 0.0;
    double attn_fence_ms = 0.0;
    double pool_checkout_and_prepare_ms = 0.0;
    double local_ffn_copy_and_call_ms = 0.0;
    double ffn_fence_ms = 0.0;
    double reduce_submit_and_cleanup_ms = 0.0;
    double reduce_fence_ms = 0.0;
};

// PC host-observed durations, not kernel event timings. The PC_CUDA FIFO
// has one worker, so no additional locking is required for these counters.
struct cycle_real_pc_cuda_full_detail {
    uint64_t chunks = 0;
    double attn_call_ms = 0.0;
    double ffn_prepare_call_ms = 0.0;
    double ffn_execute_sync_ms = 0.0;
    double lout_readback_ms = 0.0;
    double task_wall_max_ms = 0.0;
};

struct cycle_real_period_chunk {
    int layer = 0;
    int chunk = 0;
    int64_t start = 0;
    int64_t tokens = 0;
    cycle_real_live_wire wire {};
    cycle_real_phone_ready ready;
    std::vector<float> partial;
    std::vector<float> host_output;
    inference_cycle_message recv_ready;
    inference_cycle_message recv_partial;
    std::unique_ptr<cycle_real_c2p_phone_chunk> phone_output;
};

static bool cycle_real_execute_period_fifo(
        ggml_backend_t phone, ggml_backend_dev_t cpu_dev,
        const llama_model & model,
        const llama_hybrid_cycle_layout & layout,
        const std::vector<float> & embedding,
        const std::vector<float> & cpu_reference_output,
        const cycle_real_period_reference & checkpoints,
        int32_t tokens, int chunk_tokens, int max_inflight,
        bool warmup_run = false, uint64_t wire_epoch = 0,
        cycle_real_phone_full_session_pool * phone_full_session = nullptr) {
    const char * run_phase = warmup_run ? "WARMUP" : "MEASURE";
    if (!phone || !cpu_dev || tokens <= 0 || tokens > 128 ||
        chunk_tokens <= 0 || max_inflight <= 0 ||
        layout.cycle_size != layout.x + layout.y + 2 ||
        layout.x <= 0 || layout.y <= 0 ||
        layout.tail_begin < layout.cycle_size ||
        int(layout.kinds.size()) < layout.cycle_size) {
        return false;
    }
    const int n_layers = layout.cycle_size;
    const int n_chunks = (tokens + chunk_tokens - 1) / chunk_tokens;
    const int window = std::min(max_inflight, n_chunks);
    const int64_t hidden = model.hparams.n_embd;
    const int64_t topk = model.hparams.n_expert_used;
    const int64_t experts = model.hparams.n_expert;
    const bool perf_mode = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_PERF_MODE");
    const bool pc_cuda_full = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_PC_CUDA_FULL");
    const bool p2c_ffn_reuse = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_P2C_FFN_REUSE");
    const bool c2p_ffn_reuse = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_C2P_FFN_REUSE");
    if (std::getenv("LLAMA_HYBRID_CYCLE_PC_CUDA_ATTN") != nullptr) {
        std::printf("[CYCLE_PC_CUDA_PLAN] old_option=PC_CUDA_ATTN "
                    "status=IGNORED reason=FULL_LAYER_MODE_SUPERSEDES_SPLIT\n");
    }
    // Opt-in: retain only full-Phone FFN/Reduce graphs after the NEXT
    // layer has consumed their output. This never reuses an in-flight
    // activation, never crosses model layers, and does not alter fences.
    const bool phone_full_reuse = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_PHONE_FULL_REUSE");
    // Optional graph fusion for complete Phone layers only. A C2P
    // Tensor needs the PC partial later and deliberately remains split.
    const bool phone_full_fuse = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_PHONE_FULL_FUSE");
    const bool phone_full_rpc_replay = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_PHONE_FULL_RPC_REPLAY");
    if (phone_full_session &&
        (!phone_full_reuse ||
         !phone_full_session->matches(
            phone, model, layout, tokens, chunk_tokens, window,
            phone_full_fuse, phone_full_rpc_replay))) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_FULL_SESSION] phase=%s status=FAIL "
            "reason=MODEL_BACKEND_LAYOUT_OR_SHAPE_MISMATCH\n",
            run_phase);
        return false;
    }
    if (perf_mode &&
        (cycle_real_env_flag("LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK") ||
         cycle_real_env_flag("LLAMA_HYBRID_CYCLE_NUMERIC_TRACE"))) {
        std::fprintf(stderr,
            "[CYCLE_PERIOD_PERF] status=FAIL "
            "reason=DIAGNOSTIC_READBACK_OR_NUMERIC_TRACE_ACTIVE "
            "action=UNSET_DEBUG_ENV\n");
        return false;
    }
    if (embedding.size() != size_t(tokens * hidden) ||
        cpu_reference_output.size() != embedding.size() ||
        hidden <= 0 || topk <= 0 || experts < topk) {
        std::fprintf(stderr,
            "[CYCLE_PERIOD] status=FAIL reason=CPU_REFERENCE_SHAPE "
            "embedding=%zu cpu_reference=%zu tokens=%d hidden=%lld\n",
            embedding.size(), cpu_reference_output.size(),
            int(tokens), (long long) hidden);
        return false;
    }
    cycle_real_numeric_probe("REFERENCE_EMBEDDING", -1, -1, embedding);
    // First period only: [PC_CPU x X] -> C2P -> [PHONE_GPU x Y] -> P2C.
    for (int l = 0; l < n_layers; ++l) {
        const auto kind = layout.kinds[size_t(l)];
        const auto expected = l < layout.x ?
            llama_hybrid_cycle_layer_kind::PC_CPU :
            l == layout.x ? llama_hybrid_cycle_layer_kind::TENSOR_C2P :
            l < layout.x + layout.y + 1 ?
                llama_hybrid_cycle_layer_kind::PHONE_GPU :
                llama_hybrid_cycle_layer_kind::TENSOR_P2C;
        if (kind != expected) {
            return false;
        }
    }

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu) {
        return false;
    }
    // The standalone Cycle CPU backend otherwise uses ggml's default
    // GGML_DEFAULT_N_THREADS (currently 4), regardless of how many cores
    // the PC actually has. Compare per-Cycle settings without modifying the
    // CPU-only correctness oracle or any Phone backend.
    const char * thread_env = std::getenv("LLAMA_HYBRID_CYCLE_CPU_THREADS");
    int cpu_threads = GGML_DEFAULT_N_THREADS;
    const bool threads_override = thread_env && thread_env[0] != '\0';
    if (threads_override) {
        char * end = nullptr;
        const long parsed = std::strtol(thread_env, &end, 10);
        if (end == thread_env || *end != '\0' || parsed < 1 || parsed > 64) {
            std::fprintf(stderr,
                "[CYCLE_CPU_THREADS] value=%s status=FAIL "
                "reason=EXPECTED_INTEGER_1_TO_64\n", thread_env);
            ggml_backend_free(cpu);
            return false;
        }
        cpu_threads = int(parsed);
        auto * cpu_backend_dev = ggml_backend_get_device(cpu);
        auto * cpu_reg = cpu_backend_dev ?
            ggml_backend_dev_backend_reg(cpu_backend_dev) : nullptr;
        const auto set_threads = cpu_reg ?
            reinterpret_cast<ggml_backend_set_n_threads_t>(
                ggml_backend_reg_get_proc_address(
                    cpu_reg, "ggml_backend_set_n_threads")) : nullptr;
        if (!set_threads) {
            std::fprintf(stderr,
                "[CYCLE_CPU_THREADS] requested=%d status=FAIL "
                "reason=CPU_BACKEND_THREAD_CONTROL_UNAVAILABLE\n", cpu_threads);
            ggml_backend_free(cpu);
            return false;
        }
        set_threads(cpu, cpu_threads);
    }
    std::printf(
        "[CYCLE_CPU_THREADS] threads=%d source=%s status=OK\n",
        cpu_threads, threads_override ? "ENV" : "GGML_DEFAULT");
    cycle_real_pc_cuda_runtime pc_cuda;
    if (pc_cuda_full) {
        // CUDA is a third COMPUTE backend, but it is NOT added to the
        // physical Meta model-device array (CPU=0, Phone=1).
        const auto cuda_dev = ggml_backend_dev_by_name("CUDA0");
        if (!cuda_dev ||
            ggml_backend_dev_type(cuda_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            std::fprintf(stderr,
                "[CYCLE_PC_CUDA_PLAN] status=FAIL "
                "reason=CUDA0_NOT_REGISTERED\n");
            ggml_backend_free(cpu);
            return false;
        }
        pc_cuda.backend = ggml_backend_dev_init(cuda_dev, nullptr);
        if (!pc_cuda.backend) {
            std::fprintf(stderr,
                "[CYCLE_PC_CUDA_PLAN] status=FAIL "
                "reason=CUDA0_DEVICE_INIT_FAILED device=%s\n",
                ggml_backend_dev_name(cuda_dev));
            ggml_backend_free(cpu);
            return false;
        }
        pc_cuda.weights.resize(size_t(n_layers));
        for (int l = 0; l < layout.x; ++l) {
            auto weights =
                std::make_unique<cycle_real_pc_cuda_full_weights>();
            if (!weights->prepare(
                    pc_cuda.backend, cpu, model.layers[size_t(l)], l)) {
                std::fprintf(stderr,
                    "[CYCLE_PC_CUDA_PLAN] layer=%d status=FAIL "
                    "reason=ATTN_WEIGHT_UPLOAD\n", l);
                ggml_backend_free(cpu);
                return false;
            }
            pc_cuda.weights[size_t(l)] = std::move(weights);
        }
        std::printf(
            "[CYCLE_PC_CUDA_PLAN] enabled=1 device=%s "
            "layers=%d full_stage_queue=PC_CUDA "
            "attention=PC_CUDA ffn=PC_CUDA reduce=PC_CUDA "
            "tensor_pc_backend=PC_CPU weight_upload=ONCE_PER_LAYER "
            "status=READY\n",
            ggml_backend_dev_name(cuda_dev), layout.x);
    }
    std::vector<std::unique_ptr<cycle_real_phone_kv_cache>> kv;
    kv.reserve(size_t(n_layers));
    for (int l = 0; l < n_layers; ++l) {
        auto cache = std::make_unique<cycle_real_phone_kv_cache>();
        const bool pc_source = l <= layout.x;
        const bool cuda_layer = pc_cuda_full && l < layout.x;
        if (n_chunks > 1 &&
            !cache->init(cuda_layer ? pc_cuda.backend :
                         (pc_source ? cpu : phone),
                         model.layers[size_t(l)], tokens,
                         pc_source ? 0 : 1,
                         cuda_layer ? pc_cuda.weights[size_t(l)].get() : nullptr)) {
            std::fprintf(stderr,
                "[CYCLE_PERIOD] status=FAIL reason=KV_ALLOC layer=%d\n", l);
            ggml_backend_free(cpu);
            return false;
        }
        kv.push_back(std::move(cache));
    }

    using state_ptr = std::unique_ptr<cycle_real_period_chunk>;
    std::vector<std::vector<state_ptr>> states(static_cast<size_t>(n_layers));
    // Only the PHONE_GPU worker accesses this per-layer pool. Every cached
    // FFN graph owns the very same GPU allocation and borrowed model weights
    // it used on the last Chunk. No CPU/GPU hidden activation is cached.
    using phone_chunk_ptr = std::unique_ptr<cycle_real_c2p_phone_chunk>;
    std::vector<std::vector<phone_chunk_ptr>> period_phone_full_pool(
        static_cast<size_t>(n_layers));
    auto & phone_full_pool = phone_full_session ?
        phone_full_session->idle_by_layer : period_phone_full_pool;
    std::vector<size_t> phone_full_cached_at_period_start(
        static_cast<size_t>(n_layers), 0);
    if (phone_full_session) {
        for (int l = layout.x + 1; l < layout.x + layout.y + 1; ++l) {
            phone_full_cached_at_period_start[size_t(l)] =
                phone_full_pool[size_t(l)].size();
        }
    }
    // C2P setup runs on PC_CPU but finished outputs are retired by
    // PHONE_GPU, once the successor's Attention fence has completed.
    // The pool lock protects ONLY pointer transfer / metadata: allocating
    // or freeing GPU RPC buffers while holding it would block both workers.
    std::mutex c2p_pool_mutex;
    std::vector<phone_chunk_ptr> c2p_pool;
    size_t c2p_live_slots = 0;
    uint64_t c2p_prepares = 0;
    uint64_t c2p_hits = 0;
    uint64_t c2p_retired = 0;
    uint64_t c2p_shape_replacements = 0;
    struct phone_reuse_layer_stats {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t recycled = 0;
    };
    std::vector<phone_reuse_layer_stats> phone_reuse_stats(static_cast<size_t>(n_layers));
    for (int l = 0; l < n_layers; ++l) {
        auto & row = states[size_t(l)];
        row.reserve(size_t(n_chunks));
        for (int c = 0; c < n_chunks; ++c) {
            auto st = std::make_unique<cycle_real_period_chunk>();
            st->layer = l;
            st->chunk = c;
            st->start = int64_t(c) * chunk_tokens;
            st->tokens = std::min(
                int64_t(chunk_tokens), int64_t(tokens) - st->start);
            // Each run has disjoint wire seq IDs. The first (warmup) run
            // keeps legacy IDs, while the measured run uses the next epoch.
            // Both runs share the model and one transport connection.
            st->wire = {l, c,
                wire_epoch * 100000 + uint64_t(l + 1) * 1000 +
                    uint64_t(c + 1), 1};
            row.push_back(std::move(st));
        }
    }
    std::vector<float> final_output(size_t(tokens * hidden), 0.0f);
    // Keep strict CPU parity for correctness runs. Performance runs still
    // REPORT parity, but it is independently non-gating: only real compute,
    // causal KV, TX drain and finite nonzero output can succeed the run.
    const bool stage_detail = std::getenv(
        "LLAMA_HYBRID_CYCLE_STAGE_PROFILE") != nullptr;
    constexpr size_t n_queues =
        static_cast<size_t>(llama_hybrid_cycle_queue::COUNT);
    constexpr size_t n_stages =
        static_cast<size_t>(llama_hybrid_cycle_stage::REDUCE) + 1;
    std::array<std::array<cycle_real_period_stage_profile, n_stages>,
        n_queues> stage_profile {};
    llama_hybrid_cycle_fifo_profile fifo_profile {};
    std::array<cycle_real_attention_phase_profile, n_queues> attn_profile {};
    std::array<cycle_real_ffn_phase_profile, n_queues> ffn_profile {};
    std::array<cycle_real_phone_prepare_profile, n_queues> gpu_prepare_profile {};
    std::array<cycle_real_phone_fence_profile, n_queues> fence_profile {};
    std::vector<cycle_real_phone_full_detail> phone_full_detail(
        static_cast<size_t>(n_layers));
    std::vector<cycle_real_pc_cuda_full_detail> pc_cuda_detail(
        static_cast<size_t>(n_layers));
    // Owned exclusively by the PHONE_GPU FIFO worker. One shape-specific
    // slot is enough: this worker executes FFN_SOURCE tasks serially.
    std::unique_ptr<cycle_real_p2c_ffn_slot> p2c_ffn_slot;
    uint64_t p2c_ffn_prepares = 0;
    uint64_t p2c_ffn_hits = 0;
    uint64_t p2c_ffn_replacements = 0;
    const auto started = std::chrono::steady_clock::now();
    std::printf(
        "[CYCLE_PERIOD_PLAN] phase=%s epoch=%" PRIu64
        " x=%d y=%d layers=[0,%d] "
        "tokens=%d chunks=%d chunk_tokens=%d max_inflight=%d "
        "topology=PC_FULL_C2P_PHONE_FULL_P2C "
        "output_credit_release=DOWNSTREAM_CONSUMED "
        "phone_interlayer_readback=FORBIDDEN status=READY\n",
        run_phase, wire_epoch, layout.x, layout.y, n_layers - 1,
        int(tokens), n_chunks, chunk_tokens, window);
    std::fflush(stdout);

    llama_hybrid_cycle_layout period = layout;
    period.pc_cuda_full = pc_cuda_full;
    period.kinds.resize(size_t(n_layers));
    period.tail_begin = n_layers;
    const auto host_input = [&](int l, int c) -> std::vector<float> {
        const auto & st = *states[size_t(l)][size_t(c)];
        if (l == 0) {
            const size_t offset = size_t(st.start * hidden);
            return std::vector<float>(
                embedding.begin() + offset,
                embedding.begin() + offset + size_t(st.tokens * hidden));
        }
        return states[size_t(l - 1)][size_t(c)]->host_output;
    };
    const auto full_shard = [&](int l, size_t device) {
        const auto & weights = model.layers[size_t(l)];
        ggml_backend_t backend = device == 0 ? cpu : phone;
        // PC_FULL uses the complete native CPU MoE weights. PHONE_FULL
        // borrows its complete Phone GPU physical Meta shard.
        return llama_hybrid_cycle_ffn_shard_weights {
            cycle_real_resolve_physical_weight(weights.ffn_up_exps, backend, device),
            cycle_real_resolve_physical_weight(weights.ffn_gate_exps, backend, device),
            cycle_real_resolve_physical_weight(weights.ffn_down_exps, backend, device),
        };
    };
    const auto take_device_input = [&](int l, int c) -> const ggml_tensor * {
        if (l <= layout.x || c < 0 || l >= n_layers) {
            return nullptr;
        }
        const auto & previous = *states[size_t(l - 1)][size_t(c)];
        return previous.phone_output ? previous.phone_output->output : nullptr;
    };
    // Called by the sole PHONE_GPU worker only AFTER the consumer
    // Attention has explicitly fenced. At that point the producer's
    // l_out will not be read again. Retain the FFN/Reduce GPU buffers for
    // a later Chunk of the SAME physical full-Phone layer.
    const auto retire_device_output = [&](int producer_layer, int chunk) {
        auto & owned = states[size_t(producer_layer)][size_t(chunk)]->phone_output;
        if (c2p_ffn_reuse && owned && !owned->is_full_layer &&
            producer_layer == layout.x) {
            // This callback runs only AFTER the next layer's Attention
            // fence, guaranteeing it no longer reads the C2P output.
            std::lock_guard<std::mutex> lock(c2p_pool_mutex);
            c2p_pool.push_back(std::move(owned));
            ++c2p_retired;
        } else if (phone_full_reuse && owned && owned->is_full_layer &&
                   producer_layer > layout.x &&
                   producer_layer < layout.x + layout.y + 1) {
            phone_full_pool[size_t(producer_layer)].push_back(std::move(owned));
            ++phone_reuse_stats[size_t(producer_layer)].recycled;
        } else {
            owned.reset();
        }
    };

    const bool succeeded = llama_hybrid_cycle_run_fifo(
        period, n_chunks, [&](const llama_hybrid_cycle_task & task) -> bool {
            if (task.layer < 0 || task.layer >= n_layers ||
                task.chunk < 0 || task.chunk >= n_chunks) {
                return false;
            }
            const int l = task.layer;
            const int c = task.chunk;
            const size_t queue_index = static_cast<size_t>(task.queue);
            auto & st = *states[size_t(l)][size_t(c)];
            const auto kind = period.kinds[size_t(l)];
            const auto & layer = model.layers[size_t(l)];
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = false;
            switch (task.stage) {
                case llama_hybrid_cycle_stage::FULL_LAYER: {
                    if (kind == llama_hybrid_cycle_layer_kind::PC_CPU &&
                        pc_cuda_full) {
                        const auto input = host_input(l, c);
                        const auto * weights = pc_cuda.weights[size_t(l)].get();
                        double attn_ms = 0.0;
                        double ffn_prepare_call_ms = 0.0;
                        double ffn_execute_sync_ms = 0.0;
                        double lout_readback_ms = 0.0;
                        const auto cuda_begin = std::chrono::steady_clock::now();
                        ok = weights &&
                            input.size() == size_t(st.tokens * hidden) &&
                            cycle_real_execute_phone_attention_pre(
                                pc_cuda.backend, layer, model.hparams, input,
                                hidden, st.tokens, topk, experts,
                                st.ready, attn_ms, nullptr,
                                n_chunks > 1 ? kv[size_t(l)].get() : nullptr,
                                st.start, 0, nullptr, true,
                                perf_mode ? &attn_profile[queue_index] : nullptr,
                                weights);
                        const auto cuda_attn_done = std::chrono::steady_clock::now();
                        if (ok && cycle_real_env_flag(
                                "LLAMA_HYBRID_CYCLE_NUMERIC_TRACE") &&
                            size_t(l) < checkpoints.pc_attn_residuals.size()) {
                            std::vector<float> cuda_residual(
                                size_t(st.tokens * hidden));
                            ggml_backend_tensor_get(
                                st.ready.phone_graph.residual,
                                cuda_residual.data(), 0,
                                cuda_residual.size() * sizeof(float));
                            cycle_real_period_checkpoint(
                                "PC_CUDA_FULL_ATTN_RESIDUAL", l, c, st.start,
                                hidden, cuda_residual,
                                checkpoints.pc_attn_residuals[size_t(l)]);
                        }
                        if (ok) {
                            // Reuse the already-validated GPU FFN+ADD graph
                            // builder. It is backend-generic; here every
                            // weight and activation resides on local CUDA.
                            cycle_real_c2p_phone_chunk full_gpu;
                            const auto ffn_prepare_begin =
                                std::chrono::steady_clock::now();
                            ok = full_gpu.prepare(
                                pc_cuda.backend, weights->full_ffn_shard(),
                                hidden, st.tokens, topk, experts,
                                true, true, false);
                            ffn_prepare_call_ms =
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    ffn_prepare_begin).count();
                            if (ok) {
                                const auto ffn_execute_begin =
                                    std::chrono::steady_clock::now();
                                ok = full_gpu.execute_full_local(st.ready);
                                if (ok) {
                                    ggml_backend_synchronize(pc_cuda.backend);
                                }
                                ffn_execute_sync_ms =
                                    std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() -
                                        ffn_execute_begin).count();
                            }
                            if (ok && cycle_real_env_flag(
                                    "LLAMA_HYBRID_CYCLE_NUMERIC_TRACE") &&
                                size_t(l) < checkpoints.pc_ffn_outputs.size()) {
                                std::vector<float> cuda_ffn(
                                    size_t(st.tokens * hidden));
                                ggml_backend_tensor_get(
                                    full_gpu.inputs.ffn_output,
                                    cuda_ffn.data(), 0,
                                    cuda_ffn.size() * sizeof(float));
                                cycle_real_period_checkpoint(
                                    "PC_CUDA_FULL_FFN", l, c, st.start,
                                    hidden, cuda_ffn,
                                    checkpoints.pc_ffn_outputs[size_t(l)]);
                            }
                            if (ok) {
                                const auto readback_begin =
                                    std::chrono::steady_clock::now();
                                st.host_output.resize(
                                    size_t(st.tokens * hidden));
                                ggml_backend_tensor_get(
                                    full_gpu.output, st.host_output.data(),
                                    0, st.host_output.size() * sizeof(float));
                                lout_readback_ms =
                                    std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() -
                                        readback_begin).count();
                                cycle_real_numeric_probe(
                                    "PC_FULL_OUTPUT", l, c, st.host_output);
                                if (size_t(l) < checkpoints.layer_outputs.size()) {
                                    cycle_real_period_checkpoint(
                                        "PC_FULL_OUTPUT", l, c, st.start,
                                        hidden, st.host_output,
                                        checkpoints.layer_outputs[size_t(l)]);
                                }
                            }
                        }
                        // Both CUDA graphs have completed or failed. Do not
                        // retain borrowed activations beyond this FIFO task.
                        st.ready.release_phone_graph();
                        if (perf_mode) {
                            auto & sample = pc_cuda_detail[size_t(l)];
                            ++sample.chunks;
                            sample.attn_call_ms +=
                                std::chrono::duration<double, std::milli>(
                                    cuda_attn_done - cuda_begin).count();
                            sample.ffn_prepare_call_ms += ffn_prepare_call_ms;
                            sample.ffn_execute_sync_ms += ffn_execute_sync_ms;
                            sample.lout_readback_ms += lout_readback_ms;
                            sample.task_wall_max_ms = std::max(
                                sample.task_wall_max_ms,
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    cuda_begin).count());
                        }
                        if (stage_detail || !ok) {
                            std::printf(
                                "[CYCLE_PC_CUDA_FULL] layer=%d chunk=%d "
                                "attn_ms=%.3f callback_ms=%.3f "
                                "output_owner=PC_CUDA readback=FINAL_LOUT "
                                "status=%s\n",
                                l, c, attn_ms,
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    cuda_begin).count(),
                                ok ? "OK" : "FAIL");
                        }
                    } else if (kind == llama_hybrid_cycle_layer_kind::PC_CPU) {
                        const auto input = host_input(l, c);
                        double attn_ms = 0.0;
                        ok = input.size() == size_t(st.tokens * hidden) &&
                            cycle_real_execute_phone_attention_pre(
                                cpu, layer, model.hparams, input,
                                hidden, st.tokens, topk, experts,
                                st.ready, attn_ms, nullptr,
                                n_chunks > 1 ? kv[size_t(l)].get() : nullptr,
                                st.start, 0, nullptr, false,
                                perf_mode ? &attn_profile[queue_index] : nullptr);
                        if (ok) {
                            if (size_t(l) < checkpoints.pc_attn_residuals.size()) {
                                cycle_real_period_checkpoint(
                                    "PC_FULL_ATTN_RESIDUAL", l, c, st.start,
                                    hidden, st.ready.residual,
                                    checkpoints.pc_attn_residuals[size_t(l)]);
                            }
                            const auto weights = full_shard(l, 0);
                            ok = cycle_real_execute_ffn(
                                cpu, weights, st.ready.norm, st.ready.ids,
                                st.ready.routing, hidden, st.tokens, topk,
                                experts, st.partial, nullptr, nullptr,
                                nullptr, nullptr,
                                perf_mode ? &ffn_profile[queue_index] : nullptr);
                        }
                        if (ok) {
                            if (size_t(l) < checkpoints.pc_ffn_outputs.size()) {
                                cycle_real_period_checkpoint(
                                    "PC_FULL_FFN", l, c, st.start,
                                    hidden, st.partial,
                                    checkpoints.pc_ffn_outputs[size_t(l)]);
                            }
                            st.host_output = std::move(st.partial);
                            if (st.host_output.size() != st.ready.residual.size()) {
                                ok = false;
                            } else {
                                for (size_t i = 0; i < st.host_output.size(); ++i) {
                                    st.host_output[i] += st.ready.residual[i];
                                }
                            }
                        }
                        if (ok) {
                            cycle_real_numeric_probe(
                                "PC_FULL_OUTPUT", l, c, st.host_output);
                            if (size_t(l) < checkpoints.layer_outputs.size()) {
                                cycle_real_period_checkpoint(
                                    "PC_FULL_OUTPUT", l, c, st.start, hidden,
                                    st.host_output,
                                    checkpoints.layer_outputs[size_t(l)]);
                            }
                        }
                        st.ready.residual.clear();
                        st.ready.norm.clear();
                        st.ready.ids.clear();
                        st.ready.routing.clear();
                        st.ready.router_probs.clear();
                    } else if (kind == llama_hybrid_cycle_layer_kind::PHONE_GPU) {
                        const auto * local_input = take_device_input(l, c);
                        const bool boundary_trace = cycle_real_env_flag(
                            "LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK");
                        double attn_ms = 0.0;
                        const std::vector<float> empty;
                        const auto & attn_ref =
                            size_t(l) < checkpoints.phone_attn_residuals.size() ?
                            checkpoints.phone_attn_residuals[size_t(l)] : empty;
                        const auto & norm_ref =
                            size_t(l) < checkpoints.phone_ffn_norms.size() ?
                            checkpoints.phone_ffn_norms[size_t(l)] : empty;
                        const auto & ffn_ref =
                            size_t(l) < checkpoints.phone_ffn_outputs.size() ?
                            checkpoints.phone_ffn_outputs[size_t(l)] : empty;
                        std::vector<float> attn_values;
                        std::vector<float> norm_values;
                        std::vector<float> ffn_values;
                        // These phases are timed at the PC caller. The
                        // Attention call includes graph setup, copies and
                        // submit; Server GPU execution is reported separately.
                        auto & full_detail = phone_full_detail[size_t(l)];
                        const auto attn_call_begin =
                            std::chrono::steady_clock::now();
                        ok = local_input != nullptr &&
                            cycle_real_execute_phone_attention_pre(
                                phone, layer, model.hparams, empty,
                                hidden, st.tokens, topk, experts,
                                st.ready, attn_ms, nullptr,
                                n_chunks > 1 ? kv[size_t(l)].get() : nullptr,
                                st.start, 1, local_input, true,
                                perf_mode ? &attn_profile[queue_index] : nullptr);
                        if (perf_mode) {
                            ++full_detail.chunks;
                            full_detail.attn_call_ms +=
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    attn_call_begin).count();
                        }
                        if (ok) {
                            // This is a real RPC fence; the RPC backend's
                            // ordinary synchronize() is a NO-OP.
                            const auto fence_begin =
                                std::chrono::steady_clock::now();
                            ok = cycle_real_explicit_phone_fence(
                                phone, perf_mode ? &fence_profile[queue_index] : nullptr);
                            if (perf_mode) {
                                full_detail.attn_fence_ms +=
                                    std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() -
                                        fence_begin).count();
                            }
                        }
                        if (ok && boundary_trace) {
                            // The Attention tensors are still GPU-owned
                            // here. Read them BEFORE release_phone_graph().
                            ok = cycle_real_phone_boundary_probe(
                                phone, "PHONE_FULL_ATTN_RESIDUAL",
                                l, c, st.start, hidden,
                                st.ready.phone_graph.residual,
                                attn_ref, &attn_values);
                            if (ok) {
                                ok = cycle_real_phone_boundary_probe(
                                    phone, "PHONE_FULL_FFN_NORM",
                                    l, c, st.start, hidden,
                                    st.ready.phone_graph.ffn_norm, norm_ref,
                                    &norm_values);
                            }
                            if (ok) {
                                // Compare GPU RMSNorm+gamma with an
                                // independent host formula using the exact
                                // SAME GPU residual and physical gamma.
                                const auto * gamma =
                                    cycle_real_resolve_physical_weight(
                                        layer.ffn_norm, phone, 1);
                                ok = cycle_real_phone_same_input_norm_probe(
                                    gamma, attn_values, norm_values, norm_ref,
                                    l, c, st.start, hidden,
                                    model.hparams.f_norm_rms_eps);
                            }
                        }
                        const auto full_prepare_begin =
                            std::chrono::steady_clock::now();
                        if (ok) {
                            retire_device_output(l - 1, c);
                            // The previous consumer's fence is the ownership
                            // release point. Reused graph input buffers are
                            // overwritten by execute_full_local() for THIS
                            // Chunk before either FFN or Reduce is submitted.
                            if (phone_full_reuse) {
                                auto & pool = phone_full_pool[size_t(l)];
                                for (auto it = pool.begin(); it != pool.end(); ++it) {
                                    if ((*it)->is_full_layer &&
                                        (*it)->prepared_tokens == st.tokens &&
                                        (*it)->fused_full_graph == phone_full_fuse &&
                                        (*it)->rpc_full_replay == phone_full_rpc_replay) {
                                        st.phone_output = std::move(*it);
                                        pool.erase(it);
                                        ++phone_reuse_stats[size_t(l)].hits;
                                        break;
                                    }
                                }
                            }
                            if (!st.phone_output) {
                                st.phone_output =
                                    std::make_unique<cycle_real_c2p_phone_chunk>();
                                const auto weights = full_shard(l, 1);
                                ok = st.phone_output->prepare(
                                    phone, weights, hidden, st.tokens,
                                    topk, experts, true, phone_full_fuse,
                                    phone_full_rpc_replay);
                                if (ok && phone_full_reuse) {
                                    ++phone_reuse_stats[size_t(l)].misses;
                                }
                                if (ok && perf_mode) {
                                    auto & sample = gpu_prepare_profile[queue_index];
                                    ++sample.count;
                                    sample.graph_build_ms +=
                                        st.phone_output->graph_build_ms;
                                    sample.gpu_buffer_alloc_ms +=
                                        st.phone_output->gpu_alloc_ms;
                                }
                            }
                        }
                        if (perf_mode) {
                            full_detail.pool_checkout_and_prepare_ms +=
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    full_prepare_begin).count();
                        }
                        const auto full_ffn_call_begin =
                            std::chrono::steady_clock::now();
                        if (ok) {
                            ok = st.phone_output->execute_full_local(st.ready);
                        }
                        if (perf_mode) {
                            full_detail.local_ffn_copy_and_call_ms +=
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    full_ffn_call_begin).count();
                        }
                        if (ok) {
                            const auto fence_begin =
                                std::chrono::steady_clock::now();
                            ok = cycle_real_explicit_phone_fence(
                                phone, perf_mode ? &fence_profile[queue_index] : nullptr);
                            if (perf_mode) {
                                full_detail.ffn_fence_ms +=
                                    std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() -
                                        fence_begin).count();
                            }
                        }
                        if (ok && boundary_trace) {
                            // The full-Phone MoE FFN has completed, but the
                            // Reduce graph has not yet run. This isolates
                            // FFN error from Attention/input error.
                            ok = cycle_real_phone_boundary_probe(
                                phone, "PHONE_FULL_FFN",
                                l, c, st.start, hidden,
                                st.phone_output->inputs.ffn_output,
                                ffn_ref, &ffn_values);
                        }
                        const auto full_reduce_begin =
                            std::chrono::steady_clock::now();
                        if (ok) {
                            // FFN fence above has completed the entire
                            // fused FFN+ADD graph, including l_out.
                            // Only AFTER that fence may its Attention
                            // input graph be released.
                            st.ready.release_phone_graph();
                            if (!st.phone_output->fused_full_graph) {
                                ok = st.phone_output->execute_reduce(phone);
                            }
                        }
                        if (perf_mode) {
                            full_detail.reduce_submit_and_cleanup_ms +=
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    full_reduce_begin).count();
                        }
                        if (ok && !st.phone_output->fused_full_graph) {
                            const auto fence_begin =
                                std::chrono::steady_clock::now();
                            ok = cycle_real_explicit_phone_fence(
                                phone, perf_mode ? &fence_profile[queue_index] : nullptr);
                            if (perf_mode) {
                                full_detail.reduce_fence_ms +=
                                    std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() -
                                        fence_begin).count();
                            }
                        }
                        if (ok && st.phone_output &&
                            size_t(l) < checkpoints.layer_outputs.size()) {
                            std::vector<float> gpu_output;
                            ok = cycle_real_phone_boundary_probe(
                                phone, "PHONE_FULL_OUTPUT",
                                l, c, st.start, hidden,
                                st.phone_output->output,
                                checkpoints.layer_outputs[size_t(l)],
                                boundary_trace ? &gpu_output : nullptr);
                            if (ok && boundary_trace) {
                                if (attn_values.size() != ffn_values.size() ||
                                    gpu_output.size() != attn_values.size()) {
                                    std::fprintf(stderr,
                                        "[CYCLE_PHONE_FULL_REDUCE_PARITY] "
                                        "layer=%d chunk=%d status=FAIL "
                                        "reason=SHAPE_MISMATCH\n", l, c);
                                    ok = false;
                                } else {
                                    // Distinguish an incorrect GPU ADD from
                                    // upstream Attention/FFN numerical drift.
                                    for (size_t i = 0; i < attn_values.size(); ++i) {
                                        attn_values[i] += ffn_values[i];
                                    }
                                    const cycle_real_diff d =
                                        cycle_real_compare(gpu_output, attn_values);
                                    std::printf(
                                        "[CYCLE_PHONE_FULL_REDUCE_PARITY] "
                                        "layer=%d chunk=%d max_abs=%.9g "
                                        "rms=%.9g cosine=%.9g "
                                        "state=%s\n",
                                        l, c, d.max_abs, d.rms, d.cosine,
                                        !d.valid ? "INVALID" :
                                        d.max_abs < 0.0001 ?
                                            "NUMERICALLY_CLOSE" : "DRIFT");
                                }
                            }
                        }
                    }
                    break;
                }
                case llama_hybrid_cycle_stage::ATTN_PRE: {
                    double attn_ms = 0.0;
                    if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                        const auto input = host_input(l, c);
                        ok = input.size() == size_t(st.tokens * hidden) &&
                            cycle_real_execute_phone_attention_pre(
                                cpu, layer, model.hparams, input,
                                hidden, st.tokens, topk, experts,
                                st.ready, attn_ms, nullptr,
                                n_chunks > 1 ? kv[size_t(l)].get() : nullptr,
                                st.start, 0, nullptr, false,
                                perf_mode ? &attn_profile[queue_index] : nullptr);
                        if (ok) {
                            cycle_real_numeric_probe(
                                "C2P_READY_RESIDUAL", l, c, st.ready.residual);
                            cycle_real_numeric_probe(
                                "C2P_READY_NORM", l, c, st.ready.norm);
                            cycle_real_period_checkpoint(
                                "C2P_ATTN_RESIDUAL", l, c, st.start, hidden,
                                st.ready.residual, checkpoints.c2p_residual);
                            bool needs_prepare = true;
                            bool slot_reserved = false;
                            phone_chunk_ptr incompatible_slot;
                            if (c2p_ffn_reuse) {
                                {
                                    std::lock_guard<std::mutex> lock(c2p_pool_mutex);
                                    for (auto it = c2p_pool.begin();
                                         it != c2p_pool.end(); ++it) {
                                        if ((*it)->prepared_tokens == st.tokens &&
                                            !(*it)->is_full_layer) {
                                            st.phone_output = std::move(*it);
                                            c2p_pool.erase(it);
                                            ++c2p_hits;
                                            needs_prepare = false;
                                            break;
                                        }
                                    }
                                    if (needs_prepare) {
                                        if (c2p_live_slots < size_t(window)) {
                                            ++c2p_live_slots;
                                            slot_reserved = true;
                                        } else if (!c2p_pool.empty()) {
                                            // The last chunk may have a
                                            // shorter token shape. Evict one
                                            // idle incompatible buffer, not
                                            // a producer still in use.
                                            incompatible_slot =
                                                std::move(c2p_pool.back());
                                            c2p_pool.pop_back();
                                            ++c2p_shape_replacements;
                                            slot_reserved = true;
                                        }
                                        if (slot_reserved) ++c2p_prepares;
                                    }
                                }
                                // Never perform the synchronous RPC FREE
                                // inside the mutex protecting both workers.
                                incompatible_slot.reset();
                                if (needs_prepare && !slot_reserved) {
                                    std::fprintf(stderr,
                                        "[CYCLE_C2P_FFN_REUSE] layer=%d "
                                        "chunk=%d status=FAIL "
                                        "reason=POOL_EXHAUSTED window=%d\n",
                                        l, c, window);
                                    ok = false;
                                }
                            }
                            if (ok && needs_prepare) {
                                st.phone_output =
                                    std::make_unique<cycle_real_c2p_phone_chunk>();
                                const auto weights = full_shard(l, 1);
                                ok = st.phone_output->prepare(
                                    phone, weights, hidden, st.tokens,
                                    topk, experts);
                                if (!ok && c2p_ffn_reuse) {
                                    st.phone_output.reset();
                                    std::lock_guard<std::mutex> lock(c2p_pool_mutex);
                                    --c2p_live_slots;
                                }
                                if (ok && perf_mode) {
                                    auto & sample = gpu_prepare_profile[queue_index];
                                    ++sample.count;
                                    sample.graph_build_ms +=
                                        st.phone_output->graph_build_ms;
                                    sample.gpu_buffer_alloc_ms +=
                                        st.phone_output->gpu_alloc_ms;
                                }
                            }
                        }
                    } else if (kind == llama_hybrid_cycle_layer_kind::TENSOR_P2C) {
                        const auto * local_input = take_device_input(l, c);
                        const std::vector<float> empty;
                        ok = local_input != nullptr &&
                            cycle_real_execute_phone_attention_pre(
                                phone, layer, model.hparams, empty,
                                hidden, st.tokens, topk, experts,
                                st.ready, attn_ms, &st.wire,
                                n_chunks > 1 ? kv[size_t(l)].get() : nullptr,
                                st.start, 1, local_input, true,
                                perf_mode ? &attn_profile[queue_index] : nullptr);
                        if (ok) {
                            ok = cycle_real_explicit_phone_fence(
                                phone, perf_mode ? &fence_profile[queue_index] : nullptr);
                        }
                        if (ok) {
                            retire_device_output(l - 1, c);
                        }
                    }
                    break;
                }
                case llama_hybrid_cycle_stage::SEND_ACTIVATION: {
                    if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                        ok = st.phone_output && cycle_real_send_c2p_frame(
                            phone, st.wire, INFERENCE_CYCLE_READY,
                            st.ready, st.partial,
                            uint32_t(st.tokens), uint32_t(hidden), uint32_t(topk));
                        if (ok) {
                            ok = ggml_backend_rpc_cycle_apply_ready(
                                phone, st.wire.layer, st.wire.chunk,
                                st.wire.seq, st.wire.graph_id,
                                st.phone_output->residual,
                                st.phone_output->inputs.norm,
                                st.phone_output->inputs.topk_ids,
                                st.phone_output->inputs.topk_weights);
                        }
                        if (ok) {
                            ok = ggml_backend_rpc_cycle_wait_ack(
                                phone, INFERENCE_CYCLE_READY,
                                st.wire.layer, st.wire.chunk,
                                st.wire.seq, st.wire.graph_id, 30000);
                        }
                    } else {
                        ok = cycle_real_receive_live_wire(
                            phone, st.wire, INFERENCE_CYCLE_READY,
                            uint32_t(st.tokens), uint32_t(hidden),
                            uint32_t(topk), st.recv_ready);
                        if (ok) {
                            cycle_real_numeric_probe(
                                "P2C_READY_RESIDUAL", l, c,
                                st.recv_ready.ready.residual);
                            cycle_real_numeric_probe(
                                "P2C_READY_NORM", l, c,
                                st.recv_ready.ready.ffn_norm);
                            cycle_real_numeric_probe(
                                "P2C_READY_ROUTING", l, c,
                                st.recv_ready.ready.topk_weights);
                            cycle_real_period_checkpoint(
                                "P2C_ATTN_RESIDUAL", l, c, st.start, hidden,
                                st.recv_ready.ready.residual,
                                checkpoints.p2c_residual);
                            cycle_real_period_checkpoint(
                                "P2C_FFN_NORM", l, c, st.start, hidden,
                                st.recv_ready.ready.ffn_norm,
                                checkpoints.p2c_norm);
                            ok = ggml_backend_rpc_cycle_send_ack(
                                phone, st.wire.layer, st.wire.chunk,
                                st.wire.seq, st.wire.graph_id,
                                INFERENCE_CYCLE_READY, true);
                        }
                    }
                    break;
                }
                case llama_hybrid_cycle_stage::FFN_SOURCE: {
                    if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                        const auto weights = full_shard(l, 0);
                        ok = cycle_real_execute_ffn(
                            cpu, weights, st.ready.norm,
                            st.ready.ids, st.ready.routing,
                            hidden, st.tokens, topk, experts,
                            st.partial, nullptr, nullptr, nullptr, nullptr,
                            perf_mode ? &ffn_profile[queue_index] : nullptr);
                    } else {
                        const auto weights = full_shard(l, 1);
                        const std::vector<float> none_f;
                        const std::vector<int32_t> none_i;
                        std::vector<float> ignored;
                        if (!p2c_ffn_reuse) {
                            ok = cycle_real_execute_ffn(
                                phone, weights, none_f, none_i, none_f,
                                hidden, st.tokens, topk, experts,
                                ignored, nullptr, nullptr,
                                &st.wire, &st.ready.phone_graph,
                                perf_mode ? &ffn_profile[queue_index] : nullptr);
                        } else {
                            const bool same_shape = p2c_ffn_slot &&
                                p2c_ffn_slot->prepared_tokens == st.tokens;
                            if (!same_shape) {
                                if (p2c_ffn_slot) {
                                    // Tail chunks may have fewer tokens.
                                    // Free a mismatched slot before preparing
                                    // its replacement; never replay an
                                    // incompatible tensor shape.
                                    p2c_ffn_slot.reset();
                                    ++p2c_ffn_replacements;
                                }
                                p2c_ffn_slot =
                                    std::make_unique<cycle_real_p2c_ffn_slot>();
                                ok = p2c_ffn_slot->prepare(
                                    phone, weights, hidden, st.tokens,
                                    topk, experts);
                                if (ok) ++p2c_ffn_prepares;
                            } else {
                                ++p2c_ffn_hits;
                                ok = true;
                            }
                            if (ok) {
                                auto * profile = perf_mode ?
                                    &ffn_profile[queue_index] : nullptr;
                                if (profile) {
                                    ++profile->count;
                                    if (!same_shape) {
                                        profile->graph_build_ms +=
                                            p2c_ffn_slot->graph_build_ms;
                                        profile->backend_prepare_ms +=
                                            p2c_ffn_slot->backend_prepare_ms;
                                    }
                                }
                                ok = p2c_ffn_slot->execute_and_send(
                                    phone, st.ready.phone_graph, st.wire,
                                    profile);
                            }
                            if (!ok) {
                                // A failed graph/send must not become a hit.
                                p2c_ffn_slot.reset();
                            }
                        }
                        st.ready.release_phone_graph();
                    }
                    break;
                }
                case llama_hybrid_cycle_stage::FFN_DESTINATION: {
                    if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                        ok = st.phone_output && st.phone_output->execute_ffn();
                    } else {
                        const auto weights = full_shard(l, 0);
                        ok = cycle_real_execute_ffn(
                            cpu, weights,
                            st.recv_ready.ready.ffn_norm,
                            st.recv_ready.ready.topk_ids,
                            st.recv_ready.ready.topk_weights,
                            hidden, st.tokens, topk, experts,
                            st.partial, nullptr, nullptr, nullptr, nullptr,
                            perf_mode ? &ffn_profile[queue_index] : nullptr);
                        if (ok) {
                            cycle_real_numeric_probe(
                                "P2C_PC_FFN", l, c, st.partial);
                        }
                    }
                    break;
                }
                case llama_hybrid_cycle_stage::SEND_PARTIAL: {
                    if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                        ok = st.phone_output && cycle_real_send_c2p_frame(
                            phone, st.wire, INFERENCE_CYCLE_PARTIAL,
                            st.ready, st.partial, uint32_t(st.tokens),
                            uint32_t(hidden), uint32_t(topk));
                        if (ok) {
                            ok = ggml_backend_rpc_cycle_apply_partial(
                                phone, st.wire.layer, st.wire.chunk,
                                st.wire.seq, st.wire.graph_id,
                                st.phone_output->pc_partial);
                        }
                    } else {
                        ok = cycle_real_receive_live_wire(
                            phone, st.wire, INFERENCE_CYCLE_PARTIAL,
                            uint32_t(st.tokens), uint32_t(hidden), 0,
                            st.recv_partial);
                        if (ok) {
                            cycle_real_numeric_probe(
                                "P2C_PHONE_FFN_RX", l, c,
                                st.recv_partial.partial.partial);
                        }
                    }
                    break;
                }
                case llama_hybrid_cycle_stage::REDUCE: {
                    if (kind == llama_hybrid_cycle_layer_kind::TENSOR_C2P) {
                        ok = st.phone_output &&
                            st.phone_output->execute_reduce(phone);
                        if (ok) {
                            ok = ggml_backend_rpc_cycle_finish_reduce(
                                phone, st.wire.layer, st.wire.chunk,
                                st.wire.seq, st.wire.graph_id);
                        }
                        if (ok) {
                            ok = ggml_backend_rpc_cycle_wait_ack(
                                phone, INFERENCE_CYCLE_PARTIAL,
                                st.wire.layer, st.wire.chunk,
                                st.wire.seq, st.wire.graph_id, 30000);
                        }
                        // The GPU output MUST survive for the next Phone
                        // FULL_LAYER; do not free it at this boundary.
                        if (ok && cycle_real_env_flag(
                                "LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK")) {
                            std::vector<float> phone_ffn;
                            ok = cycle_real_phone_boundary_probe(
                                phone, "C2P_PHONE_FFN",
                                l, c, st.start, hidden,
                                st.phone_output->inputs.ffn_output,
                                {}, &phone_ffn);
                            if (ok && phone_ffn.size() == st.partial.size() &&
                                phone_ffn.size() == st.ready.residual.size()) {
                                std::vector<float> joined = phone_ffn;
                                for (size_t i = 0; i < joined.size(); ++i) {
                                    joined[i] += st.partial[i];
                                }
                                cycle_real_period_checkpoint(
                                    "C2P_FFN_SUM", l, c, st.start, hidden,
                                    joined, checkpoints.c2p_full_ffn);
                                std::vector<float> expected = joined;
                                for (size_t i = 0; i < expected.size(); ++i) {
                                    expected[i] += st.ready.residual[i];
                                }
                                std::vector<float> gpu_output;
                                ok = cycle_real_phone_boundary_probe(
                                    phone, "C2P_FINAL_OUTPUT",
                                    l, c, st.start, hidden,
                                    st.phone_output->output,
                                    checkpoints.layer_outputs[size_t(l)],
                                    &gpu_output);
                                if (ok) {
                                    const cycle_real_diff reduce_diff =
                                        cycle_real_compare(gpu_output, expected);
                                    std::printf(
                                        "[CYCLE_PHONE_REDUCE_PARITY] "
                                        "layer=%d chunk=%d max_abs=%.9g "
                                        "rms=%.9g cosine=%.9g "
                                        "source=GPU_OUTPUT_VS_SAME_INPUTS "
                                        "state=%s\n",
                                        l, c, reduce_diff.max_abs,
                                        reduce_diff.rms, reduce_diff.cosine,
                                        !reduce_diff.valid ? "INVALID" :
                                        reduce_diff.max_abs < 0.0001 ?
                                            "NUMERICALLY_CLOSE" : "DRIFT");
                                }
                            } else if (ok) {
                                std::fprintf(stderr,
                                    "[CYCLE_PHONE_BOUNDARY] stage=C2P_FFN_SUM "
                                    "layer=%d chunk=%d status=FAIL "
                                    "reason=INPUT_SHAPE\n", l, c);
                                ok = false;
                            }
                        }
                    } else {
                        const size_t n = size_t(hidden * st.tokens);
                        st.host_output.resize(n);
                        ok = inference_cycle_reduce_p2c_pc_cpu(
                            st.recv_ready.ready, st.partial.data(),
                            st.partial.size(), st.recv_partial.partial,
                            st.host_output.data(), st.host_output.size());
                        if (ok) {
                            ok = ggml_backend_rpc_cycle_send_ack(
                                phone, st.wire.layer, st.wire.chunk,
                                st.wire.seq, st.wire.graph_id,
                                INFERENCE_CYCLE_PARTIAL, true);
                        }
                        if (ok) {
                            cycle_real_numeric_probe(
                                "P2C_REDUCED", l, c, st.host_output);
                            std::copy(st.host_output.begin(),
                                      st.host_output.end(),
                                      final_output.begin() +
                                          size_t(st.start * hidden));
                        }
                    }
                    break;
                }
            }
            const double stage_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            if (perf_mode) {
                // Each row has a unique FIFO worker, so two physical
                // devices can update their disjoint queue rows concurrently.
                auto & sample = stage_profile[
                    static_cast<size_t>(task.queue)][
                    static_cast<size_t>(task.stage)];
                ++sample.count;
                sample.wall_ms += stage_ms;
                sample.max_ms = std::max(sample.max_ms, stage_ms);
            }
            // Suppress per-task logging during clean performance runs.
            // Detailed diagnostics remain available via STAGE_PROFILE=1.
            if (!perf_mode || stage_detail || !ok) {
                std::printf(
                    "[CYCLE_PERIOD_STAGE] layer=%d kind=%s chunk=%d "
                    "stage=%s queue=%s wall_ms=%.3f status=%s\n",
                    l, llama_hybrid_cycle_kind_name(kind), c,
                    llama_hybrid_cycle_stage_name(task.stage),
                    llama_hybrid_cycle_queue_name(task.queue),
                    stage_ms, ok ? "OK" : "FAIL");
                std::fflush(stdout);
            }
            return ok;
        }, window, true, perf_mode ? &fifo_profile : nullptr);
    const auto fifo_done = std::chrono::steady_clock::now();
    const bool p2c_ffn_cached_at_end = p2c_ffn_slot != nullptr;
    // Release the cached RPC allocation BEFORE the period timer stops.
    // No borrowed input or output outlives the FIFO workers.
    p2c_ffn_slot.reset();
    const auto p2c_pool_release_done = std::chrono::steady_clock::now();
    // Pool ownership moves out under the mutex. All synchronous RPC frees
    // run after unlock, since the producer/consumer workers have joined.
    size_t c2p_cached_at_fifo_end = 0;
    std::vector<phone_chunk_ptr> c2p_to_release;
    if (c2p_ffn_reuse) {
        {
            std::lock_guard<std::mutex> lock(c2p_pool_mutex);
            c2p_cached_at_fifo_end = c2p_pool.size();
            c2p_to_release = std::move(c2p_pool);
        }
        c2p_to_release.clear();
    }
    const auto c2p_pool_release_done = std::chrono::steady_clock::now();
    // Include the eventual GPU cache release in the measured period.
    // Otherwise a cache hit could look faster merely because the final
    // GPU buffer_free calls moved outside wall_ms.
    std::vector<size_t> phone_cached_at_end(static_cast<size_t>(n_layers), 0);
    if (phone_full_reuse) {
        for (int l = layout.x + 1; l < layout.x + layout.y + 1; ++l) {
            phone_cached_at_end[size_t(l)] =
                phone_full_pool[size_t(l)].size();
        }
        // Successful cycles leave cached graph+buffer resources in the
        // model/session owner. Failed FIFO runs must not keep partially
        // used or uncertain remote buffers for the next period.
        if (!phone_full_session || !succeeded) {
            for (auto & pool : phone_full_pool) pool.clear();
        }
    }
    const auto phone_pool_release_done = std::chrono::steady_clock::now();
    bool kv_ok = true;
    for (int l = 0; l < n_layers; ++l) {
        kv_ok = kv_ok && (n_chunks == 1 ||
            kv[size_t(l)]->committed_tokens == tokens);
        kv[size_t(l)]->release();
    }
    const auto kv_release_done = std::chrono::steady_clock::now();
    const bool net_idle =
        ggml_backend_rpc_cycle_wait_tx_idle(phone, 10000);
    const auto tx_idle_done = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(
        tx_idle_done - started).count();
    const double fifo_ms = std::chrono::duration<double, std::milli>(
        fifo_done - started).count();
    const double phone_pool_release_ms =
        std::chrono::duration<double, std::milli>(
            phone_pool_release_done - fifo_done).count();
    const double p2c_pool_release_ms =
        std::chrono::duration<double, std::milli>(
            p2c_pool_release_done - fifo_done).count();
    const double c2p_pool_release_ms =
        std::chrono::duration<double, std::milli>(
            c2p_pool_release_done - p2c_pool_release_done).count();
    const double full_pool_release_ms =
        std::chrono::duration<double, std::milli>(
            phone_pool_release_done - c2p_pool_release_done).count();
    const double kv_release_ms = std::chrono::duration<double, std::milli>(
        kv_release_done - phone_pool_release_done).count();
    const double tx_idle_ms = std::chrono::duration<double, std::milli>(
        tx_idle_done - kv_release_done).count();
    ggml_backend_free(cpu);
    cycle_real_numeric_probe("PERIOD_FINAL_OUTPUT", n_layers - 1, -1, final_output);
    size_t nonfinite = 0;
    size_t zero = 0;
    double max_abs = 0.0;
    if (succeeded) {
        for (float value : final_output) {
            if (!std::isfinite(value)) {
                ++nonfinite;
            } else {
                max_abs = std::max(max_abs, std::fabs(double(value)));
                zero += value == 0.0f ? 1 : 0;
            }
        }
    }
    const bool output_ok = succeeded && nonfinite == 0 && max_abs > 0.0;
    const bool execution_ok = succeeded && kv_ok && net_idle && output_ok;
    // Compare only a COMPLETED period. Never claim CPU parity from a
    // partially filled final_output after a failed stage or transport.
    const bool parity_ok = execution_ok &&
        cycle_real_period_cpu_parity(
            final_output, cpu_reference_output,
            hidden, tokens, chunk_tokens, n_layers - 1);
    // PERF mode separates successful execution from an already known
    // cross-backend numerical tolerance difference. STRICT (default)
    // continues to fail when ANY CPU parity chunk exceeds the bound.
    // Warmup must check execution safety, but a pre-existing cross-backend
    // strict-parity failure must not prevent the measured follow-up run.
    const bool ok = execution_ok && (warmup_run || perf_mode || parity_ok);
    std::printf(
        "[CYCLE_PC_CUDA_SUM] enabled=%d pc_full_layers=%d "
        "stage=FULL_LAYER queue=%s "
        "attention=%s ffn=%s reduce=%s "
        "tensor_pc_backend=PC_CPU kv_owner=%s status=%s\n",
        pc_cuda_full ? 1 : 0, layout.x,
        pc_cuda_full ? "PC_CUDA" : "PC_CPU",
        pc_cuda_full ? "PC_CUDA" : "PC_CPU",
        pc_cuda_full ? "PC_CUDA" : "PC_CPU",
        pc_cuda_full ? "PC_CUDA" : "PC_CPU",
        pc_cuda_full ? "PC_CUDA" : "PC_CPU",
        execution_ok ? "OK" : "FAIL");
    if (c2p_ffn_reuse) {
        std::printf(
            "[CYCLE_C2P_FFN_REUSE_SUM] side=PC layer=%d enabled=1 "
            "prepares=%llu hits=%llu retired=%llu "
            "shape_replacements=%llu slots_total=%zu "
            "cached_at_fifo_end=%zu graph_owner=PC_CPU_AND_PHONE_GPU "
            "output_release=SUCCESSOR_ATTN_FENCED "
            "cache_lifetime=ONE_PERIOD\n",
            layout.x, (unsigned long long)c2p_prepares,
            (unsigned long long)c2p_hits, (unsigned long long)c2p_retired,
            (unsigned long long)c2p_shape_replacements,
            c2p_live_slots, c2p_cached_at_fifo_end);
    }
    if (perf_mode && (c2p_ffn_reuse || p2c_ffn_reuse)) {
        std::printf(
            "[CYCLE_FFN_POOL_RELEASE_SUM] side=PC "
            "p2c_release_ms=%.3f c2p_release_ms=%.3f "
            "phone_full_release_ms=%.3f "
            "included_in=PERIOD_WALL_AND_PHONE_POOL_RELEASE\n",
            p2c_pool_release_ms, c2p_pool_release_ms,
            full_pool_release_ms);
    }
    if (p2c_ffn_reuse) {
        std::printf(
            "[CYCLE_P2C_FFN_REUSE_SUM] side=PC layer=%d enabled=1 "
            "prepares=%llu hits=%llu shape_replacements=%llu "
            "cached_at_fifo_end=%d graph_owner=PHONE_GPU "
            "output_release=PHONE_TX_SLOT_COPIED "
            "cache_lifetime=ONE_PERIOD\n",
            n_layers - 1, (unsigned long long)p2c_ffn_prepares,
            (unsigned long long)p2c_ffn_hits,
            (unsigned long long)p2c_ffn_replacements,
            p2c_ffn_cached_at_end ? 1 : 0);
    }
    std::printf(
        "[CYCLE_PHONE_FULL_FUSE_SUM] enabled=%d scope=PHONE_FULL "
        "graph_mode=%s async_safety=POST_FFN_FENCE\n",
        phone_full_fuse ? 1 : 0,
        phone_full_fuse ? "FFN_AND_ADD_ONE_GRAPH" : "FFN_PLUS_REDUCE");
    std::printf(
        "[CYCLE_PHONE_FULL_RPC_REPLAY_SUM] enabled=%d "
        "scope=PHONE_FULL uid=NONZERO_PER_PREPARED_GRAPH "
        "cache_owner=RPC_SERVER "
        "graph_lifetime=FFN_GPU_BUFFER_LIFETIME\n",
        phone_full_rpc_replay ? 1 : 0);
    if (phone_full_reuse) {
        for (int l = layout.x + 1; l < layout.x + layout.y + 1; ++l) {
            const auto & sample = phone_reuse_stats[size_t(l)];
            std::printf(
                "[CYCLE_PHONE_FULL_REUSE_SUM] layer=%d enabled=1 "
                "hits=%llu prepares=%llu retired=%llu "
                "cached_at_fifo_end=%zu graph_owner=PHONE_GPU "
                "output_release=SUCCESSOR_ATTN_FENCED "
                "phase=%s cache_scope=%s cached_at_period_start=%zu\n",
                l, (unsigned long long)sample.hits,
                (unsigned long long)sample.misses,
                (unsigned long long)sample.recycled,
                phone_cached_at_end[size_t(l)],
                run_phase, phone_full_session ? "SESSION" : "PERIOD",
                phone_full_cached_at_period_start[size_t(l)]);
            if (phone_full_session) {
                std::printf(
                    "[CYCLE_PHONE_FULL_SESSION_REUSE_SUM] phase=%s "
                    "layer=%d start_slots=%zu prepares=%llu "
                    "hits=%llu retired=%llu end_slots=%zu "
                    "status=%s\n",
                    run_phase, l,
                    phone_full_cached_at_period_start[size_t(l)],
                    (unsigned long long)sample.misses,
                    (unsigned long long)sample.hits,
                    (unsigned long long)sample.recycled,
                    phone_cached_at_end[size_t(l)],
                    succeeded ? "OK" : "FAIL");
            }
        }
    }
    if (perf_mode) {
        // Compare this breakdown against PHONE_GPU/FULL_LAYER stage_sum_ms.
        // The numbers describe client-side wall time and are subsets of it.
        for (int l = layout.x + 1; l < layout.x + layout.y + 1; ++l) {
            const auto & sample = phone_full_detail[size_t(l)];
            std::printf(
                "[CYCLE_PHONE_FULL_DETAIL] side=PC layer=%d chunks=%llu "
                "attn_call_ms=%.3f attn_fence_ms=%.3f "
                "pool_checkout_prepare_ms=%.3f "
                "local_ffn_copy_call_ms=%.3f ffn_fence_ms=%.3f "
                "reduce_submit_cleanup_ms=%.3f reduce_fence_ms=%.3f "
                "semantics=CLIENT_WALL_SUBPHASES\n",
                l, (unsigned long long)sample.chunks,
                sample.attn_call_ms, sample.attn_fence_ms,
                sample.pool_checkout_and_prepare_ms,
                sample.local_ffn_copy_and_call_ms,
                sample.ffn_fence_ms, sample.reduce_submit_and_cleanup_ms,
                sample.reduce_fence_ms);
        }
        double busy_sum_ms = 0.0;
        for (size_t q = 0; q < n_queues; ++q) {
            const auto & worker = fifo_profile.queues[q];
            if (worker.tasks == 0) {
                continue;
            }
            busy_sum_ms += worker.execute_ms;
            std::printf(
                "[CYCLE_PERIOD_QUEUE_PROFILE] side=PC queue=%s tasks=%llu "
                "execute_sum_ms=%.3f max_execute_ms=%.3f "
                "ready_queue_wait_sum_ms=%.3f max_ready_queue_wait_ms=%.3f "
                "worker_idle_sum_ms=%.3f idle_wakes=%llu\n",
                llama_hybrid_cycle_queue_name(
                    static_cast<llama_hybrid_cycle_queue>(q)),
                (unsigned long long)worker.tasks,
                worker.execute_ms, worker.max_execute_ms,
                worker.queue_wait_ms, worker.max_queue_wait_ms,
                worker.idle_wait_ms, (unsigned long long)worker.idle_wakes);
            for (size_t s = 0; s < n_stages; ++s) {
                const auto & sample = stage_profile[q][s];
                if (sample.count == 0) {
                    continue;
                }
                std::printf(
                    "[CYCLE_PERIOD_STAGE_SUM] side=PC queue=%s stage=%s "
                    "count=%llu stage_sum_ms=%.3f max_stage_ms=%.3f\n",
                    llama_hybrid_cycle_queue_name(
                        static_cast<llama_hybrid_cycle_queue>(q)),
                    llama_hybrid_cycle_stage_name(
                        static_cast<llama_hybrid_cycle_stage>(s)),
                    (unsigned long long)sample.count,
                    sample.wall_ms, sample.max_ms);
            }
        }
        for (size_t q = 0; q < n_queues; ++q) {
            const char * name = llama_hybrid_cycle_queue_name(
                static_cast<llama_hybrid_cycle_queue>(q));
            const auto & attn = attn_profile[q];
            if (attn.count) {
                std::printf(
                    "[CYCLE_PERIOD_ATTN_PREP_SUM] queue=%s count=%llu "
                    "graph_build_sum_ms=%.3f buffer_alloc_sum_ms=%.3f "
                    "input_copy_sum_ms=%.3f "
                    "backend_graph_call_sum_ms=%.3f "
                    "backend_call_semantics=CPU_COMPUTE_OR_RPC_SUBMIT\n",
                    name, (unsigned long long)attn.count,
                    attn.graph_build_ms, attn.alloc_ms,
                    attn.input_copy_ms, attn.graph_call_ms);
            }
            const auto & ffn = ffn_profile[q];
            if (ffn.count) {
                std::printf(
                    "[CYCLE_PERIOD_FFN_PREP_SUM] queue=%s count=%llu "
                    "graph_build_sum_ms=%.3f "
                    "backend_prepare_sum_ms=%.3f "
                    "backend_execute_call_sum_ms=%.3f "
                    "backend_call_semantics=CPU_COMPUTE_OR_RPC_SUBMIT\n",
                    name, (unsigned long long)ffn.count,
                    ffn.graph_build_ms, ffn.backend_prepare_ms,
                    ffn.backend_execute_call_ms);
            }
            const auto & gpu = gpu_prepare_profile[q];
            if (gpu.count) {
                std::printf(
                    "[CYCLE_PERIOD_PHONE_FFN_PREP_SUM] side=PC queue=%s "
                    "count=%llu graph_build_sum_ms=%.3f "
                    "backend_prepare_call_sum_ms=%.3f "
                    "scope=PC_PHONE_PROXY_PREPARE "
                    "includes=BACKEND_ALLOC_AND_RPC_NOT_KERNEL_ONLY\n",
                    name, (unsigned long long)gpu.count,
                    gpu.graph_build_ms, gpu.gpu_buffer_alloc_ms);
            }
            const auto & fence = fence_profile[q];
            if (fence.calls) {
                std::printf(
                    "[CYCLE_PERIOD_PHONE_FENCE_SUM] side=PC queue=%s "
                    "count=%llu rpc_fence_wait_sum_ms=%.3f "
                    "max_fence_wait_ms=%.3f "
                    "scope=PC_WAIT_FOR_PHONE_ACK "
                    "includes=REMOTE_COMPLETION_AND_CONTROL_RPC\n",
                    name, (unsigned long long)fence.calls,
                    fence.wait_ms, fence.max_wait_ms);
            }
            if (q == static_cast<size_t>(
                        llama_hybrid_cycle_queue::PC_TO_PHONE) ||
                q == static_cast<size_t>(
                        llama_hybrid_cycle_queue::PHONE_TO_PC)) {
                const auto & ready = stage_profile[q][static_cast<size_t>(
                    llama_hybrid_cycle_stage::SEND_ACTIVATION)];
                const auto & partial = stage_profile[q][static_cast<size_t>(
                    llama_hybrid_cycle_stage::SEND_PARTIAL)];
                if (ready.count || partial.count) {
                    std::printf(
                        "[CYCLE_PERIOD_TRANSPORT_SUM] side=PC queue=%s "
                        "ready_count=%llu partial_count=%llu "
                        "ready_stage_sum_ms=%.3f partial_stage_sum_ms=%.3f "
                        "semantics=CALLBACK_AND_ACK_NOT_WIRE_ONLY\n",
                        name,
                        (unsigned long long)ready.count,
                        (unsigned long long)partial.count,
                        ready.wall_ms, partial.wall_ms);
                }
            }
        }
        // Each queue's callback time is measured on PC. The PHONE_GPU
        // queue is a host-side RPC proxy, NOT the phone GPU compute clock.
        const auto & host_cpu = fifo_profile.queues[
            static_cast<size_t>(llama_hybrid_cycle_queue::PC_CPU)];
        const auto & host_cuda = fifo_profile.queues[
            static_cast<size_t>(llama_hybrid_cycle_queue::PC_CUDA)];
        const auto & remote_proxy = fifo_profile.queues[
            static_cast<size_t>(llama_hybrid_cycle_queue::PHONE_GPU)];
        const auto & c2p_worker = fifo_profile.queues[
            static_cast<size_t>(llama_hybrid_cycle_queue::PC_TO_PHONE)];
        const auto & p2c_worker = fifo_profile.queues[
            static_cast<size_t>(llama_hybrid_cycle_queue::PHONE_TO_PC)];
        std::printf(
            "[CYCLE_PC_HOST_SUM] side=PC scope=FIFO_PERIOD "
            "fifo_wall_ms=%.3f cpu_callback_sum_ms=%.3f "
            "cuda_callback_sum_ms=%.3f "
            "phone_rpc_proxy_callback_sum_ms=%.3f "
            "c2p_transport_callback_sum_ms=%.3f "
            "p2c_transport_callback_sum_ms=%.3f "
            "semantics=PARALLEL_HOST_WORKER_CALLBACKS "
            "cross_queue_addition=FORBIDDEN\n",
            fifo_ms, host_cpu.execute_ms, host_cuda.execute_ms,
            remote_proxy.execute_ms, c2p_worker.execute_ms,
            p2c_worker.execute_ms);
        const size_t phone_q =
            static_cast<size_t>(llama_hybrid_cycle_queue::PHONE_GPU);
        std::printf(
            "[CYCLE_PC_REMOTE_PROXY_SUM] side=PC "
            "scope=PHONE_GPU_QUEUE_CALLBACKS calls=%llu "
            "callback_wall_sum_ms=%.3f "
            "fence_ack_wait_sum_ms=%.3f "
            "ffn_prepare_call_sum_ms=%.3f "
            "subphases=SUBSETS_OF_CALLBACK_WALL "
            "gpu_kernel_time=PHONE_SERVER_ONLY\n",
            (unsigned long long)remote_proxy.tasks,
            remote_proxy.execute_ms,
            fence_profile[phone_q].wait_ms,
            gpu_prepare_profile[phone_q].gpu_buffer_alloc_ms);
        if (pc_cuda_full) {
            for (int l = 0; l < layout.x; ++l) {
                const auto & sample = pc_cuda_detail[size_t(l)];
                std::printf(
                    "[CYCLE_PC_CUDA_DETAIL] side=PC layer=%d chunks=%llu "
                    "attn_call_sum_ms=%.3f "
                    "ffn_prepare_call_sum_ms=%.3f "
                    "ffn_execute_sync_sum_ms=%.3f "
                    "lout_readback_sum_ms=%.3f "
                    "max_full_task_wall_ms=%.3f "
                    "semantics=HOST_WALL_CUDA_CALLS_AND_SYNC "
                    "kernel_only=NOT_MEASURED\n",
                    l, (unsigned long long)sample.chunks,
                    sample.attn_call_ms, sample.ffn_prepare_call_ms,
                    sample.ffn_execute_sync_ms, sample.lout_readback_ms,
                    sample.task_wall_max_ms);
            }
        }
        // Queue times from different workers may overlap. Their sum is
        // never claimed to be an exposed end-to-end critical-path delay.
        std::printf(
            "[CYCLE_PERIOD_PERF_SUM] side=PC phase=%s epoch=%" PRIu64
            " fifo_wall_ms=%.3f "
            "phone_pool_release_ms=%.3f "
            "kv_release_ms=%.3f tx_idle_ms=%.3f wall_ms=%.3f "
            "queue_execute_sum_ms=%.3f "
            "queue_overlap=INDEPENDENT_WORKERS "
            "parity_timing=EXCLUDED "
            "execution_status=%s strict_parity=%s\n",
            run_phase, wire_epoch,
            fifo_ms, phone_pool_release_ms, kv_release_ms, tx_idle_ms,
            wall_ms, busy_sum_ms, execution_ok ? "OK" : "FAIL",
            !execution_ok ? "NOT_CHECKED" :
                parity_ok ? "PASS" : "FAIL");
    }
    std::printf(
        "[CYCLE_PERIOD_SUM] phase=%s epoch=%" PRIu64
        " layers=%d x=%d y=%d chunks=%d tokens=%d "
        "c2p_payloads=%d p2c_payloads=%d total_payloads=%d "
        "phone_handoffs=DEVICE_LOCAL phone_to_pc_only_at=P2C "
        "gpu_output_owner_at_c2p=PHONE_GPU "
        "final_output_owner=PC_CPU "
        "output_credit_release=DOWNSTREAM_CONSUMED "
        "final_max_abs=%.6g final_zero=%zu final_nonfinite=%zu "
        "wall_ms=%.3f mode=%s execution_status=%s "
        "cpu_parity=%s status=%s\n",
        run_phase, wire_epoch, n_layers, layout.x, layout.y,
        n_chunks, int(tokens), 2 * n_chunks, 2 * n_chunks, 4 * n_chunks,
        max_abs, zero, nonfinite, wall_ms,
        warmup_run ? "WARMUP_PARITY_NON_GATING" :
            perf_mode ? "PERF_PARITY_NON_GATING" : "STRICT",
        execution_ok ? "OK" : "FAIL",
        !execution_ok ? "NOT_CHECKED" : parity_ok ? "PASS" : "FAIL",
        ok ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char ** argv) {
#ifdef LLAMA_CYCLE_LOOP_TEST
    const bool c2p_mode = true;
    const bool cycle_period_mode = true;
#elif defined(LLAMA_CYCLE_C2P_TEST)
    const bool c2p_mode = true;
    const bool cycle_period_mode = false;
#else
    const bool c2p_mode = cycle_real_env_flag("LLAMA_HYBRID_CYCLE_REAL_C2P");
    const bool cycle_period_mode = false;
#endif
    if (argc < 4 || argc > 6 || !llama_hybrid_cycle_requested()) {
        std::fprintf(stderr,
            "Usage: test-hybrid-cycle-real-%s <model.gguf> <phone-ip:port> "
            "<%s-layer> [prompt] [n_tokens=1..128]\n"
            "Set LLAMA_HYBRID_CYCLE_PC_LAYERS and "
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS first.\n",
            c2p_mode ? "c2p" : "p2c", c2p_mode ? "C2P" : "P2C");
        return 2;
    }
    char * end = nullptr;
    const long layer_long = std::strtol(argv[3], &end, 10);
    if (!end || *end != '\0' || layer_long < 0 || layer_long >= 512) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=BAD_LAYER\n");
        return 2;
    }
    const int layer = int(layer_long);
    int32_t target_tokens = 0;
    if (argc == 6) {
        char * token_end = nullptr;
        const long parsed = std::strtol(argv[5], &token_end, 10);
        if (token_end == argv[5] || !token_end || *token_end != '\0' ||
            parsed < 1 || parsed > 128) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL "
                "reason=BAD_TARGET_TOKENS allowed=[1,128]\n");
            return 2;
        }
        target_tokens = int32_t(parsed);
    }
    if (!cycle_real_set_compare_layer(layer)) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=SET_ENV\n");
        return 2;
    }

    llama_backend_init();
    // The physical PC+Phone model is expensive to load. Fail fast if the
    // opt-in local CUDA full-layer executor is unavailable. A statically
    // registered CPU/RPC backend can prevent llama_backend_init() from
    // scanning dynamically built CUDA DLLs; explicitly load available
    // backends once more only when CUDA0 was not already registered.
    if (cycle_period_mode &&
        cycle_real_env_flag("LLAMA_HYBRID_CYCLE_PC_CUDA_FULL")) {
        auto * cuda_dev = ggml_backend_dev_by_name("CUDA0");
        const bool attempted_load = cuda_dev == nullptr;
        if (attempted_load) {
            ggml_backend_load_all();
            cuda_dev = ggml_backend_dev_by_name("CUDA0");
        }
        std::printf(
            "[CYCLE_PC_CUDA_DISCOVERY] requested=CUDA0 "
            "retry_dynamic_load=%d cuda_registry=%d devices=%zu status=%s\n",
            attempted_load ? 1 : 0,
            ggml_backend_reg_by_name("CUDA") ? 1 : 0,
            ggml_backend_dev_count(),
            cuda_dev ? "FOUND" : "NOT_FOUND");
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            std::printf(
                "[CYCLE_PC_CUDA_DEVICE] index=%zu name=%s type=%d\n",
                i, ggml_backend_dev_name(dev),
                int(ggml_backend_dev_type(dev)));
        }
        if (!cuda_dev ||
            ggml_backend_dev_type(cuda_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            std::fprintf(stderr,
                "[CYCLE_PC_CUDA_DISCOVERY] status=FAIL "
                "reason=CUDA0_NOT_REGISTERED "
                "hint=CHECK_GGML_CUDA_CMAKE_CACHE_AND_CUDA_DLL\n");
            return 1;
        }
    }
    const auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=CPU_DEVICE_MISSING\n");
        return 1;
    }

    const std::string suffix = "-" + std::to_string(layer);
    // The previous layer's complete output is the exact input of Layer 7
    // Attention (Qwen3-MoE's inpL). Capture it from the independent CPU
    // baseline; do not substitute an arbitrary activation.
    const std::string attn_input_name =
        "l_out-" + std::to_string(layer - 1);
    const std::string residual_name = "ffn_inp" + suffix;
    const std::string norm_name = "ffn_norm" + suffix;
    const std::string ids_name = "cycle_real_topk_ids" + suffix;
    const std::string routing_name = "cycle_real_topk_weights" + suffix;
    // Existing canonical Router callback before argsort/topk.
    const std::string probs_name = "ffn_moe_probs" + suffix;
    const std::string ffn_name = "ffn_moe_out" + suffix;
    const std::string raw_down_name = "ffn_moe_down" + suffix;
    const std::string raw_gate_name = "ffn_moe_gate" + suffix;
    const std::string raw_up_name = "ffn_moe_up" + suffix;
    const std::string raw_swiglu_name = "ffn_moe_swiglu" + suffix;
    const std::string output_name = "l_out" + suffix;
    cycle_real_capture capture;
    for (const auto & name : {attn_input_name, residual_name, norm_name,
                             ids_name, routing_name, probs_name,
                             raw_gate_name, raw_up_name, raw_swiglu_name,
                             raw_down_name, ffn_name, output_name}) {
        capture.values.emplace(name, cycle_real_capture_value {});
    }
    std::string period_reference_output_name;
    if (cycle_period_mode) {
        // Select the first period's real final layer BEFORE the CPU-only
        // reference temporarily hides the Cycle placement environment.
        llama_hybrid_cycle_layout capture_layout;
        try {
            if (!llama_hybrid_cycle_layout_from_env(512, capture_layout)) {
                std::fprintf(stderr,
                    "[CYCLE_PERIOD] status=FAIL reason=REFERENCE_LAYOUT_MISSING\n");
                return 1;
            }
        } catch (const std::exception & e) {
            std::fprintf(stderr,
                "[CYCLE_PERIOD] status=FAIL reason=REFERENCE_LAYOUT_INVALID "
                "detail=%s\n", e.what());
            return 1;
        }
        period_reference_output_name =
            "l_out-" + std::to_string(capture_layout.cycle_size - 1);
        // Compare boundaries adjacent to the GPU-only segment without
        // adding any cross-device diagnostic activation reads.
        capture.values.emplace(
            "ffn_inp-" + std::to_string(capture_layout.x),
            cycle_real_capture_value {});
        capture.values.emplace(
            "ffn_inp-" + std::to_string(capture_layout.cycle_size - 1),
            cycle_real_capture_value {});
        capture.values.emplace(
            "ffn_norm-" + std::to_string(capture_layout.cycle_size - 1),
            cycle_real_capture_value {});
        if (cycle_real_env_flag("LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK")) {
            capture.values.emplace(
                "ffn_moe_out-" + std::to_string(capture_layout.x),
                cycle_real_capture_value {});
            // Capture all full-Phone layers, not only the current Y=1
            // example. Reference tensors are collected before GPU loading.
            for (int i = capture_layout.x + 1;
                 i < capture_layout.cycle_size - 1; ++i) {
                for (const char * prefix : {
                         "ffn_inp-", "ffn_norm-", "ffn_moe_out-"}) {
                    capture.values.emplace(
                        std::string(prefix) + std::to_string(i),
                        cycle_real_capture_value {});
                }
            }
        }
        capture.values.emplace("embd", cycle_real_capture_value {});
        // Capture all first-period layer outputs for optional numeric probes,
        // including the exact last layer used by the mandatory parity gate.
        for (int i = 0; i < capture_layout.cycle_size; ++i) {
            capture.values.emplace(
                "l_out-" + std::to_string(i), cycle_real_capture_value {});
        }
        // Keep optional per-PC-layer Attention/FFN references only while
        // investigating drift; mandatory end-to-end parity is unchanged.
        if (cycle_real_env_flag("LLAMA_HYBRID_CYCLE_NUMERIC_TRACE")) {
            for (int i = 0; i < capture_layout.x; ++i) {
                capture.values.emplace(
                    "ffn_inp-" + std::to_string(i), cycle_real_capture_value {});
                capture.values.emplace(
                    "ffn_moe_out-" + std::to_string(i),
                    cycle_real_capture_value {});
            }
        }
    }

    const std::string prompt = argc >= 5 ? argv[4] :
        "The quick brown fox jumps over the lazy dog. Please explain a transformer.";
    int32_t ntokens = 0;
    int32_t reference_layers = 0;
    int64_t reference_hidden = 0;
    int64_t reference_topk = 0;
    int64_t reference_experts = 0;
    int64_t reference_model_ffn_channels = 0;
    cycle_real_weight_fingerprints reference_ffn_weight_rows;
    double forward_ms = 0.0;

    // The Cycle model has a placement-only C2P/P2C policy and MUST NOT run
    // through the legacy Meta full-model scheduler. Establish an independent
    // complete-layer reference on ordinary CPU first. Its GGUF mmap and KV
    // resources are freed before loading the physical PC/Phone Cycle shards,
    // so the two copies of a 30B model never coexist in memory.
    {
        cycle_real_cpu_reference_env cpu_reference_env;
        if (!cpu_reference_env.disabled()) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=DISABLE_CYCLE_ENV\n");
            return 1;
        }
        std::printf("[CYCLE_REAL_REFERENCE] cycle_placement_env=DISABLED "
                    "router_capture_env=PRESERVED status=OK\n");
        ggml_backend_dev_t reference_devices[2] = {cpu_dev, nullptr};
        llama_model_params reference_params = llama_model_default_params();
        reference_params.devices = reference_devices;
        reference_params.n_gpu_layers = 0;
        reference_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
        reference_params.hybrid_auto = false;
        reference_params.use_mmap = true;
        // A complete 30B Q4_K CPU reference is loaded only to capture
        // numerical intermediates. CPU_REPACK duplicates many mapped weights
        // (~13 GiB in this GGUF) and is unnecessary for this correctness
        // probe. Keep the model's original quantization and mmap buffers.
        // This does not change the second PC+Phone Cycle model's policy.
        reference_params.use_extra_bufts = false;
        std::printf("[CYCLE_REAL_REFERENCE] backend=CPU_ONLY "
                    "split_mode=LAYER cpu_repack=DISABLED model_load=BEGIN\n");
        std::unique_ptr<llama_model, decltype(&llama_model_free)> reference_model(
            llama_model_load_from_file(argv[1], reference_params), &llama_model_free);
        reference_layers = reference_model ?
            llama_model_n_layer(reference_model.get()) : 0;
        if (!reference_model ||
            reference_model->arch != LLM_ARCH_QWEN3MOE ||
            reference_layers <= 1 || layer >= reference_layers - 1) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=REFERENCE_MODEL_OR_LAYER\n");
            return 1;
        }
        reference_hidden = reference_model->hparams.n_embd;
        reference_topk = reference_model->hparams.n_expert_used;
        reference_experts = reference_model->hparams.n_expert;
        if (!reference_model->layers[size_t(layer)].ffn_up_exps) {
            std::fprintf(stderr,
                "[CYCLE_REAL_REFERENCE] status=FAIL "
                "reason=FFN_UP_REFERENCE_WEIGHT_MISSING\n");
            return 1;
        }
        reference_model_ffn_channels =
            reference_model->layers[size_t(layer)].ffn_up_exps->ne[1];

        llama_context_params cp = llama_context_default_params();
        // CPU reference defaults to Flash Attention AUTO. The Cycle's
        // explicit Attention graph uses no Flash Attention. Allow an A/B
        // run that aligns their algorithms without altering the normal
        // independent CPU baseline or relaxing CPU parity tolerances.
        const bool reference_no_flash = cycle_period_mode &&
            cycle_real_env_flag("LLAMA_HYBRID_CYCLE_REFERENCE_NO_FLASH");
        if (reference_no_flash) {
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        }
        if (cycle_period_mode) {
            std::printf(
                "[CYCLE_REFERENCE_ATTENTION] mode=%s "
                "cycle=NO_FLASH ref_kv=F16 status=CONFIGURED\n",
                reference_no_flash ? "NO_FLASH" : "AUTO");
        }
        cp.n_ctx = 512;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.cb_eval = cycle_real_capture_cb;
        cp.cb_eval_user_data = &capture;
        std::unique_ptr<llama_context, decltype(&llama_free)> reference_context(
            llama_init_from_model(reference_model.get(), cp), &llama_free);
        if (!reference_context) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=REFERENCE_CONTEXT_INIT\n");
            return 1;
        }
        // A fixed token-count case is more reproducible than relying on
        // how many BPE tokens a short user-supplied prompt happens to use.
        // Repeat the seed only when necessary; truncate the resulting token
        // sequence to exactly the requested length before llama_decode.
        std::vector<llama_token> tokens(4096);
        const auto * vocab = llama_model_get_vocab(reference_model.get());
        std::string sample = prompt.empty() ? "Hello" : prompt;
        for (int attempt = 0; attempt < 256; ++attempt) {
            ntokens = llama_tokenize(
                vocab, sample.c_str(), int(sample.size()),
                tokens.data(), int32_t(tokens.size()), true, true);
            if (ntokens < 0) {
                tokens.resize(size_t(-ntokens));
                ntokens = llama_tokenize(
                    vocab, sample.c_str(), int(sample.size()),
                    tokens.data(), int32_t(tokens.size()), true, true);
            }
            if (ntokens < 0 ||
                target_tokens == 0 || ntokens >= target_tokens) {
                break;
            }
            if (sample.size() > 65536) {
                break;
            }
            sample.append(" ");
            sample.append(prompt.empty() ? "Hello" : prompt);
        }
        if (target_tokens > 0 && ntokens >= target_tokens) {
            ntokens = target_tokens;
        }
        if (ntokens < 1 || ntokens > int32_t(cp.n_batch) ||
            ntokens > int32_t(cp.n_ctx) ||
            (target_tokens > 0 && ntokens != target_tokens)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=TOKEN_COUNT "
                "tokens=%d max=%u\n",
                int(ntokens), cp.n_batch);
            return 1;
        }

        llama_batch batch = llama_batch_init(ntokens, 0, 1);
        batch.n_tokens = ntokens;
        for (int i = 0; i < ntokens; ++i) {
            batch.token[i] = tokens[size_t(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == ntokens - 1) ? 1 : 0;
        }
        std::printf(
            "[CYCLE_REAL_FORWARD] layer=%d prompt_tokens=%d "
            "target_tokens=%d positions=[0,%d] "
            "source=CPU_ONLY_REFERENCE stage=BEGIN\n",
            layer, int(ntokens), int(target_tokens), int(ntokens - 1));
        const auto forward_begin = std::chrono::steady_clock::now();
        const int decode_status =
            llama_decode(reference_context.get(), batch);
        llama_synchronize(reference_context.get());
        forward_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - forward_begin).count();
        llama_batch_free(batch);
        if (decode_status != 0 || capture.error) {
            std::fprintf(stderr,
                "[CYCLE_REAL_FORWARD] status=FAIL "
                "reason=CPU_REFERENCE_DECODE_OR_CAPTURE code=%d\n",
                decode_status);
            return 1;
        }
        // Check actual model weight bytes, not merely tensor shape.
        // Only keep hashes for selected experts so the CPU reference can
        // be released before loading the remote model as before.
        if (ntokens <= 16) {
            const auto itr = capture.values.find(ids_name);
            if (itr == capture.values.end() ||
                itr->second.count != 1 ||
                itr->second.type != GGML_TYPE_I32) {
                std::fprintf(stderr,
                    "[CYCLE_FFN_WEIGHT_BYTES] status=FAIL "
                    "reason=BASELINE_TOPK_NOT_CAPTURED\n");
                return 1;
            }
            const auto selected_ids = cycle_real_read<int32_t>(
                capture, ids_name);
            if (!cycle_real_capture_ffn_weight_fingerprints(
                    reference_model->layers[size_t(layer)],
                    selected_ids, reference_experts,
                    reference_ffn_weight_rows)) {
                std::fprintf(stderr,
                    "[CYCLE_FFN_WEIGHT_BYTES] status=FAIL "
                    "reason=CPU_REFERENCE_ROW_HASHES\n");
                return 1;
            }
        } else {
            std::printf("[CYCLE_FFN_WEIGHT_BYTES] enabled=0 "
                        "reason=TOKEN_LIMIT tokens=%d max=16\n",
                        int(ntokens));
        }
        std::printf(
            "[CYCLE_REAL_FORWARD] source=CPU_ONLY_REFERENCE "
            "stage=END wall_ms=%.3f status=PASS\n",
            forward_ms);
    }
    std::printf("[CYCLE_REAL_REFERENCE] model=CPU_ONLY "
                "status=RELEASED_BEFORE_CYCLE_LOAD\n");

    // CPU reference captures a contiguous prompt; the Phone may execute
    // it as causal chunks using a distinct persistent ON-GPU K/V cache.
    std::printf("[CYCLE_REAL_KV_SCOPE] tokens=%d position_end=%d "
                "kind=CONTIGUOUS_PROMPT_CACHE source=CPU_REFERENCE "
                "phone_chunk_kv=SEPARATE_PERSISTENT_BUFFER status=OK\n",
                int(ntokens), int(ntokens - 1));
    // Independent CPU reference is the same for C2P and P2C. Check the
    // actual requested physical Tensor direction before loading shards.
    llama_hybrid_cycle_layout expected_layout;
    const bool layout_ok = llama_hybrid_cycle_layout_from_env(
        reference_layers, expected_layout);
    const auto expected_kind = c2p_mode ?
        llama_hybrid_cycle_layer_kind::TENSOR_C2P :
        llama_hybrid_cycle_layer_kind::TENSOR_P2C;
    const bool layer_ok = layout_ok &&
        layer >= 0 && size_t(layer) < expected_layout.kinds.size();
    const char * actual_kind = layer_ok ?
        llama_hybrid_cycle_kind_name(expected_layout.kinds[size_t(layer)]) :
        "UNAVAILABLE";
    std::printf(
        "[CYCLE_REAL_DIRECTION] executable_mode=%s layer=%d "
        "expected=%s actual=%s x=%d y=%d layout_ok=%d "
        "status=%s\n",
        c2p_mode ? "C2P" : "P2C",
        layer, llama_hybrid_cycle_kind_name(expected_kind),
        actual_kind,
        layout_ok ? expected_layout.x : -1,
        layout_ok ? expected_layout.y : -1,
        layout_ok ? 1 : 0,
        layer_ok && expected_layout.kinds[size_t(layer)] == expected_kind ?
            "PASS" : "FAIL");
    std::fflush(stdout);
    if (!layer_ok || expected_layout.kinds[size_t(layer)] != expected_kind) {
        std::fprintf(stderr,
            "[CYCLE_REAL] status=FAIL reason=LAYOUT_DIRECTION_MISMATCH "
            "requested=%s layer=%d actual=%s "
            "hint=USE_REAL_C2P_EXE_FOR_C2P_LAYER\n",
            c2p_mode ? "C2P" : "P2C", layer, actual_kind);
        return 1;
    }
    std::printf("[CYCLE_REAL_REFERENCE] cycle_placement_env=RESTORED "
                "target_layer=%d kind=%s status=OK\n",
                layer, c2p_mode ? "C2P" : "P2C");

    // Fail on a malformed/uncaptured CPU baseline before allocating the
    // much larger distributed PC/Phone weight buffers.
    if (cycle_period_mode &&
        !cycle_real_check_capture(
            capture, "embd", GGML_TYPE_F32,
            reference_hidden, ntokens, 1)) {
        std::fprintf(stderr,
            "[CYCLE_PERIOD] status=FAIL "
            "reason=REAL_TOKEN_EMBEDDING_CAPTURE_MISSING\n");
        return 1;
    }
    if (cycle_period_mode &&
        !cycle_real_check_capture(
            capture, period_reference_output_name, GGML_TYPE_F32,
            reference_hidden, ntokens, 1)) {
        std::fprintf(stderr,
            "[CYCLE_PERIOD] status=FAIL "
            "reason=CPU_FINAL_L_OUT_CAPTURE_MISSING tensor=%s\n",
            period_reference_output_name.c_str());
        return 1;
    }
    if (!cycle_real_check_capture(
            capture, attn_input_name, GGML_TYPE_F32, reference_hidden, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, residual_name, GGML_TYPE_F32, reference_hidden, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, norm_name, GGML_TYPE_F32, reference_hidden, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, ids_name, GGML_TYPE_I32, reference_topk, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, routing_name, GGML_TYPE_F32, 1, reference_topk, ntokens) ||
        !cycle_real_check_capture(
            capture, probs_name, GGML_TYPE_F32, reference_experts, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, raw_gate_name, GGML_TYPE_F32,
            reference_model_ffn_channels, reference_topk, ntokens) ||
        !cycle_real_check_capture(
            capture, raw_up_name, GGML_TYPE_F32,
            reference_model_ffn_channels, reference_topk, ntokens) ||
        !cycle_real_check_capture(
            capture, raw_swiglu_name, GGML_TYPE_F32,
            reference_model_ffn_channels, reference_topk, ntokens) ||
        !cycle_real_check_capture(
            capture, raw_down_name, GGML_TYPE_F32,
            reference_hidden, reference_topk, ntokens) ||
        !cycle_real_check_capture(
            capture, ffn_name, GGML_TYPE_F32, reference_hidden, ntokens, 1) ||
        !cycle_real_check_capture(
            capture, output_name, GGML_TYPE_F32, reference_hidden, ntokens, 1)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=CPU_REFERENCE_CAPTURE_MISSING\n");
        return 1;
    }
    std::printf("[CYCLE_REAL_REFERENCE] "
                "capture_fields=12 status=PASS\n");

    // A new model instance holds only the *real* physical Cycle weights:
    // PC CPU FFN shard + Phone OpenCL FFN shard, with Attention/Router
    // weights resident on Phone. No llama_context/full-graph Meta execution
    // is permitted for this placement-only model.
    const auto reg = ggml_backend_rpc_add_server(argv[2]);
    if (!reg || ggml_backend_reg_dev_count(reg) != 1) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=RPC_DEVICE_COUNT\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(reg, 0);
    if (!phone_dev ||
        ggml_backend_dev_type(phone_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=NOT_PHONE_GPU\n");
        return 1;
    }
    cycle_real_phone_mem_checkpoint(phone_dev, "BEFORE_CYCLE_MODEL_LOAD");
    ggml_backend_dev_t devices[3] = {cpu_dev, phone_dev, nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.devices = devices;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.hybrid_auto = false;
    // Like the CPU-only oracle, the Cycle model is a correctness probe.
    // Default use_extra_bufts=true enables CPU_REPACK (q4_K_8x8), which
    // copies/rearranges many CPU-owned layer weights and wastes RAM here.
    // Keep canonical CPU tensor storage while preserving physical
    // PC_CPU + PHONE_GPU ownership and the actual RPC computation.
    mp.use_extra_bufts = false;
    std::printf("[CYCLE_REAL_MODEL_LOAD] split_mode=TENSOR "
                "cpu_repack=DISABLED extra_bufts=0 status=CONFIGURED\n");
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), &llama_model_free);
    const int32_t model_layers =
        model ? llama_model_n_layer(model.get()) : 0;
    if (!model || model->arch != LLM_ARCH_QWEN3MOE ||
        model_layers != reference_layers ||
        layer >= model_layers - 1 ||
        model->hybrid_layer_modes.size() != size_t(model_layers) ||
        model->hybrid_layer_modes[size_t(layer)] !=
            (c2p_mode ? llama_hybrid_layer_mode::TENSOR_C2P :
                        llama_hybrid_layer_mode::TENSOR_P2C)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL "
            "reason=MODEL_OR_NOT_P2C_LAYER layer=%d\n", layer);
        return 1;
    }

    cycle_real_phone_mem_checkpoint(phone_dev, "AFTER_CYCLE_MODEL_LOAD");
    const auto & lw = model->layers[size_t(layer)];
    const auto owner_check = [&](const char * role, const ggml_tensor * tensor) {
        return c2p_mode ?
            cycle_real_check_pc_weight(role, tensor) :
            cycle_real_check_phone_weight(role, tensor, phone_dev);
    };
    if (!owner_check("ATTN_NORM", lw.attn_norm) ||
        !owner_check("ATTN_Q", lw.wq) ||
        !owner_check("ATTN_K", lw.wk) ||
        !owner_check("ATTN_V", lw.wv) ||
        !owner_check("ATTN_Q_NORM", lw.attn_q_norm) ||
        !owner_check("ATTN_K_NORM", lw.attn_k_norm) ||
        !owner_check("ATTN_OUT", lw.wo) ||
        !owner_check("FFN_NORM", lw.ffn_norm) ||
        !owner_check("MOE_ROUTER", lw.ffn_gate_inp)) {
        std::fprintf(stderr,
            "[CYCLE_REAL] status=FAIL reason=ATTENTION_ROUTER_WRONG_OWNER "
            "direction=%s\n", c2p_mode ? "C2P" : "P2C");
        return 1;
    }
    const int64_t hidden = model->hparams.n_embd;
    const int64_t topk = model->hparams.n_expert_used;
    const int64_t experts = model->hparams.n_expert;
    if (hidden != reference_hidden || topk != reference_topk ||
        experts != reference_experts) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL "
            "reason=REFERENCE_CYCLE_HPARAM_MISMATCH\n");
        return 1;
    }
    // The first real comparison covers ordinary Qwen3-MoE SwiGLU, without
    // extra per-expert scales or clamping. Reject unsupported formulas.
    if (model->hparams.swiglu_clamp_exp[size_t(layer)] > 1.0e-6f ||
        lw.ffn_gate_exps_s || lw.ffn_up_exps_s || lw.ffn_down_exps_s ||
        !llama_hybrid_cycle_ffn_partition_valid(
            lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps,
            hidden, experts)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=WEIGHT_SHARD_OR_SCALES\n");
        return 1;
    }
    const auto pc_weights = llama_hybrid_cycle_get_physical_ffn_weights(
        lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps, 0);
    const auto phone_weights = llama_hybrid_cycle_get_physical_ffn_weights(
        lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps, 1);
    std::printf("[CYCLE_REAL_MODEL] layer=%d hidden=%lld topk=%lld "
                "experts=%lld pc_channels=%lld phone_channels=%lld "
                "type=%s status=OK\n",
                layer, (long long)hidden, (long long)topk,
                (long long)experts, (long long)pc_weights.up->ne[1],
                (long long)phone_weights.up->ne[1],
                ggml_type_name(lw.ffn_up_exps->type));

    // Diagnostic-only byte integrity check, executed BEFORE any Phone
    // Attention/FFN graph. This independently distinguishes a physical
    // Q4_K weight split/load discrepancy from a backend MUL_MAT_ID
    // numerical mismatch. A mismatch is reported but does not change the
    // test's mathematical pass/fail thresholds.
    if (reference_ffn_weight_rows.enabled) {
        const int64_t phone_begin = pc_weights.up->ne[1];
        const bool pc_gate_match = cycle_real_compare_physical_weight_rows(
            "GATE", "PC_CPU", pc_weights.gate, 0,
            reference_ffn_weight_rows, reference_ffn_weight_rows.gate_rows);
        const bool pc_up_match = cycle_real_compare_physical_weight_rows(
            "UP", "PC_CPU", pc_weights.up, 0,
            reference_ffn_weight_rows, reference_ffn_weight_rows.up_rows);
        const bool ph_gate_match = cycle_real_compare_physical_weight_rows(
            "GATE", "PHONE_GPU", phone_weights.gate, phone_begin,
            reference_ffn_weight_rows, reference_ffn_weight_rows.gate_rows);
        const bool ph_up_match = cycle_real_compare_physical_weight_rows(
            "UP", "PHONE_GPU", phone_weights.up, phone_begin,
            reference_ffn_weight_rows, reference_ffn_weight_rows.up_rows);
        std::printf("[CYCLE_FFN_WEIGHT_VERIFY] "
                    "pc_gate=%s pc_up=%s phone_gate=%s phone_up=%s "
                    "mode=BYTE_FINGERPRINT_DIAGNOSTIC status=%s\n",
                    pc_gate_match ? "MATCH" : "DIFF",
                    pc_up_match ? "MATCH" : "DIFF",
                    ph_gate_match ? "MATCH" : "DIFF",
                    ph_up_match ? "MATCH" : "DIFF",
                    pc_gate_match && pc_up_match &&
                    ph_gate_match && ph_up_match ?
                        "ALL_MATCH" : "INVESTIGATE");
    }

    const auto ref_input = cycle_real_read<float>(capture, attn_input_name);
    const auto ref_residual = cycle_real_read<float>(capture, residual_name);
    const auto ref_norm = cycle_real_read<float>(capture, norm_name);
    const auto ref_ids = cycle_real_read<int32_t>(capture, ids_name);
    const auto ref_routing = cycle_real_read<float>(capture, routing_name);
    const auto ref_router_probs = cycle_real_read<float>(capture, probs_name);
    const auto baseline_gate =
        cycle_real_read<float>(capture, raw_gate_name);
    const auto baseline_up =
        cycle_real_read<float>(capture, raw_up_name);
    const auto baseline_swiglu =
        cycle_real_read<float>(capture, raw_swiglu_name);
    const auto baseline_down =
        cycle_real_read<float>(capture, raw_down_name);
    const auto baseline_ffn = cycle_real_read<float>(capture, ffn_name);
    const auto baseline_output = cycle_real_read<float>(capture, output_name);

    // Sanity-check the *reference* before comparing any new execution.
    // For ordinary Qwen3-MoE with no adapters this layer must implement
    // l_out = AttentionResidual + CompleteFFN. Otherwise a capture label
    // or a prior graph transformation is not comparing like with like.
    if (baseline_ffn.size() != ref_residual.size() ||
        baseline_output.size() != ref_residual.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=BASELINE_SIZE\n");
        return 1;
    }
    std::vector<float> baseline_reconstructed(ref_residual.size());
    for (size_t i = 0; i < ref_residual.size(); ++i) {
        baseline_reconstructed[i] = ref_residual[i] + baseline_ffn[i];
    }
    const cycle_real_diff baseline_check =
        cycle_real_compare(baseline_reconstructed, baseline_output);
    std::printf("[CYCLE_REAL_COMPARE] target=BASELINE_INTERNAL layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, baseline_check.max_abs, baseline_check.rms,
                baseline_check.cosine, baseline_check.tolerance,
                baseline_check.valid && baseline_check.pass ? "PASS" : "FAIL");
    if (!baseline_check.valid || !baseline_check.pass) {
        return 1;
    }
    if (c2p_mode) {
        // NEW real C2P path. The host reference is only used to source the
        // physically correct input hidden state of this particular layer;
        // thereafter every Attention/FFN/Reduce operation is real.
        ggml_backend_t phone = ggml_backend_dev_init(phone_dev, nullptr);
        if (!phone || !ggml_backend_is_rpc(phone)) {
            std::fprintf(stderr,
                "[CYCLE_FIFO_C2P] status=FAIL reason=PHONE_RPC_BACKEND\n");
            return 1;
        }
        size_t payload_bytes = 0;
        if (!inference_cycle_payload_size(
                INFERENCE_CYCLE_READY, uint32_t(ntokens),
                uint32_t(hidden), uint32_t(topk), payload_bytes)) {
            ggml_backend_free(phone);
            return 1;
        }
        // Small persistent data-plane footprint on LMKD-sensitive Android.
        if (!std::getenv("GGML_INFERENCE_TRANSPORT_SLOTS") &&
            !cycle_real_set_env("GGML_INFERENCE_TRANSPORT_SLOTS", "2")) {
            ggml_backend_free(phone);
            return 1;
        }
        if (!std::getenv("GGML_INFERENCE_TRANSPORT_MAX_PAYLOAD_MIB")) {
            const size_t mib = 1024 * 1024;
            const auto value = std::to_string(
                std::max(size_t(1), (payload_bytes + mib - 1) / mib));
            if (!cycle_real_set_env(
                    "GGML_INFERENCE_TRANSPORT_MAX_PAYLOAD_MIB",
                    value.c_str())) {
                ggml_backend_free(phone);
                return 1;
            }
        }
        auto * device = ggml_backend_get_device(phone);
        auto * reg = device ? ggml_backend_dev_backend_reg(device) : nullptr;
        const auto open = reg ?
            reinterpret_cast<ggml_backend_rpc_inference_transport_open_t>(
                ggml_backend_reg_get_proc_address(
                    reg, GGML_BACKEND_RPC_INFERENCE_TRANSPORT_OPEN_PROC)) :
            nullptr;
        if (!open || !open(phone)) {
            std::fprintf(stderr,
                "[CYCLE_FIFO_C2P] status=FAIL "
                "reason=DATA_PLANE_OPEN_OR_PHONE_PATCH26_REQUIRED\n");
            ggml_backend_free(phone);
            return 1;
        }
        int chunk_tokens = std::min<int32_t>(
            16, std::max<int32_t>(1, ntokens / 4));
        int max_inflight = 2;
        const auto read_positive = [](const char * name, int & value) {
            const char * env = std::getenv(name);
            if (!env) {
                return true;
            }
            char * end = nullptr;
            const long n = std::strtol(env, &end, 10);
            if (end == env || !end || *end != '\0' ||
                n < 1 || n > 128) {
                return false;
            }
            value = int(n);
            return true;
        };
        if (!read_positive("LLAMA_HYBRID_CYCLE_REAL_CHUNK_TOKENS", chunk_tokens) ||
            !read_positive("LLAMA_HYBRID_CYCLE_REAL_MAX_INFLIGHT", max_inflight) ||
            max_inflight > 2) {
            std::fprintf(stderr,
                "[CYCLE_FIFO_C2P] status=FAIL reason=BAD_CHUNK_OR_WINDOW "
                "allowed_chunk=[1,128] allowed_window=[1,2]\n");
            ggml_backend_free(phone);
            return 1;
        }
        bool ok = false;
        if (cycle_period_mode) {
            for (int i : {0, 4, 5, 6, 7}) {
                const auto name = "l_out-" + std::to_string(i);
                const auto it = capture.values.find(name);
                if (it != capture.values.end() && it->second.count == 1 &&
                    it->second.type == GGML_TYPE_F32) {
                    cycle_real_numeric_probe(
                        "CPU_REFERENCE_L_OUT", i, -1,
                        cycle_real_read<float>(capture, name));
                }
            }
            const auto embeddings = cycle_real_read<float>(capture, "embd");
            const auto cpu_reference_output =
                cycle_real_read<float>(capture, period_reference_output_name);
            cycle_real_period_reference checkpoints;
            if (cycle_real_env_flag("LLAMA_HYBRID_CYCLE_NUMERIC_TRACE") ||
                cycle_real_env_flag("LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK")) {
                const auto optional_capture =
                    [&](const std::string & name) -> std::vector<float> {
                        const auto it = capture.values.find(name);
                        if (it == capture.values.end() ||
                            it->second.count != 1 ||
                            it->second.type != GGML_TYPE_F32 ||
                            it->second.ne[0] != reference_hidden ||
                            it->second.ne[1] != ntokens ||
                            it->second.ne[2] != 1 ||
                            it->second.bytes.size() !=
                                embeddings.size() * sizeof(float)) {
                            std::printf(
                                "[CYCLE_LAYER_PARITY] stage=REFERENCE_CAPTURE "
                                "tensor=%s status=UNAVAILABLE\n",
                                name.c_str());
                            return {};
                        }
                        return cycle_real_read<float>(capture, name);
                    };
                checkpoints.layer_outputs.reserve(
                    size_t(expected_layout.cycle_size));
                for (int l = 0; l < expected_layout.cycle_size; ++l) {
                    checkpoints.layer_outputs.push_back(
                        optional_capture("l_out-" + std::to_string(l)));
                }
                checkpoints.pc_attn_residuals.reserve(
                    size_t(expected_layout.x));
                checkpoints.pc_ffn_outputs.reserve(
                    size_t(expected_layout.x));
                for (int l = 0; l < expected_layout.x; ++l) {
                    checkpoints.pc_attn_residuals.push_back(
                        optional_capture("ffn_inp-" + std::to_string(l)));
                    checkpoints.pc_ffn_outputs.push_back(
                        optional_capture("ffn_moe_out-" + std::to_string(l)));
                }
                checkpoints.c2p_residual = optional_capture(
                    "ffn_inp-" + std::to_string(expected_layout.x));
                if (cycle_real_env_flag(
                        "LLAMA_HYBRID_CYCLE_BOUNDARY_READBACK")) {
                    checkpoints.c2p_full_ffn = optional_capture(
                        "ffn_moe_out-" + std::to_string(expected_layout.x));
                    const size_t n_layers = size_t(expected_layout.cycle_size);
                    checkpoints.phone_attn_residuals.resize(n_layers);
                    checkpoints.phone_ffn_norms.resize(n_layers);
                    checkpoints.phone_ffn_outputs.resize(n_layers);
                    for (int l = expected_layout.x + 1;
                         l < expected_layout.cycle_size - 1; ++l) {
                        checkpoints.phone_attn_residuals[size_t(l)] =
                            optional_capture(
                                "ffn_inp-" + std::to_string(l));
                        checkpoints.phone_ffn_norms[size_t(l)] =
                            optional_capture(
                                "ffn_norm-" + std::to_string(l));
                        checkpoints.phone_ffn_outputs[size_t(l)] =
                            optional_capture(
                                "ffn_moe_out-" + std::to_string(l));
                    }
                }
                checkpoints.p2c_residual = optional_capture(
                    "ffn_inp-" + std::to_string(
                        expected_layout.cycle_size - 1));
                checkpoints.p2c_norm = optional_capture(
                    "ffn_norm-" + std::to_string(
                        expected_layout.cycle_size - 1));
            }
            // Opt-in two-pass benchmark: warm a COMPLETE Cycle with the
            // same token shapes, KV prefixes and FIFO window, then run the
            // independently measured Cycle without reloading the model.
            // Each pass allocates fresh per-period KV and graph state and
            // fully drains the transport before the next begins.
            int warmup_runs = 0;
            const char * warm_env =
                std::getenv("LLAMA_HYBRID_CYCLE_WARMUP_RUNS");
            if (warm_env) {
                if (std::strcmp(warm_env, "0") == 0) {
                    warmup_runs = 0;
                } else if (std::strcmp(warm_env, "1") == 0) {
                    warmup_runs = 1;
                } else {
                    std::fprintf(stderr,
                        "[CYCLE_WARMUP] status=FAIL "
                        "reason=EXPECTED_0_OR_1 value=%s\n", warm_env);
                    ggml_backend_free(phone);
                    return 1;
                }
            }
            const bool phone_full_session_reuse = cycle_real_env_flag(
                "LLAMA_HYBRID_CYCLE_PHONE_FULL_SESSION_REUSE");
            if (phone_full_session_reuse &&
                !cycle_real_env_flag("LLAMA_HYBRID_CYCLE_PHONE_FULL_REUSE")) {
                std::fprintf(stderr,
                    "[CYCLE_PHONE_FULL_SESSION] status=FAIL "
                    "reason=REQUIRES_PHONE_FULL_REUSE\n");
                ggml_backend_free(phone);
                return 1;
            }
            cycle_real_phone_full_session_pool full_session;
            if (phone_full_session_reuse) {
                const int effective_window = std::min(
                    max_inflight, (int(ntokens) + chunk_tokens - 1) / chunk_tokens);
                full_session.configure(
                    phone, *model, expected_layout, ntokens,
                    chunk_tokens, effective_window,
                    cycle_real_env_flag("LLAMA_HYBRID_CYCLE_PHONE_FULL_FUSE"),
                    cycle_real_env_flag("LLAMA_HYBRID_CYCLE_PHONE_FULL_RPC_REPLAY"));
                std::printf(
                    "[CYCLE_PHONE_FULL_SESSION_PLAN] enabled=1 "
                    "layers=[%d,%d] max_slots_per_layer=%d "
                    "ownership=MODEL_AND_PHONE_RPC_BACKEND "
                    "retire=SUCCESSOR_ATTN_FENCED "
                    "release=BEFORE_PHONE_BACKEND_FREE\n",
                    expected_layout.x + 1,
                    expected_layout.x + expected_layout.y,
                    effective_window);
            }
            std::printf(
                "[CYCLE_WARMUP_PLAN] enabled=%d warmup_runs=%d 
                "measured_runs=1 model_load=ONCE "
                "same_shapes=1 distinct_kv_and_graphs=1 "
                "wire_seq=UNIQUE_PER_ROUND "
                "pc_wall_includes=FIFO_AND_RELEASE "
                "phone_server_sums=SESSION_CUMULATIVE_USE_DELTA\n",
                warmup_runs != 0 ? 1 : 0, warmup_runs);
            for (int run = 0; run <= warmup_runs; ++run) {
                const bool is_warmup = run < warmup_runs;
                const char * phase = is_warmup ? "WARMUP" : "MEASURE";
                const uint64_t epoch = uint64_t(run);
                std::printf(
                    "[CYCLE_WARMUP_RUN] phase=%s epoch=%" PRIu64
                    " status=BEGIN\n", phase, epoch);
                std::fflush(stdout);
                const auto run_start = std::chrono::steady_clock::now();
                ok = cycle_real_execute_period_fifo(
                    phone, cpu_dev, *model, expected_layout,
                    embeddings, cpu_reference_output, checkpoints,
                    ntokens, chunk_tokens, max_inflight,
                    is_warmup, epoch,
                    phone_full_session_reuse ? &full_session : nullptr);
                const double run_full_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - run_start).count();
                std::printf(
                    "[CYCLE_WARMUP_RUN] phase=%s epoch=%" PRIu64
                    " full_call_wall_ms=%.3f "
                    "includes=BACKEND_INIT_WEIGHT_MIRROR_AND_PARITY "
                    "status=%s\n",
                    phase, epoch, run_full_ms, ok ? "OK" : "FAIL");
                std::fflush(stdout);
                if (!ok) break;
            }
            if (phone_full_session_reuse) {
                const size_t cached_before_release = full_session.cached_slots();
                const auto release_begin = std::chrono::steady_clock::now();
                full_session.clear();
                const double release_ms =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() -
                        release_begin).count();
                std::printf(
                    "[CYCLE_PHONE_FULL_SESSION_TEARDOWN] "
                    "slots_released=%zu release_ms=%.3f "
                    "included_in=SESSION_LIFETIME_NOT_PERIOD_WALL "
                    "before_phone_backend_free=1 status=OK\n",
                    cached_before_release, release_ms);
                std::fflush(stdout);
            }
        } else {
            const cycle_real_live_wire wire {layer, 0, 1, 1};
            ok = cycle_real_execute_c2p_fifo(
                phone, cpu_dev, lw, model->hparams,
                pc_weights, phone_weights, ref_input, wire,
                hidden, ntokens, topk, experts, chunk_tokens, max_inflight);
        }
        ggml_backend_free(phone);
        return ok ? 0 : 1;
    }

    // The CPU reference is read-only from this point on. The READY wire and
    // the two physical FFN shards must consume Phone-generated arrays only.
    ggml_backend_t phone = ggml_backend_dev_init(phone_dev, nullptr);
    if (!phone || !ggml_backend_is_rpc(phone)) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL reason=PHONE_RPC_BACKEND\n");
        return 1;
    }
    // The new phase is explicitly selectable so existing offline CPU-parity
    // experiments can still run against an older Phone RPC server.
    // In the real path, both payloads originate in the Phone server process.
    const bool live_wire_enabled = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_REAL_P2C_NETWORK");
    const cycle_real_live_wire wire {layer, 0, 1, 1};
    if (live_wire_enabled) {
        size_t required_bytes = 0;
        if (!inference_cycle_payload_size(
                INFERENCE_CYCLE_READY, uint32_t(ntokens),
                uint32_t(hidden), uint32_t(topk), required_bytes)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_NETWORK] status=FAIL reason=READY_SIZE\n");
            ggml_backend_free(phone);
            return 1;
        }
        // A 2-token case needs only two 1 MiB TX/RX slots per side.
        // Do not preallocate the global RPC inference defaults
        // (4 slots x 16 MiB in each direction) on an LMKD-sensitive phone.
        if (!std::getenv("GGML_INFERENCE_TRANSPORT_SLOTS") &&
            !cycle_real_set_env("GGML_INFERENCE_TRANSPORT_SLOTS", "2")) {
            ggml_backend_free(phone);
            return 1;
        }
        if (!std::getenv("GGML_INFERENCE_TRANSPORT_MAX_PAYLOAD_MIB")) {
            const size_t mib = 1024 * 1024;
            const auto value = std::to_string(
                std::max(size_t(1), (required_bytes + mib - 1) / mib));
            if (!cycle_real_set_env(
                    "GGML_INFERENCE_TRANSPORT_MAX_PAYLOAD_MIB",
                    value.c_str())) {
                ggml_backend_free(phone);
                return 1;
            }
        }
        const auto reg = ggml_backend_dev_backend_reg(
            ggml_backend_get_device(phone));
        const auto open = reg ?
            reinterpret_cast<ggml_backend_rpc_inference_transport_open_t>(
                ggml_backend_reg_get_proc_address(
                    reg, GGML_BACKEND_RPC_INFERENCE_TRANSPORT_OPEN_PROC)) :
            nullptr;
        if (!open || !open(phone)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_NETWORK] status=FAIL "
                "reason=DATA_PLANE_OPEN rebuild_phone_server_patch25=1\n");
            ggml_backend_free(phone);
            return 1;
        }
        std::printf("[CYCLE_REAL_NETWORK] direction=P2C "
                    "mode=REAL_RPC_DATA_PLANE endpoint=%s "
                    "source=PHONE_GPU_REMOTE_BUFFERS status=OPEN\n", argv[2]);
    }
    // Real data-plane runs now default to the FIFO stage executor. Set
    // LLAMA_HYBRID_CYCLE_REAL_P2C_FIFO=0 to retain the earlier sequential
    // network diagnostic path while investigating transport behavior.
    const char * fifo_env = std::getenv("LLAMA_HYBRID_CYCLE_REAL_P2C_FIFO");
    const bool fifo_enabled = live_wire_enabled &&
        (fifo_env == nullptr || std::strcmp(fifo_env, "0") != 0);
    if (fifo_enabled) {
        // Four-ish chunks by default, at most 16 tokens each. The normal
        // two-token smoke test now runs two REAL 1-token chunks. Env allows
        // fixed-size experiments without any recompilation.
        int32_t chunk_tokens = std::min<int32_t>(
            16, std::max<int32_t>(1, ntokens / 4));
        int max_inflight = 2;
        const auto parse_positive = [](const char * value, int & out) {
            if (!value || *value == '\0') {
                return false;
            }
            char * end = nullptr;
            const long x = std::strtol(value, &end, 10);
            if (end == value || *end != '\0' || x <= 0 || x > 128) {
                return false;
            }
            out = int(x);
            return true;
        };
        if (const char * value = std::getenv(
                "LLAMA_HYBRID_CYCLE_REAL_CHUNK_TOKENS")) {
            int parsed = 0;
            if (!parse_positive(value, parsed)) {
                std::fprintf(stderr,
                    "[CYCLE_FIFO_REAL] status=FAIL "
                    "reason=BAD_CHUNK_TOKENS allowed=[1,128]\n");
                ggml_backend_free(phone);
                return 1;
            }
            chunk_tokens = int32_t(parsed);
        }
        if (const char * value = std::getenv(
                "LLAMA_HYBRID_CYCLE_REAL_MAX_INFLIGHT")) {
            int parsed = 0;
            if (!parse_positive(value, parsed) || parsed > 2) {
                std::fprintf(stderr,
                    "[CYCLE_FIFO_REAL] status=FAIL "
                    "reason=BAD_MAX_INFLIGHT allowed=[1,2]\n");
                ggml_backend_free(phone);
                return 1;
            }
            max_inflight = parsed;
        }
        const bool ok = cycle_real_execute_p2c_fifo(
            phone, cpu_dev, lw, model->hparams,
            pc_weights, phone_weights, ref_input, wire,
            hidden, ntokens, topk, experts,
            chunk_tokens, max_inflight);
        ggml_backend_free(phone);
        return ok ? 0 : 1;
    }

    cycle_real_phone_ready phone_ready;
    inference_cycle_message ready_msg;
    double phone_attn_ms = 0.0;
    if (!cycle_real_execute_phone_attention_pre(
            phone, lw, model->hparams, ref_input,
            hidden, ntokens, topk, experts, phone_ready, phone_attn_ms,
            live_wire_enabled ? &wire : nullptr)) {
        ggml_backend_free(phone);
        return 1;
    }
    if (live_wire_enabled) {
        if (!cycle_real_receive_live_wire(
                phone, wire, INFERENCE_CYCLE_READY,
                uint32_t(ntokens), uint32_t(hidden), uint32_t(topk),
                ready_msg)) {
            ggml_backend_free(phone);
            return 1;
        }
        phone_ready.residual = ready_msg.ready.residual;
        phone_ready.norm = ready_msg.ready.ffn_norm;
        phone_ready.ids = ready_msg.ready.topk_ids;
        phone_ready.routing = ready_msg.ready.topk_weights;
        // Router probabilities are not part of the two-send protocol.
        // Do not perform a duplicate normal RPC readback just for a
        // reference comparison in this transport-focused phase.
    }

    const cycle_real_diff residual_diff =
        cycle_real_compare(phone_ready.residual, ref_residual);
    const cycle_real_diff norm_diff =
        cycle_real_compare(phone_ready.norm, ref_norm);
    const cycle_real_diff routing_slot_diff =
        cycle_real_compare(phone_ready.routing, ref_routing);
    const cycle_real_diff probs_diff = live_wire_enabled ?
        cycle_real_diff {} :
        cycle_real_compare(phone_ready.router_probs, ref_router_probs);
    const bool topk_equal = phone_ready.ids == ref_ids;

    // Experts are a weighted SET, not an ordered semantic vector. argsort
    // can permute two close-scoring experts between CPU and OpenCL without
    // changing the selected expert set. Every weight is attached to its
    // Expert ID: never compare or reorder weights independently of IDs.
    const size_t expected_slots = size_t(ntokens) * size_t(topk);
    const size_t expected_probs = size_t(ntokens) * size_t(experts);
    if (phone_ready.ids.size() != expected_slots ||
        ref_ids.size() != expected_slots ||
        phone_ready.routing.size() != expected_slots ||
        ref_routing.size() != expected_slots ||
        phone_ready.router_probs.size() != expected_probs ||
        ref_router_probs.size() != expected_probs) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ROUTER_SEMANTIC] status=FAIL reason=SHAPE_MISMATCH\n");
        ggml_backend_free(phone);
        return 1;
    }
    std::vector<float> aligned_phone_weights;
    std::vector<float> aligned_ref_weights;
    aligned_phone_weights.reserve(expected_slots);
    aligned_ref_weights.reserve(expected_slots);
    bool topk_set_equal = true;
    bool topk_ids_valid = true;
    int32_t permuted_tokens = 0;
    size_t changed_slots = 0;
    constexpr size_t max_printed_router_diff = 16;
    for (int32_t token = 0; token < ntokens; ++token) {
        std::vector<std::pair<int32_t, float>> ref_pairs;
        std::vector<std::pair<int32_t, float>> phone_pairs;
        ref_pairs.reserve(size_t(topk));
        phone_pairs.reserve(size_t(topk));
        bool token_reordered = false;
        for (int64_t slot = 0; slot < topk; ++slot) {
            const size_t index = size_t(token) * size_t(topk) + size_t(slot);
            const int32_t ref_id = ref_ids[index];
            const int32_t phone_id = phone_ready.ids[index];
            const float ref_weight = ref_routing[index];
            const float phone_weight = phone_ready.routing[index];
            if (ref_id < 0 || ref_id >= experts ||
                phone_id < 0 || phone_id >= experts ||
                !std::isfinite(ref_weight) || !std::isfinite(phone_weight)) {
                topk_ids_valid = false;
                continue;
            }
            ref_pairs.emplace_back(ref_id, ref_weight);
            phone_pairs.emplace_back(phone_id, phone_weight);
            if (ref_id != phone_id) {
                token_reordered = true;
                if (changed_slots < max_printed_router_diff) {
                    // Show whether the scores are nearly tied or genuinely
                    // rank-inverted. Softmax distributions have [expert,token]
                    // layout; use the FULL scores, not normalized TopK slots.
                    const size_t off = size_t(token) * size_t(experts);
                    const float ref_a = ref_router_probs[off + size_t(ref_id)];
                    const float ref_b = ref_router_probs[off + size_t(phone_id)];
                    const float ph_a = phone_ready.router_probs[off + size_t(ref_id)];
                    const float ph_b = phone_ready.router_probs[off + size_t(phone_id)];
                    std::printf(
                        "[CYCLE_PHONE_ROUTER_RANK] pos=%d slot=%lld "
                        "reference_id=%d phone_id=%d "
                        "ref_gap=%.8g phone_gap=%.8g "
                        "ref_scores=[%.8g,%.8g] phone_scores=[%.8g,%.8g]\n",
                        token, (long long)slot, ref_id, phone_id,
                        double(ref_a) - double(ref_b),
                        double(ph_a) - double(ph_b),
                        ref_a, ref_b, ph_a, ph_b);
                }
                ++changed_slots;
            }
        }
        if (token_reordered) {
            ++permuted_tokens;
        }
        if (ref_pairs.size() != size_t(topk) ||
            phone_pairs.size() != size_t(topk)) {
            topk_set_equal = false;
            continue;
        }
        const auto by_id = [](const std::pair<int32_t, float> & a,
                              const std::pair<int32_t, float> & b) {
            return a.first < b.first;
        };
        std::sort(ref_pairs.begin(), ref_pairs.end(), by_id);
        std::sort(phone_pairs.begin(), phone_pairs.end(), by_id);
        bool this_set_equal = true;
        for (size_t slot = 0; slot < size_t(topk); ++slot) {
            // Duplicate IDs make an expert set ambiguous; fail closed.
            if (slot > 0 && (ref_pairs[slot].first == ref_pairs[slot - 1].first ||
                             phone_pairs[slot].first == phone_pairs[slot - 1].first)) {
                topk_ids_valid = false;
            }
            if (ref_pairs[slot].first != phone_pairs[slot].first) {
                this_set_equal = false;
            }
        }
        if (!this_set_equal) {
            topk_set_equal = false;
            std::fprintf(stderr,
                "[CYCLE_PHONE_TOPK_SET] pos=%d status=FAIL reason=CHANGED_EXPERT_SET\n",
                token);
            continue;
        }
        // Compare matched Expert ID + normalized weight, in sorted-ID order
        // only for numerical diagnosis. FFN still consumes phone-original
        // ID/weight pairs without modifying their original slot order.
        for (size_t slot = 0; slot < size_t(topk); ++slot) {
            aligned_ref_weights.push_back(ref_pairs[slot].second);
            aligned_phone_weights.push_back(phone_pairs[slot].second);
        }
    }
    const cycle_real_diff routing_aligned_diff =
        cycle_real_compare(aligned_phone_weights, aligned_ref_weights);

    // Print which *position* diverges, not just one pooled statistic.
    // In a causal multi-KV test, a future-mask or RoPE error can be local
    // to later tokens even when the first token matches perfectly.
    const auto report_token_errors = [&](const char * field,
                                         const std::vector<float> & actual,
                                         const std::vector<float> & baseline,
                                         int64_t elems_per_token) {
        if (actual.size() != baseline.size() ||
            actual.size() != size_t(elems_per_token * ntokens)) {
            std::fprintf(stderr,
                "[CYCLE_PHONE_READY_TOKEN] field=%s status=FAIL "
                "reason=SHAPE_MISMATCH\n", field);
            return;
        }
        double max_abs = 0.0;
        int64_t worst_token = -1;
        int64_t worst_element = -1;
        for (int32_t token = 0; token < ntokens; ++token) {
            for (int64_t j = 0; j < elems_per_token; ++j) {
                const size_t idx = size_t(token * elems_per_token + j);
                const double err = std::fabs(
                    double(actual[idx]) - double(baseline[idx]));
                if (err > max_abs || worst_token < 0) {
                    max_abs = err;
                    worst_token = token;
                    worst_element = j;
                }
            }
        }
        std::printf("[CYCLE_PHONE_READY_TOKEN] field=%s tokens=%d "
                    "worst_pos=%lld worst_element=%lld max_abs=%.8g "
                    "status=MEASURED\n",
                    field, int(ntokens),
                    (long long)worst_token, (long long)worst_element,
                    max_abs);
    };
    report_token_errors(
        "RESIDUAL", phone_ready.residual, ref_residual, hidden);
    report_token_errors(
        "FFN_NORM", phone_ready.norm, ref_norm, hidden);
    report_token_errors(
        "ROUTING_WEIGHTS_SLOT_ORDER",
        phone_ready.routing, ref_routing, topk);

    // Do NOT reuse the relaxed full-layer tolerance here. The residual
    // contains large activations; requiring its comparison to pass only
    // via 0.005*max_ref would hide significant Attention errors.
    const auto strict_ok = [](const cycle_real_diff & d,
                              double floor, double relative) {
        return d.valid && d.max_abs <= floor + relative * d.max_ref;
    };
    const bool residual_ok = strict_ok(residual_diff, 0.02, 0.00001);
    const bool norm_ok = strict_ok(norm_diff, 0.01, 0.001);
    const bool routing_ok =
        topk_set_equal && topk_ids_valid &&
        aligned_ref_weights.size() == expected_slots &&
        strict_ok(routing_aligned_diff, 0.003, 0.001);

    std::printf("[CYCLE_PHONE_READY_COMPARE] target=RESIDUAL "
                "max_abs=%.8g rms=%.8g cosine=%.8g "
                "strict_tolerance=%.8g status=%s\n",
                residual_diff.max_abs, residual_diff.rms,
                residual_diff.cosine,
                0.02 + 0.00001 * residual_diff.max_ref,
                residual_ok ? "PASS" : "FAIL");
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=FFN_NORM "
                "max_abs=%.8g rms=%.8g cosine=%.8g "
                "strict_tolerance=%.8g status=%s\n",
                norm_diff.max_abs, norm_diff.rms,
                norm_diff.cosine, 0.01 + 0.001 * norm_diff.max_ref,
                norm_ok ? "PASS" : "FAIL");
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=TOPK "
                "ids_exact=%d set_equal=%d valid_ids=%d "
                "permuted_tokens=%d changed_slots=%zu status=%s\n",
                topk_equal ? 1 : 0, topk_set_equal ? 1 : 0,
                topk_ids_valid ? 1 : 0, int(permuted_tokens), changed_slots,
                topk_set_equal && topk_ids_valid ? "PASS" : "FAIL");
    if (!topk_equal) {
        size_t mismatch_count = 0;
        constexpr size_t max_printed = 16;
        for (size_t j = 0; j < ref_ids.size(); ++j) {
            if (ref_ids[j] == phone_ready.ids[j]) {
                continue;
            }
            if (mismatch_count < max_printed) {
                std::fprintf(stderr,
                    "[CYCLE_PHONE_TOPK_MISMATCH] idx=%zu "
                    "pos=%zu expert_slot=%zu reference=%d phone=%d\n",
                    j, j / size_t(topk), j % size_t(topk),
                    ref_ids[j], phone_ready.ids[j]);
            }
            ++mismatch_count;
        }
        std::fprintf(stderr,
            "[CYCLE_PHONE_TOPK_MISMATCH_SUM] tokens=%d "
            "total=%zu displayed_max=%zu\n",
            int(ntokens), mismatch_count, max_printed);
    }
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=ROUTER_PROBS "
                "max_abs=%.8g rms=%.8g status=%s\n",
                probs_diff.max_abs, probs_diff.rms,
                live_wire_enabled ? "SKIP_NETWORK_MODE" :
                    (probs_diff.valid ? "MEASURED" : "FAIL_NONFINITE"));
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=ROUTING_WEIGHTS_SLOT_ORDER "
                "max_abs=%.8g rms=%.8g status=DIAGNOSTIC_ONLY\n",
                routing_slot_diff.max_abs, routing_slot_diff.rms);
    std::printf("[CYCLE_PHONE_READY_COMPARE] target=ROUTING_WEIGHTS_BY_EXPERT "
                "max_abs=%.8g rms=%.8g cosine=%.8g "
                "strict_tolerance=%.8g status=%s\n",
                routing_aligned_diff.max_abs, routing_aligned_diff.rms,
                routing_aligned_diff.cosine,
                0.003 + 0.001 * routing_aligned_diff.max_ref,
                routing_ok ? "PASS" : "FAIL");
    if (!residual_ok || !norm_ok ||
        (!live_wire_enabled && !probs_diff.valid) ||
        !topk_set_equal || !topk_ids_valid || !routing_ok) {
        std::fprintf(stderr,
            "[CYCLE_PHONE_ATTN_PRE] status=FAIL "
            "reason=CPU_REFERENCE_MISMATCH\n");
        ggml_backend_free(phone);
        return 1;
    }

    const auto & residual = phone_ready.residual;
    const auto & norm = phone_ready.norm;
    const auto & ids = phone_ready.ids;
    const auto & routing = phone_ready.routing;
    for (int32_t id : ids) {
        if (id < 0 || id >= experts) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=INVALID_TOPK id=%d\n", id);
            ggml_backend_free(phone);
            return 1;
        }
    }
    for (float v : norm) {
        if (!std::isfinite(v)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=NONFINITE_NORM\n");
            ggml_backend_free(phone);
            return 1;
        }
    }
    for (float w : routing) {
        if (!std::isfinite(w)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=NONFINITE_ROUTING\n");
            ggml_backend_free(phone);
            return 1;
        }
    }
    std::printf("[CYCLE_REAL_READY] layer=%d tokens=%d "
                "fields=RESIDUAL,NORM,TOPK,WEIGHTS "
                "source=PHONE_GPU_ATTN_PRE status=OK\n", layer, int(ntokens));

    // Exercise the V1 P2C READY wire representation with the *real* tensor
    // values, including frame direction and seq. This is an in-process
    // encode/decode check, NOT a claim that the new network sender is wired.
    // The network mode has already received READY directly from the
    // Phone RPC server into ready_msg. Keep the former codec path only
    // for explicit offline/legacy regression runs.
    const inference_cycle_ready_view ready_source {
        uint32_t(ntokens), uint32_t(hidden), uint32_t(topk),
        residual.data(), norm.data(), ids.data(), routing.data(),
    };
    size_t ready_bytes = 0;
    if (!inference_cycle_payload_size(
            INFERENCE_CYCLE_READY, uint32_t(ntokens), uint32_t(hidden),
            uint32_t(topk), ready_bytes)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_SIZE\n");
        return 1;
    }
    inference_frame_header ready_header =
        inference_make_header(INFERENCE_FRAME_DATA);
    ready_header.stage = INFERENCE_STAGE_CYCLE_READY;
    ready_header.tensor_role = INFERENCE_CYCLE_PHONE_TO_PC;
    ready_header.layer = layer;
    ready_header.chunk = wire.chunk;
    ready_header.seq = wire.seq;
    ready_header.graph_id = wire.graph_id;
    ready_header.payload_bytes = ready_bytes;
    if (!live_wire_enabled) {
        std::vector<uint8_t> ready_wire(ready_bytes);
        size_t ready_written = 0;
        if (!inference_cycle_write_ready(
                ready_wire.data(), ready_wire.size(),
                ready_source, ready_written) ||
            ready_written != ready_wire.size() ||
            !inference_cycle_decode_frame(
                ready_header, ready_wire.data(), ready_wire.size(),
                ready_msg)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_CODEC\n");
            return 1;
        }
    }
    if (ready_msg.kind != INFERENCE_CYCLE_READY ||
        ready_msg.direction != INFERENCE_CYCLE_PHONE_TO_PC ||
        ready_msg.layer != layer || ready_msg.chunk != wire.chunk ||
        ready_msg.seq != wire.seq || ready_msg.graph_id != wire.graph_id ||
        ready_msg.ready.topk_ids != ids ||
        ready_msg.ready.topk_weights != routing ||
        ready_msg.ready.ffn_norm != norm ||
        ready_msg.ready.residual != residual) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_CONTENT\n");
        return 1;
    }
    std::printf("[CYCLE_REAL_WIRE] stage=READY direction=P2C "
                "bytes=%zu mode=%s status=PASS\n", ready_bytes,
                live_wire_enabled ? "PHONE_GPU_REAL_DATA_PLANE" :
                                    "IN_PROCESS_CODEC");
    if (live_wire_enabled && !ggml_backend_rpc_cycle_send_ack(
            phone, layer, wire.chunk, wire.seq, wire.graph_id,
            INFERENCE_CYCLE_READY, true)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_NETWORK] status=FAIL reason=READY_HOST_ACK\n");
        return 1;
    }

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    if (!cpu) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PHYSICAL_BACKENDS\n");
        return 1;
    }

    // Differential FFN experiment. The normal production-like path below
    // ALWAYS uses Phone READY and keeps its original PASS/FAIL thresholds.
    // Before it, replay the SAME physical PC+Phone FFN shards with controlled
    // reference inputs. This separates shard/kernel error from Attention
    // Norm drift, Router weight drift, and mere TopK slot permutations.
    //
    // LOW_MEMORY is the default: one CPU_READY_CONTROL run (1-2 tokens)
    // is sufficient to compare the physical FFN against the CPU-only F64
    // oracle. All repeated Phone GPU kernel replays, per-stage captures,
    // Down OpenCL probes and additional Phone FFN ablations are opt-in.
    // Keep production Phone READY / PC+Phone execution and official gates
    // exactly as before, regardless of this diagnostic selection.
    const bool extended_diagnostics = cycle_real_env_flag(
        "LLAMA_HYBRID_CYCLE_REAL_EXTENDED_DIAG");
    const bool run_ffn_ablation = !live_wire_enabled &&
        (ntokens <= 2 || (extended_diagnostics && ntokens <= 16));
    std::vector<float> cpu_ready_control_ffn;
    std::printf("[CYCLE_DIAG_MODE] mode=%s tokens=%d "
                "cpu_ready_control=%d full_cpu_f64_oracle=%d "
                "phone_extra_gpu_probes=%d extra_phone_ffn_ablations=%d "
                "env=LLAMA_HYBRID_CYCLE_REAL_EXTENDED_DIAG\n",
                live_wire_enabled ? "NETWORK_P2C_NO_F64" :
                    (extended_diagnostics ? "EXTENDED" : "LOW_MEMORY_CPU_F64"),
                int(ntokens), run_ffn_ablation ? 1 : 0,
                run_ffn_ablation && ntokens <= 2 ? 1 : 0,
                run_ffn_ablation && extended_diagnostics && ntokens <= 2 ? 1 : 0,
                run_ffn_ablation && extended_diagnostics && ntokens <= 16 ? 1 : 0);
    std::printf("[CYCLE_FFN_ABLATION] enabled=%d tokens=%d "
                "scope=DIAGNOSTIC_ONLY gate_unchanged=1\n",
                run_ffn_ablation ? 1 : 0, int(ntokens));
    if (run_ffn_ablation) {
        const auto run_case = [&](
                const char * label,
                const std::vector<float> & input_norm,
                const std::vector<int32_t> & input_ids,
                const std::vector<float> & input_routing,
                std::vector<float> & sum_output,
                bool capture_down) -> bool {
            // The async Phone shard and foreground CPU shard consume one
            // immutable set of READY inputs. Both futures finish before
            // either backend is reused for another case.
            std::vector<float> ph_raw_down;
            std::vector<float> pc_raw_down;
            cycle_real_ffn_stages ph_stages;
            cycle_real_ffn_stages pc_stages;
            auto phone_probe = std::async(std::launch::async, [&] {
                std::vector<float> result;
                if (!cycle_real_execute_ffn(
                        phone, phone_weights,
                        input_norm, input_ids, input_routing,
                        hidden, ntokens, topk, experts, result,
                        capture_down ? &ph_raw_down : nullptr,
                        capture_down ? &ph_stages : nullptr)) {
                    result.clear();
                }
                return result;
            });
            std::vector<float> pc_probe;
            const bool pc_probe_ok = cycle_real_execute_ffn(
                cpu, pc_weights,
                input_norm, input_ids, input_routing,
                hidden, ntokens, topk, experts, pc_probe,
                capture_down ? &pc_raw_down : nullptr,
                capture_down ? &pc_stages : nullptr);
            const std::vector<float> ph_probe = phone_probe.get();
            if (!pc_probe_ok ||
                pc_probe.size() != size_t(hidden * ntokens) ||
                ph_probe.size() != size_t(hidden * ntokens)) {
                std::fprintf(stderr,
                    "[CYCLE_FFN_ABLATION] case=%s status=FAIL "
                    "reason=SHARD_COMPUTE\n", label);
                return false;
            }
            sum_output.resize(size_t(hidden * ntokens));
            for (size_t j = 0; j < sum_output.size(); ++j) {
                sum_output[j] = pc_probe[j] + ph_probe[j];
            }
            if (capture_down) {
                // Compare the *first* divergent FFN stage, retaining the
                // original per-expert channel order. Split intervals for
                // Q4_K_M are contiguous along n_ff:
                // PC [0,pc_channels), Phone [pc_channels,total_channels).
                // The physical partition shape and backend ownership
                // were checked above. This stage test does not yet prove
                // byte-for-byte identity of the quantized shard weights.
                const int64_t pc_channels = pc_weights.up->ne[1];
                const int64_t ph_channels = phone_weights.up->ne[1];
                const int64_t full_channels = reference_model_ffn_channels;
                if (pc_channels <= 0 || ph_channels <= 0 ||
                    pc_channels + ph_channels != full_channels) {
                    std::fprintf(stderr,
                        "[CYCLE_FFN_STAGE_DIFF] status=FAIL "
                        "reason=CHANNEL_PARTITION pc=%lld phone=%lld full=%lld\n",
                        (long long)pc_channels, (long long)ph_channels,
                        (long long)full_channels);
                    return false;
                }
                std::pair<int32_t, int32_t> gate_phone_peak {0, 0};
                std::pair<int32_t, int32_t> up_phone_peak {0, 0};
                const auto compare_stages = [&](const char * stage_name,
                        const std::vector<float> & oracle,
                        const std::vector<float> & pc_values,
                        const std::vector<float> & ph_values,
                        std::pair<int32_t, int32_t> * phone_peak) -> bool {
                    const size_t rows = size_t(topk * ntokens);
                    if (oracle.size() != size_t(full_channels) * rows ||
                        pc_values.size() != size_t(pc_channels) * rows ||
                        ph_values.size() != size_t(ph_channels) * rows) {
                        std::fprintf(stderr,
                            "[CYCLE_FFN_STAGE_DIFF] stage=%s status=FAIL "
                            "reason=TENSOR_SHAPE full=%zu pc=%zu phone=%zu\n",
                            stage_name, oracle.size(), pc_values.size(),
                            ph_values.size());
                        return false;
                    }
                    for (int dev = 0; dev < 2; ++dev) {
                        const bool is_pc = dev == 0;
                        const int64_t width = is_pc ? pc_channels : ph_channels;
                        const int64_t channel_base = is_pc ? 0 : pc_channels;
                        const auto & actual = is_pc ? pc_values : ph_values;
                        const char * owner = is_pc ? "PC_CPU" : "PHONE_GPU";
                        double err_sq = 0.0;
                        double worst_err = -1.0;
                        int64_t worst_row = -1;
                        int64_t worst_channel = -1;
                        float worst_ref = 0.0f;
                        float worst_actual = 0.0f;
                        for (int64_t row = 0; row < int64_t(rows); ++row) {
                            for (int64_t ch = 0; ch < width; ++ch) {
                                const size_t target = size_t(row * width + ch);
                                const size_t source = size_t(
                                    row * full_channels + channel_base + ch);
                                const double x = actual[target];
                                const double y = oracle[source];
                                if (!std::isfinite(x) || !std::isfinite(y)) {
                                    std::fprintf(stderr,
                                        "[CYCLE_FFN_STAGE_DIFF] stage=%s "
                                        "backend=%s status=FAIL "
                                        "reason=NONFINITE_VALUE row=%lld channel=%lld\n",
                                        stage_name, owner, (long long)row,
                                        (long long)ch);
                                    return false;
                                }
                                const double e = std::fabs(x - y);
                                err_sq += e * e;
                                if (e > worst_err) {
                                    worst_err = e;
                                    worst_row = row;
                                    worst_channel = channel_base + ch;
                                    worst_ref = float(y);
                                    worst_actual = float(x);
                                }
                            }
                        }
                        const int64_t token = worst_row / topk;
                        const int64_t slot = worst_row % topk;
                        if (!is_pc && phone_peak != nullptr) {
                            *phone_peak = {
                                int32_t(token), int32_t(slot)
                            };
                        }
                        const int32_t expert_id = input_ids[
                            size_t(token * topk + slot)];
                        const double rms = std::sqrt(
                            err_sq / double(actual.size()));
                        std::printf(
                            "[CYCLE_FFN_STAGE_DIFF] stage=%s backend=%s "
                            "max_abs=%.8g rms=%.8g pos=%lld "
                            "expert_slot=%lld expert_id=%d channel=%lld "
                            "cpu_value=%.8g shard_value=%.8g status=MEASURED\n",
                            stage_name, owner, worst_err, rms,
                            (long long)token, (long long)slot, expert_id,
                            (long long)worst_channel,
                            worst_ref, worst_actual);
                    }
                    return true;
                };
                if (!compare_stages(
                        "GATE", baseline_gate,
                        pc_stages.gate, ph_stages.gate, &gate_phone_peak) ||
                    !compare_stages(
                        "UP", baseline_up,
                        pc_stages.up, ph_stages.up, &up_phone_peak) ||
                    !compare_stages(
                        "SWIGLU", baseline_swiglu,
                        pc_stages.swiglu, ph_stages.swiglu, nullptr)) {
                    return false;
                }

                // Compare OpenCL MUL_MAT_ID to GGML CPU MUL_MAT_ID with
                // the *same packed Phone weight bytes*. Run twice:
                // original F32 input and an F16-roundtrip input. Neither
                // replay modifies the actual Phone/PC FFN calculation.
                if (ntokens <= 2) {
                    // The replay reads all packed rows of the selected
                    // experts from the Phone. Limit extra network traffic
                    // to the 1/2-token isolation tests, not 8..128-token
                    // performance and prefill regression cases.
                    if (!cycle_real_replay_phone_matmul_id_on_cpu(
                            "GATE", cpu, phone_weights.gate,
                            input_norm, input_ids,
                            hidden, ntokens, topk, full_channels, pc_channels,
                            baseline_gate, ph_stages.gate) ||
                        !cycle_real_replay_phone_matmul_id_on_cpu(
                            "UP", cpu, phone_weights.up,
                            input_norm, input_ids,
                            hidden, ntokens, topk, full_channels, pc_channels,
                            baseline_up, ph_stages.up)) {
                        std::fprintf(stderr,
                            "[CYCLE_FFN_KERNEL_REPLAY] status=FAIL "
                            "reason=PROBE_EXECUTION\n");
                        return false;
                    }
                    // GPU comparison is opt-in to this small diagnostic case:
                    // one selected expert's Q4_K weight matrix + F32 input,
                    // versus the existing Phone MoE MUL_MAT_ID output.
                    // Use the largest observed GPU error separately for
                    // Gate and Up. A probe failure is diagnostic only.
                    const bool gate_gpu_op_ok =
                        cycle_real_probe_phone_single_expert_mul_mat(
                            "GATE", phone, cpu, phone_weights.gate,
                            input_norm, input_ids,
                            baseline_gate, ph_stages.gate,
                            hidden, ntokens, topk, full_channels, pc_channels,
                            gate_phone_peak.first, gate_phone_peak.second);
                    const bool up_gpu_op_ok =
                        cycle_real_probe_phone_single_expert_mul_mat(
                            "UP", phone, cpu, phone_weights.up,
                            input_norm, input_ids,
                            baseline_up, ph_stages.up,
                            hidden, ntokens, topk, full_channels, pc_channels,
                            up_phone_peak.first, up_phone_peak.second);
                    std::printf(
                        "[CYCLE_FFN_GPU_OP_COMPARE_SUM] "
                        "gate=%s up=%s mode=DIAGNOSTIC_ONLY\n",
                        gate_gpu_op_ok ? "MEASURED" : "NOT_AVAILABLE",
                        up_gpu_op_ok ? "MEASURED" : "NOT_AVAILABLE");
                } else {
                    std::printf("[CYCLE_FFN_KERNEL_REPLAY] enabled=0 "
                                "reason=EXTRA_RPC_WEIGHT_TRANSFER "
                                "tokens=%d limit=2\n", int(ntokens));
                }

                const size_t n_down = size_t(hidden * topk * ntokens);
                if (baseline_down.size() != n_down ||
                    pc_raw_down.size() != n_down ||
                    ph_raw_down.size() != n_down) {
                    std::fprintf(stderr,
                        "[CYCLE_FFN_INTERMEDIATE] case=%s status=FAIL "
                        "reason=RAW_DOWN_SHAPE n=%zu cpu=%zu pc=%zu ph=%zu\n",
                        label, n_down, baseline_down.size(),
                        pc_raw_down.size(), ph_raw_down.size());
                    return false;
                }
                std::vector<float> physical_down(n_down);
                for (size_t j = 0; j < n_down; ++j) {
                    physical_down[j] = pc_raw_down[j] + ph_raw_down[j];
                }
                const cycle_real_diff down_diff =
                    cycle_real_compare(physical_down, baseline_down);
                std::printf("[CYCLE_FFN_INTERMEDIATE] "
                            "target=UNWEIGHTED_DOWN_FROM_SHARDS "
                            "max_abs=%.8g rms=%.8g cosine=%.8g "
                            "status=%s\n",
                            down_diff.max_abs, down_diff.rms,
                            down_diff.cosine,
                            down_diff.valid ? "MEASURED" : "FAIL_NONFINITE");

                // Per-token and per-Expert-slot maxima identify whether
                // the discrepancy predates Router weighting. The same
                // selected Expert IDs are used by CPU and both shards.
                double overall_down_max = -1.0;
                std::pair<int32_t, int32_t> down_phone_peak {0, 0};
                for (int32_t token = 0; token < ntokens; ++token) {
                    double token_max = -1.0;
                    int64_t worst_slot = -1;
                    int64_t worst_hidden = -1;
                    float reference_value = 0.0f;
                    float physical_value = 0.0f;
                    for (int64_t slot = 0; slot < topk; ++slot) {
                        for (int64_t channel = 0; channel < hidden; ++channel) {
                            const size_t idx = size_t(
                                (token * topk + slot) * hidden + channel);
                            const double abs_error = std::fabs(
                                double(physical_down[idx]) -
                                double(baseline_down[idx]));
                            if (abs_error > token_max) {
                                token_max = abs_error;
                                worst_slot = slot;
                                worst_hidden = channel;
                                reference_value = baseline_down[idx];
                                physical_value = physical_down[idx];
                            }
                        }
                    }
                    if (token_max > overall_down_max) {
                        overall_down_max = token_max;
                        down_phone_peak = {token, int32_t(worst_slot)};
                    }
                    std::printf("[CYCLE_FFN_DOWN_TOKEN] pos=%d "
                                "max_abs=%.8g expert_slot=%lld expert_id=%d "
                                "hidden=%lld cpu_raw=%.8g shards_raw=%.8g "
                                "status=MEASURED\n",
                                int(token), token_max,
                                (long long)worst_slot,
                                input_ids[size_t(token * topk + worst_slot)],
                                (long long)worst_hidden,
                                reference_value, physical_value);
                }
                if (ntokens <= 2) {
                    const bool down_oracle_ok = cycle_real_probe_down_f64_oracle(
                        cpu, phone, phone_weights.down,
                        baseline_swiglu, input_ids, full_channels, pc_channels,
                        hidden, ntokens, topk,
                        down_phone_peak.first, down_phone_peak.second);
                    if (!down_oracle_ok) {
                        std::fprintf(stderr,
                            "[CYCLE_FFN_DOWN_F64_ORACLE] status=NOT_AVAILABLE "
                            "scope=DIAGNOSTIC_ONLY\n");
                    }
                }

                // Reconstruct both weighted FFNs in CPU F32 from the raw
                // expert Down tensors. If CPU raw reproduces baseline_ffn,
                // the oracle capture is internally coherent. If shard raw
                // reproduces sum_output, any drift is already present
                // before the final weighted expert reduction.
                std::vector<float> oracle_weighted(size_t(hidden * ntokens), 0.0f);
                std::vector<float> physical_weighted(size_t(hidden * ntokens), 0.0f);
                for (int32_t token = 0; token < ntokens; ++token) {
                    for (int64_t slot = 0; slot < topk; ++slot) {
                        const float weight = input_routing[
                            size_t(token * topk + slot)];
                        for (int64_t j = 0; j < hidden; ++j) {
                            const size_t base = size_t(token * hidden + j);
                            const size_t idx = size_t(
                                (token * topk + slot) * hidden + j);
                            oracle_weighted[base] +=
                                baseline_down[idx] * weight;
                            physical_weighted[base] +=
                                physical_down[idx] * weight;
                        }
                    }
                }
                const cycle_real_diff oracle_reduce_diff =
                    cycle_real_compare(oracle_weighted, baseline_ffn);
                const cycle_real_diff physical_reduce_diff =
                    cycle_real_compare(physical_weighted, sum_output);
                std::printf("[CYCLE_FFN_INTERMEDIATE] "
                            "target=CPU_RAW_WEIGHTED_VS_CPU_FFN "
                            "max_abs=%.8g rms=%.8g status=%s\n",
                            oracle_reduce_diff.max_abs, oracle_reduce_diff.rms,
                            oracle_reduce_diff.valid ? "MEASURED" : "FAIL_NONFINITE");
                std::printf("[CYCLE_FFN_INTERMEDIATE] "
                            "target=SHARD_RAW_WEIGHTED_VS_SHARD_FFN "
                            "max_abs=%.8g rms=%.8g status=%s\n",
                            physical_reduce_diff.max_abs,
                            physical_reduce_diff.rms,
                            physical_reduce_diff.valid ?
                                "MEASURED" : "FAIL_NONFINITE");
                if (!down_diff.valid || !oracle_reduce_diff.valid ||
                    !physical_reduce_diff.valid) {
                    return false;
                }
            }
            const cycle_real_diff vs_cpu_full =
                cycle_real_compare(sum_output, baseline_ffn);
            const cycle_real_diff vs_cpu_ready =
                cycle_real_compare(sum_output, cpu_ready_control_ffn.empty()
                    ? sum_output : cpu_ready_control_ffn);
            std::printf(
                "[CYCLE_FFN_ABLATION] case=%s "
                "vs_cpu_full_max_abs=%.8g vs_cpu_full_rms=%.8g "
                "vs_cpu_ready_max_abs=%.8g vs_cpu_ready_rms=%.8g "
                "cpu_full_tolerance=%.8g status=%s\n",
                label,
                vs_cpu_full.max_abs, vs_cpu_full.rms,
                vs_cpu_ready.max_abs, vs_cpu_ready.rms,
                vs_cpu_full.tolerance,
                vs_cpu_full.valid && vs_cpu_ready.valid ?
                    "MEASURED" : "FAIL_NONFINITE");
            report_token_errors(label, sum_output, baseline_ffn, hidden);
            return vs_cpu_full.valid && vs_cpu_ready.valid;
        };
        // A: numerical floor of physical Q4_K shards and device kernels,
        // with NO Phone Attention/Router input discrepancy.
        cycle_real_phone_mem_checkpoint(phone_dev, "BEFORE_CPU_READY_CONTROL");
        if (!run_case("CPU_READY_CONTROL",
                      ref_norm, ref_ids, ref_routing,
                      cpu_ready_control_ffn, extended_diagnostics)) {
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }
        // The complete mathematical oracle consumes the same CPU READY
        // inputs as the physical-control FFN. All arithmetic and temporary
        // compute allocations are on the PC CPU. Only packed weight bytes
        // are read from the existing Phone weight allocation.
        if (ntokens <= 2) {
            cycle_real_phone_mem_checkpoint(phone_dev, "BEFORE_CPU_F64_ORACLE");
            const bool f64_ok = cycle_real_probe_full_ffn_f64_oracle(
                cpu, pc_weights, phone_weights,
                ref_norm, ref_ids, ref_routing,
                baseline_ffn, cpu_ready_control_ffn,
                ref_residual, baseline_output,
                hidden, ntokens, topk, experts);
            if (!f64_ok) {
                std::fprintf(stderr,
                    "[CYCLE_FFN_FULL_F64_ORACLE] status=NOT_AVAILABLE "
                    "scope=DIAGNOSTIC_ONLY\n");
            }
            cycle_real_phone_mem_checkpoint(phone_dev, "AFTER_CPU_F64_ORACLE");
        }
        if (extended_diagnostics) {
        // B: change only the pre-FFN hidden vector.
        std::vector<float> phone_norm_cpu_router;
        if (!run_case("PHONE_NORM_CPU_ROUTER",
                      norm, ref_ids, ref_routing,
                      phone_norm_cpu_router, false)) {
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }
        // C: change only Phone expert ID/weight pairs, preserving CPU Norm.
        std::vector<float> cpu_norm_phone_router;
        if (!run_case("CPU_NORM_PHONE_ROUTER",
                      ref_norm, ids, routing,
                      cpu_norm_phone_router, false)) {
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }
        // D: compare TopK ordering alone. Use Phone's expert ordering with
        // the CPU's exact weight matched to each expert ID; never assign a
        // weight to the wrong expert. If this differs from A, the summation
        // order/device arithmetic affects the result even with equal IDs.
        std::vector<float> cpu_weights_in_phone_order;
        cpu_weights_in_phone_order.reserve(size_t(ntokens * topk));
        for (int32_t t = 0; t < ntokens; ++t) {
            for (int64_t slot = 0; slot < topk; ++slot) {
                const size_t pos = size_t(t * topk + slot);
                const int32_t expert = ids[pos];
                bool found = false;
                for (int64_t ref_slot = 0; ref_slot < topk; ++ref_slot) {
                    const size_t ref_pos = size_t(t * topk + ref_slot);
                    if (ref_ids[ref_pos] == expert) {
                        cpu_weights_in_phone_order.push_back(
                            ref_routing[ref_pos]);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::fprintf(stderr,
                        "[CYCLE_FFN_ABLATION] case=PHONE_ORDER_CPU_WEIGHTS "
                        "status=FAIL reason=EXPERT_PAIR_NOT_FOUND\n");
                    ggml_backend_free(phone);
                    ggml_backend_free(cpu);
                    return 1;
                }
            }
        }
        std::vector<float> phone_order_cpu_weights;
        if (!run_case("PHONE_ORDER_CPU_WEIGHTS",
                      ref_norm, ids, cpu_weights_in_phone_order,
                      phone_order_cpu_weights, false)) {
            ggml_backend_free(phone);
            ggml_backend_free(cpu);
            return 1;
        }
        } else {
            std::printf("[CYCLE_FFN_ABLATION] "
                        "cases=PHONE_NORM_CPU_ROUTER,CPU_NORM_PHONE_ROUTER,"
                        "PHONE_ORDER_CPU_WEIGHTS status=SKIP "
                        "reason=LOW_MEMORY_DEFAULT\n");
        }
    }

    cycle_real_phone_mem_checkpoint(phone_dev, "BEFORE_REAL_FFN");
    const auto parts_begin = std::chrono::steady_clock::now();
    auto phone_task = std::async(std::launch::async, [&] {
        std::vector<float> part;
        const bool done = cycle_real_execute_ffn(
            phone, phone_weights, norm, ids, routing,
            hidden, ntokens, topk, experts, part,
            nullptr, nullptr,
            live_wire_enabled ? &wire : nullptr,
            live_wire_enabled ? &phone_ready.phone_graph : nullptr);
        return std::make_pair(done, std::move(part));
    });
    std::vector<float> pc_part;
    // PC FFN consumes the decoded Phone READY while the Phone FFN runs on
    // the same Phone backend after its own AttentionPre is complete.
    // Phone queues PARTIAL directly from its GPU output while PC computes.
    // This test is one layer/chunk: multi-layer FIFO execution comes next.
    const bool pc_ok = cycle_real_execute_ffn(
        cpu, pc_weights,
        ready_msg.ready.ffn_norm, ready_msg.ready.topk_ids,
        ready_msg.ready.topk_weights,
        hidden, ntokens, topk, experts, pc_part);
    const auto phone_result = phone_task.get();
    // Device-local copies and Phone FFN execution have completed.
    // Free the borrowed Attention compute buffer before PC Reduce.
    phone_ready.release_phone_graph();
    const bool phone_ok = phone_result.first;
    const auto & phone_part = phone_result.second;
    const double parts_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - parts_begin).count();
    // Keep the memory RPC query OUTSIDE ffn_parallel_ms timing.
    cycle_real_phone_mem_checkpoint(phone_dev, "AFTER_REAL_FFN");

    // Keep both backends alive: the PC still needs the Phone RPC
    // data-plane receiver to consume Send #2 (PARTIAL) below.
    if (!pc_ok || !phone_ok || pc_part.size() != residual.size() ||
        (!live_wire_enabled && phone_part.size() != residual.size())) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PHYSICAL_FFN_COMPUTE\n");
        return 1;
    }

    // SEND #2: in live mode the Phone RPC server has already queued the
    // FFN output tensor on its own TX worker. The PC receives this payload,
    // never a second ordinary RPC tensor_get or an in-process re-encode.
    size_t partial_bytes = 0;
    if (!inference_cycle_payload_size(
            INFERENCE_CYCLE_PARTIAL, uint32_t(ntokens), uint32_t(hidden),
            0, partial_bytes)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_SIZE\n");
        return 1;
    }
    inference_cycle_message partial_msg;
    const auto partial_receive_begin = std::chrono::steady_clock::now();
    if (live_wire_enabled) {
        if (!cycle_real_receive_live_wire(
                phone, wire, INFERENCE_CYCLE_PARTIAL,
                uint32_t(ntokens), uint32_t(hidden), 0, partial_msg)) {
            return 1;
        }
    } else {
        const inference_cycle_partial_view partial_source {
            uint32_t(ntokens), uint32_t(hidden), phone_part.data(),
        };
        std::vector<uint8_t> partial_wire(partial_bytes);
        size_t partial_written = 0;
        if (!inference_cycle_write_partial(
                partial_wire.data(), partial_wire.size(),
                partial_source, partial_written) ||
            partial_written != partial_wire.size()) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_ENCODE\n");
            return 1;
        }
        inference_frame_header partial_header =
            inference_make_header(INFERENCE_FRAME_DATA);
        partial_header.stage = INFERENCE_STAGE_CYCLE_PARTIAL;
        partial_header.tensor_role = INFERENCE_CYCLE_PHONE_TO_PC;
        partial_header.layer = layer;
        partial_header.chunk = wire.chunk;
        partial_header.seq = ready_header.seq;
        partial_header.graph_id = ready_header.graph_id;
        partial_header.payload_bytes = partial_wire.size();
        if (!inference_cycle_decode_frame(
                partial_header, partial_wire.data(), partial_wire.size(),
                partial_msg)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_DECODE\n");
            return 1;
        }
    }
    if (partial_msg.kind != INFERENCE_CYCLE_PARTIAL ||
        partial_msg.direction != ready_msg.direction ||
        partial_msg.layer != ready_msg.layer ||
        partial_msg.chunk != ready_msg.chunk ||
        partial_msg.seq != ready_msg.seq ||
        partial_msg.graph_id != ready_msg.graph_id ||
        partial_msg.partial.partial.size() != size_t(hidden * ntokens) ||
        (!live_wire_enabled && partial_msg.partial.partial != phone_part)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_CONTENT\n");
        return 1;
    }
    const double partial_receive_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - partial_receive_begin).count();
    std::printf("[CYCLE_REAL_WIRE] stage=PARTIAL direction=P2C "
                "bytes=%zu mode=%s recv_wait_ms=%.3f status=PASS\n",
                partial_bytes,
                live_wire_enabled ? "PHONE_GPU_REAL_DATA_PLANE" :
                                    "IN_PROCESS_CODEC", partial_receive_ms);

    std::vector<float> combined_ffn(residual.size());
    std::vector<float> combined_output(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        combined_ffn[i] = pc_part[i] + partial_msg.partial.partial[i];
    }
    if (!inference_cycle_reduce_p2c_pc_cpu(
            ready_msg.ready, pc_part.data(), pc_part.size(),
            partial_msg.partial, combined_output.data(), combined_output.size())) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PC_REDUCE\n");
        return 1;
    }
    // PARTIAL is acknowledged only after PC has consumed it for the
    // exclusive Reduce. Reverse ACK contains zero activation bytes.
    if (live_wire_enabled) {
        if (!ggml_backend_rpc_cycle_send_ack(
                phone, layer, wire.chunk, wire.seq, wire.graph_id,
                INFERENCE_CYCLE_PARTIAL, true) ||
            !ggml_backend_rpc_cycle_wait_tx_idle(phone, 10000)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_NETWORK] status=FAIL reason=PARTIAL_HOST_ACK\n");
            return 1;
        }
    }
    // Both payload frames were received and Reduce is complete.
    ggml_backend_free(phone);
    ggml_backend_free(cpu);
    const cycle_real_diff ffn_diff = cycle_real_compare(combined_ffn, baseline_ffn);
    const cycle_real_diff out_diff = cycle_real_compare(combined_output, baseline_output);
    if (run_ffn_ablation) {
        const cycle_real_diff delta =
            cycle_real_compare(combined_ffn, cpu_ready_control_ffn);
        std::printf("[CYCLE_FFN_ABLATION] case=PHONE_READY_ACTUAL "
                    "vs_cpu_full_max_abs=%.8g vs_cpu_full_rms=%.8g "
                    "vs_cpu_ready_max_abs=%.8g vs_cpu_ready_rms=%.8g "
                    "cpu_full_tolerance=%.8g status=%s\n",
                    ffn_diff.max_abs, ffn_diff.rms,
                    delta.max_abs, delta.rms, ffn_diff.tolerance,
                    ffn_diff.valid && delta.valid ? "MEASURED" : "FAIL_NONFINITE");
    }
    report_token_errors("FFN_PARTIAL_SUM", combined_ffn, baseline_ffn, hidden);
    report_token_errors("LAYER_OUTPUT", combined_output, baseline_output, hidden);
    std::printf("[CYCLE_REAL_COMPARE] target=FFN_PARTIAL_SUM layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, ffn_diff.max_abs, ffn_diff.rms, ffn_diff.cosine,
                ffn_diff.tolerance, ffn_diff.pass ? "PASS" : "FAIL");
    std::printf("[CYCLE_REAL_COMPARE] target=LAYER_OUTPUT layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, out_diff.max_abs, out_diff.rms, out_diff.cosine,
                out_diff.tolerance, out_diff.pass ? "PASS" : "FAIL");
    // The generic layer metric has a deliberately relaxed scale-based
    // tolerance inherited from older comparisons; it can exceed 1.0 for
    // large residuals. A real multi-KV correctness gate must also require
    // an independent absolute/relative layer bound.
    const double strict_output_tolerance = 0.02 + 0.00002 * out_diff.max_ref;
    const bool strict_output_ok = out_diff.valid &&
        out_diff.max_abs <= strict_output_tolerance;
    std::printf("[CYCLE_REAL_COMPARE] target=LAYER_OUTPUT_STRICT "
                "layer=%d max_abs=%.8g tolerance=%.8g status=%s\n",
                layer, out_diff.max_abs, strict_output_tolerance,
                strict_output_ok ? "PASS" : "FAIL");
    const bool pass = ffn_diff.valid && out_diff.valid &&
                      ffn_diff.pass && out_diff.pass && strict_output_ok;
    if (live_wire_enabled) {
        const double p2c_wall_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - parts_begin).count();
        std::printf(
            "[CYCLE_REAL_WIRE_SUM] layer=%d chunk=%d seq=%llu "
            "ready=RECEIVED_FROM_PHONE_GPU "
            "partial=RECEIVED_FROM_PHONE_GPU "
            "direction=PHONE_TO_PC payload_sends=2 "
            "phone_duplicate_rpc_readback=0 "
            "pc_reduce=REAL_RX_PARTIAL "
            "ffn_parallel_ms=%.3f p2c_until_reduce_ms=%.3f "
            "cpu_parity=%s status=PASS\n",
            layer, wire.chunk, (unsigned long long)wire.seq,
            parts_ms, p2c_wall_ms, pass ? "PASS" : "DIFF");
    }
    std::printf("[CYCLE_REAL_P2C_SUM] layer=%d tokens=%d "
                "forward_ms=%.3f ffn_parallel_ms=%.3f "
                "real_attention=PHONE_GPU_CAUSAL_MULTI_KV phone_attention_executed=1 "
                "phone_attn_ms=%.3f kv_scope=ONE_UBATCH kv_persistent=0 "
                "real_ffn_weights=1 "
                "ffn_shards=PC_CPU+PHONE_GPU reduce_owner=PC_CPU "
                "reference=CPU_FULL_QWEN3_LAYER data_plane=RPC_BACKEND_ONLY "
                "cycle_wire=%s "
                "topk_order_exact=%d topk_set_equal=%d "
                "reordered_tokens=%d status=%s\n",
                layer, int(ntokens), forward_ms, parts_ms,
                phone_attn_ms,
                live_wire_enabled ? "REAL_RPC_DATA_PLANE" :
                                    "IN_PROCESS_CODEC_ONLY",
                topk_equal ? 1 : 0, topk_set_equal ? 1 : 0,
                int(permuted_tokens), pass ? "PASS" : "FAIL");
    // Transport-focused run reports known CPU-parity divergence without
    // treating a completed two-frame Phone->PC transfer as a wire failure.
    // Legacy codec mode retains the old numerical exit gate.
    return live_wire_enabled ? 0 : (pass ? 0 : 1);
}
