// Real, isolated PC Qwen3 prefix profiling with original GGUF weights.
// Deliberately no llama_context, Region Ring, RPC, embeddings or LM Head.
// Only ONE native backend is constructed (CPU or GPU). Unsupported ops
// fail closed; there is no CPU fallback from the CUDA path.
//
// The same Qwen3 layer graph builder is used by the Phone NPU path.
// Each measurement graph contains the model's actual layers [0,N).
#include "ggml.h"
// Access to ggml_cgraph::uid is intentionally GGML-internal: Hexagon's
// fixed-Graph precompiler uses the reserved high-bit UID namespace.
// Keep this dependency private to the in-tree V2 executable.
#include "ggml-impl.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "phone-qwen3-local-builder.h"
#include "llama-region-ring-v2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using clock_t = std::chrono::steady_clock;

template <typename T, void (*Free)(T *)>
using ptr = std::unique_ptr<T, decltype(Free)>;

using ggml_ptr = ptr<ggml_context, ggml_free>;
using gguf_ptr = ptr<gguf_context, gguf_free>;
using backend_ptr = ptr<ggml_backend, ggml_backend_free>;
using buffer_ptr = ptr<ggml_backend_buffer, ggml_backend_buffer_free>;

struct parameters {
    std::string model;
    char backend = '\0';
    int layers = 0;
    int chunk = 0;
    int kv = 0;
    int warmups = 2;
    int repeats = 8;
    int threads = 8;
    int bank_repeats = 0;
    int host_stage_cap_mib = 1024;
    bool verify_cpu = false;
    bool diagnose_layers = false;
    double max_relative_rmse = 0.02;
    double max_abs_error = 0.5;
};

int parse_int(const char * value, const char * label) {
    char * end = nullptr;
    const long n = std::strtol(value, &end, 10);
    if (value == end || *end != '\0' || n < 0 ||
            n > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string("INVALID_") + label);
    }
    return static_cast<int>(n);
}

parameters parse(int argc, char ** argv) {
    parameters p;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "v2-stage-profile --model FILE --backend G|C|N|A "
                         "--layers N --chunk X --kv K [--warmups N] "
                         "[--repeats N] [--threads N] [--verify-cpu] "
                         "[--relative-rmse F] [--max-abs-error F] "
                         "[--diagnose-layers] [--bank-repeats N] "
                         "[--host-stage-cap-mib N]\n";
            std::exit(0);
        }
        if (key == "--verify-cpu") {
            p.verify_cpu = true;
            continue;
        }
        if (key == "--diagnose-layers") {
            p.diagnose_layers = true;
            p.verify_cpu = true;
            continue;
        }
        if (i + 1 >= argc) { throw std::runtime_error("MISSING_ARGUMENT:" + key); }
        const char * value = argv[++i];
        if (key == "--model") { p.model = value; }
        else if (key == "--backend") {
            if (std::strlen(value) != 1) {
                throw std::runtime_error("BACKEND_MUST_BE_G_C_N_OR_A");
            }
            p.backend = value[0];
        } else if (key == "--layers") { p.layers = parse_int(value, "LAYERS"); }
        else if (key == "--chunk") { p.chunk = parse_int(value, "CHUNK"); }
        else if (key == "--kv") { p.kv = parse_int(value, "KV"); }
        else if (key == "--warmups") { p.warmups = parse_int(value, "WARMUPS"); }
        else if (key == "--repeats") { p.repeats = parse_int(value, "REPEATS"); }
        else if (key == "--threads") { p.threads = parse_int(value, "THREADS"); }
        else if (key == "--bank-repeats") {
            p.bank_repeats = parse_int(value, "BANK_REPEATS");
        } else if (key == "--host-stage-cap-mib") {
            p.host_stage_cap_mib = parse_int(value, "HOST_STAGE_CAP_MIB");
        }
        else if (key == "--relative-rmse") {
            p.max_relative_rmse = std::stod(value);
        } else if (key == "--max-abs-error") {
            p.max_abs_error = std::stod(value);
        } else { throw std::runtime_error("UNKNOWN_ARGUMENT:" + key); }
    }
    if (p.model.empty() ||
            (p.backend != 'G' && p.backend != 'C' &&
             p.backend != 'N' && p.backend != 'A') ||
            p.layers < 1 || p.chunk < 1 || p.kv < p.chunk ||
            p.repeats < 1 || p.threads < 1 ||
            p.bank_repeats > 20 || p.host_stage_cap_mib < 1 ||
            (p.backend != 'N' && p.bank_repeats > 0) ||
            !std::isfinite(p.max_relative_rmse) ||
            !std::isfinite(p.max_abs_error) ||
            p.max_relative_rmse < 0 || p.max_abs_error < 0 ||
            (p.backend == 'N' && !p.verify_cpu)) {
        throw std::runtime_error(
            "INVALID_STAGE_PROFILE_SHAPE_OR_NPU_CPU_PARITY_REQUIRED");
    }
    return p;
}

double elapsed_ms(clock_t::time_point begin) {
    return std::chrono::duration<double, std::milli>(
        clock_t::now() - begin).count();
}

double median(std::vector<double> xs) {
    if (xs.empty()) { throw std::runtime_error("NO_COMPUTE_SAMPLES"); }
    std::sort(xs.begin(), xs.end());
    const size_t k = xs.size() / 2;
    return xs.size() & 1 ? xs[k] : (xs[k-1] + xs[k]) * 0.5;
}

uint32_t field_u32(const gguf_context * g, const char * name) {
    const auto id = gguf_find_key(g, name);
    if (id < 0) { throw std::runtime_error(std::string("MISSING_GGUF_KEY:") + name); }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("INVALID_GGUF_UINT32:") + name);
    }
    return gguf_get_val_u32(g, id);
}

float field_f32(const gguf_context * g, const char * name) {
    const auto id = gguf_find_key(g, name);
    if (id < 0) { throw std::runtime_error(std::string("MISSING_GGUF_KEY:") + name); }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) {
        throw std::runtime_error(std::string("INVALID_GGUF_FLOAT:") + name);
    }
    return gguf_get_val_f32(g, id);
}

uint32_t optional_u32(const gguf_context * g, const char * name, uint32_t fallback) {
    const auto id = gguf_find_key(g, name);
    if (id < 0) { return fallback; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("INVALID_GGUF_UINT32:") + name);
    }
    return gguf_get_val_u32(g, id);
}

float optional_f32(const gguf_context * g, const char * name, float fallback) {
    const auto id = gguf_find_key(g, name);
    return id < 0 ? fallback : field_f32(g, name);
}

struct to_load {
    ggml_tensor * tensor = nullptr;
    size_t offset = 0;
    size_t bytes = 0;
};

ggml_phone_qwen3::layer_weights load_layer(
        ggml_context * plain_ctx, ggml_context * quant_ctx,
        ggml_context * model_ctx,
        const gguf_context * gguf, uint32_t layer,
        std::vector<to_load> & todo) {
    ggml_phone_qwen3::layer_weights w {};
    const auto create = [&](const char * suffix) -> ggml_tensor * {
        const std::string name = "blk." + std::to_string(layer) + "." + suffix;
        const int64_t id = gguf_find_tensor(gguf, name.c_str());
        const auto * source = ggml_get_tensor(model_ctx, name.c_str());
        if (id < 0 || !source || source->type != gguf_get_tensor_type(gguf,id)) {
            throw std::runtime_error("MISSING_REAL_GGUF_WEIGHT:" + name);
        }
        auto * target = ggml_new_tensor(
            quant_ctx && ggml_is_quantized(source->type) ?
                quant_ctx : plain_ctx,
            source->type, GGML_MAX_DIMS, source->ne);
        if (!target || ggml_nbytes(target) != gguf_get_tensor_size(gguf,id)) {
            throw std::runtime_error("INVALID_GGUF_WEIGHT:" + name);
        }
        ggml_set_name(target, name.c_str());
        todo.push_back({target,
            gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf,id),
            ggml_nbytes(target)});
        return target;
    };
    w.attn_norm = create("attn_norm.weight");
    w.wq        = create("attn_q.weight");
    w.wk        = create("attn_k.weight");
    w.wv        = create("attn_v.weight");
    w.wo        = create("attn_output.weight");
    w.ffn_norm  = create("ffn_norm.weight");
    w.ffn_gate  = create("ffn_gate.weight");
    w.ffn_up    = create("ffn_up.weight");
    w.ffn_down  = create("ffn_down.weight");
    w.q_norm    = create("attn_q_norm.weight");
    w.k_norm    = create("attn_k_norm.weight");
    return w;
}

struct upload_stats {
    double file_read_ms = 0;
    double backend_set_ms = 0;
    size_t original_bytes = 0;
    std::vector<std::vector<uint8_t>> host_staged;
};

upload_stats upload_weights(
        const std::string & path, const std::vector<to_load> & todo,
        bool keep_staged) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("MODEL_GGUF_OPEN_FAILED"); }
    upload_stats result {};
    if (keep_staged) { result.host_staged.reserve(todo.size()); }
    // Without --bank-repeats only one Tensor's raw GGUF bytes are resident
    // in host memory. This preserves the memory behavior of normal Profile.
    std::vector<uint8_t> scratch;
    for (const auto & item : todo) {
        if (item.bytes > static_cast<size_t>(
                std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("GGUF_TENSOR_EXCEEDS_IO_LIMIT");
        }
        result.original_bytes += item.bytes;
        scratch.resize(item.bytes);
        auto begin = clock_t::now();
        in.clear();
        in.seekg(static_cast<std::streamoff>(item.offset));
        if (!in.read(reinterpret_cast<char *>(scratch.data()),
                     static_cast<std::streamsize>(item.bytes))) {
            throw std::runtime_error("GGUF_READ_FAILED:" +
                                     std::string(item.tensor->name));
        }
        result.file_read_ms += elapsed_ms(begin);
        begin = clock_t::now();
        ggml_backend_tensor_set(item.tensor, scratch.data(), 0, item.bytes);
        result.backend_set_ms += elapsed_ms(begin);
        if (keep_staged) {
            result.host_staged.emplace_back(std::move(scratch));
            scratch = {};
        }
    }
    return result;
}

ggml_phone_owned::region_plan model_plan(
        const gguf_context * g, const parameters & args) {
    const auto arch_key = gguf_find_key(g, "general.architecture");
    if (arch_key < 0 || gguf_get_kv_type(g, arch_key) != GGUF_TYPE_STRING ||
            std::strcmp(gguf_get_val_str(g, arch_key), "qwen3") != 0) {
        throw std::runtime_error("V2_PROFILE_QWEN3_DENSE_ONLY");
    }
    if (const int64_t rope_scaling = gguf_find_key(g, "qwen3.rope.scaling.type");
            rope_scaling >= 0 &&
            std::strcmp(gguf_get_val_str(g, rope_scaling), "none") != 0) {
        throw std::runtime_error("UNSUPPORTED_ROPE_SCALING_PROFILE");
    }
    ggml_phone_owned::region_plan p {};
    p.run_id = 1;
    p.p2c_run_id = 1;
    p.region = 0;
    p.layer = 0;
    p.total_tokens = static_cast<uint32_t>(args.kv);
    p.chunk_tokens = static_cast<uint32_t>(args.chunk);
    p.position_base = 0;
    p.kv_capacity = static_cast<uint32_t>(args.kv);
    p.n_kv = static_cast<uint32_t>(args.kv);
    p.n_embd = field_u32(g, "qwen3.embedding_length");
    p.n_heads = field_u32(g, "qwen3.attention.head_count");
    p.n_heads_kv = optional_u32(g, "qwen3.attention.head_count_kv", p.n_heads);
    if (p.n_heads == 0 || p.n_embd % p.n_heads != 0) {
        throw std::runtime_error("INVALID_QWEN3_HEAD_COUNT");
    }
    // Match llama_model_base::load_hparams(): Qwen3 GGUF is NOT required
    // to carry rope.dimension_count. Key head dim defaults to n_embd/n_head,
    // can be overridden by attention.key_length, and n_rot defaults to
    // that key dimension. An explicit n_rot must agree for Qwen3.
    const uint32_t default_head_dim = p.n_embd / p.n_heads;
    const uint32_t key_dim = optional_u32(
        g, "qwen3.attention.key_length", default_head_dim);
    const uint32_t value_dim = optional_u32(
        g, "qwen3.attention.value_length", default_head_dim);
    const uint32_t n_rot = optional_u32(
        g, "qwen3.rope.dimension_count", key_dim);
    if (n_rot != key_dim || key_dim != value_dim) {
        throw std::runtime_error("QWEN3_UNSUPPORTED_HEAD_OR_ROPE_DIMENSIONS");
    }
    p.head_dim = key_dim;
    p.rope_type = ggml_phone_owned::k_rope_neox;
    p.kv_type = ggml_phone_owned::k_f16;
    p.n_ctx_orig = field_u32(g, "qwen3.context_length");
    p.rms_norm_eps = field_f32(g, "qwen3.attention.layer_norm_rms_epsilon");
    // Also optional in the official loader (fallback = 10000).
    p.rope_freq_base = optional_f32(g, "qwen3.rope.freq_base", 10000.0f);
    p.rope_freq_scale = 1.0f;
    p.rope_ext_factor = -1.0f;
    p.rope_attn_factor = 1.0f;
    p.rope_beta_fast = 32.0f;
    p.rope_beta_slow = 1.0f;
    const char * why = "NONE";
    if (!ggml_phone_owned::validate(p, why)) {
        throw std::runtime_error(std::string("INVALID_QWEN3_STAGE_PLAN:") + why);
    }
    return p;
}

// Must outlive the fixed HTP graph, but be destroyed BEFORE the graph
// metadata and quantized Bank buffers. Otherwise a cached HTP opnode can
// hold a pointer into freed graph storage on abnormal exit.
struct htp_cache_guard {
    ggml_backend_t backend = nullptr;
    void (*clear)(ggml_backend_t) = nullptr;
    ~htp_cache_guard() {
        if (backend && clear) {
            ggml_backend_synchronize(backend);
            clear(backend);
        }
    }
};

struct sample_result {
    double graph_build_ms = 0;
    double weight_upload_ms = 0;
    double prepare_ms = 0;
    double compute_ms = 0;
    double htp_precompile_ms = 0;
    double activation_upload_median_ms = 0;
    double activation_readback_median_ms = 0;
    double weight_file_read_ms = 0;
    double weight_backend_set_ms = 0;
    double same_bank_refill_median_ms = 0;
    size_t raw_weight_bytes = 0;
    size_t activation_bytes = 0;
    size_t weight_bytes = 0;
    size_t kv_bytes = 0;
    size_t graph_workspace_bytes = 0;
    std::vector<float> output;
    // Filled only in diagnostic mode. Intermediate outputs are retained
    // in the measured Graph rather than reconstructed with prefix reruns.
    std::vector<std::vector<float>> layer_outputs;
};

sample_result run(const parameters & args) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    if (args.backend == 'N') {
        dev = ggml_backend_dev_by_name("HTP0");
    } else if (args.backend == 'C') {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    } else {
        const char * needle = args.backend == 'G' ? "CUDA" : "OpenCL";
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto d = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                    std::string(ggml_backend_dev_name(d)).find(needle) !=
                        std::string::npos) {
                dev = d;
                break;
            }
        }
    }
    if (!dev) { throw std::runtime_error("REQUESTED_BACKEND_NOT_AVAILABLE"); }
    backend_ptr backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
    if (!backend) { throw std::runtime_error("BACKEND_INIT_FAILED"); }
    if (args.backend == 'C') {
        ggml_backend_cpu_set_n_threads(backend.get(), args.threads);
    }
    const bool htp = args.backend == 'N';
    ggml_backend_buffer_type_t repack = nullptr;
    using precompile_fn = bool (*)(ggml_backend_t, ggml_cgraph *);
    precompile_fn precompile = nullptr;
    if (htp) {
        const auto reg = ggml_backend_dev_backend_reg(dev);
        const auto extra = reg == nullptr ? nullptr :
            reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts"));
        auto * bufts = extra == nullptr ? nullptr : extra(dev);
        if (bufts != nullptr) { repack = bufts[0]; }
        precompile = reg == nullptr ? nullptr :
            reinterpret_cast<precompile_fn>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_hexagon_precompile_fixed_graph"));
        if (!repack || !precompile) {
            throw std::runtime_error("HTP_REPACK_OR_PRECOMPILER_NOT_AVAILABLE");
        }
    }
    ggml_context * raw_model_ctx = nullptr;
    gguf_ptr gguf(gguf_init_from_file(
        args.model.c_str(), {true, &raw_model_ctx}), gguf_free);
    ggml_ptr model_ctx(raw_model_ctx, ggml_free);
    if (!gguf || !model_ctx) {
        throw std::runtime_error("GGUF_MODEL_METADATA_FAILED");
    }
    auto p = model_plan(gguf.get(), args);
    llama_ring_v2::profile_request request {};
    request.device = args.backend == 'G' ? llama_ring_v2::backend::G :
                     args.backend == 'C' ? llama_ring_v2::backend::C :
                     args.backend == 'N' ? llama_ring_v2::backend::N :
                                           llama_ring_v2::backend::A;
    request.model_layers = static_cast<int>(field_u32(gguf.get(),"qwen3.block_count"));
    request.layers = args.layers;
    request.chunk_tokens = static_cast<uint32_t>(args.chunk);
    request.kv_tokens = static_cast<uint32_t>(args.kv);
    request.repeats = static_cast<uint32_t>(args.repeats);
    request.warmups = static_cast<uint32_t>(args.warmups);
    const auto error = llama_ring_v2::validate(request);
    if (!error.empty()) { throw std::runtime_error(error); }

    const auto build_begin = clock_t::now();
    const size_t weight_meta = ggml_tensor_overhead() *
        (static_cast<size_t>(args.layers) * 32 + 128) + 65536;
    ggml_ptr weight_ctx(ggml_init({weight_meta, nullptr, true}), ggml_free);
    ggml_ptr quant_ctx(htp ?
        ggml_init({weight_meta, nullptr, true}) : nullptr, ggml_free);
    if (!weight_ctx || (htp && !quant_ctx)) {
        throw std::runtime_error("WEIGHT_CONTEXT_ALLOC_FAILED");
    }
    std::vector<to_load> todo;
    std::vector<ggml_phone_qwen3::layer_weights> weights;
    for (int layer = 0; layer < args.layers; ++layer) {
        weights.push_back(load_layer(weight_ctx.get(), quant_ctx.get(),
                                     model_ctx.get(), gguf.get(), layer, todo));
    }
    buffer_ptr weight_buffer(
        ggml_backend_alloc_ctx_tensors(weight_ctx.get(), backend.get()),
        ggml_backend_buffer_free);
    buffer_ptr repack_buffer(htp ?
        ggml_backend_alloc_ctx_tensors_from_buft(quant_ctx.get(), repack) :
        nullptr, ggml_backend_buffer_free);
    if (!weight_buffer || (htp && !repack_buffer)) {
        throw std::runtime_error("PREFIX_WEIGHTS_EXCEED_BACKEND_MEMORY");
    }
    ggml_backend_buffer_set_usage(
        weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (repack_buffer) {
        ggml_backend_buffer_set_usage(
            repack_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    size_t total_raw_bytes = 0;
    for (const auto & item : todo) {
        if (item.bytes > std::numeric_limits<size_t>::max() - total_raw_bytes) {
            throw std::runtime_error("STAGE_RAW_WEIGHTS_OVERFLOW");
        }
        total_raw_bytes += item.bytes;
    }
    if (args.bank_repeats > 0 &&
            total_raw_bytes > size_t(args.host_stage_cap_mib) * 1024 * 1024) {
        throw std::runtime_error("HTP_HOST_STAGE_CAP_EXCEEDED");
    }
    const auto upload_begin = clock_t::now();
    auto uploaded = upload_weights(args.model, todo, args.bank_repeats > 0);
    ggml_backend_synchronize(backend.get());
    const double weight_upload_ms = elapsed_ms(upload_begin);

    double same_bank_refill_ms = 0;
    if (args.bank_repeats > 0) {
        // Controlled HTP quantized-weight repack/copy into the SAME
        // allocated Bank using already-staged host bytes. This excludes
        // storage IO and is not a dual-Bank prefetch/overlap benchmark.
        std::vector<double> refill;
        for (int rep = -1; rep < args.bank_repeats; ++rep) {
            const auto started = clock_t::now();
            for (size_t i = 0; i < todo.size(); ++i) {
                ggml_backend_tensor_set(
                    todo[i].tensor, uploaded.host_staged[i].data(),
                    0, todo[i].bytes);
            }
            ggml_backend_synchronize(backend.get());
            if (rep >= 0) { refill.push_back(elapsed_ms(started)); }
        }
        same_bank_refill_ms = median(refill);
        std::cerr << "[V2_HTP_BANK] mode=HOST_STAGED_SAME_BANK_REPACK"
                  << " layers=" << args.layers
                  << " raw_bytes=" << total_raw_bytes
                  << " median_ms=" << same_bank_refill_ms
                  << " status=MEASURED_NOT_DUAL_BANK" << '\n';
    }
    // Release the optional host staging allocation before building the
    // actual full-layer Graph and its GPU/NPU intermediate workspace.
    uploaded.host_staged.clear();
    uploaded.host_staged.shrink_to_fit();

    // Prefix K/V is real populated tensor memory, not a nominal KV label.
    // The final X positions are overwritten by the measured Chunk, while
    // [0,KV-X) are nonzero deterministic historical cache entries.
    const int64_t kv_width = int64_t(p.n_heads_kv) * p.head_dim;
    ggml_ptr kv_ctx(ggml_init({
        ggml_tensor_overhead() * (static_cast<size_t>(args.layers)*4+32)+4096,
        nullptr, true}), ggml_free);
    if (!kv_ctx) { throw std::runtime_error("KV_CONTEXT_ALLOC_FAILED"); }
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> kv;
    for (int layer = 0; layer < args.layers; ++layer) {
        auto * k = ggml_new_tensor_2d(kv_ctx.get(), GGML_TYPE_F16,
                                      kv_width, args.kv);
        auto * v = ggml_new_tensor_2d(kv_ctx.get(), GGML_TYPE_F16,
                                      kv_width, args.kv);
        if (!k || !v) { throw std::runtime_error("KV_METADATA_FAILED"); }
        ggml_set_name(k, ("cache_k_l"+std::to_string(layer)).c_str());
        ggml_set_name(v, ("cache_v_l"+std::to_string(layer)).c_str());
        kv.emplace_back(k, v);
    }
    buffer_ptr kv_buffer(
        ggml_backend_alloc_ctx_tensors(kv_ctx.get(), backend.get()),
        ggml_backend_buffer_free);
    if (!kv_buffer) { throw std::runtime_error("KV_DEVICE_MEMORY_EXCEEDED"); }
    ggml_backend_buffer_clear(kv_buffer.get(), 0);
    const size_t history_elements =
        static_cast<size_t>(kv_width) * (args.kv - args.chunk);
    std::vector<ggml_fp16_t> history(history_elements);
    for (int layer = 0; layer < args.layers; ++layer) {
        for (size_t j = 0; j < history.size(); ++j) {
            history[j] = ggml_fp32_to_fp16(
                0.015f * std::sin(float((j % 97) + layer * 3)));
        }
        for (auto * t : {kv[layer].first, kv[layer].second}) {
            if (!history.empty()) {
                ggml_backend_tensor_set(t, history.data(), 0,
                                        history.size() * sizeof(ggml_fp16_t));
            }
        }
    }

    const size_t nodes_hint = static_cast<size_t>(args.layers) * 1024 + 1024;
    ggml_ptr graph_ctx(ggml_init({
        ggml_tensor_overhead() * nodes_hint +
        ggml_graph_overhead_custom(512*args.layers, false) + 65536,
        nullptr, true}), ggml_free);
    if (!graph_ctx) { throw std::runtime_error("GRAPH_CONTEXT_ALLOC_FAILED"); }
    // Always profile the final EXACT X tokens at KV-X..KV-1, not the
    // final possibly-short scheduling chunk. This also works for
    // KV512/X96 and keeps the number of computed tokens honest.
    const uint32_t chunk = 0;
    const uint32_t token_begin = static_cast<uint32_t>(args.kv - args.chunk);
    ggml_phone_qwen3::multi_graph_result graph {};
    const char * why = "NONE";
    if (!ggml_phone_qwen3::build_graph_layers(
            graph_ctx.get(), p, chunk, weights, graph, why, kv,
            args.diagnose_layers, token_begin)) {
        throw std::runtime_error(std::string("PREFIX_GRAPH_FAILED:")+why);
    }
    using gallocr_ptr = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;
    gallocr_ptr allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())),
        ggml_gallocr_free);
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph.graph)) {
        throw std::runtime_error("PREFIX_GRAPH_WORKSPACE_EXCEEDED");
    }
    htp_cache_guard cache {};
    if (htp) {
        const auto reg = ggml_backend_dev_backend_reg(dev);
        cache.backend = backend.get();
        cache.clear = reg == nullptr ? nullptr :
            reinterpret_cast<void (*)(ggml_backend_t)>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_hexagon_clear_fixed_graph_cache"));
        if (!cache.clear) {
            throw std::runtime_error("HTP_FIXED_GRAPH_CACHE_LIFETIME_UNSUPPORTED");
        }
    }
    const double graph_build_ms = elapsed_ms(build_begin) - weight_upload_ms;
    std::vector<float> activation(
        static_cast<size_t>(p.n_embd) * args.chunk);
    for (size_t i = 0; i < activation.size(); ++i) {
        activation[i] = 0.02f * std::sin(float(i % 101));
    }
    // Position/mask/KV-index input is replicated per layer but the same
    // synthetic sequence geometry applies to every Transformer layer.
    ggml_phone_qwen3::side_inputs side {};
    if (!ggml_phone_qwen3::make_side_inputs(
            p, chunk, side, why, token_begin)) {
        throw std::runtime_error(std::string("PROFILE_SIDE_INPUT_FAILED:")+why);
    }
    auto prepare = [&]() {
        ggml_backend_tensor_set(graph.input, activation.data(), 0,
                                activation.size() * sizeof(float));
        for (const auto & layer : graph.layers) {
            ggml_backend_tensor_set(layer.positions, side.positions.data(), 0,
                                    side.positions.size() * sizeof(int32_t));
            ggml_backend_tensor_set(layer.k_indices, side.k_indices.data(), 0,
                                    side.k_indices.size() * sizeof(int64_t));
            ggml_backend_tensor_set(layer.v_indices, side.v_indices.data(), 0,
                                    side.v_indices.size() * sizeof(int64_t));
            ggml_backend_tensor_set(layer.mask, side.causal_mask_f16.data(), 0,
                                    side.causal_mask_f16.size() * sizeof(uint16_t));
        }
        ggml_backend_synchronize(backend.get());
    };
    double precompile_ms = 0.0;
    if (htp) {
        // Hexagon's precompile_fixed_graph() explicitly rejects ordinary
        // GGML Graph UIDs: bit 63 is the fixed-Graph cache namespace.
        // The legacy N1 register_chunk() path sets this bit before calling
        // the same proc. This standalone profiler owns one HTP Graph per
        // process, so serial=1 is unique within the fresh HTP session.
        // Assign the UID only after the fully bound Graph is finalized.
        constexpr uint64_t fixed_graph_namespace = UINT64_C(1) << 63;
        graph.graph->uid = fixed_graph_namespace | UINT64_C(1);
        if (ggml_graph_n_nodes(graph.graph) <= 0) {
            throw std::runtime_error("HTP_MULTI_LAYER_GRAPH_EMPTY");
        }
        // HTP specializes opnodes after all repacked weights, KV and
        // fixed per-Chunk input addresses exist. No PC/Adreno fallback.
        prepare();
        std::cerr << "[V2_HTP_GRAPH] event=PRECOMPILE_START uid="
                  << graph.graph->uid << " nodes=" << ggml_graph_n_nodes(graph.graph)
                  << " layers=" << args.layers << '\n';
        const auto start = clock_t::now();
        if (!precompile(backend.get(), graph.graph)) {
            throw std::runtime_error(
                "HTP_MULTI_LAYER_PRECOMPILE_FAILED uid=" +
                std::to_string(graph.graph->uid) + " nodes=" +
                std::to_string(ggml_graph_n_nodes(graph.graph)));
        }
        ggml_backend_synchronize(backend.get());
        precompile_ms = elapsed_ms(start);
        std::cerr << "[V2_HTP_GRAPH] event=PRECOMPILE_DONE status=PASS"
                  << " elapsed_ms=" << precompile_ms << '\n';
    }
    std::vector<double> prepare_ms, compute_ms;
    for (int repeat = -args.warmups; repeat < args.repeats; ++repeat) {
        const auto preparation = clock_t::now();
        prepare();
        const double prep = elapsed_ms(preparation);
        const auto computing = clock_t::now();
        const auto code = ggml_backend_graph_compute(
            backend.get(), graph.graph);
        ggml_backend_synchronize(backend.get());
        if (code != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("NATIVE_STAGE_COMPUTE_FAILED_NO_FALLBACK");
        }
        if (repeat >= 0) {
            prepare_ms.push_back(prep);
            compute_ms.push_back(elapsed_ms(computing));
        }
    }
    sample_result result {};
    result.output.resize(activation.size());
    ggml_backend_tensor_get(graph.output, result.output.data(), 0,
                            result.output.size() * sizeof(float));
    // Isolate host<->backend buffer copies from Graph compute. These timings
    // include backend synchronization and do NOT include PC<->Phone TCP.
    // Run after the final compute to avoid changing its measurement.
    const size_t activation_bytes = activation.size() * sizeof(float);
    std::vector<double> upload_samples, readback_samples;
    for (int rep = 0; rep < 5; ++rep) {
        auto start = clock_t::now();
        ggml_backend_tensor_get(
            graph.output, result.output.data(), 0, activation_bytes);
        ggml_backend_synchronize(backend.get());
        if (rep > 0) {
            readback_samples.push_back(elapsed_ms(start));
        }
        start = clock_t::now();
        ggml_backend_tensor_set(
            graph.input, activation.data(), 0, activation_bytes);
        ggml_backend_synchronize(backend.get());
        if (rep > 0) {
            upload_samples.push_back(elapsed_ms(start));
        }
    }
    result.activation_bytes = activation_bytes;
    result.activation_upload_median_ms = median(upload_samples);
    result.activation_readback_median_ms = median(readback_samples);
    if (args.diagnose_layers) {
        result.layer_outputs.reserve(graph.layers.size());
        for (const auto & layer : graph.layers) {
            std::vector<float> captured(activation.size());
            ggml_backend_tensor_get(layer.output, captured.data(), 0,
                                    captured.size() * sizeof(float));
            result.layer_outputs.push_back(std::move(captured));
        }
    }
    for (float x : result.output) {
        if (!std::isfinite(x)) {
            throw std::runtime_error("NONFINITE_NATIVE_STAGE_OUTPUT");
        }
    }
    result.graph_build_ms = graph_build_ms;
    result.weight_upload_ms = weight_upload_ms;
    result.prepare_ms = median(prepare_ms);
    result.compute_ms = median(compute_ms);
    result.htp_precompile_ms = precompile_ms;
    result.weight_file_read_ms = uploaded.file_read_ms;
    result.weight_backend_set_ms = uploaded.backend_set_ms;
    result.same_bank_refill_median_ms = same_bank_refill_ms;
    result.raw_weight_bytes = uploaded.original_bytes;
    result.weight_bytes = ggml_backend_buffer_get_size(weight_buffer.get()) +
        (repack_buffer ? ggml_backend_buffer_get_size(repack_buffer.get()) : 0);
    result.kv_bytes = ggml_backend_buffer_get_size(kv_buffer.get());
    result.graph_workspace_bytes =
        ggml_gallocr_get_buffer_size(allocator.get(), 0);
    return result;
}

// Compares outputs after both physical devices synchronize; errors refer to
// the SAME layer output in the SAME full multi-layer Graph, not to isolated
// reruns of N1/N2/N3 that can have different allocation and cache lifetimes.
struct parity_metrics {
    double relative_rmse = 0.0;
    double max_abs = 0.0;
    double cosine = 0.0;
    double reference_rms = 0.0;
    size_t worst_index = 0;
    float at_worst = 0;
    float ref_at_worst = 0;
};

parity_metrics parity_of(const std::vector<float> & actual,
                         const std::vector<float> & reference) {
    if (actual.empty() || actual.size() != reference.size()) {
        throw std::runtime_error("CPU_REFERENCE_OUTPUT_SHAPE_MISMATCH");
    }
    parity_metrics result {};
    double squared_error = 0.0;
    double sum_reference_squares = 0.0;
    double sum_actual_squares = 0.0;
    double dot = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) {
            throw std::runtime_error("NONFINITE_STAGE_PARITY_OUTPUT");
        }
        const double difference = double(actual[i]) - double(reference[i]);
        const double abs_error = std::abs(difference);
        if (abs_error > result.max_abs) {
            result.max_abs = abs_error;
            result.worst_index = i;
            result.at_worst = actual[i];
            result.ref_at_worst = reference[i];
        }
        squared_error += difference * difference;
        sum_reference_squares += double(reference[i]) * double(reference[i]);
        sum_actual_squares += double(actual[i]) * double(actual[i]);
        dot += double(actual[i]) * double(reference[i]);
    }
    result.relative_rmse = std::sqrt(
        squared_error / std::max(sum_reference_squares, 1e-20));
    result.reference_rms = std::sqrt(
        sum_reference_squares / reference.size());
    result.cosine = dot / std::sqrt(
        std::max(sum_reference_squares * sum_actual_squares, 1e-40));
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const parameters args = parse(argc, argv);
        const sample_result measured = run(args);
        double max_abs = 0.0;
        double relative_rmse = 0.0;
        const char * parity = "NOT_REQUESTED";
        bool parity_failed = false;
        if (args.verify_cpu) {
            parameters cpu = args;
            cpu.backend = 'C';
            cpu.verify_cpu = false;
            // Bank refill benchmarking belongs ONLY to the actual HTP
            // execution, never to the CPU numerical reference.
            cpu.bank_repeats = 0;
            cpu.warmups = 0;
            cpu.repeats = 1;
            const sample_result reference = run(cpu);
            const auto final = parity_of(measured.output, reference.output);
            relative_rmse = final.relative_rmse;
            max_abs = final.max_abs;
            if (args.diagnose_layers) {
                if (measured.layer_outputs.size() != static_cast<size_t>(args.layers) ||
                        reference.layer_outputs.size() != static_cast<size_t>(args.layers)) {
                    throw std::runtime_error("INCOMPLETE_SAME_GRAPH_LAYER_OUTPUT_CAPTURE");
                }
                for (int i = 0; i < args.layers; ++i) {
                    const auto m = parity_of(measured.layer_outputs[i],
                                             reference.layer_outputs[i]);
                    std::cerr << "[V2_LAYER_PARITY] layer=" << i
                              << " backend=" << args.backend
                              << " rel_rmse=" << m.relative_rmse
                              << " max_abs=" << m.max_abs
                              << " cosine=" << m.cosine
                              << " ref_rms=" << m.reference_rms
                              << " worst_index=" << m.worst_index
                              << " actual=" << m.at_worst
                              << " reference=" << m.ref_at_worst
                              << " status=" <<
                        (m.relative_rmse <= args.max_relative_rmse &&
                         m.max_abs <= args.max_abs_error ? "PASS" : "FAIL")
                              << '\n';
                }
            }
            parity_failed = !std::isfinite(relative_rmse) ||
                relative_rmse > args.max_relative_rmse ||
                max_abs > args.max_abs_error;
            parity = parity_failed ? "FAIL" : "PASS";
            if (parity_failed) {
                // Keep actual, independently measured HTP timing available
                // for research, but return nonzero and prohibit this sample
                // from planner-approved profiles. Never relax parity silently.
                std::cerr << "[V2_STAGE_PARITY] status=FAIL rel_rmse="
                          << relative_rmse << " max_abs=" << max_abs
                          << " layers=" << args.layers
                          << " backend=" << args.backend << '\n';
            }
        }
        std::cout << "backend,layers,chunk,kv_tokens,repeats,graph_build_ms,"
                     "weight_upload_ms,htp_precompile_ms,prepare_median_ms,"
                     "compute_median_ms,weight_bytes,kv_bytes,"
                     "graph_workspace_bytes,relative_rmse,max_abs,parity,source,"
                     "token_begin,activation_bytes,activation_upload_median_ms,"
                     "activation_readback_median_ms,weight_file_read_ms,"
                     "weight_backend_set_ms,same_bank_refill_median_ms,"
                     "raw_weight_bytes\n";
        std::cout << args.backend << ',' << args.layers << ',' << args.chunk
                  << ',' << args.kv << ',' << args.repeats << ','
                  << measured.graph_build_ms << ','
                  << measured.weight_upload_ms << ','
                  << measured.htp_precompile_ms << ','
                  << measured.prepare_ms << ',' << measured.compute_ms << ','
                  << measured.weight_bytes << ',' << measured.kv_bytes << ','
                  << measured.graph_workspace_bytes << ','
                  << relative_rmse << ',' << max_abs << ',' << parity
                  << (args.diagnose_layers ?
                      ",NATIVE_SINGLE_BACKEND_GGUF_PREFIX_DIAGNOSTIC," :
                      ",NATIVE_SINGLE_BACKEND_GGUF_PREFIX,")
                  << (args.kv - args.chunk) << ','
                  << measured.activation_bytes << ','
                  << measured.activation_upload_median_ms << ','
                  << measured.activation_readback_median_ms << ','
                  << measured.weight_file_read_ms << ','
                  << measured.weight_backend_set_ms << ','
                  << measured.same_bank_refill_median_ms << ','
                  << measured.raw_weight_bytes << '\n';
        // Exit 2 means native compute succeeded, but numerical accuracy
        // failed. It is NOT a PASS and will never enter the accepted CSV.
        return parity_failed ? 2 : 0;
    } catch (const std::exception & ex) {
        std::cerr << "[V2_STAGE_PROFILE] status=FAIL reason="
                  << ex.what() << '\n';
        return 1;
    }
}
