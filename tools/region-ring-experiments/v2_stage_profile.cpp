// Real, isolated PC Qwen3 prefix profiling with original GGUF weights.
// Deliberately no llama_context, Region Ring, RPC, embeddings or LM Head.
// Only ONE native backend is constructed (CPU or GPU). Unsupported ops
// fail closed; there is no CPU fallback from the CUDA path.
//
// The same Qwen3 layer graph builder is used by the Phone NPU path.
// Each measurement graph contains the model's actual layers [0,N).
#include "ggml.h"
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
            std::cout << "v2-stage-profile --model FILE --backend G|C "
                         "--layers N --chunk X --kv K [--warmups N] "
                         "[--repeats N] [--threads N]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) { throw std::runtime_error("MISSING_ARGUMENT:" + key); }
        const char * value = argv[++i];
        if (key == "--model") { p.model = value; }
        else if (key == "--backend") {
            if (std::strlen(value) != 1) {
                throw std::runtime_error("BACKEND_MUST_BE_G_OR_C");
            }
            p.backend = value[0];
        } else if (key == "--layers") { p.layers = parse_int(value, "LAYERS"); }
        else if (key == "--chunk") { p.chunk = parse_int(value, "CHUNK"); }
        else if (key == "--kv") { p.kv = parse_int(value, "KV"); }
        else if (key == "--warmups") { p.warmups = parse_int(value, "WARMUPS"); }
        else if (key == "--repeats") { p.repeats = parse_int(value, "REPEATS"); }
        else if (key == "--threads") { p.threads = parse_int(value, "THREADS"); }
        else { throw std::runtime_error("UNKNOWN_ARGUMENT:" + key); }
    }
    if (p.model.empty() || (p.backend != 'G' && p.backend != 'C') ||
            p.layers < 1 || p.chunk < 1 || p.kv < p.chunk ||
            p.kv % p.chunk != 0 || p.repeats < 1 || p.threads < 1) {
        throw std::runtime_error("INVALID_PC_STAGE_PROFILE_SHAPE");
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

struct to_load {
    ggml_tensor * tensor = nullptr;
    size_t offset = 0;
    size_t bytes = 0;
};

ggml_phone_qwen3::layer_weights load_layer(
        ggml_context * weight_ctx, const ggml_context * model_ctx,
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
            weight_ctx, source->type, GGML_MAX_DIMS, source->ne);
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

void upload_weights(
        const std::string & path, const std::vector<to_load> & todo) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("MODEL_GGUF_OPEN_FAILED"); }
    // One Tensor at a time: avoid duplicating N layers of quantized weights
    // in host memory. The backend owns the already-allocated destination.
    std::vector<uint8_t> scratch;
    for (const auto & item : todo) {
        if (item.bytes > static_cast<size_t>(
                std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("GGUF_TENSOR_EXCEEDS_IO_LIMIT");
        }
        scratch.resize(item.bytes);
        in.clear();
        in.seekg(static_cast<std::streamoff>(item.offset));
        if (!in.read(reinterpret_cast<char *>(scratch.data()),
                     static_cast<std::streamsize>(item.bytes))) {
            throw std::runtime_error("GGUF_READ_FAILED:" +
                                     std::string(item.tensor->name));
        }
        ggml_backend_tensor_set(item.tensor, scratch.data(), 0, item.bytes);
    }
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
    p.n_heads_kv = field_u32(g, "qwen3.attention.head_count_kv");
    p.head_dim = field_u32(g, "qwen3.rope.dimension_count");
    p.rope_type = ggml_phone_owned::k_rope_neox;
    p.kv_type = ggml_phone_owned::k_f16;
    p.n_ctx_orig = field_u32(g, "qwen3.context_length");
    p.rms_norm_eps = field_f32(g, "qwen3.attention.layer_norm_rms_epsilon");
    p.rope_freq_base = field_f32(g, "qwen3.rope.freq_base");
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

int run(const parameters & args) {
    ggml_backend_load_all();
    const auto type = args.backend == 'G' ?
        GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU;
    auto dev = ggml_backend_dev_by_type(type);
    if (!dev) { throw std::runtime_error("REQUESTED_BACKEND_NOT_AVAILABLE"); }
    backend_ptr backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
    if (!backend) { throw std::runtime_error("BACKEND_INIT_FAILED"); }
    if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        ggml_backend_cpu_set_n_threads(backend.get(), args.threads);
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
    request.device = args.backend == 'G' ?
        llama_ring_v2::backend::G : llama_ring_v2::backend::C;
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
    if (!weight_ctx) { throw std::runtime_error("WEIGHT_CONTEXT_ALLOC_FAILED"); }
    std::vector<to_load> todo;
    std::vector<ggml_phone_qwen3::layer_weights> weights;
    for (int layer = 0; layer < args.layers; ++layer) {
        weights.push_back(load_layer(weight_ctx.get(), model_ctx.get(),
                                     gguf.get(), layer, todo));
    }
    buffer_ptr weight_buffer(
        ggml_backend_alloc_ctx_tensors(weight_ctx.get(), backend.get()),
        ggml_backend_buffer_free);
    if (!weight_buffer) { throw std::runtime_error("PREFIX_WEIGHTS_EXCEED_BACKEND_MEMORY"); }
    ggml_backend_buffer_set_usage(
        weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const auto upload_begin = clock_t::now();
    upload_weights(args.model, todo);
    ggml_backend_synchronize(backend.get());
    const double weight_upload_ms = elapsed_ms(upload_begin);

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
    const auto chunk = static_cast<uint32_t>(args.kv/args.chunk - 1);
    ggml_phone_qwen3::multi_graph_result graph {};
    const char * why = "NONE";
    if (!ggml_phone_qwen3::build_graph_layers(
            graph_ctx.get(), p, chunk, weights, graph, why, kv)) {
        throw std::runtime_error(std::string("PREFIX_GRAPH_FAILED:")+why);
    }
    using gallocr_ptr = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;
    gallocr_ptr allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())),
        ggml_gallocr_free);
    if (!allocator || !ggml_gallocr_alloc_graph(allocator.get(), graph.graph)) {
        throw std::runtime_error("PREFIX_GRAPH_WORKSPACE_EXCEEDED");
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
    if (!ggml_phone_qwen3::make_side_inputs(p, chunk, side, why)) {
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
    std::vector<float> output(activation.size());
    ggml_backend_tensor_get(graph.output, output.data(), 0,
                            output.size() * sizeof(float));
    for (float x : output) {
        if (!std::isfinite(x)) {
            throw std::runtime_error("NONFINITE_NATIVE_STAGE_OUTPUT");
        }
    }
    std::cout << "backend,layers,chunk,kv_tokens,repeats,graph_build_ms,"
                 "weight_upload_ms,prepare_median_ms,compute_median_ms,"
                 "weight_bytes,kv_bytes,graph_workspace_bytes,source\n";
    std::cout << args.backend << ',' << args.layers << ',' << args.chunk
              << ',' << args.kv << ',' << args.repeats << ','
              << graph_build_ms << ',' << weight_upload_ms << ','
              << median(prepare_ms) << ',' << median(compute_ms) << ','
              << ggml_backend_buffer_get_size(weight_buffer.get()) << ','
              << ggml_backend_buffer_get_size(kv_buffer.get()) << ','
              << ggml_gallocr_get_buffer_size(allocator.get(), 0)
              << ",NATIVE_SINGLE_BACKEND_GGUF_PREFIX\n";
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse(argc, argv));
    } catch (const std::exception & ex) {
        std::cerr << "[V2_STAGE_PROFILE] status=FAIL reason="
                  << ex.what() << '\n';
        return 1;
    }
}
