#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// Standalone microbenchmark for the PC Attention-entry split discussed in the
// hybrid scheduler work.
//
// Baseline:
//     GPU RMSNorm -> gamma*x -> Q/K/V GEMMs
//
// Split:
//     copy stream: GPU x -> pinned host -> CPU RMS reduction -> inv_r -> GPU
//     compute stream:             gamma*x -> Q/K/V GEMMs
//                                        \           /
//                                         join(inv_r)
//                                             |
////                                      scale Q/K/V
//
// The benchmark deliberately does not use the hybrid planner, local ring, or
// chunk scheduler.  It measures one 128-token (configurable) Attention-entry
// stage with Qwen3-30B-A3B-like dimensions and quantized Q/K/V weights.
//
// The two CUDA backend instances target the same device but own independent
// streams.  This is intentional: D2H/H2D traffic must be able to overlap the
// GPU numerator graph instead of being serialized onto the compute stream.

using clock_type = std::chrono::steady_clock;

static double elapsed_ms(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct options {
    int device = 0;
    int64_t tokens = 128;
    int64_t n_embd = 2048;
    int64_t q_dim = 4096;
    int64_t kv_dim = 512;
    int warmup = 10;
    int iters = 100;
    float eps = 1.0e-6f;
    enum ggml_type weight_type = GGML_TYPE_Q4_K;
};

static void usage(const char * argv0) {
    std::printf(
        "usage: %s [options]\n"
        "  --device N          CUDA device (default: 0)\n"
        "  --tokens N          token count (default: 128)\n"
        "  --embd N            hidden size (default: 2048)\n"
        "  --q-dim N           Q projection width (default: 4096)\n"
        "  --kv-dim N          K/V projection width (default: 512)\n"
        "  --warmup N          warmup iterations (default: 10)\n"
        "  --iters N           measured iterations (default: 100)\n"
        "  --eps X             RMSNorm epsilon (default: 1e-6)\n"
        "  --weight-type TYPE  q4_k | q6_k | q4_0 (default: q4_k)\n",
        argv0);
}

static bool parse_i64(const char * s, int64_t & v) {
    char * end = nullptr;
    const long long x = std::strtoll(s, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    v = (int64_t) x;
    return true;
}

static bool parse_i32(const char * s, int & v) {
    int64_t tmp = 0;
    if (!parse_i64(s, tmp)) {
        return false;
    }
    v = (int) tmp;
    return true;
}

static bool parse_options(int argc, char ** argv, options & opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "--device") {
            const char * s = need("--device");
            if (!s || !parse_i32(s, opt.device)) return false;
        } else if (a == "--tokens") {
            const char * s = need("--tokens");
            if (!s || !parse_i64(s, opt.tokens)) return false;
        } else if (a == "--embd") {
            const char * s = need("--embd");
            if (!s || !parse_i64(s, opt.n_embd)) return false;
        } else if (a == "--q-dim") {
            const char * s = need("--q-dim");
            if (!s || !parse_i64(s, opt.q_dim)) return false;
        } else if (a == "--kv-dim") {
            const char * s = need("--kv-dim");
            if (!s || !parse_i64(s, opt.kv_dim)) return false;
        } else if (a == "--warmup") {
            const char * s = need("--warmup");
            if (!s || !parse_i32(s, opt.warmup)) return false;
        } else if (a == "--iters") {
            const char * s = need("--iters");
            if (!s || !parse_i32(s, opt.iters)) return false;
        } else if (a == "--eps") {
            const char * s = need("--eps");
            if (!s) return false;
            char * end = nullptr;
            opt.eps = std::strtof(s, &end);
            if (!end || *end != '\0') return false;
        } else if (a == "--weight-type") {
            const char * s = need("--weight-type");
            if (!s) return false;
            if (std::strcmp(s, "q4_k") == 0) {
                opt.weight_type = GGML_TYPE_Q4_K;
            } else if (std::strcmp(s, "q6_k") == 0) {
                opt.weight_type = GGML_TYPE_Q6_K;
            } else if (std::strcmp(s, "q4_0") == 0) {
                opt.weight_type = GGML_TYPE_Q4_0;
            } else {
                std::fprintf(stderr, "unsupported --weight-type: %s\n", s);
                return false;
            }
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return false;
        }
    }

    if (opt.device < 0 || opt.tokens <= 0 || opt.n_embd <= 0 || opt.q_dim <= 0 ||
        opt.kv_dim <= 0 || opt.warmup < 0 || opt.iters <= 0 || opt.eps <= 0.0f) {
        return false;
    }

    const int64_t bs = ggml_blck_size(opt.weight_type);
    if (opt.n_embd % bs != 0) {
        std::fprintf(stderr,
            "hidden size %" PRId64 " must be divisible by weight block size %" PRId64 "\n",
            opt.n_embd, bs);
        return false;
    }
    return true;
}

static void set_f32_random(
        ggml_tensor * t,
        std::mt19937 & rng,
        float lo,
        float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> data((size_t) ggml_nelements(t));
    for (float & v : data) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

static void set_gamma_random(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(0.8f, 1.2f);
    std::vector<float> data((size_t) ggml_nelements(t));
    for (float & v : data) {
        v = dist(rng);
    }
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

static bool set_quantized_weight(
        ggml_tensor * t,
        std::mt19937 & rng) {
    const int64_t n_per_row = t->ne[0];
    const int64_t nrows = ggml_nrows(t);

    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::vector<float> src((size_t) n_per_row * (size_t) nrows);
    for (float & v : src) {
        v = dist(rng);
    }

    std::vector<uint8_t> quant(ggml_nbytes(t));
    const size_t written = ggml_quantize_chunk(
        t->type, src.data(), quant.data(), 0, nrows, n_per_row, nullptr);

    if (written != quant.size()) {
        std::fprintf(stderr,
            "quantized size mismatch for %s: wrote=%zu expected=%zu\n",
            t->name, written, quant.size());
        return false;
    }

    ggml_backend_tensor_set(t, quant.data(), 0, quant.size());
    return true;
}

static void cpu_rms_inv(
        const float * x,
        float * inv_r,
        int64_t n_embd,
        int64_t tokens,
        float eps) {
    for (int64_t t = 0; t < tokens; ++t) {
        const float * row = x + t * n_embd;
        double sum_sq = 0.0;
        for (int64_t i = 0; i < n_embd; ++i) {
            const double v = row[i];
            sum_sq += v * v;
        }
        inv_r[t] = 1.0f / std::sqrt((float) (sum_sq / (double) n_embd) + eps);
    }
}

struct sample_stats {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double min = 0.0;
};

static sample_stats summarize(std::vector<double> v) {
    sample_stats s;
    if (v.empty()) {
        return s;
    }

    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
    s.min = *std::min_element(v.begin(), v.end());

    std::sort(v.begin(), v.end());
    auto percentile = [&](double p) {
        const double pos = p * (double) (v.size() - 1);
        const size_t lo = (size_t) std::floor(pos);
        const size_t hi = (size_t) std::ceil(pos);
        const double f = pos - (double) lo;
        return v[lo] * (1.0 - f) + v[hi] * f;
    };

    s.p50 = percentile(0.50);
    s.p95 = percentile(0.95);
    return s;
}

struct diff_stats {
    double max_abs = 0.0;
    double peak_ref = 0.0;
    double cosine = 0.0;
    bool finite = true;
};

static diff_stats compare_vectors(const std::vector<float> & ref, const std::vector<float> & got) {
    diff_stats s;
    if (ref.size() != got.size() || ref.empty()) {
        s.finite = false;
        return s;
    }

    long double dot = 0.0;
    long double aa = 0.0;
    long double bb = 0.0;

    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = ref[i];
        const double b = got[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            s.finite = false;
        }
        s.max_abs = std::max(s.max_abs, std::abs(a - b));
        s.peak_ref = std::max(s.peak_ref, std::abs(a));
        dot += (long double) a * (long double) b;
        aa += (long double) a * (long double) a;
        bb += (long double) b * (long double) b;
    }

    if (aa > 0.0 && bb > 0.0) {
        s.cosine = (double) (dot / std::sqrt(aa * bb));
    }
    return s;
}

static bool numerically_close(const diff_stats & s) {
    const double scale = std::max(1.0, s.peak_ref);
    return s.finite && s.cosine >= 0.9999 && s.max_abs <= 0.02 * scale;
}

struct split_sample {
    double wall_ms = 0.0;
    double d2h_ready_ms = 0.0;
    double cpu_rms_ms = 0.0;
};

int main(int argc, char ** argv) {
    options opt;
    if (!parse_options(argc, argv, opt)) {
        usage(argv[0]);
        return 1;
    }

    const int n_cuda = ggml_backend_cuda_get_device_count();
    if (opt.device >= n_cuda) {
        std::fprintf(stderr, "CUDA device %d unavailable (count=%d)\n", opt.device, n_cuda);
        return 2;
    }

    char dev_desc[256] = {};
    ggml_backend_cuda_get_device_description(opt.device, dev_desc, sizeof(dev_desc));

    ggml_backend_t compute_backend = ggml_backend_cuda_init(opt.device);
    ggml_backend_t copy_backend = ggml_backend_cuda_init(opt.device);
    if (!compute_backend || !copy_backend) {
        std::fprintf(stderr, "failed to initialize two CUDA backend contexts\n");
        return 3;
    }

    // Main context: inputs, quantized weights, GPU-serial graph, and numerator graph.
    ggml_init_params p = {};
    p.mem_size = 16 * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc = true;
    ggml_context * ctx = ggml_init(p);
    if (!ctx) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return 4;
    }

    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, opt.n_embd, opt.tokens);
    ggml_tensor * gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, opt.n_embd);
    ggml_tensor * wq = ggml_new_tensor_2d(ctx, opt.weight_type, opt.n_embd, opt.q_dim);
    ggml_tensor * wk = ggml_new_tensor_2d(ctx, opt.weight_type, opt.n_embd, opt.kv_dim);
    ggml_tensor * wv = ggml_new_tensor_2d(ctx, opt.weight_type, opt.n_embd, opt.kv_dim);

    ggml_set_name(x, "pc_rms_x");
    ggml_set_name(gamma, "pc_rms_gamma");
    ggml_set_name(wq, "pc_rms_wq");
    ggml_set_name(wk, "pc_rms_wk");
    ggml_set_name(wv, "pc_rms_wv");

    // A: normal GPU path.
    ggml_tensor * x_norm = ggml_rms_norm(ctx, x, opt.eps);
    ggml_set_name(x_norm, "pc_rms_gpu_norm");
    ggml_tensor * x_norm_gamma = ggml_mul(ctx, x_norm, gamma);
    ggml_set_name(x_norm_gamma, "pc_rms_gpu_norm_gamma");

    ggml_tensor * q_ref = ggml_mul_mat(ctx, wq, x_norm_gamma);
    ggml_tensor * k_ref = ggml_mul_mat(ctx, wk, x_norm_gamma);
    ggml_tensor * v_ref = ggml_mul_mat(ctx, wv, x_norm_gamma);
    ggml_set_name(q_ref, "pc_rms_q_ref");
    ggml_set_name(k_ref, "pc_rms_k_ref");
    ggml_set_name(v_ref, "pc_rms_v_ref");

    ggml_cgraph * graph_serial = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph_serial, q_ref);
    ggml_build_forward_expand(graph_serial, k_ref);
    ggml_build_forward_expand(graph_serial, v_ref);

    // B compute branch: gamma*x followed immediately by Q/K/V GEMMs.
    ggml_tensor * x_gamma = ggml_mul(ctx, x, gamma);
    ggml_set_name(x_gamma, "pc_rms_split_gamma_x");

    ggml_tensor * q_num = ggml_mul_mat(ctx, wq, x_gamma);
    ggml_tensor * k_num = ggml_mul_mat(ctx, wk, x_gamma);
    ggml_tensor * v_num = ggml_mul_mat(ctx, wv, x_gamma);
    ggml_set_name(q_num, "pc_rms_q_num");
    ggml_set_name(k_num, "pc_rms_k_num");
    ggml_set_name(v_num, "pc_rms_v_num");

    ggml_cgraph * graph_num = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph_num, q_num);
    ggml_build_forward_expand(graph_num, k_num);
    ggml_build_forward_expand(graph_num, v_num);

    ggml_backend_buffer_t main_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cuda_buffer_type(opt.device));
    if (!main_buffer) {
        std::fprintf(stderr, "failed to allocate main CUDA tensor buffer\n");
        return 5;
    }

    // Scale context. q/k/v leaves alias the actual numerator outputs, so there
    // is no extra D2D copy between the numerator graph and the scale graph.
    ggml_init_params sp = {};
    sp.mem_size = 8 * 1024 * 1024;
    sp.mem_buffer = nullptr;
    sp.no_alloc = true;
    ggml_context * scale_ctx = ggml_init(sp);
    if (!scale_ctx) {
        std::fprintf(stderr, "failed to create scale context\n");
        return 6;
    }

    ggml_tensor * q_leaf = ggml_new_tensor_2d(scale_ctx, GGML_TYPE_F32, opt.q_dim, opt.tokens);
    ggml_tensor * k_leaf = ggml_new_tensor_2d(scale_ctx, GGML_TYPE_F32, opt.kv_dim, opt.tokens);
    ggml_tensor * v_leaf = ggml_new_tensor_2d(scale_ctx, GGML_TYPE_F32, opt.kv_dim, opt.tokens);
    ggml_tensor * inv_r = ggml_new_tensor_2d(scale_ctx, GGML_TYPE_F32, 1, opt.tokens);

    ggml_set_name(q_leaf, "pc_rms_q_num_alias");
    ggml_set_name(k_leaf, "pc_rms_k_num_alias");
    ggml_set_name(v_leaf, "pc_rms_v_num_alias");
    ggml_set_name(inv_r, "pc_rms_inv_r");

    if (ggml_backend_tensor_alloc(q_num->buffer, q_leaf, q_num->data) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(k_num->buffer, k_leaf, k_num->data) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(v_num->buffer, v_leaf, v_num->data) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "failed to alias numerator tensors into scale context\n");
        return 7;
    }

    ggml_tensor * q_split = ggml_mul(scale_ctx, q_leaf, inv_r);
    ggml_tensor * k_split = ggml_mul(scale_ctx, k_leaf, inv_r);
    ggml_tensor * v_split = ggml_mul(scale_ctx, v_leaf, inv_r);
    ggml_set_name(q_split, "pc_rms_q_split");
    ggml_set_name(k_split, "pc_rms_k_split");
    ggml_set_name(v_split, "pc_rms_v_split");

    ggml_cgraph * graph_scale = ggml_new_graph_custom(scale_ctx, 32, false);
    ggml_build_forward_expand(graph_scale, q_split);
    ggml_build_forward_expand(graph_scale, k_split);
    ggml_build_forward_expand(graph_scale, v_split);

    ggml_backend_buffer_t scale_buffer =
        ggml_backend_alloc_ctx_tensors_from_buft(scale_ctx, ggml_backend_cuda_buffer_type(opt.device));
    if (!scale_buffer) {
        std::fprintf(stderr, "failed to allocate scale CUDA tensor buffer\n");
        return 8;
    }

    // Pinned host buffers make the copy stream experiment meaningful.
    const size_t x_bytes = ggml_nbytes(x);
    const size_t r_bytes = (size_t) opt.tokens * sizeof(float);
    ggml_backend_buffer_type_t host_buft = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_t host_x_buffer = ggml_backend_buft_alloc_buffer(host_buft, x_bytes);
    ggml_backend_buffer_t host_r_buffer = ggml_backend_buft_alloc_buffer(host_buft, r_bytes);
    if (!host_x_buffer || !host_r_buffer) {
        std::fprintf(stderr, "failed to allocate CUDA pinned host buffers\n");
        return 9;
    }

    float * host_x = (float *) ggml_backend_buffer_get_base(host_x_buffer);
    float * host_inv_r = (float *) ggml_backend_buffer_get_base(host_r_buffer);

    ggml_backend_event_t d2h_done = ggml_backend_event_new(ggml_backend_get_device(copy_backend));
    ggml_backend_event_t r_ready = ggml_backend_event_new(ggml_backend_get_device(copy_backend));
    if (!d2h_done || !r_ready) {
        std::fprintf(stderr, "CUDA backend events unavailable\n");
        return 10;
    }

    std::mt19937 rng(1234);
    set_f32_random(x, rng, -1.0f, 1.0f);
    set_gamma_random(gamma, rng);
    if (!set_quantized_weight(wq, rng) ||
        !set_quantized_weight(wk, rng) ||
        !set_quantized_weight(wv, rng)) {
        return 11;
    }

    auto run_serial = [&]() -> double {
        const auto t0 = clock_type::now();
        const ggml_status st = ggml_backend_graph_compute_async(compute_backend, graph_serial);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "serial graph failed: %d\n", (int) st);
            std::exit(20);
        }
        ggml_backend_synchronize(compute_backend);
        return elapsed_ms(t0, clock_type::now());
    };

    auto run_num_only = [&]() -> double {
        const auto t0 = clock_type::now();
        const ggml_status st = ggml_backend_graph_compute_async(compute_backend, graph_num);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "numerator graph failed: %d\n", (int) st);
            std::exit(21);
        }
        ggml_backend_synchronize(compute_backend);
        return elapsed_ms(t0, clock_type::now());
    };

    auto run_scale_only = [&]() -> double {
        // inv_r=1 keeps this component timing independent of CPU reduction.
        std::fill(host_inv_r, host_inv_r + opt.tokens, 1.0f);
        ggml_backend_tensor_set(inv_r, host_inv_r, 0, r_bytes);

        const auto t0 = clock_type::now();
        const ggml_status st = ggml_backend_graph_compute_async(compute_backend, graph_scale);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "scale graph failed: %d\n", (int) st);
            std::exit(22);
        }
        ggml_backend_synchronize(compute_backend);
        return elapsed_ms(t0, clock_type::now());
    };

    auto run_split = [&]() -> split_sample {
        split_sample out;
        const auto t0 = clock_type::now();

        // Start the independent copy path first, then immediately launch the
        // numerator graph on another CUDA stream/context.
        ggml_backend_tensor_get_async(copy_backend, x, host_x, 0, x_bytes);
        ggml_backend_event_record(d2h_done, copy_backend);

        const ggml_status num_st = ggml_backend_graph_compute_async(compute_backend, graph_num);
        if (num_st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "split numerator graph failed: %d\n", (int) num_st);
            std::exit(23);
        }

        // CPU waits only for its D2H dependency. The GPU numerator graph keeps
        // running while this host thread waits and then performs the reduction.
        ggml_backend_event_synchronize(d2h_done);
        const auto td2h = clock_type::now();
        out.d2h_ready_ms = elapsed_ms(t0, td2h);

        const auto tcpu0 = clock_type::now();
        cpu_rms_inv(host_x, host_inv_r, opt.n_embd, opt.tokens, opt.eps);
        const auto tcpu1 = clock_type::now();
        out.cpu_rms_ms = elapsed_ms(tcpu0, tcpu1);

        // Return only one scalar per token.  The compute stream waits on this
        // event at the true join point; there is no device-wide synchronize.
        ggml_backend_tensor_set_async(copy_backend, inv_r, host_inv_r, 0, r_bytes);
        ggml_backend_event_record(r_ready, copy_backend);
        ggml_backend_event_wait(compute_backend, r_ready);

        const ggml_status scale_st = ggml_backend_graph_compute_async(compute_backend, graph_scale);
        if (scale_st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "split scale graph failed: %d\n", (int) scale_st);
            std::exit(24);
        }

        ggml_backend_synchronize(compute_backend);
        out.wall_ms = elapsed_ms(t0, clock_type::now());
        return out;
    };

    std::printf(
        "[PC_RMS_BENCH_CONFIG] device=%d device_name=\"%s\" tokens=%" PRId64
        " embd=%" PRId64 " q_dim=%" PRId64 " kv_dim=%" PRId64
        " weight=%s eps=%.9g warmup=%d iters=%d x_bytes=%zu return_bytes=%zu\n",
        opt.device, dev_desc, opt.tokens, opt.n_embd, opt.q_dim, opt.kv_dim,
        ggml_type_name(opt.weight_type), opt.eps, opt.warmup, opt.iters,
        x_bytes, r_bytes);

    // Warm all three graphs and both copy directions before collecting data.
    for (int i = 0; i < opt.warmup; ++i) {
        (void) run_serial();
        (void) run_split();
    }

    // Correctness: the split graph consumes the real numerator outputs through
    // zero-copy aliases, so this compares the actual algebraic reordering.
    (void) run_serial();

    std::vector<float> q_ref_h((size_t) ggml_nelements(q_ref));
    std::vector<float> k_ref_h((size_t) ggml_nelements(k_ref));
    std::vector<float> v_ref_h((size_t) ggml_nelements(v_ref));
    ggml_backend_tensor_get(q_ref, q_ref_h.data(), 0, q_ref_h.size() * sizeof(float));
    ggml_backend_tensor_get(k_ref, k_ref_h.data(), 0, k_ref_h.size() * sizeof(float));
    ggml_backend_tensor_get(v_ref, v_ref_h.data(), 0, v_ref_h.size() * sizeof(float));

    (void) run_split();

    std::vector<float> q_split_h((size_t) ggml_nelements(q_split));
    std::vector<float> k_split_h((size_t) ggml_nelements(k_split));
    std::vector<float> v_split_h((size_t) ggml_nelements(v_split));
    ggml_backend_tensor_get(q_split, q_split_h.data(), 0, q_split_h.size() * sizeof(float));
    ggml_backend_tensor_get(k_split, k_split_h.data(), 0, k_split_h.size() * sizeof(float));
    ggml_backend_tensor_get(v_split, v_split_h.data(), 0, v_split_h.size() * sizeof(float));

    const diff_stats dq = compare_vectors(q_ref_h, q_split_h);
    const diff_stats dk = compare_vectors(k_ref_h, k_split_h);
    const diff_stats dv = compare_vectors(v_ref_h, v_split_h);
    const bool correct = numerically_close(dq) && numerically_close(dk) && numerically_close(dv);

    std::printf(
        "[PC_RMS_CORRECTNESS] status=%s"
        " q_max_abs=%.8g q_cos=%.9f"
        " k_max_abs=%.8g k_cos=%.9f"
        " v_max_abs=%.8g v_cos=%.9f\n",
        correct ? "PASS" : "FAIL",
        dq.max_abs, dq.cosine,
        dk.max_abs, dk.cosine,
        dv.max_abs, dv.cosine);

    std::vector<double> serial_ms;
    std::vector<double> split_ms;
    std::vector<double> d2h_ready_ms;
    std::vector<double> cpu_rms_ms;
    std::vector<double> num_ms;
    std::vector<double> scale_ms;

    serial_ms.reserve(opt.iters);
    split_ms.reserve(opt.iters);
    d2h_ready_ms.reserve(opt.iters);
    cpu_rms_ms.reserve(opt.iters);
    num_ms.reserve(opt.iters);
    scale_ms.reserve(opt.iters);

    for (int i = 0; i < opt.iters; ++i) {
        serial_ms.push_back(run_serial());
        const split_sample sp_sample = run_split();
        split_ms.push_back(sp_sample.wall_ms);
        d2h_ready_ms.push_back(sp_sample.d2h_ready_ms);
        cpu_rms_ms.push_back(sp_sample.cpu_rms_ms);

        // Components are sampled independently; they are diagnostics, not
        // terms that should be summed to reconstruct split wall time.
        num_ms.push_back(run_num_only());
        scale_ms.push_back(run_scale_only());
    }

    const sample_stats s_serial = summarize(serial_ms);
    const sample_stats s_split = summarize(split_ms);
    const sample_stats s_d2h = summarize(d2h_ready_ms);
    const sample_stats s_cpu = summarize(cpu_rms_ms);
    const sample_stats s_num = summarize(num_ms);
    const sample_stats s_scale = summarize(scale_ms);

    const double speedup = s_split.mean > 0.0 ? s_serial.mean / s_split.mean : 0.0;
    const double saved = s_serial.mean - s_split.mean;

    std::printf(
        "[PC_RMS_BENCH] mode=GPU_SERIAL"
        " mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f min_ms=%.6f\n",
        s_serial.mean, s_serial.p50, s_serial.p95, s_serial.min);

    std::printf(
        "[PC_RMS_BENCH] mode=CPU_OVERLAP"
        " mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f min_ms=%.6f"
        " d2h_ready_mean_ms=%.6f cpu_rms_mean_ms=%.6f\n",
        s_split.mean, s_split.p50, s_split.p95, s_split.min,
        s_d2h.mean, s_cpu.mean);

    std::printf(
        "[PC_RMS_COMPONENT] gpu_numerator_mean_ms=%.6f"
        " gpu_scale_mean_ms=%.6f"
        " cpu_branch_pre_h2d_mean_ms=%.6f\n",
        s_num.mean, s_scale.mean, s_d2h.mean + s_cpu.mean);

    std::printf(
        "[PC_RMS_RESULT] correctness=%s delta_ms=%.6f speedup=%.6fx"
        " verdict=%s\n",
        correct ? "PASS" : "FAIL",
        saved, speedup,
        correct && saved > 0.0 ? "CPU_OVERLAP_FASTER" :
        correct ? "GPU_SERIAL_FASTER_OR_EQUAL" : "INVALID_CORRECTNESS");

    ggml_backend_event_free(r_ready);
    ggml_backend_event_free(d2h_done);
    ggml_backend_buffer_free(host_r_buffer);
    ggml_backend_buffer_free(host_x_buffer);
    ggml_backend_buffer_free(scale_buffer);
    ggml_backend_buffer_free(main_buffer);
    ggml_free(scale_ctx);
    ggml_free(ctx);
    ggml_backend_free(copy_backend);
    ggml_backend_free(compute_backend);
    ggml_quantize_free();

    return correct ? 0 : 30;
}
