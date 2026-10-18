#include "llama.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif

// Real-weight companion to test-pc-attn-rms-overlap.
//
// This intentionally benchmarks only the Attention entry:
//   baseline: RMSNorm(x) * gamma -> real Wq/Wk/Wv
//   split:    CPU RMS(x) || GPU (gamma*x -> real Wq/Wk/Wv)
//
// It does NOT claim to be a full-layer benchmark. The purpose is to answer a
// narrower question with actual GGUF weights from several consecutive layers:
// can the CPU helper branch stay under the real GPU QKV branch, and what is the
// block-level upper bound if the final token scale is fused into downstream
// Q/K norm and V consumption rather than emitted as an extra scale kernel?

using clock_type = std::chrono::steady_clock;

static double elapsed_ms(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct options {
    std::string model_path;
    int device = 0;
    int start_layer = 0;
    int layers = 4;
    int64_t tokens = 128;
    int warmup = 5;
    int iters = 50;
    int cpu_threads = 0;
    int copy_prime = 16;
};

static void usage(const char * argv0) {
    std::printf(
        "usage: %s -m model.gguf [options]\n"
        "  -m, --model PATH     GGUF model path\n"
        "  --device N           CUDA device (default: 0)\n"
        "  --start-layer N      first real model layer (default: 0)\n"
        "  --layers N           number of consecutive layers (default: 4)\n"
        "  --tokens N           activation tokens (default: 128)\n"
        "  --warmup N           warmup block iterations (default: 5)\n"
        "  --iters N            measured block iterations per phase (default: 50)\n"
        "  --cpu-threads N      CPU RMS threads (default: min(8, HW threads))\n"
        "  --copy-prime N       D2H/H2D prime iterations before fused-only phase (default: 16)\n",
        argv0);
}

static bool parse_i32(const char * s, int & v) {
    char * end = nullptr;
    const long x = std::strtol(s, &end, 10);
    if (!end || *end != '\0') {
        return false;
    }
    v = (int) x;
    return true;
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
        } else if (a == "-m" || a == "--model") {
            const char * s = need("--model");
            if (!s) return false;
            opt.model_path = s;
        } else if (a == "--device") {
            const char * s = need("--device");
            if (!s || !parse_i32(s, opt.device)) return false;
        } else if (a == "--start-layer") {
            const char * s = need("--start-layer");
            if (!s || !parse_i32(s, opt.start_layer)) return false;
        } else if (a == "--layers") {
            const char * s = need("--layers");
            if (!s || !parse_i32(s, opt.layers)) return false;
        } else if (a == "--tokens") {
            const char * s = need("--tokens");
            if (!s || !parse_i64(s, opt.tokens)) return false;
        } else if (a == "--warmup") {
            const char * s = need("--warmup");
            if (!s || !parse_i32(s, opt.warmup)) return false;
        } else if (a == "--iters") {
            const char * s = need("--iters");
            if (!s || !parse_i32(s, opt.iters)) return false;
        } else if (a == "--cpu-threads") {
            const char * s = need("--cpu-threads");
            if (!s || !parse_i32(s, opt.cpu_threads)) return false;
        } else if (a == "--copy-prime") {
            const char * s = need("--copy-prime");
            if (!s || !parse_i32(s, opt.copy_prime)) return false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return false;
        }
    }

    if (opt.model_path.empty() || opt.device < 0 || opt.start_layer < 0 ||
        opt.layers <= 0 || opt.tokens <= 0 || opt.warmup < 0 ||
        opt.iters <= 0 || opt.cpu_threads < 0 || opt.copy_prime < 0) {
        return false;
    }

    if (opt.cpu_threads == 0) {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        opt.cpu_threads = std::min<int>(8, std::min<int>((int) hw, (int) opt.tokens));
    } else {
        opt.cpu_threads = std::min<int>(opt.cpu_threads, (int) opt.tokens);
    }

    return true;
}

static float sum_sq_fast(const float * row, int64_t n) {
#if defined(__AVX2__) || defined(_M_AVX2)
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256 x0 = _mm256_loadu_ps(row + i);
        const __m256 x1 = _mm256_loadu_ps(row + i + 8);
#if defined(__FMA__) || defined(_M_FMA)
        s0 = _mm256_fmadd_ps(x0, x0, s0);
        s1 = _mm256_fmadd_ps(x1, x1, s1);
#else
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, x0));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(x1, x1));
#endif
    }
    const __m256 s = _mm256_add_ps(s0, s1);
    alignas(32) float lane[8];
    _mm256_store_ps(lane, s);
    float total = lane[0] + lane[1] + lane[2] + lane[3] +
                  lane[4] + lane[5] + lane[6] + lane[7];
    for (; i < n; ++i) {
        total += row[i] * row[i];
    }
    return total;
#else
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += row[i + 0] * row[i + 0];
        s1 += row[i + 1] * row[i + 1];
        s2 += row[i + 2] * row[i + 2];
        s3 += row[i + 3] * row[i + 3];
    }
    float total = s0 + s1 + s2 + s3;
    for (; i < n; ++i) {
        total += row[i] * row[i];
    }
    return total;
#endif
}

class rms_pool {
public:
    explicit rms_pool(int n_threads) : n_threads_(std::max(1, n_threads)) {
        for (int tid = 1; tid < n_threads_; ++tid) {
            workers_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    ~rms_pool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (auto & t : workers_) {
            t.join();
        }
    }

    void run(const float * x, float * inv_r, int64_t n_embd, int64_t tokens, float eps) {
        if (n_threads_ == 1) {
            compute_range(0, x, inv_r, n_embd, tokens, eps);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            x_ = x;
            inv_r_ = inv_r;
            n_embd_ = n_embd;
            tokens_ = tokens;
            eps_ = eps;
            done_ = 0;
            ++generation_;
        }
        start_cv_.notify_all();

        compute_range(0, x, inv_r, n_embd, tokens, eps);

        std::unique_lock<std::mutex> lock(mu_);
        done_cv_.wait(lock, [&]() { return done_ == n_threads_ - 1; });
    }

    int n_threads() const { return n_threads_; }

private:
    void compute_range(
            int tid,
            const float * x,
            float * inv_r,
            int64_t n_embd,
            int64_t tokens,
            float eps) const {
        const int64_t t0 = tokens * tid / n_threads_;
        const int64_t t1 = tokens * (tid + 1) / n_threads_;
        for (int64_t t = t0; t < t1; ++t) {
            const float ss = sum_sq_fast(x + t * n_embd, n_embd);
            inv_r[t] = 1.0f / std::sqrt(ss / (float) n_embd + eps);
        }
    }

    void worker_loop(int tid) {
        uint64_t seen = 0;
        for (;;) {
            const float * x = nullptr;
            float * inv_r = nullptr;
            int64_t n_embd = 0;
            int64_t tokens = 0;
            float eps = 0.0f;

            {
                std::unique_lock<std::mutex> lock(mu_);
                start_cv_.wait(lock, [&]() { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                x = x_;
                inv_r = inv_r_;
                n_embd = n_embd_;
                tokens = tokens_;
                eps = eps_;
            }

            compute_range(tid, x, inv_r, n_embd, tokens, eps);

            {
                std::lock_guard<std::mutex> lock(mu_);
                ++done_;
            }
            done_cv_.notify_one();
        }
    }

    int n_threads_;
    std::vector<std::thread> workers_;

    std::mutex mu_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    bool stop_ = false;
    uint64_t generation_ = 0;
    int done_ = 0;

    const float * x_ = nullptr;
    float * inv_r_ = nullptr;
    int64_t n_embd_ = 0;
    int64_t tokens_ = 0;
    float eps_ = 0.0f;
};

static const char * cpu_kernel_name() {
#if defined(__AVX2__) || defined(_M_AVX2)
#if defined(__FMA__) || defined(_M_FMA)
    return "AVX2_FMA";
#else
    return "AVX2";
#endif
#else
    return "SCALAR_UNROLLED";
#endif
}

struct stats {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

static stats summarize(std::vector<double> v) {
    stats s;
    if (v.empty()) return s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    std::sort(v.begin(), v.end());

    auto pct = [&](double p) {
        const double pos = p * (double) (v.size() - 1);
        const size_t lo = (size_t) std::floor(pos);
        const size_t hi = (size_t) std::ceil(pos);
        const double f = pos - (double) lo;
        return v[lo] * (1.0 - f) + v[hi] * f;
    };
    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    s.p99 = pct(0.99);
    return s;
}

struct diff_stats {
    double max_abs = 0.0;
    double peak_ref = 0.0;
    double cosine = 0.0;
    bool finite = true;
};

static diff_stats compare_vec(const std::vector<float> & a, const std::vector<float> & b) {
    diff_stats s;
    if (a.size() != b.size() || a.empty()) {
        s.finite = false;
        return s;
    }

    long double dot = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = a[i];
        const double y = b[i];
        if (!std::isfinite(x) || !std::isfinite(y)) s.finite = false;
        s.max_abs = std::max(s.max_abs, std::abs(x - y));
        s.peak_ref = std::max(s.peak_ref, std::abs(x));
        dot += (long double) x * y;
        aa += (long double) x * x;
        bb += (long double) y * y;
    }
    if (aa > 0.0 && bb > 0.0) {
        s.cosine = (double) (dot / std::sqrt(aa * bb));
    }
    return s;
}

static bool close_enough(const diff_stats & s) {
    return s.finite && s.cosine >= 0.9999 &&
           s.max_abs <= 0.02 * std::max(1.0, s.peak_ref);
}

static bool copy_tensor_raw(const ggml_tensor * src, ggml_tensor * dst) {
    if (src == nullptr || dst == nullptr ||
        src->type != dst->type ||
        ggml_nbytes(src) != ggml_nbytes(dst)) {
        return false;
    }

    std::vector<uint8_t> bytes(ggml_nbytes(src));
    ggml_backend_tensor_get(src, bytes.data(), 0, bytes.size());
    ggml_backend_tensor_set(dst, bytes.data(), 0, bytes.size());
    return true;
}

struct layer_case {
    int il = -1;
    int64_t n_embd = 0;
    int64_t q_dim = 0;
    int64_t k_dim = 0;
    int64_t v_dim = 0;
    float eps = 0.0f;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ggml_tensor * x = nullptr;
    ggml_tensor * gamma = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;

    ggml_tensor * q_ref = nullptr;
    ggml_tensor * k_ref = nullptr;
    ggml_tensor * v_ref = nullptr;
    ggml_cgraph * graph_serial = nullptr;

    ggml_tensor * q_num = nullptr;
    ggml_tensor * k_num = nullptr;
    ggml_tensor * v_num = nullptr;
    ggml_cgraph * graph_num = nullptr;

    ggml_context * scale_ctx = nullptr;
    ggml_backend_buffer_t scale_buffer = nullptr;
    ggml_tensor * inv_r = nullptr;
    ggml_tensor * q_split = nullptr;
    ggml_tensor * k_split = nullptr;
    ggml_tensor * v_split = nullptr;
    ggml_cgraph * graph_scale = nullptr;
};

static bool make_layer_case(
        layer_case & out,
        const llama_model * model,
        int il,
        int64_t tokens,
        int device,
        std::mt19937 & rng) {
    if (il < 0 || il >= (int) model->layers.size()) return false;
    const llama_layer & src = model->layers[il];
    if (!src.attn_norm || !src.wq || !src.wk || !src.wv) return false;
    if (src.wqkv != nullptr) {
        std::fprintf(stderr, "layer %d uses fused WQKV; this probe currently expects separate Wq/Wk/Wv\n", il);
        return false;
    }

    out.il = il;
    out.n_embd = src.wq->ne[0];
    out.q_dim = src.wq->ne[1];
    out.k_dim = src.wk->ne[1];
    out.v_dim = src.wv->ne[1];
    out.eps = model->hparams.f_norm_rms_eps;

    if (src.wk->ne[0] != out.n_embd || src.wv->ne[0] != out.n_embd ||
        src.attn_norm->ne[0] != out.n_embd) {
        return false;
    }

    ggml_init_params p = {};
    p.mem_size = 32 * 1024 * 1024;
    p.no_alloc = true;
    out.ctx = ggml_init(p);
    if (!out.ctx) return false;

    out.x = ggml_new_tensor_2d(out.ctx, GGML_TYPE_F32, out.n_embd, tokens);
    out.gamma = ggml_new_tensor_1d(out.ctx, src.attn_norm->type, out.n_embd);
    out.wq = ggml_new_tensor_2d(out.ctx, src.wq->type, src.wq->ne[0], src.wq->ne[1]);
    out.wk = ggml_new_tensor_2d(out.ctx, src.wk->type, src.wk->ne[0], src.wk->ne[1]);
    out.wv = ggml_new_tensor_2d(out.ctx, src.wv->type, src.wv->ne[0], src.wv->ne[1]);

    ggml_set_name(out.x, "real_rms_x");
    ggml_set_name(out.gamma, "real_rms_gamma");
    ggml_set_name(out.wq, "real_rms_wq");
    ggml_set_name(out.wk, "real_rms_wk");
    ggml_set_name(out.wv, "real_rms_wv");

    ggml_tensor * norm = ggml_rms_norm(out.ctx, out.x, out.eps);
    ggml_tensor * norm_gamma = ggml_mul(out.ctx, norm, out.gamma);
    out.q_ref = ggml_mul_mat(out.ctx, out.wq, norm_gamma);
    out.k_ref = ggml_mul_mat(out.ctx, out.wk, norm_gamma);
    out.v_ref = ggml_mul_mat(out.ctx, out.wv, norm_gamma);

    out.graph_serial = ggml_new_graph_custom(out.ctx, 64, false);
    ggml_build_forward_expand(out.graph_serial, out.q_ref);
    ggml_build_forward_expand(out.graph_serial, out.k_ref);
    ggml_build_forward_expand(out.graph_serial, out.v_ref);

    ggml_tensor * x_gamma = ggml_mul(out.ctx, out.x, out.gamma);
    out.q_num = ggml_mul_mat(out.ctx, out.wq, x_gamma);
    out.k_num = ggml_mul_mat(out.ctx, out.wk, x_gamma);
    out.v_num = ggml_mul_mat(out.ctx, out.wv, x_gamma);

    out.graph_num = ggml_new_graph_custom(out.ctx, 64, false);
    ggml_build_forward_expand(out.graph_num, out.q_num);
    ggml_build_forward_expand(out.graph_num, out.k_num);
    ggml_build_forward_expand(out.graph_num, out.v_num);

    out.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        out.ctx, ggml_backend_cuda_buffer_type(device));
    if (!out.buffer) return false;

    if (!copy_tensor_raw(src.attn_norm, out.gamma) ||
        !copy_tensor_raw(src.wq, out.wq) ||
        !copy_tensor_raw(src.wk, out.wk) ||
        !copy_tensor_raw(src.wv, out.wv)) {
        return false;
    }

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> xh((size_t) out.n_embd * (size_t) tokens);
    for (float & v : xh) v = dist(rng);
    ggml_backend_tensor_set(out.x, xh.data(), 0, xh.size() * sizeof(float));

    ggml_init_params sp = {};
    sp.mem_size = 16 * 1024 * 1024;
    sp.no_alloc = true;
    out.scale_ctx = ggml_init(sp);
    if (!out.scale_ctx) return false;

    ggml_tensor * q_leaf = ggml_new_tensor_2d(out.scale_ctx, GGML_TYPE_F32, out.q_dim, tokens);
    ggml_tensor * k_leaf = ggml_new_tensor_2d(out.scale_ctx, GGML_TYPE_F32, out.k_dim, tokens);
    ggml_tensor * v_leaf = ggml_new_tensor_2d(out.scale_ctx, GGML_TYPE_F32, out.v_dim, tokens);
    out.inv_r = ggml_new_tensor_2d(out.scale_ctx, GGML_TYPE_F32, 1, tokens);

    if (ggml_backend_tensor_alloc(out.q_num->buffer, q_leaf, out.q_num->data) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(out.k_num->buffer, k_leaf, out.k_num->data) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(out.v_num->buffer, v_leaf, out.v_num->data) != GGML_STATUS_SUCCESS) {
        return false;
    }

    out.q_split = ggml_mul(out.scale_ctx, q_leaf, out.inv_r);
    out.k_split = ggml_mul(out.scale_ctx, k_leaf, out.inv_r);
    out.v_split = ggml_mul(out.scale_ctx, v_leaf, out.inv_r);

    out.graph_scale = ggml_new_graph_custom(out.scale_ctx, 32, false);
    ggml_build_forward_expand(out.graph_scale, out.q_split);
    ggml_build_forward_expand(out.graph_scale, out.k_split);
    ggml_build_forward_expand(out.graph_scale, out.v_split);

    out.scale_buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        out.scale_ctx, ggml_backend_cuda_buffer_type(device));
    return out.scale_buffer != nullptr;
}

static void free_layer_case(layer_case & c) {
    if (c.scale_buffer) ggml_backend_buffer_free(c.scale_buffer);
    if (c.buffer) ggml_backend_buffer_free(c.buffer);
    if (c.scale_ctx) ggml_free(c.scale_ctx);
    if (c.ctx) ggml_free(c.ctx);
    c = {};
}

struct fused_sample {
    double wall_ms = 0.0;
    double d2h_ms = 0.0;
    double cpu_ms = 0.0;
    double join_tail_ms = 0.0;
};

int main(int argc, char ** argv) {
    options opt;
    if (!parse_options(argc, argv, opt)) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = true;

    std::printf("[PC_RMS_REAL_LOAD] model=%s mode=CPU_MMAP\n", opt.model_path.c_str());
    llama_model * model = llama_model_load_from_file(opt.model_path.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", opt.model_path.c_str());
        llama_backend_free();
        return 2;
    }

    if (model->arch != LLM_ARCH_QWEN3MOE) {
        // llama_model::arch_name() is an internal, non-exported C++ method
        // when llama is built as a Windows DLL. Keep this standalone probe
        // compatible with shared builds by reading the already-visible enum.
        std::fprintf(stderr, "expected Qwen3-MoE, got arch enum=%d\n", (int) model->arch);
        llama_model_free(model);
        llama_backend_free();
        return 3;
    }

    const int n_model_layers = (int) model->layers.size();
    if (opt.start_layer >= n_model_layers ||
        opt.start_layer + opt.layers > n_model_layers) {
        std::fprintf(stderr,
            "invalid layer range [%d,%d), model has %d layers\n",
            opt.start_layer, opt.start_layer + opt.layers, n_model_layers);
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    const int n_cuda = ggml_backend_cuda_get_device_count();
    if (opt.device >= n_cuda) {
        std::fprintf(stderr, "CUDA device %d unavailable, count=%d\n", opt.device, n_cuda);
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    char dev_desc[256] = {};
    ggml_backend_cuda_get_device_description(opt.device, dev_desc, sizeof(dev_desc));

    ggml_backend_t compute_backend = ggml_backend_cuda_init(opt.device);
    ggml_backend_t copy_backend = ggml_backend_cuda_init(opt.device);
    if (!compute_backend || !copy_backend) {
        std::fprintf(stderr, "failed to initialize CUDA benchmark backends\n");
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    std::mt19937 rng(1234);
    std::vector<layer_case> cases((size_t) opt.layers);
    for (int i = 0; i < opt.layers; ++i) {
        const int il = opt.start_layer + i;
        if (!make_layer_case(cases[(size_t) i], model, il, opt.tokens, opt.device, rng)) {
            std::fprintf(stderr, "failed to construct real-weight probe for layer %d\n", il);
            for (auto & c : cases) free_layer_case(c);
            ggml_backend_free(copy_backend);
            ggml_backend_free(compute_backend);
            llama_model_free(model);
            llama_backend_free();
            return 7;
        }

        const auto & c = cases[(size_t) i];
        std::printf(
            "[PC_RMS_REAL_LAYER] layer=%d embd=%" PRId64
            " q=%" PRId64 " k=%" PRId64 " v=%" PRId64
            " gamma=%s wq=%s wk=%s wv=%s weight_bytes=%zu\n",
            il, c.n_embd, c.q_dim, c.k_dim, c.v_dim,
            ggml_type_name(c.gamma->type),
            ggml_type_name(c.wq->type),
            ggml_type_name(c.wk->type),
            ggml_type_name(c.wv->type),
            ggml_nbytes(c.gamma) + ggml_nbytes(c.wq) +
            ggml_nbytes(c.wk) + ggml_nbytes(c.wv));
    }

    const int64_t n_embd = cases[0].n_embd;
    for (const auto & c : cases) {
        if (c.n_embd != n_embd) {
            std::fprintf(stderr, "probe layers do not share hidden width\n");
            return 8;
        }
    }

    const size_t x_bytes = (size_t) n_embd * (size_t) opt.tokens * sizeof(float);
    const size_t r_bytes = (size_t) opt.tokens * sizeof(float);

    ggml_backend_buffer_type_t host_buft = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_t host_x_buffer = ggml_backend_buft_alloc_buffer(host_buft, x_bytes);
    ggml_backend_buffer_t host_r_buffer = ggml_backend_buft_alloc_buffer(host_buft, r_bytes);
    if (!host_x_buffer || !host_r_buffer) {
        std::fprintf(stderr, "failed to allocate pinned host buffers\n");
        return 9;
    }
    float * host_x = (float *) ggml_backend_buffer_get_base(host_x_buffer);
    float * host_r = (float *) ggml_backend_buffer_get_base(host_r_buffer);

    rms_pool cpu_pool(opt.cpu_threads);

    ggml_backend_event_t d2h_done = ggml_backend_event_new(ggml_backend_get_device(copy_backend));
    ggml_backend_event_t r_ready = ggml_backend_event_new(ggml_backend_get_device(copy_backend));
    if (!d2h_done || !r_ready) {
        std::fprintf(stderr, "CUDA events unavailable\n");
        return 10;
    }

    auto run_serial_layer = [&](layer_case & c) -> double {
        const auto t0 = clock_type::now();
        const ggml_status st = ggml_backend_graph_compute_async(compute_backend, c.graph_serial);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "serial layer %d failed: %d\n", c.il, (int) st);
            std::exit(20);
        }
        ggml_backend_synchronize(compute_backend);
        return elapsed_ms(t0, clock_type::now());
    };

    auto run_fused_layer = [&](layer_case & c) -> fused_sample {
        fused_sample s;
        const auto t0 = clock_type::now();

        ggml_backend_tensor_get_async(copy_backend, c.x, host_x, 0, x_bytes);
        ggml_backend_event_record(d2h_done, copy_backend);

        const ggml_status st = ggml_backend_graph_compute_async(compute_backend, c.graph_num);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "numerator layer %d failed: %d\n", c.il, (int) st);
            std::exit(21);
        }

        ggml_backend_event_synchronize(d2h_done);
        const auto td2h = clock_type::now();
        s.d2h_ms = elapsed_ms(t0, td2h);

        const auto tc0 = clock_type::now();
        cpu_pool.run(host_x, host_r, c.n_embd, opt.tokens, c.eps);
        const auto tc1 = clock_type::now();
        s.cpu_ms = elapsed_ms(tc0, tc1);

        ggml_backend_tensor_set_async(copy_backend, c.inv_r, host_r, 0, r_bytes);
        ggml_backend_event_record(r_ready, copy_backend);
        ggml_backend_event_wait(compute_backend, r_ready);
        ggml_backend_synchronize(compute_backend);

        const auto tend = clock_type::now();
        s.join_tail_ms = elapsed_ms(tc1, tend);
        s.wall_ms = elapsed_ms(t0, tend);
        return s;
    };

    auto run_explicit_split_layer = [&](layer_case & c) {
        (void) run_fused_layer(c);
        const ggml_status st = ggml_backend_graph_compute_async(compute_backend, c.graph_scale);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "scale layer %d failed: %d\n", c.il, (int) st);
            std::exit(22);
        }
        ggml_backend_synchronize(compute_backend);
    };

    auto prime_copy_path = [&](int reps) {
        if (reps <= 0) {
            std::printf("[PC_RMS_REAL_COPY_PRIME] reps=0 status=SKIP\n");
            return;
        }

        std::fill(host_r, host_r + opt.tokens, 1.0f);
        std::vector<double> d2h_ms;
        std::vector<double> h2d_ms;
        d2h_ms.reserve((size_t) reps);
        h2d_ms.reserve((size_t) reps);

        for (int i = 0; i < reps; ++i) {
            const auto d0 = clock_type::now();
            ggml_backend_tensor_get_async(
                copy_backend, cases[0].x, host_x, 0, x_bytes);
            ggml_backend_event_record(d2h_done, copy_backend);
            ggml_backend_event_synchronize(d2h_done);
            d2h_ms.push_back(elapsed_ms(d0, clock_type::now()));

            const auto h0 = clock_type::now();
            ggml_backend_tensor_set_async(
                copy_backend, cases[0].inv_r, host_r, 0, r_bytes);
            ggml_backend_event_record(r_ready, copy_backend);
            ggml_backend_event_synchronize(r_ready);
            h2d_ms.push_back(elapsed_ms(h0, clock_type::now()));
        }

        const stats d = summarize(d2h_ms);
        const stats h = summarize(h2d_ms);
        std::printf(
            "[PC_RMS_REAL_COPY_PRIME] reps=%d"
            " d2h_mean_ms=%.6f d2h_p50_ms=%.6f d2h_p95_ms=%.6f d2h_p99_ms=%.6f d2h_max_ms=%.6f"
            " h2d_mean_ms=%.6f h2d_p50_ms=%.6f h2d_p95_ms=%.6f h2d_p99_ms=%.6f h2d_max_ms=%.6f\n",
            reps,
            d.mean, d.p50, d.p95, d.p99, d.max,
            h.mean, h.p50, h.p95, h.p99, h.max);
    };

    std::printf(
        "[PC_RMS_REAL_CONFIG] device=%d device_name=\"%s\""
        " layers=[%d,%d) tokens=%" PRId64
        " warmup=%d iters=%d cpu_threads=%d cpu_kernel=%s copy_prime=%d"
        " x_bytes=%zu return_bytes=%zu eps=%.9g\n",
        opt.device, dev_desc,
        opt.start_layer, opt.start_layer + opt.layers, opt.tokens,
        opt.warmup, opt.iters, cpu_pool.n_threads(), cpu_kernel_name(), opt.copy_prime,
        x_bytes, r_bytes, cases[0].eps);

    // Real-weight correctness of the algebraic split. The timed fused mode
    // intentionally stops before the explicit scale, but correctness is
    // checked once with that scale restored.
    bool all_correct = true;
    for (auto & c : cases) {
        (void) run_serial_layer(c);
        std::vector<float> qr((size_t) ggml_nelements(c.q_ref));
        std::vector<float> kr((size_t) ggml_nelements(c.k_ref));
        std::vector<float> vr((size_t) ggml_nelements(c.v_ref));
        ggml_backend_tensor_get(c.q_ref, qr.data(), 0, qr.size() * sizeof(float));
        ggml_backend_tensor_get(c.k_ref, kr.data(), 0, kr.size() * sizeof(float));
        ggml_backend_tensor_get(c.v_ref, vr.data(), 0, vr.size() * sizeof(float));

        run_explicit_split_layer(c);
        std::vector<float> qs((size_t) ggml_nelements(c.q_split));
        std::vector<float> ks((size_t) ggml_nelements(c.k_split));
        std::vector<float> vs((size_t) ggml_nelements(c.v_split));
        ggml_backend_tensor_get(c.q_split, qs.data(), 0, qs.size() * sizeof(float));
        ggml_backend_tensor_get(c.k_split, ks.data(), 0, ks.size() * sizeof(float));
        ggml_backend_tensor_get(c.v_split, vs.data(), 0, vs.size() * sizeof(float));

        const diff_stats dq = compare_vec(qr, qs);
        const diff_stats dk = compare_vec(kr, ks);
        const diff_stats dv = compare_vec(vr, vs);
        const bool ok = close_enough(dq) && close_enough(dk) && close_enough(dv);
        all_correct = all_correct && ok;

        std::printf(
            "[PC_RMS_REAL_CORRECTNESS] layer=%d status=%s"
            " q_max_abs=%.8g q_cos=%.9f"
            " k_max_abs=%.8g k_cos=%.9f"
            " v_max_abs=%.8g v_cos=%.9f\n",
            c.il, ok ? "PASS" : "FAIL",
            dq.max_abs, dq.cosine,
            dk.max_abs, dk.cosine,
            dv.max_abs, dv.cosine);
    }

    std::vector<std::vector<double>> layer_serial((size_t) opt.layers);
    std::vector<std::vector<double>> layer_fused((size_t) opt.layers);
    std::vector<std::vector<double>> layer_d2h((size_t) opt.layers);
    std::vector<std::vector<double>> layer_cpu((size_t) opt.layers);
    std::vector<std::vector<double>> layer_join_tail((size_t) opt.layers);

    std::vector<double> block_serial_pre;
    std::vector<double> block_serial_post;
    std::vector<double> block_serial_all;
    std::vector<double> block_fused;
    block_serial_pre.reserve(opt.iters);
    block_serial_post.reserve(opt.iters);
    block_serial_all.reserve((size_t) opt.iters * 2);
    block_fused.reserve(opt.iters);

    // Phase A: baseline only. Do not touch the copy stream while measuring.
    for (int w = 0; w < opt.warmup; ++w) {
        for (auto & lc : cases) (void) run_serial_layer(lc);
    }
    for (int it = 0; it < opt.iters; ++it) {
        const auto b0 = clock_type::now();
        for (int i = 0; i < opt.layers; ++i) {
            layer_serial[(size_t) i].push_back(
                run_serial_layer(cases[(size_t) i]));
        }
        const double wall = elapsed_ms(b0, clock_type::now());
        block_serial_pre.push_back(wall);
        block_serial_all.push_back(wall);
    }

    // Prime the copy stream immediately before the fused-only phase. This
    // distinguishes steady-state transfer cost from copy-engine/stream wakeup.
    prime_copy_path(opt.copy_prime);

    // Phase B: fused-only continuous blocks. Layer 0 now follows layer N-1
    // from the previous block instead of following a GPU-serial block.
    for (int w = 0; w < opt.warmup; ++w) {
        for (auto & lc : cases) (void) run_fused_layer(lc);
    }
    for (int it = 0; it < opt.iters; ++it) {
        const auto b0 = clock_type::now();
        for (int i = 0; i < opt.layers; ++i) {
            const fused_sample s =
                run_fused_layer(cases[(size_t) i]);
            layer_fused[(size_t) i].push_back(s.wall_ms);
            layer_d2h[(size_t) i].push_back(s.d2h_ms);
            layer_cpu[(size_t) i].push_back(s.cpu_ms);
            layer_join_tail[(size_t) i].push_back(s.join_tail_ms);
        }
        block_fused.push_back(elapsed_ms(b0, clock_type::now()));
    }

    // Phase C: baseline recheck to expose GPU clock / thermal drift rather
    // than accidentally crediting that drift to the fused helper path.
    for (int w = 0; w < opt.warmup; ++w) {
        for (auto & lc : cases) (void) run_serial_layer(lc);
    }
    for (int it = 0; it < opt.iters; ++it) {
        const auto b0 = clock_type::now();
        for (int i = 0; i < opt.layers; ++i) {
            layer_serial[(size_t) i].push_back(
                run_serial_layer(cases[(size_t) i]));
        }
        const double wall = elapsed_ms(b0, clock_type::now());
        block_serial_post.push_back(wall);
        block_serial_all.push_back(wall);
    }

    for (int i = 0; i < opt.layers; ++i) {
        const stats a = summarize(layer_serial[(size_t) i]);
        const stats b = summarize(layer_fused[(size_t) i]);
        const stats d = summarize(layer_d2h[(size_t) i]);
        const stats cp = summarize(layer_cpu[(size_t) i]);
        const stats j = summarize(layer_join_tail[(size_t) i]);

        std::printf(
            "[PC_RMS_REAL_PER_LAYER] layer=%d"
            " gpu_serial_mean_ms=%.6f gpu_serial_p50_ms=%.6f gpu_serial_p95_ms=%.6f"
            " fused_mean_ms=%.6f fused_p50_ms=%.6f fused_p95_ms=%.6f fused_p99_ms=%.6f fused_max_ms=%.6f"
            " d2h_mean_ms=%.6f d2h_p50_ms=%.6f d2h_p95_ms=%.6f d2h_p99_ms=%.6f d2h_max_ms=%.6f"
            " cpu_mean_ms=%.6f cpu_p50_ms=%.6f cpu_p95_ms=%.6f cpu_p99_ms=%.6f cpu_max_ms=%.6f"
            " join_tail_mean_ms=%.6f join_tail_p50_ms=%.6f join_tail_p95_ms=%.6f join_tail_p99_ms=%.6f join_tail_max_ms=%.6f"
            " delta_ms=%.6f speedup=%.6fx\n",
            cases[(size_t) i].il,
            a.mean, a.p50, a.p95,
            b.mean, b.p50, b.p95, b.p99, b.max,
            d.mean, d.p50, d.p95, d.p99, d.max,
            cp.mean, cp.p50, cp.p95, cp.p99, cp.max,
            j.mean, j.p50, j.p95, j.p99, j.max,
            a.mean - b.mean,
            b.mean > 0.0 ? a.mean / b.mean : 0.0);
    }

    const stats bs_pre = summarize(block_serial_pre);
    const stats bs_post = summarize(block_serial_post);
    const stats bs = summarize(block_serial_all);
    const stats bf = summarize(block_fused);

    std::printf(
        "[PC_RMS_REAL_BLOCK] mode=GPU_SERIAL_PRE layers=%d"
        " mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f p99_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
        opt.layers, bs_pre.mean, bs_pre.p50, bs_pre.p95,
        bs_pre.p99, bs_pre.min, bs_pre.max);

    std::printf(
        "[PC_RMS_REAL_BLOCK] mode=FUSED_ONLY layers=%d"
        " mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f p99_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
        opt.layers, bf.mean, bf.p50, bf.p95,
        bf.p99, bf.min, bf.max);

    std::printf(
        "[PC_RMS_REAL_BLOCK] mode=GPU_SERIAL_POST layers=%d"
        " mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f p99_ms=%.6f min_ms=%.6f max_ms=%.6f\n",
        opt.layers, bs_post.mean, bs_post.p50, bs_post.p95,
        bs_post.p99, bs_post.min, bs_post.max);

    std::printf(
        "[PC_RMS_REAL_DRIFT]"
        " serial_pre_mean_ms=%.6f serial_post_mean_ms=%.6f"
        " serial_post_over_pre=%.6fx"
        " serial_pre_p50_ms=%.6f serial_post_p50_ms=%.6f\n",
        bs_pre.mean, bs_post.mean,
        bs_pre.mean > 0.0 ? bs_post.mean / bs_pre.mean : 0.0,
        bs_pre.p50, bs_post.p50);

    std::printf(
        "[PC_RMS_REAL_RESULT] correctness=%s"
        " layers=[%d,%d) tokens=%" PRId64
        " baseline_combined_mean_ms=%.6f fused_only_mean_ms=%.6f"
        " delta_ms=%.6f speedup=%.6fx verdict=%s\n",
        all_correct ? "PASS" : "FAIL",
        opt.start_layer, opt.start_layer + opt.layers, opt.tokens,
        bs.mean, bf.mean,
        bs.mean - bf.mean,
        bf.mean > 0.0 ? bs.mean / bf.mean : 0.0,
        all_correct && bf.mean < bs.mean ?
            "REAL_WEIGHT_FUSED_BOUND_FASTER" :
        all_correct ? "REAL_WEIGHT_GPU_SERIAL_FASTER_OR_EQUAL" :
            "INVALID_CORRECTNESS");

    ggml_backend_event_free(r_ready);
    ggml_backend_event_free(d2h_done);
    ggml_backend_buffer_free(host_r_buffer);
    ggml_backend_buffer_free(host_x_buffer);

    for (auto & c : cases) free_layer_case(c);

    ggml_backend_free(copy_backend);
    ggml_backend_free(compute_backend);
    llama_model_free(model);
    llama_backend_free();

    return all_correct ? 0 : 30;
}
