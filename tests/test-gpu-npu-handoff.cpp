#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

static int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

struct stats {
    double mean = 0.0;
    double p50  = 0.0;
    double p95  = 0.0;
    double min  = 0.0;
    double max  = 0.0;
};

static stats summarize(const std::vector<double> & values) {
    stats s;
    if (values.empty()) {
        return s;
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    s.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    s.min  = sorted.front();
    s.max  = sorted.back();

    auto percentile = [&](double q) {
        const double pos = q * (sorted.size() - 1);
        const size_t lo = (size_t) std::floor(pos);
        const size_t hi = (size_t) std::ceil(pos);
        if (lo == hi) {
            return sorted[lo];
        }
        const double frac = pos - lo;
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    };

    s.p50 = percentile(0.50);
    s.p95 = percentile(0.95);
    return s;
}

static double gib_per_s(size_t bytes, double us) {
    if (us <= 0.0) {
        return 0.0;
    }
    return ((double) bytes / (1024.0 * 1024.0 * 1024.0)) / (us / 1.0e6);
}

struct tensor_holder {
    ggml_context * ctx = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    void release() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
            buffer = nullptr;
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
            ctx = nullptr;
            tensor = nullptr;
        }
    }

    ~tensor_holder() {
        release();
    }
};

static bool alloc_tensor_on_backend(tensor_holder & holder, ggml_backend_t backend, int hidden, int chunk, const char * name) {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 4,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    holder.ctx = ggml_init(params);
    if (holder.ctx == nullptr) {
        return false;
    }

    holder.tensor = ggml_new_tensor_2d(holder.ctx, GGML_TYPE_F32, hidden, chunk);
    ggml_set_name(holder.tensor, name);

    holder.buffer = ggml_backend_alloc_ctx_tensors(holder.ctx, backend);
    if (holder.buffer == nullptr) {
        holder.release();
        return false;
    }

    return true;
}

static bool verify_copy(
        const char * direction,
        ggml_backend_t src_backend,
        ggml_backend_t dst_backend,
        ggml_tensor * src,
        ggml_tensor * dst,
        size_t bytes) {
    std::vector<uint8_t> expected(bytes);
    std::vector<uint8_t> actual(bytes);

    for (size_t i = 0; i < bytes; ++i) {
        expected[i] = (uint8_t) ((i * 131u + 17u) & 0xffu);
    }

    ggml_backend_tensor_set(src, expected.data(), 0, bytes);
    ggml_backend_synchronize(src_backend);
    ggml_backend_synchronize(dst_backend);

    ggml_backend_tensor_copy(src, dst);
    ggml_backend_synchronize(dst_backend);
    ggml_backend_tensor_get(dst, actual.data(), 0, bytes);

    const bool ok = std::memcmp(expected.data(), actual.data(), bytes) == 0;
    printf("[HANDOFF_VERIFY] direction=%s bytes=%zu ok=%d\n", direction, bytes, ok ? 1 : 0);
    return ok;
}

static void warmup_sync(
        ggml_backend_t src_backend,
        ggml_backend_t dst_backend,
        ggml_tensor * src,
        ggml_tensor * dst,
        int warmup) {
    for (int i = 0; i < warmup; ++i) {
        ggml_backend_synchronize(src_backend);
        ggml_backend_synchronize(dst_backend);
        ggml_backend_tensor_copy(src, dst);
    }
    ggml_backend_synchronize(dst_backend);
}

static void bench_sync(
        const char * direction,
        ggml_backend_t src_backend,
        ggml_backend_t dst_backend,
        ggml_tensor * src,
        ggml_tensor * dst,
        int chunk,
        size_t bytes,
        int reps,
        int warmup) {
    warmup_sync(src_backend, dst_backend, src, dst, warmup);

    std::vector<double> src_sync_us;
    std::vector<double> dst_sync_us;
    std::vector<double> copy_us;
    std::vector<double> total_us;
    src_sync_us.reserve(reps);
    dst_sync_us.reserve(reps);
    copy_us.reserve(reps);
    total_us.reserve(reps);

    for (int i = 0; i < reps; ++i) {
        const int64_t t0 = now_us();

        const int64_t s0 = now_us();
        ggml_backend_synchronize(src_backend);
        const int64_t s1 = now_us();

        const int64_t d0 = now_us();
        ggml_backend_synchronize(dst_backend);
        const int64_t d1 = now_us();

        const int64_t c0 = now_us();
        ggml_backend_tensor_copy(src, dst);
        const int64_t c1 = now_us();

        src_sync_us.push_back((double) (s1 - s0));
        dst_sync_us.push_back((double) (d1 - d0));
        copy_us.push_back((double) (c1 - c0));
        total_us.push_back((double) (c1 - t0));
    }

    ggml_backend_synchronize(dst_backend);

    const stats ss = summarize(src_sync_us);
    const stats ds = summarize(dst_sync_us);
    const stats cs = summarize(copy_us);
    const stats ts = summarize(total_us);

    printf("[HANDOFF_SYNC] direction=%s chunk=%d bytes=%zu reps=%d "
           "src_sync_p50_us=%.3f dst_sync_p50_us=%.3f "
           "copy_p50_us=%.3f copy_p95_us=%.3f "
           "total_p50_us=%.3f total_p95_us=%.3f "
           "copy_gib_s=%.3f total_gib_s=%.3f\n",
           direction, chunk, bytes, reps,
           ss.p50, ds.p50,
           cs.p50, cs.p95,
           ts.p50, ts.p95,
           gib_per_s(bytes, cs.p50), gib_per_s(bytes, ts.p50));
}

static void bench_async_api(
        const char * direction,
        ggml_backend_t src_backend,
        ggml_backend_t dst_backend,
        ggml_tensor * src,
        ggml_tensor * dst,
        int chunk,
        size_t bytes,
        int reps,
        int warmup) {
    for (int i = 0; i < warmup; ++i) {
        ggml_backend_tensor_copy_async(src_backend, dst_backend, src, dst);
        ggml_backend_synchronize(dst_backend);
    }

    std::vector<double> call_us;
    std::vector<double> post_sync_us;
    std::vector<double> total_us;
    call_us.reserve(reps);
    post_sync_us.reserve(reps);
    total_us.reserve(reps);

    for (int i = 0; i < reps; ++i) {
        const int64_t t0 = now_us();
        ggml_backend_tensor_copy_async(src_backend, dst_backend, src, dst);
        const int64_t t1 = now_us();
        ggml_backend_synchronize(dst_backend);
        const int64_t t2 = now_us();

        call_us.push_back((double) (t1 - t0));
        post_sync_us.push_back((double) (t2 - t1));
        total_us.push_back((double) (t2 - t0));
    }

    const stats cs = summarize(call_us);
    const stats ss = summarize(post_sync_us);
    const stats ts = summarize(total_us);

    printf("[HANDOFF_ASYNC_API] direction=%s chunk=%d bytes=%zu reps=%d "
           "call_p50_us=%.3f call_p95_us=%.3f "
           "post_sync_p50_us=%.3f post_sync_p95_us=%.3f "
           "total_p50_us=%.3f total_p95_us=%.3f "
           "total_gib_s=%.3f\n",
           direction, chunk, bytes, reps,
           cs.p50, cs.p95,
           ss.p50, ss.p95,
           ts.p50, ts.p95,
           gib_per_s(bytes, ts.p50));
}

static void print_devices() {
    fprintf(stderr, "Available devices:\n");
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        fprintf(stderr, "  %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    }
}

int main(int argc, char ** argv) {
    const char * gpu_name = "GPUOpenCL";
    const char * npu_name = "HTP0";
    int reps = 128;
    int warmup = 8;
    int hidden = 2048;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            gpu_name = argv[++i];
        } else if (strcmp(argv[i], "--npu") == 0 && i + 1 < argc) {
            npu_name = argv[++i];
        } else if (strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            hidden = atoi(argv[++i]);
        } else {
            fprintf(stderr, "Usage: %s [--gpu GPUOpenCL] [--npu HTP0] [--reps N] [--warmup N] [--hidden 2048]\n", argv[0]);
            return 2;
        }
    }

    if (reps <= 0 || warmup < 0 || hidden <= 0) {
        fprintf(stderr, "Invalid benchmark arguments\n");
        return 2;
    }

    ggml_backend_load_all();

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_name(gpu_name);
    ggml_backend_dev_t npu_dev = ggml_backend_dev_by_name(npu_name);
    if (gpu_dev == nullptr || npu_dev == nullptr) {
        fprintf(stderr, "Could not find requested GPU/NPU devices: gpu=%s npu=%s\n", gpu_name, npu_name);
        print_devices();
        return 1;
    }

    ggml_backend_t gpu = ggml_backend_dev_init(gpu_dev, nullptr);
    ggml_backend_t npu = ggml_backend_dev_init(npu_dev, nullptr);
    if (gpu == nullptr || npu == nullptr) {
        fprintf(stderr, "Failed to initialize gpu=%s or npu=%s\n", gpu_name, npu_name);
        if (gpu != nullptr) {
            ggml_backend_free(gpu);
        }
        if (npu != nullptr) {
            ggml_backend_free(npu);
        }
        return 1;
    }

    const char * hostbuf_env = std::getenv("GGML_HEXAGON_HOSTBUF");
    printf("[HANDOFF_CONFIG] gpu=%s npu=%s hidden=%d reps=%d warmup=%d GGML_HEXAGON_HOSTBUF=%s\n",
           gpu_name, npu_name, hidden, reps, warmup, hostbuf_env ? hostbuf_env : "<default>");

    const int chunks[] = {64, 128, 256};
    bool all_ok = true;

    for (int chunk : chunks) {
        tensor_holder gpu_tensor;
        tensor_holder npu_tensor;

        if (!alloc_tensor_on_backend(gpu_tensor, gpu, hidden, chunk, "handoff_gpu") ||
            !alloc_tensor_on_backend(npu_tensor, npu, hidden, chunk, "handoff_npu")) {
            fprintf(stderr, "Failed to allocate handoff tensors for chunk=%d\n", chunk);
            all_ok = false;
            break;
        }

        const size_t bytes = ggml_nbytes(gpu_tensor.tensor);
        if (ggml_nbytes(npu_tensor.tensor) != bytes) {
            fprintf(stderr, "Tensor byte-size mismatch for chunk=%d\n", chunk);
            all_ok = false;
            break;
        }

        printf("[HANDOFF_BUFFER] chunk=%d bytes=%zu "
               "gpu_buffer=%s gpu_host=%d "
               "npu_buffer=%s npu_host=%d\n",
               chunk, bytes,
               ggml_backend_buffer_name(gpu_tensor.buffer),
               ggml_backend_buffer_is_host(gpu_tensor.buffer) ? 1 : 0,
               ggml_backend_buffer_name(npu_tensor.buffer),
               ggml_backend_buffer_is_host(npu_tensor.buffer) ? 1 : 0);

        if (!verify_copy("GPU_TO_NPU", gpu, npu, gpu_tensor.tensor, npu_tensor.tensor, bytes)) {
            all_ok = false;
            break;
        }
        if (!verify_copy("NPU_TO_GPU", npu, gpu, npu_tensor.tensor, gpu_tensor.tensor, bytes)) {
            all_ok = false;
            break;
        }

        bench_sync("GPU_TO_NPU", gpu, npu, gpu_tensor.tensor, npu_tensor.tensor,
                   chunk, bytes, reps, warmup);
        bench_async_api("GPU_TO_NPU", gpu, npu, gpu_tensor.tensor, npu_tensor.tensor,
                        chunk, bytes, reps, warmup);

        bench_sync("NPU_TO_GPU", npu, gpu, npu_tensor.tensor, gpu_tensor.tensor,
                   chunk, bytes, reps, warmup);
        bench_async_api("NPU_TO_GPU", npu, gpu, npu_tensor.tensor, gpu_tensor.tensor,
                        chunk, bytes, reps, warmup);
    }

    ggml_backend_free(gpu);
    ggml_backend_free(npu);

    printf("[HANDOFF_DONE] ok=%d\n", all_ok ? 1 : 0);
    return all_ok ? 0 : 1;
}
