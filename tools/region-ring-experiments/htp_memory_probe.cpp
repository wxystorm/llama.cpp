// Standalone HTP0 allocation probe. No GGUF, inference graph, or RPC server.
// This measures GGML -> rpcmem_alloc2 -> FastRPC host-side mapping only.
// It does NOT establish the DSP's active HAP_mmap/graph capacity.
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __ANDROID__
#  include <dlfcn.h>
#endif

namespace {
constexpr uint64_t MIB = 1024ULL * 1024;
struct options {
    std::vector<uint64_t> sizes {256, 512, 768, 1024, 1280, 1536, 1792};
    std::string mode = "both";
    uint64_t aggregate_cap_mib = 0; // 0: conservative auto cap
    bool touch_pages = false;       // optional; substantially increases pressure
    bool dsp_check = false;         // execute a real HTP F32 ADD using both arenas
    bool tight_dsp_vmem = false;    // explicit reduced mapping reserve (64 MiB)
    bool allow_large_single = false; // opt-in 2-3 GiB single arena, only with rpcmem_alloc2
};

uint64_t parse_u64(const std::string & s) {
    if (s.empty() || s[0] == '-') { throw std::runtime_error("INVALID_INTEGER"); }
    size_t pos = 0;
    const auto n = std::stoull(s, &pos, 10);
    if (pos != s.size() || n == 0 || n > 8192) {
        throw std::runtime_error("INVALID_MIB_VALUE");
    }
    return n;
}

std::vector<uint64_t> sizes_from_csv(const std::string & csv) {
    std::vector<uint64_t> out;
    size_t start = 0;
    while (start < csv.size()) {
        const size_t comma = csv.find(',', start);
        const std::string part = csv.substr(start,
            comma == std::string::npos ? std::string::npos : comma - start);
        const uint64_t n = parse_u64(part);
        // A large (>1984 MiB) request requires a separate explicit
        // opt-in and runtime proof of the size_t-based rpcmem_alloc2 symbol.
        if (n < 64 || n > 3072) {
            throw std::runtime_error("SIZE_MUST_BE_64_TO_3072_MIB");
        }
        out.push_back(n);
        if (comma == std::string::npos) { break; }
        start = comma + 1;
    }
    if (out.empty() || csv.back() == ',') {
        throw std::runtime_error("EMPTY_OR_TRAILING_SIZE");
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

uint64_t available_mib() {
    std::ifstream in("/proc/meminfo");
    std::string key, unit;
    uint64_t value = 0;
    while (in >> key >> value >> unit) {
        if (key == "MemAvailable:") { return value / 1024; }
    }
    return 0;
}

void touch_pages(ggml_backend_buffer_t buffer, uint64_t requested_bytes) {
    auto * base = static_cast<volatile unsigned char *>(
        ggml_backend_buffer_get_base(buffer));
    if (!base) { throw std::runtime_error("BUFFER_HAS_NO_HOST_BASE"); }
    for (uint64_t offset = 0; offset < requested_bytes; offset += 4096) {
        base[offset] = 0;
    }
}

struct buffer_guard {
    std::vector<ggml_backend_buffer_t> handles;
    ~buffer_guard() {
        for (auto it = handles.rbegin(); it != handles.rend(); ++it) {
            ggml_backend_buffer_free(*it);
        }
    }
};

void usage() {
    std::cout <<
        "Usage: ggml-htp-memory-probe [--mode single|dual|both] "
        "[--sizes-mib 256,512,...] [--max-aggregate-mib N] "
        "[--touch-pages] [--dsp-check] [--allow-tight-dsp-vmem] "
        "[--allow-large-single]\n"
        "By default measures RPCMEM/FastRPC host mapping only. "
        "--dsp-check additionally executes a tiny HTP ADD graph referencing "
        "the complete large-buffer descriptors. Stop the RPC server first.\n";
}

options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        }
        if (arg == "--touch-pages") { o.touch_pages = true; continue; }
        if (arg == "--dsp-check") { o.dsp_check = true; continue; }
        if (arg == "--allow-large-single") {
            o.allow_large_single = true;
            continue;
        }
        if (arg == "--allow-tight-dsp-vmem") {
            o.tight_dsp_vmem = true;
            continue;
        }
        if (i + 1 >= argc) { throw std::runtime_error("MISSING_ARGUMENT:" + arg); }
        const std::string value = argv[++i];
        if (arg == "--mode") {
            if (value != "both" && value != "single" && value != "dual") {
                throw std::runtime_error("INVALID_MODE");
            }
            o.mode = value;
        } else if (arg == "--sizes-mib") {
            o.sizes = sizes_from_csv(value);
        } else if (arg == "--max-aggregate-mib") {
            o.aggregate_cap_mib = parse_u64(value);
        } else {
            throw std::runtime_error("UNKNOWN_ARGUMENT:" + arg);
        }
    }
    if (o.tight_dsp_vmem && !o.dsp_check) {
        throw std::runtime_error("TIGHT_DSP_VMEM_REQUIRES_DSP_CHECK");
    }
    if (o.allow_large_single &&
            (o.mode != "single" || !o.dsp_check)) {
        throw std::runtime_error(
            "LARGE_SINGLE_REQUIRES_SINGLE_MODE_AND_DSP_CHECK");
    }
    for (uint64_t mib : o.sizes) {
        if (mib > 1984 &&
                (!o.allow_large_single || o.mode != "single")) {
            throw std::runtime_error(
                "OVER_1984_MIB_REQUIRES_ALLOW_LARGE_SINGLE");
        }
    }
    return o;
}

// The old rpcmem_alloc(heap, flags, int size) API cannot represent a 3 GiB
// allocation. On Android refuse the experiment unless libcdsprpc.so exports
// rpcmem_alloc2(heap, flags, size_t size), avoiding silent int truncation
// inside the Hexagon backend's fallback code path.
bool has_rpcmem_alloc2() {
#ifdef __ANDROID__
    void * driver = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!driver) { return false; }
    const bool available = dlsym(driver, "rpcmem_alloc2") != nullptr;
    dlclose(driver);
    return available;
#else
    return false;
#endif
}

void record(const char * mode, uint64_t size, const char * status,
            int failed_index, double elapsed_ms, uint64_t avail_after) {
    std::cout << "[HTP_MEM_PROBE] event=RESULT"
              << " mode=" << mode << " each_mib=" << size
              << " total_mib=" << size * (std::string(mode).find("dual") != std::string::npos ? 2 : 1)
              << " status=" << status
              << " failed_buffer=" << failed_index
              << " elapsed_ms=" << elapsed_ms
              << " mem_available_after_mib=" << avail_after
              << " scope=RPCMEM_FASTRPC_HOST_MAP" << std::endl;
}
// Execute a real HTP graph that references both giant RPCMEM arenas.
// Hexagon's op serializer passes the entire buffer descriptor (not merely
// the 256-element F32 tensor) to the DSP and charges that whole allocation
// against the active mapping budget. This is a mapping test, not a bandwidth
// or full-layer accuracy/performance benchmark.
const char * run_dsp_mapping_check(
        ggml_backend_t backend, ggml_backend_buffer_type_t buft,
        const std::vector<ggml_backend_buffer_t> & buffers) {
    if (buffers.empty()) { return "NO_INPUT_BUFFERS"; }
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({1024 * 1024, nullptr, true}), &ggml_free);
    if (!ctx) { return "GRAPH_METADATA_ALLOC_FAILED"; }
    ggml_tensor * a = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 256);
    ggml_tensor * b = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 256);
    ggml_tensor * dst = ggml_add(ctx.get(), a, b);
    if (!a || !b || !dst) { return "GRAPH_TENSOR_SETUP_FAILED"; }
    buffer_guard output_buf;
    ggml_backend_buffer_t out =
        ggml_backend_buft_alloc_buffer(buft, 64 * 1024);
    if (!out) { return "OUTPUT_BUFFER_ALLOC_FAILED"; }
    output_buf.handles.push_back(out);
    const ggml_backend_buffer_t bbuf = buffers.size() > 1 ?
        buffers[1] : buffers[0];
    if (ggml_backend_tensor_alloc(
            buffers[0], a,
            ggml_backend_buffer_get_base(buffers[0])) !=
            GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(
            bbuf, b,
            static_cast<unsigned char *>(
                ggml_backend_buffer_get_base(bbuf)) +
                (buffers.size() > 1 ? 0 : 4096)) !=
            GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(
            out, dst, ggml_backend_buffer_get_base(out)) !=
            GGML_STATUS_SUCCESS) {
        return "GRAPH_TENSOR_BIND_FAILED";
    }
    std::array<float, 256> av, bv, result {};
    av.fill(1.25f);
    bv.fill(2.50f);
    ggml_backend_tensor_set(a, av.data(), 0, sizeof(av));
    ggml_backend_tensor_set(b, bv.data(), 0, sizeof(bv));
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    if (!graph) { return "GRAPH_CREATE_FAILED"; }
    ggml_build_forward_expand(graph, dst);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        return "HTP_GRAPH_COMPUTE_FAILED";
    }
    ggml_backend_tensor_get(dst, result.data(), 0, sizeof(result));
    for (float value : result) {
        if (!std::isfinite(value) || std::fabs(value - 3.75f) > 1e-3f) {
            return "HTP_GRAPH_OUTPUT_MISMATCH";
        }
    }
    return "PASS";
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse(argc, argv);
        const bool large_single = opts.allow_large_single &&
            std::any_of(opts.sizes.begin(), opts.sizes.end(),
                [](uint64_t size) { return size > 1984; });
        if (large_single) {
            const bool has_alloc2 = has_rpcmem_alloc2();
            std::cout << "[HTP_MEM_PROBE] event=RPCMEM_ALLOC2"
                      << " available=" << (has_alloc2 ? 1 : 0)
                      << " large_single_requested=1" << std::endl;
            if (!has_alloc2) {
                throw std::runtime_error(
                    "OVER_2_GIB_REFUSED_RPCMEM_ALLOC2_SYMBOL_MISSING");
            }
        }
        const uint64_t available = available_mib();
        if (available < 3072) {
            throw std::runtime_error("MEMAVAILABLE_BELOW_3072_MIB:REFUSE_PRESSURE_TEST");
        }
        // An explicit cap is allowed up to (MemAvailable - 2 GiB). The
        // default never consumes more than one third of currently available
        // RAM, nor more than 3 GiB in a single attempt.
        const uint64_t safe_cap = available - 2048;
        const uint64_t aggregate_cap = opts.aggregate_cap_mib ?
            opts.aggregate_cap_mib : std::min<uint64_t>(3072, available / 3);
        if (aggregate_cap > safe_cap) {
            throw std::runtime_error("CAP_WOULD_LEAVE_LESS_THAN_2_GIB_AVAILABLE");
        }
        std::cout << "[HTP_MEM_PROBE] event=START"
                  << " mem_available_mib=" << available
                  << " aggregate_cap_mib=" << aggregate_cap
                  << " touch_pages=" << (opts.touch_pages ? 1 : 0)
                  << " dsp_check=" << (opts.dsp_check ? 1 : 0)
                  << " tight_dsp_vmem=" << (opts.tight_dsp_vmem ? 1 : 0)
                  << " allow_large_single=" << (opts.allow_large_single ? 1 : 0)
                  << " scope=RPCMEM_FASTRPC_HOST_MAP" << std::endl;

        ggml_backend_load_all();
        auto * dev = ggml_backend_dev_by_name("HTP0");
        if (!dev) { throw std::runtime_error("HTP0_DEVICE_NOT_FOUND"); }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) { throw std::runtime_error("HTP0_BACKEND_INIT_FAILED"); }
        const auto reg = ggml_backend_dev_backend_reg(dev);
        const auto get_extra = reg ? reinterpret_cast<
            ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts")) : nullptr;
        auto ** types = get_extra ? get_extra(dev) : nullptr;
        auto * repack = types ? types[0] : nullptr;
        if (!repack) {
            ggml_backend_free(backend);
            throw std::runtime_error("HTP0_REPACK_BUFFER_TYPE_NOT_FOUND");
        }
        const uint64_t per_buffer_cap = ggml_backend_buft_get_max_size(repack) / MIB;
        std::cout << "[HTP_MEM_PROBE] event=LIMIT"
                  << " buft=" << ggml_backend_buft_name(repack)
                  << " reported_single_buffer_max_mib=" << per_buffer_cap
                  << " hexagon_mbuf_env="
                  << (std::getenv("GGML_HEXAGON_MBUF") ?
                      std::getenv("GGML_HEXAGON_MBUF") : "default")
                  << " hexagon_vmem_env="
                  << (std::getenv("GGML_HEXAGON_VMEM") ?
                      std::getenv("GGML_HEXAGON_VMEM") : "default")
                  << " note=NO_DSP_GRAPH_MAPPING" << std::endl;

        int passed = 0;
        for (const std::string mode : {"single", "dual"}) {
            if (opts.mode != "both" && opts.mode != mode) { continue; }
            const int count = mode == "dual" ? 2 : 1;
            for (const uint64_t each : opts.sizes) {
                if (each > per_buffer_cap) {
                    record(mode.c_str(), each, "SKIP_SOFTWARE_BUFFER_LIMIT",
                           -1, 0, available_mib());
                    continue;
                }
                if (each > std::numeric_limits<uint64_t>::max() / count ||
                        each * count > aggregate_cap) {
                    record(mode.c_str(), each, "SKIP_AGGREGATE_SAFETY_CAP",
                           -1, 0, available_mib());
                    continue;
                }
                if (available_mib() < each * count + 2048) {
                    record(mode.c_str(), each, "STOP_LOW_MEMAVAILABLE",
                           -1, 0, available_mib());
                    break;
                }
                std::cout << "[HTP_MEM_PROBE] event=ATTEMPT"
                          << " mode=" << mode << " each_mib=" << each
                          << " count=" << count << std::endl;
                const auto start = std::chrono::steady_clock::now();
                int failure = -1;
                const char * status = "PASS";
                bool dsp_failed = false;
                {
                    buffer_guard buffers;
                    for (int b = 0; b < count; ++b) {
                        ggml_backend_buffer_t handle =
                            ggml_backend_buft_alloc_buffer(repack, each * MIB);
                        if (!handle) {
                            failure = b;
                            status = "ALLOC_OR_FASTRPC_MAP_FAILED";
                            break;
                        }
                        buffers.handles.push_back(handle);
                        ggml_backend_buffer_set_usage(
                            handle, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                        if (opts.touch_pages) {
                            touch_pages(handle, each * MIB);
                        }
                    }
                    if (failure < 0 && opts.dsp_check) {
                        // Default: 320 MiB DSP mapping reserve. Explicit
                        // opt-in: 64 MiB for near-limit experiments such
                        // as two 1536 MiB arenas within a 3200 MiB VMEM
                        // limit. This is intentionally NOT the default:
                        // real inference graphs need additional KV/workspace.
                        const uint64_t dsp_reserve_mib =
                            opts.tight_dsp_vmem ? 64 : 320;
                        const uint64_t vmem_mib = std::getenv(
                            "GGML_HEXAGON_VMEM") ?
                            std::strtoull(std::getenv(
                                "GGML_HEXAGON_VMEM"), nullptr, 10) : 3200;
                        const uint64_t active = each * count + 1;
                        if (active > vmem_mib ||
                                dsp_reserve_mib > vmem_mib - active) {
                            std::cout << "[HTP_MEM_PROBE] event=DSP_RESULT"
                                      << " mode=" << mode
                                      << " each_mib=" << each
                                      << " total_mib=" << each * count
                                      << " status=SKIP_DSP_VMEM_HEADROOM"
                                      << " vmem_mib=" << vmem_mib
                                      << " dsp_reserve_mib=" << dsp_reserve_mib
                                      << std::endl;
                        } else {
                            std::cout << "[HTP_MEM_PROBE] event=DSP_ATTEMPT"
                                      << " mode=" << mode
                                      << " each_mib=" << each
                                      << " total_mib=" << each * count
                                      << " vmem_mib=" << vmem_mib
                                      << " dsp_reserve_mib=" << dsp_reserve_mib
                                      << " scope=HTP_DSP_GRAPH"
                                      << std::endl;
                            const auto dsp_start =
                                std::chrono::steady_clock::now();
                            const char * dsp_status =
                                run_dsp_mapping_check(
                                    backend, repack, buffers.handles);
                            dsp_failed = std::string(dsp_status) != "PASS";
                            const double dsp_elapsed =
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() -
                                    dsp_start).count();
                            std::cout << "[HTP_MEM_PROBE] event=DSP_RESULT"
                                      << " mode=" << mode
                                      << " each_mib=" << each
                                      << " total_mib=" << each * count
                                      << " status=" << dsp_status
                                      << " elapsed_ms=" << dsp_elapsed
                                      << " scope=HTP_DSP_GRAPH"
                                      << std::endl;
                        }
                    }
                } // Always release all buffers before the next attempt.
                const double elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                record(mode.c_str(), each, status, failure,
                       elapsed, available_mib());
                if (failure >= 0 || dsp_failed) {
                    break; // Do not stress larger sizes after a real failure.
                }
                ++passed;
            }
        }
        ggml_backend_free(backend);
        std::cout << "[HTP_MEM_PROBE] event=DONE pass_cases=" << passed
                  << " scope=RPCMEM_FASTRPC_HOST_MAP" << std::endl;
        return passed > 0 ? 0 : 3;
    } catch (const std::exception & exc) {
        std::cerr << "[HTP_MEM_PROBE] event=ERROR reason=" << exc.what()
                  << std::endl;
        return 2;
    }
}
