// Standalone HTP0 allocation probe. No GGUF, inference graph, or RPC server.
// This measures GGML -> rpcmem_alloc2 -> FastRPC host-side mapping only.
// It does NOT establish the DSP's active HAP_mmap/graph capacity.
#include "ggml-backend.h"

#include <algorithm>
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

namespace {
constexpr uint64_t MIB = 1024ULL * 1024;
struct options {
    std::vector<uint64_t> sizes {256, 512, 768, 1024, 1280, 1536, 1792, 2048};
    std::string mode = "both";
    uint64_t aggregate_cap_mib = 0; // 0: conservative auto cap
    bool touch_pages = false;       // optional; substantially increases pressure
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
        if (n < 64) { throw std::runtime_error("MINIMUM_SIZE_IS_64_MIB"); }
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
        "[--sizes-mib 256,512,...] [--max-aggregate-mib N] [--touch-pages]\n"
        "Measures rpcmem allocation and FastRPC host mapping only, not DSP "
        "active graph mapping. Stop inference server before running.\n";
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
    return o;
}

void record(const char * mode, uint64_t size, const char * status,
            int failed_index, double elapsed_ms, uint64_t avail_after) {
    std::cout << "[HTP_MEM_PROBE] event=RESULT"
              << " mode=" << mode << " each_mib=" << size
              << " total_mib=" << size * (std::string(mode) == "dual" ? 2 : 1)
              << " status=" << status
              << " failed_buffer=" << failed_index
              << " elapsed_ms=" << elapsed_ms
              << " mem_available_after_mib=" << avail_after
              << " scope=RPCMEM_FASTRPC_HOST_MAP" << std::endl;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse(argc, argv);
        const uint64_t available = available_mib();
        if (available < 3072) {
            throw std::runtime_error("MEMAVAILABLE_BELOW_3072_MIB:REFUSE_PRESSURE_TEST");
        }
        // An explicit cap is allowed up to (MemAvailable - 2 GiB). Never
        // automatically consume more than one third of currently available
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
                } // Always release all buffers before the next attempt.
                const double elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                record(mode.c_str(), each, status, failure,
                       elapsed, available_mib());
                if (failure >= 0) { break; } // Do not stress larger sizes.
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
