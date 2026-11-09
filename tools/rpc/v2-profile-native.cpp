// Native V2 GCNA Phone Profile over the *existing* ggml-rpc command 57.
// The complete N/A graph and CPU parity check run in the RPC process.
// No Python, popen, fork, exec, sockets, external profiler, or sidecar.
#include "v2-profile-native.h"
#include "v2-profile-sha256.h"
#include "v2_stage_profile_embedded.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#include <sys/utsname.h>
#include <unistd.h>

namespace {
using json = nlohmann::json;
constexpr size_t max_frame = 256 * 1024;

std::string fail_response(const std::string & reason) {
    json response = {
        {"version", 1}, {"status", "ERROR"},
        {"error", reason.substr(0, 3000)}
    };
    return response.dump() + "\n";
}

std::string property(const char * key) {
#if defined(__ANDROID__)
    char value[PROP_VALUE_MAX] = {};
    return __system_property_get(key, value) > 0 ? value : "";
#else
    (void) key;
    return "";
#endif
}

std::filesystem::path executable_path() {
    char buffer[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (n <= 0 || size_t(n) >= sizeof(buffer) - 1) {
        throw std::runtime_error("RPC_EXECUTABLE_PATH_UNAVAILABLE");
    }
    buffer[n] = '\0';
    return std::filesystem::path(buffer);
}

json device_identity() {
    utsname name {};
    if (uname(&name) != 0) {
        throw std::runtime_error("PHONE_UNAME_FAILED");
    }
    return {
        {"machine", name.machine},
        {"product", property("ro.product.model")},
        {"soc", property("ro.soc.model")},
        {"abi", property("ro.product.cpu.abi")},
        {"os_build", property("ro.build.fingerprint")},
        {"vendor_build", property("ro.vendor.build.fingerprint")},
    };
}

std::string hash_library(
        const std::filesystem::path & folder,
        const std::string & library) {
    const std::filesystem::path in_deployment = folder / "lib" / library;
    if (std::filesystem::is_regular_file(in_deployment)) {
        return v2_profile_hash::file_sha256(in_deployment.string());
    }
    const char * raw_path = std::getenv("LD_LIBRARY_PATH");
    if (raw_path != nullptr) {
        std::istringstream paths(raw_path);
        std::string base;
        while (std::getline(paths, base, ':')) {
            if (base.empty()) { continue; }
            const std::filesystem::path candidate =
                std::filesystem::path(base) / library;
            if (std::filesystem::is_regular_file(candidate)) {
                return v2_profile_hash::file_sha256(candidate.string());
            }
        }
    }
    return "NOT_FOUND";
}

int bounded_int(const json & shape,
                const char * key, int min, int max) {
    auto it = shape.find(key);
    if (it == shape.end() || !it->is_number_integer()) {
        throw std::runtime_error(std::string("INVALID_PROFILE_") + key);
    }
    const int64_t value = it->get<int64_t>();
    if (value < min || value > max) {
        throw std::runtime_error(std::string("INVALID_PROFILE_") + key);
    }
    return static_cast<int>(value);
}

double bounded_threshold(const json & shape, const char * key) {
    auto it = shape.find(key);
    if (it == shape.end() || !it->is_number()) {
        throw std::runtime_error(std::string("INVALID_PROFILE_") + key);
    }
    const double value = it->get<double>();
    if (!std::isfinite(value) || value < 0 || value > 100) {
        throw std::runtime_error(std::string("INVALID_PROFILE_") + key);
    }
    return value;
}

std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> out;
    std::istringstream stream(line);
    std::string value;
    while (std::getline(stream, value, ',')) {
        out.push_back(value);
    }
    return out;
}

json native_csv_sample(
        const v2_stage_profile_native_result & result,
        const v2_stage_profile_native_options & options) {
    std::istringstream input(result.csv);
    std::string header, sample_line, trailing;
    if (!std::getline(input, header) ||
            !std::getline(input, sample_line) ||
            std::getline(input, trailing)) {
        throw std::runtime_error("NATIVE_PROFILE_NO_SINGLE_CSV_RESULT");
    }
    const auto names = split_csv(header);
    const auto values = split_csv(sample_line);
    if (names.empty() || names.size() != values.size()) {
        throw std::runtime_error("NATIVE_PROFILE_CSV_SCHEMA_MISMATCH");
    }
    json row = json::object();
    for (size_t i = 0; i < names.size(); ++i) {
        if (names[i].empty() || row.contains(names[i])) {
            throw std::runtime_error("NATIVE_PROFILE_CSV_DUPLICATE_COLUMN");
        }
        row[names[i]] = values[i];
    }
    const auto get = [&row](const char * key) -> std::string {
        if (!row.contains(key) || !row.at(key).is_string()) {
            throw std::runtime_error(std::string("NATIVE_PROFILE_MISSING_") + key);
        }
        return row.at(key).get<std::string>();
    };
    if (get("backend") != std::string(1, options.backend) ||
            std::stoi(get("layers")) != options.layers ||
            std::stoi(get("chunk")) != options.chunk ||
            std::stoi(get("kv_tokens")) != options.kv ||
            std::stoi(get("repeats")) != options.repeats ||
            get("source") != "NATIVE_SINGLE_BACKEND_GGUF_PREFIX") {
        throw std::runtime_error("NATIVE_PROFILE_RESULT_SHAPE_MISMATCH");
    }
    const auto positive = [&get](const char * key, bool require_positive) {
        const double n = std::stod(get(key));
        if (!std::isfinite(n) || n < 0 || (require_positive && n == 0)) {
            throw std::runtime_error(std::string("NATIVE_PROFILE_BAD_") + key);
        }
        return n;
    };
    positive("compute_median_ms", true);
    positive("prepare_median_ms", false);
    positive("weight_upload_ms", false);
    positive("htp_precompile_ms", false);
    const double relative_rmse = positive("relative_rmse", false);
    const double max_abs = positive("max_abs", false);
    const bool numeric_pass =
        relative_rmse <= options.max_relative_rmse &&
        max_abs <= options.max_abs_error;
    const std::string parity = get("parity");
    if (result.exit_code == 0 &&
            ((options.verify_cpu && (parity != "PASS" || !numeric_pass)) ||
             (!options.verify_cpu && parity == "FAIL"))) {
        throw std::runtime_error("NATIVE_PROFILE_FALSE_PARITY_PASS");
    }
    if (result.exit_code == 2 &&
            (!options.verify_cpu || parity != "FAIL" || numeric_pass)) {
        throw std::runtime_error("NATIVE_PROFILE_PARITY_FAILURE_INCONSISTENT");
    }
    if (result.exit_code != 0 && result.exit_code != 2) {
        throw std::runtime_error("NATIVE_PROFILE_EXECUTION_FAILED");
    }
    if (options.backend == 'N' && parity == "NOT_REQUESTED") {
        throw std::runtime_error("HTP_PARITY_REQUIRED");
    }
    if (options.bank_repeats > 0) {
        positive("same_bank_refill_median_ms", true);
        positive("raw_weight_bytes", true);
    }
    return row;
}

} // namespace

v2_profile_native_service::v2_profile_native_service(std::string model_path)
    : model_path_(std::move(model_path)) {}

void v2_profile_native_service::ensure_identity() {
    if (identity_ready_) { return; }
    if (model_path_.empty() || !std::filesystem::is_regular_file(model_path_)) {
        throw std::runtime_error("PHONE_PROFILE_MODEL_UNAVAILABLE");
    }
    // Run once for the entire RPC process; never hash the multi-GB GGUF
    // again for every independent NPU/Adreno point.
    model_sha256_ = v2_profile_hash::file_sha256(model_path_);
    const std::filesystem::path executable = executable_path();
    exe_sha256_ = v2_profile_hash::file_sha256(executable.string());
    const json device = device_identity();
    device_json_ = device.dump();
    const std::vector<std::string> keys = {
        device.at("machine").get<std::string>(),
        device.at("product").get<std::string>(),
        device.at("soc").get<std::string>(),
        device.at("abi").get<std::string>(),
        device.at("os_build").get<std::string>(),
        device.at("vendor_build").get<std::string>()
    };
    std::string raw_id;
    for (const auto & item : keys) {
        if (!raw_id.empty()) { raw_id += "|"; }
        raw_id += item;
    }
    device_id_ = v2_profile_hash::string_sha256(raw_id);

    json libs = json::object();
    const std::filesystem::path folder = executable.parent_path().parent_path();
    for (const char * name : {"libggml-htp-v79.so", "libggml-hexagon.so",
                              "libggml-opencl.so", "libggml-cpu.so",
                              "libggml.so"}) {
        libs[name] = hash_library(folder, name);
    }
    library_fingerprint_json_ = libs.dump();
    const json runtime = {
        {"exe_sha256", exe_sha256_},
        {"library_sha256", libs},
        {"device_id", device_id_},
        {"transport", "V2_PHONE_PROFILE_NATIVE_IN_PROCESS"}
    };
    runtime_sha256_ = v2_profile_hash::string_sha256(runtime.dump());
    identity_ready_ = true;
}

std::string v2_profile_native_service::handle(
        const uint8_t * data, size_t size) {
    try {
        if (data == nullptr || size < 2 || size > max_frame ||
                data[size - 1] != '\n') {
            throw std::runtime_error("INVALID_PROFILE_REQUEST_FRAME");
        }
        const json request = json::parse(
            std::string(reinterpret_cast<const char *>(data), size - 1));
        if (!request.is_object() || !request.contains("version") ||
                !request.at("version").is_number_integer() ||
                request.at("version").get<int>() != 1) {
            throw std::runtime_error("V2_PROFILE_PROTOCOL_VERSION_MISMATCH");
        }
        const std::string action = request.at("action").get<std::string>();
        if (action != "hello" && action != "profile") {
            throw std::runtime_error("UNKNOWN_PROFILE_ACTION");
        }
        // Optional Phone-side token. Constant-time length-matched equality,
        // no client-controlled path, shell, code or command execution.
        const char * expected = std::getenv("V2_PROFILE_TOKEN");
        if (expected != nullptr && *expected) {
            const std::string supplied = request.value("token", std::string());
            const size_t n = std::strlen(expected);
            unsigned mismatch = supplied.size() == n ? 0u : 1u;
            for (size_t i = 0; i < n; ++i) {
                const unsigned char a = static_cast<unsigned char>(expected[i]);
                const unsigned char b = i < supplied.size() ?
                    static_cast<unsigned char>(supplied[i]) : 0u;
                mismatch |= a ^ b;
            }
            if (mismatch != 0) {
                throw std::runtime_error("UNAUTHORIZED_PROFILE_REQUEST");
            }
        }
        ensure_identity();
        const json libs = json::parse(library_fingerprint_json_);
        const json device = json::parse(device_json_);
        if (action == "hello") {
            const json response = {
                {"version", 1}, {"status", "OK"},
                {"model_sha256", model_sha256_},
                {"exe_sha256", exe_sha256_},
                {"runtime_sha256", runtime_sha256_},
                {"device_id", device_id_},
                {"device", device},
                {"library_sha256", libs},
                {"backends", {"N", "A"}},
                {"transport", "V2_PHONE_PROFILE_RPC50052"}
            };
            return response.dump() + "\n";
        }
        if (request.value("model_sha256", std::string()) != model_sha256_) {
            throw std::runtime_error("PHONE_PC_GGUF_SHA256_MISMATCH");
        }
        const json shape = request.at("shape");
        if (!shape.is_object()) {
            throw std::runtime_error("INVALID_PROFILE_SHAPE");
        }
        const std::string backend = shape.at("backend").get<std::string>();
        if (backend != "N" && backend != "A") {
            throw std::runtime_error("PHONE_PROFILE_ONLY_N_OR_A");
        }
        v2_stage_profile_native_options opts {};
        opts.model = model_path_;
        opts.backend = backend[0];
        opts.layers = bounded_int(shape, "layers", 1, 32);
        opts.chunk = bounded_int(shape, "chunk", 1, 4096);
        opts.kv = bounded_int(shape, "kv_tokens", 1, 16384);
        opts.warmups = bounded_int(shape, "warmups", 0, 20);
        opts.repeats = bounded_int(shape, "repeats", 1, 100);
        opts.threads = bounded_int(shape, "threads", 1, 64);
        opts.bank_repeats = shape.contains("bank_repeats") ?
            bounded_int(shape, "bank_repeats", 0, 20) : 0;
        opts.host_stage_cap_mib = shape.contains("host_stage_cap_mib") ?
            bounded_int(shape, "host_stage_cap_mib", 1, 8192) : 1024;
        opts.max_relative_rmse = bounded_threshold(shape, "max_relative_rmse");
        opts.max_abs_error = bounded_threshold(shape, "max_abs_error");
        if (!shape.contains("verify_cpu") ||
                !shape.at("verify_cpu").is_boolean()) {
            throw std::runtime_error("INVALID_PROFILE_VERIFY_CPU");
        }
        opts.verify_cpu = shape.at("verify_cpu").get<bool>() || opts.backend == 'N';
        if (opts.kv < opts.chunk ||
                (opts.backend != 'N' && opts.bank_repeats != 0)) {
            throw std::runtime_error("INVALID_PROFILE_CHUNK_KV_OR_BANK");
        }
        std::fprintf(stderr,
            "[V2_PROFILE_NATIVE] event=RUN backend=%c layers=%d chunk=%d kv=%d\n",
            opts.backend, opts.layers, opts.chunk, opts.kv);
        const auto native = v2_stage_profile_run_embedded(opts);
        if (native.exit_code == 1) {
            throw std::runtime_error("PHONE_NATIVE_PROFILE_FAILED:" + native.error);
        }
        const json sample = native_csv_sample(native, opts);
        const json response = {
            {"version", 1},
            {"status", native.exit_code == 2 ? "PARITY_FAIL" : "PASS"},
            {"sample", sample},
            {"model_sha256", model_sha256_},
            {"exe_sha256", exe_sha256_},
            {"runtime_sha256", runtime_sha256_},
            {"device_id", device_id_},
            {"diagnostics", native.error.substr(0, 2500)},
        };
        std::fprintf(stderr,
            "[V2_PROFILE_NATIVE] event=DONE backend=%c layers=%d "
            "status=%s\n", opts.backend, opts.layers,
            native.exit_code == 2 ? "PARITY_FAIL" : "PASS");
        return response.dump() + "\n";
    } catch (const std::exception & ex) {
        std::fprintf(stderr,
            "[V2_PROFILE_NATIVE] event=FAIL reason=%s\n", ex.what());
        return fail_response(ex.what());
    }
}

bool v2_profile_native_service::rpc_callback(
        void * context, const uint8_t * request, size_t request_size,
        uint8_t * response, size_t response_capacity, size_t * response_size) {
    if (context == nullptr || response == nullptr || response_size == nullptr) {
        return false;
    }
    auto & service = *static_cast<v2_profile_native_service *>(context);
    std::lock_guard<std::mutex> guard(service.mutex_);
    const std::string result = service.handle(request, request_size);
    if (result.empty() || result.size() > response_capacity) {
        return false;
    }
    std::memcpy(response, result.data(), result.size());
    *response_size = result.size();
    return true;
}
