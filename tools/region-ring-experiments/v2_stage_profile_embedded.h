#pragma once

// Same real Qwen3 isolated-GGUF graph profiler used by the standalone
// region-ring-v2-stage-profile CLI, exposed as an in-process call.
// Compiled into Android ggml-rpc-server; no Python, fork, exec or sidecar.
#include <string>

struct v2_stage_profile_native_options {
    std::string model;
    char backend = 'N'; // N = HTP0, A = GPUOpenCL
    int layers = 1;
    int chunk = 32;
    int kv = 512;
    int warmups = 2;
    int repeats = 5;
    int threads = 8;
    int bank_repeats = 0;
    int host_stage_cap_mib = 1024;
    bool verify_cpu = true;
    double max_relative_rmse = 0.02;
    double max_abs_error = 0.5;
};

struct v2_stage_profile_native_result {
    // 0 = complete PASS, 2 = numerically measured PARITY_FAIL, 1 = error.
    int exit_code = 1;
    std::string csv;   // exact standalone CLI header + one data row
    std::string error; // error diagnostic, never a fabricated timing
};

v2_stage_profile_native_result v2_stage_profile_run_embedded(
    const v2_stage_profile_native_options & options);
