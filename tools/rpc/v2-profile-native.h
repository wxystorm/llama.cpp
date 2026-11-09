#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

class v2_profile_native_service {
public:
    explicit v2_profile_native_service(std::string model_path);
    v2_profile_native_service(const v2_profile_native_service &) = delete;
    v2_profile_native_service & operator=(const v2_profile_native_service &) = delete;

    // Called only by GGML RPC command 57, after the normal RPC HELLO.
    // No script, subprocess, listener, or remotely selected file path.
    static bool rpc_callback(
        void * context, const uint8_t * request, size_t request_size,
        uint8_t * response, size_t response_capacity, size_t * response_size);

private:
    std::string handle(const uint8_t * request, size_t size);
    std::string model_path_;
    std::mutex mutex_; // serialize expensive N/A graph probes
    bool identity_ready_ = false;
    std::string model_sha256_;
    std::string exe_sha256_;
    std::string runtime_sha256_;
    std::string device_id_;
    std::string library_fingerprint_json_;
    std::string device_json_;
    void ensure_identity();
};
