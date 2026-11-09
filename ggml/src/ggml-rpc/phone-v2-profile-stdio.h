#pragma once

// Single-port V2 Profile adapter for the existing ggml-rpc-server.
// Android RPC process owns TCP :50052. This adapter launches an internal
// long-lived stdio worker only on demand; it does NOT bind a second port.
// All GGUF/Profiler paths are server-owned. The worker validates every shape,
// verifies HTP parity, and uses the existing independent native Stage profiler.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <cerrno>
#include <csignal>
#include <filesystem>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

class phone_v2_profile_stdio {
public:
    explicit phone_v2_profile_stdio(const char * model_path)
        : model_path_(model_path ? model_path : "") {}

    phone_v2_profile_stdio(const phone_v2_profile_stdio &) = delete;
    phone_v2_profile_stdio & operator=(const phone_v2_profile_stdio &) = delete;

    ~phone_v2_profile_stdio() {
        std::lock_guard<std::mutex> guard(mutex_);
        stop_worker();
    }

    // A single bound JSON request/response, transported as an ordinary RPC
    // command AFTER the normal HELLO. Never accepts an arbitrary executable
    // or GGUF path from the network.
    bool request(const std::vector<uint8_t> & bytes, std::vector<uint8_t> & result) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (bytes.empty() || bytes.size() > max_frame ||
                bytes.back() != '\n' ||
                std::find(bytes.begin(), bytes.end() - 1, '\n') != bytes.end() - 1) {
            error(result, "BAD_PROFILE_REQUEST_FRAME");
            return true;
        }
#if defined(__ANDROID__)
        if (model_path_.empty()) {
            error(result, "RPC_SERVER_REQUIRES_MODEL_PATH");
            return true;
        }
        if (!read_ || !write_) {
            if (!start_worker()) {
                error(result, "PHONE_PROFILE_WORKER_START_FAILED");
                return true;
            }
        }
        if (std::fwrite(bytes.data(), 1, bytes.size(), write_) != bytes.size() ||
                std::fflush(write_) != 0) {
            stop_worker();
            error(result, "PHONE_PROFILE_WORKER_WRITE_FAILED");
            return true;
        }
        // Worker emits exactly one bounded JSON line. Its subprocess timeout
        // is enforced in v2_phone_profile_server.py, not by a socket timeout.
        std::vector<char> line(max_frame + 2, 0);
        if (!std::fgets(line.data(), (int) line.size(), read_)) {
            stop_worker();
            error(result, "PHONE_PROFILE_WORKER_READ_FAILED");
            return true;
        }
        const size_t len = std::strlen(line.data());
        if (len < 2 || len > max_frame || line[len - 1] != '\n') {
            stop_worker();
            error(result, "PHONE_PROFILE_WORKER_BAD_RESPONSE");
            return true;
        }
        result.assign(line.data(), line.data() + len);
#else
        error(result, "PHONE_PROFILE_ONLY_SUPPORTED_ON_ANDROID");
#endif
        return true;
    }

    static constexpr size_t max_frame = 256 * 1024;

private:
    static void error(std::vector<uint8_t> & out, const char * reason) {
        const std::string json = std::string(
            "{\"version\":1,\"status\":\"ERROR\",\"error\":\"") +
            reason + "\"}\n";
        out.assign(json.begin(), json.end());
    }

#if defined(__ANDROID__)
    bool start_worker() {
        char executable[4096] = {};
        const ssize_t path_length = readlink("/proc/self/exe",
                                              executable, sizeof(executable) - 1);
        if (path_length <= 0) {
            return false;
        }
        executable[path_length] = '\0';
        const std::filesystem::path root =
            std::filesystem::path(executable).parent_path().parent_path();
        const char * script_env = std::getenv("GGML_RPC_V2_PROFILE_SCRIPT");
        const char * exe_env = std::getenv("GGML_RPC_V2_PROFILE_EXE");
        const std::string script = script_env && *script_env ? script_env :
            (root / "v2_phone_profile_server.py").string();
        const std::string profiler = exe_env && *exe_env ? exe_env :
            (root / "bin/region-ring-v2-stage-profile").string();
        if (!std::filesystem::is_regular_file(script) ||
                !std::filesystem::is_regular_file(profiler) ||
                !std::filesystem::is_regular_file(model_path_)) {
            std::fprintf(stderr,
                "[V2_PROFILE_RPC] status=UNAVAILABLE reason=PROFILE_FILES_MISSING "
                "script=%s profiler=%s\n", script.c_str(), profiler.c_str());
            return false;
        }
        int parent_to_child[2] = {-1, -1};
        int child_to_parent[2] = {-1, -1};
        if (pipe(parent_to_child) != 0) {
            return false;
        }
        if (pipe(child_to_parent) != 0) {
            close(parent_to_child[0]);
            close(parent_to_child[1]);
            return false;
        }
        pid_t child = fork();
        if (child == 0) {
            // In a multi-threaded RPC process, perform only async-signal-safe
            // operations between fork and exec.
            dup2(parent_to_child[0], STDIN_FILENO);
            dup2(child_to_parent[1], STDOUT_FILENO);
            close(parent_to_child[0]);
            close(parent_to_child[1]);
            close(child_to_parent[0]);
            close(child_to_parent[1]);
            execlp("python", "python", "-u", script.c_str(),
                   "--model", model_path_.c_str(), "--exe",
                   profiler.c_str(), "--stdio", (char *) nullptr);
            _exit(127);
        }
        close(parent_to_child[0]);
        close(child_to_parent[1]);
        if (child < 0) {
            close(parent_to_child[1]);
            close(child_to_parent[0]);
            return false;
        }
        write_ = fdopen(parent_to_child[1], "w");
        read_ = fdopen(child_to_parent[0], "r");
        child_ = child;
        if (!write_ || !read_) {
            if (write_) { std::fclose(write_); write_ = nullptr; }
            else { close(parent_to_child[1]); }
            if (read_) { std::fclose(read_); read_ = nullptr; }
            else { close(child_to_parent[0]); }
            kill(child_, SIGTERM);
            waitpid(child_, nullptr, 0);
            child_ = -1;
            return false;
        }
        std::fprintf(stderr,
            "[V2_PROFILE_RPC] status=WORKER_STARTED port=SHARED_50052\n");
        return true;
    }

    void stop_worker() {
        if (write_) { std::fclose(write_); write_ = nullptr; }
        if (read_) { std::fclose(read_); read_ = nullptr; }
        if (child_ > 0) {
            kill(child_, SIGTERM);
            waitpid(child_, nullptr, 0);
            child_ = -1;
        }
    }
    FILE * read_ = nullptr;
    FILE * write_ = nullptr;
    pid_t child_ = -1;
#else
    void stop_worker() {}
#endif
    std::mutex mutex_;
    std::string model_path_;
};
