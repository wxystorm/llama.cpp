#pragma once

// Single-port V2 Profile adapter for ggml-rpc-server.
// A server-owned, on-demand local Python worker has NO listening socket:
// only the existing inference RPC port 50052 is exposed. It dispatches
// bounded requests to the existing native N/A graph profiler, preserving
// model fingerprint and HTP parity checks.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#if defined(__ANDROID__)
#include <cerrno>
#include <csignal>
#include <filesystem>
#include <sys/socket.h>
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

    static constexpr size_t max_frame = 256 * 1024;

    // Wire: one JSON line in, one JSON line out. All requests are
    // serialized, even across concurrent inference RPC client sockets.
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
        if (worker_fd_ < 0 && !start_worker()) {
            error(result, "PHONE_PROFILE_WORKER_START_FAILED");
            return true;
        }
        size_t sent = 0;
        while (sent < bytes.size()) {
            // MSG_NOSIGNAL: a crashed child must never kill the inference
            // RPC server with SIGPIPE.
            const ssize_t n = send(worker_fd_, bytes.data() + sent,
                                    bytes.size() - sent, MSG_NOSIGNAL);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                const int saved_errno = n < 0 ? errno : 0;
                const std::string detail = worker_failure_detail(
                    "PHONE_PROFILE_WORKER_WRITE_FAILED", saved_errno);
                stop_worker();
                error(result, detail.c_str());
                return true;
            }
            sent += (size_t) n;
        }
        result.clear();
        std::array<uint8_t, 4096> buffer {};
        while (result.size() < max_frame) {
            const size_t wanted = std::min(buffer.size(),
                                           max_frame - result.size());
            const ssize_t n = recv(worker_fd_, buffer.data(), wanted, 0);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                const int saved_errno = n < 0 ? errno : 0;
                const std::string detail = worker_failure_detail(
                    "PHONE_PROFILE_WORKER_READ_FAILED", saved_errno);
                stop_worker();
                error(result, detail.c_str());
                return true;
            }
            const auto end = buffer.begin() + n;
            const auto newline = std::find(buffer.begin(), end, uint8_t('\n'));
            if (newline != end && newline + 1 != end) {
                stop_worker();
                error(result, "PHONE_PROFILE_WORKER_EXTRA_RESPONSE_DATA");
                return true;
            }
            result.insert(result.end(), buffer.begin(), end);
            if (newline != end) {
                return true;
            }
        }
        stop_worker();
        error(result, "PHONE_PROFILE_WORKER_RESPONSE_TOO_LARGE");
#else
        error(result, "PHONE_PROFILE_ONLY_SUPPORTED_ON_ANDROID");
#endif
        return true;
    }

private:
    static void error(std::vector<uint8_t> & out, const char * reason) {
        const std::string json = std::string(
            "{\"version\":1,\"status\":\"ERROR\",\"error\":\"") +
            reason + "\"}\n";
        out.assign(json.begin(), json.end());
    }

#if defined(__ANDROID__)
    std::string worker_failure_detail(const char * failure, int io_errno) {
        int status = 0;
        const pid_t waited = child_ > 0 ? waitpid(child_, &status, WNOHANG) : -1;
        std::string detail(failure);
        if (waited == child_) {
            child_ = -1; // Already reaped: stop_worker must not wait twice.
            if (WIFEXITED(status)) {
                detail += "_EXIT_" + std::to_string(WEXITSTATUS(status));
            } else if (WIFSIGNALED(status)) {
                detail += "_SIGNAL_" + std::to_string(WTERMSIG(status));
            }
        }
        std::fprintf(stderr,
            "[V2_PROFILE_RPC] event=WORKER_FAILURE reason=%s errno=%d "
            "child_wait=%ld status=%d\n", detail.c_str(), io_errno,
            (long) waited, status);
        return detail;
    }

    bool start_worker() {
        char pathname[4096] = {};
        const ssize_t n = readlink("/proc/self/exe",
                                   pathname, sizeof(pathname) - 1);
        if (n <= 0) {
            return false;
        }
        pathname[n] = '\0';
        const std::filesystem::path root =
            std::filesystem::path(pathname).parent_path().parent_path();
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
        int socket_pair[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, socket_pair) != 0) {
            return false;
        }
        const pid_t child = fork();
        if (child == 0) {
            // In the multi-threaded RPC process use only async-signal-safe
            // functions between fork and exec.
            dup2(socket_pair[1], STDIN_FILENO);
            dup2(socket_pair[1], STDOUT_FILENO);
            close(socket_pair[0]);
            close(socket_pair[1]);
            // The worker is an internal subprocess, not another TCP server.
            // Termux may expose either python or python3. Allow an explicit
            // phone-local interpreter override, never an RPC-supplied path.
            const char * configured =
                std::getenv("GGML_RPC_V2_PROFILE_PYTHON");
            if (configured != nullptr && *configured) {
                execlp(configured, configured, "-u", script.c_str(),
                       "--model", model_path_.c_str(), "--exe",
                       profiler.c_str(), "--stdio", (char *) nullptr);
            } else {
                execlp("python", "python", "-u", script.c_str(),
                       "--model", model_path_.c_str(), "--exe",
                       profiler.c_str(), "--stdio", (char *) nullptr);
                execlp("python3", "python3", "-u", script.c_str(),
                       "--model", model_path_.c_str(), "--exe",
                       profiler.c_str(), "--stdio", (char *) nullptr);
            }
            // exec failed; child stderr is captured by the RPC server log.
            // Use write() only: the RPC server may have other threads at fork.
            static const char message[] =
                "[V2_PROFILE_RPC] event=WORKER_EXEC_FAILED reason=PYTHON_UNAVAILABLE_OR_NOT_EXECUTABLE\n";
            (void) write(STDERR_FILENO, message, sizeof(message) - 1);
            _exit(127);
        }
        close(socket_pair[1]);
        if (child < 0) {
            close(socket_pair[0]);
            return false;
        }
        worker_fd_ = socket_pair[0];
        child_ = child;
        std::fprintf(stderr,
            "[V2_PROFILE_RPC] status=WORKER_STARTED port=SHARED_50052\n");
        return true;
    }

    void stop_worker() {
        if (worker_fd_ >= 0) {
            shutdown(worker_fd_, SHUT_RDWR);
            close(worker_fd_);
            worker_fd_ = -1;
        }
        if (child_ > 0) {
            kill(child_, SIGTERM);
            waitpid(child_, nullptr, 0);
            child_ = -1;
        }
    }
    int worker_fd_ = -1;
    pid_t child_ = -1;
#else
    void stop_worker() {}
#endif
    std::mutex mutex_;
    std::string model_path_;
};
