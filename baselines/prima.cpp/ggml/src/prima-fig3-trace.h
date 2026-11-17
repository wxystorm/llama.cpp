// Optional, append-only per-rank telemetry for the Prima Fig.3 experiment.
// All timestamps use CLOCK_MONOTONIC / steady_clock (nanoseconds). This
// header does not alter the planner, the ring order or normal measurements.
#ifndef PRIMA_FIG3_TRACE_H
#define PRIMA_FIG3_TRACE_H

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace prima_fig3 {

inline bool enabled() {
    const char * path = std::getenv("PRIMA_FIG3_FILE");
    return path != nullptr && path[0] != '\0';
}

inline int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int rank() {
    const char * value = std::getenv("PRIMA_FIG3_RANK");
    return value ? std::atoi(value) : -1;
}

// Each line is emitted with ONE append write; independent static instances
// across shared libraries cannot corrupt one another by interleaving offsets.
// Category/kind/detail are source-controlled identifiers, not free-form text.
inline void record(const char * category, const char * kind,
                   int64_t start_ns, int64_t end_ns, int64_t nbytes = 0,
                   const char * detail = "") {
    if (!enabled() || start_ns <= 0 || end_ns < start_ns) return;
    const char * path = std::getenv("PRIMA_FIG3_FILE");
    const char * run  = std::getenv("PRIMA_FIG3_RUN_ID");
    if (!run || !*run || !category || !kind || !detail) return;
    char line[768];
    const int count = std::snprintf(line, sizeof(line),
        "%s,%d,%lld,%lld,%lld,%s,%s,%lld,%s\n",
        run, rank(),
#if defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
        static_cast<long long>(getpid()),
#else
        static_cast<long long>(0),
#endif
        static_cast<long long>(start_ns),
        static_cast<long long>(end_ns), category, kind,
        static_cast<long long>(nbytes), detail);
    if (count <= 0 || count >= static_cast<int>(sizeof(line))) return;
#if defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) {
        (void) write(fd, line, static_cast<size_t>(count));
        (void) close(fd);
    }
#else
    std::FILE * fp = std::fopen(path, "ab");
    if (fp) { (void) std::fwrite(line, 1, static_cast<size_t>(count), fp); std::fclose(fp); }
#endif
}

class scope {
public:
    scope(const char * category, const char * kind, int64_t nbytes = 0,
          const char * detail = "")
        : category_(category), kind_(kind), nbytes_(nbytes), detail_(detail),
          start_(enabled() ? now_ns() : 0) {}
    ~scope() {
        if (start_) record(category_, kind_, start_, now_ns(), nbytes_, detail_);
    }
    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;
private:
    const char * category_;
    const char * kind_;
    int64_t nbytes_;
    const char * detail_;
    int64_t start_;
};

} // namespace prima_fig3
#endif // PRIMA_FIG3_TRACE_H
