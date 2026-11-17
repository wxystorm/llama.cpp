#pragma once

// Jupiter Fig.3 native instrumentation. Completely opt-in:
//   PC:    LLAMA_JUPITER_FIG3_DIR=existing-or-new-directory
//   Phone: LLAMA_JUPITER_FIG3_PHONE_FILE=/writable/path.csv
//
// Steady-clock timestamps (nanoseconds) are comparable between local modules.
// Phone timestamps MUST be calibrated to PC using JUPITER_FIG3_CLOCK_SYNC.
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>

namespace jupiter_fig3 {
using clock = std::chrono::steady_clock;

inline int64_t ns(clock::time_point t = clock::now()) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}
inline bool pc_enabled() {
    const auto * p = std::getenv("LLAMA_JUPITER_FIG3_DIR");
    return p && *p;
}
inline bool phone_enabled() {
    const auto * p = std::getenv("LLAMA_JUPITER_FIG3_PHONE_FILE");
    return p && *p;
}
inline bool append_file(const std::filesystem::path & path,
                        const std::string & header, const std::string & line) {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    try {
        std::filesystem::create_directories(path.parent_path());
        std::error_code ec;
        bool create_header = !std::filesystem::exists(path, ec) ||
                             std::filesystem::file_size(path, ec) == 0;
        std::ofstream out(path, std::ios::app);
        if (!out) return false;
        if (create_header) out << header << '\n';
        if (!line.empty()) out << line << '\n';
        return bool(out);
    } catch (...) {
        return false;
    }
}
inline void pc_row(const char * filename, const char * header,
                   const std::string & data = "") {
    const auto * dir = std::getenv("LLAMA_JUPITER_FIG3_DIR");
    if (!dir || !*dir) return;
    append_file(std::filesystem::path(dir) / filename, header, data);
}
inline void device(const char * file, const char * category,
                   int64_t begin_ns, int64_t end_ns, const char * source,
                   int device_id = -1, uint64_t bytes = 0) {
    if (!pc_enabled() || end_ns <= begin_ns) return;
    std::ostringstream row;
    row << category << ',' << begin_ns << ',' << end_ns << ','
        << source << ',' << device_id << ',' << bytes;
    pc_row(file, "category,start_ns,end_ns,source,device,bytes", row.str());
}
inline void window(int64_t begin_ns, int64_t end_ns, int tokens) {
    if (end_ns <= begin_ns) return;
    std::ostringstream row;
    row << begin_ns << ',' << end_ns << ',' << tokens;
    pc_row("pc_prefill_windows.csv", "start_ns,end_ns,tokens", row.str());
}
} // namespace jupiter_fig3
