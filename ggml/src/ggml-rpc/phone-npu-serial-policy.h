#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>

// Opt-in Phone-local *serial* Region selection. These are the same Region
// indices used by the existing PC scheduler and P2C frames. No extra RPC
// device is exposed, and GPU/NPU work is never issued by separate compute
// threads. The value is intentionally explicit: never auto-route a Region
// to HTP based on estimated performance.
namespace ggml_phone_npu_serial {

inline bool parse_regions(const char * raw, std::set<uint32_t> & regions) {
    regions.clear();
    if (raw == nullptr || *raw == '\0') {
        return true;  // GPU-only compatibility mode
    }
    const char * p = raw;
    for (;;) {
        if (!std::isdigit(static_cast<unsigned char>(*p))) {
            return false;
        }
        char * end = nullptr;
        const unsigned long value = std::strtoul(p, &end, 10);
        if (end == p || value >= 64 ||
                !regions.insert(static_cast<uint32_t>(value)).second) {
            return false;
        }
        if (*end == '\0') {
            return true;
        }
        if (*end != ',') {
            return false;
        }
        p = end + 1;
    }
}

inline bool htp_region(const std::set<uint32_t> & regions, uint32_t region) {
    return regions.find(region) != regions.end();
}

// Ranks 0 and 1 have resident HTP weights in the two physical Banks.
// Rank 2+ is only a planned GGUF layer until the Bank Mover refills it.
// Its registration-time Graph shape check must use the saved GGUF tensor
// metadata, never require absent HTP weight tensor allocations.
inline bool use_deferred_weight_metadata(bool banked, int rank, bool bank_ready) {
    return banked && rank >= 2 && !bank_ready;
}

// Match a numbered Transformer layer without confusing e.g. blk.1 with
// blk.10. The tokenizer format is the ggml weight naming used by llama.cpp.
inline int layer_from_tensor_name(const char * name) {
    if (name == nullptr) {
        return -1;
    }
    std::string s(name);
    const size_t at = s.find("blk.");
    if (at == std::string::npos) {
        return -1;
    }
    const size_t first = at + 4;
    size_t pos = first;
    while (pos < s.size() &&
            std::isdigit(static_cast<unsigned char>(s[pos]))) {
        ++pos;
    }
    if (pos == first || pos >= s.size() || s[pos] != '.') {
        return -1;
    }
    const long layer = std::strtol(s.substr(first, pos - first).c_str(),
                                   nullptr, 10);
    return layer >= 0 && layer < 4096 ? static_cast<int>(layer) : -1;
}

}  // namespace ggml_phone_npu_serial
