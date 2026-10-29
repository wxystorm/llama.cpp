#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama_jupiter {

struct plan_data {
    int pc_layers = -1;
    std::unordered_map<uint32_t, std::vector<uint32_t>> sequence;
};

inline bool enabled() {
    const char * value = std::getenv("LLAMA_JUPITER_PREFILL");
    return value != nullptr && std::atoi(value) != 0;
}

inline std::string trim(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

inline std::vector<uint32_t> parse_u32_list(const std::string & value) {
    std::vector<uint32_t> result;
    std::stringstream ss(value);
    std::string item;

    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            continue;
        }
        char * end = nullptr;
        const unsigned long parsed = std::strtoul(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
            return {};
        }
        result.push_back((uint32_t) parsed);
    }

    return result;
}

inline plan_data load_plan_file() {
    plan_data result;

    const char * path = std::getenv("LLAMA_JUPITER_PLAN_FILE");
    if (path == nullptr || path[0] == '\0') {
        return result;
    }

    std::ifstream in(path);
    if (!in.good()) {
        return result;
    }

    std::string line;
    while (std::getline(in, line)) {
        const size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line.resize(hash);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const size_t eq = line.find('=');
        if (eq == std::string::npos) {
            continue;
        }

        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));

        if (key == "model.pc_layers") {
            result.pc_layers = std::atoi(value.c_str());
            continue;
        }

        static const std::string prefix = "sequence.";
        if (key.rfind(prefix, 0) == 0) {
            const std::string total_str = key.substr(prefix.size());
            char * end = nullptr;
            const unsigned long total = std::strtoul(total_str.c_str(), &end, 10);
            if (end == total_str.c_str() || *end != '\0' || total == 0 || total > UINT32_MAX) {
                continue;
            }

            auto chunks = parse_u32_list(value);
            if (!chunks.empty()) {
                result.sequence[(uint32_t) total] = std::move(chunks);
            }
        }
    }

    return result;
}

inline int model_pc_layers(int fallback) {
    if (const char * value = std::getenv("LLAMA_JUPITER_PC_LAYERS")) {
        return std::atoi(value);
    }

    const auto plan = load_plan_file();
    return plan.pc_layers > 0 ? plan.pc_layers : fallback;
}

inline std::vector<uint32_t> sequence_plan(uint32_t total_tokens, uint32_t max_ubatch) {
    if (!enabled() || total_tokens <= 1) {
        return {};
    }

    std::vector<uint32_t> chunks;

    // Explicit per-run override. It is applied only when the sum exactly matches
    // this batch, so warm-up/decode batches are unaffected.
    if (const char * value = std::getenv("LLAMA_JUPITER_SEQUENCE_PLAN")) {
        auto candidate = parse_u32_list(value);
        uint64_t sum = 0;
        for (uint32_t x : candidate) {
            sum += x;
        }
        if (!candidate.empty() && sum == total_tokens) {
            chunks = std::move(candidate);
        }
    }

    if (chunks.empty()) {
        const auto plan = load_plan_file();
        auto it = plan.sequence.find(total_tokens);
        if (it != plan.sequence.end()) {
            chunks = it->second;
        }
    }

    if (chunks.empty()) {
        return {};
    }

    uint64_t sum = 0;
    for (uint32_t x : chunks) {
        if (x == 0 || x > max_ubatch) {
            return {};
        }
        sum += x;
    }

    if (sum != total_tokens) {
        return {};
    }

    return chunks;
}

inline std::string format_chunks(const std::vector<uint32_t> & chunks) {
    std::string result;
    for (size_t i = 0; i < chunks.size(); ++i) {
        if (i != 0) {
            result += ",";
        }
        result += std::to_string(chunks[i]);
    }
    return result;
}

} // namespace llama_jupiter
