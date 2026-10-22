#pragma once

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

// V1: a repeating PC -> Phone -> PC ownership cycle. This is a placement
// description only; the two one-way Tensor executors are implemented separately.
enum class llama_hybrid_cycle_layer_kind {
    PC_CPU,
    TENSOR_C2P,
    PHONE_GPU,
    TENSOR_P2C,
    PC_CUDA_TAIL,
};

enum class llama_hybrid_cycle_owner {
    PC_CPU,
    PHONE_GPU,
    PC_CUDA,
};

inline const char * llama_hybrid_cycle_kind_name(llama_hybrid_cycle_layer_kind kind) {
    switch (kind) {
        case llama_hybrid_cycle_layer_kind::PC_CPU:       return "PC_CPU";
        case llama_hybrid_cycle_layer_kind::TENSOR_C2P:   return "TENSOR_C2P";
        case llama_hybrid_cycle_layer_kind::PHONE_GPU:    return "PHONE_GPU";
        case llama_hybrid_cycle_layer_kind::TENSOR_P2C:   return "TENSOR_P2C";
        case llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL: return "PC_CUDA_TAIL";
    }
    return "UNKNOWN";
}

inline const char * llama_hybrid_cycle_owner_name(llama_hybrid_cycle_owner owner) {
    switch (owner) {
        case llama_hybrid_cycle_owner::PC_CPU:    return "PC_CPU";
        case llama_hybrid_cycle_owner::PHONE_GPU: return "PHONE_GPU";
        case llama_hybrid_cycle_owner::PC_CUDA:   return "PC_CUDA";
    }
    return "UNKNOWN";
}

inline llama_hybrid_cycle_owner llama_hybrid_cycle_attn_owner(llama_hybrid_cycle_layer_kind kind) {
    switch (kind) {
        case llama_hybrid_cycle_layer_kind::PC_CPU:
        case llama_hybrid_cycle_layer_kind::TENSOR_C2P:
            return llama_hybrid_cycle_owner::PC_CPU;
        case llama_hybrid_cycle_layer_kind::PHONE_GPU:
        case llama_hybrid_cycle_layer_kind::TENSOR_P2C:
            return llama_hybrid_cycle_owner::PHONE_GPU;
        case llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL:
            return llama_hybrid_cycle_owner::PC_CUDA;
    }
    return llama_hybrid_cycle_owner::PC_CPU;
}

inline llama_hybrid_cycle_owner llama_hybrid_cycle_output_owner(llama_hybrid_cycle_layer_kind kind) {
    switch (kind) {
        case llama_hybrid_cycle_layer_kind::PC_CPU:
        case llama_hybrid_cycle_layer_kind::TENSOR_P2C:
            return llama_hybrid_cycle_owner::PC_CPU;
        case llama_hybrid_cycle_layer_kind::PHONE_GPU:
        case llama_hybrid_cycle_layer_kind::TENSOR_C2P:
            return llama_hybrid_cycle_owner::PHONE_GPU;
        case llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL:
            return llama_hybrid_cycle_owner::PC_CUDA;
    }
    return llama_hybrid_cycle_owner::PC_CPU;
}

struct llama_hybrid_cycle_layout {
    int x = 0;
    int y = 0;
    int cycle_size = 0;
    int cycles = 0;
    int tail_begin = 0;
    // Opt-in execution policy for PC full layers in the real Cycle period:
    // PC CUDA Attention/Router -> PC CPU FFN -> PC CPU residual Reduce.
    // Placement kinds remain unchanged; the default path stays CPU-only.
    bool pc_cuda_attn = false;
    std::vector<llama_hybrid_cycle_layer_kind> kinds;

    bool enabled() const { return !kinds.empty(); }
    int remainder() const { return int(kinds.size()) - tail_begin; }
};

inline llama_hybrid_cycle_layout llama_hybrid_build_cycle_layout(int n_layers, int x, int y) {
    const int64_t period = int64_t(x) + y + 2;
    if (n_layers <= 0 || x <= 0 || y <= 0 || period > n_layers) {
        throw std::invalid_argument("cycle layout requires x>=1, y>=1 and x+y+2<=n_layers");
    }

    llama_hybrid_cycle_layout result;
    result.x = x;
    result.y = y;
    result.cycle_size = int(period);
    result.cycles = n_layers / result.cycle_size;
    result.tail_begin = result.cycles * result.cycle_size;
    result.kinds.reserve(n_layers);

    for (int c = 0; c < result.cycles; ++c) {
        result.kinds.insert(result.kinds.end(), x, llama_hybrid_cycle_layer_kind::PC_CPU);
        result.kinds.push_back(llama_hybrid_cycle_layer_kind::TENSOR_C2P);
        result.kinds.insert(result.kinds.end(), y, llama_hybrid_cycle_layer_kind::PHONE_GPU);
        result.kinds.push_back(llama_hybrid_cycle_layer_kind::TENSOR_P2C);
    }
    result.kinds.insert(result.kinds.end(), n_layers - result.tail_begin,
                        llama_hybrid_cycle_layer_kind::PC_CUDA_TAIL);
    return result;
}

inline bool llama_hybrid_cycle_requested() {
    return std::getenv("LLAMA_HYBRID_CYCLE_PC_LAYERS") != nullptr ||
           std::getenv("LLAMA_HYBRID_CYCLE_PHONE_LAYERS") != nullptr;
}

inline int llama_hybrid_cycle_parse_positive(const char * name, const char * value) {
    if (value == nullptr || *value == '\0') {
        throw std::invalid_argument(name);
    }
    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' || parsed <= 0 || parsed > INT_MAX) {
        throw std::invalid_argument(name);
    }
    return int(parsed);
}

inline bool llama_hybrid_cycle_layout_from_env(int n_layers, llama_hybrid_cycle_layout & result) {
    const char * pc = std::getenv("LLAMA_HYBRID_CYCLE_PC_LAYERS");
    const char * phone = std::getenv("LLAMA_HYBRID_CYCLE_PHONE_LAYERS");
    if (pc == nullptr && phone == nullptr) {
        return false;
    }
    const int x = llama_hybrid_cycle_parse_positive("LLAMA_HYBRID_CYCLE_PC_LAYERS must be a positive integer", pc);
    const int y = llama_hybrid_cycle_parse_positive("LLAMA_HYBRID_CYCLE_PHONE_LAYERS must be a positive integer", phone);
    result = llama_hybrid_build_cycle_layout(n_layers, x, y);
    return true;
}
