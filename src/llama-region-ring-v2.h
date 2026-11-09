#pragma once

// Region Ring V2: a hardware-agnostic description of physical whole-layer
// Stages. It has NO dependency on GGML, RPC, the old G/C/P ring scheduler,
// Phone graph templates, or a particular final backend.
//
// A Cycle is an ordered SUBSEQUENCE of G -> C -> N -> A. Zero depths do not
// create phantom Regions. A final partial Cycle is allowed, and the final
// Transformer Stage is NOT required to be on the Phone. The CPU output Head
// is deliberately outside this Transformer layout.
//
// This header is the shared contract for V2 profiling, planning and runtime.
// The legacy Region Ring retains its original behavior during migration.
// Use a path relative to this header: standalone Planner tests do not
// all add ggml/include to their compiler include directories.
#include "../ggml/include/ggml-phone-npu-budget.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace llama_ring_v2 {

inline constexpr uint32_t plan_version = 2;
inline constexpr uint32_t profile_version = 1;

enum class backend : uint8_t {
    G = 0,  // PC CUDA
    C = 1,  // PC CPU
    N = 2,  // Phone HTP NPU
    A = 3,  // Phone Adreno OpenCL
};

inline constexpr bool is_phone(backend b) {
    return b == backend::N || b == backend::A;
}

inline constexpr char backend_letter(backend b) {
    return b == backend::G ? 'G' :
           b == backend::C ? 'C' :
           b == backend::N ? 'N' : 'A';
}

struct cycle {
    int g = 0;
    int c = 0;
    int n = 0;
    int a = 0;
};

struct plan_spec {
    uint32_t version = plan_version;
    int model_layers = 0;
    uint32_t chunk_tokens = 0;       // Shared GPU/NPU/Adreno Chunk size (X)
    uint32_t cpu_chunk_tokens = 0;   // 0 means same X; optional XC
    std::vector<cycle> cycles;
};

struct stage {
    uint32_t id = 0;
    uint32_t cycle_id = 0;
    backend device = backend::G;
    int layer_begin = 0;  // inclusive
    int layer_end = 0;    // exclusive
    uint32_t chunk_tokens = 0;

    int depth() const { return layer_end - layer_begin; }
};

enum class handoff : uint8_t {
    SAME_DEVICE,
    GPU_TO_CPU,       // CUDA D2H to CPU
    CPU_TO_GPU,       // CPU H2D to CUDA
    PC_TO_PHONE,      // Runtime C2P
    PHONE_TO_PC,      // Runtime P2C
    HTP_TO_ADRENO,    // Same Phone Host activation, plus OpenCL upload
    ADRENO_TO_HTP,    // Requires measured local transfer
    PHONE_LOCAL,      // Other local Phone path; do NOT assume zero cost
};

inline constexpr handoff classify_handoff(backend src, backend dst) {
    if (src == dst) { return handoff::SAME_DEVICE; }
    if (src == backend::G && dst == backend::C) {
        return handoff::GPU_TO_CPU;
    }
    if (src == backend::C && dst == backend::G) {
        return handoff::CPU_TO_GPU;
    }
    if (is_phone(src) && !is_phone(dst)) {
        return handoff::PHONE_TO_PC;
    }
    if (!is_phone(src) && is_phone(dst)) {
        return handoff::PC_TO_PHONE;
    }
    if (src == backend::N && dst == backend::A) {
        return handoff::HTP_TO_ADRENO;
    }
    if (src == backend::A && dst == backend::N) {
        return handoff::ADRENO_TO_HTP;
    }
    return handoff::PHONE_LOCAL;
}

struct boundary {
    uint32_t from = 0; // indexes stages, never phantom zero-depth Stages
    uint32_t to = 0;
    handoff kind = handoff::SAME_DEVICE;
};

struct compiled_plan {
    std::vector<stage> stages;
    std::vector<boundary> boundaries;
    std::string error;

    bool valid() const { return error.empty() && !stages.empty(); }
    backend final_backend() const { return stages.back().device; }
    bool final_head_needs_d2h() const { return final_backend() == backend::G; }
    bool final_head_needs_p2c() const { return is_phone(final_backend()); }
};

inline compiled_plan compile(const plan_spec & p) {
    compiled_plan result {};
    if (p.version != plan_version) {
        result.error = "UNSUPPORTED_PLAN_VERSION";
        return result;
    }
    if (p.model_layers <= 0 || p.chunk_tokens == 0 || p.cycles.empty()) {
        result.error = "INVALID_PLAN_SHAPE";
        return result;
    }
    // Keep descriptors small enough for the current Region indexing ABI.
    if (p.cycles.size() > 256) {
        result.error = "TOO_MANY_CYCLES";
        return result;
    }
    int64_t next = 0;
    for (std::size_t ci = 0; ci < p.cycles.size(); ++ci) {
        const cycle & c = p.cycles[ci];
        const std::array<int, 4> depths {c.g, c.c, c.n, c.a};
        bool occupied = false;
        for (std::size_t di = 0; di < depths.size(); ++di) {
            const int count = depths[di];
            if (count < 0) {
                result.error = "NEGATIVE_STAGE_DEPTH";
                return result;
            }
            if (count == 0) {
                continue;
            }
            occupied = true;
            if (next + count > p.model_layers || result.stages.size() >= 256) {
                result.error = "STAGE_RANGE_EXCEEDS_MODEL";
                return result;
            }
            stage s {};
            s.id = static_cast<uint32_t>(result.stages.size());
            s.cycle_id = static_cast<uint32_t>(ci);
            s.device = static_cast<backend>(di);
            s.layer_begin = static_cast<int>(next);
            next += count;
            s.layer_end = static_cast<int>(next);
            s.chunk_tokens = s.device == backend::C && p.cpu_chunk_tokens != 0 ?
                p.cpu_chunk_tokens : p.chunk_tokens;
            result.stages.push_back(s);
        }
        if (!occupied) {
            result.error = "EMPTY_CYCLE";
            return result;
        }
    }
    if (next != p.model_layers) {
        result.error = "INCOMPLETE_MODEL_COVERAGE";
        return result;
    }
    for (std::size_t i = 1; i < result.stages.size(); ++i) {
        result.boundaries.push_back({
            static_cast<uint32_t>(i - 1),
            static_cast<uint32_t>(i),
            classify_handoff(result.stages[i - 1].device,
                             result.stages[i].device)
        });
    }
    return result;
}

// Select the physical HTP weight policy from the complete Plan, rather
// than assuming that Region count > 2 implies weight streaming.  We budget
// total shared Phone memory separately from the maximum HTP allocation.
// KV and resident Adreno storage must not be evicted by weight refills.
enum class npu_weight_policy : uint8_t {
    DISABLED = 0,
    RESIDENT = 1,
    STREAMING_2BANK = 2,
};

inline const char * npu_weight_policy_name(npu_weight_policy p) {
    switch (p) {
        case npu_weight_policy::DISABLED:        return "DISABLED";
        case npu_weight_policy::RESIDENT:        return "RESIDENT";
        case npu_weight_policy::STREAMING_2BANK: return "STREAMING_2BANK";
    }
    return "INVALID";
}

struct npu_memory_request {
    std::vector<size_t> stage_weight_bytes; // actual HTP-repacked bytes per N stage
    // Optional per-stage KV + activation + graph workspace, estimated at
    // the same Chunk/KV geometry as the selected Plan. This is the per-cycle
    // NPU footprint, NOT just its quantized weights.
    std::vector<size_t> stage_auxiliary_bytes;
    size_t npu_kv_bytes = 0;              // total persistent NPU-owned K/V
    size_t npu_graph_workspace_bytes = 0; // peak reusable NPU graph workspace
    size_t npu_activation_bytes = 0;      // persistent NPU activation slots
    size_t host_stage_bytes = 0;          // CPU-side GGUF staging, Phone RAM
    size_t adreno_resident_bytes = 0;     // weights + KV + workspace
    size_t other_shared_bytes = 0;       // other Phone allocations
    size_t phone_budget_bytes = 0;       // aggregate shared RAM budget
    size_t max_htp_allocation_bytes = 0; // actual HTP repack buffer ceiling
    size_t npu_total_budget_bytes = ggml_phone_npu_budget::total_bytes;
    size_t npu_bank_budget_bytes = ggml_phone_npu_budget::bank_bytes;
};

struct npu_memory_decision {
    bool feasible = false;
    npu_weight_policy policy = npu_weight_policy::DISABLED;
    size_t resident_total_bytes = 0;  // all Phone shared memory
    size_t streaming_total_bytes = 0; // all Phone shared memory
    size_t selected_total_bytes = 0;
    size_t resident_npu_bytes = 0;    // NPU weights + KV + workspace + activation
    size_t streaming_npu_bytes = 0;
    size_t npu_auxiliary_bytes = 0;
    std::array<size_t, 2> bank_weight_bytes {};
    size_t all_npu_weight_bytes = 0;
    const char * reason = "INVALID_BUDGET";
};

inline npu_memory_decision choose_npu_weight_policy(
        const npu_memory_request & m) {
    npu_memory_decision d {};
    if (m.phone_budget_bytes == 0 ||
            m.npu_total_budget_bytes == 0 ||
            m.npu_bank_budget_bytes == 0) {
        return d;
    }
    const auto add = [](size_t a, size_t b, size_t & out) -> bool {
        if (b > std::numeric_limits<size_t>::max() - a) {
            return false;
        }
        out = a + b;
        return true;
    };
    size_t phone_fixed = 0;
    for (size_t bytes : {
            m.npu_kv_bytes, m.npu_graph_workspace_bytes,
            m.npu_activation_bytes, m.host_stage_bytes,
            m.adreno_resident_bytes, m.other_shared_bytes}) {
        if (!add(phone_fixed, bytes, phone_fixed)) {
            d.reason = "MEMORY_OVERFLOW";
            return d;
        }
    }
    if (m.stage_weight_bytes.empty()) {
        d.policy = npu_weight_policy::DISABLED;
        d.feasible = phone_fixed <= m.phone_budget_bytes;
        d.resident_total_bytes = phone_fixed;
        d.streaming_total_bytes = phone_fixed;
        d.selected_total_bytes = phone_fixed;
        d.reason = d.feasible ? "NONE" : "PHONE_BUDGET_EXCEEDED";
        return d;
    }
    if (m.max_htp_allocation_bytes == 0) {
        d.reason = "UNKNOWN_HTP_SINGLE_ALLOCATION_LIMIT";
        return d;
    }
    if (!m.stage_auxiliary_bytes.empty() &&
            m.stage_auxiliary_bytes.size() != m.stage_weight_bytes.size()) {
        d.reason = "NPU_STAGE_AUXILIARY_SHAPE";
        return d;
    }
    // NPU-private budget: do not count Adreno's own residency or CPU-side
    // raw GGUF staging. Both remain charged to the Phone RAM budget above.
    for (size_t bytes : {
            m.npu_kv_bytes, m.npu_graph_workspace_bytes,
            m.npu_activation_bytes}) {
        if (!add(d.npu_auxiliary_bytes, bytes, d.npu_auxiliary_bytes)) {
            d.reason = "MEMORY_OVERFLOW";
            return d;
        }
    }
    for (size_t i = 0; i < m.stage_weight_bytes.size(); ++i) {
        const size_t weights = m.stage_weight_bytes[i];
        if (weights == 0 || weights > m.max_htp_allocation_bytes ||
                weights > m.npu_bank_budget_bytes) {
            d.reason = "HTP_STAGE_ALLOCATION_EXCEEDED";
            return d;
        }
        const size_t auxiliary = m.stage_auxiliary_bytes.empty() ?
            0 : m.stage_auxiliary_bytes[i];
        size_t stage_total = 0;
        if (!add(weights, auxiliary, stage_total)) {
            d.reason = "MEMORY_OVERFLOW";
            return d;
        }
        if (stage_total > m.npu_bank_budget_bytes) {
            d.reason = "NPU_STAGE_BUDGET_EXCEEDED";
            return d;
        }
        if (!add(d.all_npu_weight_bytes, weights, d.all_npu_weight_bytes)) {
            d.reason = "MEMORY_OVERFLOW";
            return d;
        }
        d.bank_weight_bytes[i % 2] =
            std::max(d.bank_weight_bytes[i % 2], weights);
    }
    size_t bank_weight_total = 0;
    if (!add(d.bank_weight_bytes[0], d.bank_weight_bytes[1],
             bank_weight_total) ||
            !add(phone_fixed, bank_weight_total, d.streaming_total_bytes) ||
            !add(phone_fixed, d.all_npu_weight_bytes,
                 d.resident_total_bytes) ||
            !add(d.npu_auxiliary_bytes, bank_weight_total,
                 d.streaming_npu_bytes) ||
            !add(d.npu_auxiliary_bytes, d.all_npu_weight_bytes,
                 d.resident_npu_bytes)) {
        d.reason = "MEMORY_OVERFLOW";
        return d;
    }
    // Crucial: if ALL HTP stages fit in 3072 MiB inclusive of NPU
    // non-weight storage, keep every stage resident. No rolling Banks.
    if (d.resident_npu_bytes <= m.npu_total_budget_bytes &&
            d.resident_total_bytes <= m.phone_budget_bytes) {
        d.policy = npu_weight_policy::RESIDENT;
        d.selected_total_bytes = d.resident_total_bytes;
        d.feasible = true;
    } else if (m.stage_weight_bytes.size() >= 2 &&
            d.streaming_npu_bytes <= m.npu_total_budget_bytes &&
            d.streaming_total_bytes <= m.phone_budget_bytes) {
        // Two weight arenas alternate by rank; persistent KV, graph
        // scratch and activation slots stay outside the weight Banks.
        d.policy = npu_weight_policy::STREAMING_2BANK;
        d.selected_total_bytes = d.streaming_total_bytes;
        d.feasible = true;
    }
    d.reason = d.feasible ? "NONE" :
        d.streaming_npu_bytes > m.npu_total_budget_bytes ?
            "NPU_TOTAL_BUDGET_EXCEEDED" : "PHONE_BUDGET_EXCEEDED";
    return d;
}

// A profile request is independent of a final inference Plan. It describes
// exactly the first N layers on ONE backend, with no hidden feeder Stage and
// no output Head. The executor must populate real Position/KV/Mask inputs.
struct profile_request {
    uint32_t version = profile_version;
    backend device = backend::G;
    int model_layers = 0;
    int layers = 0;               // Always [0, layers)
    uint32_t chunk_tokens = 0;
    uint32_t kv_tokens = 0;      // Real attention read span, not a label
    uint32_t warmups = 1;
    uint32_t repeats = 5;
};

inline std::string validate(const profile_request & p) {
    if (p.version != profile_version) {
        return "UNSUPPORTED_PROFILE_VERSION";
    }
    if (static_cast<uint8_t>(p.device) > static_cast<uint8_t>(backend::A)) {
        return "UNSUPPORTED_PROFILE_BACKEND";
    }
    if (p.model_layers <= 0 || p.layers <= 0 || p.layers > p.model_layers) {
        return "PROFILE_LAYERS_MUST_BE_MODEL_PREFIX";
    }
    if (p.chunk_tokens == 0 || p.kv_tokens < p.chunk_tokens ||
        p.repeats == 0) {
        return "INVALID_PROFILE_TOKENS_OR_REPEATS";
    }
    return {};
}

} // namespace llama_ring_v2
