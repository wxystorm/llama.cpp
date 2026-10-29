#pragma once

#include "llama-arch.h"

// One authoritative capability set for the Region Ring / Hybrid data path.
// Qwen3-MoE remains the MoE specialization, while Llama/Qwen dense models
// share the same topology, ownership, transport and GPU-weight-streaming
// machinery with dense FFN graph construction.
inline bool llama_hybrid_arch_supported(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_QWEN3:
        case LLM_ARCH_QWEN3MOE:
            return true;
        default:
            return false;
    }
}

inline bool llama_hybrid_arch_is_moe(llm_arch arch) {
    return arch == LLM_ARCH_QWEN3MOE;
}

inline bool llama_hybrid_arch_is_dense(llm_arch arch) {
    return llama_hybrid_arch_supported(arch) &&
           !llama_hybrid_arch_is_moe(arch);
}

// These capabilities intentionally use the same architecture set. Keeping
// named helpers makes future backend-specific restrictions explicit instead
// of silently reintroducing Qwen3-MoE-only gates at call sites.
inline bool llama_hybrid_arch_supports_region_gpu_weight_streaming(
        llm_arch arch) {
    return llama_hybrid_arch_supported(arch);
}

inline bool llama_hybrid_arch_supports_region_embedding_prelude(
        llm_arch arch) {
    return llama_hybrid_arch_supported(arch);
}
