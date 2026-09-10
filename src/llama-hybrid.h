#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

struct llama_hybrid_transfer_point {
    size_t bytes = 0;
    double ms    = 0.0;
};

struct llama_hybrid_dual_transfer_point {
    size_t bytes0  = 0;
    size_t bytes1  = 0;
    double wall_ms = 0.0;
};

struct llama_hybrid_ffn_compute_point {
    int    tokens = 0;
    float  ratio  = 1.0f;
    double ms     = 0.0;
};

struct llama_hybrid_ffn_desc {
    int64_t n_embd = 0;
    int64_t n_ff   = 0;

    ggml_type gate_type = GGML_TYPE_COUNT;
    ggml_type up_type   = GGML_TYPE_COUNT;
    ggml_type down_type = GGML_TYPE_COUNT;
};

struct llama_hybrid_profile {
    int probe_tokens           = 0;
    int probe_chunk_min_tokens = 0;

    int n_layer = 0;
    int n_embd  = 0;
    int n_ff    = 0;

    std::vector<llama_hybrid_ffn_compute_point> cpu_ffn;
    std::vector<llama_hybrid_ffn_compute_point> phone_ffn;

    double cpu_attn_ms       = 0.0;
    double cpu_full_layer_ms = 0.0;

    double phone_attn_ms       = 0.0;
    double phone_full_layer_ms = 0.0;

    double gpu_full_layer_ms = 0.0;

    std::vector<llama_hybrid_transfer_point> pc_to_phone;
    std::vector<llama_hybrid_transfer_point> phone_to_pc;

    std::vector<llama_hybrid_dual_transfer_point> dual_phone_to_pc;

    double reduce_ms     = 0.0;
    double rpc_submit_ms = 0.0;

    size_t pc_free_mem    = 0;
    size_t phone_free_mem = 0;
    size_t gpu_free_mem   = 0;

    size_t attn_weight_bytes_per_layer  = 0;
    size_t ffn_weight_bytes_per_layer   = 0;
    size_t total_weight_bytes_per_layer = 0;
    size_t non_layer_weight_bytes       = 0;
};

struct llama_hybrid_plan {
    int tensor_layers = 0;
    int phone_layers  = 0;
    int pc_layers     = 0;

    float tensor_pc_ratio = 0.0f;

    int gpu_pc_layers            = 0;
    int tensor_chunks_per_ubatch = 1;

    double predicted_ms = 0.0;

    size_t pc_memory    = 0;
    size_t phone_memory = 0;
    size_t gpu_memory   = 0;
};

struct llama_hybrid_constraints {
    size_t pc_memory_budget    = 0;
    size_t phone_memory_budget = 0;
    size_t gpu_memory_budget   = 0;

    std::optional<int> fixed_tensor_layers;
    std::optional<int> fixed_phone_layers;
    std::optional<int> fixed_pc_layers;

    std::optional<float> fixed_tensor_pc_ratio;
    std::optional<int>   fixed_gpu_pc_layers;
    std::optional<int>   fixed_tensor_chunks_per_ubatch;
};

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;

LLAMA_API void llama_hybrid_profile_print(const llama_hybrid_profile & profile);
LLAMA_API void llama_hybrid_plan_print(const llama_hybrid_plan & plan);

LLAMA_API bool llama_hybrid_profile_rpc(llama_hybrid_profile & profile, ggml_backend_t pc_backend, ggml_backend_t phone_backend);

LLAMA_API std::vector<int> llama_hybrid_split_chunks(int tokens, int n_chunks);
LLAMA_API std::vector<int> llama_hybrid_probe_chunk_tokens(const llama_hybrid_profile & profile);

LLAMA_API int64_t llama_hybrid_ffn_shard_size(int64_t n_ff, ggml_type down_type, float pc_ratio, int backend_index);

LLAMA_API bool llama_hybrid_profile_ffn(llama_hybrid_profile &        profile,
                                        const llama_hybrid_ffn_desc & desc,
                                        ggml_backend_t                cpu_backend,
                                        ggml_backend_t                phone_backend);
