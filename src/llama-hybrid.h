#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

struct llama_hybrid_transfer_point {
    size_t bytes             = 0;
    double ms                = 0.0;
    double confirmed_wall_ms = 0.0;
};

struct llama_hybrid_dual_transfer_point {
    size_t bytes0  = 0;
    size_t bytes1  = 0;
    double wall_ms = 0.0;
};

struct llama_hybrid_ffn_compute_point {
    int    tokens      = 0;
    float  local_ratio = 1.0f;
    double ms          = 0.0;
};

struct llama_hybrid_ffn_desc {
    int64_t n_embd = 0;
    int64_t n_ff   = 0;

    ggml_type ffn_norm_type = GGML_TYPE_COUNT;

    ggml_type gate_type = GGML_TYPE_COUNT;
    ggml_type up_type   = GGML_TYPE_COUNT;
    ggml_type down_type = GGML_TYPE_COUNT;

    size_t weight_bytes = 0;
};

struct llama_hybrid_attn_compute_point {
    int    tokens    = 0;
    int    kv_tokens = 0;
    double ms        = 0.0;
};

struct llama_hybrid_phone_block_point {
    int layers    = 0;
    int tokens    = 0;
    int kv_tokens = 0;

    double wall_ms        = 0.0;
    double compute_est_ms = 0.0;
};

struct llama_hybrid_attn_desc {
    int64_t n_embd = 0;

    int64_t q_dim    = 0;
    int64_t k_dim    = 0;
    int64_t v_dim    = 0;
    int64_t o_in_dim = 0;

    int n_head    = 0;
    int n_head_kv = 0;

    float rms_eps = 1e-5f;

    llama_rope_type rope_type = LLAMA_ROPE_TYPE_NONE;

    int32_t n_rot      = 0;
    int32_t n_ctx_orig = 0;

    float freq_base  = 10000.0f;
    float freq_scale = 1.0f;

    ggml_type attn_norm_type = GGML_TYPE_COUNT;

    ggml_type q_type = GGML_TYPE_COUNT;
    ggml_type k_type = GGML_TYPE_COUNT;
    ggml_type v_type = GGML_TYPE_COUNT;
    ggml_type o_type = GGML_TYPE_COUNT;

    bool has_q_norm = false;
    bool has_k_norm = false;

    ggml_type q_norm_type = GGML_TYPE_COUNT;
    ggml_type k_norm_type = GGML_TYPE_COUNT;

    size_t weight_bytes = 0;
};

struct llama_hybrid_profile {
    int probe_tokens           = 0;
    int probe_chunk_min_tokens = 0;

    int n_layer = 0;
    int n_embd  = 0;
    int n_ff    = 0;

    std::vector<llama_hybrid_ffn_compute_point> cpu_ffn;
    std::vector<llama_hybrid_ffn_compute_point> phone_ffn;

    std::vector<llama_hybrid_attn_compute_point> cpu_attn;
    std::vector<llama_hybrid_attn_compute_point> phone_attn;
    std::vector<llama_hybrid_attn_compute_point> gpu_attn;

    std::vector<llama_hybrid_phone_block_point> phone_blocks;

    double cpu_attn_ms       = 0.0;
    double cpu_full_layer_ms = 0.0;

    double phone_attn_ms                   = 0.0;
    double phone_full_layer_ms             = 0.0;
    double phone_full_layer_compute_est_ms = 0.0;

    double gpu_full_layer_ms = 0.0;

    std::vector<llama_hybrid_transfer_point> pc_to_phone;
    std::vector<llama_hybrid_transfer_point> snapshot_phone_to_pc;
    std::vector<llama_hybrid_transfer_point> phone_to_pc;

    std::vector<llama_hybrid_dual_transfer_point> dual_phone_to_pc;

    double reduce_ms     = 0.0;
    double rpc_submit_ms = 0.0;
    double rpc_fence_ms  = 0.0;

    size_t pc_free_mem    = 0;
    size_t phone_free_mem = 0;
    size_t gpu_free_mem   = 0;

    size_t pc_total_mem    = 0;
    size_t phone_total_mem = 0;
    size_t gpu_total_mem   = 0;

    size_t model_weight_bytes = 0;

    size_t attn_weight_bytes_per_layer  = 0;
    size_t ffn_weight_bytes_per_layer   = 0;
    size_t total_weight_bytes_per_layer = 0;
    size_t non_layer_weight_bytes       = 0;

    std::vector<size_t> layer_weight_bytes;
    std::vector<size_t> layer_attn_forced_bytes;
    std::vector<size_t> layer_ffn_split_bytes;
    std::vector<size_t> layer_mirrored_bytes;

    size_t kv_bytes_per_token_per_layer = 0;
    int    n_ctx_train                  = 0;
};

struct llama_hybrid_plan {
    int tensor_layers = 0;
    int phone_layers  = 0;
    int pc_layers     = 0;

    float tensor_pc_ratio = 0.0f;

    int gpu_pc_layers            = 0;
    int tensor_chunks_per_ubatch = 1;

    double predicted_ms = 0.0;

    double predicted_tensor_ms  = 0.0;
    double predicted_phone_ms   = 0.0;
    double predicted_pc_cpu_ms  = 0.0;
    double predicted_pc_gpu_ms  = 0.0;
    double predicted_handoff_ms = 0.0;

    size_t pc_memory    = 0;
    size_t phone_memory = 0;
    size_t gpu_memory   = 0;
};

struct llama_hybrid_constraints {
    size_t pc_memory_budget    = 0;
    size_t phone_memory_budget = 0;
    size_t gpu_memory_budget   = 0;

    size_t gpu_runtime_reserve_bytes = size_t(512) * 1024 * 1024;

    int target_ctx      = 0;
    int score_kv_tokens = 0;

    std::optional<int> fixed_tensor_layers;
    std::optional<int> fixed_phone_layers;
    std::optional<int> fixed_pc_layers;

    std::optional<float> fixed_tensor_pc_ratio;
    std::optional<int>   fixed_gpu_pc_layers;
    std::optional<int>   fixed_tensor_chunks_per_ubatch;
};

struct ggml_backend;
typedef struct ggml_backend * ggml_backend_t;
class llama_model_loader;

LLAMA_API void llama_hybrid_profile_print(const llama_hybrid_profile & profile);
LLAMA_API void llama_hybrid_plan_print(const llama_hybrid_plan & plan);

LLAMA_API bool llama_hybrid_profile_rpc(llama_hybrid_profile & profile,
                                        ggml_backend_t         pc_backend,
                                        ggml_backend_t         phone_backend);

LLAMA_API std::vector<int> llama_hybrid_split_chunks(int tokens, int n_chunks);
LLAMA_API std::vector<int> llama_hybrid_probe_chunk_tokens(const llama_hybrid_profile & profile);

LLAMA_API int64_t llama_hybrid_ffn_shard_size(int64_t n_ff, ggml_type down_type, float pc_ratio, int backend_index);

LLAMA_API bool llama_hybrid_profile_ffn(llama_hybrid_profile &        profile,
                                        const llama_hybrid_ffn_desc & desc,
                                        ggml_backend_t                cpu_backend,
                                        ggml_backend_t                phone_backend);

LLAMA_API bool llama_hybrid_profile_attention(llama_hybrid_profile &         profile,
                                              const llama_hybrid_attn_desc & desc,
                                              ggml_backend_t                 cpu_backend,
                                              ggml_backend_t                 phone_backend,
                                              ggml_backend_t                 gpu_backend);

LLAMA_API bool llama_hybrid_profile_full_layer(llama_hybrid_profile &         profile,
                                               const llama_hybrid_attn_desc & attn_desc,
                                               const llama_hybrid_ffn_desc &  ffn_desc,
                                               ggml_backend_t                 cpu_backend,
                                               ggml_backend_t                 phone_backend,
                                               ggml_backend_t                 gpu_backend);

LLAMA_API bool llama_hybrid_profile_phone_blocks(llama_hybrid_profile &         profile,
                                                 const llama_hybrid_attn_desc & attn_desc,
                                                 const llama_hybrid_ffn_desc &  ffn_desc,
                                                 ggml_backend_t                 phone_backend);

LLAMA_API bool llama_hybrid_profile_memory(llama_hybrid_profile & profile,
                                           ggml_backend_t         pc_backend,
                                           ggml_backend_t         phone_backend,
                                           ggml_backend_t         gpu_backend);

LLAMA_API std::vector<llama_hybrid_plan> llama_hybrid_enumerate_feasible_plans(
    const llama_hybrid_profile &     profile,
    const llama_hybrid_constraints & constraints);

LLAMA_API bool llama_hybrid_score_plan(const llama_hybrid_profile &     profile,
                                       const llama_hybrid_constraints & constraints,
                                       llama_hybrid_plan &              plan);

LLAMA_API bool llama_hybrid_runtime_plan_set(const llama_hybrid_plan & plan);
LLAMA_API bool llama_hybrid_runtime_plan_get(llama_hybrid_plan & plan);
LLAMA_API void llama_hybrid_runtime_plan_clear();
LLAMA_API int  llama_hybrid_runtime_prefill_chunks();

LLAMA_API bool llama_hybrid_autoplan(llama_model_loader &       ml,
                                     const llama_model_params & params,
                                     llama_hybrid_plan &        best_plan);
