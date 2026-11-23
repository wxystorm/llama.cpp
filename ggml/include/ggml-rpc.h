#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    7

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 97, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// Dynamically retrieved procedures for the opt-in Jupiter native Phone-owned
// Prefill bridge. These POD types keep MSVC/Android DLL boundaries independent
// of C++ std::string and std::vector ABIs.
#define GGML_BACKEND_RPC_PHONE_OWNED_PREPARE_PROC "ggml_backend_rpc_phone_owned_native_prepare"
#define GGML_BACKEND_RPC_PHONE_OWNED_RUN_PROC     "ggml_backend_rpc_phone_owned_native_run"
#define GGML_BACKEND_RPC_PHONE_OWNED_RESET_PROC   "ggml_backend_rpc_phone_owned_native_reset"

struct ggml_rpc_phone_owned_plan {
    uint32_t version;
    uint32_t flags;
    uint64_t run_id;
    uint64_t p2c_run_id;
    uint32_t region;
    uint32_t layer;
    uint32_t total_tokens;
    uint32_t chunk_tokens;
    // Exact Jupiter Sequence DP cuts (0 = legacy uniform-width plan).
    uint32_t schedule_count;
    uint32_t schedule[16];
    uint32_t n_embd;
    uint32_t position_base;
    uint32_t kv_capacity;
    uint32_t n_heads;
    uint32_t n_heads_kv;
    uint32_t head_dim;
    uint32_t n_kv;
    uint32_t rope_type;
    uint32_t n_ctx_orig;
    uint32_t kv_type;
    float rms_norm_eps;
    float rope_freq_base;
    float rope_freq_scale;
    float rope_ext_factor;
    float rope_attn_factor;
    float rope_beta_fast;
    float rope_beta_slow;
};
struct ggml_rpc_phone_owned_prepare_stats {
    uint32_t graphs;
    double model_load_ms;
    double build_ms;
    double buffer_alloc_ms;
    double precompile_ms;
    double wall_ms;
};
struct ggml_rpc_phone_owned_run_stats {
    uint32_t layers;
    uint32_t tokens;
    double upload_ms;
    double compute_ms;
    double output_copy_ms;
    double wall_ms;
};
typedef bool (*ggml_backend_rpc_phone_owned_prepare_t)(
    const char * endpoint,
    const struct ggml_rpc_phone_owned_plan * plan,
    const uint32_t * layer_devices,
    uint32_t n_devices,
    struct ggml_rpc_phone_owned_prepare_stats * stats);
typedef bool (*ggml_backend_rpc_phone_owned_run_t)(
    const char * endpoint,
    uint64_t run_id, uint32_t region, uint32_t chunk,
    const float * input, size_t input_floats,
    float * output, size_t output_capacity_floats,
    struct ggml_rpc_phone_owned_run_stats * stats);
typedef bool (*ggml_backend_rpc_phone_owned_reset_t)(
    const char * endpoint, uint64_t run_id);

// Opt-in Phone-side autonomous Prefill. START arms a single bounded input
// stream, PUSH transfers an independent F32 Chunk and only acknowledges
// queue admission, and P2C_WAIT subscribes to the final Phone-owned hidden
// state on a dedicated socket. No Decode or production KV handoff.
#define GGML_BACKEND_RPC_PHONE_OWNED_AUTO_START_PROC \
    "ggml_backend_rpc_phone_owned_auto_start"
#define GGML_BACKEND_RPC_PHONE_OWNED_AUTO_PUSH_PROC \
    "ggml_backend_rpc_phone_owned_auto_push"
#define GGML_BACKEND_RPC_PHONE_OWNED_AUTO_P2C_WAIT_PROC \
    "ggml_backend_rpc_phone_owned_auto_p2c_wait"

typedef bool (*ggml_backend_rpc_phone_owned_auto_start_t)(
    const char * endpoint, uint64_t run_id, uint64_t p2c_run_id,
    uint32_t region, uint32_t max_in_flight);
typedef bool (*ggml_backend_rpc_phone_owned_auto_push_t)(
    const char * endpoint, uint64_t run_id, uint64_t p2c_run_id,
    uint32_t region, uint32_t chunk, const float * input,
    size_t input_floats);
typedef bool (*ggml_backend_rpc_phone_owned_auto_p2c_wait_t)(
    const char * endpoint, uint64_t run_id, uint64_t p2c_run_id,
    uint32_t region, float * output, size_t output_capacity_floats,
    struct ggml_rpc_phone_owned_run_stats * stats);

struct ggml_rpc_local_tensor_info {
    uint64_t nbytes;      //张量的字节数
    uint32_t type;      //张量的数据类型
    uint32_t n_dims;    //张量的维度数
    uint64_t ne[GGML_MAX_DIMS]; //张量的每个维度的大小
};

typedef bool (*ggml_rpc_get_local_tensor_info_fn)(
        void * user_data,
        const char * name,
        struct ggml_rpc_local_tensor_info * info);
//回调函数模版
typedef bool (*ggml_rpc_read_local_tensor_fn)(
        void * user_data,
        const char * name,
        uint64_t tensor_offset,
        void * dest,
        size_t size);

struct ggml_rpc_local_tensor_source {
    void * user_data;

    ggml_rpc_get_local_tensor_info_fn get_info;
    ggml_rpc_read_local_tensor_fn     read;
};

#define GGML_BACKEND_RPC_SET_LOCAL_2D_PROC \
    "ggml_backend_rpc_set_tensor_from_local_file_2d"
    
enum ggml_backend_local_file_result {
    GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED,
    GGML_BACKEND_LOCAL_FILE_HANDLED,
    GGML_BACKEND_LOCAL_FILE_ERROR,
};

using ggml_backend_rpc_set_local_2d_t =
    ggml_backend_local_file_result (*)(
        ggml_tensor * tensor,
        uint64_t src_offset,
        uint64_t dst_offset,
        uint64_t copy_size,
        uint64_t n_copies,
        uint64_t src_stride,
        uint64_t dst_stride);
// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

GGML_BACKEND_API void ggml_backend_rpc_start_server_ex(const char * endpoint, const char * cache_dir, size_t n_threads,
        size_t n_devices, ggml_backend_dev_t * devices, const struct ggml_rpc_local_tensor_source * tensor_source,
        const char * model_path, int tp_rank, int tp_world_size);

#ifdef  __cplusplus
}
#endif
