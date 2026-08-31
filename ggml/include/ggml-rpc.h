#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    2

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 97, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

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

#define GGML_BACKEND_RPC_SET_STAGE_READY_PROC \
    "ggml_backend_rpc_set_stage_ready_callback"

#define GGML_BACKEND_RPC_FENCE_PROC \
    "ggml_backend_rpc_fence"
    
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

using ggml_backend_rpc_stage_ready_callback_t = void (*)(void * user_data);
using ggml_backend_rpc_set_stage_ready_t = void (*)(
        ggml_backend_rpc_stage_ready_callback_t callback,
        void * user_data);

using ggml_backend_rpc_fence_t = void (*)(ggml_backend_t backend);
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
