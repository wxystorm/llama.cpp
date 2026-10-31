#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    19

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

#define GGML_BACKEND_RPC_SET_ROUTE_TRANSFER_LANE_PROC \
    "ggml_backend_rpc_set_route_transfer_lane"

#define GGML_BACKEND_RPC_SET_ROUTE_WAIT_SEQ_PROC \
    "ggml_backend_rpc_set_route_wait_seq"

#define GGML_BACKEND_RPC_ROUTE_MARK_READY_PROC \
    "ggml_backend_rpc_route_mark_ready"

#define GGML_BACKEND_RPC_ROUTE_SNAPSHOT_READY_PROC \
    "ggml_backend_rpc_route_snapshot_ready"

#define GGML_BACKEND_RPC_GET_ROUTE_SNAPSHOT_PROC \
    "ggml_backend_rpc_get_route_snapshot"

#define GGML_BACKEND_RPC_FENCE_PROC \
    "ggml_backend_rpc_fence"

#define GGML_BACKEND_RPC_SNAPSHOT_ARM_PROC \
    "ggml_backend_rpc_snapshot_arm"

#define GGML_BACKEND_RPC_PREPARE_GRAPH_SNAPSHOT_PROC \
    "ggml_backend_rpc_prepare_graph_snapshot"

#define GGML_BACKEND_RPC_SET_SNAPSHOT_READ_PROC \
    "ggml_backend_rpc_set_snapshot_read"

#define GGML_BACKEND_RPC_WAIT_SNAPSHOT_READY_PROC \
    "ggml_backend_rpc_wait_snapshot_ready"

#define GGML_BACKEND_RPC_GET_SNAPSHOT_STATS_PROC \
    "ggml_backend_rpc_get_snapshot_stats"

#define GGML_BACKEND_RPC_GET_GRAPH_PROFILE_PROC \
    "ggml_backend_rpc_get_graph_profile"

// Experimental Region Ring stream: one Phone-side sender, one PC-side receiver.
// All snapshot captures are published only after the GPU->Host read completes.
#define GGML_BACKEND_RPC_P2C_STREAM_START_PROC \
    "ggml_backend_rpc_p2c_stream_start"
#define GGML_BACKEND_RPC_P2C_STREAM_CAPTURE_PROC \
    "ggml_backend_rpc_p2c_stream_capture"
#define GGML_BACKEND_RPC_P2C_STREAM_STOP_PROC \
    "ggml_backend_rpc_p2c_stream_stop"
    
#define GGML_BACKEND_RPC_PREPARE_FUSED_FFN_INPUT_PROC \
    "ggml_backend_rpc_prepare_fused_ffn_input"

#define GGML_BACKEND_RPC_INIT_ZERO_TENSOR_PROC \
    "ggml_backend_rpc_init_zero_tensor"

#define GGML_BACKEND_RPC_GET_TENSOR_BATCH3_PROC \
    "ggml_backend_rpc_get_tensor_batch3"

#define GGML_BACKEND_RPC_SET_TENSOR_GRAPH_PROC \
    "ggml_backend_rpc_set_tensor_graph"

#define GGML_BACKEND_RPC_SET_TENSOR_ASYNC_RETURN_PROC \
    "ggml_backend_rpc_set_tensor_async_return"

#define GGML_BACKEND_RPC_PHONE_FFN_MARK_READY_PROC \
    "ggml_backend_rpc_phone_ffn_mark_ready"

#define GGML_BACKEND_RPC_SET_TENSOR_ASYNC_RETURN_WAIT_PROC \
    "ggml_backend_rpc_set_tensor_async_return_wait"

#define GGML_BACKEND_RPC_WAIT_READY_SEQ_PROC \
    "ggml_backend_rpc_wait_ready_seq"

#define GGML_BACKEND_RPC_GET_TENSOR_RETURN_WAIT_PROC \
    "ggml_backend_rpc_get_tensor_return_wait"

#define GGML_BACKEND_RPC_PHONE_REGION_LOCAL_SUBMIT_PROC \
    "ggml_backend_rpc_phone_region_local_submit"

#define GGML_BACKEND_RPC_PHONE_REGION_PARTIAL_GET_PROC \
    "ggml_backend_rpc_phone_region_partial_get"

// Region Ring: register a Phone-only fused graph, then publish input READY.
// The Phone MAIN worker executes it; the PC waits only at the output bridge.
#define GGML_BACKEND_RPC_PHONE_RING_AUTO_READY_PROC \
    "ggml_backend_rpc_phone_ring_auto_ready"

#define GGML_BACKEND_RPC_PHONE_RING_AUTO_WAIT_PROC \
    "ggml_backend_rpc_phone_ring_auto_wait"

typedef bool (*ggml_backend_rpc_init_zero_tensor_t)(
    ggml_tensor * tensor);

typedef bool (*ggml_backend_rpc_prepare_fused_ffn_input_t)(
    ggml_backend_t backend,
    const ggml_tensor * tensor);

using ggml_backend_rpc_get_tensor_batch3_t = bool (*)(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src0,
        ggml_tensor * dst0,
        const ggml_tensor * src1,
        ggml_tensor * dst1,
        const ggml_tensor * src2,
        ggml_tensor * dst2);

using ggml_backend_rpc_set_tensor_graph_t = bool (*)(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src,
        ggml_tensor * dst,
        const struct ggml_cgraph * graph);

using ggml_backend_rpc_set_tensor_async_return_t = bool (*)(
        ggml_backend_t backend_dst,
        ggml_tensor * dst,
        const void * data,
        size_t size);

using ggml_backend_rpc_phone_ffn_mark_ready_t = bool (*)(
        ggml_backend_t backend,
        uint64_t seq);

using ggml_backend_rpc_wait_ready_seq_t = bool (*)(
        ggml_backend_t backend,
        uint64_t seq);

using ggml_backend_rpc_get_tensor_return_wait_t = bool (*)(
        ggml_backend_t backend_src,
        const ggml_tensor * src,
        void * data,
        size_t size,
        uint64_t ready_seq,
        size_t return_lane);

using ggml_backend_rpc_set_tensor_async_return_wait_t = bool (*)(
        ggml_backend_t backend_dst,
        ggml_tensor * dst,
        const void * data,
        size_t size,
        uint64_t phone_ffn_seq,
        size_t return_lane);

using ggml_backend_rpc_phone_region_local_submit_t = bool (*)(
        ggml_backend_t backend,
        int chunk,
        const struct ggml_cgraph * region_graph,
        const struct ggml_cgraph * attn_graph,
        const ggml_tensor * ready_hidden,
        const ggml_tensor * ready_topk,
        const ggml_tensor * ready_weights,
        uint64_t ready_seq,
        uint32_t ready_lane,
        const struct ggml_cgraph * ffn_graph,
        const struct ggml_cgraph * fold_graph,
        const ggml_tensor * partial,
        uint64_t partial_seq,
        uint32_t partial_slot,
        uint64_t scratch_reuse_seq);

using ggml_backend_rpc_phone_ring_auto_ready_t = bool (*)(
        ggml_backend_t backend,
        const struct ggml_cgraph * graph,
        int first_layer,
        uint64_t completion_seq);

using ggml_backend_rpc_phone_ring_auto_wait_t = bool (*)(
        ggml_backend_t backend,
        uint64_t completion_seq);

using ggml_backend_rpc_phone_region_partial_get_t = bool (*)(
        ggml_backend_t backend,
        uint32_t slot,
        uint64_t seq,
        void * data,
        size_t size);

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

using ggml_backend_rpc_set_route_transfer_lane_t = void (*)(
        int lane);

using ggml_backend_rpc_set_route_wait_seq_t = void (*)(
        uint64_t seq);

using ggml_backend_rpc_route_mark_ready_t = bool (*)(
        ggml_backend_t backend,
        uint64_t seq);

using ggml_backend_rpc_route_snapshot_ready_t = bool (*)(
        ggml_backend_t backend,
        uint64_t seq,
        uint32_t lane,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * src2);

using ggml_backend_rpc_get_route_snapshot_t = bool (*)(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        uint64_t seq,
        uint32_t lane,
        ggml_tensor * dst0,
        ggml_tensor * dst1,
        ggml_tensor * dst2);

using ggml_backend_rpc_fence_t = void (*)(ggml_backend_t backend);

using ggml_backend_rpc_snapshot_arm_t = bool (*)(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        size_t offset,
        size_t size,
        uint32_t slot,
        uint64_t seq);

using ggml_backend_rpc_prepare_graph_snapshot_t = bool (*)(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        size_t offset,
        size_t size,
        uint32_t slot,
        uint64_t seq);

using ggml_backend_rpc_set_snapshot_read_t = void (*)(
        bool enabled,
        uint32_t slot,
        uint64_t seq);

using ggml_backend_rpc_wait_snapshot_ready_t = bool (*)(
        ggml_backend_t backend,
        uint64_t seq);

struct ggml_backend_rpc_snapshot_stats {
    uint64_t transfer_count;
    uint64_t payload_bytes;
    int64_t  request_us;
    int64_t  ready_first_byte_us;
    int64_t  recv_payload_us;
    int64_t  total_us;

    // Ordinary GET_TENSOR, not Snapshot. Header wait includes remote
    // producer/queue/device-read time, not just TCP latency.
    uint64_t tensor_get_count;
    uint64_t tensor_get_bytes;
    int64_t  tensor_get_request_us;
    int64_t  tensor_get_header_wait_us;
    int64_t  tensor_get_payload_recv_us;
    int64_t  tensor_get_total_us;

    // Client-side Region Ring RPC command accounting; all values are
    // cumulative and sampled by the Phone Full worker at Chunk boundaries.
    // SET_TENSOR bytes exclude protocol headers, hash hits and fused commands.
    uint64_t set_tensor_count;
    uint64_t set_tensor_bytes;
    uint64_t graph_compute_count;
    uint64_t graph_recompute_count;
    uint64_t graph_snapshot_count;
    uint64_t graph_multi_count;
    uint64_t graph_multi_nodes;
    uint64_t graph_single_add_count;
    uint64_t graph_single_other_count;
};

using ggml_backend_rpc_get_snapshot_stats_t = bool (*)(
        ggml_backend_t backend,
        struct ggml_backend_rpc_snapshot_stats * stats);

// Native Phone clock, timed around one full graph with device sync.
// Query only after the normal warmup/measurement loop has finished.
struct ggml_backend_rpc_graph_profile {
    uint32_t samples = 0;
    uint32_t n_nodes = 0;
    int64_t server_median_us = 0;
};

using ggml_backend_rpc_get_graph_profile_t = bool (*)(
        ggml_backend_t backend,
        uint64_t graph_uid,
        struct ggml_backend_rpc_graph_profile * profile);

// This is a logical frame header: no remote pointer or Arena address is sent.
// The receiver resolves (run_id, seq, region_id, chunk_id) to its own host
// output buffer before reading exactly payload_bytes bytes.
struct ggml_backend_rpc_p2c_frame {
    uint64_t run_id;
    uint64_t seq;
    uint32_t region_id;
    uint32_t chunk_id;
    uint32_t token_begin;
    uint32_t n_tokens;
    uint32_t kind;
    uint32_t reserved;
    uint64_t payload_bytes;
};
enum {
    GGML_BACKEND_RPC_P2C_FULL = 1,
    GGML_BACKEND_RPC_P2C_TENSOR = 2,
    GGML_BACKEND_RPC_P2C_END = 0xFFFFFFFFu,
    GGML_BACKEND_RPC_P2C_ERROR = 0xFFFFFFFEu,
};

// resolve() must return a stable PC destination with capacity payload_bytes.
// complete() is called only after a full successful recv, or with ok=false on
// stream errors. Neither callback may synchronously wait for the sender.
using ggml_backend_rpc_p2c_resolve_t = void * (*)(
        void * userdata,
        const struct ggml_backend_rpc_p2c_frame * frame);
using ggml_backend_rpc_p2c_complete_t = void (*)(
        void * userdata,
        const struct ggml_backend_rpc_p2c_frame * frame,
        bool ok);

using ggml_backend_rpc_p2c_stream_start_t = void * (*)(
        ggml_backend_t backend,
        uint64_t run_id,
        uint32_t n_regions,
        uint32_t n_tokens,
        uint32_t n_embd,
        ggml_backend_rpc_p2c_resolve_t resolve,
        ggml_backend_rpc_p2c_complete_t complete,
        void * userdata);
using ggml_backend_rpc_p2c_stream_capture_t = bool (*)(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        uint64_t run_id,
        uint64_t seq,
        uint32_t region_id,
        uint32_t chunk_id,
        uint32_t token_begin,
        uint32_t n_tokens);
using ggml_backend_rpc_p2c_stream_stop_t = bool (*)(
        ggml_backend_t backend,
        void * stream_handle);
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
