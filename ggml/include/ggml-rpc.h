#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    30

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

// Opt-in RPC transfer calibration. Two independent sockets issue the
// existing SET_TENSOR and GET_TENSOR commands concurrently; the server
// protocol and real inference data path are unchanged.
#define GGML_BACKEND_RPC_PROFILE_DUPLEX_PROC \
    "ggml_backend_rpc_profile_duplex"

struct ggml_rpc_duplex_profile {
    double c2p_solo_ms = 0.0;
    double p2c_solo_ms = 0.0;
    double c2p_concurrent_ms = 0.0;
    double p2c_concurrent_ms = 0.0;
    double concurrent_wall_ms = 0.0;
    double concurrent_overlap_ms = 0.0;
};

using ggml_backend_rpc_profile_duplex_t = bool (*)(
    ggml_backend_t backend,
    ggml_tensor * upload_destination,
    const ggml_tensor * download_source,
    size_t bytes,
    ggml_rpc_duplex_profile * result);

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

// Experimental ACK-free Phone input stream (RPC patch 23).
#define GGML_BACKEND_RPC_PHONE_INPUT_STREAM_RUN_PROC \
    "ggml_backend_rpc_phone_input_stream_run"
#define GGML_BACKEND_RPC_PHONE_INPUT_STREAM_PUSH_PROC \
    "ggml_backend_rpc_phone_input_stream_push"

using ggml_backend_rpc_phone_input_stream_run_t = bool (*)(
    ggml_backend_t backend, uint64_t run_id, bool active);
using ggml_backend_rpc_phone_input_stream_push_t = bool (*)(
    ggml_backend_t backend, uint64_t run_id, uint32_t region,
    uint32_t chunk, const void * input, size_t bytes,
    struct ggml_backend_rpc_phone_input_push_timing * timing);

// Experimental Phone-owned compute: PC registers the exact per-Chunk
// graph before streaming; a verified PUSH is the only compute trigger.
// The Phone prepares two local GPU slots, then executes chunks in KV order.
// Non-final Phone Regions retain the existing PC-side output handoff.
// Patch 24 Graph registration, patch 25 direct WAIT_CAPTURE, patch 26
// Phone-owned output publication on compute completion. Older peers fail closed.
#define GGML_BACKEND_RPC_PHONE_AUTO_GRAPH_REGISTER_PROC \
    "ggml_backend_rpc_phone_auto_graph_register"
#define GGML_BACKEND_RPC_PHONE_AUTO_TASK_REGISTER_PROC \
    "ggml_backend_rpc_phone_auto_task_register"
#define GGML_BACKEND_RPC_PHONE_AUTO_WAIT_PROC \
    "ggml_backend_rpc_phone_auto_wait"
// Patch 25: non-final Phone Full runtime avoids Meta graph execution.
// WAIT_CAPTURE validates the exact registered output and publishes it into
// the existing Phone->PC Arena before acknowledging the PC Phone Worker.
#define GGML_BACKEND_RPC_PHONE_AUTO_WAIT_CAPTURE_PROC \
    "ggml_backend_rpc_phone_auto_wait_capture"
using ggml_backend_rpc_phone_auto_graph_register_t = bool (*)(
    ggml_backend_t backend, const ggml_cgraph * graph,
    uint64_t * registered_uid);
// Output metadata is registered once, before input PUSH. No per-Chunk
// WAIT_CAPTURE RPC is necessary in proactive mode.
using ggml_backend_rpc_phone_auto_task_register_t = bool (*)(
    ggml_backend_t backend, uint64_t run_id, uint32_t region,
    uint32_t chunk, uint64_t graph_uid, size_t expected_input_bytes,
    uint64_t p2c_run_id, uint32_t token_begin, uint32_t n_tokens,
    size_t expected_output_bytes);
using ggml_backend_rpc_phone_auto_wait_t = bool (*)(
    ggml_backend_t backend, uint64_t run_id, uint32_t region,
    uint32_t chunk);
// A zeroed request is a capability probe (RPC patch >=25). Real requests
// must identify both the Phone input run and the independent P2C stream run.
using ggml_backend_rpc_phone_auto_wait_capture_t = bool (*)(
    ggml_backend_t backend, uint64_t input_run_id, uint32_t region,
    uint32_t chunk, uint64_t graph_uid, uint64_t p2c_run_id,
    uint32_t token_begin, uint32_t n_tokens, size_t expected_bytes);

// Patch 28: optional shadow registration of an immutable Phone-owned
// execution plan. This does not switch off the PC-built Graph path; a local
// Qwen3 Layer Builder and parity validation are required before that cutover.
struct ggml_backend_rpc_phone_owned_plan {
    uint32_t version;
    uint32_t flags;
    uint64_t run_id;
    uint64_t p2c_run_id;
    uint32_t region;
    uint32_t layer;
    uint32_t total_tokens;
    uint32_t chunk_tokens;
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
#define GGML_BACKEND_RPC_PHONE_OWNED_PLAN_REGISTER_PROC \
    "ggml_backend_rpc_phone_owned_plan_register"
using ggml_backend_rpc_phone_owned_plan_register_t = bool (*)(
    ggml_backend_t backend,
    const ggml_backend_rpc_phone_owned_plan * plan);

// Planner-driven Phone HTP Region/Layer mapping. No per-layout HTP
// environment variables are required on the Phone; PC passes all selected
// P1 Region/Layer pairs before registering any Phone-owned Graph/KV.
// Version 1 requires at least two HTP Regions for physical two-Bank mode.
struct ggml_backend_rpc_phone_npu_region_layer {
    uint32_t region;
    uint32_t layer;
};
#define GGML_BACKEND_RPC_PHONE_NPU_PLAN_CONFIGURE_PROC \
    "ggml_backend_rpc_phone_npu_plan_configure"
using ggml_backend_rpc_phone_npu_plan_configure_t = bool (*)(
    ggml_backend_t backend, uint64_t run_id,
    const ggml_backend_rpc_phone_npu_region_layer * entries,
    size_t count);

// One-time canonical Decode KV destination binding; no per-Chunk Graph.
#define GGML_BACKEND_RPC_PHONE_OWNED_KV_BIND_PROC \
    "ggml_backend_rpc_phone_owned_kv_bind"
using ggml_backend_rpc_phone_owned_kv_bind_t = bool (*)(
    ggml_backend_t backend, uint64_t run_id, uint32_t region,
    uint32_t layer, uint32_t total_tokens,
    const ggml_tensor * canonical_k, const ggml_tensor * canonical_v,
    bool verified_identity_slots);

// Experiment: producer-driven PC->Phone host landing slots. No current Graph
// tensors are touched until the ordinary Phone Worker enters split_prepare().
#define GGML_BACKEND_RPC_PHONE_INPUT_PUSH_PROC \
    "ggml_backend_rpc_phone_input_push"
#define GGML_BACKEND_RPC_PHONE_INPUT_CONSUME_BEGIN_PROC \
    "ggml_backend_rpc_phone_input_consume_begin"
#define GGML_BACKEND_RPC_PHONE_INPUT_CONSUME_END_PROC \
    "ggml_backend_rpc_phone_input_consume_end"
#define GGML_BACKEND_RPC_PHONE_INPUT_DROP_PROC \
    "ggml_backend_rpc_phone_input_drop"

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
    // Optional client-side graph I/O timing. Command wall time may include
    // remote execution/ack; it is NOT pure graph registration cost.
    uint64_t graph_serialize_us;
    uint64_t graph_serialized_bytes;
    uint64_t graph_compute_cmd_us;
    uint64_t graph_recompute_cmd_us;
};

using ggml_backend_rpc_get_snapshot_stats_t = bool (*)(
        ggml_backend_t backend,
        struct ggml_backend_rpc_snapshot_stats * stats);

// Durations from two independent monotonic clocks. Do NOT subtract PC
// timestamps from Phone timestamps or add overlapping send/receive calls.
// This lightweight breakdown is returned by the PUSH RPC ACK; no per-chunk
// terminal logging or extra control-plane round trips are required.
struct ggml_backend_rpc_phone_input_push_timing {
    int64_t pc_socket_lock_wait_us = 0;
    int64_t pc_hash_us = 0;
    int64_t pc_pack_us = 0;
    int64_t pc_header_send_us = 0;
    int64_t pc_payload_send_us = 0;
    // PC-clock absolute boundaries, for TASK_TRACE JSON only. The values
    // are not sent to the Phone and must never be subtracted from Phone time.
    int64_t pc_header_begin_us = 0;
    int64_t pc_header_end_us = 0;
    int64_t pc_payload_begin_us = 0;
    int64_t pc_payload_end_us = 0;
    int64_t pc_ack_wait_us = 0;
    int64_t phone_recv_header_us = 0;
    int64_t phone_recv_alloc_us = 0;
    int64_t phone_recv_payload_us = 0;
    int64_t phone_hash_us = 0;
    int64_t phone_slot_lock_wait_us = 0;
    int64_t phone_host_stage_us = 0;
    int64_t phone_handler_us = 0;
};

using ggml_backend_rpc_phone_input_push_t = bool (*)(
        ggml_backend_t backend, uint64_t run_id,
        uint32_t region, uint32_t chunk,
        const void * input, size_t bytes,
        struct ggml_backend_rpc_phone_input_push_timing * timing);
// PC Phone Worker thread-local scope for one already staged input. The
// ordinary RPC set_tensor() transparently applies the matching staged bytes
// to its original destination tensor; all other tensors use legacy SET.
using ggml_backend_rpc_phone_input_consume_begin_t = void (*)(
        uint64_t run_id, uint32_t region, uint32_t chunk, size_t bytes);
using ggml_backend_rpc_phone_input_consume_end_t = bool (*)();
using ggml_backend_rpc_phone_input_drop_t = bool (*)(
        ggml_backend_t backend, uint64_t run_id,
        uint32_t region, uint32_t chunk);

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
#ifdef __cplusplus
static_assert(sizeof(ggml_backend_rpc_p2c_frame) == 48,
    "P2C frame wire header must be 48 bytes");
#endif
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
