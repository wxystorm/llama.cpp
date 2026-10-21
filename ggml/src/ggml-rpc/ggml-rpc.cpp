#include "ggml-rpc.h"
#include "ggml-opencl.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-quants.h"
#include "transport.h"
#include "inference-transport.h"
#include "inference-cycle-transport.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <set>
#include <tuple>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <functional>
#include <algorithm>
#include <climits>
#include <cstdlib>

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

static constexpr size_t RPC_ROUTE_TRANSFER_LANES = 4;
static constexpr size_t RPC_RETURN_TRANSFER_LANES = 2;

static bool rpc_tensor_phone_stage_profile_enabled() {
    return std::getenv("GGML_META_TENSOR_PHONE_STAGE_PROFILE") != nullptr;
}

struct rpc_tensor_phone_graph_stage {
    const char * stage = "graph_compute";
    const char * mode = "unknown";
    int layer = -1;
    int chunk = -1;
};

static rpc_tensor_phone_graph_stage rpc_tensor_phone_graph_stage_identity(
        const ggml_cgraph * graph) {
    rpc_tensor_phone_graph_stage result;
    if (graph == nullptr) {
        return result;
    }

    // Prefer the FFN output boundary when a graph contains more than one
    // recognizable Tensor-Phone marker.
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        const char * name = node != nullptr ? node->name : nullptr;
        if (name == nullptr) {
            continue;
        }

        int chunk = -1;
        int layer = -1;
        int parsed = 0;
        if (std::sscanf(
                name,
                "prefill_ffn_down_chunk_%d-%d%n",
                &chunk,
                &layer,
                &parsed) == 2 &&
                name[parsed] == '\0') {
            result.stage = "ffn_device_compute";
            result.mode = "prefill";
            result.layer = layer;
            result.chunk = chunk;
            return result;
        }

        parsed = 0;
        if (std::sscanf(
                name,
                "ffn_down_chunk_%d-%d%n",
                &chunk,
                &layer,
                &parsed) == 2 &&
                name[parsed] == '\0') {
            result.stage = "ffn_device_compute";
            result.mode = "decode";
            result.layer = layer;
            result.chunk = chunk;
            return result;
        }

        parsed = 0;
        if (std::sscanf(name, "ffn_moe_out-%d%n", &layer, &parsed) == 1 &&
                name[parsed] == '\0') {
            result.stage = "ffn_device_compute";
            result.mode = "decode";
            result.layer = layer;
            result.chunk = 0;
            return result;
        }
    }

    // Route/norm graphs are useful too, but are secondary to the FFN marker.
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        const char * name = node != nullptr ? node->name : nullptr;
        if (name == nullptr) {
            continue;
        }

        int chunk = -1;
        int layer = -1;
        int parsed = 0;
        if ((std::sscanf(
                 name,
                 "phone_prefill_route_weights_chunk_%d-%d%n",
                 &chunk,
                 &layer,
                 &parsed) == 2 ||
             std::sscanf(
                 name,
                 "phone_prefill_route_topk_chunk_%d-%d%n",
                 &chunk,
                 &layer,
                 &parsed) == 2 ||
             std::sscanf(
                 name,
                 "prefill_ffn_norm_chunk_%d-%d%n",
                 &chunk,
                 &layer,
                 &parsed) == 2) &&
                name[parsed] == '\0') {
            result.stage = "pre_route_device_compute";
            result.mode = "prefill";
            result.layer = layer;
            result.chunk = chunk;
            return result;
        }

        parsed = 0;
        if ((std::sscanf(
                 name,
                 "phone_moe_route_weights-%d%n",
                 &layer,
                 &parsed) == 1 ||
             std::sscanf(
                 name,
                 "phone_moe_route_topk-%d%n",
                 &layer,
                 &parsed) == 1) &&
                name[parsed] == '\0') {
            result.stage = "pre_route_device_compute";
            result.mode = "decode";
            result.layer = layer;
            result.chunk = 0;
            return result;
        }
    }

    return result;
}

static void rpc_tensor_phone_log_graph_compute(
        const char * kind,
        uint64_t graph_uid,
        const ggml_cgraph * graph,
        int64_t compute_us,
        int64_t post_compute_us) {
    if (!rpc_tensor_phone_stage_profile_enabled()) {
        return;
    }

    const rpc_tensor_phone_graph_stage stage =
        rpc_tensor_phone_graph_stage_identity(graph);
    const char * first =
        graph != nullptr && graph->n_nodes > 0 && graph->nodes[0] != nullptr ?
            graph->nodes[0]->name : "(none)";
    const char * last =
        graph != nullptr && graph->n_nodes > 0 &&
                graph->nodes[graph->n_nodes - 1] != nullptr ?
            graph->nodes[graph->n_nodes - 1]->name : "(none)";

    GGML_LOG_INFO(
        "[TENSOR_PHONE_RPC_STAGE] side=phone stage=%s kind=%s "
        "mode=%s layer=%d chunk=%d graph_uid=%" PRIu64
        " nodes=%d first=%s last=%s compute_sync_ms=%.3f "
        "post_compute_ms=%.3f\n",
        stage.stage,
        kind,
        stage.mode,
        stage.layer,
        stage.chunk,
        graph_uid,
        graph != nullptr ? graph->n_nodes : 0,
        first,
        last,
        compute_us / 1000.0,
        post_compute_us / 1000.0);
}

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id; //服务端会根据id来重建图
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;    //张量节点要进行的计算
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4]; //填充字段
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_SET_TENSOR_FROM_LOCAL_FILE,
    RPC_CMD_SYNCHRONIZE,
    RPC_CMD_SNAPSHOT_TENSOR,
    RPC_CMD_GET_SNAPSHOT,
    RPC_CMD_GRAPH_RECOMPUTE_SNAPSHOT,
    RPC_CMD_SET_TENSOR_RECOMPUTE_SNAPSHOT,
    RPC_CMD_INIT_ZERO_TENSOR,
    RPC_CMD_GET_TENSOR_BATCH3,
    RPC_CMD_SET_TENSOR_GRAPH_COMPUTE,
    RPC_CMD_ROUTE_MARK_READY,
    RPC_CMD_GET_TENSOR_BATCH3_WAIT,
    RPC_CMD_SET_TENSOR_ASYNC_RETURN,
    RPC_CMD_ROUTE_SNAPSHOT_READY,
    RPC_CMD_GET_ROUTE_SNAPSHOT,
    RPC_CMD_PHONE_FFN_MARK_READY,
    RPC_CMD_SET_TENSOR_ASYNC_RETURN_WAIT,
    RPC_CMD_ROUTE_SNAPSHOT_READY_ASYNC,
    RPC_CMD_RETURN_BULK_ACQUIRE,
    RPC_CMD_INFERENCE_DATA_PLANE_OPEN,
    RPC_CMD_INFERENCE_RETURN_BIND,
    RPC_CMD_INFERENCE_ROUTE_ARM,
    RPC_CMD_INFERENCE_CYCLE_SEND,
    RPC_CMD_INFERENCE_CYCLE_APPLY,
    RPC_CMD_INFERENCE_CYCLE_FINISH,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");
static_assert(RPC_CMD_SET_TENSOR_FROM_LOCAL_FILE == 17, "RPC_CMD_SET_TENSOR_FROM_LOCAL_FILE must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_SYNCHRONIZE == 18, "RPC_CMD_SYNCHRONIZE must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_SNAPSHOT_TENSOR == 19, "RPC_CMD_SNAPSHOT_TENSOR must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_GET_SNAPSHOT == 20, "RPC_CMD_GET_SNAPSHOT must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_GRAPH_RECOMPUTE_SNAPSHOT == 21, "RPC_CMD_GRAPH_RECOMPUTE_SNAPSHOT must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_SET_TENSOR_RECOMPUTE_SNAPSHOT == 22,"RPC_CMD_SET_TENSOR_RECOMPUTE_SNAPSHOT must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_INIT_ZERO_TENSOR == 23, "RPC_CMD_INIT_ZERO_TENSOR must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_GET_TENSOR_BATCH3 == 24, "RPC_CMD_GET_TENSOR_BATCH3 must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_SET_TENSOR_GRAPH_COMPUTE == 25, "RPC_CMD_SET_TENSOR_GRAPH_COMPUTE must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_ROUTE_MARK_READY == 26, "RPC_CMD_ROUTE_MARK_READY must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_GET_TENSOR_BATCH3_WAIT == 27, "RPC_CMD_GET_TENSOR_BATCH3_WAIT must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_SET_TENSOR_ASYNC_RETURN == 28, "RPC_CMD_SET_TENSOR_ASYNC_RETURN must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_ROUTE_SNAPSHOT_READY == 29, "RPC_CMD_ROUTE_SNAPSHOT_READY must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_GET_ROUTE_SNAPSHOT == 30, "RPC_CMD_GET_ROUTE_SNAPSHOT must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_PHONE_FFN_MARK_READY == 31, "RPC_CMD_PHONE_FFN_MARK_READY must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_SET_TENSOR_ASYNC_RETURN_WAIT == 32, "RPC_CMD_SET_TENSOR_ASYNC_RETURN_WAIT must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_ROUTE_SNAPSHOT_READY_ASYNC == 33, "RPC_CMD_ROUTE_SNAPSHOT_READY_ASYNC must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_RETURN_BULK_ACQUIRE == 34, "RPC_CMD_RETURN_BULK_ACQUIRE must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_INFERENCE_DATA_PLANE_OPEN == 35, "RPC_CMD_INFERENCE_DATA_PLANE_OPEN must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_INFERENCE_RETURN_BIND == 36, "RPC_CMD_INFERENCE_RETURN_BIND must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_INFERENCE_ROUTE_ARM == 37, "RPC_CMD_INFERENCE_ROUTE_ARM must be before RPC_CMD_COUNT");
static_assert(RPC_CMD_INFERENCE_CYCLE_SEND == 38, "RPC_CMD_INFERENCE_CYCLE_SEND must remain 38");
static_assert(RPC_CMD_INFERENCE_CYCLE_APPLY == 39, "RPC_CMD_INFERENCE_CYCLE_APPLY must be 39");
static_assert(RPC_CMD_INFERENCE_CYCLE_FINISH == 40, "RPC_CMD_INFERENCE_CYCLE_FINISH must be last");
// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_inference_data_plane_open_req {
    uint32_t tx_slots;
    uint32_t rx_slots;
    uint64_t max_payload_bytes;
};

struct rpc_msg_inference_data_plane_open_rsp {
    uint32_t result;
    uint32_t tx_slots;
    uint32_t rx_slots;
    uint32_t reserved;
    uint64_t max_payload_bytes;
};

static_assert(sizeof(rpc_msg_inference_data_plane_open_req) == 16,
              "inference data-plane open request wire size changed");
static_assert(sizeof(rpc_msg_inference_data_plane_open_rsp) == 24,
              "inference data-plane open response wire size changed");

// Control-plane instruction only: the payload originates from the PHONE
// server's existing GPU tensor buffers, never from host-uploaded PC values.
// One matching READY and one PARTIAL may use the same layer/chunk/seq/graph.
struct rpc_msg_inference_cycle_send_req {
    uint32_t device;
    uint32_t kind;
    int32_t layer;
    int32_t chunk;
    uint64_t seq;
    uint64_t graph_id;
    uint32_t tokens;
    uint32_t hidden;
    uint32_t topk;
    uint32_t reserved;
    rpc_tensor tensors[4];
};
struct rpc_msg_inference_cycle_send_rsp {
    uint32_t result;
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_init_zero_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_init_zero_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_get_tensor_batch3_req {
    rpc_msg_get_tensor_req items[3];
};

struct rpc_msg_get_tensor_batch3_wait_req {
    uint32_t device;
    uint32_t reserved;
    uint64_t wait_seq;
    rpc_msg_get_tensor_batch3_req batch;
};

struct rpc_msg_route_mark_ready_req {
    uint32_t device;
    uint64_t seq;
};

struct rpc_msg_phone_ffn_mark_ready_req {
    uint32_t device;
    uint64_t seq;
};

// A pre-compute marker used only by the Phone-primary Chunk Ring.  Meta sends
// seq|ARM before the Phone FFN graph.  The normal unmodified seq is sent again
// after the graph completes.  Keeping the mode in the seq avoids widening the
// existing RPC command and preserves old clients/servers.
static constexpr uint64_t RPC_PHONE_FFN_CPU_REDUCE_ARM =
    UINT64_C(1) << 63;

static constexpr uint64_t RPC_RETURN_WIRE_MAGIC =
    UINT64_C(0x4650313652455455);

enum rpc_return_wire_type : uint32_t {
    RPC_RETURN_WIRE_RAW = 0,
    RPC_RETURN_WIRE_F16 = 1,
};

// v19: the sender asks the Phone server for an immediate receipt ACK after the
// complete Return payload has reached Phone userspace. This ACK releases the
// PC-side one-Return-at-a-time wire mutex before Phone FFN wait/reduce.
static constexpr uint32_t RPC_RETURN_WIRE_EARLY_ACK_FLAG =
    UINT32_C(1) << 0;

struct rpc_msg_set_tensor_async_return_wait_req_v14 {
    uint32_t device;
    uint64_t phone_ffn_seq;
    rpc_tensor tensor;
    uint64_t offset;
};

struct rpc_msg_set_tensor_async_return_wait_req {
    uint32_t device;
    uint64_t phone_ffn_seq;
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t wire_magic;
    uint32_t wire_type;
    uint32_t reserved;
    uint64_t original_size;
};

struct rpc_msg_inference_return_bind_req {
    rpc_msg_set_tensor_async_return_wait_req return_req;
    uint64_t graph_id;
    uint64_t wire_size;
    int32_t layer;
    int32_t chunk;
};

struct rpc_msg_inference_return_bind_rsp {
    uint8_t result;
};

static_assert(
    offsetof(rpc_msg_set_tensor_async_return_wait_req, wire_magic) ==
        sizeof(rpc_msg_set_tensor_async_return_wait_req_v14),
    "return-wait v15 request must preserve the v14 prefix");

struct rpc_msg_route_snapshot_ready_req {
    uint32_t device;
    uint32_t lane;
    uint64_t seq;
    rpc_tensor tensors[3];
    uint64_t sizes[3];
};

struct rpc_msg_route_snapshot_ready_rsp {
    uint8_t result;
};

struct rpc_msg_get_route_snapshot_req {
    uint32_t device;
    uint32_t lane;
    uint64_t seq;
    uint64_t sizes[3];
};

struct rpc_msg_inference_route_arm_req {
    rpc_msg_get_route_snapshot_req snapshot;
    int32_t layer;
    int32_t chunk;
};

struct rpc_msg_inference_route_arm_rsp {
    uint8_t result;
    uint8_t reserved[7];
    uint64_t wire_size;
};

static_assert(
    sizeof(rpc_msg_inference_route_arm_req) == 48,
    "inference route arm request wire size changed");
static_assert(
    sizeof(rpc_msg_inference_route_arm_rsp) == 16,
    "inference route arm response wire size changed");

struct rpc_msg_return_bulk_acquire_req {
    uint64_t seq;
};

struct rpc_msg_return_bulk_acquire_rsp {
    uint8_t active;
    uint8_t reserved[7];
    uint64_t inflight;
    uint64_t limit;
};

struct rpc_msg_set_tensor_graph_req_v6 {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t graph_size;
};

struct rpc_msg_set_tensor_graph_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t graph_size;
    uint64_t graph_uid;
    uint32_t device;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t graph_uid;
};

struct rpc_msg_graph_recompute_snapshot_req {
    uint32_t device;
    uint32_t slot;
    uint64_t graph_uid;
    uint64_t seq;
    uint32_t output_node;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_set_tensor_recompute_snapshot_req {
    rpc_msg_graph_recompute_snapshot_req graph;
    rpc_tensor input_tensor;
    uint64_t input_offset;
    uint64_t input_size;
};

static constexpr size_t FUSED_RPC_PREFIX = 1 + sizeof(uint64_t);
static constexpr size_t FUSED_REQ_OFFSET = FUSED_RPC_PREFIX;
static constexpr size_t FUSED_DATA_OFFSET = FUSED_REQ_OFFSET + sizeof(rpc_msg_set_tensor_recompute_snapshot_req);

struct rpc_msg_synchronize_req {
    uint32_t device;
};

struct rpc_msg_snapshot_tensor_req {
    uint32_t device;
    uint32_t slot;
    uint64_t seq;
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_get_snapshot_req {
    uint32_t device;
    uint32_t slot;
    uint64_t seq;
    uint64_t size;
};

struct rpc_msg_set_tensor_from_local_file_rsp {
    // 0:不支持本地文件加载
    // 1:已经成功处理
    // 2:处理失败
    uint8_t result;
 };
//新加的
struct rpc_msg_set_tensor_from_local_file_req {
    // 含远端目标 buffer/data、形状、类型、名称等
    rpc_tensor tensor;

    // 相对于本地完整源张量起点
    uint64_t src_offset;

    // 相对于远端目标 tensor.data 起点
    uint64_t dst_offset;

    // 每次复制的字节数
    uint64_t copy_size;

    // 复制次数
    uint64_t n_copies;

    // 相邻两次复制的源起点间距
    uint64_t src_stride;

    // 相邻两次复制的目标起点间距
    uint64_t dst_stride;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct rpc_pending_fused_ffn_input {
    std::mutex mutex;

    // Meta 告诉 RPC：
    // 下一次针对这个 tensor 的 SET_TENSOR 要暂存。
    bool capture_next_set = false;
    const ggml_tensor * expected_tensor = nullptr;

    // 已经截获到的数据
    bool pending = false;

    const ggml_tensor * tensor = nullptr;
    size_t data_size = 0;
    std::vector<uint8_t> packet;
};

struct rpc_snapshot_ready_context {
    std::mutex mutex;
    std::condition_variable cv;
    std::unordered_set<uint64_t> ready_seqs;
};

struct rpc_snapshot_client_stats {
    std::atomic<uint64_t> transfer_count { 0 };
    std::atomic<uint64_t> payload_bytes { 0 };
    std::atomic<int64_t> request_us { 0 };
    std::atomic<int64_t> ready_first_byte_us { 0 };
    std::atomic<int64_t> recv_payload_us { 0 };
    std::atomic<int64_t> total_us { 0 };
};

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    std::unordered_set<uint64_t> graph_uids;

    std::mutex fused_set_graph_cache_mutex;
    std::unordered_set<uint64_t> fused_set_graph_uids;
    const socket_t * fused_set_graph_cache_socket = nullptr;

    rpc_pending_fused_ffn_input fused_ffn;
    rpc_snapshot_ready_context snapshot_ready;
    rpc_snapshot_client_stats snapshot_client_stats;
    std::array<std::shared_ptr<socket_t>, RPC_ROUTE_TRANSFER_LANES> route_transfer_socks {};
    std::array<std::shared_ptr<socket_t>, RPC_RETURN_TRANSFER_LANES> return_transfer_socks {};

    // v19 directional network policy:
    //   - at most one PC->Phone Return payload on the wire;
    //   - Route is limited independently on the Phone server;
    //   - one Route and one Return may overlap.
    // A lane mutex remains held through the final RPC response so the same
    // socket cannot be reused while its previous request is still draining.
    std::mutex return_wire_mutex;
    std::array<std::mutex, RPC_RETURN_TRANSFER_LANES> return_lane_mutex {};

    // Legacy v18 shared-credit discovery/cache, kept only for old servers.
    std::atomic<int> network_bulk_shared_active { -1 };
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;

    // RPC GET_ALLOC_SIZE is called repeatedly by gallocr for the same logical
    // allocation request. Keep two bounded caches:
    //   exact:      includes tensor identities and catches reserve/alloc repeats;
    //   structural: ignores graph-local identities and catches equal op/shape
    //               requests across layers/chunks.
    std::mutex alloc_size_cache_mutex;
    std::unordered_map<std::string, size_t> alloc_size_exact_cache;
    std::unordered_map<std::string, size_t> alloc_size_structural_cache;

    uint64_t alloc_size_lookups          = 0;
    uint64_t alloc_size_exact_hits       = 0;
    uint64_t alloc_size_structural_hits  = 0;
    uint64_t alloc_size_misses           = 0;
    uint64_t alloc_size_exact_clears      = 0;
    uint64_t alloc_size_structural_clears = 0;
    uint64_t alloc_size_connection_resets = 0;
    int64_t  alloc_size_miss_rpc_us       = 0;
    const socket_t * alloc_size_cache_socket = nullptr;
};

struct rpc_graph_snapshot_context {
    std::mutex mutex;
    bool active = false;
    const ggml_tensor * tensor = nullptr;
    size_t offset = 0;
    size_t size = 0;
    uint32_t slot = 0;
    uint64_t seq = 0;
    bool snapshot_was_fused = false;
    uint64_t fused_snapshot_seq = 0;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    rpc_graph_snapshot_context graph_snapshot;

    // Diagnostic-only record of the most recent graph successfully submitted
    // on this RPC backend.  Used by GGML_META_XLAYER_BOUNDARY_DIAG to tell
    // whether a later coarse RPC fence is still draining the predecessor FFN
    // or has also caught next-layer ahead work.  This state never participates
    // in execution ordering or correctness.
    std::mutex fence_diag_mutex;
    uint64_t fence_diag_submit_seq = 0;
    uint64_t fence_diag_graph_uid = 0;
    int fence_diag_nodes = 0;
    std::string fence_diag_cmd;
    std::string fence_diag_first_node;
    std::string fence_diag_marker_node;
    std::string fence_diag_last_node;
};

static bool rpc_xlayer_boundary_diag_enabled() {
    return std::getenv("GGML_META_XLAYER_BOUNDARY_DIAG") != nullptr;
}

static bool rpc_fence_diag_marker_name(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return false;
    }

    return std::strstr(name, "prefill_wave_") != nullptr ||
           std::strstr(name, "prefill_ffn_") != nullptr ||
           std::strstr(name, "phone_prefill_route_") != nullptr ||
           std::strstr(name, "ffn_moe_") != nullptr ||
           std::strstr(name, "attn_") != nullptr;
}

static void rpc_fence_diag_record_graph(
        ggml_backend_rpc_context * rpc_ctx,
        uint64_t                   graph_uid,
        const ggml_cgraph        * cgraph,
        const char               * cmd) {
    if (!rpc_xlayer_boundary_diag_enabled() ||
            rpc_ctx == nullptr ||
            cgraph == nullptr) {
        return;
    }

    std::string first_node;
    std::string marker_node;
    std::string last_node;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (node == nullptr || node->name[0] == '\0') {
            continue;
        }

        if (first_node.empty()) {
            first_node = node->name;
        }
        last_node = node->name;

        if (rpc_fence_diag_marker_name(node->name)) {
            marker_node = node->name;
        }
    }

    std::lock_guard<std::mutex> lock(rpc_ctx->fence_diag_mutex);
    rpc_ctx->fence_diag_submit_seq += 1;
    rpc_ctx->fence_diag_graph_uid = graph_uid;
    rpc_ctx->fence_diag_nodes = cgraph->n_nodes;
    rpc_ctx->fence_diag_cmd = cmd != nullptr ? cmd : "unknown";
    rpc_ctx->fence_diag_first_node = std::move(first_node);
    rpc_ctx->fence_diag_marker_node = std::move(marker_node);
    rpc_ctx->fence_diag_last_node = std::move(last_node);
}

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    std::shared_ptr<socket_t> transfer_sock;

    std::string endpoint;
    uint32_t device;

    void * base_ptr;
    uint64_t remote_ptr;

    ggml_backend_rpc_device_context * device_ctx = nullptr;
    std::array<std::shared_ptr<socket_t>, 2> snapshot_transfer_socks {};
};

static bool ggml_backend_rpc_set_tensor_graph(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src,
        ggml_tensor * dst,
        const ggml_cgraph * graph);

static bool ggml_backend_rpc_prepare_fused_ffn_input(
        ggml_backend_t backend,
        const ggml_tensor * tensor) {

    auto * rpc_dev_ctx =
        static_cast<ggml_backend_rpc_device_context *>(
            ggml_backend_get_device(backend)->context);

    auto & fused = rpc_dev_ctx->fused_ffn;

    std::lock_guard<std::mutex> lock(
        fused.mutex);

    // 第一版这里严格一点，发现旧状态说明时序有问题。
    GGML_ASSERT(!fused.capture_next_set);
    GGML_ASSERT(!fused.pending);

    fused.capture_next_set = true;
    fused.expected_tensor = tensor;

    return true;
}
// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}
template <typename T>
static bool send_rpc_cmd_compact_small(
        socket_ptr sock,
        enum rpc_cmd cmd,
        const T & request) {

    constexpr size_t request_size = sizeof(T);

    std::array<uint8_t,
        1 + sizeof(uint64_t) + request_size> packet {};

    size_t pos = 0;

    packet[pos++] = static_cast<uint8_t>(cmd);

    const uint64_t size64 = request_size;

    memcpy(
        packet.data() + pos,
        &size64,
        sizeof(size64));

    pos += sizeof(size64);

    memcpy(
        packet.data() + pos,
        &request,
        request_size);

    pos += request_size;

    GGML_ASSERT(pos == packet.size());

    return sock->send_data(
        packet.data(),
        packet.size());
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(
        socket_ptr sock,
        enum rpc_cmd cmd,
        const void * input,
        size_t input_size) {

    std::array<uint8_t, 1 + sizeof(uint64_t)> header {};

    header[0] =
        static_cast<uint8_t>(cmd);

    const uint64_t wire_size =
        static_cast<uint64_t>(input_size);

    memcpy(
        header.data() + 1,
        &wire_size,
        sizeof(wire_size));

    // cmd + request_size 一次发送
    if (!sock->send_data(
            header.data(),
            header.size())) {
        return false;
    }

    // payload 为空时不要再调用 send_data
    if (input_size == 0) {
        return true;
    }

    // payload 单独一次发送
    return sock->send_data(
        input,
        input_size);
}

struct rpc_cmd_roundtrip_profile {
    int64_t request_send_us         = 0;
    int64_t response_header_wait_us = 0;
    int64_t response_body_wait_us   = 0;
};

static thread_local rpc_cmd_roundtrip_profile *
    rpc_cmd_roundtrip_profile_active = nullptr;

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    rpc_cmd_roundtrip_profile * const profile =
        rpc_cmd_roundtrip_profile_active;

    const int64_t request_send_begin_us =
        profile != nullptr ? ggml_time_us() : 0;
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    const int64_t request_send_done_us =
        profile != nullptr ? ggml_time_us() : 0;

    uint64_t out_size;
    const int64_t response_header_begin_us =
        profile != nullptr ? ggml_time_us() : 0;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    const int64_t response_header_done_us =
        profile != nullptr ? ggml_time_us() : 0;

    if (out_size != output_size) {
        return false;
    }

    const int64_t response_body_begin_us =
        profile != nullptr ? ggml_time_us() : 0;
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    const int64_t response_body_done_us =
        profile != nullptr ? ggml_time_us() : 0;

    if (profile != nullptr) {
        profile->request_send_us =
            request_send_done_us - request_send_begin_us;
        profile->response_header_wait_us =
            response_header_done_us - response_header_begin_us;
        profile->response_body_wait_us =
            response_body_done_us - response_body_begin_us;
    }

    return true;
}

static thread_local ggml_backend_rpc_stage_ready_callback_t rpc_stage_ready_callback = nullptr;
static thread_local void * rpc_stage_ready_user_data = nullptr;
static thread_local int rpc_route_transfer_lane = -1;
static thread_local uint64_t rpc_route_wait_seq = 0;

struct rpc_snapshot_read_context {
    bool active = false;
    uint32_t slot = 0;
    uint64_t seq = 0;
};

static thread_local rpc_snapshot_read_context rpc_snapshot_read;

static void ggml_backend_rpc_set_stage_ready_callback(
        ggml_backend_rpc_stage_ready_callback_t callback,
        void * user_data) {
    rpc_stage_ready_callback = callback;
    rpc_stage_ready_user_data = user_data;
}

static void ggml_backend_rpc_set_route_transfer_lane(int lane) {
    GGML_ASSERT(lane >= -1 && lane < static_cast<int>(RPC_ROUTE_TRANSFER_LANES));
    rpc_route_transfer_lane = lane;
}

static void ggml_backend_rpc_set_route_wait_seq(uint64_t seq) {
    rpc_route_wait_seq = seq;
}

static void ggml_backend_rpc_set_snapshot_read(bool enabled, uint32_t slot, uint64_t seq) {
    rpc_snapshot_read.active = enabled;
    rpc_snapshot_read.slot = slot;
    rpc_snapshot_read.seq = seq;
}

static void rpc_mark_snapshot_ready(
        ggml_backend_rpc_device_context * device_ctx,
        uint64_t seq) {
    if (device_ctx == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(device_ctx->snapshot_ready.mutex);
        device_ctx->snapshot_ready.ready_seqs.insert(seq);
    }
    device_ctx->snapshot_ready.cv.notify_all();
}

static bool ggml_backend_rpc_get_snapshot_stats(
        ggml_backend_t backend,
        ggml_backend_rpc_snapshot_stats * stats) {
    if (backend == nullptr || stats == nullptr) {
        return false;
    }
    auto * dev = ggml_backend_get_device(backend);
    if (dev == nullptr || dev->context == nullptr) {
        return false;
    }
    auto * ctx =
        static_cast<ggml_backend_rpc_device_context *>(dev->context);
    stats->transfer_count =
        ctx->snapshot_client_stats.transfer_count.load(std::memory_order_relaxed);
    stats->payload_bytes =
        ctx->snapshot_client_stats.payload_bytes.load(std::memory_order_relaxed);
    stats->request_us =
        ctx->snapshot_client_stats.request_us.load(std::memory_order_relaxed);
    stats->ready_first_byte_us =
        ctx->snapshot_client_stats.ready_first_byte_us.load(std::memory_order_relaxed);
    stats->recv_payload_us =
        ctx->snapshot_client_stats.recv_payload_us.load(std::memory_order_relaxed);
    stats->total_us =
        ctx->snapshot_client_stats.total_us.load(std::memory_order_relaxed);
    return true;
}

static bool ggml_backend_rpc_wait_snapshot_ready(
        ggml_backend_t backend,
        uint64_t seq) {
    auto * dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return false;
    }

    auto * device_ctx =
        static_cast<ggml_backend_rpc_device_context *>(dev->context);
    if (device_ctx == nullptr) {
        return false;
    }

    std::unique_lock<std::mutex> lock(device_ctx->snapshot_ready.mutex);
    device_ctx->snapshot_ready.cv.wait(lock, [&]() {
        return device_ctx->snapshot_ready.ready_seqs.find(seq) !=
               device_ctx->snapshot_ready.ready_seqs.end();
    });
    device_ctx->snapshot_ready.ready_seqs.erase(seq);
    return true;
}

static bool send_rpc_cmd_staged(
        socket_ptr sock,
        enum rpc_cmd cmd,
        const void * input,
        size_t input_size,
        void * output,
        size_t output_size,
        rpc_cmd_roundtrip_profile * profile = nullptr) {

    const int64_t t0 = ggml_time_us();

    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }

    const int64_t t_request_done = ggml_time_us();

    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }

    const int64_t t_stage = ggml_time_us();

    if (out_size != output_size) {
        return false;
    }

    if (rpc_stage_ready_callback != nullptr) {
        rpc_stage_ready_callback(
            rpc_stage_ready_user_data);
    }

    if (!sock->recv_data(output, output_size)) {
        return false;
    }

    const int64_t t_done = ggml_time_us();

    if (profile != nullptr) {
        profile->request_send_us =
            t_request_done - t0;
        profile->response_header_wait_us =
            t_stage - t_request_done;
        profile->response_body_wait_us =
            t_done - t_stage;
    }

    if (RPC_DEBUG) {
        printf(
            "[RPC_STAGED_GET] bytes=%zu "
            "request=%.3f ms "
            "to_stage=%.3f ms "
            "payload=%.3f ms "
            "total=%.3f ms\n",
            output_size,
            (t_request_done - t0) / 1000.0,
            (t_stage - t_request_done) / 1000.0,
            (t_done - t_stage) / 1000.0,
            (t_done - t0) / 1000.0);
    }

    return true;
}

// RPC client-side implementation

static std::mutex rpc_remote_version_mutex;
static std::unordered_map<std::string, uint8_t> rpc_remote_patch_by_socket_key;

static void rpc_record_remote_patch(
        const std::string & socket_key,
        uint8_t patch) {
    std::lock_guard<std::mutex> lock(rpc_remote_version_mutex);
    rpc_remote_patch_by_socket_key[socket_key] = patch;
}

static uint8_t rpc_get_remote_patch(
        const std::string & socket_key) {
    std::lock_guard<std::mutex> lock(rpc_remote_version_mutex);
    auto it = rpc_remote_patch_by_socket_key.find(socket_key);
    return it == rpc_remote_patch_by_socket_key.end() ? 0 : it->second;
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(
        const std::shared_ptr<socket_t> & sock,
        uint8_t * remote_patch) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    if (remote_patch != nullptr) {
        *remote_patch = response.patch;
    }

    sock->update_caps(response.conn_caps);
    return true;
}
enum class rpc_socket_role {
    COMPUTE,
    TRANSFER,
    SNAPSHOT_TRANSFER_0,
    SNAPSHOT_TRANSFER_1,
    ROUTE_TRANSFER_0,
    ROUTE_TRANSFER_1,
    ROUTE_TRANSFER_2,
    ROUTE_TRANSFER_3,
    RETURN_TRANSFER_0,
    RETURN_TRANSFER_1,
};

static std::shared_ptr<socket_t> get_socket_role(const std::string & endpoint, rpc_socket_role role) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;
    const char * suffix = nullptr;
    switch (role) {
        case rpc_socket_role::COMPUTE:
            suffix = "_compute";
            break;
        case rpc_socket_role::TRANSFER:
            suffix = "_transfer";
            break;
        case rpc_socket_role::SNAPSHOT_TRANSFER_0:
            suffix = "_snapshot_transfer_0";
            break;
        case rpc_socket_role::SNAPSHOT_TRANSFER_1:
            suffix = "_snapshot_transfer_1";
            break;
        case rpc_socket_role::ROUTE_TRANSFER_0:
            suffix = "_route_transfer_0";
            break;
        case rpc_socket_role::ROUTE_TRANSFER_1:
            suffix = "_route_transfer_1";
            break;
        case rpc_socket_role::ROUTE_TRANSFER_2:
            suffix = "_route_transfer_2";
            break;
        case rpc_socket_role::ROUTE_TRANSFER_3:
            suffix = "_route_transfer_3";
            break;
        case rpc_socket_role::RETURN_TRANSFER_0:
            suffix = "_return_transfer_0";
            break;
        case rpc_socket_role::RETURN_TRANSFER_1:
            suffix = "_return_transfer_1";
            break;
    }
    std::string key = endpoint + suffix;
    auto it = sockets.find(key);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    uint8_t remote_patch = 0;
    if (!negotiate_hello(sock, &remote_patch)) {
        return nullptr;
    }
    rpc_record_remote_patch(key, remote_patch);
    LOG_DBG(
        "[%s] connected to %s remote_rpc=%d.%d.%d\n",
        __func__, endpoint.c_str(),
        RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION, remote_patch);
    sockets[key] = sock;
    return sock;
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    return get_socket_role(endpoint, rpc_socket_role::COMPUTE);
}

static std::shared_ptr<socket_t> get_transfer_socket(const std::string & endpoint) {
    return get_socket_role(endpoint, rpc_socket_role::TRANSFER);
}

static std::shared_ptr<socket_t> get_snapshot_transfer_socket(const std::string & endpoint, size_t lane) {
    GGML_ASSERT(lane < 2);
    return get_socket_role(endpoint, lane == 0 ? rpc_socket_role::SNAPSHOT_TRANSFER_0
                                               : rpc_socket_role::SNAPSHOT_TRANSFER_1);
}

static std::shared_ptr<socket_t> get_route_transfer_socket(const std::string & endpoint, size_t lane) {
    GGML_ASSERT(lane < RPC_ROUTE_TRANSFER_LANES);
    switch (lane) {
        case 0:
            return get_socket_role(endpoint, rpc_socket_role::ROUTE_TRANSFER_0);
        case 1:
            return get_socket_role(endpoint, rpc_socket_role::ROUTE_TRANSFER_1);
        case 2:
            return get_socket_role(endpoint, rpc_socket_role::ROUTE_TRANSFER_2);
        case 3:
            return get_socket_role(endpoint, rpc_socket_role::ROUTE_TRANSFER_3);
        default:
            GGML_ABORT("invalid route transfer lane");
    }
}

static std::shared_ptr<socket_t> get_return_transfer_socket(
        const std::string & endpoint,
        size_t lane) {
    GGML_ASSERT(lane < RPC_RETURN_TRANSFER_LANES);
    return get_socket_role(
        endpoint,
        lane == 0 ?
            rpc_socket_role::RETURN_TRANSFER_0 :
            rpc_socket_role::RETURN_TRANSFER_1);
}

static inference_transport_config rpc_inference_transport_config() {
    inference_transport_config config;
    config.tx_slots = 4;
    config.rx_slots = 4;
    config.max_payload_bytes = 16 * 1024 * 1024;

    if (const char * slots_env =
            std::getenv("GGML_INFERENCE_TRANSPORT_SLOTS")) {
        const long value = std::strtol(slots_env, nullptr, 10);
        if (value >= 1 && value <= 64) {
            config.tx_slots = static_cast<size_t>(value);
            config.rx_slots = static_cast<size_t>(value);
        }
    }

    if (const char * mib_env =
            std::getenv("GGML_INFERENCE_TRANSPORT_MAX_PAYLOAD_MIB")) {
        const long value = std::strtol(mib_env, nullptr, 10);
        if (value >= 1 && value <= 256) {
            config.max_payload_bytes =
                static_cast<size_t>(value) * 1024 * 1024;
        }
    }

    return config;
}

static std::shared_ptr<inference_transport>
get_inference_transport(const std::string & endpoint) {
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        std::shared_ptr<inference_transport>> transports;

    std::lock_guard<std::mutex> lock(mutex);

    auto it = transports.find(endpoint);
    if (it != transports.end()) {
        if (it->second && it->second->healthy()) {
            return it->second;
        }
        if (it->second) {
            it->second->stop();
        }
        transports.erase(it);
    }

    std::string host;
    int port = 0;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR(
            "[INFERENCE_DATA_PLANE] side=client status=FAIL "
            "reason=bad_endpoint endpoint=%s\n",
            endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }

    auto sock = socket_t::connect(host.c_str(), port);
    if (!sock) {
        GGML_LOG_ERROR(
            "[INFERENCE_DATA_PLANE] side=client status=FAIL "
            "reason=connect endpoint=%s\n",
            endpoint.c_str());
        return nullptr;
    }

    uint8_t remote_patch = 0;
    if (!negotiate_hello(sock, &remote_patch)) {
        return nullptr;
    }

    if (remote_patch < 20) {
        GGML_LOG_ERROR(
            "[INFERENCE_DATA_PLANE] side=client status=FAIL "
            "reason=server_too_old endpoint=%s patch=%u required=20\n",
            endpoint.c_str(),
            static_cast<unsigned>(remote_patch));
        return nullptr;
    }

    const inference_transport_config config =
        rpc_inference_transport_config();

    rpc_msg_inference_data_plane_open_req request {};
    request.tx_slots =
        static_cast<uint32_t>(config.tx_slots);
    request.rx_slots =
        static_cast<uint32_t>(config.rx_slots);
    request.max_payload_bytes =
        config.max_payload_bytes;

    rpc_msg_inference_data_plane_open_rsp response {};
    if (!send_rpc_cmd(
            sock,
            RPC_CMD_INFERENCE_DATA_PLANE_OPEN,
            &request,
            sizeof(request),
            &response,
            sizeof(response)) ||
        response.result == 0) {
        GGML_LOG_ERROR(
            "[INFERENCE_DATA_PLANE] side=client status=FAIL "
            "reason=open_rejected endpoint=%s\n",
            endpoint.c_str());
        return nullptr;
    }

    auto adopted = inference_transport::adopt(sock, config);
    if (!adopted) {
        return nullptr;
    }

    std::shared_ptr<inference_transport> transport =
        std::move(adopted);
    transports[endpoint] = transport;

    GGML_LOG_INFO(
        "[INFERENCE_DATA_PLANE] side=client status=OPEN "
        "endpoint=%s tx_slots=%zu rx_slots=%zu "
        "max_payload_mib=%.3f server_tx_slots=%u "
        "server_rx_slots=%u\n",
        endpoint.c_str(),
        config.tx_slots,
        config.rx_slots,
        config.max_payload_bytes / (1024.0 * 1024.0),
        response.tx_slots,
        response.rx_slots);

    return transport;
}

struct rpc_inference_owned_frame {
    inference_frame_header header {};
    std::vector<uint8_t> data;
};

struct rpc_inference_receive_state {
    std::mutex mutex;
    std::condition_variable cv;
    const inference_transport * transport_id = nullptr;
    uint64_t generation = 0;
    bool failed = false;
    std::deque<rpc_inference_owned_frame> pending;
};

static std::shared_ptr<rpc_inference_receive_state>
rpc_inference_get_receive_state(const std::string & endpoint) {
    static std::mutex map_mutex;
    static std::unordered_map<
        std::string,
        std::shared_ptr<rpc_inference_receive_state>> states;

    std::lock_guard<std::mutex> lock(map_mutex);
    auto & state = states[endpoint];
    if (!state) {
        state = std::make_shared<rpc_inference_receive_state>();
    }
    return state;
}

static void rpc_inference_ensure_receiver(
        const std::string & endpoint,
        const std::shared_ptr<inference_transport> & transport,
        const std::shared_ptr<rpc_inference_receive_state> & state) {
    if (!transport || !state) {
        return;
    }

    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->transport_id == transport.get() &&
                !state->failed) {
            return;
        }

        state->transport_id = transport.get();
        generation = ++state->generation;
        state->failed = false;
        state->pending.clear();
    }

    // One application-level dispatcher per live data-plane connection.
    // inference_transport already owns the socket RX thread; this dispatcher
    // only drains ready RX slots and demultiplexes them to Route/Return
    // waiters so no waiter monopolizes receive().
    std::thread(
        [endpoint, transport, state, generation]() {
            constexpr size_t MAX_PENDING_FRAMES = 256;

            while (transport->healthy()) {
                inference_rx_frame frame;
                if (!transport->receive(frame, -1)) {
                    break;
                }

                rpc_inference_owned_frame owned;
                owned.header = frame.header;
                if (frame.size > 0) {
                    owned.data.assign(
                        frame.data,
                        frame.data + frame.size);
                }
                transport->release_rx(std::move(frame));

                bool overflow = false;
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    if (state->generation != generation ||
                            state->transport_id != transport.get()) {
                        return;
                    }

                    if (state->pending.size() >= MAX_PENDING_FRAMES) {
                        state->failed = true;
                        overflow = true;
                    } else {
                        state->pending.push_back(std::move(owned));
                    }
                }
                state->cv.notify_all();

                if (overflow) {
                    GGML_LOG_ERROR(
                        "[INFERENCE_DATA_PLANE] side=client status=FAIL "
                        "reason=demux_overflow endpoint=%s pending=%zu\n",
                        endpoint.c_str(),
                        MAX_PENDING_FRAMES);
                    transport->stop();
                    break;
                }
            }

            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (state->generation == generation &&
                        state->transport_id == transport.get()) {
                    state->failed = true;
                }
            }
            state->cv.notify_all();
        }).detach();
}

static bool rpc_inference_frame_matches(
        const inference_frame_header & header,
        uint16_t type,
        uint16_t stage,
        uint32_t stream_id,
        uint64_t seq) {
    return header.type == type &&
        header.stage == stage &&
        header.stream_id == stream_id &&
        header.seq == seq;
}

static bool rpc_inference_receive_matching(
        const std::string & endpoint,
        uint16_t type,
        uint16_t stage,
        uint32_t stream_id,
        uint64_t seq,
        size_t expected_size,
        std::vector<uint8_t> & output,
        inference_frame_header * header_out = nullptr,
        int timeout_ms = -1) {
    auto transport = get_inference_transport(endpoint);
    if (!transport) {
        return false;
    }

    auto state = rpc_inference_get_receive_state(endpoint);
    rpc_inference_ensure_receiver(endpoint, transport, state);

    std::unique_lock<std::mutex> lock(state->mutex);
    const uint64_t generation = state->generation;

    auto find_match = [&]() {
        return std::find_if(
            state->pending.begin(),
            state->pending.end(),
            [&](const rpc_inference_owned_frame & owned) {
                return rpc_inference_frame_matches(
                    owned.header,
                    type,
                    stage,
                    stream_id,
                    seq);
            });
    };

    auto ready = [&]() {
        return find_match() != state->pending.end() ||
            state->failed ||
            state->generation != generation;
    };

    if (timeout_ms < 0) {
        state->cv.wait(lock, ready);
    } else {
        const bool woke = state->cv.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            ready);
        if (!woke) {
            return false;
        }
    }

    auto it = find_match();
    if (it == state->pending.end()) {
        return false;
    }

    rpc_inference_owned_frame owned = std::move(*it);
    state->pending.erase(it);
    lock.unlock();

    if (owned.data.size() != expected_size) {
        GGML_LOG_ERROR(
            "[INFERENCE_DATA_PLANE] side=client status=FAIL "
            "reason=size_mismatch type=%u stage=%u stream=%u "
            "seq=%" PRIu64 " expected=%zu actual=%zu\n",
            static_cast<unsigned>(type),
            static_cast<unsigned>(stage),
            stream_id,
            seq,
            expected_size,
            owned.data.size());
        return false;
    }

    if (header_out != nullptr) {
        *header_out = owned.header;
    }
    output = std::move(owned.data);
    return true;
}

static bool rpc_inference_receive_any(
        const std::string & endpoint,
        rpc_inference_owned_frame & output,
        int timeout_ms) {
    auto transport = get_inference_transport(endpoint);
    if (!transport) {
        return false;
    }

    auto state = rpc_inference_get_receive_state(endpoint);
    rpc_inference_ensure_receiver(endpoint, transport, state);

    std::unique_lock<std::mutex> lock(state->mutex);
    const uint64_t generation = state->generation;
    auto ready = [&]() {
        return !state->pending.empty() ||
            state->failed ||
            state->generation != generation;
    };

    if (timeout_ms < 0) {
        state->cv.wait(lock, ready);
    } else {
        const bool woke = state->cv.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            ready);
        if (!woke) {
            return false;
        }
    }

    if (state->pending.empty()) {
        return false;
    }

    output = std::move(state->pending.front());
    state->pending.pop_front();
    return true;
}

static ggml_backend_rpc_device_context *
rpc_inference_device_context(ggml_backend_t backend) {
    if (backend == nullptr) {
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr || dev->context == nullptr) {
        return nullptr;
    }

    return static_cast<ggml_backend_rpc_device_context *>(
        dev->context);
}

static bool ggml_backend_rpc_inference_transport_open(
        ggml_backend_t backend) {
    auto * ctx = rpc_inference_device_context(backend);
    return ctx != nullptr &&
        get_inference_transport(ctx->endpoint) != nullptr;
}

static bool ggml_backend_rpc_inference_transport_submit(
        ggml_backend_t backend,
        const ggml_backend_rpc_inference_task * task,
        const void * data,
        size_t size) {
    auto * ctx = rpc_inference_device_context(backend);
    if (ctx == nullptr || task == nullptr ||
        (size > 0 && data == nullptr)) {
        return false;
    }

    auto transport =
        get_inference_transport(ctx->endpoint);
    if (!transport) {
        return false;
    }

    inference_tx_buffer tx =
        transport->acquire_tx(size);
    if (!tx) {
        return false;
    }

    if (size > 0) {
        std::memcpy(tx.data, data, size);
    }

    inference_frame_header header =
        inference_make_header(INFERENCE_FRAME_DATA);
    header.flags = task->flags;
    header.stage = task->stage;
    header.tensor_role = task->tensor_role;
    header.priority = task->priority;
    header.layer = task->layer;
    header.chunk = task->chunk;
    header.stream_id = task->stream_id;
    header.seq = task->seq;
    header.graph_id = task->graph_id;

    return transport->submit_tx(
        std::move(tx),
        header,
        size);
}

static bool ggml_backend_rpc_inference_transport_receive(
        ggml_backend_t backend,
        ggml_backend_rpc_inference_task * task,
        void * data,
        size_t capacity,
        size_t * size,
        int timeout_ms) {
    auto * ctx = rpc_inference_device_context(backend);
    if (ctx == nullptr || task == nullptr ||
        size == nullptr || (capacity > 0 && data == nullptr)) {
        return false;
    }

    rpc_inference_owned_frame frame;
    if (!rpc_inference_receive_any(
            ctx->endpoint,
            frame,
            timeout_ms)) {
        return false;
    }

    const bool fits = frame.data.size() <= capacity;
    if (fits && !frame.data.empty()) {
        std::memcpy(
            data,
            frame.data.data(),
            frame.data.size());
    }

    task->graph_id = frame.header.graph_id;
    task->seq = frame.header.seq;
    task->layer = frame.header.layer;
    task->chunk = frame.header.chunk;
    task->stream_id = frame.header.stream_id;
    task->stage = frame.header.stage;
    task->tensor_role = frame.header.tensor_role;
    task->priority = frame.header.priority;
    task->flags =
        static_cast<uint16_t>(frame.header.flags & 0xffffu);
    *size = frame.data.size();

    return fits;
}

static bool ggml_backend_rpc_inference_transport_get_stats(
        ggml_backend_t backend,
        ggml_backend_rpc_inference_transport_stats * out) {
    auto * ctx = rpc_inference_device_context(backend);
    if (ctx == nullptr || out == nullptr) {
        return false;
    }

    auto transport =
        get_inference_transport(ctx->endpoint);
    if (!transport) {
        return false;
    }

    const inference_transport_stats stats =
        transport->stats();
    out->frames_sent = stats.frames_sent;
    out->frames_received = stats.frames_received;
    out->bytes_sent = stats.bytes_sent;
    out->bytes_received = stats.bytes_received;
    out->tx_acquire_wait_count =
        stats.tx_acquire_wait_count;
    out->rx_slot_wait_count =
        stats.rx_slot_wait_count;
    out->tx_acquire_wait_ms =
        stats.tx_acquire_wait_ms;
    out->rx_slot_wait_ms =
        stats.rx_slot_wait_ms;
    out->tx_queue_wait_ms =
        stats.tx_queue_wait_ms;
    out->tx_queue_wait_max_ms =
        stats.tx_queue_wait_max_ms;
    out->tx_send_ms = stats.tx_send_ms;
    out->rx_recv_ms = stats.rx_recv_ms;
    out->rx_ready_wait_ms =
        stats.rx_ready_wait_ms;
    out->rx_ready_wait_max_ms =
        stats.rx_ready_wait_max_ms;
    out->tx_queue_max = stats.tx_queue_max;
    out->tx_inflight_max = stats.tx_inflight_max;
    out->rx_ready_max = stats.rx_ready_max;
    return true;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer);

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}
// 将ggml_tensor序列化为rpc_tensor结构体，便于在RPC中传输
static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    memset(result.padding, 0, sizeof(result.padding));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

// Attach the independent persistent data-plane socket before issuing the
// phone-local send command. All four tensors must still own live RPC buffers.
// Protocol 26 C2P: ask the Phone RPC server to consume an already-received
// data-plane frame into its own GPU tensors, never upload host activation
// bytes through the control RPC connection.
static bool rpc_cycle_apply_inbound_tensors(
        ggml_backend_t backend, inference_cycle_payload_kind kind,
        int32_t layer, int32_t chunk, uint64_t seq, uint64_t graph_id,
        const ggml_tensor * const * targets,
        uint32_t tokens, uint32_t hidden, uint32_t topk) {
    if (!ggml_backend_is_rpc(backend) || layer < 0 || chunk < 0 ||
        seq == 0 || tokens == 0 || hidden == 0 ||
        (kind == INFERENCE_CYCLE_READY ? topk == 0 : topk != 0)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_rpc_context *>(backend->context);
    if (!ctx || rpc_get_remote_patch(ctx->endpoint + "_compute") < 26 ||
        !get_inference_transport(ctx->endpoint)) {
        GGML_LOG_ERROR("[CYCLE_C2P] status=FAIL reason=PHONE_SERVER_PATCH26_REQUIRED\n");
        return false;
    }
    rpc_msg_inference_cycle_send_req request {};
    request.device = ctx->device;
    request.kind = kind;
    request.layer = layer;
    request.chunk = chunk;
    request.seq = seq;
    request.graph_id = graph_id;
    request.tokens = tokens;
    request.hidden = hidden;
    request.topk = topk;
    const size_t count = kind == INFERENCE_CYCLE_READY ? 4 : 1;
    for (size_t i = 0; i < count; ++i) {
        const ggml_tensor * t = targets[i];
        if (!t || !t->buffer || !ggml_backend_buffer_is_rpc(t->buffer) ||
            !ggml_is_contiguous(t)) {
            return false;
        }
        auto * bctx = static_cast<ggml_backend_rpc_buffer_context *>(t->buffer->context);
        if (!bctx || bctx->endpoint != ctx->endpoint || bctx->device != ctx->device) {
            return false;
        }
        const uint32_t type = i == 2 && kind == INFERENCE_CYCLE_READY ?
            GGML_TYPE_I32 : GGML_TYPE_F32;
        const size_t elements = kind == INFERENCE_CYCLE_READY && i >= 2 ?
            size_t(tokens) * topk : size_t(tokens) * hidden;
        if (t->type != type || ggml_nelements(t) != int64_t(elements) ||
            ggml_nbytes(t) != elements * sizeof(float)) {
            return false;
        }
        request.tensors[i] = serialize_tensor(t);
    }
    auto sock = get_socket(ctx->endpoint);
    if (!sock) {
        return false;
    }
    rpc_msg_inference_cycle_send_rsp response {};
    const bool ok = send_rpc_cmd(
        sock, RPC_CMD_INFERENCE_CYCLE_APPLY,
        &request, sizeof(request), &response, sizeof(response)) &&
        response.result == 1;
    GGML_LOG_INFO("[CYCLE_C2P] side=pc action=PHONE_GPU_APPLY kind=%s "
                  "layer=%d chunk=%d seq=%" PRIu64 " status=%s\n",
                  kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
                  layer, chunk, seq, ok ? "DEVICE_READY" : "FAIL");
    return ok;
}

bool ggml_backend_rpc_cycle_apply_ready(
        ggml_backend_t backend, int32_t layer, int32_t chunk,
        uint64_t seq, uint64_t graph_id,
        const ggml_tensor * residual, const ggml_tensor * norm,
        const ggml_tensor * ids, const ggml_tensor * routing) {
    if (!residual || !norm || !ids || !routing ||
        residual->ne[0] <= 0 || residual->ne[1] <= 0 ||
        ids->ne[0] <= 0 || residual->ne[0] > UINT32_MAX ||
        residual->ne[1] > UINT32_MAX || ids->ne[0] > UINT32_MAX) {
        return false;
    }
    const ggml_tensor * targets[4] = {residual, norm, ids, routing};
    return rpc_cycle_apply_inbound_tensors(
        backend, INFERENCE_CYCLE_READY, layer, chunk, seq, graph_id,
        targets, uint32_t(residual->ne[1]), uint32_t(residual->ne[0]),
        uint32_t(ids->ne[0]));
}

bool ggml_backend_rpc_cycle_apply_partial(
        ggml_backend_t backend, int32_t layer, int32_t chunk,
        uint64_t seq, uint64_t graph_id, const ggml_tensor * partial) {
    if (!partial || partial->ne[0] <= 0 || partial->ne[1] <= 0 ||
        partial->ne[0] > UINT32_MAX || partial->ne[1] > UINT32_MAX) {
        return false;
    }
    const ggml_tensor * targets[4] = {partial};
    return rpc_cycle_apply_inbound_tensors(
        backend, INFERENCE_CYCLE_PARTIAL, layer, chunk, seq, graph_id,
        targets, uint32_t(partial->ne[1]), uint32_t(partial->ne[0]), 0);
}

bool ggml_backend_rpc_cycle_wait_ack(
        ggml_backend_t backend, uint32_t kind,
        int32_t layer, int32_t chunk, uint64_t seq, uint64_t graph_id,
        int timeout_ms) {
    if (!ggml_backend_is_rpc(backend) ||
        layer < 0 || chunk < 0 || seq == 0 ||
        (kind != INFERENCE_CYCLE_READY &&
         kind != INFERENCE_CYCLE_PARTIAL)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_rpc_context *>(backend->context);
    if (!ctx) {
        return false;
    }
    inference_frame_header header {};
    std::vector<uint8_t> payload;
    const bool received = rpc_inference_receive_matching(
        ctx->endpoint, INFERENCE_FRAME_ACK,
        kind == INFERENCE_CYCLE_READY ?
            INFERENCE_STAGE_CYCLE_READY : INFERENCE_STAGE_CYCLE_PARTIAL,
        0, seq, 0, payload, &header, timeout_ms);
    inference_cycle_ack ack {};
    const bool ok = received && inference_cycle_decode_ack(
        header, payload.size(), ack) &&
        ack.direction == INFERENCE_CYCLE_PC_TO_PHONE &&
        ack.kind == kind && ack.layer == layer &&
        ack.chunk == chunk && ack.seq == seq &&
        ack.graph_id == graph_id && ack.device_ready;
    GGML_LOG_INFO(
        "[CYCLE_C2P_ACK] side=pc kind=%s layer=%d chunk=%d "
        "seq=%" PRIu64 " device_ready=%d status=%s\n",
        kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
        layer, chunk, seq, ok ? 1 : 0, ok ? "PASS" : "FAIL");
    return ok;
}

bool ggml_backend_rpc_cycle_finish_reduce(
        ggml_backend_t backend, int32_t layer, int32_t chunk,
        uint64_t seq, uint64_t graph_id) {
    if (!ggml_backend_is_rpc(backend) || layer < 0 || chunk < 0 || seq == 0) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_rpc_context *>(backend->context);
    if (!ctx || rpc_get_remote_patch(ctx->endpoint + "_compute") < 26) {
        return false;
    }
    rpc_msg_inference_cycle_send_req request {};
    request.device = ctx->device;
    request.kind = INFERENCE_CYCLE_PARTIAL;
    request.layer = layer;
    request.chunk = chunk;
    request.seq = seq;
    request.graph_id = graph_id;
    auto sock = get_socket(ctx->endpoint);
    rpc_msg_inference_cycle_send_rsp response {};
    return sock && send_rpc_cmd(
        sock, RPC_CMD_INFERENCE_CYCLE_FINISH,
        &request, sizeof(request), &response, sizeof(response)) &&
        response.result == 1;
}

static bool rpc_cycle_send_remote_tensors(
        ggml_backend_t backend, inference_cycle_payload_kind kind,
        int32_t layer, int32_t chunk, uint64_t seq, uint64_t graph_id,
        const ggml_tensor * const * tensors,
        uint32_t tokens, uint32_t hidden, uint32_t topk) {
    if (!ggml_backend_is_rpc(backend) || layer < 0 || chunk < 0 ||
        seq == 0 || tokens == 0 || hidden == 0 ||
        (kind == INFERENCE_CYCLE_READY && topk == 0) ||
        (kind == INFERENCE_CYCLE_PARTIAL && topk != 0)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_rpc_context *>(backend->context);
    if (!ctx) {
        return false;
    }

    rpc_msg_inference_cycle_send_req request {};
    request.device = ctx->device;
    request.kind = kind;
    request.layer = layer;
    request.chunk = chunk;
    request.seq = seq;
    request.graph_id = graph_id;
    request.tokens = tokens;
    request.hidden = hidden;
    request.topk = topk;
    const size_t count = kind == INFERENCE_CYCLE_READY ? 4 : 1;
    for (size_t i = 0; i < count; ++i) {
        const ggml_tensor * tensor = tensors[i];
        if (!tensor || !tensor->buffer ||
            !ggml_backend_buffer_is_rpc(tensor->buffer) ||
            !ggml_is_contiguous(tensor)) {
            return false;
        }
        auto * buffer_ctx = static_cast<ggml_backend_rpc_buffer_context *>(
            tensor->buffer->context);
        if (!buffer_ctx || buffer_ctx->endpoint != ctx->endpoint ||
            buffer_ctx->device != ctx->device) {
            return false;
        }
        const uint32_t type = i == 2 && kind == INFERENCE_CYCLE_READY ?
            GGML_TYPE_I32 : GGML_TYPE_F32;
        const size_t elements = (kind == INFERENCE_CYCLE_READY && i >= 2) ?
            size_t(tokens) * topk : size_t(tokens) * hidden;
        if (tensor->type != type || ggml_nelements(tensor) != int64_t(elements) ||
            ggml_nbytes(tensor) != elements * sizeof(float)) {
            return false;
        }
        request.tensors[i] = serialize_tensor(tensor);
    }

    // Protocol 25 adds the Phone-owned send command. Never issue it to an
    // older running Termux RPC server.
    auto sock = get_socket(ctx->endpoint);
    if (!sock || rpc_get_remote_patch(ctx->endpoint + "_compute") < 25 ||
        !get_inference_transport(ctx->endpoint)) {
        GGML_LOG_ERROR("[CYCLE_REAL_WIRE] side=pc status=FAIL "
                       "reason=RPC_SERVER_NEEDS_PATCH25_OR_DATA_PLANE\n");
        return false;
    }
    rpc_msg_inference_cycle_send_rsp response {};
    const bool ok = send_rpc_cmd(
        sock, RPC_CMD_INFERENCE_CYCLE_SEND,
        &request, sizeof(request), &response, sizeof(response)) &&
        response.result == 1;
    GGML_LOG_INFO(
        "[CYCLE_REAL_WIRE] side=pc action=ARM_SEND kind=%s "
        "layer=%d chunk=%d seq=%" PRIu64 " status=%s\n",
        kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
        layer, chunk, seq, ok ? "QUEUED_PHONE_TX" : "FAIL");
    return ok;
}

bool ggml_backend_rpc_cycle_send_ready(
        ggml_backend_t backend, int32_t layer, int32_t chunk,
        uint64_t seq, uint64_t graph_id,
        const ggml_tensor * residual, const ggml_tensor * norm,
        const ggml_tensor * ids, const ggml_tensor * routing) {
    if (!residual || !norm || !ids || !routing ||
        residual->ne[0] <= 0 || residual->ne[1] <= 0 ||
        ids->ne[0] <= 0 || ggml_nelements(ids) % residual->ne[1] != 0 ||
        ggml_nelements(ids) / residual->ne[1] > UINT32_MAX ||
        residual->ne[0] > UINT32_MAX || residual->ne[1] > UINT32_MAX) {
        return false;
    }
    const uint32_t tokens = uint32_t(residual->ne[1]);
    const uint32_t hidden = uint32_t(residual->ne[0]);
    const uint32_t topk = uint32_t(ggml_nelements(ids) / tokens);
    const ggml_tensor * sources[4] = {residual, norm, ids, routing};
    return rpc_cycle_send_remote_tensors(
        backend, INFERENCE_CYCLE_READY, layer, chunk, seq, graph_id,
        sources, tokens, hidden, topk);
}

bool ggml_backend_rpc_cycle_send_partial(
        ggml_backend_t backend, int32_t layer, int32_t chunk,
        uint64_t seq, uint64_t graph_id, const ggml_tensor * partial) {
    if (!partial || partial->ne[0] <= 0 || partial->ne[1] <= 0 ||
        partial->ne[0] > UINT32_MAX || partial->ne[1] > UINT32_MAX ||
        ggml_nelements(partial) != partial->ne[0] * partial->ne[1]) {
        return false;
    }
    const ggml_tensor * sources[4] = {partial};
    return rpc_cycle_send_remote_tensors(
        backend, INFERENCE_CYCLE_PARTIAL, layer, chunk, seq, graph_id,
        sources, uint32_t(partial->ne[1]),
        uint32_t(partial->ne[0]), 0);
}

bool ggml_backend_rpc_cycle_send_ack(
        ggml_backend_t backend, int32_t layer, int32_t chunk,
        uint64_t seq, uint64_t graph_id,
        uint32_t kind, bool device_ready) {
    if (!ggml_backend_is_rpc(backend) ||
        layer < 0 || chunk < 0 || seq == 0 ||
        (kind != INFERENCE_CYCLE_READY &&
         kind != INFERENCE_CYCLE_PARTIAL)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_rpc_context *>(backend->context);
    auto transport = ctx ? get_inference_transport(ctx->endpoint) : nullptr;
    if (!transport) {
        return false;
    }
    inference_cycle_message msg {};
    msg.direction = INFERENCE_CYCLE_PHONE_TO_PC;
    msg.kind = static_cast<inference_cycle_payload_kind>(kind);
    msg.layer = layer;
    msg.chunk = chunk;
    msg.seq = seq;
    msg.graph_id = graph_id;
    const bool ok = inference_cycle_send_ack(*transport, msg, device_ready);
    GGML_LOG_INFO(
        "[CYCLE_REAL_WIRE_ACK] side=pc kind=%s layer=%d chunk=%d "
        "seq=%" PRIu64 " device_ready=%d status=%s\n",
        kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
        layer, chunk, seq, device_ready ? 1 : 0,
        ok ? "QUEUED" : "FAIL");
    return ok;
}

bool ggml_backend_rpc_cycle_wait_tx_idle(
        ggml_backend_t backend, int timeout_ms) {
    if (!ggml_backend_is_rpc(backend)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_rpc_context *>(backend->context);
    auto transport = ctx ? get_inference_transport(ctx->endpoint) : nullptr;
    return transport != nullptr && transport->wait_tx_idle(timeout_ms);
}

static constexpr size_t RPC_ALLOC_SIZE_EXACT_CACHE_MAX      = 16384;
static constexpr size_t RPC_ALLOC_SIZE_STRUCTURAL_CACHE_MAX = 4096;
static constexpr uint64_t RPC_ALLOC_SIZE_CACHE_LOG_INTERVAL = 512;

static std::string rpc_alloc_size_key_bytes(const rpc_msg_get_alloc_size_req & request) {
    return std::string(
        reinterpret_cast<const char *>(&request),
        sizeof(request));
}

static bool rpc_alloc_size_structural_cache_eligible(
        const rpc_msg_get_alloc_size_req & request) {
    // Be conservative: structural reuse is enabled only for planning tensors
    // that are not tied to a concrete remote buffer/data address.
    if (request.tensor.buffer != 0 || request.tensor.data != 0) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (request.srcs[i].id == 0) {
            continue;
        }
        if (request.srcs[i].buffer != 0 || request.srcs[i].data != 0) {
            return false;
        }
    }
    return true;
}

static void rpc_alloc_size_normalize_tensor(rpc_tensor & tensor) {
    // These fields describe graph-local identity, not the allocation size.
    tensor.id = 0;
    tensor.buffer = 0;
    tensor.data = 0;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        tensor.src[i] = 0;
    }

    // Preserve whether this is a view, while removing the graph-local pointer.
    tensor.view_src = tensor.view_src != 0 ? 1 : 0;

    // Tensor names do not affect backend allocation requirements.
    memset(tensor.name, 0, sizeof(tensor.name));
    memset(tensor.padding, 0, sizeof(tensor.padding));
}

static std::string rpc_alloc_size_structural_key(
        const rpc_msg_get_alloc_size_req & request) {
    rpc_msg_get_alloc_size_req normalized = request;
    rpc_alloc_size_normalize_tensor(normalized.tensor);
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        rpc_alloc_size_normalize_tensor(normalized.srcs[i]);
    }
    return rpc_alloc_size_key_bytes(normalized);
}

static bool rpc_alloc_size_cache_enabled() {
    const char * value = std::getenv("GGML_RPC_ALLOC_SIZE_CACHE");
    return value == nullptr || strcmp(value, "0") != 0;
}

static void rpc_alloc_size_cache_log_locked(
        const ggml_backend_rpc_buffer_type_context * ctx) {
    const uint64_t hits = ctx->alloc_size_exact_hits + ctx->alloc_size_structural_hits;
    const double hit_rate =
        ctx->alloc_size_lookups > 0 ?
            100.0 * hits / ctx->alloc_size_lookups :
            0.0;
    const double avg_miss_rpc_ms =
        ctx->alloc_size_misses > 0 ?
            (ctx->alloc_size_miss_rpc_us / 1000.0) / ctx->alloc_size_misses :
            0.0;

    GGML_LOG_INFO(
        "[RPC_ALLOC_SIZE_CACHE] device=%u lookups=%" PRIu64
        " exact_hits=%" PRIu64 " structural_hits=%" PRIu64
        " misses=%" PRIu64 " saved_rpc=%" PRIu64
        " hit_rate=%.1f miss_rpc_ms=%.3f avg_miss_rpc_ms=%.3f"
        " exact_entries=%zu structural_entries=%zu"
        " exact_clears=%" PRIu64 " structural_clears=%" PRIu64
        " connection_resets=%" PRIu64 "\n",
        ctx->device,
        ctx->alloc_size_lookups,
        ctx->alloc_size_exact_hits,
        ctx->alloc_size_structural_hits,
        ctx->alloc_size_misses,
        hits,
        hit_rate,
        ctx->alloc_size_miss_rpc_us / 1000.0,
        avg_miss_rpc_ms,
        ctx->alloc_size_exact_cache.size(),
        ctx->alloc_size_structural_cache.size(),
        ctx->alloc_size_exact_clears,
        ctx->alloc_size_structural_clears,
        ctx->alloc_size_connection_resets);
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd(
            ctx->sock,
            RPC_CMD_INIT_TENSOR,
            &request,
            sizeof(request),
            nullptr,
            0);

        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

/*static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    if (size > HASH_THRESHOLD) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        // 发送的信息和接收的信息打印,比如发送接收多少字节
        //GGML_LOG_INFO("ggml_backend_rpc_buffer_set_tensor: sent %zu bytes, received %zu bytes\n", sizeof(request), sizeof(response));
        RPC_STATUS_ASSERT(status);
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    //GGML_LOG_INFO("ggml_backend_rpc_buffer_set_tensor: sent %zu bytes\n", input.size());
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    RPC_STATUS_ASSERT(status);
}
*/
static bool rpc_is_k_quant_type(enum ggml_type type) {
    return type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K;
}

static bool rpc_opencl_extra_debug_enabled() {
    return std::getenv("GGML_RPC_OPENCL_EXTRA_DEBUG") != nullptr;
}

#define RPC_OPENCL_EXTRA_DBG(...) \
    do { \
        if (rpc_opencl_extra_debug_enabled()) { \
            GGML_LOG_DEBUG(__VA_ARGS__); \
        } \
    } while (0)

//用于判断是否需要传权重
static bool should_use_local_file_tensor(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->name[0] == '\0') {
        return false;
    }

    const char * name = tensor->name;

    // 第一阶段只处理模型权重，先不要处理 KV cache / 临时 tensor
    if (strncmp(name, "blk.", 4) == 0) {
        return true;
    }

    if (strcmp(name, "output.weight") == 0) {
        return true;
    }

    if (strcmp(name, "token_embd.weight") == 0) {
        return true;
    }

    return false;
}

static ggml_backend_local_file_result
ggml_backend_rpc_set_tensor_from_local_file_2d(
        ggml_tensor * tensor,
        uint64_t src_offset,
        uint64_t dst_offset,
        uint64_t copy_size,
        uint64_t n_copies,
        uint64_t src_stride,
        uint64_t dst_stride) {

    if (tensor == nullptr ||
        tensor->buffer == nullptr ||
        !ggml_backend_buffer_is_rpc(tensor->buffer)) {
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    // An allocated static tensor is not necessarily a GGUF model weight.
    // The Cycle FFN probe creates up.exps/gate.exps/down.exps in memory on PC.
    // The Phone RPC server has no corresponding local-file source; never
    // dispatch a SET_TENSOR_FROM_LOCAL_FILE request for such a tensor.
    // Preserve local GGUF fast-path loading for actual model weight names.
    if (!should_use_local_file_tensor(tensor)) {
        if (std::getenv("GGML_CYCLE_LIVE_FFN_DIAG") != nullptr) {
            GGML_LOG_INFO(
                "[CYCLE_WEIGHT_UPLOAD] tensor=%s route=HOST_RPC "
                "reason=NOT_GGUF_MODEL_WEIGHT\n", tensor->name);
        }
        return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
    }

    auto * ctx =
        static_cast<ggml_backend_rpc_buffer_context *>(
            tensor->buffer->context);

    rpc_msg_set_tensor_from_local_file_req request {};

    request.tensor      = serialize_tensor(tensor);
    request.src_offset  = src_offset;
    request.dst_offset  = dst_offset;
    request.copy_size   = copy_size;
    request.n_copies    = n_copies;
    request.src_stride  = src_stride;
    request.dst_stride  = dst_stride;

    rpc_msg_set_tensor_from_local_file_rsp response {};

    const bool status = send_rpc_cmd(
        ctx->sock,
        RPC_CMD_SET_TENSOR_FROM_LOCAL_FILE,
        &request,
        sizeof(request),
        &response,
        sizeof(response));

    if (!status) {
        return GGML_BACKEND_LOCAL_FILE_ERROR;
    }

    switch (response.result) {
        case 0:
            return GGML_BACKEND_LOCAL_FILE_NOT_SUPPORTED;
        case 1:
            return GGML_BACKEND_LOCAL_FILE_HANDLED;
        case 2:
            return GGML_BACKEND_LOCAL_FILE_ERROR;
        default:
            return GGML_BACKEND_LOCAL_FILE_ERROR;
    }
}
// 这个有问题
static void ggml_backend_rpc_buffer_set_tensor(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size) {

    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(data != nullptr || size == 0);

    auto * ctx =
        static_cast<ggml_backend_rpc_buffer_context *>(
            buffer->context);

    GGML_ASSERT(ctx != nullptr);

    const rpc_tensor serialized_tensor =
        serialize_tensor(tensor);

    if (rpc_opencl_extra_debug_enabled() &&
        rpc_is_k_quant_type(tensor->type)) {
        GGML_LOG_DEBUG(
            "[RPC_KQUANT_CLIENT_SET_BEGIN] name=%s type=%d "
            "local_buffer=%p remote_buffer=0x%" PRIx64
            " data=0x%" PRIx64 " offset=%zu size=%zu\n",
            tensor->name,
            (int) tensor->type,
            (void *) buffer,
            serialized_tensor.buffer,
            serialized_tensor.data,
            offset,
            size);
    }

    if (ctx->device_ctx != nullptr) {
        auto & fused = ctx->device_ctx->fused_ffn;
        std::lock_guard<std::mutex> lock(fused.mutex);

        if (fused.capture_next_set && fused.expected_tensor == tensor) {
            GGML_ASSERT(!fused.pending);

            fused.capture_next_set = false;
            fused.expected_tensor = nullptr;
            fused.pending = true;
            fused.tensor = tensor;

            rpc_msg_set_tensor_recompute_snapshot_req request {};
            request.input_tensor = serialized_tensor;
            request.input_offset = static_cast<uint64_t>(offset);
            request.input_size = size;

            const uint64_t payload_size = sizeof(request) + size;
            fused.packet.resize(FUSED_RPC_PREFIX + payload_size);
            fused.packet[0] = static_cast<uint8_t>(RPC_CMD_SET_TENSOR_RECOMPUTE_SNAPSHOT);
            memcpy(fused.packet.data() + 1, &payload_size, sizeof(payload_size));
            memcpy(fused.packet.data() + FUSED_REQ_OFFSET, &request, sizeof(request));

            if (size > 0) {
                memcpy(fused.packet.data() + FUSED_DATA_OFFSET, data, size);
            }

            fused.data_size = size;

            return;
        }
    }

    /*
     * 保留原有哈希去重逻辑。
     * 服务端已经有相同数据时，无须再次传输。
     */
    if (size > HASH_THRESHOLD) {
        rpc_msg_set_tensor_hash_req request {};

        request.tensor = serialized_tensor;
        request.offset = static_cast<uint64_t>(offset);
        request.hash   = fnv_hash(
            static_cast<const uint8_t *>(data),
            size);

        rpc_msg_set_tensor_hash_rsp response {};

        const bool status = send_rpc_cmd(
            ctx->sock,
            RPC_CMD_SET_TENSOR_HASH,
            &request,
            sizeof(request),
            &response,
            sizeof(response));

        RPC_STATUS_ASSERT(status);

        if (response.result) {
            return;
        }
    }

    /*
     * 普通网络写入协议：
     *
     * | rpc_tensor | uint64_t offset | data[size] |
     */
    const uint64_t wire_offset =
        static_cast<uint64_t>(offset);

    const size_t input_size =
        sizeof(serialized_tensor) +
        sizeof(wire_offset) +
        size;

    std::vector<uint8_t> input(input_size);

    size_t cursor = 0;

    memcpy(
        input.data() + cursor,
        &serialized_tensor,
        sizeof(serialized_tensor));

    cursor += sizeof(serialized_tensor);

    memcpy(
        input.data() + cursor,
        &wire_offset,
        sizeof(wire_offset));

    cursor += sizeof(wire_offset);

    if (size > 0) {
        memcpy(
            input.data() + cursor,
            data,
            size);
    }

    const bool status = send_rpc_cmd(
        ctx->sock,
        RPC_CMD_SET_TENSOR,
        input.data(),
        input.size());

    RPC_STATUS_ASSERT(status);

    if (rpc_opencl_extra_debug_enabled() &&
        rpc_is_k_quant_type(tensor->type)) {
        GGML_LOG_DEBUG(
            "[RPC_KQUANT_CLIENT_SET_DONE] name=%s type=%d "
            "remote_buffer=0x%" PRIx64 " data=0x%" PRIx64
            " offset=%zu size=%zu\n",
            tensor->name,
            (int) tensor->type,
            serialized_tensor.buffer,
            serialized_tensor.data,
            offset,
            size);
    }
}
static bool ggml_backend_rpc_init_zero_tensor(ggml_tensor * tensor) {
    if (tensor == nullptr ||
        tensor->buffer == nullptr ||
        !ggml_backend_buffer_is_rpc(tensor->buffer)) {
        return false;
    }

    auto * ctx =
        static_cast<ggml_backend_rpc_buffer_context *>(
            tensor->buffer->context);
    if (ctx == nullptr || ctx->sock == nullptr) {
        return false;
    }

    rpc_msg_init_zero_tensor_req request {};
    request.tensor = serialize_tensor(tensor);

    rpc_msg_init_zero_tensor_rsp response {};
    const bool status = send_rpc_cmd(
        ctx->sock,
        RPC_CMD_INIT_ZERO_TENSOR,
        &request,
        sizeof(request),
        &response,
        sizeof(response));

    return status && response.result != 0;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = static_cast<ggml_backend_rpc_buffer_context *>(buffer->context);

    if (rpc_snapshot_read.active) {
        rpc_msg_get_snapshot_req request {};
        request.device = ctx->device;
        request.slot = rpc_snapshot_read.slot;
        request.seq = rpc_snapshot_read.seq;
        request.size = size;

        const size_t lane = rpc_snapshot_read.slot & 1u;
        auto & snapshot_sock = ctx->snapshot_transfer_socks[lane];
        if (snapshot_sock == nullptr) {
            snapshot_sock = get_snapshot_transfer_socket(ctx->endpoint, lane);
            RPC_STATUS_ASSERT(snapshot_sock != nullptr);
        }

        const int64_t t0 = ggml_time_us();

        bool status =
            send_rpc_cmd_compact_small(
                snapshot_sock,
                RPC_CMD_GET_SNAPSHOT,
                request);

        RPC_STATUS_ASSERT(status);

        const int64_t request_done_us = ggml_time_us();

        int64_t ready_first_byte_us = 0;
        int64_t recv_rest_us = 0;

        if (size > 0) {
            const int64_t ready_wait_start_us = ggml_time_us();
            status = snapshot_sock->recv_data(data, 1);
            RPC_STATUS_ASSERT(status);
            const int64_t ready_byte_us = ggml_time_us();
            ready_first_byte_us = ready_byte_us - ready_wait_start_us;

            // The server only starts sending snapshot bytes after the slot
            // transitions to READY. Receiving the first byte is therefore the
            // producer-complete signal. Publish it immediately so Phone
            // compute credit does not wait for the rest of the payload.
            rpc_mark_snapshot_ready(ctx->device_ctx, request.seq);

            if (size > 1) {
                const int64_t recv_rest_start_us = ggml_time_us();
                status = snapshot_sock->recv_data(
                    static_cast<uint8_t *>(data) + 1,
                    size - 1);
                RPC_STATUS_ASSERT(status);
                recv_rest_us = ggml_time_us() - recv_rest_start_us;
            }
        } else {
            rpc_mark_snapshot_ready(ctx->device_ctx, request.seq);
        }

        const int64_t done_us = ggml_time_us();

        if (ctx->device_ctx != nullptr) {
            auto & stats = ctx->device_ctx->snapshot_client_stats;
            stats.transfer_count.fetch_add(1, std::memory_order_relaxed);
            stats.payload_bytes.fetch_add(size, std::memory_order_relaxed);
            stats.request_us.fetch_add(
                request_done_us - t0, std::memory_order_relaxed);
            stats.ready_first_byte_us.fetch_add(
                ready_first_byte_us, std::memory_order_relaxed);
            stats.recv_payload_us.fetch_add(
                recv_rest_us, std::memory_order_relaxed);
            stats.total_us.fetch_add(
                done_us - t0, std::memory_order_relaxed);
        }

        if (RPC_DEBUG || std::getenv("GGML_RETURN_PATH_DEBUG") != nullptr) {
            printf(
                "[RPC_SNAPSHOT_CLIENT] "
                "slot=%u lane=%zu seq=%" PRIu64
                " bytes=%zu request=%.3f ready_first_byte=%.3f recv_rest=%.3f "
                "recv=%.3f total=%.3f ms\n",
                request.slot,
                lane,
                request.seq,
                size,
                (request_done_us - t0) / 1000.0,
                ready_first_byte_us / 1000.0,
                recv_rest_us / 1000.0,
                (done_us - request_done_us) / 1000.0,
                (done_us - t0) / 1000.0);
            fflush(stdout);
        }

        return;
    }

    rpc_msg_get_tensor_req request {};
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    GGML_LOG_INFO("ggml_backend_rpc_buffer_get_tensor: sent %zu bytes\n", sizeof(request));

    bool status;
    if (rpc_stage_ready_callback != nullptr) {

    if (ctx->transfer_sock == nullptr) {
        ctx->transfer_sock =
            get_transfer_socket(ctx->endpoint);

        RPC_STATUS_ASSERT(
            ctx->transfer_sock != nullptr);
    }

    status = send_rpc_cmd_staged(
        ctx->transfer_sock,
        RPC_CMD_GET_TENSOR,
        &request,
        sizeof(request),
        data,
        size);

} else {
    status = send_rpc_cmd(
        ctx->sock,
        RPC_CMD_GET_TENSOR,
        &request,
        sizeof(request),
        data,
        size);
}

    RPC_STATUS_ASSERT(status);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->sock != dst_ctx->sock) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    const int64_t total_begin_us = ggml_time_us();

    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_dev_t dev =
    ggml_backend_buft_get_device(buft);

auto * dev_ctx =
    static_cast<ggml_backend_rpc_device_context *>(
        dev->context);
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response {};

    const int64_t socket_begin_us = ggml_time_us();
    auto sock = get_socket(buft_ctx->endpoint);
    const int64_t socket_us = ggml_time_us() - socket_begin_us;

    const int64_t rpc_begin_us = ggml_time_us();
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    const int64_t rpc_us = ggml_time_us() - rpc_begin_us;
    const int64_t total_us = ggml_time_us() - total_begin_us;

    const bool timing_debug = std::getenv("GGML_ALLOC_TIMING_DEBUG") != nullptr;
    if (timing_debug || total_us >= 10000) {
        GGML_LOG_DEBUG(
            "[RPC_ALLOC_BUFFER] device=%u requested_mib=%.3f returned_mib=%.3f status=%d "
            "total_ms=%.3f socket_ms=%.3f rpc_ms=%.3f\n",
            buft_ctx->device,
            size / 1048576.0,
            status ? response.remote_size / 1048576.0 : 0.0,
            status ? 1 : 0,
            total_us / 1000.0,
            socket_us / 1000.0,
            rpc_us / 1000.0);
    }

    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, buft_ctx->endpoint, buft_ctx->device, nullptr, response.remote_ptr, dev_ctx},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx =
            (ggml_backend_rpc_buffer_type_context *)buft->context;

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        auto sock = get_socket(buft_ctx->endpoint);
        RPC_STATUS_ASSERT(sock != nullptr);

        const bool cache_enabled = rpc_alloc_size_cache_enabled();
        const bool structural_eligible =
            cache_enabled && rpc_alloc_size_structural_cache_eligible(request);
        std::string exact_key;
        std::string structural_key;

        if (cache_enabled) {
            exact_key = rpc_alloc_size_key_bytes(request);
            if (structural_eligible) {
                structural_key = rpc_alloc_size_structural_key(request);
            }

            std::lock_guard<std::mutex> lock(buft_ctx->alloc_size_cache_mutex);
            if (buft_ctx->alloc_size_cache_socket != sock.get()) {
                buft_ctx->alloc_size_exact_cache.clear();
                buft_ctx->alloc_size_structural_cache.clear();
                buft_ctx->alloc_size_cache_socket = sock.get();
                buft_ctx->alloc_size_connection_resets += 1;
            }
            buft_ctx->alloc_size_lookups += 1;

            auto exact_it = buft_ctx->alloc_size_exact_cache.find(exact_key);
            if (exact_it != buft_ctx->alloc_size_exact_cache.end()) {
                buft_ctx->alloc_size_exact_hits += 1;
                if (buft_ctx->alloc_size_lookups % RPC_ALLOC_SIZE_CACHE_LOG_INTERVAL == 0) {
                    rpc_alloc_size_cache_log_locked(buft_ctx);
                }
                return exact_it->second;
            }

            if (structural_eligible) {
                auto structural_it =
                    buft_ctx->alloc_size_structural_cache.find(structural_key);
                if (structural_it != buft_ctx->alloc_size_structural_cache.end()) {
                    buft_ctx->alloc_size_structural_hits += 1;

                    if (buft_ctx->alloc_size_exact_cache.size() >= RPC_ALLOC_SIZE_EXACT_CACHE_MAX) {
                        buft_ctx->alloc_size_exact_cache.clear();
                        buft_ctx->alloc_size_exact_clears += 1;
                    }
                    buft_ctx->alloc_size_exact_cache.emplace(
                        std::move(exact_key), structural_it->second);

                    if (buft_ctx->alloc_size_lookups % RPC_ALLOC_SIZE_CACHE_LOG_INTERVAL == 0) {
                        rpc_alloc_size_cache_log_locked(buft_ctx);
                    }
                    return structural_it->second;
                }
            }
        }

        rpc_msg_get_alloc_size_rsp response {};
        const bool cycle_ffn_diag = std::getenv("GGML_CYCLE_LIVE_FFN_DIAG") != nullptr;
        if (cycle_ffn_diag) {
            GGML_LOG_INFO(
                "[CYCLE_LIVE_FFN_RPC] side=client cmd=GET_ALLOC_SIZE stage=BEGIN "
                "tensor=%s op=%s type=%s ne=[%" PRId64 ",%" PRId64 ",%" PRId64 "] "
                "src0=%s src1=%s src2=%s\n",
                tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
                tensor->ne[0], tensor->ne[1], tensor->ne[2],
                tensor->src[0] ? tensor->src[0]->name : "none",
                tensor->src[1] ? tensor->src[1]->name : "none",
                tensor->src[2] ? tensor->src[2]->name : "none");
        }
        const int64_t rpc_begin_us = ggml_time_us();
        bool status = send_rpc_cmd(
            sock,
            RPC_CMD_GET_ALLOC_SIZE,
            &request,
            sizeof(request),
            &response,
            sizeof(response));
        const int64_t rpc_us = ggml_time_us() - rpc_begin_us;
        if (cycle_ffn_diag) {
            GGML_LOG_INFO(
                "[CYCLE_LIVE_FFN_RPC] side=client cmd=GET_ALLOC_SIZE stage=END "
                "tensor=%s ok=%d bytes=%" PRIu64 " ms=%.3f\n",
                tensor->name, status ? 1 : 0,
                status ? response.alloc_size : uint64_t(0), rpc_us / 1000.0);
        }
        RPC_STATUS_ASSERT(status);

        if (cache_enabled) {
            std::lock_guard<std::mutex> lock(buft_ctx->alloc_size_cache_mutex);
            buft_ctx->alloc_size_misses += 1;
            buft_ctx->alloc_size_miss_rpc_us += rpc_us;

            if (buft_ctx->alloc_size_exact_cache.size() >= RPC_ALLOC_SIZE_EXACT_CACHE_MAX) {
                buft_ctx->alloc_size_exact_cache.clear();
                buft_ctx->alloc_size_exact_clears += 1;
            }
            buft_ctx->alloc_size_exact_cache.emplace(
                std::move(exact_key), response.alloc_size);

            if (structural_eligible) {
                if (buft_ctx->alloc_size_structural_cache.size() >= RPC_ALLOC_SIZE_STRUCTURAL_CACHE_MAX) {
                    buft_ctx->alloc_size_structural_cache.clear();
                    buft_ctx->alloc_size_structural_clears += 1;
                }
                buft_ctx->alloc_size_structural_cache.emplace(
                    std::move(structural_key), response.alloc_size);
            }

            if (buft_ctx->alloc_size_lookups % RPC_ALLOC_SIZE_CACHE_LOG_INTERVAL == 0 ||
                buft_ctx->alloc_size_misses == 1) {
                rpc_alloc_size_cache_log_locked(buft_ctx);
            }
        }

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
}

static void ggml_backend_rpc_fence(ggml_backend_t backend) {
    auto * rpc_ctx = static_cast<ggml_backend_rpc_context *>(backend->context);
    rpc_msg_synchronize_req request {};
    request.device = rpc_ctx->device;

    const bool fence_diag = rpc_xlayer_boundary_diag_enabled();
    uint64_t diag_submit_seq = 0;
    uint64_t diag_graph_uid = 0;
    int diag_nodes = 0;
    std::string diag_cmd;
    std::string diag_first_node;
    std::string diag_marker_node;
    std::string diag_last_node;

    if (fence_diag) {
        std::lock_guard<std::mutex> lock(rpc_ctx->fence_diag_mutex);
        diag_submit_seq = rpc_ctx->fence_diag_submit_seq;
        diag_graph_uid = rpc_ctx->fence_diag_graph_uid;
        diag_nodes = rpc_ctx->fence_diag_nodes;
        diag_cmd = rpc_ctx->fence_diag_cmd;
        diag_first_node = rpc_ctx->fence_diag_first_node;
        diag_marker_node = rpc_ctx->fence_diag_marker_node;
        diag_last_node = rpc_ctx->fence_diag_last_node;

        GGML_LOG_ERROR(
            "[RPC_PHONE_FENCE_DIAG] event=BEGIN endpoint=%s device=%u "
            "last_submit_seq=%" PRIu64 " graph_uid=%" PRIu64 " cmd=%s "
            "nodes=%d first=%s marker=%s last=%s\n",
            rpc_ctx->endpoint.c_str(),
            rpc_ctx->device,
            diag_submit_seq,
            diag_graph_uid,
            diag_cmd.empty() ? "none" : diag_cmd.c_str(),
            diag_nodes,
            diag_first_node.empty() ? "none" : diag_first_node.c_str(),
            diag_marker_node.empty() ? "none" : diag_marker_node.c_str(),
            diag_last_node.empty() ? "none" : diag_last_node.c_str());
    }

    auto sock = get_socket(rpc_ctx->endpoint);
    RPC_STATUS_ASSERT(sock != nullptr);

    const int64_t fence_begin_us = fence_diag ? ggml_time_us() : 0;
    const bool status = send_rpc_cmd(
        sock,
        RPC_CMD_SYNCHRONIZE,
        &request,
        sizeof(request),
        nullptr,
        0);
    const int64_t fence_us =
        fence_diag ? ggml_time_us() - fence_begin_us : 0;

    RPC_STATUS_ASSERT(status);

    if (fence_diag) {
        GGML_LOG_ERROR(
            "[RPC_PHONE_FENCE_DIAG] event=END endpoint=%s device=%u "
            "last_submit_seq=%" PRIu64 " graph_uid=%" PRIu64
            " wait_ms=%.3f\n",
            rpc_ctx->endpoint.c_str(),
            rpc_ctx->device,
            diag_submit_seq,
            diag_graph_uid,
            fence_us / 1000.0);
    }
}

static bool ggml_backend_rpc_route_mark_ready(
        ggml_backend_t backend,
        uint64_t seq) {
    if (backend == nullptr) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend->context);
    if (rpc_ctx == nullptr) {
        return false;
    }

    constexpr uint8_t RPC_ROUTE_SEQ_MIN_PATCH = 8;
    const std::string compute_key = rpc_ctx->endpoint + "_compute";
    if (rpc_get_remote_patch(compute_key) < RPC_ROUTE_SEQ_MIN_PATCH) {
        return false;
    }
    if (seq == 0) {
        // Capability query used by Meta before it rebinds producer storage.
        return true;
    }

    rpc_msg_route_mark_ready_req request {};
    request.device = rpc_ctx->device;
    request.seq = seq;

    auto sock = get_socket(rpc_ctx->endpoint);
    RPC_STATUS_ASSERT(sock != nullptr);
    return send_rpc_cmd_compact_small(
        sock,
        RPC_CMD_ROUTE_MARK_READY,
        request);
}

static constexpr uint64_t RPC_ROUTE_SNAPSHOT_COMPACT_TOPK_FLAG =
    UINT64_C(1) << 63;

// GET_ROUTE_SNAPSHOT patch-17 extension. The client advertises ACK support in
// sizes[2]. The server mirrors the same bit in the response-size header only
// when a sub-lane bulk limit is active.
static constexpr uint64_t RPC_ROUTE_SNAPSHOT_WIRE_ACK_FLAG =
    UINT64_C(1) << 63;

struct rpc_route_topk_compact_layout {
    bool supported = false;
    size_t rows = 0;
    size_t row_bytes = 0;
    size_t stride_bytes = 0;
    size_t wire_bytes = 0;
};

static rpc_route_topk_compact_layout
rpc_route_topk_compact_layout_for_tensor(
        const ggml_tensor * tensor) {
    rpc_route_topk_compact_layout layout {};
    if (tensor == nullptr ||
            tensor->type != GGML_TYPE_I32 ||
            tensor->ne[0] <= 0 ||
            tensor->ne[1] <= 0 ||
            tensor->ne[2] != 1 ||
            tensor->ne[3] != 1 ||
            tensor->nb[0] != sizeof(int32_t)) {
        return layout;
    }

    const size_t cols = static_cast<size_t>(tensor->ne[0]);
    const size_t rows = static_cast<size_t>(tensor->ne[1]);
    if (cols > SIZE_MAX / sizeof(int32_t)) {
        return layout;
    }
    const size_t row_bytes = cols * sizeof(int32_t);
    if (tensor->nb[1] < row_bytes ||
            rows > SIZE_MAX / row_bytes) {
        return layout;
    }

    const size_t wire_bytes = rows * row_bytes;
    const size_t storage_bytes = ggml_nbytes(tensor);
    if (wire_bytes > storage_bytes ||
            (rows > 0 &&
             (rows - 1) > (storage_bytes - row_bytes) / tensor->nb[1])) {
        return layout;
    }

    layout.supported = true;
    layout.rows = rows;
    layout.row_bytes = row_bytes;
    layout.stride_bytes = tensor->nb[1];
    layout.wire_bytes = wire_bytes;
    return layout;
}


static bool ggml_backend_rpc_route_snapshot_ready(
        ggml_backend_t backend,
        uint64_t seq,
        uint32_t lane,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * src2) {
    if (backend == nullptr || lane >= RPC_ROUTE_TRANSFER_LANES) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend->context);
    if (rpc_ctx == nullptr) {
        return false;
    }

    constexpr uint8_t RPC_ROUTE_SNAPSHOT_MIN_PATCH = 12;
    constexpr uint8_t RPC_ROUTE_SNAPSHOT_ASYNC_MIN_PATCH = 14;
    const std::string compute_key = rpc_ctx->endpoint + "_compute";
    const uint8_t remote_patch =
        rpc_get_remote_patch(compute_key);
    if (remote_patch < RPC_ROUTE_SNAPSHOT_MIN_PATCH) {
        return false;
    }

    if (seq == 0) {
        // Capability query used by Meta before enabling producer snapshots.
        return true;
    }

    const ggml_tensor * srcs[3] = { src0, src1, src2 };
    rpc_msg_route_snapshot_ready_req request {};
    request.device = rpc_ctx->device;
    request.lane = lane;
    request.seq = seq;

    for (size_t i = 0; i < 3; ++i) {
        if (srcs[i] == nullptr ||
                srcs[i]->buffer == nullptr ||
                !ggml_backend_buffer_is_rpc(srcs[i]->buffer)) {
            return false;
        }

        auto * buffer_ctx =
            static_cast<ggml_backend_rpc_buffer_context *>(
                srcs[i]->buffer->context);
        if (buffer_ctx == nullptr ||
                buffer_ctx->endpoint != rpc_ctx->endpoint ||
                buffer_ctx->device != rpc_ctx->device) {
            return false;
        }

        request.tensors[i] = serialize_tensor(srcs[i]);
        request.sizes[i] = ggml_nbytes(srcs[i]);
    }

    const bool async_requested =
        std::getenv("GGML_META_PHONE_PREFILL_ASYNC_SNAPSHOT_READY") != nullptr;
    const bool async_enabled =
        async_requested &&
        remote_patch >= RPC_ROUTE_SNAPSHOT_ASYNC_MIN_PATCH;
    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const int64_t client_begin_us =
        stage_profile ? ggml_time_us() : 0;

    const int64_t socket_begin_us =
        stage_profile ? ggml_time_us() : 0;
    auto sock = get_socket(rpc_ctx->endpoint);
    const int64_t socket_done_us =
        stage_profile ? ggml_time_us() : 0;
    RPC_STATUS_ASSERT(sock != nullptr);

    if (async_enabled) {
        const int64_t send_begin_us =
            stage_profile ? ggml_time_us() : 0;
        const bool status =
            send_rpc_cmd_compact_small(
                sock,
                RPC_CMD_ROUTE_SNAPSHOT_READY_ASYNC,
                request);
        const int64_t send_done_us =
            stage_profile ? ggml_time_us() : 0;

        if (stage_profile) {
            GGML_LOG_INFO(
                "[TENSOR_PHONE_RPC_STAGE] side=pc "
                "stage=route_snapshot_ready_client lane=%u "
                "seq=%" PRIu64 " request_bytes=%zu response_bytes=0 "
                "async_requested=1 async_enabled=1 remote_patch=%u "
                "get_socket_ms=%.3f request_send_ms=%.3f "
                "response_header_wait_ms=0.000 "
                "response_body_wait_ms=0.000 "
                "rpc_roundtrip_ms=%.3f total_ms=%.3f "
                "status=%d result=%d\n",
                lane,
                seq,
                sizeof(request),
                static_cast<unsigned>(remote_patch),
                (socket_done_us - socket_begin_us) / 1000.0,
                (send_done_us - send_begin_us) / 1000.0,
                (send_done_us - send_begin_us) / 1000.0,
                (send_done_us - client_begin_us) / 1000.0,
                status ? 1 : 0,
                status ? 1 : 0);
            std::fflush(stderr);
        }

        return status;
    }

    rpc_msg_route_snapshot_ready_rsp response {};
    rpc_cmd_roundtrip_profile roundtrip_profile {};
    rpc_cmd_roundtrip_profile * const previous_profile =
        rpc_cmd_roundtrip_profile_active;
    if (stage_profile) {
        rpc_cmd_roundtrip_profile_active =
            &roundtrip_profile;
    }

    const int64_t rpc_begin_us =
        stage_profile ? ggml_time_us() : 0;
    const bool status = send_rpc_cmd(
        sock,
        RPC_CMD_ROUTE_SNAPSHOT_READY,
        &request,
        sizeof(request),
        &response,
        sizeof(response));
    const int64_t rpc_done_us =
        stage_profile ? ggml_time_us() : 0;

    if (stage_profile) {
        rpc_cmd_roundtrip_profile_active =
            previous_profile;
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=pc "
            "stage=route_snapshot_ready_client lane=%u "
            "seq=%" PRIu64 " request_bytes=%zu response_bytes=%zu "
            "async_requested=%d async_enabled=0 remote_patch=%u "
            "get_socket_ms=%.3f request_send_ms=%.3f "
            "response_header_wait_ms=%.3f "
            "response_body_wait_ms=%.3f "
            "rpc_roundtrip_ms=%.3f total_ms=%.3f "
            "status=%d result=%d\n",
            lane,
            seq,
            sizeof(request),
            sizeof(response),
            async_requested ? 1 : 0,
            static_cast<unsigned>(remote_patch),
            (socket_done_us - socket_begin_us) / 1000.0,
            roundtrip_profile.request_send_us / 1000.0,
            roundtrip_profile.response_header_wait_us / 1000.0,
            roundtrip_profile.response_body_wait_us / 1000.0,
            (rpc_done_us - rpc_begin_us) / 1000.0,
            (rpc_done_us - client_begin_us) / 1000.0,
            status ? 1 : 0,
            status && response.result != 0 ? 1 : 0);
        std::fflush(stderr);
    }

    return status && response.result != 0;
}

static bool ggml_backend_rpc_get_route_snapshot(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        uint64_t seq,
        uint32_t lane,
        ggml_tensor * dst0,
        ggml_tensor * dst1,
        ggml_tensor * dst2) {
    if (backend_src == nullptr ||
            backend_dst == nullptr ||
            seq == 0 ||
            lane >= RPC_ROUTE_TRANSFER_LANES) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend_src->context);
    if (rpc_ctx == nullptr) {
        return false;
    }

    constexpr uint8_t RPC_ROUTE_SNAPSHOT_MIN_PATCH = 12;
    constexpr uint8_t RPC_ROUTE_SNAPSHOT_COMPACT_TOPK_MIN_PATCH = 13;
    constexpr uint8_t RPC_ROUTE_SNAPSHOT_WIRE_ACK_MIN_PATCH = 17;
    constexpr uint8_t RPC_INFERENCE_ROUTE_MIN_PATCH = 23;

    const std::string compute_key = rpc_ctx->endpoint + "_compute";
    const uint8_t remote_patch =
        rpc_get_remote_patch(compute_key);
    if (remote_patch < RPC_ROUTE_SNAPSHOT_MIN_PATCH) {
        return false;
    }

    const char * inference_transport_env =
        std::getenv("GGML_INFERENCE_TRANSPORT");
    const bool inference_force_new =
        inference_transport_env != nullptr &&
        std::strcmp(
            inference_transport_env,
            "force-new") == 0;
    const bool inference_new =
        inference_force_new ||
        (inference_transport_env != nullptr &&
         std::strcmp(
             inference_transport_env,
             "new") == 0);

    ggml_tensor * dsts[3] = { dst0, dst1, dst2 };
    rpc_msg_get_route_snapshot_req request {};
    request.device = rpc_ctx->device;
    request.lane = lane;
    request.seq = seq;

    std::array<size_t, 3> storage_sizes {};
    size_t storage_total_size = 0;
    for (size_t i = 0; i < 3; ++i) {
        if (dsts[i] == nullptr) {
            return false;
        }
        const size_t size = ggml_nbytes(dsts[i]);
        if (size > SIZE_MAX - storage_total_size ||
                size >= RPC_ROUTE_SNAPSHOT_COMPACT_TOPK_FLAG) {
            return false;
        }
        storage_sizes[i] = size;
        request.sizes[i] = size;
        storage_total_size += size;
    }

    const rpc_route_topk_compact_layout topk_layout =
        rpc_route_topk_compact_layout_for_tensor(dst1);
    const bool compact_topk =
        std::getenv("GGML_RPC_DISABLE_ROUTE_TOPK_COMPACT") == nullptr &&
        remote_patch >=
            RPC_ROUTE_SNAPSHOT_COMPACT_TOPK_MIN_PATCH &&
        topk_layout.supported &&
        topk_layout.wire_bytes < storage_sizes[1];

    size_t wire_total_size = storage_total_size;
    size_t topk_wire_bytes = storage_sizes[1];
    if (compact_topk) {
        topk_wire_bytes = topk_layout.wire_bytes;
        wire_total_size =
            storage_total_size -
            storage_sizes[1] +
            topk_wire_bytes;
        request.sizes[1] |=
            RPC_ROUTE_SNAPSHOT_COMPACT_TOPK_FLAG;
    }

    const bool wire_ack_supported =
        remote_patch >= RPC_ROUTE_SNAPSHOT_WIRE_ACK_MIN_PATCH;
    if (wire_ack_supported) {
        request.sizes[2] |=
            RPC_ROUTE_SNAPSHOT_WIRE_ACK_FLAG;
    }

    auto * rpc_dev_ctx =
        static_cast<ggml_backend_rpc_device_context *>(
            ggml_backend_get_device(backend_src)->context);
    if (rpc_dev_ctx == nullptr) {
        return false;
    }

    auto & route_sock =
        rpc_dev_ctx->route_transfer_socks[lane];
    if (route_sock == nullptr) {
        route_sock =
            get_route_transfer_socket(
                rpc_ctx->endpoint,
                lane);
    }
    RPC_STATUS_ASSERT(route_sock != nullptr);

    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const bool route_profile =
        std::getenv("GGML_RPC_ROUTE_STAGE_PROFILE") != nullptr;
    const bool measure_profile =
        stage_profile || route_profile;
    const int64_t client_begin_us =
        ggml_time_us();

    // Destination staging is private per chunk/lane. Keep the generic
    // destination-safety contract explicit for correctness.
    const int64_t dst_sync_begin_us =
        measure_profile ? ggml_time_us() : 0;
    ggml_backend_synchronize(backend_dst);
    const int64_t dst_sync_us =
        measure_profile ? ggml_time_us() - dst_sync_begin_us : 0;

    bool use_inference_route = false;
    std::shared_ptr<inference_transport> inference_transport_ptr;
    if (inference_new &&
            remote_patch >= RPC_INFERENCE_ROUTE_MIN_PATCH) {
        inference_transport_ptr =
            get_inference_transport(rpc_ctx->endpoint);
        use_inference_route =
            inference_transport_ptr != nullptr &&
            inference_transport_ptr->healthy();
    }

    if (inference_new && !use_inference_route) {
        if (inference_force_new) {
            GGML_LOG_ERROR(
                "[INFERENCE_ROUTE] side=client status=FAIL "
                "reason=force_new_unavailable endpoint=%s patch=%u "
                "seq=%" PRIu64 " lane=%u\n",
                rpc_ctx->endpoint.c_str(),
                static_cast<unsigned>(remote_patch),
                seq,
                lane);
            return false;
        }

        GGML_LOG_INFO(
            "[INFERENCE_ROUTE] side=client status=FALLBACK "
            "reason=unavailable endpoint=%s patch=%u "
            "seq=%" PRIu64 " lane=%u\n",
            rpc_ctx->endpoint.c_str(),
            static_cast<unsigned>(remote_patch),
            seq,
            lane);
    }

    std::vector<uint8_t> response;
    int64_t request_send_us = 0;
    int64_t response_header_wait_us = 0;
    int64_t payload_recv_us = 0;
    int64_t ack_send_us = 0;
    int64_t inference_arm_us = 0;

    if (use_inference_route) {
        int route_layer = -1;
        int route_chunk = -1;
        if (dst0 != nullptr && dst0->name[0] != '\0') {
            int parsed = 0;
            if (std::sscanf(
                    dst0->name,
                    "prefill_ffn_norm_chunk_%d-%d%n",
                    &route_chunk,
                    &route_layer,
                    &parsed) != 2 ||
                    dst0->name[parsed] != '\0') {
                route_layer = -1;
                route_chunk = -1;
            }
        }

        rpc_msg_inference_route_arm_req arm {};
        arm.snapshot = request;
        arm.layer = route_layer;
        arm.chunk = route_chunk;

        rpc_msg_inference_route_arm_rsp arm_response {};
        rpc_cmd_roundtrip_profile roundtrip_profile {};
        rpc_cmd_roundtrip_profile * const previous_profile =
            rpc_cmd_roundtrip_profile_active;
        if (measure_profile) {
            rpc_cmd_roundtrip_profile_active =
                &roundtrip_profile;
        }

        const int64_t control_begin_us = ggml_time_us();
        bool status = send_rpc_cmd(
            route_sock,
            RPC_CMD_INFERENCE_ROUTE_ARM,
            &arm,
            sizeof(arm),
            &arm_response,
            sizeof(arm_response));
        const int64_t control_done_us = ggml_time_us();
        inference_arm_us =
            control_done_us - control_begin_us;

        if (measure_profile) {
            rpc_cmd_roundtrip_profile_active =
                previous_profile;
            request_send_us =
                roundtrip_profile.request_send_us;
            response_header_wait_us =
                roundtrip_profile.response_header_wait_us +
                roundtrip_profile.response_body_wait_us;
        }

        if (status) {
            status = arm_response.result != 0;
        }

        if (status && rpc_stage_ready_callback != nullptr) {
            rpc_stage_ready_callback(
                rpc_stage_ready_user_data);
        }

        const int64_t recv_begin_us = ggml_time_us();
        inference_frame_header received_header {};
        if (status) {
            status = rpc_inference_receive_matching(
                rpc_ctx->endpoint,
                INFERENCE_FRAME_DATA,
                INFERENCE_STAGE_ROUTE,
                rpc_ctx->device,
                seq,
                wire_total_size,
                response,
                &received_header);
        }
        const int64_t recv_done_us = ggml_time_us();
        payload_recv_us = recv_done_us - recv_begin_us;

        if (status &&
                route_layer >= 0 &&
                (received_header.layer != route_layer ||
                 received_header.chunk != route_chunk)) {
            GGML_LOG_ERROR(
                "[INFERENCE_ROUTE] side=client status=FAIL "
                "reason=identity_mismatch seq=%" PRIu64
                " expected=%d/%d actual=%d/%d\n",
                seq,
                route_layer,
                route_chunk,
                received_header.layer,
                received_header.chunk);
            status = false;
        }

        if (!status) {
            GGML_LOG_ERROR(
                "[INFERENCE_ROUTE] side=client "
                "seq=%" PRIu64 " lane=%u layer=%d chunk=%d "
                "bytes=%zu arm_ms=%.3f wait_recv_ms=%.3f "
                "status=FAIL\n",
                seq,
                lane,
                route_layer,
                route_chunk,
                wire_total_size,
                inference_arm_us / 1000.0,
                payload_recv_us / 1000.0);
            return false;
        }
    } else {
        response.resize(wire_total_size);
        const int64_t rpc_begin_us = ggml_time_us();

        bool status = send_rpc_cmd(
            route_sock,
            RPC_CMD_GET_ROUTE_SNAPSHOT,
            &request,
            sizeof(request));
        const int64_t request_done_us = ggml_time_us();

        uint64_t wire_out_size = 0;
        if (status) {
            status = route_sock->recv_data(
                &wire_out_size,
                sizeof(wire_out_size));
        }
        const int64_t response_header_us = ggml_time_us();

        const bool wire_ack_required =
            (wire_out_size &
             RPC_ROUTE_SNAPSHOT_WIRE_ACK_FLAG) != 0;
        const uint64_t out_size =
            wire_out_size &
            ~RPC_ROUTE_SNAPSHOT_WIRE_ACK_FLAG;

        if (status && out_size != response.size()) {
            status = false;
        }

        if (status && rpc_stage_ready_callback != nullptr) {
            rpc_stage_ready_callback(
                rpc_stage_ready_user_data);
        }

        if (status && !response.empty()) {
            status = route_sock->recv_data(
                response.data(),
                response.size());
        }
        const int64_t payload_done_us = ggml_time_us();

        // Legacy route credit release is driven by PC userspace receive
        // completion. The new data plane uses its bounded RX/TX rings instead.
        const int64_t ack_send_begin_us =
            measure_profile ? ggml_time_us() : 0;
        if (status && wire_ack_required) {
            const uint64_t ack_seq = seq;
            status = route_sock->send_data(
                &ack_seq,
                sizeof(ack_seq));
        }
        const int64_t ack_send_done_us =
            measure_profile ? ggml_time_us() : 0;

        const int64_t rpc_done_us = ggml_time_us();
        GGML_UNUSED(rpc_done_us);
        RPC_STATUS_ASSERT(status);

        request_send_us =
            request_done_us - rpc_begin_us;
        response_header_wait_us =
            response_header_us - request_done_us;
        payload_recv_us =
            payload_done_us - response_header_us;
        ack_send_us =
            wire_ack_required ?
                ack_send_done_us - ack_send_begin_us :
                0;
    }

    const int64_t dst_set_begin_us =
        measure_profile ? ggml_time_us() : 0;
    size_t offset = 0;

    ggml_backend_tensor_set(
        dst0,
        response.data() + offset,
        0,
        storage_sizes[0]);
    offset += storage_sizes[0];

    if (compact_topk) {
        std::vector<uint8_t> expanded_topk(
            storage_sizes[1],
            uint8_t(0));
        for (size_t row = 0;
                row < topk_layout.rows;
                ++row) {
            std::memcpy(
                expanded_topk.data() +
                    row * topk_layout.stride_bytes,
                response.data() +
                    offset +
                    row * topk_layout.row_bytes,
                topk_layout.row_bytes);
        }
        ggml_backend_tensor_set(
            dst1,
            expanded_topk.data(),
            0,
            expanded_topk.size());
        offset += topk_wire_bytes;
    } else {
        ggml_backend_tensor_set(
            dst1,
            response.data() + offset,
            0,
            storage_sizes[1]);
        offset += storage_sizes[1];
    }

    ggml_backend_tensor_set(
        dst2,
        response.data() + offset,
        0,
        storage_sizes[2]);
    offset += storage_sizes[2];

    GGML_ASSERT(offset == response.size());
    const int64_t dst_set_us =
        measure_profile ?
            ggml_time_us() - dst_set_begin_us :
            0;

    if (use_inference_route) {
        GGML_LOG_INFO(
            "[INFERENCE_ROUTE] side=client "
            "seq=%" PRIu64 " lane=%u bytes=%zu "
            "sync=%.3f arm=%.3f wait_recv=%.3f "
            "set=%.3f total=%.3f status=OK\n",
            seq,
            lane,
            wire_total_size,
            dst_sync_us / 1000.0,
            inference_arm_us / 1000.0,
            payload_recv_us / 1000.0,
            dst_set_us / 1000.0,
            (ggml_time_us() - client_begin_us) / 1000.0);
    } else {
        if (route_profile) {
            std::fprintf(
                stderr,
                "[ROUTE_RPC_RAW] seq=%" PRIu64
                " lane=%u bytes=%zu "
                "sync=%.3f req=%.3f hdr=%.3f recv=%.3f "
                "ack=%.3f set=%.3f total=%.3f\n",
                seq,
                lane,
                wire_total_size,
                dst_sync_us / 1000.0,
                request_send_us / 1000.0,
                response_header_wait_us / 1000.0,
                payload_recv_us / 1000.0,
                ack_send_us / 1000.0,
                dst_set_us / 1000.0,
                (ggml_time_us() - client_begin_us) / 1000.0);
        }

        if (stage_profile) {
            GGML_LOG_INFO(
                "[ROUTE_RPC] seq=%" PRIu64
                " lane=%u bytes=%zu "
                "sync=%.3f req=%.3f hdr=%.3f recv=%.3f "
                "ack=%.3f set=%.3f total=%.3f\n",
                seq,
                lane,
                wire_total_size,
                dst_sync_us / 1000.0,
                request_send_us / 1000.0,
                response_header_wait_us / 1000.0,
                payload_recv_us / 1000.0,
                ack_send_us / 1000.0,
                dst_set_us / 1000.0,
                (ggml_time_us() - client_begin_us) / 1000.0);
        }
    }

    return true;
}

static bool ggml_backend_rpc_get_tensor_batch3(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src0,
        ggml_tensor * dst0,
        const ggml_tensor * src1,
        ggml_tensor * dst1,
        const ggml_tensor * src2,
        ggml_tensor * dst2) {
    if (backend_src == nullptr || backend_dst == nullptr) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend_src->context);
    if (rpc_ctx == nullptr) {
        return false;
    }

    const ggml_tensor * srcs[3] = { src0, src1, src2 };
    ggml_tensor * dsts[3] = { dst0, dst1, dst2 };
    rpc_msg_get_tensor_batch3_req request {};
    size_t sizes[3] = {};
    size_t total_size = 0;

    for (size_t i = 0; i < 3; ++i) {
        if (srcs[i] == nullptr || dsts[i] == nullptr ||
                srcs[i]->buffer == nullptr ||
                !ggml_backend_buffer_is_rpc(srcs[i]->buffer) ||
                !ggml_are_same_layout(srcs[i], dsts[i])) {
            return false;
        }

        auto * buffer_ctx =
            static_cast<ggml_backend_rpc_buffer_context *>(
                srcs[i]->buffer->context);
        if (buffer_ctx == nullptr ||
                buffer_ctx->endpoint != rpc_ctx->endpoint ||
                buffer_ctx->device != rpc_ctx->device) {
            return false;
        }

        sizes[i] = ggml_nbytes(srcs[i]);
        if (sizes[i] > SIZE_MAX - total_size) {
            return false;
        }

        request.items[i].tensor = serialize_tensor(srcs[i]);
        request.items[i].offset = 0;
        request.items[i].size = sizes[i];
        total_size += sizes[i];
    }

    const bool staged_route =
        rpc_stage_ready_callback != nullptr &&
        rpc_route_transfer_lane >= 0;
    std::shared_ptr<socket_t> sock;
    if (staged_route) {
        auto * rpc_dev_ctx =
            static_cast<ggml_backend_rpc_device_context *>(
                ggml_backend_get_device(backend_src)->context);
        RPC_STATUS_ASSERT(rpc_dev_ctx != nullptr);
        const size_t route_lane =
            static_cast<size_t>(rpc_route_transfer_lane);
        auto & route_sock =
            rpc_dev_ctx->route_transfer_socks[route_lane];
        if (route_sock == nullptr) {
            route_sock =
                get_route_transfer_socket(
                    rpc_ctx->endpoint,
                    route_lane);
        }
        sock = route_sock;
    } else {
        sock = get_socket(rpc_ctx->endpoint);
    }
    RPC_STATUS_ASSERT(sock != nullptr);

    // Batch3 was added in RPC protocol patch 5.  Older servers intentionally
    // remain connectable at the same major/minor version, so never send the new
    // command unless the HELLO response explicitly proves support.  Returning
    // false lets Meta fall back to the already-correct three-GET path.
    constexpr uint8_t RPC_BATCH3_MIN_PATCH = 5;
    constexpr uint8_t RPC_ROUTE_SEQ_MIN_PATCH = 8;
    const std::string compute_key = rpc_ctx->endpoint + "_compute";
    const uint8_t remote_patch = rpc_get_remote_patch(compute_key);
    if (remote_patch < RPC_BATCH3_MIN_PATCH ||
            (rpc_route_wait_seq != 0 &&
             remote_patch < RPC_ROUTE_SEQ_MIN_PATCH)) {
        LOG_DBG(
            "[RPC_GET_BATCH3_FALLBACK] endpoint=%s remote_patch=%u required=%u\n",
            rpc_ctx->endpoint.c_str(),
            (unsigned) remote_patch,
            (unsigned) RPC_BATCH3_MIN_PATCH);
        return false;
    }

    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const bool route_profile =
        std::getenv("GGML_RPC_ROUTE_STAGE_PROFILE") != nullptr;
    const bool measure_profile =
        stage_profile || route_profile;
    const int64_t client_begin_us =
        measure_profile ? ggml_time_us() : 0;

    // Match the generic copy fallback's destination-safety contract, but do it
    // once for the whole Router packet instead of once per tensor.
    const int64_t dst_sync_begin_us =
        measure_profile ? ggml_time_us() : 0;
    if (!staged_route) {
        ggml_backend_synchronize(backend_dst);
    }
    const int64_t dst_sync_us =
        measure_profile ? ggml_time_us() - dst_sync_begin_us : 0;

    std::vector<uint8_t> response(total_size);

    rpc_cmd_roundtrip_profile route_roundtrip {};
    rpc_cmd_roundtrip_profile * const previous_roundtrip_profile =
        rpc_cmd_roundtrip_profile_active;
    if (route_profile && !staged_route) {
        rpc_cmd_roundtrip_profile_active =
            &route_roundtrip;
    }

    const int64_t rpc_begin_us = ggml_time_us();
    bool status = false;
    if (rpc_route_wait_seq != 0) {
        rpc_msg_get_tensor_batch3_wait_req wait_request {};
        wait_request.device = rpc_ctx->device;
        wait_request.wait_seq = rpc_route_wait_seq;
        wait_request.batch = request;
        status = staged_route ?
            send_rpc_cmd_staged(
                sock,
                RPC_CMD_GET_TENSOR_BATCH3_WAIT,
                &wait_request,
                sizeof(wait_request),
                response.data(),
                response.size(),
                route_profile ? &route_roundtrip : nullptr) :
            send_rpc_cmd(
                sock,
                RPC_CMD_GET_TENSOR_BATCH3_WAIT,
                &wait_request,
                sizeof(wait_request),
                response.data(),
                response.size());
    } else {
        status = staged_route ?
            send_rpc_cmd_staged(
                sock,
                RPC_CMD_GET_TENSOR_BATCH3,
                &request,
                sizeof(request),
                response.data(),
                response.size(),
                route_profile ? &route_roundtrip : nullptr) :
            send_rpc_cmd(
                sock,
                RPC_CMD_GET_TENSOR_BATCH3,
                &request,
                sizeof(request),
                response.data(),
                response.size());
    }
    const int64_t rpc_us = ggml_time_us() - rpc_begin_us;

    if (route_profile && !staged_route) {
        rpc_cmd_roundtrip_profile_active =
            previous_roundtrip_profile;
    }

    RPC_STATUS_ASSERT(status);

    const int64_t dst_set_begin_us =
        measure_profile ? ggml_time_us() : 0;
    size_t response_offset = 0;
    for (size_t i = 0; i < 3; ++i) {
        ggml_backend_tensor_set(
            dsts[i],
            response.data() + response_offset,
            0,
            sizes[i]);
        response_offset += sizes[i];
    }
    GGML_ASSERT(response_offset == response.size());
    const int64_t dst_set_us =
        measure_profile ? ggml_time_us() - dst_set_begin_us : 0;

    if (RPC_DEBUG) {
        GGML_LOG_INFO(
            "[RPC_GET_BATCH3_CLIENT] bytes=%zu total=%.3f ms\n",
            total_size,
            rpc_us / 1000.0);
    }
    if (route_profile) {
        std::fprintf(
            stderr,
            "[RPC_ROUTE_STAGE] side=pc stage=router_batch "
            "bytes=%zu staged=%d lane=%d wait_seq=%" PRIu64 " "
            "dst_sync_ms=%.3f request_send_ms=%.3f "
            "response_header_wait_ms=%.3f payload_recv_ms=%.3f "
            "rpc_roundtrip_ms=%.3f dst_set_ms=%.3f total_ms=%.3f\n",
            total_size,
            staged_route ? 1 : 0,
            staged_route ? rpc_route_transfer_lane : -1,
            rpc_route_wait_seq,
            dst_sync_us / 1000.0,
            route_roundtrip.request_send_us / 1000.0,
            route_roundtrip.response_header_wait_us / 1000.0,
            route_roundtrip.response_body_wait_us / 1000.0,
            rpc_us / 1000.0,
            dst_set_us / 1000.0,
            (ggml_time_us() - client_begin_us) / 1000.0);
        std::fflush(stderr);
    }

    if (stage_profile) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=pc stage=router_batch "
            "bytes=%zu staged=%d lane=%d dst_sync_ms=%.3f "
            "rpc_roundtrip_ms=%.3f dst_set_ms=%.3f total_ms=%.3f\n",
            total_size,
            staged_route ? 1 : 0,
            staged_route ? rpc_route_transfer_lane : -1,
            dst_sync_us / 1000.0,
            rpc_us / 1000.0,
            dst_set_us / 1000.0,
            (ggml_time_us() - client_begin_us) / 1000.0);
    }

    return true;
}

static bool ggml_backend_rpc_snapshot_arm(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        size_t offset,
        size_t size,
        uint32_t slot,
        uint64_t seq) {
    auto * rpc_ctx = static_cast<ggml_backend_rpc_context *>(backend->context);

    {
        auto & graph_snapshot = rpc_ctx->graph_snapshot;
        std::lock_guard<std::mutex> lock(graph_snapshot.mutex);

        if (graph_snapshot.snapshot_was_fused &&
            graph_snapshot.fused_snapshot_seq == seq) {
            graph_snapshot.snapshot_was_fused = false;
            return true;
        }
    }

    auto sock = get_socket(rpc_ctx->endpoint);
    RPC_STATUS_ASSERT(sock != nullptr);

    rpc_msg_snapshot_tensor_req request {};
    request.device = rpc_ctx->device;
    request.slot = slot;
    request.seq = seq;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;

    return send_rpc_cmd_compact_small(
        sock,
        RPC_CMD_SNAPSHOT_TENSOR,
        request);
}

static bool ggml_backend_rpc_prepare_graph_snapshot(
        ggml_backend_t backend,
        const ggml_tensor * tensor,
        size_t offset,
        size_t size,
        uint32_t slot,
        uint64_t seq) {
    auto * rpc_ctx = static_cast<ggml_backend_rpc_context *>(backend->context);
    auto & graph_snapshot = rpc_ctx->graph_snapshot;
    std::lock_guard<std::mutex> lock(graph_snapshot.mutex);

    GGML_ASSERT(!graph_snapshot.active);

    graph_snapshot.active = true;
    graph_snapshot.tensor = tensor;
    graph_snapshot.offset = offset;
    graph_snapshot.size = size;
    graph_snapshot.slot = slot;
    graph_snapshot.seq = seq;

    return true;
}

static bool rpc_is_prefill_external_input(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return false;
    }

    int chunk = -1;
    int layer = -1;
    int n = 0;
    if (std::sscanf(
            tensor->name,
            "prefill_ffn_norm_chunk_%d-%d%n",
            &chunk,
            &layer,
            &n) != 2) {
        return false;
    }
    return tensor->name[n] == '\0';
}

static void add_tensor(
        ggml_tensor * tensor,
        std::vector<rpc_tensor> & tensors,
        std::unordered_set<ggml_tensor *> & visited,
        const std::unordered_set<ggml_tensor *> & graph_nodes) {
    if (tensor == nullptr) {
        return;
    }
    if (!visited.insert(tensor).second) {
        return;
    }

    if (rpc_is_prefill_external_input(tensor)) {
        GGML_ASSERT(tensor->buffer != nullptr && ggml_backend_buffer_is_rpc(tensor->buffer));

        rpc_tensor result = serialize_tensor(tensor);
        result.op = GGML_OP_NONE;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            result.src[i] = 0;
        }
        result.view_src = 0;
        result.view_offs = 0;
        result.flags &= ~GGML_TENSOR_FLAG_COMPUTE;
        tensors.push_back(result);

        if (RPC_DEBUG) {
            printf(
                "[RPC_GRAPH_CUT_INPUT] name=%s ne=[%u,%u,%u,%u]\n",
                result.name, result.ne[0], result.ne[1], result.ne[2], result.ne[3]);
        }
        return;
    }

    const bool external_input =
        graph_nodes.find(tensor) == graph_nodes.end() &&
        tensor->buffer != nullptr &&
        ggml_backend_buffer_is_rpc(tensor->buffer);
    if (external_input) {
        if (tensor->view_src != nullptr) {
            add_tensor(tensor->view_src, tensors, visited, graph_nodes);
        }

        rpc_tensor result = serialize_tensor(tensor);
        result.op = GGML_OP_NONE;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            result.src[i] = 0;
        }
        result.flags &= ~GGML_TENSOR_FLAG_COMPUTE;

        if (rpc_opencl_extra_debug_enabled() &&
            rpc_is_k_quant_type(tensor->type)) {
            GGML_LOG_DEBUG(
                "[RPC_KQUANT_GRAPH_EXTERNAL] name=%s ptr=%p type=%d "
                "local_buffer=%p remote_buffer=0x%" PRIx64
                " data=0x%" PRIx64 " bytes=%zu view_src=%p\n",
                tensor->name,
                (void *) tensor,
                (int) tensor->type,
                (void *) tensor->buffer,
                result.buffer,
                result.data,
                ggml_nbytes(tensor),
                (void *) tensor->view_src);
        }

        tensors.push_back(result);
        return;
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited, graph_nodes);
    }
    add_tensor(tensor->view_src, tensors, visited, graph_nodes);
    tensors.push_back(serialize_tensor(tensor));
}
// 将ggml_cgraph序列化为字节流，便于在RPC中传输
static uint64_t rpc_hash_u64(uint64_t h, uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

static uint64_t rpc_graph_effective_uid(const ggml_cgraph * cgraph) {
    if (cgraph->uid != 0) {
        return cgraph->uid;
    }

    uint64_t uid = 0x7bf5d03899c4e5b9ULL;
    uid = rpc_hash_u64(uid, (uint64_t) cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        uid = rpc_hash_u64(uid, (uint64_t) (uintptr_t) node);
        uid = rpc_hash_u64(uid, (uint64_t) node->op);
        uid = rpc_hash_u64(uid, (uint64_t) node->type);
        uid = rpc_hash_u64(uid, (uint64_t) (uintptr_t) node->view_src);
        for (int j = 0; j < GGML_MAX_DIMS; ++j) {
            uid = rpc_hash_u64(uid, (uint64_t) node->ne[j]);
            uid = rpc_hash_u64(uid, (uint64_t) node->nb[j]);
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            uid = rpc_hash_u64(uid, (uint64_t) (uintptr_t) node->src[j]);
        }
    }
    return uid != 0 ? uid : 0x7bf5d03899c4e5b9ULL;
}

static void serialize_graph(uint32_t device, uint64_t graph_uid, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    std::unordered_set<ggml_tensor *> graph_nodes;
    graph_nodes.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; ++i) {
        graph_nodes.insert(cgraph->nodes[i]);
    }
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited, graph_nodes);
    }
    // serialization format:
    // | device (4 bytes) | graph_uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &graph_uid, sizeof(graph_uid));
    dest += sizeof(graph_uid);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));

    if (RPC_DEBUG) {
        const char * first = n_nodes > 0 ? cgraph->nodes[0]->name : "";
        const char * last = n_nodes > 0 ? cgraph->nodes[n_nodes - 1]->name : "";
        printf(
            "[RPC_GRAPH_SERIALIZE] uid=%" PRIu64
            " nodes=%u tensors=%zu bytes=%zu first=%s last=%s\n",
            graph_uid, n_nodes, tensors.size(), output.size(), first, last);
    }
}

static bool ggml_backend_rpc_set_tensor_graph(
        ggml_backend_t backend_src,
        ggml_backend_t backend_dst,
        const ggml_tensor * src,
        ggml_tensor * dst,
        const ggml_cgraph * graph) {
    if (backend_src == nullptr || backend_dst == nullptr ||
            src == nullptr || dst == nullptr || graph == nullptr ||
            graph->n_nodes <= 0 || src->buffer == nullptr ||
            dst->buffer == nullptr ||
            !ggml_backend_buffer_is_rpc(dst->buffer) ||
            !ggml_are_same_layout(src, dst)) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend_dst->context);
    auto * dst_buffer_ctx =
        static_cast<ggml_backend_rpc_buffer_context *>(dst->buffer->context);
    auto * rpc_dev_ctx =
        static_cast<ggml_backend_rpc_device_context *>(
            ggml_backend_get_device(backend_dst)->context);
    if (rpc_ctx == nullptr || dst_buffer_ctx == nullptr ||
            rpc_dev_ctx == nullptr ||
            dst_buffer_ctx->endpoint != rpc_ctx->endpoint ||
            dst_buffer_ctx->device != rpc_ctx->device) {
        return false;
    }

    auto sock = get_socket(rpc_ctx->endpoint);
    RPC_STATUS_ASSERT(sock != nullptr);

    constexpr uint8_t RPC_SET_TENSOR_GRAPH_MIN_PATCH = 6;
    constexpr uint8_t RPC_SET_TENSOR_GRAPH_CACHE_MIN_PATCH = 7;
    const std::string compute_key = rpc_ctx->endpoint + "_compute";
    const uint8_t remote_patch = rpc_get_remote_patch(compute_key);
    if (remote_patch < RPC_SET_TENSOR_GRAPH_MIN_PATCH) {
        LOG_DBG(
            "[RPC_SET_GRAPH_FALLBACK] endpoint=%s remote_patch=%u required=%u\n",
            rpc_ctx->endpoint.c_str(),
            (unsigned) remote_patch,
            (unsigned) RPC_SET_TENSOR_GRAPH_MIN_PATCH);
        return false;
    }

    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const int64_t client_begin_us =
        stage_profile ? ggml_time_us() : 0;

    // Preserve producer completion before materializing the PC partial on host.
    const int64_t src_sync_begin_us =
        stage_profile ? ggml_time_us() : 0;
    ggml_backend_synchronize(backend_src);
    const int64_t src_sync_us =
        stage_profile ? ggml_time_us() - src_sync_begin_us : 0;

    const int64_t dst_sync_begin_us =
        stage_profile ? ggml_time_us() : 0;
    ggml_backend_synchronize(backend_dst);
    const int64_t dst_sync_us =
        stage_profile ? ggml_time_us() - dst_sync_begin_us : 0;

    const size_t data_size = ggml_nbytes(src);
    std::vector<uint8_t> data(data_size);
    const int64_t src_get_begin_us =
        stage_profile ? ggml_time_us() : 0;
    ggml_backend_tensor_get(src, data.data(), 0, data_size);
    const int64_t src_get_us =
        stage_profile ? ggml_time_us() - src_get_begin_us : 0;

    const uint64_t graph_uid = rpc_graph_effective_uid(graph);
    const bool cache_supported =
        remote_patch >= RPC_SET_TENSOR_GRAPH_CACHE_MIN_PATCH &&
        std::getenv("GGML_RPC_DISABLE_FUSED_ADD_GRAPH_CACHE") == nullptr;

    bool cache_hit = false;
    std::unique_lock<std::mutex> cache_lock(
        rpc_dev_ctx->fused_set_graph_cache_mutex,
        std::defer_lock);
    if (cache_supported) {
        cache_lock.lock();
        if (rpc_dev_ctx->fused_set_graph_cache_socket != sock.get()) {
            rpc_dev_ctx->fused_set_graph_uids.clear();
            rpc_dev_ctx->fused_set_graph_cache_socket = sock.get();
        }
        cache_hit =
            rpc_dev_ctx->fused_set_graph_uids.find(graph_uid) !=
            rpc_dev_ctx->fused_set_graph_uids.end();
    }

    std::vector<uint8_t> graph_data;
    const int64_t serialize_begin_us =
        stage_profile ? ggml_time_us() : 0;
    if (!cache_hit) {
        serialize_graph(
            rpc_ctx->device,
            graph_uid,
            graph,
            graph_data);
    }
    const int64_t serialize_us =
        stage_profile ? ggml_time_us() - serialize_begin_us : 0;

    const int64_t pack_begin_us =
        stage_profile ? ggml_time_us() : 0;
    std::vector<uint8_t> input;
    if (cache_supported) {
        rpc_msg_set_tensor_graph_req request {};
        request.tensor = serialize_tensor(dst);
        request.offset = 0;
        request.size = data_size;
        request.graph_size = graph_data.size();
        request.graph_uid = graph_uid;
        request.device = rpc_ctx->device;

        if (data_size > SIZE_MAX - sizeof(request) ||
                graph_data.size() > SIZE_MAX - sizeof(request) - data_size) {
            return false;
        }

        input.resize(sizeof(request) + data_size + graph_data.size());
        size_t cursor = 0;
        memcpy(input.data() + cursor, &request, sizeof(request));
        cursor += sizeof(request);
        if (data_size > 0) {
            memcpy(input.data() + cursor, data.data(), data_size);
            cursor += data_size;
        }
        if (!graph_data.empty()) {
            memcpy(input.data() + cursor, graph_data.data(), graph_data.size());
            cursor += graph_data.size();
        }
        GGML_ASSERT(cursor == input.size());
    } else {
        // Patch-6 server compatibility: preserve the original header exactly.
        rpc_msg_set_tensor_graph_req_v6 request {};
        request.tensor = serialize_tensor(dst);
        request.offset = 0;
        request.size = data_size;
        request.graph_size = graph_data.size();

        if (data_size > SIZE_MAX - sizeof(request) ||
                graph_data.size() > SIZE_MAX - sizeof(request) - data_size) {
            return false;
        }

        input.resize(sizeof(request) + data_size + graph_data.size());
        size_t cursor = 0;
        memcpy(input.data() + cursor, &request, sizeof(request));
        cursor += sizeof(request);
        if (data_size > 0) {
            memcpy(input.data() + cursor, data.data(), data_size);
            cursor += data_size;
        }
        if (!graph_data.empty()) {
            memcpy(input.data() + cursor, graph_data.data(), graph_data.size());
            cursor += graph_data.size();
        }
        GGML_ASSERT(cursor == input.size());
    }

    const int64_t pack_us =
        stage_profile ? ggml_time_us() - pack_begin_us : 0;
    const int64_t send_begin_us = ggml_time_us();
    const bool status = send_rpc_cmd(
        sock,
        RPC_CMD_SET_TENSOR_GRAPH_COMPUTE,
        input.data(),
        input.size());
    const int64_t send_us = ggml_time_us() - send_begin_us;

    if (status && cache_supported && !cache_hit) {
        // Same control socket is strictly ordered: once the full graph command
        // has been sent, a later cached command cannot overtake its server-side
        // reconstruction.  Socket replacement clears this cache above.
        rpc_dev_ctx->fused_set_graph_uids.insert(graph_uid);
    }

    if (RPC_DEBUG) {
        GGML_LOG_INFO(
            "[RPC_SET_GRAPH_CLIENT] tensor=%s bytes=%zu graph_uid=%" PRIu64
            " cache=%d graph_bytes=%zu total=%.3f ms\n",
            dst->name,
            data_size,
            graph_uid,
            cache_hit ? 1 : 0,
            graph_data.size(),
            send_us / 1000.0);
    }
    if (stage_profile) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=pc stage=return_set_add "
            "bytes=%zu graph_uid=%" PRIu64 " cache=%d graph_bytes=%zu "
            "src_sync_ms=%.3f dst_sync_ms=%.3f src_get_ms=%.3f "
            "serialize_ms=%.3f pack_ms=%.3f send_ms=%.3f total_ms=%.3f\n",
            data_size,
            graph_uid,
            cache_hit ? 1 : 0,
            graph_data.size(),
            src_sync_us / 1000.0,
            dst_sync_us / 1000.0,
            src_get_us / 1000.0,
            serialize_us / 1000.0,
            pack_us / 1000.0,
            send_us / 1000.0,
            (ggml_time_us() - client_begin_us) / 1000.0);
        std::fflush(stderr);
    }

    return status;
}

static bool ggml_backend_rpc_phone_ffn_mark_ready(
        ggml_backend_t backend,
        uint64_t seq) {
    if (backend == nullptr) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend->context);
    if (rpc_ctx == nullptr) {
        return false;
    }

    constexpr uint8_t RPC_PHONE_FFN_JOIN_MIN_PATCH = 11;
    constexpr uint8_t RPC_PHONE_FFN_CPU_REDUCE_MIN_PATCH = 16;
    const std::string compute_key = rpc_ctx->endpoint + "_compute";
    const uint8_t remote_patch =
        rpc_get_remote_patch(compute_key);
    if (remote_patch < RPC_PHONE_FFN_JOIN_MIN_PATCH) {
        return false;
    }

    // ARM alone is a side-effect-free capability query used by Meta when
    // deciding whether Chunk Ring CPU join may be activated.
    if (seq == RPC_PHONE_FFN_CPU_REDUCE_ARM) {
        return remote_patch >=
            RPC_PHONE_FFN_CPU_REDUCE_MIN_PATCH;
    }

    if ((seq & RPC_PHONE_FFN_CPU_REDUCE_ARM) != 0 &&
            remote_patch <
                RPC_PHONE_FFN_CPU_REDUCE_MIN_PATCH) {
        return false;
    }

    if (seq == 0) {
        return true;
    }

    rpc_msg_phone_ffn_mark_ready_req request {};
    request.device = rpc_ctx->device;
    request.seq = seq;

    auto sock = get_socket(rpc_ctx->endpoint);
    RPC_STATUS_ASSERT(sock != nullptr);

    // This command intentionally has no response. It is sent on the compute
    // socket immediately after the Phone FFN graph command. The server cannot
    // process this marker until the preceding graph_compute has completed on
    // that socket, making seq a per-chunk Phone-FFN completion dependency.
    return send_rpc_cmd_compact_small(
        sock,
        RPC_CMD_PHONE_FFN_MARK_READY,
        request);
}

static thread_local bool rpc_inference_return_enqueue_only = false;

static bool ggml_backend_rpc_set_tensor_async_return_wait(
        ggml_backend_t backend_dst,
        ggml_tensor * dst,
        const void * data,
        size_t data_size,
        uint64_t phone_ffn_seq,
        size_t return_lane) {
    if (backend_dst == nullptr || !ggml_backend_is_rpc(backend_dst)) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend_dst->context);
    auto * rpc_dev_ctx =
        static_cast<ggml_backend_rpc_device_context *>(
            ggml_backend_get_device(backend_dst)->context);
    if (rpc_ctx == nullptr || rpc_dev_ctx == nullptr) {
        return false;
    }

    constexpr uint8_t RPC_PHONE_FFN_JOIN_MIN_PATCH = 11;
    constexpr uint8_t RPC_RETURN_F16_MIN_PATCH = 15;
    constexpr uint8_t RPC_DIRECTIONAL_BULK_MIN_PATCH = 19;
    const std::string compute_key = rpc_ctx->endpoint + "_compute";
    const uint8_t remote_patch = rpc_get_remote_patch(compute_key);
    if (remote_patch < RPC_PHONE_FFN_JOIN_MIN_PATCH) {
        return false;
    }

    const char * inference_transport_env =
        std::getenv("GGML_INFERENCE_TRANSPORT");
    const bool inference_force_new =
        inference_transport_env != nullptr &&
        std::strcmp(
            inference_transport_env,
            "force-new") == 0;
    const bool inference_new =
        inference_force_new ||
        (inference_transport_env != nullptr &&
         std::strcmp(
             inference_transport_env,
             "new") == 0);
    constexpr uint8_t RPC_INFERENCE_RETURN_MIN_PATCH = 24;

    if (dst == nullptr && data == nullptr &&
            data_size == 0 && phone_ffn_seq == 0) {
        if (return_lane >= RPC_RETURN_TRANSFER_LANES) {
            return false;
        }
        if (!inference_new) {
            return true;
        }

        const bool available =
            remote_patch >= RPC_INFERENCE_RETURN_MIN_PATCH &&
            get_inference_transport(rpc_ctx->endpoint) != nullptr;
        if (!available && !inference_force_new) {
            GGML_LOG_INFO(
                "[INFERENCE_RETURN] side=client status=FALLBACK "
                "reason=capability endpoint=%s patch=%u\n",
                rpc_ctx->endpoint.c_str(),
                static_cast<unsigned>(remote_patch));
            return true;
        }
        return available;
    }

    if (dst == nullptr || data == nullptr ||
            phone_ffn_seq == 0 ||
            dst->buffer == nullptr ||
            !ggml_backend_buffer_is_rpc(dst->buffer) ||
            data_size != ggml_nbytes(dst)) {
        return false;
    }

    auto * dst_buffer_ctx =
        static_cast<ggml_backend_rpc_buffer_context *>(
            dst->buffer->context);
    if (dst_buffer_ctx == nullptr ||
            dst_buffer_ctx->endpoint != rpc_ctx->endpoint ||
            dst_buffer_ctx->device != rpc_ctx->device) {
        return false;
    }

    if (return_lane >= RPC_RETURN_TRANSFER_LANES) {
        return false;
    }

    auto & return_sock =
        rpc_dev_ctx->return_transfer_socks[return_lane];
    if (return_sock == nullptr) {
        return_sock =
            get_return_transfer_socket(
                rpc_ctx->endpoint,
                return_lane);
        rpc_dev_ctx->network_bulk_shared_active.store(
            -1,
            std::memory_order_relaxed);
    }
    auto sock = return_sock;
    RPC_STATUS_ASSERT(sock != nullptr);

    const char * f16_env =
        std::getenv("GGML_RPC_RETURN_F16");
    const bool f16_requested =
        f16_env != nullptr && std::atoi(f16_env) != 0;
    const bool use_extended =
        remote_patch >= RPC_RETURN_F16_MIN_PATCH;
    const bool use_f16 =
        f16_requested &&
        use_extended &&
        dst->type == GGML_TYPE_F32 &&
        data_size % sizeof(float) == 0;

    std::vector<ggml_fp16_t> f16_payload;
    const void * wire_data = data;
    size_t wire_size = data_size;
    if (use_f16) {
        const size_t n_elements =
            data_size / sizeof(float);
        f16_payload.resize(n_elements);
        ggml_fp32_to_fp16_row(
            static_cast<const float *>(data),
            f16_payload.data(),
            static_cast<int64_t>(n_elements));
        wire_data = f16_payload.data();
        wire_size =
            n_elements * sizeof(ggml_fp16_t);
    }

    if (inference_new) {
        bool new_available =
            remote_patch >= RPC_INFERENCE_RETURN_MIN_PATCH;
        std::shared_ptr<inference_transport> transport;
        if (new_available) {
            transport =
                get_inference_transport(rpc_ctx->endpoint);
            new_available = transport != nullptr;
        }

        if (!new_available) {
            if (inference_force_new) {
                GGML_LOG_ERROR(
                    "[INFERENCE_RETURN] side=client status=FAIL "
                    "reason=force_new_unavailable endpoint=%s patch=%u\n",
                    rpc_ctx->endpoint.c_str(),
                    static_cast<unsigned>(remote_patch));
                return false;
            }

            GGML_LOG_INFO(
                "[INFERENCE_RETURN] side=client status=FALLBACK "
                "reason=unavailable endpoint=%s patch=%u\n",
                rpc_ctx->endpoint.c_str(),
                static_cast<unsigned>(remote_patch));
        } else {
            std::unique_lock<std::mutex> lane_lock(
                rpc_dev_ctx->return_lane_mutex[return_lane]);

            rpc_msg_inference_return_bind_req bind {};
            bind.return_req.device = rpc_ctx->device;
            bind.return_req.phone_ffn_seq =
                phone_ffn_seq;
            bind.return_req.tensor =
                serialize_tensor(dst);
            bind.return_req.offset = 0;
            bind.return_req.wire_magic =
                RPC_RETURN_WIRE_MAGIC;
            bind.return_req.wire_type =
                use_f16 ?
                    RPC_RETURN_WIRE_F16 :
                    RPC_RETURN_WIRE_RAW;
            bind.return_req.reserved = 0;
            bind.return_req.original_size =
                data_size;
            bind.graph_id = 0;
            bind.wire_size = wire_size;
            bind.layer = -1;
            bind.chunk = -1;

            const bool stage_profile =
                rpc_tensor_phone_stage_profile_enabled();
            const bool throttle_profile =
                std::getenv(
                    "GGML_META_XLAYER_THROTTLE_PROFILE") != nullptr;
            const bool measure_profile =
                stage_profile || throttle_profile;
            const int64_t begin_us =
                measure_profile ? ggml_time_us() : 0;

            const int64_t bind_begin_us =
                measure_profile ? ggml_time_us() : 0;
            rpc_msg_inference_return_bind_rsp response {};
            bool status = send_rpc_cmd(
                sock,
                RPC_CMD_INFERENCE_RETURN_BIND,
                &bind,
                sizeof(bind),
                &response,
                sizeof(response));
            if (status && response.result == 0) {
                status = false;
            }
            const int64_t bind_done_us =
                measure_profile ? ggml_time_us() : 0;

            if (status) {
                inference_tx_buffer tx =
                    transport->acquire_tx(wire_size);
                status = static_cast<bool>(tx);
                if (status && wire_size > 0) {
                    std::memcpy(
                        tx.data,
                        wire_data,
                        wire_size);
                }

                if (status) {
                    inference_frame_header header =
                        inference_make_header(
                            INFERENCE_FRAME_DATA);
                    header.stage =
                        INFERENCE_STAGE_RETURN;
                    header.stream_id =
                        rpc_ctx->device;
                    header.seq = phone_ffn_seq;
                    header.graph_id = 0;
                    header.layer = -1;
                    header.chunk = -1;

                    status = transport->submit_tx(
                        std::move(tx),
                        header,
                        wire_size);
                } else if (tx) {
                    transport->release_tx(
                        std::move(tx));
                }
            }

            const int64_t submit_done_us =
                measure_profile ? ggml_time_us() : 0;

            int64_t completion_wait_us = 0;
            if (status &&
                    !rpc_inference_return_enqueue_only) {
                const int64_t completion_begin_us =
                    measure_profile ? ggml_time_us() : 0;
                std::vector<uint8_t> ack_payload;
                inference_frame_header ack {};
                status = rpc_inference_receive_matching(
                    rpc_ctx->endpoint,
                    INFERENCE_FRAME_ACK,
                    INFERENCE_STAGE_RETURN,
                    rpc_ctx->device,
                    phone_ffn_seq,
                    0,
                    ack_payload,
                    &ack);
                if (measure_profile) {
                    completion_wait_us =
                        ggml_time_us() -
                        completion_begin_us;
                }
            }

            if (!status) {
                // Once BIND is accepted, a missing DATA/ACK frame would leave
                // the peer state ambiguous. Fail loudly instead of silently
                // reverting to legacy semantics.
                transport->stop();
            }

            const int64_t done_us =
                measure_profile ? ggml_time_us() : 0;

            if (stage_profile || throttle_profile) {
                GGML_LOG_INFO(
                    "[INFERENCE_RETURN] side=client "
                    "seq=%" PRIu64 " lane=%zu "
                    "bytes=%zu wire=%zu wire_type=%s "
                    "bind_ms=%.3f submit_ms=%.3f "
                    "completion_wait_ms=%.3f total_ms=%.3f "
                    "status=%s\n",
                    phone_ffn_seq,
                    return_lane,
                    data_size,
                    wire_size,
                    use_f16 ? "f16" : "raw",
                    (bind_done_us - bind_begin_us) /
                        1000.0,
                    (submit_done_us - bind_done_us) /
                        1000.0,
                    completion_wait_us / 1000.0,
                    (done_us - begin_us) / 1000.0,
                    status ?
                        (rpc_inference_return_enqueue_only ?
                            "QUEUED" : "OK") :
                        "FAIL");
            }

            return status;
        }
    }

    std::vector<uint8_t> input;
    if (use_extended) {
        rpc_msg_set_tensor_async_return_wait_req request {};
        request.device = rpc_ctx->device;
        request.phone_ffn_seq = phone_ffn_seq;
        request.tensor = serialize_tensor(dst);
        request.offset = 0;
        request.wire_magic = RPC_RETURN_WIRE_MAGIC;
        request.wire_type =
            use_f16 ? RPC_RETURN_WIRE_F16 :
                RPC_RETURN_WIRE_RAW;
        if (remote_patch >= RPC_DIRECTIONAL_BULK_MIN_PATCH) {
            request.reserved |=
                RPC_RETURN_WIRE_EARLY_ACK_FLAG;
        }
        request.original_size = data_size;

        input.resize(sizeof(request) + wire_size);
        memcpy(input.data(), &request, sizeof(request));
        if (wire_size > 0) {
            memcpy(
                input.data() + sizeof(request),
                wire_data,
                wire_size);
        }
    } else {
        rpc_msg_set_tensor_async_return_wait_req_v14 request {};
        request.device = rpc_ctx->device;
        request.phone_ffn_seq = phone_ffn_seq;
        request.tensor = serialize_tensor(dst);
        request.offset = 0;

        input.resize(sizeof(request) + data_size);
        memcpy(input.data(), &request, sizeof(request));
        if (data_size > 0) {
            memcpy(
                input.data() + sizeof(request),
                data,
                data_size);
        }
        wire_size = data_size;
    }

    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const bool throttle_profile =
        std::getenv("GGML_META_XLAYER_THROTTLE_PROFILE") != nullptr;
    const bool measure_profile =
        stage_profile || throttle_profile;

    constexpr uint8_t RPC_NETWORK_BULK_MIN_PATCH = 18;
    const bool directional_bulk =
        remote_patch >= RPC_DIRECTIONAL_BULK_MIN_PATCH;

    int64_t bulk_admit_us = 0;
    int64_t request_send_us = 0;
    int64_t receipt_wait_us = 0;
    int64_t tail_wait_us = 0;
    int64_t legacy_grant_us = 0;
    uint64_t legacy_grant_inflight = 0;
    uint64_t legacy_grant_limit = 0;

    // Keep each Return socket single-owner until its final RPC response. The
    // direction-wide mutex below is released earlier, on the Phone receipt ACK.
    std::unique_lock<std::mutex> return_lane_lock;
    if (directional_bulk) {
        return_lane_lock =
            std::unique_lock<std::mutex>(
                rpc_dev_ctx->return_lane_mutex[return_lane]);
    }

    // v18 fallback only: preserve the old Phone-granted shared-credit protocol
    // when talking to an older server.
    if (!directional_bulk &&
            remote_patch >= RPC_NETWORK_BULK_MIN_PATCH &&
            rpc_dev_ctx->network_bulk_shared_active.load(
                std::memory_order_relaxed) != 0) {
        rpc_msg_return_bulk_acquire_req bulk_request {};
        bulk_request.seq = phone_ffn_seq;
        rpc_msg_return_bulk_acquire_rsp bulk_response {};

        const int64_t grant_begin_us =
            measure_profile ? ggml_time_us() : 0;
        const bool grant_status = send_rpc_cmd(
            sock,
            RPC_CMD_RETURN_BULK_ACQUIRE,
            &bulk_request,
            sizeof(bulk_request),
            &bulk_response,
            sizeof(bulk_response));
        const int64_t grant_done_us =
            measure_profile ? ggml_time_us() : 0;
        RPC_STATUS_ASSERT(grant_status);

        const bool shared_bulk_active =
            bulk_response.active != 0;
        rpc_dev_ctx->network_bulk_shared_active.store(
            shared_bulk_active ? 1 : 0,
            std::memory_order_relaxed);
        legacy_grant_us =
            measure_profile ?
                grant_done_us - grant_begin_us :
                0;
        legacy_grant_inflight = bulk_response.inflight;
        legacy_grant_limit = bulk_response.limit;
    }

    const int64_t begin_us =
        measure_profile ? ggml_time_us() : 0;
    bool status = true;

    if (directional_bulk) {
        const int64_t admit_begin_us =
            measure_profile ? ggml_time_us() : 0;
        std::unique_lock<std::mutex> return_wire_lock(
            rpc_dev_ctx->return_wire_mutex);
        const int64_t admit_done_us =
            measure_profile ? ggml_time_us() : 0;
        bulk_admit_us =
            measure_profile ?
                admit_done_us - admit_begin_us :
                0;

        const int64_t request_begin_us =
            measure_profile ? ggml_time_us() : 0;
        status = send_rpc_cmd(
            sock,
            RPC_CMD_SET_TENSOR_ASYNC_RETURN_WAIT,
            input.data(),
            input.size());
        const int64_t request_done_us =
            measure_profile ? ggml_time_us() : 0;
        request_send_us =
            measure_profile ?
                request_done_us - request_begin_us :
                0;

        uint64_t receipt_seq = 0;
        const int64_t receipt_begin_us =
            measure_profile ? ggml_time_us() : 0;
        if (status) {
            status = recv_msg(
                sock,
                &receipt_seq,
                sizeof(receipt_seq));
        }
        const int64_t receipt_done_us =
            measure_profile ? ggml_time_us() : 0;
        receipt_wait_us =
            measure_profile ?
                receipt_done_us - receipt_begin_us :
                0;
        if (status && receipt_seq != phone_ffn_seq) {
            GGML_LOG_ERROR(
                "[RETURN_DIRECTIONAL_ACK_MISMATCH] "
                "expected=%" PRIu64 " actual=%" PRIu64 "\n",
                phone_ffn_seq,
                receipt_seq);
            status = false;
        }

        // The full Return payload is now in Phone userspace. Let the next
        // Return payload use the opposite lane while this RPC waits for Phone
        // FFN dependency/reduce and its final response.
        return_wire_lock.unlock();

        const int64_t tail_begin_us =
            measure_profile ? ggml_time_us() : 0;
        if (status) {
            uint64_t final_size = UINT64_MAX;
            status = sock->recv_data(
                &final_size,
                sizeof(final_size));
            if (status && final_size != 0) {
                status = false;
            }
        }
        const int64_t tail_done_us =
            measure_profile ? ggml_time_us() : 0;
        tail_wait_us =
            measure_profile ?
                tail_done_us - tail_begin_us :
                0;
    } else {
        rpc_cmd_roundtrip_profile roundtrip_profile {};
        rpc_cmd_roundtrip_profile * const previous_profile =
            rpc_cmd_roundtrip_profile_active;
        if (measure_profile) {
            rpc_cmd_roundtrip_profile_active =
                &roundtrip_profile;
        }

        status = send_rpc_cmd(
            sock,
            RPC_CMD_SET_TENSOR_ASYNC_RETURN_WAIT,
            input.data(),
            input.size(),
            nullptr,
            0);

        if (measure_profile) {
            rpc_cmd_roundtrip_profile_active =
                previous_profile;
            request_send_us =
                roundtrip_profile.request_send_us;
            tail_wait_us =
                roundtrip_profile.response_header_wait_us +
                roundtrip_profile.response_body_wait_us;
        }
    }

    const int64_t end_us =
        measure_profile ? ggml_time_us() : 0;

    if (throttle_profile) {
        std::fprintf(
            stderr,
            "[XLAYER_RETURN_RPC] seq=%" PRIu64
            " lane=%zu bytes=%zu wire=%zu mode=%s "
            "admit=%.3f req=%.3f receipt=%.3f tail=%.3f "
            "legacy_grant=%.3f inflight=%" PRIu64 "/%" PRIu64 " "
            "total=%.3f status=%d\n",
            phone_ffn_seq,
            return_lane,
            data_size,
            wire_size,
            directional_bulk ? "DIR1" : "LEGACY",
            bulk_admit_us / 1000.0,
            request_send_us / 1000.0,
            receipt_wait_us / 1000.0,
            tail_wait_us / 1000.0,
            legacy_grant_us / 1000.0,
            legacy_grant_inflight,
            legacy_grant_limit,
            (end_us - begin_us) / 1000.0,
            status ? 1 : 0);
    }

    if (stage_profile) {
        GGML_LOG_INFO(
            "[RETURN_RPC] seq=%" PRIu64
            " lane=%zu bytes=%zu wire=%zu mode=%s "
            "admit=%.3f req=%.3f receipt=%.3f tail=%.3f "
            "legacy_grant=%.3f inflight=%" PRIu64 "/%" PRIu64 " "
            "total=%.3f status=%d\n",
            phone_ffn_seq,
            return_lane,
            data_size,
            wire_size,
            directional_bulk ? "DIR1" : "LEGACY",
            bulk_admit_us / 1000.0,
            request_send_us / 1000.0,
            receipt_wait_us / 1000.0,
            tail_wait_us / 1000.0,
            legacy_grant_us / 1000.0,
            legacy_grant_inflight,
            legacy_grant_limit,
            (end_us - begin_us) / 1000.0,
            status ? 1 : 0);
    }

    return status;
}
static bool ggml_backend_rpc_set_tensor_async_return_enqueue(
        ggml_backend_t backend_dst,
        ggml_tensor * dst,
        const void * data,
        size_t data_size,
        uint64_t phone_ffn_seq,
        size_t return_lane) {
    const bool previous =
        rpc_inference_return_enqueue_only;
    rpc_inference_return_enqueue_only = true;
    const bool status =
        ggml_backend_rpc_set_tensor_async_return_wait(
            backend_dst,
            dst,
            data,
            data_size,
            phone_ffn_seq,
            return_lane);
    rpc_inference_return_enqueue_only = previous;
    return status;
}

static bool ggml_backend_rpc_inference_return_wait(
        ggml_backend_t backend,
        uint64_t phone_ffn_seq) {
    if (backend == nullptr || !ggml_backend_is_rpc(backend)) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend->context);
    if (rpc_ctx == nullptr) {
        return false;
    }

    const char * inference_transport_env =
        std::getenv("GGML_INFERENCE_TRANSPORT");
    const bool inference_new =
        inference_transport_env != nullptr &&
        (std::strcmp(inference_transport_env, "new") == 0 ||
         std::strcmp(inference_transport_env, "force-new") == 0);
    if (!inference_new) {
        return false;
    }

    constexpr uint8_t RPC_INFERENCE_RETURN_ACK_MIN_PATCH = 24;
    const std::string compute_key =
        rpc_ctx->endpoint + "_compute";
    const uint8_t remote_patch =
        rpc_get_remote_patch(compute_key);

    if (phone_ffn_seq == 0) {
        return remote_patch >=
                RPC_INFERENCE_RETURN_ACK_MIN_PATCH &&
            get_inference_transport(
                rpc_ctx->endpoint) != nullptr;
    }

    if (remote_patch <
            RPC_INFERENCE_RETURN_ACK_MIN_PATCH) {
        return false;
    }

    const int64_t begin_us = ggml_time_us();
    std::vector<uint8_t> payload;
    inference_frame_header ack {};
    const bool status =
        rpc_inference_receive_matching(
            rpc_ctx->endpoint,
            INFERENCE_FRAME_ACK,
            INFERENCE_STAGE_RETURN,
            rpc_ctx->device,
            phone_ffn_seq,
            0,
            payload,
            &ack);

    if (rpc_tensor_phone_stage_profile_enabled() ||
            std::getenv(
                "GGML_META_XLAYER_THROTTLE_PROFILE") != nullptr) {
        GGML_LOG_INFO(
            "[INFERENCE_RETURN_ACK] side=client "
            "seq=%" PRIu64 " wait_ms=%.3f status=%s\n",
            phone_ffn_seq,
            (ggml_time_us() - begin_us) / 1000.0,
            status ? "OK" : "FAIL");
    }

    return status;
}

static bool ggml_backend_rpc_set_tensor_async_return(
        ggml_backend_t backend_dst,
        ggml_tensor * dst,
        const void * data,
        size_t data_size) {
    if (backend_dst == nullptr || !ggml_backend_is_rpc(backend_dst)) {
        return false;
    }

    auto * rpc_ctx =
        static_cast<ggml_backend_rpc_context *>(backend_dst->context);
    auto * rpc_dev_ctx =
        static_cast<ggml_backend_rpc_device_context *>(
            ggml_backend_get_device(backend_dst)->context);
    if (rpc_ctx == nullptr || rpc_dev_ctx == nullptr) {
        return false;
    }

    constexpr uint8_t RPC_ASYNC_RETURN_MIN_PATCH = 9;
    const std::string compute_key = rpc_ctx->endpoint + "_compute";
    const uint8_t remote_patch = rpc_get_remote_patch(compute_key);
    if (remote_patch < RPC_ASYNC_RETURN_MIN_PATCH) {
        return false;
    }

    // Null destination/data is a cheap capability query used by Meta.
    if (dst == nullptr && data == nullptr && data_size == 0) {
        return true;
    }
    if (dst == nullptr || data == nullptr ||
            dst->buffer == nullptr ||
            !ggml_backend_buffer_is_rpc(dst->buffer) ||
            data_size != ggml_nbytes(dst)) {
        return false;
    }

    auto * dst_buffer_ctx =
        static_cast<ggml_backend_rpc_buffer_context *>(
            dst->buffer->context);
    if (dst_buffer_ctx == nullptr ||
            dst_buffer_ctx->endpoint != rpc_ctx->endpoint ||
            dst_buffer_ctx->device != rpc_ctx->device) {
        return false;
    }

    auto & return_sock =
        rpc_dev_ctx->return_transfer_socks[0];
    if (return_sock == nullptr) {
        return_sock =
            get_return_transfer_socket(
                rpc_ctx->endpoint,
                0);
    }
    auto sock = return_sock;
    RPC_STATUS_ASSERT(sock != nullptr);

    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const int64_t client_begin_us =
        stage_profile ? ggml_time_us() : 0;

    const int64_t pack_begin_us =
        stage_profile ? ggml_time_us() : 0;
    const rpc_tensor serialized_tensor = serialize_tensor(dst);
    const uint64_t wire_offset = 0;
    const size_t input_size =
        sizeof(serialized_tensor) + sizeof(wire_offset) + data_size;
    std::vector<uint8_t> input(input_size);
    size_t cursor = 0;
    memcpy(input.data() + cursor, &serialized_tensor, sizeof(serialized_tensor));
    cursor += sizeof(serialized_tensor);
    memcpy(input.data() + cursor, &wire_offset, sizeof(wire_offset));
    cursor += sizeof(wire_offset);
    if (data_size > 0) {
        memcpy(input.data() + cursor, data, data_size);
        cursor += data_size;
    }
    GGML_ASSERT(cursor == input.size());
    const int64_t pack_us =
        stage_profile ? ggml_time_us() - pack_begin_us : 0;

    const int64_t send_begin_us = ggml_time_us();
    // Wait only for the server to enqueue the asynchronous OpenCL write.  The
    // layer-tail RPC fence later guarantees device completion before ADD.
    const bool status = send_rpc_cmd(
        sock,
        RPC_CMD_SET_TENSOR_ASYNC_RETURN,
        input.data(),
        input.size(),
        nullptr,
        0);
    const int64_t send_us = ggml_time_us() - send_begin_us;

    if (stage_profile) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=pc stage=return_stage_send "
            "bytes=%zu pack_ms=%.3f send_ms=%.3f total_ms=%.3f\n",
            data_size,
            pack_us / 1000.0,
            send_us / 1000.0,
            (ggml_time_us() - client_begin_us) / 1000.0);
    }

    return status;
}

static int rpc_find_graph_node(
        const ggml_cgraph * graph,
        const ggml_tensor * target) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == target) {
            return i;
        }
    }

    return -1;
}

struct rpc_fused_ffn_input_data {
    const ggml_tensor * tensor = nullptr;
    size_t data_size = 0;
    std::vector<uint8_t> packet;
};

static bool rpc_tensor_depends_on(
        const ggml_tensor * tensor,
        const ggml_tensor * target,
        std::unordered_set<const ggml_tensor *> & visited) {
    if (tensor == nullptr) {
        return false;
    }
    if (tensor == target) {
        return true;
    }
    if (!visited.insert(tensor).second) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (rpc_tensor_depends_on(tensor->src[i], target, visited)) {
            return true;
        }
    }
    return rpc_tensor_depends_on(tensor->view_src, target, visited);
}

static bool rpc_graph_uses_tensor(
        const ggml_cgraph * graph,
        const ggml_tensor * tensor) {
    std::unordered_set<const ggml_tensor *> visited;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (rpc_tensor_depends_on(graph->nodes[i], tensor, visited)) {
            return true;
        }
    }
    return false;
}

static bool rpc_flush_pending_fused_input(
        ggml_backend_rpc_device_context * rpc_dev_ctx) {
    rpc_fused_ffn_input_data pending {};

    {
        auto & fused = rpc_dev_ctx->fused_ffn;
        std::lock_guard<std::mutex> lock(fused.mutex);
        if (!fused.pending) {
            return true;
        }

        pending.tensor = fused.tensor;
        pending.data_size = fused.data_size;
        pending.packet = std::move(fused.packet);

        fused.pending = false;
        fused.tensor = nullptr;
    }

    if (pending.packet.size() != FUSED_DATA_OFFSET + pending.data_size) {
        return false;
    }

    rpc_msg_set_tensor_recompute_snapshot_req request {};
    memcpy(&request, pending.packet.data() + FUSED_REQ_OFFSET, sizeof(request));

    const size_t input_size = sizeof(request.input_tensor) + sizeof(request.input_offset) + pending.data_size;
    std::vector<uint8_t> input(input_size);
    memcpy(input.data(), &request.input_tensor, sizeof(request.input_tensor));
    memcpy(input.data() + sizeof(request.input_tensor), &request.input_offset, sizeof(request.input_offset));
    if (pending.data_size > 0) {
        memcpy(
            input.data() + sizeof(request.input_tensor) + sizeof(request.input_offset),
            pending.packet.data() + FUSED_DATA_OFFSET,
            pending.data_size);
    }

    auto sock = get_socket(rpc_dev_ctx->endpoint);
    RPC_STATUS_ASSERT(sock != nullptr);
    return send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    const uint64_t graph_uid = rpc_graph_effective_uid(cgraph); //这个是客户端的
    const bool cacheable = cgraph->uid != 0;
    bool reuse = cacheable && rpc_dev_ctx->graph_uids.find(graph_uid) != rpc_dev_ctx->graph_uids.end();
    //新的
    //reuse = false;
    if (reuse) {
        rpc_graph_snapshot_context graph_snapshot {};
        bool use_fused = false;

       int output_node = -1;

        {
            auto & prepared = rpc_ctx->graph_snapshot;
            std::lock_guard<std::mutex> lock(prepared.mutex);

            if (prepared.active) {
                output_node =
                    rpc_find_graph_node(
                        cgraph,
                        prepared.tensor);

                if (output_node >= 0) {
                    graph_snapshot.tensor = prepared.tensor;
                    graph_snapshot.offset = prepared.offset;
                    graph_snapshot.size   = prepared.size;
                    graph_snapshot.slot   = prepared.slot;
                    graph_snapshot.seq    = prepared.seq;
                    use_fused = true;
                }

                prepared.active = false;
                prepared.tensor = nullptr;
            }
        }
        rpc_fused_ffn_input_data fused_input {};
        bool have_fused_input = false;
        if (use_fused && output_node >= 0) {
            auto & pending = rpc_dev_ctx->fused_ffn;
            std::lock_guard<std::mutex> lock(pending.mutex);
            if (pending.pending && rpc_graph_uses_tensor(cgraph, pending.tensor)) {
                fused_input.tensor = pending.tensor;
                fused_input.data_size = pending.data_size;
                fused_input.packet = std::move(pending.packet);

                pending.pending = false;
                pending.tensor = nullptr;
                have_fused_input = true;
            }
        }

        if (output_node >= 0 && have_fused_input) {
            RPC_STATUS_ASSERT(fused_input.packet.size() == FUSED_DATA_OFFSET + fused_input.data_size);

            rpc_msg_set_tensor_recompute_snapshot_req request {};
            memcpy(&request, fused_input.packet.data() + FUSED_REQ_OFFSET, sizeof(request));

            request.graph.device = rpc_ctx->device;
            request.graph.slot = graph_snapshot.slot;
            request.graph.graph_uid = graph_uid;
            request.graph.seq = graph_snapshot.seq;
            request.graph.output_node = output_node;
            request.graph.offset = graph_snapshot.offset;
            request.graph.size = graph_snapshot.size;
            memcpy(fused_input.packet.data() + FUSED_REQ_OFFSET, &request, sizeof(request));

            auto sock = get_socket(rpc_ctx->endpoint);
            const bool status = sock->send_data(
                fused_input.packet.data(),
                fused_input.packet.size());
            RPC_STATUS_ASSERT(status);
            rpc_fence_diag_record_graph(
                rpc_ctx, graph_uid, cgraph,
                "SET_TENSOR_RECOMPUTE_SNAPSHOT");

            auto & prepared = rpc_ctx->graph_snapshot;
            std::lock_guard<std::mutex> lock(prepared.mutex);
            prepared.snapshot_was_fused = true;
            prepared.fused_snapshot_seq = request.graph.seq;

            return GGML_STATUS_SUCCESS;
        }

        RPC_STATUS_ASSERT(rpc_flush_pending_fused_input(rpc_dev_ctx));

        if (output_node >= 0) {
            rpc_msg_graph_recompute_snapshot_req request {};
            request.device = rpc_ctx->device;
            request.slot = graph_snapshot.slot;
            request.graph_uid = graph_uid;
            request.seq = graph_snapshot.seq;
            request.output_node = output_node;
            request.offset = graph_snapshot.offset;
            request.size = graph_snapshot.size;

            auto sock = get_socket(rpc_ctx->endpoint);
            const bool status = send_rpc_cmd_compact_small(
                sock,
                RPC_CMD_GRAPH_RECOMPUTE_SNAPSHOT,
                request);
            RPC_STATUS_ASSERT(status);
            rpc_fence_diag_record_graph(
                rpc_ctx, graph_uid, cgraph,
                "GRAPH_RECOMPUTE_SNAPSHOT");

            auto & prepared = rpc_ctx->graph_snapshot;
            std::lock_guard<std::mutex> lock(prepared.mutex);
            prepared.snapshot_was_fused = true;
            prepared.fused_snapshot_seq = request.seq;

            return GGML_STATUS_SUCCESS;
        }

        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        request.graph_uid = graph_uid;
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd_compact_small(
            sock,
            RPC_CMD_GRAPH_RECOMPUTE,
            request);
        GGML_LOG_INFO(
            "[RPC_GRAPH] endpoint=%s device=%u graph=%p uid=%" PRIu64
            " nodes=%d tensors=reused bytes=%zu cmd=GRAPH_RECOMPUTE\n",
            rpc_ctx->endpoint.c_str(),
            rpc_ctx->device,
            (void *) cgraph,
            graph_uid,
            cgraph->n_nodes,
            sizeof(request));
        RPC_STATUS_ASSERT(status);
        rpc_fence_diag_record_graph(
            rpc_ctx, graph_uid, cgraph,
            "GRAPH_RECOMPUTE");
    } else {
        RPC_STATUS_ASSERT(rpc_flush_pending_fused_input(rpc_dev_ctx));

        {
            auto & prepared = rpc_ctx->graph_snapshot;
            std::lock_guard<std::mutex> lock(prepared.mutex);
            prepared.active = false;
            prepared.tensor = nullptr;
        }

        if (cacheable) {
            rpc_dev_ctx->graph_uids.insert(graph_uid);
        }
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, graph_uid, cgraph, input);
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        uint32_t n_tensors = 0;
        const size_t n_tensors_offset = 2*sizeof(uint32_t) + sizeof(uint64_t) + (size_t) cgraph->n_nodes*sizeof(uint64_t);
        if (input.size() >= n_tensors_offset + sizeof(n_tensors)) {
            memcpy(&n_tensors, input.data() + n_tensors_offset, sizeof(n_tensors));
        }
        // 打印发送和接收的字节数，计算图
        //GGML_LOG_INFO("ggml_backend_rpc_graph_compute: sent %zu bytes\n", input.size());
        GGML_LOG_INFO(
            "[RPC_GRAPH] endpoint=%s device=%u graph=%p uid=%" PRIu64
            " nodes=%d tensors=%u bytes=%zu cmd=GRAPH_COMPUTE\n",
            rpc_ctx->endpoint.c_str(),
            rpc_ctx->device,
            (void *) cgraph,
            graph_uid,
            cgraph->n_nodes,
            n_tensors,
            input.size());
        RPC_STATUS_ASSERT(status);
        rpc_fence_diag_record_graph(
            rpc_ctx, graph_uid, cgraph,
            "GRAPH_COMPUTE");
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint       = */ endpoint,
        /* .device         = */ device,
        /* .name           = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };

    const char * inference_env =
        std::getenv("GGML_INFERENCE_TRANSPORT");
    const bool inference_new =
        inference_env != nullptr &&
        (std::strcmp(inference_env, "new") == 0 ||
         std::strcmp(inference_env, "force-new") == 0);

    const char * probe_env =
        std::getenv("GGML_INFERENCE_TRANSPORT_PROBE");
    const bool probe =
        probe_env != nullptr && std::atoi(probe_env) != 0;

    if (inference_env != nullptr || probe) {
        GGML_LOG_INFO(
            "[INFERENCE_TRANSPORT_CONFIG] endpoint=%s device=%u "
            "mode=%s probe=%d\n",
            endpoint,
            device,
            inference_env != nullptr ? inference_env : "legacy",
            probe ? 1 : 0);
    }

    if (inference_new || probe) {
        const bool opened =
            get_inference_transport(endpoint) != nullptr;
        GGML_LOG_INFO(
            "[INFERENCE_DATA_PLANE_PROBE] endpoint=%s device=%u "
            "mode=%s status=%s\n",
            endpoint,
            device,
            inference_env != nullptr ? inference_env : "legacy",
            opened ? "OK" : "FAIL");

        if (!opened &&
            inference_env != nullptr &&
            std::strcmp(inference_env, "force-new") == 0) {
            GGML_LOG_ERROR(
                "[INFERENCE_TRANSPORT_CONFIG] endpoint=%s device=%u "
                "mode=force-new status=FAIL reason=open_failed\n",
                endpoint,
                device);
        }
    }

    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

// RPC server-side implementation
enum class rpc_snapshot_state {
    FREE,
    FILLING,
    READY,
    SENDING,
};

struct rpc_snapshot_slot {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<uint8_t> data;
    uint64_t seq = 0;
    int64_t free_wait_us = 0;
    int64_t fill_us = 0;
    rpc_snapshot_state state = rpc_snapshot_state::FREE;
};

struct rpc_snapshot_device {
    std::array<rpc_snapshot_slot, 2> slots;
};


struct rpc_route_snapshot_slot {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<uint8_t> data;
    std::array<uint64_t, 3> sizes {};
    uint64_t seq = 0;
    int64_t fill_us = 0;
    bool topk_compact_supported = false;
    size_t topk_rows = 0;
    size_t topk_row_bytes = 0;
    size_t topk_stride_bytes = 0;
    size_t topk_wire_bytes = 0;
    rpc_snapshot_state state = rpc_snapshot_state::FREE;
};

struct rpc_route_snapshot_device {
    std::array<rpc_route_snapshot_slot, RPC_ROUTE_TRANSFER_LANES> slots;
};

struct rpc_snapshot_breakdown {
    uint64_t requests      = 0;
    uint64_t bytes         = 0;
    int64_t  ready_wait_us = 0;
    int64_t  send_us       = 0;
};

// OpenCL keeps backend-private tensor state in tensor->extra (cl_mem,
// quantized sub-buffers/images, offsets, etc.). RPC reconstructs ggml_tensor
// objects for each request, so that pointer is otherwise lost. Keep this
// cache strictly OpenCL-only so the existing CPU/CUDA RPC paths are untouched.
struct rpc_opencl_tensor_extra_key {
    ggml_backend_buffer_t buffer = nullptr;
    uint64_t data = 0;
    enum ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, GGML_MAX_DIMS> ne {};
    std::array<size_t, GGML_MAX_DIMS> nb {};
    std::string name;

    bool operator==(const rpc_opencl_tensor_extra_key & other) const {
        return buffer == other.buffer &&
               data == other.data &&
               type == other.type &&
               ne == other.ne &&
               nb == other.nb &&
               name == other.name;
    }
};

struct rpc_opencl_tensor_extra_key_hash {
    size_t operator()(const rpc_opencl_tensor_extra_key & key) const {
        size_t h = std::hash<uintptr_t>{}(
            reinterpret_cast<uintptr_t>(key.buffer));
        auto mix = [&](size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(std::hash<uint64_t>{}(key.data));
        mix(std::hash<int>{}((int) key.type));
        for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
            mix(std::hash<int64_t>{}(key.ne[i]));
            mix(std::hash<size_t>{}(key.nb[i]));
        }
        mix(std::hash<std::string>{}(key.name));
        return h;
    }
};

// Graph reconstruction can legitimately change tensor metadata such as name,
// ne/nb while still referring to the same OpenCL allocation. Keep a second,
// storage-level key so a strict-key miss does not silently replace a quantized
// type-specific extra with the generic ggml_tensor_extra_cl layout.
struct rpc_opencl_tensor_storage_key {
    ggml_backend_buffer_t buffer = nullptr;
    uint64_t data = 0;
    enum ggml_type type = GGML_TYPE_F32;

    bool operator==(const rpc_opencl_tensor_storage_key & other) const {
        return buffer == other.buffer &&
               data == other.data &&
               type == other.type;
    }
};

struct rpc_opencl_tensor_storage_key_hash {
    size_t operator()(const rpc_opencl_tensor_storage_key & key) const {
        size_t h = std::hash<uintptr_t>{}(
            reinterpret_cast<uintptr_t>(key.buffer));
        auto mix = [&](size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(std::hash<uint64_t>{}(key.data));
        mix(std::hash<int>{}((int) key.type));
        return h;
    }
};

struct rpc_opencl_tensor_storage_entry {
    void * extra = nullptr;
    bool ambiguous = false;
};

static struct ggml_cgraph rpc_graph_view(
        struct ggml_cgraph * cgraph0,
        int i0,
        int i1) {
    GGML_ASSERT(cgraph0 != nullptr);
    GGML_ASSERT(i0 >= 0);
    GGML_ASSERT(i1 >= i0);
    GGML_ASSERT(i1 <= cgraph0->n_nodes);

    struct ggml_cgraph cgraph = {
        /* .size             = */ 0,
        /* .n_nodes          = */ i1 - i0,
        /* .n_leafs          = */ 0,
        /* .nodes            = */ cgraph0->nodes + i0,
        /* .grads            = */ nullptr,
        /* .grad_accs        = */ nullptr,
        /* .leafs            = */ nullptr,
        /* .use_counts       = */ cgraph0->use_counts,
        /* .visited_hash_set = */ cgraph0->visited_hash_set,
        /* .order            = */ cgraph0->order,
        /* .uid              = */ 0,
    };
    return cgraph;
}

struct rpc_phone_npu_ffn_stage {
    int layer = -1;
    int chunk = -1;
    bool prefill = false;
    ggml_tensor * norm = nullptr;
    ggml_tensor * topk = nullptr;
    ggml_tensor * route_weights = nullptr;
    ggml_tensor * down = nullptr;
    int front_end = 0;
    int down_index = -1;
};

struct rpc_phone_npu_ffn_cache {
    int layer = -1;
    ggml_context_ptr ctx_w;
    ggml_backend_buffer_ptr wbuf;
    ggml_tensor * w_up = nullptr;
    ggml_tensor * w_gate = nullptr;
    ggml_tensor * w_down = nullptr;
};

struct rpc_phone_npu_cpu_reduce_entry {
    int layer = -1;
    int chunk = -1;
    ggml_tensor * down = nullptr;
    std::vector<float> phone_partial;
};

struct rpc_phone_npu_async_job {
    uint32_t device = 0;
    uint64_t seq = 0;
    int layer = -1;
    int chunk = -1;
    bool prefill = false;

    // Stable direct-bound destination owned by the Ring.  The NPU worker does
    // not write it; CPU reduce writes the complete PC+NPU result once.
    ggml_tensor * down = nullptr;

    // Per-layer resident private-HTP weights.  The pointee remains stable even
    // if the unordered_map holding the unique_ptr rehashes.
    rpc_phone_npu_ffn_cache * cache = nullptr;

    int64_t n_embd = 0;
    int64_t n_tokens = 0;
    int64_t n_expert_used = 0;
    int64_t enqueue_us = 0;

    // GPU allocator-backed stage tensors must not escape graph_compute().
    // Freeze all NPU inputs into host-owned storage before enqueueing.
    std::vector<uint8_t> hidden_dense;
    std::vector<uint8_t> topk_dense;
    std::vector<uint8_t> route_dense;
};

static ggml_backend_buffer_type_t rpc_phone_npu_repack_buft(
        ggml_backend_t backend) {
    if (backend == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return nullptr;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return nullptr;
    }
    auto fn = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(
            reg, "ggml_backend_dev_get_extra_bufts"));
    if (fn == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_type_t * bufts = fn(dev);
    return bufts != nullptr ? bufts[0] : nullptr;
}

static ggml_context_ptr rpc_phone_npu_make_ctx(
        size_t bytes = 8u * 1024u * 1024u) {
    ggml_init_params params {
        /* .mem_size   = */ bytes,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_context_ptr { ggml_init(params) };
}

static bool rpc_phone_npu_read_dense(
        const ggml_tensor * tensor,
        std::vector<uint8_t> & dense) {
    if (tensor == nullptr || tensor->buffer == nullptr ||
        ggml_blck_size(tensor->type) != 1) {
        return false;
    }

    const size_t elem = ggml_type_size(tensor->type);
    const size_t span = ggml_nbytes(tensor);
    const size_t n = (size_t) ggml_nelements(tensor);

    std::vector<uint8_t> raw(span);
    ggml_backend_tensor_get(
        tensor, raw.data(), 0, raw.size());

    dense.resize(n * elem);
    size_t out = 0;

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t off =
                        (size_t) i0 * tensor->nb[0] +
                        (size_t) i1 * tensor->nb[1] +
                        (size_t) i2 * tensor->nb[2] +
                        (size_t) i3 * tensor->nb[3];

                    if (off + elem > raw.size()) {
                        return false;
                    }

                    memcpy(
                        dense.data() + out * elem,
                        raw.data() + off,
                        elem);
                    ++out;
                }
            }
        }
    }

    return out == n;
}

static bool rpc_phone_npu_supported_k_quant(
        enum ggml_type type) {
    return type == GGML_TYPE_Q4_K ||
           type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_Q6_K;
}

static bool rpc_phone_npu_convert_k_to_q4_0(
        const ggml_tensor * weight,
        std::vector<uint8_t> & converted) {
    if (weight == nullptr || weight->buffer == nullptr ||
        !rpc_phone_npu_supported_k_quant(weight->type) ||
        !ggml_is_contiguous(weight) ||
        weight->ne[0] <= 0 ||
        weight->ne[0] % QK_K != 0 ||
        weight->ne[0] % ggml_blck_size(GGML_TYPE_Q4_0) != 0) {
        return false;
    }

    const int64_t row_elems = weight->ne[0];
    const int64_t nrows = ggml_nelements(weight) / row_elems;
    const size_t src_row =
        ggml_row_size(weight->type, row_elems);
    const size_t dst_row =
        ggml_row_size(GGML_TYPE_Q4_0, row_elems);
    const size_t src_bytes = src_row * (size_t) nrows;

    if (src_bytes != ggml_nbytes(weight)) {
        return false;
    }

    std::vector<uint8_t> src(src_bytes);
    ggml_backend_tensor_get(
        weight, src.data(), 0, src.size());

    converted.resize(dst_row * (size_t) nrows);
    std::vector<float> row((size_t) row_elems);

    for (int64_t i = 0; i < nrows; ++i) {
        const uint8_t * src_row_data =
            src.data() + (size_t) i * src_row;
        auto * q4_row =
            reinterpret_cast<block_q4_0 *>(
                converted.data() + (size_t) i * dst_row);

        switch (weight->type) {
            case GGML_TYPE_Q4_K:
                dequantize_row_q4_K(
                    reinterpret_cast<const block_q4_K *>(
                        src_row_data),
                    row.data(), row_elems);
                break;
            case GGML_TYPE_Q5_K:
                dequantize_row_q5_K(
                    reinterpret_cast<const block_q5_K *>(
                        src_row_data),
                    row.data(), row_elems);
                break;
            case GGML_TYPE_Q6_K:
                dequantize_row_q6_K(
                    reinterpret_cast<const block_q6_K *>(
                        src_row_data),
                    row.data(), row_elems);
                break;
            default:
                return false;
        }

        quantize_row_q4_0_ref(
            row.data(), q4_row, row_elems);
    }

    return true;
}

static ggml_tensor * rpc_phone_npu_find_named_rec(
        ggml_tensor * tensor,
        const char * name,
        std::unordered_set<ggml_tensor *> & seen) {
    if (tensor == nullptr || name == nullptr) {
        return nullptr;
    }
    if (std::strcmp(tensor->name, name) == 0) {
        return tensor;
    }
    if (!seen.insert(tensor).second) {
        return nullptr;
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (ggml_tensor * found =
                rpc_phone_npu_find_named_rec(
                    tensor->src[i], name, seen)) {
            return found;
        }
    }
    return rpc_phone_npu_find_named_rec(
        tensor->view_src, name, seen);
}

static ggml_tensor * rpc_phone_npu_find_named(
        ggml_cgraph * graph,
        const char * name) {
    if (graph == nullptr || name == nullptr) {
        return nullptr;
    }

    std::unordered_set<ggml_tensor *> seen;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (ggml_tensor * found =
                rpc_phone_npu_find_named_rec(
                    graph->nodes[i], name, seen)) {
            return found;
        }
    }
    return nullptr;
}


static int rpc_phone_npu_graph_node_index(
        const ggml_cgraph * graph,
        const ggml_tensor * tensor) {
    if (graph == nullptr || tensor == nullptr) {
        return -1;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == tensor) {
            return i;
        }
    }
    return -1;
}

static bool rpc_phone_npu_is_ancestor(
        const ggml_tensor * root,
        const ggml_tensor * needle,
        int depth,
        std::unordered_set<const ggml_tensor *> & seen) {
    if (root == nullptr || needle == nullptr || depth < 0) {
        return false;
    }
    if (root == needle) {
        return true;
    }
    if (!seen.insert(root).second) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (rpc_phone_npu_is_ancestor(
                root->src[i], needle, depth - 1, seen)) {
            return true;
        }
    }
    return rpc_phone_npu_is_ancestor(
        root->view_src, needle, depth - 1, seen);
}


static bool rpc_phone_npu_finalize_stage(
        ggml_cgraph * graph,
        rpc_phone_npu_ffn_stage & stage) {
    if (graph == nullptr ||
        stage.norm == nullptr ||
        stage.topk == nullptr ||
        stage.route_weights == nullptr ||
        stage.down == nullptr ||
        stage.norm->type != GGML_TYPE_F32 ||
        stage.topk->type != GGML_TYPE_I32 ||
        stage.route_weights->type != GGML_TYPE_F32 ||
        stage.down->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(stage.down)) {
        return false;
    }

    const int norm_index =
        rpc_phone_npu_graph_node_index(graph, stage.norm);
    const int topk_index =
        rpc_phone_npu_graph_node_index(graph, stage.topk);
    const int route_index =
        rpc_phone_npu_graph_node_index(
            graph, stage.route_weights);
    stage.down_index =
        rpc_phone_npu_graph_node_index(graph, stage.down);
    if (stage.down_index < 0) {
        return false;
    }

    stage.front_end =
        std::max({ norm_index, topk_index, route_index }) + 1;
    if (stage.front_end < 0 ||
        stage.front_end > stage.down_index) {
        return false;
    }

    for (int i = stage.front_end;
         i <= stage.down_index;
         ++i) {
        std::unordered_set<const ggml_tensor *> seen;
        if (!rpc_phone_npu_is_ancestor(
                stage.down, graph->nodes[i],
                128, seen)) {
            return false;
        }
    }

    return true;
}

static std::vector<rpc_phone_npu_ffn_stage>
rpc_phone_npu_find_ffn_stages(
        ggml_cgraph * graph) {
    std::vector<rpc_phone_npu_ffn_stage> result;
    if (graph == nullptr) {
        return result;
    }

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        if (node == nullptr || node->name[0] == '\0') {
            continue;
        }

        int parsed = 0;
        int chunk = -1;
        int layer = -1;

        if (std::sscanf(
                node->name,
                "prefill_ffn_down_chunk_%d-%d%n",
                &chunk, &layer, &parsed) == 2 &&
            node->name[parsed] == '\0') {
            char name[GGML_MAX_NAME] = {};

            rpc_phone_npu_ffn_stage stage;
            stage.layer = layer;
            stage.chunk = chunk;
            stage.prefill = true;
            stage.down = node;

            std::snprintf(
                name, sizeof(name),
                "prefill_ffn_norm_chunk_%d-%d",
                chunk, layer);
            stage.norm =
                rpc_phone_npu_find_named(graph, name);

            std::snprintf(
                name, sizeof(name),
                "phone_prefill_route_topk_chunk_%d-%d",
                chunk, layer);
            stage.topk =
                rpc_phone_npu_find_named(graph, name);

            std::snprintf(
                name, sizeof(name),
                "phone_prefill_route_weights_chunk_%d-%d",
                chunk, layer);
            stage.route_weights =
                rpc_phone_npu_find_named(graph, name);

            // These explicit markers are emitted only for
            // TENSOR_PHONE_PRIMARY layers.  This makes HTP ownership follow
            // the planner's Tensor region automatically.
            if (stage.topk == nullptr ||
                stage.route_weights == nullptr) {
                continue;
            }

            if (rpc_phone_npu_finalize_stage(
                    graph, stage)) {
                result.push_back(stage);
            }
            continue;
        }

        parsed = 0;
        layer = -1;
        if (std::sscanf(
                node->name,
                "ffn_moe_out-%d%n",
                &layer, &parsed) == 1 &&
            node->name[parsed] == '\0') {
            char name[GGML_MAX_NAME] = {};

            std::snprintf(
                name, sizeof(name),
                "ffn_down_chunk_0-%d", layer);
            if (rpc_phone_npu_find_named(
                    graph, name) != nullptr) {
                continue;
            }

            rpc_phone_npu_ffn_stage stage;
            stage.layer = layer;
            stage.chunk = 0;
            stage.down = node;

            std::snprintf(
                name, sizeof(name),
                "ffn_norm-%d", layer);
            stage.norm =
                rpc_phone_npu_find_named(graph, name);
            stage.prefill =
                stage.norm != nullptr &&
                stage.norm->ne[1] > 1;

            std::snprintf(
                name, sizeof(name),
                "phone_moe_route_topk-%d", layer);
            stage.topk =
                rpc_phone_npu_find_named(graph, name);

            std::snprintf(
                name, sizeof(name),
                "phone_moe_route_weights-%d", layer);
            stage.route_weights =
                rpc_phone_npu_find_named(graph, name);

            // PHONE_ONLY and PC-primary Tensor layers do not carry these
            // phone_moe_route_* markers and remain on their existing path.
            if (stage.topk == nullptr ||
                stage.route_weights == nullptr) {
                continue;
            }

            if (rpc_phone_npu_finalize_stage(
                    graph, stage)) {
                result.push_back(stage);
            }
        }
    }

    std::sort(
        result.begin(), result.end(),
        [](const rpc_phone_npu_ffn_stage & a,
           const rpc_phone_npu_ffn_stage & b) {
            return a.down_index < b.down_index;
        });

    int cursor = 0;
    for (const auto & stage : result) {
        if (stage.front_end < cursor ||
            stage.down_index < stage.front_end) {
            result.clear();
            break;
        }
        cursor = stage.down_index + 1;
    }

    return result;
}

class rpc_server {
public:
    rpc_server(
            std::vector<ggml_backend_t> all_backends,
            const char * cache_dir,
            const ggml_rpc_local_tensor_source * tensor_source = nullptr,
            ggml_backend_t phone_npu_backend = nullptr)
        : backends(std::move(all_backends)),
          phone_npu_backend(phone_npu_backend),
          cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        snapshot_devices.reserve(backends.size());
        route_snapshot_devices.reserve(backends.size());
        for (size_t i = 0; i < backends.size(); ++i) {
            snapshot_devices.emplace_back(std::make_unique<rpc_snapshot_device>());
            route_snapshot_devices.emplace_back(
                std::make_unique<rpc_route_snapshot_device>());
        }
        if (tensor_source != nullptr) {
            local_tensor_source = *tensor_source;
        }

        // v19 formal policy: one Route + one Return may overlap, but never
        // more than one bulk payload in either direction. Route wire
        // completion is still tracked by the PC full-receive ACK.
        network_bulk_shared = false;
        route_bulk_limit = 1;

        const char * legacy_network_bulk_env =
            std::getenv("GGML_RPC_NETWORK_BULK_INFLIGHT");
        const char * legacy_route_bulk_env =
            std::getenv("GGML_RPC_ROUTE_BULK_INFLIGHT");
        GGML_LOG_INFO(
            "[RPC_DIRECTIONAL_BULK] route_inflight=1 "
            "return_inflight=1 route_slots=%zu return_slots=%zu"
            "%s%s\n",
            RPC_ROUTE_TRANSFER_LANES,
            RPC_RETURN_TRANSFER_LANES,
            legacy_network_bulk_env != nullptr ?
                " legacy_network_env_ignored=1" : "",
            legacy_route_bulk_env != nullptr ?
                " legacy_route_env_ignored=1" : "");
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_direct(const rpc_tensor & in_tensor, uint64_t offset, const void * data, size_t size);
    bool set_tensor_direct_opencl_async(
        const rpc_tensor & in_tensor,
        uint64_t offset,
        const void * data,
        size_t size);
    bool set_tensor_recompute_snapshot(const std::vector<uint8_t> & input);
    bool set_tensor_async_return(const std::vector<uint8_t> & input);
    bool set_tensor_async_return_wait(const std::vector<uint8_t> & input);
    bool phone_ffn_mark_ready(const rpc_msg_phone_ffn_mark_ready_req & request);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor_into(const rpc_msg_get_tensor_req & request, void * response_data);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool get_tensor_batch3(const rpc_msg_get_tensor_batch3_req & request, std::vector<uint8_t> & response);
    bool get_tensor_batch3_wait(const rpc_msg_get_tensor_batch3_wait_req & request, std::vector<uint8_t> & response);
    bool route_mark_ready(const rpc_msg_route_mark_ready_req & request);
    bool route_snapshot_ready(
        const rpc_msg_route_snapshot_ready_req & request,
        rpc_msg_route_snapshot_ready_rsp & response);
    bool get_route_snapshot(
        const rpc_msg_get_route_snapshot_req & request,
        std::vector<uint8_t> & response);
    size_t acquire_route_bulk_credit() {
        std::unique_lock<std::mutex> lock(route_bulk_mutex);
        route_bulk_cv.wait(
            lock,
            [&]() {
                return route_bulk_inflight <
                    route_bulk_limit;
            });
        ++route_bulk_inflight;
        return route_bulk_inflight;
    }
    void release_route_bulk_credit() {
        {
            std::lock_guard<std::mutex> lock(route_bulk_mutex);
            GGML_ASSERT(route_bulk_inflight > 0);
            --route_bulk_inflight;
        }
        route_bulk_cv.notify_one();
    }
    size_t route_bulk_transfer_limit() const {
        return route_bulk_limit;
    }
    bool network_bulk_shared_enabled() const {
        return network_bulk_shared;
    }
    bool route_bulk_wire_ack_required() const {
        return route_bulk_limit < RPC_ROUTE_TRANSFER_LANES ||
            network_bulk_shared;
    }
    bool set_tensor_graph_compute(const std::vector<uint8_t> & input);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool graph_recompute_snapshot(const rpc_msg_graph_recompute_snapshot_req & request);
    bool synchronize(uint32_t device);
    bool snapshot_tensor(const rpc_msg_snapshot_tensor_req & request);
    bool snapshot_tensor_direct(uint32_t device, uint32_t slot, uint64_t seq, ggml_tensor * tensor, uint64_t offset, uint64_t size);
    bool send_snapshot(const rpc_msg_get_snapshot_req & request, socket_ptr sock);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool init_zero_tensor(
        const rpc_msg_init_zero_tensor_req & request,
        rpc_msg_init_zero_tensor_rsp & response);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
    bool set_tensor_from_local_file(const rpc_msg_set_tensor_from_local_file_req & request, rpc_msg_set_tensor_from_local_file_rsp & response); //新加的

    bool send_inference_cycle(const rpc_msg_inference_cycle_send_req & request);
    bool apply_inference_cycle(const rpc_msg_inference_cycle_send_req & request);
    bool finish_inference_cycle(const rpc_msg_inference_cycle_send_req & request);
    using cycle_rx_key = std::tuple<int32_t, int32_t, uint64_t, uint64_t, uint32_t>;
    static cycle_rx_key cycle_key(int32_t layer, int32_t chunk,
                                   uint64_t seq, uint64_t graph_id, uint32_t kind) {
        return {layer, chunk, seq, graph_id, kind};
    }
    bool receive_c2p_inference_frame(const inference_frame_header & header,
                                      const uint8_t * data, size_t size) {
        inference_cycle_message msg;
        if (header.type != INFERENCE_FRAME_DATA ||
            !inference_cycle_decode_frame(header, data, size, msg) ||
            msg.direction != INFERENCE_CYCLE_PC_TO_PHONE) {
            return false;
        }
        const auto key = cycle_key(
            msg.layer, msg.chunk, msg.seq, msg.graph_id, msg.kind);
        {
            std::lock_guard<std::mutex> lock(cycle_rx_mutex);
            if (cycle_rx_failed || cycle_rx_frames.size() >= 32 ||
                !cycle_rx_frames.emplace(key, std::move(msg)).second) {
                return false;
            }
        }
        cycle_rx_cv.notify_all();
        GGML_LOG_INFO(
            "[CYCLE_C2P_RX] side=phone layer=%d chunk=%d seq=%" PRIu64
            " kind=%s bytes=%zu state=HOST_READY\n",
            header.layer, header.chunk, header.seq,
            header.stage == INFERENCE_STAGE_CYCLE_READY ? "READY" : "PARTIAL",
            size);
        return true;
    }
    void fail_cycle_inbound() {
        {
            std::lock_guard<std::mutex> lock(cycle_rx_mutex);
            cycle_rx_failed = true;
            cycle_rx_frames.clear();
            cycle_rx_applied_partial.clear();
        }
        cycle_rx_cv.notify_all();
    }

    bool inference_data_plane_available() {
        std::lock_guard<std::mutex> lock(inference_data_plane_mutex);
        return !inference_data_plane || !inference_data_plane->healthy();
    }

    bool attach_inference_data_plane(
        const std::shared_ptr<inference_transport> & transport) {
        if (!transport) {
            return false;
        }

        std::lock_guard<std::mutex> lock(inference_data_plane_mutex);
        if (inference_data_plane && inference_data_plane->healthy()) {
            return false;
        }

        inference_data_plane = transport;
        {
            std::lock_guard<std::mutex> inbound_lock(cycle_rx_mutex);
            cycle_rx_failed = false;
            cycle_rx_frames.clear();
            cycle_rx_applied_partial.clear();
        }
        return true;
    }

    void detach_inference_data_plane(
        const std::shared_ptr<inference_transport> & transport) {
        std::lock_guard<std::mutex> lock(inference_data_plane_mutex);
        if (inference_data_plane == transport) {
            inference_data_plane.reset();
        }
    }

    std::shared_ptr<inference_transport> get_inference_data_plane() {
        std::lock_guard<std::mutex> lock(inference_data_plane_mutex);
        return inference_data_plane;
    }

    bool bind_inference_return(
        const rpc_msg_inference_return_bind_req & request) {
        auto transport = get_inference_data_plane();
        if (!transport || !transport->healthy() ||
            request.return_req.device >= backends.size() ||
            request.return_req.phone_ffn_seq == 0 ||
            request.wire_size == 0) {
            return false;
        }

        std::lock_guard<std::mutex> lock(
            inference_return_mutex);
        auto & by_seq =
            inference_return_bindings[
                request.return_req.device];
        if (by_seq.find(
                request.return_req.phone_ffn_seq) !=
                by_seq.end()) {
            return false;
        }

        inference_return_binding binding;
        binding.request = request;
        by_seq.emplace(
            request.return_req.phone_ffn_seq,
            std::move(binding));
        inference_return_cv.notify_all();
        return true;
    }

    void fail_inference_returns() {
        {
            std::lock_guard<std::mutex> lock(
                inference_return_mutex);
            inference_return_bindings.clear();
        }
        inference_return_cv.notify_all();
    }

    bool handle_inference_frame(
        const inference_frame_header & header,
        const uint8_t * data,
        size_t size) {
        // REAL C2P frames use the existing sole server-side RX dispatcher.
        // The control-only APPLY later consumes the HOST-ready frame and
        // uploads it inside the Phone RPC server, not through the PC.
        if (header.type == INFERENCE_FRAME_DATA &&
            (header.stage == INFERENCE_STAGE_CYCLE_READY ||
             header.stage == INFERENCE_STAGE_CYCLE_PARTIAL)) {
            return header.tensor_role == INFERENCE_CYCLE_PC_TO_PHONE &&
                receive_c2p_inference_frame(header, data, size);
        }
        // Dedicated Cycle ACKs travel only as tiny control frames in the
        // reverse direction. The PC sends one after READY is usable and
        // another after PARTIAL has been consumed by the PC Reduce.
        // Never try to parse them as legacy INFERENCE_STAGE_RETURN data.
        if (header.type == INFERENCE_FRAME_ACK &&
            (header.stage == INFERENCE_STAGE_CYCLE_READY ||
             header.stage == INFERENCE_STAGE_CYCLE_PARTIAL)) {
            inference_cycle_ack ack {};
            const bool valid =
                inference_cycle_decode_ack(header, size, ack) &&
                ack.direction == INFERENCE_CYCLE_PHONE_TO_PC &&
                ack.device_ready;
            GGML_LOG_INFO(
                "[CYCLE_REAL_WIRE_ACK] side=phone kind=%s "
                "layer=%d chunk=%d seq=%" PRIu64 " "
                "host_ready=%d status=%s\n",
                header.stage == INFERENCE_STAGE_CYCLE_READY ?
                    "READY" : "PARTIAL",
                header.layer, header.chunk, header.seq,
                valid ? 1 : 0, valid ? "RECEIVED" : "FAIL");
            return valid;
        }
        if (header.type != INFERENCE_FRAME_DATA ||
            header.stage != INFERENCE_STAGE_RETURN) {
            GGML_LOG_ERROR(
                "[INFERENCE_DATA_PLANE] side=server "
                "status=FAIL reason=unsupported_frame "
                "type=%u stage=%u seq=%" PRIu64 "\n",
                static_cast<unsigned>(header.type),
                static_cast<unsigned>(header.stage),
                header.seq);
            return false;
        }

        const uint32_t device = header.stream_id;
        rpc_msg_inference_return_bind_req binding_request {};

        {
            std::unique_lock<std::mutex> lock(
                inference_return_mutex);
            inference_return_cv.wait(
                lock,
                [&]() {
                    auto device_it =
                        inference_return_bindings.find(device);
                    return device_it !=
                            inference_return_bindings.end() &&
                        device_it->second.find(header.seq) !=
                            device_it->second.end();
                });

            auto & binding =
                inference_return_bindings[device][header.seq];
            binding_request = binding.request;
        }

        bool success = true;
        if (binding_request.return_req.phone_ffn_seq !=
                header.seq ||
            binding_request.graph_id != header.graph_id ||
            binding_request.wire_size != size) {
            success = false;
            GGML_LOG_ERROR(
                "[INFERENCE_RETURN] side=server "
                "status=FAIL reason=metadata_mismatch "
                "device=%u seq=%" PRIu64
                " graph=%" PRIu64 "/%" PRIu64
                " bytes=%zu/%" PRIu64 "\n",
                device,
                header.seq,
                header.graph_id,
                binding_request.graph_id,
                size,
                binding_request.wire_size);
        }

        if (success) {
            std::vector<uint8_t> input(
                sizeof(
                    rpc_msg_set_tensor_async_return_wait_req) +
                size);
            std::memcpy(
                input.data(),
                &binding_request.return_req,
                sizeof(
                    rpc_msg_set_tensor_async_return_wait_req));
            if (size > 0) {
                std::memcpy(
                    input.data() +
                        sizeof(
                            rpc_msg_set_tensor_async_return_wait_req),
                    data,
                    size);
            }

            success =
                set_tensor_async_return_wait(input);
        }

        {
            std::lock_guard<std::mutex> lock(
                inference_return_mutex);
            auto device_it =
                inference_return_bindings.find(device);
            if (device_it !=
                    inference_return_bindings.end()) {
                device_it->second.erase(header.seq);
                if (device_it->second.empty()) {
                    inference_return_bindings.erase(device_it);
                }
            }
        }
        inference_return_cv.notify_all();

        int64_t ack_submit_us = 0;
        if (success) {
            auto transport = get_inference_data_plane();
            const int64_t ack_begin_us = ggml_time_us();
            inference_tx_buffer ack_tx =
                transport ?
                    transport->acquire_tx(0) :
                    inference_tx_buffer {};
            success =
                transport != nullptr &&
                transport->healthy() &&
                static_cast<bool>(ack_tx);

            if (success) {
                inference_frame_header ack =
                    inference_make_header(
                        INFERENCE_FRAME_ACK);
                ack.stage = INFERENCE_STAGE_RETURN;
                ack.stream_id = device;
                ack.seq = header.seq;
                ack.graph_id = header.graph_id;
                ack.layer = binding_request.layer;
                ack.chunk = binding_request.chunk;

                success = transport->submit_tx(
                    std::move(ack_tx),
                    ack,
                    0);
            } else if (ack_tx && transport) {
                transport->release_tx(
                    std::move(ack_tx));
            }
            ack_submit_us =
                ggml_time_us() - ack_begin_us;
        }

        GGML_LOG_INFO(
            "[INFERENCE_RETURN] side=server "
            "device=%u seq=%" PRIu64
            " layer=%d chunk=%d bytes=%zu "
            "ack_submit_ms=%.3f status=%s\n",
            device,
            header.seq,
            binding_request.layer,
            binding_request.chunk,
            size,
            ack_submit_us / 1000.0,
            success ? "OK" : "FAIL");

        return success;
    }

    bool arm_inference_route(
        const rpc_msg_inference_route_arm_req & request) {
        if (request.snapshot.device >= backends.size() ||
                request.snapshot.lane >= RPC_ROUTE_TRANSFER_LANES ||
                request.snapshot.seq == 0) {
            return false;
        }

        auto transport = get_inference_data_plane();
        if (!transport || !transport->healthy()) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(
                inference_route_mutex);

            auto & by_seq =
                inference_route_arms[
                    request.snapshot.device];
            if (by_seq.find(request.snapshot.seq) !=
                    by_seq.end()) {
                return false;
            }

            inference_route_arm_entry entry;
            entry.request = request;
            entry.arm_us = ggml_time_us();
            by_seq.emplace(
                request.snapshot.seq,
                std::move(entry));

            if (!inference_route_thread.joinable()) {
                inference_route_stop = false;
                inference_route_thread =
                    std::thread(
                        &rpc_server::inference_route_sender_loop,
                        this);
            }
        }

        // Normally ARM arrives before the producer graph. Handle the inverse
        // race as well: if the immutable snapshot is already READY, queue it
        // immediately. Exact-once removal from inference_route_arms prevents
        // duplicate jobs when READY and ARM race each other.
        return queue_inference_route_if_ready(
            request.snapshot.device,
            request.snapshot.lane,
            request.snapshot.seq);
    }

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };
private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    bool is_opencl_tensor(const ggml_tensor * tensor) const;
    rpc_opencl_tensor_extra_key opencl_tensor_extra_key(const ggml_tensor * tensor) const;
    rpc_opencl_tensor_storage_key opencl_tensor_storage_key(const ggml_tensor * tensor) const;
    bool restore_opencl_tensor_extra(ggml_tensor * tensor);
    bool ensure_opencl_tensor_extra(
        ggml_tensor * tensor,
        bool allow_quantized_weight_init = true);
    void remember_opencl_tensor_extra(ggml_tensor * tensor);
    void erase_opencl_tensor_extras_for_buffer(ggml_backend_buffer_t buffer);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);
    bool prepare_phone_npu_ffn_cache(
        ggml_cgraph * graph,
        const rpc_phone_npu_ffn_stage & stage,
        rpc_phone_npu_ffn_cache *& cache);
    bool run_phone_npu_ffn_stage(
        ggml_cgraph * graph,
        const rpc_phone_npu_ffn_stage & stage,
        std::vector<float> * host_output);
    bool prepare_phone_npu_async_job(
        uint32_t device,
        uint64_t seq,
        ggml_cgraph * graph,
        const rpc_phone_npu_ffn_stage & stage,
        rpc_phone_npu_async_job & job);
    bool execute_phone_npu_async_job(
        const rpc_phone_npu_async_job & job,
        std::vector<float> & output);
    void ensure_phone_npu_async_worker();
    void phone_npu_async_loop();
    ggml_status graph_compute_phone_npu_ffn(
        uint32_t device,
        uint64_t graph_uid,
        ggml_cgraph * graph);

    bool queue_inference_route_if_ready(
        uint32_t device,
        uint32_t lane,
        uint64_t seq);
    void inference_route_sender_loop();

    std::vector<ggml_backend_t> backends;
    // Private phone-local execution backend. It is deliberately not part of
    // the RPC-visible backend vector and therefore cannot become an Outer
    // Meta tensor-split participant.
    ggml_backend_t phone_npu_backend = nullptr;
    std::mutex phone_npu_mutex;
    std::unordered_map<
        int,
        std::unique_ptr<rpc_phone_npu_ffn_cache>>
        phone_npu_ffn_caches;

    // Chunk Ring CPU-reduce ownership.  "armed" seqs are announced by Meta
    // before a Phone FFN graph is submitted.  A completed private-HTP partial
    // is then bound to that seq and kept in host memory until the matching PC
    // partial arrives on SET_TENSOR_ASYNC_RETURN_WAIT.
    std::mutex phone_npu_cpu_reduce_mutex;
    std::unordered_map<uint32_t, std::deque<uint64_t>>
        phone_npu_cpu_reduce_armed;
    std::unordered_map<
        uint32_t,
        std::unordered_map<uint64_t, rpc_phone_npu_cpu_reduce_entry>>
        phone_npu_cpu_reduce_pending;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>>
        phone_npu_cpu_reduce_active;
    // One ordinary post-graph PHONE_FFN_MARK_READY is expected for every
    // armed Ring seq.  Track that marker independently from active ownership
    // so a very fast NPU+PC completion cannot make a late marker stale-ready.
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>>
        phone_npu_cpu_reduce_marker_pending;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>>
        phone_npu_cpu_reduce_failed;

    // One private HTP execution lane.  GPU front graphs only freeze their
    // inputs and enqueue here; this worker serializes HTP jobs while the RPC
    // compute socket is free to submit later Attention chunks.
    std::mutex phone_npu_async_mutex;
    std::condition_variable phone_npu_async_cv;
    std::deque<rpc_phone_npu_async_job> phone_npu_async_jobs;
    bool phone_npu_async_stop = false;
    std::thread phone_npu_async_thread;

    const char * cache_dir;
    std::unordered_set<ggml_backend_buffer_t> buffers;

    // Non-owning OpenCL-only tensor state. Entries are removed before the
    // corresponding backend buffer is freed.
    std::unordered_map<
        rpc_opencl_tensor_extra_key,
        void *,
        rpc_opencl_tensor_extra_key_hash> opencl_tensor_extras;
    std::unordered_map<
        rpc_opencl_tensor_storage_key,
        rpc_opencl_tensor_storage_entry,
        rpc_opencl_tensor_storage_key_hash> opencl_tensor_storage_extras;

    // Tracks synthetic profiling tensors that have actually run through the
    // OpenCL quantized set_tensor path. Do not infer this from the ordinary
    // extra cache: INIT_TENSOR can legitimately cache only the generic
    // ggml_tensor_extra_cl object before SoA conversion has happened.
    std::unordered_set<
        rpc_opencl_tensor_storage_key,
        rpc_opencl_tensor_storage_key_hash> opencl_profile_zero_initialized;

    std::mutex opencl_tensor_extras_mutex;

    // store computed graphs for each backend by graph uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;

    std::mutex route_ready_mutex;
    std::condition_variable route_ready_cv;
    std::unordered_map<uint32_t, uint64_t> route_ready_seq;

    std::mutex phone_ffn_ready_mutex;
    std::condition_variable phone_ffn_ready_cv;
    std::unordered_map<
        uint32_t,
        std::unordered_set<uint64_t>>
        phone_ffn_ready_seqs;

    std::vector<std::unique_ptr<rpc_snapshot_device>> snapshot_devices;
    std::vector<std::unique_ptr<rpc_route_snapshot_device>>
        route_snapshot_devices;

    // Phone->PC Route wire admission. v19 keeps this independent from Return:
    // one Route may overlap one Return, but a second Route waits for the first
    // PC full-receive ACK.
    std::mutex route_bulk_mutex;
    std::condition_variable route_bulk_cv;
    size_t route_bulk_inflight = 0;
    size_t route_bulk_limit = 1;

    // Legacy v18 shared-credit state. New v19 clients do not use it.
    bool network_bulk_shared = false;

    // V20 inference data plane. Control RPC keeps its request/response socket;
    // one separate connection is upgraded after HELLO and then owned solely by
    // inference_transport TX/RX workers.
    std::mutex inference_data_plane_mutex;
    std::shared_ptr<inference_transport> inference_data_plane;

    struct inference_route_arm_entry {
        rpc_msg_inference_route_arm_req request {};
        int64_t arm_us = 0;
    };
    struct inference_route_send_job {
        rpc_msg_inference_route_arm_req request {};
        int64_t arm_us = 0;
        int64_t ready_us = 0;
    };
    std::mutex inference_route_mutex;
    std::condition_variable inference_route_cv;
    std::unordered_map<
        uint32_t,
        std::unordered_map<uint64_t, inference_route_arm_entry>>
        inference_route_arms;
    std::deque<inference_route_send_job>
        inference_route_jobs;
    bool inference_route_stop = false;
    std::thread inference_route_thread;

    struct inference_return_binding {
        rpc_msg_inference_return_bind_req request {};
    };
    std::mutex inference_return_mutex;
    std::condition_variable inference_return_cv;
    std::unordered_map<
        uint32_t,
        std::unordered_map<uint64_t, inference_return_binding>>
        inference_return_bindings;

    // A bounded HOST-ready mailbox owned by the Phone server's one RX
    // dispatcher. READY/PARTIAL move to GPU only through control APPLY.
    // An applied PARTIAL is acknowledged only after GPU Reduce completion.
    std::mutex cycle_rx_mutex;
    std::condition_variable cycle_rx_cv;
    std::map<cycle_rx_key, inference_cycle_message> cycle_rx_frames;
    std::set<cycle_rx_key> cycle_rx_applied_partial;
    bool cycle_rx_failed = false;

    std::mutex async_return_payload_mutex;
    std::unordered_map<
        ggml_backend_dev_t,
        std::vector<std::shared_ptr<std::vector<uint8_t>>>>
        async_return_payloads;

    std::array<rpc_snapshot_breakdown, 2> snapshot_breakdown {};
    std::mutex snapshot_breakdown_mutex;
    //新加的
    ggml_rpc_local_tensor_source local_tensor_source {};
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::is_opencl_tensor(const ggml_tensor * tensor) const {
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->buffer->buft == nullptr) {
        return false;
    }

    const char * name = ggml_backend_buft_name(tensor->buffer->buft);
    return name != nullptr && std::strcmp(name, "OpenCL") == 0;
}

rpc_opencl_tensor_extra_key rpc_server::opencl_tensor_extra_key(
        const ggml_tensor * tensor) const {
    rpc_opencl_tensor_extra_key key;
    key.buffer = tensor->buffer;
    key.data = reinterpret_cast<uint64_t>(tensor->data);
    key.type = tensor->type;
    for (size_t i = 0; i < GGML_MAX_DIMS; ++i) {
        key.ne[i] = tensor->ne[i];
        key.nb[i] = tensor->nb[i];
    }
    key.name = tensor->name;
    return key;
}

rpc_opencl_tensor_storage_key rpc_server::opencl_tensor_storage_key(
        const ggml_tensor * tensor) const {
    rpc_opencl_tensor_storage_key key;
    key.buffer = tensor->buffer;
    key.data = reinterpret_cast<uint64_t>(tensor->data);
    key.type = tensor->type;
    return key;
}

bool rpc_server::restore_opencl_tensor_extra(ggml_tensor * tensor) {
    if (!is_opencl_tensor(tensor)) {
        return false;
    }

    const rpc_opencl_tensor_extra_key exact_key =
        opencl_tensor_extra_key(tensor);
    const rpc_opencl_tensor_storage_key storage_key =
        opencl_tensor_storage_key(tensor);

    std::lock_guard<std::mutex> lock(opencl_tensor_extras_mutex);

    const auto exact_it = opencl_tensor_extras.find(exact_key);
    if (exact_it != opencl_tensor_extras.end()) {
        tensor->extra = exact_it->second;
        RPC_OPENCL_EXTRA_DBG(
            "[RPC_OPENCL_EXTRA] action=RESTORE_EXACT name=%s buffer=%p data=%p extra=%p\n",
            tensor->name, (void *) tensor->buffer, tensor->data, tensor->extra);
        return true;
    }

    const auto storage_it =
        opencl_tensor_storage_extras.find(storage_key);
    if (storage_it == opencl_tensor_storage_extras.end()) {
        // Report the nearest cached storage entry with the same buffer/type.
        // This is intentionally diagnostics-only and remains OpenCL-local.
        bool have_nearest = false;
        uint64_t nearest_data = 0;
        uint64_t nearest_delta = UINT64_MAX;
        void * nearest_extra = nullptr;
        bool nearest_ambiguous = false;
        size_t same_buffer_type = 0;

        for (const auto & entry : opencl_tensor_storage_extras) {
            if (entry.first.buffer != storage_key.buffer ||
                entry.first.type != storage_key.type) {
                continue;
            }

            ++same_buffer_type;
            const uint64_t candidate_data = entry.first.data;
            const uint64_t delta =
                candidate_data >= storage_key.data ?
                    candidate_data - storage_key.data :
                    storage_key.data - candidate_data;

            if (!have_nearest || delta < nearest_delta) {
                have_nearest = true;
                nearest_data = candidate_data;
                nearest_delta = delta;
                nearest_extra = entry.second.extra;
                nearest_ambiguous = entry.second.ambiguous;
            }
        }

        if (have_nearest) {
            RPC_OPENCL_EXTRA_DBG(
                "[RPC_OPENCL_EXTRA] action=RESTORE_STORAGE_MISS "
                "name=%s buffer=%p data=0x%" PRIx64 " type=%d "
                "same_buffer_type=%zu nearest_data=0x%" PRIx64
                " delta=%" PRIu64 " nearest_extra=%p ambiguous=%d\n",
                tensor->name,
                (void *) tensor->buffer,
                storage_key.data,
                (int) tensor->type,
                same_buffer_type,
                nearest_data,
                nearest_delta,
                nearest_extra,
                nearest_ambiguous ? 1 : 0);
        } else {
            RPC_OPENCL_EXTRA_DBG(
                "[RPC_OPENCL_EXTRA] action=RESTORE_STORAGE_MISS "
                "name=%s buffer=%p data=0x%" PRIx64 " type=%d "
                "same_buffer_type=0\n",
                tensor->name,
                (void *) tensor->buffer,
                storage_key.data,
                (int) tensor->type);
        }
        return false;
    }

    if (storage_it->second.ambiguous ||
        storage_it->second.extra == nullptr) {
        RPC_OPENCL_EXTRA_DBG(
            "[RPC_OPENCL_EXTRA] action=RESTORE_STORAGE_CONFLICT "
            "name=%s buffer=%p data=%p type=%d\n",
            tensor->name,
            (void *) tensor->buffer,
            tensor->data,
            (int) tensor->type);
        return false;
    }

    tensor->extra = storage_it->second.extra;

    // Remember the reconstructed metadata variant as an exact alias. Future
    // reconstructions then take the strict path without another fallback.
    opencl_tensor_extras.emplace(exact_key, tensor->extra);

    RPC_OPENCL_EXTRA_DBG(
        "[RPC_OPENCL_EXTRA] action=RESTORE_STORAGE name=%s "
        "buffer=%p data=%p type=%d extra=%p\n",
        tensor->name,
        (void *) tensor->buffer,
        tensor->data,
        (int) tensor->type,
        tensor->extra);
    return true;
}

bool rpc_server::ensure_opencl_tensor_extra(
        ggml_tensor * tensor,
        bool allow_quantized_weight_init) {
    // Deliberately leave every non-OpenCL backend on the pre-existing RPC path.
    if (!is_opencl_tensor(tensor)) {
        return true;
    }

    // Meta may retain empty shards in RPC graphs. They have no uploaded
    // weights or OpenCL extra, and OpenCL skips empty nodes at execution.
    if (ggml_is_empty(tensor)) {
        if (rpc_opencl_extra_debug_enabled()) {
            RPC_OPENCL_EXTRA_DBG(
                "[RPC_OPENCL_EXTRA] action=SKIP_EMPTY name=%s type=%d\n",
                tensor->name, (int) tensor->type);
        }
        return true;
    }

    if (tensor->extra != nullptr || restore_opencl_tensor_extra(tensor)) {
        return true;
    }

    // During graph reconstruction, K-quant leaf weights must recover the
    // type-specific OpenCL extra produced by set_tensor. Scheduler/RPC graph
    // reconstruction may rename them to "leaf_N", so model-name matching is
    // not reliable here. A non-view GGML_OP_NONE K-quant tensor is a leaf
    // storage tensor; letting it fall back to generic ggml_tensor_extra_cl
    // would later reinterpret offset/size bytes as cl_mem handles.
    const bool requires_k_quant_extra =
        tensor->view_src == nullptr &&
        tensor->op == GGML_OP_NONE &&
        (tensor->type == GGML_TYPE_Q4_K ||
         tensor->type == GGML_TYPE_Q5_K ||
         tensor->type == GGML_TYPE_Q6_K);

    if (!allow_quantized_weight_init && requires_k_quant_extra) {
        RPC_OPENCL_EXTRA_DBG(
            "[RPC_OPENCL_EXTRA] action=RESTORE_MISS_K_QUANT_LEAF "
            "name=%s buffer=%p data=%p type=%d ne=[%" PRId64 ",%" PRId64
            ",%" PRId64 ",%" PRId64 "]\n",
            tensor->name,
            (void *) tensor->buffer,
            tensor->data,
            (int) tensor->type,
            tensor->ne[0],
            tensor->ne[1],
            tensor->ne[2],
            tensor->ne[3]);
        return false;
    }

    const ggml_status status =
        ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
    if (status != GGML_STATUS_SUCCESS || tensor->extra == nullptr) {
        RPC_OPENCL_EXTRA_DBG(
            "[RPC_OPENCL_EXTRA] action=INIT_FAILED name=%s status=%d extra=%p\n",
            tensor->name, (int) status, tensor->extra);
        return false;
    }

    RPC_OPENCL_EXTRA_DBG(
        "[RPC_OPENCL_EXTRA] action=INIT name=%s buffer=%p data=%p extra=%p\n",
        tensor->name, (void *) tensor->buffer, tensor->data, tensor->extra);
    return true;
}

void rpc_server::remember_opencl_tensor_extra(ggml_tensor * tensor) {
    if (!is_opencl_tensor(tensor) || tensor->extra == nullptr) {
        return;
    }

    const rpc_opencl_tensor_extra_key exact_key =
        opencl_tensor_extra_key(tensor);
    const rpc_opencl_tensor_storage_key storage_key =
        opencl_tensor_storage_key(tensor);

    bool storage_conflict = false;
    {
        std::lock_guard<std::mutex> lock(opencl_tensor_extras_mutex);

        void * previous_exact = nullptr;
        bool had_exact = false;

        auto exact_it = opencl_tensor_extras.find(exact_key);
        if (exact_it != opencl_tensor_extras.end()) {
            had_exact = true;
            previous_exact = exact_it->second;
            exact_it->second = tensor->extra;
        } else {
            opencl_tensor_extras.emplace(exact_key, tensor->extra);
        }

        auto storage_it =
            opencl_tensor_storage_extras.find(storage_key);
        if (storage_it == opencl_tensor_storage_extras.end()) {
            opencl_tensor_storage_extras.emplace(
                storage_key,
                rpc_opencl_tensor_storage_entry {
                    /* .extra = */ tensor->extra,
                    /* .ambiguous = */ false,
                });
        } else if (!storage_it->second.ambiguous &&
                   storage_it->second.extra != tensor->extra) {
            // The same exact tensor can legitimately transition from the
            // generic allocation extra to a type-specific quantized extra
            // during set_tensor. Treat that as replacement, not ambiguity.
            if (had_exact &&
                storage_it->second.extra == previous_exact) {
                storage_it->second.extra = tensor->extra;
            } else {
                storage_it->second.extra = nullptr;
                storage_it->second.ambiguous = true;
                storage_conflict = true;
            }
        }
    }

    if (storage_conflict) {
        RPC_OPENCL_EXTRA_DBG(
            "[RPC_OPENCL_EXTRA] action=REMEMBER_STORAGE_CONFLICT "
            "name=%s buffer=%p data=%p type=%d extra=%p\n",
            tensor->name,
            (void *) tensor->buffer,
            tensor->data,
            (int) tensor->type,
            tensor->extra);
    } else {
        RPC_OPENCL_EXTRA_DBG(
            "[RPC_OPENCL_EXTRA] action=REMEMBER name=%s "
            "buffer=%p data=%p type=%d extra=%p\n",
            tensor->name,
            (void *) tensor->buffer,
            tensor->data,
            (int) tensor->type,
            tensor->extra);
    }
}

void rpc_server::erase_opencl_tensor_extras_for_buffer(
        ggml_backend_buffer_t buffer) {
    std::lock_guard<std::mutex> lock(opencl_tensor_extras_mutex);
    for (auto it = opencl_tensor_extras.begin();
         it != opencl_tensor_extras.end();) {
        if (it->first.buffer == buffer) {
            it = opencl_tensor_extras.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = opencl_tensor_storage_extras.begin();
         it != opencl_tensor_storage_extras.end();) {
        if (it->first.buffer == buffer) {
            it = opencl_tensor_storage_extras.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = opencl_profile_zero_initialized.begin();
         it != opencl_profile_zero_initialized.end();) {
        if (it->buffer == buffer) {
            it = opencl_profile_zero_initialized.erase(it);
        } else {
            ++it;
        }
    }
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    const bool cycle_ffn_diag = std::getenv("GGML_CYCLE_LIVE_FFN_DIAG") != nullptr;
    if (cycle_ffn_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=GET_ALLOC_SIZE stage=BEGIN "
            "device=%u tensor=%.*s type=%u op=%u ne=[%" PRIu32 ",%" PRIu32 ",%" PRIu32 "]\n",
            request.device, (int) GGML_MAX_NAME, request.tensor.name,
            (unsigned) request.tensor.type, (unsigned) request.tensor.op,
            request.tensor.ne[0], request.tensor.ne[1], request.tensor.ne[2]);
    }
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    if (cycle_ffn_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=GET_ALLOC_SIZE stage=BACKEND_BEGIN "
            "tensor=%s op=%s type=%s buft=%s\n",
            tensor->name, ggml_op_name(tensor->op), ggml_type_name(tensor->type),
            ggml_backend_buft_name(buft));
    }
    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
    if (cycle_ffn_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=GET_ALLOC_SIZE stage=END "
            "tensor=%s bytes=%" PRIu64 "\n",
            tensor->name, response.alloc_size);
    }

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        GGML_LOG_INFO("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n", __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        //LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    erase_opencl_tensor_extras_for_buffer(buffer);
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}
// 把收到的rpc_tensor反序列化为ggml_tensor结构体
ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);
    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);

    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
    }

    return set_tensor_direct(*in_tensor, offset, data, size);
}

bool rpc_server::set_tensor_direct(
        const rpc_tensor & in_tensor,
        uint64_t offset,
        const void * data,
        size_t size) {

    const bool cycle_ffn_diag = std::getenv("GGML_CYCLE_LIVE_FFN_DIAG") != nullptr;
    if (cycle_ffn_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=SET_TENSOR stage=BEGIN "
            "tensor=%.*s type=%u offset=%" PRIu64 " bytes=%zu\n",
            (int) GGML_MAX_NAME, in_tensor.name, (unsigned) in_tensor.type,
            offset, size);
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    //打印张量信息，包括名字
    if (RPC_DEBUG) {
        printf(
            "[RPC_SET_DIRECT] offset=%" PRIu64 " size=%zu name=%s\n",
            offset,
            size,
            in_tensor.name);
    }

    const bool debug_k_quant =
        rpc_opencl_extra_debug_enabled() &&
        is_opencl_tensor(tensor) &&
        rpc_is_k_quant_type(tensor->type);

    if (debug_k_quant) {
        GGML_LOG_DEBUG(
            "[RPC_KQUANT_SERVER_SET_BEGIN] name=%s type=%d "
            "buffer=%p data=%p offset=%" PRIu64 " size=%zu extra=%p\n",
            tensor->name,
            (int) tensor->type,
            (void *) tensor->buffer,
            tensor->data,
            offset,
            size,
            tensor->extra);
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor.data + offset < p0 || in_tensor.data + offset >= p1 || size > (p1 - in_tensor.data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor.data, offset, size, p0, p1);
            return false;
        }
    }

    if (cycle_ffn_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=SET_TENSOR stage=INIT_BEGIN "
            "tensor=%s type=%s\n",
            tensor->name, ggml_type_name(tensor->type));
    }
    if (!ensure_opencl_tensor_extra(tensor)) {
        GGML_LOG_ERROR(
            "[%s] failed to initialize OpenCL tensor state for '%s'\n",
            __func__, tensor->name);
        return false;
    }
    if (cycle_ffn_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=SET_TENSOR stage=INIT_END "
            "tensor=%s extra=%p\n",
            tensor->name, tensor->extra);
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=SET_TENSOR stage=WRITE_BEGIN "
            "tensor=%s bytes=%zu\n",
            tensor->name, size);
    }

    ggml_backend_tensor_set(tensor, data, offset, size);
    if (cycle_ffn_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=SET_TENSOR stage=WRITE_END "
            "tensor=%s\n",
            tensor->name);
    }

    if (debug_k_quant) {
        GGML_LOG_DEBUG(
            "[RPC_KQUANT_SERVER_SET_AFTER_BACKEND] name=%s type=%d "
            "buffer=%p data=%p extra=%p\n",
            tensor->name,
            (int) tensor->type,
            (void *) tensor->buffer,
            tensor->data,
            tensor->extra);
    }

    // OpenCL quantized set_tensor may replace the generic extra with a
    // type-specific object. Preserve the post-write pointer.
    remember_opencl_tensor_extra(tensor);
    if (cycle_ffn_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_RPC] side=phone cmd=SET_TENSOR stage=END tensor=%s\n",
            tensor->name);
    }

    if (debug_k_quant) {
        GGML_LOG_DEBUG(
            "[RPC_KQUANT_SERVER_SET_REMEMBERED] name=%s type=%d "
            "buffer=%p data=%p extra=%p\n",
            tensor->name,
            (int) tensor->type,
            (void *) tensor->buffer,
            tensor->data,
            tensor->extra);
    }
    return true;
}

bool rpc_server::set_tensor_direct_opencl_async(
        const rpc_tensor & in_tensor,
        uint64_t offset,
        const void * data,
        size_t size) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);

    ggml_tensor * tensor =
        deserialize_tensor(ctx_ptr.get(), &in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr ||
            !is_opencl_tensor(tensor)) {
        return false;
    }

    const size_t p0 =
        (size_t) ggml_backend_buffer_get_base(tensor->buffer);
    const size_t p1 =
        p0 + ggml_backend_buffer_get_size(tensor->buffer);
    if (in_tensor.data + offset < p0 ||
            in_tensor.data + offset >= p1 ||
            size > (p1 - in_tensor.data - offset)) {
        return false;
    }

    if (!ensure_opencl_tensor_extra(tensor)) {
        return false;
    }

    ggml_backend_dev_t dev =
        ggml_backend_buft_get_device(tensor->buffer->buft);
    ggml_backend_reg_t reg =
        dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    const ggml_backend_opencl_set_tensor_async_t set_async =
        reg != nullptr ?
            reinterpret_cast<ggml_backend_opencl_set_tensor_async_t>(
                ggml_backend_reg_get_proc_address(
                    reg,
                    GGML_BACKEND_OPENCL_SET_TENSOR_ASYNC_PROC)) :
            nullptr;

    if (set_async == nullptr ||
            !set_async(
                tensor,
                data,
                static_cast<size_t>(offset),
                size)) {
        return false;
    }

    remember_opencl_tensor_extra(tensor);
    return true;
}

bool rpc_server::set_tensor_async_return(
        const std::vector<uint8_t> & input) {
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }

    rpc_tensor in_tensor {};
    memcpy(&in_tensor, input.data(), sizeof(in_tensor));
    uint64_t offset = 0;
    memcpy(
        &offset,
        input.data() + sizeof(in_tensor),
        sizeof(offset));
    const size_t data_size =
        input.size() - sizeof(in_tensor) - sizeof(offset);
    const void * data =
        input.data() + sizeof(in_tensor) + sizeof(offset);

    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const int64_t begin_us =
        stage_profile ? ggml_time_us() : 0;
    const int64_t write_begin_us =
        stage_profile ? ggml_time_us() : 0;

    // CL_FALSE writes require the host source pointer to remain valid until
    // the write completes. The old path pointed directly into this RPC
    // request vector, which is destroyed when the handler returns. Keep an
    // owned copy alive on the server until a synchronize on this device proves
    // every previously-enqueued write has completed.
    bool async_write = false;
    bool status = false;
    std::shared_ptr<std::vector<uint8_t>> owned_payload;
    ggml_backend_dev_t payload_dev = nullptr;

    ggml_backend_buffer_t payload_buffer =
        reinterpret_cast<ggml_backend_buffer_t>(in_tensor.buffer);
    if (payload_buffer != nullptr &&
            buffers.find(payload_buffer) != buffers.end() &&
            payload_buffer->buft != nullptr) {
        payload_dev =
            ggml_backend_buft_get_device(payload_buffer->buft);
    }

    if (payload_dev != nullptr) {
        owned_payload =
            std::make_shared<std::vector<uint8_t>>(data_size);
        if (data_size > 0) {
            memcpy(
                owned_payload->data(),
                data,
                data_size);
        }

        std::lock_guard<std::mutex> lock(
            async_return_payload_mutex);
        async_write = set_tensor_direct_opencl_async(
            in_tensor,
            offset,
            owned_payload->data(),
            data_size);
        if (async_write) {
            async_return_payloads[payload_dev].
                push_back(owned_payload);
            status = true;
        }
    }

    if (!status) {
        status = set_tensor_direct(
            in_tensor, offset, data, data_size);
        owned_payload.reset();
    }

    const int64_t write_us =
        stage_profile ? ggml_time_us() - write_begin_us : 0;
    if (stage_profile) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=phone stage=return_stage_write "
            "bytes=%zu async_write=%d write_ms=%.3f total_ms=%.3f status=%d\n",
            data_size,
            async_write ? 1 : 0,
            write_us / 1000.0,
            (ggml_time_us() - begin_us) / 1000.0,
            status ? 1 : 0);
    }

    return status;
}

bool rpc_server::phone_ffn_mark_ready(
        const rpc_msg_phone_ffn_mark_ready_req & request) {
    if (request.device >= backends.size() || request.seq == 0) {
        return false;
    }

    if ((request.seq & RPC_PHONE_FFN_CPU_REDUCE_ARM) != 0) {
        const uint64_t seq =
            request.seq & ~RPC_PHONE_FFN_CPU_REDUCE_ARM;
        if (seq == 0) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(
                phone_npu_cpu_reduce_mutex);
            auto & active =
                phone_npu_cpu_reduce_active[request.device];
            if (!active.insert(seq).second) {
                GGML_LOG_ERROR(
                    "[CHUNK_RING_CPU_ARM] device=%u seq=%" PRIu64
                    " status=FAIL reason=duplicate_seq\n",
                    request.device,
                    seq);
                return false;
            }
            phone_npu_cpu_reduce_marker_pending[
                request.device].insert(seq);
            phone_npu_cpu_reduce_armed[request.device].
                push_back(seq);
        }

        GGML_LOG_INFO(
            "[CHUNK_RING_CPU_ARM] device=%u seq=%" PRIu64 "\n",
            request.device,
            seq);
        return true;
    }

    bool ring_graph_marker = false;
    {
        std::lock_guard<std::mutex> join_lock(
            phone_npu_cpu_reduce_mutex);
        auto marker_it =
            phone_npu_cpu_reduce_marker_pending.find(
                request.device);
        if (marker_it !=
                phone_npu_cpu_reduce_marker_pending.end()) {
            ring_graph_marker =
                marker_it->second.erase(request.seq) != 0;
            if (marker_it->second.empty()) {
                phone_npu_cpu_reduce_marker_pending.erase(
                    marker_it);
            }
        }
    }

    if (ring_graph_marker) {
        const bool ring_npu_async =
            std::getenv("GGML_RPC_DISABLE_CHUNK_RING_NPU_ASYNC") == nullptr;

        if (!ring_npu_async) {
            {
                std::lock_guard<std::mutex> lock(phone_ffn_ready_mutex);
                phone_ffn_ready_seqs[request.device].insert(request.seq);
            }
            phone_ffn_ready_cv.notify_all();
        }

        if (rpc_tensor_phone_stage_profile_enabled()) {
            GGML_LOG_INFO(
                "[TENSOR_PHONE_RPC_STAGE] side=phone "
                "stage=phone_ffn_graph_returned device=%u "
                "seq=%" PRIu64 " ring_async=%d\n",
                request.device,
                request.seq,
                ring_npu_async ? 1 : 0);
        }
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(phone_ffn_ready_mutex);
        phone_ffn_ready_seqs[request.device].insert(request.seq);
    }
    phone_ffn_ready_cv.notify_all();

    if (rpc_tensor_phone_stage_profile_enabled()) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=phone "
            "stage=phone_ffn_chunk_ready device=%u seq=%" PRIu64 "\n",
            request.device,
            request.seq);
    }
    return true;
}

bool rpc_server::set_tensor_async_return_wait(
        const std::vector<uint8_t> & input) {
    if (input.size() <
            sizeof(rpc_msg_set_tensor_async_return_wait_req_v14)) {
        return false;
    }

    rpc_msg_set_tensor_async_return_wait_req_v14 request {};
    memcpy(&request, input.data(), sizeof(request));

    bool extended = false;
    uint32_t wire_type = RPC_RETURN_WIRE_RAW;
    size_t header_size = sizeof(request);
    size_t original_size =
        input.size() - header_size;

    if (input.size() >=
            sizeof(rpc_msg_set_tensor_async_return_wait_req)) {
        rpc_msg_set_tensor_async_return_wait_req request_v15 {};
        memcpy(
            &request_v15,
            input.data(),
            sizeof(request_v15));
        if (request_v15.wire_magic ==
                RPC_RETURN_WIRE_MAGIC &&
                (request_v15.wire_type ==
                    RPC_RETURN_WIRE_RAW ||
                 request_v15.wire_type ==
                    RPC_RETURN_WIRE_F16)) {
            extended = true;
            wire_type = request_v15.wire_type;
            header_size = sizeof(request_v15);
            original_size =
                static_cast<size_t>(
                    request_v15.original_size);
        }
    }

    if (request.device >= backends.size() ||
            request.phone_ffn_seq == 0) {
        return false;
    }

    const size_t wire_size =
        input.size() - header_size;
    if (wire_type == RPC_RETURN_WIRE_RAW) {
        if (wire_size != original_size) {
            return false;
        }
    } else {
        if (!extended ||
                request.tensor.type !=
                    static_cast<uint32_t>(GGML_TYPE_F32) ||
                original_size == 0 ||
                original_size % sizeof(float) != 0 ||
                wire_size !=
                    (original_size / sizeof(float)) *
                        sizeof(ggml_fp16_t)) {
            return false;
        }
    }

    const int64_t wait_begin_us = ggml_time_us();
    {
        std::unique_lock<std::mutex> lock(phone_ffn_ready_mutex);
        phone_ffn_ready_cv.wait(
            lock,
            [&]() {
                auto it = phone_ffn_ready_seqs.find(request.device);
                return it != phone_ffn_ready_seqs.end() &&
                    it->second.find(request.phone_ffn_seq) !=
                        it->second.end();
            });

        auto it = phone_ffn_ready_seqs.find(request.device);
        GGML_ASSERT(it != phone_ffn_ready_seqs.end());
        const size_t erased =
            it->second.erase(request.phone_ffn_seq);
        GGML_ASSERT(erased == 1);
        if (it->second.empty()) {
            phone_ffn_ready_seqs.erase(it);
        }
    }
    const int64_t wait_us = ggml_time_us() - wait_begin_us;

    rpc_phone_npu_cpu_reduce_entry cpu_entry;
    bool cpu_reduce = false;
    bool cpu_reduce_failed = false;
    {
        std::lock_guard<std::mutex> lock(
            phone_npu_cpu_reduce_mutex);

        auto failed_it =
            phone_npu_cpu_reduce_failed.find(request.device);
        if (failed_it != phone_npu_cpu_reduce_failed.end()) {
            cpu_reduce_failed =
                failed_it->second.erase(
                    request.phone_ffn_seq) != 0;
            if (failed_it->second.empty()) {
                phone_npu_cpu_reduce_failed.erase(failed_it);
            }
        }

        auto device_it =
            phone_npu_cpu_reduce_pending.find(request.device);
        if (!cpu_reduce_failed &&
                device_it != phone_npu_cpu_reduce_pending.end()) {
            auto entry_it =
                device_it->second.find(request.phone_ffn_seq);
            if (entry_it != device_it->second.end()) {
                cpu_entry = std::move(entry_it->second);
                device_it->second.erase(entry_it);
                if (device_it->second.empty()) {
                    phone_npu_cpu_reduce_pending.erase(device_it);
                }
                cpu_reduce = true;
            }
        }

        if (cpu_reduce_failed) {
            auto active_it =
                phone_npu_cpu_reduce_active.find(request.device);
            if (active_it !=
                    phone_npu_cpu_reduce_active.end()) {
                active_it->second.erase(
                    request.phone_ffn_seq);
                if (active_it->second.empty()) {
                    phone_npu_cpu_reduce_active.erase(active_it);
                }
            }
        }
    }

    if (cpu_reduce_failed) {
        GGML_LOG_ERROR(
            "[CHUNK_RING_CPU_REDUCE] seq=%" PRIu64
            " status=FAIL reason=npu_async_failed "
            "ffn_wait_ms=%.3f\n",
            request.phone_ffn_seq,
            wait_us / 1000.0);
        return false;
    }

    const uint8_t * wire_data =
        input.data() + header_size;

    if (cpu_reduce) {
        if (cpu_entry.down == nullptr ||
                cpu_entry.down->type != GGML_TYPE_F32 ||
                request.tensor.type !=
                    static_cast<uint32_t>(GGML_TYPE_F32) ||
                request.offset != 0 ||
                original_size !=
                    cpu_entry.phone_partial.size() *
                        sizeof(float) ||
                original_size !=
                    ggml_nbytes(cpu_entry.down)) {
            GGML_LOG_ERROR(
                "[CHUNK_RING_CPU_REDUCE] seq=%" PRIu64
                " layer=%d chunk=%d status=FAIL "
                "reason=shape_or_type bytes=%zu phone_bytes=%zu\n",
                request.phone_ffn_seq,
                cpu_entry.layer,
                cpu_entry.chunk,
                original_size,
                cpu_entry.phone_partial.size() * sizeof(float));
            return false;
        }

        std::vector<float> pc_partial(
            cpu_entry.phone_partial.size());
        if (wire_type == RPC_RETURN_WIRE_F16) {
            ggml_fp16_to_fp32_row(
                reinterpret_cast<const ggml_fp16_t *>(
                    wire_data),
                pc_partial.data(),
                static_cast<int64_t>(
                    pc_partial.size()));
        } else if (original_size > 0) {
            memcpy(
                pc_partial.data(),
                wire_data,
                original_size);
        }

        const int64_t reduce_begin_us =
            ggml_time_us();
        for (size_t i = 0;
             i < cpu_entry.phone_partial.size();
             ++i) {
            cpu_entry.phone_partial[i] +=
                pc_partial[i];
        }
        const int64_t reduce_us =
            ggml_time_us() - reduce_begin_us;

        const int64_t upload_begin_us =
            ggml_time_us();
        ggml_backend_tensor_set(
            cpu_entry.down,
            cpu_entry.phone_partial.data(),
            0,
            original_size);
        const int64_t upload_us =
            ggml_time_us() - upload_begin_us;

        {
            std::lock_guard<std::mutex> lock(
                phone_npu_cpu_reduce_mutex);
            auto active_it =
                phone_npu_cpu_reduce_active.find(
                    request.device);
            if (active_it !=
                    phone_npu_cpu_reduce_active.end()) {
                active_it->second.erase(
                    request.phone_ffn_seq);
                if (active_it->second.empty()) {
                    phone_npu_cpu_reduce_active.erase(
                        active_it);
                }
            }
        }

        GGML_LOG_INFO(
            "[CHUNK_RING_CPU_REDUCE] "
            "seq=%" PRIu64 " layer=%d chunk=%d bytes=%zu "
            "ffn_wait_ms=%.3f reduce_ms=%.3f upload_ms=%.3f "
            "status=OK\n",
            request.phone_ffn_seq,
            cpu_entry.layer,
            cpu_entry.chunk,
            original_size,
            wait_us / 1000.0,
            reduce_us / 1000.0,
            upload_us / 1000.0);
        return true;
    }

    std::vector<uint8_t> legacy_input(
        sizeof(rpc_tensor) +
        sizeof(uint64_t) +
        original_size);
    size_t cursor = 0;
    memcpy(
        legacy_input.data() + cursor,
        &request.tensor,
        sizeof(request.tensor));
    cursor += sizeof(request.tensor);
    memcpy(
        legacy_input.data() + cursor,
        &request.offset,
        sizeof(request.offset));
    cursor += sizeof(request.offset);

    if (wire_type == RPC_RETURN_WIRE_F16) {
        const size_t n_elements =
            original_size / sizeof(float);
        ggml_fp16_to_fp32_row(
            reinterpret_cast<const ggml_fp16_t *>(wire_data),
            reinterpret_cast<float *>(
                legacy_input.data() + cursor),
            static_cast<int64_t>(n_elements));
        cursor += original_size;
    } else if (original_size > 0) {
        memcpy(
            legacy_input.data() + cursor,
            wire_data,
            original_size);
        cursor += original_size;
    }
    GGML_ASSERT(cursor == legacy_input.size());

    const bool status =
        set_tensor_async_return(legacy_input);

    if (rpc_tensor_phone_stage_profile_enabled()) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=phone "
            "stage=return_chunk_join device=%u seq=%" PRIu64
            " ffn_wait_ms=%.3f bytes=%zu wire_bytes=%zu wire=%s "
            "status=%d\n",
            request.device,
            request.phone_ffn_seq,
            wait_us / 1000.0,
            original_size,
            wire_size,
            wire_type == RPC_RETURN_WIRE_F16 ? "f16" : "raw",
            status ? 1 : 0);
    }

    return status;
}
bool rpc_server::set_tensor_recompute_snapshot(const std::vector<uint8_t> & input) {
    if (input.size() < sizeof(rpc_msg_set_tensor_recompute_snapshot_req)) {
        return false;
    }

    rpc_msg_set_tensor_recompute_snapshot_req request {};
    memcpy(&request, input.data(), sizeof(request));

    const size_t data_size = input.size() - sizeof(request);
    if (data_size != request.input_size) {
        return false;
    }

    if (!set_tensor_direct(
            request.input_tensor,
            request.input_offset,
            input.data() + sizeof(request),
            data_size)) {
        return false;
    }

    return graph_recompute_snapshot(request.graph);
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    if (!ensure_opencl_tensor_extra(tensor)) {
        GGML_LOG_ERROR(
            "[%s] failed to initialize OpenCL tensor state for '%s'\n",
            __func__, tensor->name);
        return false;
    }

    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    remember_opencl_tensor_extra(tensor);
    response.result = 1;
    return true;
}

bool rpc_server::init_zero_tensor(
        const rpc_msg_init_zero_tensor_req & request,
        rpc_msg_init_zero_tensor_rsp & response) {
    response.result = 0;

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);

    ggml_tensor * tensor =
        deserialize_tensor(ctx_ptr.get(), &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR(
            "[%s] error deserializing synthetic tensor\n",
            __func__);
        return false;
    }

    // This command exists only to materialize backend-private state required
    // by synthetic profiling weights. CPU/CUDA RPC servers already have valid
    // zero-filled storage after BUFFER_CLEAR, so leave those paths untouched.
    if (!is_opencl_tensor(tensor) || !ggml_is_quantized(tensor->type)) {
        response.result = 1;
        return true;
    }

    const rpc_opencl_tensor_storage_key storage_key =
        opencl_tensor_storage_key(tensor);

    // Repeated initialization of the same profiling tensor is unnecessary and
    // unsafe for SoA quantization: set_tensor expects the generic allocation
    // extra as its input and replaces it with a type-specific extra. Track
    // successful profile initialization explicitly instead of treating an
    // ordinary extra-cache hit as proof that SoA conversion already happened.
    {
        std::lock_guard<std::mutex> lock(opencl_tensor_extras_mutex);
        if (opencl_profile_zero_initialized.find(storage_key) !=
            opencl_profile_zero_initialized.end()) {
            response.result = 1;
            return true;
        }
    }

    if (!ensure_opencl_tensor_extra(tensor)) {
        GGML_LOG_ERROR(
            "[%s] failed to create generic OpenCL state for '%s'\n",
            __func__, tensor->name);
        return false;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    if (nbytes == 0) {
        remember_opencl_tensor_extra(tensor);
        response.result = 1;
        return true;
    }

    std::vector<uint8_t> zero_data;
    try {
        zero_data.resize(nbytes, 0);
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR(
            "[%s] failed to allocate %zu zero bytes for '%s'\n",
            __func__, nbytes, tensor->name);
        return false;
    }

    ggml_backend_tensor_set(
        tensor,
        zero_data.data(),
        0,
        nbytes);

    // OpenCL quantized set_tensor replaces the generic extra with the
    // Q4_K/Q5_K/Q6_K/etc. SoA object. Cache that final pointer so subsequent
    // graph reconstruction restores the correct layout.
    remember_opencl_tensor_extra(tensor);

    {
        std::lock_guard<std::mutex> lock(opencl_tensor_extras_mutex);
        opencl_profile_zero_initialized.insert(storage_key);
    }

    if (rpc_opencl_extra_debug_enabled()) {
        GGML_LOG_DEBUG(
            "[RPC_OPENCL_PROFILE_INIT] name=%s type=%d buffer=%p "
            "data=%p bytes=%zu extra=%p\n",
            tensor->name,
            (int) tensor->type,
            (void *) tensor->buffer,
            tensor->data,
            nbytes,
            tensor->extra);
    }

    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);

    if (is_opencl_tensor(tensor)) {
        if (!ensure_opencl_tensor_extra(tensor)) {
            GGML_LOG_ERROR(
                "[%s] failed to initialize OpenCL tensor state for '%s'\n",
                __func__, tensor->name);
            return false;
        }
        remember_opencl_tensor_extra(tensor);
        return true;
    }

    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

// Executed in the Phone RPC server process. GPU outputs are read here,
// packed directly into the existing inference transport TX slots, and sent
// Phone->PC by its TX worker. The client control RPC contains descriptors
// only and is not a second host payload transfer.
// Called on an ordinary RPC control connection after the data-plane worker
// has received the corresponding real C2P frame. The request contains GPU
// destination DESCRIPTORS only; no payload is retransmitted by PC RPC.
bool rpc_server::apply_inference_cycle(
        const rpc_msg_inference_cycle_send_req & request) {
    const auto kind = static_cast<inference_cycle_payload_kind>(request.kind);
    const bool ready = kind == INFERENCE_CYCLE_READY;
    if ((kind != INFERENCE_CYCLE_READY &&
         kind != INFERENCE_CYCLE_PARTIAL) ||
        request.device >= backends.size() ||
        request.layer < 0 || request.chunk < 0 || request.seq == 0 ||
        request.tokens == 0 || request.tokens > 128 ||
        request.hidden == 0 || request.hidden > 65536 ||
        (ready ? request.topk == 0 || request.topk > 128 :
                 request.topk != 0)) {
        return false;
    }
    inference_cycle_message msg;
    const auto key = cycle_key(
        request.layer, request.chunk, request.seq, request.graph_id,
        request.kind);
    {
        std::unique_lock<std::mutex> lock(cycle_rx_mutex);
        const bool awoke = cycle_rx_cv.wait_for(
            lock, std::chrono::seconds(30), [&] {
                return cycle_rx_failed || cycle_rx_frames.count(key) > 0;
            });
        if (!awoke || cycle_rx_failed) {
            GGML_LOG_ERROR("[CYCLE_C2P_APPLY] status=FAIL reason=RX_TIMEOUT_OR_CLOSED "
                           "layer=%d chunk=%d seq=%" PRIu64 "\n",
                           request.layer, request.chunk, request.seq);
            return false;
        }
        auto it = cycle_rx_frames.find(key);
        if (it == cycle_rx_frames.end()) {
            return false;
        }
        msg = std::move(it->second);
        cycle_rx_frames.erase(it);
    }

    const size_t n = size_t(request.tokens) * request.hidden;
    const size_t routes = size_t(request.tokens) * request.topk;
    if ((ready &&
         (msg.ready.tokens != request.tokens ||
          msg.ready.hidden != request.hidden ||
          msg.ready.topk != request.topk ||
          msg.ready.residual.size() != n ||
          msg.ready.ffn_norm.size() != n ||
          msg.ready.topk_ids.size() != routes ||
          msg.ready.topk_weights.size() != routes)) ||
        (!ready &&
         (msg.partial.tokens != request.tokens ||
          msg.partial.hidden != request.hidden ||
          msg.partial.partial.size() != n))) {
        return false;
    }

    const void * source[4] {};
    if (ready) {
        source[0] = msg.ready.residual.data();
        source[1] = msg.ready.ffn_norm.data();
        source[2] = msg.ready.topk_ids.data();
        source[3] = msg.ready.topk_weights.data();
    } else {
        source[0] = msg.partial.partial.data();
    }

    const size_t count = ready ? 4 : 1;
    for (size_t i = 0; i < count; ++i) {
        const rpc_tensor & dst = request.tensors[i];
        const uint32_t type = ready && i == 2 ?
            GGML_TYPE_I32 : GGML_TYPE_F32;
        const size_t elements = ready && i >= 2 ? routes : n;
        if (dst.buffer == 0 || dst.data == 0 || dst.type != type ||
            dst.nb[0] != sizeof(float)) {
            return false;
        }
        uint64_t actual = 1;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (dst.ne[d] == 0 || actual > elements / dst.ne[d]) {
                return false;
            }
            actual *= dst.ne[d];
            if (d > 0 && dst.nb[d] !=
                    uint64_t(dst.nb[d - 1]) * dst.ne[d - 1]) {
                return false;
            }
        }
        if (actual != elements ||
            !set_tensor_direct(dst, 0, source[i], elements * sizeof(float))) {
            GGML_LOG_ERROR("[CYCLE_C2P_APPLY] side=phone status=FAIL "
                           "reason=GPU_BUFFER_SET kind=%s tensor=%zu\n",
                           ready ? "READY" : "PARTIAL", i);
            return false;
        }
    }
    ggml_backend_synchronize(backends[request.device]);
    // READY ACK means Phone GPU inputs can be consumed. PARTIAL needs the
    // subsequent GPU Reduce before ACK may be sent to the PC sender.
    if (ready) {
        auto transport = get_inference_data_plane();
        if (!transport || !inference_cycle_send_ack(*transport, msg, true)) {
            return false;
        }
    } else {
        std::lock_guard<std::mutex> lock(cycle_rx_mutex);
        if (!cycle_rx_applied_partial.insert(key).second) {
            return false;
        }
    }
    GGML_LOG_INFO(
        "[CYCLE_C2P_APPLY] side=phone kind=%s layer=%d chunk=%d "
        "seq=%" PRIu64 " bytes=%zu state=PHONE_GPU_DEVICE_READY\n",
        ready ? "READY" : "PARTIAL",
        request.layer, request.chunk, request.seq,
        (ready ? 2 * n + 2 * routes : n) * sizeof(float));
    return true;
}

bool rpc_server::finish_inference_cycle(
        const rpc_msg_inference_cycle_send_req & request) {
    if (request.device >= backends.size() ||
        request.kind != INFERENCE_CYCLE_PARTIAL ||
        request.layer < 0 || request.chunk < 0 || request.seq == 0) {
        return false;
    }
    const auto key = cycle_key(
        request.layer, request.chunk, request.seq, request.graph_id,
        INFERENCE_CYCLE_PARTIAL);
    {
        std::lock_guard<std::mutex> lock(cycle_rx_mutex);
        if (cycle_rx_applied_partial.count(key) == 0) {
            return false;
        }
    }
    ggml_backend_synchronize(backends[request.device]);
    inference_cycle_message msg;
    msg.direction = INFERENCE_CYCLE_PC_TO_PHONE;
    msg.kind = INFERENCE_CYCLE_PARTIAL;
    msg.layer = request.layer;
    msg.chunk = request.chunk;
    msg.seq = request.seq;
    msg.graph_id = request.graph_id;
    auto transport = get_inference_data_plane();
    if (!transport || !inference_cycle_send_ack(*transport, msg, true)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(cycle_rx_mutex);
        cycle_rx_applied_partial.erase(key);
    }
    GGML_LOG_INFO(
        "[CYCLE_C2P_FINISH] side=phone layer=%d chunk=%d seq=%" PRIu64
        " reduce=PHONE_GPU ack=PARTIAL_DEVICE_READY status=OK\n",
        request.layer, request.chunk, request.seq);
    return true;
}

bool rpc_server::send_inference_cycle(
        const rpc_msg_inference_cycle_send_req & request) {
    const auto kind =
        static_cast<inference_cycle_payload_kind>(request.kind);
    const bool ready = kind == INFERENCE_CYCLE_READY;
    if ((kind != INFERENCE_CYCLE_READY &&
         kind != INFERENCE_CYCLE_PARTIAL) ||
        request.device >= backends.size() ||
        request.layer < 0 || request.chunk < 0 || request.seq == 0 ||
        request.tokens == 0 || request.tokens > 128 ||
        request.hidden == 0 || request.hidden > 65536 ||
        (ready ? request.topk == 0 || request.topk > 128 :
                 request.topk != 0)) {
        return false;
    }
    auto transport = get_inference_data_plane();
    if (!transport || !transport->healthy()) {
        return false;
    }
    size_t wire_bytes = 0;
    if (!inference_cycle_payload_size(
            kind, request.tokens, request.hidden, request.topk,
            wire_bytes)) {
        return false;
    }

    // Restrict the descriptors to contiguous, non-aliased value layouts
    // before get_tensor_into's own allocated-buffer/bounds checks.
    const size_t n = size_t(request.tokens) * request.hidden;
    const size_t r = size_t(request.tokens) * request.topk;
    auto read = [&](size_t index, ggml_type type, size_t elements, void * dst) {
        const rpc_tensor & tensor = request.tensors[index];
        if (tensor.type != uint32_t(type) || tensor.buffer == 0 ||
            tensor.data == 0 || tensor.nb[0] != sizeof(float)) {
            return false;
        }
        size_t actual = 1;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (tensor.ne[d] == 0 ||
                actual > elements / tensor.ne[d]) {
                return false;
            }
            actual *= tensor.ne[d];
            if (d > 0 && tensor.nb[d] !=
                    uint64_t(tensor.nb[d - 1]) * tensor.ne[d - 1]) {
                return false;
            }
        }
        if (actual != elements) {
            return false;
        }
        rpc_msg_get_tensor_req get {};
        get.tensor = tensor;
        get.offset = 0;
        get.size = elements * sizeof(float);
        return get_tensor_into(get, dst);
    };

    bool ok = false;
    if (ready) {
        std::vector<float> residual(n), norm(n), weights(r);
        std::vector<int32_t> ids(r);
        ok = read(0, GGML_TYPE_F32, n, residual.data()) &&
             read(1, GGML_TYPE_F32, n, norm.data()) &&
             read(2, GGML_TYPE_I32, r, ids.data()) &&
             read(3, GGML_TYPE_F32, r, weights.data());
        if (ok) {
            const inference_cycle_ready_view view {
                request.tokens, request.hidden, request.topk,
                residual.data(), norm.data(), ids.data(), weights.data()
            };
            ok = inference_cycle_send_ready(
                *transport, INFERENCE_CYCLE_PHONE_TO_PC,
                request.layer, request.chunk, request.seq, request.graph_id,
                view);
        }
    } else {
        std::vector<float> partial(n);
        ok = read(0, GGML_TYPE_F32, n, partial.data());
        if (ok) {
            const inference_cycle_partial_view view {
                request.tokens, request.hidden, partial.data()
            };
            ok = inference_cycle_send_partial(
                *transport, INFERENCE_CYCLE_PHONE_TO_PC,
                request.layer, request.chunk, request.seq, request.graph_id,
                view);
        }
    }

    GGML_LOG_INFO(
        "[CYCLE_REAL_WIRE] side=phone action=SEND kind=%s "
        "layer=%d chunk=%d seq=%" PRIu64 " wire_bytes=%zu "
        "phone_source=RPC_GPU_TENSORS tx=INFERENCE_TRANSPORT "
        "status=%s\n",
        ready ? "READY" : "PARTIAL",
        request.layer, request.chunk, request.seq,
        wire_bytes, ok ? "QUEUED" : "FAIL");
    return ok;
}

bool rpc_server::get_tensor_into(
        const rpc_msg_get_tensor_req & request,
        void * response_data) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    if (!ensure_opencl_tensor_extra(tensor)) {
        GGML_LOG_ERROR(
            "[%s] failed to initialize OpenCL tensor state for '%s'\n",
            __func__, tensor->name);
        return false;
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    const int64_t t0 = ggml_time_us();
    ggml_backend_tensor_get(
        tensor,
        response_data,
        request.offset,
        request.size);
    const int64_t get_us = ggml_time_us() - t0;

    if (RPC_DEBUG) {
        GGML_LOG_INFO(
            "[RPC_SERVER_GET] tensor=%s bytes=%" PRIu64
            " tensor_get=%.3f ms\n",
            tensor->name,
            request.size,
            get_us / 1000.0);
    }
    return true;
}

bool rpc_server::get_tensor(
        const rpc_msg_get_tensor_req & request,
        std::vector<uint8_t> & response) {
    response.resize(request.size, 0);
    return get_tensor_into(request, response.data());
}

bool rpc_server::route_snapshot_ready(
        const rpc_msg_route_snapshot_ready_req & request,
        rpc_msg_route_snapshot_ready_rsp & response) {
    response.result = 0;
    if (request.device >= backends.size() ||
            request.lane >= RPC_ROUTE_TRANSFER_LANES ||
            request.seq == 0) {
        return true;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ 3 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);

    ggml_tensor * tensors[3] = {};
    size_t total_size = 0;
    for (size_t i = 0; i < 3; ++i) {
        tensors[i] =
            deserialize_tensor(
                ctx_ptr.get(),
                &request.tensors[i]);
        if (tensors[i] == nullptr ||
                tensors[i]->buffer == nullptr ||
                request.sizes[i] != ggml_nbytes(tensors[i])) {
            return true;
        }
        if (!ensure_opencl_tensor_extra(tensors[i])) {
            return true;
        }
        if (request.sizes[i] > SIZE_MAX - total_size) {
            return true;
        }
        total_size +=
            static_cast<size_t>(request.sizes[i]);
    }

    const rpc_route_topk_compact_layout topk_layout =
        rpc_route_topk_compact_layout_for_tensor(tensors[1]);

    rpc_route_snapshot_slot & slot =
        route_snapshot_devices[request.device]->
            slots[request.lane];

    const int64_t snapshot_begin_us = ggml_time_us();
    const int64_t free_wait_begin_us = snapshot_begin_us;
    int64_t free_wait_us = 0;
    {
        std::unique_lock<std::mutex> lock(slot.mutex);
        slot.cv.wait(
            lock,
            [&]() {
                return slot.state ==
                    rpc_snapshot_state::FREE;
            });
        free_wait_us = ggml_time_us() - free_wait_begin_us;
        slot.state = rpc_snapshot_state::FILLING;
        slot.seq = request.seq;
        slot.sizes = {
            request.sizes[0],
            request.sizes[1],
            request.sizes[2],
        };
        slot.topk_compact_supported =
            topk_layout.supported &&
            topk_layout.wire_bytes <
                static_cast<size_t>(request.sizes[1]);
        slot.topk_rows = topk_layout.rows;
        slot.topk_row_bytes = topk_layout.row_bytes;
        slot.topk_stride_bytes = topk_layout.stride_bytes;
        slot.topk_wire_bytes = topk_layout.wire_bytes;
        slot.data.resize(total_size);
    }

    const int64_t fill_begin_us = ggml_time_us();
    size_t offset = 0;
    bool ok = true;
    for (size_t i = 0; i < 3; ++i) {
        const size_t size =
            static_cast<size_t>(request.sizes[i]);
        ggml_backend_tensor_get(
            tensors[i],
            slot.data.data() + offset,
            0,
            size);
        offset += size;
    }
    GGML_ASSERT(offset == total_size);
    const int64_t fill_us =
        ggml_time_us() - fill_begin_us;

    {
        std::lock_guard<std::mutex> lock(slot.mutex);
        slot.fill_us = fill_us;
        if (ok) {
            slot.state = rpc_snapshot_state::READY;
        } else {
            slot.state = rpc_snapshot_state::FREE;
            slot.data.clear();
            slot.seq = 0;
            slot.fill_us = 0;
            slot.topk_compact_supported = false;
            slot.topk_rows = 0;
            slot.topk_row_bytes = 0;
            slot.topk_stride_bytes = 0;
            slot.topk_wire_bytes = 0;
        }
    }
    slot.cv.notify_all();

    if (rpc_tensor_phone_stage_profile_enabled()) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=phone "
            "stage=route_snapshot_fill lane=%u "
            "seq=%" PRIu64 " bytes=%zu "
            "topk_storage_bytes=%" PRIu64 " topk_wire_bytes=%zu "
            "compact_topk_eligible=%d free_wait_ms=%.3f "
            "fill_ms=%.3f total_ms=%.3f status=%d\n",
            request.lane,
            request.seq,
            total_size,
            request.sizes[1],
            topk_layout.wire_bytes,
            (topk_layout.supported &&
             topk_layout.wire_bytes <
                static_cast<size_t>(request.sizes[1])) ? 1 : 0,
            free_wait_us / 1000.0,
            fill_us / 1000.0,
            (ggml_time_us() - snapshot_begin_us) / 1000.0,
            ok ? 1 : 0);
        std::fflush(stderr);
    }

    bool route_dispatch_ok = true;
    if (ok) {
        route_dispatch_ok =
            queue_inference_route_if_ready(
                request.device,
                request.lane,
                request.seq);
    }

    response.result =
        ok && route_dispatch_ok ? 1 : 0;
    return true;
}

bool rpc_server::get_route_snapshot(
        const rpc_msg_get_route_snapshot_req & request,
        std::vector<uint8_t> & response) {
    if (request.device >= backends.size() ||
            request.lane >= RPC_ROUTE_TRANSFER_LANES ||
            request.seq == 0) {
        return false;
    }

    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const bool route_profile =
        std::getenv("GGML_RPC_ROUTE_STAGE_PROFILE") != nullptr;
    const bool measure_profile =
        stage_profile || route_profile;
    const int64_t server_enter_us =
        measure_profile ? ggml_time_us() : 0;

    const bool compact_topk_requested =
        (request.sizes[1] &
         RPC_ROUTE_SNAPSHOT_COMPACT_TOPK_FLAG) != 0;
    std::array<uint64_t, 3> requested_sizes = {
        request.sizes[0],
        request.sizes[1] &
            ~RPC_ROUTE_SNAPSHOT_COMPACT_TOPK_FLAG,
        request.sizes[2] &
            ~RPC_ROUTE_SNAPSHOT_WIRE_ACK_FLAG,
    };

    rpc_route_snapshot_slot & slot =
        route_snapshot_devices[request.device]->
            slots[request.lane];

    const int64_t lock_begin_us =
        measure_profile ? ggml_time_us() : 0;
    int64_t lock_wait_us = 0;
    int64_t cv_wait_us = 0;
    int64_t response_copy_us = 0;
    int64_t slot_fill_us = 0;
    int64_t ready_wait_us = 0;
    size_t storage_total_size = 0;
    size_t topk_storage_bytes = 0;
    size_t topk_wire_bytes = 0;
    bool compact_topk_used = false;
    {
        std::unique_lock<std::mutex> lock(slot.mutex);
        const int64_t lock_acquired_us =
            measure_profile ? ggml_time_us() : lock_begin_us;
        lock_wait_us = lock_acquired_us - lock_begin_us;

        const int64_t cv_wait_begin_us =
            measure_profile ? lock_acquired_us : ggml_time_us();
        slot.cv.wait(
            lock,
            [&]() {
                return slot.state ==
                        rpc_snapshot_state::READY &&
                    slot.seq == request.seq;
            });
        const int64_t ready_us = ggml_time_us();
        cv_wait_us = ready_us - cv_wait_begin_us;
        ready_wait_us = ready_us - lock_begin_us;
        slot_fill_us = slot.fill_us;

        for (size_t i = 0; i < 3; ++i) {
            if (slot.sizes[i] != requested_sizes[i]) {
                GGML_LOG_ERROR(
                    "[RPC_ROUTE_SNAPSHOT_MISMATCH] "
                    "lane=%u seq=%" PRIu64
                    " item=%zu expected=%" PRIu64
                    " actual=%" PRIu64 "\n",
                    request.lane,
                    request.seq,
                    i,
                    requested_sizes[i],
                    slot.sizes[i]);
                return false;
            }
            if (slot.sizes[i] > SIZE_MAX - storage_total_size) {
                return false;
            }
            storage_total_size +=
                static_cast<size_t>(slot.sizes[i]);
        }

        topk_storage_bytes =
            static_cast<size_t>(slot.sizes[1]);
        topk_wire_bytes = topk_storage_bytes;
        compact_topk_used =
            compact_topk_requested &&
            slot.topk_compact_supported;

        if (compact_topk_requested &&
                !slot.topk_compact_supported) {
            GGML_LOG_ERROR(
                "[RPC_ROUTE_SNAPSHOT_COMPACT_TOPK_UNSUPPORTED] "
                "lane=%u seq=%" PRIu64 "\n",
                request.lane,
                request.seq);
            return false;
        }

        slot.state = rpc_snapshot_state::SENDING;
        const int64_t copy_begin_us =
            measure_profile ? ggml_time_us() : ready_us;

        if (compact_topk_used) {
            topk_wire_bytes = slot.topk_wire_bytes;
            const size_t hidden_bytes =
                static_cast<size_t>(slot.sizes[0]);
            const size_t weights_bytes =
                static_cast<size_t>(slot.sizes[2]);
            if (hidden_bytes > SIZE_MAX - topk_wire_bytes ||
                    hidden_bytes + topk_wire_bytes >
                        SIZE_MAX - weights_bytes) {
                return false;
            }

            response.resize(
                hidden_bytes +
                topk_wire_bytes +
                weights_bytes);

            std::memcpy(
                response.data(),
                slot.data.data(),
                hidden_bytes);

            const uint8_t * topk_src =
                slot.data.data() + hidden_bytes;
            uint8_t * topk_dst =
                response.data() + hidden_bytes;
            for (size_t row = 0;
                    row < slot.topk_rows;
                    ++row) {
                std::memcpy(
                    topk_dst + row * slot.topk_row_bytes,
                    topk_src + row * slot.topk_stride_bytes,
                    slot.topk_row_bytes);
            }

            std::memcpy(
                response.data() +
                    hidden_bytes +
                    topk_wire_bytes,
                slot.data.data() +
                    hidden_bytes +
                    topk_storage_bytes,
                weights_bytes);
        } else {
            response = slot.data;
        }

        const int64_t copy_done_us =
            measure_profile ? ggml_time_us() : copy_begin_us;
        response_copy_us = copy_done_us - copy_begin_us;
        slot.data.clear();
        slot.seq = 0;
        slot.sizes = {};
        slot.fill_us = 0;
        slot.topk_compact_supported = false;
        slot.topk_rows = 0;
        slot.topk_row_bytes = 0;
        slot.topk_stride_bytes = 0;
        slot.topk_wire_bytes = 0;
        slot.state = rpc_snapshot_state::FREE;
    }
    slot.cv.notify_all();

    if (route_profile) {
        const int64_t producer_wait_est_us =
            std::max<int64_t>(0, ready_wait_us - slot_fill_us);
        std::fprintf(
            stderr,
            "[ROUTE_WAIT_RAW] seq=%" PRIu64 " lane=%u "
            "lock=%.3f cv=%.3f fill=%.3f copy=%.3f "
            "prod=%.3f ready=%.3f\n",
            request.seq,
            request.lane,
            lock_wait_us / 1000.0,
            cv_wait_us / 1000.0,
            slot_fill_us / 1000.0,
            response_copy_us / 1000.0,
            producer_wait_est_us / 1000.0,
            ready_wait_us / 1000.0);
    }

    if (stage_profile) {
        const int64_t producer_wait_est_us =
            std::max<int64_t>(0, ready_wait_us - slot_fill_us);
        GGML_LOG_INFO(
            "[ROUTE_WAIT] seq=%" PRIu64 " lane=%u "
            "lock=%.3f cv=%.3f fill=%.3f copy=%.3f "
            "prod=%.3f ready=%.3f\n",
            request.seq,
            request.lane,
            lock_wait_us / 1000.0,
            cv_wait_us / 1000.0,
            slot_fill_us / 1000.0,
            response_copy_us / 1000.0,
            producer_wait_est_us / 1000.0,
            ready_wait_us / 1000.0);
    }

    return true;
}

bool rpc_server::queue_inference_route_if_ready(
        uint32_t device,
        uint32_t lane,
        uint64_t seq) {
    if (device >= route_snapshot_devices.size() ||
            lane >= RPC_ROUTE_TRANSFER_LANES ||
            seq == 0) {
        return false;
    }

    rpc_route_snapshot_slot & slot =
        route_snapshot_devices[device]->slots[lane];

    {
        std::lock_guard<std::mutex> slot_lock(slot.mutex);
        if (slot.state != rpc_snapshot_state::READY ||
                slot.seq != seq) {
            // ARM can legitimately arrive before READY.
            return true;
        }
    }

    bool queued = false;
    {
        std::lock_guard<std::mutex> lock(
            inference_route_mutex);

        auto device_it =
            inference_route_arms.find(device);
        if (device_it == inference_route_arms.end()) {
            // READY can legitimately arrive before ARM.
            return true;
        }

        auto seq_it =
            device_it->second.find(seq);
        if (seq_it == device_it->second.end()) {
            return true;
        }

        if (seq_it->second.request.snapshot.lane != lane) {
            GGML_LOG_ERROR(
                "[INFERENCE_ROUTE_ARM] side=server status=FAIL "
                "reason=lane_mismatch device=%u seq=%" PRIu64
                " expected=%u actual=%u\n",
                device,
                seq,
                seq_it->second.request.snapshot.lane,
                lane);
            device_it->second.erase(seq_it);
            if (device_it->second.empty()) {
                inference_route_arms.erase(device_it);
            }
            return false;
        }

        inference_route_send_job job;
        job.request = seq_it->second.request;
        job.arm_us = seq_it->second.arm_us;
        job.ready_us = ggml_time_us();

        inference_route_jobs.push_back(std::move(job));
        device_it->second.erase(seq_it);
        if (device_it->second.empty()) {
            inference_route_arms.erase(device_it);
        }
        queued = true;
    }

    if (queued) {
        inference_route_cv.notify_one();
    }
    return true;
}

void rpc_server::inference_route_sender_loop() {
    for (;;) {
        inference_route_send_job job;
        size_t queue_depth = 0;

        {
            std::unique_lock<std::mutex> lock(
                inference_route_mutex);
            inference_route_cv.wait(
                lock,
                [&]() {
                    return inference_route_stop ||
                        !inference_route_jobs.empty();
                });

            if (inference_route_stop &&
                    inference_route_jobs.empty()) {
                return;
            }

            job = std::move(
                inference_route_jobs.front());
            inference_route_jobs.pop_front();
            queue_depth = inference_route_jobs.size();
        }

        const int64_t worker_begin_us =
            ggml_time_us();

        std::vector<uint8_t> payload;
        bool status =
            get_route_snapshot(
                job.request.snapshot,
                payload);
        const int64_t snapshot_done_us =
            ggml_time_us();

        auto transport =
            status ?
                get_inference_data_plane() :
                nullptr;
        status =
            status &&
            transport != nullptr &&
            transport->healthy();

        inference_tx_buffer tx;
        const int64_t acquire_begin_us =
            ggml_time_us();
        if (status) {
            tx = transport->acquire_tx(
                payload.size());
            status = static_cast<bool>(tx);
        }
        const int64_t acquire_done_us =
            ggml_time_us();

        if (status && !payload.empty()) {
            std::memcpy(
                tx.data,
                payload.data(),
                payload.size());
        }

        const int64_t submit_begin_us =
            ggml_time_us();
        if (status) {
            inference_frame_header header =
                inference_make_header(
                    INFERENCE_FRAME_DATA);
            header.stage =
                INFERENCE_STAGE_ROUTE;
            header.stream_id =
                job.request.snapshot.device;
            header.seq =
                job.request.snapshot.seq;
            header.graph_id = 0;
            header.layer = job.request.layer;
            header.chunk = job.request.chunk;

            status = transport->submit_tx(
                std::move(tx),
                header,
                payload.size());
        } else if (tx) {
            transport->release_tx(
                std::move(tx));
        }
        const int64_t submit_done_us =
            ggml_time_us();

        GGML_LOG_INFO(
            "[INFERENCE_ROUTE_SEND] side=server "
            "device=%u lane=%u seq=%" PRIu64
            " layer=%d chunk=%d bytes=%zu "
            "arm_to_ready_ms=%.3f queue_ms=%.3f "
            "snapshot_ms=%.3f acquire_ms=%.3f "
            "submit_ms=%.3f pending_jobs=%zu status=%s\n",
            job.request.snapshot.device,
            job.request.snapshot.lane,
            job.request.snapshot.seq,
            job.request.layer,
            job.request.chunk,
            payload.size(),
            (job.ready_us - job.arm_us) / 1000.0,
            (worker_begin_us - job.ready_us) / 1000.0,
            (snapshot_done_us - worker_begin_us) / 1000.0,
            (acquire_done_us - acquire_begin_us) / 1000.0,
            (submit_done_us - submit_begin_us) / 1000.0,
            queue_depth,
            status ? "OK" : "FAIL");

        if (!status) {
            if (transport) {
                transport->stop();
            }
            // A failed data-plane send makes any waiting PC route consumer
            // fail via the transport rather than silently falling back after
            // ARM was accepted.
            return;
        }
    }
}

bool rpc_server::route_mark_ready(
        const rpc_msg_route_mark_ready_req & request) {
    {
        std::lock_guard<std::mutex> lock(route_ready_mutex);
        uint64_t & ready = route_ready_seq[request.device];
        ready = std::max(ready, request.seq);
    }
    route_ready_cv.notify_all();
    return true;
}

struct rpc_route_batch_server_profile {
    bool valid = false;
    uint64_t wait_seq = 0;
    int64_t wait_us = 0;
    int64_t prepare_us = 0;
    int64_t device_read_us = 0;
    size_t bytes = 0;
    bool opencl_batch_used = false;
};

static thread_local rpc_route_batch_server_profile
    rpc_route_batch_server_stage {};

static bool rpc_route_stage_profile_enabled() {
    return std::getenv("GGML_RPC_ROUTE_STAGE_PROFILE") != nullptr;
}

bool rpc_server::get_tensor_batch3_wait(
        const rpc_msg_get_tensor_batch3_wait_req & request,
        std::vector<uint8_t> & response) {
    const bool route_profile =
        rpc_route_stage_profile_enabled();
    const int64_t wait_begin_us =
        route_profile ? ggml_time_us() : 0;

    if (request.wait_seq != 0) {
        std::unique_lock<std::mutex> lock(route_ready_mutex);
        route_ready_cv.wait(
            lock,
            [&]() {
                auto it = route_ready_seq.find(request.device);
                return it != route_ready_seq.end() &&
                       it->second >= request.wait_seq;
            });
    }

    if (route_profile) {
        rpc_route_batch_server_stage.wait_seq =
            request.wait_seq;
        rpc_route_batch_server_stage.wait_us =
            ggml_time_us() - wait_begin_us;
    }

    return get_tensor_batch3(request.batch, response);
}

bool rpc_server::get_tensor_batch3(
        const rpc_msg_get_tensor_batch3_req & request,
        std::vector<uint8_t> & response) {
    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const bool route_profile =
        rpc_route_stage_profile_enabled();
    const bool measure_profile =
        stage_profile || route_profile;
    const int64_t server_begin_us =
        measure_profile ? ggml_time_us() : 0;

    size_t sizes[3] = {};
    size_t response_offsets[3] = {};
    size_t total_size = 0;
    for (size_t i = 0; i < 3; ++i) {
        if (request.items[i].size > SIZE_MAX - total_size) {
            return false;
        }
        response_offsets[i] = total_size;
        sizes[i] = static_cast<size_t>(request.items[i].size);
        total_size += sizes[i];
    }

    response.resize(total_size);
    const int64_t read_begin_us = ggml_time_us();
    const int64_t prepare_us =
        measure_profile ? read_begin_us - server_begin_us : 0;

    bool opencl_batch_used = false;
    if (std::getenv("GGML_RPC_DISABLE_OPENCL_BATCH3_READ") == nullptr) {
        struct ggml_init_params params {
            /*.mem_size   =*/ 3 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context_ptr ctx_ptr { ggml_init(params) };
        GGML_ASSERT(ctx_ptr != nullptr);

        ggml_tensor * tensors[3] = {};
        bool valid = true;
        for (size_t i = 0; i < 3; ++i) {
            tensors[i] =
                deserialize_tensor(ctx_ptr.get(), &request.items[i].tensor);
            if (tensors[i] == nullptr || tensors[i]->buffer == nullptr ||
                    !is_opencl_tensor(tensors[i]) ||
                    !ensure_opencl_tensor_extra(tensors[i])) {
                valid = false;
                break;
            }
        }

        if (valid) {
            ggml_backend_dev_t dev =
                ggml_backend_buft_get_device(tensors[0]->buffer->buft);
            for (size_t i = 1; i < 3 && valid; ++i) {
                valid =
                    ggml_backend_buft_get_device(tensors[i]->buffer->buft) ==
                    dev;
            }

            ggml_backend_reg_t reg =
                valid && dev != nullptr ?
                    ggml_backend_dev_backend_reg(dev) : nullptr;
            const ggml_backend_opencl_get_tensor_batch3_t opencl_batch =
                reg != nullptr ?
                    reinterpret_cast<ggml_backend_opencl_get_tensor_batch3_t>(
                        ggml_backend_reg_get_proc_address(
                            reg,
                            GGML_BACKEND_OPENCL_GET_TENSOR_BATCH3_PROC)) :
                    nullptr;

            if (opencl_batch != nullptr) {
                opencl_batch_used = opencl_batch(
                    tensors[0],
                    response.data() + response_offsets[0],
                    static_cast<size_t>(request.items[0].offset),
                    sizes[0],
                    tensors[1],
                    response.data() + response_offsets[1],
                    static_cast<size_t>(request.items[1].offset),
                    sizes[1],
                    tensors[2],
                    response.data() + response_offsets[2],
                    static_cast<size_t>(request.items[2].offset),
                    sizes[2]);
            }
        }
    }

    if (!opencl_batch_used) {
        for (size_t i = 0; i < 3; ++i) {
            if (!get_tensor_into(
                    request.items[i],
                    response.data() + response_offsets[i])) {
                return false;
            }
        }
    }

    const int64_t read_us = ggml_time_us() - read_begin_us;

    if (route_profile) {
        rpc_route_batch_server_stage.valid = true;
        rpc_route_batch_server_stage.prepare_us = prepare_us;
        rpc_route_batch_server_stage.device_read_us = read_us;
        rpc_route_batch_server_stage.bytes = total_size;
        rpc_route_batch_server_stage.opencl_batch_used =
            opencl_batch_used;
    }

    if (RPC_DEBUG) {
        GGML_LOG_INFO(
            "[RPC_SERVER_GET_BATCH3] bytes=%zu opencl_wait1=%d total=%.3f ms\n",
            total_size,
            opencl_batch_used ? 1 : 0,
            read_us / 1000.0);
    }
    if (stage_profile) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=phone stage=router_batch "
            "bytes=%zu opencl_wait1=%d prepare_ms=%.3f "
            "device_read_ms=%.3f total_ms=%.3f\n",
            total_size,
            opencl_batch_used ? 1 : 0,
            prepare_us / 1000.0,
            read_us / 1000.0,
            (ggml_time_us() - server_begin_us) / 1000.0);
    }
    return true;
}

bool rpc_server::snapshot_tensor(const rpc_msg_snapshot_tensor_req & request) {
    if (request.device >= backends.size() || request.slot >= 2) {
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);

    ggml_tensor * tensor = deserialize_tensor(ctx_ptr.get(), &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return false;
    }
    if (!ensure_opencl_tensor_extra(tensor)) {
        GGML_LOG_ERROR(
            "[%s] failed to initialize OpenCL tensor state for '%s'\n",
            __func__, tensor->name);
        return false;
    }

    return snapshot_tensor_direct(
        request.device,
        request.slot,
        request.seq,
        tensor,
        request.offset,
        request.size);
}

bool rpc_server::snapshot_tensor_direct(
        uint32_t device,
        uint32_t slot_id,
        uint64_t seq,
        ggml_tensor * tensor,
        uint64_t offset,
        uint64_t size) {
    if (device >= backends.size() ||
        slot_id >= 2 ||
        tensor == nullptr ||
        tensor->buffer == nullptr) {
        return false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    if (offset > tensor_size || size > tensor_size - offset) {
        return false;
    }

    rpc_snapshot_slot & slot = snapshot_devices[device]->slots[slot_id];
    const int64_t free_wait_start_us = ggml_time_us();
    {
        std::unique_lock<std::mutex> lock(slot.mutex);
        slot.cv.wait(lock, [&]() { return slot.state == rpc_snapshot_state::FREE; });
        slot.free_wait_us = ggml_time_us() - free_wait_start_us;
        slot.fill_us = 0;
        slot.state = rpc_snapshot_state::FILLING;
        slot.seq = seq;
        slot.data.resize(size);
    }

    const int64_t t0 = ggml_time_us();

    ggml_backend_tensor_get(
        tensor,
        slot.data.data(),
        offset,
        size);

    const int64_t t1 = ggml_time_us();
    const int64_t fill_us = t1 - t0;
    int64_t free_wait_us = 0;

    {
        std::lock_guard<std::mutex> lock(slot.mutex);
        slot.fill_us = fill_us;
        free_wait_us = slot.free_wait_us;
        slot.state = rpc_snapshot_state::READY;
    }

    if (RPC_DEBUG || std::getenv("GGML_RETURN_PATH_DEBUG") != nullptr) {
        printf(
            "[RPC_SNAPSHOT_FILL] "
            "slot=%u seq=%" PRIu64 " bytes=%" PRIu64
            " free_wait=%.3f tensor_get=%.3f ms\n",
            slot_id,
            seq,
            size,
            free_wait_us / 1000.0,
            fill_us / 1000.0);
        fflush(stdout);
    }

    slot.cv.notify_all();
    return true;
}

bool rpc_server::send_snapshot(const rpc_msg_get_snapshot_req & request, socket_ptr sock) {
    if (request.device >= backends.size() || request.slot >= 2) {
        return false;
    }

    rpc_snapshot_slot & slot = snapshot_devices[request.device]->slots[request.slot];
    const uint8_t * data;
    size_t size;
    int64_t slot_free_wait_us = 0;
    int64_t slot_fill_us = 0;
    const int64_t wait_start_us = ggml_time_us();
    {
        std::unique_lock<std::mutex> lock(slot.mutex);
        slot.cv.wait(lock, [&]() { return slot.state == rpc_snapshot_state::READY; });

        if (slot.seq != request.seq ||
            slot.state != rpc_snapshot_state::READY ||
            slot.data.size() != request.size) {
            GGML_LOG_ERROR(
                "[RPC_SNAPSHOT_MISMATCH] device=%u slot=%u request_seq=%" PRIu64
                " slot_seq=%" PRIu64 " state=%d request_size=%" PRIu64 " slot_size=%zu\n",
                request.device,
                request.slot,
                request.seq,
                slot.seq,
                (int) slot.state,
                request.size,
                slot.data.size());
            return false;
        }

        slot.state = rpc_snapshot_state::SENDING;
        data = slot.data.data();
        size = slot.data.size();
        slot_free_wait_us = slot.free_wait_us;
        slot_fill_us = slot.fill_us;
    }

    const int64_t ready_us = ggml_time_us();
    const int64_t send_start_us = ggml_time_us();

        // GET_SNAPSHOT 是固定长度协议。
        // request.size 已经由 client 和 server 校验过，
        // 不再额外发送 uint64_t response_size。
        const bool status = sock->send_data(data, size);

        const int64_t send_us = ggml_time_us() - send_start_us;

        {
            const size_t lane = request.slot & 1u;
            std::lock_guard<std::mutex> lock(snapshot_breakdown_mutex);
            auto & stats = snapshot_breakdown[lane];
            ++stats.requests;
            stats.bytes += size;
            stats.ready_wait_us += ready_us - wait_start_us;
            stats.send_us += send_us;

            const uint64_t total_requests =
                snapshot_breakdown[0].requests + snapshot_breakdown[1].requests;
            if (total_requests % 100 == 0) {
                for (size_t report_lane = 0; report_lane < snapshot_breakdown.size(); ++report_lane) {
                    const auto & report = snapshot_breakdown[report_lane];
                    printf(
                        "[SNAPSHOT_BREAKDOWN] total_requests=%" PRIu64 " lane=%zu requests=%" PRIu64
                        " ready_wait=%.3f send=%.3f bytes=%" PRIu64 "\n",
                        total_requests, report_lane, report.requests,
                        report.ready_wait_us / 1000.0, report.send_us / 1000.0, report.bytes);
                }
                fflush(stdout);
            }
        }

        if (RPC_DEBUG || std::getenv("GGML_RETURN_PATH_DEBUG") != nullptr) {
            printf(
                "[RPC_SNAPSHOT_SERVER] slot=%u lane=%u seq=%" PRIu64
                " bytes=%zu free_wait=%.3f fill=%.3f ready_wait=%.3f send=%.3f ms\n",
                request.slot,
                request.slot & 1u,
                request.seq,
                size,
                slot_free_wait_us / 1000.0,
                slot_fill_us / 1000.0,
                (ready_us - wait_start_us) / 1000.0,
                send_us / 1000.0);
            fflush(stdout);
        }

        {
            std::lock_guard<std::mutex> lock(slot.mutex);
            slot.state = rpc_snapshot_state::FREE;
        }

        slot.cv.notify_all();
        return status;
        }

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }
    const bool debug_k_quant =
        rpc_opencl_extra_debug_enabled() &&
        (rpc_is_k_quant_type(src->type) ||
         rpc_is_k_quant_type(dst->type));

    if (debug_k_quant) {
        GGML_LOG_DEBUG(
            "[RPC_KQUANT_SERVER_COPY_BEGIN] "
            "src_name=%s src_type=%d src_buffer=%p src_data=%p "
            "dst_name=%s dst_type=%d dst_buffer=%p dst_data=%p\n",
            src->name,
            (int) src->type,
            (void *) src->buffer,
            src->data,
            dst->name,
            (int) dst->type,
            (void *) dst->buffer,
            dst->data);
    }

    if (!ensure_opencl_tensor_extra(src) ||
        !ensure_opencl_tensor_extra(dst)) {
        GGML_LOG_ERROR(
            "[%s] failed to initialize OpenCL tensor state\n",
            __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);

    // Cycle Phone->Phone FFN handoff: some OpenCL buffers do not provide
    // a device-native cpy_tensor implementation. Recover ONLY on this RPC
    // server, never through the generic client's fallback (which would
    // download Phone->PC and re-upload PC->Phone). For small contiguous
    // F32/I32 activation tensors this bounded Phone-local staging preserves
    // one-direction payload semantics and avoids allocating another GPU
    // graph. Quantized model weights are never copied by this fallback.
    constexpr size_t cycle_local_copy_max_bytes = 4 * 1024 * 1024;
    const size_t nbytes = ggml_nbytes(src);
    const bool phone_local_copy_safe =
        !response.result &&
        nbytes != 0 && nbytes <= cycle_local_copy_max_bytes &&
        (src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_I32) &&
        ggml_are_same_layout(src, dst) &&
        ggml_is_contiguous(src) && ggml_is_contiguous(dst) &&
        ggml_backend_buffer_get_type(src->buffer) ==
            ggml_backend_buffer_get_type(dst->buffer) &&
        ggml_backend_buffer_get_type(src->buffer) !=
            ggml_backend_cpu_buffer_type();

    if (phone_local_copy_safe) {
        // Both buffers have been deserialized/validated by this server.
        // Their read/write happens on the Phone process only.
        std::vector<uint8_t> local_scratch(nbytes);
        ggml_backend_tensor_get(src, local_scratch.data(), 0, nbytes);
        ggml_backend_tensor_set(dst, local_scratch.data(), 0, nbytes);
        response.result = 1;
        GGML_LOG_INFO(
            "[CYCLE_PHONE_LOCAL_COPY] path=PHONE_HOST_STAGING "
            "src=%s dst=%s bytes=%zu cross_network_payload=0 status=OK\n",
            src->name, dst->name, nbytes);
    }

    if (debug_k_quant) {
        GGML_LOG_DEBUG(
            "[RPC_KQUANT_SERVER_COPY_DONE] result=%d "
            "src_name=%s src_extra=%p dst_name=%s dst_extra=%p\n",
            response.result ? 1 : 0,
            src->name,
            src->extra,
            dst->name,
            dst->extra);
    }

    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }   //如果找到对应的，则返回
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);  //反序列化rpc_tensor为ggml_tensor
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {  //指针值
            result->src[i] = nullptr;
        } else {    //递归调用create_node创建源节点
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;

    if (!ensure_opencl_tensor_extra(
            result,
            /* allow_quantized_weight_init = */ false)) {
        GGML_LOG_ERROR(
            "[%s] failed to restore OpenCL tensor state for node '%s'\n",
            __func__, result->name);
        return nullptr;
    }
    return result;
}

bool rpc_server::set_tensor_graph_compute(
        const std::vector<uint8_t> & input) {
    const bool stage_profile =
        rpc_tensor_phone_stage_profile_enabled();
    const int64_t server_begin_us =
        stage_profile ? ggml_time_us() : 0;

    if (input.size() < sizeof(rpc_msg_set_tensor_graph_req_v6)) {
        return false;
    }

    rpc_msg_set_tensor_graph_req_v6 request_v6 {};
    memcpy(&request_v6, input.data(), sizeof(request_v6));

    const size_t data_size = static_cast<size_t>(request_v6.size);
    const size_t graph_size = static_cast<size_t>(request_v6.graph_size);

    size_t header_size = sizeof(request_v6);
    bool cached_graph_request = false;
    uint64_t graph_uid = 0;
    uint32_t device = 0;

    const bool v6_size_ok =
        data_size <= SIZE_MAX - sizeof(request_v6) &&
        graph_size <= SIZE_MAX - sizeof(request_v6) - data_size &&
        input.size() == sizeof(request_v6) + data_size + graph_size;

    if (!v6_size_ok) {
        if (input.size() < sizeof(rpc_msg_set_tensor_graph_req)) {
            return false;
        }

        rpc_msg_set_tensor_graph_req request {};
        memcpy(&request, input.data(), sizeof(request));
        if (request.size != request_v6.size ||
                request.graph_size != request_v6.graph_size) {
            return false;
        }

        header_size = sizeof(request);
        graph_uid = request.graph_uid;
        device = request.device;

        if (data_size > SIZE_MAX - header_size ||
                graph_size > SIZE_MAX - header_size - data_size ||
                input.size() != header_size + data_size + graph_size) {
            return false;
        }

        cached_graph_request = graph_size == 0;
        if (cached_graph_request &&
                (graph_uid == 0 || device >= backends.size())) {
            return false;
        }
    }

    const uint8_t * data = input.data() + header_size;
    const int64_t write_begin_us = ggml_time_us();
    const int64_t parse_us =
        stage_profile ? write_begin_us - server_begin_us : 0;
    bool async_write_used = false;
    if (std::getenv("GGML_RPC_DISABLE_OPENCL_ASYNC_SET_ADD") == nullptr) {
        async_write_used = set_tensor_direct_opencl_async(
            request_v6.tensor,
            request_v6.offset,
            data,
            data_size);
    }

    if (!async_write_used &&
            !set_tensor_direct(
                request_v6.tensor,
                request_v6.offset,
                data,
                data_size)) {
        return false;
    }
    const int64_t write_us = ggml_time_us() - write_begin_us;

    const int64_t graph_begin_us = ggml_time_us();
    bool ok = false;

    if (cached_graph_request) {
        rpc_msg_graph_recompute_req recompute {};
        recompute.device = device;
        recompute.graph_uid = graph_uid;
        ok = graph_recompute(recompute);
    } else {
        std::vector<uint8_t> graph_input(graph_size);
        if (graph_size > 0) {
            memcpy(
                graph_input.data(),
                data + data_size,
                graph_size);
        }
        ok = graph_compute(graph_input);
    }

    const int64_t graph_us = ggml_time_us() - graph_begin_us;
    if (RPC_DEBUG) {
        GGML_LOG_INFO(
            "[RPC_SET_GRAPH_SERVER] tensor=%s bytes=%zu graph_uid=%" PRIu64
            " cache=%d graph_bytes=%zu async_write=%d total=%.3f ms status=%d\n",
            request_v6.tensor.name,
            data_size,
            graph_uid,
            cached_graph_request ? 1 : 0,
            graph_size,
            async_write_used ? 1 : 0,
            graph_us / 1000.0,
            ok ? 1 : 0);
    }
    if (stage_profile) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=phone stage=return_set_add "
            "bytes=%zu graph_uid=%" PRIu64 " cache=%d graph_bytes=%zu "
            "async_write=%d parse_ms=%.3f write_ms=%.3f "
            "graph_ms=%.3f total_ms=%.3f status=%d\n",
            data_size,
            graph_uid,
            cached_graph_request ? 1 : 0,
            graph_size,
            async_write_used ? 1 : 0,
            parse_us / 1000.0,
            write_us / 1000.0,
            graph_us / 1000.0,
            (ggml_time_us() - server_begin_us) / 1000.0,
            ok ? 1 : 0);
    }
    return ok;
}


bool rpc_server::prepare_phone_npu_ffn_cache(
        ggml_cgraph * graph,
        const rpc_phone_npu_ffn_stage & stage,
        rpc_phone_npu_ffn_cache *& cache) {
    cache = nullptr;
    auto it = phone_npu_ffn_caches.find(stage.layer);
    if (it != phone_npu_ffn_caches.end()) {
        cache = it->second.get();
        return cache != nullptr;
    }

    char name[GGML_MAX_NAME] = {};
    std::snprintf(name, sizeof(name),
        "blk.%d.ffn_up_exps.weight", stage.layer);
    ggml_tensor * gpu_up =
        rpc_phone_npu_find_named(graph, name);

    std::snprintf(name, sizeof(name),
        "blk.%d.ffn_gate_exps.weight", stage.layer);
    ggml_tensor * gpu_gate =
        rpc_phone_npu_find_named(graph, name);

    std::snprintf(name, sizeof(name),
        "blk.%d.ffn_down_exps.weight", stage.layer);
    ggml_tensor * gpu_down =
        rpc_phone_npu_find_named(graph, name);

    if (gpu_up == nullptr || gpu_gate == nullptr ||
        gpu_down == nullptr ||
        !rpc_phone_npu_supported_k_quant(gpu_up->type) ||
        !rpc_phone_npu_supported_k_quant(gpu_gate->type) ||
        !rpc_phone_npu_supported_k_quant(gpu_down->type) ||
        gpu_up->ne[0] != stage.norm->ne[0] ||
        gpu_gate->ne[0] != stage.norm->ne[0] ||
        gpu_up->ne[1] != gpu_gate->ne[1] ||
        gpu_up->ne[2] != gpu_gate->ne[2] ||
        gpu_down->ne[0] != gpu_up->ne[1] ||
        gpu_down->ne[1] != stage.norm->ne[0] ||
        gpu_down->ne[2] != gpu_up->ne[2]) {
        GGML_LOG_ERROR(
            "[PHONE_NPU_PRELOAD] layer=%d status=FAIL reason=weight_shape_or_type "
            "norm={type:%d ne:[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]} "
            "up={type:%d ne:[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]} "
            "gate={type:%d ne:[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]} "
            "down={type:%d ne:[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]}\n",
            stage.layer,
            stage.norm != nullptr ? (int) stage.norm->type : -1,
            stage.norm != nullptr ? stage.norm->ne[0] : -1,
            stage.norm != nullptr ? stage.norm->ne[1] : -1,
            stage.norm != nullptr ? stage.norm->ne[2] : -1,
            stage.norm != nullptr ? stage.norm->ne[3] : -1,
            gpu_up != nullptr ? (int) gpu_up->type : -1,
            gpu_up != nullptr ? gpu_up->ne[0] : -1,
            gpu_up != nullptr ? gpu_up->ne[1] : -1,
            gpu_up != nullptr ? gpu_up->ne[2] : -1,
            gpu_up != nullptr ? gpu_up->ne[3] : -1,
            gpu_gate != nullptr ? (int) gpu_gate->type : -1,
            gpu_gate != nullptr ? gpu_gate->ne[0] : -1,
            gpu_gate != nullptr ? gpu_gate->ne[1] : -1,
            gpu_gate != nullptr ? gpu_gate->ne[2] : -1,
            gpu_gate != nullptr ? gpu_gate->ne[3] : -1,
            gpu_down != nullptr ? (int) gpu_down->type : -1,
            gpu_down != nullptr ? gpu_down->ne[0] : -1,
            gpu_down != nullptr ? gpu_down->ne[1] : -1,
            gpu_down != nullptr ? gpu_down->ne[2] : -1,
            gpu_down != nullptr ? gpu_down->ne[3] : -1);
        return false;
    }

    ggml_backend_buffer_type_t repack =
        rpc_phone_npu_repack_buft(phone_npu_backend);
    if (repack == nullptr) {
        GGML_LOG_ERROR(
            "[PHONE_NPU_PRELOAD] layer=%d status=FAIL reason=no_repack_buffer\n",
            stage.layer);
        return false;
    }

    auto entry =
        std::make_unique<rpc_phone_npu_ffn_cache>();
    entry->layer = stage.layer;
    entry->ctx_w = rpc_phone_npu_make_ctx();
    if (!entry->ctx_w) {
        return false;
    }

    entry->w_up = ggml_new_tensor(
        entry->ctx_w.get(), GGML_TYPE_Q4_0,
        ggml_n_dims(gpu_up), gpu_up->ne);
    entry->w_gate = ggml_new_tensor(
        entry->ctx_w.get(), GGML_TYPE_Q4_0,
        ggml_n_dims(gpu_gate), gpu_gate->ne);
    entry->w_down = ggml_new_tensor(
        entry->ctx_w.get(), GGML_TYPE_Q4_0,
        ggml_n_dims(gpu_down), gpu_down->ne);

    ggml_set_name(entry->w_up, "phone_npu_ffn_up_q4_0");
    ggml_set_name(entry->w_gate, "phone_npu_ffn_gate_q4_0");
    ggml_set_name(entry->w_down, "phone_npu_ffn_down_q4_0");

    entry->wbuf.reset(
        ggml_backend_alloc_ctx_tensors_from_buft(
            entry->ctx_w.get(), repack));
    if (!entry->wbuf) {
        GGML_LOG_ERROR(
            "[PHONE_NPU_PRELOAD] layer=%d status=FAIL reason=weight_alloc\n",
            stage.layer);
        return false;
    }
    ggml_backend_buffer_set_usage(
        entry->wbuf.get(),
        GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const struct {
        ggml_tensor * gpu;
        ggml_tensor * npu;
    } weights[] = {
        { gpu_up, entry->w_up },
        { gpu_gate, entry->w_gate },
        { gpu_down, entry->w_down },
    };

    size_t total_bytes = 0;
    for (const auto & weight : weights) {
        std::vector<uint8_t> converted;
        if (!rpc_phone_npu_convert_k_to_q4_0(
                weight.gpu, converted) ||
            converted.size() != ggml_nbytes(weight.npu)) {
            GGML_LOG_ERROR(
                "[PHONE_NPU_PRELOAD] layer=%d status=FAIL reason=weight_convert name=%s\n",
                stage.layer, weight.gpu->name);
            return false;
        }
        ggml_backend_tensor_set(
            weight.npu, converted.data(),
            0, converted.size());
        total_bytes += converted.size();
    }

    cache = entry.get();
    phone_npu_ffn_caches.emplace(
        stage.layer, std::move(entry));

    GGML_LOG_INFO(
        "[PHONE_NPU_PRELOAD] layer=%d format=q4_0 bytes=%zu status=OK\n",
        stage.layer, total_bytes);
    return true;
}

bool rpc_server::run_phone_npu_ffn_stage(
        ggml_cgraph * graph,
        const rpc_phone_npu_ffn_stage & stage,
        std::vector<float> * host_output) {
    if (phone_npu_backend == nullptr || graph == nullptr ||
        stage.norm == nullptr || stage.topk == nullptr ||
        stage.route_weights == nullptr || stage.down == nullptr) {
        return false;
    }

    rpc_phone_npu_ffn_cache * cache = nullptr;
    if (!prepare_phone_npu_ffn_cache(
            graph, stage, cache) || cache == nullptr) {
        return false;
    }

    const int64_t n_embd = stage.norm->ne[0];
    const int64_t n_tokens = stage.norm->ne[1];
    const int64_t n_expert_used = stage.topk->ne[0];

    if (n_embd <= 0 || n_tokens <= 0 ||
        n_expert_used <= 0 ||
        stage.topk->ne[1] != n_tokens ||
        stage.route_weights->ne[0] != 1 ||
        stage.route_weights->ne[1] != n_expert_used ||
        stage.route_weights->ne[2] != n_tokens ||
        cache->w_up->ne[0] != n_embd ||
        cache->w_down->ne[1] != n_embd) {
        return false;
    }

    std::vector<uint8_t> hidden_dense;
    std::vector<uint8_t> topk_dense;
    std::vector<uint8_t> route_dense;
    if (!rpc_phone_npu_read_dense(stage.norm, hidden_dense) ||
        !rpc_phone_npu_read_dense(stage.topk, topk_dense) ||
        !rpc_phone_npu_read_dense(
            stage.route_weights, route_dense)) {
        return false;
    }

    if (hidden_dense.size() !=
            (size_t) n_embd * (size_t) n_tokens * sizeof(float) ||
        topk_dense.size() !=
            (size_t) n_expert_used * (size_t) n_tokens *
                sizeof(int32_t) ||
        route_dense.size() !=
            (size_t) n_expert_used * (size_t) n_tokens *
                sizeof(float)) {
        return false;
    }

    ggml_context_ptr ctx_a = rpc_phone_npu_make_ctx();
    if (!ctx_a) {
        return false;
    }

    ggml_tensor * hidden = ggml_new_tensor_2d(
        ctx_a.get(), GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx_a.get(), GGML_TYPE_I32,
        n_expert_used, n_tokens);
    ggml_tensor * rweights = ggml_new_tensor_3d(
        ctx_a.get(), GGML_TYPE_F32,
        1, n_expert_used, n_tokens);

    ggml_tensor * cur = ggml_reshape_3d(
        ctx_a.get(), hidden, n_embd, 1, n_tokens);
    ggml_tensor * up = ggml_mul_mat_id(
        ctx_a.get(), cache->w_up, cur, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
        ctx_a.get(), cache->w_gate, cur, ids);
    ggml_tensor * act = ggml_swiglu_split(
        ctx_a.get(), gate, up);
    ggml_tensor * down = ggml_mul_mat_id(
        ctx_a.get(), cache->w_down, act, ids);
    ggml_tensor * weighted = ggml_mul(
        ctx_a.get(), down, rweights);

    std::vector<ggml_tensor *> expert_views;
    expert_views.reserve((size_t) n_expert_used);
    for (int64_t e = 0; e < n_expert_used; ++e) {
        expert_views.push_back(
            ggml_view_2d(
                ctx_a.get(), weighted,
                n_embd, n_tokens,
                weighted->nb[2],
                (size_t) e * weighted->nb[1]));
    }

    ggml_tensor * out = expert_views[0];
    for (int64_t e = 1; e < n_expert_used; ++e) {
        out = ggml_add(
            ctx_a.get(), out,
            expert_views[(size_t) e]);
    }
    ggml_set_name(out, "phone_npu_ffn_out");

    ggml_backend_buffer_ptr abuf {
        ggml_backend_alloc_ctx_tensors(
            ctx_a.get(), phone_npu_backend)
    };
    if (!abuf) {
        return false;
    }

    ggml_backend_tensor_set(
        hidden, hidden_dense.data(),
        0, hidden_dense.size());
    ggml_backend_tensor_set(
        ids, topk_dense.data(),
        0, topk_dense.size());
    ggml_backend_tensor_set(
        rweights, route_dense.data(),
        0, route_dense.size());

    ggml_cgraph * gf =
        ggml_new_graph_custom(ctx_a.get(), 2048, false);
    ggml_build_forward_expand(gf, out);

    for (int i = 0; i < gf->n_nodes; ++i) {
        if (!ggml_backend_supports_op(
                phone_npu_backend, gf->nodes[i])) {
            GGML_LOG_ERROR(
                "[PHONE_NPU_EXEC] layer=%d chunk=%d mode=%s status=FAIL unsupported=%s\n",
                stage.layer, stage.chunk,
                stage.prefill ? "prefill" : "decode",
                ggml_op_desc(gf->nodes[i]));
            return false;
        }
    }

    if (ggml_backend_graph_compute(
            phone_npu_backend, gf) != GGML_STATUS_SUCCESS) {
        return false;
    }

    const size_t output_bytes =
        (size_t) ggml_nelements(out) * sizeof(float);
    if (output_bytes != ggml_nbytes(stage.down)) {
        return false;
    }

    std::vector<float> output(
        (size_t) ggml_nelements(out));
    ggml_backend_tensor_get(
        out, output.data(), 0, output_bytes);
    ggml_backend_tensor_set(
        stage.down, output.data(), 0, output_bytes);

    if (host_output != nullptr) {
        *host_output = std::move(output);
    }

    GGML_LOG_INFO(
        "[PHONE_NPU_EXEC] layer=%d chunk=%d mode=%s tokens=%" PRId64 " status=OK\n",
        stage.layer, stage.chunk,
        stage.prefill ? "prefill" : "decode",
        n_tokens);
    return true;
}


bool rpc_server::prepare_phone_npu_async_job(
        uint32_t device,
        uint64_t seq,
        ggml_cgraph * graph,
        const rpc_phone_npu_ffn_stage & stage,
        rpc_phone_npu_async_job & job) {
    if (phone_npu_backend == nullptr || graph == nullptr ||
            seq == 0 ||
            stage.norm == nullptr || stage.topk == nullptr ||
            stage.route_weights == nullptr || stage.down == nullptr) {
        return false;
    }

    rpc_phone_npu_ffn_cache * cache = nullptr;

    // Cached weights are immutable after construction, so a hit does not need
    // the HTP execution mutex.  A miss performs the one-time repack/upload
    // under that mutex to avoid colliding with the running NPU job.
    auto cache_it = phone_npu_ffn_caches.find(stage.layer);
    if (cache_it != phone_npu_ffn_caches.end()) {
        cache = cache_it->second.get();
    } else {
        std::lock_guard<std::mutex> npu_lock(phone_npu_mutex);
        if (!prepare_phone_npu_ffn_cache(
                graph, stage, cache)) {
            return false;
        }
    }
    if (cache == nullptr) {
        return false;
    }

    const int64_t n_embd = stage.norm->ne[0];
    const int64_t n_tokens = stage.norm->ne[1];
    const int64_t n_expert_used = stage.topk->ne[0];

    if (n_embd <= 0 || n_tokens <= 0 ||
            n_expert_used <= 0 ||
            stage.topk->ne[1] != n_tokens ||
            stage.route_weights->ne[0] != 1 ||
            stage.route_weights->ne[1] != n_expert_used ||
            stage.route_weights->ne[2] != n_tokens ||
            cache->w_up->ne[0] != n_embd ||
            cache->w_down->ne[1] != n_embd ||
            stage.down->type != GGML_TYPE_F32 ||
            ggml_nbytes(stage.down) !=
                (size_t) n_embd * (size_t) n_tokens *
                    sizeof(float)) {
        return false;
    }

    rpc_phone_npu_async_job prepared;
    prepared.device = device;
    prepared.seq = seq;
    prepared.layer = stage.layer;
    prepared.chunk = stage.chunk;
    prepared.prefill = stage.prefill;
    prepared.down = stage.down;
    prepared.cache = cache;
    prepared.n_embd = n_embd;
    prepared.n_tokens = n_tokens;
    prepared.n_expert_used = n_expert_used;

    // This readback is the ownership boundary: after it returns, no
    // allocator-backed stage tensor is referenced by the NPU worker.
    if (!rpc_phone_npu_read_dense(
            stage.norm, prepared.hidden_dense) ||
        !rpc_phone_npu_read_dense(
            stage.topk, prepared.topk_dense) ||
        !rpc_phone_npu_read_dense(
            stage.route_weights, prepared.route_dense)) {
        return false;
    }

    if (prepared.hidden_dense.size() !=
            (size_t) n_embd * (size_t) n_tokens *
                sizeof(float) ||
        prepared.topk_dense.size() !=
            (size_t) n_expert_used * (size_t) n_tokens *
                sizeof(int32_t) ||
        prepared.route_dense.size() !=
            (size_t) n_expert_used * (size_t) n_tokens *
                sizeof(float)) {
        return false;
    }

    prepared.enqueue_us = ggml_time_us();
    job = std::move(prepared);
    return true;
}

bool rpc_server::execute_phone_npu_async_job(
        const rpc_phone_npu_async_job & job,
        std::vector<float> & output) {
    if (phone_npu_backend == nullptr ||
            job.cache == nullptr ||
            job.down == nullptr ||
            job.n_embd <= 0 ||
            job.n_tokens <= 0 ||
            job.n_expert_used <= 0) {
        return false;
    }

    ggml_context_ptr ctx_a = rpc_phone_npu_make_ctx();
    if (!ctx_a) {
        return false;
    }

    ggml_tensor * hidden = ggml_new_tensor_2d(
        ctx_a.get(), GGML_TYPE_F32,
        job.n_embd, job.n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(
        ctx_a.get(), GGML_TYPE_I32,
        job.n_expert_used, job.n_tokens);
    ggml_tensor * rweights = ggml_new_tensor_3d(
        ctx_a.get(), GGML_TYPE_F32,
        1, job.n_expert_used, job.n_tokens);

    ggml_tensor * cur = ggml_reshape_3d(
        ctx_a.get(), hidden,
        job.n_embd, 1, job.n_tokens);
    ggml_tensor * up = ggml_mul_mat_id(
        ctx_a.get(), job.cache->w_up, cur, ids);
    ggml_tensor * gate = ggml_mul_mat_id(
        ctx_a.get(), job.cache->w_gate, cur, ids);
    ggml_tensor * act = ggml_swiglu_split(
        ctx_a.get(), gate, up);
    ggml_tensor * down = ggml_mul_mat_id(
        ctx_a.get(), job.cache->w_down, act, ids);
    ggml_tensor * weighted = ggml_mul(
        ctx_a.get(), down, rweights);

    std::vector<ggml_tensor *> expert_views;
    expert_views.reserve((size_t) job.n_expert_used);
    for (int64_t e = 0; e < job.n_expert_used; ++e) {
        expert_views.push_back(
            ggml_view_2d(
                ctx_a.get(), weighted,
                job.n_embd, job.n_tokens,
                weighted->nb[2],
                (size_t) e * weighted->nb[1]));
    }

    ggml_tensor * out = expert_views[0];
    for (int64_t e = 1; e < job.n_expert_used; ++e) {
        out = ggml_add(
            ctx_a.get(), out,
            expert_views[(size_t) e]);
    }
    ggml_set_name(out, "phone_npu_ffn_out_async");

    ggml_backend_buffer_ptr abuf {
        ggml_backend_alloc_ctx_tensors(
            ctx_a.get(), phone_npu_backend)
    };
    if (!abuf) {
        return false;
    }

    ggml_backend_tensor_set(
        hidden,
        job.hidden_dense.data(),
        0,
        job.hidden_dense.size());
    ggml_backend_tensor_set(
        ids,
        job.topk_dense.data(),
        0,
        job.topk_dense.size());
    ggml_backend_tensor_set(
        rweights,
        job.route_dense.data(),
        0,
        job.route_dense.size());

    ggml_cgraph * gf =
        ggml_new_graph_custom(ctx_a.get(), 2048, false);
    ggml_build_forward_expand(gf, out);

    for (int i = 0; i < gf->n_nodes; ++i) {
        if (!ggml_backend_supports_op(
                phone_npu_backend, gf->nodes[i])) {
            GGML_LOG_ERROR(
                "[CHUNK_RING_NPU_ASYNC] "
                "seq=%" PRIu64 " layer=%d chunk=%d "
                "status=FAIL unsupported=%s\n",
                job.seq,
                job.layer,
                job.chunk,
                ggml_op_desc(gf->nodes[i]));
            return false;
        }
    }

    if (ggml_backend_graph_compute(
            phone_npu_backend, gf) !=
            GGML_STATUS_SUCCESS) {
        return false;
    }

    const size_t output_bytes =
        (size_t) ggml_nelements(out) * sizeof(float);
    if (output_bytes != ggml_nbytes(job.down)) {
        return false;
    }

    output.resize((size_t) ggml_nelements(out));
    ggml_backend_tensor_get(
        out,
        output.data(),
        0,
        output_bytes);
    return true;
}

void rpc_server::ensure_phone_npu_async_worker() {
    std::lock_guard<std::mutex> lock(phone_npu_async_mutex);
    if (phone_npu_async_thread.joinable()) {
        return;
    }

    phone_npu_async_stop = false;
    phone_npu_async_thread =
        std::thread([this]() {
            phone_npu_async_loop();
        });
}

void rpc_server::phone_npu_async_loop() {
    for (;;) {
        rpc_phone_npu_async_job job;
        {
            std::unique_lock<std::mutex> lock(
                phone_npu_async_mutex);
            phone_npu_async_cv.wait(
                lock,
                [&]() {
                    return phone_npu_async_stop ||
                        !phone_npu_async_jobs.empty();
                });

            if (phone_npu_async_stop &&
                    phone_npu_async_jobs.empty()) {
                return;
            }

            job = std::move(
                phone_npu_async_jobs.front());
            phone_npu_async_jobs.pop_front();
        }

        const int64_t start_us = ggml_time_us();
        std::vector<float> output;
        bool ok = false;
        {
            // The private HTP backend remains single-owner.  Only this worker
            // uses it for Ring jobs; the mutex also protects legacy/decode
            // synchronous calls.
            std::lock_guard<std::mutex> npu_lock(
                phone_npu_mutex);
            ok = execute_phone_npu_async_job(
                job, output);
        }
        const int64_t done_us = ggml_time_us();

        {
            std::lock_guard<std::mutex> join_lock(
                phone_npu_cpu_reduce_mutex);

            if (ok) {
                rpc_phone_npu_cpu_reduce_entry entry;
                entry.layer = job.layer;
                entry.chunk = job.chunk;
                entry.down = job.down;
                entry.phone_partial = std::move(output);

                auto & pending =
                    phone_npu_cpu_reduce_pending[
                        job.device];
                if (pending.find(job.seq) !=
                        pending.end()) {
                    ok = false;
                } else {
                    pending.emplace(
                        job.seq,
                        std::move(entry));
                }
            }

            if (!ok) {
                phone_npu_cpu_reduce_failed[
                    job.device].insert(job.seq);
            }
        }

        // NPU completion, not the client-side post-graph marker, releases the
        // PC return waiter for Ring-owned seqs.
        {
            std::lock_guard<std::mutex> ready_lock(
                phone_ffn_ready_mutex);
            phone_ffn_ready_seqs[
                job.device].insert(job.seq);
        }
        phone_ffn_ready_cv.notify_all();

        GGML_LOG_INFO(
            "[CHUNK_RING_NPU_ASYNC_DONE] "
            "seq=%" PRIu64 " layer=%d chunk=%d tokens=%" PRId64
            " queue_ms=%.3f compute_ms=%.3f status=%s\n",
            job.seq,
            job.layer,
            job.chunk,
            job.n_tokens,
            (start_us - job.enqueue_us) / 1000.0,
            (done_us - start_us) / 1000.0,
            ok ? "OK" : "FAIL");
    }
}

ggml_status rpc_server::graph_compute_phone_npu_ffn(
        uint32_t device,
        uint64_t graph_uid,
        ggml_cgraph * graph) {
    if (device >= backends.size() || graph == nullptr) {
        return GGML_STATUS_FAILED;
    }

    ggml_backend_t gpu = backends[device];

    uint64_t cpu_reduce_seq = 0;
    {
        std::lock_guard<std::mutex> join_lock(
            phone_npu_cpu_reduce_mutex);
        auto armed_it =
            phone_npu_cpu_reduce_armed.find(device);
        if (armed_it != phone_npu_cpu_reduce_armed.end() &&
                !armed_it->second.empty()) {
            cpu_reduce_seq = armed_it->second.front();
            armed_it->second.pop_front();
            if (armed_it->second.empty()) {
                phone_npu_cpu_reduce_armed.erase(armed_it);
            }
        }
    }

    if (phone_npu_backend == nullptr) {
        if (cpu_reduce_seq != 0) {
            GGML_LOG_ERROR(
                "[CHUNK_RING_CPU_JOIN] seq=%" PRIu64
                " status=FAIL reason=no_private_htp\n",
                cpu_reduce_seq);
            return GGML_STATUS_FAILED;
        }
        return ggml_backend_graph_compute(gpu, graph);
    }

    // TENSOR_PHONE_PRIMARY is already encoded in the graph through the
    // phone_* router markers, so every eligible Tensor FFN is discovered
    // automatically.  No separate layer-range configuration is needed.
    std::vector<rpc_phone_npu_ffn_stage> stages =
        rpc_phone_npu_find_ffn_stages(graph);

    if (stages.empty()) {
        if (cpu_reduce_seq != 0) {
            GGML_LOG_ERROR(
                "[CHUNK_RING_CPU_JOIN] seq=%" PRIu64
                " status=FAIL reason=armed_graph_has_no_htp_stage\n",
                cpu_reduce_seq);
            return GGML_STATUS_FAILED;
        }
        return ggml_backend_graph_compute(gpu, graph);
    }

    // One ring seq describes one (layer, chunk) FFN completion.  Keep this
    // invariant explicit before making the HTP worker asynchronous later.
    if (cpu_reduce_seq != 0 &&
            (stages.size() != 1 ||
             !stages[0].prefill ||
             stages[0].down_index != graph->n_nodes - 1)) {
        GGML_LOG_ERROR(
            "[CHUNK_RING_CPU_JOIN] seq=%" PRIu64
            " status=FAIL reason=armed_graph_shape "
            "stages=%zu down=%d nodes=%d\n",
            cpu_reduce_seq,
            stages.size(),
            stages.empty() ? -1 : stages[0].down_index,
            graph->n_nodes);
        return GGML_STATUS_FAILED;
    }

    int cursor = 0;
    for (const auto & stage : stages) {
        if (stage.front_end < cursor ||
            stage.down_index < stage.front_end ||
            stage.down_index >= graph->n_nodes) {
            if (cpu_reduce_seq != 0) {
                GGML_LOG_ERROR(
                    "[CHUNK_RING_CPU_JOIN] seq=%" PRIu64
                    " layer=%d chunk=%d status=FAIL "
                    "reason=invalid_stage_span front=%d down=%d nodes=%d\n",
                    cpu_reduce_seq,
                    stage.layer,
                    stage.chunk,
                    stage.front_end,
                    stage.down_index,
                    graph->n_nodes);
                return GGML_STATUS_FAILED;
            }
            return ggml_backend_graph_compute(gpu, graph);
        }
        cursor = stage.down_index + 1;
    }

    const bool ring_npu_async =
        std::getenv("GGML_RPC_DISABLE_CHUNK_RING_NPU_ASYNC") == nullptr;

    if (cpu_reduce_seq != 0 && ring_npu_async) {
        GGML_ASSERT(stages.size() == 1);
        const rpc_phone_npu_ffn_stage & stage = stages[0];

        if (stage.front_end > 0) {
            ggml_cgraph front = rpc_graph_view(
                graph, 0, stage.front_end);
            const ggml_status front_status =
                ggml_backend_graph_compute(gpu, &front);
            if (front_status != GGML_STATUS_SUCCESS) {
                return front_status;
            }
        }

        rpc_phone_npu_async_job job;
        if (!prepare_phone_npu_async_job(
                device,
                cpu_reduce_seq,
                graph,
                stage,
                job)) {
            GGML_LOG_ERROR(
                "[CHUNK_RING_NPU_ASYNC] "
                "seq=%" PRIu64 " layer=%d chunk=%d "
                "status=FAIL reason=prepare_job\n",
                cpu_reduce_seq,
                stage.layer,
                stage.chunk);
            return GGML_STATUS_FAILED;
        }

        ensure_phone_npu_async_worker();
        size_t queue_depth = 0;
        {
            std::lock_guard<std::mutex> async_lock(
                phone_npu_async_mutex);
            if (phone_npu_async_stop) {
                return GGML_STATUS_FAILED;
            }
            phone_npu_async_jobs.push_back(
                std::move(job));
            queue_depth = phone_npu_async_jobs.size();
        }
        phone_npu_async_cv.notify_one();

        GGML_LOG_INFO(
            "[CHUNK_RING_NPU_ASYNC_ENQUEUE] "
            "seq=%" PRIu64 " layer=%d chunk=%d tokens=%" PRId64
            " queue_depth=%zu\n",
            cpu_reduce_seq,
            stage.layer,
            stage.chunk,
            stage.norm->ne[1],
            queue_depth);

        // The Ring-owned FFN stage is terminal by construction.  Returning
        // here frees the compute socket so the next Attention chunk can run
        // while the dedicated HTP worker processes this job.
        return GGML_STATUS_SUCCESS;
    }

    std::lock_guard<std::mutex> lock(phone_npu_mutex);

    cursor = 0;
    for (const auto & stage : stages) {
        if (cursor < stage.front_end) {
            ggml_cgraph front = rpc_graph_view(
                graph, cursor, stage.front_end);
            const ggml_status status =
                ggml_backend_graph_compute(gpu, &front);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }
        }

        std::vector<float> ring_phone_partial;
        const bool htp_ok =
            run_phone_npu_ffn_stage(
                graph,
                stage,
                cpu_reduce_seq != 0 ?
                    &ring_phone_partial :
                    nullptr);

        if (!htp_ok) {
            GGML_LOG_ERROR(
                "[PHONE_NPU_EXEC] graph_uid=%" PRIu64
                " layer=%d chunk=%d mode=%s status=FALLBACK_GPU\n",
                graph_uid, stage.layer, stage.chunk,
                stage.prefill ? "prefill" : "decode");

            ggml_cgraph fallback = rpc_graph_view(
                graph, stage.front_end,
                stage.down_index + 1);
            const ggml_status status =
                ggml_backend_graph_compute(gpu, &fallback);
            if (status != GGML_STATUS_SUCCESS) {
                return status;
            }

            if (cpu_reduce_seq != 0) {
                const size_t count =
                    (size_t) ggml_nelements(stage.down);
                const size_t bytes =
                    count * sizeof(float);
                if (stage.down->type != GGML_TYPE_F32 ||
                        bytes != ggml_nbytes(stage.down)) {
                    return GGML_STATUS_FAILED;
                }
                ring_phone_partial.resize(count);
                ggml_backend_tensor_get(
                    stage.down,
                    ring_phone_partial.data(),
                    0,
                    bytes);
            }
        }

        if (cpu_reduce_seq != 0) {
            const size_t expected =
                (size_t) ggml_nelements(stage.down);
            if (stage.down->type != GGML_TYPE_F32 ||
                    ring_phone_partial.size() != expected) {
                GGML_LOG_ERROR(
                    "[CHUNK_RING_CPU_JOIN] seq=%" PRIu64
                    " layer=%d chunk=%d status=FAIL "
                    "reason=phone_partial_shape\n",
                    cpu_reduce_seq,
                    stage.layer,
                    stage.chunk);
                return GGML_STATUS_FAILED;
            }

            rpc_phone_npu_cpu_reduce_entry entry;
            entry.layer = stage.layer;
            entry.chunk = stage.chunk;
            entry.down = stage.down;
            entry.phone_partial =
                std::move(ring_phone_partial);

            {
                std::lock_guard<std::mutex> join_lock(
                    phone_npu_cpu_reduce_mutex);
                auto & pending =
                    phone_npu_cpu_reduce_pending[device];
                if (pending.find(cpu_reduce_seq) !=
                        pending.end()) {
                    GGML_LOG_ERROR(
                        "[CHUNK_RING_CPU_JOIN] seq=%" PRIu64
                        " status=FAIL reason=duplicate_seq\n",
                        cpu_reduce_seq);
                    return GGML_STATUS_FAILED;
                }
                pending.emplace(
                    cpu_reduce_seq,
                    std::move(entry));
            }

            GGML_LOG_INFO(
                "[CHUNK_RING_CPU_NPU_READY] "
                "seq=%" PRIu64 " layer=%d chunk=%d bytes=%zu\n",
                cpu_reduce_seq,
                stage.layer,
                stage.chunk,
                ggml_nbytes(stage.down));
        }

        cursor = stage.down_index + 1;
    }

    if (cursor < graph->n_nodes) {
        ggml_cgraph tail = rpc_graph_view(
            graph, cursor, graph->n_nodes);
        return ggml_backend_graph_compute(gpu, &tail);
    }

    return GGML_STATUS_SUCCESS;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | graph_uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint64_t graph_uid;
    memcpy(&graph_uid, src, sizeof(graph_uid));
    src += sizeof(graph_uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, graph_uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, graph_uid, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false); //计算缓冲区大小
    stored_graph & graph_entry = stored_graphs[device][graph_uid];  //存储计算图以用于复用
    if (graph_entry.buffer.size() < buf_size) {
        graph_entry.buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ graph_entry.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) }; //智能指针
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
    }   //恢复原来的计算图

    // Only for the Cycle live-FFN test. The PC RPC device stub advertises
    // all GGML ops as supported. Validate on the REAL OpenCL device.
    const bool cycle_probe_diag =
        std::getenv("GGML_CYCLE_LIVE_FFN_DIAG") != nullptr;
    if (cycle_probe_diag) {
        const auto real_device = ggml_backend_get_device(backends[device]);
        bool supported = true;
        for (uint32_t i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = graph->nodes[i];
            const bool node_ok =
                node != nullptr &&
                ggml_backend_dev_supports_op(real_device, node);
            GGML_LOG_INFO(
                "[CYCLE_LIVE_FFN_REMOTE_OP] uid=%" PRIu64
                " node=%u op=%s type=%s supported=%d tensor=%s\n",
                graph_uid, i,
                node != nullptr ? ggml_op_name(node->op) : "(null)",
                node != nullptr ? ggml_type_name(node->type) : "(null)",
                node_ok ? 1 : 0,
                node != nullptr ? node->name : "(null)");
            supported = supported && node_ok;
        }
        if (!supported) {
            GGML_LOG_ERROR(
                "[CYCLE_LIVE_FFN_REMOTE] uid=%" PRIu64
                " status=REJECT reason=UNSUPPORTED_GPU_OP\n", graph_uid);
            return false;
        }
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_REMOTE] uid=%" PRIu64
            " stage=COMPUTE_BEGIN nodes=%u\n", graph_uid, n_nodes);
    }
    const int64_t t0 = ggml_time_us();

    ggml_status status =
        graph_compute_phone_npu_ffn(
            device,
            graph_uid,
            graph);

    if (cycle_probe_diag) {
        GGML_LOG_INFO(
            "[CYCLE_LIVE_FFN_REMOTE] uid=%" PRIu64
            " stage=COMPUTE_END status=%d ms=%.3f\n",
            graph_uid, (int) status,
            (ggml_time_us() - t0) / 1000.0);
        if (status != GGML_STATUS_SUCCESS) {
            return false;
        }
    }

    const int64_t t1 = ggml_time_us();

    // This timer runs on the actual Phone RPC server and includes the
    // synchronous backend graph execution. Unlike the client-side
    // FFN_DESTINATION timer, it is NOT merely RPC graph submission.
    // Classify the single-layer Cycle test graphs by their node shape;
    // log UNKNOWN rather than attributing a different graph incorrectly.
    if (std::getenv("GGML_CYCLE_GPU_STAGE_TIMING") != nullptr) {
        const char * phase = "OTHER";
        if (n_nodes == 2) {
            phase = "REDUCE";
        } else if (n_nodes >= 20 && n_nodes <= 26) {
            phase = "FFN";
        } else if (n_nodes >= 40 && n_nodes <= 70) {
            phase = "ATTN";
        }
        GGML_LOG_INFO(
            "[CYCLE_GPU_SERVER] phase_hint=%s graph_uid=%" PRIu64
            " nodes=%u gpu_compute_wall_ms=%.3f "
            "source=PHONE_SERVER_SYNC_BACKEND_GRAPH status=%s\n",
            phase, graph_uid, n_nodes, (t1 - t0) / 1000.0,
            status == GGML_STATUS_SUCCESS ? "OK" : "FAIL");
    }

    rpc_tensor_phone_log_graph_compute(
        "compute",
        graph_uid,
        graph,
        t1 - t0,
        0);

    if (RPC_DEBUG || std::getenv("GGML_RETURN_PATH_DEBUG") != nullptr) {
        printf(
            "[RPC_GRAPH_SERVER] uid=%" PRIu64
            " nodes=%d compute=%.3f ms\n",
            graph_uid,
            graph->n_nodes,
            (t1 - t0) / 1000.0);
        fflush(stdout);
    }
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    graph_entry.graph = graph;
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    auto it = stored_graphs[device].find(request.graph_uid);
    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        return false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, graph_uid: %" PRIu64 "\n", __func__, device, request.graph_uid);
    const int64_t t0 = ggml_time_us();

    ggml_status status =
        graph_compute_phone_npu_ffn(
            device,
            request.graph_uid,
            graph);

    const int64_t t1 = ggml_time_us();

    if (std::getenv("GGML_CYCLE_GPU_STAGE_TIMING") != nullptr) {
        GGML_LOG_INFO(
            "[CYCLE_GPU_SERVER] phase_hint=RECOMPUTE "
            "graph_uid=%" PRIu64 " nodes=%d gpu_compute_wall_ms=%.3f "
            "source=PHONE_SERVER_SYNC_BACKEND_GRAPH status=%s\n",
            request.graph_uid, graph->n_nodes, (t1 - t0) / 1000.0,
            status == GGML_STATUS_SUCCESS ? "OK" : "FAIL");
    }

    rpc_tensor_phone_log_graph_compute(
        "recompute",
        request.graph_uid,
        graph,
        t1 - t0,
        0);

    if (RPC_DEBUG || std::getenv("GGML_RETURN_PATH_DEBUG") != nullptr) {
        printf(
            "[RPC_GRAPH_SERVER] uid=%" PRIu64
            " nodes=%d compute=%.3f ms\n",
            request.graph_uid,
            graph->n_nodes,
            (t1 - t0) / 1000.0);
        fflush(stdout);
    }
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

bool rpc_server::graph_recompute_snapshot(
        const rpc_msg_graph_recompute_snapshot_req & request) {
    if (request.device >= backends.size() || request.slot >= 2) {
        return false;
    }

    auto it = stored_graphs[request.device].find(request.graph_uid);
    if (it == stored_graphs[request.device].end() || it->second.graph == nullptr) {
        return false;
    }

    ggml_cgraph * graph = it->second.graph;
    if (request.output_node >= static_cast<uint32_t>(graph->n_nodes)) {
        return false;
    }

    const int64_t t0 = ggml_time_us();
    const ggml_status status =
        graph_compute_phone_npu_ffn(
            request.device,
            request.graph_uid,
            graph);
    const int64_t t1 = ggml_time_us();
    if (status != GGML_STATUS_SUCCESS) {
        return false;
    }

    ggml_tensor * output = graph->nodes[request.output_node];
    const bool snapshot_ok = snapshot_tensor_direct(
        request.device,
        request.slot,
        request.seq,
        output,
        request.offset,
        request.size);
    const int64_t t2 = ggml_time_us();

    rpc_tensor_phone_log_graph_compute(
        "recompute_snapshot",
        request.graph_uid,
        graph,
        t1 - t0,
        t2 - t1);

    if (RPC_DEBUG || std::getenv("GGML_RETURN_PATH_DEBUG") != nullptr) {
        printf(
            "[RPC_GRAPH_SNAPSHOT_SERVER] uid=%" PRIu64
            " node=%u seq=%" PRIu64
            " compute=%.3f ms snapshot=%.3f ms\n",
            request.graph_uid,
            request.output_node,
            request.seq,
            (t1 - t0) / 1000.0,
            (t2 - t1) / 1000.0);
        fflush(stdout);
    }

    return snapshot_ok;
}

bool rpc_server::synchronize(uint32_t device) {
    if (device >= backends.size()) {
        return false;
    }

    ggml_backend_t backend = backends[device];
    ggml_backend_dev_t dev =
        ggml_backend_get_device(backend);

    size_t released_payloads = 0;
    {
        std::lock_guard<std::mutex> lock(
            async_return_payload_mutex);
        ggml_backend_synchronize(backend);

        if (dev != nullptr) {
            auto it = async_return_payloads.find(dev);
            if (it != async_return_payloads.end()) {
                released_payloads = it->second.size();
                async_return_payloads.erase(it);
            }
        }
    }

    if (released_payloads != 0 &&
            rpc_tensor_phone_stage_profile_enabled()) {
        GGML_LOG_INFO(
            "[TENSOR_PHONE_RPC_STAGE] side=phone "
            "stage=async_return_payload_release "
            "device=%u count=%zu\n",
            device,
            released_payloads);
    }
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;

    const double mib = 1024.0 * 1024.0;
    printf(
        "[RPC_DEVICE_MEMORY] device=%u backend=%s description=%s "
        "free=%" PRIu64 " total=%" PRIu64 " free_mib=%.1f total_mib=%.1f\n",
        dev_id,
        ggml_backend_dev_name(dev),
        ggml_backend_dev_description(dev),
        response.free_mem,
        response.total_mem,
        response.free_mem / mib,
        response.total_mem / mib);
    fflush(stdout);

    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

rpc_server::~rpc_server() {
    if (auto transport = get_inference_data_plane()) {
        transport->stop();
    }

    {
        std::lock_guard<std::mutex> lock(
            inference_route_mutex);
        inference_route_stop = true;
    }
    inference_route_cv.notify_all();
    if (inference_route_thread.joinable()) {
        inference_route_thread.join();
    }

    {
        std::lock_guard<std::mutex> lock(
            phone_npu_async_mutex);
        phone_npu_async_stop = true;
    }
    phone_npu_async_cv.notify_all();
    if (phone_npu_async_thread.joinable()) {
        phone_npu_async_thread.join();
    }

    {
        std::lock_guard<std::mutex> lock(
            phone_npu_cpu_reduce_mutex);
        phone_npu_cpu_reduce_armed.clear();
        phone_npu_cpu_reduce_pending.clear();
        phone_npu_cpu_reduce_active.clear();
        phone_npu_cpu_reduce_marker_pending.clear();
        phone_npu_cpu_reduce_failed.clear();
    }
    phone_npu_ffn_caches.clear();
    for (size_t lane = 0; lane < snapshot_breakdown.size(); ++lane) {
        const auto & stats = snapshot_breakdown[lane];
        if (stats.requests == 0) {
            continue;
        }
        printf(
            "[SNAPSHOT_BREAKDOWN] lane=%zu requests=%" PRIu64 " ready_wait=%.3f send=%.3f bytes=%" PRIu64 "\n",
            lane, stats.requests, stats.ready_wait_us / 1000.0, stats.send_us / 1000.0, stats.bytes);
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

//新加的函数
static bool range_2d_fits(
        uint64_t offset,
        uint64_t copy_size,
        uint64_t n_copies,
        uint64_t stride,
        uint64_t limit) {

    // 空复制不访问任何数据
    if (n_copies == 0 || copy_size == 0) {
        return offset <= limit;
    }

    if (offset > limit) {
        return false;
    }

    // 先检查第一次复制，避免减法下溢
    if (copy_size > limit - offset) {
        return false;
    }

    if (n_copies == 1) {
        return true;
    }

    // 最后一次复制的结束位置：
    // offset + (n_copies - 1) * stride + copy_size
    //
    // 使用除法形式避免乘法和加法溢出。
    const uint64_t remaining =
        limit - offset - copy_size;

    return stride <= remaining / (n_copies - 1);
}

bool rpc_server::set_tensor_from_local_file(
        const rpc_msg_set_tensor_from_local_file_req & request,
 rpc_msg_set_tensor_from_local_file_rsp & response) {

    struct ggml_init_params params {
        /* .mem_size   = */ ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    response.result = 2;

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);

    ggml_tensor * tensor =
        deserialize_tensor(ctx_ptr.get(), &request.tensor);

    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR(
            "[%s] error deserializing tensor\n",
            __func__);
        return false;
    }

    if (local_tensor_source.read == nullptr) {
        GGML_LOG_ERROR(
            "[%s] no local tensor source registered\n",
            __func__);
        response.result = 0;
        return true;
    }

    const size_t name_len =
        strnlen(request.tensor.name, GGML_MAX_NAME);

    if (name_len == GGML_MAX_NAME) {
        GGML_LOG_ERROR(
            "[%s] tensor name is not null-terminated\n",
            __func__);
        return false;
    }

    const std::string name(
        request.tensor.name,
        name_len);

    /*
     * 空操作。
     *
     * 与 set_tensor_2d 的语义保持一致：
     * 没有复制次数或单次大小为 0 时，不访问源和目标。
     */
    if (request.n_copies == 0 ||
        request.copy_size == 0) {
        response.result = 1;
        return true;
    }

    /*
     * 后面需要把 copy_size 和目标偏移传给接受 size_t 的
     * ggml_backend_tensor_set，因此先检查是否可表示。
     */
    if (request.copy_size > SIZE_MAX ||
        request.dst_offset > SIZE_MAX ||
        request.dst_stride > SIZE_MAX) {

        GGML_LOG_ERROR(
            "[%s] request values exceed size_t: "
            "name='%s'\n",
            __func__,
            name.c_str());
        return false;
    }

    const uint64_t tensor_nbytes =
        static_cast<uint64_t>(ggml_nbytes(tensor));

    // OpenCL quantized set_tensor performs layout conversion and replaces the
    // generic extra with a type-specific object. It is therefore not a safe
    // partial/repeated write primitive. Gather a complete destination before
    // writing once; other backends keep the existing 2D behavior unchanged.
    const bool opencl_quantized =
        is_opencl_tensor(tensor) && ggml_is_quantized(tensor->type);
    const bool complete_destination =
        request.dst_offset == 0 &&
        (request.n_copies == 1 || request.dst_stride == request.copy_size) &&
        tensor_nbytes % request.copy_size == 0 &&
        tensor_nbytes / request.copy_size == request.n_copies;
    if (opencl_quantized && !complete_destination) {
        RPC_OPENCL_EXTRA_DBG(
            "[RPC_OPENCL_EXTRA] action=REJECT_PARTIAL_QUANT_SET "
            "name=%s type=%d dst_offset=%" PRIu64
            " copy_size=%" PRIu64 " n_copies=%" PRIu64
            " tensor_size=%" PRIu64 "\n",
            tensor->name,
            (int) tensor->type,
            request.dst_offset,
            request.copy_size,
            request.n_copies,
            tensor_nbytes);
        response.result = 2;
        return true;
    }

    if (!ensure_opencl_tensor_extra(tensor)) {
        GGML_LOG_ERROR(
            "[%s] failed to initialize OpenCL tensor state for '%s'\n",
            __func__, tensor->name);
        return false;
    }

    /*
     * 验证最后一次目标写入不会越过局部目标张量。
     *
     * 最后写入范围：
     * dst_offset
     * + (n_copies - 1) * dst_stride
     * + copy_size
     */
    if (!range_2d_fits(
            request.dst_offset,
            request.copy_size,
            request.n_copies,
            request.dst_stride,
            tensor_nbytes)) {

        GGML_LOG_ERROR(
            "[%s] destination 2D range is out of bounds: "
            "name='%s', dst_offset=%" PRIu64
            ", copy_size=%" PRIu64
            ", n_copies=%" PRIu64
            ", dst_stride=%" PRIu64
            ", tensor_size=%" PRIu64 "\n",
            __func__,
            name.c_str(),
            request.dst_offset,
            request.copy_size,
            request.n_copies,
            request.dst_stride,
            tensor_nbytes);
        return false;
    }

    /*
     * 查询服务端本地完整源张量的信息，
     * 检查最后一次源读取是否越界。
     */
    if (local_tensor_source.get_info != nullptr) {
        ggml_rpc_local_tensor_info info {};

        if (!local_tensor_source.get_info(
                local_tensor_source.user_data,
                name.c_str(),
                &info)) {

            GGML_LOG_ERROR(
                "[%s] local tensor '%s' not found\n",
                __func__,
                name.c_str());
            return false;
        }

        if (!range_2d_fits(
                request.src_offset,
                request.copy_size,
                request.n_copies,
                request.src_stride,
                info.nbytes)) {

            GGML_LOG_ERROR(
                "[%s] source 2D range is out of bounds: "
                "name='%s', src_offset=%" PRIu64
                ", copy_size=%" PRIu64
                ", n_copies=%" PRIu64
                ", src_stride=%" PRIu64
                ", tensor_size=%" PRIu64 "\n",
                __func__,
                name.c_str(),
                request.src_offset,
                request.copy_size,
                request.n_copies,
                request.src_stride,
                info.nbytes);
            return false;
        }
    }

    // Even without source metadata, do not allow source offset arithmetic to wrap.
    if (opencl_quantized && !range_2d_fits(
            request.src_offset, request.copy_size, request.n_copies,
            request.src_stride, UINT64_MAX)) {
        return false;
    }

    // Quantized OpenCL writes need the whole shard for layout conversion.
    // Other backends retain one-row staging and incremental writes.
    const size_t staging_size = static_cast<size_t>(
        opencl_quantized ? tensor_nbytes : request.copy_size);
    std::vector<uint8_t> staging;

    try {
        staging.resize(staging_size);
    } catch (const std::bad_alloc &) {
        GGML_LOG_ERROR(
            "[%s] failed to allocate %" PRIu64
            " staging bytes\n",
            __func__,
            static_cast<uint64_t>(staging_size));
        return false;
    }

    for (uint64_t i = 0;
         i < request.n_copies;
         ++i) {

        /*
         * 这些乘加已经由前面的 range_2d_fits()
         * 验证不会越界或溢出。
         */
        const uint64_t src_i =
            request.src_offset +
            i * request.src_stride;

        const uint64_t dst_i =
            request.dst_offset +
            i * request.dst_stride;

        if (dst_i > SIZE_MAX) {
            GGML_LOG_ERROR(
                "[%s] destination offset exceeds size_t: "
                "name='%s', copy=%" PRIu64
                ", dst=%" PRIu64 "\n",
                __func__,
                name.c_str(),
                i,
                dst_i);
            return false;
        }

        /*
         * 从服务端本地完整源张量的第 src_i 字节读取。
         *
         * read() 中应当根据 name 找到完整源张量，
         * 再从它的相对偏移 src_i 开始读取。
         */
        if (!local_tensor_source.read(
                local_tensor_source.user_data,
                name.c_str(),
                src_i,
                staging.data() + (opencl_quantized ? static_cast<size_t>(dst_i) : 0),
                static_cast<size_t>(request.copy_size))) {

            GGML_LOG_ERROR(
                "[%s] failed to read local tensor: "
                "name='%s', copy=%" PRIu64
                ", src=%" PRIu64
                ", size=%" PRIu64 "\n",
                __func__,
                name.c_str(),
                i,
                src_i,
                request.copy_size);
            return false;
        }

        /*
         * tensor->data 已经是远端目标张量基址。
         * dst_i 是相对于该基址的目标偏移。
         */
        if (!opencl_quantized) {
            ggml_backend_tensor_set(
                tensor,
                staging.data(),
                static_cast<size_t>(dst_i),
                staging.size());
            remember_opencl_tensor_extra(tensor);
        }
    }

    if (opencl_quantized) {
        ggml_backend_tensor_set(tensor, staging.data(), 0, staging.size());
        remember_opencl_tensor_extra(tensor);
        const char * verify = std::getenv("GGML_RPC_VERIFY_QUANT_WEIGHTS");
        if (verify != nullptr &&
            (std::strcmp(verify, "1") == 0 || std::strstr(tensor->name, verify) != nullptr)) {
            std::vector<uint8_t> readback(staging.size());
            ggml_backend_tensor_get(tensor, readback.data(), 0, readback.size());
            const auto mismatch = std::mismatch(staging.begin(), staging.end(), readback.begin());
            if (mismatch.first != staging.end()) {
                const size_t pos = static_cast<size_t>(mismatch.first - staging.begin());
                GGML_LOG_DEBUG(
                    "[RPC_OPENCL_EXTRA] action=VERIFY_QUANT_MISMATCH "
                    "name=%s type=%d offset=%zu expected=%u actual=%u bytes=%zu\n",
                    tensor->name, (int) tensor->type, pos,
                    (unsigned) staging[pos], (unsigned) readback[pos], staging.size());
                response.result = 2;
                return true;
            }
            GGML_LOG_DEBUG(
                "[RPC_OPENCL_EXTRA] action=VERIFY_QUANT_OK name=%s type=%d bytes=%zu\n",
                tensor->name, (int) tensor->type, staging.size());
        }
        if (rpc_opencl_extra_debug_enabled()) {
            RPC_OPENCL_EXTRA_DBG(
                "[RPC_OPENCL_EXTRA] action=SET_QUANT_FULL_STAGED "
                "name=%s bytes=%zu n_copies=%" PRIu64 "\n",
                tensor->name, staging.size(), request.n_copies);
        }
    }

    GGML_LOG_DEBUG(
        "[%s] loaded local tensor: "
        "name='%s', src_offset=%" PRIu64
        ", dst_offset=%" PRIu64
        ", copy_size=%" PRIu64
        ", n_copies=%" PRIu64
        ", src_stride=%" PRIu64
        ", dst_stride=%" PRIu64 "\n",
        __func__,
        name.c_str(),
        request.src_offset,
        request.dst_offset,
        request.copy_size,
        request.n_copies,
        request.src_stride,
        request.dst_stride);
    
    response.result = 1;
    
    return true;
}/*
static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}
*/

static void rpc_serve_client(std::shared_ptr<rpc_server> server_ptr, socket_ptr sock, size_t device_count) {
    rpc_server & server = *server_ptr;
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);

    bool return_bulk_credit_held = false;
    uint64_t return_bulk_credit_seq = 0;
    int64_t return_bulk_credit_wait_us = 0;
    size_t return_bulk_credit_inflight = 0;
    size_t return_bulk_credit_limit = 0;

    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            if (return_bulk_credit_held) {
                server.release_route_bulk_credit();
                return_bulk_credit_held = false;
            }
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            if (return_bulk_credit_held) {
                server.release_route_bulk_credit();
                return_bulk_credit_held = false;
            }
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        if (return_bulk_credit_held &&
                cmd != RPC_CMD_SET_TENSOR_ASYNC_RETURN_WAIT) {
            GGML_LOG_ERROR(
                "[RETURN_BULK_SRV] seq=%" PRIu64
                " status=FAIL reason=unexpected_cmd cmd=%u\n",
                return_bulk_credit_seq,
                static_cast<unsigned>(cmd));
            server.release_route_bulk_credit();
            return_bulk_credit_held = false;
            return;
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = device_count;
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_BATCH3: {
                const bool route_profile =
                    std::getenv("GGML_RPC_ROUTE_STAGE_PROFILE") != nullptr;
                rpc_route_batch_server_stage = {};
                const int64_t request_begin_us =
                    route_profile ? ggml_time_us() : 0;

                rpc_msg_get_tensor_batch3_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                const int64_t request_done_us =
                    route_profile ? ggml_time_us() : 0;

                std::vector<uint8_t> response;
                const int64_t handler_begin_us =
                    route_profile ? ggml_time_us() : 0;
                if (!server.get_tensor_batch3(request, response)) {
                    return;
                }
                const int64_t handler_done_us =
                    route_profile ? ggml_time_us() : 0;

                const int64_t send_begin_us =
                    route_profile ? ggml_time_us() : 0;
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                const int64_t send_done_us =
                    route_profile ? ggml_time_us() : 0;

                if (route_profile &&
                        rpc_route_batch_server_stage.valid) {
                    const auto & p =
                        rpc_route_batch_server_stage;
                    std::fprintf(
                        stderr,
                        "[RPC_ROUTE_STAGE] side=phone stage=router_batch "
                        "bytes=%zu wait_seq=%" PRIu64 " wait_ms=%.3f "
                        "opencl_batch=%d prepare_ms=%.3f device_read_ms=%.3f "
                        "request_recv_ms=%.3f handler_ms=%.3f "
                        "response_send_ms=%.3f total_dispatch_ms=%.3f\n",
                        p.bytes,
                        p.wait_seq,
                        p.wait_us / 1000.0,
                        p.opencl_batch_used ? 1 : 0,
                        p.prepare_us / 1000.0,
                        p.device_read_us / 1000.0,
                        (request_done_us - request_begin_us) / 1000.0,
                        (handler_done_us - handler_begin_us) / 1000.0,
                        (send_done_us - send_begin_us) / 1000.0,
                        (send_done_us - request_begin_us) / 1000.0);
                    std::fflush(stderr);
                }
                break;
            }
            case RPC_CMD_ROUTE_MARK_READY: {
                rpc_msg_route_mark_ready_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.route_mark_ready(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_PHONE_FFN_MARK_READY: {
                rpc_msg_phone_ffn_mark_ready_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.phone_ffn_mark_ready(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_BATCH3_WAIT: {
                const bool route_profile =
                    std::getenv("GGML_RPC_ROUTE_STAGE_PROFILE") != nullptr;
                rpc_route_batch_server_stage = {};
                const int64_t request_begin_us =
                    route_profile ? ggml_time_us() : 0;

                rpc_msg_get_tensor_batch3_wait_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                const int64_t request_done_us =
                    route_profile ? ggml_time_us() : 0;

                std::vector<uint8_t> response;
                const int64_t handler_begin_us =
                    route_profile ? ggml_time_us() : 0;
                if (!server.get_tensor_batch3_wait(request, response)) {
                    return;
                }
                const int64_t handler_done_us =
                    route_profile ? ggml_time_us() : 0;

                const int64_t send_begin_us =
                    route_profile ? ggml_time_us() : 0;
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                const int64_t send_done_us =
                    route_profile ? ggml_time_us() : 0;

                if (route_profile &&
                        rpc_route_batch_server_stage.valid) {
                    const auto & p =
                        rpc_route_batch_server_stage;
                    std::fprintf(
                        stderr,
                        "[RPC_ROUTE_STAGE] side=phone stage=router_batch "
                        "bytes=%zu wait_seq=%" PRIu64 " wait_ms=%.3f "
                        "opencl_batch=%d prepare_ms=%.3f device_read_ms=%.3f "
                        "request_recv_ms=%.3f handler_ms=%.3f "
                        "response_send_ms=%.3f total_dispatch_ms=%.3f\n",
                        p.bytes,
                        p.wait_seq,
                        p.wait_us / 1000.0,
                        p.opencl_batch_used ? 1 : 0,
                        p.prepare_us / 1000.0,
                        p.device_read_us / 1000.0,
                        (request_done_us - request_begin_us) / 1000.0,
                        (handler_done_us - handler_begin_us) / 1000.0,
                        (send_done_us - send_begin_us) / 1000.0,
                        (send_done_us - request_begin_us) / 1000.0);
                    std::fflush(stderr);
                }
                break;
            }
            case RPC_CMD_ROUTE_SNAPSHOT_READY: {
                const bool stage_profile =
                    rpc_tensor_phone_stage_profile_enabled();
                const int64_t request_recv_begin_us =
                    stage_profile ? ggml_time_us() : 0;

                rpc_msg_route_snapshot_ready_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                const int64_t request_recv_done_us =
                    stage_profile ? ggml_time_us() : 0;

                rpc_msg_route_snapshot_ready_rsp response {};
                const int64_t handler_begin_us =
                    stage_profile ? ggml_time_us() : 0;
                if (!server.route_snapshot_ready(request, response)) {
                    return;
                }
                const int64_t handler_done_us =
                    stage_profile ? ggml_time_us() : 0;

                const int64_t response_send_begin_us =
                    stage_profile ? ggml_time_us() : 0;
                if (!send_msg(
                        sock,
                        &response,
                        sizeof(response))) {
                    return;
                }
                const int64_t response_send_done_us =
                    stage_profile ? ggml_time_us() : 0;

                if (stage_profile) {
                    GGML_LOG_INFO(
                        "[TENSOR_PHONE_RPC_STAGE] side=phone "
                        "stage=route_snapshot_ready_server lane=%u "
                        "seq=%" PRIu64 " request_bytes=%zu response_bytes=%zu "
                        "request_recv_ms=%.3f handler_ms=%.3f "
                        "response_send_ms=%.3f total_ms=%.3f "
                        "result=%d\n",
                        request.lane,
                        request.seq,
                        sizeof(request),
                        sizeof(response),
                        (request_recv_done_us -
                            request_recv_begin_us) / 1000.0,
                        (handler_done_us -
                            handler_begin_us) / 1000.0,
                        (response_send_done_us -
                            response_send_begin_us) / 1000.0,
                        (response_send_done_us -
                            request_recv_begin_us) / 1000.0,
                        response.result != 0 ? 1 : 0);
                    std::fflush(stderr);
                }
                break;
            }
            case RPC_CMD_ROUTE_SNAPSHOT_READY_ASYNC: {
                const bool stage_profile =
                    rpc_tensor_phone_stage_profile_enabled();
                const int64_t request_recv_begin_us =
                    stage_profile ? ggml_time_us() : 0;

                rpc_msg_route_snapshot_ready_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                const int64_t request_recv_done_us =
                    stage_profile ? ggml_time_us() : 0;

                rpc_msg_route_snapshot_ready_rsp response {};
                const int64_t handler_begin_us =
                    stage_profile ? ggml_time_us() : 0;
                if (!server.route_snapshot_ready(request, response)) {
                    return;
                }
                const int64_t handler_done_us =
                    stage_profile ? ggml_time_us() : 0;

                if (stage_profile) {
                    GGML_LOG_INFO(
                        "[TENSOR_PHONE_RPC_STAGE] side=phone "
                        "stage=route_snapshot_ready_async_server lane=%u "
                        "seq=%" PRIu64 " request_bytes=%zu response_bytes=0 "
                        "request_recv_ms=%.3f handler_ms=%.3f "
                        "response_send_ms=0.000 total_ms=%.3f "
                        "result=%d\n",
                        request.lane,
                        request.seq,
                        sizeof(request),
                        (request_recv_done_us -
                            request_recv_begin_us) / 1000.0,
                        (handler_done_us -
                            handler_begin_us) / 1000.0,
                        (handler_done_us -
                            request_recv_begin_us) / 1000.0,
                        response.result != 0 ? 1 : 0);
                    std::fflush(stderr);
                }
                break;
            }
            case RPC_CMD_GET_ROUTE_SNAPSHOT: {
                const bool stage_profile =
                    rpc_tensor_phone_stage_profile_enabled();
                const bool route_profile =
                    std::getenv("GGML_RPC_ROUTE_STAGE_PROFILE") != nullptr;
                const bool measure_profile =
                    stage_profile || route_profile;
                const int64_t request_recv_begin_us =
                    measure_profile ? ggml_time_us() : 0;

                rpc_msg_get_route_snapshot_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                const int64_t request_recv_done_us =
                    measure_profile ? ggml_time_us() : 0;

                std::vector<uint8_t> response;
                const int64_t dispatch_begin_us =
                    measure_profile ? ggml_time_us() : 0;
                if (!server.get_route_snapshot(request, response)) {
                    return;
                }
                const int64_t dispatch_done_us =
                    measure_profile ? ggml_time_us() : 0;

                // Preserve send_msg() wire semantics exactly, but expose the
                // response-size header and payload writes separately.  This
                // lets the client-side response_header_wait be reconciled with
                // server wait/copy time without changing the RPC protocol.
                const uint64_t out_size =
                    static_cast<uint64_t>(response.size());
                const bool wire_ack_supported =
                    (request.sizes[2] &
                     RPC_ROUTE_SNAPSHOT_WIRE_ACK_FLAG) != 0;

                // Do not gate the request or the READY wait. Multiple GETs
                // remain pre-armed and all four snapshot slots remain usable.
                // Only READY snapshots entering the large TCP response are
                // credit-limited.
                const int64_t credit_wait_begin_us =
                    measure_profile ? ggml_time_us() : 0;
                const size_t bulk_inflight =
                    server.acquire_route_bulk_credit();
                const int64_t credit_wait_done_us =
                    measure_profile ? ggml_time_us() : 0;
                const size_t bulk_limit =
                    server.route_bulk_transfer_limit();
                const bool wire_ack_required =
                    wire_ack_supported &&
                    server.route_bulk_wire_ack_required();
                const uint64_t wire_out_size =
                    out_size |
                    (wire_ack_required ?
                        RPC_ROUTE_SNAPSHOT_WIRE_ACK_FLAG :
                        UINT64_C(0));

                const int64_t header_send_begin_us =
                    measure_profile ? ggml_time_us() : 0;
                if (!sock->send_data(
                        &wire_out_size,
                        sizeof(wire_out_size))) {
                    server.release_route_bulk_credit();
                    return;
                }
                const int64_t header_send_done_us =
                    measure_profile ? ggml_time_us() : 0;

                if (!response.empty() &&
                        !sock->send_data(
                            response.data(),
                            response.size())) {
                    server.release_route_bulk_credit();
                    return;
                }
                const int64_t payload_send_done_us =
                    measure_profile ? ggml_time_us() : 0;

                int64_t ack_done_us = payload_send_done_us;
                if (wire_ack_required) {
                    uint64_t ack_seq = 0;
                    if (!sock->recv_data(
                            &ack_seq,
                            sizeof(ack_seq))) {
                        server.release_route_bulk_credit();
                        return;
                    }
                    ack_done_us =
                        measure_profile ? ggml_time_us() : payload_send_done_us;
                    if (ack_seq != request.seq) {
                        GGML_LOG_ERROR(
                            "[RPC_ROUTE_BULK_ACK_MISMATCH] "
                            "expected=%" PRIu64 " actual=%" PRIu64 "\n",
                            request.seq,
                            ack_seq);
                        server.release_route_bulk_credit();
                        return;
                    }
                }
                server.release_route_bulk_credit();

                if (route_profile) {
                    std::fprintf(
                        stderr,
                        "[ROUTE_SRV_RAW] seq=%" PRIu64 " lane=%u bytes=%zu "
                        "recv=%.3f queue=%.3f get=%.3f "
                        "credit=%.3f inflight=%zu/%zu "
                        "hsend=%.3f psend=%.3f ack=%.3f hold=%.3f "
                        "total=%.3f\n",
                        request.seq,
                        request.lane,
                        response.size(),
                        (request_recv_done_us -
                            request_recv_begin_us) / 1000.0,
                        (dispatch_begin_us -
                            request_recv_done_us) / 1000.0,
                        (dispatch_done_us -
                            dispatch_begin_us) / 1000.0,
                        (credit_wait_done_us -
                            credit_wait_begin_us) / 1000.0,
                        bulk_inflight,
                        bulk_limit,
                        (header_send_done_us -
                            header_send_begin_us) / 1000.0,
                        (payload_send_done_us -
                            header_send_done_us) / 1000.0,
                        (ack_done_us -
                            payload_send_done_us) / 1000.0,
                        (ack_done_us -
                            header_send_begin_us) / 1000.0,
                        (payload_send_done_us -
                            header_send_begin_us) / 1000.0);
                }

                if (stage_profile) {
                    GGML_LOG_INFO(
                        "[ROUTE_SRV] seq=%" PRIu64 " lane=%u bytes=%zu "
                        "recv=%.3f queue=%.3f get=%.3f "
                        "credit=%.3f inflight=%zu/%zu "
                        "hsend=%.3f psend=%.3f ack=%.3f hold=%.3f "
                        "total=%.3f\n",
                        request.seq,
                        request.lane,
                        response.size(),
                        (request_recv_done_us -
                            request_recv_begin_us) / 1000.0,
                        (dispatch_begin_us -
                            request_recv_done_us) / 1000.0,
                        (dispatch_done_us -
                            dispatch_begin_us) / 1000.0,
                        (credit_wait_done_us -
                            credit_wait_begin_us) / 1000.0,
                        bulk_inflight,
                        bulk_limit,
                        (header_send_done_us -
                            header_send_begin_us) / 1000.0,
                        (payload_send_done_us -
                            header_send_done_us) / 1000.0,
                        (ack_done_us -
                            payload_send_done_us) / 1000.0,
                        (ack_done_us -
                            header_send_begin_us) / 1000.0,
                        (payload_send_done_us -
                            header_send_begin_us) / 1000.0);
                }
                break;
            }
            case RPC_CMD_RETURN_BULK_ACQUIRE: {
                rpc_msg_return_bulk_acquire_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (request.seq == 0 || return_bulk_credit_held) {
                    return;
                }

                rpc_msg_return_bulk_acquire_rsp response {};
                response.active =
                    server.network_bulk_shared_enabled() ? 1 : 0;
                if (response.active != 0) {
                    const int64_t wait_begin_us = ggml_time_us();
                    response.inflight =
                        server.acquire_route_bulk_credit();
                    return_bulk_credit_wait_us =
                        ggml_time_us() - wait_begin_us;
                    response.limit =
                        server.route_bulk_transfer_limit();
                    return_bulk_credit_held = true;
                    return_bulk_credit_seq = request.seq;
                    return_bulk_credit_inflight =
                        static_cast<size_t>(response.inflight);
                    return_bulk_credit_limit =
                        static_cast<size_t>(response.limit);
                }

                if (!send_msg(
                        sock,
                        &response,
                        sizeof(response))) {
                    if (return_bulk_credit_held) {
                        server.release_route_bulk_credit();
                        return_bulk_credit_held = false;
                    }
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_ASYNC_RETURN_WAIT: {
                const bool stage_profile =
                    rpc_tensor_phone_stage_profile_enabled();
                const int64_t recv_begin_us =
                    stage_profile ? ggml_time_us() : 0;

                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    if (return_bulk_credit_held) {
                        server.release_route_bulk_credit();
                        return_bulk_credit_held = false;
                    }
                    return;
                }
                const int64_t recv_done_us =
                    stage_profile ? ggml_time_us() : 0;

                bool directional_early_ack = false;
                uint64_t directional_seq = 0;
                if (input.size() >=
                        sizeof(
                            rpc_msg_set_tensor_async_return_wait_req)) {
                    rpc_msg_set_tensor_async_return_wait_req
                        request_v19 {};
                    std::memcpy(
                        &request_v19,
                        input.data(),
                        sizeof(request_v19));
                    directional_early_ack =
                        request_v19.wire_magic ==
                            RPC_RETURN_WIRE_MAGIC &&
                        (request_v19.reserved &
                         RPC_RETURN_WIRE_EARLY_ACK_FLAG) != 0;
                    directional_seq =
                        request_v19.phone_ffn_seq;
                }

                const int64_t early_ack_begin_us =
                    stage_profile ? ggml_time_us() : 0;
                if (directional_early_ack) {
                    if (!send_msg(
                            sock,
                            &directional_seq,
                            sizeof(directional_seq))) {
                        return;
                    }
                }
                const int64_t early_ack_done_us =
                    stage_profile ? ggml_time_us() : 0;

                if (stage_profile && directional_early_ack) {
                    GGML_LOG_INFO(
                        "[RETURN_DIRECTIONAL_SRV] seq=%" PRIu64
                        " recv=%.3f ack=%.3f\n",
                        directional_seq,
                        (recv_done_us - recv_begin_us) / 1000.0,
                        (early_ack_done_us -
                            early_ack_begin_us) / 1000.0);
                }

                if (return_bulk_credit_held) {
                    if (input.size() <
                            sizeof(
                                rpc_msg_set_tensor_async_return_wait_req_v14)) {
                        server.release_route_bulk_credit();
                        return_bulk_credit_held = false;
                        return;
                    }

                    rpc_msg_set_tensor_async_return_wait_req_v14
                        return_request {};
                    std::memcpy(
                        &return_request,
                        input.data(),
                        sizeof(return_request));
                    if (return_request.phone_ffn_seq !=
                            return_bulk_credit_seq) {
                        GGML_LOG_ERROR(
                            "[RETURN_BULK_SRV] expected=%" PRIu64
                            " actual=%" PRIu64
                            " status=FAIL reason=seq_mismatch\n",
                            return_bulk_credit_seq,
                            return_request.phone_ffn_seq);
                        server.release_route_bulk_credit();
                        return_bulk_credit_held = false;
                        return;
                    }

                    // Full PC->Phone payload has reached Phone userspace.
                    // Release before Phone-FFN dependency wait and reduce.
                    server.release_route_bulk_credit();

                    if (stage_profile) {
                        GGML_LOG_INFO(
                            "[RETURN_BULK_SRV] seq=%" PRIu64
                            " credit=%.3f inflight=%zu/%zu recv=%.3f\n",
                            return_bulk_credit_seq,
                            return_bulk_credit_wait_us / 1000.0,
                            return_bulk_credit_inflight,
                            return_bulk_credit_limit,
                            (recv_done_us - recv_begin_us) / 1000.0);
                    }

                    return_bulk_credit_held = false;
                    return_bulk_credit_seq = 0;
                    return_bulk_credit_wait_us = 0;
                    return_bulk_credit_inflight = 0;
                    return_bulk_credit_limit = 0;
                }

                if (!server.set_tensor_async_return_wait(input)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_ASYNC_RETURN: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_async_return(input)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_FROM_LOCAL_FILE: {
                rpc_msg_set_tensor_from_local_file_req request {};

                if (!recv_msg(
                        sock,
                        &request,
                        sizeof(request))) {
                    return;
                }

                rpc_msg_set_tensor_from_local_file_rsp response {};

                if (!server.set_tensor_from_local_file(
                        request,
                        response)) {
                    return;
                }

                if (!send_msg(
                        sock,
                        &response,
                        sizeof(response))) {
                    return;
                }

                break;
            }
            case RPC_CMD_INIT_ZERO_TENSOR: {
                rpc_msg_init_zero_tensor_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }

                rpc_msg_init_zero_tensor_rsp response {};
                if (!server.init_zero_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SYNCHRONIZE: {
                rpc_msg_synchronize_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.synchronize(request.device)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SNAPSHOT_TENSOR: {
                rpc_msg_snapshot_tensor_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.snapshot_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_SNAPSHOT: {
                rpc_msg_get_snapshot_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.send_snapshot(request, sock)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE_SNAPSHOT: {
                rpc_msg_graph_recompute_snapshot_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute_snapshot(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_RECOMPUTE_SNAPSHOT: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_recompute_snapshot(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_INFERENCE_RETURN_BIND: {
                rpc_msg_inference_return_bind_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }

                rpc_msg_inference_return_bind_rsp response {};
                response.result =
                    server.bind_inference_return(request) ?
                        1 : 0;

                if (!send_msg(
                        sock,
                        &response,
                        sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INFERENCE_ROUTE_ARM: {
                rpc_msg_inference_route_arm_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }

                const int64_t arm_begin_us = ggml_time_us();
                rpc_msg_inference_route_arm_rsp response {};
                const bool status =
                    server.arm_inference_route(request);
                const int64_t arm_done_us = ggml_time_us();

                response.result = status ? 1 : 0;
                response.wire_size = 0;

                GGML_LOG_INFO(
                    "[INFERENCE_ROUTE_ARM] side=server "
                    "device=%u lane=%u seq=%" PRIu64
                    " layer=%d chunk=%d arm_ms=%.3f "
                    "status=%s\n",
                    request.snapshot.device,
                    request.snapshot.lane,
                    request.snapshot.seq,
                    request.layer,
                    request.chunk,
                    (arm_done_us - arm_begin_us) / 1000.0,
                    status ? "OK" : "FAIL");

                if (!send_msg(
                        sock,
                        &response,
                        sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INFERENCE_CYCLE_SEND: {
                rpc_msg_inference_cycle_send_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_inference_cycle_send_rsp response {};
                response.result = server.send_inference_cycle(request) ? 1 : 0;
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INFERENCE_CYCLE_APPLY: {
                rpc_msg_inference_cycle_send_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_inference_cycle_send_rsp response {};
                response.result = server.apply_inference_cycle(request) ? 1 : 0;
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INFERENCE_CYCLE_FINISH: {
                rpc_msg_inference_cycle_send_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_inference_cycle_send_rsp response {};
                response.result = server.finish_inference_cycle(request) ? 1 : 0;
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INFERENCE_DATA_PLANE_OPEN: {
                rpc_msg_inference_data_plane_open_req request {};
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }

                rpc_msg_inference_data_plane_open_rsp response {};
                const bool config_ok =
                    request.tx_slots >= 1 &&
                    request.tx_slots <= 64 &&
                    request.rx_slots >= 1 &&
                    request.rx_slots <= 64 &&
                    request.max_payload_bytes >= 1024 &&
                    request.max_payload_bytes <=
                        UINT64_C(256) * 1024 * 1024;

                if (!config_ok ||
                    !server.inference_data_plane_available()) {
                    response.result = 0;
                    if (!send_msg(sock, &response, sizeof(response))) {
                        return;
                    }
                    break;
                }

                inference_transport_config config;
                // The peer's TX capacity is this side's RX capacity and vice
                // versa. Keep the same bounded in-flight contract in both
                // directions.
                config.tx_slots = request.rx_slots;
                config.rx_slots = request.tx_slots;
                config.max_payload_bytes =
                    static_cast<size_t>(request.max_payload_bytes);

                response.result = 1;
                response.tx_slots =
                    static_cast<uint32_t>(config.tx_slots);
                response.rx_slots =
                    static_cast<uint32_t>(config.rx_slots);
                response.max_payload_bytes =
                    config.max_payload_bytes;

                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }

                auto adopted =
                    inference_transport::adopt(sock, config);
                if (!adopted) {
                    return;
                }

                std::shared_ptr<inference_transport> transport =
                    std::move(adopted);
                if (!server.attach_inference_data_plane(transport)) {
                    transport->stop();
                    return;
                }

                GGML_LOG_INFO(
                    "[INFERENCE_DATA_PLANE] side=server status=OPEN "
                    "tx_slots=%zu rx_slots=%zu max_payload_mib=%.3f\n",
                    config.tx_slots,
                    config.rx_slots,
                    config.max_payload_bytes /
                        (1024.0 * 1024.0));

                while (transport->healthy()) {
                    inference_rx_frame frame;
                    if (!transport->receive(frame, -1)) {
                        break;
                    }

                    const bool handled =
                        server.handle_inference_frame(
                            frame.header,
                            frame.data,
                            frame.size);
                    transport->release_rx(std::move(frame));

                    if (!handled) {
                        transport->stop();
                        break;
                    }
                }

                server.fail_inference_returns();
                server.fail_cycle_inbound();
                server.detach_inference_data_plane(transport);

                const auto stats = transport->stats();
                GGML_LOG_INFO(
                    "[INFERENCE_DATA_PLANE] side=server status=CLOSED "
                    "tx_frames=%" PRIu64 " rx_frames=%" PRIu64
                    " tx_mib=%.3f rx_mib=%.3f\n",
                    stats.frames_sent,
                    stats.frames_received,
                    stats.bytes_sent / (1024.0 * 1024.0),
                    stats.bytes_received / (1024.0 * 1024.0));
                return;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}
/*void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}
    */
// Internal extended server entry used for TP-specific experiments.
void ggml_backend_rpc_start_server_ex(const char * endpoint, const char * cache_dir,
                                      size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices,
                                      const ggml_rpc_local_tensor_source * tensor_source,
                                      const char * model_path, int tp_rank, int tp_world_size,
                                      ggml_backend_dev_t phone_npu_device) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    if (model_path != nullptr && model_path[0] != '\0') {
    if (tp_world_size <= 0 || tp_rank < 0 || tp_rank >= tp_world_size) {
        fprintf(stderr, "Invalid TP arguments: rank=%d, world_size=%d\n", tp_rank, tp_world_size);
        return;
        }
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  model          : %s\n", model_path && model_path[0] ? model_path : "n/a");
    printf("  tp rank        : %d\n", tp_rank);
    printf("  tp world size  : %d\n", tp_world_size);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);  //读取设备总内存和可用内存
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads); //只要有这个函数，才调用它
            }
        }
    }

    ggml_backend_t phone_npu_backend = nullptr;
    if (phone_npu_device != nullptr) {
        for (size_t i = 0; i < n_devices; ++i) {
            if (devices[i] == phone_npu_device) {
                fprintf(
                    stderr,
                    "Invalid phone NPU stage configuration: private device is RPC-visible\n");
                for (ggml_backend_t backend : backends) {
                    ggml_backend_free(backend);
                }
                return;
            }
        }

        phone_npu_backend = ggml_backend_dev_init(phone_npu_device, nullptr);
        if (phone_npu_backend == nullptr) {
            fprintf(
                stderr,
                "Failed to create private phone NPU backend for device %s\n",
                ggml_backend_dev_name(phone_npu_device));
            for (ggml_backend_t backend : backends) {
                ggml_backend_free(backend);
            }
            return;
        }

    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port); //这个是监听用的socket
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    auto shared_server = std::make_shared<rpc_server>(
        backends, cache_dir, tensor_source, phone_npu_backend);
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        std::thread([shared_server, client_socket, n_devices]() {
            rpc_serve_client(shared_server, client_socket, n_devices);
            printf("Client connection closed\n");
            fflush(stdout);
        }).detach();
    }
    rpc_transport_shutdown();
    if (phone_npu_backend != nullptr) {
        ggml_backend_free(phone_npu_backend);
    }
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    ggml_backend_rpc_start_server_ex(endpoint, cache_dir, n_threads, n_devices, devices,
                                     nullptr, nullptr, 0, 1, nullptr);
}
static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_LOCAL_2D_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_set_tensor_from_local_file_2d);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_STAGE_READY_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_set_stage_ready_callback);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_ROUTE_TRANSFER_LANE_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_set_route_transfer_lane);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_ROUTE_WAIT_SEQ_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_set_route_wait_seq);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_ROUTE_MARK_READY_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_route_mark_ready);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_ROUTE_SNAPSHOT_READY_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_route_snapshot_ready);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_GET_ROUTE_SNAPSHOT_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_get_route_snapshot);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_FENCE_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_fence);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SNAPSHOT_ARM_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_snapshot_arm);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_PREPARE_GRAPH_SNAPSHOT_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_prepare_graph_snapshot);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_SNAPSHOT_READ_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_set_snapshot_read);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_WAIT_SNAPSHOT_READY_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_wait_snapshot_ready);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_GET_SNAPSHOT_STATS_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_get_snapshot_stats);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_PREPARE_FUSED_FFN_INPUT_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_prepare_fused_ffn_input);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_INIT_ZERO_TENSOR_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_init_zero_tensor);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_GET_TENSOR_BATCH3_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_get_tensor_batch3);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_TENSOR_GRAPH_PROC) == 0) {
        return reinterpret_cast<void *>(ggml_backend_rpc_set_tensor_graph);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_TENSOR_ASYNC_RETURN_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_set_tensor_async_return);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_PHONE_FFN_MARK_READY_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_phone_ffn_mark_ready);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_TENSOR_ASYNC_RETURN_WAIT_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_set_tensor_async_return_wait);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_SET_TENSOR_ASYNC_RETURN_ENQUEUE_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_set_tensor_async_return_enqueue);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_INFERENCE_RETURN_WAIT_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_inference_return_wait);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_INFERENCE_TRANSPORT_OPEN_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_inference_transport_open);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_INFERENCE_TRANSPORT_SUBMIT_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_inference_transport_submit);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_INFERENCE_TRANSPORT_RECEIVE_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_inference_transport_receive);
    }
    if (std::strcmp(
            name,
            GGML_BACKEND_RPC_INFERENCE_TRANSPORT_STATS_PROC) == 0) {
        return reinterpret_cast<void *>(
            ggml_backend_rpc_inference_transport_get_stats);
    }
    GGML_UNUSED(reg);

    return NULL;
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
