#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lt {

enum class side : uint8_t {
    pc = 0,
    phone = 1,
};

enum class mode : uint8_t {
    tensor = 0,
    layer = 1,
};

enum class message_kind : uint8_t {
    ready = 1,
    partial = 2,
    layer_hidden = 3,
    done = 4,
    stop = 5,
    initial = 6,
    model_ready = 7,
    debug_partial = 8,
    bench_ping = 9,
    bench_pong = 10,
    stream_bench_request = 11,
    stream_bench_result = 12,
    sequence_ffn_request = 13,
    sequence_ffn_result = 14,
    bench_pull = 15,
    bench_pull_result = 16,
};

struct key {
    int32_t layer = -1;
    int32_t chunk = -1;

    bool operator==(const key & other) const {
        return layer == other.layer && chunk == other.chunk;
    }
};

struct key_hash {
    size_t operator()(const key & value) const {
        return (static_cast<size_t>(static_cast<uint32_t>(value.layer)) << 32) ^
               static_cast<size_t>(static_cast<uint32_t>(value.chunk));
    }
};

struct ready_payload {
    std::vector<float> residual;
    std::vector<float> hidden;
    std::vector<int32_t> topk;
    std::vector<float> weights;
};

struct message {
    message_kind kind = message_kind::stop;
    key task;
    uint64_t seq = 0;

    ready_payload ready;
    std::vector<float> partial;
    std::vector<float> hidden;
    double checksum = 0.0;
};

struct plan {
    mode execution_mode = mode::tensor;
    int32_t layers = 0;
    std::vector<int32_t> chunk_tokens;

    side first_tensor_owner = side::pc;
    std::vector<side> layer_owners;

    // Physical PC share of the MoE expert hidden dimension.  Directional
    // ratios are independent so the crossover search can tune C2P/P2C
    // separately without changing the runtime protocol.
    float c2p_pc_ratio = 0.5f;
    float p2c_pc_ratio = 0.5f;

    // OWNER has priority, but REMOTE_FFN must not starve. With the default
    // value 1, a waiting REMOTE_FFN is serviced after at most one OWNER task.
    int32_t owner_burst_limit = 1;

    // Admission gate for diagonal Tensor wavefront. A local owner layer may
    // stay at most this many chunks ahead of the next (remote-owned) layer
    // having published READY back to this endpoint.
    int32_t max_owner_ahead = 1;

    // Diagnostic only. Phone reads its REMOTE_FFN shard back to host and
    // publishes DEBUG_PARTIAL so PC can compare against a CPU reference.
    bool debug_phone_partial = false;

    bool valid(std::string * reason = nullptr) const;
    side owner_for_layer(int32_t layer) const;
    float pc_ratio_for_layer(int32_t layer) const;
};

class transport {
public:
    virtual ~transport() = default;
    virtual bool send(const message & value) = 0;
};

class compute {
public:
    virtual ~compute() = default;

    // Attention owner:
    //   Attention -> residual -> FFN norm -> Router/TopK.
    // READY is published immediately after this call and before local_ffn().
    virtual bool owner_prefix(
        const key & task,
        const std::vector<float> & input,
        ready_payload & ready) = 0;

    virtual bool local_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) = 0;

    // The non-owner receives READY and computes its FFN shard independently.
    virtual bool remote_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) = 0;

    // REDUCE is deliberately owned by a separate worker.  For an alternating
    // C2P/P2C Tensor chain, REDUCE(L,C) runs on the next layer's owner and
    // directly unlocks OWNER(L+1,C).
    virtual bool reduce(
        const key & task,
        const ready_payload & ready,
        const std::vector<float> & local_partial,
        const std::vector<float> & remote_partial,
        std::vector<float> & output) = 0;

    // Full-layer baseline.
    virtual bool full_layer(
        const key & task,
        const std::vector<float> & input,
        std::vector<float> & output) = 0;
};

struct stats {
    uint64_t owner_tasks = 0;
    uint64_t remote_ffn_tasks = 0;
    uint64_t reduce_tasks = 0;
    uint64_t full_layer_tasks = 0;

    uint64_t ready_sent = 0;
    uint64_t partial_sent = 0;
    uint64_t layer_hidden_sent = 0;
    uint64_t done_sent = 0;
    uint64_t initial_sent = 0;

    uint64_t ready_received = 0;
    uint64_t partial_received = 0;
    uint64_t layer_hidden_received = 0;
    uint64_t done_received = 0;
    uint64_t initial_received = 0;

    uint64_t final_chunks = 0;
    double final_checksum_sum = 0.0;
};

class runtime {
public:
    runtime(side local_side,
            plan runtime_plan,
            std::shared_ptr<compute> compute_impl,
            std::shared_ptr<transport> transport_impl,
            bool log_events = false);
    ~runtime();

    runtime(const runtime &) = delete;
    runtime & operator=(const runtime &) = delete;

    bool start();
    void stop();

    bool submit_initial(int32_t chunk, std::vector<float> input);

    // RX thread entry point.  It only publishes readiness; it never computes
    // and never performs blocking TX.
    bool on_message(message value);

    bool wait_until_complete(uint64_t expected_chunks);

    bool failed() const;
    std::string failure_reason() const;
    stats snapshot_stats() const;

private:
    struct owner_work {
        key task;
        std::vector<float> input;
    };

    struct tensor_state {
        ready_payload ready;
        std::vector<float> local_partial;
        std::vector<float> remote_partial;

        bool ready_received = false;
        bool local_partial_ready = false;
        bool remote_partial_ready = false;
        bool reduce_queued = false;
    };

    void compute_loop();
    void reduce_loop();
    void tx_loop();

    bool run_owner(owner_work work);
    bool run_remote(const key & task);
    bool run_full_layer(owner_work work);

    void enqueue_tx(message value);
    void maybe_queue_reduce_locked(const key & task, tensor_state & state);
    void mark_final(const key & task, const std::vector<float> & output);

    void set_failure(const std::string & reason);
    void set_failure_locked(const std::string & reason);
    void notify_all();
    void log(const char * event, const key & task, const char * detail = nullptr) const;

private:
    const side local_side_;
    const plan plan_;
    const std::shared_ptr<compute> compute_;
    const std::shared_ptr<transport> transport_;
    const bool log_events_;

    mutable std::mutex mutex_;
    std::condition_variable compute_cv_;
    std::condition_variable reduce_cv_;
    std::condition_variable tx_cv_;
    std::condition_variable done_cv_;

    std::deque<owner_work> owner_queue_;
    std::deque<key> remote_queue_;
    std::deque<key> reduce_queue_;
    std::deque<message> tx_queue_;

    std::unordered_map<key, tensor_state, key_hash> tensor_states_;

    std::thread compute_thread_;
    std::thread reduce_thread_;
    std::thread tx_thread_;

    bool started_ = false;
    bool stopping_ = false;
    bool failed_ = false;
    std::string failure_reason_;

    int32_t consecutive_owner_ = 0;
    std::vector<int32_t> next_owner_chunk_;
    std::vector<int32_t> remote_owner_started_;
    uint64_t next_seq_ = 1;
    uint64_t completed_chunks_ = 0;

    stats stats_;
};

// Deterministic implementation used only for scheduler/protocol self-tests.
// Performance runs will bind real GGML kernels through the same interface.
class reference_compute final : public compute {
public:
    explicit reference_compute(float side_bias);

    bool owner_prefix(
        const key & task,
        const std::vector<float> & input,
        ready_payload & ready) override;

    bool local_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) override;

    bool remote_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) override;

    bool reduce(
        const key & task,
        const ready_payload & ready,
        const std::vector<float> & local_partial,
        const std::vector<float> & remote_partial,
        std::vector<float> & output) override;

    bool full_layer(
        const key & task,
        const std::vector<float> & input,
        std::vector<float> & output) override;

private:
    float side_bias_;
};

double checksum(const std::vector<float> & values);
const char * side_name(side value);
const char * mode_name(mode value);

} // namespace lt
