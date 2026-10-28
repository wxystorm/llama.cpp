#pragma once

// Causal sequence-FFN task scheduler.
//
// This scheduler deliberately knows nothing about GGML.  It owns only local
// host-side readiness state:
//   * normal Region tasks run on CPU / GPU / Phone;
//   * a Sequence-FFN stage expands into CPU Attention, CPU FFN, and Phone FFN;
//   * dependencies are token intervals, never chunk ids;
//   * Phone work is aggregated locally and completion is the returned payload
//     itself -- there is no READY/ACK polling protocol.
//
// The outer G/C/P layout remains a placement/planner concern.  Sequence-FFN is
// a runtime task expansion and therefore does not add a fourth physical device.

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

enum class llama_sequence_pipe_resource : uint8_t {
    CPU = 0,
    GPU = 1,
    PHONE = 2,
};

enum class llama_sequence_pipe_stage_kind : uint8_t {
    CPU,
    GPU,
    PHONE,
    SEQUENCE_FFN,
};

enum class llama_sequence_pipe_task_kind : uint8_t {
    REGION,
    SEQ_ATTN,
    SEQ_PC_FFN,
    SEQ_PHONE_FFN,
};

struct llama_sequence_pipe_stage {
    llama_sequence_pipe_stage_kind kind;
    int layer_begin = 0;
    int layer_end = 0;
    int chunk_tokens = 0;
};

struct llama_sequence_pipe_task {
    llama_sequence_pipe_task_kind kind =
        llama_sequence_pipe_task_kind::REGION;
    llama_sequence_pipe_resource resource =
        llama_sequence_pipe_resource::CPU;

    size_t stage = 0;
    size_t chunk = 0;
    int layer = -1;

    int token_begin = 0;
    int n_tokens = 0;

    uint64_t id = 0;
};

inline const char * llama_sequence_pipe_task_name(
        llama_sequence_pipe_task_kind kind) {
    switch (kind) {
        case llama_sequence_pipe_task_kind::REGION:
            return "REGION";
        case llama_sequence_pipe_task_kind::SEQ_ATTN:
            return "SEQ_ATTN";
        case llama_sequence_pipe_task_kind::SEQ_PC_FFN:
            return "SEQ_PC_FFN";
        case llama_sequence_pipe_task_kind::SEQ_PHONE_FFN:
            return "SEQ_PHONE_FFN";
    }
    return "UNKNOWN";
}

inline const char * llama_sequence_pipe_resource_name(
        llama_sequence_pipe_resource resource) {
    switch (resource) {
        case llama_sequence_pipe_resource::CPU:
            return "CPU";
        case llama_sequence_pipe_resource::GPU:
            return "GPU";
        case llama_sequence_pipe_resource::PHONE:
            return "PHONE";
    }
    return "UNKNOWN";
}

// One scheduler instance covers one ubatch.  Stages are ordered by model
// execution order and may use different chunk sizes.  Producer completions
// publish exact token intervals; consumer chunks become ready once their entire
// input interval is covered.  This is what makes Region chunks and the global
// Sequence split independent.
class llama_sequence_pipe_ready {
public:
    llama_sequence_pipe_ready(
            std::vector<llama_sequence_pipe_stage> stages,
            int n_tokens,
            size_t sequence_stage,
            int sequence_pc_end,
            int phone_min_task_tokens,
            int phone_target_task_tokens) :
        stages_(std::move(stages)),
        n_tokens_(n_tokens),
        sequence_stage_(sequence_stage),
        sequence_pc_end_(sequence_pc_end),
        phone_min_task_tokens_(std::max(1, phone_min_task_tokens)),
        phone_target_task_tokens_(
            std::max(phone_min_task_tokens_, phone_target_task_tokens)) {
        if (stages_.empty() || n_tokens_ <= 0 ||
                sequence_stage_ >= stages_.size() ||
                stages_[sequence_stage_].kind !=
                    llama_sequence_pipe_stage_kind::SEQUENCE_FFN ||
                sequence_pc_end_ <= 0 ||
                sequence_pc_end_ >= n_tokens_) {
            throw std::runtime_error(
                "invalid sequence pipeline dimensions");
        }

        chunk_begins_.resize(stages_.size());
        chunk_sizes_.resize(stages_.size());
        input_ready_tokens_.resize(stages_.size());
        chunk_state_.resize(stages_.size());

        for (size_t s = 0; s < stages_.size(); ++s) {
            const int x = stages_[s].chunk_tokens;
            if (x <= 0 || stages_[s].layer_end <=
                    stages_[s].layer_begin) {
                throw std::runtime_error(
                    "invalid sequence pipeline stage");
            }
            for (int begin = 0; begin < n_tokens_; begin += x) {
                chunk_begins_[s].push_back(begin);
                chunk_sizes_[s].push_back(
                    std::min(x, n_tokens_ - begin));
            }
            input_ready_tokens_[s].assign(
                chunk_sizes_[s].size(), 0);
            chunk_state_[s].assign(
                chunk_sizes_[s].size(), 0);
        }

        // Stage 0 consumes the model input, which is immediately available.
        for (size_t c = 0; c < chunk_sizes_[0].size(); ++c) {
            input_ready_tokens_[0][c] = chunk_sizes_[0][c];
        }
        offer_stage_chunk_locked(0, 0);
    }

    size_t n_stages() const {
        return stages_.size();
    }

    int sequence_pc_end() const {
        return sequence_pc_end_;
    }

    int sequence_phone_tokens() const {
        return n_tokens_ - sequence_pc_end_;
    }

    size_t finished_output_tokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return finished_output_tokens_;
    }

    bool failed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return failed_;
    }

    bool all_done() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return all_done_locked();
    }

    bool resource_busy(
            llama_sequence_pipe_resource resource) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return resource_busy_[(size_t) resource];
    }

    size_t queue_size(
            llama_sequence_pipe_resource resource) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queues_[(size_t) resource].size();
    }

    int phone_pending_tokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return phone_pending_begin_ < 0 ?
            0 : phone_pending_end_ - phone_pending_begin_;
    }

    bool phone_has_submitted_work() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return
            resource_busy_[(size_t) llama_sequence_pipe_resource::PHONE] ||
            !queues_[(size_t) llama_sequence_pipe_resource::PHONE].empty();
    }

    std::optional<llama_sequence_pipe_task> wait_take(
            llama_sequence_pipe_resource resource) {
        const size_t r = (size_t) resource;
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() {
            return failed_ || all_done_locked() ||
                has_claimable_locked(resource);
        });
        if (failed_ || all_done_locked()) {
            return std::nullopt;
        }

        auto task = take_locked(resource);
        if (!task.has_value()) {
            return std::nullopt;
        }
        resource_busy_[r] = true;
        running_task_[r] = task;
        return task;
    }

    // Completion is entirely local scheduler state.  For Phone tasks the
    // caller invokes this only after the P2C payload has arrived, so the return
    // data itself is the completion notification.
    void complete(const llama_sequence_pipe_task & task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const size_t r = (size_t) task.resource;
            if (!resource_busy_[r] ||
                    !running_task_[r].has_value() ||
                    running_task_[r]->id != task.id) {
                throw std::runtime_error(
                    "invalid sequence pipeline completion");
            }

            resource_busy_[r] = false;
            running_task_[r].reset();

            switch (task.kind) {
                case llama_sequence_pipe_task_kind::REGION:
                    complete_region_locked(task);
                    break;
                case llama_sequence_pipe_task_kind::SEQ_ATTN:
                    complete_attention_locked(task);
                    break;
                case llama_sequence_pipe_task_kind::SEQ_PC_FFN:
                case llama_sequence_pipe_task_kind::SEQ_PHONE_FFN:
                    publish_interval_locked(
                        task.stage,
                        task.token_begin,
                        task.n_tokens);
                    break;
            }

            maybe_flush_phone_locked();
        }
        cv_.notify_all();
    }

    void fail(const llama_sequence_pipe_task & task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const size_t r = (size_t) task.resource;
            resource_busy_[r] = false;
            running_task_[r].reset();
            failed_ = true;
        }
        cv_.notify_all();
    }

    void cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
        }
        cv_.notify_all();
    }

private:
    static llama_sequence_pipe_resource stage_resource(
            llama_sequence_pipe_stage_kind kind) {
        switch (kind) {
            case llama_sequence_pipe_stage_kind::CPU:
                return llama_sequence_pipe_resource::CPU;
            case llama_sequence_pipe_stage_kind::GPU:
                return llama_sequence_pipe_resource::GPU;
            case llama_sequence_pipe_stage_kind::PHONE:
                return llama_sequence_pipe_resource::PHONE;
            case llama_sequence_pipe_stage_kind::SEQUENCE_FFN:
                return llama_sequence_pipe_resource::CPU;
        }
        throw std::runtime_error(
            "invalid sequence stage resource");
    }

    bool all_done_locked() const {
        return !failed_ &&
            finished_output_tokens_ == (size_t) n_tokens_ &&
            std::none_of(
                resource_busy_.begin(), resource_busy_.end(),
                [](bool busy) { return busy; }) &&
            std::all_of(
                queues_.begin(), queues_.end(),
                [](const auto & q) { return q.empty(); }) &&
            phone_pending_begin_ < 0;
    }

    bool has_claimable_locked(
            llama_sequence_pipe_resource resource) {
        maybe_flush_phone_locked();

        const auto & q = queues_[(size_t) resource];
        return !q.empty();
    }

    bool should_feed_phone_locked() const {
        // CPU Attention is promoted only while Phone has no submitted work and
        // the global Phone suffix has not yet been completely materialized.
        // Once Phone is busy/queued, CPU advances the oldest causal
        // continuation instead.
        const bool phone_submitted =
            resource_busy_[(size_t) llama_sequence_pipe_resource::PHONE] ||
            !queues_[(size_t) llama_sequence_pipe_resource::PHONE].empty();
        if (phone_submitted ||
                sequence_phone_materialized_tokens_ >=
                    n_tokens_ - sequence_pc_end_) {
            return false;
        }

        const int pending =
            phone_pending_begin_ < 0 ?
                0 : phone_pending_end_ - phone_pending_begin_;
        return pending < phone_min_task_tokens_;
    }

    std::optional<llama_sequence_pipe_task> take_locked(
            llama_sequence_pipe_resource resource) {
        auto & q = queues_[(size_t) resource];
        if (q.empty()) {
            return std::nullopt;
        }

        auto choose_oldest =
            [&](auto predicate) -> typename std::deque<
                    llama_sequence_pipe_task>::iterator {
                auto best = q.end();
                for (auto it = q.begin(); it != q.end(); ++it) {
                    if (!predicate(*it)) {
                        continue;
                    }
                    if (best == q.end() ||
                            it->token_begin < best->token_begin ||
                            (it->token_begin == best->token_begin &&
                             it->id < best->id)) {
                        best = it;
                    }
                }
                return best;
            };

        auto it = q.end();

        if (resource == llama_sequence_pipe_resource::CPU) {
            if (should_feed_phone_locked()) {
                it = choose_oldest(
                    [](const llama_sequence_pipe_task & task) {
                        return task.kind ==
                            llama_sequence_pipe_task_kind::SEQ_ATTN;
                    });
            }

            if (it == q.end()) {
                // Continuations first once Phone is fed.  A PC FFN task and a
                // normal CPU Region task both advance already-materialized
                // tokens.  Oldest token interval wins.
                it = choose_oldest(
                    [](const llama_sequence_pipe_task & task) {
                        return task.kind !=
                            llama_sequence_pipe_task_kind::SEQ_ATTN;
                    });
            }

            if (it == q.end()) {
                it = choose_oldest(
                    [](const llama_sequence_pipe_task &) {
                        return true;
                    });
            }
        } else if (resource ==
                llama_sequence_pipe_resource::PHONE) {
            // Oldest causal interval wins; at the same begin offset the
            // Sequence FFN completion is more direct, so prefer it.
            it = q.begin();
            for (auto cur = q.begin(); cur != q.end(); ++cur) {
                const bool cur_seq =
                    cur->kind ==
                    llama_sequence_pipe_task_kind::SEQ_PHONE_FFN;
                const bool best_seq =
                    it->kind ==
                    llama_sequence_pipe_task_kind::SEQ_PHONE_FFN;
                if (cur->token_begin < it->token_begin ||
                        (cur->token_begin == it->token_begin &&
                         cur_seq && !best_seq)) {
                    it = cur;
                }
            }
        } else {
            it = choose_oldest(
                [](const llama_sequence_pipe_task &) {
                    return true;
                });
        }

        if (it == q.end()) {
            return std::nullopt;
        }

        llama_sequence_pipe_task task = *it;
        q.erase(it);
        return task;
    }

    void enqueue_locked(llama_sequence_pipe_task task) {
        task.id = next_task_id_++;
        queues_[(size_t) task.resource].push_back(
            std::move(task));
    }

    void offer_stage_chunk_locked(
            size_t stage,
            size_t chunk) {
        if (stage >= stages_.size() ||
                chunk >= chunk_sizes_[stage].size() ||
                chunk_state_[stage][chunk] != 0 ||
                input_ready_tokens_[stage][chunk] !=
                    chunk_sizes_[stage][chunk] ||
                (chunk > 0 &&
                 chunk_state_[stage][chunk - 1] != 3)) {
            return;
        }

        chunk_state_[stage][chunk] = 1;
        const auto & desc = stages_[stage];
        const auto task_kind =
            desc.kind ==
                llama_sequence_pipe_stage_kind::SEQUENCE_FFN ?
                llama_sequence_pipe_task_kind::SEQ_ATTN :
                llama_sequence_pipe_task_kind::REGION;

        enqueue_locked({
            task_kind,
            stage_resource(desc.kind),
            stage,
            chunk,
            desc.kind ==
                llama_sequence_pipe_stage_kind::SEQUENCE_FFN ?
                desc.layer_begin : -1,
            chunk_begins_[stage][chunk],
            chunk_sizes_[stage][chunk],
            0,
        });
        chunk_state_[stage][chunk] = 2;
    }

    void complete_region_locked(
            const llama_sequence_pipe_task & task) {
        if (task.stage >= chunk_state_.size() ||
                task.chunk >= chunk_state_[task.stage].size() ||
                chunk_state_[task.stage][task.chunk] != 2) {
            throw std::runtime_error(
                "invalid Region task completion");
        }

        chunk_state_[task.stage][task.chunk] = 3;
        publish_interval_locked(
            task.stage,
            task.token_begin,
            task.n_tokens);
        offer_stage_chunk_locked(
            task.stage,
            task.chunk + 1);
    }

    void complete_attention_locked(
            const llama_sequence_pipe_task & task) {
        if (task.stage != sequence_stage_ ||
                task.chunk >= chunk_state_[task.stage].size() ||
                chunk_state_[task.stage][task.chunk] != 2) {
            throw std::runtime_error(
                "invalid Sequence Attention completion");
        }

        chunk_state_[task.stage][task.chunk] = 3;
        ++sequence_attention_completed_;

        const int begin = task.token_begin;
        const int end = begin + task.n_tokens;

        const int pc_begin = begin;
        const int pc_end =
            std::min(end, sequence_pc_end_);
        if (pc_end > pc_begin) {
            enqueue_locked({
                llama_sequence_pipe_task_kind::SEQ_PC_FFN,
                llama_sequence_pipe_resource::CPU,
                task.stage,
                task.chunk,
                stages_[task.stage].layer_begin,
                pc_begin,
                pc_end - pc_begin,
                0,
            });
        }

        const int phone_begin =
            std::max(begin, sequence_pc_end_);
        const int phone_end = end;
        if (phone_end > phone_begin) {
            if (phone_pending_begin_ < 0) {
                phone_pending_begin_ = phone_begin;
                phone_pending_end_ = phone_end;
            } else {
                if (phone_pending_end_ != phone_begin) {
                    throw std::runtime_error(
                        "non-contiguous Phone Sequence pending interval");
                }
                phone_pending_end_ = phone_end;
            }
            sequence_phone_materialized_tokens_ +=
                phone_end - phone_begin;
        }

        offer_stage_chunk_locked(
            task.stage,
            task.chunk + 1);
        maybe_flush_phone_locked();
    }

    void maybe_flush_phone_locked() {
        if (phone_pending_begin_ < 0) {
            return;
        }

        const int pending =
            phone_pending_end_ - phone_pending_begin_;
        const bool phone_idle =
            !resource_busy_[
                (size_t) llama_sequence_pipe_resource::PHONE] &&
            queues_[
                (size_t) llama_sequence_pipe_resource::PHONE].empty();
        const bool all_phone_materialized =
            sequence_phone_materialized_tokens_ >=
                n_tokens_ - sequence_pc_end_;

        const bool flush =
            pending >= phone_target_task_tokens_ ||
            (phone_idle && pending >= phone_min_task_tokens_) ||
            all_phone_materialized;

        if (!flush) {
            return;
        }

        enqueue_locked({
            llama_sequence_pipe_task_kind::SEQ_PHONE_FFN,
            llama_sequence_pipe_resource::PHONE,
            sequence_stage_,
            std::numeric_limits<size_t>::max(),
            stages_[sequence_stage_].layer_begin,
            phone_pending_begin_,
            pending,
            0,
        });

        phone_pending_begin_ = -1;
        phone_pending_end_ = -1;
    }

    void publish_interval_locked(
            size_t producer_stage,
            int begin,
            int n_tokens) {
        if (n_tokens <= 0) {
            return;
        }

        if (producer_stage + 1 >= stages_.size()) {
            finished_output_tokens_ += (size_t) n_tokens;
            if (finished_output_tokens_ > (size_t) n_tokens_) {
                throw std::runtime_error(
                    "sequence pipeline final coverage overflow");
            }
            return;
        }

        const size_t consumer = producer_stage + 1;
        const int end = begin + n_tokens;
        for (size_t c = 0;
             c < chunk_sizes_[consumer].size(); ++c) {
            const int cb = chunk_begins_[consumer][c];
            const int ce = cb + chunk_sizes_[consumer][c];
            const int overlap =
                std::max(0, std::min(end, ce) -
                    std::max(begin, cb));
            if (overlap <= 0) {
                continue;
            }

            int & ready =
                input_ready_tokens_[consumer][c];
            ready += overlap;
            if (ready > chunk_sizes_[consumer][c]) {
                throw std::runtime_error(
                    "sequence pipeline input coverage overlap");
            }
            if (ready == chunk_sizes_[consumer][c]) {
                offer_stage_chunk_locked(
                    consumer, c);
            }
        }
    }

private:
    const std::vector<llama_sequence_pipe_stage> stages_;
    const int n_tokens_ = 0;
    const size_t sequence_stage_ = 0;
    const int sequence_pc_end_ = 0;
    const int phone_min_task_tokens_ = 1;
    const int phone_target_task_tokens_ = 1;

    std::vector<std::vector<int>> chunk_begins_;
    std::vector<std::vector<int>> chunk_sizes_;
    std::vector<std::vector<int>> input_ready_tokens_;
    std::vector<std::vector<uint8_t>> chunk_state_;

    std::array<std::deque<llama_sequence_pipe_task>, 3> queues_;
    std::array<bool, 3> resource_busy_ = {false, false, false};
    std::array<std::optional<llama_sequence_pipe_task>, 3>
        running_task_;

    int phone_pending_begin_ = -1;
    int phone_pending_end_ = -1;
    int sequence_phone_materialized_tokens_ = 0;
    size_t sequence_attention_completed_ = 0;

    size_t finished_output_tokens_ = 0;
    uint64_t next_task_id_ = 1;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool failed_ = false;
};
