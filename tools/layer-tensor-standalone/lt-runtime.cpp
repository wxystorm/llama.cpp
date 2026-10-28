#include "lt-runtime.h"

#include <algorithm>
#include <cstdio>
#include <utility>

namespace lt {

const char * side_name(side value) {
    return value == side::pc ? "PC" : "PHONE";
}

const char * mode_name(mode value) {
    return value == mode::tensor ? "TENSOR" : "LAYER";
}

double checksum(const std::vector<float> & values) {
    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
        sum += static_cast<double>(values[i]) *
               static_cast<double>((i % 17) + 1);
    }
    return sum;
}

bool plan::valid(std::string * reason) const {
    auto fail = [&](const char * text) {
        if (reason != nullptr) {
            *reason = text;
        }
        return false;
    };

    if (layers <= 0) {
        return fail("layers must be > 0");
    }
    if (chunk_tokens.empty()) {
        return fail("at least one chunk is required");
    }
    for (const int32_t tokens : chunk_tokens) {
        if (tokens <= 0) {
            return fail("chunk token counts must be > 0");
        }
    }
    if (owner_burst_limit <= 0) {
        return fail("owner_burst_limit must be > 0");
    }
    if (max_owner_ahead < 0) {
        return fail("max_owner_ahead must be >= 0");
    }
    if (execution_mode == mode::layer &&
        layer_owners.size() != static_cast<size_t>(layers)) {
        return fail("layer mode requires one owner per layer");
    }
    if (execution_mode == mode::tensor &&
        (!(c2p_pc_ratio > 0.0f && c2p_pc_ratio < 1.0f) ||
         !(p2c_pc_ratio > 0.0f && p2c_pc_ratio < 1.0f))) {
        return fail("tensor ratios must be strictly between 0 and 1");
    }
    return true;
}

side plan::owner_for_layer(int32_t layer) const {
    if (execution_mode == mode::layer) {
        return layer_owners.at(static_cast<size_t>(layer));
    }

    const bool even = (layer % 2) == 0;
    if (even) {
        return first_tensor_owner;
    }
    return first_tensor_owner == side::pc ? side::phone : side::pc;
}

float plan::pc_ratio_for_layer(int32_t layer) const {
    if (execution_mode != mode::tensor) {
        return 1.0f;
    }
    return owner_for_layer(layer) == side::pc ?
        c2p_pc_ratio : p2c_pc_ratio;
}

runtime::runtime(
        side local_side,
        plan runtime_plan,
        std::shared_ptr<compute> compute_impl,
        std::shared_ptr<transport> transport_impl,
        bool log_events)
    : local_side_(local_side),
      plan_(std::move(runtime_plan)),
      compute_(std::move(compute_impl)),
      transport_(std::move(transport_impl)),
      log_events_(log_events) {
    next_owner_chunk_.assign(
        static_cast<size_t>(std::max(0, plan_.layers)),
        0);
    remote_owner_started_.assign(
        static_cast<size_t>(std::max(0, plan_.layers)),
        0);
}

runtime::~runtime() {
    stop();
}

bool runtime::start() {
    std::string reason;
    if (!plan_.valid(&reason)) {
        set_failure("invalid plan: " + reason);
        return false;
    }
    if (compute_ == nullptr || transport_ == nullptr) {
        set_failure("compute and transport must be non-null");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (started_) {
            return true;
        }
        started_ = true;
        stopping_ = false;
    }

    compute_thread_ = std::thread([this]() { compute_loop(); });
    if (plan_.execution_mode == mode::tensor) {
        reduce_thread_ = std::thread([this]() { reduce_loop(); });
    }
    tx_thread_ = std::thread([this]() { tx_loop(); });

    if (log_events_) {
        std::fprintf(stderr,
            "[LT_RUNTIME] side=%s mode=%s layers=%d chunks=%zu "
            "owner_burst=%d max_owner_ahead=%d status=STARTED\n",
            side_name(local_side_), mode_name(plan_.execution_mode),
            plan_.layers, plan_.chunk_tokens.size(),
            plan_.owner_burst_limit,
            plan_.max_owner_ahead);
    }
    return true;
}

void runtime::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!started_) {
            return;
        }
        stopping_ = true;
    }
    notify_all();

    if (compute_thread_.joinable()) {
        compute_thread_.join();
    }
    if (reduce_thread_.joinable()) {
        reduce_thread_.join();
    }
    if (tx_thread_.joinable()) {
        tx_thread_.join();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
}

bool runtime::submit_initial(int32_t chunk, std::vector<float> input) {
    if (chunk < 0 ||
        static_cast<size_t>(chunk) >= plan_.chunk_tokens.size()) {
        set_failure("submit_initial: invalid chunk");
        return false;
    }
    if (plan_.owner_for_layer(0) != local_side_) {
        set_failure("submit_initial: layer-0 owner is remote");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        owner_queue_.push_back({ { 0, chunk }, std::move(input) });
    }
    compute_cv_.notify_one();
    return true;
}

bool runtime::on_message(message value) {
    const key task = value.task;

    if (value.kind == message_kind::stop) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        notify_all();
        return true;
    }

    if (task.chunk < 0 ||
        static_cast<size_t>(task.chunk) >= plan_.chunk_tokens.size()) {
        set_failure("RX: invalid chunk");
        return false;
    }

    if (value.kind == message_kind::initial) {
        if (task.layer != 0 ||
            plan_.owner_for_layer(0) != local_side_) {
            set_failure("INITIAL delivered to non-owner");
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.initial_received;
            owner_queue_.push_back({ task, std::move(value.hidden) });
        }
        compute_cv_.notify_one();
        log("INITIAL_RX", task);
        return true;
    }

    if (value.kind == message_kind::done) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++completed_chunks_;
            ++stats_.done_received;
            ++stats_.final_chunks;
            stats_.final_checksum_sum += value.checksum;
        }
        done_cv_.notify_all();
        log("DONE_RX", task);
        return true;
    }

    if (task.layer < 0 || task.layer >= plan_.layers) {
        set_failure("RX: invalid layer");
        return false;
    }

    if (plan_.execution_mode == mode::layer) {
        if (value.kind != message_kind::layer_hidden) {
            set_failure("layer mode received non-layer message");
            return false;
        }
        if (plan_.owner_for_layer(task.layer) != local_side_) {
            set_failure("layer hidden delivered to non-owner");
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.layer_hidden_received;
            owner_queue_.push_back({ task, std::move(value.hidden) });
        }
        compute_cv_.notify_one();
        log("LAYER_HIDDEN_RX", task);
        return true;
    }

    const side attention_owner = plan_.owner_for_layer(task.layer);
    if (attention_owner == local_side_) {
        set_failure("Tensor RX delivered to Attention owner");
        return false;
    }

    if (value.kind == message_kind::ready) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tensor_state & state = tensor_states_[task];
            if (state.ready_received) {
                set_failure_locked("duplicate READY");
                notify_all();
                return false;
            }
            state.ready = std::move(value.ready);
            state.ready_received = true;
            ++stats_.ready_received;

            int32_t & started =
                remote_owner_started_[
                    static_cast<size_t>(task.layer)];
            started = std::max(
                started,
                task.chunk + 1);

            remote_queue_.push_back(task);
            maybe_queue_reduce_locked(task, state);
        }
        compute_cv_.notify_one();
        log("READY_RX", task);
        return true;
    }

    if (value.kind == message_kind::partial) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tensor_state & state = tensor_states_[task];
            if (state.remote_partial_ready) {
                set_failure_locked("duplicate PARTIAL");
                notify_all();
                return false;
            }
            state.remote_partial = std::move(value.partial);
            state.remote_partial_ready = true;
            ++stats_.partial_received;
            maybe_queue_reduce_locked(task, state);
        }
        log("PARTIAL_RX", task);
        return true;
    }

    set_failure("Tensor RX: unsupported message kind");
    return false;
}

bool runtime::wait_until_complete(uint64_t expected_chunks) {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [&]() {
        return failed_ || stopping_ ||
            completed_chunks_ >= expected_chunks;
    });
    return !failed_ && completed_chunks_ >= expected_chunks;
}

bool runtime::failed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failed_;
}

std::string runtime::failure_reason() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return failure_reason_;
}

stats runtime::snapshot_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void runtime::compute_loop() {
    while (true) {
        owner_work owner;
        key remote;
        bool have_owner = false;
        bool have_remote = false;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            const auto eligible_owner =
                [&]() {
                    return std::find_if(
                        owner_queue_.begin(),
                        owner_queue_.end(),
                        [&](const owner_work & work) {
                            if (work.task.layer < 0 ||
                                work.task.layer >= plan_.layers) {
                                return false;
                            }
                            const size_t layer_index =
                                static_cast<size_t>(
                                    work.task.layer);

                            if (work.task.chunk !=
                                next_owner_chunk_[layer_index]) {
                                return false;
                            }

                            if (plan_.execution_mode == mode::tensor &&
                                work.task.layer + 1 < plan_.layers) {
                                const size_t next_layer_index =
                                    static_cast<size_t>(
                                        work.task.layer + 1);

                                if (next_owner_chunk_[layer_index] >
                                    remote_owner_started_[
                                        next_layer_index] +
                                    plan_.max_owner_ahead) {
                                    return false;
                                }
                            }

                            return true;
                        });
                };

            compute_cv_.wait(lock, [&]() {
                return stopping_ ||
                    failed_ ||
                    !remote_queue_.empty() ||
                    eligible_owner() !=
                        owner_queue_.end();
            });

            const auto owner_it =
                eligible_owner();
            const bool owner_ready =
                owner_it !=
                    owner_queue_.end();

            if ((stopping_ || failed_) &&
                !owner_ready &&
                remote_queue_.empty()) {
                return;
            }

            if (plan_.execution_mode ==
                mode::layer) {
                if (owner_ready) {
                    owner =
                        std::move(*owner_it);
                    owner_queue_.erase(
                        owner_it);
                    have_owner = true;
                }
            } else {
                const bool prefer_owner =
                    owner_ready &&
                    (remote_queue_.empty() ||
                     consecutive_owner_ <
                        plan_.owner_burst_limit);

                if (prefer_owner) {
                    owner =
                        std::move(*owner_it);
                    owner_queue_.erase(
                        owner_it);
                    ++consecutive_owner_;
                    have_owner = true;
                } else if (!remote_queue_.empty()) {
                    remote =
                        remote_queue_.front();
                    remote_queue_.pop_front();
                    consecutive_owner_ = 0;
                    have_remote = true;
                } else if (owner_ready) {
                    owner =
                        std::move(*owner_it);
                    owner_queue_.erase(
                        owner_it);
                    ++consecutive_owner_;
                    have_owner = true;
                }
            }
        }

        if (have_owner) {
            const key completed_task =
                owner.task;

            const bool ok =
                plan_.execution_mode ==
                    mode::tensor ?
                    run_owner(
                        std::move(owner)) :
                    run_full_layer(
                        std::move(owner));

            if (!ok) {
                return;
            }

            {
                std::lock_guard<std::mutex> lock(
                    mutex_);
                int32_t & next =
                    next_owner_chunk_[
                        static_cast<size_t>(
                            completed_task.layer)];
                if (completed_task.chunk != next) {
                    set_failure_locked(
                        "OWNER chunk order state mismatch");
                    notify_all();
                    return;
                }
                ++next;
            }

            // A queued later chunk of this same layer may have just become
            // eligible. Different layers remain independently ready.
            compute_cv_.notify_one();
        } else if (have_remote) {
            if (!run_remote(remote)) {
                return;
            }
        }
    }
}

void runtime::reduce_loop() {
    while (true) {
        key task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            reduce_cv_.wait(lock, [&]() {
                return stopping_ || failed_ ||
                    !reduce_queue_.empty();
            });

            if ((stopping_ || failed_) &&
                reduce_queue_.empty()) {
                return;
            }

            task = reduce_queue_.front();
            reduce_queue_.pop_front();
        }

        ready_payload ready;
        std::vector<float> local_partial;
        std::vector<float> remote_partial;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = tensor_states_.find(task);
            if (it == tensor_states_.end()) {
                set_failure_locked("REDUCE state missing");
                notify_all();
                return;
            }
            ready = it->second.ready;
            local_partial = it->second.local_partial;
            remote_partial = it->second.remote_partial;
        }

        std::vector<float> output;
        log("REDUCE_BEGIN", task);
        if (!compute_->reduce(
                task, ready,
                local_partial, remote_partial,
                output)) {
            set_failure("REDUCE compute failed");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.reduce_tasks;
            tensor_states_.erase(task);
        }
        log("REDUCE_END", task);

        if (task.layer + 1 == plan_.layers) {
            mark_final(task, output);
            continue;
        }

        const key next { task.layer + 1, task.chunk };
        if (plan_.owner_for_layer(next.layer) != local_side_) {
            set_failure(
                "REDUCE did not unlock local next owner");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            owner_queue_.push_back({
                next, std::move(output)
            });
        }
        compute_cv_.notify_one();
        log("OWNER_UNLOCK", next);
    }
}

void runtime::tx_loop() {
    while (true) {
        message value;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tx_cv_.wait(lock, [&]() {
                return stopping_ || failed_ ||
                    !tx_queue_.empty();
            });

            if ((stopping_ || failed_) &&
                tx_queue_.empty()) {
                return;
            }

            value = std::move(tx_queue_.front());
            tx_queue_.pop_front();
        }

        if (!transport_->send(value)) {
            set_failure("transport send failed");
            return;
        }
    }
}

bool runtime::run_owner(owner_work work) {
    if (plan_.owner_for_layer(work.task.layer) != local_side_) {
        set_failure("OWNER queued on wrong side");
        return false;
    }

    ready_payload ready;
    log("OWNER_PREFIX_BEGIN", work.task);
    if (!compute_->owner_prefix(
            work.task, work.input, ready)) {
        set_failure("OWNER prefix failed");
        return false;
    }

    message ready_msg;
    ready_msg.kind = message_kind::ready;
    ready_msg.task = work.task;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_msg.seq = next_seq_++;
        ++stats_.owner_tasks;
        ++stats_.ready_sent;
    }
    ready_msg.ready = ready;
    enqueue_tx(std::move(ready_msg));
    log("READY_PUBLISH", work.task);

    std::vector<float> partial;
    log("OWNER_FFN_BEGIN", work.task);
    if (!compute_->local_ffn(
            work.task, ready, partial)) {
        set_failure("OWNER local FFN failed");
        return false;
    }

    message partial_msg;
    partial_msg.kind = message_kind::partial;
    partial_msg.task = work.task;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        partial_msg.seq = next_seq_++;
        ++stats_.partial_sent;
    }
    partial_msg.partial = std::move(partial);
    enqueue_tx(std::move(partial_msg));
    log("PARTIAL_PUBLISH", work.task);
    return true;
}

bool runtime::run_remote(const key & task) {
    ready_payload ready;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tensor_states_.find(task);
        if (it == tensor_states_.end() ||
            !it->second.ready_received) {
            set_failure_locked(
                "REMOTE_FFN missing READY");
            notify_all();
            return false;
        }
        ready = it->second.ready;
    }

    std::vector<float> partial;
    log("REMOTE_FFN_BEGIN", task);
    if (!compute_->remote_ffn(
            task, ready, partial)) {
        set_failure("REMOTE_FFN failed");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        tensor_state & state = tensor_states_[task];
        state.local_partial = std::move(partial);
        state.local_partial_ready = true;
        ++stats_.remote_ffn_tasks;
        maybe_queue_reduce_locked(task, state);
    }
    log("REMOTE_FFN_END", task);
    return true;
}

bool runtime::run_full_layer(owner_work work) {
    if (plan_.owner_for_layer(work.task.layer) != local_side_) {
        set_failure("FULL_LAYER queued on wrong side");
        return false;
    }

    std::vector<float> output;
    log("FULL_LAYER_BEGIN", work.task);
    if (!compute_->full_layer(
            work.task, work.input, output)) {
        set_failure("FULL_LAYER failed");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.full_layer_tasks;
    }
    log("FULL_LAYER_END", work.task);

    if (work.task.layer + 1 == plan_.layers) {
        mark_final(work.task, output);
        return true;
    }

    const key next {
        work.task.layer + 1,
        work.task.chunk
    };
    const side next_owner =
        plan_.owner_for_layer(next.layer);

    if (next_owner == local_side_) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            owner_queue_.push_back({
                next, std::move(output)
            });
        }
        compute_cv_.notify_one();
        return true;
    }

    message hidden_msg;
    hidden_msg.kind = message_kind::layer_hidden;
    hidden_msg.task = next;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        hidden_msg.seq = next_seq_++;
        ++stats_.layer_hidden_sent;
    }
    hidden_msg.hidden = std::move(output);
    enqueue_tx(std::move(hidden_msg));
    log("LAYER_HIDDEN_PUBLISH", next);
    return true;
}

void runtime::enqueue_tx(message value) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tx_queue_.push_back(std::move(value));
    }
    tx_cv_.notify_one();
}

void runtime::maybe_queue_reduce_locked(
        const key & task,
        tensor_state & state) {
    if (!state.reduce_queued &&
        state.ready_received &&
        state.local_partial_ready &&
        state.remote_partial_ready) {
        state.reduce_queued = true;
        reduce_queue_.push_back(task);
        reduce_cv_.notify_one();
    }
}

void runtime::mark_final(
        const key & task,
        const std::vector<float> & output) {
    const double value = checksum(output);

    if (local_side_ == side::pc) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++completed_chunks_;
            ++stats_.final_chunks;
            stats_.final_checksum_sum += value;
        }
        done_cv_.notify_all();
        log("FINAL_LOCAL", task);
        return;
    }

    message done;
    done.kind = message_kind::done;
    done.task = task;
    done.checksum = value;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        done.seq = next_seq_++;
        ++stats_.done_sent;
    }
    enqueue_tx(std::move(done));
    log("FINAL_REMOTE", task);
}

void runtime::set_failure(
        const std::string & reason) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        set_failure_locked(reason);
    }

    std::fprintf(stderr,
        "[LT_RUNTIME_ERROR] side=%s reason=%s\n",
        side_name(local_side_),
        reason.c_str());

    notify_all();
}

void runtime::set_failure_locked(
        const std::string & reason) {
    if (failed_) {
        return;
    }
    failed_ = true;
    failure_reason_ = reason;
}

void runtime::notify_all() {
    compute_cv_.notify_all();
    reduce_cv_.notify_all();
    tx_cv_.notify_all();
    done_cv_.notify_all();
}

void runtime::log(
        const char * event,
        const key & task,
        const char * detail) const {
    if (!log_events_) {
        return;
    }

    std::fprintf(stderr,
        "[LT_TASK] side=%s mode=%s event=%s "
        "layer=%d chunk=%d%s%s\n",
        side_name(local_side_),
        mode_name(plan_.execution_mode),
        event,
        task.layer,
        task.chunk,
        detail != nullptr ? " " : "",
        detail != nullptr ? detail : "");
}

reference_compute::reference_compute(float side_bias)
    : side_bias_(side_bias) {
}

bool reference_compute::owner_prefix(
        const key & task,
        const std::vector<float> & input,
        ready_payload & ready) {
    ready.residual = input;
    ready.hidden.resize(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        ready.hidden[i] =
            input[i] * 1.01f +
            0.001f * static_cast<float>(task.layer + 1) +
            side_bias_;
    }

    const size_t tokens =
        std::max<size_t>(1, input.size() / 8);
    constexpr int k = 4;

    ready.topk.resize(tokens * k);
    ready.weights.resize(tokens * k);
    for (size_t t = 0; t < tokens; ++t) {
        for (int j = 0; j < k; ++j) {
            ready.topk[
                t * k + static_cast<size_t>(j)] =
                static_cast<int32_t>(
                    (t + static_cast<size_t>(j) +
                     static_cast<size_t>(task.layer)) %
                    128);
            ready.weights[
                t * k + static_cast<size_t>(j)] =
                1.0f / k;
        }
    }
    return true;
}

bool reference_compute::local_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) {
    partial.resize(ready.hidden.size());
    const float bias =
        0.03f +
        0.0001f * static_cast<float>(task.chunk) +
        side_bias_;

    for (size_t i = 0; i < partial.size(); ++i) {
        partial[i] =
            ready.hidden[i] * 0.5f + bias;
    }
    return true;
}

bool reference_compute::remote_ffn(
        const key & task,
        const ready_payload & ready,
        std::vector<float> & partial) {
    partial.resize(ready.hidden.size());
    const float bias =
        0.07f +
        0.0001f * static_cast<float>(task.layer) +
        side_bias_;

    for (size_t i = 0; i < partial.size(); ++i) {
        partial[i] =
            ready.hidden[i] * 0.25f + bias;
    }
    return true;
}

bool reference_compute::reduce(
        const key &,
        const ready_payload & ready,
        const std::vector<float> & local_partial,
        const std::vector<float> & remote_partial,
        std::vector<float> & output) {
    if (ready.residual.size() != local_partial.size() ||
        ready.residual.size() != remote_partial.size()) {
        return false;
    }

    output.resize(ready.residual.size());
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] =
            ready.residual[i] +
            local_partial[i] +
            remote_partial[i];
    }
    return true;
}

bool reference_compute::full_layer(
        const key & task,
        const std::vector<float> & input,
        std::vector<float> & output) {
    output.resize(input.size());
    const float gain =
        1.02f +
        0.001f * static_cast<float>(
            task.layer + 1);

    for (size_t i = 0; i < input.size(); ++i) {
        output[i] =
            input[i] * gain +
            0.1f + side_bias_;
    }
    return true;
}

} // namespace lt
