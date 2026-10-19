#include "inference-transport.h"
#include "transport.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

using inference_clock = std::chrono::steady_clock;

static double inference_elapsed_ms(
        inference_clock::time_point begin,
        inference_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct inference_transport::impl {
    struct tx_slot {
        std::vector<uint8_t> data;
        inference_frame_header header = {};
        size_t payload_bytes = 0;
        inference_clock::time_point enqueue_time = {};
    };

    struct rx_slot {
        std::vector<uint8_t> data;
        inference_frame_header header = {};
        size_t payload_bytes = 0;
        inference_clock::time_point ready_time = {};
    };

    socket_ptr socket;
    inference_transport_config config;

    std::vector<tx_slot> tx_slots;
    std::vector<rx_slot> rx_slots;

    mutable std::mutex tx_mu;
    mutable std::mutex rx_mu;
    mutable std::mutex stats_mu;
    mutable std::mutex state_mu;

    std::condition_variable tx_free_cv;
    std::condition_variable tx_pending_cv;
    std::condition_variable tx_idle_cv;
    std::condition_variable rx_free_cv;
    std::condition_variable rx_ready_cv;
    std::condition_variable state_cv;

    std::deque<uint32_t> tx_free;
    std::deque<uint32_t> tx_pending;
    std::deque<uint32_t> rx_free;
    std::deque<uint32_t> rx_ready;

    std::thread tx_thread;
    std::thread rx_thread;

    std::atomic<bool> stopping { false };
    std::atomic<bool> ok { true };

    size_t tx_in_progress = 0;

    inference_transport_stats stats = {};

    impl(socket_ptr socket_, const inference_transport_config & config_)
        : socket(std::move(socket_)), config(config_) {
        config.tx_slots = std::max<size_t>(config.tx_slots, 1);
        config.rx_slots = std::max<size_t>(config.rx_slots, 1);
        config.max_payload_bytes = std::max<size_t>(config.max_payload_bytes, 1);

        tx_slots.resize(config.tx_slots);
        rx_slots.resize(config.rx_slots);

        for (uint32_t i = 0; i < tx_slots.size(); ++i) {
            tx_slots[i].data.resize(config.max_payload_bytes);
            tx_free.push_back(i);
        }

        for (uint32_t i = 0; i < rx_slots.size(); ++i) {
            rx_slots[i].data.resize(config.max_payload_bytes);
            rx_free.push_back(i);
        }
    }

    ~impl() {
        stop();
    }

    void start() {
        tx_thread = std::thread([this]() { tx_loop(); });
        rx_thread = std::thread([this]() { rx_loop(); });
    }

    void notify_all() {
        tx_free_cv.notify_all();
        tx_pending_cv.notify_all();
        tx_idle_cv.notify_all();
        rx_free_cv.notify_all();
        rx_ready_cv.notify_all();
        state_cv.notify_all();
    }

    void fail() {
        ok.store(false, std::memory_order_release);
        stopping.store(true, std::memory_order_release);

        if (socket) {
            socket->shutdown();
        }

        notify_all();
    }

    void stop() {
        const bool was_stopping = stopping.exchange(true, std::memory_order_acq_rel);
        if (!was_stopping && socket) {
            socket->shutdown();
        }

        notify_all();

        if (tx_thread.joinable() && tx_thread.get_id() != std::this_thread::get_id()) {
            tx_thread.join();
        }
        if (rx_thread.joinable() && rx_thread.get_id() != std::this_thread::get_id()) {
            rx_thread.join();
        }
    }

    void drain_pending_tx() {
        {
            std::lock_guard<std::mutex> lock(tx_mu);
            while (!tx_pending.empty()) {
                const uint32_t slot_id = tx_pending.front();
                tx_pending.pop_front();
                tx_slots[slot_id].payload_bytes = 0;
                tx_free.push_back(slot_id);
            }
            if (tx_in_progress == 0) {
                tx_idle_cv.notify_all();
            }
        }
        tx_free_cv.notify_all();
    }

    void tx_loop() {
        while (true) {
            uint32_t slot_id = UINT32_MAX;

            {
                std::unique_lock<std::mutex> lock(tx_mu);
                tx_pending_cv.wait(lock, [this]() {
                    return stopping.load(std::memory_order_acquire) || !tx_pending.empty();
                });

                if (stopping.load(std::memory_order_acquire)) {
                    break;
                }

                slot_id = tx_pending.front();
                tx_pending.pop_front();
                ++tx_in_progress;
            }

            tx_slot & slot = tx_slots[slot_id];

            const auto send_begin = inference_clock::now();
            const double queue_wait_ms = inference_elapsed_ms(slot.enqueue_time, send_begin);

            const bool sent = socket &&
                socket->send_data_pair(
                    &slot.header,
                    sizeof(slot.header),
                    slot.data.data(),
                    slot.payload_bytes);

            const auto send_end = inference_clock::now();

            {
                std::lock_guard<std::mutex> lock(stats_mu);
                stats.tx_queue_wait_ms += queue_wait_ms;
                stats.tx_queue_wait_max_ms = std::max(stats.tx_queue_wait_max_ms, queue_wait_ms);
                stats.tx_send_ms += inference_elapsed_ms(send_begin, send_end);

                if (sent) {
                    ++stats.frames_sent;
                    stats.bytes_sent += slot.payload_bytes;
                }
            }

            {
                std::lock_guard<std::mutex> lock(tx_mu);
                slot.payload_bytes = 0;
                tx_free.push_back(slot_id);

                if (tx_in_progress > 0) {
                    --tx_in_progress;
                }

                if (tx_pending.empty() && tx_in_progress == 0) {
                    tx_idle_cv.notify_all();
                }
            }

            tx_free_cv.notify_one();

            if (!sent) {
                fail();
                break;
            }
        }

        drain_pending_tx();
    }

    bool read_frame_into_slot(uint32_t slot_id) {
        rx_slot & slot = rx_slots[slot_id];
        const auto recv_begin = inference_clock::now();

        if (!socket || !socket->recv_data(&slot.header, sizeof(slot.header))) {
            return false;
        }

        const bool valid =
            slot.header.magic == INFERENCE_TRANSPORT_MAGIC &&
            slot.header.version == INFERENCE_TRANSPORT_VERSION &&
            slot.header.header_bytes == sizeof(inference_frame_header) &&
            slot.header.payload_bytes <= config.max_payload_bytes;

        if (!valid) {
            return false;
        }

        slot.payload_bytes = static_cast<size_t>(slot.header.payload_bytes);

        if (slot.payload_bytes > 0 &&
            !socket->recv_data(slot.data.data(), slot.payload_bytes)) {
            return false;
        }

        const auto recv_end = inference_clock::now();
        slot.ready_time = recv_end;

        {
            std::lock_guard<std::mutex> lock(stats_mu);
            stats.rx_recv_ms += inference_elapsed_ms(recv_begin, recv_end);
            ++stats.frames_received;
            stats.bytes_received += slot.payload_bytes;
        }

        return true;
    }

    void rx_loop() {
        while (!stopping.load(std::memory_order_acquire)) {
            uint32_t slot_id = UINT32_MAX;
            bool blocked_for_slot = false;
            inference_clock::time_point wait_begin = {};

            {
                std::unique_lock<std::mutex> lock(rx_mu);
                blocked_for_slot = rx_free.empty();
                if (blocked_for_slot) {
                    wait_begin = inference_clock::now();
                }

                rx_free_cv.wait(lock, [this]() {
                    return stopping.load(std::memory_order_acquire) || !rx_free.empty();
                });

                if (stopping.load(std::memory_order_acquire)) {
                    break;
                }

                slot_id = rx_free.front();
                rx_free.pop_front();
            }

            if (blocked_for_slot) {
                const double wait_ms = inference_elapsed_ms(wait_begin, inference_clock::now());
                std::lock_guard<std::mutex> lock(stats_mu);
                ++stats.rx_slot_wait_count;
                stats.rx_slot_wait_ms += wait_ms;
            }

            if (!read_frame_into_slot(slot_id)) {
                {
                    std::lock_guard<std::mutex> lock(rx_mu);
                    rx_slots[slot_id].payload_bytes = 0;
                    rx_free.push_back(slot_id);
                }
                rx_free_cv.notify_one();

                if (!stopping.load(std::memory_order_acquire)) {
                    fail();
                }
                break;
            }

            size_t ready_size = 0;
            {
                std::lock_guard<std::mutex> lock(rx_mu);
                rx_ready.push_back(slot_id);
                ready_size = rx_ready.size();
            }

            {
                std::lock_guard<std::mutex> lock(stats_mu);
                stats.rx_ready_max = std::max(stats.rx_ready_max, ready_size);
            }

            rx_ready_cv.notify_one();
        }
    }

    inference_tx_buffer acquire_tx(size_t payload_bytes) {
        if (payload_bytes > config.max_payload_bytes ||
            stopping.load(std::memory_order_acquire)) {
            return {};
        }

        uint32_t slot_id = UINT32_MAX;
        bool blocked = false;
        inference_clock::time_point wait_begin = {};

        {
            std::unique_lock<std::mutex> lock(tx_mu);
            blocked = tx_free.empty();
            if (blocked) {
                wait_begin = inference_clock::now();
            }

            tx_free_cv.wait(lock, [this]() {
                return stopping.load(std::memory_order_acquire) || !tx_free.empty();
            });

            if (stopping.load(std::memory_order_acquire) || tx_free.empty()) {
                return {};
            }

            slot_id = tx_free.front();
            tx_free.pop_front();
        }

        if (blocked) {
            const double wait_ms = inference_elapsed_ms(wait_begin, inference_clock::now());
            std::lock_guard<std::mutex> lock(stats_mu);
            ++stats.tx_acquire_wait_count;
            stats.tx_acquire_wait_ms += wait_ms;
        }

        inference_tx_buffer result;
        result.slot = slot_id;
        result.data = tx_slots[slot_id].data.data();
        result.capacity = tx_slots[slot_id].data.size();
        return result;
    }

    void release_tx(inference_tx_buffer && tx) {
        if (!tx || tx.slot >= tx_slots.size()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(tx_mu);
            tx_slots[tx.slot].payload_bytes = 0;
            tx_free.push_back(tx.slot);
        }

        tx.slot = UINT32_MAX;
        tx.data = nullptr;
        tx.capacity = 0;

        tx_free_cv.notify_one();
    }

    bool submit_tx(
            inference_tx_buffer && tx,
            inference_frame_header header,
            size_t payload_bytes) {
        if (!tx || tx.slot >= tx_slots.size()) {
            return false;
        }

        if (payload_bytes > tx.capacity ||
            payload_bytes > config.max_payload_bytes ||
            stopping.load(std::memory_order_acquire)) {
            release_tx(std::move(tx));
            return false;
        }

        tx_slot & slot = tx_slots[tx.slot];

        header.magic         = INFERENCE_TRANSPORT_MAGIC;
        header.version       = INFERENCE_TRANSPORT_VERSION;
        header.header_bytes  = sizeof(inference_frame_header);
        header.payload_bytes = payload_bytes;

        slot.header = header;
        slot.payload_bytes = payload_bytes;
        slot.enqueue_time = inference_clock::now();

        size_t queue_size = 0;
        size_t inflight = 0;

        bool cancelled = false;
        {
            std::lock_guard<std::mutex> lock(tx_mu);
            if (stopping.load(std::memory_order_acquire)) {
                slot.payload_bytes = 0;
                tx_free.push_back(tx.slot);
                cancelled = true;
            } else {
                tx_pending.push_back(tx.slot);
                queue_size = tx_pending.size();
                inflight = tx_pending.size() + tx_in_progress;
            }
        }

        tx.slot = UINT32_MAX;
        tx.data = nullptr;
        tx.capacity = 0;

        if (cancelled) {
            tx_free_cv.notify_one();
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(stats_mu);
            stats.tx_queue_max = std::max(stats.tx_queue_max, queue_size);
            stats.tx_inflight_max = std::max(stats.tx_inflight_max, inflight);
        }

        tx_pending_cv.notify_one();
        return true;
    }

    bool receive(inference_rx_frame & frame, int timeout_ms) {
        uint32_t slot_id = UINT32_MAX;
        inference_clock::time_point ready_time = {};

        {
            std::unique_lock<std::mutex> lock(rx_mu);

            const auto ready_or_stopped = [this]() {
                return stopping.load(std::memory_order_acquire) || !rx_ready.empty();
            };

            bool woke = true;
            if (timeout_ms < 0) {
                rx_ready_cv.wait(lock, ready_or_stopped);
            } else {
                woke = rx_ready_cv.wait_for(
                    lock,
                    std::chrono::milliseconds(timeout_ms),
                    ready_or_stopped);
            }

            if (!woke || rx_ready.empty()) {
                return false;
            }

            slot_id = rx_ready.front();
            rx_ready.pop_front();
            ready_time = rx_slots[slot_id].ready_time;
        }

        const double ready_wait_ms = inference_elapsed_ms(ready_time, inference_clock::now());
        {
            std::lock_guard<std::mutex> lock(stats_mu);
            stats.rx_ready_wait_ms += ready_wait_ms;
            stats.rx_ready_wait_max_ms = std::max(stats.rx_ready_wait_max_ms, ready_wait_ms);
        }

        const rx_slot & slot = rx_slots[slot_id];
        frame.slot = slot_id;
        frame.header = slot.header;
        frame.data = slot.data.data();
        frame.size = slot.payload_bytes;
        return true;
    }

    void release_rx(inference_rx_frame && frame) {
        if (!frame || frame.slot >= rx_slots.size()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(rx_mu);
            rx_slots[frame.slot].payload_bytes = 0;
            rx_free.push_back(frame.slot);
        }

        frame.slot = UINT32_MAX;
        frame.data = nullptr;
        frame.size = 0;

        rx_free_cv.notify_one();
    }

    bool wait_tx_idle(int timeout_ms) {
        std::unique_lock<std::mutex> lock(tx_mu);

        const auto idle_or_stopped = [this]() {
            return stopping.load(std::memory_order_acquire) ||
                   (tx_pending.empty() && tx_in_progress == 0);
        };

        bool woke = true;
        if (timeout_ms < 0) {
            tx_idle_cv.wait(lock, idle_or_stopped);
        } else {
            woke = tx_idle_cv.wait_for(
                lock,
                std::chrono::milliseconds(timeout_ms),
                idle_or_stopped);
        }

        return woke &&
               !stopping.load(std::memory_order_acquire) &&
               tx_pending.empty() &&
               tx_in_progress == 0;
    }

    bool wait_closed(int timeout_ms) {
        std::unique_lock<std::mutex> lock(state_mu);
        const auto closed = [this]() {
            return stopping.load(std::memory_order_acquire);
        };

        if (timeout_ms < 0) {
            state_cv.wait(lock, closed);
            return true;
        }

        return state_cv.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            closed);
    }

    inference_transport_stats snapshot_stats() const {
        std::lock_guard<std::mutex> lock(stats_mu);
        return stats;
    }
};

inference_transport::inference_transport(std::unique_ptr<impl> impl_)
    : pimpl(std::move(impl_)) {
}

inference_transport::~inference_transport() = default;

std::unique_ptr<inference_transport> inference_transport::connect(
        const char * host,
        int port,
        const inference_transport_config & config) {
    if (!rpc_transport_init()) {
        return nullptr;
    }

    socket_ptr socket = socket_t::connect(host, port);
    if (!socket) {
        return nullptr;
    }

    auto pimpl = std::make_unique<impl>(std::move(socket), config);
    auto result = std::unique_ptr<inference_transport>(
        new inference_transport(std::move(pimpl)));
    result->pimpl->start();
    return result;
}

std::unique_ptr<inference_transport> inference_transport::accept(
        const char * host,
        int port,
        const inference_transport_config & config) {
    if (!rpc_transport_init()) {
        return nullptr;
    }

    socket_ptr listener = socket_t::create_server(host, port);
    if (!listener) {
        return nullptr;
    }

    socket_ptr socket = listener->accept();
    if (!socket) {
        return nullptr;
    }

    auto pimpl = std::make_unique<impl>(std::move(socket), config);
    auto result = std::unique_ptr<inference_transport>(
        new inference_transport(std::move(pimpl)));
    result->pimpl->start();
    return result;
}

std::unique_ptr<inference_transport> inference_transport::adopt(
        socket_ptr socket,
        const inference_transport_config & config) {
    if (!socket) {
        return nullptr;
    }

    auto pimpl = std::make_unique<impl>(std::move(socket), config);
    auto result = std::unique_ptr<inference_transport>(
        new inference_transport(std::move(pimpl)));
    result->pimpl->start();
    return result;
}

inference_tx_buffer inference_transport::acquire_tx(size_t payload_bytes) {
    return pimpl ? pimpl->acquire_tx(payload_bytes) : inference_tx_buffer {};
}

void inference_transport::release_tx(inference_tx_buffer && tx) {
    if (pimpl) {
        pimpl->release_tx(std::move(tx));
    }
}

bool inference_transport::submit_tx(
        inference_tx_buffer && tx,
        inference_frame_header header,
        size_t payload_bytes) {
    return pimpl && pimpl->submit_tx(std::move(tx), header, payload_bytes);
}

bool inference_transport::receive(inference_rx_frame & frame, int timeout_ms) {
    return pimpl && pimpl->receive(frame, timeout_ms);
}

void inference_transport::release_rx(inference_rx_frame && frame) {
    if (pimpl) {
        pimpl->release_rx(std::move(frame));
    }
}

bool inference_transport::wait_tx_idle(int timeout_ms) {
    return pimpl && pimpl->wait_tx_idle(timeout_ms);
}

bool inference_transport::wait_closed(int timeout_ms) {
    return pimpl && pimpl->wait_closed(timeout_ms);
}

bool inference_transport::healthy() const {
    return pimpl && pimpl->ok.load(std::memory_order_acquire);
}

void inference_transport::stop() {
    if (pimpl) {
        pimpl->stop();
    }
}

inference_transport_stats inference_transport::stats() const {
    return pimpl ? pimpl->snapshot_stats() : inference_transport_stats {};
}
