#pragma once

#include "inference-transport-protocol.h"

#include <cstddef>
#include <cstdint>
#include <memory>

struct inference_transport_config {
    size_t tx_slots          = 8;
    size_t rx_slots          = 8;
    size_t max_payload_bytes = 4 * 1024 * 1024;
};

struct inference_tx_buffer {
    uint32_t slot     = UINT32_MAX;
    uint8_t * data    = nullptr;
    size_t   capacity = 0;

    explicit operator bool() const {
        return slot != UINT32_MAX && data != nullptr;
    }
};

struct inference_rx_frame {
    uint32_t slot = UINT32_MAX;
    inference_frame_header header = {};
    const uint8_t * data = nullptr;
    size_t size = 0;

    explicit operator bool() const {
        return slot != UINT32_MAX && data != nullptr;
    }
};

struct inference_transport_stats {
    uint64_t frames_sent     = 0;
    uint64_t frames_received = 0;
    uint64_t bytes_sent      = 0;
    uint64_t bytes_received  = 0;

    uint64_t tx_acquire_wait_count = 0;
    uint64_t rx_slot_wait_count    = 0;

    double tx_acquire_wait_ms = 0.0;
    double rx_slot_wait_ms    = 0.0;
    double tx_queue_wait_ms   = 0.0;
    double tx_queue_wait_max_ms = 0.0;
    double tx_send_ms         = 0.0;
    double rx_recv_ms         = 0.0;
    double rx_ready_wait_ms   = 0.0;
    double rx_ready_wait_max_ms = 0.0;

    size_t tx_queue_max    = 0;
    size_t tx_inflight_max = 0;
    size_t rx_ready_max    = 0;
};

class inference_transport {
public:
    ~inference_transport();

    inference_transport(const inference_transport &) = delete;
    inference_transport & operator=(const inference_transport &) = delete;

    static std::unique_ptr<inference_transport> connect(
        const char * host,
        int port,
        const inference_transport_config & config = {});

    static std::unique_ptr<inference_transport> accept(
        const char * host,
        int port,
        const inference_transport_config & config = {});

    // Producer owns the returned slot until submit_tx() or release_tx().
    inference_tx_buffer acquire_tx(size_t payload_bytes);
    void release_tx(inference_tx_buffer && tx);

    bool submit_tx(
        inference_tx_buffer && tx,
        inference_frame_header header,
        size_t payload_bytes);

    // Consumer owns an RX slot until release_rx().
    bool receive(inference_rx_frame & frame, int timeout_ms = -1);
    void release_rx(inference_rx_frame && frame);

    bool wait_tx_idle(int timeout_ms = -1);

    bool healthy() const;
    void stop();

    inference_transport_stats stats() const;

private:
    struct impl;
    explicit inference_transport(std::unique_ptr<impl> impl);
    std::unique_ptr<impl> pimpl;
};
