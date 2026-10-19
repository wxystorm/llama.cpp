#include "inference-transport.h"
#include "transport.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using bench_clock = std::chrono::steady_clock;

struct bench_options {
    bool server = false;
    std::string host;
    int port = 0;

    std::string engine = "new";
    std::string mode = "recv";

    size_t frames = 256;
    size_t bytes = 1024 * 1024;
    size_t slots = 8;
};

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage:\n"
        "  %s --server HOST PORT [--engine new|legacy] [--mode recv|send|bidir]\n"
        "       [--frames N] [--bytes N] [--slots N]\n"
        "  %s --client HOST PORT [--engine new|legacy] [--mode recv|send|bidir]\n"
        "       [--frames N] [--bytes N] [--slots N]\n",
        argv0,
        argv0);
}

static bool parse_size(const char * s, size_t & out) {
    if (s == nullptr || s[0] == '\0') {
        return false;
    }

    char * end = nullptr;
    const unsigned long long value = std::strtoull(s, &end, 10);
    if (end == nullptr || *end != '\0') {
        return false;
    }

    out = static_cast<size_t>(value);
    return true;
}

static bool parse_args(int argc, char ** argv, bench_options & opt) {
    if (argc < 4) {
        return false;
    }

    int i = 1;
    if (std::strcmp(argv[i], "--server") == 0) {
        opt.server = true;
    } else if (std::strcmp(argv[i], "--client") == 0) {
        opt.server = false;
    } else {
        return false;
    }

    opt.host = argv[++i];
    opt.port = std::atoi(argv[++i]);
    ++i;

    while (i < argc) {
        if (std::strcmp(argv[i], "--engine") == 0 && i + 1 < argc) {
            opt.engine = argv[++i];
        } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            opt.mode = argv[++i];
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], opt.frames)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], opt.bytes)) {
                return false;
            }
        } else if (std::strcmp(argv[i], "--slots") == 0 && i + 1 < argc) {
            if (!parse_size(argv[++i], opt.slots)) {
                return false;
            }
        } else {
            return false;
        }
        ++i;
    }

    return opt.port > 0 &&
           opt.frames > 0 &&
           opt.bytes > 0 &&
           opt.slots > 0 &&
           (opt.engine == "new" || opt.engine == "legacy") &&
           (opt.mode == "send" || opt.mode == "recv" || opt.mode == "bidir");
}

static uint8_t payload_pattern(uint64_t seq) {
    return static_cast<uint8_t>((seq * 1315423911ULL) & 0xff);
}

static void fill_payload(uint8_t * data, size_t size, uint64_t seq) {
    std::memset(data, payload_pattern(seq), size);
}

static bool verify_payload(const uint8_t * data, size_t size, uint64_t seq) {
    if (size == 0) {
        return true;
    }

    const uint8_t expected = payload_pattern(seq);
    const size_t q1 = size / 4;
    const size_t q2 = size / 2;
    const size_t q3 = (size * 3) / 4;

    return data[0] == expected &&
           data[q1] == expected &&
           data[q2] == expected &&
           data[q3] == expected &&
           data[size - 1] == expected;
}

static inference_frame_header make_data_header(uint64_t seq) {
    inference_frame_header h = inference_make_header(INFERENCE_FRAME_DATA);
    h.graph_id = 1;
    h.seq = seq;
    h.chunk = static_cast<int32_t>(seq);
    h.stage = INFERENCE_STAGE_ATTN_OUT;
    return h;
}

static bool new_send_data_frames(
        inference_transport & transport,
        const bench_options & opt) {
    for (size_t i = 0; i < opt.frames; ++i) {
        inference_tx_buffer tx = transport.acquire_tx(opt.bytes);
        if (!tx) {
            return false;
        }

        fill_payload(tx.data, opt.bytes, i);

        if (!transport.submit_tx(
                std::move(tx),
                make_data_header(i),
                opt.bytes)) {
            return false;
        }
    }

    return transport.wait_tx_idle(-1);
}

static bool new_send_ack(inference_transport & transport, uint64_t seq) {
    inference_tx_buffer tx = transport.acquire_tx(0);
    if (!tx) {
        return false;
    }

    inference_frame_header h = inference_make_header(INFERENCE_FRAME_ACK);
    h.graph_id = 1;
    h.seq = seq;

    return transport.submit_tx(std::move(tx), h, 0);
}

static bool run_new(
        const bench_options & opt,
        inference_transport_stats & out_stats) {
    inference_transport_config config;
    config.tx_slots = opt.slots;
    config.rx_slots = opt.slots;
    config.max_payload_bytes = opt.bytes;

    std::unique_ptr<inference_transport> transport;
    if (opt.server) {
        std::printf(
            "[INFERENCE_TRANSPORT_BENCH] engine=new listening=%s:%d\n",
            opt.host.c_str(),
            opt.port);
        std::fflush(stdout);
        transport = inference_transport::accept(opt.host.c_str(), opt.port, config);
    } else {
        transport = inference_transport::connect(opt.host.c_str(), opt.port, config);
    }

    if (!transport) {
        std::fprintf(stderr, "failed to create new inference transport\n");
        return false;
    }

    const bool need_send = opt.mode == "send" || opt.mode == "bidir";
    const bool need_recv = opt.mode == "recv" || opt.mode == "bidir";

    bool sender_ok = true;
    std::thread sender;

    if (need_send) {
        sender = std::thread([&]() {
            sender_ok = new_send_data_frames(*transport, opt);
        });
    }

    size_t data_received = 0;
    bool ack_received = !need_send;
    bool ack_sent = !need_recv;
    bool receiver_ok = true;

    while (receiver_ok &&
           (data_received < (need_recv ? opt.frames : 0) || !ack_received)) {
        inference_rx_frame frame;
        if (!transport->receive(frame, -1)) {
            receiver_ok = false;
            break;
        }

        if (frame.header.type == INFERENCE_FRAME_DATA) {
            const uint64_t expected_seq = data_received;

            if (!need_recv ||
                frame.header.seq != expected_seq ||
                frame.size != opt.bytes ||
                !verify_payload(frame.data, frame.size, frame.header.seq)) {
                receiver_ok = false;
            } else {
                ++data_received;
            }
        } else if (frame.header.type == INFERENCE_FRAME_ACK) {
            if (!need_send) {
                receiver_ok = false;
            } else {
                ack_received = true;
            }
        } else {
            receiver_ok = false;
        }

        transport->release_rx(std::move(frame));

        if (receiver_ok &&
            need_recv &&
            !ack_sent &&
            data_received == opt.frames) {
            ack_sent = new_send_ack(*transport, opt.frames - 1);
            if (!ack_sent) {
                receiver_ok = false;
            }
        }
    }

    if (sender.joinable()) {
        sender.join();
    }

    if (receiver_ok && ack_sent) {
        receiver_ok = transport->wait_tx_idle(-1);
    }

    const bool success =
        sender_ok &&
        receiver_ok &&
        ack_received &&
        transport->healthy();

    out_stats = transport->stats();
    transport->stop();

    return success;
}

static bool legacy_recv_one(
        socket_t & socket,
        const bench_options & opt,
        uint64_t expected_seq,
        std::vector<uint8_t> & payload,
        inference_transport_stats & stats) {
    inference_frame_header h = {};

    if (!socket.recv_data(&h, sizeof(h))) {
        return false;
    }

    if (h.magic != INFERENCE_TRANSPORT_MAGIC ||
        h.version != INFERENCE_TRANSPORT_VERSION ||
        h.header_bytes != sizeof(inference_frame_header) ||
        h.type != INFERENCE_FRAME_DATA ||
        h.seq != expected_seq ||
        h.payload_bytes != opt.bytes) {
        return false;
    }

    if (!socket.recv_data(payload.data(), payload.size())) {
        return false;
    }

    if (!verify_payload(payload.data(), payload.size(), h.seq)) {
        return false;
    }

    uint64_t ack = h.seq;
    if (!socket.send_data(&ack, sizeof(ack))) {
        return false;
    }

    ++stats.frames_received;
    stats.bytes_received += payload.size();
    return true;
}

static bool legacy_send_one(
        socket_t & socket,
        const bench_options & opt,
        uint64_t seq,
        std::vector<uint8_t> & payload,
        inference_transport_stats & stats) {
    fill_payload(payload.data(), payload.size(), seq);

    inference_frame_header h = make_data_header(seq);
    h.payload_bytes = payload.size();

    if (!socket.send_data(&h, sizeof(h)) ||
        !socket.send_data(payload.data(), payload.size())) {
        return false;
    }

    uint64_t ack = UINT64_MAX;
    if (!socket.recv_data(&ack, sizeof(ack)) || ack != seq) {
        return false;
    }

    ++stats.frames_sent;
    stats.bytes_sent += payload.size();
    return true;
}

static bool run_legacy(
        const bench_options & opt,
        inference_transport_stats & stats) {
    if (!rpc_transport_init()) {
        return false;
    }

    socket_ptr socket;

    if (opt.server) {
        std::printf(
            "[INFERENCE_TRANSPORT_BENCH] engine=legacy listening=%s:%d\n",
            opt.host.c_str(),
            opt.port);
        std::fflush(stdout);

        socket_ptr listener = socket_t::create_server(opt.host.c_str(), opt.port);
        if (!listener) {
            return false;
        }

        socket = listener->accept();
    } else {
        socket = socket_t::connect(opt.host.c_str(), opt.port);
    }

    if (!socket) {
        return false;
    }

    std::vector<uint8_t> payload(opt.bytes);

    if (opt.mode == "send") {
        for (size_t i = 0; i < opt.frames; ++i) {
            if (!legacy_send_one(*socket, opt, i, payload, stats)) {
                return false;
            }
        }
        return true;
    }

    if (opt.mode == "recv") {
        for (size_t i = 0; i < opt.frames; ++i) {
            if (!legacy_recv_one(*socket, opt, i, payload, stats)) {
                return false;
            }
        }
        return true;
    }

    for (size_t i = 0; i < opt.frames; ++i) {
        if (opt.server) {
            if (!legacy_recv_one(*socket, opt, i, payload, stats) ||
                !legacy_send_one(*socket, opt, i, payload, stats)) {
                return false;
            }
        } else {
            if (!legacy_send_one(*socket, opt, i, payload, stats) ||
                !legacy_recv_one(*socket, opt, i, payload, stats)) {
                return false;
            }
        }
    }

    return true;
}

static void print_stats(
        const bench_options & opt,
        const inference_transport_stats & s,
        double wall_ms,
        bool success) {
    const double tx_mib = static_cast<double>(s.bytes_sent) / (1024.0 * 1024.0);
    const double rx_mib = static_cast<double>(s.bytes_received) / (1024.0 * 1024.0);
    const double wall_s = wall_ms / 1000.0;

    const double tx_mib_s = wall_s > 0.0 ? tx_mib / wall_s : 0.0;
    const double rx_mib_s = wall_s > 0.0 ? rx_mib / wall_s : 0.0;
    const double aggregate_mib_s =
        wall_s > 0.0 ? (tx_mib + rx_mib) / wall_s : 0.0;

    std::printf(
        "[INFERENCE_TRANSPORT_BENCH] "
        "status=%s engine=%s role=%s mode=%s frames=%zu payload=%zu slots=%zu "
        "wall_ms=%.3f "
        "tx_frames=%llu rx_frames=%llu tx_mib=%.3f rx_mib=%.3f "
        "tx_mib_s=%.3f rx_mib_s=%.3f aggregate_mib_s=%.3f "
        "tx_queue_max=%zu tx_inflight_max=%zu rx_ready_max=%zu "
        "tx_acquire_wait_count=%llu tx_acquire_wait_ms=%.3f "
        "rx_slot_wait_count=%llu rx_slot_wait_ms=%.3f "
        "tx_queue_wait_ms=%.3f tx_queue_wait_max_ms=%.3f "
        "tx_send_ms=%.3f rx_recv_ms=%.3f "
        "rx_ready_wait_ms=%.3f rx_ready_wait_max_ms=%.3f\n",
        success ? "OK" : "FAIL",
        opt.engine.c_str(),
        opt.server ? "server" : "client",
        opt.mode.c_str(),
        opt.frames,
        opt.bytes,
        opt.slots,
        wall_ms,
        static_cast<unsigned long long>(s.frames_sent),
        static_cast<unsigned long long>(s.frames_received),
        tx_mib,
        rx_mib,
        tx_mib_s,
        rx_mib_s,
        aggregate_mib_s,
        s.tx_queue_max,
        s.tx_inflight_max,
        s.rx_ready_max,
        static_cast<unsigned long long>(s.tx_acquire_wait_count),
        s.tx_acquire_wait_ms,
        static_cast<unsigned long long>(s.rx_slot_wait_count),
        s.rx_slot_wait_ms,
        s.tx_queue_wait_ms,
        s.tx_queue_wait_max_ms,
        s.tx_send_ms,
        s.rx_recv_ms,
        s.rx_ready_wait_ms,
        s.rx_ready_wait_max_ms);
}

int main(int argc, char ** argv) {
    bench_options opt;

    if (!parse_args(argc, argv, opt)) {
        usage(argv[0]);
        return 2;
    }

    inference_transport_stats stats = {};

    const auto begin = bench_clock::now();

    const bool success =
        opt.engine == "new"
            ? run_new(opt, stats)
            : run_legacy(opt, stats);

    const auto end = bench_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();

    print_stats(opt, stats, wall_ms, success);
    return success ? 0 : 1;
}
