#include "inference-cycle-transport.h"
#include "inference-cycle-reduce.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#ifdef _WIN32
#  include <process.h>
#else
#  include <unistd.h>
#endif

struct cycle_fixture {
    uint32_t tokens = 2;
    uint32_t hidden = 8;
    uint32_t topk = 2;
    std::vector<float> residual = std::vector<float>(16, 1.0f);
    std::vector<float> norm = std::vector<float>(16, 0.5f);
    std::vector<float> pc_ffn = std::vector<float>(16, 0.25f);
    std::vector<float> phone_ffn = std::vector<float>(16, 0.5f);
    std::vector<int32_t> ids = {3, 5, 3, 5};
    std::vector<float> weights = {0.75f, 0.25f, 0.75f, 0.25f};

    inference_cycle_ready_view ready() const {
        return {tokens, hidden, topk,
                residual.data(), norm.data(), ids.data(), weights.data()};
    }
    inference_cycle_partial_view partial() const {
        return {tokens, hidden, phone_ffn.data()};
    }
};

static bool send_pair(
        inference_transport & io, const cycle_fixture & f,
        inference_cycle_direction direction, int layer, uint64_t seq) {
    return inference_cycle_send_ready(
               io, direction, layer, 0, seq, 5, f.ready()) &&
           inference_cycle_send_partial(
               io, direction, layer, 0, seq, 5, f.partial());
}

static bool recv_pair(
        inference_transport & io, const cycle_fixture & f,
        inference_cycle_direction direction, int layer, uint64_t seq,
        bool reduce_pc) {
    inference_cycle_message ready;
    inference_cycle_message partial;
    if (!inference_cycle_receive(io, ready, 3000) ||
        ready.kind != INFERENCE_CYCLE_READY ||
        ready.direction != direction ||
        ready.layer != layer || ready.chunk != 0 || ready.seq != seq ||
        ready.graph_id != 5 ||
        ready.ready.residual != f.residual ||
        ready.ready.ffn_norm != f.norm ||
        ready.ready.topk_ids != f.ids ||
        ready.ready.topk_weights != f.weights) {
        return false;
    }
    // On the actual Phone GPU, the upload event/fence must be complete
    // before this ACK. This socket test uses CPU-owned host buffers only.
    if (!inference_cycle_send_ack(io, ready, true)) {
        return false;
    }
    if (!inference_cycle_receive(io, partial, 3000) ||
        partial.kind != INFERENCE_CYCLE_PARTIAL ||
        partial.direction != direction ||
        partial.layer != layer || partial.chunk != 0 || partial.seq != seq ||
        partial.graph_id != 5 ||
        partial.partial.partial != f.phone_ffn) {
        return false;
    }

    if (reduce_pc) {
        std::vector<float> output(f.residual.size(), 0.0f);
        if (!inference_cycle_reduce_p2c_pc_cpu(
                ready.ready,
                f.pc_ffn.data(), f.pc_ffn.size(),
                partial.partial, output.data(), output.size())) {
            return false;
        }
        for (float value : output) {
            if (std::fabs(value - 1.75f) > 1.0e-6f) {
                return false;
            }
        }
    }
    return inference_cycle_send_ack(io, partial, true);
}

static bool recv_acks(
        inference_transport & io, inference_cycle_direction direction,
        int layer, uint64_t seq) {
    for (int i = 0; i < 2; ++i) {
        inference_rx_frame frame;
        if (!io.receive(frame, 3000)) {
            return false;
        }
        inference_cycle_ack ack;
        const bool decoded = inference_cycle_decode_ack(
            frame.header, frame.size, ack);
        io.release_rx(std::move(frame));
        if (!decoded || !ack.device_ready ||
            ack.direction != direction ||
            ack.layer != layer || ack.chunk != 0 ||
            ack.seq != seq || ack.graph_id != 5 ||
            ack.kind != (i == 0 ? INFERENCE_CYCLE_READY : INFERENCE_CYCLE_PARTIAL)) {
            return false;
        }
    }
    return true;
}

static bool run_socket_test() {
    if (!rpc_transport_init()) {
        return false;
    }
#ifdef _WIN32
    const unsigned pid = static_cast<unsigned>(_getpid());
#else
    const unsigned pid = static_cast<unsigned>(getpid());
#endif
    const int first_port = 20000 + int(pid % 20000);

    socket_ptr listener;
    int port = 0;
    for (int i = 0; i < 32 && !listener; ++i) {
        port = first_port + i;
        listener = socket_t::create_server("127.0.0.1", port);
    }
    if (!listener) {
        return false;
    }

    inference_transport_config config;
    config.tx_slots = 2;
    config.rx_slots = 2;
    config.max_payload_bytes = 128 * 1024;
    const cycle_fixture fixture;

    bool server_ok = false;
    std::thread server([&]() {
        const socket_ptr socket = listener->accept();
        if (!socket) {
            return;
        }
        auto io = inference_transport::adopt(socket, config);
        if (!io) {
            return;
        }
        // First, Phone->PC: Receive the real two Tensor data frames, reduce
        // on the simulated PC CPU, then return only tiny ready ACKs.
        const bool p2c_ok = recv_pair(
            *io, fixture, INFERENCE_CYCLE_PHONE_TO_PC, 19, 100, true);
        // Next, PC->Phone: Send two DATA frames, receive their reverse ACKs.
        const bool c2p_ok = p2c_ok &&
            send_pair(*io, fixture, INFERENCE_CYCLE_PC_TO_PHONE, 10, 101) &&
            recv_acks(*io, INFERENCE_CYCLE_PC_TO_PHONE, 10, 101);
        server_ok = p2c_ok && c2p_ok && io->wait_tx_idle(3000);
        io->stop();
    });

    const socket_ptr socket = socket_t::connect("127.0.0.1", port);
    if (!socket) {
        listener->shutdown();
        server.join();
        return false;
    }
    auto io = inference_transport::adopt(socket, config);
    if (!io) {
        listener->shutdown();
        server.join();
        return false;
    }
    const bool p2c_ok =
        send_pair(*io, fixture, INFERENCE_CYCLE_PHONE_TO_PC, 19, 100) &&
        recv_acks(*io, INFERENCE_CYCLE_PHONE_TO_PC, 19, 100);
    const bool c2p_ok = p2c_ok &&
        recv_pair(*io, fixture, INFERENCE_CYCLE_PC_TO_PHONE, 10, 101, false);
    const bool client_ok = p2c_ok && c2p_ok && io->wait_tx_idle(3000);
    server.join();
    io->stop();

    std::printf("[CYCLE_WIRE_SMOKE] p2c=%d c2p=%d pc_reduce=%d "
                "client_ok=%d server_ok=%d\n",
                p2c_ok ? 1 : 0, c2p_ok ? 1 : 0, p2c_ok ? 1 : 0,
                client_ok ? 1 : 0, server_ok ? 1 : 0);
    return client_ok && server_ok;
}

int main() {
    return run_socket_test() ? 0 : 1;
}
