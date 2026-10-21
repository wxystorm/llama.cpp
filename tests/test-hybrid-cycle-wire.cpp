#include "inference-cycle-transport.h"
#include "inference-cycle-reduce.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

static inference_frame_header make_cycle_frame(
        uint16_t stage, inference_cycle_direction direction,
        uint64_t payload, int layer = 19, int chunk = 2,
        uint64_t seq = 42, uint64_t graph_id = 123) {
    inference_frame_header h = inference_make_header(INFERENCE_FRAME_DATA);
    h.stage = stage;
    h.tensor_role = direction;
    h.payload_bytes = payload;
    h.layer = layer;
    h.chunk = chunk;
    h.seq = seq;
    h.graph_id = graph_id;
    return h;
}

static void test_p2c_payload_and_cpu_reduce() {
    constexpr uint32_t tokens = 3;
    constexpr uint32_t hidden = 8;
    constexpr uint32_t topk = 2;
    constexpr size_t n = size_t(tokens) * hidden;

    std::vector<float> residual(n);
    std::vector<float> norm(n);
    std::vector<float> pc_ffn(n);
    std::vector<float> phone_ffn(n);
    std::vector<int32_t> ids(size_t(tokens) * topk);
    std::vector<float> weights(size_t(tokens) * topk);

    for (size_t i = 0; i < n; ++i) {
        residual[i] = static_cast<float>(i + 1) * 0.5f;
        norm[i] = static_cast<float>(i + 1) / 32.0f;
        pc_ffn[i] = static_cast<float>(i + 1) / 4.0f;
        phone_ffn[i] = static_cast<float>(i + 1) / 8.0f;
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = int32_t(100 + i);
        weights[i] = (i % 2) == 0 ? 0.75f : 0.25f;
    }

    const inference_cycle_ready_view ready {tokens, hidden, topk,
        residual.data(), norm.data(), ids.data(), weights.data()};
    const inference_cycle_partial_view partial {tokens, hidden, phone_ffn.data()};
    size_t ready_size = 0;
    size_t partial_size = 0;
    assert(inference_cycle_payload_size(INFERENCE_CYCLE_READY,
        tokens, hidden, topk, ready_size));
    assert(inference_cycle_payload_size(INFERENCE_CYCLE_PARTIAL,
        tokens, hidden, 0, partial_size));

    std::vector<uint8_t> ready_wire(ready_size);
    std::vector<uint8_t> partial_wire(partial_size);
    size_t written = 0;
    assert(inference_cycle_write_ready(ready_wire.data(), ready_wire.size(), ready, written));
    assert(written == ready_size);
    assert(inference_cycle_write_partial(partial_wire.data(), partial_wire.size(), partial, written));
    assert(written == partial_size);

    const auto header1 = make_cycle_frame(
        INFERENCE_STAGE_CYCLE_READY, INFERENCE_CYCLE_PHONE_TO_PC, ready_size);
    const auto header2 = make_cycle_frame(
        INFERENCE_STAGE_CYCLE_PARTIAL, INFERENCE_CYCLE_PHONE_TO_PC, partial_size);
    inference_cycle_message msg1;
    inference_cycle_message msg2;
    assert(inference_cycle_decode_frame(header1, ready_wire.data(), ready_wire.size(), msg1));
    assert(inference_cycle_decode_frame(header2, partial_wire.data(), partial_wire.size(), msg2));
    assert(msg1.kind == INFERENCE_CYCLE_READY);
    assert(msg2.kind == INFERENCE_CYCLE_PARTIAL);
    assert(msg1.direction == INFERENCE_CYCLE_PHONE_TO_PC);
    assert(msg2.direction == INFERENCE_CYCLE_PHONE_TO_PC);
    assert(msg1.seq == 42 && msg2.seq == 42);
    assert(msg1.layer == 19 && msg2.layer == 19);
    assert(msg1.ready.residual == residual);
    assert(msg1.ready.ffn_norm == norm);
    assert(msg1.ready.topk_ids == ids);
    assert(msg1.ready.topk_weights == weights);
    assert(msg2.partial.partial == phone_ffn);

    std::vector<float> result(n, 0.0f);
    assert(inference_cycle_reduce_p2c_pc_cpu(
        msg1.ready, pc_ffn.data(), pc_ffn.size(),
        msg2.partial, result.data(), result.size()));
    for (size_t i = 0; i < n; ++i) {
        assert(std::fabs(result[i] -
            (residual[i] + (pc_ffn[i] + phone_ffn[i]))) < 1.0e-6f);
    }

    // Invalid size/dimensions must fail rather than truncate a partial.
    assert(!inference_cycle_reduce_p2c_pc_cpu(
        msg1.ready, pc_ffn.data(), pc_ffn.size() - 1,
        msg2.partial, result.data(), result.size()));
    auto mismatch = msg2.partial;
    ++mismatch.hidden;
    assert(!inference_cycle_reduce_p2c_pc_cpu(
        msg1.ready, pc_ffn.data(), pc_ffn.size(),
        mismatch, result.data(), result.size()));

    // Correct frame type, incorrect framing length.
    inference_cycle_message bad;
    assert(!inference_cycle_decode_frame(
        header1, ready_wire.data(), ready_wire.size() - 1, bad));
    auto wrong_stage = header1;
    wrong_stage.stage = INFERENCE_STAGE_CYCLE_PARTIAL;
    assert(!inference_cycle_decode_frame(
        wrong_stage, ready_wire.data(), ready_wire.size(), bad));
    auto wrong_role = header1;
    wrong_role.tensor_role = 0;
    assert(!inference_cycle_decode_frame(
        wrong_role, ready_wire.data(), ready_wire.size(), bad));
    auto wrong_seq = header1;
    wrong_seq.seq = 0;
    assert(!inference_cycle_decode_frame(
        wrong_seq, ready_wire.data(), ready_wire.size(), bad));
    ready_wire[0] ^= 0xff;
    assert(!inference_cycle_decode_frame(
        header1, ready_wire.data(), ready_wire.size(), bad));
}

static void test_ack_after_device_ready() {
    inference_frame_header ack = inference_make_header(INFERENCE_FRAME_ACK);
    ack.stage = INFERENCE_STAGE_CYCLE_READY;
    // Original tensor direction is Phone->PC, although ACK travels PC->Phone.
    ack.tensor_role = INFERENCE_CYCLE_PHONE_TO_PC;
    ack.flags = INFERENCE_CYCLE_ACK_DEVICE_READY;
    ack.layer = 19;
    ack.chunk = 2;
    ack.seq = 42;
    ack.graph_id = 123;

    inference_cycle_ack received;
    assert(inference_cycle_decode_ack(ack, 0, received));
    assert(received.device_ready);
    assert(received.kind == INFERENCE_CYCLE_READY);
    assert(received.direction == INFERENCE_CYCLE_PHONE_TO_PC);
    assert(received.layer == 19 && received.chunk == 2);
    assert(received.seq == 42 && received.graph_id == 123);

    ack.stage = INFERENCE_STAGE_CYCLE_PARTIAL;
    ack.flags = INFERENCE_CYCLE_ACK_DEVICE_FAILED;
    assert(inference_cycle_decode_ack(ack, 0, received));
    assert(!received.device_ready);
    assert(received.kind == INFERENCE_CYCLE_PARTIAL);

    ack.flags = 0;
    assert(!inference_cycle_decode_ack(ack, 0, received));
    ack.flags = INFERENCE_CYCLE_ACK_DEVICE_READY;
    assert(!inference_cycle_decode_ack(ack, 1, received));
    ack.seq = 0;
    assert(!inference_cycle_decode_ack(ack, 0, received));
}

static void test_large_chunk_and_c2p() {
    constexpr uint32_t tokens = 128;
    constexpr uint32_t hidden = 2048;
    constexpr uint32_t topk = 8;
    const size_t n = size_t(tokens) * hidden;
    const size_t route_n = size_t(tokens) * topk;
    std::vector<float> residual(n, 1.25f);
    std::vector<float> norm(n, 0.5f);
    std::vector<int32_t> topk_ids(route_n, 13);
    std::vector<float> weights(route_n, 0.125f);
    const inference_cycle_ready_view v {tokens, hidden, topk,
        residual.data(), norm.data(), topk_ids.data(), weights.data()};
    size_t ready_bytes = 0;
    assert(inference_cycle_payload_size(INFERENCE_CYCLE_READY,
        tokens, hidden, topk, ready_bytes));
    assert(ready_bytes < 4u * 1024u * 1024u);
    std::vector<uint8_t> buffer(ready_bytes);
    size_t written = 0;
    assert(inference_cycle_write_ready(buffer.data(), buffer.size(), v, written));
    assert(written == ready_bytes);

    const auto header = make_cycle_frame(
        INFERENCE_STAGE_CYCLE_READY, INFERENCE_CYCLE_PC_TO_PHONE,
        ready_bytes, 10, 3, 2);
    inference_cycle_message parsed;
    assert(inference_cycle_decode_frame(header, buffer.data(), buffer.size(), parsed));
    assert(parsed.direction == INFERENCE_CYCLE_PC_TO_PHONE);
    assert(parsed.ready.residual == residual);
    assert(parsed.ready.ffn_norm == norm);
    assert(parsed.ready.topk_ids == topk_ids);
    assert(parsed.ready.topk_weights == weights);
}

// C2P uses the SAME two-send wire encoding, but the Phone GPU is now the
// destination and the sole Reduce owner. Check per-chunk frame identity
// and that the reverse ACKs preserve the ORIGINAL PC->Phone direction.
static void test_c2p_two_sends_per_chunk_and_reverse_ack() {
    constexpr uint32_t tokens = 16;
    constexpr uint32_t hidden = 2048;
    constexpr uint32_t topk = 8;
    const size_t n = size_t(tokens) * hidden;
    const size_t k = size_t(tokens) * topk;
    std::vector<float> residual(n, 0.25f);
    std::vector<float> norm(n, 0.5f);
    std::vector<int32_t> ids(k, 7);
    std::vector<float> weights(k, 0.125f);
    std::vector<float> pc_partial(n, 0.375f);
    const inference_cycle_ready_view ready {
        tokens, hidden, topk, residual.data(),
        norm.data(), ids.data(), weights.data()
    };
    const inference_cycle_partial_view partial {
        tokens, hidden, pc_partial.data()
    };
    size_t ready_size = 0, partial_size = 0;
    assert(inference_cycle_payload_size(
        INFERENCE_CYCLE_READY, tokens, hidden, topk, ready_size));
    assert(inference_cycle_payload_size(
        INFERENCE_CYCLE_PARTIAL, tokens, hidden, 0, partial_size));

    for (int chunk = 0; chunk < 4; ++chunk) {
        const uint64_t seq = uint64_t(chunk + 1);
        std::vector<uint8_t> ready_wire(ready_size);
        std::vector<uint8_t> partial_wire(partial_size);
        size_t written = 0;
        assert(inference_cycle_write_ready(
            ready_wire.data(), ready_wire.size(), ready, written));
        assert(written == ready_size);
        assert(inference_cycle_write_partial(
            partial_wire.data(), partial_wire.size(), partial, written));
        assert(written == partial_size);

        const auto rframe = make_cycle_frame(
            INFERENCE_STAGE_CYCLE_READY,
            INFERENCE_CYCLE_PC_TO_PHONE, ready_size, 5, chunk, seq, 123);
        const auto pframe = make_cycle_frame(
            INFERENCE_STAGE_CYCLE_PARTIAL,
            INFERENCE_CYCLE_PC_TO_PHONE, partial_size, 5, chunk, seq, 123);
        inference_cycle_message rmsg, pmsg;
        assert(inference_cycle_decode_frame(
            rframe, ready_wire.data(), ready_wire.size(), rmsg));
        assert(inference_cycle_decode_frame(
            pframe, partial_wire.data(), partial_wire.size(), pmsg));
        assert(rmsg.direction == INFERENCE_CYCLE_PC_TO_PHONE);
        assert(pmsg.direction == INFERENCE_CYCLE_PC_TO_PHONE);
        assert(rmsg.kind == INFERENCE_CYCLE_READY);
        assert(pmsg.kind == INFERENCE_CYCLE_PARTIAL);
        assert(rmsg.chunk == chunk && pmsg.chunk == chunk);
        assert(rmsg.seq == seq && pmsg.seq == seq);
        assert(rmsg.ready.ffn_norm == norm);
        assert(pmsg.partial.partial == pc_partial);

        for (const auto stage : {INFERENCE_STAGE_CYCLE_READY,
                                 INFERENCE_STAGE_CYCLE_PARTIAL}) {
            auto ack = inference_make_header(INFERENCE_FRAME_ACK);
            ack.stage = stage;
            ack.tensor_role = INFERENCE_CYCLE_PC_TO_PHONE;
            ack.flags = INFERENCE_CYCLE_ACK_DEVICE_READY;
            ack.layer = 5;
            ack.chunk = chunk;
            ack.seq = seq;
            ack.graph_id = 123;
            inference_cycle_ack received;
            assert(inference_cycle_decode_ack(ack, 0, received));
            assert(received.direction == INFERENCE_CYCLE_PC_TO_PHONE);
            assert(received.kind == (stage == INFERENCE_STAGE_CYCLE_READY ?
                INFERENCE_CYCLE_READY : INFERENCE_CYCLE_PARTIAL));
            assert(received.device_ready);
        }
    }
}

int main() {
    test_p2c_payload_and_cpu_reduce();
    test_ack_after_device_ready();
    test_large_chunk_and_c2p();
    test_c2p_two_sends_per_chunk_and_reverse_ack();
    return 0;
}
