#include "llama-hybrid-cycle-wire.h"
#include "../ggml/include/ggml-rpc.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>

// Existing RPC data-plane, independent matching RX dispatcher.
bool llama_hybrid_cycle_receive_wire(
        ggml_backend_t phone, const llama_hybrid_cycle_live_wire & wire,
        inference_cycle_payload_kind kind,
        uint32_t tokens, uint32_t hidden, uint32_t topk,
        inference_cycle_message & output) {
    auto * dev = ggml_backend_get_device(phone);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return false;
    }
    auto receive = reinterpret_cast<
        ggml_backend_rpc_inference_transport_receive_matching_t>(
            ggml_backend_reg_get_proc_address(
                reg, GGML_BACKEND_RPC_INFERENCE_TRANSPORT_RECEIVE_MATCHING_PROC));
    if (!receive) {
        return false;
    }
    size_t bytes = 0;
    if (!inference_cycle_payload_size(
            kind, tokens, hidden,
            kind == INFERENCE_CYCLE_READY ? topk : 0, bytes)) {
        return false;
    }
    std::vector<uint8_t> frame_data(bytes);
    const uint16_t stage = kind == INFERENCE_CYCLE_READY ?
        INFERENCE_STAGE_CYCLE_READY : INFERENCE_STAGE_CYCLE_PARTIAL;
    ggml_backend_rpc_inference_task task {};
    task.stage = stage;
    task.stream_id = 0; // Cycle frames use the default inference stream.
    task.tensor_role = INFERENCE_CYCLE_PHONE_TO_PC;
    task.layer = wire.layer;
    task.chunk = wire.chunk;
    task.seq = wire.seq;
    task.graph_id = wire.graph_id;
    size_t received = 0;
    if (!receive(phone, &task, frame_data.data(), frame_data.size(),
                 &received, 30000) || received != bytes) {
        std::fprintf(stderr,
            "[CYCLE_REAL_WIRE] side=pc stage=%s status=FAIL "
            "reason=RECEIVE_MATCH_OR_SIZE expected=%zu got=%zu "
            "layer=%d chunk=%d seq=%llu\n",
            kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
            bytes, received, wire.layer, wire.chunk,
            (unsigned long long) wire.seq);
        return false;
    }
    inference_frame_header header =
        inference_make_header(INFERENCE_FRAME_DATA);
    header.stage = task.stage;
    header.tensor_role = task.tensor_role;
    header.layer = task.layer;
    header.chunk = task.chunk;
    header.seq = task.seq;
    header.graph_id = task.graph_id;
    header.payload_bytes = received;
    if (task.stage != stage ||
        task.tensor_role != INFERENCE_CYCLE_PHONE_TO_PC ||
        task.layer != wire.layer || task.chunk != wire.chunk ||
        task.seq != wire.seq || task.graph_id != wire.graph_id ||
        !inference_cycle_decode_frame(
            header, frame_data.data(), frame_data.size(), output) ||
        output.direction != INFERENCE_CYCLE_PHONE_TO_PC ||
        output.kind != kind) {
        std::fprintf(stderr,
            "[CYCLE_REAL_WIRE] side=pc stage=%s status=FAIL "
            "reason=HEADER_OR_DECODE\n",
            kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL");
        return false;
    }
    std::printf(
        "[CYCLE_REAL_WIRE] side=pc stage=%s direction=P2C "
        "layer=%d chunk=%d seq=%llu bytes=%zu "
        "source=PHONE_GPU_RPC_TENSORS mode=REAL_RPC_DATA_PLANE status=PASS\n",
        kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
        wire.layer, wire.chunk, (unsigned long long)wire.seq, received);
    return true;
}

// PC -> Phone READY and PARTIAL are two ordered, unidirectional frames.
// This function uses only a proc resolved from the attached backend.
bool llama_hybrid_cycle_send_c2p_frame(
        ggml_backend_t phone, const llama_hybrid_cycle_live_wire & wire,
        inference_cycle_payload_kind kind,
        const llama_hybrid_cycle_attention_ready & ready,
        const std::vector<float> & partial,
        uint32_t tokens, uint32_t hidden, uint32_t topk,
        llama_hybrid_cycle_wire_submit_timing * timing) {
    const auto encode_begin = std::chrono::steady_clock::now();
    size_t bytes = 0;
    if (!inference_cycle_payload_size(
            kind, tokens, hidden,
            kind == INFERENCE_CYCLE_READY ? topk : 0, bytes)) {
        return false;
    }
    std::vector<uint8_t> frame(bytes);
    size_t written = 0;
    bool ok = false;
    if (kind == INFERENCE_CYCLE_READY) {
        const inference_cycle_ready_view value {
            tokens, hidden, topk,
            ready.residual.data(), ready.norm.data(),
            ready.ids.data(), ready.routing.data()
        };
        ok = ready.residual.size() == size_t(tokens) * hidden &&
             ready.norm.size() == size_t(tokens) * hidden &&
             ready.ids.size() == size_t(tokens) * topk &&
             ready.routing.size() == size_t(tokens) * topk &&
             inference_cycle_write_ready(
                 frame.data(), frame.size(), value, written);
    } else {
        const inference_cycle_partial_view value {
            tokens, hidden, partial.data()
        };
        ok = partial.size() == size_t(tokens) * hidden &&
             inference_cycle_write_partial(
                 frame.data(), frame.size(), value, written);
    }
    if (!ok || written != bytes) {
        return false;
    }
    auto * dev = ggml_backend_get_device(phone);
    auto * reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    auto submit = reg ?
        reinterpret_cast<ggml_backend_rpc_inference_transport_submit_t>(
            ggml_backend_reg_get_proc_address(
                reg, GGML_BACKEND_RPC_INFERENCE_TRANSPORT_SUBMIT_PROC)) :
        nullptr;
    ggml_backend_rpc_inference_task task {};
    task.layer = wire.layer;
    task.chunk = wire.chunk;
    task.seq = wire.seq;
    task.graph_id = wire.graph_id;
    task.stage = kind == INFERENCE_CYCLE_READY ?
        INFERENCE_STAGE_CYCLE_READY : INFERENCE_STAGE_CYCLE_PARTIAL;
    task.tensor_role = INFERENCE_CYCLE_PC_TO_PHONE;
    const auto enqueue_begin = std::chrono::steady_clock::now();
    const bool submitted = submit != nullptr &&
        submit(phone, &task, frame.data(), frame.size());
    if (timing) {
        timing->payload_bytes = bytes;
        timing->encode_ms = std::chrono::duration<double, std::milli>(
            enqueue_begin - encode_begin).count();
        timing->enqueue_call_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - enqueue_begin).count();
    }
    std::printf(
        "[CYCLE_C2P_WIRE] side=pc action=SEND kind=%s layer=%d chunk=%d "
        "seq=%llu bytes=%zu direction=PC_TO_PHONE status=%s\n",
        kind == INFERENCE_CYCLE_READY ? "READY" : "PARTIAL",
        wire.layer, wire.chunk, (unsigned long long)wire.seq,
        bytes, submitted ? "QUEUED" : "FAIL");
    return submitted;
}
