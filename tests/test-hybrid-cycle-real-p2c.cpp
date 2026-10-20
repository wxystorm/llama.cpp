// Opt-in real-Qwen3-MoE P2C numerical comparison, not a production Cycle
// executor. The regular model forward pass is the independent reference:
// it supplies REAL Attention/Norm/Router READY values and full-layer output.
// PC CPU + Phone GPU then run the two borrowed physical FFN weight shards
// independently. Only PC sums the partials and adds the captured residual.
//
// READY/PARTIAL use the existing RPC backend tensor transport in this test;
// the dedicated Cycle inference data plane / FIFO send tasks are NOT active.
//
// Usage (choose a TENSOR_P2C layer in the fixed cycle):
//   test-hybrid-cycle-real-p2c <Qwen3-MoE.gguf> <phone:rpc-port> <layer> [prompt]
// Set LLAMA_HYBRID_CYCLE_PC_LAYERS and LLAMA_HYBRID_CYCLE_PHONE_LAYERS
// before launching (e.g. 5 and 1, then the first P2C layer is 7).
#include "llama.h"
#include "llama-model.h"
#include "llama-hybrid-cycle.h"
#include "llama-hybrid-cycle-ffn-executor.h"
#include "inference-cycle-transport.h"
#include "inference-cycle-reduce.h"
#include "ggml-rpc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct cycle_real_capture_value {
    std::vector<uint8_t> bytes;
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t ne[GGML_MAX_DIMS] = {};
    int count = 0;
};

struct cycle_real_capture {
    std::map<std::string, cycle_real_capture_value> values;
    bool error = false;
};

// This is the ordinary llama_context scheduler callback. On ask=false,
// the scheduler has already synchronized the node's producer backend.
static bool cycle_real_capture_cb(ggml_tensor * tensor, bool ask, void * userdata) {
    auto & state = *static_cast<cycle_real_capture *>(userdata);
    if (!tensor) {
        state.error = true;
        return false;
    }
    auto it = state.values.find(tensor->name);
    if (it == state.values.end()) {
        return ask ? false : true;
    }
    if (ask) {
        return true;
    }
    if (!tensor->buffer || !ggml_is_contiguous(tensor)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=NO_BUFFER_OR_NONCONTIG\n",
            tensor->name);
        state.error = true;
        return false;
    }
    auto & output = it->second;
    if (output.count != 0) {
        // A single prompt ubatch should deliver exactly one complete tensor.
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=DUPLICATE_CAPTURE\n",
            tensor->name);
        state.error = true;
        return false;
    }
    output.type = tensor->type;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        output.ne[i] = tensor->ne[i];
    }
    output.bytes.resize(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, output.bytes.data(), 0, output.bytes.size());
    ++output.count;
    std::printf("[CYCLE_REAL_CAPTURE] tensor=%s type=%s bytes=%zu "
                "ne=[%lld,%lld,%lld] status=OK\n",
                tensor->name, ggml_type_name(tensor->type), output.bytes.size(),
                (long long)tensor->ne[0], (long long)tensor->ne[1],
                (long long)tensor->ne[2]);
    return true;
}

static bool cycle_real_set_compare_layer(int layer) {
    const std::string value = std::to_string(layer);
#ifdef _WIN32
    return _putenv_s("LLAMA_HYBRID_CYCLE_REAL_COMPARE_LAYER", value.c_str()) == 0;
#else
    return setenv("LLAMA_HYBRID_CYCLE_REAL_COMPARE_LAYER", value.c_str(), 1) == 0;
#endif
}

// The chosen P2C layer must keep Attention, both norms and MoE Router
// exclusively on the sender. A numerically matching FFN is not sufficient
// if the reference graph silently computes its Attention on the PC.
static bool cycle_real_check_phone_weight(
        const char * role,
        const ggml_tensor * tensor,
        ggml_backend_dev_t phone_device) {
    if (!tensor || !phone_device) {
        std::fprintf(stderr,
            "[CYCLE_REAL_OWNER] role=%s status=FAIL reason=MISSING_WEIGHT\n", role);
        return false;
    }
    const auto * pc = ggml_backend_meta_get_static_weight_shard(tensor, 0);
    const auto * phone = ggml_backend_meta_get_static_weight_shard(tensor, 1);
    const bool valid =
        pc && phone &&
        pc->buffer && phone->buffer &&
        ggml_nelements(pc) == 0 &&
        ggml_nelements(phone) == ggml_nelements(tensor) &&
        ggml_backend_buft_get_device(
            ggml_backend_buffer_get_type(phone->buffer)) == phone_device;
    std::printf(
        "[CYCLE_REAL_OWNER] role=%s source=PHONE_GPU "
        "pc_elements=%lld phone_elements=%lld expected=%lld status=%s\n",
        role,
        (long long)(pc ? ggml_nelements(pc) : -1),
        (long long)(phone ? ggml_nelements(phone) : -1),
        (long long)ggml_nelements(tensor),
        valid ? "PASS" : "FAIL");
    return valid;
}

static bool cycle_real_check_capture(
        const cycle_real_capture & state,
        const std::string & name,
        ggml_type type,
        int64_t d0, int64_t d1, int64_t d2) {
    const auto it = state.values.find(name);
    if (it == state.values.end() || it->second.count != 1) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL reason=NOT_CAPTURED\n",
            name.c_str());
        return false;
    }
    const auto & entry = it->second;
    const size_t expected = size_t(d0) * size_t(d1) * size_t(d2) *
        (type == GGML_TYPE_F32 ? sizeof(float) : sizeof(int32_t));
    const bool ok =
        entry.type == type &&
        entry.ne[0] == d0 && entry.ne[1] == d1 &&
        entry.ne[2] == d2 && entry.ne[3] == 1 &&
        entry.bytes.size() == expected;
    if (!ok) {
        std::fprintf(stderr,
            "[CYCLE_REAL_CAPTURE] tensor=%s status=FAIL "
            "reason=TYPE_OR_SHAPE actual=[%lld,%lld,%lld,%lld] bytes=%zu\n",
            name.c_str(), (long long)entry.ne[0], (long long)entry.ne[1],
            (long long)entry.ne[2], (long long)entry.ne[3],
            entry.bytes.size());
    }
    return ok;
}

template <typename T>
static std::vector<T> cycle_real_read(
        const cycle_real_capture & state, const std::string & name) {
    const auto & bytes = state.values.at(name).bytes;
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

static bool cycle_real_execute_ffn(
        ggml_backend_t backend,
        const llama_hybrid_cycle_ffn_shard_weights & shard,
        const std::vector<float> & norm,
        const std::vector<int32_t> & ids,
        const std::vector<float> & routing,
        int64_t hidden, int64_t tokens,
        int64_t topk, int64_t experts,
        std::vector<float> & partial) {
    ggml_init_params params {};
    params.mem_size = 2 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    auto in = llama_hybrid_cycle_make_ffn_graph_inputs(
        ctx, hidden, tokens, topk);
    auto * x = ggml_reshape_3d(ctx, in.norm, hidden, 1, tokens);
    auto * gate = ggml_mul_mat_id(ctx, shard.gate, x, in.topk_ids);
    auto * up = ggml_mul_mat_id(ctx, shard.up, x, in.topk_ids);
    auto * activation = ggml_swiglu_split(ctx, gate, up);
    auto * down = ggml_mul_mat_id(ctx, shard.down, activation, in.topk_ids);
    auto * weighted = ggml_mul(ctx, down, in.topk_weights);

    ggml_tensor * sum = nullptr;
    for (int64_t k = 0; k < topk; ++k) {
        auto * view = ggml_view_2d(ctx, weighted, hidden, tokens,
                                  weighted->nb[2], size_t(k) * weighted->nb[1]);
        sum = sum ? ggml_add(ctx, sum, view) : view;
    }
    in.ffn_output = ggml_cont(ctx, sum);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 512, false);
    ggml_build_forward_expand(graph, in.ffn_output);

    bool ok = false;
    {
        const char * backend_name = ggml_backend_name(backend);
        std::printf("[CYCLE_REAL_FFN_SHARD] backend=%s channels=%lld nodes=%d "
                    "stage=PREPARE_BEGIN\n",
                    backend_name, (long long)shard.up->ne[1], ggml_graph_n_nodes(graph));
        llama_hybrid_cycle_ffn_executor executor;
        if (!executor.prepare(backend, ctx, graph, in, shard,
                              hidden, tokens, topk, experts)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_FFN_SHARD] backend=%s stage=PREPARE_FAIL "
                "weight_buft=%s compute_buft=%s\n",
                backend_name,
                shard.up->buffer != nullptr ?
                    ggml_backend_buft_name(ggml_backend_buffer_get_type(shard.up->buffer)) :
                    "(unallocated)",
                ggml_backend_buft_name(ggml_backend_get_default_buffer_type(backend)));
        } else {
            partial.resize(size_t(hidden * tokens));
            ok = executor.execute(
                norm.data(), norm.size(),
                ids.data(), ids.size(),
                routing.data(), routing.size(),
                partial.data(), partial.size());
            std::printf("[CYCLE_REAL_FFN_SHARD] backend=%s stage=COMPUTE_END "
                        "elements=%zu status=%s\n",
                        backend_name, partial.size(), ok ? "OK" : "FAIL");
        }
    }
    ggml_free(ctx);
    return ok;
}

struct cycle_real_diff {
    double max_abs = 0;
    double max_ref = 0;
    double rms = 0;
    double cosine = 0;
    double tolerance = 0;
    bool valid = false;
    bool pass = false;
};

static cycle_real_diff cycle_real_compare(
        const std::vector<float> & actual,
        const std::vector<float> & reference) {
    cycle_real_diff result {};
    if (actual.empty() || actual.size() != reference.size()) {
        return result;
    }
    double sum_sq = 0;
    double sum_x_sq = 0;
    double sum_y_sq = 0;
    double dot = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double x = actual[i], y = reference[i];
        if (!std::isfinite(x) || !std::isfinite(y)) {
            return result;
        }
        const double err = std::fabs(x - y);
        result.max_abs = std::max(result.max_abs, err);
        result.max_ref = std::max(result.max_ref, std::fabs(y));
        sum_sq += err * err;
        sum_x_sq += x * x;
        sum_y_sq += y * y;
        dot += x * y;
    }
    result.rms = std::sqrt(sum_sq / actual.size());
    result.cosine = (sum_x_sq > 0 && sum_y_sq > 0) ?
        dot / std::sqrt(sum_x_sq * sum_y_sq) : 0;
    result.tolerance = 0.001 + 0.005 * result.max_ref;
    result.valid = true;
    result.pass = result.max_abs <= result.tolerance;
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 4 || argc > 5 || !llama_hybrid_cycle_requested()) {
        std::fprintf(stderr,
            "Usage: test-hybrid-cycle-real-p2c <model.gguf> <phone-ip:port> "
            "<P2C-layer> [prompt]\n"
            "Set LLAMA_HYBRID_CYCLE_PC_LAYERS and "
            "LLAMA_HYBRID_CYCLE_PHONE_LAYERS first.\n");
        return 2;
    }
    char * end = nullptr;
    const long layer_long = std::strtol(argv[3], &end, 10);
    if (!end || *end != '\0' || layer_long < 0 || layer_long >= 512) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=BAD_LAYER\n");
        return 2;
    }
    const int layer = int(layer_long);
    if (!cycle_real_set_compare_layer(layer)) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=SET_ENV\n");
        return 2;
    }

    llama_backend_init();
    const auto cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    const auto reg = ggml_backend_rpc_add_server(argv[2]);
    if (!cpu_dev || !reg || ggml_backend_reg_dev_count(reg) != 1) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=RPC_DEVICE_COUNT\n");
        return 1;
    }
    const auto phone_dev = ggml_backend_reg_dev_get(reg, 0);
    if (!phone_dev || ggml_backend_dev_type(phone_dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=NOT_PHONE_GPU\n");
        return 1;
    }

    ggml_backend_dev_t devices[3] = {cpu_dev, phone_dev, nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.devices = devices;
    mp.split_mode = LLAMA_SPLIT_MODE_TENSOR;
    mp.hybrid_auto = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], mp), &llama_model_free);
    if (!model || model->arch != LLM_ARCH_QWEN3MOE ||
        layer >= model->hparams.n_layer() - 1 ||
        model->hybrid_layer_mode(layer) != llama_hybrid_layer_mode::TENSOR_P2C) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=MODEL_OR_NOT_P2C_LAYER layer=%d\n",
            layer);
        return 1;
    }

    const auto & lw = model->layers[size_t(layer)];
    if (!cycle_real_check_phone_weight("ATTN_NORM", lw.attn_norm, phone_dev) ||
        !cycle_real_check_phone_weight("ATTN_OUT", lw.wo, phone_dev) ||
        !cycle_real_check_phone_weight("FFN_NORM", lw.ffn_norm, phone_dev) ||
        !cycle_real_check_phone_weight("MOE_ROUTER", lw.ffn_gate_inp, phone_dev)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=ATTENTION_ROUTER_NOT_PHONE_OWNED\n");
        return 1;
    }
    const int64_t hidden = model->hparams.n_embd;
    const int64_t topk = model->hparams.n_expert_used;
    const int64_t experts = model->hparams.n_expert;
    // The first real comparison intentionally covers ordinary Qwen3-MoE
    // SwiGLU without extra expert scaling/clamping. Reject unsupported
    // architectures/layouts rather than silently comparing unlike formulas.
    if (model->hparams.swiglu_clamp_exp[size_t(layer)] > 1.0e-6f ||
        lw.ffn_gate_exps_s || lw.ffn_up_exps_s || lw.ffn_down_exps_s ||
        !llama_hybrid_cycle_ffn_partition_valid(
            lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps,
            hidden, experts)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=WEIGHT_SHARD_OR_SCALES\n");
        return 1;
    }
    const auto pc_weights = llama_hybrid_cycle_get_physical_ffn_weights(
        lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps, 0);
    const auto phone_weights = llama_hybrid_cycle_get_physical_ffn_weights(
        lw.ffn_up_exps, lw.ffn_gate_exps, lw.ffn_down_exps, 1);
    std::printf("[CYCLE_REAL_MODEL] layer=%d hidden=%lld topk=%lld experts=%lld "
                "pc_channels=%lld phone_channels=%lld type=%s status=OK\n",
                layer, (long long)hidden, (long long)topk, (long long)experts,
                (long long)pc_weights.up->ne[1],
                (long long)phone_weights.up->ne[1],
                ggml_type_name(lw.ffn_up_exps->type));

    const std::string suffix = "-" + std::to_string(layer);
    const std::string residual_name = "ffn_inp" + suffix;
    const std::string norm_name = "ffn_norm" + suffix;
    const std::string ids_name = "cycle_real_topk_ids" + suffix;
    const std::string routing_name = "cycle_real_topk_weights" + suffix;
    const std::string ffn_name = "ffn_moe_out" + suffix;
    const std::string output_name = "l_out" + suffix;
    cycle_real_capture capture;
    for (const auto & name : {residual_name, norm_name, ids_name,
                             routing_name, ffn_name, output_name}) {
        capture.values.emplace(name, cycle_real_capture_value {});
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    cp.cb_eval = cycle_real_capture_cb;
    cp.cb_eval_user_data = &capture;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), cp), &llama_free);
    if (!context) {
        std::fprintf(stderr, "[CYCLE_REAL_P2C] status=FAIL reason=CONTEXT_INIT\n");
        return 1;
    }

    const std::string prompt = argc == 5 ? argv[4] :
        "The quick brown fox jumps over the lazy dog. Please explain a transformer.";
    std::vector<llama_token> tokens(4096);
    int32_t ntokens = llama_tokenize(
        llama_model_get_vocab(model.get()), prompt.c_str(), int(prompt.size()),
        tokens.data(), int32_t(tokens.size()), true, true);
    if (ntokens < 0) {
        tokens.resize(size_t(-ntokens));
        ntokens = llama_tokenize(
            llama_model_get_vocab(model.get()), prompt.c_str(), int(prompt.size()),
            tokens.data(), int32_t(tokens.size()), true, true);
    }
    if (ntokens < 1 || ntokens > int32_t(cp.n_batch) || ntokens > int32_t(cp.n_ctx)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=TOKEN_COUNT tokens=%d max=%u\n",
            int(ntokens), cp.n_batch);
        return 1;
    }

    llama_batch batch = llama_batch_init(ntokens, 0, 1);
    batch.n_tokens = ntokens;
    for (int i = 0; i < ntokens; ++i) {
        batch.token[i] = tokens[size_t(i)];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == ntokens - 1) ? 1 : 0;
    }
    std::printf("[CYCLE_REAL_FORWARD] layer=%d prompt_tokens=%d stage=BEGIN\n",
                layer, int(ntokens));
    const auto forward_begin = std::chrono::steady_clock::now();
    const int decode_status = llama_decode(context.get(), batch);
    llama_synchronize(context.get());
    const double forward_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - forward_begin).count();
    llama_batch_free(batch);
    if (decode_status != 0 || capture.error) {
        std::fprintf(stderr,
            "[CYCLE_REAL_FORWARD] status=FAIL reason=DECODE_OR_CAPTURE code=%d\n",
            decode_status);
        return 1;
    }
    if (!cycle_real_check_capture(capture, residual_name, GGML_TYPE_F32, hidden, ntokens, 1) ||
        !cycle_real_check_capture(capture, norm_name, GGML_TYPE_F32, hidden, ntokens, 1) ||
        !cycle_real_check_capture(capture, ids_name, GGML_TYPE_I32, topk, ntokens, 1) ||
        !cycle_real_check_capture(capture, routing_name, GGML_TYPE_F32, 1, topk, ntokens) ||
        !cycle_real_check_capture(capture, ffn_name, GGML_TYPE_F32, hidden, ntokens, 1) ||
        !cycle_real_check_capture(capture, output_name, GGML_TYPE_F32, hidden, ntokens, 1)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=REAL_READY_OR_REFERENCE_MISSING\n");
        return 1;
    }

    const auto residual = cycle_real_read<float>(capture, residual_name);
    const auto norm = cycle_real_read<float>(capture, norm_name);
    const auto ids = cycle_real_read<int32_t>(capture, ids_name);
    const auto routing = cycle_real_read<float>(capture, routing_name);
    const auto baseline_ffn = cycle_real_read<float>(capture, ffn_name);
    const auto baseline_output = cycle_real_read<float>(capture, output_name);

    // Sanity-check the *reference* before comparing any new execution.
    // For ordinary Qwen3-MoE with no adapters this layer must implement
    // l_out = AttentionResidual + CompleteFFN. Otherwise a capture label
    // or a prior graph transformation is not comparing like with like.
    if (baseline_ffn.size() != residual.size() ||
        baseline_output.size() != residual.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=BASELINE_SIZE\n");
        return 1;
    }
    std::vector<float> baseline_reconstructed(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        baseline_reconstructed[i] = residual[i] + baseline_ffn[i];
    }
    const cycle_real_diff baseline_check =
        cycle_real_compare(baseline_reconstructed, baseline_output);
    std::printf("[CYCLE_REAL_COMPARE] target=BASELINE_INTERNAL layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, baseline_check.max_abs, baseline_check.rms,
                baseline_check.cosine, baseline_check.tolerance,
                baseline_check.valid && baseline_check.pass ? "PASS" : "FAIL");
    if (!baseline_check.valid || !baseline_check.pass) {
        return 1;
    }
    for (int32_t id : ids) {
        if (id < 0 || id >= experts) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=INVALID_TOPK id=%d\n", id);
            return 1;
        }
    }
    for (float v : norm) {
        if (!std::isfinite(v)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=NONFINITE_NORM\n");
            return 1;
        }
    }
    for (float w : routing) {
        if (!std::isfinite(w)) {
            std::fprintf(stderr,
                "[CYCLE_REAL_READY] status=FAIL reason=NONFINITE_ROUTING\n");
            return 1;
        }
    }
    std::printf("[CYCLE_REAL_READY] layer=%d tokens=%d fields=RESIDUAL,NORM,TOPK,WEIGHTS "
                "source=REAL_MODEL status=OK\n", layer, int(ntokens));

    // Exercise the V1 P2C READY wire representation with the *real* tensor
    // values, including frame direction and seq. This is an in-process
    // encode/decode check, NOT a claim that the new network sender is wired.
    const inference_cycle_ready_view ready_source {
        uint32_t(ntokens), uint32_t(hidden), uint32_t(topk),
        residual.data(), norm.data(), ids.data(), routing.data(),
    };
    size_t ready_bytes = 0;
    if (!inference_cycle_payload_size(
            INFERENCE_CYCLE_READY, uint32_t(ntokens), uint32_t(hidden),
            uint32_t(topk), ready_bytes)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_SIZE\n");
        return 1;
    }
    std::vector<uint8_t> ready_wire(ready_bytes);
    size_t ready_written = 0;
    if (!inference_cycle_write_ready(
            ready_wire.data(), ready_wire.size(), ready_source, ready_written) ||
        ready_written != ready_wire.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_ENCODE\n");
        return 1;
    }
    inference_frame_header ready_header = inference_make_header(INFERENCE_FRAME_DATA);
    ready_header.stage = INFERENCE_STAGE_CYCLE_READY;
    ready_header.tensor_role = INFERENCE_CYCLE_PHONE_TO_PC;
    ready_header.layer = layer;
    ready_header.chunk = 0;
    ready_header.seq = 1;
    ready_header.graph_id = 1;
    ready_header.payload_bytes = ready_wire.size();
    inference_cycle_message ready_msg;
    if (!inference_cycle_decode_frame(
            ready_header, ready_wire.data(), ready_wire.size(), ready_msg) ||
        ready_msg.kind != INFERENCE_CYCLE_READY ||
        ready_msg.direction != INFERENCE_CYCLE_PHONE_TO_PC ||
        ready_msg.layer != layer || ready_msg.seq != 1 ||
        ready_msg.ready.topk_ids != ids ||
        ready_msg.ready.topk_weights != routing ||
        ready_msg.ready.ffn_norm != norm ||
        ready_msg.ready.residual != residual) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=READY_WIRE_DECODE\n");
        return 1;
    }
    std::printf("[CYCLE_REAL_WIRE] stage=READY direction=P2C "
                "bytes=%zu mode=IN_PROCESS_CODEC status=PASS\n", ready_bytes);

    ggml_backend_t cpu = ggml_backend_dev_init(cpu_dev, nullptr);
    ggml_backend_t phone = ggml_backend_dev_init(phone_dev, nullptr);
    if (!cpu || !phone || !ggml_backend_is_rpc(phone)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PHYSICAL_BACKENDS\n");
        return 1;
    }

    const auto parts_begin = std::chrono::steady_clock::now();
    auto phone_task = std::async(std::launch::async, [&] {
        std::vector<float> part;
        if (!cycle_real_execute_ffn(
                phone, phone_weights, norm, ids, routing,
                hidden, ntokens, topk, experts, part)) {
            part.clear();
        }
        return part;
    });
    std::vector<float> pc_part;
    // The receiving PC consumes only the decoded READY, not its source arrays.
    // The Phone FFN above continues to use the sender's original inputs.
    const bool pc_ok = cycle_real_execute_ffn(
        cpu, pc_weights,
        ready_msg.ready.ffn_norm, ready_msg.ready.topk_ids,
        ready_msg.ready.topk_weights,
        hidden, ntokens, topk, experts, pc_part);
    const auto phone_part = phone_task.get();
    const double parts_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - parts_begin).count();

    ggml_backend_free(phone);
    ggml_backend_free(cpu);

    if (!pc_ok || pc_part.size() != residual.size() ||
        phone_part.size() != residual.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PHYSICAL_FFN_COMPUTE\n");
        return 1;
    }

    // SEND #2: serialize the actual Phone GPU FFN contribution and decode it
    // as the PC receiver would. Both payload frames have the SAME direction.
    const inference_cycle_partial_view partial_source {
        uint32_t(ntokens), uint32_t(hidden), phone_part.data(),
    };
    size_t partial_bytes = 0;
    if (!inference_cycle_payload_size(
            INFERENCE_CYCLE_PARTIAL, uint32_t(ntokens), uint32_t(hidden),
            0, partial_bytes)) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_SIZE\n");
        return 1;
    }
    std::vector<uint8_t> partial_wire(partial_bytes);
    size_t partial_written = 0;
    if (!inference_cycle_write_partial(
            partial_wire.data(), partial_wire.size(),
            partial_source, partial_written) ||
        partial_written != partial_wire.size()) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_ENCODE\n");
        return 1;
    }
    inference_frame_header partial_header = inference_make_header(INFERENCE_FRAME_DATA);
    partial_header.stage = INFERENCE_STAGE_CYCLE_PARTIAL;
    partial_header.tensor_role = INFERENCE_CYCLE_PHONE_TO_PC;
    partial_header.layer = layer;
    partial_header.chunk = 0;
    partial_header.seq = ready_header.seq;
    partial_header.graph_id = ready_header.graph_id;
    partial_header.payload_bytes = partial_wire.size();
    inference_cycle_message partial_msg;
    if (!inference_cycle_decode_frame(
            partial_header, partial_wire.data(), partial_wire.size(), partial_msg) ||
        partial_msg.kind != INFERENCE_CYCLE_PARTIAL ||
        partial_msg.direction != ready_msg.direction ||
        partial_msg.layer != ready_msg.layer ||
        partial_msg.seq != ready_msg.seq ||
        partial_msg.graph_id != ready_msg.graph_id ||
        partial_msg.partial.partial != phone_part) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PARTIAL_WIRE_DECODE\n");
        return 1;
    }
    std::printf("[CYCLE_REAL_WIRE] stage=PARTIAL direction=P2C "
                "bytes=%zu mode=IN_PROCESS_CODEC status=PASS\n", partial_bytes);

    std::vector<float> combined_ffn(residual.size());
    std::vector<float> combined_output(residual.size());
    for (size_t i = 0; i < residual.size(); ++i) {
        combined_ffn[i] = pc_part[i] + partial_msg.partial.partial[i];
    }
    if (!inference_cycle_reduce_p2c_pc_cpu(
            ready_msg.ready, pc_part.data(), pc_part.size(),
            partial_msg.partial, combined_output.data(), combined_output.size())) {
        std::fprintf(stderr,
            "[CYCLE_REAL_P2C] status=FAIL reason=PC_REDUCE\n");
        return 1;
    }
    const cycle_real_diff ffn_diff = cycle_real_compare(combined_ffn, baseline_ffn);
    const cycle_real_diff out_diff = cycle_real_compare(combined_output, baseline_output);
    std::printf("[CYCLE_REAL_COMPARE] target=FFN_PARTIAL_SUM layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, ffn_diff.max_abs, ffn_diff.rms, ffn_diff.cosine,
                ffn_diff.tolerance, ffn_diff.pass ? "PASS" : "FAIL");
    std::printf("[CYCLE_REAL_COMPARE] target=LAYER_OUTPUT layer=%d "
                "max_abs=%.8g rms=%.8g cosine=%.8g tolerance=%.8g status=%s\n",
                layer, out_diff.max_abs, out_diff.rms, out_diff.cosine,
                out_diff.tolerance, out_diff.pass ? "PASS" : "FAIL");
    const bool pass = ffn_diff.valid && out_diff.valid &&
                      ffn_diff.pass && out_diff.pass;
    std::printf("[CYCLE_REAL_P2C_SUM] layer=%d tokens=%d "
                "forward_ms=%.3f ffn_parallel_ms=%.3f "
                "real_attention=FULL_MODEL_REFERENCE real_ffn_weights=1 "
                "ffn_shards=PC_CPU+PHONE_GPU reduce_owner=PC_CPU "
                "reference=FULL_QWEN3_LAYER data_plane=RPC_BACKEND_ONLY "
                "cycle_wire=IN_PROCESS_CODEC_ONLY status=%s\n",
                layer, int(ntokens), forward_ms, parts_ms,
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
