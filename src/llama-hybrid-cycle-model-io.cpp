#include "llama-hybrid-cycle-model-io.h"

#include "llama-hybrid-cycle-physical.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

struct cycle_io_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;

    cycle_io_graph() {
        ggml_init_params p {};
        p.mem_size = 16 * 1024 * 1024;
        p.mem_buffer = nullptr;
        p.no_alloc = true;
        ctx = ggml_init(p);
        if (ctx) graph = ggml_new_graph_custom(ctx, 256, false);
    }

    ~cycle_io_graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
    bool alloc(ggml_backend_t cpu) {
        if (!ctx || !graph || !cpu) return false;
        auto * buft = ggml_backend_get_default_buffer_type(cpu);
        if (!buft) return false;
        buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        return buffer != nullptr;
    }
    bool compute(ggml_backend_t cpu) {
        return graph &&
            ggml_backend_graph_compute(cpu, graph) == GGML_STATUS_SUCCESS;
    }
    cycle_io_graph(const cycle_io_graph &) = delete;
    cycle_io_graph & operator=(const cycle_io_graph &) = delete;
};

bool cpu_weight(ggml_backend_t cpu, const ggml_tensor * src,
                ggml_tensor *& out) {
    out = nullptr;
    if (!cpu || !src || !ggml_backend_get_device(cpu) ||
        ggml_backend_dev_type(ggml_backend_get_device(cpu)) !=
            GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }
    out = llama_hybrid_cycle_resolve_physical_weight(src, cpu, 0);
    if (!out || !out->buffer) return false;
    auto * dev = ggml_backend_get_device(cpu);
    auto * wt_buft = ggml_backend_buffer_get_type(out->buffer);
    return wt_buft &&
           ggml_backend_dev_supports_buft(dev, wt_buft);
}

bool finite(const std::vector<float> & data) {
    for (float x : data) if (!std::isfinite(x)) return false;
    return true;
}

} // namespace

bool llama_hybrid_cycle_embed_tokens(
        const llama_model & model, ggml_backend_t cpu,
        const std::vector<llama_token> & tokens,
        std::vector<float> & embeddings, std::string & reason) {
    embeddings.clear();
    reason.clear();
    const int64_t hidden = model.hparams.n_embd;
    const int64_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(&model));
    if (model.arch != LLM_ARCH_QWEN3MOE || hidden <= 0 ||
        tokens.empty() || tokens.size() > 64 ||
        size_t(hidden) > std::numeric_limits<size_t>::max()/tokens.size()) {
        reason = "EMBED_UNSUPPORTED_SHAPE";
        return false;
    }
    std::vector<int32_t> ids(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] < 0 || int64_t(tokens[i]) >= vocab) {
            reason = "EMBED_INVALID_TOKEN_ID";
            return false;
        }
        ids[i] = int32_t(tokens[i]);
    }
    ggml_tensor * weight = nullptr;
    if (!cpu_weight(cpu, model.tok_embd, weight) ||
        weight->ne[0] != hidden || weight->ne[1] != vocab) {
        reason = "EMBED_CPU_WEIGHT_UNAVAILABLE";
        return false;
    }
    cycle_io_graph g;
    if (!g.graph) {
        reason = "EMBED_GRAPH_METADATA_ALLOCATION";
        return false;
    }
    ggml_tensor * input_ids = ggml_new_tensor_1d(
        g.ctx, GGML_TYPE_I32, int64_t(tokens.size()));
    ggml_set_input(input_ids);
    ggml_tensor * e = ggml_get_rows(g.ctx, weight, input_ids);
    if (!e || e->type != GGML_TYPE_F32 ||
        e->ne[0] != hidden || e->ne[1] != int64_t(tokens.size())) {
        reason = "EMBED_GRAPH_OUTPUT_SHAPE";
        return false;
    }
    ggml_build_forward_expand(g.graph, e);
    if (!g.alloc(cpu)) {
        reason = "EMBED_CPU_ALLOC";
        return false;
    }
    ggml_backend_tensor_set(input_ids, ids.data(), 0,
                            ids.size()*sizeof(int32_t));
    if (!g.compute(cpu)) {
        ggml_backend_synchronize(cpu);
        reason = "EMBED_CPU_COMPUTE";
        return false;
    }
    embeddings.resize(size_t(hidden)*tokens.size());
    ggml_backend_tensor_get(e, embeddings.data(), 0,
                            embeddings.size()*sizeof(float));
    if (!finite(embeddings)) {
        embeddings.clear();
        reason = "EMBED_NONFINITE";
        return false;
    }
    return true;
}

bool llama_hybrid_cycle_project_last_token(
        const llama_model & model, ggml_backend_t cpu,
        const std::vector<float> & hidden, int n_tokens,
        std::vector<float> & logits, std::string & reason) {
    logits.clear();
    reason.clear();
    const int64_t n_embd = model.hparams.n_embd;
    const int64_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(&model));
    if (model.arch != LLM_ARCH_QWEN3MOE ||
        n_tokens < 1 || n_embd <= 0 || n_vocab <= 0 ||
        size_t(n_tokens)*size_t(n_embd) != hidden.size() ||
        !finite(hidden)) {
        reason = "LOGITS_INVALID_HIDDEN_SHAPE";
        return false;
    }
    ggml_tensor * norm_weight = nullptr;
    ggml_tensor * output_weight = nullptr;
    if (!cpu_weight(cpu, model.output_norm, norm_weight) ||
        !cpu_weight(cpu, model.output, output_weight) ||
        norm_weight->ne[0] != n_embd ||
        output_weight->ne[0] != n_embd ||
        output_weight->ne[1] != n_vocab) {
        reason = "LOGITS_CPU_HEAD_WEIGHTS_UNAVAILABLE";
        return false;
    }
    cycle_io_graph g;
    if (!g.graph) {
        reason = "LOGITS_GRAPH_METADATA_ALLOCATION";
        return false;
    }
    ggml_tensor * last = ggml_new_tensor_2d(
        g.ctx, GGML_TYPE_F32, n_embd, 1);
    ggml_set_input(last);
    ggml_tensor * rms = ggml_rms_norm(
        g.ctx, last, model.hparams.f_norm_rms_eps);
    ggml_tensor * normed = ggml_mul(g.ctx, rms, norm_weight);
    ggml_tensor * output = ggml_mul_mat(g.ctx, output_weight, normed);
    if (!output || output->type != GGML_TYPE_F32 ||
        output->ne[0] != n_vocab || output->ne[1] != 1) {
        reason = "LOGITS_OUTPUT_GRAPH_SHAPE";
        return false;
    }
    ggml_build_forward_expand(g.graph, output);
    if (!g.alloc(cpu)) {
        reason = "LOGITS_CPU_ALLOC";
        return false;
    }
    const float * last_hidden =
        hidden.data() + size_t(n_tokens - 1)*size_t(n_embd);
    ggml_backend_tensor_set(last, last_hidden, 0,
                            size_t(n_embd)*sizeof(float));
    if (!g.compute(cpu)) {
        ggml_backend_synchronize(cpu);
        reason = "LOGITS_CPU_COMPUTE";
        return false;
    }
    logits.resize(size_t(n_vocab));
    ggml_backend_tensor_get(output, logits.data(), 0,
                            size_t(n_vocab)*sizeof(float));
    if (!finite(logits)) {
        logits.clear();
        reason = "LOGITS_NONFINITE";
        return false;
    }
    return true;
}
