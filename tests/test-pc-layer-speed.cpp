#include "llama.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

using clock_type = std::chrono::steady_clock;

static double ms_between(clock_type::time_point a, clock_type::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct options {
    std::string model_path;
    int device = 0;
    int start_layer = 0;
    int layers = 4;
    int tokens = 128;
    int warmup = 2;
    int iters = 10;
    int cpu_threads = 8;
    double link_gbs = 11.0;
};

static void usage(const char * argv0) {
    std::printf(
        "usage: %s -m model.gguf [options]\n"
        "  -m, --model PATH      GGUF model path\n"
        "  --device N            CUDA device (default: 0)\n"
        "  --start-layer N       first layer (default: 0)\n"
        "  --layers N            consecutive layers in one graph (default: 4)\n"
        "  --tokens N            prefill tokens, KV=tokens (default: 128)\n"
        "  --warmup N            warmup runs per backend (default: 2)\n"
        "  --iters N             timed runs per backend (default: 10)\n"
        "  --cpu-threads N       CPU threads (default: 8)\n"
        "  --link-gbs X          reference one-way bandwidth in GB/s (default: 11)\n",
        argv0);
}

static bool parse_i32(const char * s, int & v) {
    char * end = nullptr;
    long x = std::strtol(s, &end, 10);
    if (!end || *end != '\0') return false;
    v = (int) x;
    return true;
}

static bool parse_f64(const char * s, double & v) {
    char * end = nullptr;
    double x = std::strtod(s, &end);
    if (!end || *end != '\0') return false;
    v = x;
    return true;
}

static bool parse_options(int argc, char ** argv, options & opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "-m" || a == "--model") {
            const char * s = need("--model");
            if (!s) return false;
            opt.model_path = s;
        } else if (a == "--device") {
            const char * s = need("--device");
            if (!s || !parse_i32(s, opt.device)) return false;
        } else if (a == "--start-layer") {
            const char * s = need("--start-layer");
            if (!s || !parse_i32(s, opt.start_layer)) return false;
        } else if (a == "--layers") {
            const char * s = need("--layers");
            if (!s || !parse_i32(s, opt.layers)) return false;
        } else if (a == "--tokens") {
            const char * s = need("--tokens");
            if (!s || !parse_i32(s, opt.tokens)) return false;
        } else if (a == "--warmup") {
            const char * s = need("--warmup");
            if (!s || !parse_i32(s, opt.warmup)) return false;
        } else if (a == "--iters") {
            const char * s = need("--iters");
            if (!s || !parse_i32(s, opt.iters)) return false;
        } else if (a == "--cpu-threads") {
            const char * s = need("--cpu-threads");
            if (!s || !parse_i32(s, opt.cpu_threads)) return false;
        } else if (a == "--link-gbs") {
            const char * s = need("--link-gbs");
            if (!s || !parse_f64(s, opt.link_gbs)) return false;
        } else {
            std::fprintf(stderr, "unknown option: %s\n", a.c_str());
            return false;
        }
    }

    return !opt.model_path.empty() &&
           opt.device >= 0 &&
           opt.start_layer >= 0 &&
           opt.layers > 0 &&
           opt.tokens > 0 &&
           opt.warmup >= 0 &&
           opt.iters > 0 &&
           opt.cpu_threads > 0 &&
           opt.link_gbs > 0.0;
}

struct stats {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double min = 0.0;
    double max = 0.0;
};

static stats summarize(std::vector<double> v) {
    stats s;
    if (v.empty()) return s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / (double) v.size();
    s.min = *std::min_element(v.begin(), v.end());
    s.max = *std::max_element(v.begin(), v.end());
    std::sort(v.begin(), v.end());

    auto pct = [&](double p) {
        const double x = p * (double) (v.size() - 1);
        const size_t lo = (size_t) std::floor(x);
        const size_t hi = (size_t) std::ceil(x);
        const double f = x - (double) lo;
        return v[lo] * (1.0 - f) + v[hi] * f;
    };

    s.p50 = pct(0.50);
    s.p95 = pct(0.95);
    return s;
}

struct qwen_layer_weights {
    const ggml_tensor * attn_norm = nullptr;
    const ggml_tensor * wq = nullptr;
    const ggml_tensor * wk = nullptr;
    const ggml_tensor * wv = nullptr;
    const ggml_tensor * wo = nullptr;
    const ggml_tensor * q_norm = nullptr;
    const ggml_tensor * k_norm = nullptr;

    const ggml_tensor * ffn_norm = nullptr;
    const ggml_tensor * router = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * up = nullptr;
    const ggml_tensor * down = nullptr;
};

static bool get_layer_weights(
        const llama_model * model,
        int il,
        qwen_layer_weights & w) {
    if (il < 0 || il >= (int) model->layers.size()) return false;
    const llama_layer & l = model->layers[(size_t) il];

    w.attn_norm = l.attn_norm;
    w.wq = l.wq;
    w.wk = l.wk;
    w.wv = l.wv;
    w.wo = l.wo;
    w.q_norm = l.attn_q_norm;
    w.k_norm = l.attn_k_norm;

    w.ffn_norm = l.ffn_norm;
    w.router = l.ffn_gate_inp;
    w.gate = l.ffn_gate_exps;
    w.up = l.ffn_up_exps;
    w.down = l.ffn_down_exps;

    return w.attn_norm && w.wq && w.wk && w.wv && w.wo &&
           w.q_norm && w.k_norm &&
           w.ffn_norm && w.router && w.gate && w.up && w.down;
}

static size_t tensor_bytes(const ggml_tensor * t) {
    return t ? ggml_nbytes(t) : 0;
}

static size_t layer_attn_bytes(const qwen_layer_weights & w) {
    return tensor_bytes(w.attn_norm) +
           tensor_bytes(w.wq) +
           tensor_bytes(w.wk) +
           tensor_bytes(w.wv) +
           tensor_bytes(w.wo) +
           tensor_bytes(w.q_norm) +
           tensor_bytes(w.k_norm);
}

static size_t layer_moe_bytes(const qwen_layer_weights & w) {
    return tensor_bytes(w.ffn_norm) +
           tensor_bytes(w.router) +
           tensor_bytes(w.gate) +
           tensor_bytes(w.up) +
           tensor_bytes(w.down);
}

struct copied_weight {
    const ggml_tensor * src = nullptr;
    ggml_tensor * dst = nullptr;
};

static ggml_tensor * copy_weight_meta(
        ggml_context * ctx,
        const ggml_tensor * src,
        std::vector<copied_weight> & copies) {
    ggml_tensor * dst = ggml_dup_tensor(ctx, src);
    if (!dst) return nullptr;
    ggml_set_name(dst, src->name);
    copies.push_back({src, dst});
    return dst;
}

struct owned_layer_weights {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * wq = nullptr;
    ggml_tensor * wk = nullptr;
    ggml_tensor * wv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;

    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * router = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
};

struct layer_graph {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * pos = nullptr;
    ggml_tensor * mask = nullptr;

    // Per-layer stage views over the full graph. ATTN ends exactly at
    // ffn_inp (attention output + residual), and FFN ends at layer output.
    std::vector<ggml_cgraph *> attn_graphs;
    std::vector<ggml_cgraph *> ffn_graphs;

    std::vector<copied_weight> copies;
    size_t weight_bytes = 0;
    size_t buffer_bytes = 0;
};

static bool set_cpu_threads(ggml_backend_t backend, int n_threads) {
    if (!backend || n_threads <= 0) return false;
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (!reg) return false;

    using setter_t = void (*)(ggml_backend_t, int);
    auto fn = reinterpret_cast<setter_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
    if (!fn) return false;
    fn(backend, n_threads);
    return true;
}

static bool build_full_layer_graph(
        layer_graph & out,
        const llama_model * model,
        int start_layer,
        int n_layers,
        int tokens,
        ggml_backend_t backend,
        bool duplicate_weights) {
    const size_t graph_size = (size_t) n_layers * 96 + 64;
    const size_t tensor_budget = (size_t) n_layers * 128 + 256;

    ggml_init_params p = {};
    const size_t graph_count = 1 + 2 * (size_t) n_layers;
    p.mem_size =
        tensor_budget * ggml_tensor_overhead() +
        graph_count * ggml_graph_overhead_custom(graph_size, false);
    p.no_alloc = true;

    out.ctx = ggml_init(p);
    if (!out.ctx) return false;

    std::vector<qwen_layer_weights> src_layers((size_t) n_layers);
    std::vector<owned_layer_weights> owned((size_t) n_layers);

    for (int li = 0; li < n_layers; ++li) {
        const int il = start_layer + li;
        if (!get_layer_weights(model, il, src_layers[(size_t) li])) {
            std::fprintf(stderr, "missing Qwen3-MoE tensors for layer %d\n", il);
            return false;
        }

        const auto & s = src_layers[(size_t) li];
        out.weight_bytes += layer_attn_bytes(s) + layer_moe_bytes(s);

        if (duplicate_weights) {
            auto & d = owned[(size_t) li];
            d.attn_norm = copy_weight_meta(out.ctx, s.attn_norm, out.copies);
            d.wq = copy_weight_meta(out.ctx, s.wq, out.copies);
            d.wk = copy_weight_meta(out.ctx, s.wk, out.copies);
            d.wv = copy_weight_meta(out.ctx, s.wv, out.copies);
            d.wo = copy_weight_meta(out.ctx, s.wo, out.copies);
            d.q_norm = copy_weight_meta(out.ctx, s.q_norm, out.copies);
            d.k_norm = copy_weight_meta(out.ctx, s.k_norm, out.copies);
            d.ffn_norm = copy_weight_meta(out.ctx, s.ffn_norm, out.copies);
            d.router = copy_weight_meta(out.ctx, s.router, out.copies);
            d.gate = copy_weight_meta(out.ctx, s.gate, out.copies);
            d.up = copy_weight_meta(out.ctx, s.up, out.copies);
            d.down = copy_weight_meta(out.ctx, s.down, out.copies);

            if (!d.attn_norm || !d.wq || !d.wk || !d.wv || !d.wo ||
                !d.q_norm || !d.k_norm || !d.ffn_norm || !d.router ||
                !d.gate || !d.up || !d.down) {
                return false;
            }
        }
    }

    const qwen_layer_weights & w0 = src_layers.front();
    const int64_t n_embd = w0.wq->ne[0];

    out.input = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F32, n_embd, tokens);
    out.pos = ggml_new_tensor_1d(
        out.ctx, GGML_TYPE_I32, tokens);
    out.mask = ggml_new_tensor_2d(
        out.ctx, GGML_TYPE_F16, tokens, tokens);

    ggml_set_name(out.input, "layer_bench_input");
    ggml_set_name(out.pos, "layer_bench_pos");
    ggml_set_name(out.mask, "layer_bench_mask");

    ggml_tensor * cur = out.input;

    std::vector<ggml_tensor *> attn_ends;
    std::vector<ggml_tensor *> layer_ends;
    attn_ends.reserve((size_t) n_layers);
    layer_ends.reserve((size_t) n_layers);

    const float rms_eps = model->hparams.f_norm_rms_eps;
    const int n_expert = (int) model->hparams.n_expert;
    const int n_expert_used = (int) model->hparams.n_expert_used;

    for (int li = 0; li < n_layers; ++li) {
        const int il = start_layer + li;
        const auto & s = src_layers[(size_t) li];
        const auto & d = owned[(size_t) li];

        auto pick = [&](const ggml_tensor * cpu_t, ggml_tensor * gpu_t) -> ggml_tensor * {
            return duplicate_weights ? gpu_t : const_cast<ggml_tensor *>(cpu_t);
        };

        ggml_tensor * attn_norm_w = pick(s.attn_norm, d.attn_norm);
        ggml_tensor * wq = pick(s.wq, d.wq);
        ggml_tensor * wk = pick(s.wk, d.wk);
        ggml_tensor * wv = pick(s.wv, d.wv);
        ggml_tensor * wo = pick(s.wo, d.wo);
        ggml_tensor * q_norm_w = pick(s.q_norm, d.q_norm);
        ggml_tensor * k_norm_w = pick(s.k_norm, d.k_norm);

        ggml_tensor * ffn_norm_w = pick(s.ffn_norm, d.ffn_norm);
        ggml_tensor * router_w = pick(s.router, d.router);
        ggml_tensor * gate_w = pick(s.gate, d.gate);
        ggml_tensor * up_w = pick(s.up, d.up);
        ggml_tensor * down_w = pick(s.down, d.down);

        const int n_head = (int) model->hparams.n_head_arr[(size_t) il];
        const int n_head_kv = (int) model->hparams.n_head_kv_arr[(size_t) il];
        if (n_head <= 0 || n_head_kv <= 0) return false;

        const int64_t q_dim = wq->ne[1];
        const int64_t k_dim = wk->ne[1];
        const int64_t v_dim = wv->ne[1];
        const int64_t head_dim_q = q_dim / n_head;
        const int64_t head_dim_k = k_dim / n_head_kv;
        const int64_t head_dim_v = v_dim / n_head_kv;
        if (head_dim_q <= 0 || head_dim_q != head_dim_k ||
            wo->ne[0] != head_dim_v * n_head) {
            return false;
        }

        ggml_tensor * norm = ggml_rms_norm(out.ctx, cur, rms_eps);
        norm = ggml_mul(out.ctx, norm, attn_norm_w);

        ggml_tensor * q = ggml_mul_mat(out.ctx, wq, norm);
        ggml_tensor * k = ggml_mul_mat(out.ctx, wk, norm);
        ggml_tensor * v = ggml_mul_mat(out.ctx, wv, norm);

        q = ggml_reshape_3d(out.ctx, q, head_dim_q, n_head, tokens);
        k = ggml_reshape_3d(out.ctx, k, head_dim_k, n_head_kv, tokens);
        v = ggml_reshape_3d(out.ctx, v, head_dim_v, n_head_kv, tokens);

        q = ggml_rms_norm(out.ctx, q, rms_eps);
        q = ggml_mul(out.ctx, q, q_norm_w);
        k = ggml_rms_norm(out.ctx, k, rms_eps);
        k = ggml_mul(out.ctx, k, k_norm_w);

        const int n_rot = (int) model->hparams.n_rot_full;
        const int n_ctx_orig =
            model->hparams.n_ctx_orig_yarn > 0 ?
                (int) model->hparams.n_ctx_orig_yarn :
                (int) model->hparams.n_ctx_train;
        const float freq_base = model->hparams.rope_freq_base_train;
        const float freq_scale = model->hparams.rope_freq_scale_train;

        q = ggml_rope_ext(
            out.ctx, q, out.pos, nullptr,
            n_rot, (int) LLAMA_ROPE_TYPE_NEOX,
            n_ctx_orig, freq_base, freq_scale,
            0.0f, 1.0f, 32.0f, 1.0f);
        k = ggml_rope_ext(
            out.ctx, k, out.pos, nullptr,
            n_rot, (int) LLAMA_ROPE_TYPE_NEOX,
            n_ctx_orig, freq_base, freq_scale,
            0.0f, 1.0f, 32.0f, 1.0f);

        q = ggml_permute(out.ctx, q, 0, 2, 1, 3);
        k = ggml_permute(out.ctx, k, 0, 2, 1, 3);
        v = ggml_permute(out.ctx, v, 0, 2, 1, 3);
        k = ggml_cast(out.ctx, k, GGML_TYPE_F16);
        v = ggml_cast(out.ctx, v, GGML_TYPE_F16);

        ggml_tensor * attn = ggml_flash_attn_ext(
            out.ctx, q, k, v, out.mask,
            1.0f / std::sqrt((float) head_dim_q),
            0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
        attn = ggml_reshape_2d(
            out.ctx, attn, wo->ne[0], tokens);

        ggml_tensor * attn_out = ggml_mul_mat(out.ctx, wo, attn);
        ggml_tensor * ffn_inp = ggml_add(out.ctx, attn_out, cur);
        const std::string attn_name =
            "layer_bench_attn_out_" + std::to_string(il);
        ggml_set_name(ffn_inp, attn_name.c_str());
        attn_ends.push_back(ffn_inp);

        ggml_tensor * ffn_norm = ggml_rms_norm(
            out.ctx, ffn_inp, rms_eps);
        ffn_norm = ggml_mul(out.ctx, ffn_norm, ffn_norm_w);

        ggml_tensor * logits = ggml_mul_mat(
            out.ctx, router_w, ffn_norm);
        ggml_tensor * probs = ggml_soft_max(out.ctx, logits);
        ggml_tensor * selected = ggml_argsort_top_k(
            out.ctx, probs, n_expert_used);

        probs = ggml_reshape_3d(
            out.ctx, probs, 1, n_expert, tokens);
        ggml_tensor * mix_weights = ggml_get_rows(
            out.ctx, probs, selected);
        mix_weights = ggml_reshape_2d(
            out.ctx, mix_weights, n_expert_used, tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(
            out.ctx, mix_weights);
        weights_sum = ggml_clamp(
            out.ctx, weights_sum, 6.103515625e-5f, INFINITY);
        mix_weights = ggml_div(
            out.ctx, mix_weights, weights_sum);
        mix_weights = ggml_reshape_3d(
            out.ctx, mix_weights, 1, n_expert_used, tokens);

        ggml_tensor * ffn3 = ggml_reshape_3d(
            out.ctx, ffn_norm, n_embd, 1, tokens);
        ggml_tensor * gate = ggml_mul_mat_id(
            out.ctx, gate_w, ffn3, selected);
        gate = ggml_silu(out.ctx, gate);
        ggml_tensor * up = ggml_mul_mat_id(
            out.ctx, up_w, ffn3, selected);
        ggml_tensor * hidden = ggml_mul(
            out.ctx, gate, up);
        ggml_tensor * experts = ggml_mul_mat_id(
            out.ctx, down_w, hidden, selected);
        experts = ggml_mul(
            out.ctx, experts, mix_weights);

        std::vector<ggml_tensor *> expert_views;
        expert_views.reserve((size_t) n_expert_used);
        for (int ie = 0; ie < n_expert_used; ++ie) {
            expert_views.push_back(
                ggml_view_2d(
                    out.ctx,
                    experts,
                    n_embd,
                    tokens,
                    experts->nb[2],
                    (size_t) ie * experts->nb[1]));
        }

        ggml_tensor * moe = expert_views.front();
        for (size_t ie = 1; ie < expert_views.size(); ++ie) {
            moe = ggml_add(out.ctx, moe, expert_views[ie]);
        }
        if (n_expert_used == 1) {
            moe = ggml_cont(out.ctx, moe);
        }

        cur = ggml_add(out.ctx, moe, ffn_inp);

        const std::string name =
            "layer_bench_out_" + std::to_string(il);
        ggml_set_name(cur, name.c_str());
        layer_ends.push_back(cur);
    }

    out.output = cur;
    out.graph = ggml_new_graph_custom(
        out.ctx, graph_size, false);
    ggml_build_forward_expand(out.graph, out.output);

    const int n_nodes = ggml_graph_n_nodes(out.graph);

    // Split the already-built full graph at the exact Transformer boundary:
    //
    //   ATTN: layer input
    //         -> RMS/QKV/QK-norm/RoPE/FA/O-proj/residual
    //         -> ffn_inp
    //
    //   FFN:  ffn_inp
    //         -> FFN RMS/router/top-k/experts/combine/residual
    //         -> layer output
    //
    // The subgraphs reuse the same tensors and buffers. No copies are inserted.
    out.attn_graphs.reserve((size_t) n_layers);
    out.ffn_graphs.reserve((size_t) n_layers);

    int previous_layer_end = -1;
    for (int li = 0; li < n_layers; ++li) {
        int attn_end_index = -1;
        int layer_end_index = -1;
        for (int i = previous_layer_end + 1; i < n_nodes; ++i) {
            ggml_tensor * node = ggml_graph_node(out.graph, i);
            if (node == attn_ends[(size_t) li]) {
                attn_end_index = i;
            }
            if (node == layer_ends[(size_t) li]) {
                layer_end_index = i;
                break;
            }
        }

        if (attn_end_index < 0 ||
            layer_end_index <= attn_end_index) {
            std::fprintf(
                stderr,
                "failed to split layer %d graph: prev=%d attn_end=%d layer_end=%d\n",
                start_layer + li,
                previous_layer_end,
                attn_end_index,
                layer_end_index);
            return false;
        }

        const int attn_nodes =
            attn_end_index - previous_layer_end;
        const int ffn_nodes =
            layer_end_index - attn_end_index;

        ggml_cgraph * attn_graph = ggml_new_graph_custom(
            out.ctx, std::max(16, attn_nodes + 4), false);
        ggml_cgraph * ffn_graph = ggml_new_graph_custom(
            out.ctx, std::max(16, ffn_nodes + 4), false);
        if (!attn_graph || !ffn_graph) {
            return false;
        }

        for (int i = previous_layer_end + 1;
             i <= attn_end_index; ++i) {
            ggml_graph_add_node(
                attn_graph,
                ggml_graph_node(out.graph, i));
        }
        for (int i = attn_end_index + 1;
             i <= layer_end_index; ++i) {
            ggml_graph_add_node(
                ffn_graph,
                ggml_graph_node(out.graph, i));
        }

        out.attn_graphs.push_back(attn_graph);
        out.ffn_graphs.push_back(ffn_graph);

        std::printf(
            "[PC_LAYER_BENCH_SPLIT] backend=%s layer=%d"
            " attn_nodes=%d ffn_nodes=%d\n",
            ggml_backend_name(backend),
            start_layer + li,
            attn_nodes,
            ffn_nodes);

        previous_layer_end = layer_end_index;
    }

    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(out.graph, i);
        if (!ggml_backend_supports_op(backend, node)) {
            std::fprintf(
                stderr,
                "backend=%s unsupported op=%s node=%s\n",
                ggml_backend_name(backend),
                ggml_op_name(node->op),
                node->name);
            return false;
        }
    }

    out.buffer = ggml_backend_alloc_ctx_tensors_from_buft(
        out.ctx,
        ggml_backend_get_default_buffer_type(backend));
    if (!out.buffer) {
        std::fprintf(
            stderr,
            "failed to allocate graph on %s\n",
            ggml_backend_name(backend));
        return false;
    }
    out.buffer_bytes = ggml_backend_buffer_get_size(out.buffer);

    if (duplicate_weights) {
        std::printf(
            "[PC_LAYER_BENCH_COPY] backend=%s tensors=%zu weight_mib=%.3f status=BEGIN\n",
            ggml_backend_name(backend),
            out.copies.size(),
            out.weight_bytes / 1048576.0);
        for (size_t i = 0; i < out.copies.size(); ++i) {
            ggml_backend_tensor_copy(
                out.copies[i].src,
                out.copies[i].dst);
        }
        ggml_backend_synchronize(backend);
        std::printf(
            "[PC_LAYER_BENCH_COPY] backend=%s tensors=%zu status=END\n",
            ggml_backend_name(backend),
            out.copies.size());
    }

    std::vector<float> input_data(
        (size_t) n_embd * (size_t) tokens);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float & x : input_data) x = dist(rng);
    ggml_backend_tensor_set(
        out.input,
        input_data.data(),
        0,
        input_data.size() * sizeof(float));

    std::vector<int32_t> pos_data((size_t) tokens);
    for (int i = 0; i < tokens; ++i) pos_data[(size_t) i] = i;
    ggml_backend_tensor_set(
        out.pos,
        pos_data.data(),
        0,
        pos_data.size() * sizeof(int32_t));

    std::vector<ggml_fp16_t> mask_data(
        (size_t) tokens * (size_t) tokens);
    const float neg_inf = -INFINITY;
    for (int iq = 0; iq < tokens; ++iq) {
        for (int ik = 0; ik < tokens; ++ik) {
            mask_data[(size_t) iq * (size_t) tokens + (size_t) ik] =
                ggml_fp32_to_fp16(ik <= iq ? 0.0f : neg_inf);
        }
    }
    ggml_backend_tensor_set(
        out.mask,
        mask_data.data(),
        0,
        mask_data.size() * sizeof(ggml_fp16_t));

    return true;
}

static void free_layer_graph(layer_graph & g) {
    if (g.buffer) ggml_backend_buffer_free(g.buffer);
    if (g.ctx) ggml_free(g.ctx);
    g = {};
}

static stats run_graph(
        const char * label,
        layer_graph & g,
        ggml_backend_t backend,
        int warmup,
        int iters) {
    for (int i = 0; i < warmup; ++i) {
        const ggml_status st =
            ggml_backend_graph_compute_async(backend, g.graph);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s warmup failed: %d\n", label, (int) st);
            std::exit(20);
        }
        ggml_backend_synchronize(backend);
    }

    std::vector<double> samples;
    samples.reserve((size_t) iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = clock_type::now();
        const ggml_status st =
            ggml_backend_graph_compute_async(backend, g.graph);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "%s run failed: %d\n", label, (int) st);
            std::exit(21);
        }
        ggml_backend_synchronize(backend);
        samples.push_back(ms_between(t0, clock_type::now()));
    }
    return summarize(samples);
}

static stats run_stage_block(
        const char * label,
        const std::vector<ggml_cgraph *> & graphs,
        ggml_backend_t backend,
        int warmup,
        int iters) {
    if (graphs.empty()) {
        return {};
    }

    auto submit_block = [&]() {
        for (ggml_cgraph * graph : graphs) {
            const ggml_status st =
                ggml_backend_graph_compute_async(backend, graph);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(
                    stderr,
                    "%s stage graph failed: %d\n",
                    label, (int) st);
                std::exit(22);
            }
        }
        ggml_backend_synchronize(backend);
    };

    for (int i = 0; i < warmup; ++i) {
        submit_block();
    }

    std::vector<double> samples;
    samples.reserve((size_t) iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = clock_type::now();
        submit_block();
        samples.push_back(ms_between(t0, clock_type::now()));
    }
    return summarize(samples);
}

int main(int argc, char ** argv) {
    options opt;
    if (!parse_options(argc, argv, opt)) {
        usage(argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_mmap = true;

    std::printf(
        "[PC_LAYER_BENCH_LOAD] model=%s mode=CPU_MMAP\n",
        opt.model_path.c_str());

    llama_model * model =
        llama_model_load_from_file(opt.model_path.c_str(), mp);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        llama_backend_free();
        return 2;
    }

    if (model->arch != LLM_ARCH_QWEN3MOE) {
        std::fprintf(
            stderr,
            "expected Qwen3-MoE, got arch enum=%d\n",
            (int) model->arch);
        llama_model_free(model);
        llama_backend_free();
        return 3;
    }

    if (opt.start_layer + opt.layers > (int) model->layers.size()) {
        std::fprintf(stderr, "layer range exceeds model\n");
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }

    if (opt.device >= ggml_backend_cuda_get_device_count()) {
        std::fprintf(stderr, "CUDA device unavailable\n");
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    ggml_backend_t cpu =
        ggml_backend_init_by_type(
            GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu =
        ggml_backend_cuda_init(opt.device);
    if (!cpu || !gpu) {
        std::fprintf(stderr, "failed to initialize CPU/CUDA backends\n");
        if (cpu) ggml_backend_free(cpu);
        if (gpu) ggml_backend_free(gpu);
        llama_model_free(model);
        llama_backend_free();
        return 6;
    }

    const bool set_threads_ok =
        set_cpu_threads(cpu, opt.cpu_threads);

    char gpu_desc[256] = {};
    ggml_backend_cuda_get_device_description(
        opt.device, gpu_desc, sizeof(gpu_desc));

    std::printf(
        "[PC_LAYER_BENCH_CONFIG] device=%d device_name=\"%s\""
        " layers=[%d,%d) tokens=%d warmup=%d iters=%d"
        " cpu_threads=%d thread_setter=%d link_gbs=%.3f\n",
        opt.device, gpu_desc,
        opt.start_layer, opt.start_layer + opt.layers,
        opt.tokens, opt.warmup, opt.iters,
        opt.cpu_threads, set_threads_ok ? 1 : 0,
        opt.link_gbs);

    size_t block_weight_bytes = 0;
    for (int li = 0; li < opt.layers; ++li) {
        const int il = opt.start_layer + li;
        qwen_layer_weights w;
        if (!get_layer_weights(model, il, w)) {
            std::fprintf(stderr, "missing layer %d weights\n", il);
            return 7;
        }
        const size_t attn = layer_attn_bytes(w);
        const size_t moe = layer_moe_bytes(w);
        const size_t total = attn + moe;
        block_weight_bytes += total;

        std::printf(
            "[PC_LAYER_BENCH_LAYER] layer=%d"
            " attn_mib=%.3f moe_mib=%.3f total_mib=%.3f"
            " q=%s k=%s v=%s gate=%s up=%s down=%s\n",
            il,
            attn / 1048576.0,
            moe / 1048576.0,
            total / 1048576.0,
            ggml_type_name(w.wq->type),
            ggml_type_name(w.wk->type),
            ggml_type_name(w.wv->type),
            ggml_type_name(w.gate->type),
            ggml_type_name(w.up->type),
            ggml_type_name(w.down->type));
    }

    const double avg_layer_bytes =
        (double) block_weight_bytes / (double) opt.layers;
    const double avg_layer_mib = avg_layer_bytes / 1048576.0;
    const double transfer_ms =
        avg_layer_bytes / (opt.link_gbs * 1.0e9) * 1000.0;

    std::printf(
        "[PC_LAYER_BENCH_TRANSFER] block_weight_mib=%.3f"
        " avg_layer_weight_mib=%.3f link_gbs=%.3f"
        " ideal_one_layer_h2d_ms=%.3f\n",
        block_weight_bytes / 1048576.0,
        avg_layer_mib,
        opt.link_gbs,
        transfer_ms);

    layer_graph cpu_graph;
    std::printf("[PC_LAYER_BENCH_BUILD] backend=CPU status=BEGIN\n");
    if (!build_full_layer_graph(
            cpu_graph, model,
            opt.start_layer, opt.layers, opt.tokens,
            cpu, false)) {
        std::fprintf(stderr, "failed to build CPU graph\n");
        free_layer_graph(cpu_graph);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        llama_model_free(model);
        llama_backend_free();
        return 8;
    }
    std::printf(
        "[PC_LAYER_BENCH_BUILD] backend=CPU status=END runtime_buffer_mib=%.3f\n",
        cpu_graph.buffer_bytes / 1048576.0);

    layer_graph gpu_graph;
    std::printf("[PC_LAYER_BENCH_BUILD] backend=CUDA status=BEGIN\n");
    if (!build_full_layer_graph(
            gpu_graph, model,
            opt.start_layer, opt.layers, opt.tokens,
            gpu, true)) {
        std::fprintf(stderr, "failed to build CUDA graph\n");
        free_layer_graph(gpu_graph);
        free_layer_graph(cpu_graph);
        ggml_backend_free(gpu);
        ggml_backend_free(cpu);
        llama_model_free(model);
        llama_backend_free();
        return 9;
    }
    std::printf(
        "[PC_LAYER_BENCH_BUILD] backend=CUDA status=END"
        " buffer_mib=%.3f weight_mib=%.3f\n",
        gpu_graph.buffer_bytes / 1048576.0,
        gpu_graph.weight_bytes / 1048576.0);

    const stats cs = run_graph(
        "CPU", cpu_graph, cpu, opt.warmup, opt.iters);
    const stats gs = run_graph(
        "CUDA", gpu_graph, gpu, opt.warmup, opt.iters);

    // Stage timing intentionally submits all selected layers' ATTN graphs as
    // one block, then all selected layers' FFN graphs as one block. This
    // keeps distinct real layer weights in flight while paying only one host
    // synchronize per measured block, so the result is not dominated by
    // per-layer synchronization overhead.
    const stats cpu_attn = run_stage_block(
        "CPU_ATTN", cpu_graph.attn_graphs,
        cpu, opt.warmup, opt.iters);
    const stats cpu_ffn = run_stage_block(
        "CPU_FFN", cpu_graph.ffn_graphs,
        cpu, opt.warmup, opt.iters);
    const stats gpu_attn = run_stage_block(
        "CUDA_ATTN", gpu_graph.attn_graphs,
        gpu, opt.warmup, opt.iters);
    const stats gpu_ffn = run_stage_block(
        "CUDA_FFN", gpu_graph.ffn_graphs,
        gpu, opt.warmup, opt.iters);

    std::printf(
        "[PC_LAYER_BENCH_RESULT] backend=CPU"
        " block_layers=%d tokens=%d"
        " mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f min_ms=%.6f max_ms=%.6f"
        " per_layer_mean_ms=%.6f per_layer_p50_ms=%.6f\n",
        opt.layers, opt.tokens,
        cs.mean, cs.p50, cs.p95, cs.min, cs.max,
        cs.mean / opt.layers,
        cs.p50 / opt.layers);

    std::printf(
        "[PC_LAYER_BENCH_RESULT] backend=CUDA"
        " block_layers=%d tokens=%d"
        " mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f min_ms=%.6f max_ms=%.6f"
        " per_layer_mean_ms=%.6f per_layer_p50_ms=%.6f\n",
        opt.layers, opt.tokens,
        gs.mean, gs.p50, gs.p95, gs.min, gs.max,
        gs.mean / opt.layers,
        gs.p50 / opt.layers);

    const auto print_stage = [&](const char * backend_name,
                                 const char * stage,
                                 const stats & s) {
        std::printf(
            "[PC_LAYER_BENCH_STAGE] backend=%s stage=%s"
            " block_layers=%d tokens=%d"
            " mean_ms=%.6f p50_ms=%.6f p95_ms=%.6f min_ms=%.6f max_ms=%.6f"
            " per_layer_mean_ms=%.6f per_layer_p50_ms=%.6f\n",
            backend_name, stage,
            opt.layers, opt.tokens,
            s.mean, s.p50, s.p95, s.min, s.max,
            s.mean / opt.layers,
            s.p50 / opt.layers);
    };

    print_stage("CPU",  "ATTN", cpu_attn);
    print_stage("CPU",  "FFN",  cpu_ffn);
    print_stage("CUDA", "ATTN", gpu_attn);
    print_stage("CUDA", "FFN",  gpu_ffn);

    std::printf(
        "[PC_LAYER_BENCH_STAGE_SUM] backend=CPU"
        " attn_per_layer_ms=%.6f ffn_per_layer_ms=%.6f"
        " split_sum_per_layer_ms=%.6f full_per_layer_ms=%.6f"
        " split_over_full=%.4fx\n",
        cpu_attn.mean / opt.layers,
        cpu_ffn.mean / opt.layers,
        (cpu_attn.mean + cpu_ffn.mean) / opt.layers,
        cs.mean / opt.layers,
        cs.mean > 0.0 ?
            (cpu_attn.mean + cpu_ffn.mean) / cs.mean : 0.0);

    std::printf(
        "[PC_LAYER_BENCH_STAGE_SUM] backend=CUDA"
        " attn_per_layer_ms=%.6f ffn_per_layer_ms=%.6f"
        " split_sum_per_layer_ms=%.6f full_per_layer_ms=%.6f"
        " split_over_full=%.4fx\n",
        gpu_attn.mean / opt.layers,
        gpu_ffn.mean / opt.layers,
        (gpu_attn.mean + gpu_ffn.mean) / opt.layers,
        gs.mean / opt.layers,
        gs.mean > 0.0 ?
            (gpu_attn.mean + gpu_ffn.mean) / gs.mean : 0.0);

    std::printf(
        "[PC_LAYER_BENCH_COMPARE]"
        " cpu_per_layer_ms=%.6f gpu_per_layer_ms=%.6f"
        " cpu_over_gpu=%.3fx"
        " ideal_weight_h2d_ms=%.6f"
        " h2d_over_gpu_compute=%.3fx"
        " h2d_over_cpu_compute=%.3fx\n",
        cs.mean / opt.layers,
        gs.mean / opt.layers,
        gs.mean > 0.0 ? cs.mean / gs.mean : 0.0,
        transfer_ms,
        gs.mean > 0.0 ?
            transfer_ms / (gs.mean / opt.layers) : 0.0,
        cs.mean > 0.0 ?
            transfer_ms / (cs.mean / opt.layers) : 0.0);

    free_layer_graph(gpu_graph);
    free_layer_graph(cpu_graph);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
