#pragma once

// Independent, normal llama_decode() CPU oracle for the six physical Cycle
// boundaries. This is a TEST fixture, never part of the inference runtime.
// CPU model/context must be fully destroyed before loading hybrid RPC shards.
//
// Opt in from test-hybrid-cycle-real-pc-ring with
// LLAMA_HYBRID_CYCLE_REAL_REFERENCE=1 and ALL real stage flags enabled.
// Compare the SAME token embedding and F16-KV causal prompt across six
// physical layer outputs. Use the V5 strict period-parity tolerance verbatim.

#include "llama.h"
#include "llama-model.h"
#include "ggml-backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace cycle_full_reference {

inline bool env_set(const char * name, const char * value) {
#ifdef _WIN32
    return _putenv_s(name, value ? value : "") == 0;
#else
    return value ? setenv(name, value, 1) == 0 :
                   unsetenv(name) == 0;
#endif
}

class no_hybrid_env {
public:
    no_hybrid_env() {
        const char * pc = std::getenv("LLAMA_HYBRID_CYCLE_PC_LAYERS");
        const char * phone = std::getenv("LLAMA_HYBRID_CYCLE_PHONE_LAYERS");
        pc_present_ = pc != nullptr;
        phone_present_ = phone != nullptr;
        if (pc_present_) pc_ = pc;
        if (phone_present_) phone_ = phone;
        ok_ = env_set("LLAMA_HYBRID_CYCLE_PC_LAYERS", nullptr) &&
              env_set("LLAMA_HYBRID_CYCLE_PHONE_LAYERS", nullptr);
    }
    no_hybrid_env(const no_hybrid_env &) = delete;
    no_hybrid_env & operator=(const no_hybrid_env &) = delete;
    ~no_hybrid_env() {
        if (!env_set("LLAMA_HYBRID_CYCLE_PC_LAYERS",
                     pc_present_ ? pc_.c_str() : nullptr) ||
            !env_set("LLAMA_HYBRID_CYCLE_PHONE_LAYERS",
                     phone_present_ ? phone_.c_str() : nullptr)) {
            std::fprintf(stderr,
                "[CYCLE_FULL_REFERENCE] status=FAIL "
                "reason=RESTORE_PLACEMENT_ENV\n");
        }
    }
    bool ok() const { return ok_; }

private:
    std::string pc_;
    std::string phone_;
    bool pc_present_ = false;
    bool phone_present_ = false;
    bool ok_ = false;
};

struct capture {
    int64_t hidden = 0;
    int tokens = 0;
    bool error = false;
    std::vector<std::string> name;
    std::vector<int> counts;
    std::vector<std::vector<float>> values;

    capture(int64_t n_hidden, int n_tokens, bool all_physical_layers)
        : hidden(n_hidden), tokens(n_tokens) {
        name.push_back("embd");
        for (int l = 0; l < 48; ++l) {
            if (all_physical_layers || (l % 8 == 7)) {
                name.push_back("l_out-" + std::to_string(l));
            }
        }
        counts.resize(name.size(), 0);
        values.resize(name.size());
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * user) {
        auto & self = *static_cast<capture *>(user);
        if (!tensor) {
            self.error = true;
            return false;
        }
        int found = -1;
        for (size_t i = 0; i < self.name.size(); ++i) {
            if (std::strcmp(tensor->name, self.name[i].c_str()) == 0) {
                found = int(i);
                break;
            }
        }
        if (found < 0) return !ask;
        if (ask) return true;
        const size_t index = size_t(found);
        const bool valid =
            self.counts[index] == 0 && tensor->buffer &&
            tensor->type == GGML_TYPE_F32 &&
            ggml_is_contiguous(tensor) &&
            tensor->ne[0] == self.hidden &&
            tensor->ne[1] == self.tokens &&
            ggml_nelements(tensor) == self.hidden * self.tokens &&
            ggml_nbytes(tensor) == sizeof(float) *
                size_t(self.hidden) * size_t(self.tokens);
        if (!valid) {
            // The last Transformer layer can be pruned to n_outputs=1
            // when llama_batch::logits only requests the final position.
            // Print the actual tensor contract rather than hiding whether
            // the failure was shape, type, buffer, or a duplicate callback.
            std::fprintf(stderr,
                "[CYCLE_FULL_REFERENCE_CAPTURE] name=%s "
                "status=FAIL reason=TYPE_SHAPE_OR_DUPLICATE "
                "count=%d actual_type=%s actual_ne=[%lld,%lld,%lld,%lld] "
                "actual_contiguous=%d buffer=%d actual_bytes=%zu "
                "expected_type=F32 expected_ne=[%lld,%d] "
                "expected_bytes=%zu\n",
                self.name[index].c_str(), self.counts[index],
                ggml_type_name(tensor->type),
                (long long) tensor->ne[0],
                (long long) tensor->ne[1],
                (long long) tensor->ne[2],
                (long long) tensor->ne[3],
                ggml_is_contiguous(tensor) ? 1 : 0,
                tensor->buffer ? 1 : 0,
                ggml_nbytes(tensor),
                (long long) self.hidden, self.tokens,
                sizeof(float) * size_t(self.hidden) * size_t(self.tokens));
            self.error = true;
            return false;
        }
        auto & dest = self.values[index];
        dest.resize(size_t(self.hidden) * size_t(self.tokens));
        ggml_backend_tensor_get(tensor, dest.data(), 0,
                                dest.size() * sizeof(float));
        ++self.counts[index];
        std::printf(
            "[CYCLE_FULL_REFERENCE_CAPTURE] name=%s tokens=%d "
            "hidden=%lld status=PASS\n",
            self.name[index].c_str(), self.tokens,
            (long long) self.hidden);
        return true;
    }

    bool complete() const {
        if (error) return false;
        for (size_t i = 0; i < name.size(); ++i) {
            if (counts[i] != 1 ||
                values[i].size() != size_t(hidden) * size_t(tokens)) {
                return false;
            }
            for (float v : values[i]) {
                if (!std::isfinite(v)) return false;
            }
        }
        return true;
    }
};

struct baseline {
    int n_tokens = 0;
    int64_t hidden = 0;
    std::vector<float> embedding;
    std::array<std::vector<float>, 6> after_period;
    // Only retained with the expensive, explicit 48-layer diagnostic.
    std::array<std::vector<float>, 48> after_layer;

    bool run(const char * model_path, ggml_backend_dev_t cpu_device,
             int required_tokens, bool trace_all_layers = false) {
        if (!model_path || !cpu_device ||
            required_tokens < 1 || required_tokens > 64) {
            return false;
        }
        no_hybrid_env saved_env;
        if (!saved_env.ok()) {
            std::fprintf(stderr,
                "[CYCLE_FULL_REFERENCE] status=FAIL "
                "reason=DISABLE_CYCLE_ENV\n");
            return false;
        }
        ggml_backend_dev_t devices[] = {cpu_device, nullptr};
        llama_model_params mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        mp.hybrid_auto = false;
        mp.use_mmap = true;
        mp.use_extra_bufts = false; // No CPU_REPACK duplication for 30B.
        std::printf(
            "[CYCLE_FULL_REFERENCE] source=CPU_ONLY_LLAMA_DECODE "
            "model_load=BEGIN tokens=%d no_gpu=1\n", required_tokens);
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(model_path, mp), &llama_model_free);
        if (!model || model->arch != LLM_ARCH_QWEN3MOE ||
            llama_model_n_layer(model.get()) != 48 ||
            model->hparams.n_embd <= 0) {
            std::fprintf(stderr,
                "[CYCLE_FULL_REFERENCE] status=FAIL reason=MODEL\n");
            return false;
        }
        capture state(model->hparams.n_embd, required_tokens, trace_all_layers);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = 128;
        cp.n_ubatch = 128;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        // Explicit Cycle Attention does not use Flash Attention; match
        // the legacy V5 reference A/B path without relaxing parity bounds.
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.cb_eval = &capture::callback;
        cp.cb_eval_user_data = &state;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), cp), &llama_free);
        if (!context) {
            std::fprintf(stderr,
                "[CYCLE_FULL_REFERENCE] status=FAIL reason=CONTEXT\n");
            return false;
        }
        // Reproduce the V5 fixed-token prompt construction, including BOS.
        const std::string seed =
            "The quick brown fox jumps over the lazy dog. "
            "Please explain a transformer.";
        std::string prompt = seed;
        std::vector<llama_token> tokens(4096);
        int32_t count = -1;
        const auto * vocab = llama_model_get_vocab(model.get());
        for (int attempt = 0; attempt < 256; ++attempt) {
            count = llama_tokenize(
                vocab, prompt.c_str(), int32_t(prompt.size()),
                tokens.data(), int32_t(tokens.size()), true, true);
            if (count < 0) {
                tokens.resize(size_t(-count));
                count = llama_tokenize(
                    vocab, prompt.c_str(), int32_t(prompt.size()),
                    tokens.data(), int32_t(tokens.size()), true, true);
            }
            if (count < 0 || count >= required_tokens) break;
            if (prompt.size() > 65536) break;
            prompt.append(" ");
            prompt.append(seed);
        }
        if (count < required_tokens) {
            std::fprintf(stderr,
                "[CYCLE_FULL_REFERENCE] status=FAIL reason=TOKENIZE\n");
            return false;
        }
        llama_batch batch = llama_batch_init(required_tokens, 0, 1);
        batch.n_tokens = required_tokens;
        for (int i = 0; i < required_tokens; ++i) {
            batch.token[i] = tokens[size_t(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            // The CPU oracle captures l_out-47 for ALL input tokens.
            // Request every logit so llama.cpp's last-layer output pruning
            // cannot reduce l_out-47 to only the final position. Other
            // captured periods do not need this, but the final one does.
            batch.logits[i] = 1;
        }
        std::printf(
            "[CYCLE_FULL_REFERENCE] phase=DECODE "
            "tokens=%d logits_requested=%d "
            "last_layer_output_pruning=DISABLED\n",
            required_tokens, required_tokens);
        std::fflush(stdout);
        const int status = llama_decode(context.get(), batch);
        llama_synchronize(context.get());
        llama_batch_free(batch);
        if (status != 0 || !state.complete()) {
            std::fprintf(stderr,
                "[CYCLE_FULL_REFERENCE] status=FAIL "
                "reason=DECODE_OR_CAPTURE code=%d\n", status);
            return false;
        }
        n_tokens = required_tokens;
        hidden = state.hidden;
        embedding = std::move(state.values[0]);
        if (trace_all_layers) {
            // Every layer is retained for optional numerical localization.
            // The additional CPU host storage is at most 48*64*2048*4B,
            // far less than a second 30B model allocation.
            for (int l = 0; l < 48; ++l) {
                after_layer[size_t(l)] =
                    std::move(state.values[size_t(l + 1)]);
            }
            for (int period = 0; period < 6; ++period) {
                after_period[size_t(period)] =
                    after_layer[size_t(8 * period + 7)];
            }
        } else {
            for (int period = 0; period < 6; ++period) {
                after_period[size_t(period)] =
                    std::move(state.values[size_t(period + 1)]);
            }
        }
        std::printf(
            "[CYCLE_FULL_REFERENCE] model=CPU_ONLY "
            "capture=%s "
            "model_lifetime=SEPARATE_BEFORE_HYBRID "
            "tokens=%d hidden=%lld status=PASS\n",
            trace_all_layers ? "EMBEDDING_AND_ALL_48_L_OUT" :
                "EMBEDDING_AND_SIX_L_OUT",
            n_tokens, (long long) hidden);
        return true;
    }
};

// Exact V5 period parity tolerance. Do not silently replace this by a
// permissive cosine-only criterion: one bad high-amplitude channel matters.
// Both full prompt and every 16-token Chunk must individually meet the bound.
inline bool compare_period(const std::vector<float> & actual,
                           const std::vector<float> & ref,
                           int physical_layer,
                           int tokens, int chunk_tokens, int64_t hidden) {
    const size_t expected = size_t(tokens) * size_t(hidden);
    if (tokens <= 0 || hidden <= 0 || chunk_tokens <= 0 ||
        actual.size() != expected || ref.size() != expected) {
        std::fprintf(stderr,
            "[CYCLE_FULL_PARITY] layer=%d status=FAIL "
            "reason=REFERENCE_SHAPE_MISMATCH\n", physical_layer);
        return false;
    }
    bool passed = true;
    const int n_chunks = (tokens + chunk_tokens - 1) / chunk_tokens;
    for (int c = -1; c < n_chunks; ++c) {
        const size_t begin = c < 0 ? 0 :
            size_t(c) * size_t(chunk_tokens) * size_t(hidden);
        const size_t end = c < 0 ? expected : std::min(expected,
            size_t(c + 1) * size_t(chunk_tokens) * size_t(hidden));
        double error2 = 0.0;
        double dot = 0.0, sum_a2 = 0.0, sum_r2 = 0.0;
        double max_abs = 0.0, max_ref = 0.0;
        size_t worst = begin;
        bool finite = true;
        for (size_t i = begin; i < end; ++i) {
            const double a = actual[i], b = ref[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                finite = false;
                break;
            }
            const double delta = std::fabs(a - b);
            if (delta > max_abs) {
                max_abs = delta;
                worst = i;
            }
            max_ref = std::max(max_ref, std::fabs(b));
            error2 += delta * delta;
            dot += a * b;
            sum_a2 += a * a;
            sum_r2 += b * b;
        }
        const double rms = finite ?
            std::sqrt(error2 / double(end - begin)) : 0.0;
        const double cosine = finite && sum_a2 > 0 && sum_r2 > 0 ?
            dot / std::sqrt(sum_a2 * sum_r2) : 0.0;
        const double tolerance = 0.02 + 0.00002 * max_ref;
        const bool ok = finite && max_abs <= tolerance;
        passed = passed && ok;
        std::printf(
            "[CYCLE_FULL_PARITY] layer=%d scope=%s chunk=%d "
            "elements=%zu max_abs=%.9g rms=%.9g cosine=%.9g "
            "max_ref=%.9g strict_tolerance=%.9g "
            "worst_token=%lld worst_channel=%lld "
            "status=%s\n",
            physical_layer, c < 0 ? "FULL" : "CHUNK", c,
            end - begin, max_abs, rms, cosine, max_ref, tolerance,
            (long long)(worst / size_t(hidden)),
            (long long)(worst % size_t(hidden)),
            ok ? "PASS" : "FAIL");
    }
    return passed;
}

// Trace-only, fine-grained check for one immutable output Chunk of an
// absolute physical layer. Threshold is EXACTLY the V5 strict period bound,
// without relaxed cosine-only acceptance or hardware-specific exceptions.
inline bool compare_layer_chunk(
        const std::vector<float> & actual,
        const std::vector<float> & reference,
        int layer, int chunk, int64_t start, int64_t tokens,
        int64_t hidden, const char * backend) {
    const bool shape_ok = layer >= 0 && layer < 48 && chunk >= 0 &&
        start >= 0 && tokens > 0 && hidden > 0 &&
        actual.size() == size_t(tokens) * size_t(hidden) &&
        size_t(start + tokens) * size_t(hidden) <= reference.size();
    if (!shape_ok) {
        std::printf(
            "[CYCLE_LAYER_PARITY] layer=%d chunk=%d backend=%s "
            "status=FAIL reason=SHAPE_OR_REFERENCE "
            "actual=%zu reference=%zu tokens=%lld start=%lld\n",
            layer, chunk, backend, actual.size(), reference.size(),
            (long long) tokens, (long long) start);
        return false;
    }
    const size_t n = actual.size();
    const size_t first = size_t(start) * size_t(hidden);
    double worst_abs = 0, max_ref = 0;
    double squared = 0, dot = 0, actual2 = 0, reference2 = 0;
    size_t worst = 0;
    bool finite = true;
    for (size_t i = 0; i < n; ++i) {
        const double a = actual[i], b = reference[first + i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            finite = false;
            worst = i;
            break;
        }
        const double delta = std::fabs(a - b);
        if (delta > worst_abs) {
            worst_abs = delta;
            worst = i;
        }
        max_ref = std::max(max_ref, std::fabs(b));
        squared += delta * delta;
        dot += a * b;
        actual2 += a * a;
        reference2 += b * b;
    }
    const double rms = finite ? std::sqrt(squared / double(n)) : 0;
    const double cosine = finite && actual2 > 0 && reference2 > 0 ?
        dot / std::sqrt(actual2 * reference2) : 0;
    const double tolerance = 0.02 + 0.00002 * max_ref;
    const bool passed = finite && worst_abs <= tolerance;
    std::printf(
        "[CYCLE_LAYER_PARITY] layer=%d chunk=%d backend=%s "
        "positions=[%lld,%lld] max_abs=%.9g rms=%.9g "
        "cosine=%.9g max_ref=%.9g strict_tolerance=%.9g "
        "worst_token=%lld worst_channel=%lld "
        "worst_actual=%.9g worst_reference=%.9g status=%s\n",
        layer, chunk, backend,
        (long long) start, (long long)(start + tokens - 1),
        worst_abs, rms, cosine, max_ref, tolerance,
        (long long) (start + int64_t(worst / size_t(hidden))),
        (long long) (worst % size_t(hidden)),
        actual[worst], reference[first + worst],
        passed ? "PASS" : "FAIL");
    return passed;
}

} // namespace cycle_full_reference
