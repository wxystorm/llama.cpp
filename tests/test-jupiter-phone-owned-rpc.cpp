// Standalone Phone-owned HTP prepare/run smoke. Does not alter the normal
// llama-cli inference path or claim output-token correctness.
#include "../ggml/src/ggml-rpc/phone-owned-jupiter-api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static ggml_phone_owned::region_plan qwen3_14b_plan() {
    ggml_phone_owned::region_plan p {};
    p.version         = ggml_phone_owned::k_multi_plan_version;
    p.flags           = ggml_phone_owned::flags_for_layers(8);
    p.run_id          = 20261122;
    p.p2c_run_id      = 20261122;
    p.region          = 0;
    p.layer           = 32;
    p.total_tokens    = 175;
    p.chunk_tokens    = 88;
    p.n_embd          = 5120;
    p.position_base   = 0;
    p.kv_capacity     = 512;
    p.n_heads         = 40;
    p.n_heads_kv      = 8;
    p.head_dim        = 128;
    p.n_kv            = 512;
    p.rope_type       = ggml_phone_owned::k_rope_neox;
    p.n_ctx_orig      = 32768;
    p.kv_type         = ggml_phone_owned::k_f16;
    p.rms_norm_eps    = 1e-6f;
    p.rope_freq_base  = 1000000.0f;
    p.rope_freq_scale = 1.0f;
    p.rope_ext_factor = -1.0f;
    p.rope_attn_factor = 1.0f;
    p.rope_beta_fast  = 32.0f;
    p.rope_beta_slow  = 1.0f;
    return p;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4 ||
            (std::string(argv[2]) != "1" && std::string(argv[2]) != "4")) {
        std::fprintf(stderr,
            "usage: %s <phone-host:port> <sessions:1|4> [--run]\n"
            "Requires a rebuilt phone RPC server exposing HTP devices\n"
            "and a local Qwen3-14B GGUF tensor source.\n", argv[0]);
        return 2;
    }
    if (argc == 4 && std::string(argv[3]) != "--run") {
        return 2;
    }
    const std::string endpoint(argv[1]);
    const bool execute = argc == 4;
    const int sessions = std::atoi(argv[2]);
    auto p = qwen3_14b_plan();
    const char * reason = "NONE";
    if (!ggml_phone_owned::validate(p, reason)) {
        std::fprintf(stderr, "invalid plan: %s\n", reason);
        return 2;
    }
    // The current Jupiter HTP8_S4 stage layout is 2 layers per device.
    std::vector<uint32_t> devices(8, 0);
    if (sessions == 4) {
        for (size_t i = 0; i < devices.size(); ++i) {
            devices[i] = static_cast<uint32_t>(i / 2);
        }
    }
    ggml_jupiter_phone_owned::prepare_stats prep {};
    if (!ggml_jupiter_phone_owned::prepare(
            endpoint.c_str(), p, devices, prep)) {
        std::fprintf(stderr, "native phone-owned prepare FAILED\n");
        return 1;
    }
    std::printf("[PHONE_OWNED_SMOKE] event=PREPARE sessions=%d"
                " graphs=%u model_load_ms=%.3f build_ms=%.3f"
                " buffer_ms=%.3f precompile_ms=%.3f wall_ms=%.3f status=PASS\n",
                sessions, prep.graphs, prep.model_load_ms, prep.build_ms,
                prep.buffer_alloc_ms, prep.precompile_ms, prep.wall_ms);

    bool ok = true;
    if (execute) {
        for (uint32_t chunk = 0;
                chunk < ggml_phone_owned::chunk_count(p); ++chunk) {
            const uint32_t n = std::min(
                p.chunk_tokens, p.total_tokens - chunk * p.chunk_tokens);
            // Synthetic finite input is a functional smoke only. Numeric
            // parity must be tested against the real PC layer boundary.
            std::vector<float> input(size_t(n) * p.n_embd, 0.0f);
            std::vector<float> output;
            ggml_jupiter_phone_owned::run_stats st {};
            if (!ggml_jupiter_phone_owned::run_chunk(
                    endpoint.c_str(), p.run_id, p.region, chunk,
                    input.data(), input.size(), output, st) ||
                    output.size() != input.size()) {
                std::fprintf(stderr, "[PHONE_OWNED_SMOKE] RUN_FAILED chunk=%u\n", chunk);
                ok = false;
                break;
            }
            const bool finite = std::all_of(output.begin(), output.end(),
                [](float x) { return std::isfinite(x); });
            std::printf("[PHONE_OWNED_SMOKE] event=RUN chunk=%u tokens=%u"
                        " layers=%u upload_ms=%.3f compute_ms=%.3f"
                        " copy_ms=%.3f wall_ms=%.3f status=%s\n",
                        chunk, st.tokens, st.layers, st.upload_ms,
                        st.compute_ms, st.output_copy_ms, st.wall_ms,
                        finite ? "PASS" : "FAIL");
            if (!finite) { ok = false; break; }
        }
    }

    if (!ggml_jupiter_phone_owned::reset(endpoint.c_str(), p.run_id)) {
        std::fprintf(stderr, "[PHONE_OWNED_SMOKE] RESET_FAILED\n");
        ok = false;
    }
    return ok ? 0 : 1;
}
