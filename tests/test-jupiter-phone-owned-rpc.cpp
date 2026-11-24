// Standalone Phone-owned HTP prepare/run smoke. Does not alter the normal
// llama-cli inference path or claim output-token correctness.
#include "../ggml/src/ggml-rpc/phone-owned-jupiter-api.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static ggml_phone_owned::region_plan qwen3_14b_plan(uint32_t phone_layers) {
    ggml_phone_owned::region_plan p {};
    p.version         = phone_layers == 1 ?
        ggml_phone_owned::k_plan_version : ggml_phone_owned::k_multi_plan_version;
    p.flags           = phone_layers == 1 ?
        ggml_phone_owned::k_flag_flash_attn :
        ggml_phone_owned::flags_for_layers(phone_layers);
    p.run_id          = 20261122;
    p.p2c_run_id      = 20261122;
    p.region          = 0;
    p.layer           = 40 - phone_layers;
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
    if (argc < 3 || std::atoi(argv[2]) < 1 || std::atoi(argv[2]) > 4) {
        std::fprintf(stderr,
            "usage: %s <phone-host:port> <sessions:1..4>"
            " [--layers <1..39>] [--run|--auto-run]\n"
            "Requires a rebuilt phone RPC server exposing HTP devices\n"
            "and a local Qwen3-14B GGUF tensor source.\n", argv[0]);
        return 2;
    }
    // Preserve the old eight-layer smoke as the default, while allowing
    // any 14B suffix with a nonempty PC prefix.
    uint32_t phone_layers = 8;
    bool execute = false;
    bool autonomous = false;
    for (int i = 3; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--run" || arg == "--auto-run") {
            if (execute) { return 2; }
            execute = true;
            autonomous = arg == "--auto-run";
        } else if (arg == "--layers" && i + 1 < argc) {
            char * end = nullptr;
            const char * value = argv[++i];
            const unsigned long count = std::strtoul(value, &end, 10);
            if (end == value || *end != '\0' || count < 1 || count >= 40 ||
                    count > ggml_phone_owned::k_max_phone_layers_per_region) {
                std::fprintf(stderr, "invalid 14B Phone layer count: %s\n", value);
                return 2;
            }
            phone_layers = static_cast<uint32_t>(count);
        } else {
            return 2;
        }
    }
    // Refuse older server protocol before sending native opcodes.
#ifdef _WIN32
    _putenv_s("LLAMA_JUPITER_PHONE_OWNED_NATIVE", "1");
    if (autonomous) { _putenv_s("LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS", "1"); }
#else
    setenv("LLAMA_JUPITER_PHONE_OWNED_NATIVE", "1", 1);
    if (autonomous) { setenv("LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS", "1", 1); }
#endif
    const std::string endpoint(argv[1]);
    const int sessions = std::atoi(argv[2]);
    auto p = qwen3_14b_plan(phone_layers);
    const char * reason = "NONE";
    if (!ggml_phone_owned::validate(p, reason)) {
        std::fprintf(stderr, "invalid plan: %s\n", reason);
        return 2;
    }
    // Map every Phone layer to its balanced HTP Session; the original
    // eight-layer S4 layout remains [0,0,1,1,2,2,3,3].
    std::vector<uint32_t> devices(phone_layers, 0);
    for (size_t i = 0; i < devices.size(); ++i) {
        devices[i] = static_cast<uint32_t>(i*sessions/devices.size());
    }
    ggml_jupiter_phone_owned::prepare_stats prep {};
    if (!ggml_jupiter_phone_owned::prepare(
            endpoint.c_str(), p, devices, prep)) {
        std::fprintf(stderr, "native phone-owned prepare FAILED\n");
        return 1;
    }
    std::printf("[PHONE_OWNED_SMOKE] event=PREPARE sessions=%d layers=%u"
                " graphs=%u model_load_ms=%.3f build_ms=%.3f"
                " buffer_ms=%.3f precompile_ms=%.3f wall_ms=%.3f status=PASS\n",
                sessions, phone_layers, prep.graphs, prep.model_load_ms, prep.build_ms,
                prep.buffer_alloc_ms, prep.precompile_ms, prep.wall_ms);

    bool ok = true;
    if (autonomous) {
        if (!ggml_jupiter_phone_owned::auto_start(
                endpoint.c_str(), p.run_id, p.p2c_run_id, p.region, 2)) {
            std::fprintf(stderr, "[PHONE_OWNED_SMOKE] AUTO_START_FAILED\n");
            ok = false;
        }
        for (uint32_t chunk = 0;
                ok && chunk < ggml_phone_owned::chunk_count(p); ++chunk) {
            const uint32_t n = std::min(
                p.chunk_tokens, p.total_tokens - chunk*p.chunk_tokens);
            std::vector<float> input(size_t(n)*p.n_embd, 0.0f);
            if (!ggml_jupiter_phone_owned::auto_push(
                    endpoint.c_str(), p.run_id, p.p2c_run_id, p.region,
                    chunk, input.data(), input.size())) {
                std::fprintf(stderr,
                    "[PHONE_OWNED_SMOKE] AUTO_PUSH_FAILED chunk=%u\n", chunk);
                ok = false;
                break;
            }
            std::printf("[PHONE_OWNED_SMOKE] AUTO_PUSH chunk=%u"
                        " tokens=%u status=QUEUED\n", chunk, n);
        }
        if (ok) {
            std::vector<float> output;
            ggml_jupiter_phone_owned::run_stats st {};
            const uint32_t last_tokens = p.total_tokens % p.chunk_tokens ?
                p.total_tokens % p.chunk_tokens : p.chunk_tokens;
            ok = ggml_jupiter_phone_owned::auto_p2c_wait(
                endpoint.c_str(), p.run_id, p.p2c_run_id,
                p.region, output, st);
            ok = ok && output.size() == size_t(last_tokens)*p.n_embd &&
                st.layers == phone_layers && st.tokens == last_tokens &&
                std::all_of(output.begin(), output.end(),
                    [](float x) { return std::isfinite(x); });
            std::printf("[PHONE_OWNED_SMOKE] AUTO_P2C"
                        " tokens=%u layers=%u status=%s\n",
                        st.tokens, st.layers, ok ? "PASS" : "FAIL");
        }
    } else if (execute) {
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
            if (!finite || st.layers != phone_layers) { ok = false; break; }
        }
    }

    if (!ggml_jupiter_phone_owned::reset(endpoint.c_str(), p.run_id)) {
        std::fprintf(stderr, "[PHONE_OWNED_SMOKE] RESET_FAILED\n");
        ok = false;
    }
    return ok ? 0 : 1;
}
