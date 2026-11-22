#pragma once

// Opt-in Jupiter bridge for native Phone-owned Qwen3 prefill graphs.
// The native phone HTP Graph is prepared and executed without the PC-built
// GGML RPC Graph. This pilot returns the last Transformer hidden state;
// it intentionally does not replace the llama.cpp final norm/head or Decode.
#include "ggml-rpc.h"
#include "phone-owned-execution-plan.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml_jupiter_phone_owned {

struct prepare_stats {
    uint32_t graphs = 0;
    double model_load_ms = 0.0;
    double build_ms = 0.0;
    double buffer_alloc_ms = 0.0;
    double precompile_ms = 0.0;
    double wall_ms = 0.0;
};

struct run_stats {
    uint32_t layers = 0;
    uint32_t tokens = 0;
    double upload_ms = 0.0;
    double compute_ms = 0.0;
    double output_copy_ms = 0.0;
    double wall_ms = 0.0;
};

// Pass a device ID for each consecutive Transformer layer. Examples:
// S1: [0,0,0,0,0,0,0,0]; S4: [0,0,1,1,2,2,3,3].
// The server loads the selected weights from its local GGUF source,
// builds/cache-compiles one native HTP graph per layer and chunk, and
// retains private KV. A registered run must be explicitly reset.
GGML_BACKEND_API bool prepare(
    const char * endpoint,
    const ggml_phone_owned::region_plan & plan,
    const std::vector<uint32_t> & layer_devices,
    prepare_stats & stats);

// Phone-side per-Chunk full-layer chain. Input and output are F32
// [n_embd, n_tokens] hidden states. No final norm/head is executed.
GGML_BACKEND_API bool run_chunk(
    const char * endpoint,
    uint64_t run_id, uint32_t region, uint32_t chunk,
    const float * input, size_t input_floats,
    std::vector<float> & output, run_stats & stats);

// Releases all pilot Phone-owned graphs and private KV for the sole active
// run. Do not call concurrently with native chunk execution.
GGML_BACKEND_API bool reset(
    const char * endpoint, uint64_t run_id);

} // namespace ggml_jupiter_phone_owned
