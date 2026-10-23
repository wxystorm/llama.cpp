#pragma once

#include "llama.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

// Qwen3-MoE Cycle's host-side model I/O. These are REAL operations on the
// already-loaded and physically mapped model, not an extra CPU reference
// model. Device placement for these two endpoint weights must be CPU-readable.
//
// This module owns only temporary GGML input/compute buffers. It never
// creates a second GGUF model, KV cache, scheduler, or inference session.
bool llama_hybrid_cycle_embed_tokens(
    const llama_model & model,
    ggml_backend_t cpu,
    const std::vector<llama_token> & tokens,
    std::vector<float> & embeddings,
    std::string & reason);

// Applies actual output_norm + output weight to the *final token* hidden
// state. Returns n_vocab F32 logits. For generation, the next-token sampler
// consumes these logits; never treat the L47 raw hidden state as logits.
bool llama_hybrid_cycle_project_last_token(
    const llama_model & model,
    ggml_backend_t cpu,
    const std::vector<float> & hidden,
    int n_tokens,
    std::vector<float> & logits,
    std::string & reason);
