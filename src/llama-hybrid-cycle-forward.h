#pragma once

#include "ggml-backend.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

struct llama_model;

// Executable 48-physical-layer Cycle prefill, independent of the test binary.
// The caller owns its already-loaded *physically sharded* model and the CPU
// and Phone RPC backend handles. The model and backends must outlive this call.
//
// Real path ONLY: PC Full(4) -> C2P(1) -> Phone Full(2) -> P2C(1), x6.
// Uses one global five-queue FIFO, the existing Region/Buffer Ownership pools,
// persistent per-physical-layer prompt KV, and the existing bidirectional RPC
// transport. The result is [n_tokens,n_embd] F32 final hidden on the host.
//
// Boundaries of THIS API:
// - Prefill starts at absolute token position zero; <=64 tokens, <=16/chunk.
// - K/V belongs to this invocation and is reclaimed after all FIFO joins.
// - It is not yet a persistent autoregressive decode session. Do not loop
//   this API for one-token decode and assume it retains the old KV state.
// - Embedding, final RMSNorm, output projection and sampling are separate.
// - No CPU oracle, synthetic inputs, or test-only parity is performed here.
struct llama_hybrid_cycle_forward_metrics {
    int n_tokens = 0;
    int n_chunks = 0;
    int completed_c2p = 0;
    int completed_phone_full = 0;
    int completed_p2c = 0;
    int p2c_ready_received = 0;
    int p2c_partial_received = 0;
    int p2c_reduce_ack = 0;
    bool all_kv_complete = false;
    bool all_host_outputs_published = false;
    bool all_phone_slots_retired = false;
};

bool llama_hybrid_cycle_forward_prefill(
    const llama_model & model,
    ggml_backend_t pc_cpu,
    ggml_backend_t phone_rpc_gpu,
    const std::vector<float> & token_embeddings,
    int n_tokens,
    std::vector<float> & final_hidden,
    std::string & reason,
    llama_hybrid_cycle_forward_metrics * metrics = nullptr,
    int chunk_tokens = 16,
    int max_inflight = 2);
