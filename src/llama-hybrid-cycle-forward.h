#pragma once

#include "ggml-backend.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <vector>

struct llama_model;
class llama_hybrid_cycle_physical_resource_pool;

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
// - Prefill may cover the full <=512-token physical context in one call; <=64/chunk.
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

LLAMA_API bool llama_hybrid_cycle_forward_prefill(
    const llama_model & model,
    ggml_backend_t pc_cpu,
    ggml_backend_t phone_rpc_gpu,
    const std::vector<float> & token_embeddings,
    int n_tokens,
    std::vector<float> & final_hidden,
    std::string & reason,
    llama_hybrid_cycle_forward_metrics * metrics = nullptr,
    int chunk_tokens = 64,
    int max_inflight = 0);

// Persistent physical KV session for Prefill and incremental one-token Decode.
// Every append is a new global FIFO invocation with absolute position and a
// distinct wire epoch; all 48 physical KV tensors survive between calls.
// Physical K/V supports 512 total tokens. Prefill is one global FIFO append
// with 64-token Chunks; no KV shifting or
// multi-sequence manager yet.
class llama_hybrid_cycle_session {
public:
    LLAMA_API llama_hybrid_cycle_session();
    LLAMA_API ~llama_hybrid_cycle_session();
    llama_hybrid_cycle_session(const llama_hybrid_cycle_session &) = delete;
    llama_hybrid_cycle_session & operator=(const llama_hybrid_cycle_session &) = delete;

    LLAMA_API bool prepare(const llama_model & model, ggml_backend_t cpu,
                 ggml_backend_t phone_rpc_gpu, int64_t max_total_tokens,
                 std::string & reason,
                 int chunk_tokens = 64, int max_inflight = 0,
                 ggml_backend_t pc_cuda = nullptr,
                 size_t cuda_weight_budget_bytes = 0);

    // After prompt evaluation, call with ONE next-token embedding.
    // Every successful call commits the absolute physical KV prefix.
    LLAMA_API bool evaluate(const std::vector<float> & embeddings, int tokens,
                  std::vector<float> & final_hidden, std::string & reason,
                  llama_hybrid_cycle_forward_metrics * metrics = nullptr);

    LLAMA_API int64_t committed_tokens() const;
    LLAMA_API bool usable() const;

    // Cycle has no regular GGML scheduler. Summarize actual buffers owned
    // by this persistent Session, grouped by their backend buffer type.
    LLAMA_API void memory_allocations(
        std::map<ggml_backend_buffer_type_t, size_t> & kv,
        std::map<ggml_backend_buffer_type_t, size_t> & scratch,
        std::map<ggml_backend_buffer_type_t, size_t> & cuda_mirrors) const;

    LLAMA_API void reset();

private:
    const llama_model * model_ = nullptr;
    ggml_backend_t cpu_ = nullptr;
    ggml_backend_t phone_ = nullptr;
    ggml_backend_t cuda_ = nullptr;
    size_t cuda_weight_budget_bytes_ = 0;
    std::unique_ptr<llama_hybrid_cycle_physical_resource_pool> resources_;
    int64_t prefix_ = 0;
    int64_t capacity_ = 0;
    uint64_t wire_epoch_ = 0;
    int chunk_tokens_ = 64;
    int max_inflight_ = 0;
    bool poisoned_ = false;
};
