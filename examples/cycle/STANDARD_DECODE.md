# Standard llama.cpp Context → native Cycle FIFO (opt-in)

This mode routes the **public** `llama_decode()` API into the SAME
`llama_hybrid_cycle_session` already used by `llama-cycle-cli`, with no
duplicate forward implementation and no legacy Meta scheduler execution.

The standard API adapter owns a long-lived 48-layer physical K/V pool and
exposes the final token's model Logits through normal
`llama_get_logits()` / `llama_get_logits_ith(ctx, -1)`.

## Build and run

```powershell
git pull --ff-only origin feat/cycle-runtime-integration-20261023
cmake -S . -B .\build-cycle-test -DLLAMA_BUILD_EXAMPLES=ON
cmake --build .\build-cycle-test --config Debug --target llama-cycle-standard

$env:LLAMA_HYBRID_CYCLE_PC_LAYERS = "4"
$env:LLAMA_HYBRID_CYCLE_PHONE_LAYERS = "2"
$env:LLAMA_HYBRID_CYCLE_STANDARD_DECODE = "1"

.\build-cycle-test\bin\Debug\llama-cycle-standard.exe `
    E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
    10.39.52.114:50053 `
    "Hello, how are you?" 4
```

`llama-cycle-standard` only calls PUBLIC model/context/decode/logits APIs.
The original `llama-cycle-cli` remains available for A/B regression.

## Contract and safety

1. Requires precisely `Qwen3-MoE, 48 layers` with the existing Cycle physical
   CPU/RPC weight layout (4 PC Full + 1 C2P + 2 Phone Full + 1 P2C) × 6.
2. The activation is **explicit**: `LLAMA_HYBRID_CYCLE_STANDARD_DECODE=1`
   and the fixed Cycle placement environment. Other models' `llama_decode`
   implementation is unchanged.
3. Single causal sequence 0, sequential **absolute** token positions.
   A decode batch is limited to 64 tokens. Max total KV is 512 tokens (128 by default).
   Only the last token may request output Logits. Full all-position logits,
   multi-sequence, packed multi-user batches and layer embeddings are rejected.
4. The physical KV state is owned by the Cycle Session, not by ordinary
   `llama_memory`. Therefore `llama_get_memory(ctx)` does NOT expose a
   mutable standard KV-cache object for Cycle contexts yet. Do not call
   standard KV editing/defragmentation/snapshot/rollback APIs expecting them
   to control Cycle KV. To start another sequence in a healthy Cycle
   Context, call `llama_cycle_context_reset(ctx)` and restart at position 0,
   or create a fresh Context. Reset returns false on ordinary Contexts
   and on poisoned Cycle Contexts.
5. This mode does NOT support speculative sampling, backend samplers,
   LoRA hot-swap, noncausal attention, extra Context features or >512 KV.
   The default Context path must not attempt to execute the fixed Cycle
   layout using Meta.
6. Failure after partial physical KV execution is fatal for that context.
   A new Context is required; trying to append to partially-committed KV
   is unsafe.
7. This is an implementation handoff, not a guarantee of CPU numerical
   equivalence. Previously observed MoE numeric differences remain known.

The stage dispatcher, physical KV, GPU/CPU ownership, and two-way RPC remain
exactly those used by `llama-cycle-cli`. The old test harness is preserved
until this public API route passes compilation and real PC/Phone tests.


The public Context accepts `n_ctx=256` or `n_ctx=512` for an explicit
larger physical KV session; `n_ctx=0` still selects the previously verified
128-token default. Values above 512 are rejected rather than silently
clamped. Physical Attention and CPU/Phone GPU precision at >128 positions
still require real-device verification.

## Public API performance counters

`llama_perf_context(ctx)` and `llama_perf_context_print(ctx)` now
report real Cycle elapsed times through the original
`t_p_eval_ms`, `t_eval_ms`, `n_p_eval`, and `n_eval` fields.

When using the public C API directly, optionally classify Prompt and
Decode by calling:

    llama_cycle_context_set_prompt_phase(ctx, true);   // Before Prompt
    // llama_decode(ctx, prompt_chunk) ... perhaps multiple <=64-token calls
    llama_cycle_context_set_prompt_phase(ctx, false);  // Before generation
    // llama_decode(ctx, one_generated_token) ...

The helper returns false for ordinary Contexts or an invalid Cycle
Session and only changes timing classification. It does **not**
affect KV, device ownership, scheduling, or numerical results.
Without the explicit hint, Cycle uses the native fallback:
`n_tokens > 1` or first KV prefix zero is counted as Prompt. This
cannot distinguish the final one-token Prompt chunk of a long Prompt.

Timing covers Embedding, the synchronous physical FIFO, RPC/device
waiting, and output Head. The standard `no_perf` flag is respected.
A finished generation does not need to decode the last sampled token,
so reported eval runs can be one less than emitted tokens. Standard
`graphs reused` does not represent Cycle-specific Attention/FFN caches.

The `llama-cycle-standard` hardware regression now checks that these
public counters are populated and have consistent Prompt / Decode
classification.

## Physical memory breakdown safety

Cycle Contexts have no regular GGML Scheduler. Their public
`llama_get_memory_breakdown(ctx)` is routed to owned, physical
K/V, Attention Scratch, and optional CUDA mirror allocations.
Normal Contexts continue to query their regular Scheduler.
The Cycle totals intentionally exclude transient buffers outside the
session-owned resource pool.
