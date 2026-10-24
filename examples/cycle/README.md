# Physical Cycle integration

**Official `llama-cli` entrypoint:** see
[OFFICIAL_CLI.md](OFFICIAL_CLI.md). Explicitly enabling
`LLAMA_HYBRID_CYCLE_STANDARD_DECODE=1` routes the official executable
through the public Context API and real five-queue Cycle, without
starting its normal HTTP server. This is currently single-prompt
completion, not full interactive server chat.

## Earlier native prefill runner

`llama-cycle-prefill` is the first **non-test** host entry point for the
real 48-layer Qwen3-30B-A3B Cycle. It links the reusable physical executor in
`src/llama-hybrid-cycle-forward.cpp` plus real model embedding/LM-head graph
helpers in `src/llama-hybrid-cycle-model-io.cpp`.

It runs tokenizer -> real token embedding -> one 48-layer global five-worker
FIFO -> actual output RMSNorm and output projection -> first greedy token.

This program does **not** use the CPU oracle, test mock Phone stages, artificial
Hidden States, or single-period Meta re-execution. It intentionally keeps the
six 4-PC/1-C2P/2-Phone/1-P2C physical regions from the verified test.

## Build and run on Windows

Enable `LLAMA_BUILD_EXAMPLES=ON` and the existing RPC backend:

```powershell
cmake -S . -B .\build-cycle-test -DLLAMA_BUILD_EXAMPLES=ON
cmake --build .\build-cycle-test --config Debug --target llama-cycle-prefill
$env:LLAMA_HYBRID_CYCLE_PC_LAYERS = "4"
$env:LLAMA_HYBRID_CYCLE_PHONE_LAYERS = "2"
.\build-cycle-test\bin\Debug\llama-cycle-prefill.exe `
    E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
    10.39.52.114:50053 `
    "Explain how GPUs and NPUs cooperate."
```

The model must already load with physical CPU/Phone tensor placement matching
the layout. Unsupported model, token count, backend, weight placement and
partial KV/ACK states fail closed. The input is 1..64 tokens; the first version
returns **one next-token logit decision**.

## Boundaries still to integrate

- Long-lived incremental generation is now available through the
  separate `llama-cycle-cli` target; see [DECODE.md](DECODE.md).
  The one-shot `llama_hybrid_cycle_forward_prefill` remains intentionally
  a one-call API and must not be repeatedly invoked as decode.
- An **experimental, explicit** standard `llama_decode()` and
  `llama_get_logits()` adapter is now provided; see
  [STANDARD_DECODE.md](STANDARD_DECODE.md). The Context uses the very same
  physical Cycle forward and owns its own physical KV. Standard
  `llama_memory` editing/snapshot APIs are not yet supported for Cycle KV.
- Larger context windows, multi-user batches, checkpoint/rollback and
  sampler selection remain unimplemented.
- The previously reported CPU/GPU numeric parity differences are not fixed
  or hidden; this production migration does not relax numeric tolerances.

The existing `test-hybrid-cycle-real-pc-ring` remains the separate structural
and numerical regression harness. Normal `llama_decode()` remains unchanged
unless both Cycle placement and the explicit standard adapter opt-in are set.
