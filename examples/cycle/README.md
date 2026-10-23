# Native physical Cycle prefill

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

- A long-lived physical-KV evaluation session for incremental decode. The
  one-shot prefill API must NOT be called repeatedly for generation.
- Standard `llama_context::decode()` and `llama_get_logits()` interoperability:
  the native Cycle output is currently owned by this standalone CLI rather
  than the regular llama_context memory manager.
- Larger context windows, multi-user batches, checkpoint/rollback and
  sampler selection remain unimplemented.
- The previously reported CPU/GPU numeric parity differences are not fixed
  or hidden; this production migration does not relax numeric tolerances.

The existing `test-hybrid-cycle-real-pc-ring` remains the separate structural
and numerical regression harness. Neither it nor `llama_decode()` is silently
replaced by the native runner.
