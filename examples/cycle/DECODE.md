## Incremental 48-layer native Cycle decode

Build `llama-cycle-cli` after enabling `LLAMA_BUILD_EXAMPLES=ON` and
the `ggml-rpc` backend:

```powershell
cmake -S . -B .\build-cycle-test -DLLAMA_BUILD_EXAMPLES=ON
cmake --build .\build-cycle-test --config Debug --target llama-cycle-cli
$env:LLAMA_HYBRID_CYCLE_PC_LAYERS = "4"
$env:LLAMA_HYBRID_CYCLE_PHONE_LAYERS = "2"
.\build-cycle-test\bin\Debug\llama-cycle-cli.exe `
    E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
    10.39.52.114:50053 `
    "Explain how GPUs and NPUs cooperate." 8
```

The long-lived `llama_hybrid_cycle_session` owns all 48 physical KV
buffers. Each evaluation runs one physical six-period FIFO with absolute
positions and a new wire epoch, then advances KV only on success. A failure
poisons the session: a partially committed physical KV cannot be retried.

Current constraints: Qwen3-MoE 48-layer layout 4/2, 16-token Chunk,
1..64 token prompt, 128-token total causal KV, one sequence, greedy decoding.
No CPU oracle/model duplicate is loaded. The normal `llama_context` API,
multi-sequence KV management, and longer contexts remain separate integration
tasks. Numerical CPU/GPU parity differences remain unresolved and are not
suppressed by these changes.
