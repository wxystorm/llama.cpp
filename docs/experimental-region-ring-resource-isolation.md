# Region Ring: executor resource isolation (experimental)

## Current modes

- `full`, `serial`, `ready`: unchanged correctness reference modes.
- `fifo`: original single-thread, device-specific READY queue simulation.
- `fifo-isolated`: opt-in single-thread simulation with **separate CPU and CUDA GGML schedulers and graph results**. Phone uses the canonical `sched` / `gf_res_prev` so the final logits extraction continues to use the correct scheduler.

No worker thread or asynchronous Device FIFO compute is enabled by this change. In particular **GGML backend objects, the KV memory context, threadpools, `llama_context::n_outputs`, and Meta/RPC backend state are shared**. Independent GGML scheduler handles must not be confused with fully independent compute execution.

## Resource ownership

| Logical device | GGML scheduler | Graph Result | Notes |
| --- | --- | --- | --- |
| CPU | auxiliary persistent scheduler 0 | auxiliary result 0 | Lazily allocated |
| GPU | auxiliary persistent scheduler 1 | auxiliary result 1 | Lazily allocated |
| Phone | canonical `sched` | canonical `gf_res_prev` | Required for caller's final result/logits path |

Each auxiliary scheduler is built with the **same** `backend_ptrs` / `backend_buft` as the canonical scheduler. This preserves the existing layer assignment and Meta routing while testing Graph Result / GGML allocator isolation. Auxiliary resources are invalidated when the primary scheduler is re-reserved and their graph-reuse caches are reset after a memory module update.

Region outputs still use the existing synchronous host hidden-state buffers. Phone final `l_out` residual handling is unchanged.

## Validation on Windows

```powershell
# On this experimental branch, use the existing build directory.
cmake --build build-tensor-v2 --config Debug --target llama-server test-region-ring -j 8
ctest --test-dir build-tensor-v2 -C Debug -R "^test-region-ring$" --output-on-failure

# Keep all existing model / RPC / context parameters unchanged.
$env:LLAMA_HYBRID_REGION_RING_EXEC="fifo-isolated"
# Launch the same llama-server invocation used for the correct fifo baseline.
```

For a 358-token / 64-token-chunk prefill, verify:

- `[REGION_RING_RESOURCE] ... status=ISOLATED_SCHED_ONLY`, with three distinct scheduler and graph-result addresses, and `backend=SHARED kv=SHARED threads=1`.
- `[REGION_RING_SUM] ... executed=54 complete=1 mode=DEVICE_FIFO_ISOLATED_SIMULATION status=0`.
- Identical top logits and generated text compared with the proven `fifo` baseline under the same prompt and settings.
- Successful `[REGION_RING_EXIT]` handoffs at layers 15 and 31 (`KEEP_PHONE_FFN_NEXT_L_OUT`, then `COPY_COMPUTED_L_OUT`).
- No RPC disconnect, allocation failure, or per-chunk KV order violation.

If memory pressure prevents allocating the auxiliary schedulers, switch back to `fifo`; do not reduce correctness guards or alter Phone `l_out` handling to make the experiment fit.

## Still required before starting three threads

1. Remove or synchronize shared mutable `n_outputs`, `set_stage_range`, `mctx->apply`, and CPU threadpool changes.
2. Prove separate, safe backend/Meta/RPC contexts, GPU stream and device Buffer ownership. Schedulers alone do not make shared backend APIs thread-safe.
3. Move host hidden state to credit/slot ownership and publish READY only after compute and transfers complete.
4. Repeat numerical and output parity tests, then profile actual CPU/CUDA/Phone overlapping wall time.
