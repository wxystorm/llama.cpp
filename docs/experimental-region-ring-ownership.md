# Region Ring stage ownership: experimental

## Opt-in mode

`LLAMA_HYBRID_REGION_RING_EXEC=fifo-owned`

All previous full/serial/ready/fifo/fifo-isolated modes are unchanged. This remains single-threaded GGML execution.

## Ownership

- The canonical KV context applies the ubatch once before any task. Three device-specific `make_stage_view()` copies hold their own mutable stage-range metadata; each refuses `apply()` to avoid applying the KV ubatch twice. KV tensors themselves remain shared.
- CPU and CUDA use the auxiliary schedulers and graph results from fifo-isolated; Phone uses the canonical graph and scheduler to preserve final logits extraction. Every `run_hybrid_stage_block()` holds `region_ring_backend_execution_mutex` because actual backend handles, Meta/RPC state, and `n_outputs` are still shared.
- Each nonfinal Region edge owns two fixed host F32 output slots (`window=2`). Slot states transition `FREE -> WRITING -> READY -> READING -> FREE`. A slot is freed only after its downstream consumer synchronously finishes. Credit-blocked jobs remain queued without blocking jobs from other Regions.
- All Phone `l_out` residual/return code is unchanged. `run_hybrid_stage_block` still synchronizes before `ready.complete()`.

## Windows validation

```powershell
git fetch origin
git switch exp/region-ring-ownership-v1-20261026
cmake --build build-tensor-v2 --config Debug --target llama-server test-region-ring -j 8
ctest --test-dir build-tensor-v2 -C Debug -R "^test-region-ring$" --output-on-failure
$env:LLAMA_HYBRID_REGION_RING_V1="1"
$env:LLAMA_HYBRID_REGION_RING_LAYOUT="C7,G4,P5,C7,G4,P5,C8,G3,P5"
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS="64"
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX="1"
$env:LLAMA_HYBRID_REGION_RING_EXEC="fifo-owned"
# Run the same model and prompt as the validated fifo-isolated mode.
```

Expected logs for 358 tokens, six chunks, nine Regions:

```text
[REGION_RING_OWNERSHIP] window=2 slots_per_edge=2 kv_views=3 kv_storage=SHARED backend=SHARED_GATED graph_threads=1 status=READY
[REGION_RING_OWNERSHIP_SUM] window=2 live_slots=0 finished=54 status=DRAINED
[REGION_RING_SUM] tokens=358 chunk_tokens=64 regions=9 chunks=6 executed=54 complete=1 ... mode=DEVICE_FIFO_OWNED_SIMULATION status=0
```

Check L15/L31 `KEEP_PHONE_FFN_NEXT_L_OUT` then `COPY_COMPUTED_L_OUT`, first-token Top-10 logits, multi-token Decode and repeated prompts against `fifo-isolated`. CTest exercises scheduler credits, queue wakeup, slot release and concurrent **mock** workers; it is not a real GPU/RPC concurrency test.

## Not yet solved

Backend pointers are shared and the broad lock prevents any concurrent GGML compute. To achieve true overlap, localize `llama_context` graph-build state, provide safely isolated Backend/Meta-RPC contexts or fine-grained resource-scoped locks and async completions, and expand Slot ownership to transport events. `window=2` is only host buffer backpressure, not an async network engine.

If this experimental mode fails, use `fifo-isolated` as the correctness fallback without altering Phone residual handling.
