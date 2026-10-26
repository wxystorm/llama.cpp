# Region Ring: explicit task Graph state (experimental)

Branch: `exp/region-ring-exec-state-20261026`, based on the validated `fifo-owned` branch at `ab6a5aeb`.

## New opt-in mode

`LLAMA_HYBRID_REGION_RING_EXEC=fifo-state`

The original `ready`, `serial`, `full`, `fifo`, `fifo-isolated` and `fifo-owned` execution paths retain their previous behavior. The new mode inherits the three KV Stage Views, Window=2 host Slot credit system, separated CPU/CUDA GGML schedulers/results and the existing broad Backend execution mutex from `fifo-owned`.

## What changed

- Previously `run_hybrid_stage_block()` assigned `llama_context::n_outputs = block_outputs` for each `(Region, Chunk)` task. `graph_params()` then implicitly read that shared value.
- In `fifo-state` only, `block_outputs` is passed explicitly through `run_hybrid_stage_block()` -> `prepare_ubatch()` -> `graph_params()` and becomes `llm_graph_params::n_outputs` for that task. No stage task needs to mutate `llama_context::n_outputs`.
- `llm_graph_params::allow_reuse()` already includes `n_outputs`; explicit output counts therefore continue to invalidate Graph reuse when necessary.
- `n_outputs` is checked after the Region Ring execution and must equal the original value supplied by the parent Decode loop. All other modes still use their original shared-field path.
- The diagnostics enumerate the real `ggml_backend_t` handles used by all three schedulers, including names, addresses and device types. These handles remain shared. The broad Backend gate must not be removed merely because `n_outputs` is now task-local.
- `ggml_backend_cpu_set_threadpool`, `ggml_backend_set_n_threads`, `model.build_graph` and the Meta/RPC execution path still need thread-safety/resource-ownership analysis before actual parallel Graph execution.

## Windows verification

```powershell
git fetch origin
git switch exp/region-ring-exec-state-20261026
cmake --build build-tensor-v2 --config Debug --target llama-server test-region-ring -j 8
ctest --test-dir build-tensor-v2 -C Debug -R '^test-region-ring$' --output-on-failure

$env:LLAMA_HYBRID_REGION_RING_V1 = '1'
$env:LLAMA_HYBRID_REGION_RING_LAYOUT = 'C7,G4,P5,C7,G4,P5,C8,G3,P5'
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS = '64'
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX = '1'
$env:LLAMA_HYBRID_REGION_RING_EXEC = 'fifo-state'
# Use the same Qwen3 model, RPC server, prompt and seed as the validated fifo-owned run.
```

Expected for the 358-token Prompt and 64-token chunks:

```text
[REGION_RING_BACKEND] index=... handle=... name=... device_type=... schedulers=3 state=SHARED
[REGION_RING_BACKEND_SUM] ... backend=SHARED_GATED ... graph_parallelism=DISABLED status=AUDITED
[REGION_RING_TASK_STATE] outputs=TASK_LOCAL context_outputs=READ_ONLY kv_views=3 graph_threads=1 status=READY
[REGION_RING_OWNERSHIP_SUM] window=2 live_slots=0 finished=54 status=DRAINED
[REGION_RING_TASK_STATE_SUM] tasks=54 task_outputs=EXPLICIT context_outputs_unchanged=1 final_graph=MAIN backend=SHARED_GATED status=PASS
[REGION_RING_SUM] tokens=358 ... executed=54 complete=1 ... mode=DEVICE_FIFO_TASK_STATE_SIMULATION status=0
```

Compare `decoded=0` Top-10 Logits against the earlier `fifo-owned` result (Top-1 token 151667, logit 28.507137299) and check subsequent Decode tokens and final output. Confirm L15/L31 have `KEEP_PHONE_FFN_NEXT_L_OUT` and `COPY_COMPUTED_L_OUT` with `status=OK`. Do not infer numerical parity from `complete=1` alone.

## Work still required before concurrent CPU/CUDA/Phone compute

1. Any remaining mutable `llama_context` fields called by graph preparation, including graph reuse counters and CPU threadpool configuration, must be worker-owned or safely synchronized.
2. Actual CPU/CUDA/Meta-RPC Backend state and execution queues must be independently safe; three GGML Scheduler pointers do not mean three independently owned Backends.
3. Start real Worker threads under the existing Backend gate first, verify correctness, then identify which Backend operations can overlap and narrow the gate only where ownership permits. This mode does not start Worker threads.
4. Keep Slot Credit `window=2` and synchronous transfer completion semantics until real async fence/event ownership is implemented and tested.

Rollback: use the previously validated `LLAMA_HYBRID_REGION_RING_EXEC=fifo-owned` on the previous branch to compare numerical output.
