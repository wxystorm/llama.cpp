# Region Ring V1: three gated device workers

This experimental mode is based on the validated `fifo-state` commit `32e3a234`.

## Mode and scope

`LLAMA_HYBRID_REGION_RING_EXEC=fifo-workers` launches **three real host threads**, named CPU, GPU/CUDA, and PHONE. Each thread owns a device-specific READY queue consumer, KV Stage View, logical Scheduler/Graph Result and per-worker status/timing counters. The canonical CPU/GPU schedulers and the main Phone scheduler retain the exact model layer assignment from `fifo-state`.

Tasks are acquired with the condition-variable `ready.wait_take(device)`. Only a completed synchronous Graph/transfer may call `ready.complete(task)`, which publishes the downstream Region's READY and returns the consumed input credit. The two Host F32 output Slots per Region boundary obey `FREE -> WRITING -> READY -> READING -> FREE`.

**Backend execution is intentionally still serialized within the llama_context instance**: each worker acquires `region_ring_backend_execution_mutex` before calling `run_hybrid_stage_block()` and holds it through Graph preparation, Meta RPC, CUDA/CPU compute and backend synchronization. `ready.complete` runs outside the Backend lock so scheduling and credit releases are asynchronous to other host workers. This is a concurrency **correctness** experiment, not true overlapping CPU/CUDA/Phone compute.

## Thread exit and failures

- A failed Graph, invalid final Graph owner, or cancelled task calls `ready.fail(task)`; worker exceptions call `ready.cancel()`. Both notify all waiting Worker threads.
- Already claimed work can finish or exit after cancellation, but no new dependent task can be published after failure.
- The Decode caller always `join()`s every started thread, including partial thread-creation failures, before freeing any stage views, host slots or Graph Results.
- Each task uses local `ggml_status` and local `apply_mctx`; the canonical KV ubatch is still applied once before the Workers start.
- The Phone Worker must return the final Graph Result from the **main** scheduler so the ordinary logits extraction path remains unchanged.

## Build and run

```powershell
git fetch origin
git switch exp/region-ring-workers-gated-20261026
cmake --build build-tensor-v2 --config Debug --target llama-server test-region-ring -j 8
ctest --test-dir build-tensor-v2 -C Debug -R '^test-region-ring$' --output-on-failure

$env:LLAMA_HYBRID_REGION_RING_V1 = '1'
$env:LLAMA_HYBRID_REGION_RING_LAYOUT = 'C7,G4,P5,C7,G4,P5,C8,G3,P5'
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS = '64'
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX = '1'
$env:LLAMA_HYBRID_REGION_RING_EXEC = 'fifo-workers'
# Run the same llama-server args and Qwen3 model/prompt/seed as fifo-state.
```

## Required checks

```text
[REGION_RING_WORKERS_PLAN] workers=3 fifo=PER_DEVICE kv_views=PER_DEVICE window=2 backend=SHARED_GATED graph_overlap=0 status=READY
[REGION_RING_WORKER] device=CPU event=START ...
[REGION_RING_WORKER] device=GPU event=START ...
[REGION_RING_WORKER] device=PHONE event=START ...
[REGION_RING_WORKER_SUM] device=CPU completed=18 ... status=0
[REGION_RING_WORKER_SUM] device=GPU completed=18 ... status=0
[REGION_RING_WORKER_SUM] device=PHONE completed=18 ... status=0
[REGION_RING_WORKERS_SUM] created=3 joined=3 finished=54 completed=54 cpu_tasks=18 gpu_tasks=18 phone_tasks=18 backend_gate=CONTEXT_SHARED graph_overlap=0 final_owner=MAIN status=PASS
[REGION_RING_TASK_STATE_SUM] tasks=54 ... context_outputs_unchanged=1 ... status=PASS
[REGION_RING_OWNERSHIP_SUM] window=2 live_slots=0 finished=54 status=DRAINED
[REGION_RING_SUM] tokens=358 chunks=6 executed=54 complete=1 ... mode=DEVICE_FIFO_GATED_WORKERS status=0
```

For the preceding two-token one-chunk warmups, each Worker completes 3 tasks, not 18.

Use this PowerShell filter on a captured log:

```powershell
$log = '.\fifo-workers.log'
Select-String -Path $log -Pattern '\[REGION_RING_WORKERS', '\[REGION_RING_WORKER', '\[REGION_RING_TASK_STATE_SUM\]', '\[REGION_RING_OWNERSHIP_SUM\]', '\[REGION_RING_SUM\]', '\[REGION_RING_EXIT\]', '^decoded=', '^top\[0\]:', 'event=DEADLOCK', 'event=FAIL', 'event=EXCEPTION' |
    ForEach-Object { $_.Line } | Set-Content '.\workers-check.txt'
```

Compare against the earlier validated `fifo-state`: prefill first Top-1 token 151667, logit 28.507137299; decoded 1 token 198, decoded 2 token 32313. Verify further generated tokens, L15/L31 `KEEP_PHONE_FFN_NEXT_L_OUT`/`COPY_COMPUTED_L_OUT`, and CTest including `window_failure_wakeup=PASS`. A few matching top tokens are not strict full Tensor parity.

## Next step: independently safe Backend instances

Current PC Backend inventory is `Meta(CPU,RPC0)`, `CUDA0`, and `CPU`, and **all three GGML Schedulers point to those same Backend handles**. `std::thread` isolation alone does not isolate backend state.

Before relaxing the global lock: (1) audit and separate Meta CPU/RPC temporary Buffer and dispatch state, (2) establish independent CUDA execution queues and their completion events, (3) give CPU worker stable threadpool state, (4) ensure any shared KV weights are either readonly or guarded by exact Layer/Chunk dependencies, (5) tie async transfer completion to output Slot release, and (6) test numerical parity with both Window=1 and Window=2 cases. A copied Backend pointer is not sufficient.

Rollback without touching the validated baseline: use `fifo-state` on `exp/region-ring-exec-state-20261026`.
