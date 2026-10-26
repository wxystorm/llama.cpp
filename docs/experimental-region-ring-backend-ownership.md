# Region Ring Backend ownership: split and nested Meta audit

## Purpose

**This step does not remove the Backend execution gate.** It records which exact GGML Backend instances are touched by every (Region, Chunk) graph, including the backend that *produces each split input*. That cross-backend source is important because `ggml_backend_sched_prepare_split` may synchronize and copy from a different backend even if it does not execute a node in this split.

Based on `fifo-workers` at `2b7021c3`, new opt-in mode:

```powershell
$env:LLAMA_HYBRID_REGION_RING_EXEC='fifo-backend-audit'
```

The old `fifo-workers`, `fifo-state`, `fifo-owned`, and other modes remain unchanged.

## Newly exposed read-only GGML inspection

- `ggml_backend_sched_get_split_resource_mask(sched, split_id)` returns a uint64 bit mask. Each bit `i` corresponds to the backend returned by `ggml_backend_sched_get_backend(sched, i)`. It ORs the compute backend and all source backends of that split's cross-backend inputs. Bit 63 indicates an unresolved input source. The API does not execute, copy or lock any tensor.
- `ggml_backend_is_meta`, `ggml_backend_meta_n_backends`, and `ggml_backend_meta_simple_backend` expose existing read-only Meta inspection functions through the public GGML Backend header. **Do not free Meta-owned child Backend pointers.**

The actual PC backend inventory has three **outer** handles (by index): `0=Meta(CPU,RPC0)`, `1=CUDA0`, and `2=CPU`. Critically, Meta already constructs its **own** CPU and RPC backend objects inside `ggml_backend_meta_context`: they are not references to the stand-alone CPU Backend. The audit logs the inner handle addresses and whether either aliases an outer handle. Different C++ objects do not imply independent hardware or threadpool, and the RPC client may still share endpoint/device caches and sockets. **Do not clone Meta or RPC blindly**.

## Validation

```powershell
git fetch origin
git switch exp/region-ring-backend-ownership-20261026
cmake --build build-tensor-v2 --config Debug --target llama-server test-region-ring -j 8
ctest --test-dir build-tensor-v2 -C Debug -R '^test-region-ring$' --output-on-failure

$env:LLAMA_HYBRID_REGION_RING_V1='1'
$env:LLAMA_HYBRID_REGION_RING_LAYOUT='C7,G4,P5,C7,G4,P5,C8,G3,P5'
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS='64'
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX='1'
$env:LLAMA_HYBRID_REGION_RING_EXEC='fifo-backend-audit'
# Keep the same PC llama-server / Phone RPC invocation and model/prompt/seed.
```

Filter the output:

```powershell
$log = '.\fifo-backend-audit.log'
Select-String -Path $log -Pattern '\[REGION_RING_META_CHILD\]', '\[REGION_RING_BACKEND_SPLIT\]', '\[REGION_RING_BACKEND_TASK\]', '\[REGION_RING_BACKEND_OWNERSHIP_SUM\]', '\[REGION_RING_WORKERS_SUM\]', '\[REGION_RING_OWNERSHIP_SUM\]', '\[REGION_RING_SUM\]', '^decoded=', '^top\[0\]:', 'event=EXCEPTION', 'event=FAIL' | ForEach-Object { $_.Line } | Set-Content '.\backend-ownership-check.txt'
```

For 358 tokens, expect 54 tasks and three sets of 18 per-device jobs, each with `window=2 live_slots=0 status=DRAINED`, `context_outputs_unchanged=1`, the Phone L15/L31 residual handoff unaffected, and the same first Top-1 token `151667` / logit `28.507137299` as `fifo-workers`. Verify Decode and CTest independently.

### How to interpret the diagnostic masks

Each `[REGION_RING_BACKEND_SPLIT]` shows the compute backend plus its true Scheduler input-source resource mask (printed for chunk 0 of every Region; also printed for any unresolved source). `[REGION_RING_BACKEND_TASK]` logs the union mask of all splits in every task. `[REGION_RING_BACKEND_OWNERSHIP_SUM]` ORs the masks across 18 tasks per worker, with `cross` and `unresolved` counts.

Example *interpretation* (not a predicted result): `resource_mask=0x000...004` means only outer Backend 2 (CPU); `0x000...002` only CUDA0; `0x000...001` only Meta; a mask like `0x000...003` means both Meta and CUDA are touched. `0x800...` means at least one source backend could not be resolved, and any further gate narrowing must conservatively keep the shared gate.

**Audited masks are a necessary input, not proof that concurrency is safe.** They do not describe every allocator, the `llama_context` shared graph-build state, Meta's PC/RPC internal workers, RPC device singleton cache, all asynchronous fences, CPU threadpool changes, or CUDA stream operations. `graph_overlap=0` and the context-wide Backend execution gate remain intentional.

## Next implementation after evidence

After we confirm actual per-Region resource masks: (1) define per-Backend resource ownership and transfer dependency leases, (2) keep graph preparation and any shared allocator activity synchronized until their own lifetime is isolated, (3) replace the broad gate incrementally with locks on the complete resource set, (4) tie each asynchronous device fence and RPC copy to Window=2 output-Slot consumption, (5) compare numerical parity and stability before enabling overlapping device compute.
