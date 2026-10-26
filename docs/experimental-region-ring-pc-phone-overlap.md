# PC CPU+CUDA and Phone Meta/RPC concurrent Region Ring (experimental)

## Source of truth and purpose

This branch is based on the successfully run `fifo-backend-audit` reference (`8bdc38f6e059eababa6533b75a2d99b462693b96`), not the slower `fifo-cpu-only` branch. The tested PC CPU Regions really used mixed CPU+CUDA splits (`resource_mask=0x6`) and the Phone Regions used one Meta(CPU,RPC0) split with a CPU Host input source (`resource_mask=0x5`).

The goal is not three physically isolated devices. **PC is one CPU+CUDA execution domain; Phone Meta/RPC is another.** The two PC device FIFO workers continue sharing their execution gate, with GGML automatic CUDA offload enabled. Phone only holds that gate for Graph preparation and input handoff, then executes its single Meta split without the PC gate. This does **not** imply Meta runs exclusively on the phone: the Meta context contains its own separate PC CPU child backend, which may contend for physical PC CPU cores.

## Opt-in execution

```powershell
git fetch origin
git switch exp/region-ring-pc-phone-overlap-20261026
cmake --build build-tensor-v2 --config Debug --target llama-server test-region-ring -j 8
ctest --test-dir build-tensor-v2 -C Debug -R '^test-region-ring$' --output-on-failure

$env:LLAMA_HYBRID_REGION_RING_V1='1'
$env:LLAMA_HYBRID_REGION_RING_LAYOUT='C7,G4,P5,C7,G4,P5,C8,G3,P5'
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS='64'
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX='1'
$env:LLAMA_HYBRID_REGION_RING_EXEC='fifo-pc-phone'
# Run the same Qwen3 GGUF, RPC phone server, prompt, seed and llama-server or llama-cli flags as the audit baseline.
```

Only the new opt-in mode changes execution. `fifo-backend-audit`, `fifo-workers` and `fifo-state` remain available and continue to use a single shared graph gate.

## Exact safety boundary

1. Both PC workers receive independent Schedulers that register only **CUDA0 + outer CPU** (`n_backends=2`, `op_offload=cparams.op_offload`). They never call Meta via the Scheduler. They still share one PC gate because they can both submit CUDA work.
2. Phone uses the original canonical Scheduler and Graph Result. The worker holds the PC gate during `prepare_ubatch()` (Graph build, buffer allocation, input setup), validates that there is **exactly one compute split** on outer `Meta(CPU,RPC0)` and validates the source mask includes only Meta and outer CPU. Any unexpected CUDA split/unknown source fails closed. `ggml_backend_sched_prepare_split()` is executed under the PC gate, including Host input transfer staging.
3. Only after successful input preparation does the Phone worker `unlock()` the PC gate and call `ggml_backend_sched_compute_split()` for its one Meta split. It then synchronizes **the Meta Backend only**, copies its output to the dedicated Window=2 Host Slot, and finally calls `ready.complete()` to release the upstream Slot and publish downstream READY.
4. The ordinary `graph_compute_range()` configures *all* Backends' threads each time it runs. In this mode CPU threadpool / set_n_threads are applied once before any workers are launched. PC workers call `ggml_backend_sched_compute_range()` directly; the Phone worker calls `ggml_backend_sched_compute_split()` directly. This prevents a PC worker changing Meta's threads during remote work.
5. PC intermediate output synchronization still covers only the two PC Scheduler handles (CUDA0 and CPU); Phone output synchronization does not call `ggml_backend_sched_synchronize()` on the original 3-backend Scheduler, which would touch running PC resources.
6. Existing per-device KV Stage Views, Layer/Chunk dependencies, Window=2 Host Slot credits, Phone L15/L31 l_out fixes, all 3 Worker joins and final MAIN Graph Result ownership are kept.

These are **resource-scoped execution barriers**, not a blanket guarantee about every platform/driver, shared system memory or RPC bug. This stage still requires Windows+phone testing, full token parity comparison, repeatability and failure-path testing. If unexpected split ownership occurs the mode rejects the task; do not weaken that guard to force a run.

## Diagnostic output to validate

```text
[REGION_RING_PC_SCHED] worker=CPU backends=2 first=CUDA0 last=CPU offload=1 meta_registered=0 status=READY
[REGION_RING_PC_SCHED] worker=GPU backends=2 first=CUDA0 last=CPU offload=1 meta_registered=0 status=READY
[REGION_RING_PC_PHONE_THREADS] threads=... backend_config=ONCE_BEFORE_WORKERS status=READY
[REGION_RING_WORKERS_PLAN] workers=3 ... backend=PC_GATE_PLUS_META graph_overlap=MEASURED status=READY
[REGION_RING_PC_PHONE_SUM] cpu_intervals=18 gpu_intervals=18 phone_intervals=18 overlap_ms=... pc_gate=CPU_CUDA phone_compute=DETACHED ... status=PASS
[REGION_RING_WORKERS_SUM] created=3 joined=3 finished=54 completed=54 cpu_tasks=18 gpu_tasks=18 phone_tasks=18 ... status=PASS
[REGION_RING_TASK_STATE_SUM] tasks=54 ... context_outputs_unchanged=1 ... status=PASS
[REGION_RING_OWNERSHIP_SUM] window=2 live_slots=0 finished=54 status=DRAINED
[REGION_RING_SUM] tokens=358 chunks=6 executed=54 complete=1 ... mode=DEVICE_FIFO_PC_PHONE_OVERLAP status=0
```

`overlap_ms` is calculated from the intersections of PC task execution intervals with the intervals between Phone's release of the PC gate and completion of its Meta synchronization/output copy. It is **observed wall-clock overlap**, not necessarily GPU kernel overlap or a measured performance gain. Zero means the current task/credit schedule yielded no overlap; nonzero proves time intervals coincide, not perfect numeric parity.

Use a compact filter:

```powershell
$log = '.\fifo-pc-phone.log'
Select-String -Path $log -Pattern '\[REGION_RING_PC_SCHED\]', '\[REGION_RING_PC_PHONE', '\[REGION_RING_WORKERS_SUM\]', '\[REGION_RING_BACKEND_OWNERSHIP_SUM\]', '\[REGION_RING_TASK_STATE_SUM\]', '\[REGION_RING_OWNERSHIP_SUM\]', '\[REGION_RING_SUM\]', '\[REGION_RING_EXIT\]', '^decoded=', '^top\[0\]:', 'event=REJECT', 'event=FAIL', 'event=EXCEPTION' |
    ForEach-Object { $_.Line } | Set-Content '.\pc-phone-check.txt'
```

## A/B procedure and rollback

Use identical model/RPC server/prompt/seed flags. First verify `fifo-backend-audit` for 358 tokens (previous run `wall_ms=5581.101`, Top-1 token `151667`, logit `28.507137299`), then `fifo-pc-phone`. Prefill timing can fluctuate; perform repeated runs after warmup and compare medians. Check all Decode logits and boundary correctness, not just Top-1. Test both 2-token warmups and 358-token Prefill with 6 chunks. For a serious run, enable existing parity/trace diagnostics as necessary and compare exact numerics / per-stage drift.

On any failed topology check, graph_compute error, slot leak, stall, or unexpected logit drift, use the known-good mode/branch; do not silently fall back. Old baseline:

```powershell
git switch exp/region-ring-backend-ownership-20261026
$env:LLAMA_HYBRID_REGION_RING_EXEC='fifo-backend-audit'
```

## Next changes after this experiment passes

- Break down time in Phone input staging vs Meta execution vs output copy, to move more transfer work outside the PC gate without sacrificing Host Slot lifetime.
- Add explicit transfer completion events and resource-specific Graph build locks, keeping one PC CUDA execution queue and an independent Phone RPC compute queue.
- Keep Metal/CUDA/CPU shader/kernels, backend threadpool lifecycle and driver-specific assumptions behind explicit guards.
