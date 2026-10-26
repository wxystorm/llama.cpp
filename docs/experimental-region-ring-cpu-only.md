# Region Ring: strict CPU Backend placement experiment

This branch continues from the previously tested `fifo-backend-audit` commit `8bdc38f6`, without changing that reference branch or its execution mode.

## Why this change

On the tested 358-token / six-chunk Prefill, the CPU worker reported `resource_mask=0x6` (outer CPU + CUDA). More importantly, CPU Regions 0 and 3 ran 22 GGML splits each, and CPU Region 6 ran 25: many of those splits had `compute=CUDA0` even though their owning Ring regions were CPU.

Source analysis found that the model loader assigns PC_ONLY CPU Region layer weights to `CPU_DIRECT`, but the previous CPU worker Scheduler was created from the full `[Meta, CUDA0, CPU]` backend list and with `cparams.op_offload`. Therefore GGML could move some CPU-weight operations to CUDA and expand those assignments onto nearby graph nodes. This contradicts the intended independent CPU compute worker and prevents correctly narrowing the Backend lock.

## Opt-in CPU-only Scheduler

Set:

```powershell
$env:LLAMA_HYBRID_REGION_RING_EXEC='fifo-cpu-only'
```

The CPU worker Scheduler is now constructed with **only the canonical outer CPU backend** and its matching buffer type, `n_backends=1`, `op_offload=false`. A new `REGION_RING_CPU_LAYOUT` check first confirms that all CPU Region layers really have model weight placement `CPU_DIRECT`; a mismatch rejects the mode rather than copying weights silently.

The CUDA worker retains its previous full Scheduler, the Phone worker still uses the canonical Scheduler and Meta(CPU,RPC0), and all the previous KV Stage Views, Window=2 output slots, three real host Worker threads, and final Phone logits ownership are retained.

**The shared Backend graph execution lock is intentionally unchanged.** This experiment only enforces CPU compute placement. It does not yet allow overlapping CPU/CUDA/Phone graph execution.

## Backend mask interpretation

The split resource mask API returns indices local to each Scheduler. With a new CPU-only Scheduler, bit 0 would mean CPU instead of Meta. In this mode the audit explicitly converts every local split mask back to the canonical outer `[Meta, CUDA0, CPU]` order before aggregating or printing. All logged masks have `scope=OUTER`.

Expected first-chunk CPU diagnostics (example, not yet measured):

```text
[REGION_RING_CPU_LAYOUT] layers=22 device=CPU_DIRECT status=READY
[REGION_RING_CPU_BACKEND] ... backends=1 op_offload=0 mode=CPU_ONLY status=READY
[REGION_RING_CPU_PLACEMENT] region=0 chunk=0 splits=1 non_cpu_splits=0 status=PASS
[REGION_RING_CPU_PLACEMENT] region=3 chunk=0 splits=1 non_cpu_splits=0 status=PASS
[REGION_RING_CPU_PLACEMENT] region=6 chunk=0 splits=1 non_cpu_splits=0 status=PASS
[REGION_RING_BACKEND_OWNERSHIP_SUM] cpu_mask=0x0000000000000004 gpu_mask=0x0000000000000006 phone_mask=0x0000000000000005 ...
```

For the `C7,G4,P5,C7,G4,P5,C8,G3,P5` layout, there are indeed 22 CPU Region layers (7 + 7 + 8). The CPU Region graph may contain more than one split, but **every CPU split must compute on CPU**; the runtime checks that condition for every task and fails instead of silently proceeding.

## Build and validation

```powershell
git fetch origin
git switch exp/region-ring-cpu-only-20261026
cmake --build build-tensor-v2 --config Debug --target llama-server test-region-ring -j 8
ctest --test-dir build-tensor-v2 -C Debug -R '^test-region-ring$' --output-on-failure

$env:LLAMA_HYBRID_REGION_RING_V1='1'
$env:LLAMA_HYBRID_REGION_RING_LAYOUT='C7,G4,P5,C7,G4,P5,C8,G3,P5'
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS='64'
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX='1'
$env:LLAMA_HYBRID_REGION_RING_EXEC='fifo-cpu-only'
# Keep the exact same Qwen3 GGUF, Phone RPC server, Prompt and Seed.
```

Collect the important output:

```powershell
$log = '.\fifo-cpu-only.log'
Select-String -Path $log -Pattern '\[REGION_RING_CPU_LAYOUT\]', '\[REGION_RING_CPU_BACKEND\]', '\[REGION_RING_CPU_PLACEMENT\]', '\[REGION_RING_BACKEND_SPLIT\]', '\[REGION_RING_BACKEND_OWNERSHIP_SUM\]', '\[REGION_RING_WORKERS_SUM\]', '\[REGION_RING_OWNERSHIP_SUM\]', '\[REGION_RING_SUM\]', '^decoded=', '^top\[0\]:', 'status=FAIL', 'event=EXCEPTION' |
    ForEach-Object { $_.Line } | Set-Content '.\cpu-placement-check.txt'
```

Expected for 358 tokens: 54 tasks completed, 18 on each worker, `window=2 live_slots=0 status=DRAINED`, `context_outputs_unchanged=1`, Phone L15/L31 output handoff unaffected, and the previous deterministic first Top-1 token `151667`, logit `28.507137299` (if all settings match). Compare further decoded tokens as well; Top-1 equality does not establish full tensor parity.

## Fallback and next step

If the CPU Scheduler cannot build/allocate a CPU graph, this mode intentionally stops. Return to the known-good `fifo-backend-audit` on `exp/region-ring-backend-ownership-20261026` or `fifo-workers` on `exp/region-ring-workers-gated-20261026`; do not add CUDA as a silent fallback to the strict CPU worker.

After CPU-only placement passes, the cross-backend masks for CUDA and Phone must be separated into compute locks vs. input-transfer and buffer-credit synchronization. CPU-only placement alone does not prove that Meta/RPC and CUDA resources can run concurrently.
