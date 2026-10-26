# Region Ring plan-driven runtime: general topology (experimental)

Branch: `feat/region-ring-plan-runtime-v1-20261026`

**Status: Implementation committed, unit tests and actual Windows CUDA +
Phone inference NOT yet verified on this revision.** Keep the previously
tested fixed-Ring branch as the fallback. This is a testable implementation,
not a claim that any predicted performance improvement has materialized.

## Supported automatic topology in this experimental mode

`LLAMA_HYBRID_REGION_RING_PLANNER=apply-topology` chooses a model-owned
plan before weight loading, using measured CPU/CUDA/Phone performance,
transfer timings and the existing memory/DAG simulator.

- 2-4 complete CPU -> GPU -> Phone cycles
- each Phone span has 2-8 layers
- an optional CPU or CPU+CUDA suffix, **required when applying** a
  general plan, so the final logits owner is the PC
- 48-layer Qwen3-MoE profile, Chunk=64, Window=2, PC CPU/CUDA gate
- no dynamic replanning or live weight migration
- minimum predicted gain defaults to 2%, controlled by
  `LLAMA_HYBRID_REGION_RING_PLAN_MIN_GAIN_PCT`
- if the Planner lacks enough measured data or doesn't beat the
  threshold, keep the original fixed layout without changing weights

Example *candidate only*:
`C6,G4,P8,C6,G4,P8,C8,G4`. It has 8 stages and for 358
prompt tokens runs 6 chunks -> 48 stage-chunk tasks. Candidate scores
are model estimates, not hardware measurements.

`topology-preview` still performs a non-applying search.
`apply-safe` still retains the original fixed Phone P5 and 9-stage
restrictions. No generic changes are applied to either old mode.

## Implementation paths

- `src/llama-region-ring.h`: variable-topology shape validation
- `src/llama-region-ring-planner.h`: executable topology search
  with an optional required PC tail
- `src/llama.cpp`: mode validation, measured Profile and immutable
  pre-weight-load plan decision
- `src/llama-model.cpp`: exact per-layer CPU/CUDA/Phone weights
- `src/llama-context.cpp`: same Stage layout, Phone mask registered
  with Meta, and main-scheduler CPU/CUDA final logits ownership
- `ggml/src/ggml-backend-meta.cpp`: dynamic contiguous Phone fusion,
  exact terminal `l_out` bridge checks, per-span auto-READY, graph
  capacity guard and fail-closed incomplete prefill fusion
- `ggml/src/ggml-rpc/ggml-rpc.cpp`: PC client and Phone server
  validate registered 2..8 layer Phone graphs before dispatching
  them to Phone MAIN

## CRITICAL: PC and Phone must run the same new RPC build

RPC protocol patch was bumped from **17 to 18**, to enforce matched
client and server binaries after changing Phone graph validation.
The HELLO handshake rejects a mismatch. Update the PC and Termux
`ggml-rpc-server` from this branch together; never mix an old Phone
server (patch 17) with a new PC client (patch 18).

### Windows PowerShell build and unit tests

```powershell
git fetch origin
git switch feat/region-ring-plan-runtime-v1-20261026
git pull --ff-only
cmake --build build-tensor-v2 --config Release --target llama-cli test-region-ring test-region-ring-planner -j 8
ctest --test-dir build-tensor-v2 -C Release -R "^test-region-ring(-planner)?$" --output-on-failure
```

If the test targets don't exist, reconfigure your existing build with
`LLAMA_BUILD_TESTS=ON` while retaining its working CUDA and RPC options.

### Termux Phone server

In the Phone source directory, fetch/switch/pull the **same branch**,
then reuse the original working Termux OpenCL/RPC build configuration:

```sh
git fetch origin
git switch feat/region-ring-plan-runtime-v1-20261026
git pull --ff-only
cmake --build build-region-v1 --target ggml-rpc-server -j 8
./build-region-v1/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -t 8 -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

Do not reuse an already-running patch-17 Phone process. Restart after
rebuilding. Keep your established server flags and model path if they
differ from this example.

### Apply topology on PC

Retain the working `--split-mode tensor`, RPC address, model path,
context, output, and sampling arguments. **Do not set
`--hybrid-auto`**, and do not manually set
`LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT`.

```powershell
$env:LLAMA_HYBRID_REGION_RING_V1 = "1"
$env:LLAMA_HYBRID_REGION_RING_EXEC = "fifo-pc-phone"
$env:LLAMA_HYBRID_REGION_RING_LAYOUT = "C7,G4,P5,C7,G4,P5,C8,G3,P5"
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS = "64"
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX = "1"
$env:GGML_META_RING_PHONE_FUSION = "1"
$env:GGML_META_RING_PHONE_AUTOSTART = "1"
$env:LLAMA_HYBRID_REGION_RING_PLANNER = "apply-topology"
$env:LLAMA_HYBRID_REGION_RING_PLAN_TOKENS = "358"
$env:LLAMA_HYBRID_REGION_RING_PLAN_MIN_GAIN_PCT = "2"
Remove-Item Env:LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT -ErrorAction SilentlyContinue
```

`LLAMA_HYBRID_REGION_RING_LAYOUT` here is only the original fallback
baseline; the new Planner's selected model-owned Plan overrides it.

## Independent Region Ring device Profile (2026-10-27)

New `llama_hybrid_region_ring_collect_profile()` measures and caches
CPU/CUDA/Phone compute, attention, available memory, and PC/Phone
transfer curves **without** running the legacy
`llama_hybrid_enumerate_feasible_plans` /
`llama_hybrid_score_plan` memory-and-score filter.

The shared hardware probe implementation is
`llama_hybrid_collect_device_profile()`. The separate entrypoints are:

- Legacy `--hybrid-auto`: collect hardware measurements, then enumerate,
  score, memory-filter and publish the legacy placement Plan.
- Region `dry-run`, `topology-preview`, `apply-safe`,
  `apply-topology` (without `--hybrid-auto`): collect hardware
  measurements **only**, cache the Profile, and let the Region planner
  independently search/simulate/validate its own topology.

This fixes a missed mode check: `apply-topology` was not among the
modes that cached Profile even though it had fully measured devices.
A legacy `coarse_scoreable=0` or `no scoreable hybrid plans` can no
longer block `apply-topology` by withholding the shared Profile.

Expected new logging at load time:

```text
[REGION_RING_PLAN_PROFILE] phase=MEASURE_BEGIN source=INDEPENDENT_DEVICE_PROBES mode=APPLY_TOPOLOGY legacy_candidates=SKIPPED
[REGION_RING_PLAN_PROFILE] phase=MEASURED status=CACHED source=INDEPENDENT_DEVICE_PROBES ... legacy_candidates=NOT_REQUIRED
[REGION_RING_TOPOLOGY_SEARCH] mode=APPLY_TOPOLOGY ...
```

`phase=MEASURED status=FAILED` still retains the old fixed Ring,
but it now identifies `DEVICE_PROBE_FAILED` or
`INCOMPLETE_DEVICE_PROFILE` instead of hiding behind a legacy
memory-filter failure. In particular, require usable CPU, CUDA,
Phone, GPU->PC, PC->Phone and Phone->PC measurement points.

Device Profile caching is in-memory and valid for the current model
load; it is **not** an on-disk cache. Independent measurement still
takes time on every new process/model-load.

Verification: rebuild the **PC** `llama-cli` from this revision, keep
the existing RPC patch-18 Phone binary (rebuild Phone only if it is
older), and run exactly the same `apply-topology` command as below.
Inspect `REGION_RING_PLAN_PROFILE`, `REGION_RING_TOPOLOGY_SEARCH`,
`REGION_RING_PLAN_SELECTED` and all three
`REGION_RING_PLAN_APPLY` phases. Never infer actual execution from
candidate listing alone. Correctness and performance need real-device
A/B testing; this refactor has not been hardware-validated.

## P8 output-bridge false rejection (2026-10-27)

On the first real `apply-topology` run with an internal Phone span
`[10,18)`, Meta successfully constructed the 503-node P8 graph:

```text
[PHONE_BLOCK_FUSED] sg=0..15 layers=10..17 nodes=503
[PHONE_RING_AUTO] side=pc event=OUTPUT_READY layer=17 sg=16 status=OK
[REGION_RING_PHONE_FUSION] event=REJECT sg=16 reason=PAIR_OUT_OF_RANGE
```

The reject was a **misclassification**. SG16 is the deliberately
separate single-node `l_out-17` output bridge; it is not an incomplete
Phone Attention/FFN pair. The previous fail-closed guard rejected any
Phone-only graph that could not start a complete pair, including this
already completed bridge.

The fix tracks the output bridge **only** after (a) the exact mirrored
single-node `l_out-N` marker is validated and (b) the outstanding
Phone graph completion sequence was consumed successfully by
`phone_ring_auto_wait`. That SG follows Meta's preexisting bridge
communication path and does not call `find_phone_block` again. All
other incomplete or unexpected Phone-only prefill graphs remain
fail-closed, and single-token decode remains on the original per-SG
path.

On the next real run, after each fused internal Phone region, expect:

```text
[PHONE_RING_AUTO] side=pc event=OUTPUT_READY layer=17 sg=16 ... status=OK
[REGION_RING_PHONE_FUSION] event=OUTPUT_BRIDGE sg=16 layer=17 action=PASS_THROUGH status=READY
```

No `event=REJECT reason=PAIR_OUT_OF_RANGE` is expected at that same
terminal bridge. This fix is **only PC-side Meta code**; a working RPC
patch-18 Phone server does not need rebuilding for this particular
change. Build the PC and rerun the model. Verify both Phone regions,
`REGION_RING_FINAL_OWNER result=MAIN`, drained output slots,
`REGION_RING_SUM complete=1`, and numeric/token parity. Neither full
success nor performance improvement has yet been established by the
previous failed run.

## General-topology PC tail must own its output head (2026-10-27)

An applied 8-Region Plan completed the P8 `l_out-17` bridge and
progressed into GPU tail `[46,48)`, but then failed:

```text
[REGION_RING_PHONE_FUSION] event=OUTPUT_BRIDGE sg=16 layer=17 action=PASS_THROUGH status=READY
[REGION_RING_FINAL_OWNER] stage=7 status=REJECT reason=PC_TAIL_HAS_META_SPLIT
```

Root cause: `llama_model_base::load_tensors()` historically placed
**all Hybrid output weights** under Meta(CPU,RPC), regardless of the
last Region's CPU/CUDA device. The final output-norm / LM-head graph
therefore included Meta even though the last transformer layers were
on the PC. In contrast, `apply-topology` explicitly uses a canonical
PC-owned final output stage.

Fix: only when `region_ring && hybrid_ring_plan_topology`, set
`pimpl->dev_output = {cpu_dev, &pimpl->cpu_buft_list}`. CPU_DIRECT
output ownership is deliberately conservative for both CPU and CUDA
tails. It avoids unexpected GPU VRAM allocations from moving the
large output projection to CUDA. Existing fixed Ring/Legacy Hybrid
placements are unaffected.

The final stage still verifies **every** scheduler split is non-Meta.
No Meta/RPC split is silently allowed. After rebuilding the PC client,
expect:

```text
[REGION_RING_OUTPUT_PLACEMENT] mode=APPLY_TOPOLOGY output=CPU_DIRECT ... meta_rpc=DISABLED status=READY
[REGION_RING_FINAL_GRAPH] stage=7 layers=[46,48) cpu_splits=... gpu_splits=... meta_splits=0 output=CPU_DIRECT status=PC_ONLY_READY
[REGION_RING_FINAL_OWNER] device=GPU region=7 ... result=MAIN status=READY
```

`device=GPU` describes the tail **Region worker**; the final output
projection actually runs on CPU_DIRECT. A CUDA-placed output head
would require an explicit GPU-memory-aware Plan and is deferred.

The previously observed `REGION_RING_BACKEND_SPLIT` output also showed
CPU regions borrowing CUDA compute due to scheduler offload. That
is a separate performance/ownership optimization, not this output
placement bug.

Correctness of final logits / Decode and an actual successful Prefill
remain **unverified** after this change. Rebuild PC from the new
commit, retain a working RPC patch-18 Phone, and compare final output
against the fixed layout only after all 48 Region×Chunk tasks (for an
8-Region layout with six chunks) complete.

## First unresolved failure after PC-owned output head (2026-10-27)

The next real run successfully used the model-owned CPU_DIRECT output
placement and constructed the PC-only final graph:

```text
[REGION_RING_FINAL_GRAPH] stage=7 layers=[46,48) cpu_splits=1 gpu_splits=1 meta_splits=0 output=CPU_DIRECT status=PC_ONLY_READY
[REGION_RING_FINAL_OWNER] device=GPU region=7 result=MAIN status=READY
```

The run **still failed**, but on a different worker:

```text
[REGION_RING_WORKER] device=PHONE event=FAIL region=2 chunk=3 status=-1
[REGION_RING_SUM] tokens=358 regions=8 chunks=6 executed=23 complete=0 status=-1
```

The user-supplied filtered excerpt contains no underlying
Phone/RPC/graph-allocation failure line for this task. Therefore the
root cause is **not established**; do not alter KV/FIFO credit ordering,
Graph cache identity, or RPC synchronization based on guesswork.

Added diagnostics without changing execution:

- `REGION_RING_STAGE_FAIL phase=PREPARE`: failure allocating or preparing
  the stage graph
- `phase=COMPUTE`: scheduler or Meta backend returned non-success
- `phase=OUTPUT_COPY`: intermediate hidden output validation/copy failed
- `REGION_RING_WORKER event=FAIL ... reason=...`: distinguishes a null
  stage result, bad return owner, non-success status and cancellation
- `PHONE_RING_AUTO side=pc event=REGISTER ... status=FAIL`: report both
  RPC transport success and remote registration acknowledgement
- GPU terminal resource audit now inspects the actual **MAIN** scheduler,
  rather than stale GPU worker graph split state; prints
  `scheduler=MAIN` or `scheduler=WORKER`.

For a useful failure reproduction, capture unfiltered PC stdout **and**
stderr. Keep the Phone server log separately. Look for the **first**
`REGION_RING_STAGE_FAIL` and preceding Meta/RPC error; end-of-run
`FINAL_OWNER result=MAIN` only means the canonical output pointer was
selected, not that all workers succeeded.

```powershell
Select-String -Path ".\\ring-pc-full.log" `
    -Pattern 'REGION_RING_STAGE_FAIL|REGION_RING_WORKER|PHONE_RING_AUTO|REGION_RING_PHONE_FUSION|failed to allocate graph|GGML_ASSERT|RPC.*FAIL|REGION_RING_FINAL_GRAPH' `
    -Context 4,6 | Out-File ".\\ring-pc-failure-context.log"
```

No phone server rebuild is required for these PC-only diagnostics.

## Phone chunk 3 intermediate output shape diagnostic (2026-10-27)

An 8-stage `C6,G4,P8,C6,G4,P8,C8,G4` Prefill run with 358 tokens
completed the Phone P8 compute and `OUTPUT_WAIT status=OK`, then failed
on the **first Phone region**, stage 2, chunk 3:

```text
[REGION_RING_STAGE_FAIL] phase=OUTPUT_COPY stage=2 device=PHONE layers=[10,18) chunk=3 tokens=64 detached_phone=1 status=-1
[REGION_RING_SUM] ... regions=8 chunks=6 executed=23 complete=0 status=-1
```

This is **not** a GPU output-head error. Code inspection confirms
`copy_hybrid_stage_output()` returns false only if
`res->get_stage_output()` is absent, not F32, not contiguous, or does
not match the required `[n_embd,n_tokens]` 2D shape. The actual
`ggml_backend_tensor_get()` happens *after* this check, so no
evidence currently shows a failed data copy or a failed Phone RPC
graph. It is unsafe to relax these shape checks without identifying
the mismatch.

This revision logs:

- `[REGION_RING_STAGE_OUTPUT] phase=PRE_COMPUTE`: Phone result pointer,
  tensor pointer, type, shape, contiguity **before** detached execution
- `phase=POST_COMPUTE`: the same after all Phone completion waits,
  with `metadata_changed=1` if fields/pointer changed
- `[REGION_RING_OUTPUT_COPY_REJECT]`: precise failure reason, actual
  `ne[0..3]`, strides, buffer/data pointers and expected shape

If PRE and POST differ, investigate graph ownership and metadata
lifetime while the Phone worker runs concurrently with PC workers.
If both are invalid in exactly the same way, investigate Qwen3 MoE
Stage output graph construction and output pruning instead.

Rebuild only the PC executable, then capture both stdout and stderr
in one log. Suggested PowerShell command to filter it:

```powershell
Select-String -Path ".\\region-pair.log" -Pattern 'REGION_RING_STAGE_OUTPUT|REGION_RING_OUTPUT_COPY_REJECT|REGION_RING_STAGE_FAIL|PHONE_RING_AUTO.*OUTPUT_WAIT|REGION_RING_SUM' |
    ForEach-Object { $_.Line }
```

These extra checks do not alter Phone Graph execution or Relax CPU/Phone
buffer ownership. PC and Phone RPC patch remain unchanged. **No complete
Prefill or numerical parity validation has been claimed.**

## Meta Graph ownership and Phone Worker scheduler fix (2026-10-27)

A subsequent run completed both fused Phone P8 graphs, including the
second `l_out-35` bridge, but the Meta backend then asserted while
trying to split a **PC-only** compute tensor:

```text
[META_SIMPLE_NONMETA] name=norm-36 op=RMS_NORM buffer=0x0 index=1 ne=[2048,64,1,1]
GGML_ASSERT(ggml_backend_buffer_is_meta(tensor->buffer)) failed
```

The selected Plan has Phone `[28,36)`, then CPU `[36,44)`,
then GPU `[44,48)`. `norm-36` is *not* Phone-owned and must not
be mapped into a Meta CPU/RPC compute graph.

A code review exposed a concrete scheduler ownership bug:
`src/llama-context.cpp` constructed **all three worker schedulers**
with backend-list offset 1 when `pc_phone_overlap` was enabled.
That intentionally removes Meta for **CPU/GPU** workers but
incorrectly removed it for the **Phone** worker too. The Phone
scheduler now gets the full `[Meta,CUDA0,CPU]` list, while CPU and
GPU workers retain `[CUDA0,CPU]`. At initialization, require:

```text
[REGION_RING_PC_SCHED] worker=CPU ... meta_registered=0 status=READY
[REGION_RING_PC_SCHED] worker=GPU ... meta_registered=0 status=READY
[REGION_RING_PHONE_SCHED] worker=PHONE ... meta_registered=1 status=READY
```

If the Phone scheduler cannot include Meta, it is rejected before
any Graph compute. Correct device ownership is required even though
the Meta backend may use its separate local CPU child internally.

The Meta backend now also fails closed **before rotating the Meta
tensor-container graph cache** if any model-owned Phone graph has a
non-Meta compute tensor without the preexisting host-view exception.
This replaces the raw `META_SIMPLE_NONMETA` assert with a diagnostic
showing the graph UID, first/last node, offending op/buffer, and
`norm_layer`/Phone ownership:

```text
[REGION_RING_META_GRAPH_REJECT] ... name=norm-36 ... norm_layer=36 phone_owned=0 reason=NON_META_COMPUTE_NODE status=FAIL
```

It is **not** safe to pretend `norm-36` is allocated in Meta or to
skip its RMS_NORM calculation. If the new rejection still appears,
the remaining root cause needs investigation with the graph UID and
scheduler ownership logs; the scheduler-list fix is a hypothesis
supported by source inspection, not yet hardware-proven.

A diagnostic-only A/B env var can temporarily serialize Phone
execution under the shared backend gate:

```powershell
$env:LLAMA_HYBRID_REGION_RING_STRICT_BACKEND_GATE = "1"
```

This disables PC/Phone overlap for the experiment; **do not measure
performance** with it. Remove the variable to restore normal overlap:

```powershell
Remove-Item Env:LLAMA_HYBRID_REGION_RING_STRICT_BACKEND_GATE -ErrorAction SilentlyContinue
```

The `expected_ne` logging for Phone Stage outputs also now casts
the uint32 `model.hparams.n_embd` to int64 before printing with
`PRId64`; the earlier enormous printed dimension was a printf
format mismatch, not an actual tensor size.

Rebuild the **PC** executable from this revision. The Phone RPC
patch-18 binary is unchanged. Start with the normal concurrent
`apply-topology` test, inspect the three scheduler initialization
lines and `REGION_RING_META_GRAPH_REJECT`. Only if needed run the
strict-gate A/B test. Real-device correctness/parity and throughput
remain **unverified**.

## Verification gates

1. Both binaries accept RPC protocol patch 18 and Phone server loads weights.
2. `[REGION_RING_PLAN_SELECTED] mode=APPLY_TOPOLOGY source=AUTO` and
   `[REGION_RING_PLAN_APPLY] phase=MODEL_LOAD applied=1`.
3. `[REGION_RING_PLAN_APPLY] phase=WEIGHT_PLACEMENT applied=1`
   and `phase=RUNTIME_STAGES applied=1`, with identical layouts.
4. `[REGION_RING_PHONE_MASK]` and `[REGION_RING_PHONE_PLAN]`
   register the exact new Phone layer spans.
5. Phone logs `[PHONE_RING_AUTO] side=phone event=VALIDATE` and
   `READY_ENQUEUED` / `EXECUTE_DONE`; Meta handles each terminal
   `l_out` exactly once.
6. `[REGION_RING_FINAL_OWNER]` reports CPU or GPU with `result=MAIN`.
7. `[REGION_RING_OWNERSHIP_SUM] live_slots=0 status=DRAINED`.
8. `[REGION_RING_SUM] ... complete=1 status=0`, with
   `executed = regions * chunks`. Not every Plan has 54 tasks.
9. Compare generated output and layer/logit numeric parity with the
   fixed-Ring branch. Device changes can cause small floating-point
   differences; large drift or unexpected tokens require investigation.
10. Only after correctness passes, compare warmed repeated 358-token
    prefill wall-time medians against the original fixed layout.

If Phone fusion is incomplete, graph capacity is exceeded, or a PC-tail
graph unexpectedly includes a Meta split, this experiment **fails
closed** rather than inventing a different ownership path. A failure
during weight loading is not automatically recoverable in the same
process; restart with the previously tested branch and layout.

For the old standalone `topology-preview`, setting that mode rather
than `apply-topology` leaves all weights/stages unchanged. New Ring-only profiling bypasses legacy candidate scoring; that
warning should not be emitted in these modes when
`--hybrid-auto` is disabled. If it appears, verify the CLI flags
and executable revision rather than assuming it is harmless.
