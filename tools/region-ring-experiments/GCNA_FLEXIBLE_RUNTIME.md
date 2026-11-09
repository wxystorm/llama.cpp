# GCNA flexible tail / multi-layer HTP / adaptive weight residency

Development branch: `feat/gcna-v2-flexible-tail-npu-resident-20261109`.
Keep the previously verified V2 branch unchanged until both PC and Phone
have compiled and completed an end-to-end inference run.

## Planner and execution contract

- A cycle follows `G -> C -> N -> A`; absent backends create no phantom
  Region. The last Transformer Region may be on PC CUDA, PC CPU, Phone HTP,
  or Phone OpenCL. The final normalization / LM Head remains PC CPU-owned,
  separate from the last Transformer Region.
- A selected NPU Stage carries the exact **contiguous** interval
  `[layer_begin,layer_end)`. For example, `G8,N3,G8` executes all three
  HTP layers in **one** Phone Stage and publishes its output only after
  the complete interval. This is not three new PC->Phone boundary Regions.
- A Plan may choose NPU0. In this case Phone HTP Region registration and
  its weight arenas are absent; A Regions use the normal OpenCL path.
- The native NPU graph builder supports Stage depths 1..32, but the
  executable Planner selects only **measured, supported NPU depths** in the
  native Stage Profile. An unmeasured NPU depth is not silently estimated;
  the Phone also checks the real HTP allocation and Graph compilation.
- Every selected HTP Layer owns its own persistent K/V tensors. The PC
  sends the canonical Decode K/V binding for every layer before the Phone
  registers the Stage's Chunk tasks.

## Phone memory policy

The Planner evaluates the complete candidate's shared Phone memory
footprint, including HTP weights, HTP K/V, NPU Graph workspace, Host
staging, resident Adreno buffers and shared overhead. Each NPU Stage must
also fit an individual physical HTP allocation.

- `DISABLED`: no HTP Regions; no NPU weight allocations or mover.
- `RESIDENT`: the complete sum of Plan-selected NPU Stage weights fits;
  allocate one stable HTP repack arena per NPU Stage at setup, load once,
  and emit no rolling Bank-refill events.
- `STREAMING_2BANK`: full residency is too large but the two rank-parity
  Bank peaks fit; allocate the two stable arenas, keep KV independent,
  and replace Stage weights after their previous Bank user drains.
- If neither mode can fit, the candidate is rejected or an NPU suffix may
  be assigned to Adreno without changing Transformer layer coverage.

The current Android Hexagon default single-buffer limit is 1024 MiB
(`GGML_HEXAGON_MBUF`). Native GCNA planning conservatively uses that default
as its physical allocation ceiling until the Phone reports a measured
per-buffer cap over RPC. A **single NPU Stage per Cycle** must occupy no
more than half of BOTH the remaining shared Phone weight budget (after
KV/workspace/Host/Adreno reserves) and the physical HTP single-buffer limit.
With the default Hexagon cap, that is at most 512 MiB per N Stage. The
constraint also applies to RESIDENT plans; a larger Stage is never admitted
just because the entire Plan's weights fit in aggregate. Planner can shrink
an N Stage and place the remaining Phone layers on Adreno if profiled and
memory-feasible; it does not hardcode a maximum NPU layer count.

Planning budget is an admission estimate, not a promise that arbitrary
HTP allocations will succeed. The Phone's repack-buffer allocations are
the final capacity check.

## RPC version / deployment

GGML RPC protocol patch **36** changes both sides of the native contract.
The Plan configure request sends `region,layer_begin,layer_end`, an
optional immediate Adreno successor, and the selected NPU weight policy.
The Phone-owned HTP Stage descriptor now carries an exclusive
`layer_end` and uses Plan version 3. Both PC and Android RPC binaries
must be rebuilt from the same revision. The V2 native Profile wire also
expects RPC version 4.0.36.

The final Transformer Stage must publish full `l_out` when a separate
Head is enabled. `prepare_ubatch()` suppresses the inline Head for that
Stage, and the canonical result is built once after every task and return
frame has finished. The separate Head is validated to run **only on PC CPU**,
whether the final Transformer Stage is G, C, N or A. GCNA PC-final execution
currently requires the autonomous-final output mode and its Phone stream
prerequisites; it is not a universal legacy inference fallback.

## Build and host regression checks

After switching to this branch, rebuild the **PC executable and Phone RPC
server from the same patch-36 source revision**. On PC PowerShell (use
an existing configured `build-tensor-v2` with RPC and tests enabled):

```powershell
cmake --build build-tensor-v2 --config Release --parallel 8 --target llama-cli ggml-rpc-server test-region-ring-v2-stage test-region-ring-planner test-region-ring test-phone-plan-region-map test-phone-npu-weight-banks test-phone-owned-execution-plan test-phone-qwen3-local-builder
ctest --test-dir build-tensor-v2 -C Release --output-on-failure -R "^(test-region-ring-v2-stage|test-region-ring-planner|test-region-ring|test-phone-plan-region-map|test-phone-npu-weight-banks|test-phone-owned-execution-plan|test-phone-qwen3-local-builder)$"
```

Use the repository's Android/NDK build procedure for
`ggml-rpc-server` and install matching libraries on Phone; an older binary
using RPC 4.0.35 is **not** compatible. Keep the manual
`NPU_SERIAL_REGIONS` and `MODEL_LOAD_LAYERS` lists unset in
plan-controlled mode. Inspect the per-run logs for
`[REGION_RING_NPU_MEMORY]`, `[PHONE_NPU_PLAN]` and
`[PHONE_OWNED_KV]` before relying on any speed measurement.

The supported diagnostic logs include
`[REGION_RING_NPU_MEMORY] policy=...`,
`[PHONE_NPU_PLAN] ... weights=...`,
`[PHONE_MIXED_BACKENDS] region=... layers=[...)`, and
`[REGION_RING_PLAN_APPLY] phase=WEIGHT_PLACEMENT`.

## Verification boundary

These changes are committed as development source and host-side
regression fixtures. They have not yet been compiled on the target
Windows/Android toolchains or executed on the Snapdragon device.
Rebuild PC and Phone together, run the new C++ tests, then check that the
logged model-owned layout and actual HTP Stage ranges agree.
