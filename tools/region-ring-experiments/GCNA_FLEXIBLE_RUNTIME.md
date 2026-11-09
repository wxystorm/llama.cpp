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
- The native NPU graph execution ABI supports Stage depths 1..32. The
  Planner may estimate larger Stage service demands from measured smaller
  depths within that bound. The Phone still verifies exact HTP buffers.
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
