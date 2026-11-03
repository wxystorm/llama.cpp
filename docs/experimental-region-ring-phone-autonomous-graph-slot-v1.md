# Experimental Phase B: Phone-owned Full Region compute + two-slot lookahead

Branch: `feat/phone-autonomous-graph-slot-v1-20261103`.

This is a **correctness-first, opt-in experiment**. It does not replace the
existing Region Ring planner, PC CPU/CUDA scheduling, or Phone-to-PC output
return. Its two goals are:

1. Verified PC->Phone PUSH is sufficient to launch the registered Phone
   Full Region graph, without a per-Chunk PC `PHONE_INPUT_APPLY` or
   `GRAPH_COMPUTE` command.
2. The Phone-side prepare worker can upload Chunk N+1 to the other
   OpenCL Stage Slot while Chunk N computes. KV-dependent graph execution
   remains in strict Chunk order.

## Runtime Meta bypass (patch 25)

Meta stays on PC for **initial Phone Full graph building** and RPC graph
registration. For **non-final** Phone Full Regions only, PC runtime does not
call Meta Graph Compute or look up Meta KV View mappings. The existing Phone
Autonomous thread handles VERIFIED input -> Slot PREPARE -> Compute; the PC
Phone Worker uses a dedicated completion socket for a direct
`PHONE_AUTO_WAIT_CAPTURE` request keyed by (input run, region, chunk,
graph UID) plus the separate P2C stream run ID. Phone validates exactly
one contiguous F32 output with the expected token count and bytes, and
uses the existing Arena to snapshot GPU output before replying. The PC
P2C receiver (unchanged) fills the downstream host edge and publishes
READY. The global PC PREPARE lock is released before WAIT_CAPTURE.

The final Phone Region and PC vocabulary head still use the legacy Meta
compute path; final Phone graphs are forcibly reset and rebuilt just-in-time
before each final-region Chunk to avoid reusing long-lived KV View mappings. Decode, directional/Tensor paths, and
legacy Phone execution remain unchanged. This phase intentionally retains
one PC WAIT_CAPTURE RPC per Chunk; proactive Phone P2C publication is a
separate next phase, after data correctness is validated.

New diagnostics:
- `[PHONE_AUTO_DIRECT] event=WAIT_CAPTURE ... meta_compute=0 status=PASS` on PC
- `[PHONE_AUTO_DIRECT] event=WAIT_CAPTURE ... status=PASS` on Phone
- `[PHONE_AUTO_DIRECT] event=WAIT_CAPTURE_REJECT ... reason=...` on failure

## Enabling

**Build both PC and Phone from this branch.** RPC patch 25 adds command 52
(Phone autonomous WAIT_CAPTURE), preserving the patch24 49/50/51 commands.
Older patch24 Phone binaries are rejected during the preflight probe;
PC and Phone MUST both be rebuilt from the same version.

Start the Phone RPC server with:

```sh
export GGML_RPC_PHONE_BINDING_PREFLIGHT=1
export GGML_RPC_PHONE_STAGE_SLOT0_EXEC=1
export GGML_RPC_PHONE_STAGE_DUAL_SLOT=1
export GGML_RPC_PHONE_STAGE_PREPARE_AHEAD=1
# Run your normal RPC server from this exact branch.
```

Binding preflight is required to populate each registered Graph's Slot plan.
The Phone accepts either plain `inp_stage` or the exact Meta-serialized
`Meta(...)#inp_stage#0` leaf, and rejects ambiguous/invalid candidates.
A task binding failure now logs `[PHONE_AUTO] event=REGISTER_REJECT` with
`reason=...` on the Phone (e.g. `SLOT0_NOT_READY`, `INPUT_NAME_NOT_FOUND`,
`INPUT_BYTES_MISMATCH`, or `INPUT_RANGE_OVERLAP`). The server uses
native OpenCL storage addresses for its registered input leaves: the generic
client serializer alone zeroes non-RPC buffer addresses and must not be used
as an authoritative Phone-local tensor descriptor. If an actual address
conflict remains, `event=INPUT_RANGE_CONFLICT` reports both UID and address
ranges; the collision is still rejected rather than silently overwritten.

Before launching `run_prompt_sweep.ps1` on Windows:

```powershell
$env:GGML_META_PHONE_FULL_REGION_GRAPH = "1"
$env:LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES = "1"
$env:LLAMA_REGION_RING_PHONE_INPUT_STREAM = "1"
$env:LLAMA_REGION_RING_PHONE_AUTONOMOUS = "1"
$env:LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM = "1"
$env:LLAMA_REGION_RING_TASK_TRACE = "1"
$env:LLAMA_REGION_RING_PREPARE_BREAKDOWN = "1"
```

Keep `GGML_META_RING_PHONE_FUSION` / `GGML_META_RING_PHONE_AUTOSTART`
**disabled** for this experiment. The complete Full Region proof path is
required. Use a fixed layout with **non-final Phone Full Regions and a final
Phone Full Region** (for example `G10,C1,P3,G11,C1,P1,G11,C1,P1`)
at first. No directional Tensor stage, multi-output batch, or mixed
Tail Graph may be admitted unless the registered exact graph validates.

## Lifecycle

At setup, PC builds each non-final Phone Region/Chunk with that Chunk's
true KV position, mask and token shape. Each Chunk gets its own Scheduler
and RPC graph/input storage. The completed Phone Full Graph is registered
with a uniquely salted UID (Region + Chunk). The setup sends dynamic side
inputs but *does not* upload the hidden `inp_stage` placeholder.

The PC opens the input stream and binds each registered graph UID to
`run_id,region,chunk,expected_input_bytes`. The Phone checks the exact
OpenCL input leaf, registered Graph, complete two-Slot plan and distinct
input allocation before accepting the task.

When a pushed frame passes its hash check, it enters the existing bounded
Phone Host queue, then wakes the Phone-owned prepare worker:

```
PUSH/verify -> Host READY -> local APPLY original OpenCL input
                          -> prepare next free GPU Slot
                          -> PREPARED
                          -> strict per-Region KV-order Graph Compute
                          -> COMPLETE
```

At most one subsequent Chunk may be in preparation beyond the next
KV-eligible Chunk. Compute runs on a separate Phone worker. It reuses the
original Stage binding and Slot policy `phone_stage_prepare_input()`
and `phone_stage_slot0_compute()`, not a new kernel executor.

For this first implementation, the local upload is deliberately
conservative: Host -> private original OpenCL input -> staged Slot.
This extra local OpenCL copy is not yet optimized away; the objective is
to validate local triggering and upload/compute overlap first.

PC's Phone Worker now invokes a `PHONE_AUTO_WAIT` RPC for its matching
logical job, **not** APPLY or Graph Compute. On success it resumes the
unchanged Meta output handoff (including optional asynchronous return).
This is **not yet** Phase C's fully independent output event: the PC still
owns downstream READY publication.

Only **non-final Phone Full Regions** run autonomously in this phase.
The final Phone Full Region, any tiny startup/warmup request and one-token
decode continue with the original path.

## Safety

- Never infer a Graph from a token shape alone. Missing complete graph,
  ambiguous `inp_stage`, duplicate input allocation, mismatched bytes,
  unavailable two-slot plan, or invalid UID are hard rejects.
- Host PUSH validation precedes local upload. No hidden input is written
  into an actively computing tensor.
- Both GPU Slots use existing tickets/epochs and binding-map checks.
- The same Region executes Chunk N+1 only after Chunk N has returned a
  successful Graph completion (which includes KV writes). Input upload
  for N+1 may overlap Chunk N compute.
- Session teardown removes bound jobs and Host buffers after PC Workers
  have joined. The legacy RPC behavior remains the default.
- A failed task produces a failed WAIT rather than an implicit duplicate
  Graph execution. This mode currently targets one explicit PC/Phone pair
  and one stream run at a time.

## Validation

First build and run the standalone policy test:

```powershell
cmake --build .\build-tensor --config Release --target test-phone-autonomous-policy
ctest --test-dir .\build-tensor -C Release -R test-phone-autonomous-policy --output-on-failure
```

If using a different existing CMake build directory, substitute it. The
policy checks one-Chunk lookahead, strict KV FIFO, and 32-bit overflow.

Run the same fixed layout with autonomous mode OFF as control; then ON
using the same model/prompt/Threads/UBatch. Compare logits and KV parity
before comparing prefill. Check Phone RPC log:

```powershell
Select-String .\rpc.log -Pattern 'PHONE_AUTO_SUM|PHONE_INPUT_STREAM_SUM|PHONE_AUTO'
```

For 358 tokens, XP=64 and the example three-Phone-Region layout, expected
autonomous setup: **12 registered and completed jobs** (Regions 2 and 5,
6 Chunk each); Region 8 stays on the legacy path.
`PHONE_INPUT_STREAM_SUM` still includes 18 input frames.

`[PHONE_AUTO_SUM]` contains `registered`, `completed`, `failed`,
`upload_ms`, `prepare_ms`, `compute_ms`,
`prepare_compute_overlap_ms`, and `prepared_ahead`.
The overlap metric measures intersection of next Chunk upload/prepare with
the previous Chunk's compute for the same Region, not guaranteed wall-time
savings. Detailed Phone execute logs are opt-in via
`GGML_RPC_PHONE_AUTO_TRACE=1`.

On PC, ensure non-final Phone Regions' `PUSH_APPLY` Copy count drops,
with Graph Compute RPCs limited to the terminal Phone Region. The
autonomous registration cost happens in setup and may dominate
short tests, since this v1 allocates a separate Graph/Buffer per Chunk.

**Verification status:** source integration is experimental. Linux/Windows
whole-project builds and Android OpenCL end-to-end numerical parity must
be performed before treating this as a performance or production baseline.
