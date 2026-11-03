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

## Runtime Meta bypass + Phone-owned P2C publication (patch 26)

Meta stays on PC for **initial Phone Full graph building** and RPC graph
registration. For **non-final** Phone Full Regions only, PC runtime does not
call Meta Graph Compute or look up Meta KV View mappings. The existing Phone
Autonomous thread handles VERIFIED input -> Slot PREPARE -> Compute and
**automatically captures** the registered output into P2C Arena on successful
compute completion, without a per-Chunk PC WAIT/CAPTURE request. The separate
network sender publishes the ready frame immediately. PC pre-registers every
non-final Phone Chunk's destination **before starting input PUSH** so fast
outputs never reach the receiver before destination registration. PC's Ring
Worker only delegates the chunk and defers it; PC Receiver (unchanged)
publishes downstream READY **after the entire frame is received**.
The global PC PREPARE lock is not held during Phone compute or P2C.

The final Phone Region and PC vocabulary head still use the legacy Meta
compute path; final Phone graphs are forcibly reset and rebuilt just-in-time
before each final-region Chunk to avoid reusing long-lived KV View mappings. Decode, directional/Tensor paths, and
legacy Phone execution remain unchanged. There is no per-Chunk PC WAIT_CAPTURE RPC in patch26 autonomous mode.
The final Phone Region and PC Head continue on the legacy path.

New diagnostics:
- `[PHONE_AUTO_P2C] event=PREBIND ... status=READY` on PC
- `[PHONE_AUTO_DIRECT] event=DELEGATE ... per_chunk_rpc=0` (debug level)
- `[PHONE_AUTO_DIRECT] event=AUTO_CAPTURE ... status=PASS` on Phone
- `[PHONE_AUTO_P2C] event=ABORT ...` on Phone failure (P2C ERROR to PC)
- `[P2C_STREAM] side=PC phase=RECEIVER_DONE ... status=OK` once at teardown

## Enabling

**Build both PC and Phone from this branch.** RPC patch 26 extends task
registration (command 50) with P2C run ID, token interval and output byte
contract. Commands 49/51/52 remain allocated, but autonomous mode no longer
issues the per-Chunk 52. Older Phone binaries are rejected by the registration
capability check; PC and Phone MUST both be rebuilt from the same version.

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

PC's Phone Worker no longer invokes `PHONE_AUTO_WAIT` or
`PHONE_AUTO_WAIT_CAPTURE`. Phone's Compute Worker captures the verified
output into the Arena, while the dedicated P2C sender streams the frame.
PC Receiver remains the only owner of downstream READY, after full frame
receipt. A failed Phone input verification, prepare, compute or capture aborts
the Arena, sending a P2C ERROR rather than silently hanging the PC Receiver.

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
savings. `capture_ms` and `published` report Phone-owned Arena capture
cost and output frame count (not TCP send completion). Detailed Phone execute logs are opt-in via
`GGML_RPC_PHONE_AUTO_TRACE=1`.

On PC, ensure non-final Phone Regions' `PUSH_APPLY` Copy count drops,
with Graph Compute RPCs limited to the terminal Phone Region. The
autonomous registration cost happens in setup and may dominate
short tests, since this v1 allocates a separate Graph/Buffer per Chunk.

**Verification status:** source integration is experimental. Linux/Windows
whole-project builds and Android OpenCL end-to-end numerical parity must
be performed before treating this as a performance or production baseline.
