# Phone autonomous Region: transport foundation (2026-11-03)

This branch implements **Phase A**, the ACK-free PC-to-Phone input stream. It does
**not** yet move Full-Region Graph launch/argument binding from the PC Meta
executor to the Phone. Keep that distinction explicit: an ACK-free send is not
the same as autonomous Phone computation.

## Enable and run

Compile **both** Windows and Android binaries from the same commit. RPC protocol
patch is now **23**. Existing protocol patch 22 clients/servers cannot be mixed.

The old PUSH and its ACK timing remain available when stream is off.

```powershell
$env:LLAMA_REGION_RING_PHONE_INPUT_STREAM = "1"
$env:LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES = "1"
$env:GGML_META_PHONE_FULL_REGION_GRAPH = "1"
$env:LLAMA_REGION_RING_TASK_TRACE = "1"
# Reuse your exact fixed layout, model, prompt, batching and sweep command.
```

Stream mode is explicit and fails early if the Phone RPC extension is unavailable.
Do not enable it for decode, directional Tensor stages, or unsupported runtime
configurations.

## Data ownership and ordering

- PC sends one framed `run_id/region/chunk/bytes/hash/payload` on the
  existing dedicated PUSH socket and proceeds immediately after `send_data`.
  There is no per-Chunk success ACK.
- Phone receives at most 128 MiB of outstanding framed data, including a frame
  during receive and a buffer in GPU upload. Once full, its socket worker
  stops reading and TCP provides backpressure. No explicit credit messages.
- Phone verifies the received payload's FNV-1a hash before publishing staged
  host bytes. The buffer is **moved**, not copied, into the staging queue.
  Failed validation is remembered and can be observed by APPLY.
- The current PC-owned Phone Compute Worker still invokes APPLY. APPLY waits
  for matching *Phone-verified* bytes, with a bounded 30-second timeout;
  `send_data` returning on PC is **not** considered verified input.
- A failed or mismatched APPLY retains the old SET_TENSOR recovery behavior.
  Check both `applied` and `fallback`, not only `accepted`. In stream mode,
  `accepted` counts successful TCP sends, **not** remote verification.
- At RUN end the PC sends one teardown control RPC which drops any remaining
  staged Host buffers. Device activation slots remain separately managed.
- Per-region KV ordering stays owned by the existing PC-side Region scheduler.

## Status and limitations

Phase A is an experimental transport path and has **not** been validated on the
Windows/Android hardware yet. In particular, a full 128 MiB backlog of chunks
with unfavorable causal ordering can still need a protocol-level scheduling
guarantee to avoid an input-dependency deadlock. Until the autonomous local
scheduler is implemented, this build therefore rejects a single ubatch whose
total predicted Phone input payload exceeds 120 MiB (8 MiB safety headroom).
Initial validation should use the 358-token / 20.977 MiB case, then 1024-token
experiments. The Phone emits one [PHONE_INPUT_STREAM_SUM] line at run end with
receiver, hash and APPLY-wait costs. The aggregate Phone recv/hash timing fields in the old ACK are zero
for stream mode, because there is no ACK. Do not interpret them as zero cost.

## Phase B requirements: true Phone-local execution

1. At prefill setup, register *every* actual Phone Full-Region graph UID,
   bound device resources, the input/output tensor map, KV/position state, and
   the standard and terminal Chunk shapes. PC-built Meta outer templates
   **alone do not** satisfy this requirement.
2. Add a Phone-owned Region Executor state machine:
   `RECEIVED -> VERIFIED -> READY -> COMPUTING -> OUTPUT_READY`. A queued
   input may execute only when its per-Region preceding KV update is committed,
   its graph/bindings are valid and the GPU slot is available.
3. Move per-Chunk graph binding and launch out of
   `llama_context::run_hybrid_stage_block` on the PC for this opt-in mode.
   Phone input arrival, not a PC GRAPH_COMPUTE or APPLY command, must wake
   Phone execution. Preserve strict local KV correctness.
4. Publish Phone Full-Region intermediate outputs through P2C streaming into
   the existing PC downstream-ready event mechanism. The final Phone Region
   requires explicit result/PC Head ownership, not a fake intermediate frame.
5. Only after numerical parity and FIFO/decode tests, calibrate the Planner's
   new overlapping C2P service-time model.

Never report Phase A as Phone-autonomous inference.
