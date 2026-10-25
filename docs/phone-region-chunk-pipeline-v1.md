# Phone Region Chunk Pipeline V1 (experimental, opt-in)

This branch derives from `feat/phone-region-graph-20261025` at `585893a`, retaining the verified (pre-pipeline) C2P/P2C directional pair and the experimental complete Phone Region L35-L45 graph. The original correct baseline branch remains unchanged.

## Purpose and status

The baseline Meta execution order for a 358-token prefill (XT=128) is:

```
C2P(34) chunk 0, C2P chunk 1, C2P chunk 2
Phone Region(35..45) chunk 0, Region chunk 1, Region chunk 2
P2C(46) chunk 0, P2C chunk 1, P2C chunk 2
```

With the V1 pipeline enabled, the **Meta invocation order** becomes:

```
PC prefix (unchanged)
C2P(34) chunk 0 -> Phone Region chunk 0 [async RPC submit]
C2P(34) chunk 1 -> Phone Region chunk 1 [async RPC submit]
C2P(34) chunk 2 -> Phone Region chunk 2 [async RPC submit]
Phone L45 -> L46 graph join / P2C(46) (unchanged)
PC tail (unchanged)
```

Only *one* L35-L45 graph is issued for each chunk, with KV work ordered chunk 0, 1, 2. PC-side C2P norm/router preparation for chunk N+1 may run while the earlier Phone graph has been submitted and the Phone GPU continues working.

The stock gallocr arena is **not used by an in-flight Region graph**. Meta maintains two persistent Phone scratch slots. Every transient Region tensor is rebound at submit time to a private clone of its original Phone compute arena, preserving all gallocr offsets and VIEW relationships. Chunk N uses slot `N % 2`; before slot reuse, Meta explicitly fences the Phone backend. The final `phone_region_chunk_end_N-45` is excluded from scratch and gets its own compact per-chunk Phone output buffer so it remains valid until L46 consumes it.

This is a bounded, correctness-first **window=2** stage pipeline, not a completely independent dual-queue GPU execution engine. P2C per-chunk pipelining remains a future step.

Important: `PHONE_REGION_GRAPH_DONE` and `PHONE_REGION_PIPE_STAGE ... event=SUBMITTED` mean the host worker returned from asynchronous graph submission; they are NOT GPU execution timings.

## Enable

Set both flags on the **PC** launcher:

```powershell
$env:LLAMA_HYBRID_PHONE_REGION_GRAPH = "1"
$env:GGML_META_PHONE_REGION_PIPELINE = "1"
$env:GGML_META_PIPELINE_DEBUG = "1"
```

Do not change the Phone RPC protocol or the L34 / L46 split ratio for the initial test. The CLI, prompt, seed, temperature and output tokens must match the non-pipeline run.

To disable only scheduling (keeping the same per-chunk Phone Region graph):

```powershell
Remove-Item Env:GGML_META_PHONE_REGION_PIPELINE -ErrorAction SilentlyContinue
```

To return to the serial, non-Region execution path, disable both flags.

## Eligibility and fail-closed checks

The execution schedule changes **only** if the already-allocated Meta graph contains:

- Two or more consecutive C2P L34 `prefill_ffn_norm_chunk_N-34` and `prefill_ffn_down_chunk_N-34` pairs, with N = 0, 1, ...
- Matching ordered Phone Region graphs `phone_region_attn_chunk_N-35` through `phone_region_chunk_end_N-45`.
- One exact uninterrupted run of C2P chunks, followed by one exact run of Phone chunks, with no unaccounted Meta subgraphs and no outer scheduler split inside a chunk.
- The expected C2P L34 + P2C L46 directional owner layout.
- **Private Region memory:** gallocr aliasing with future C2P is expected and reported as `PHONE_REGION_PIPE_ALIAS ... action=ISOLATE`. When the pipeline is active, two private Region scratch slots isolate all transient L35-L45 tensors from the serial arena, and per-chunk output buffers preserve L45 outputs until L46. A scratch/output binding failure is fatal for the experimental path rather than silently falling back after execution has begun.

When a pipeline is activated, the previous legacy L34 `l_out-34` producer is disabled **at C2P chunk 0**, not at the final chunk. That producer would otherwise overwrite the Phone-owned, partially accumulated L34 output before Region chunk 0 reads it.

The graph's PC/Phone ownership and real COMPUTE checks continue to run in the existing Phone Region fused execution branch.

## Expected diagnostics, 358-token prefill

```
[PHONE_REGION_PIPE_PLAN] active=1 window=2 chunks=3 ... scratch_pool=1 reason=READY_SCRATCH_POOL
[PHONE_REGION_PIPE_CHUNK] chunk=0 c2p_norm=1 c2p_down=2 region=[7,29] status=READY
[PHONE_REGION_PIPE_CHUNK] chunk=1 c2p_norm=3 c2p_down=4 region=[30,52] status=READY
[PHONE_REGION_PIPE_CHUNK] chunk=2 c2p_norm=5 c2p_down=6 region=[53,75] status=READY

[PHONE_REGION_PIPE_STAGE] ... chunk=0 stage=C2P_FFN_REDUCE event=BEGIN
[REGION_C2P_PREFILL] layer=34 chunk=0 ... owner=PHONE status=OK
[PHONE_REGION_PIPE_STAGE] ... chunk=0 stage=PHONE_REGION event=BEGIN
[PHONE_REGION_GRAPH] chunk=0 ... layers=[35,46) status=DEVICE_LOCAL_READY
[PHONE_REGION_PIPE_STAGE] ... chunk=0 stage=PHONE_REGION event=SUBMITTED

[PHONE_REGION_PIPE_STAGE] ... chunk=1 stage=C2P_NORM event=BEGIN
...
```

The exact SG ranges are specific to the current 358-token graph and can change with graph construction; the structural validator does not hard-code them.

For three chunks, expect scratch/output ownership similar to:

```
[PHONE_REGION_OUTPUT_SLOT] chunk=0 ... status=BOUND
[PHONE_REGION_SCRATCH] chunk=0 slot=0 ... status=BOUND
[PHONE_REGION_OUTPUT_SLOT] chunk=1 ... status=BOUND
[PHONE_REGION_SCRATCH] chunk=1 slot=1 ... status=BOUND
[PHONE_REGION_OUTPUT_SLOT] chunk=2 ... status=BOUND
[PHONE_REGION_SCRATCH] chunk=2 slot=0 ... reuse_wait_ms=... status=BOUND
```

The non-zero `reuse_wait_ms` on chunk 2 is the explicit window=2 credit: slot 0 cannot be overwritten until the Phone backend has completed the earlier graph that used it.

### Pool prewarm and host-stage profiling

The first scratch-pool implementation allocated a slot/output lazily when its Region stage began. On the first 358-token hardware run this exposed roughly 795 ms and 720 ms in Region chunk 1/2 even though `PHONE_REGION_GRAPH_DONE` itself was only a few milliseconds. A plausible cause is remote OpenCL/RPC allocation serializing behind the already-submitted previous Region graph.

The current branch therefore allocates **all two scratch slots and every per-chunk L45 output buffer before the first C2P/Region pipeline stage starts**. The buffers remain persistent in the Meta backend config and subsequent graph executions reuse them.

Expected preflight:

```
[PHONE_REGION_POOL_PREP] ready=1 outputs=3 output_mib=... slot0_mib=... slot1_mib=... alloc_ms=... reason=READY
```

No `alloc_buffer` attributable to these Region slots should occur between `PHONE_REGION_PIPE_STAGE ... stage=PHONE_REGION event=BEGIN` and `event=SUBMITTED`.

A second hardware trace showed that allocation prewarm alone was not sufficient: chunk 1/2 still spent about 808 ms / 646 ms specifically in `output_bind_ms`. The RPC buffer backend resolves a remote buffer base lazily: the first `ggml_backend_buffer_get_base()` sends `RPC_CMD_BUFFER_GET_BASE` and caches `base_ptr`. Doing that first lookup after an earlier Phone Region graph had been submitted made the control request wait behind Phone work.

The pool prewarm now also:
- calls `ggml_backend_buffer_get_base()` for every Region output buffer and every private scratch arena before the pipeline starts;
- precomputes the complete output alias family and its byte offsets once per chunk;
- stores a local `phone_region_output_bind_plan` for the current Meta execution.

The Region hot path therefore performs descriptor-only pointer rebinding and logs `mode=PREBOUND`; it must not issue a lazy `BUFFER_GET_BASE` request or rescan all Meta subgraphs. Expected preflight now includes `output_plans=<chunks>`.

### Per-slot completion credit (RPC patch 13)

After prebinding removed the output stall, hardware showed `chunk=2 slot=0 reuse_wait_ms≈1315`. The old scratch reuse path called `RPC_CMD_SYNCHRONIZE`, which waits for the **current whole Phone queue**. With Region0 and Region1 both submitted, slot0 reuse therefore waited for Region1 as well even though only Region0 owns slot0.

RPC patch 13 adds a dedicated completion-wait socket and `RPC_CMD_WAIT_READY_SEQ`. Region chunks whose slot will be reused enqueue an existing `PHONE_FFN_MARK_READY(seq)` marker immediately after their graph on the compute socket. Because the server processes that marker only after the preceding graph completes, it precisely marks that slot's producer completion. A later slot reuse waits for that seq through the independent completion socket; Region1 may continue running while slot0 becomes reusable.

On the standalone per-slot-fence branch, the expected three-chunk trace was:

```
[PHONE_REGION_POOL_PREP] ... seq_fence=1 ... reason=READY
[PHONE_REGION_SLOT_CREDIT] chunk=0 slot=0 seq=... marked=1 future_chunk=2
...
[PHONE_REGION_SCRATCH] chunk=2 slot=0 ... reuse_wait_ms=... wait_mode=SEQ status=BOUND
```

The continuation branch below consumes that credit earlier at L46 and releases the slot, so its later reuse is expected to switch from `wait_mode=SEQ` to `wait_mode=NONE`.

If the seq feature is unavailable, the experimental pipeline fails preflight. At runtime, any missing marker/wait falls back to the previous global fence for correctness. Patch 13 changes the RPC protocol, so **PC and Phone server must be rebuilt from the same commit**.

### Per-chunk Region -> L46 continuation

The per-slot fence removed the false whole-queue drain, but a hardware trace still showed about 1.02 s waiting for Region0 itself and a later L45 boundary charged about 1.74 s to Phone->PC copy wait. The remaining problem was control-flow ordering: Meta submitted all Region chunks before it entered the post-Region L46 Tensor/P2C subgraphs.

The continuation branch discovers, for each Region chunk, the exact post-Region L46 subgraph range that consumes that chunk's private `phone_region_chunk_end_<chunk>-45` output. Each Region publishes a completion seq, and the matching L46 continuation waits for only that seq before it starts.

For three chunks the intended scheduling order is:

```
C2P_NORM(0)
C2P_FFN_REDUCE(0)
REGION(0)

C2P_NORM(1)          # PC-owned prework may overlap Region0
P2C_CONTINUE(0)      # waits only Region0, then immediately enters L46
C2P_FFN_REDUCE(1)
REGION(1)

C2P_NORM(2)
P2C_CONTINUE(1)
C2P_FFN_REDUCE(2)
REGION(2)

P2C_CONTINUE(2)
```

The L46 Phone branch may start PC FFN/return work asynchronously using the existing prefill workers, so after the Phone part of the continuation has been submitted/completed, later Region work can proceed while the previous chunk's PC/return path remains in flight.

The continuation consumes the same Region completion credit that previously existed only to protect scratch reuse. Once `P2C_CONTINUE(c)` has observed Region(c) complete, its scratch slot is explicitly released. Therefore a later chunk reusing that slot should no longer wait on the old producer:

```
[PHONE_REGION_CONTINUATION_CREDIT] chunk=0 seq=... marked=1 p2c_sg=...
[PHONE_REGION_CONTINUATION_WAIT] chunk=0 seq=... slot=0 wait_ms=... slot_released=1 status=READY
...
[PHONE_REGION_SCRATCH] chunk=2 slot=0 ... reuse_wait_ms=0... wait_mode=NONE status=BOUND
```

The pipeline plan now reports `continuation=1`, and each chunk reports its discovered continuation range as `p2c=[begin,end]`. Discovery fails closed on ambiguous or overlapping post-Region topology rather than reordering unknown subgraphs.

This continuation change does **not** change the RPC protocol; it still uses RPC 4.0.13 and the patch-13 completion-wait primitive.

Each Region stage now also prints:

```
[PHONE_REGION_HOST_BREAKDOWN] chunk=N
  pre_graph_ms=...
  graph_build_ms=...
  output_bind_ms=...
  scratch_bind_ms=...
  submit_ms=...
  post_submit_ms=...
  other_ms=...
  total_ms=...
```

This separates structural ownership scanning, fused graph assembly, descriptor rebinding, RPC graph submission, and post-submit Meta work. Do not interpret `submit_ms` as Phone GPU execution time; the RPC compute path is asynchronous.

The RPC backend already caches non-zero graph UIDs and sends `GRAPH_RECOMPUTE` on a cache hit. Region submissions now log a stable per chunk/slot key:

```
[PHONE_REGION_CACHE_KEY] chunk=N slot=S uid=... policy=PERSISTENT_SLOT_UID
```

Because scratch/output buffers are persistent, that UID describes stable remote addresses across repeated executions of the same graph shape. A new RPC graph-registration protocol is intentionally deferred: the measured `PHONE_REGION_GRAPH_DONE` time was already small, so prewarming allocation is the higher-priority fix. If later logs still show large `submit_ms`, inspect PC `[RPC_GRAPH] ... cmd=GRAPH_COMPUTE/GRAPH_RECOMPUTE` before adding protocol complexity.

If `[PHONE_REGION_PIPE_PLAN] active=0`, inspect the `reason`:
- `UNSUPPORTED_GRAPH`: missing, non-consecutive, or otherwise unsuitable SG arrangement.
- `OUTPUT_BUFFER_UNAVAILABLE`: the graph does not expose a materialized L45 chunk output that can be rebound safely.
- `PHONE_REGION_SCRATCH_REJECT`: the experimental pipeline was selected but private scratch/output binding could not be proven safe.
- `DISABLED`: both opt-in flags not set or another prerequisite not present.

A fallback is *not* a failure of the existing serial Phone Region graph. It intentionally runs the previous order to avoid silent buffer corruption.

## Chunk-local L46 Attention continuation

The first continuation attempt exposed a real graph barrier: L46 built one
full-batch Attention and only split the FFN afterward. Consequently FFN chunk0
still depended on Region1/2, and the dependency guard correctly fell back with
`CONTINUATION_FUTURE_REGION_INPUT`.

The Region path now carries each L45 chunk directly into an L46 chunk-local
Attention graph. L46 reuses the same range-specific KV index/mask input that
already drives L35-L45:

```
Region0 -> L46 Attn0 -> Router0 -> FFN0
Region1 -> L46 Attn1 -> Router1 -> FFN1
Region2 -> L46 Attn2 -> Router2 -> FFN2
```

For every chunk, the L46 graph performs Attention norm, Q/K/V, RoPE, KV writes,
causal Attention, residual, FFN norm, Router/Top-K and the P2C FFN boundary.
The model logs:

```
[PHONE_TENSOR_ATTN_CHUNK_PLAN] layer=46 chunks=... tokens=... mode=P2C status=READY
[PHONE_TENSOR_ATTN_CHUNK] layer=46 chunk=N begin=... tokens=... source=REGION_L45 status=BUILT
```

Meta starts each continuation at
`phone_tensor_attn_norm_chunk_N-46` and ends it at
`prefill_ffn_down_chunk_N-46`. The future-Region dependency walk remains
enabled as a correctness guard; seeing
`CONTINUATION_FUTURE_REGION_INPUT` is now a failure of the chunk-local
topology, not an expected fallback.

The shared L46 concat/residual tail remains after all chunk continuations so the
existing P2C reduction protocol keeps its previous final-output shape. For
three L46 chunks, `REGION_P2C_PREFILL` must report `last=0` for chunks 0/1
and `last=1` only for chunk 2.

## Required hardware validation

1. Build PC and Phone server from compatible commits as applicable; test 2-token warmup first. **Both the Region Graph and V1 pipeline now stay disabled for a single chunk**, so warmup follows the validated legacy Phone path. This is intentional: the single-chunk Region shape was not fused and exposed a Phone-side `GGML_OP_SET_ROWS` index assertion without providing any pipeline benefit.
2. Run the 358-token known prompt with `GGML_META_PHONE_REGION_PIPELINE=1`. Verify the stage order and all three `PHONE_REGION_GRAPH` submissions. Check full L46 P2C reduce, final logits and generated text.
3. Disable only the pipeline flag and repeat with **Phone Region graph still enabled**. This is the controlled serial-vs-pipeline comparison using identical graph mathematics.
4. Verify Layer 34, Layer 45 and Layer 46 outputs, chunk boundaries and KV positions if a numerical drift appears. Do not treat plausible text as conclusive parity.
5. Compare prompt eval throughput and the `META_LAYER_SUM` / `META_COPY_SUM` sections; note that Phone Region host submission walltime is not GPU kernel time.
6. Test token counts 128, 256, 358, 512 and repeated prompts. Decode remains on the existing non-pipeline path.

**No PC/Termux hardware run or build has been performed through this GitHub connector.** Do not assume a speedup until stage ordering, correctness, and actual GPU time have been measured.
