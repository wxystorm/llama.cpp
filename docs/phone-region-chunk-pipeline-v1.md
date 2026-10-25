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

## Required hardware validation

1. Build PC and Phone server from compatible commits as applicable; test 2-token warmup first. **Both the Region Graph and V1 pipeline now stay disabled for a single chunk**, so warmup follows the validated legacy Phone path. This is intentional: the single-chunk Region shape was not fused and exposed a Phone-side `GGML_OP_SET_ROWS` index assertion without providing any pipeline benefit.
2. Run the 358-token known prompt with `GGML_META_PHONE_REGION_PIPELINE=1`. Verify the stage order and all three `PHONE_REGION_GRAPH` submissions. Check full L46 P2C reduce, final logits and generated text.
3. Disable only the pipeline flag and repeat with **Phone Region graph still enabled**. This is the controlled serial-vs-pipeline comparison using identical graph mathematics.
4. Verify Layer 34, Layer 45 and Layer 46 outputs, chunk boundaries and KV positions if a numerical drift appears. Do not treat plausible text as conclusive parity.
5. Compare prompt eval throughput and the `META_LAYER_SUM` / `META_COPY_SUM` sections; note that Phone Region host submission walltime is not GPU kernel time.
6. Test token counts 128, 256, 358, 512 and repeated prompts. Decode remains on the existing non-pipeline path.

**No PC/Termux hardware run or build has been performed through this GitHub connector.** Do not assume a speedup until stage ordering, correctness, and actual GPU time have been measured.
