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

Only *one* L35-L45 graph is issued for each chunk, with KV work ordered chunk 0, 1, 2. PC-side C2P norm/router preparation for chunk N+1 may run while the earlier Phone graph has been submitted and the Phone GPU continues working; a **pre-existing Phone synchronize** during the next PC->Phone router transfer provides the GPU credit and ensures no C2P Phone FFN overlaps unsafe live Phone scratch. This is a bounded, correctness-first **window=2** stage pipeline, not a completely independent dual-queue GPU execution engine. In particular, full C2P vs full Phone overlap and P2C per-chunk pipelining are future steps, *not implemented here*.

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
- **Distinct memory:** each early Phone Region output must not alias a future C2P Phone writer. Missing backing pointers or alias collisions trigger a return to the original Meta execution order.

When a pipeline is activated, the previous legacy L34 `l_out-34` producer is disabled **at C2P chunk 0**, not at the final chunk. That producer would otherwise overwrite the Phone-owned, partially accumulated L34 output before Region chunk 0 reads it.

The graph's PC/Phone ownership and real COMPUTE checks continue to run in the existing Phone Region fused execution branch.

## Expected diagnostics, 358-token prefill

```
[PHONE_REGION_PIPE_PLAN] active=1 window=2 chunks=3 ... reason=READY
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

If `[PHONE_REGION_PIPE_PLAN] active=0`, inspect the `reason`:
- `UNSUPPORTED_GRAPH`: missing, non-consecutive, or otherwise unsuitable SG arrangement.
- `OUTPUT_BUFFER_ALIAS`: a live Region output shares storage with a future C2P Phone writer.
- `OUTPUT_BUFFER_UNAVAILABLE` or `UNKNOWN_FUTURE_BUFFER`: unable to prove memory safety.
- `DISABLED`: both opt-in flags not set or another prerequisite not present.

A fallback is *not* a failure of the existing serial Phone Region graph. It intentionally runs the previous order to avoid silent buffer corruption.

## Required hardware validation

1. Build PC and Phone server from compatible commits as applicable; test 2-token warmup first (V1 pipeline will stay disabled with one chunk).
2. Run the 358-token known prompt with `GGML_META_PHONE_REGION_PIPELINE=1`. Verify the stage order and all three `PHONE_REGION_GRAPH` submissions. Check full L46 P2C reduce, final logits and generated text.
3. Disable only the pipeline flag and repeat with **Phone Region graph still enabled**. This is the controlled serial-vs-pipeline comparison using identical graph mathematics.
4. Verify Layer 34, Layer 45 and Layer 46 outputs, chunk boundaries and KV positions if a numerical drift appears. Do not treat plausible text as conclusive parity.
5. Compare prompt eval throughput and the `META_LAYER_SUM` / `META_COPY_SUM` sections; note that Phone Region host submission walltime is not GPU kernel time.
6. Test token counts 128, 256, 358, 512 and repeated prompts. Decode remains on the existing non-pipeline path.

**No PC/Termux hardware run or build has been performed through this GitHub connector.** Do not assume a speedup until stage ordering, correctness, and actual GPU time have been measured.
