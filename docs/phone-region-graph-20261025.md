# Phone Full Region Graph (experimental)

Base commit: `d9706d8eb4895d87c58c72479487ace9428ee8dc` (verified directional C2P/P2C rollback).

This change deliberately leaves L34 C2P and L46 P2C math, partial transfer, and final-reduce ownership unchanged. It only changes the L35-L45 Qwen3MoE Phone Full middle.

## Execution

Set `LLAMA_HYBRID_PHONE_REGION_GRAPH=1` on the PC when running the 48-layer directional pair. This is off by default. It is eligible only for the single-sequence, prefill, non-LoRA, C2P(L34) + Phone Full(L35-L45) + P2C(L46) topology.

The Qwen3MoE graph now builds **chunk-major**: L35 through L45 in one contiguous forward DAG for chunk 0, then L35 through L45 for chunk 1, and so on. Every attention call uses `build_attn_inp_kv_range(token_begin, token_count)`, including its range-specific mask and KV insertion positions. We expand each completed chunk into the top-level graph in order, to preserve KV operations' graph order. `phone_region_chunk_end_{chunk}-45` is a deliberate Meta split point. The resulting device-local Phone chunk graph should be submitted once for that chunk, without any host intervention between L35 and L45.

At the Meta execution boundary, `PARTIAL` state markers can still create *internal* Meta subgraphs. The execution engine now detects a chunk by `phone_region_attn_chunk_{chunk}-35`, scans until `phone_region_chunk_end_{chunk}-45`, and fuses all of the Phone subgraphs into **one RPC graph submission** with no intermediate handoff or all-reduce. Every node is checked for PC compute ownership before fusion; an unexpected PC compute node prints `[PHONE_REGION_GRAPH_REJECT] ... reason=PC_COMPUTE_ACTIVE` and stops, instead of silently dropping required PC work.

For a **single-chunk** request, `l_out-45` uses a separate zero-copy view to preserve the terminal `phone_region_chunk_end_0-45` marker. This fixes the missing end marker discovered in the 2-token warmup path.

The changes are still uncompiled and unverified on Windows/Phone hardware. They specifically address the earlier failure in which an internal `Kcur-41 (view)` was mistaken for a Phone-to-PC layer output at `ggml-backend-meta.cpp:8885`.

## Fix after first 358-token execution failure

The previous test successfully completed C2P prefill chunks 0/1/2 but stopped at:
`[PHONE_REGION_GRAPH_REJECT] chunk=0 sg=7 node=phone_region_input-35 op=CONT reason=PC_COMPUTE_ACTIVE`.

That `CONT` made a new full-size activation from `l_out-34`. On this Meta layout it was flagged COMPUTE on both PC and Phone; the PC copy would have used a stale shadow of a Phone-owned C2P output, so disabling the ownership guard would be unsafe.

The graph builder now **removes the full-batch CONT** and directly makes chunk views of the contiguous Phone-owned C2P result. Non-contiguous input is rejected with an explicit assertion rather than silently copied. The Meta fusion ownership validator still rejects every real PC compute node (especially CONT/CPY/MUL_MAT); only inherently metadata-only PC operations (VIEW/RESHAPE/PERMUTE/TRANSPOSE) may have their inherited COMPUTE flags ignored, with a `[PHONE_REGION_GRAPH_OWNERSHIP]` summary.

Next run should report `[PHONE_REGION_GRAPH_OWNERSHIP] chunk=0 ... pc_compute=0 status=SAFE` and `[PHONE_REGION_GRAPH_DONE] chunk=0 ... status=0`, followed by chunks 1 and 2. Continue to check output parity, KV ordering, and buffer lifetime before treating the fused graph as verified.

## First hardware verification (REQUIRED; not yet run)

1. Build and restart matching PC and Phone server binaries at the same branch commit. Compare protocol versions and hashes.
2. Enable `GGML_META_PIPELINE_DEBUG=1`; compare the output to the same prompt on the unchanged baseline with `LLAMA_HYBRID_PHONE_REGION_GRAPH` unset.
3. First verify 2-token warmup: expect `[PHONE_REGION_GRAPH] chunk=0 ... layers=[35,46) ... status=DEVICE_LOCAL_READY` and `[PHONE_REGION_GRAPH_DONE] ... status=0`; then test 358 tokens with XT=128 and expect `[PHONE_REGION_GRAPH_PLAN] enabled=1 ... chunks=3 ... status=READY` plus three fused region submissions. If any `PHONE_REGION_GRAPH_REJECT` appears, inspect the exact PC-active node and do not ignore the error.
4. Verify no `[PHONE_ONLY_SG]` markers inside a Phone Region chunk, unless a deliberate subgraph boundary must be investigated. Check L34 final `l_out`, L45 output, L46 input, L46 P2C parity, full-vocabulary logits and generated text.
5. Run 1, 2, 16, 120, 128, 358 and 512 prompt-token cases; also run incremental decode. Decode uses the existing baseline path.
6. Inspect KV per-layer positions for all chunks and compare the last token of each chunk against the baseline. Ensure position/mask validity for the following chunks.
7. Check that graph UID/reuse, buffer liveness and scratch use are stable across repeated prompts; measure `META_GRAPH_TIMING`, `META_COPY_SUM`, `PHONE_REGION_GRAPH`, and final prompt throughput.

## Important limits

* This first implementation is a **single top-level graph** containing per-chunk Phone Region subgraphs. The outer C2P stage still processes the batch before the Phone Region starts, and the outer P2C stage still runs afterwards. **Inter-device per-chunk pipeline overlap is not implemented here**.
* This change does not provide explicit Phone scheduler registration, persistent buffers across different graph invocations, or independent concurrent OpenCL command queues.
* The branch has been edited through the GitHub connector; it has **not** been compiled or validated on PC/CUDA or Termux/OpenCL.
* The environment flag defaults off. Do not enable production traffic until correctness and stage profiling succeed.
