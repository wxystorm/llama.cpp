# Phone Full Region Graph (experimental)

Base commit: `d9706d8eb4895d87c58c72479487ace9428ee8dc` (verified directional C2P/P2C rollback).

This change deliberately leaves L34 C2P and L46 P2C math, partial transfer, and final-reduce ownership unchanged. It only changes the L35-L45 Qwen3MoE Phone Full middle.

## Execution

Set `LLAMA_HYBRID_PHONE_REGION_GRAPH=1` on the PC when running the 48-layer directional pair. This is off by default. It is eligible only for the single-sequence, prefill, non-LoRA, C2P(L34) + Phone Full(L35-L45) + P2C(L46) topology.

The Qwen3MoE graph now builds **chunk-major**: L35 through L45 in one contiguous forward DAG for chunk 0, then L35 through L45 for chunk 1, and so on. Every attention call uses `build_attn_inp_kv_range(token_begin, token_count)`, including its range-specific mask and KV insertion positions. We expand each completed chunk into the top-level graph in order, to preserve KV operations' graph order. `phone_region_chunk_end_{chunk}-45` is a deliberate Meta split point. The resulting device-local Phone chunk graph should be submitted once for that chunk, without any host intervention between L35 and L45.

At the Meta execution boundary a per-node check rejects any explicit Phone region graph containing a PC compute node. This is a correctness guard, not a recovery mechanism. If a guard trips, do not silence it: analyze the unexpected owner or spurious Meta boundary.

## First hardware verification (REQUIRED; not yet run)

1. Build and restart matching PC and Phone server binaries at the same branch commit. Compare protocol versions and hashes.
2. Enable `GGML_META_PIPELINE_DEBUG=1`; compare the output to the same prompt on the unchanged baseline with `LLAMA_HYBRID_PHONE_REGION_GRAPH` unset.
3. For 358 tokens with XT=128, expect `[PHONE_REGION_GRAPH_PLAN] ... chunks=3 ... status=READY`; each explicit `[PHONE_REGION_GRAPH]` should have exactly one Phone graph covering L35-L45 and zero PC compute nodes.
4. Verify no `[PHONE_ONLY_SG]` markers inside a Phone Region chunk, unless a deliberate subgraph boundary must be investigated. Check L34 final `l_out`, L45 output, L46 input, L46 P2C parity, full-vocabulary logits and generated text.
5. Run 1, 2, 16, 120, 128, 358 and 512 prompt-token cases; also run incremental decode. Decode uses the existing baseline path.
6. Inspect KV per-layer positions for all chunks and compare the last token of each chunk against the baseline. Ensure position/mask validity for the following chunks.
7. Check that graph UID/reuse, buffer liveness and scratch use are stable across repeated prompts; measure `META_GRAPH_TIMING`, `META_COPY_SUM`, `PHONE_REGION_GRAPH`, and final prompt throughput.

## Important limits

* This first implementation is a **single top-level graph** containing per-chunk Phone Region subgraphs. The outer C2P stage still processes the batch before the Phone Region starts, and the outer P2C stage still runs afterwards. **Inter-device per-chunk pipeline overlap is not implemented here**.
* This change does not provide explicit Phone scheduler registration, persistent buffers across different graph invocations, or independent concurrent OpenCL command queues.
* The branch has been edited through the GitHub connector; it has **not** been compiled or validated on PC/CUDA or Termux/OpenCL.
* The environment flag defaults off. Do not enable production traffic until correctness and stage profiling succeed.
