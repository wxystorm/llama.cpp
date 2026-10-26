# Region Ring P5 Phone graph fusion (experimental, opt-in)

Base: `exp/region-ring-pc-phone-overlap-20261026` @ `2f4f8e63f9a8affd7aa088e4f7fd40b000cae0e8`. The previous successful 358-token run measured ~4007 ms Prefill and ~1669 ms of PC/Phone wall-clock overlap. Neither baseline branch nor baseline flags were modified.

## Scope: one remote execution per complete five-layer Region

`LLAMA_HYBRID_REGION_RING_EXEC=fifo-pc-phone` already permits PC CPU+CUDA work to overlap Meta/Phone work. This change adds an **optional client-side Meta graph fusion** of each complete five-layer Phone Region (for the validated P5 layout). The original Meta splitter still produces about 10 Phone-only attention/FFN SGs plus the terminal output bridge. Instead of 10 separate remote graph launches for those 10 SGs, the existing `build_phone_block_graph()` concatenates their nodes in original order and submits **one** RPC graph containing the five layers. The terminal `l_out` return/PC logits bridge remains under the existing communication code.

This uses the phone RPC server's **existing** UID-based `GRAPH_COMPUTE` / `GRAPH_RECOMPUTE` cache. It does not add or change the RPC wire protocol, and the server already has its own local GGML GPU computation. **It does not yet move the llama.cpp model Graph Builder to the phone, nor does the phone autonomously trigger a region upon arrival of READY.** Those are later, higher-risk steps. The immediate goal is to reduce redundant per-layer PC-to-phone graph command submission while preserving the already-observed PC/Phone overlap.

## Eligibility (reject partial fusion, preserve original path)

- Both flags must be present on PC: exact `fifo-pc-phone` mode and `GGML_META_RING_PHONE_FUSION=1`; the default is disabled.
- Meta topology must be exactly two simple Backends: PC CPU plus Phone RPC. Phone SG pairs must be individually Phone-only. A mirrored PC compute node causes **the whole fusion to be skipped**, never partially dropped.
- Each pair must represent one Qwen3-style attention subgraph starting at `norm-L` or `l_out-(L-1)`, followed by an `ffn_inp-L` to `ffn_moe_out-L` Phone subgraph, and have multi-token input (single-token Decode stays on the unmodified path).
- Five pairs must be consecutive (10 SGs, five consecutive layers). There must be a following output bridge or PC consumer. Partial P5 graphs continue through the original un-fused SG execution.
- The existing Tensor Phone-primary protection remains intact; do not fuse a split FFN or graph that requires intermediate PC work.
- `Window=2` Slot credits, KV chunk ordering, Meta output bridge, Phone `l_out` residual handling, and final MAIN Graph Result remain unchanged.

## PC build and run

```powershell
git fetch origin
git switch exp/region-ring-phone-fused-20261026
cmake --build build-tensor-v2 --config Release --target llama-cli -j 8
cmake --build build-tensor-v2 --config Debug --target test-region-ring -j 8
ctest --test-dir build-tensor-v2 -C Debug -R '^test-region-ring$' --output-on-failure

$env:LLAMA_HYBRID_REGION_RING_V1='1'
$env:LLAMA_HYBRID_REGION_RING_LAYOUT='C7,G4,P5,C7,G4,P5,C8,G3,P5'
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS='64'
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX='1'
$env:LLAMA_HYBRID_REGION_RING_EXEC='fifo-pc-phone'
$env:GGML_META_RING_PHONE_FUSION='1'

# Use exactly the same Qwen3-30B-A3B-Q4_K_M GGUF, Prompt, seed, num tokens
# and existing Phone RPC server. No protocol changes are required.
```

Observe diagnostics in both PC and optionally phone logs:

```text
[REGION_RING_PHONE_FUSION] event=SUBMIT sg=0..9 layers=[11,16) meta_sgs=10 rpc_graphs=1 nodes=... uid=... status=READY
[REGION_RING_PHONE_FUSION] event=DONE sg=0..9 layers=[11,16) submit_ms=... status=0
[RPC_GRAPH] ... nodes=... cmd=GRAPH_COMPUTE
[RPC_GRAPH] ... nodes=... cmd=GRAPH_RECOMPUTE
[REGION_RING_PC_PHONE_SUM] ... overlap_ms=... status=PASS
[REGION_RING_WORKERS_SUM] created=3 joined=3 finished=54 completed=54 ... status=PASS
[REGION_RING_OWNERSHIP_SUM] window=2 live_slots=0 finished=54 status=DRAINED
[REGION_RING_SUM] tokens=358 chunks=6 executed=54 complete=1 ... status=0
```

For 358 prompt tokens there are six Chunks and three P5 Regions, i.e. up to 18 P5 submissions (three additional P5 submissions per 2-token warmup run). Fusion logs may report `event=SKIP ... fallback=LEGACY_PHONE_SGS` if structural validation is not met. A skipped case is a deliberate correctness-preserving fallback. The Meta logical `subgraphs=11` count **does not** become 1: fusion reduces remote graph submissions, not the original Meta graph splitter's logical count.

### Compact PowerShell log filter

```powershell
$log='.\ring-phone-fused.log'
Select-String -Path $log -Pattern '\[REGION_RING_PHONE_FUSION\]', '\[RPC_GRAPH\]', '\[REGION_RING_PC_PHONE_SUM\]', '\[REGION_RING_WORKERS_SUM\]', '\[REGION_RING_OWNERSHIP_SUM\]', '\[REGION_RING_SUM\]', '\[REGION_RING_EXIT\]', '^decoded=', '^top\[0\]:', 'event=FAIL', 'event=REJECT', 'event=EXCEPTION' | ForEach-Object { $_.Line } | Set-Content '.\phone-fusion-check.txt'
```

## Acceptance and performance comparison

1. Run the **same** Release binary / RPC server / model / prompt with the new fusion flag removed, then with it set. Repeat both after warmup; compare medians of `REGION_RING_SUM wall_ms`, `prompt eval time`, and `REGION_RING_PC_PHONE_SUM overlap_ms`. One 4.007 s run is a useful starting benchmark, not a guaranteed speed.
2. Require `finished=54 completed=54`, no Slot leak, no `status=FAIL`, and valid L15/L31 output bridges. Verify the first three Top-1 results against the validated run: `151667 @ 28.507137299`, `198 @ 43.090492249`, `32313 @ 38.111267090`. Passing Top-1 alone does **not** prove full numerical parity. Check more token/logit values and, when enabled, per-layer tensor parity.
3. On the first successful fused run, inspect `[RPC_GRAPH] nodes=...` to verify the server receives one large graph per eligible P5 region instead of multiple tiny per-layer graphs; expect later repeated-UID submissions to use `GRAPH_RECOMPUTE` when cache addresses and graph identities remain valid.
4. For any wrong logits, memory pressure, KV position regression, remote EOF or Phone `l_out` mismatch: unset the fusion flag and recheck known-good `fifo-pc-phone`. Do **not** hide an ownership failure by turning off its eligibility checks.

## Next: actual on-phone autonomous execution

Once fused Graph parity and performance pass, the following distinct feature can be developed: pre-register a **shape/slot/KV-safe** graph template on the Phone server, bind incoming Chunk data in a Phone-owned Buffer, use transfer-complete READY plus ordered KV credits to enqueue that template in `phone_region_main_loop()`, and return a completion event without a new PC Graph command. Only after that should the model-building logic be moved to Phone CPU to construct P Regions locally from Qwen3 metadata and GGUF weights. Existing `phone_region_main_loop()` and `stored_graphs` are foundations but do not yet implement this for Region Ring.
