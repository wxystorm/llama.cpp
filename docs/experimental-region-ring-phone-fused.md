# Region Ring P5 Phone graph fusion (experimental, opt-in)

Base: `exp/region-ring-pc-phone-overlap-20261026` @ `2f4f8e63f9a8affd7aa088e4f7fd40b000cae0e8`. The previous successful 358-token run measured ~4007 ms Prefill and ~1669 ms of PC/Phone wall-clock overlap. Neither baseline branch nor baseline flags were modified.

## Scope: one remote execution per complete five-layer Region

`LLAMA_HYBRID_REGION_RING_EXEC=fifo-pc-phone` already permits PC CPU+CUDA work to overlap Meta/Phone work. This change adds an **optional client-side Meta graph fusion** of each complete five-layer Phone Region (for the validated P5 layout). The original Meta splitter still produces about 10 Phone-only attention/FFN SGs plus the terminal output bridge. Instead of 10 separate remote graph launches for those 10 SGs, the existing `build_phone_block_graph()` concatenates their nodes in original order and submits **one** RPC graph containing the five layers. The terminal `l_out` return/PC logits bridge remains under the existing communication code.

This uses the phone RPC server's **existing** UID-based `GRAPH_COMPUTE` / `GRAPH_RECOMPUTE` cache. It does not add or change the RPC wire protocol, and the server already has its own local GGML GPU computation. **It does not yet move the llama.cpp model Graph Builder to the phone, nor does the phone autonomously trigger a region upon arrival of READY.** Those are later, higher-risk steps. The immediate goal is to reduce redundant per-layer PC-to-phone graph command submission while preserving the already-observed PC/Phone overlap.

## Eligibility (reject partial fusion, preserve original path)

- Both flags must be present on PC: exact `fifo-pc-phone` mode and `GGML_META_RING_PHONE_FUSION=1`; the default is disabled. A safer eligibility-only mode is `GGML_META_RING_PHONE_FUSION=probe`: it prints `event=PROBE` for structurally valid P5 blocks but still executes every original unfused SG.
- Meta topology must be exactly two simple Backends: PC CPU plus Phone RPC. Phone SG pairs must be individually Phone-only. A mirrored PC compute node causes **the whole fusion to be skipped**, never partially dropped.
- Each pair must represent one Qwen3-style Attention/FFN layer. The recognizer now searches the paired SGs for the actual `ffn_inp-L` and `ffn_moe_out-L` nodes, rather than demanding that these be literal first/last SG nodes; the MoE output must still be in the second SG. The first SG must contain `norm-L`/`attn_norm-L` or previous `l_out-(L-1)`. Only multi-token graph shapes are eligible; Decode stays unmodified.
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
$env:GGML_META_RING_PHONE_FUSION='probe' # first run: eligibility only
$env:GGML_META_RING_PHONE_FUSION='1'     # second run: enable actual fusion

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

First run with `probe`: look for `[REGION_RING_PHONE_FUSION] event=PROBE ... action=KEEP_LEGACY status=ELIGIBLE`. This run should not produce `event=SUBMIT`. Then set the flag to `1` and compare. For 358 prompt tokens there are six Chunks and three P5 Regions, i.e. up to 18 P5 submissions (three additional P5 submissions per 2-token warmup run). Fusion logs may report `event=SKIP ... fallback=LEGACY_PHONE_SGS` if structural validation is not met. A skipped case is a deliberate correctness-preserving fallback. The Meta logical `subgraphs=11` count **does not** become 1: fusion reduces remote graph submissions, not the original Meta graph splitter's logical count.

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


## Diagnostic update after first SKIP-only run

The first opt-in test reported repeated `event=SKIP sg=0 layer=11/27/43` for all Phone Regions. The 358-token run still completed 54/54 jobs in **3772.754 ms**, with **1728.482 ms** observed PC–Phone interval overlap, no leaked Window=2 slots, and matching first three Top-1 token/logit pairs. **No Phone fusion `event=SUBMIT` was observed; do not attribute this run's timing to fusion.** Earlier broad `INCOMPLETE_OR_MIXED_OWNERSHIP` messages did not pinpoint the rejected condition.

The follow-up diagnostic commit `05c7222557251c1d5308804362d40c60035ce017` preserves the same execution and fusion eligibility, but reports **the exact first rejected SG** and its failure reason. For `probe`, it additionally prints up to 12 consecutive Meta SG descriptors: first/last node names, token dimension, phone-ownership classification, PC/Phone active node counts, and an example PC active operation if any. `probe` does not fuse and does not change the original graph execution.

Suggested short diagnostic run:

```powershell
git fetch origin
git switch exp/region-ring-phone-fused-20261026
cmake --build build-tensor-v2 --config Release --target llama-cli -j 8

$env:LLAMA_HYBRID_REGION_RING_EXEC = 'fifo-pc-phone'
$env:GGML_META_RING_PHONE_FUSION = 'probe'
# Run the same 358-token Prompt and server configuration.
Select-String -Path '.\region-pair.log' -Pattern '\[REGION_RING_PHONE_FUSION\]', '\[REGION_RING_PHONE_SG\]', '\[REGION_RING_SUM\]', '^top\[0\]:'
```

`GGML_META_RING_PHONE_FUSION=1` still activates real fusion only when *all* existing ownership checks pass. `GGML_META_RING_PHONE_FUSION_TRACE=1` enables the SG descriptors even when fusion is set to `1`. If failure says `PC_ACTIVE_NODE`, **do not** remove the guard until that operation's role in Meta's reduction/transfer semantics is understood. If it says `RING_FFN_MARKERS_OR_SHAPE`, inspect the logged actual first/last tensor names and adjust the recognizer to the real Qwen3-MoE graph rather than assuming a fixed SG boundary. `FFN_SG_NOT_PHONE_OWNED` can likewise reflect an actual cross-backend step; fusion must not skip it.

Do not change the verified `fifo-pc-phone` reference branch. Repeat the same A/B parity and timing tests only *after* eligibility is correctly established.

## Follow-up implementation: correct Qwen3-MoE markers (2026-10-26)

Previously every P5 candidate fell back with `event=SKIP` even though the independent PC/Phone workload completed correctly (358-token Prefill `3772.754 ms`, `1728.482 ms` concurrent PC/Phone intervals, `54/54` tasks, zero Window=2 slot leaks and the expected first three Top-1 logits). That run **did not** contain actual graph fusion.

The initial recognizer was stricter than the real Qwen3-MoE graph: `src/models/qwen3moe.cpp` labels the Attention residual as `ffn_inp-L` and MoE output as `ffn_moe_out-L`; `ggml-backend-meta.cpp` splits Meta subgraphs at `PARTIAL` and other graph boundaries, so those labels need not be at a subgraph's first or final node. The experimental recognizer now scans the actual nodes in the adjacent pair, checks that its two markers agree on layer `L`, requires `ffn_moe_out-L` inside the second subgraph with multi-token shape, and checks the first subgraph has a current-layer attention input marker. Existing PC compute ownership and complete-P5 checks remain enforced.

The `REGION_RING_PHONE_SG` diagnostic now records layer IDs for `norm`, `attn_norm`, `ffn_inp`, `ffn_moe_out`, and `l_out` in each SG. A `-1` means that marker was absent from the SG. The trace still includes PC and Phone active compute counts and the first PC-active operation, when present.

First run safely with `GGML_META_RING_PHONE_FUSION=probe` and confirm candidate classification without changing graph execution:

```powershell
$env:LLAMA_HYBRID_REGION_RING_EXEC = 'fifo-pc-phone'
$env:GGML_META_RING_PHONE_FUSION = 'probe'
# Keep the original model, RPC server, layout, prompt, seed and other flags
Select-String -Path '.\region-pair.log' -Pattern '\[REGION_RING_PHONE_FUSION\]', '\[REGION_RING_PHONE_SG\]', '\[REGION_RING_SUM\]', '^top\[0\]:'
```

`event=PROBE ... status=ELIGIBLE` means a complete five-layer candidate passed. The probe retains legacy per-SG execution, and an eligible PROBE is no longer followed by a misleading generic `SKIP`. If a candidate still prints `SKIP`, preserve its `failed_sg`, `reason` and the following `REGION_RING_PHONE_SG` records; do not disable guards just to force a fusion.

After successful probe, set `GGML_META_RING_PHONE_FUSION=1`, then verify `event=SUBMIT` / `event=DONE`, RPC graph command counts, output parity and repeated Prefill timing against `fifo-pc-phone` with the fusion flag unset. Because this graph-fusion code is newly changed and not yet built or tested on the user's devices, **no new performance gain or correct fused result is claimed**.


## Experiment: Phone READY-driven Region Ring execution (protocol v17)

This experiment is built on top of the validated first two fused P5 regions. It is intentionally separate from `GGML_META_RING_PHONE_FUSION=1`, and is **disabled by default**. It does not change `exp/region-ring-phone-fused-20261026` or the original overlap baseline.

### Design

- **PC prepares the graph**: the existing Meta fusion builds an exact ten-SG / five-layer Phone graph for L11–L15 or L27–L31. PC registers the serialized graph on the Phone RPC server by its UID. For a repeated UID on the same compute socket, registration is skipped. This is *lazy registration when each shape/slot graph is first encountered*; it is not yet a separate model-load-time prewarm pass.
- **PC transfers the input** using the existing Region boundary path. An explicit RPC Fence (not ggml_backend_synchronize(), which is a no-op for the RPC client backend) confirms the preceding input writes have reached the remote backend before READY is published. This conservative fence can be optimized only after Buffer Ownership is verified.
- **PC publishes `INPUT_READY`**, containing `device, first_layer, graph_uid, completion_seq`. This event does **not** invoke `GRAPH_COMPUTE` / `GRAPH_RECOMPUTE` on the client. It is a small control notification of input completion, not a compute command.
- **Phone MAIN worker receives READY and starts the already registered graph**. Execution uses the shared RPC graph/GPU gate so other Graph execution cannot interleave. After the device synchronizes, the Phone publishes a success/failure completion for the given sequence.
- **PC consumes the completion only at the exact next `l_out-15` or `l_out-31` bridge**, then executes the original `l_out` calculation and Phone→PC handoff. This prevents stale-output races. Other GPU/CPU tasks remain scheduled under the original Region Ring worker model.
- **Fallbacks remain unchanged**: L43–L47, single-token Decode, nonmatching shapes/ownership, and disabled auto mode all use the existing path. The server's legacy Region local queue remains intact.
- **KV/slot order** remains controlled by the existing outer Region Ring dependencies, `window=2` host buffer lifetime, and Phone MAIN FIFO. This first stage does not yet independently rebuild KV dependencies from serialized graph nodes. Never bypass the upstream Region task/slot credits.

### Build BOTH PC and Phone

The new RPC wire commands require **protocol 4.0.17**. The current RPC HELLO rejects mismatched versions, so PC and Phone must use the same new branch; do not reuse the Phone server binary from the previous protocol 4.0.16 branch.

PC (PowerShell):

```powershell
git fetch origin
git switch exp/region-ring-phone-ready-autostart-20261026
git pull --ff-only
cmake --build build-tensor-v2 --config Release --target llama-cli -j 8
```

Phone (Termux, adapt build directory if necessary):

```sh
git fetch origin
git switch exp/region-ring-phone-ready-autostart-20261026
git pull --ff-only
cmake --build build-region-v1 --target ggml-rpc-server -j 6
```

**Restart the Phone RPC server** after rebuilding, and keep the same model, Prompt, Seed, Chunk size, Region layout, PC and Phone flags from the known-good fused run.

### Mode selection

```powershell
$env:LLAMA_HYBRID_REGION_RING_V1 = '1'
$env:LLAMA_HYBRID_REGION_RING_EXEC = 'fifo-pc-phone'
$env:LLAMA_HYBRID_REGION_RING_LAYOUT = 'C7,G4,P5,C7,G4,P5,C8,G3,P5'
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX = '1'
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS = '64'
$env:GGML_META_RING_PHONE_FUSION = '1'
$env:GGML_META_RING_PHONE_AUTOSTART = '1'
```

Unset `GGML_META_RING_PHONE_AUTOSTART` to revert **only execution control** to the prior PC-initiated fused graph dispatch (after using matching protocol binaries); or switch both PC and Phone back to the prior branch for a wholly unchanged reference.

### Expected logs (two eligible P5 regions per Chunk)

```text
[PHONE_RING_AUTO] side=pc event=REGISTER layer=11 ... status=OK
[PHONE_RING_AUTO] side=phone event=GRAPH_REGISTERED ... status=OK
[PHONE_RING_AUTO] side=pc event=INPUT_READY layer=11 ... status=QUEUED
[PHONE_RING_AUTO] side=phone event=READY_ENQUEUED layer=11 ...
[PHONE_RING_AUTO] side=phone event=EXECUTE_DONE layer=11 ... status=OK
[PHONE_RING_AUTO] side=pc event=OUTPUT_WAIT ... status=OK
[PHONE_RING_AUTO] side=pc event=OUTPUT_READY layer=15 sg=10 ... status=OK
[REGION_RING_EXIT] layer=15 sg=10 action=COPY_COMPUTED_L_OUT ... status=OK
[REGION_RING_WORKERS_SUM] ... finished=54 completed=54 ...
[REGION_RING_OWNERSHIP_SUM] window=2 live_slots=0 ... status=DRAINED
```

For the 358-token Prefill (6 Chunks) expect 12 `INPUT_READY` / `EXECUTE_DONE` pairs, split across L11–L15 and L27–L31. `REGISTER` may occur fewer than 12 times if UID/shape/slot cache hits. `REGION_RING_PHONE_FUSION event=SUBMIT/DONE` is intentionally suppressed for this auto mode to avoid confusing *queued* with *computed*.

### Acceptance gates

1. Start with a 2-token warmup; verify registration, READY enqueue, Phone MAIN execution and only-then `l_out` output consumption.
2. Run a 358-token full Prefill. Require 54/54 Region jobs complete, `window=2 live_slots=0`, L15/L31 handoffs intact, no RPC EOF/deadlock or worker failure.
3. Compare generated output text, token IDs and as many logits as available against the known-good fused mode. Top-1 consistency alone is insufficient for numerical parity.
4. Compare multiple warmup-adjusted Release A/B Prefill timings. The conservative pre-READY synchronization and lazy first-time registration may cost more than the previous ~4 s result. Do not claim speedup until measured.
5. Confirm Phone server logs `READY_ENQUEUED` and `EXECUTE_DONE`, not just client `INPUT_READY`. If the Phone error flag is reported, the completion wait returns failure; do not continue to `l_out` with stale data.

This mode is the **first** autonomy milestone, not full Phone local Graph Builder. PC still constructs and registers each new graph template, and still sends the small INPUT_READY event after completing the transfer. Future phases may pre-register all shape/slot templates at model setup and attach READY directly to the final data-plane write.
