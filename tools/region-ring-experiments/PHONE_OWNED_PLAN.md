# Phone-owned Qwen3 Layer Graph: phase 2 (shadow)

This branch moves the **graph description and construction logic** toward the
Phone without changing the currently working PC-built Graph execution path.

## Implemented

- RPC patch **28** has a versioned per-Region `region_plan`. No graph or
  tensor pointers cross this control RPC.
- The Phone records the plan for the active input stream, rejects conflicting
  re-registration, validates every legacy Chunk registration against it, and
  clears descriptors on stream teardown.
- `phone-qwen3-local-builder.h` builds a complete dense Qwen3 Transformer
  Layer *from Phone-resident GGUF weights*, using GGML operations:
  RMSNorm, Q/K/V matmul, per-head Q/K Norm, NeoX RoPE, K/V `SET_ROWS`,
  F16 Flash Attention, output projection, parallel SwiGLU FFN, and residuals.
- Side inputs (position, K/V indices, F16 causal mask) are generated locally
  from the immutable plan. No PC-supplied attention mask is required in this
  **fresh single-sequence prefill** subset.
- An optional Phone-side check creates one local graph per exact Chunk shape
  and verifies operator counts without HTP allocations.
- A **separate opt-in** HTP prepare step allocates persistent Phone-owned
  KV and fixed input/side-input/output buffers, fills positions/masks locally,
  and precompiles every local Chunk graph via the Hexagon fixed-graph API.
  It still does **not execute** these graphs or publish their results; the
  existing PC-built Graph path remains the only live execution path.

## Strict first-slice restrictions

Dense Qwen3, one complete Transformer Layer per Phone Region, F32 hidden
activations, F16 non-transposed K/V cache, Flash Attention, NeoX RoPE, one
sequence starting at position 0, no KV prefix reuse or shifting, no LoRA,
no additional per-layer features, and a single-output Phone Layer.

The PC still constructs and registers all old RPC Graphs. Keep the original
graph path for baseline performance and numerical parity.

## Enable both shadow layers

Build **both PC and Phone from the same commit** (patch 28 HELLO rejects
other versions). The Phone must start with a local copy of the *same model*
and have the matching layers resident in HTP before the PC registers plans.

Phone environment, shown for a four-Region layout with Phone Layers
9, 19, 29, 39 (adjust Region indices to match the actual runtime plan):

```sh
export GGML_RPC_PHONE_NPU_SERIAL_REGIONS=1,3,5,7
export GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS=9,19,29,39
export GGML_RPC_PHONE_QWEN3_LOCAL_BUILDER_SHADOW=1
# Additional memory/time-intensive HTP precompile smoke only:
# export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
```

PC environment (in addition to existing autonomous, input PUSH stream, and
Phone return stream experiment settings):

```powershell
$env:LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW = "1"
$env:LLAMA_REGION_RING_PHONE_ASYNC_GRAPH_SETUP = "1"
```

For the proposed first smoke test, use 512 prompt tokens, 128-token
macro-chunks, and `G9,P1,G9,P1,G9,P1,G9,P1`. Ensure the same GGUF weights
are available on both devices and Flash Attention is enabled.

Expected new Phone log:

```
[PHONE_QWEN3_LOCAL] event=GRAPH_BUILD ... mode=SHADOW_NO_HTP_ALLOC status=PASS
# Only with GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1:
[PHONE_QWEN3_LOCAL] event=HTP_PRECOMPILE ... mode=SHADOW_FIXED_STORAGE status=PASS
[PHONE_OWNED_PLAN] event=REGISTER ... graph_mode=SHADOW_PC_BUILT local_builder=GRAPH_BUILT status=READY
```

The GRAPH_BUILD message indicates **local graph construction validated**.
The optional HTP_PRECOMPILE message indicates backend allocation and graph
precompilation, **not** numerical parity or local compute execution.
Both modes retain the original PC-built Graph as the only compute owner.
Precompilation currently occurs synchronously during plan registration and
is therefore intentionally not a performance optimization yet.

## Model-free unit tests

```sh
cmake -S . -B build-phone-owned -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-phone-owned --target test-phone-owned-execution-plan test-phone-qwen3-local-builder
ctest --test-dir build-phone-owned -R 'test-phone-(owned-execution-plan|qwen3-local-builder)' --output-on-failure
```

Tests cover plan validation/teardown, exact Chunk intervals including the tail,
per-position F16 causal masks, weight shapes, graph output dimensions and
presence of two `SET_ROWS`, two RoPE nodes, one Flash Attention and one SwiGLU.


## Phase 3: REAL Phone-local HTP A/B execution (opt-in)

After the above GRAPH_BUILD/HTP_PRECOMPILE phases pass, enable one extra
environment variable on the **Phone** (not on the PC), then restart the RPC
server:

```sh
export GGML_RPC_PHONE_QWEN3_LOCAL_BUILDER_SHADOW=1
export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
export GGML_RPC_PHONE_QWEN3_LOCAL_EXEC_SHADOW=1
```

Use the *same* PC run command, fixed Region layout, model, and prompt.
Every verified 128-token Chunk now undergoes **two real HTP computations**:
1. Baseline PC-built Graph executes first, updating its canonical KV and
   OpenCL output. This remains the only output published over P2C.
2. Phone-local Graph reads the same verified OpenCL input into its own
   pre-bound HTP activation buffer, executes with independent Phone-owned KV
   (strict Region FIFO), then compares against the baseline OpenCL output.

The new per-Chunk logs are:

```
[PHONE_QWEN3_LOCAL] event=EXECUTE_AB ... htp_compute_ms=... max_abs=... mean_abs=... rmse=... cosine=... nonfinite=0 parity=MEASURED_ONLY p2c=LEGACY status=PASS reason=NONE
[PHONE_QWEN3_LOCAL] event=FEATURE_DIAG ... worst_feature=731 worst_token=... ref_worst=... local_worst=... feature_rmse=... feature_mse_share=... dominant_feature=... dominant_mse_share=... rmse_excl=... cosine_excl=... max_abs_excl=... status=PASS
```

The additional FEATURE_DIAG line reuses the already downloaded F32
activations. It identifies the feature at the worst absolute difference,
attributes total squared error to that feature, finds the highest-energy
mismatched feature, and recomputes RMSE/Cosine while excluding the
worst-index feature across **all** tokens. In Qwen3-14B, the observed
`worst_index % 5120` was 731 for all 16 Region-Chunks; this is
not proof of corrupted weights, and these new metrics help distinguish
an isolated outlier channel from a widespread computation mismatch.

`status=PASS` only means local HTP compute succeeded and both outputs are
finite; `parity=MEASURED_ONLY` means no numerical tolerance is being
silently asserted. Check all 16 Chunk rows for each metric, especially
Chunk 0 and Chunk 1 after the KV cache starts accumulating history.

Safety and limitations:
- A/B is serial, intentionally slower, and does NOT measure the final
  standalone Phone-owned inference speed.
- A failure in the opted-in Phone-local Graph aborts the experimental run;
  leaving the variable unset completely preserves the known-good path.
- A/B still uses the PC-built Graph to bind the reference activation, hold
  legacy KV for decode, and expose the P2C output. The next change is to
  replace these dependencies after parity results are understood.
- Do not reuse or merge Phone-local KV into canonical KV without a validated
  ownership and decode transition.
- This stage does not implement two-slot overlap or local-output P2C.

## Diagnosing HTP stage mismatch (after FEATURE_DIAG)

The 512-token Qwen3-14B A/B test showed the worst absolute error at
hidden feature `731` across all 16 (Region, Chunk) outputs. The FEATURE_DIAG
metrics confirmed that this feature accounts for only 17-19% of L39
squared error, and excluding it does not repair L39 cosine similarity.

To locate the **first divergent HTP stage**, optionally trace one Layer/Chunk:

```sh
# On Phone, in addition to the existing three local A/B switches:
export GGML_RPC_PHONE_QWEN3_STAGE_DIAG=1
export GGML_RPC_PHONE_QWEN3_STAGE_DIAG_LAYER=39
export GGML_RPC_PHONE_QWEN3_STAGE_DIAG_CHUNK=0
```

Restart the Phone RPC server, and run the exact same 512-token PC sweep.
Inspect:

```powershell
Select-String -Path ".\\npu.log" -Pattern "event=STAGE_DIAG" |
    ForEach-Object { $_.Line }
```

The stage rows (where present in both Graphs) are:
`inp_stage`, `attn_norm`, `Qcur`, `Kcur`, `Vcur`,
`attn_out`, `ffn_inp`, `ffn_norm`, `ffn_up`,
`ffn_gate`, `ffn_swiglu`, `ffn_down`.
Every row reports element count, MaxAbs, RMSE, Cosine,
reference/local RMS and a non-numeric `state`.

This diagnosis reads **both sets of intermediate tensors from the HTP
backend** after the two graphs have executed. It requires the original
PC-derived graph to use `GGML_RPC_PHONE_NPU_FIXED_CHUNK_GRAPHS=1`;
otherwise the baseline scratch graph no longer exists for comparison,
and a `REFERENCE_FIXED_GRAPH_NOT_AVAILABLE` row is reported.
Intermediate tensors with missing/ambiguous labels or incompatible
layout are reported explicitly, never treated as equal.

First examine `inp_stage`. A mismatch here means the two HTP graphs
consumed different input activations, independent of attention math.
If input matches but `attn_norm` diverges, focus on RMSNorm/weights;
if `Qcur`/`Kcur` diverge, examine Q/K projections, normalization and
RoPE; if `attn_out` is the first divergence, check KV/mask/FA.
If attention matches but FFN diverges, inspect gate/up/swiglu/down.

CAUTION: this is an expensive, diagnostic-only readback of intermediate
HTP tensors. Target L39 Chunk 0 first rather than all 16 Chunk graphs.
Stage data are observational only: the original Graph remains the
sole P2C publisher.

## A/B input alias regression fix (2026-11-06)

The first targeted L39 Chunk 0 stage trace showed:

- `inp_stage`: cosine 0.867666, RMSE 15.3106513, max abs 6615.25.
- `attn_norm` and all subsequent observable FFN stages also diverged.
- `attn_out` read as zero on BOTH sides, which is not trustworthy
  intermediate parity; zeroed/unmaterialized HTP scratch can look identical.

Root-cause hypothesis, supported by the source: the old A/B path **ran
the PC-built HTP graph first**, then read `task->input_graph_tensor`
from OpenCL for the Phone-local HTP graph. That OpenCL storage can be
reused or aliased by the canonical Graph's output; the local Graph may
have received a different activation than the baseline.

The A/B harness now does a safer and directly verifiable sequence:

1. After the baseline's input leaf is copied into its **actual HTP
   target Tensor**, capture the target F32 bytes before executing the
   baseline HTP graph, with a backend sync.
2. Execute the baseline graph (unchanged canonical KV + P2C semantics).
3. Upload the immutable pre-compute snapshot to Phone-local HTP input.
4. Execute local graph, compare outputs, and (if opted in) inspect stages.

When `GGML_RPC_PHONE_QWEN3_LOCAL_EXEC_SHADOW=1` is set, a *fixed*
baseline graph and an exact input leaf match are required. The test
fails closed with `PHONE_LOCAL_AB_REQUIRES_FIXED_REFERENCE_GRAPH` or
`HTP_FIXED_AB_STAGE_INPUT_NOT_FOUND_OR_INVALID` if the safe capture
is unavailable. It does NOT silently fall back to reading the OpenCL
input after compute.

New `EXECUTE_AB` lines include
`input_source=BASELINE_HTP_PRE_COMPUTE`. The `input_read_ms` field
now measures this pre-compute HTP snapshot, not the old post-compute
OpenCL read.

Re-run the **same 512-token prompt** and first check `STAGE_DIAG
stage=inp_stage`: its RMSE should approach zero with cosine
approaching one, provided the backend preserves these input buffers
for diagnostic readback. If the input matches but output still diverges,
diagnose Attention/KV/FFN math. Any
`state=ZERO_INTERMEDIATE_UNVERIFIED` is not a pass.

## Phase 4: Phone-local HTP as the actual P2C source (opt-in)

The 2026-11-06 Qwen3-14B, 512-token A/B smoke test produced 16/16
finite output comparisons with exact `MaxAbs=0, RMSE=0, Cosine=1`
for Phone Layers 9, 19, 29 and 39 (four 128-token chunks per Region).
The immutable baseline-HTP input capture is required for this result.

After rebuilding/deploying the latest Phone server, enable **on Phone**:

```sh
export GGML_RPC_PHONE_NPU_FIXED_CHUNK_GRAPHS=1
export GGML_RPC_PHONE_QWEN3_LOCAL_BUILDER_SHADOW=1
export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
export GGML_RPC_PHONE_QWEN3_LOCAL_EXEC_SHADOW=1
export GGML_RPC_PHONE_QWEN3_LOCAL_P2C=1
# The expensive intermediate Stage diagnostics are optional:
export GGML_RPC_PHONE_QWEN3_STAGE_DIAG=0
```

Run the **same PC** 512-token, Chunk 128, forced 4-Phone-Region
smoke case. The Phone compute thread now:

1. Runs the reference PC-built Graph to maintain canonical KV for
   subsequent decode and snapshot the exact HTP input.
2. Runs the independently built Phone-local HTP Graph, using its own
   KV and the frozen input snapshot, and computes A/B parity.
3. Requires **exact F32 parity**, `count > 0`, `nonfinite=0` and
   `max_abs=0`; any failure **aborts** (no fallback).
4. Copies the local HTP `l_out` Tensor into the existing per-Chunk
   OpenCL output buffer and synchronizes the OpenCL backend.
5. Uses the **unchanged Phone autonomous P2C capture** to deliver that
   buffer. The output source now comes from Phone-local HTP, while
   the PC-side graph registration and canonical KV are still used.

Expected per-Chunk log:

```
[PHONE_QWEN3_LOCAL] event=EXECUTE_AB ... p2c=LOCAL_CANDIDATE max_abs=0 status=PASS
[PHONE_QWEN3_LOCAL] event=P2C_SOURCE ... source=PHONE_LOCAL_HTP canonical_kv=LEGACY strict_parity=EXACT max_abs=0 copy_ms=... status=PASS reason=NONE
```

There should be **16 `event=P2C_SOURCE` lines** with `status=PASS` and
normal PC summary output. Anything else means cutover has not passed.
These lines are generated before the actual P2C publication call; also
check the existing `PHONE_AUTO_P2C` log for absence of `ABORT`.

Note that this is a **P2C output-source cutover**, not yet a full Phone-only
compute cutover. The baseline graph still computes on HTP for each
Chunk and PC still registers its graphs. This first phase deliberately
preserves decode and local KV state semantics. It is slower than
standalone execution because it runs two graphs and compares outputs.

To restore reference-only P2C, set:

```sh
export GGML_RPC_PHONE_QWEN3_LOCAL_P2C=0
```

Restart the Phone RPC server after changing any of the switches.

## Phase 5: one Phone-local HTP compute per Chunk, local KV to Decode KV

After the Phase 4 output-source cutover passes **both prefill and decode**,
the next experimental switch eliminates the redundant PC-derived HTP
Graph computation. PC still provides Region Plan and the registered
OpenCL input/output metadata, so this is *local compute-only*, not
yet a protocol-level PC Graph-registration removal.

Recompile and redeploy the Phone server, and use this **Phone-only**
configuration (retain the known-good NPU Region and model GGUF settings):

```sh
export GGML_RPC_PHONE_NPU_SERIAL_REGIONS=1,3,5,7
export GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS=9,19,29,39
export GGML_RPC_PHONE_NPU_PRELOAD_WEIGHTS=1
export GGML_RPC_PHONE_NPU_PRELOAD_ASYNC=1

export GGML_RPC_PHONE_QWEN3_LOCAL_BUILDER_SHADOW=1
export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1

# Crucial: no baseline HTP execution, no A/B, no old fixed-graph compile.
export GGML_RPC_PHONE_QWEN3_LOCAL_EXEC_SHADOW=0
export GGML_RPC_PHONE_QWEN3_LOCAL_P2C=0
export GGML_RPC_PHONE_NPU_FIXED_CHUNK_GRAPHS=0
export GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1

export GGML_RPC_PHONE_QWEN3_STAGE_DIAG=0
```

PC settings and the 512-token fixed-layout test remain unchanged.
Keep `LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW=1` on the PC:
it still sends the immutable per-Region Plan.

The new execution is:

1. PC PUSH / verified local OpenCL Activation (unchanged).
2. Phone validates the original PC Graph's canonical Decode K/V
   Tensor storage and each Chunk's actual I64 KV slot indices.
   **Both** physical K and V indices must exactly equal the plan's
   fresh positions `chunk * 128 + [0..127]`. A different physical slot
   mapping aborts the run instead of silently corrupting Decode.
3. Phone reads the input **before any computation**, uploads it to its
   separately precompiled HTP graph, computes **once**, and writes the
   local `l_out` into the existing per-Chunk OpenCL output.
4. The proven autonomous P2C captures that OpenCL output normally.
5. On end-of-prefill `phone_input_stream_run(active=false)`, the
   server copies **only the populated F16 KV prefix** from the Phone-local
   HTP K/V buffers into the authoritative OpenCL Decode K/V tensors,
   synchronizes OpenCL, then releases the local Graphs/KV.
   The ordinary PC-controlled decode keeps working without modification.

Expected on the Phone:

```
[PHONE_QWEN3_LOCAL] event=KV_BIND ... source=PC_GRAPH_METADATA_ONLY status=PASS
[PHONE_QWEN3_LOCAL] event=EXECUTE_ONLY ... legacy_htp_compute=SKIPPED kv_owner=PHONE_LOCAL status=PASS
[PHONE_NPU_SERIAL] event=EXECUTE ... mode=PHONE_LOCAL_ONLY status=PASS
[PHONE_QWEN3_LOCAL] event=KV_SYNC ... from=PHONE_LOCAL_HTP to=OPENCL_DECODE mode=LOCAL_ONLY regions=4 status=PASS
```

For a 512-token, 128-token Chunk, four Region test: **16 KV_BIND,
16 EXECUTE_ONLY**, one end-of-prefill **KV_SYNC** with four Regions,
**no ABORT**, and PC `summary.csv` showing successful prefill and
nonzero, correct decode. The result must be compared with Phase 4 for
the SAME prompt; especially verify decode output/logits equivalence,
not only the `status=PASS` and timing fields.

Important limitations:
- This first local-only experiment is restricted to Qwen3, Flash
  Attention, F16 KV, single fresh sequence starting at position 0,
  contiguous identity KV slot assignment, and the tested Region Plan.
  Prefix reuse, non-identity slots, sliding-window KV, mixed sequences,
  and noncontiguous K/V storage are rejected.
- PC still registers its Graph as **metadata** for input/output
  ownership, canonical KV destination binding and P2C capture. It does
  not execute a PC-derived HTP Graph. The next phase removes that
  registration when it has a dedicated Phone-owned output capture path.
- This mode is opt-in and mutually exclusive with `LOCAL_EXEC_SHADOW=1`
  and `LOCAL_P2C=1`. Set `LOCAL_ONLY=0` to restore the tested A/B
  workflow.
- If physical KV indices are absent or not identity-mapped, the run
  fails closed with `PHONE_LOCAL_KV_SLOT_INDICES_UNAVAILABLE` or
  `PHONE_LOCAL_ONLY_NONCONTIGUOUS_PC_KV_SLOTS`. Do not remove
  that guard just to make the test run.

## Phase 6: direct Phone-owned HTP output to P2C Arena (opt-in)

After the Phase 5 local-only Prefill/Decode comparison passes, keep
`GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1` and additionally enable:

```sh
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=1
```

This mode reads each F32 `l_out` from the precompiled Phone-local HTP
Graph into a private per-task host payload and copies those bytes straight
into the existing P2C Arena. It **does not** write the PC Graph's OpenCL
output Tensor, re-run the legacy HTP graph, or use `stored_graphs`
to locate the P2C output. The existing Arena performs ordered, backpressured
snapshot publication. The original PC-registered Graph **still** provides
Activation input storage, physical KV slot indices and canonical OpenCL
Decode KV destinations. Do not disable that registration yet.

This experiment fails closed when the immutable Region Plan, Chunk shape,
payload byte count or finite HTP output check fails. Only use it with the
Phase 5 Qwen3/F16 fresh-sequence restrictions. To revert, set
`GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=0` and restart the Phone server.

With four NPU Regions and 512 tokens at Chunk=128, expect **16**
`event=EXECUTE_ONLY p2c=PHONE_OWNED_HOST status=PASS` and **16**
`event=P2C_DIRECT source=PHONE_LOCAL_HTP_HOST graph_lookup=SKIPPED
status=PASS`, followed by one `KV_SYNC regions=4 status=PASS`.
Empty warmup/decode-only streams with no Phone-owned Plan now log
`KV_SYNC regions=0 status=SKIPPED reason=NO_PHONE_LOCAL_PLAN`.

Compare PC-side Prefill, Decode and generated tokens or logits against
the same-prompt Phase 5 baseline before proceeding. **A successful P2C
capture alone does not validate Decode KV.**

## Remaining work before runtime cutover

1. Validate HTP precompilation on the real Phone and establish a
   Graph READY barrier, then move compilation off the synchronous RPC path.
2. Feed verified PC PUSH activations into the fixed Phone-owned input tensors;
   respect the existing per-Layer KV FIFO before executing the HTP Graph.
3. Validate the opt-in direct Phone-owned HTP output -> P2C Arena path;
   then replace PC-registered input Tensor lookup with independent
   Activation/PUSH address binding while retaining Arena backpressure.
4. Preserve PC decode compatibility (KV ownership/synchronization) and
   compare per-Layer activations and final logits against the old path.
5. Only then make the PC skip building/registering Phone RPC Graphs.
