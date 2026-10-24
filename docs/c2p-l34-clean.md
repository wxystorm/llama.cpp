# C2P L34 clean correctness baseline

This branch starts from `ca551103` and restores **only the serial, correctness-first**
C2P implementation from the pre-parallel Region Boundary work. It deliberately
excludes the later experimental Router/FFN overlap changes.

## Fixed layout for Qwen3-30B-A3B (48 layers)

- **PC Full**: L0-L33 (inclusive); PC can use CUDA/CPU subject to ordinary backend configuration.
- **C2P Tensor**: L34. PC owns Attention, residual, FFN norm, Router/TopK and its FFN shard.
- **Phone Full**: L35-L47 (inclusive).
- Default L34 FFN expert-weight share: PC one-third, Phone two-thirds.
- L34 final `l_out` is **Phone-owned**; do not mirror it to PC.

The C2P boundary stays PC-to-Phone:

1. PC computes Attention, the residual `x = ffn_inp`, FFN Norm and Router.
2. Send normalized hidden activation (the FFN input) and the matching Router TopK/weights PC-to-Phone.
3. Compute the two FFN shards. **This baseline serializes the FFN launch** to
   preserve correctness; simultaneous PC/Phone FFN is a separate future change.
4. PC folds `pc_ffn_partial + x` locally and sends that **one** tensor to Phone.
5. Phone adds its local `phone_ffn_partial`, materializes L34 `l_out`, and
   starts L35. No Phone-to-PC large activation return is allowed at this boundary.

TopK/weights are currently separate small messages. This is not yet a combined
READY packet, and this revision does not claim bandwidth or throughput gains.

## Windows PC environment (PowerShell)

Use the same new branch SHA on both PC and Phone; rebuild the PC RPC backend
and Phone RPC server, then restart the server before testing.

```powershell
$env:LLAMA_HYBRID_REGION_BOUNDARY_V1 = "1"
$env:LLAMA_HYBRID_REGION_C2P_ONLY = "1"
$env:LLAMA_HYBRID_REGION_C2P_LAYER = "34"
$env:LLAMA_HYBRID_REGION_C2P_PC_RATIO = "0.333333"
$env:GGML_META_PIPELINE_DEBUG = "1"
$env:GGML_META_REGION_C2P_NUMERIC_TRACE = "1"
```

Run the existing llama-cli command with tensor split and RPC endpoints. Avoid
reusing a Phone server process or build directory from a Cycle or V5 branch.

The `LLAMA_HYBRID_REGION_BOUNDARY_V1` switch is **off by default**. When
enabled here, only C2P-only layout is permitted; repeating C2P/P2C cycles are
intentionally disabled in this clean branch.

## RPC protocol fail-fast

The PC and Phone must both report `RPC_PROTO_MAJOR.MINOR.PATCH = 4.0.12`.
The client now refuses a different RPC patch version during HELLO and emits
`[RPC_PROTOCOL_MISMATCH]` instead of allowing graph serialization against a
different server implementation. Matching protocol numbers do not prove identical
binaries; verify both Git commit SHAs before runtime tests.

## Verification gates (not yet executed on hardware)

1. Confirm layout log `pc=[0,34) c2p_layer=34 phone=[35,48)`.
2. Confirm L34 FFN physical partition contains nonzero PC and Phone weight units.
3. Compare PC/Phone Router TopK/weights for every chunk; confirm normalized
   hidden arrives before Phone expert execution.
4. Verify `[REGION_C2P_PREFILL]` and `[REGION_C2P_NUMERIC]`, including all
   chunks and final `l_out-34`; follow with L35 input numerical parity.
5. Repeat with single-token decode, multiple chunks and an incremental decode
   sequence before enabling parallel FFN.
6. Only after correctness: measure per-stage PC, Phone, transfer and exposed
   wait times. Keep the existing serial path as the fallback for any future
   parallel experiment.

There is no claim that this commit was compiled, run on Windows/Termux or
validated for numerical parity yet.


## End-to-end correctness localization

Do not use a Phone-only layer such as L46 as the A/B trace target for this
layout. Trace the actual C2P boundary:

```powershell
$env:GGML_META_AB_TRACE_LAYER = "34"
$env:GGML_META_AB_MIN_TOKENS = "16"
$env:GGML_META_AB_DUMP_DIR = "ab-c2p"
```

Run the same prompt, context, token limit, temperature and seed once with this
C2P branch and once with a known-correct baseline. Use a different dump
directory for the baseline (for example `ab-baseline`).

The diagnostic now captures the authoritative Phone-owned L34 `l_out`
immediately after the final C2P reduce. It also captures `next_input` and
`next_norm` only after the subgraph containing `norm-35` has completed; the
old pre-compute `l47_input` capture could observe stale arena storage.

```powershell
python tools/compare_region_ab.py ab-c2p ab-baseline --tokens 358
python tools/compare_token_logits.py c2p.log baseline.log
```

Interpret the first mismatch in order:

1. `ffn_norm`: L34 activation/normalization is already wrong.
2. `router_topk` / `router_weights`: Router handoff is the first divergence.
3. `partial` / combined FFN: expert shard computation diverges.
4. `l_out`: C2P fold/reduce or ownership is wrong.
5. `next_norm`: the L35 consumer path is wrong even if L34 `l_out` is valid.
6. Tensor stages match but token logits diverge: continue downstream toward the
   Phone-only tail, L47 handoff, output norm and LM head.

Do not begin FFN overlap/windowing work until the first token-divergence point is
understood.


## Direct L34 self-check (no dump files)

Set only:

```powershell
$env:GGML_META_REGION_C2P_SELF_CHECK = "1"
$env:GGML_META_PIPELINE_DEBUG = "1"
```

No `GGML_META_AB_DUMP_DIR` is required. One C2P run prints:

- `REGION_C2P_WEIGHT_PLACEMENT`: Attention must be PC-only; MoE FFN expert weights must exist on both PC and Phone.
- `REGION_C2P_WIRE_PARITY stage=FFN_NORM/READY_GENERIC`: first large PC->Phone handoff.
- `REGION_C2P_WIRE_PARITY stage=ROUTER_TOPK`: exact I32 route parity.
- `REGION_C2P_WIRE_PARITY stage=ROUTER_WEIGHTS`: exact F32 route-weight parity.
- `REGION_C2P_WIRE_PARITY stage=PC_PARTIAL_PLUS_RESIDUAL`: second large PC->Phone handoff.
- `REGION_C2P_NUMERIC`: Prefill reduction check.
- `REGION_C2P_DECODE_NUMERIC`: per-token Decode reduction check.

The wire checks compare logical tensor elements in memory after the destination
backend is synchronized; they strip view strides/padding, so a PASS is stronger
than comparing physical RPC byte counts. If all wire/reduce checks pass but
generation is still wrong, the next suspect is the mathematical producer before
the handoff (Attention/FFN Norm/Router/FFN), not serialization.


## Directional pair: PC -> C2P -> Phone -> P2C -> PC

The correctness-first topology is now a single directional pair. For the
48-layer Qwen3 MoE model the defaults are:

- PC full: L0-L33
- C2P: L34
- Phone full: L35-L45
- P2C: L46
- PC full: L47

P2C mirrors C2P. Phone owns Attention, residual, FFN norm and Router. It sends
normalized hidden + TopK/weights Phone-to-PC, both sides compute their FFN
shards, then Phone folds `phone_partial + residual` and sends that one tensor
Phone-to-PC. PC adds its local partial and owns the final P2C `l_out`.

Use:

```powershell
$env:LLAMA_HYBRID_REGION_BOUNDARY_V1 = "1"
$env:LLAMA_HYBRID_REGION_C2P_ONLY = "0"
$env:LLAMA_HYBRID_REGION_C2P_LAYER = "34"
$env:LLAMA_HYBRID_REGION_P2C_LAYER = "46"

$env:LLAMA_HYBRID_REGION_C2P_PC_RATIO = "0.333333"
$env:LLAMA_HYBRID_REGION_P2C_PC_RATIO = "0.333333"

$env:GGML_META_REGION_C2P_SELF_CHECK = "1"
$env:GGML_META_PIPELINE_DEBUG = "1"
$env:LLAMA_DECODE_TOP5_TRACE = "1"
```

Do not enable the legacy Phone-primary single-owner/one-way variables for this
test. Directional C2P/P2C owns its reductions explicitly.

Expected policy logs:

```text
[REGION_BOUNDARY_V1] mode=DIRECTIONAL_PAIR pc0=[0,34) c2p=34 phone=[35,46) p2c=46 pc1=[47,48)
[META_C2P_POLICY_MASK] ... selected=1 first=34 last=35
[META_P2C_POLICY_MASK] ... selected=1 first=46 last=47
```

P2C correctness logs should all PASS:

```text
[REGION_P2C_WEIGHT_PLACEMENT] ... class=ATTENTION pc_units=0 phone_units=...
[REGION_P2C_WEIGHT_PLACEMENT] ... class=FFN pc_units=256 phone_units=512
[REGION_P2C_WIRE_PARITY] ... stage=FFN_NORM ... status=PASS
[REGION_P2C_WIRE_PARITY] ... stage=ROUTER_TOPK ... status=PASS
[REGION_P2C_WIRE_PARITY] ... stage=ROUTER_WEIGHTS ... status=PASS
[REGION_P2C_WIRE_PARITY] ... stage=PHONE_PARTIAL_PLUS_RESIDUAL ... status=PASS
[REGION_P2C_NUMERIC] ... status=PASS
[REGION_P2C_DECODE_NUMERIC] ... status=PASS
```

Decode diagnostics print the raw Top-5 logits and the actual vocabulary piece
for each token id:

```text
[DECODE_TOP5] decoded=0 rank=0 token=... logit=... piece="..." raw_argmax=1
[DECODE_TOP5] decoded=0 rank=1 token=... logit=... piece="..." raw_argmax=0
...
```

The `piece` field escapes newline/tab/quote/backslash so repeated or malformed
generation can be inspected directly from the log. `raw_argmax=1` marks the
highest raw logit, not necessarily the final sampled token when non-greedy
sampling transforms logits.


## Region FFN parallel phase 1

The first performance step keeps the directional topology and both final owners
unchanged. Enable it explicitly:

```powershell
$env:LLAMA_HYBRID_REGION_PARALLEL_FFN = "1"
```

C2P previously executed one PC graph containing both Router and the PC FFN
shard, copied Router TopK/weights, and only then launched the Phone FFN shard.
With this switch the PC graph is split at the pinned Router outputs:

1. execute the PC Router prefix;
2. copy pinned TopK/weights PC->Phone;
3. submit the PC FFN suffix and Phone FFN graph together;
4. wait for both before the existing fold/copy/final-reduce path.

The serial C2P implementation remains the fallback when the switch is unset or
when the Router split point cannot be identified.

P2C already reaches its FFN boundary through the generic dual-worker path, so
phase 1 does not rewrite its compute graph. No extra P2C marker is added here;
the existing P2C correctness/timing logs remain the source of truth.

Expected new debug markers:

```text
[REGION_C2P_FFN_PARALLEL] phase=PREFILL ...
[REGION_C2P_FFN_PARALLEL] phase=DECODE ...
```

For the first hardware run, keep the correctness diagnostics enabled and verify
C2P/P2C wire/numeric checks and generated text before measuring clean throughput.
