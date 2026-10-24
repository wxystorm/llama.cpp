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


## Region route overlap V2

After `LLAMA_HYBRID_REGION_PARALLEL_FFN=1` has passed correctness, enable:

```powershell
$env:LLAMA_HYBRID_REGION_ROUTE_OVERLAP = "1"
```

V2 keeps the directional owners and final partial/reduce protocol unchanged.

### C2P

After the PC Router prefix completes, TopK/weights are snapshotted to host.
The PC FFN suffix is submitted immediately, then the host snapshot is pushed
PC->Phone while the PC FFN is running. Phone FFN starts after those route
control tensors arrive.

Expected marker:

```text
[REGION_C2P_FFN_PARALLEL] phase=PREFILL ... route_overlap=1 ...
[REGION_C2P_FFN_PARALLEL] phase=DECODE  ... route_overlap=1 ...
```

### P2C Prefill

The existing RPC immutable Router snapshot mailbox is prearmed before the Phone
Router producer graph. After the Router graph completes, the server captures
hidden + TopK + weights into an immutable snapshot. The next Phone FFN graph is
submitted immediately while the dedicated route worker transfers that snapshot
Phone->PC. Once the route transfer finishes, the PC staging tensors are copied
into the normal PC FFN inputs and the PC FFN starts. Both FFN workers are joined
before the existing Phone-partial+residual -> PC final-reduce path.

Expected markers:

```text
[REGION_P2C_ROUTE_OVERLAP] phase=PHONE_FFN_EARLY ... status=SUBMITTED
[REGION_P2C_ROUTE_OVERLAP] phase=HANDOFF ... status=IN_FLIGHT
[REGION_P2C_FFN_PARALLEL] phase=PREFILL ... status=OK
```

If the endpoint does not support the immutable route mailbox, V2 logs:

```text
[REGION_P2C_ROUTE_OVERLAP] phase=PREARM ... status=FALLBACK_SYNC
```

and retains the proven synchronous P2C handoff.

For the first V2 hardware run keep the self-check enabled. The required
correctness gates remain `REGION_C2P_NUMERIC`, `REGION_P2C_NUMERIC`,
their Decode variants, wire parity, and normal generated text. Only after those
pass should the self-check and Top-5 diagnostics be disabled for throughput A/B.


## Directional priority queues V1

The directional pair is intentionally asymmetric on each device even though
C2P and P2C are mirrors of each other.

Physical weight roles remain:

- C2P L34: Attention/Router execution is PC-owned; MoE expert weights are
  split with backend 0 = PC and backend 1 = Phone; final reduce is Phone-owned.
- P2C L46: Attention/Router execution is Phone-owned; MoE expert weights use
  the same backend 0 = PC / backend 1 = Phone split convention; final reduce
  is PC-owned.
- Small control weights such as FFN norm/router weights may remain mirrored.
  Compute ownership, not storage duplication, selects the C2P/P2C control side.

Enable the first role-queue implementation with:

```powershell
$env:LLAMA_HYBRID_REGION_PRIORITY_QUEUES = "1"
```

The logical queues are:

```text
PC_MAIN        (high): outgoing C2P work
PC_P2C_FFN     (low):  incoming/reducer-side P2C FFN shard

PHONE_MAIN     (high): outgoing P2C work
PHONE_C2P_FFN  (low):  incoming/reducer-side C2P FFN shard
```

This distinction is important. Phone L46 FFN belongs to PHONE_MAIN: Phone
does not reduce P2C and therefore should compute its L46 partial and send it
to PC without waiting for the PC reduce. Symmetrically, PC L34 FFN belongs to
PC_MAIN.

V1 only establishes the mirrored priority executors at the already-proven
Prefill FFN overlap points. It does not change Decode, does not move reduce to
CPU, and does not yet relax the current dependency/join window. MAIN is chosen
before a queued incoming-FFN shard, but an already submitted GPU graph is not
preempted.

Expected debug markers:

```text
[REGION_EXEC_QUEUE] side=PC    queue=MAIN    priority=HIGH direction=C2P role=OUTGOING_FFN
[REGION_EXEC_QUEUE] side=PHONE queue=C2P_FFN priority=LOW  direction=C2P role=INCOMING_FFN
[REGION_EXEC_QUEUE] side=PHONE queue=MAIN    priority=HIGH direction=P2C role=OUTGOING_FFN
[REGION_EXEC_QUEUE] side=PC    queue=P2C_FFN priority=LOW  direction=P2C role=INCOMING_FFN
```

The next step, after correctness passes, is to feed the post-C2P Phone
continuation (L35-L45 -> L46 Attention/Router -> L46 Phone FFN -> P2C send)
through PHONE_MAIN and the mirrored PC continuation through PC_MAIN, then
delay incoming-shard joins to introduce a two-credit Prefill window.


### Directional MAIN coverage V2

With `LLAMA_HYBRID_REGION_PRIORITY_QUEUES=1`, the high-priority MAIN
executor now also covers the already-proven synchronous Prefill continuation:

```text
PC_MAIN:
  PC full-region Prefill work
  C2P outgoing FFN/fold
  P2C final reduce
  PC continuation after P2C

PHONE_MAIN:
  C2P final reduce
  Phone full-region Prefill work
  P2C Attention/Router path
  P2C outgoing FFN/fold
```

The reducer-side waiting shards remain low-priority only:

```text
PHONE_C2P_FFN
PC_P2C_FFN
```

This stage intentionally preserves every existing join. It does not yet
introduce a multi-chunk credit window. The purpose is to establish the
correct priority topology before incoming-FFN waits are deferred.

Reduce remains on the existing GPU backend in this version. CPU Reduce can be
A/B tested later after the queue/window topology is correct.


### Directional boundary window=2 V1

Enable the first two-credit Region boundary window with:

```powershell
$env:LLAMA_HYBRID_REGION_WINDOW = "2"
```

It requires the already-proven switches:

```powershell
$env:LLAMA_HYBRID_REGION_PRIORITY_QUEUES = "1"
$env:LLAMA_HYBRID_REGION_ASYNC_PARTIAL = "1"
```

V1 gives each directional boundary two independent credits:

```text
lane = chunk % 2
```

Each credit pins both pieces required by the final reducer:

- the outgoing remote partial in a private per-lane staging buffer;
- the reducer-side local FFN partial in a separate private per-lane staging
  buffer, so the next chunk cannot overwrite it through graph-arena reuse.

The asynchronous task for each credit performs:

```text
PARTIAL transfer
  -> destination visible
  -> enqueue final Reduce on MAIN/HIGH
  -> Reduce complete
  -> credit may retire
```

Therefore C2P can overlap:

```text
chunk0: PC->Phone PARTIAL / Phone MAIN Reduce
chunk1: C2P Router + PC FFN + PHONE_C2P_FFN
```

and P2C mirrors it:

```text
chunk0: Phone->PC PARTIAL / PC MAIN Reduce
chunk1: P2C Router + Phone FFN + PC_P2C_FFN
```

A credit is released only after final Reduce has completed. Lane reuse waits for
the previous owner of that lane, and the final boundary chunk drains both
credits before downstream ownership is consumed.

Expected markers:

```text
[REGION_WINDOW_CONFIG] requested=2 active=1 credits=2
[REGION_CREDIT] direction=C2P event=ACQUIRE ... inflight=1/2
[REGION_CREDIT] direction=C2P event=ACQUIRE ... inflight=2/2
[REGION_RETURN_TX] direction=C2P ... lane=0/1
[REGION_EXEC_QUEUE] ... queue=MAIN ... role=REDUCE ... ENQUEUED_ASYNC
[REGION_CREDIT] direction=C2P event=RELEASE ... reason=LANE_REUSE/BOUNDARY_DRAIN
```

P2C has the same mirrored markers.

This V1 pipelines the directional boundaries themselves. The current graph
still finishes all L34 chunks before entering the full Phone L35-L45 graph.
The later full Region pipeline step will chunk the local full-region graph so
that an earlier chunk can already advance through L35-L46 while the next
chunk is still in the incoming C2P FFN queue.


### Keep Phone-local before P2C

For the directional pair

```text
L35-L45 Phone Full
  -> L46 P2C, Attention/Router on Phone
```

the L45 full activation remains Phone-owned. A generic Phone-owner-exit mirror
must not copy the complete `l_out-45` back to PC before L46, because the PC
side of L46 only consumes the explicit P2C READY/Router payload and its FFN
shard inputs.

The Prefill-only guard emits:

```text
[REGION_KEEP_PHONE_EXIT] ... layer=45 next_layer=46
reason=NEXT_P2C_PHONE_ATTN status=OK
```

Expected effect: the full L45 `1->0` mirror disappears; the intentional L46
P2C READY/PARTIAL transfers remain unchanged. Decode and non-directional
Phone-owner exits retain the legacy handoff behavior.


## Directional Region Chunk Continuation

The continuation path is isolated behind:

```text
LLAMA_HYBRID_REGION_CHUNK_CONTINUATION=1
```

It is Prefill-only and currently gated to Qwen3-MoE directional-pair
policies with exactly one C2P and one P2C boundary.

Unlike the earlier staged prototype, this implementation keeps one normal
full Qwen3 graph. There is no `process_ubatch_staged()` handoff and no host
hidden buffer between the PC prefix, Phone Region, and PC tail.

For the current L34/L46 policy the single graph is:

```text
L0 ... L33
      |
      v
L34 C2P
      |
      +--> chunk0 -> L35 -> ... -> L45 -> L46 P2C
      +--> chunk1 -> L35 -> ... -> L45 -> L46 P2C
      +--> chunk2 -> L35 -> ... -> L45 -> L46 P2C
                                      |
                                      v
                                     L47
                                      |
                                      v
                                    logits
```

The ordinary Qwen3 layer loop builds L0-L33. When it reaches the C2P layer,
it calls the directional Region builder, which emits the chunk-major
L34-L46 sub-DAG and returns `l_out-46`. The same ordinary loop then resumes
at L47.

C2P final output uses one stable Phone-side `l_out-34` backing. The ordinary
producer is disabled before chunk 0 continuation. Each C2P Reduce writes only
its chunk into that backing. With `LLAMA_HYBRID_REGION_WINDOW=2`,
`Attention(L35,C)` waits only for the exact `Reduce(L34,C)` credit; the old
boundary-wide C2P drain is suppressed. Graph-end drain remains a safety net.

The existing P2C stable boundary-output commit remains unchanged, so the PC
tail sees the completed `l_out-46` before L47 executes.

Expected diagnostic markers:

```text
[REGION_CONT_FULL_GRAPH_CAP] ... status=READY
[REGION_CONT_GRAPH] ... mode=BOUNDARY_SEED_THEN_CHUNK_MAJOR
[REGION_CONT_DEP] direction=C2P producer=34/C consumer=35/C ... status=OK
[REGION_BOUNDARY_OUTPUT_COMMIT] direction=P2C layer=46 ... status=OK
```

This removes the staged host round-trip and the full-Region layer barrier
from the graph dataflow. The priority worker is still non-preemptive: a
low-priority C2P FFN that is already running is not interrupted by newly
ready MAIN work.

