# Region Ring V2: rebuild from the known code base, not the last patches

Base commit: \`6dee1ea93630717f6cc2433a859c9f56ec63044d\`
Branch: \`feat/region-ring-v2-stage-profile-rebuild-20261108\`

## Design constraints

- Four physical compute backends: \`G\` = PC CUDA, \`C\` = PC CPU,
  \`N\` = Phone HTP NPU, \`A\` = Phone Adreno OpenCL.
- In each Cycle the stage *order* is \`G -> C -> N -> A\`.
  Each backend can have zero layers; omit zero-width physical Stages.
  A Cycle may start/end on any backend and the final Cycle may be partial.
  There is no mandatory Phone stage, including at the end of the model.
- A physical Stage owns one contiguous Transformer layer interval.
  The intervals over all Cycles cover \`[0, n_model_layers)\` exactly once.
  The output RMSNorm/LM Head belongs to the **PC CPU** and is
  logically separate from the final Transformer Stage.
- \`X\` is the common G/N/A Chunk size; CPU may use a separate \`XC\`
  (when zero, \`XC=X\`). Do not introduce independent \`XG\` or \`XN\`.
- A Phone NPU Region may own \`[layer_begin,layer_end)\`, including N2,
  N4, etc. It must truly execute the consecutive layers locally. N4 is
  not four CPU-issued N1 Stages and is never estimated as \`4 * N1\`
  without an explicit model fitting policy and truthful provenance.
- Do not silently fallback among physical backends. Unsupported
  compute/memory/transfer behavior invalidates a candidate.
- KV is owned by the execution backend and survives weight-bank
  recycling. Phone HTP Graph retirement and two-Bank replacement
  remain coupled to safe output/P2C fences.

## Single ownership contract

Header: \`src/llama-region-ring-v2.h\`

- \`plan_spec\`: exact per-Cycle G/C/N/A layer depths and common X/XC.
- \`compile()\`: emits ordered physical Stages, contiguous layer
  intervals and actual activation boundaries. No phantom zero Stages,
  mandatory G, mandatory N->A adjacency, or Phone-last restriction.
- \`profile_request\`: *only* the first N layers of one backend with
  real Chunk and KV read-span dimensions. There is no feeder Stage
  and no output Head.
- \`ggml/src/ggml-rpc/phone-plan-region-map.h\` now also provides
  a V2 range-based NPU Region map; the legacy single-layer RPC map
  is deliberately unchanged until the new versioned RPC transport
  has a corresponding consumer.

These are **contracts**, not a claim that the old scheduler executes
all such topologies. Planning and execution capability must be tracked
separately.

## Runtime and Profiler responsibilities

1. Main program reads the original GGUF metadata, probes PC/Phone
   backend identity and checks an exact-versioned persisted Profile.
2. If the Profile is absent/stale/incomplete, run only the missing
   compute/transfer/bank probes. \`--force-profile\` explicitly refreshes.
3. For compute, each requested \`backend, N, X, KV\` constructs and
   executes a **single native Stage graph for layers [0,N)**. The
   other three devices must execute no Transformer operations.
   Real positions, causal mask and initialized KV reads are mandatory.
   Graph build, prepare, compute, synchronization and memory peaks
   are measured separately; warmups are excluded from medians.
4. Phone runs N/A probes locally, then reports paged typed results
   through PC-initiated RPC, reusing the verified Patch32 telemetry
   transport primitives. No manual Android log copying and no
   operator-authored HTP Region/layer environment variables.
5. Transfer profiling uses real PUSH/P2C and PC CUDA H2D/D2H paths
   with several payload sizes, recording fixed latency and size cost.
   Phone NPU->Adreno Host handoff and device upload are **different**
   costs; the shared Host activation must not be copied twice.
   HTP weight-host staging and HTP device Bank repack are separate
   memory/bank calibration lanes.
6. Save complete, versioned Profile data keyed by model fingerprint,
   device identities, runtime/graph implementation version, quant,
   KV types and other relevant compute settings. Never mark a Profile
   complete when a required measured path is missing.
7. Every new Prompt is tokenized and Planned anew. A Profile cache hit
   skips benchmarking but NEVER reuses an old Prompt's Plan.

The public CLI can still be \`llama-completion\` with options to force
refresh; the components should live behind a small coordinator rather
than extending \`llama-context.cpp\` or the old PowerShell probes.

## Planner candidate search

Phase A: enumerate \`G/C/N\` layer-depth/number-of-cycle seeds with
optional zero counts and partial final Cycle. Do not require a fixed
\`G,C,N,A\` template. The first version does not proactively search A.

Phase B: evaluate HTP A/B Bank capacities, persistent per-layer KV,
HTP scratch, Host staging, Adreno GPU memory and Phone shared budget.
When an NPU Stage exceeds an HTP Bank, enumerate *contiguous NPU suffix
spill* alternatives (\`N4\` -> \`N3,A1\` -> \`N2,A2\`, etc.)
before rejecting. Keep exact total layer coverage. Reject only when
both NPU and Adreno placement fail. A is an optional performance
optimization in a later iteration.

Phase C: rank feasible candidates using a cheap lower bound, then send
a generously sized shortlist to the **single** exact event-DAG planner.
Initial targets are 500-2000 full DAG evaluations; tune against actual
planning wall time. The DAG owns compute lanes, activation transfers,
Bank k-2 refill fences, KV order, local Phone handoff, and PC CPU Head.

Only plans supported by **real Runtime capability** can be executed.
The offline search is free to show additional candidates as
\`PLAN_ONLY\`, never \`RUNTIME_PASS\`.

## Migration, not a parallel production runtime

Reuse:
- existing Qwen3 layer math and PC stage Graph machinery;
- Phone-owned Graphless HTP, persistent KV, single-HOST N->A handoff;
- Phone dual-weight-Bank lifecycle, preload and safe graph retirement;
- PC/Phone RPC telemetry page/checkpoint primitives;
- exact DAG dependency semantics and existing tests.

Replace over time:
- \`G/C/P\`-only physical Region types and layout parsers;
- dynamic Phone backend selection by a magic Region-ID list;
- two distinct three-/four-backend Planners and duplicated layer
  placement rules;
- the mixed-pipeline compact-GGUF Stage proxy
  (\`run_gcna_profile.ps1\`, \`run_backend_profile.ps1\` from the
  abandoned experimental branch) for native isolated compute profiling;
- static Phone-final assumptions and separately forced XG.

Do **not** replace the old runtime immediately: keep it as a
regression baseline while migrating one capability at a time.

## Current implementation / acceptance gates

Implemented in this V2 branch:
- pure four-backend ordered Stage contract with zero counts and any tail;
- pure first-N-layer independent Profile request contract;
- Phone NPU multi-layer Qwen3 **graph metadata builder**, chaining
  consecutive layers with separate per-layer KV and unchanged
  single-layer builder API;
- pure multi-layer HTP Region range-map validation;
- model-free regression tests for all above.

**Not yet implemented or verified:**
- running a V2 G/C/N/A isolated compute Profile on real hardware;
- HTP multi-layer Bank weight binding, KV residence, precompile,
  native execution, parity and benchmark RPC; metadata is NOT runtime;
- the native Profile cache/control flow in the main executable;
- complete runtime-matched transfer-only benchmarking;
- the V2 Planner, memory spill DAG, and V2 Runtime adoption;
- Windows+Android builds and real Qwen3 logits/Prefill/Decode parity.

Gates: (1) host-only tests/build; (2) PC G/C prefix Graph timing;
(3) Phone A prefix Graph; (4) N2/N4 local parity and HTP Bank safety;
(5) RPC, transfer, bank and cache integration; (6) V2 Planner;
(7) runtime topology and final Head parity; (8) performance comparisons.
