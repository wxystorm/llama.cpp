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
   through PC-initiated RPC, reusing the verified Patch33 telemetry
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

## Incremental integration status (2026-11-09)

The original `apply-wide` search and production entry point remain the
authoritative target. These pieces are now implemented as opt-in helpers in
`llama-region-ring-planner.h`, without changing legacy G/C/P behavior:

- The final event simulator models HTP A/B Bank refill on the existing
  Phone stage queue, with rank k blocked until rank k-2 has finished and
  its physical Bank reload completes. The HTP weight-mover lane is
  independent of the existing CUDA H2D mover.
- In mixed-Phone mode NPU and Adreno must provide distinct measured native
  Stage compute/prepare curves; an unavailable shape is rejected, not
  silently charged as aggregate PHONE.
- `choose_phone_npu_preferred_spill` selects a correctness-qualified NPU
  prefix followed by an Adreno suffix. `phone_spill_memory_feasible`
  checks two alternating HTP weight Banks, persistent KV, Adreno resident
  weights, workspace and the shared Phone RAM budget before DAG admission.
- The original independent native `run_v2_stage_matrix.py` Profile now
  publishes `npu_weight_refill_points` from PASS-only repeated HTP
  same-Bank measurements. These are task durations, not observed overlap
  percentages. Measuring runtime overlap is **not** a prerequisite for
  Planner eligibility.

**Main-program integration is now committed but not device-verified:**

- `run_prompt_sweep.ps1` keeps the original `llama-completion + apply-wide`
  entry point. It checks the model-bound native `profile.json` once per
  sweep; a cache miss/force calls the existing resumable *independent*
  `run_v2_stage_matrix.py`, not a compact fixed-layout runtime benchmark.
  Native Phone samples now use command 57 on inference RPC 50052
  (protocol 4.0.35). Phone RPC directly invokes the native C++ N/A
  Graph profiler within the server process: no Python worker or second port.
- The native cache is accepted only after its model/device/binary
  fingerprints, PASS CSV/manifest consistency, and all required
  G2/C2/N1/A1 Chunk x KV shapes have been checked. Incomplete grids
  resume the independent matrix sampler instead of being treated as
  Planner-ready. For longer prompt groups, KV anchors grow from
  512/1024 to 2048/4096 only when the sweep needs them; previously
  measured shapes are reused.
- The existing Hybrid measurements (PC/CUDA memory and real ggml-rpc
  transfer curves) are cached independently as a versioned model-/device-/
  binary-bound `.rrhp` snapshot. On a hit the next
  `llama-completion` process skips those probes, but every Prompt reruns
  the original E/Beam/shortlist DAG candidate search.
- The cheap E/Beam candidate screen uses native G/C compute curves and
  a measured N1 + resident Adreno suffix service proxy for logical Phone
  depth P; this preserves the existing search structure. The exact
  shortlisted DAG uses distinct real N/A curves and rejects missing shapes.
  A3 is now among the automatically sampled Adreno depth shapes.
- `llama-region-ring-native-stage-profile.h` loads only valid native
  PASS rows; N4 parity-failed rows are excluded. The E/Beam stage-cost
  callbacks use native G/C timings; the final N/A physical Stage DAG uses
  distinct measured N/A compute/prepare costs and NPU same-Bank refill
  durations. It never requires observed physical overlap.
- For shortlisted logical P Regions, the original Planner now chooses
  correctness-qualified NPU prefixes and resident Adreno suffixes,
  checks the joint Phone memory budget and HTP A/B Bank capacity, then
  computes precise refill fences in its existing event simulator.
- The winning HTP Region IDs are stored on the model before its weights
  are placed. The existing phone-owned Plan RPC gets the actual mapping
  from that model, rather than asking the operator to write a Region-ID
  environment variable. Unsupported candidates fail instead of silently
  falling back to P-only timing.

**Known restrictions / validation still outstanding:**

1. Production mixed-Phone execution accepts **N1 per HTP Region**, with
   at least two HTP Regions and one Adreno Region. N2/N4 native Stage
   profiling is not production multi-layer HTP support. Accordingly
   the spill policy currently caps each physical HTP Region to N1; once
   production N2+ is proven, increase the cap and retest.
2. New C++ and Python regression fixtures were committed but not yet
   compiled/executed on the user's Windows PC or Snapdragon Phone. The
   first true `llama-completion` PC+Phone `apply-wide` run must verify
   Graph registration, Plan RPC, KV and final logits. Tests and sample
   commands live below.
3. Native Profile-cache hits validate the stored Phone fingerprint, not a
   fresh live Phone RPC identity query. After changing the Phone DSP/runtime/binary,
   use `-ForceProfile` until live identity checking is integrated.
4. Some independent samples (e.g. N4) have a known numeric parity failure.
   These are excluded rather than synthesized; if no executable N1/A
   combination fits, Planner fails closed.

**Main-program invocation:**

```powershell
.\tools\region-ring-experiments\run_prompt_sweep.ps1 `
  -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir ".\single-prompt-512" `
  -Out ".\planner-test-results-v2" `
  -Planner apply-wide -Streaming on -NPredict 32 `
  -Threads 8 -Batch 512 -UBatch 512 -LogVerbosity 4
```

Add `-ForceProfile` to refresh measurements or
`-NoV2Profile` to run only the legacy G/C/P Planner. Use
`-V2ProfileDir ".\profile-v2\gcna-expanded"` for an explicitly
selected persisted independent matrix. Old `-SkipProfile -ForceLayout
-ForceXC -ForceXG -ForceXP` continues to be a separate fixed A/B
benchmark bypass.

```powershell
cmake --build build-tensor-v2 --config Release --target llama-completion region-ring-v2-stage-profile test-region-ring-planner test-region-ring-profile-cache test-region-ring-native-stage-profile -j 8
ctest --test-dir build-tensor-v2 -C Release -R "test-region-ring-(planner|profile-cache|native-stage-profile)" --output-on-failure
python -m unittest discover -s tools/region-ring-experiments -p "test_*profile*.py" -v
```

## Current implementation / acceptance gates

Implemented in this V2 branch:
- pure four-backend ordered Stage contract with zero counts and any tail;
- pure first-N-layer independent Profile request contract;
- Phone NPU multi-layer Qwen3 **graph metadata builder**, chaining
  consecutive layers with separate per-layer KV and unchanged
  single-layer builder API;
- pure multi-layer HTP Region range-map validation;
- model-free regression tests for all above;
- a standalone native Stage profiler that loads ONLY [0,N) from the
  original Qwen3 Dense GGUF, builds a real shared N-layer Graph on exactly
  one device (G/C/N/A), and executes real KV/Mask with no feeder Stages;
- HTP0 native quantized-weight repack allocation, fixed-Graph precompile,
  synchronized N2/N4 compute path, automatic compiled-cache retirement,
  and CPU output parity (required for N mode);
- resumable stage matrix with a versioned model/executable/device
  fingerprint, and a synthetic tiny-Qwen3 CPU N1/N2/N4 integration-test
  fixture;
- integrated versioned V2 N/A Profile requests on GGML RPC 50052:
  Phone calls the original HTP/OpenCL native Graph sampler *in-process*
  and returns measured rows, with no Python, worker process, second
  listener or manual CSV copying;
- explicit PARITY_FAIL measurements retained for analysis but excluded
  from the planner-eligible PASS profile CSV;
- exact KV-X final-token profile window for nondivisible shapes (e.g.
  KV512/X96), plus measured compute min/median/p90/max/stddev;
- physical GGML backend Activation set/get times and real raw GGUF
  file-read versus HTP repack-write separation;
- optional host-staged HTP SAME-BANK repeated refill measurement
  (not dual-Bank overlap); optional direct-TCP socket receiver stream
  calibration for both PC->Phone and Phone->PC directions;
- buffer allocation measurements and separately labelled NOT-VALIDATED
  two-HTP-Bank allocation sizing estimates in profile.json;
- Phone device/lib/DSP/Android firmware runtime fingerprinting to
  invalidate old timing samples after a backend binary update.
See V2_NATIVE_STAGE_PROFILE.md and V2_PHONE_PROFILE_RPC.md.

Production-path timing instrumentation (RPC Patch33) now additionally
records the existing real two-weight-Bank Mover queue/Host staging/HTP
repack+Graph precompile intervals, measures overlap against completed
Phone HTP/Adreno service windows, and tracks zero-copy NPU->Adreno
local handoff until downstream upload/device service. The PC imports
these from already returned RPC log events into a separate model-bound
Profile using merge_v2_runtime_profile.py, without Android log copying.
These changes are NOT equivalent to an observed physical runtime PASS;
the double-Bank overlap and final combined Planner remain gated on
a matching PC/Phone hardware test. See V2_RUNTIME_BANK_PROFILE.md.

**Not yet implemented or verified:**
- end-to-end verification of the integrated GCNA matrix/RPC 50052 path
  on real PC and Phone hardware; source + regression tests committed;
  the user already observed Phone HTP N2 precompile/compute/parity PASS,
  while N4 precompile/compute succeeded but numerical parity FAILED;
- integration of N2/N4 native HTP graphs with PRODUCTION per-Cycle
  two-Bank refill, persistent Decode KV and inference RPC ownership;
- real-hardware N/A native in-process tests, including repeated HTP Graph
  lifetimes, CPU parity, capacity limits, and inference co-existence;
- complete runtime-matched transfer-only benchmarking;
- the V2 Planner, memory spill DAG, and V2 Runtime adoption;
- Windows+Android builds and real Qwen3 logits/Prefill/Decode parity.

Gates: (1) host-only tests/build; (2) PC G/C prefix Graph timing;
(3) Phone A prefix Graph; (4) N2/N4 local parity and HTP Bank safety;
(5) RPC, transfer, bank and cache integration; (6) V2 Planner;
(7) runtime topology and final Head parity; (8) performance comparisons.
