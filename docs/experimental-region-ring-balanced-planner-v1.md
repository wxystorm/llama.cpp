# Region Ring Balanced Planner V1

Status: design specification / implementation roadmap. **No planner code is enabled yet.** This branch is based on `exp/region-ring-phone-ready-autostart-20261026`; the verified old Ring and READY execution remain unchanged.

## 1. Design objective

**Primary idea:** each C→G→P cycle is a two-sided producer/consumer pipeline. Balance the rate at which PC (CPU+CUDA) produces a Chunk for Phone against the rate at which Phone processes and returns one to PC for the next cycle. Use measured per-layer/per-Chunk Profile costs and interpolation, then minimize the wall time of the **entire finite Prefill**. Equal layer counts or exactly equal single-task times are not themselves the objective.

Different token intervals can run concurrently on CPU, CUDA and Phone. Balance is an efficient coarse search filter; the full simulation must include pipeline fill/drain, Window=2 stalls, network direction sharing, actual CPU/CUDA overlap and graph registration.

Known 48-layer reference: `C7,G4,P5,C7,G4,P5,C8,G3,P5` (three cycles of 16 layers), common `X=64`, 358 tokens producing 6 Chunks and 54 Region tasks, Window=2. Phone fusion/autostart currently validates only L11–15 and L27–31; L43–47 remains conservative legacy execution.

The planner jointly chooses:
* the **number of complete C→G→P cycles**, and (initially repeated or near-repeated) per-cycle contiguous layer counts `(C_i,G_i,P_i)`; **do not require `C_i+G_i+P_i=16`**. All model layers are covered by cycles plus an optional PC-only suffix `C_tail/G_tail`. The full 48-layer constraint is `Σ_i(C_i+G_i+P_i)+C_tail+G_tail=48`;
* independent `X_CPU`, `X_GPU`, `X_PHONE` Chunk token sizes, optionally per cycle later;
* Token partitions, output-slot/credit requirements and reusable Phone Graph shape/slot/KV keys;
* predicted full-Prefill makespan, resource occupancy, memory peak and transfer/registration cost.

### 1.1. Producer/consumer throughput balance

Let `T_C(C,x,K)`, `T_G(G,x,K)`, `T_P(P,x,K)` be the measured/interpolated times to execute the CPU, CUDA, Phone portions for query Chunk size `x` and KV length `K`. Let `D_CP(x)` and `D_PC(x)` be the C2P/P2C transfer costs, including genuine input-ready fencing where applicable.

For **identical Chunk sizes** and limited overlap, the user's coarse cycle-balance equation is:

`T_C(C,x,K) + T_G(G,x,K) + D_CP(x) ≈ T_P(P,x,K) + D_PC(x)`.

For **different Chunk sizes**, compare effective time per token, not raw milliseconds:

`s_C=T_C(C,X_C,K)/X_C`, `s_G=T_G(G,X_G,K)/X_G`, `s_P=T_P(P,X_P,K)/X_P`,
`d_CP=D_CP(q)/q`, `d_PC=D_PC(q)/q`, for a common token reference interval `q` or by summing exact interval transfers.

A simple sequential PC-side filter is `s_PC_seq=s_C+s_G+d_CP` vs `s_PHONE=s_P+d_PC`. If separate CPU/CUDA workers truly overlap processing **different Chunks**, a throughput-oriented PC estimate can approach `max(s_C,s_G)+d_CP`. The estimator must identify the actual PC gate/overlap from profiling; neither expression can be blindly treated as final makespan.

- `s_PC >> s_PHONE`: PC is slower; Phone queue starves. Consider shifting eligible work from CPU/CUDA to Phone, or adjusting sizes.
- `s_PC << s_PHONE`: Phone is slower; PC builds a queue, Window credits eventually block. Consider shifting eligible work from Phone to CPU/CUDA or selecting more efficient Phone Chunk sizes.
- `s_PC ≈ s_PHONE`: viable steady-state balance, but still check network and finite-prompt fill/drain.

Reverse flow is not a second independent throughput target: the Phone's output rate into the next PC cycle is constrained by the same token interval dependencies. The shared Wi-Fi path can become a third bottleneck, especially when C2P and P2C contend. Use **actual two-way network measurements** instead of treating their byte-transfer times as fully independent.

For the whole multi-cycle Ring, CPU, CUDA, Phone and transfers are reused by all cycles. A necessary long-run lower bound on per-token initiation interval is approximately the **maximum of aggregate per-device/per-link service demand**, not merely the sum within one cycle. Use the interval-DAG simulator for the final score.

### 1.2. Remaining layers are PC-owned

Search `N` near-identical balanced cycles first. If the model's `n_layer` is not divisible by their total cycle width, **assign the remainder to a PC-only tail**, choosing CPU and/or CUDA by Profile and memory feasibility. The final layout can be `[C,G,P]×N → [C_tail,G_tail]`; a tail of zero layers is allowed.

Example only, not a predicted optimum: three `C7,G4,P4` cycles cover 45 of 48 layers; the remaining 3 may run on PC. Do **not** silently extend Phone P or stretch the final cycle simply to consume the remainder.

The PC tail adds final latency and consumes memory/bandwidth. Its own cost must be included in full-Prefill score. The final Phone→PC boundary is still needed to feed that suffix; the suffix must own final logits. The current Ring parser requires complete C/G/P cycles and the isolated scheduler expects a final Phone owner, so **tail execution remains DRY_RUN until parser, stage-output and final-logit ownership are generalized and parity-tested**.

Do **not** apply altered Phone layer boundaries until arbitrary P-region Graph fusion, Phone READY registration and final output ownership have been generalized. The original P5 endpoints are hard-coded safety guards today.

## 2. Reuse, not duplicate, the existing planner

Existing `src/llama-hybrid.cpp` already searches per-device `cpu_chunk_tokens`, `gpu_chunk_tokens`, `phone_chunk_tokens` and uses a `llama_hybrid_plan_boundary()` helper. Reuse its profile anchors, layer-block timing interpolation and memory filtering.

Current Ring **does not** use those independent Chunk sizes:
* `src/llama-context.cpp` gives every Ring Region the same `region_ring_chunk_tokens`.
* `src/llama-region-ring.h` builds a single global Chunk partition and orders all Regions by identical Chunk ID.
* Ring host buffer addressing uses the common `chunk % window` times `chunk_tokens`.

Therefore changing three environment variables **alone** is unsafe: it would break upstream token coverage, KV order, and Buffer/Slot lifetime. An interval-aware Ring is a prerequisite to APPLY.

## 3. Candidate search

Initial coarse X candidates (anchors, not choices made in advance):
* PC CPU & CUDA: `{32,64,128}`.
* Phone GPU: `{16,32,64,128}`.
* Refine promising cases with `{48,96}`, balanced tails and measured interpolation uncertainty.
* For 358 tokens, consider the 38-token tail under 64 and avoid an unhelpful 6-token tail under 32 by validated bounded tail coalescing.

Layer search should **not** be limited to a 16-layer sum per loop. Stage 1 enumerates candidate repeated `(C,G,P)` widths and cycle counts `N`, computes any remaining layers as a PC suffix, and ranks coarse candidates by their producer/consumer **per-token rate mismatch** and feasibility. Stage 2 refines near-repeated per-cycle counts, Chunk sizes and tail CPU/CUDA split using the finite-Prefill DAG simulator.

Known-good `C7,G4,P5` vs alternative `C6,G5,P5`, `C7,G4,P4` etc. are only search examples. Moving any P endpoints or enabling a PC tail is **score-only** until generalized graph ownership and final-output runtime guards pass.

Early pruning: both rate estimates finite and supported by Profile, Phone weight residency / remaining memory feasible, GPU residency and LRU upload estimated, no invalid direction or slot mapping, no missing Profile anchor for critical cases. Re-evaluate the best candidates with actual KV per interval and measured CPU↔CUDA concurrency.

Profile task costs as functions of actual token interval, Layer count and KV length:
`T_region(device, region, n_query, K_i)` including Graph preparation, device compute, scratch, cache misses and unavoidable copies. Use real `K_i` per interval rather than one constant KV approximation.

Profile separately:
* CPU Region and CUDA Region compute (including GPU layer weight upload/LRU misses and PC gate);
* Phone GPU server compute and READY queue, Graph REGISTER cost only once per new UID, and output `l_out` handoff;
* H2P and P2H bandwidth/latency during overlapping transfers, RPC Fence/READY latency, and slot occupancy;
* model weight residency, KV cache, Graph cache, per-slot scratch and reserve memory on each device.

## 4. Load balancing is a dependency-DAG simulation

Build task nodes `(region, token_begin, token_end)` with compute lanes `PC_CPU`, `PC_CUDA`, `PHONE_MAIN` and transfer lanes `H2P/P2H`. Model full-model layer coverage as **N variable-width C/G/P cycles, followed optionally by PC C/G tail**. Match the actual PC gate, shared resources, Phone Main FIFO and output Window credits. Include startup/fill, all steady-state intervals, tail processing and final drain.

For each physical resource d, report busy time, idle time and critical-path blocking time. Within each cycle report `C_i_ms/G_i_ms/P_i_ms`, `s_PC`, `s_PHONE`, producer/consumer rate mismatch and buffer occupancy. Approximate steady-state initiation interval is bounded by the busiest **aggregate** lane across all cycles. Rank candidates by the **simulator's total makespan** on the actual finite Prompt; use rate equality only for quick pruning / explaining bottlenecks.

Score priority:
1. Correctness + feasible memory + no deadlocks + safe graph ownership (hard constraints).
2. Minimum predicted wall time for the whole Prefill.
3. Near-ties: lower memory, smaller peak queue/wait, fewer graph templates and less boundary fragmentation.

Do not change ownership/layers in the middle of one running Prefill. Adapt between requests only, after profiling.

## 5. Heterogeneous Chunks require exact token-interval boundaries

Example `PC64 -> Phone32`: PC output interval `[0,64)` fans out into Phone intervals `[0,32)` and `[32,64)`. Its producer buffer remains live until both sub-consumers have finished reading/copying.

Example `Phone32 -> PC64`: PC cannot run `[0,64)` until **both** Phone outputs `[0,32)` and `[32,64)` have completed. The boundary operation is ordered **concatenation/gather of hidden vectors**, never arithmetic ADD/reduction.

Required runtime representation:
* For every Region task and output edge: `{region, token_begin, n_tokens, slot, generation, seq}`.
* Per-edge interval coverage; a downstream task becomes READY only when the **union of ready input ranges completely covers** its token interval and transfers have completed.
* Reference-counted reader credits: `FREE -> WRITING -> READY -> READING -> FREE`. One producer may have multiple consumers; credit returns only after the last one.
* Per-layer KV position commitment is monotonically ordered. Different physical Region Chunk counts must not bypass KV ordering or reuse a scratch/Graph slot early.
* Phone READY is emitted only after its input is truly transferred (RPC Fence / transfer event). Phone MAIN executes after READY; PC waits only at downstream output consumption.
* Distinct Shape/Slot/KV bindings may need distinct cached Graph UIDs. Reuse only when the graph and all associated buffers are valid for the intended token interval.

The first correctness target is `PC64 -> Phone32 -> PC64`; next test the inverse. Keep 2 active output slots per boundary at first; use interval/reader credits rather than global Chunk parity alone. Tail cases must cover 358 tokens exactly with no gaps/duplicates.

## 6. Development sequence

### A. MEASURE + DRY_RUN (first code milestone)

Collect per-region CPU/CUDA/Phone and transfer timings, then enumerate `(N,C_i,G_i,P_i,C_tail,G_tail,X_CPU,X_GPU,X_PHONE)` candidates. First filter by estimated PC↔Phone per-token service-rate mismatch, then score complete 358-token Prefill including PC tail and shared network effects; print the selected theoretical layout and predicted wall time, but **execute the original nine Regions with uniform 64**. Env opt-in suggestion: `LLAMA_HYBRID_REGION_RING_PLANNER=dry-run`. Compare predicted baseline wall time with repeated real measurements; reject missing/uncertain profile data.

### B. Interval-aware READY/Slot scheduler

Replace global Chunk-ID dependencies with token ranges and exact edge coverage; test many-to-one and one-to-many fanouts, Buffer ownership, cancellation and KV order independently of a real Phone. Maintain an unchanged uniform-Chunk compatibility path.

### C. Apply independent Chunk sizes under fixed known-good Region layout

First apply hetero X in a fixed layout, with P5 UID/shape/slot validation and real Phone READY. Verify correctness before measuring speed. Report actual task count (54 is valid only for uniform 64).

### D. Apply per-cycle C/G/P layer balancing

Generalize Phone graph validators and autostart beyond L11/L27; support an optional **PC-only CPU/CUDA tail** after the last P-region, including correct Phone→PC transfer, final logits ownership and a generalized Ring parser. Reallocate device weights and CUDA/Phone residency safely before starting inference. Apply a selected plan at request setup, never halfway through a Prefill.

### E. A/B and auto-select

Compare repeated Release medians of complete Prefill wall time, busy/idle, overlap, template reuse and transfer stalls against fixed uniform-64 reference. A plan is acceptable only with unchanged tokens/logits and tensor parity, no leaks, no KV violations, and lower end-to-end makespan.

## 7. Logs and acceptance tests

Diagnostic names:
* `[REGION_RING_PLAN_PROFILE]` device, region, query tokens, K_i, compute/copy/transfer times.
* `[REGION_RING_PLAN_CANDIDATE]` each candidate's cycle count, C/G/P partition, PC tail, X sizes, estimated s_PC/s_PHONE, projected wall, occupancy, memory.
* `[REGION_RING_PLAN_SELECTED]` full layout, per-Region intervals, predicted wall.
* `[REGION_RING_PLAN_APPLY] mode=DRY_RUN|APPLIED` with explicit validation and rejection reasons.
* `[REGION_RING_EDGE_READY]` producer range, consumer range, coverage, Slot credits.
* `[REGION_RING_PLAN_REJECT]` reason, required guard and fallback.

Tests: 2-token warmup, 358-token Prefill including a 38-token tail, longer 512/1024 contexts, both 64↔32 transitions, Window=2 slot reuse, duplicate/out-of-order READY, cancellation, GPU/Phone memory limits, and full output/logit/selected `l_out` parity. Dense and MoE use the same scheduler abstraction but have different profiled costs.

V1 explicitly does **not** introduce new Tensor split FFN/Reduce semantics, on-the-fly Layer migration, fabricated graph UID reuse, or bypass existing ownership/fusion guards.


## 8. Implemented Dry-run search — runnable Phase A

The planner is now implemented behind an explicit opt-in. Phases B/C/D (actual heterogeneous READY/Slot scheduling, Phone graph placement, dynamic PC-tail final output) are not yet enabled.

- `src/llama-region-ring-planner.h` implements search for repeated `[C,G,P]×N` plus optional `C_tail/G_tail`, with independent `XC/XG/XP`, interpolated costs supplied by callbacks, a conservative shared network lane, per-Region KV ordering, token-interval edge coverage, staging-buffer transfer, and 2 outstanding producer slots per edge.
- `src/llama-hybrid.cpp` connects measured `cpu_layer_blocks`, `gpu_layer_blocks`, `phone_layer_blocks`, attention-KV interpolation, directional transfer cost curves and estimated weight/KV/scratch memory budgets. It conservatively includes Phone graph registration overhead by layer count. Estimated memory is not a real weight allocator.
- `src/llama.cpp` reuses the existing Hybrid Auto hardware profiler when dry-run is requested, even if `--hybrid-auto` was not otherwise enabled, **without publishing or changing the model placement**. A measured Profile can be used even when the old legacy Hybrid layout search found no feasible plan. If hardware profiling itself fails, the fixed Ring continues.
- `src/llama-context.cpp` calls the search at the start of a large prefill ubatch. Logs include reference predicted time and the highest-ranked candidate configurations. The **actual** Stage list, KV cache and Phone RPC execution are never modified by this option.
- `tests/test-region-ring-planner.cpp` tests exact 54-task uniform reference, different PC and Phone interval sizes, PC tail, missing Profile and memory filters. `test-region-ring-planner` is registered as a standalone C++17 CTest target and its checks run even in Release with NDEBUG.

### Quick start

Checkout branch `exp/region-ring-balanced-planner-20261026`, recompile PC `llama-cli` (the existing protocol-v17 Phone RPC server from the READY-autostart branch can be reused), keep the same fixed Ring flags and add:

```powershell
$env:LLAMA_HYBRID_REGION_RING_PLANNER = "dry-run"
$env:GGML_META_RING_PHONE_FUSION = "1"
$env:GGML_META_RING_PHONE_AUTOSTART = "1"
```

To build and run the standalone test when `LLAMA_BUILD_TESTS=ON`:

```powershell
cmake --build build-tensor-v2 --config Release --target test-region-ring-planner -j 8
ctest --test-dir build-tensor-v2 -C Release -R "^test-region-ring-planner$" --output-on-failure
```

Search these PC log markers:

```powershell
Select-String -Path ".\region-pair.log" -Pattern '\[REGION_RING_PLAN_PROFILE\]','\[REGION_RING_PLAN_SEARCH\]','\[REGION_RING_PLAN_CANDIDATE\]','\[REGION_RING_PLAN_SELECTED\]','\[REGION_RING_PLAN_REJECT\]','\[REGION_RING_PLAN_APPLY\]','\[REGION_RING_SUM\]'
```

`[REGION_RING_PLAN_APPLY] mode=DRY_RUN applied=0` is expected and mandatory for this phase. `predicted_ms` is estimated and should be calibrated against repeated, Release-mode `REGION_RING_SUM wall_ms`; no claims of speedup until an applied plan is verified.

Current bounded coarse search uses `N=2..4`, `C=2..12`, `G=1..8`, `P=2..8`, PC tail up to 12 layers, `XC/XG={32,64,128}`, `XP={16,32,64,128}`, and 192 full DAG simulations after profile/balance pruning. These bounds and a conservative serialized Wi-Fi lane do not prove global optimality. The CPU↔CUDA transfer estimate uses the GPU→PC profile as a proxy and needs calibration. The conservative Phone graph registration estimate assumes new UIDs until safe template reuse is confirmed.

If the selected layout moves a Phone Region, changes XC/XG/XP or ends with PC layers, it remains a **proposed** layout only. Before application, implement the interval-aware runtime and generalize current hard-coded Phone Graph/last-logits ownership guards.


### Calibration correction after the first real Dry-run

The first observed 358-token Dry-run proposed `N=4,C3,G3,P4,PC_tail_C8,XC/XG/XP=128`, estimated `3400.191 ms`, while the **unchanged runtime** actually took `4036.169 ms`. These are **different layouts**, so the two wall-time values cannot be interpreted as a measured speedup. The 5 top-ranked candidates had aggregate layer counts close or identical (CPU20/CUDA12/Phone16) and CPU busy time dominated their estimated wall time.

Root issue in the first simulation: it treated CPU and CUDA as two unconstrained simultaneous compute lanes, but the currently validated `fifo-pc-phone` runtime serializes their Graph compute through `region_ring_backend_execution_mutex` (Phone remote GPU execution may overlap after the lock is released). The predictor now sets `pc_compute_gate=true` by default, serializes CPU/CUDA Graph compute and uses combined PC compute demand in coarse ranking. An explicit `pc_compute_gate=false` remains only for a counterfactual future-backend-isolation test.

New log marker `[REGION_RING_PLAN_BASELINE]` prints the **predicted fixed-layout** wall time and PC Gate busy estimate on a short independent line, so compare that number directly with the runtime's `[REGION_RING_SUM]` for the same request. Candidate logs now include `pc_gate_busy`. The simulator is still an approximation: input staging, GPU/Phone graph preparation within the outer PC gate, lazy graph registration, two-way network contention and exact GPU memory residency require additional calibration before real APPLY. Do not extrapolate a performance improvement from the original ungated 3400 ms prediction.
