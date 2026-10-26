# Region Ring Balanced Planner V1

Status: design specification / implementation roadmap. **No planner code is enabled yet.** This branch is based on `exp/region-ring-phone-ready-autostart-20261026`; the verified old Ring and READY execution remain unchanged.

## 1. Design objective

Optimize a **whole finite Prefill** rather than one isolated Chunk. Each cycle has consecutive CPU (C), CUDA (G), Phone GPU (P) Regions; independent device workers can pipeline adjacent token intervals. A balanced cycle does not require equal layer counts, and even equal single-Chunk Region time may not minimize fill+drain, communication and Window=2 stalls.

Known 48-layer reference: `C7,G4,P5,C7,G4,P5,C8,G3,P5` (three cycles of 16 layers), common `X=64`, 358 tokens producing 6 Chunks and 54 Region tasks, Window=2. Phone fusion/autostart currently validates only L11–15 and L27–31; L43–47 remains conservative legacy execution.

The planner jointly chooses:
* per-cycle contiguous layer counts `(C_i,G_i,P_i)` with `C_i+G_i+P_i=16` in the initial search, each count positive, and all 48 layers covered in order;
* independent `X_CPU`, `X_GPU`, `X_PHONE` Chunk token sizes, optionally per cycle later;
* Token partitions, output-slot/credit requirements and reusable Phone Graph shape/slot/KV keys;
* predicted full-Prefill makespan, resource occupancy, memory peak and transfer/registration cost.

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

Layer search first compares per-cycle partitions near the known reference, e.g. `C7,G4,P5` vs `C6,G5,P5`; broader `P4/P6` alternatives are **score-only** until they can use a genuinely dynamic Phone graph identity/ownership path. Candidate P placements must satisfy weight residency, Phone memory and graph correctness constraints, not just predicted compute speed.

Profile task costs as functions of actual token interval, Layer count and KV length:
`T_region(device, region, n_query, K_i)` including Graph preparation, device compute, scratch, cache misses and unavoidable copies. Use real `K_i` per interval rather than one constant KV approximation.

Profile separately:
* CPU Region and CUDA Region compute (including GPU layer weight upload/LRU misses and PC gate);
* Phone GPU server compute and READY queue, Graph REGISTER cost only once per new UID, and output `l_out` handoff;
* H2P and P2H bandwidth/latency during overlapping transfers, RPC Fence/READY latency, and slot occupancy;
* model weight residency, KV cache, Graph cache, per-slot scratch and reserve memory on each device.

## 4. Load balancing is a dependency-DAG simulation

Build task nodes `(region, token_begin, token_end)`, with compute lanes `PC_CPU`, `PC_CUDA`, `PHONE_MAIN`, and transfer lanes `H2P/P2H`. Match the actual PC gate, shared resources, Phone Main FIFO and output Window credits. Include all three 16-layer cycles, start-up/fill, steady state and drain.

For each physical resource d, report busy time, idle time and critical-path blocking time. Within each cycle report `C_i_ms/G_i_ms/P_i_ms` as a diagnostic; do not simply force them equal. Approximate steady-state initiation interval is bounded by the busiest lane, but rank plans by the **simulator's total makespan** on the actual finite Prompt.

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

Collect per-region CPU/CUDA/Phone and transfer timings, then enumerate (C/G/P,X_CPU,X_GPU,X_PHONE) candidates. Simulate full 358-token Prefill; print the chosen theoretical layout and predicted wall time, but **execute the original nine Regions with uniform 64**. Env opt-in suggestion: `LLAMA_HYBRID_REGION_RING_PLANNER=dry-run`. Compare predicted baseline wall time with repeated real measurements; reject missing/uncertain profile data.

### B. Interval-aware READY/Slot scheduler

Replace global Chunk-ID dependencies with token ranges and exact edge coverage; test many-to-one and one-to-many fanouts, Buffer ownership, cancellation and KV order independently of a real Phone. Maintain an unchanged uniform-Chunk compatibility path.

### C. Apply independent Chunk sizes under fixed known-good Region layout

First apply hetero X in a fixed layout, with P5 UID/shape/slot validation and real Phone READY. Verify correctness before measuring speed. Report actual task count (54 is valid only for uniform 64).

### D. Apply per-cycle C/G/P layer balancing

Generalize Phone graph validators and autostart beyond L11/L27; protect the final P-region with final logits ownership checks. Reallocate device weights and CUDA/Phone residency safely before starting inference. Apply a selected plan at request setup, never halfway through a Prefill.

### E. A/B and auto-select

Compare repeated Release medians of complete Prefill wall time, busy/idle, overlap, template reuse and transfer stalls against fixed uniform-64 reference. A plan is acceptable only with unchanged tokens/logits and tensor parity, no leaks, no KV violations, and lower end-to-end makespan.

## 7. Logs and acceptance tests

Diagnostic names:
* `[REGION_RING_PLAN_PROFILE]` device, region, query tokens, K_i, compute/copy/transfer times.
* `[REGION_RING_PLAN_CANDIDATE]` each candidate's cycle partition, X sizes, projected wall, occupancy, memory.
* `[REGION_RING_PLAN_SELECTED]` full layout, per-Region intervals, predicted wall.
* `[REGION_RING_PLAN_APPLY] mode=DRY_RUN|APPLIED` with explicit validation and rejection reasons.
* `[REGION_RING_EDGE_READY]` producer range, consumer range, coverage, Slot credits.
* `[REGION_RING_PLAN_REJECT]` reason, required guard and fallback.

Tests: 2-token warmup, 358-token Prefill including a 38-token tail, longer 512/1024 contexts, both 64↔32 transitions, Window=2 slot reuse, duplicate/out-of-order READY, cancellation, GPU/Phone memory limits, and full output/logit/selected `l_out` parity. Dense and MoE use the same scheduler abstraction but have different profiled costs.

V1 explicitly does **not** introduce new Tensor split FFN/Reduce semantics, on-the-fly Layer migration, fabricated graph UID reuse, or bypass existing ownership/fusion guards.
