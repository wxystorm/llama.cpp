# Region Ring general topology: Phase 1

Status: **topology parser and general search committed; not a working
variable-Phone runtime yet**. PC/CUDA/Phone inference correctness has NOT
been tested on hardware.

Branch: `feat/region-ring-plan-runtime-v1-20261026`
Base: `feat/region-ring-plan-apply-safe-20261026`

## Goal

Turn the model-owned Region Plan into the source of truth for all weight
placement, graph setup, FIFO scheduling and final output ownership, without
discarding the previous proven Ring path.

A candidate may be `C6,G4,P8,C6,G4,P8,C8,G4`, i.e. two complete C/G/P
cycles and a PC tail. As with the existing `apply-safe`, this is a Plan
*example*, **not a measured speed improvement**.

## What is implemented in this first commit series

- `llama_region_ring_parse(spec, n_layers, true)` can parse two to four
  CPU/GPU/Phone cycles and an optional CPU or CPU/GPU tail; the existing
  two-argument parser remains strict.
- `llama_region_ring_apply_topology_compatible` validates contiguous
  ownership, expected stage order and bounded Phone region widths.
- `llama_ring_planner::search_supported_topology` enumerates different
  cycles/Phone widths/tails while retaining Chunk=64 and Window=2. It
  filters using existing memory and compute/transfer profile functions,
  then runs the existing event simulation with PC-gate/KV/slot constraints.
- `LLAMA_HYBRID_REGION_RING_PLANNER=topology-preview` runs that
  search during model initialization, logs ranked candidates, and
  **never applies them**.
- Standalone test cases cover variable layout parsing, structural
  guards and planner simulation.

## Critical blockers before apply-topology

1. The Meta Phone graph path still contains exact P5 and fixed-L11/L27
   checks, as well as fixed intermediate terminal layers L15/L31.
2. The current threaded runtime demands Phone be the final owner and
   always reads the Phone worker's final result. A PC tail requires
   routing canonical logits ownership to the last actual Stage.
3. Meta phone fusion's mirror and terminal l_out bridge checks need to
   work on a **model-owned region mask** rather than the fixed environment
   layout. The same mask must guide remote ready/autostart handoffs.
4. Dynamic graph capacity must be proven for P8, and the real Phone
   registered graph cache must use the selected layer range.
5. The current conservative memory-feasibility checks are approximate.
   Final weight residency, KV and scratch allocation require real checks.

Never turn on unrestricted application by simply removing guards.

## PC-only standalone tests

```powershell
git fetch origin
git switch feat/region-ring-plan-runtime-v1-20261026
git pull --ff-only

cmake --build build-tensor-v2 --config Release --target test-region-ring test-region-ring-planner -j 8
ctest --test-dir build-tensor-v2 -C Release -R "^test-region-ring(-planner)?$" --output-on-failure
```

## Preview with the original tested inference layout

```powershell
$env:LLAMA_HYBRID_REGION_RING_V1 = "1"
$env:LLAMA_HYBRID_REGION_RING_EXEC = "fifo-pc-phone"
$env:LLAMA_HYBRID_REGION_RING_LAYOUT = "C7,G4,P5,C7,G4,P5,C8,G3,P5"
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS = "64"
$env:LLAMA_HYBRID_REGION_RING_PLANNER = "topology-preview"
$env:LLAMA_HYBRID_REGION_RING_PLAN_TOKENS = "358"
Remove-Item Env:LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT -ErrorAction SilentlyContinue
```

The existing model/Phone/CLI parameters remain unchanged, with
`--split-mode tensor`. The preview should show
`[REGION_RING_TOPOLOGY_SEARCH]` and
`[REGION_RING_TOPOLOGY_CANDIDATE]`, followed by
`[REGION_RING_TOPOLOGY_PREVIEW] applied=0 runtime_layout=UNCHANGED`.

This mode is solely for topology evaluation. Use the previous
`apply-safe` branch for verified inference until the remaining runtime
work and hardware A/B correctness testing are completed.
