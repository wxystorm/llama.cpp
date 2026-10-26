# Region Ring plan-driven runtime: general topology (experimental)

Branch: `feat/region-ring-plan-runtime-v1-20261026`

**Status: Implementation committed, unit tests and actual Windows CUDA +
Phone inference NOT yet verified on this revision.** Keep the previously
tested fixed-Ring branch as the fallback. This is a testable implementation,
not a claim that any predicted performance improvement has materialized.

## Supported automatic topology in this experimental mode

`LLAMA_HYBRID_REGION_RING_PLANNER=apply-topology` chooses a model-owned
plan before weight loading, using measured CPU/CUDA/Phone performance,
transfer timings and the existing memory/DAG simulator.

- 2-4 complete CPU -> GPU -> Phone cycles
- each Phone span has 2-8 layers
- an optional CPU or CPU+CUDA suffix, **required when applying** a
  general plan, so the final logits owner is the PC
- 48-layer Qwen3-MoE profile, Chunk=64, Window=2, PC CPU/CUDA gate
- no dynamic replanning or live weight migration
- minimum predicted gain defaults to 2%, controlled by
  `LLAMA_HYBRID_REGION_RING_PLAN_MIN_GAIN_PCT`
- if the Planner lacks enough measured data or doesn't beat the
  threshold, keep the original fixed layout without changing weights

Example *candidate only*:
`C6,G4,P8,C6,G4,P8,C8,G4`. It has 8 stages and for 358
prompt tokens runs 6 chunks -> 48 stage-chunk tasks. Candidate scores
are model estimates, not hardware measurements.

`topology-preview` still performs a non-applying search.
`apply-safe` still retains the original fixed Phone P5 and 9-stage
restrictions. No generic changes are applied to either old mode.

## Implementation paths

- `src/llama-region-ring.h`: variable-topology shape validation
- `src/llama-region-ring-planner.h`: executable topology search
  with an optional required PC tail
- `src/llama.cpp`: mode validation, measured Profile and immutable
  pre-weight-load plan decision
- `src/llama-model.cpp`: exact per-layer CPU/CUDA/Phone weights
- `src/llama-context.cpp`: same Stage layout, Phone mask registered
  with Meta, and main-scheduler CPU/CUDA final logits ownership
- `ggml/src/ggml-backend-meta.cpp`: dynamic contiguous Phone fusion,
  exact terminal `l_out` bridge checks, per-span auto-READY, graph
  capacity guard and fail-closed incomplete prefill fusion
- `ggml/src/ggml-rpc/ggml-rpc.cpp`: PC client and Phone server
  validate registered 2..8 layer Phone graphs before dispatching
  them to Phone MAIN

## CRITICAL: PC and Phone must run the same new RPC build

RPC protocol patch was bumped from **17 to 18**, to enforce matched
client and server binaries after changing Phone graph validation.
The HELLO handshake rejects a mismatch. Update the PC and Termux
`ggml-rpc-server` from this branch together; never mix an old Phone
server (patch 17) with a new PC client (patch 18).

### Windows PowerShell build and unit tests

```powershell
git fetch origin
git switch feat/region-ring-plan-runtime-v1-20261026
git pull --ff-only
cmake --build build-tensor-v2 --config Release --target llama-cli test-region-ring test-region-ring-planner -j 8
ctest --test-dir build-tensor-v2 -C Release -R "^test-region-ring(-planner)?$" --output-on-failure
```

If the test targets don't exist, reconfigure your existing build with
`LLAMA_BUILD_TESTS=ON` while retaining its working CUDA and RPC options.

### Termux Phone server

In the Phone source directory, fetch/switch/pull the **same branch**,
then reuse the original working Termux OpenCL/RPC build configuration:

```sh
git fetch origin
git switch feat/region-ring-plan-runtime-v1-20261026
git pull --ff-only
cmake --build build-region-v1 --target ggml-rpc-server -j 8
./build-region-v1/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -t 8 -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

Do not reuse an already-running patch-17 Phone process. Restart after
rebuilding. Keep your established server flags and model path if they
differ from this example.

### Apply topology on PC

Retain the working `--split-mode tensor`, RPC address, model path,
context, output, and sampling arguments. **Do not set
`--hybrid-auto`**, and do not manually set
`LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT`.

```powershell
$env:LLAMA_HYBRID_REGION_RING_V1 = "1"
$env:LLAMA_HYBRID_REGION_RING_EXEC = "fifo-pc-phone"
$env:LLAMA_HYBRID_REGION_RING_LAYOUT = "C7,G4,P5,C7,G4,P5,C8,G3,P5"
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS = "64"
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX = "1"
$env:GGML_META_RING_PHONE_FUSION = "1"
$env:GGML_META_RING_PHONE_AUTOSTART = "1"
$env:LLAMA_HYBRID_REGION_RING_PLANNER = "apply-topology"
$env:LLAMA_HYBRID_REGION_RING_PLAN_TOKENS = "358"
$env:LLAMA_HYBRID_REGION_RING_PLAN_MIN_GAIN_PCT = "2"
Remove-Item Env:LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT -ErrorAction SilentlyContinue
```

`LLAMA_HYBRID_REGION_RING_LAYOUT` here is only the original fallback
baseline; the new Planner's selected model-owned Plan overrides it.

## Verification gates

1. Both binaries accept RPC protocol patch 18 and Phone server loads weights.
2. `[REGION_RING_PLAN_SELECTED] mode=APPLY_TOPOLOGY source=AUTO` and
   `[REGION_RING_PLAN_APPLY] phase=MODEL_LOAD applied=1`.
3. `[REGION_RING_PLAN_APPLY] phase=WEIGHT_PLACEMENT applied=1`
   and `phase=RUNTIME_STAGES applied=1`, with identical layouts.
4. `[REGION_RING_PHONE_MASK]` and `[REGION_RING_PHONE_PLAN]`
   register the exact new Phone layer spans.
5. Phone logs `[PHONE_RING_AUTO] side=phone event=VALIDATE` and
   `READY_ENQUEUED` / `EXECUTE_DONE`; Meta handles each terminal
   `l_out` exactly once.
6. `[REGION_RING_FINAL_OWNER]` reports CPU or GPU with `result=MAIN`.
7. `[REGION_RING_OWNERSHIP_SUM] live_slots=0 status=DRAINED`.
8. `[REGION_RING_SUM] ... complete=1 status=0`, with
   `executed = regions * chunks`. Not every Plan has 54 tasks.
9. Compare generated output and layer/logit numeric parity with the
   fixed-Ring branch. Device changes can cause small floating-point
   differences; large drift or unexpected tokens require investigation.
10. Only after correctness passes, compare warmed repeated 358-token
    prefill wall-time medians against the original fixed layout.

If Phone fusion is incomplete, graph capacity is exceeded, or a PC-tail
graph unexpectedly includes a Meta split, this experiment **fails
closed** rather than inventing a different ownership path. A failure
during weight loading is not automatically recoverable in the same
process; restart with the previously tested branch and layout.

For the old standalone `topology-preview`, setting that mode rather
than `apply-topology` leaves all weights/stages unchanged. The old
`llama_hybrid_autoplan: no scoreable hybrid plans` warning refers to
legacy auto-placement and may coexist with a valid Region Planner
profile.
