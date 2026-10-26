# Region Ring Planner: first executable Plan (APPLY_SAFE)

Status: **implementation committed, pending PC/Phone compile, real-model
correctness and A/B performance validation.** Keep the earlier
~exp/region-ring-balanced-planner-20261026~ branch as the verified,
non-applying reference.

This phase changes ONLY the CPU/CUDA boundary inside each existing PC
window, while keeping the original 9-stage C/G/P Ring, exact Phone GPU
graph endpoints, 64-token Chunk, 2 output slots and Phone READY mode.

## Why APPLY_SAFE is narrower than Dry-run

The Dry-run research search may propose 2-4 loops, Phone P8,
different XC/XG/XP, or a final CPU/CUDA tail. Current runtime cannot safely
execute those shapes because Phone fusion/READY and the final logits owner
are not yet generalized.

The first executable search instead keeps:

- Three PC windows: [0,11), [16,27), [32,43), each split into CPU then CUDA
- Three Phone windows, unchanged: [11,16), [27,32), [43,48)
- Total CUDA weight allocation: 11 layers; total CPU: 22; Phone: 15
- Chunk64 on all 9 stages; Window=2; last output stays Phone-owned
- ~LLAMA_HYBRID_REGION_RING_EXEC=fifo-pc-phone~
- Phone P5 Graph fusion / READY ownership checks unchanged
- No runtime layer migration or dynamically resized KV/activation buffers

The current default remains:

~C7,G4,P5,C7,G4,P5,C8,G3,P5~

For example, the following is an **eligible manual test Plan**, NOT a
predicted winner:

~C8,G3,P5,C6,G5,P5,C8,G3,P5~

It transfers one CUDA layer from the first PC window to the second but
keeps the same eleven CUDA layers and exact Phone windows.

## Real Plan ownership across model and context

~src/llama.cpp~ profiles CPU/CUDA/Phone with existing Hybrid probes
**before loading model weights**. ~src/llama-hybrid.cpp~ searches only
safe candidates, checks simulation, memory feasibility and predicted gain
(2% default). On selection, the explicit Ring layout and uniform Chunk
are stored on ~llama_model~.

~src/llama-model.cpp~ uses the selected model-owned layout to place
CPU/CUDA/Phone model weights, not a fresh read of the baseline env layout.
~src/llama-context.cpp~ converts exactly that same model-owned layout
into live Region stages and verifies every layer's backend assignment.
Both sides reject a mismatched selected layout.

If Profile is unavailable, no supported candidate exists, or predicted
gain is below the threshold, no new layout is selected; the current fixed
Ring continues. Legacy Ring counts are inferred directly from the fixed
Ring when there is no legacy Hybrid Auto placement, so fallback does not
depend on stale PC/Phone/GPU count environment variables.

**Important:** this fallback happens during model initialization.
It is NOT automatic rollback after a real runtime kernel/Graph error.
Once weights have been placed, unexpected runtime ownership errors fail
closed: restart with the fixed Ring baseline.

## Step 1: Build and run standalone policy tests (PC)

~~~powershell
git fetch origin
git switch feat/region-ring-plan-apply-safe-20261026
git pull --ff-only

cmake --build build-tensor-v2 --config Release --target test-region-ring test-region-ring-planner -j 8
ctest --test-dir build-tensor-v2 -C Release -R "^test-region-ring(-planner)?$" --output-on-failure

cmake --build build-tensor-v2 --config Release --target llama-cli -j 8
~~~

If the build directory was configured without tests, reconfigure with
~LLAMA_BUILD_TESTS=ON~, preserving all other CUDA/RPC options.

The standalone tests validate legacy Ring layout/dependencies and safe
C/G boundary changes; they do **not** establish real Qwen3 numerical parity.
Phone RPC protocol remains v17, with no new server-side code in this phase.

## Step 2: Fixed-layout control

~~~powershell
$env:LLAMA_HYBRID_REGION_RING_V1 = "1"
$env:LLAMA_HYBRID_REGION_RING_EXEC = "fifo-pc-phone"
$env:LLAMA_HYBRID_REGION_RING_LAYOUT = "C7,G4,P5,C7,G4,P5,C8,G3,P5"
$env:LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS = "64"
$env:LLAMA_HYBRID_REGION_RING_PHONE_EXIT_FIX = "1"
$env:GGML_META_RING_PHONE_FUSION = "1"
$env:GGML_META_RING_PHONE_AUTOSTART = "1"

Remove-Item Env:LLAMA_HYBRID_REGION_RING_PLANNER -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT -ErrorAction SilentlyContinue
~~~

Run the **same** Qwen3-30B-A3B-Q4_K_M GGUF, RPC server, prompt, seed,
~--split-mode tensor~, context, batch, ~-t~, and other CLI parameters as
the verified baseline. Record the complete 358-token Prefill and output
token IDs / selected logits. No performance gain is assumed.

## Step 3: Manual same-layout Plan; verify the ownership path

~~~powershell
$env:LLAMA_HYBRID_REGION_RING_PLANNER = "apply-safe"
$env:LLAMA_HYBRID_REGION_RING_PLAN_TOKENS = "358"
$env:LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT = "C7,G4,P5,C7,G4,P5,C8,G3,P5"
~~~

This deliberately applies the original layout through the **new**
model-owned Plan path, isolating Plan integration from changed weights.
Require the following logs:

~~~text
[REGION_RING_PLAN_SELECTED] mode=APPLY_SAFE source=MANUAL ... status=SUPPORTED_OVERRIDE
[REGION_RING_PLAN_APPLY] ... phase=MODEL_LOAD applied=1 ... status=MODEL_OWNED
[REGION_RING_PLAN_APPLY] ... phase=WEIGHT_PLACEMENT applied=1 ... status=CONSISTENT
[REGION_RING_PLAN_APPLY] ... phase=RUNTIME_STAGES applied=1 ... status=CONSISTENT
[REGION_RING_OWNERSHIP_SUM] window=2 live_slots=0 ...
[REGION_RING_SUM] tokens=358 chunk_tokens=64 regions=9 chunks=6 executed=54 complete=1 ... status=0
~~~

Confirm generated output/token/logit parity with the original fixed run.
An identical layout is a functional Plan plumbing test, not a speed test.

## Step 4: Manual non-default safe Plan

~~~powershell
$env:LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT = "C8,G3,P5,C6,G5,P5,C8,G3,P5"
~~~

The manual override runs through the same structural, memory and
simulation guards as automatic APPLY; it does not waive correctness or
GPU capacity checks. It bypasses the predicted-gain threshold only to
permit deterministic A/B testing. Compare to Step 2/3. Any output drift,
Phone graph SKIP, RPC hang, GPU allocation error or slot leak requires
rolling back to the original layout; do not yet enable automatic search.

## Step 5: Automatic supported candidate selection

Remove the manual override. The Planner now picks the fastest supported
C/G split from the measured Profile and applies it only if the predicted
improvement is at least 2% (configurable).

~~~powershell
Remove-Item Env:LLAMA_HYBRID_REGION_RING_PLAN_LAYOUT -ErrorAction SilentlyContinue
$env:LLAMA_HYBRID_REGION_RING_PLAN_MIN_GAIN_PCT = "2"
$env:LLAMA_HYBRID_REGION_RING_PLAN_TOKENS = "358"
~~~

Watch for ~[REGION_RING_PLAN_SUPPORTED_SEARCH]~, up to five
~[REGION_RING_PLAN_SUPPORTED_CANDIDATE]~, and
~[REGION_RING_PLAN_SELECTED] mode=APPLY_SAFE~.

A selected candidate is only a **profile-model prediction**, NOT a
measured speedup. Earlier measured baseline ~4034 ms~ was
underestimated or overestimated depending on the model version; in the
PC-Gate-corrected comparison the baseline estimate was ~4740 ms~.
Use warmup-adjusted identical-protocol repeated Release runs and compare
the median real ~[REGION_RING_SUM].wall_ms~.

Useful full-line filter:

~~~powershell
Select-String -Path ".\region-pair.log" -Pattern 