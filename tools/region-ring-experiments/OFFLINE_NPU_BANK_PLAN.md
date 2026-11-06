# Offline DAG Planner and Phone NPU two weight banks

## Implemented in this branch

- The existing C++ Planner now accepts caller-controlled DAG budgets. The
  default online values are unchanged. A standalone command,
  **region-ring-offline-plan**, reuses the exact same DAG simulator and genuine
  Stage CSV profiles, but increases E candidates, first-cycle seeds, prefix
  beam, final beam, throughput anchors and local search. It saves plan.json
  and candidates.csv without running inference.
- **run_saved_plan.ps1** reads the saved plan and calls the existing
  apply-wide/SkipProfile runner, so subsequent experiments do not plan again.
- **phone-npu-weight-banks.h** codifies the PC GPU's simplified fixed-parity
  A/B Bank policy with per-Region Chunk FIFO, physical capacity maxima,
  strict Graph-retirement-before-refill, shared-KV residency and a single
  global HTP compute lane. There is a standalone deterministic test.

**Important runtime limit:** the NPU policy is a lifecycle and memory
module, NOT yet the real HTP weight-transfer engine. The current HTP
executor still precompiles Graphs referencing its own permanently allocated
weight tensors. Until those Graph references and physical weight buffers can
be safely recycled, a plan that requires evictions is marked
npu_streaming_required=true / npu_streaming_runtime_ready=false, and the
saved-plan runner deliberately refuses to run it. Do not claim an
out-of-memory model can already run on two device weight banks.

## GPU Bank Mover behavior to preserve exactly

1. NPU Region weight ranks are assigned in execution order, **even to Bank A,
   odd to Bank B**. Each Bank has a fixed capacity equal to the largest
   repacked Region of that parity. KV and scratch are separately charged.
2. The first two Regions are primed before the prefill. There is no fallback
   to an unlimited persistent weight cache.
3. The single compute lane may run one Chunk at a time. The prior Region's
   LAST Chunk must complete, its P2C result must be published, and its
   compiled Graph references must be retired before Bank rank k can be
   repurposed for rank k+2. Bank readiness includes a generation check.
4. A single Bank Mover stages/uploads weights into the freed Bank; only a
   successful upload and Graph recompile publishes READY. Failures propagate
   instead of using stale weights.
5. CPU-side GGUF staging can overlap HTP computation. **Do not assume HTP
   device weight upload/repack overlaps compute** without proving the
   Hexagon backend supports it; the present HTP compute mutex is necessary.
6. Per-layer HTP KV stays independent and resident throughout prefill.
   Weight refill never resets or reallocates KV.

## Build and unit test

From the repo root on Windows:

~~~powershell
cmake --build build-tensor-v2 --config Release --target region-ring-offline-plan test-phone-npu-weight-banks -j 8
ctest --test-dir build-tensor-v2 -C Release -R "test-phone-npu-weight-banks" --output-on-failure
~~~

## Offline plan once, reuse later

The Planner requires matching real Stage calibration
(real-stage-profile.csv, produced by build_real_stage_profile.py), and
**measured** HTP-repacked weight MiB per assigned layer, KV MiB per
layer, memory budgets and link throughput. Do not substitute raw GGUF
weight size for HTP repacked usage. Recalibrate when changing models.

Example: set numbers to actual measured values before running:

~~~powershell
$gpuMem = 0          # replace with safe GPU MiB budget
$npuMem = 0          # replace with safe HTP MiB budget
$gpuLayerMiB = 0     # measured CUDA layer weights
$npuLayerMiB = 0     # measured HTP repacked layer weights
$gpuKVMiB = 0        # GPU KV per layer
$npuKVMiB = 0        # NPU KV per layer
$h2d = 0             # GPU Bank upload throughput MiB/s
$c2p = 0             # PC to Phone throughput MiB/s
$p2c = 0             # Phone to PC throughput MiB/s
$local = 0           # PC local tensor copy throughput MiB/s
$opt = @(
  "--profile", "E:\llama\profiles\qwen14b-real-stage.csv",
  "--out", ".\offline-plan-512",
  "--model-id", "Qwen3-14B-Q4_K_M.gguf",
  "--layers", "40", "--tokens", "512", "--hidden", "5120",
  "--baseline", "G9,P1,G9,P1,G9,P1,G9,P1",
  "--gpu-budget-mib", "$gpuMem", "--npu-budget-mib", "$npuMem",
  "--gpu-weight-layer-mib", "$gpuLayerMiB",
  "--npu-weight-layer-mib", "$npuLayerMiB",
  "--gpu-kv-layer-mib", "$gpuKVMiB", "--npu-kv-layer-mib", "$npuKVMiB",
  "--gpu-h2d-mibps", "$h2d", "--c2p-mibps", "$c2p",
  "--p2c-mibps", "$p2c", "--local-mibps", "$local",
  "--e-margin", "0.30", "--min-dag", "256", "--max-dag", "4096",
  "--first-pool", "1500", "--first-seeds", "500",
  "--beam", "64", "--final-beam", "96", "--cpu-dag", "256",
  "--local-dag", "128", "--local-rounds", "4", "--top", "100"
)
& .\build-tensor-v2\bin\Release\region-ring-offline-plan.exe @opt
~~~

The zero-valued example measurements are **placeholders** and will
intentionally be rejected. After replacing them with measured values,
the program saves plan.json and up to 100 ranked candidates.

For the same model and token-length prompt set, execute the saved layout
without repeating any Planner search:

~~~powershell
.\tools\region-ring-experiments\run_saved_plan.ps1 -Plan .\offline-plan-512\plan.json -Model E:\llama\models\Qwen3-14B-Q4_K_M.gguf -Rpc 10.39.52.114:50052 -PromptDir D:\prima.cpp\prompts512 -Out .\offline-plan-512-runs -NoMmap
~~~

The Phone RPC server must already have the NPU Region list from the
plan.json property phone_npu_regions. The runner checks the model filename,
prompt token-size hints, one-Layer-per-HTP-Region support and bank runtime
readiness. Its planning mode is fixed apply-wide plus SkipProfile.

## Accuracy and remaining physical integration

The saved predicted_ms_lower_bound includes the existing computation,
GPU Bank streaming and C2P/P2C DAG model. **It does not yet simulate the
NPU weight refill critical path.** Its purpose is candidate ranking, not
a validated oversubscribed NPU latency forecast.

Next implement the physical HTP A/B arenas, host-stage queue and
repack/upload worker, rank-k drain and P2C completion barrier, Hexagon
fixed-Graph invalidation/rebind, and true READY publication. Test HTP
weight residency under memory pressure before enabling runtime streaming.
Then measure refill queue wait, CPU-stage, HTP-upload, graph compilation
and exposed/hidden refill time, and incorporate those measurements into
the offline DAG simulator.
