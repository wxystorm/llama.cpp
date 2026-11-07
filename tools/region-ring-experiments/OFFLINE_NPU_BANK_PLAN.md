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

**Experimental physical HTP dual-Bank runtime is now integrated**, opt-in only via
`GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2`. The Phone server allocates exactly two
mapped HTP repack-weight arenas, pre-primes the first two Regions, reserves
all NPU KV separately, and defers later Graph compilation. When all Chunks
from rank k have published their P2C output, the Bank Mover selectively
retires old Graph UIDs, stages/loads rank k+2 into the same physical arena,
recompiles its Chunk Graphs, and publishes READY. A generation counter blocks
stale Graph reuse. Host GGUF staging may overlap other-Bank HTP compute;
repack and Graph compilation remain serialized.

**Not yet Android device-verified.** The saved-plan runner still refuses
NPU oversubscription by default. After compiling and actually testing the
Phone runtime, operators may explicitly set
`--npu-banked-runtime-confirmed 1` in the offline Planner and
`-ConfirmBankedPhoneReady` when running the saved plan. Do not claim hardware
PASS solely because this source code is merged.

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

## Opt-in Phone environment

For the Qwen3-14B layout G9,P1,G9,P1,G9,P1,G9,P1, launch the existing
Android Phone RPC server with these environment variables, using a locally
available matching GGUF:

~~~sh
export GGML_RPC_PHONE_NPU_SERIAL_REGIONS=1,3,5,7
export GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS=9,19,29,39
export GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2
export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
export GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=1
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_INPUT=1
export GGML_RPC_PHONE_QWEN3_LOCAL_GRAPHLESS=1
export GGML_RPC_PHONE_NPU_HOST_STAGE_MIB=512
# Optional limit on two weight arenas only, in MiB (does NOT include KV):
# export GGML_RPC_PHONE_NPU_BANK_BUDGET_MIB=1536
~~~

The two ordered lists MUST have the same number of entries. Each Phone
Region is exactly one Qwen3 layer. The GGUF loader callbacks must remain
alive across refills. Do not combine this mode with legacy
`GGML_RPC_PHONE_NPU_PRELOAD_ASYNC=1` and
`GGML_RPC_PHONE_NPU_PRELOAD_WEIGHTS=1`.

Expect `[NPU_BANK_MEMORY] event=ALLOCATED/KV_BOUND` and
`[NPU_BANK] event=REFILL ... status=PASS`. Logs include the **real** HTP
Bank/Graph/KV allocation bytes, host-stage time, repack time, compile
time, and HTP lock wait. Hexagon's device memory query remains 0; the
two physical Bank allocations are verified by real rpcmem + FastRPC
allocation/mapping at startup, and later KV allocation must also succeed.

Host GGUF staging is bounded (default 512 MiB). Too large a Region or
host staging OOM triggers an explicitly logged serial-read fallback,
rather than silently caching a third Region's weights. HTP weights are
still repacked under the single HTP mutex.

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

If the selected plan requires HTP weight eviction, add
`--npu-banked-runtime-confirmed 1` to the offline Planner options ONLY after
the matching Phone Hexagon binary has actually passed the double-Bank test.
The saved runner also requires `-ConfirmBankedPhoneReady` for these plans.

The zero-valued example measurements are **placeholders** and will
intentionally be rejected. After replacing them with measured values,
the program saves plan.json and up to 100 ranked candidates.

For the same model and token-length prompt set, execute the saved layout
without repeating any Planner search:

~~~powershell
.\tools\region-ring-experiments\run_saved_plan.ps1 -Plan .\offline-plan-512\plan.json -Model E:\llama\models\Qwen3-14B-Q4_K_M.gguf -Rpc 10.39.52.114:50052 -PromptDir D:\prima.cpp\prompts512 -Out .\offline-plan-512-runs -NoMmap -ConfirmBankedPhoneReady
~~~

The Phone RPC server must already have the NPU Region list from the
plan.json property phone_npu_regions. The runner checks the model filename,
prompt token-size hints, one-Layer-per-HTP-Region support and bank runtime
readiness. Its planning mode is fixed apply-wide plus SkipProfile.

## Accuracy and remaining verification

The saved `predicted_ms_lower_bound` currently excludes the real HTP
refill critical path. It ranks candidates but is NOT a validated speed
prediction for oversubscribed NPU execution.

Device QA needed before enabling saved plans for large models:
1. Cross-compile Phone Hexagon + PC from this branch; run HTP double-Bank
   with 4+ NPU Regions, inspect `REFILL` and `NPU_BANK_MEMORY`.
2. Compare logits with the non-banked Qwen3 reference and confirm one
   compute per Chunk, complete P2C, KV sync and Decode correctness.
3. Run twice in the SAME Phone server to verify re-prime, old UID removal
   and stale refill cancellation. Deliberately request a too-small Bank
   budget and confirm startup fails without launching inference.
4. Compare 128/256/512/1024 prompts. Measure `host_stage_ms`,
   `repack_ms`, `compile_ms`, `htp_lock_wait_ms` and exposed Bank wait.
5. Ensure unbanked Phone mode remains unchanged with the opt-in disabled.

Only after collecting these timings should the offline DAG simulator
model HTP bank refill overlap and bottlenecks; it currently models the
two-Bank *capacity*, not the transfer schedule.
