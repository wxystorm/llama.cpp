# Model-bound Profile then offline Plan (experimental HTP two-Bank)

Calibration is explicit and kept separate from planning. The official plan-only
entrypoint never runs a real/synthetic Profile, launches inference, or opens RPC.
A missing model-bound Profile is a hard error.

This experimental path currently supports Dense Qwen3 with single-layer HTP
Phone Regions (P1) and a **real** HTP two-Bank server. It is not a generic
Phone OpenCL calibration.

## 1. Build the PC tools

From the repo root in PowerShell:

    cmake --build build-tensor-v2 --config Release --target llama-completion llama-tokenize region-ring-offline-plan -j 8

## 2. Set up the Phone server for internal calibration probes

    .\tools\region-ring-experiments\run_model_profile.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -ShowPhoneEnv

Copy the resulting export lines to the Phone Termux shell **alongside the
normal OpenCL + HTP setup**. Start ggml-rpc-server with the same GGUF. The
internal probe geometry for 40-layer Qwen3 uses P1 at layers 9,19,29,39
and Phone region indices 2,5,8,11. This is not the later planned layout.
The tool chooses three internal G/C/P geometries and chunk widths
16,32,64,128, covering CPU, CUDA, and HTP1 service samples.

Do not use the previous G9,P1 repeated Phone region list (1,3,5,7):
the probe layouts include CPU stages, so their Region indices differ.

## 3. Collect PC and Phone real-stage measurements

    .\tools\region-ring-experiments\run_model_profile.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" -Rpc "10.39.52.114:50052" -NoMmap

This is intentionally the **only** command that launches calibration
inference. The first probe also gathers the legacy Hybrid transfer/memory
Profile; remaining probe runs use forced-layout/SkipProfile. The Phone RPC
server must have GGML_RPC_REAL_STAGE_PROFILE=1 and return P1 HTP timings.

After probes finish, copy the current Phone RPC log to the PC (for example
phone-npu-bank.log) and finalize:

    .\tools\region-ring-experiments\run_model_profile.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Finalize -PhoneLog ".\phone-npu-bank.log"

The output is:

    model-profiles/Qwen3-14B-Q4_K_M/
      model_profile.json             # model fingerprint, backend, calibration
      real-stage-profile.csv         # actual CPU/CUDA/HTP Stage samples
      runs/                          # probe logs/metrics/diagnostics

An existing calibration can be replaced explicitly with -UpdateProfile on
the collection command. This does NOT silently happen during planning.

## 4. Plan from the saved Profile only

    .\tools\region-ring-experiments\run_plan.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt"

Output under offline-plans/ includes plan.json and candidates.csv.
The model is identified and verified from GGUF metadata and a size +
first/last MiB fingerprint; the Prompt is tokenized with llama-tokenize.
Layer count, hidden size, KV size and the internal baseline come from GGUF
and the model's Profile. All layer allocations and chunk sizes are searched
by the existing offline Exact DAG Planner. They are NEVER user inputs.

The Phone server is not reconfigured by a plan-only command. Before applying
a chosen plan, restart or reconfigure Phone with its planned Phone Region
indices and absolute P1 layer IDs, and validate real two-Bank readiness.
For plans that require HTP streaming, use -BankedRuntimeConfirmed ONLY after
physical dual-Bank hardware tests have passed. This switch is a human
attestation; it is not an automatic hardware health check.

## Important caveats

- Existing offline_plan.cpp is still an approximate DAG model:
  **HTP Bank refill/Graph recompilation is not part of the predicted critical
  path**. Predicted wall time is a lower bound, not validated throughput.
- The saved Profile must contain measured HTP P1 timings, Bank allocation,
  KV-bound allocation, and PC Hybrid memory and transfer measurements.
  Missing metrics produce explicit errors, not synthetic zeroes.
- HTP Bank+KV+Graph observed allocation is used as a conservative *tested
  allocation floor*, not an invented query for full HTP free memory. This
  conservative heuristic may reject larger but feasible plans.
- Device characteristics and software builds matter; regenerate the model
  Profile after changing the backend, phone mode, CPU/GPU or runtime settings.
- The calibration setup performs real multi-device inference and depends on
  compatible current-branch binaries. It has been unit-tested with simulated
  GGUF and logs; it has **not yet been run on the real Android HTP device**.
