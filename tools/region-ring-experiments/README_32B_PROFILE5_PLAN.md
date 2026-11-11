# 32B: rolling native Profile + per-Prompt APPLY_WIDE Plan (development)

Branch: `feat/32b-gcna-profile5-plan-chunks9-20261111`.

## Scope

- The existing E-window, memory lower-bound, 500/200 seed, Beam-16/24, full-DAG and local-refinement Planner is reused. It is **not** replaced by a separate Python Planner.
- The **automatic Planner** considers exactly nine Chunk widths for XC/XG/XP:
  `32,48,56,64,80,96,112,128,160`. Legacy forced diagnostic values remain valid for other tests; this setting changes the automatic search grid.
- Native G/C/N/A profiling is bootstrapped once at five measured anchor Chunk widths (32,64,96,128,160); the Planner interpolates the other four candidates, and selected anchor widths are remeasured after every **five successful Prompt runs**. All other previously validated device/model-specific samples are kept. Default refresh widths are `96,128` to cap profiling overhead; override with `-RefreshChunks`.
- All 40 ShareGPT Prompt files are processed independently. Each prompt loads the latest native Profile and runs the existing `apply-wide` search again; the PC-selected Phone NPU Stage layer ranges are installed using model-owned RPC. Adreno is still available as a **fallback** if the Planner chooses it; it is not a required manually configured Phone layer list.
- Each Prompt has an isolated attempt directory. Terminal output prints `[PROFILE]`, `[PROMPT_PLAN]`, `[PROMPT_RESULT]` including selected layout, runtime layout, XC/XG/XN (the last maps to legacy XP), NPU weight policy when logged, prediction and TPS.
- Reusing the same `-Out` skips PASS results, retries unfinished/failed prompts, and rejects different model/executable bytes or changed Prompt data. Profile versions and source logs are preserved.

## Requirements

The code is a **development integration** based on the GCNA flexible-tail branch. It is not yet a verified merge with the separately developed 32B resource-aware Drain-first scheduler. Running this command alone **does not imply** that the FIFO policy has been changed to the verified 32B Drain-first mode. Do not compare it as equivalent to that benchmark until the scheduler and DAG semantics have been ported and tested.

Both the PC and Phone must be built from the **same** source revision and support GGML RPC patch 36. The Phone's NPU Plan-controlled server, HTP and GPUOpenCL support, and direct native RPC profile endpoint must be configured as documented in `GCNA_FLEXIBLE_RUNTIME.md` and `V2_PHONE_PROFILE_RPC.md`. The original 32B GGUF bytes on Phone must match the PC model's SHA256. No `GGML_RPC_PHONE_NPU_SERIAL_REGIONS` or `MODEL_LOAD_LAYERS` manual list.

The model's NPU Region may contain 1..32 contiguous layers, but true HTP memory/graph allocation and output correctness remain required; pre-plan feasibility is not a guarantee of physical allocation. A plan that cannot be installed must fail explicitly rather than silently selecting arbitrary Phone GPU execution.

## Launch

In PowerShell from the project root, after building `llama-completion` and `region-ring-v2-stage-profile`, and starting the matching Phone RPC server:

```powershell
.\tools\region-ring-experiments\run_32b_profile5_plan.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\32b-gcna-profile5-plan-all40" `
  -ProfileEvery 5 `
  -RefreshChunks "96,128" `
  -Batch 2048 -UBatch 2048 -NPredict 1 `
  -Threads 8 -NoMmap
```

Test with `-Limit 2` first. Rerun the **same** command with the same `-Out` to resume; the limiter can be omitted when switching from quick validation to the full sweep. Existing PASS runs are not rerun.

First run: bootstrap the native G/C/N/A profiling matrix at five Chunk anchors (32,64,96,128,160) and matching KV anchors; the Planner still searches all nine requested widths. This can be expensive, especially for a 32B model. Subsequent profile epochs remeasure 96/128 only and keep the other sample rows. Native NPU measurements must pass the parity gate; unsupported shapes are not invented.

A per-Prompt terminal example (illustrative only, NOT a measured result):

```text
[PROFILE] event=REFRESH version=1 completed=5 chunks=32,48,56,64,80,96,112,128,160 refresh_chunks=96,128
[PROMPT] event=PLAN_AND_RUN prompt=prompt_512_06.txt tokens_hint=... profile=V1 attempt=1
[PROMPT_PLAN] prompt=prompt_512_06.txt status=PASS applied=True layout=... runtime=... XC=... XG=... XN=... npu_weights=RESIDENT predicted_ms=...
[PROMPT_RESULT] prompt=prompt_512_06.txt prefill=... tok/s prefill_ms=... attempt=1
```

Outputs are under `-Out`: `run-config.json`, `profile-state.json`, `summary.csv`, `runs/prompt_*/attempt-*/`. The separate model-native profile cache is under `model-profiles/<model>/native-gcna-v2`.

## Checks before interpreting results

1. `[REGION_RING_PLAN_SEARCH]` confirms candidate search rather than forced layout.
2. `[REGION_RING_PLAN_APPLY]`, `[PHONE_NPU_PLAN]`, and `[PHONE_OWNED_KV]` confirm selected layout, NPU mapping, and successful install.
3. Check `[PROMPT_PLAN] runtime=` against `layout=`. A baseline fallback is reported explicitly as `applied=False`, not misreported as a newly applied Plan.
4. Compare `predicted_ms` with actual `prefill_ms`; the model load, Plan setup and HTP graph compilation are separate costs and should not be conflated with Prefill.
5. Until the 32B Drain-first runtime and DAG simulator have been merged and tested on this branch, do not claim its performance matches the verified G8/P6 Drain-first benchmark.

## Host tests

```powershell
python -m unittest discover -s tools/region-ring-experiments -p "test_*v2*profile*.py" -v
cmake --build build-tensor-v2 --config Release --target test-region-ring-planner
ctest --test-dir build-tensor-v2 -C Release --output-on-failure -R "^test-region-ring-planner$"
```

Rebuild both Windows and Android runtimes before real device testing. Host tests and Snapdragon end-to-end runs have **not** been executed in this change environment.
