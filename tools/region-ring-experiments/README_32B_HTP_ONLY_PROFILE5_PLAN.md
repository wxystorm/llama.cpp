# 32B rolling real Stage Profile + automatic Plan on the proven HTP-only architecture

Branch: `feat/32b-htp-only-profile5-autoplan-20261111`, **based on**
`feat/32b-sharegpt40-clean-resume-20261110` (NOT on GCNA's
OpenCL-visible mixed-backend branch).

## What remains unchanged

- Phone RPC is exposed as **`-d CPU`**, acting only as an RPC transport.
  Adreno/OpenCL is **not** required or used for Phone Transformer compute.
- HTP0..HTP3 are private multi-session NPU executors. They own immutable
  Qwen3 layer weights, per-request KV and local HTP Graphs. GPU double-bank
  streaming, PC/Phone input PUSH, P2C and **resource-aware Drain-first**
  remain the existing 32B runtime paths.
- C++ APPLY_WIDE still performs E-value/pruning, real memory limits,
  prefix Beam, full-Prompt DAG, and local refinement. Its automatic Chunk
  candidates (XC/XG/XP) are now EXACTLY:
  **32, 48, 56, 64, 80, 96, 112, 128, 160**.
- The verified *fixed* 32B experiment commands remain available and do not
  opt into dynamic Phone mapping or rolling Profile.

## What's new

1. `GGML_RPC_PHONE_NPU_PLAN_CONTROLLED=1` with
   `GGML_RPC_PHONE_HTP_ONLY=1`: startup initializes four independent HTP
   sessions with **no manual NPU_SERIAL_REGIONS**, while still requiring a
   single RPC-visible CPU backend. This is NOT the GCNA mixed GPU mode.
2. In automatic HTP-only mode, model load calls the existing weight-preload
   RPC on a separate socket to transmit each PC-selected Phone Region's
   exact contiguous layer range. This installs the Region -> HTP session
   mapping **before Phone-owned Plan registration**.
3. Before each new Plan, a reserved `PHONE_NPU_WEIGHT_PRELOAD` reset
   descriptor clears the previous Plan's idle-session Graph/KV/HTP weight
   residency, then all selected NPU weights are reloaded from Phone's GGUF.
   This prevents unbounded accumulation across differently planned Prompts.
   It is outside the measured Prefill, but setup time remains important.
   Reset is explicitly rejected if an existing Phone RUN is still active.
4. An HTP-only-specific **post-Beam GPU tail check** tests 1..4 final GPU
   layers split from existing Phone-final candidates. Every candidate
   reuses the original memory constraints and full DAG simulator.
   Only a valid GPU-tail Plan can be accepted by the automatic HTP-only
   mode. Original fixed-plan and ordinary Planner paths are not changed.
5. Before the first automatic Plan, the runner performs **one real, uncounted
   fixed HTP-only calibration** using the known G8/P6/.../G8 layout.
   Its actual CPU/GPU/HTP Stage service samples seed `profile-bootstrap/v00/`
   (V0); the RPC-visible CPU is never treated as HTP compute.
   After every **5 completed PASS Prompts**, the runner creates an immutable
   `profile-epochs/vNN/real-stage-profile.csv` using ONLY the last five
   successful Prompt attempts. Phone HTP samples are returned automatically
   by RPC after RUN teardown. Subsequent Prompts consume the newest CSV,
   but each still calls the C++ APPLY_WIDE Planner independently.
   Failed attempts, calibration, and warmup do not advance an epoch.
6. Every Prompt prints the selected logical and physical HTP-only layout,
   XC/XG/XN (XN is the old XP), prediction, actual TPS and Profile epoch.
   PASS results and per-attempt traces are stored for resume.

## Important Profile qualification

This builds real Stage CSV snapshots **from actual completed inference**.
Automatic HTP-only mode **requires real Phone HTP samples**. The initial V0
fixed-layout calibration is one actual Prompt Prefill on the four HTP
sessions; it is never included in the measured ShareGPT40 totals.
The first AutoPlan is not attempted until V0 contains positive Phone compute
data. V1, V2 and later are rolling five-success windows and reject an epoch
without Phone HTP samples. Native CPU/CUDA and actual transport probes remain
available as cost fallbacks where appropriate, but the synthetic RPC CPU
Graph benchmark is skipped for Phone HTP compute.

Server-side `GGML_RPC_REAL_STAGE_PROFILE=1` is **required**. Protocol patch
**31** adds a bounded batch read of verified HTP Chunk service samples after
RUN END. PC writes session-matched `[REAL_STAGE_PROFILE] side=PHONE
source=RPC_HTP` records into its own `engine.stderr.log`. No per-Chunk
RPC messages, no manual Phone log copies, and no Phone Graph benchmark
masquerading as NPU service time.

For an unmeasured candidate width the real Stage interpolation may apply a
conservative bounded HTP estimate. This is an estimate from real anchor
samples, not a direct measurement at every width; monitor predicted-vs-actual
errors. Graph registration without an actual measured setup value falls back
to the existing registration estimate rather than being reported as zero.

`-PhoneProfileLog` remains an optional compatibility input for archival
Phone logs; it is not necessary with the new RPC sample path.

HTP mappings are virtual FastRPC domains, not four independent pools of
physical RAM. A candidate passing estimated memory constraints can still
fail real HTP weight allocation or Graph precompilation; fail closed, inspect
logs, and do not relabel the result PASS.

## Phone startup (Termux)

Build **both** PC and Android from this branch, install the same GGUF on
Phone, and start the existing RPC server as follows (not `-d GPUOpenCL`):

```sh
cd ~/phone-381
export LD_LIBRARY_PATH="$PWD/lib:$PWD/bin:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$PWD/lib"
export GGML_HEXAGON_NDEV=4
export GGML_HEXAGON_HOSTBUF=1
export GGML_RPC_PHONE_HTP_ONLY=1
export GGML_RPC_PHONE_NPU_PLAN_CONTROLLED=1
export GGML_RPC_REAL_STAGE_PROFILE=1
unset GGML_RPC_PHONE_NPU_SERIAL_REGIONS
unset GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS
unset GGML_RPC_PHONE_QWEN3_SHARED_SCRATCH
export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
export GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_INPUT=1
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=1
export GGML_RPC_PHONE_QWEN3_LOCAL_GRAPHLESS=1

./bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d CPU \
  -m "$HOME/models/Qwen3-32B-Q4_K_M.gguf" 2>&1 | tee phone-profile5.log
```

The RPC port should be accessible **only on a trusted local network**.
This experimental server does not authenticate remote clients.

Startup: four `[PHONE_NPU_SESSION] event=INIT` rows followed by
`[PHONE_NPU_SERIAL] event=BACKEND_INIT ... rpc_visible=CPU_TRANSPORT_HTP_ONLY`.
The Region count can initially be 0; PC installs it later.

Each automatic Prompt should produce `[PHONE_NPU_PLAN] event=RESET`,
then one `[PHONE_NPU_PLAN] event=MAP` per selected N Region, then verified
`[PHONE_NPU_EARLY_WEIGHTS] ... status=PASS`.
PC additionally prints `[REGION_RING_PHONE_PROFILE_PULL] status=PASS`
after the Phone RUN ends. This requires **PC and Phone both rebuilt from
the same patch-31 commit**. The fixed V0 calibration precedes the first
automatic Prompt and saves its own logs under `profile-bootstrap/v00/`.

## PC launch

```powershell
git switch feat/32b-htp-only-profile5-autoplan-20261111
cmake --build build-tensor-v2 --config Release --parallel 8 --target llama-completion test-region-ring-planner

.\tools\region-ring-experiments\run_32b_profile5_plan.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\32b-htp-only-profile5-plan-all40" `
  -ProfileEvery 5 `
  -Batch 2048 -UBatch 2048 -NPredict 1 -Threads 8 `
  -Limit 2
```

First use `-Limit 2` as a smoke test. The uncounted V0 fixed HTP calibration
runs first; its PASS must not be confused with a measured Prompt PASS.
Once both succeed, re-run
the same command without `-Limit 2` and with the **same `-Out`** to
continue. Successful Prompts are skipped, Profile versions are recovered
from completed-Prompt counts and immutable snapshots. Optional
`-NoMmap` requires enough PC RAM.

Illustrative terminal output (not a measured sample):

```text
[PROFILE] event=BOOTSTRAP_FIXED version=V0 prompt=prompt_512_01.txt counted=0
[REGION_RING_PHONE_PROFILE_PULL] ... rows=... status=PASS
[PROFILE] event=BOOTSTRAP_READY version=V0 npu_points=... csv=...
[1/40] RUN prompt_128_01.txt group=128 ctx=...
  PASS: prefill=... tok/s (...) ...
  [PLAN] prompt=prompt_128_01.txt applied=True profile=V0 physical=G8,N4,G8,N3,G... transport=CPU source=htp_gpu_tail chosen=G8,P4,G8,P3,G... XC=64 XG=96 XP=128 predicted_ms=...
[PROFILE] event=REFRESH version=V1 completed=5 cpu_points=... gpu_points=... npu_points=... csv=...
```

The terminal uses `N` for Phone HTP; native C++ internal stages retain the
legacy label `P`. Every `P` in HTP-only mode is executed on the NPU,
not Adreno. V0 is `FIXED_HTP_REAL_CALIBRATION`, NOT `NATIVE_PROBES`.
The per-session mapping guard defaults to 3584 MiB/session and may be
adjusted with `LLAMA_REGION_RING_HTP_SESSION_BUDGET_MIB` after measuring
the actual HTP address mapping. The original aggregate Phone free-memory
budget still applies because the sessions share physical RAM.

## Testing and diagnosis

```powershell
python -m unittest discover -s tools/region-ring-experiments -p "test_rolling_real_stage_profile.py" -v
python -m unittest discover -s tools/region-ring-experiments -p "test_run_32b_all40.py" -v
ctest --test-dir build-tensor-v2 -C Release --output-on-failure -R "test-region-ring-planner|test-region-ring$"
```

Check the per-Prompt `metrics.jsonl`, `engine.stderr.log` and
`result.json`. Critical milestones:

- `[REGION_RING_PLAN_SEARCH] ...` must appear per Prompt.
- `[REGION_RING_PLAN_HTP_TAIL] status=PASS` must identify the GPU-tail plan.
- `[REGION_RING_PLAN_APPLY] ... applied=1` and matching weight placement.
- `[PHONE_NPU_PLAN] event=RESET/MAP`, `[PHONE_HTP_ONLY] event=REGISTER`,
  `[PHONE_HTP_ONLY] event=RUN_END ... status=PASS`.
- PC exit `0` and complete Prefill metrics are required for PASS.

**Verification boundary:** Changes are committed and CI checks are configured, but successful Windows/Android compilation and Snapdragon end-to-end execution have not been observed here. The host-side tests must be executed, not assumed to pass. Earlier source-only
tests. They have **not** been compiled on the user's Windows/Android toolchains
or executed on the Snapdragon device here. Start with the two-Prompt smoke,
then diagnose the first failing log rather than treating source review alone
as proof that arbitrary auto Plans run.
