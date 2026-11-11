# Qwen3-32B: Independent backend Profile -> APPLY_WIDE -> rolling Profile5

Branch: `feat/32b-independent-backend-profile-20261111`

This replaces the incorrect fixed-layout V0 bootstrap. **There is NO
`G8,P6,...` inference in the initial Profile.** The original HTP-only
Phone executor, PC GPU double banks, priority Bank Mover, verified C2P
streaming, Phone-owned P2C, and Drain-first DAG are unchanged.

## Device meanings

| Logical label | Physical backend | Standalone calibration |
| --- | --- | --- |
| G | PC CUDA | real full-layer graph; GPU weight H2D |
| C | PC CPU | real full-layer graph |
| N / P | Snapdragon HTP0..HTP3 | native Phone-owned HTP graph, striped by layer |
| A | Phone Adreno OpenCL | unavailable in **this HTP-only mode**, zero layers |

The general G/C/N/A scheme may omit any backend, including A. Do not
misidentify the RPC-visible CPU transport as NPU compute. The standalone
HTP-only mode has one RPC-visible backend and four *private* HTP sessions.

## Exact workflow

1. **Independent Profile**, only when there is no cached baseline or
   `-UpdateProfile` is specified. C++ model-load startup runs the actual
   `llama_hybrid_profile_gpu_transfer` and
   `llama_hybrid_profile_rpc` transport probes, then independently
   evaluates CPU/CUDA graph blocks and Phone-native HTP Graphs. A Phone
   RPC command performs **real HTP compute** against its local GGUF,
   private KV and synthetic activations. It never starts a Prompt,
   allocates a production Ring or builds a PC Phone Graph.
2. **Persist baseline** under `<Out>/independent-profile/`:
   `profile.csv` is read by the unchanged APPLY_WIDE C++ Planner;
   `profile.json` contains device stages, preparation, upload, output
   readback, and PC↔Phone/CUDA transfer curves. Each CSV point says
   `device,layers,tokens,kv_tokens,setup_ms,upload_ms,prepare_ms,compute_ms,
   terminal,samples`. HTP `compute_ms` is **total serial Phone Stage
   service**, including the Phone-host **device upload, HTP kernel compute
   and output readback** between chained NPU layers. The JSON separately
   reports `htp_kernel_ms`, `upload_ms`, `output_read_ms`,
   `upload_mib_s`, `readback_mib_s`, `cold_weight_bytes`,
   `cold_weight_load_ms`, and effective cold-load MiB/s.
   Every PC↔Phone or GPU weight transfer curve also includes measured
   `bytes`, `ms` and calculated `mib_per_s`. The cold HTP load rate
   includes GGUF read/repack and must **not** be called raw DMA bandwidth.
3. **Plan** once per Prompt, only *after* independent Profile succeeds:
   E pruning -> memory check -> Beam -> full DAG -> GPU-tail refinement.
   The original `APPLY_WIDE` implementation chooses the layer layout
   and XC/XG/XP; all Phone Regions are dynamically mapped to HTP and
   weights loaded from the Phone GGUF. The Phone has no manual region
   configuration.
4. **Inference** using the existing 32B double-bank/HTP-only executor.
5. **Profile5**: after every five completed PASS Prompts, rebuild
   `profile-epochs/vNN/real-stage-profile.csv` from the last five real
   PC/GPU/HTP Stage logs. Only the *same measured shapes* override the
   baseline; missing backend/chunk/depth samples remain in the
   independent Profile, **never zero cost**. Epoch manifests are immutable,
   contain the baseline SHA256 and are verified on resume.

The standalone Profile grid by default is `quick`: depths 2/4/6,
Chunks 32/64/96/128; Phone padded KV spans 256/512/1024, two repeats
per HTP point. `full` additionally probes 1 layer. A CPU/GPU block's
KV=x is truly measured; longer contexts are covered by independent
attention probes/fallback, *not fabricated by renaming the short-KV
measurement*. No fixed pipeline inference is used for calibration.

## PC and Phone must have RPC protocol patch 32

Update both binaries from the **same commit**. Phone stays on
`-d CPU`, as in the previous HTP-only setup. Do not turn on OpenCL for
the HTP-only experiment. Keep your original working Phone env and add:

```bash
export GGML_HEXAGON_NDEV=4
export GGML_RPC_PHONE_HTP_ONLY=1
export GGML_RPC_PHONE_NPU_PLAN_CONTROLLED=1
export GGML_RPC_REAL_STAGE_PROFILE=1
unset GGML_RPC_PHONE_NPU_SERIAL_REGIONS
unset GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS
```

The Phone server must have a locally accessible matching Qwen3-32B
GGUF (the existing `-m` startup option), because the NPU independent
probe loads real model layers directly from that GGUF.

The PC loads the same model for shape metadata and the independent
CPU/CUDA probes. Do **not** set `LLAMA_HYBRID_REGION_RING_SKIP_PROFILE=1`
or a forced G/P layout in the automatic mode.

## Launch smoke test (PowerShell)

Use a **new Out directory**: the old V0 run configuration and artifacts
must not be mixed with the independent backend calibration.

```powershell
git fetch origin
git switch feat/32b-independent-backend-profile-20261111
git pull --ff-only
cmake --build build-tensor-v2 --config Release --parallel 8 --target llama-completion test-region-ring-planner

.\tools\region-ring-experiments\run_32b_profile5_plan.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\32b-independent-profile-all40" `
  -ProfileGrid quick `
  -ProfileEvery 5 `
  -Batch 2048 -UBatch 2048 -NPredict 1 -Threads 8 `
  -Limit 2 -StopOnFail
```

After the two PASS runs, omit `-Limit 2` to resume all 40. To
force re-profiling **before any PASS prompt**, add `-UpdateProfile`;
after PASS data exists, use a new Out to preserve epoch reproducibility.

Expected first-startup logs:

```text
[PROFILE] event=INDEPENDENT_START ... fixed_ring=SKIPPED
[INDEPENDENT_PROFILE] ... device=CPU layers=... chunk=... status=PASS
[INDEPENDENT_PROFILE] ... device=GPU layers=... chunk=... status=PASS
[HTP_INDEPENDENT_PROFILE] ... layers=... chunk=... status=PASS
[INDEPENDENT_PROFILE] ... device=HTP layers=... chunk=... status=PASS
[INDEPENDENT_PROFILE] event=READY ... source=STANDALONE_G_C_N
[REGION_RING_PLAN_APPLY] ... applied=1
[PHONE_NPU_PLAN] event=RESET ...
[PHONE_NPU_PLAN] event=MAP ...
[REGION_RING_PHONE_PROFILE_PULL] ... status=PASS
```

If the HTP probe is rejected, inspect PC
`runs/<prompt>/attempt-*/engine.stderr.log` for
`[INDEPENDENT_PROFILE] reason=REAL_HTP_PROBE_FAILED`; inspect the
Phone server log for `[HTP_INDEPENDENT_PROFILE] reason=...`.
The *first* independent calibration does not need a completed
inference or manual Phone log copy.

## 255-token APPLY_WIDE Profile mismatch fix

The observed first-prompt failure showed:

```text
[REGION_RING_REAL_STAGE_PROFILE] status=LOADED points=60
[REGION_RING_PLAN_SEARCH] enumerated=413343 profiled=0 reject_memory=378351 reject_profile=34992
[REGION_RING_PLAN_APPLY] reason=NO_SCOREABLE_RUNTIME_PLAN
[REGION_RING_PLAN_APPLY] reason=NO_EXECUTABLE_HTP_ONLY_GPU_TAIL
```

The root cause is the **coarse Planner's logical KV=255/2=127**
against the independent HTP Profile's minimum **physical padded KV=256**.
The lower-bound interpolation rejected every memory-feasible remaining
candidate, then the legacy Phone CPU fallback was intentionally empty.
Phone-owned HTP queries now map to their actual 256-token-aligned KV
shape during both E-scoring and DAG simulation. This affects **only
N/HTP**, not CPU/GPU/normal RPC, and does not change available memory
or hide missing samples by pretending zero service time.

Also, the HTP-only search no longer refuses a valid GPU-tail candidate
only because the legacy **Phone-final baseline** is unscoreable.
The selected tail must **still pass** candidate memory, topology, and
full-DAG simulation. Unmeasured 1-layer HTP Region extrapolation has a
nonzero measured-anchor floor. Diagnostic lines now include:

```text
[REGION_RING_PROFILE_KV] source=PHONE_HTP_GRAPHLESS logical_kv=255 coarse_kv=127 padded_kv=256
[REGION_RING_HTP_MEMORY_DIAG] geometry=... session_map=... bank_shape=... host_ram=... gpu_vram=... phone_ram=... pass=...
[REGION_RING_PLAN_BASELINE] ... baseline=UNSCOREABLE action=VERIFY_CANDIDATE_GPU_TAIL
```

Rebuild the **PC** from this branch. If the Phone is already running
RPC patch 32 from this independent-Profile branch, its code is unchanged
by this Planner-only fix. Re-run the **same command and same `-Out`**
for resume; earlier FAIL/INTERRUPTED attempts do not count as PASS.
The existing 36 valid HTP baseline points are reused, without another
cold calibration. Do **not** bypass the memory guard to force selection.
If no Plan is found, use the new `[PLAN_DIAG]` and
`[REGION_RING_HTP_MEMORY_DIAG]` breakdown to identify the actual budget
limiting the search.

## V1 with unchanged NPU points: real HTP timeline export fix

A live ShareGPT40 run passed five V0 prompts, then produced:

```text
[PROFILE] event=REFRESH version=V1 completed=5 cpu_points=17 gpu_points=59 npu_points=36
[PLAN_DIAG] reason=NO_EXECUTABLE_HTP_ONLY_GPU_TAIL
```

**Cause of missing Phone calibration:** the RPC post-RUN exporter
unconditionally required `job.prepared.has_value()`. HTP-only Graphless
workers intentionally never construct that legacy OpenCL prepared slot:
they set `job.prepare_done_us` directly after input upload. Therefore
the Phone ran correctly, but its per-Chunk HTP timestamps were silently
discarded. This path now exports `upload_begin_us → upload_done_us`,
`upload_done_us → prepare_done_us`, and
`compute_begin_us → compute_end_us` for successful HTP jobs without
requiring the legacy OpenCL prepared slot. The Phone logs a summary:

```text
[PHONE_STAGE_PROFILE_EXPORT] ... complete=... graphless=... exported=... status=READY
```

PC `[REGION_RING_PHONE_PROFILE_PULL]` is `PASS` only if it actually
pulled positive rows; empty or unavailable pulls show `UNAVAILABLE`.

**Fail-safe Profile5 policy:** a Stage epoch with no matched real HTP
runtime samples (`manifest.json: diagnostic.phone == 0`) is **not**
used to rank HTP-only Plans, even if its merged CSV still contains the
independent 36 baseline NPU points. The immutable V1 CSV is retained
for postmortem; the runner explicitly logs `event=FALLBACK
rejected_version=V1 active=V0` and executes with the independent
calibration. If a later epoch with Phone samples still cannot produce
a scoreable GPU tail, the remaining retry for that Prompt uses V0 and
logs `event=PLAN_RETRY_FALLBACK`. The output row records the actual
Profile version, so a V0 fallback is never mislabeled V1.

**Recovery:** rebuild both PC and Phone from this branch and restart
the Phone server with `GGML_RPC_REAL_STAGE_PROFILE=1`. Continue with
the **same `-Out` directory**; do not erase V0 or immutable V1.
Existing five PASS results remain counted. V1 will be bypassed because
its old manifest has zero Phone runtime samples. After prompts 6–10
succeed, the V2 epoch can capture fresh post-fix real HTP measurements.
If `[PHONE_STAGE_PROFILE_EXPORT] exported=0`, inspect
`missing_timestamps` and whether the Phone environment enabled
`GGML_RPC_REAL_STAGE_PROFILE`.

## Limits and verification

The HTP-only 32B branch deliberately does not benchmark Adreno OpenCL
as A, or use A in automatic Plans. The general GCNA mixed-backend
Profile can add an independent Adreno probe later. GPU-tail selection
still refines existing Beam candidates; it is not a globally exhaustive
search. Model-load native CPU/GPU and transport profiling occurs
per process, although the expensive independent depth/Chunk grid is
cached once.

The new RPC path and C++ model-load integration require a **fresh
Windows and Android compile** followed by an actual phone-device PASS.
No performance or end-to-end hardware correctness is asserted by
these source commits alone. See
`.github/workflows/htp-realprofile-autoplan-smoke.yml` for
host C++ and Python regression jobs.
