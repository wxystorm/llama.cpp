# Qwen3-32B: single independent backend Profile -> APPLY_WIDE for all prompts

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

## Crash guard: real Phone shared RAM and per-layer HTP Session memory

The previously selected layout `G7,N9,G9,N7,G7,N9,G8,N1,G7` assigned
26 Qwen3-32B layers to the Phone. Its four Region preloads finished,
and the first three NPU Regions executed, but the Phone log stopped
after HTP3's last Layer 56 weight cache hit, before its Graphless
precompile result. The log **does not prove** whether Android low-memory
killer, FastRPC mapping exhaustion, or an HTP driver fault terminated it.

The Planner now validates each candidate using the Phone's real
`MemAvailable` from `/proc/meminfo` (exposed by HTP-only RPC CPU
transport). If the snapshot cannot be read, automatic HTP planning
**fails closed**, rather than assuming `free == total`. This is a
runtime memory snapshot, not a guarantee against Android memory
pressure changing later.

The model follows the Phone runtime's actual mapping: each NPU Region
starts at the next HTP Session; each consecutive layer advances to the
next session round-robin. Weights include a conservative 15% repack
margin. Each **individual Phone layer** reserves persistent F16 KV,
Graphless scratch sized from the actual Prompt's maximum Chunk and
model dimensions, plus per-Session and shared staging headroom.
The measured crash at Chunk 167 used 121.195 MiB scratch *per layer*;
the conservative analytical estimate is about 144 MiB. The observed
HTP op-batch virtual mapping limit was 3200 MiB per Session, so the
default allowed estimate is **2880 MiB per Session**, not the previous
3584 MiB. You may reduce this with
`LLAMA_REGION_RING_HTP_SESSION_BUDGET_MIB` (256–3200); raising beyond
3200 is intentionally rejected in this hardware-specific experimental
mode. These are still estimates, not queried HTP allocator statistics.

`[REGION_RING_HTP_MEMORY]` logs the current physical/shared capacity
and `[REGION_RING_HTP_MEMORY_SELECTED]` logs the selected layout's
per-Session layer counts and estimated memory usage. A candidate that
fails either budget is rejected and Planner continues its usual search:
no fixed NPU-layer cap and no modification to Profile V0.

**Both PC and Phone must be rebuilt from the same new commit.** This
change affects the Phone RPC `GET_DEVICE_MEMORY` response semantics.
Restart the Phone server using the existing
`GGML_RPC_PHONE_NPU_PLAN_CONTROLLED=1` / `-d CPU` HTP-only setup.

After rebuilding, use a **fresh `-Out`** to keep benchmark executable
identities and old results reproducible. To avoid calibrating a second
time, reuse the already completed standalone V0 as an explicit CSV:

```powershell
.\tools\region-ring-experiments\run_32b_profile5_plan.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\32b-htp-memory-guard-all40" `
  -RealStageProfileCsv ".\32b-independent-profile-all40-final\independent-profile\profile.csv" `
  -ProfileEvery 0 -ProfileGrid quick `
  -Batch 2048 -UBatch 2048 -NPredict 1 -Threads 8 `
  -Limit 1 -StopOnFail
```

The CSV and neighboring JSON must both be present and contain the
previous successful G/C/N independent calibration. After smoke PASS,
omit `-Limit 1` to run all remaining prompts in the same new Out.
If the prior calibration is incomplete, omit `-RealStageProfileCsv`
in a fresh Out and the runner will calibrate exactly once. The existing
PC native startup fallback probes are distinct from the standalone V0.

## Optional fixed HTP-only layout and Chunk widths

The same `run_32b_profile5_plan.ps1` now accepts `-ForceLayout`,
`-ForceXC`, `-ForceXG`, and `-ForceXP`. This runs the native APPLY_WIDE
forced A/B path instead of the automatic layout search, **without**
disabling the exact DAG checks or per-layer FastRPC Session / Android
MemAvailable guard. `N` and `P` in `-ForceLayout` both select Phone HTP;
the runtime receives the canonical `P` notation.

For an initial 32B smoke, use 12 Phone layers (six P2 Regions),
52 PC CUDA layers and a final GPU tail. The topology is not guaranteed
to fit every device; the Planner must still pass its live memory check.

```powershell
.\tools\region-ring-experiments\run_32b_profile5_plan.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\32b-fixed-n12" `
  -RealStageProfileCsv ".\32b-independent-profile-all40-final\independent-profile\profile.csv" `
  -ForceLayout "G8,N2,G8,N2,G8,N2,G8,N2,G8,N2,G8,N2,G4" `
  -ForceXC 64 -ForceXG 64 -ForceXP 64 `
  -ProfileEvery 0 -Batch 2048 -UBatch 2048 -NPredict 1 -Threads 8 `
  -Limit 1 -StopOnFail
```

Remove `-Limit 1` after smoke PASS. Use the same `-Out` to resume a
fixed run, or **a new Out** when changing layout/chunk widths. Fixed
parameters are signed into the run configuration. Omitting
`-ForceLayout` restores the original automatic Planner. Legacy
`--layout` is incompatible with HTP-only automatic mode; use the
new fixed-Plan parameters above.
## First-Cycle single-Bank hard limit and later-Cycle micro-adjustments

This Planner rule applies to automatic APPLY_WIDE search. It is now
**illegal** for any first GPU Region to have a weight footprint that
fits total CUDA VRAM but exceeds the capacity of a **single** streaming
Bank. The capacity used for a complete Plan is

`B_one = (0.90 * measured_GPU_free - GPU_KV(full_plan) - GPU_slot_scratch) / 2`

and the strict test is `max_weight_region(Bank_A) < B_one` AND
`max_weight_region(Bank_B) < B_one`. In particular,
`weight(G1) < B_one`. No equal-capacity boundary is accepted.
The first-Cycle E prefilter also rejects oversize G1 **before**
ranking, using an optimistic minimum future-GPU KV reservation;
the terminal Beam and full Plan recompute actual GPU-layer KV and
the two Bank maxima. If the streaming CUDA budget/weights are unknown,
the candidate is rejected instead of skipping the Bank safety check.
The older aggregate two-Bank peak and GPU KV checks remain enabled.

**Later Cycle changes are micro-adjustments**, not a new unconstrained
layer split. With first-cycle counts `(G1,C1,P1)`, each later Cycle
must satisfy `|Gi-G1| <= 2`, `|Ci-C1| <= 2`,
`|Pi-P1| <= 2`, and
`|(Gi+Ci+Pi)-(G1+C1+P1)| <= 3` layers. The same `XC/XG/XP`
and accumulated workload E/Beam scoring remain in use. The CPU-optional
G/P pass also obeys this rule; C=0 is legal. Local refinement,
uniform candidates, and the final Top24 are gated too. HTP-only
GPU-tail is a **separate final head/tail** and is not counted as
a fifth/full Cycle in this rule.

The independent G/C/N Profile V0 is unchanged. In
`planner-first-cycle-top256.csv`, the new
`gpu_single_bank_capacity_mib` column shows the early optimistic
one-Bank maximum. Inspect `[REGION_RING_PLAN_SEARCH]` for
`gpu_first_one_bank_pruned` and `prefix_micro_pruned`.
`[REGION_RING_GPU_BANK_GUARD]` reports the final selected Plan's
first GPU Region, Bank A/B weight peaks, and exact single-Bank
capacity. A manually forced layout keeps its exact requested stages
(no automatic micro-adjustment), but it **still must pass** the
full two-Bank memory hard constraint and DAG checks.

The Bank limit is a conservative VRAM accounting rule built from
weight bytes, GPU KV and slot scratch; it is **not** a direct query
of a hardware-exposed CUDA Bank size. Physical CUDA allocation or WDDM
pressure can still cause runtime failure, so validate real execution.

## Wider Plan shortlist and CPU-optional search

Automatic APPLY_WIDE still uses the one-shot independent Profile V0.
This revision expands *offline candidate search*, not live device
profiling. To improve coverage before deriving the final HTP-only GPU
tail, the first-cycle E reservoir grows from 500 to 640, selected
first-Cycle seeds from 200 to 256, middle/last Beam width from 16/24
to 24/32, and the complete full-DAG-scored parent shortlist from
Top10 to **Top24**. The CPU-optional DAG budget rises from 64 to 96,
with more diverse three-stage anchors and standalone G/P seeds.
Consequently, planning CPU time and memory may increase, especially
for long Prompts; **no** HTP Session, shared Phone RAM, GPU VRAM or
exact DAG safety constraint is relaxed.

CPU=0 is allowed, but the main six-dimensional first-Cycle Beam still
starts with nonempty G,C,P stages. After it completes, a second pass
can merge C into the neighboring G or P and score independent G/P
layouts. Unlike the prior behavior, independent G/P seeds are **still
evaluated when the full G/C/P search produces no memory-feasible
survivor**; CPU=0 no longer depends on winning a three-device plan.

For each first-Cycle seed `(G1,C1,P1,XC,XG,XP)`, subsequent Cycles
keep `XC/XG/XP` fixed and vary the `(G,C,P)` layer counts subject to
remaining layers, bounded Cycle-width spread, GPU two-Bank peak and
terminal Phone/HTP memory feasibility. Each expansion adds service
cost to accumulated `W_CPU/W_GPU/W_PHONE` and ranks by accumulated
minimax mismatch plus a small current-Cycle penalty, not by repeating
the first Cycle. Only full layouts pass the exact Prompt-level DAG.
The HTP-only final GPU-tail step now checks up to 24 different
Phone-final parents, shifting 1..8 last Phone layers to the GPU
and re-checking each resulting layout against all memory/DAG guards.

Review `[REGION_RING_PLAN_SEARCH]` for `first_cycle_selected`,
`prefix_expanded`, `prefix_completed` and `[REGION_RING_PLAN_CPU_OPTIONAL]`
for CPU-free candidates. `[REGION_RING_PLAN_HTP_TAIL] event=PARENT_POOL`
reports parent count. The expanded first-Cycle artifact is named
`planner-first-cycle-top256.csv`. The final selected runtime layout
still uses the shortest **predicted full-DAG wall time** among safe
candidates, not the lowest first-Cycle E.

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
5. **One-shot Profile (V0 only)**: the first Prompt constructs the G/C/N
   independent grid if no valid cached baseline exists. Every later Prompt
   reuses the same `<Out>/independent-profile/profile.csv`. No rolling
   Profile5, V1/V2 refresh, or runtime Stage override is created. The
   Planner still searches independently for each Prompt based on its
   length and memory constraints.

   **Scope:** rolling profiling and the independent G/C/N grid run only
   once. The existing C++ startup still performs separate native
   CPU/GPU/transport fallback probes whenever a new `llama-completion`
   process starts. Eliminating those is a separate C++ caching change;
   this patch does not change the runtime executor or memory budget.

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
  -ProfileEvery 0 `
  -Batch 2048 -UBatch 2048 -NPredict 1 -Threads 8 `
  -Limit 2 -StopOnFail
```

After the two PASS runs, omit `-Limit 2` to resume all 40. The old file
name `run_32b_profile5_plan.ps1` is kept for compatibility, but it
**always passes `--profile-every 0`**. A legacy `-ProfileEvery 5`
argument prints a warning and is ignored.

**Resume the existing five-PASS Profile5 directory:** use the same
`-Out` and other settings. The runner permits a one-way migration only
when the Profile5 policy and its two telemetry flags are the sole
differences; it backs up the old config to
`run-config.profile5-before-once.json`. All PASS results and immutable
V1 artifacts are preserved, but V1 is never selected, and V2 is not
created. Other configuration mismatches still fail safely.

To force independent re-profiling **before any PASS prompt**, add
`-UpdateProfile`; after any PASS, use a fresh `-Out`.

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
[INDEPENDENT_PROFILE] event=REUSE ... status=READY  # subsequent prompts
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

## Historical issue: V1 missing NPU samples (rolling mode disabled)

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

The V1/V2 recovery steps in the historical section above no longer
apply to the one-shot workflow. Look for
`[PROFILE] policy=INDEPENDENT_ONCE`,
`[INDEPENDENT_PROFILE] event=REUSE`, and
`[PLAN] ... profile=V0`. Old `profile-epochs/` data is audit-only.

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
