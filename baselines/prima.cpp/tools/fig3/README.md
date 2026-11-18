# Prima.cpp Fig. 3: evidence-oriented two-rank profiling (experimental)

Branch: exp/prima-fig3-full-trace-20261118, based on
baseline/prima-qwen3-dense-20261107. The Piped-ring execution, ZMQ transport,
HiGHS scheduling, GPU assignment, memory parameters and weight placement are
unchanged. This is measurement instrumentation only. Do NOT copy Jupiter RPC
run settings into Prima.

## Differences from Galaxy and Jupiter

Prima has one WSL Rank 0 (PC CUDA + CPU) and one Termux Rank 1 (usually CPU).
Rank 1 forwards data over ZMQ. The original output head remains on Rank 0.
Its automatic placement may choose ZERO layers on a device. If Rank 1 has
zero actual compute layers, do not fabricate a Phone Compute fraction.

Observed events:
- PC GPU Compute: CUDA event durations, recorded outside Graph capture, with
  sync-completion host upper-bound positioning (not absolute timestamps).
- PC PCIe Copy: H2D/D2H CUDA events for backend buffer set/get and async
  tensor transfers. CUDA tensor/weight copies via other paths can be absent.
- CPU Compute: native PC backend Graph calls.
- Phone Compute: native Termux backend CPU Graph calls (no NPU claim).
- RPC / Coordination: ZMQ meta/tensor send/recv API walls, including blocking
  waits and remote execution dependencies; NOT pure Wi-Fi wire time.
- Waiting / Other: unmarked residual after priority projection.
- Host Prepare, Host Transfer and Prefetch: retained as separate diagnostics,
  and may overlap compute/communications.

Raw overlapping intervals are saved; mutual-exclusive percentages use
Phone Compute > PCIe Copy > GPU Compute > CPU Compute >
RPC / Coordination > Waiting / Other. Do not interpret this projection
as a causal critical path or device utilization.

**Important device-time caveat:** the legacy CUDA Graph backend does not
expose a shared absolute GPU clock to this tracer. Native device-event
DURATIONS are precise at CUDA event resolution, but their alignment to
the CPU monotonic timeline uses the existing stream sync return as an
upper-bound anchor. The audit marks paper_fidelity=PROVISIONAL_GPU_EPOCH.
A paper-quality cross-device temporal claim needs a CUPTI/Nsight trace with
absolute CUDA timestamps or a verified hardware anchor. Do not remove this
caveat or silently label the projection fully hardware-verified.

## 0. Build both ranks after switching branches

PC WSL:

    cd /mnt/d/prima-qwen3
    git fetch origin refs/heads/exp/prima-fig3-full-trace-20261118
    git switch -c exp/prima-fig3-full-trace-20261118 FETCH_HEAD
    cd baselines/prima.cpp
    make USE_HIGHS=1 GGML_CUDA=1 CUDA_DOCKER_ARCH=compute_86 CC=gcc-10 CXX=g++-10 NVCCFLAGS="-ccbin=/usr/bin/g++-10" -j4 llama-cli

Phone Termux:

    cd ~/prima-qwen3
    git fetch origin refs/heads/exp/prima-fig3-full-trace-20261118
    git switch -c exp/prima-fig3-full-trace-20261118 FETCH_HEAD
    cd baselines/prima.cpp
    make CC=clang CXX=clang++ -j4 llama-cli

Check that the PC uses the previously working CUDA toolchain and that Phone
uses its original CPU-only build. Memory limits are not modified here.
For 24B/32B replace the model path on both ranks, keeping their GGUF identical.

## 1. First run: Qwen3-14B and the exact one-prompt sample

Choose one unique run ID, e.g. prima-fig3-14b-20261118.
The PC collector and Phone supervisor MUST receive the same ID.

Phone Termux (first, leave running):

    cd ~/prima-qwen3/baselines/prima.cpp
    bash tools/fig3/phone_worker_fig3.sh \
      "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
      10.238.229.214 \
      prima-fig3-14b-20261118

The FIRST phone process is profiled. Following phone processes are completely
unprofiled. The phone supervisor also launches TCP clock port 19139. Its
persistent output is:

    ~/prima-fig3-phone/prima-fig3-14b-20261118/phone.csv
    ~/prima-fig3-phone/prima-fig3-14b-20261118/clock.log
    ~/prima-fig3-phone/prima-fig3-14b-20261118/logs/rank1-01.log

On PC WSL, preflight syntax/simulation without loading model:

    cd /mnt/d/prima-qwen3/baselines/prima.cpp
    python3 tools/fig3/test_fig3.py
    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-20261118 --dry-run

Then start the SINGLE full collection (one Profile, one normal warmup,
three normal Prefill runs):

    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-20261118 \
      --binary ./llama-cli \
      --model /mnt/e/llama/models/Qwen3-14B-Q4_K_M.gguf \
      --prompt /mnt/e/llama/datasets/sharegpt/sharegpt50_short/prompt_1024_01.txt \
      --out /mnt/d/fig3-prima14b-complete \
      --master 10.238.229.214 --next 10.238.229.239 \
      --context 2048 --batch 512 --ubatch 512 --threads 8 \
      --repeats 3 --warmups 1 --between 30 \
      --phone-user u0_a283 --ssh-port 8022

WSL clock port 19139 must reach Phone 10.238.229.239. The Phone may need
port access permission (Termux allows nonprivileged port 19139).

Do not use old automatic ShareGPT sweep logs to infer six-way proportions.
They are separate, normal throughput measurements only.

## FastTrace v1 complete: strict clock failure + diagnostic-only recovery

The completed `prima-fig3-14b-fasttrace-v1` archive contains
PC 1,573 raw events, Phone 72 raw events, Profile 11933.83ms, and
3 normal Prefill runs 8228.27/7694.00/7785.37ms (median 7785.37ms)
on the SAME fixed PC39/Phone1 (PC GPU21) layout.
The Profile/Normal ratio has improved from 2.87x to 1.53x after staging
the trace in native Linux /tmp. PC GRAPH_SUBMIT inside Prefill fell from
~15.99s in the original run to ~3.50s. The 8.82s PC Graph call
before the formal Prefill window is outside the plotted envelope.

The archived manifest status is `ANALYSIS_FAILED`, so the `profile/`
folder is empty. This is due to the STRICT clock discontinuity guard:
90 in-run samples, 88 with <=20ms uncertainty, plus before/after
calibrations. The best four before/after offset changed by 122.465ms
in ~49.675s, but DURING the Prefill a pair of neighboring anchors,
1.586s apart, differs by **111.026ms** in relative offset. The code's
100ms local guard fails and correctly refuses paper proportions.

The analyser still defaults to `--clock-mode strict`. DO NOT disable or
raise this guard for the paper. Instead, an **explicit diagnostic-only**
mode can save a dense-anchor extrapolation for debugging the event
accounting, while recording the rejected clock segment:

    cd /mnt/d/prima-qwen3/baselines/prima.cpp
    python3 tools/fig3/test_fig3.py
    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-fasttrace-v1 \
      --out /mnt/d/fig3-prima14b-fasttrace-v1 \
      --clock-mode piecewise-diagnostic --analyze-only

The resulting `paper_fidelity` label is
`DIAGNOSTIC_ONLY_DENSE_CLOCK_ANOMALY_AND_GPU_EPOCH`.
This is **NOT PAPER DATA**: inside the Prefill the sampled offset is
strongly nonstationary and CUDA absolute event time placement remains
unverified. Avoid the two-endpoint affine correction here, since it
would ignore the >350ms mid-run offset excursion. No rerun is required
for diagnostic extraction; correct device-clock and GPU epoch placement
before submitting final six-class Figure 3 data.

## Resume after Phone Termux worker was accidentally closed

A completed PC --profile-only run already saved its PC trace and 16+90+16
clock observations. If Phone was then stopped, DO NOT run the original
Phone helper in normal/start mode: it refuses a nonempty phone.csv.
Do NOT delete phone.csv and do NOT use a different run-id.

After updating the branch, recover Phone with the same run-id:

    cd ~/prima-qwen3
    git fetch origin refs/heads/exp/prima-fig3-full-trace-20261118
    git merge --ff-only FETCH_HEAD
    cd baselines/prima.cpp
    bash -n tools/fig3/phone_worker_fig3.sh
    PRIMA_FIG3_PHONE_RESUME=1 bash tools/fig3/phone_worker_fig3.sh \
      "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
      10.238.229.214 prima-fig3-14b-fasttrace-v1

The helper checks that phone.csv is nonempty and belongs to this exact
run; it skips tracing, skips re-running the clock server, and continues
at the next unused rank1-NN.log session. The phone CSV remains byte-for-
byte unchanged. Keep the helper alive while PC finishes Normal cases.

PC WSL (no change to model binaries needed):

    cd /mnt/d/prima-qwen3/baselines/prima.cpp
    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-fasttrace-v1 \
      --out /mnt/d/fig3-prima14b-fasttrace-v1 \
      --model /mnt/e/llama/models/Qwen3-14B-Q4_K_M.gguf \
      --prompt /mnt/e/llama/datasets/sharegpt/sharegpt50_short/prompt_1024_01.txt \
      --master 10.238.229.214 --next 10.238.229.239 \
      --context 2048 --batch 512 --ubatch 512 --threads 8 \
      --fixed-layers 39,1 --pc-ngl 21 --model-layers 40 \
      --warmups 1 --repeats 3 --between 30 --resume

The resume command does not re-run the Profile or re-collect clocks
because those are already saved. If the original Rank1 process is
still running independently (terminal disappeared but process survived),
do not launch a duplicate; first inspect `ps -ef | grep llama-cli`.

## PC trace write amplification and quick A/B (2026-11-18)

Full original PC logs confirm the same manual PC39/Phone1, PC GPU21,
ubatch512, ctx2048 on Profile, Warmup and Normal1-3. Exact PC Prefill
(ms): 22299.08; 8295.31; 7773.30; 7894.46; 7701.38.
The normal median is 7773.30ms (132.89 tokens/s), versus the Profile
22.30s. PC model-load times vary but are NOT included in prompt eval.

The raw source_events.csv contains 9 PC GRAPH_SUBMIT calls totaling
15985.74ms. Two especially long calls: 5550.14ms and 7878.45ms.
EACH contained 238 H2D_BUFFER events, 216 CUDA_GRAPH events,
54 D2H_BUFFER events, and 36 CPU_GRAPH events. The implementation
in prima-fig3-trace.h historically performs open+write+close for every
event. The original PC event file lived on WSL DrvFs /mnt/d; slow
per-event metadata operations could have artificially inflated these
host-side timings. This is a hypothesis, not a measured cause.

NEW: run_fig3_complete.py stages the PC raw events into a native Linux
/tmp temporary directory and copies the CSV **once** into the output
directory when the first Profile process exits. This changes only the
trace destination, not the inference or placement. Source timestamps
and source event schema are unchanged. It also supports --profile-only
to stop after Profile, without the normal repetitions. The original
file and run-id are never overwritten.

Check file I/O overhead without executing model or phone:

    cd /mnt/d/prima-qwen3/baselines/prima.cpp
    python3 tools/fig3/trace_io_probe.py \
      --linux-dir /tmp \
      --windows-dir /mnt/d/fig3-prima14b-fixed39-1 --iterations 400

Start a NEW phone worker, after stopping the previous one:

    bash tools/fig3/phone_worker_fig3.sh \
      "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
      10.238.229.214 prima-fig3-14b-fasttrace-v1

On PC WSL, a ONE-PROFILE A/B for the 39/1 layout:

    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-fasttrace-v1 \
      --out /mnt/d/fig3-prima14b-fasttrace-v1 \
      --model /mnt/e/llama/models/Qwen3-14B-Q4_K_M.gguf \
      --prompt /mnt/e/llama/datasets/sharegpt/sharegpt50_short/prompt_1024_01.txt \
      --master 10.238.229.214 --next 10.238.229.239 \
      --context 2048 --batch 512 --ubatch 512 --threads 8 \
      --fixed-layers 39,1 --pc-ngl 21 --model-layers 40 \
      --profile-only

The output ends with PROFILE_ONLY complete, Prefill=...ms. Compare
against the old 22299.08ms under equivalent conditions. If no longer
grossly inflated, the DrvFs trace write amplification is implicated.
If it stays slow, isolate CUDA per-op event timing and WSL clock behavior
next. The test cannot by itself validate absolute GPU epoch alignment.
Do not mislabel the 39/1 replay as Prima automatic placement.

Clock samples in the old run had 122 DURING observations: offset
increased broadly throughout the Profile, by around 1.30s over
64.7s of PC clock elapsed (~2% relative rate difference), not just
a single observed endpoint anomaly. This is too large to accept as
normal clock skew; verify WSL/Android clocks independently.

## Verified 14B fixed-layout diagnostics (profile(3).zip)

A real fixed-layout Qwen3-14B run yielded post-init PC39 (GPU21), Phone1,
1033 tokens, normal Prefill median 7773.30 ms, Profile Prefill
22302.92 ms, 154 clock samples (123 selected anchors) and an
**unresolved 1298.36 ms relative clock offset change**. Its
paper_fidelity=PROVISIONAL_CLOCK_RATE_AND_GPU_EPOCH: not final
publication-quality absolute overlap evidence.

The first generated six-class percentage had RPC / Coordination=70.25%
because its inclusion of Phone ZMQ_RECV_META classified 18.01s of
blocking *metadata polling* as communications. In the original trace
15.55s of those poll intervals overlapped PC GRAPH_SUBMIT. This is
not 15.55s of network traffic and not proof of CPU idle on the PC.

The offline analyzer now preserves Phone ZMQ_RECV_META in source events
as `Phone Metadata Poll Wait` but excludes it from the RPC /
Coordination numerator. Under the same priority-exclusive projection,
the saved trace's diagnostic proportions become approximately:
Phone Compute 13.17%, PCIe Copy 4.67%, GPU Compute 7.41%,
CPU Compute 4.42%, RPC / Coordination 10.79%, Waiting / Other 59.54%.
The last category is residual/unattributed host work plus device waits;
it MUST NOT be labeled "PC idle" or "Wi-Fi delay".

To reprocess the original fixed39/1 run with no model execution:

    cd /mnt/d/prima-qwen3/baselines/prima.cpp
    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-fixed39-1 \
      --out /mnt/d/fig3-prima14b-fixed39-1 \
      --analyze-only

This rebuilds only the derived `profile/` outputs. Keep a copy of
the earlier profile archive for reproducibility; neither PC nor
Phone raw CSV requires modification. The new audit includes
`polling_and_host_submit_diagnostics`.

The Profile takes 2.87x the uninstrumented median even with the
same post-init placement. The PC GRAPH_SUBMIT API wall time alone is
15.99s. The cause is NOT proven: investigate weight/page-cache
effects, backend launch/memory handling, profiling overhead, and CPU
native events. Avoid claiming the ratio is a calibrated profiler
overhead before an A/B comparison on warm/cold matched placement.

## Controlled replay: Qwen3-14B consistent layer placement

The previous 14B Figure 3 is **not** a valid Profile/Normal comparison:
the Profile assigned PC6 GPU layers and Phone34 CPU layers (109006.70ms /
1033 tokens); three Normal runs first planned PC39/Phone1 (PC GPU21) and
subsequently logged PC40-only, with a normal median of 5144.86ms.
The 21x discrepancy is a placement mismatch, **not proof of a profiling
slowdown**. That run also had 1514.126ms PC/Phone clock-offset change.
Keep its CSVs as diagnostic evidence only.

Prima supports a documented manual layout option on Rank 0:
-lw "39,1" -ngl 21. This bypasses HiGHS auto placement. The run
is therefore called **FIXED_LAYOUT_REPLAY**; never label its performance
as Prima's automatic placement baseline. The collector now records the
post-init effective layout of *every* run, including Normal, and fails
closed if the profile/normal layouts disagree or the phone was removed.

Use a fresh run-id and output directory. First STOP the old phone worker.

Phone Termux:

    cd ~/prima-qwen3
    git fetch origin refs/heads/exp/prima-fig3-full-trace-20261118
    git switch exp/prima-fig3-full-trace-20261118
    git merge --ff-only FETCH_HEAD
    cd baselines/prima.cpp
    make CC=clang CXX=clang++ -j4 llama-cli
    bash tools/fig3/phone_worker_fig3.sh \
      "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
      10.238.229.214 prima-fig3-14b-fixed39-1

PC WSL (different terminal, after Phone Worker starts):

    cd /mnt/d/prima-qwen3
    git fetch origin refs/heads/exp/prima-fig3-full-trace-20261118
    git switch exp/prima-fig3-full-trace-20261118
    git merge --ff-only FETCH_HEAD
    cd baselines/prima.cpp
    make USE_HIGHS=1 GGML_CUDA=1 \
      CUDA_DOCKER_ARCH=compute_86 CC=gcc-10 CXX=g++-10 \
      NVCCFLAGS="-ccbin=/usr/bin/g++-10" -j4 llama-cli
    python3 -m py_compile tools/fig3/*.py
    python3 tools/fig3/test_fig3.py
    python3 tools/fig3/test_layout_guard.py

    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-fixed39-1 \
      --binary ./llama-cli \
      --model /mnt/e/llama/models/Qwen3-14B-Q4_K_M.gguf \
      --prompt /mnt/e/llama/datasets/sharegpt/sharegpt50_short/prompt_1024_01.txt \
      --out /mnt/d/fig3-prima14b-fixed39-1 \
      --master 10.238.229.214 --next 10.238.229.239 \
      --context 2048 --batch 512 --ubatch 512 --threads 8 \
      --fixed-layers 39,1 --pc-ngl 21 --model-layers 40 \
      --clock-interval 0.5 --warmups 1 --repeats 3 --between 30

Rank 1 needs no -lw/-ngl option; Rank 0 supplies the split through the
existing ring. Both nodes use the same GGUF and fresh run-id.
The expected PC log line starts with:

    [PRIMA_FIG3_LAYOUT] stage=post_init rank=0 world=2 lw0=39 lw1=1 ngl=21

The runner verifies this after every session, including all 3 Normal
measurements. Using a pinned replay is more reproducible, but does NOT
substitute for rerunning a stable real auto-plan baseline. The original
normal median 5144.86ms must not be reused; the fixed replay remeasures it.

If phone.csv is not copied automatically, fetch:
    ~/prima-fig3-phone/prima-fig3-14b-fixed39-1/phone.csv
to D:\fig3-prima14b-fixed39-1\phone_events.csv, then use --analyze-only.

Even a matched replay remains provisional for cross-device overlap until
the CUDA device-epoch alignment and clock discontinuity checks pass.

## Clock discontinuity recovery and follow-up

A large discrepancy between *before* and *after* four-timestamp clock
calibrations is a scientific validity issue, not Prima's network transfer
latency. Example: PC time elapsed 164.733 s, Phone time elapsed 166.247 s;
offset changed by 1514.126 ms with individual clock-probe uncertainty
approximately 1.2 ms. Do **not** raise a 20 ms tolerance to 2000 ms.

Existing files can still be inspected WITHOUT inference or file overwrite:

    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-20261118 \
      --out /mnt/d/fig3-prima14b-complete \
      --clock-mode affine-diagnostic --analyze-only

Only use this mode to investigate graph composition and event accounting.
It **assumes** a linear clock-rate mismatch between the two endpoints.
A clock jump in the middle cannot be ruled out from two snapshots.
All generated outputs are explicitly labeled
paper_fidelity=DIAGNOSTIC_ONLY_UNVERIFIED_TWO_POINT_CLOCK, **NOT paper data**.
The original PC/Phone CSV and manifest clock samples are retained.

For newly collected experiments, the PC script records four-timestamp
clock observations **during** Profile, every 0.5s by default, in addition
to before/after samples. Default strict mode can piecewise-align densely
sampled observations only if it sees no large step and every adjacent
calibration is within 8s; gaps and jumps fail closed.
Even the piecewise path remains PROVISIONAL because CUDA device-event
absolute placement is still estimated. Check WSL timing behavior before
using absolute cross-device attribution in the paper.

## 2. Recovery without rerunning completed inference

If the phone CSV SCP failed, copy it manually to PC and run:

    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-20261118 \
      --out /mnt/d/fig3-prima14b-complete \
      --phone-file /mnt/d/phone.csv --analyze-only

If the collector stopped after a completed Profile or normal repetition:

    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-14b-20261118 \
      --out /mnt/d/fig3-prima14b-complete --resume

Use the SAME binary/model/prompt/flags as the original run on resume.
Phone helper must stay running for unfinished cases. Never overwrite the
original Phone CSV or reuse a Run ID across distinct experiments.

## 3. Results

    fig3-prima14b-complete/
      manifest.json
      pc_events.csv
      phone_events.csv
      profile_run.log
      warmup_1.log
      normal_1.log
      normal_2.log
      normal_3.log
      profile/
        fig3_prima_breakdown.csv
        fig3_prima_paper_row.csv
        fig3_prima_source_events.csv
        fig3_prima_timeline.csv
        fig3_prima_audit.json

The audit checks PC Prefill boundary, matching token counts, CPU/CUDA/phone
raw source presence, before/after four-timestamp clock offset/drift,
100% interval projection and original overlapping event coverage.
A valid CSV output is NOT itself evidence of exact CUDA absolute placement.
Do not turn approximate device/host alignment into verified percentages.

## Prima 32B Fig.3: first discover, then replay only if needed

Qwen3-32B-Q4_K_M has 64 transformer layers. Unlike the 14B 40-layer
replay, do **not** reuse `-lw 39,1 -ngl 21` for 32B: its PC/Phone
memory budgets and best ring windows differ. In Prima a manual `-lw`
describes the *per-cycle rank windows*, which can repeat across
multiple cycles. Hence for a 64-layer model, `-lw 12,10` is a valid
22-layer ring window; it is NOT a claim of only 22 model layers.
The updated fixed-layout validator accepts such windows and estimates
actual local layer counts for `--pc-ngl` bounds (accounting for a
partial last cycle).

Preflight on both devices: ensure the 32B GGUF exists, inspect
available memory with `free -h`; check PC CUDA memory with
`nvidia-smi`. Do not blindly use a 14B layout on 32B.

First attempt should preserve Prima's *automatic* HiGHS placement,
so launch the Phone Fig.3 worker with a FRESH run-id:

    cd ~/prima-qwen3/baselines/prima.cpp
    bash tools/fig3/phone_worker_fig3.sh \
      "$HOME/models/Qwen3-32B-Q4_K_M.gguf" \
      10.238.229.214 prima-fig3-32b-auto-v1

On PC WSL, profile **one 1024-group prompt** only. The collector
already stages all PC raw trace writes to native /tmp:

    cd /mnt/d/prima-qwen3/baselines/prima.cpp
    python3 tools/fig3/run_fig3_complete.py \
      --run-id prima-fig3-32b-auto-v1 \
      --binary ./llama-cli \
      --model /mnt/e/llama/models/Qwen3-32B-Q4_K_M.gguf \
      --prompt /mnt/e/llama/datasets/sharegpt/sharegpt50_short/prompt_1024_01.txt \
      --out /mnt/d/fig3-prima32b-auto-v1 \
      --master 10.238.229.214 --next 10.238.229.239 \
      --context 2048 --batch 512 --ubatch 512 --threads 8 \
      --model-layers 64 --profile-only

Notice there is NO `--fixed-layers` nor `--pc-ngl` in this
initial attempt. After it succeeds, check the post-init layout:

    grep -E 'k =|W =|Device Index|Assignment Set|N Layer Window|N GPU Layers|PRIMA_FIG3_LAYOUT' \
      /mnt/d/fig3-prima32b-auto-v1/profile_run.log

If the plan is feasible and you want to retain the original auto-plan
mode, rerun the same PC command with `--resume` INSTEAD OF
`--profile-only`; this fills 1 warmup + 3 Normal sessions. The
collector compares effective placement and rejects re-plans rather
than silently combining inconsistent data.

If HiGHS changes its layout between independent runs, stop the phone
worker, start a *new* run-id, and repeat a full collection with the
measured `--fixed-layers PC_WINDOW,PHONE_WINDOW` plus
`--pc-ngl OBSERVED_PC_GPU_LAYERS --model-layers 64`. Those
numbers MUST come from the successful 32B probe, not be invented.
This replay is explicitly named `FIXED_LAYOUT_REPLAY`, not automatic
placement performance. The 14B diagnostic was also a fixed replay.

32B may consume more RAM/VRAM or choose to remove Phone compute.
If the phone is selected out (world=1, Phone=0), do not invent a
Phone Compute fraction. Keep the raw trace and report that the
automatic plan used only the PC. Check for OOM separately.

The strict PC/Phone clock guard remains enabled by default. If it
rejects the reconstruction, the Profile and Normal raw traces stay
saved. For an explicitly non-paper diagnostic plot only, use
`--clock-mode piecewise-diagnostic --analyze-only`. CUDA event
absolute placement and clock anomaly audits still govern whether
an exclusive six-way breakdown can support the final paper figure.

## 4. 24B/32B and fairness

Reuse the tooling after the 14B instrumented run passes. Use unique run IDs,
separate directories and the same model file on both ranks. Confirm planner
k/W and actual Rank-0/Rank-1 layers, GPU layers, total RAM and prefetch.
Keep Prima's previously accepted baseline configuration; do not pin layers
just to obtain attractive Figure 3 proportions.

One caution: Mistral tokenizer may produce a different number of tokens for
the SAME text. Always display actual token count; do not label it as the
1041-token Qwen input without checking.
