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
    git fetch origin
    git switch -c exp/prima-fig3-full-trace-20261118 origin/exp/prima-fig3-full-trace-20261118
    cd baselines/prima.cpp
    make USE_HIGHS=1 GGML_CUDA=1 CUDA_DOCKER_ARCH=compute_86 CC=gcc-10 CXX=g++-10 NVCCFLAGS="-ccbin=/usr/bin/g++-10" -j4 llama-cli

Phone Termux:

    cd ~/prima-qwen3
    git fetch origin
    git switch -c exp/prima-fig3-full-trace-20261118 origin/exp/prima-fig3-full-trace-20261118
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
clock observations **during** Profile, every 2s by default, in addition
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

## 4. 24B/32B and fairness

Reuse the tooling after the 14B instrumented run passes. Use unique run IDs,
separate directories and the same model file on both ranks. Confirm planner
k/W and actual Rank-0/Rank-1 layers, GPU layers, total RAM and prefetch.
Keep Prima's previously accepted baseline configuration; do not pin layers
just to obtain attractive Figure 3 proportions.

One caution: Mistral tokenizer may produce a different number of tokens for
the SAME text. Always display actual token count; do not label it as the
1041-token Qwen input without checking.
