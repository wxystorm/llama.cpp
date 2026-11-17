# Jupiter Fig.3 - complete one-time evidence collection

Source baseline: baseline/jupiter-infocom25
Instrumentation branch: exp/jupiter-fig3-full-trace-20261118

Jupiter stays in its original PC-prefix/Phone-suffix Layer Pipeline.
Do not replace it with Galaxy TP or experimental HTP. Model partition and
chunk/window scheduling code are unchanged; all telemetry is opt-in.

## Six categories

Phone Compute: Phone native graph execution. GPU Compute: PC CUDA stream
device intervals. CPU Compute: PC CPU backend graph executions. PCIe Copy:
PC CUDA H2D/D2H device events. Wi-Fi Sync: RPC send, roundtrip and remote
completion service path (NOT pure Wi-Fi RF airtime). Waiting/Other:
unattributed CPU host work, graph preparation, queueing and idle residual,
NOT measured physical hardware idle.

The profiler retains overlapping raw events and independently emits an
EXCLUSIVE 100% timeline using the same priority as Galaxy:
Phone Compute > PCIe Copy > GPU Compute > CPU Compute >
Wi-Fi Sync > Waiting/Other. This is a projection, not a causal critical path.

Phone Graph Prepare and PC Host Prepare are saved separately as diagnostic
raw events. Phone steady-clock timestamps are translated using eight
four-timepoint RPC clock samples per lane, with uncertainty recorded.

## Phone Termux: update once

Phone and PC both need the NEW Jupiter branch, since the profiler adds
RPC opcodes 21/22 (run tag + phone clock sync). Rebuild Phone ONCE.

Stop old RPC server. From the Phone's Jupiter clone:

    cd ~/llama-jupiter
    git fetch origin exp/jupiter-fig3-full-trace-20261118
    git switch -c exp/jupiter-fig3-full-trace-20261118 FETCH_HEAD
    cmake --build build-jupiter --target ggml-rpc-server -j 4
    export LLAMA_JUPITER_PREFILL=1
    export LLAMA_JUPITER_TIMELINE=1
    export LLAMA_JUPITER_FIG3_PHONE_FILE="$HOME/jupiter-fig3-phone.csv"
    export LD_LIBRARY_PATH="$PWD/build-jupiter/bin:$LD_LIBRARY_PATH"
    ./build-jupiter/bin/ggml-rpc-server --device GPUOpenCL -H 0.0.0.0 -p 10139 -t 8 -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf"

Use your existing known-good Phone build directory, OpenCL libraries and
server command if paths differ. Keep GPUOpenCL for original Jupiter baseline;
do not silently switch to an HTP-only experimental mode.

The Phone appends ONE file at ~/jupiter-fig3-phone.csv with run_id,
device, graph UID and real native preparation and compute timestamps.
Run ID is assigned automatically by the PC.

Optional SSH for automatic transfer (another Termux session):
    sshd

## Windows PC: a single command

    cd E:\llama\llama-jupiter
    git fetch origin exp/jupiter-fig3-full-trace-20261118
    git switch -c exp/jupiter-fig3-full-trace-20261118 FETCH_HEAD
    .\tools\jupiter\run_fig3_complete.ps1

Defaults: Qwen3-14B-Q4_K_M, prompt_1024_01.txt, PC24/Phone16 layers,
context2048, batch2048, ubatch128, window2, n_predict1,
RPC 10.238.229.239:10139. Use -PcLayers N and -PlanFile PATH or
-SequencePlan CHUNKS if your previously verified baseline used those
values. Do not tune the baseline for the motivation figure.

The script runs a synthetic schema/clock analyzer self-test; builds PC
llama-cli without UI; probes the actual Phone RPC run-tag, writable CSV and
clock support BEFORE inference; then performs one instrumented profile,
one normal warmup and three normal Prefill trials. The normal median is
used for the bar top label, not the intrusive profile time.

The experimental profile sets LLAMA_JUPITER_PROFILE_STAGE=0, preserving
the asynchronous original Jupiter pipeline. All graph/tensor transfers,
GPU/CPU device events, full Prefill boundary and chunk/slot scheduling
are saved separately.

Already built? Run:
    .\tools\jupiter\run_fig3_complete.ps1 -SkipBuild

No-inference check:
    python .\tools\jupiter\test_fig3_complete.py
    .\tools\jupiter\run_fig3_complete.ps1 -DryRun -SkipBuild

## Failed SCP NEVER requires a model rerun

If Termux SSH port 8022 is not running, PC files are safe and the script
prints WAITING_FOR_PHONE_CSV. Copy one phone file manually, then run:

    .\tools\jupiter\run_fig3_complete.ps1 -AnalyzeOnly -PhoneEvents ".\jupiter-fig3-phone.csv"

This performs NO build or inference and extracts only the matching Run ID.

## Files

Output root: fig3-jupiter14b-complete
- manifest.json: run ID, exact split/configuration, runs and status
- profile_run.log: original Jupiter communication, graph and chunk logs
- warmup_1.log and normal_1.log..normal_3.log: clean Prefill timings
- phone_events.csv: archived raw Phone evidence
- fig3_jupiter_paper_row.csv: final comparison/bar data
- profile/pc_prefill_windows.csv: exact synced start/end of Prefill
- profile/pc_chunks.csv: native Jupiter chunk/slot/prefix
- profile/pc_cuda_events.csv: CUDA graph and PCIe H2D/D2H intervals
- profile/pc_cpu_events.csv: native PC CPU backend graph intervals
- profile/pc_rpc_events.csv: socket operations and RPC service timing
- profile/pc_clock_sync.csv: PC/Phone offset/uncertainty calibration
- profile/fig3_jupiter_breakdown.csv: strict 100% six-category projection
- profile/fig3_jupiter_source_events.csv: raw concurrent event intervals
- profile/fig3_jupiter_timeline.csv: full unbroken exclusive time axis
- profile/fig3_jupiter_chunks.csv: original chunk sizes and slots
- profile/fig3_jupiter_preview.svg: six lanes and projection
- profile/fig3_jupiter_audit.json: clock, overlap and semantics audit

Static checks are not a substitute for compiling CUDA on the PC and
GPUOpenCL on the Phone. First build/phone-protocol preflight must pass.
