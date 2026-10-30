# Jupiter planner (INFOCOM 2025 adaptation)

This directory implements Jupiter's offline prefill planning for the llama.cpp
PC + phone pipeline. It intentionally stays separate from the custom Region/Tensor
planners.

## Runtime plan file

Set:

```text
LLAMA_JUPITER_PLAN_FILE=/path/to/jupiter.plan
```

The plan format is deliberately simple:

```text
model.pc_layers=25
sequence.358=60,60,60,60,59,59
```

`LLAMA_JUPITER_PC_LAYERS` still overrides `model.pc_layers`.
`LLAMA_JUPITER_SEQUENCE_PLAN` can override one exact batch when its chunk sum
matches the batch length.

## q(x,y) calibration

Jupiter profiles sub-sequence latency as q(x,y), where x is the current
sub-sequence length and y is the total prefix length.

Calibration is intentionally synchronous and must not be used for performance
runs:

```powershell
$env:LLAMA_JUPITER_PREFILL="1"
$env:LLAMA_JUPITER_PROFILE_STAGE="1"
$env:LLAMA_JUPITER_TRACE="1"
$env:LLAMA_JUPITER_Q_SWEEP="1"   # include only required sampling markers at -lv 3
$env:LLAMA_JUPITER_SEQUENCE_PLAN="64,64,64,64,64,38"
```

The scheduler then prints one line per stage:

```text
[JUPITER_STAGE_PROFILE] ... backend=CUDA0 ... input_ms=... compute_ms=... wall_ms=...
[JUPITER_STAGE_PROFILE] ... backend=RPC0[...] ... input_ms=... compute_ms=... wall_ms=...
```

With the updated PC binary and `LLAMA_JUPITER_TRACE=1`, runtime also reports:

```text
[JUPITER_SEQUENCE_DECISION] batch_tokens=N requested_sum=M requested_k=...
  requested_max=... max_ubatch=... n_stream=... applied=0|1
  reason=APPLIED|TOKEN_SUM_MISMATCH|INVALID_CHUNKS|CHUNK_EXCEEDS_UBATCH|MULTIPLE_KV_STREAMS|NO_EXPLICIT_PLAN
```

This is the decisive check when a server task token count differs from the
actual prefill decode batch size. The diagnostic is compiled into the PC
runtime only; the phone RPC server does not need a rebuild.

Extract q samples from the PC log:

```bash
python tools/jupiter/jupiter_planner.py extract-q \
  --log jupiter-calibration.log \
  --output q-profile.csv \
  --metric compute_ms
```

Run several calibration plans so the q(x,y) grid covers the x and y ranges of
interest. The sequence DP interpolates between measured points, matching the
paper's offline profiling + interpolation design.

## Automatic q(x,y) sweep

Use the Python runner for automated calibration. It avoids PowerShell native
stderr/interactive-process behavior and works on Windows as well as Unix-like
hosts.

The runner first executes **one unpartitioned token probe** with the same prompt.
It prefers the longest complete `JUPITER_UBATCH event=BEGIN` chain,
because the offline sequence plan must match the actual prefill decode batch,
not necessarily the server-level `task.n_tokens`. Task token count and prompt
timing remain fallback signals and cross-checks.

If the calibration reports a different batch length from the probe, the runner
will regenerate all partitions with the observed length and retry once.
An optional `--target-tokens N` is only a hint: if the model reports a
different length, the runner prints `[JUPITER_SWEEP_TARGET_OVERRIDE]` and
uses the runtime length.

Each calibration point then starts a fresh `llama-cli` with stdin closed and
`--single-turn --simple-io`, so one completed prompt exits before the next
sweep point starts.

The Python q-sweep runner now uses `-lv 3` (not `-lv 4`) and
`LLAMA_JUPITER_Q_SWEEP=1`. The PC `common/log.cpp` logging callback
makes only the four calibration-critical records visible at verbosity 3:
`JUPITER_SEQUENCE_PLAN`, `JUPITER_SEQUENCE_DECISION`,
`JUPITER_UBATCH` and `JUPITER_STAGE_PROFILE`. Other trace logs stay
filtered, and console output is limited to plan checks, timing summaries and
errors; the complete stage samples still go into the per-run log file.
**Rebuild the PC llama-cli** after updating `common/log.cpp` or the
`-lv 3` sweep cannot see those records. This opt-in does not change
ordinary inference logging.

Example for the current 358-token Qwen3-14B experiment:

```powershell
python .\tools\jupiter\run_q_sweep.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --pc-layers 25 `
  --repeats 2 `
  --resume `
  --run-dp `
  --plan-output .\jupiter.plan
```

The default sweep paths for 358 tokens are:

```text
x=32  -> 32,32,32,32,32,32,32,32,32,32,38
x=48  -> 48,48,48,48,48,48,70
x=64  -> 64,64,64,64,64,38
x=80  -> 80,80,80,80,38
x=96  -> 96,96,96,70
x=112 -> 112,112,67,67
x=128 -> 128,128,102
```

Use `--dry-run --target-tokens 358` to preview plans without running the
token probe. Use `--x-values 32,64,96,128` for a smaller sweep.
`--resume` skips a point only when its log confirms the matching plan and
the extracted CSV contains rows.

Artifacts are isolated by actual prompt length and run configuration under
`jupiter-q-sweep\`:

```text
q-token-probe.log
tokens-<actual-token-count>-<config-hash>/
  q-x32-r1.log
  q-x32-r1.csv
  ...
  q-profile-all.csv
```

If `--run-dp` is present, the runner also writes the requested runtime plan.

Repeated samples are intentionally preserved in `q-profile-all.csv`. The
planner first takes the median of repeated measurements for each
`(x,y,device)`, then takes the slowest pipeline stage as Jupiter's
`q(x,y)`.

The older `run_q_sweep.ps1` is retained only for compatibility; the Python
runner is the recommended path.

Offline runner checks (no model or RPC needed):

```powershell
python -m unittest discover -s .\tools\jupiter -p test_run_q_sweep.py -v
```

## Sequence DP candidate refinement (targeted q calibration)

A fixed-x q sweep is **not a complete (x,prefix) grid**. Previously the DP
interpolated x *within each prefix*; if a prefix had only x=48 measured, then
q(128, that prefix) incorrectly inherited the 48-token cost. Now q first
interpolates/extrapolates **prefix y at the same chunk size x**, then interpolates
across measured chunk sizes. It also prints exact sample coverage:

```text
[JUPITER_SEQUENCE_COVERAGE] total=357 metric=latency_ms measured=... chunks=... missing=128:144;... status=NEEDS_TARGETED_Q
```

These are still estimates, not new measurements. Before accepting the plan,
run the iterative point calibration (reuses the full q sweep CSV; no need to
repeat the seven fixed-x sweeps):

```powershell
python .\tools\jupiter\refine_sequence_q.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --profile .\jupiter-q-sweep-final-lv3\tokens-357-<config-hash>\q-profile-all.csv `
  --target-tokens 357 `
  --pc-layers 25 `
  --metric compute_ms `
  --repeats 3 `
  --max-rounds 3 `
  --out-dir .\jupiter-refine-compute `
  --plan-output .\jupiter-refined-compute.plan
```

Replace `<config-hash>` with the generated folder name (or provide the
full actual CSV path). For example, in PowerShell:

```powershell
$profile = (Get-ChildItem .\jupiter-q-sweep-final-lv3 -Filter q-profile-all.csv -Recurse | Select-Object -First 1).FullName
```

Then pass `--profile "$profile"`. You can use `--metric wall_ms`
with **the same original CSV** but a different `--out-dir` and
`--plan-output` to generate a second candidate without a full re-sweep.
The original CSV remains unchanged.

Each missing `q(x,y)` is measured at **exactly** chunk length `x` and
prefix `y`, embedded within a complete plan of the observed prompt length.
The profiler uses `-lv 3`, `LLAMA_JUPITER_Q_SWEEP=1`,
`LLAMA_JUPITER_PROFILE_STAGE=1`, and `WINDOW=1`. This requires the updated
PC binary, but **not** a phone rebuild. The tool saves only target-position
samples to `q-profile-refined.csv`, re-solves DP and repeats if the candidate
moves to new unmeasured points. Every successful point is committed only
after all repeats finish.

A fully covered plan prints:

```text
[JUPITER_REFINE_DONE] status=FULLY_MEASURED ... plan=...
```

If the DP has not stabilized by `--max-rounds`, the tool exits 2 with
`[JUPITER_REFINE_INCOMPLETE]`. The intermediate `candidate.plan` and
`q-profile-refined.csv` remain in the output directory, **but the requested
final plan is NOT written**. Resume safely with
`--resume --max-rounds 6`. This never treats an extrapolated candidate as
a completed experiment.

For manual DP diagnostics, the `sequence` command now accepts
`--metric compute_ms|wall_ms` and `--require-measured`.

A fully measured *calibration* candidate is **not** a proof of improved
asynchronous Prefill latency. After refinement, compare it with default
and fixed64 using `run_prefill_bench.py` with profiling disabled
(`LLAMA_JUPITER_PROFILE_STAGE=0`). Keep the default policy unless the
formal asynchronous A/B confirms a reproducible speedup.

Offline tests:

```powershell
python -m unittest discover -s .\tools\jupiter -p test_refine_sequence_q.py -v
```

## Opt-in asynchronous Jupiter C2P socket transport

The Region Ring branches already have independent transfer workers, dedicated
RPC return lanes and exact per-chunk readiness fences. Jupiter's layer pipeline
is simpler (PC full layers followed by Phone full layers). Rather than
copying Tensor/Meta's P2C/FFN mailbox protocol, an **optional** transport path
in `ggml/src/ggml-rpc/ggml-rpc.cpp` adapts the transfer-worker idea:

- `LLAMA_JUPITER_ASYNC_C2P=1` enables the **PC-side** C2P worker only.
  The default is OFF, preserving the known-correct Jupiter RPC path.
- The GGML scheduler calls RPC's `cpy_tensor_async` for eligible C2P
  backend copies. The PC synchronizes the source Backend and stages the
  tensor into **private immutable Host storage before returning**.
  GPU-to-Host staging is still synchronous in this first version.
- The worker sends these staged bytes through its own transfer connection.
  Before dispatching a remote graph on its separate compute connection,
  it fences the transfer lane with `RPC_CMD_JUPITER_SYNC`. Graph commands
  and their completion-event markers are FIFO-ordered on the worker.
- A remote RPC output read, buffer release and backend synchronization
  fence outstanding worker jobs to keep buffer lifetimes safe. There is no
  new RPC command or model-format change; the existing Jupiter phone
  server already supports multiple connections sharing one graph/buffer
  registry.
- Unsupported copies are left on the pre-existing scheduler fallback.
  Do not enable this feature globally for general RPC: it is guarded by
  both `LLAMA_JUPITER_PREFILL=1` and `LLAMA_JUPITER_ASYNC_C2P=1`.

This is an **experimental network-only overlap**. It does **not** eliminate
GPU readback, all graph allocation synchronization, or guarantee a speedup.
Validate correctness (generated first token, absence of RPC errors), the
verification log's `[JUPITER_COMM] action=C2P_ENQUEUE`, and timing with
repeated runs before adopting it.

Pull the code and rebuild **PC only**:

```powershell
git fetch origin
git checkout origin/baseline/jupiter-infocom25 -- `
  ggml/src/ggml-rpc/ggml-rpc.cpp `
  tools/jupiter/run_prefill_bench.py `
  tools/jupiter/test_run_prefill_bench.py
cmake --build build-jupiter --config Release -j
python -m unittest discover -s .\tools\jupiter -p test_run_prefill_bench.py -v
```

Control and experiment use the **same** model/plan/prompt but different
output folders (repeats remain independent). For example:

```powershell
$bench = @(
  "--exe", ".\build-jupiter\bin\Release\llama-cli.exe",
  "--model", "E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
  "--rpc", "10.39.52.114:10139",
  "--prompt-file", ".\prompt.txt",
  "--plan", ".\jupiter-refined-compute.plan",
  "--target-tokens", "357",
  "--pc-layers", "25",
  "--modes", "default,dp",
  "--repeats", "3"
)

python .\tools\jupiter\run_prefill_bench.py @bench --out-dir .\jupiter-comm-legacy
python .\tools\jupiter\run_prefill_bench.py @bench --async-c2p --out-dir .\jupiter-comm-async
```

The benchmark runner overwrites any inherited `LLAMA_JUPITER_ASYNC_C2P`
value for reproducibility. The opt-in validation uses TRACE verbosity but
formal measurements stay at `-lv 3`, `PROFILE_STAGE=0`, `TRACE=0`.
The output `configuration.json` records the asynchronous-transport flag,
and `--resume` rejects an incompatible configuration.

**Implementation status:** source committed, but the C++ build, on-device
correctness and throughput A/B must be run on the actual PC+Phone system.

## Jupiter communication and graph timeline (diagnostic only)

To distinguish CPU waiting, GPU readback, socket sends, remote transfer
fences, graph preparation and slot lifetime waits, build the latest PC
binary and run diagnostic trials with `--comm-timing`. This sets
`LLAMA_JUPITER_TIMELINE=1` while leaving
`LLAMA_JUPITER_PROFILE_STAGE=0` (no per-split forced synchronizations).
The existing `--async-c2p` flag remains independent and OFF by default.

```powershell
git fetch origin
git checkout origin/baseline/jupiter-infocom25 -- `
  ggml/src/ggml-rpc/ggml-rpc.cpp `
  ggml/src/ggml-backend.cpp `
  src/llama-context.cpp `
  tools/jupiter/run_prefill_bench.py `
  tools/jupiter/test_run_prefill_bench.py `
  tools/jupiter/analyze_comm_timeline.py `
  tools/jupiter/test_analyze_comm_timeline.py

cmake --build build-jupiter --config Release -j
python -m unittest discover -s .\tools\jupiter -p test_analyze_comm_timeline.py -v
python -m unittest discover -s .\tools\jupiter -p test_run_prefill_bench.py -v
```

For a controlled **default-chunk** diagnostic, leave the DP plan unchanged
and compare Legacy/Async using separate directories, preferably alternating
the two runs to reduce frequency and temperature drift:

```powershell
$bench = @(
  "--exe", ".\build-jupiter\bin\Release\llama-cli.exe",
  "--model", "E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
  "--rpc", "10.39.52.114:10139",
  "--prompt-file", ".\prompt.txt",
  "--plan", ".\jupiter-refined-compute.plan",
  "--target-tokens", "357",
  "--pc-layers", "25",
  "--modes", "default",
  "--repeats", "1"
)

python .\tools\jupiter\run_prefill_bench.py @bench `
  --comm-timing --out-dir .\jupiter-timeline-legacy
python .\tools\jupiter\run_prefill_bench.py @bench `
  --comm-timing --async-c2p --out-dir .\jupiter-timeline-async
```

The PC `default-r1.log` files include, only when timeline is enabled:

- `[JUPITER_UBATCH_TIMELINE]`: chunk, prefix, slot and PC timestamp
- `[JUPITER_GRAPH_HOST]`: host graph build, allocation, inputs and submit times
- `[JUPITER_C2P_SYNC_FALLBACK]`: legacy synchronous RPC copy timing
- `[JUPITER_C2P_STAGE]`: private host allocation, source Backend sync,
  tensor readback, enqueue wait and total staging (joined by `task`)
- `[JUPITER_C2P_SEND]`: worker queue wait and client socket-send duration
- `[JUPITER_TRANSFER_FENCE]`: transfer-lane sync round-trip; prior
  SET_TENSOR requests have been processed, not merely sent
- `[JUPITER_GRAPH_PREPARE]`: PC control-lane fence and graph serialization
- `[JUPITER_GRAPH_DISPATCH]`: worker queue, transfer fence and graph-send times
- `[JUPITER_BUFFER_DRAIN]`: remote buffer-free/output-read dependency wait
- `[JUPITER_REMOTE_SYNC]`: backend-wide wait for outstanding work

Summarize the per-task timing report (default ignores prelude transfers
below 1 MiB; use `--min-bytes 0` to include them):

```powershell
python .\tools\jupiter\analyze_comm_timeline.py `
  --pc-log .\jupiter-timeline-async\default-r1.log `
  --csv .\jupiter-timeline-async\c2p-tasks.csv
python .\tools\jupiter\analyze_comm_timeline.py `
  --pc-log .\jupiter-timeline-legacy\default-r1.log
```

For **Phone-side graph reconstruction and actual device execution**,
rebuild the updated RPC server on the Phone, and start it with
`LLAMA_JUPITER_TIMELINE=1` in Termux. Its separate phone-side log emits
`[JUPITER_PHONE_TIMELINE] uid=... graph_prepare_ms=...
phone_compute_ms=...`. The Phone computation is timed using
synchronized `ggml_backend_graph_compute()`, not PC's graph enqueue.
Correlate PC and Phone records by `uid`, never subtract timestamps
from different machines. Pass `--phone-log PATH` to the report when available.

The `--comm-timing` run is **diagnostic**, not a clean throughput
measurement: taking timestamps and printing logs add overhead. For a
formal A/B comparison rerun the conditions *without* `--comm-timing`
and use at least three repeats. The benchmark manifest records
`comm_timing` and prevents resuming incompatible measurements.

**Interpretation:** `gpu_sync_ms` is Backend wait, not pure kernel time;
`device_read_ms` is API readback; `socket_send_ms` measures client send,
not Phone GPU readiness; `transfer_fence_ms` includes server round-trip;
and sums of overlapping durations are not end-to-end Prefill time.
No scheduling behavior changes when only the timing flag is enabled.

## Formal prefill A/B benchmark (async, low logging)

After q calibration and DP plan generation, **do not benchmark with
`LLAMA_JUPITER_PROFILE_STAGE=1`**; it synchronizes every backend split and
is only for q(x,y) profiling. A normal throughput run uses:

- `LLAMA_JUPITER_PREFILL=1`
- `LLAMA_JUPITER_PROFILE_STAGE=0`
- `LLAMA_JUPITER_TRACE=0`
- `LLAMA_JUPITER_WINDOW=2`
- `LLAMA_JUPITER_LOGITS_TOP10=0`
- `-lv 3` (keep server `prompt eval time` visible; Jupiter trace logs hidden)

The old unconditional first-token top-10 sorting/printing in
`tools/server/server-context.cpp` is now **opt-in** via
`LLAMA_JUPITER_LOGITS_TOP10=1`. Rebuild the **PC** CLI after pulling that
change; the phone RPC server does not need to be rebuilt.

Run all three modes with separate fresh processes, 3 repeats each:

```powershell
cmake --build build-jupiter --config Release -j

python .\tools\jupiter\run_prefill_bench.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --plan .\jupiter.plan `
  --target-tokens 357 `
  --pc-layers 25 `
  --modes default,fixed64,dp `
  --repeats 3 `
  --out-dir .\jupiter-bench-formal
```

- `default`: no plan-file or sequence override; runtime uses its default
  ubatch split.
- `fixed64`: a valid 64-token-oriented sequence, generated for the exact
  target length.
- `dp`: `LLAMA_JUPITER_PLAN_FILE` points to the generated DP plan file;
  the explicit sequence env override is absent.

Before measuring, the runner verifies the two explicit plans in separate
`-lv 4` runs (with stage profiling **off**) and stores
`verify-fixed64.log` and `verify-dp.log`. These verification timings are
never included in results. For actual measured runs, logs are captured to
files without printing full CLI output to the terminal. The mode order
rotates between repeats to reduce order/temperature bias.

Outputs:

```text
jupiter-bench-formal/
  configuration.json
  verify-fixed64.log
  verify-dp.log
  default-r1.log
  fixed64-r1.log
  dp-r1.log
  ...
  results.csv
  summary.txt
```

Use `--modes dp` to run only the DP policy, or `--resume` to reuse complete
previous logs. The `--target-tokens` value must correspond to the actual
prompt-length entry in the DP plan; the runner checks the measured token count
for every run. Use a new output directory for a new official comparison.

Offline checks:

```powershell
python -m unittest discover -s .\tools\jupiter -p test_run_prefill_bench.py -v
```


## Sequence partition DP

Input CSV:

```csv
x,y,latency_ms,device
32,0,18.2,PC
32,0,20.1,PHONE
64,0,32.0,PC
64,0,35.4,PHONE
...
```

Rows with the same `(x,y)` are reduced to the slowest stage latency, which is
the pipeline bottleneck after model balancing.

Example:

```bash
python tools/jupiter/jupiter_planner.py sequence \
  --profile q-profile.csv \
  --output jupiter.plan \
  --devices 2 \
  --min-chunk 32 \
  --max-chunk 128 \
  --target-length 358
```

The implementation follows Jupiter Eq.(2)-(4): it minimizes the slowest
sub-sequence latency for each `k`, then chooses `k <= 4|D|` minimizing the
estimated full pipeline latency.

## Model partition DP

Input CSV:

```csv
device,layer,time_ms,memory_bytes
PC,0,12.1,220000000
PHONE,0,18.7,220000000
...
```

Example:

```bash
python tools/jupiter/jupiter_planner.py model \
  --profile layer-profile.csv \
  --output jupiter.plan \
  --device-order PC,PHONE \
  --memory-budget-mib PC:5500 \
  --memory-budget-mib PHONE:7000
```

This implements Jupiter Eq.(1), minimizing the maximum stage time under optional
memory budgets. In the current two-stage runtime, the output is
`model.pc_layers=<boundary>`.

The paper treats planning as an offline one-shot process whose outputs are stored
and reused across inference requests. This plan-file design follows that model.

## Measured Model Partition profiling (PC GPU + Phone RPC)

The existing `jupiter_planner.py model` command implements the paper's
per-layer DP for a CSV of independently measured per-layer time/memory.
For the currently implemented two-stage, contiguous PC-prefix/Phone-suffix
runtime, `run_model_profile.py` provides a runnable measured-boundary
alternative: for each candidate layer boundary, run a fresh synchronized
prefill sample, read PC and Phone stage service times, and minimize
`max(PC stage service, Phone stage service)` over the *measured* feasible
boundaries. This is a discrete model-boundary search, not full arbitrary
per-layer profiling, and untested boundaries are not extrapolated.

The profiling driver uses `LLAMA_JUPITER_PROFILE_STAGE=1`,
`LLAMA_JUPITER_ASYNC_C2P=0`, `LLAMA_JUPITER_WINDOW=1`, and `-lv 4`.
These intentionally serialize stage measurements. The normal async
execution benchmark must leave stage profiling OFF.

On the PC, with the existing Phone RPC server running:

```powershell
cd E:\\llama\\llama-jupiter
git pull --ff-only origin baseline/jupiter-infocom25
cmake --build build-jupiter --config Release -j
python -m unittest discover -s .\\tools\\jupiter -p test_model_profile.py -v

python .\\tools\\jupiter\\run_model_profile.py `
  --exe .\\build-jupiter\\bin\\Release\\llama-cli.exe `
  --model E:\\llama\\models\\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\\prompt.txt `
  --target-tokens 357 `
  --boundaries 20,22,24,25,26,28 `
  --chunk 64 --repeats 2 `
  --out-dir .\\jupiter-model-profile `
  --plan-output .\\jupiter-model.plan
```

The profile uses a 357-token runtime sequence partition (64-token chunks
with a valid tail), but measures only the first formal 64-token chunk after
warmup at each boundary. The `--target-tokens` value must match the actual
prefill decode batch; if not, profiling rejects the run rather than silently
falling back. Memory columns are the observed sum of logged model and
compute/KV/output buffer sizes, not an accurate peak free-memory watermark.
For **hard** memory constraints pass `--pc-budget-mib` and
`--phone-budget-mib`; any candidate with missing memory data is rejected.
A successful process load also establishes that the specific partition was
loadable under the environment's memory conditions.

Outputs:
- `jupiter-model-profile/model-boundaries.csv`: measurements and rejected runs
- `jupiter-model-profile/pcN-rM.log`: full per-boundary logs
- `jupiter-model.plan`: selected `model.pc_layers=N` entry

The optimizer is a measured discrete two-device equivalent of the minimax
Model Partition objective. Because raw `STAGE_PROFILE compute_ms` can include
remote host submission/synchronization, it is a calibrated service proxy;
validate the chosen boundary in the normal asynchronous benchmark. The
per-layer DP `model` command remains available and unchanged.


## Minimum efficient prefill Chunk b (stage 2)

Jupiter's paper limits each sequence sub-block to at least b to avoid
accelerator under-utilization. The precise GPU/NPU occupancy counters are not
available portably through the current PC/Phone RPC setup. Therefore this
adaptation chooses a **measured throughput-saturation proxy**, not hardware
utilization: for candidate first-chunk sizes x, measure synchronized PC
accelerator and Phone RPC stage compute times at a fixed prefix y=0; compute
x / time for both stages; select the smallest x where **both** stages reach
at least a configurable fraction (default 90%) of their own measured best
throughput. The threshold is an explicit implementation heuristic, not a
paper-mandated constant. If the stages have no common plateau, output
NO_COMMON_PLATEAU and the best max-min fallback rather than claiming a
saturated optimum.

This stage reuses the q-sweep's stage instrumentation and full-prompt
sequence-plan checks. It profiles a short first block, followed by large
blocks; unlike a traditional x=8 sweep, the remaining full prompt is NOT
split into dozens of 8-token blocks. Each run uses exactly the same
model/partition/prompt and its own fresh process. The 2-token model warmup
and CPU-only bookkeeping stages are excluded from the throughput proxy.
Legacy C2P and synchronized stage measurement are used consistently.

Run on the PC, while Phone RPC is already listening at port 10139:

```powershell
cd E:\llama\llama-jupiter
git pull --ff-only origin baseline/jupiter-infocom25
cmake --build build-jupiter --config Release -j
python -m unittest discover -s .\tools\jupiter -p test_min_chunk_profile.py -v

python .\tools\jupiter\run_min_chunk_profile.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --model-plan .\jupiter-model.plan `
  --target-tokens 357 `
  --candidates 8,16,24,32,48,64,96,128 `
  --threshold 0.90 --repeats 2 `
  --out-dir .\jupiter-min-chunk
```

If Model Partition did not produce jupiter-model.plan, replace
`--model-plan .\jupiter-model.plan` with `--pc-layers 20` (or the selected
boundary from the model profiling stage). The script validates the actual
runtime token count and verifies each complete sequence plan was applied.
`--resume` skips completed samples when the configuration matches.

Output files:

- `jupiter-min-chunk/chunk-samples.csv`: per-run PC/Phone service times and failures
- `jupiter-min-chunk/chunk-summary.csv`: per-size median throughput and normalized ratios
- `jupiter-min-chunk/min-chunk.json`: selected b and selection status
- `jupiter-min-chunk/b-xN-rM.log`: individual calibration logs

For the next, Sequence DP stage, read the chosen minimum from JSON and pass
it explicitly to the existing planner. In PowerShell:

```powershell
$b = (Get-Content .\jupiter-min-chunk\min-chunk.json -Raw |
      ConvertFrom-Json).min_chunk
$xValues = (@(8,16,24,32,48,64,96,128) |
            Where-Object { $_ -ge $b }) -join ","

# After the model boundary and b are fixed, recalibrate q(x,y) for that
# boundary and these feasible sizes; use --min-chunk $b in Sequence DP.
python .\tools\jupiter\run_q_sweep.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --pc-layers 20 `
  --x-values $xValues --min-chunk $b --max-chunk 128 `
  --repeats 2 --run-dp --plan-output .\jupiter.plan
```

Replace the example PC layer count with the selected Model Partition boundary.
The chosen b is a heuristic minimum Chunk constraint for q(x,y) Sequence DP,
not a new runtime environment variable. Do not treat synchronized calibration
times as comparable to normal async Prefill benchmarks; rerun those separately.

## High-b sequence feasibility (357 tokens with max Chunk 128)

The minimum-chunk selection must be **feasible for the full prompt**, not
just efficient on an individual first chunk. A legal sequence has integer k
such that `k * b <= total_tokens <= k * max_chunk`.
For 357 tokens and `max_chunk=128`, `b=128` cannot produce a plan:
two chunks are at most 256 tokens, while three require at least 384.
The profiler now checks this before exporting the minimum. With the
measured example (PC 20 / Phone 20), the best feasible measured fallback
is `b=96`, whose PC rate is 89.43% of the measured peak (slightly under
90%). The status is `NO_FEASIBLE_PLATEAU`, not a claim of full saturation.

Re-select using the existing saved samples; no repeat of the individual
candidate profiling runs is necessary:

```powershell
python .\tools\jupiter\run_min_chunk_profile.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --model-plan .\jupiter-model.plan `
  --target-tokens 357 `
  --candidates 8,16,24,32,48,64,96,128 `
  --threshold 0.90 --repeats 2 `
  --out-dir .\jupiter-min-chunk --resume
```

A q-sweep with `--min-chunk 96 --max-chunk 128` can no longer
assume repeating every x produces a legal 357-token tail. The runner now
skips impossible x=96, falls back to an exact first x=112 plus balanced
tail `112,123,122`, and retains the legal x=128 plan
`128,128,101`. The existing targeted q refinement then measures the
unobserved `q(x,y)` points of the DP candidate and produces a
fully-measured runtime plan.

```powershell
$b = (Get-Content .\jupiter-min-chunk\min-chunk.json -Raw |
      ConvertFrom-Json).min_chunk

python .\tools\jupiter\run_q_sweep.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --pc-layers 20 `
  --x-values 96,112,128 --min-chunk $b --max-chunk 128 `
  --repeats 2 --run-dp --plan-output .\jupiter-b96.plan `
  --out-dir .\jupiter-q-b96
```

Do not run Sequence DP with `--min-chunk 128 --max-chunk 128` for
357 tokens; it has no legal partition. Prefer the fully measured
`refine_sequence_q.py` result for the final comparison.

## Stage 3: end-to-end q(x,y) -> Sequence DP -> measured final plan

`run_sequence_pipeline.py` is the minimal orchestration layer for the existing
`run_q_sweep.py`, `jupiter_planner.py` and `refine_sequence_q.py`.
It does not alter the llama.cpp executor or make any speculative q samples
count as measured. It reads the Stage-2 JSON, checks whether the selected
b permits an actual full-prompt partition, and can recompute an effective b
from the **already measured** stage-2 candidate medians. In particular,
the original 357-token JSON with `b=128` is automatically corrected to
`b=96` with `NO_FEASIBLE_PLATEAU` status; no Stage-2 sampling rerun is needed.

For the 357-token, PC20/Phone20 setup, automatic first-chunk seeds are
`x=101,112,128` (all can appear at prefix 0 in legal three-chunk plans).
The initial q-sweep creates the candidate DP plan; exact-q refinement then
samples only missing `(x,prefix)` pairs and resolves DP again. The final
runtime plan is written only after all chosen points have direct PC CUDA and
Phone RPC measurements. A final, separate inference run confirms that
llama.cpp applied the generated sequence without enabling stage profiling.

On Windows with the existing Phone RPC server listening:

```powershell
cd E:\llama\llama-jupiter
git pull --ff-only origin baseline/jupiter-infocom25
python -m unittest discover -s .\tools\jupiter -p test_sequence_pipeline.py -v

python .\tools\jupiter\run_sequence_pipeline.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --min-chunk-json .\jupiter-min-chunk\min-chunk.json `
  --target-tokens 357 `
  --q-repeats 2 --refine-repeats 2 --max-rounds 6 `
  --out-dir .\jupiter-sequence-stage3 `
  --plan-output .\jupiter-stage3.plan
```

Outputs:

- `jupiter-sequence-stage3/q-sweep/tokens-357-*/q-profile-all.csv`: initial q samples
- `jupiter-sequence-stage3/initial-dp.plan`: approximate DP candidate
- `jupiter-sequence-stage3/refinement/q-profile-refined.csv`: additional exact-q observations
- `jupiter-stage3.plan`: **final measured model + sequence partition**
- `jupiter-sequence-stage3/verify-runtime.log`: proof of runtime plan application
- `jupiter-sequence-stage3/sequence-summary.json`: result and coverage summary

If refinement exits with `JUPITER_STAGE3_INCOMPLETE`, re-run with the same
arguments plus `--resume`, increasing `--max-rounds` if desired. It will
reuse completed q-sweep samples and saved targeted measurements. A failed
or incomplete refinement does not overwrite the previous final plan.

The next stage is the normal, unprofiled Prefill A/B:

```powershell
python .\tools\jupiter\run_prefill_bench.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompt-file .\prompt.txt `
  --plan .\jupiter-stage3.plan `
  --target-tokens 357 --pc-layers 20 `
  --modes default,fixed64,dp --repeats 3 `
  --out-dir .\jupiter-stage4-bench
```

The q-sweep/refinement GPU stage timers deliberately synchronize devices;
do not interpret them as real overlap or actual throughput. Final runtime
verification is one trial, not an A/B result.

## Stage 5: multiple prompt lengths with separate calibrated Jupiter plans

The 357-token one-prompt result alone cannot establish a benefit at longer
sequences. `run_prompt_suite.py` profiles and benchmarks each supplied prompt
**independently**: actual prefill token probe, initial q(x,y), Sequence DP,
exact-q refinement, runtime plan verification, then the ordinary unprofiled
`default,fixed64,dp` Prefill A/B. The script only orchestrates existing
runners; it does **not** modify the inference executor.

Provide a directory of real UTF-8 `.txt` prompts, ideally spanning roughly
350, 700, 1000 and 1500+ tokens (these are intended groups, **not** assumed
token counts). Names and character lengths are not authoritative; every
case is tokenized by the actual `llama-cli` configuration before planning.
Longer prompts require a sufficient batch and context, and some larger
contexts may not fit the PC/Phone memory budgets.

Start with one model and the PC 20 / Phone 20 placement used in Stage 1.
The Stage-2 `min-chunk.json` contains an efficiency curve measured on the
original 357-token prompt. Reuse that curve only for the **same model and
devices**. The Stage-3 log explicitly marks reuse at a new prompt length.
With auto b, different lengths may select different feasible minimum chunk
sizes; a prompt length divisible by 128 can choose b=128 and hence force
fixed 128-token chunks (no DP degrees of freedom). To compare prompt
lengths with one common constraint, use `--min-chunk 96`, a documented
**fixed-b ablation**, not a claim that this value strictly reached the
90% saturation threshold. This example uses fixed b=96.

```powershell
cd E:\llama\llama-jupiter
git pull --ff-only origin baseline/jupiter-infocom25
python -m unittest discover -s .\tools\jupiter -p test_run_prompt_suite.py -v
python -m unittest discover -s .\tools\jupiter -p test_sequence_pipeline.py -v
python -m unittest discover -s .\tools\jupiter -p test_refine_sequence_q.py -v

python .\tools\jupiter\run_prompt_suite.py `
  --exe .\build-jupiter\bin\Release\llama-cli.exe `
  --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
  --rpc 10.39.52.114:10139 `
  --prompts-dir .\jupiter-prompts `
  --min-chunk-json .\jupiter-min-chunk\min-chunk.json `
  --min-chunk 96 `
  --pc-layers 20 --ubatch 128 `
  --batch 2048 --context 2048 --threads 8 `
  --q-repeats 2 --refine-repeats 2 --max-rounds 6 `
  --bench-repeats 7 --modes default,fixed64,dp `
  --window 2 `
  --out-dir .\jupiter-prompt-suite `
  --resume
```

You may also pass `--prompt-file path` multiple times instead of
`--prompts-dir`. `--dry-run` previews case directories without launching
the model. The pipeline uses fixed per-prompt configuration fingerprints,
supports `--resume` to preserve finished work, and optionally
`--continue-on-error` to record one failed prompt while moving on to others.

Each case directory holds `token-probe.log`, `stage3-run.log`,
`stage3/q-sweep/`, `stage3/refinement*/`, `final.plan`, `bench-run.log`,
`bench/results.csv`, `bench/summary.txt` and the exact case configuration.
At the top of the output folder, the script writes:

- `suite-results.csv`: actual token counts, per-prompt DP chunks, median
  Prefill latency, tokens/s, and DP-versus-Default speedup
- `suite-summary.json`: full experiment settings and status for every prompt

Only a fully measured and runtime-verified Stage-3 plan advances to the
Prefill benchmark; benchmark runs use `PROFILE_STAGE=0`, `TRACE=0`,
`async-c2p=0`, and no per-chunk communication timing. Each process is
an independent sample, with rotating mode order and medians.

For multi-prompt comparisons, the benchmark has a narrow opt-in
`--timing-token-tolerance 1` to accommodate the known difference between
the actual prefill batch token count and the CLI's timing token counter.
Stage 5 enables it, preserves `tokens` as the actual planned prefill
count, and writes the unmodified `timing_tokens` separately in each
benchmark CSV. Differences greater than one token still fail. The
standalone Stage-4 runner defaults to strict tolerance zero.

The Sequence DP solver now automatically searches enough chunk counts
for long prompts instead of capping at `4 * devices = 8`; each DP transition
is restricted to valid chunk sizes and prefix lengths. Example: at
`max_chunk=128`, a 1536-token prompt needs **at least 12 chunks**.

A successful suite establishes measured results for these prompts, not
faithfulness of all Jupiter system components. If the DP variant remains
slower, report that finding rather than changing the original baseline.
