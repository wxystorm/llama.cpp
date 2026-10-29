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
