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
$env:LLAMA_JUPITER_SEQUENCE_PLAN="64,64,64,64,64,38"
```

The scheduler then prints one line per stage:

```text
[JUPITER_STAGE_PROFILE] ... backend=CUDA0 ... input_ms=... compute_ms=... wall_ms=...
[JUPITER_STAGE_PROFILE] ... backend=RPC0[...] ... input_ms=... compute_ms=... wall_ms=...
```

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

## Automatic Windows q(x,y) sweep

For the Windows laptop + phone RPC setup, the sweep can now be run end-to-end
with one PowerShell command. It generates repeated-X paths that cover
`x = 32,48,64,80,96,112,128`, runs synchronous calibration, extracts each
log, merges all samples, and can immediately run the sequence DP.

The prompt file must contain the same prompt whose tokenized length is supplied
as `-TargetTokens`.

Example for the current 358-token Qwen3-14B experiment:

```powershell
.\tools\jupiter\run_q_sweep.ps1 \
  -Exe .\build-jupiter\bin\Release\llama-cli.exe \
  -Model E:\llama\models\Qwen3-14B-Q4_K_M.gguf \
  -Rpc 10.39.52.114:10139 \
  -PromptFile .\prompt.txt \
  -TargetTokens 358 \
  -PcLayers 25 \
  -Repeats 2 \
  -RunDp \
  -PlanOutput .\jupiter.plan
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

Use `-DryRun` to print the generated paths without running inference. Use
`-XValues "32,64,96,128"` for a smaller sweep.

Artifacts are written under `jupiter-q-sweep\` by default:

```text
q-x32-r1.log
q-x32-r1.csv
...
q-profile-all.csv
```

If `-RunDp` is present, the script also writes the requested runtime plan.

Repeated samples are intentionally preserved in `q-profile-all.csv`. The
planner first takes the median of repeated measurements for each
`(x,y,device)`, then takes the slowest pipeline stage as Jupiter's
`q(x,y)`.

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
