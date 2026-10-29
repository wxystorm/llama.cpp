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
