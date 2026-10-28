# CPU / Phone x-layer Layer-vs-Tensor benchmark

This experiment is independent from Region Ring and local CUDA.

It asks one narrow question:

> If we load exactly `x` transformer layers, is the fastest full-layer
> CPU/Phone placement still better than making those same `x` layers
> directional Tensor layers?

The benchmark uses real `llama-cli` prompt evaluation for every candidate.
There is no analytical performance model in v1.

## Real truncated loading

For every tested `x`, the runner sets:

```
LLAMA_HYBRID_EXPERIMENT_N_LAYERS=x
```

The model first reads the original GGUF metadata so Qwen3-MoE shapes and model
type are initialized normally.  Before tensor creation, the effective block
count is truncated to `x`.

Therefore:

```
x = 4
```

creates and loads only:

```
blk.0.*
blk.1.*
blk.2.*
blk.3.*
```

and does **not** create/materialize `blk.4.* ... blk.47.*`.

The tokenizer/embedding/output tensors are still loaded so the normal llama
runtime can execute a real prompt through the truncated model.  The runner
uses `--no-mmap` so the PC path does not mmap the entire GGUF as a shortcut.

A successful run must contain:

```
[HYBRID_EXPERIMENT_TRUNCATE] original_layers=48 active_layers=x ...
```

The Python runner rejects a candidate if that line does not report the
requested `x`.

## Backends

The benchmark deliberately uses only:

```
PC CPU
Phone RPC GPU
```

The command registers the phone with `--rpc`, then passes:

```
--device RPC0
```

so the local CUDA GPU is excluded from the offload device list.  If the phone
registers under another device name, use `--phone-device`.

## Layer planner

For the `x` loaded layers, the Layer planner searches full-layer placement.

V1 enumerates one contiguous Phone block, including the two endpoints:

```
x=4

C C C C
P C C C
P P C C
C P P C
C P P P
P P P P
...
```

where:

- `C` = PC CPU Full layer
- `P` = Phone Full layer

A single Phone block is sufficient for this first full-layer baseline: multiple
disconnected Phone blocks add extra ownership transfers to an otherwise
sequential full-layer execution.

If a candidate does not fit Phone memory or fails to execute, it is simply not
eligible to win.

## Tensor planner

All `x` loaded layers are Tensor layers with alternating ownership:

```
x=1: C2P
x=2: C2P P2C
x=3: C2P P2C C2P
x=4: C2P P2C C2P P2C
...
```

This is the intended ownership cycle:

```
PC
  --C2P--> Phone
  --P2C--> PC
  --C2P--> Phone
  ...
```

For every `x`, the Tensor planner searches:

- C2P physical PC FFN share;
- P2C physical PC FFN share;
- Tensor prefill chunk size `XT`.

The defaults are:

```
R = 0.20, 0.33, 0.50
XT = 32, 64, 128
```

so the first coarse search can find both compute balance and communication /
chunking crossover.

Use `--same-ratio` to force C2P and P2C to share the same R candidate.

## Explicit per-layer policy

This experiment branch adds:

```
LLAMA_HYBRID_EXPERIMENT_POLICY
```

with exactly one entry per **active/truncated** layer:

```
C,P,C2P,P2C,...
```

The experiment policy cannot be combined with Region Ring.

## Example

PowerShell:

```powershell
python .\tools\cpu-phone-tensor-bench\cpu_phone_tensor_bench.py `
  --exe .\build-tensor-bench\bin\Debug\llama-cli.exe `
  --model E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
  --rpc 192.168.1.123:50052 `
  --phone-device RPC0 `
  --layers 2,4,6,8 `
  --ratios 0.20,0.33,0.50 `
  --tensor-chunks 32,64,128 `
  --same-ratio `
  --prompt-file .\prompt.txt `
  --ctx 512 --batch 512 --ubatch 512 --threads 8
```

For one exact depth:

```powershell
--layers 6
```

To cap the full-layer planner's Phone residency:

```powershell
--max-phone-layers 4
```

## Output

For each `x`, both planners choose a measured winner and the runner then
re-runs the two winners.

The console summary is:

```
x= 2 layer=...ms tensor=...ms speedup=... winner=...
x= 4 layer=...ms tensor=...ms speedup=... winner=...
x= 6 layer=...ms tensor=...ms speedup=... winner=...
```

A speedup greater than 1 means:

```
best Layer placement time / best Tensor placement time > 1
```

so alternating directional Tensor is faster for that actual x-layer model.

The output directory also contains:

- one complete log per candidate run;
- `all-results.csv`;
- `summary.json`.

The crossover we care about is the smallest `x` for which the Tensor winner
consistently beats the Layer winner.
