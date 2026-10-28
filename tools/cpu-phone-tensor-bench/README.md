# CPU / Phone Layer vs Directional Tensor benchmark

This experiment is intentionally independent from Region Ring and local CUDA.

It asks one narrow question:

> Given an `x`-layer window that would otherwise run on PC CPU, is it faster
> to move whole layers to the Phone GPU, or to turn all `x` layers into
> alternating directional Tensor layers (`C2P, P2C, C2P, ...`)?

The program uses real `llama-cli` prompt evaluation for every candidate.
There is no analytical performance model in v1.

## Experiment families

For a requested window `[start, start+x)`, all layers outside the window are
kept `PC_ONLY`.

### Layer planner

The Layer planner enumerates one contiguous `PHONE_ONLY` block anywhere
inside the window:

```
CCCC
PCCC
PPCC
CPPC
CPPP
PPPP
...
```

This allows the planner to choose how many slow CPU layers to replace with
Phone Full layers and where the ownership handoff occurs, without introducing
gratuitous repeated PC/Phone handoffs.

### Tensor planner

All `x` layers are Tensor layers with alternating ownership:

```
x=1: C2P
x=2: C2P P2C
x=3: C2P P2C C2P
x=4: C2P P2C C2P P2C
...
```

C2P and P2C can use different physical PC FFN shares.  The default coarse
grid is:

```
0.20, 0.33, 0.50
```

so v1 evaluates the 3x3 C2P/P2C ratio combinations.

The winner in each family is selected by measured `prompt eval time`.
The two winners are then re-run and compared directly.

## Explicit per-layer policy

This branch adds the experimental environment variable:

```
LLAMA_HYBRID_EXPERIMENT_POLICY
```

It contains exactly one comma-separated mode per model layer:

```
C,P,C2P,P2C,...
```

where:

- `C` = PC CPU Full layer
- `P` = Phone Full layer
- `C2P` = PC-attention directional Tensor layer, output owned by Phone
- `P2C` = Phone-attention directional Tensor layer, output owned by PC

The experiment policy has priority over the legacy Hybrid policy builder and
must not be combined with Region Ring.

## Example

PowerShell:

```powershell
python .\tools\cpu-phone-tensor-bench\cpu_phone_tensor_bench.py `
  --exe .\build-tensor-bench\bin\Debug\llama-cli.exe `
  --model E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
  --rpc 192.168.1.123:50052 `
  --model-layers 48 `
  --start-layer 0 `
  --layers 1:8 `
  --ratios 0.20,0.33,0.50 `
  --prompt-file .\prompt.txt `
  --ctx 512 --batch 512 --ubatch 512 --threads 8
```

To test only a few depths:

```powershell
--layers 2,4,6,8
```

To force the same FFN ratio in both directions:

```powershell
--same-ratio
```

To cap the Layer planner's Phone Full block for memory experiments:

```powershell
--max-phone-layers 4
```

Extra llama-cli arguments can be appended after `--extra`:

```powershell
--extra --no-warmup
```

Extra environment values can be supplied with repeated `--extra-env`:

```powershell
--extra-env NAME=value
```

## Output

The output directory contains:

- one full log per real candidate run;
- `all-results.csv` with every candidate;
- `summary.json` with the best Layer and Tensor configuration for every `x`.

The console summary reports:

```
x= 2 layer=...ms tensor=...ms speedup=... winner=...
x= 3 layer=...ms tensor=...ms speedup=... winner=...
...
```

A speedup greater than 1 means alternating directional Tensor is faster than
the best full-layer placement found for the same `x`.

## Important interpretation

v1 still loads the complete GGUF and executes the complete transformer.  The
`x` value is an **experiment window**, not a truncated GGUF.  All layers
outside the window remain PC Full and therefore contribute the same constant
background to both planner families.

This is deliberate for the first experiment: it avoids introducing a second
custom graph executor while keeping the A/B comparison on the exact production
C2P/P2C implementation.  The sign of the Layer-vs-Tensor comparison is still
meaningful because the unchanged outer layers are common to both candidates.
The logs also parse `META_LAYER_SUM` when available so the window-local layer
time is recorded separately.

If the first experiment shows a clear Tensor crossover, the next step is to
move this same search into a dedicated x-layer graph executor so the constant
outer-model cost disappears.
