# Fig. 4(b): no universally best fixed Chunk width (Qwen3-32B)

This is an **experiment runner**, not generated paper results. It measures three **fixed** Chunk widths (32, 64, 128) at three ShareGPT input-length groups (128, 512, 1024), using the **same fixed 64-layer placement** and **identical full Cycle runtime policy**. It tests whether the best fixed Chunk width changes with prompt length. Crossing curves are a hypothesis, not an assumed outcome.

- **Placement:** `G8,P6,G8,P6,G8,P6,G8,P6,G8` by default; exactly identical in every case. Use `--layout` only to supply a known-feasible *common* placement, never a different placement per Chunk.
- **Chunks:** `XC = XG = XP = 32 / 64 / 128`, fixed independently of prompt length, **not** selected by the Planner. Native fixed mode skips hardware profiling and Planner search.
- **Runtime:** GPU two banks, async prefetch, Drain-First, resource-aware Chunk scheduling, streaming ON and forced small-batch mover. The forced short-prompt mover is intentionally enabled for *every* case, to avoid silently switching the 128-token case into a non-streaming runtime.
- **Inference:** one generated token, non-conversation, same model, prompt file and seed. One separate **unmeasured** prefill warmup per Chunk/repetition invocation; measured prefill comes from the engine, not wall time.
- **Repetitions:** three by default: 9 `(length, Chunk)` configuration points × 3 independent measurements = 27 measurements. Chunk order rotates by repetition to reduce monotonic thermal bias.

## PowerShell: run from the experiment branch

Use a built PC `llama-completion.exe` from the same branch and a running Phone RPC server with the matching Qwen3-32B model and HTP configuration. Adjust the executable and RPC port to your environment.

```powershell
$root = 'E:\llama\llama-tensor-v2'
cd $root

git fetch origin
git switch exp/fig4b-32b-fixed-chunks-20261118
cmake --build .\build-tensor-v2 --config Release --target llama-completion -j 8

$opts = @(
  '--exe', "$root\build-tensor-v2\bin\Release\llama-completion.exe",
  '--model', 'E:\llama\models\Qwen3-32B-Q4_K_M.gguf',
  '--rpc', '10.238.229.239:50052',
  '--prompt-dir', 'E:\llama\datasets\sharegpt\sharegpt40',
  '--out', "$root\fig4b-32b-fixed-chunks",
  '--groups', '128', '512', '1024',
  '--chunks', '32', '64', '128',
  '--batch', '2048', '--ubatch', '2048', '--threads', '8'
)

# Inspect scheduled commands first (no inference).
python .\tools\region-ring-experiments\run_32b_fig4b_fixed_chunks.py @opts --dry-run

# One repetition for a hardware smoke check (9 measurements).
python .\tools\region-ring-experiments\run_32b_fig4b_fixed_chunks.py @opts --repeats 1

# Full experiment: same --out and same settings, 3 repetitions.
# Already PASS cases from repeat 1 are resumed, not rerun.
python .\tools\region-ring-experiments\run_32b_fig4b_fixed_chunks.py @opts --repeats 3
```

If the PC runs out of physical memory, pass `--no-mmap` **consistently** and use a new output directory. Do not change runtime flags between Chunk cases. For additional diagnostics, add `--trace-tasks` uniformly (again use a new output directory). Default runs leave mmap enabled to avoid eagerly loading the entire 32B model into host RAM.

## Outputs and verification

The script writes to `--out`:

| File/directory | Meaning |
|---|---|
| `chunk_032/repeat_01/`, etc. | Native runner `run-config.json`, `summary.csv`, warmup, `metrics.jsonl`, raw logs and attempt results; resumable |
| `fig4b_raw.csv` | All attempted/missing measurements, actual model-prefill tokens, context, native latency and audit errors |
| `fig4b_summary.csv` | Per-group/per-Chunk median prefill latency across repetitions, pass counts, normalization and winner |
| `fig4b_plot_normalized.csv` | Three rows suitable for Fig. 4(b): actual tokens + normalized curves. **Empty until all requested runs pass audit** |

Validation rejects missing/failed samples, incorrect applied `layout` or runtime `XC/XG/XP`, inactive GPU bank mover, absent forced-small-batch acknowledgment, accidental dynamic Planner activation, missing evidence of the requested runtime flags and changes in *actual model token count or context* for the same prompt across configurations. A failed run is not replaced with a prediction or zero latency.

For each input group `g`, with median measured prefill latency `T(g,c)` for fixed Chunk `c`, the plotted value is

```
normalized(g,c) = T(g,c) / min(T(g,32), T(g,64), T(g,128))
```

A value of 1.0 is the fastest of the three tested fixed configurations **at that prompt length**, not an oracle or comparison with Full Cycle's dynamic Planner. Group labels denote dataset buckets; use `actual_tokens` on the final X-axis if you want the exact model-token lengths.

**Interpretation:** If the winning Chunk changes between groups, Fig. 4(b) supports request-length-dependent Chunk adaptation. If the winner stays the same, report that result; these three settings alone do not establish an absence of a universal optimum. This experiment **does not** test changing placement/Region size or the number of PC/Phone cycles.

## Offline test

```powershell
python -m unittest discover -s .\tools\region-ring-experiments -p test_run_32b_fig4b_fixed_chunks.py -v
```

Tests validate the command builder, fixed Runtime settings, 3-way curve normalization, and failure handling with synthetic CSVs. They are not a substitute for physical 32B CUDA/HTP measurements.