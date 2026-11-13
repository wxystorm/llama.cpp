# ShareGPT40 benchmark results (2026-11-13)

This branch contains **10 experimental runs** and **396 successful prompt-level observations**, spanning prompt groups 128, 256, 512 and 1024.

- **Nine** runs have 40 PASS observations each.
- **Jupiter Mistral Small 3.2-24B** has 36 PASS observations (128: 10, 256: 10, 512: 10, 1024: 6). The four uncompleted 1024 prompts are **not** counted or inferred.
- Two extra **PC-only Qwen3-32B** reference observations are recorded separately; they are not counted in the 396 complete/partial experiment observations or group statistics.

## Run inventory

| Method | Model | Run ID | Status |
| --- | --- | --- | --- |
| Prima.cpp | Qwen3-14B | `prima-14b` | 40/40 |
| Prima.cpp | Mistral Small 3.2-24B Q6_K | `prima-24b` | 40/40 |
| Prima.cpp | Qwen3-32B | `prima-32b` | 40/40 |
| Jupiter | Qwen3-14B | `jupiter-14b` | 40/40 |
| Jupiter | Mistral Small 3.2-24B Q6_K | `jupiter-24b` | 36/40 |
| Jupiter | Qwen3-32B | `jupiter-32b` | 40/40 |
| Galaxy | Qwen3-14B | `galaxy-14b` | 40/40 |
| Ours (region ring) | Qwen3-14B | `ours-14b-g9p1` | 40/40 |
| Ours (region ring) | Mistral Small 3.2-24B Q6_K | `ours-24b-htp` | 40/40 |
| Ours (region ring) | Qwen3-32B | `ours-32b-g8p6` | 40/40 |

## Dataset files

- `config.json`: model and runtime configuration for each run, including references to per-run config files.
- `results.csv`: 396 successful individual prefill measurements; primary key is `framework,model,group,prompt`.
- `group-summary.csv`: 40 group rows containing arithmetic mean and median prefill throughput and latency. Each `runs` field records the actual sample count (Jupiter 24B group 1024 has `6`).
- `archive-manifest.json`: source ZIP filenames, sizes and SHA-256 hashes.
- `runs/jupiter-24b/run-config.json`: Jupiter 24B DP configuration, sanitized to use path and RPC placeholders.
- `runs/jupiter-24b/per-prompt-details.csv`: all 36 observed DP measurements and selected chunk plans.
- `runs/ours-*/` and `runs/prima-24b/`: other available run configurations and detailed measurements.
- `runs/pc-only-32b/observations.csv`: single-PC 32B reference: **256 group = 21.58 tok/s** (266 tokens from a server log, prompt index unknown); **512 group = 32.9 tok/s** (`prompt_512_01.txt`, user-reported). Each reference is a single observation, not a Top-5 mean.

## Experimental notes

The Jupiter 24B run used Sequence DP with `graph_prebuild=false`, `context=2048`, `batch=2048`, `ubatch=128`, `pc_layers=32`, `cpu_layers=24`, and 96-128 chunk bounds. Its status is partial because four 1024-group samples lack completed DP results.

Prompt groups (128, 256, 512, 1024) are sampling buckets and **are not exact tokenizer token counts**. Engines, quantization, run settings and tokenization can differ between methods; rank-sorted Top-5 measurements need not correspond to the same prompt. Group means and medians are arithmetic calculations on measured observations, not pooled TPS. The PC-only references are not used in these means or Top-5 ranking.

The **original binary ZIP archives are not committed** to this branch because the connected GitHub writer only supports UTF-8 files. SHA-256 values of the original archives are stored in `archive-manifest.json`.
