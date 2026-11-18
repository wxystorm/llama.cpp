# Fig. 4(b) — 32B Fixed Chunk Sweep (2026-11-18, Repeat 1 of 3)

Status: **9/9 request measurements PASS**, only Repeat 1 measured. These values were transcribed from the user's Windows PowerShell / native server logs shared in the conversation. This record is not a replacement for the run's original CSV/JSONL/log files. Do not describe it as 3-repeat median results.

## Configuration

- Repository/branch: `wxystorm/llama.cpp`, `exp/fig4b-32b-main50-parity-20261118`.
- Model: `Qwen3-32B-Q4_K_M.gguf` (64 transformer layers).
- Full Cycle layer placement: `G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7` (43 GPU, 21 phone HTP).
- Execution: resident PC `llama-server` via phone RPC `10.238.229.239:50052`; no-mmap enabled.
- PC server: `--split-mode tensor --flash-attn on -c 2048 -b 2048 -ub 2048 -t 8 --parallel 1 --no-cont-batching --no-cache-prompt --cache-ram 0 --ctx-checkpoints 0 --no-context-shift --no-warmup`.
- Runtime: two GPU banks, async prefetch on, Drain-First on, resource-aware scheduler, fixed-Chunk ablation mode (native per-request dynamic Chunk search disabled), memory allocators unchanged; memory estimates reported SKIPPED.
- Inputs: `sharegpt50_short`, one prompt per group: `prompt_128_01.txt`, `prompt_512_01.txt`, `prompt_1024_01.txt`; generated-token count 1; cache_n=0 on all nine requests.
- Per-case separate resident server, with 5s between requests, excluding delay from `prompt_ms`.
- Result root: `E:\llama\llama-tensor-v2\fig4b-32b-chunk64-128-256-nommap`.

## Measured raw data: Repeat 1 of 3

All requests had `status=PASS` and matching `cache_n=0`. Times are engine Prefill `prompt_ms` from the logs, **not** application wall time.

| Prompt group | Actual input tokens | Chunk 64 (ms) | Chunk 128 (ms) | Chunk 256 (ms) | Best fixed Chunk |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 | 167 | 2435.146 | 2576.364 | 3321.995 | 64 |
| 512 | 587 | 5225.046 | 4637.428 | 5150.254 | 128 |
| 1024 | 1033 | 8991.450 | 7346.917 | 7677.362 | 128 |

| Prompt group | Chunk 64 (tok/s) | Chunk 128 (tok/s) | Chunk 256 (tok/s) |
| --- | ---: | ---: | ---: |
| 128 | 68.579 | 64.820 | 50.271 |
| 512 | 112.344 | 126.579 | 113.975 |
| 1024 | 114.887 | 140.603 | 134.551 |

## Measured normalized latency (lower is better)

Each group's fastest measurement is set to 1.0000. These are **repeat-1** ratios, not median ratios:

| Actual input tokens | Chunk 64 | Chunk 128 | Chunk 256 |
| ---: | ---: | ---: | ---: |
| 167 | 1.0000 | 1.0580 | 1.3642 |
| 587 | 1.1267 | 1.0000 | 1.1106 |
| 1033 | 1.2238 | 1.0000 | 1.0450 |

Normalization: `T(group, chunk) / min_{chunk in {64,128,256}} T(group, chunk)`.

## Native fixed Chunk / runtime physical widths

All nine requests have selected `XC = XG = XP` as shown below. `runtime_xg` and `runtime_xp` were equal within each case; physical tail coalescing may widen a selected width.

| Actual tokens | Fixed target | Native selected XC=XG=XP | Runtime XG=XP | Planner time (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 167 | 64 | 64 | 64 | 0.013 |
| 587 | 64 | 64 | 66 | 0.012 |
| 1033 | 64 | 64 | 65 | 0.017 |
| 167 | 128 | 128 | 128 | 0.011 |
| 587 | 128 | 128 | 128 | 0.012 |
| 1033 | 128 | 128 | 130 | 0.012 |
| 167 | 256 | 167 | 167 | 0.012 |
| 587 | 256 | 256 | 256 | 0.011 |
| 1033 | 256 | 256 | 259 | 0.016 |

Note: The 167-token prompt cannot execute a literal 256-token Chunk; native fixed policy clamps to a single whole-prompt 167-token Chunk. This is expected, **not an invalid run**.

## Interpretation and next step

- Repeat-1 best fixed size changes with input length: Chunk 64 is fastest for 167 tokens, Chunk 128 for 587 and 1033 tokens.
- Preliminary evidence for request-dependent Chunk choice, but **do not claim statistical stability or final paper results** from one repeat.
- Finish Repeat 2 and Repeat 3 with the same binary, Profile, server arguments, no-mmap policy, RPC, model and dataset. Generate final medians from `fig4b_summary.csv` and normalized curves from `fig4b_plot_normalized.csv`.
- The user-provided native logs explicitly reported `[SERVER_SWEEP] status=PASS completed=3/3` for each of Chunk 64, Chunk 128, and Chunk 256; the wrapper produced `[FIG4B_READY]` for the 1-repeat plot.

## Files on the benchmark PC

- `fig4b_raw.csv` (nine rows for Repeat 1)
- `fig4b_summary.csv` (one-repeat summaries; not final medians)
- `fig4b_plot_normalized.csv` (one-repeat illustration, not final Fig. 4(b))
- `chunk_064/repeat_01/requests.csv`, `chunk_128/repeat_01/requests.csv`, `chunk_256/repeat_01/requests.csv` and corresponding `requests.jsonl` / `llama-server.log`.
