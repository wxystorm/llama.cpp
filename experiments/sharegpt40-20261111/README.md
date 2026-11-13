# ShareGPT40 benchmark snapshot (2026-11-13)

Ten experimental runs cover **396 successful prompt-level records**: four prompt-length groups (128, 256, 512, 1024), with ten prompts per group and per run. Each result row is uniquely keyed by framework, model, prompt group and prompt filename. Existing run records are updated in-place when a matching key is found, never duplicated.

## Run inventory

- Prima.cpp: Qwen3-14B, Qwen3-32B, Mistral Small 3.2 24B (Q6_K).
- Jupiter: Qwen3-14B and Qwen3-32B.
- Galaxy: Qwen3-14B.
- Our distributed region-ring splits:
  - Qwen3-14B `ours-14b-g9p1`, layout `G9,P1,G9,P1,G9,P1,G10`.
  - Qwen3-32B `ours-32b-g8p6`, layout `G8,P6,G8,P6,G8,P6,G8,P6,G8`; fixed layout, chunk 96 or 128 per prompt, drain-first scheduling.
  - Mistral Small 3.2 24B `ours-24b-htp`, layout `G5,P3,G5,P3,G6,P3,G5,P4,G6`; forced HTP plan and independent-backend profiling.

## Dataset files

- `config.json`: models, hardware and per-run settings.
- `results.csv`: 360 prefill observations.
- `group-summary.csv`: 36 group means and medians (not pooled TPS).
- `archive-manifest.json`: SHA-256 and byte sizes of nine uploaded source ZIPs.
- `runs/*/run-config.json`: reproducible run settings, with host file paths and IP addresses replaced by environment placeholders.
- `runs/*/per-prompt-details.csv`: context, chunks, latency, throughput, decode and GPU-bank timing measurements.

For the new 32B source, only the top-level final `summary.csv` was used; the included separate verification runs are not double-counted. For the new 24B source, the final summary contains 40 PASS records; one earlier interrupted attempt was excluded.

**Original binary ZIP files and verbose logs are not stored in this Git branch.** The GitHub connector can write UTF-8 text data, but cannot directly push local ZIP binaries. Use the archived checksums to verify separately retained original files.

Performance comparisons require care: Prima, Jupiter, Galaxy and our fixed/layout-controlled region-ring runs differ in engine, model, quantization, planner and warmup/profile overhead. Prompt group labels are sampling buckets, not exact tokenizer counts. The `prefill_ms` field reflects the measured prefill stage, while `wall_ms` in detailed records can include setup and model preload time.

## Additional partial baseline: single-PC 32B

Two observations are documented in `runs/pc-only-32b/` and **are not part of the 360 complete-run records**. Group 256: 266 prompt tokens / 12325.37 ms / 21.58 tok/s from an engine log; group 512, `prompt_512_01.txt`: user-reported 32.9 tok/s, with no accompanying timing log. The group 256 log must not be assigned to `prompt_512_01.txt`.

## Jupiter 24B addition (partial)

The uploaded `jupiter-24b-数据.zip` contains 36 PASS Sequence DP results for Mistral Small 3.2-24B-Q6_K: groups 128/256/512 contain 10 each; group 1024 contains only 6. The additional 4 entries were not completed in the supplied DP summary and are **not imputed**. The run used `graph_prebuild=false`, 2048 context, batch 2048, ubatch 128, 32 PC layers (24 CPU), DP chunks bounded from 96 to 128, and one benchmark repeat.

For this run, `runs/jupiter-24b/run-config.json` records the parameters and `runs/jupiter-24b/per-prompt-details.csv` retains the observed per-prompt TPS, latency and selected DP chunks. The group summary `runs` column contains actual sample counts (for Jupiter 24B: 10/10/10/6). Jupiter 24B Top 5 at 1024 is based on five of the six available values.

The binary ZIP archive is not included in this repository branch; its SHA-256 is preserved in `archive-manifest.json`. Existing nine complete runs remain unchanged. PC-only 32B remains a separate two-observation reference in `runs/pc-only-32b/` and is not part of the Top 5 samples or full group statistics.
