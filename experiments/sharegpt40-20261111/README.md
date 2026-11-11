# ShareGPT40 benchmark snapshot (2026-11-11)

Seven experiment groups cover **280 successful prompt-level records**: 10 prompts in each of four sampling groups (128, 256, 512, 1024) for each run. Existing results are preserved as recorded.

## Included runs

- Prima.cpp: Qwen3-14B, Qwen3-32B, and **Mistral Small 3.2 24B (Q6_K)**
- Jupiter: Qwen3-14B and Qwen3-32B
- Galaxy: Qwen3-14B
- Our fixed region-ring: Qwen3-14B (`G9,P1,G9,P1,G9,P1,G10`)

## Files

- `config.json`: model, hardware and run configuration metadata.
- `results.csv`: 280 normalized prefill records.
- `group-summary.csv`: 28 group means and medians for latency and throughput.
- `archive-manifest.json`: filenames, original ZIP byte sizes and SHA-256 hashes.
- `runs/ours-14b-g9p1/run-config.json`: fixed region-ring configuration.
- `runs/ours-14b-g9p1/per-prompt-details.csv`: 40 prompt-level measurements and metadata.
- `runs/prima-24b/run-config.json`: Prima 24B replay configuration, replacing original absolute paths and network addresses with environment placeholders.
- `runs/prima-24b/per-prompt-details.csv`: Prima 24B prefill, decode and planner-output details for all 40 prompts.

The seven **original ZIP archives are not included in this Git branch**. The available repository connector supports submitting UTF-8 files but cannot upload local binary ZIP files directly. Use SHA-256 values in `archive-manifest.json` to verify locally retained source archives.

Run configurations are not identical: Prima uses HiGHS, Jupiter uses Sequence DP, Galaxy uses RANK_LOCAL, and our 14B system uses a fixed ring. The 24B Prima measurements use Mistral Small 3.2 Q6_K rather than Qwen3. Reported group values are unweighted means/medians, not pooled throughput. Sampling group labels are not exact tokenizer token counts.
