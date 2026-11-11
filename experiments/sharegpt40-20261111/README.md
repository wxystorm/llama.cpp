# ShareGPT40 benchmark snapshot (2026-11-11)

Five uploaded benchmark result archives were used to prepare this snapshot. The five runs cover 200 successful prompt-level records, with 10 prompts per group (128, 256, 512, 1024).

Files:
- `config.json`: model, hardware and recorded execution settings.
- `results.csv`: individual prefill measurements from the original summary CSV files (200 rows).
- `group-summary.csv`: mean and median prefill latency and throughput (20 groups).
- `archive-manifest.json`: SHA-256 and original byte size of each source archive.

This branch records the derived, numeric experiment results. The five original ZIP archives (including verbose logs, per-case planner files and other artifacts) are **not** uploaded to this branch: the available repository connection does not support uploading the local binary ZIP files. Use `archive-manifest.json` to verify archives kept separately.

The group label is a sampling bucket rather than the exact prompt token count. Measurements are the recorded per-run values, and group aggregates are arithmetic means and medians; no pooled TPS is calculated. Runtime parameters differ across Prima (HiGHS), Jupiter (Sequence DP / graph prebuild), and Galaxy (RANK_LOCAL); compare with awareness of distinct setups. The source archive also contains additional per-run details not represented in this concise table.
