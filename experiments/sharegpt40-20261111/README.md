# ShareGPT40 benchmark snapshot (2026-11-11)

Six experimental runs cover **240 successful prompt-level records**, with 10 prompts in each of four groups (128, 256, 512, 1024). The earlier five runs (200 observations) remain unchanged; the sixth is the fixed-layout 14B region-ring run (`ours-14b-g9p1`).

Files:
- `config.json`: models, hardware and run configurations, including the added run.
- `results.csv`: normalized prefill measurements, one row per prompt (240 rows).
- `group-summary.csv`: arithmetic mean and median throughput and latency (24 groups).
- `archive-manifest.json`: names, SHA-256 hashes and sizes of the six original uploaded ZIP archives.
- `runs/ours-14b-g9p1/run-config.json`: reproducible runtime settings with local executable/model paths and RPC endpoint replaced by placeholders.
- `runs/ours-14b-g9p1/per-prompt-details.csv`: 40 measured prompt records, including context, planned tokens, source runtime commit and fixed layout.

The original six ZIP archives contain verbose logs and additional per-prompt artifacts, but **are not stored in this Git branch**. The available GitHub connection supports UTF-8 repository files but cannot directly upload the local binary ZIP. Verify an archive independently using its SHA-256 entry in `archive-manifest.json`.

Measurements in `results.csv` are from experiment summaries; derived statistics are unweighted means and medians (not pooled throughput). Prompt groups are sampling buckets rather than exact prompt token counts. Runtime setups differ between Prima (HiGHS), Jupiter (Sequence DP), Galaxy (RANK_LOCAL) and the fixed-layout region-ring run, so compare results with their configurations in mind.
