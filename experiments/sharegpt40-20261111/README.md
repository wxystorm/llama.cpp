# ShareGPT benchmark snapshot

Ten experimental runs; **430 successful per-prompt results** in `results.csv` and **64 group summary rows** in `group-summary.csv`. Prompt groups are sampling labels, not exact model token counts.

## Run inventory

| Method | Model | ID | Observations |
| --- | --- | --- | ---: |
| Prima.cpp | Qwen3-14B | `prima-14b` | 40 |
| Prima.cpp | Mistral Small 3.2-24B Q6_K | `prima-24b` | 50 |
| Prima.cpp | Qwen3-32B | `prima-32b` | 50 |
| Jupiter | Qwen3-14B | `jupiter-14b` | 40 |
| Jupiter | Mistral Small 3.2-24B Q6_K | `jupiter-24b` | 50 (aggregate-only, 10/group) |
| Jupiter | Qwen3-32B | `jupiter-32b` | 50 |
| Galaxy | Qwen3-14B | `galaxy-14b` | 50 |
| Ours | Qwen3-14B | `ours-14b-server-plan` | 50 |
| Ours | Mistral Small 3.2-24B Q6_K | `ours-24b-server-plan` | 50 |
| Ours | Qwen3-32B | `ours-32b-server-plan` | 50 |

The three Ours 14B/24B/32B runs now use user-supplied 50-prompt `SERVER_PROBE` results. Each has 10 distinct prompts per group **0/128/256/512/1024**; `prompt_1024_10.txt` was run first as a long-prompt anchor, so rows are matched by filename, not output order. Earlier Ours prompt measurements remain in Git history but are excluded from the active results.

Each new run's `runs/ours-*-server-plan/per-prompt-details.csv` keeps input tokens, prefill latency and TPS, cache=0, planned XC/XG/XP, actual XG/XP and planner latency. The physical layer placements were supplied separately by the user after the SERVER_PROBE logs: **24B** `G5,P3,G5,P3,G6,P3,G5,P4,G6` (40 layers) and **32B** `G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7` (64 layers). These layouts are recorded in the run configurations and in every per-prompt detail row; the underlying probe TPS, chunk sizes and planner times are unchanged. The 32B full50 console excerpt does not repeat the launch command or binary hash; the run configuration records the preceding experiment setup separately.

Ours 14B has been fully replaced by the new 50-prompt Server Plan measurements for groups 0/128/256/512/1024, with the user-confirmed placement `G11,P5,G11,P5,G8` (40 layers). Its earlier 2048 group and old mixed per-prompt maxima are no longer part of the current summary. They remain available in Git history.

Galaxy 14B covers groups 0/128/256/512/1024 with 10 prompts per group (group 0 mean: 5.202 tok/s). Prima.cpp 24B and 32B each contain 50 observations (context 2048). Jupiter 24B was **remeasured due to a 24B prefix issue**. The user-reported replacement group-average Prefill throughput (tok/s), each based on 10 prompts, is: **0: 22.551; 128: 23.138; 256: 25.252; 512: 28.374; 1024: 30.521**. The new 50 per-prompt measurements were not provided, so this replacement is **aggregate-only**. The former 36 Jupiter 24B per-prompt rows are excluded from active `results.csv`; the old `runs/jupiter-24b/per-prompt-details.csv` and run configuration remain as **superseded historical evidence**, not valid inputs for the new means.

`group-summary.csv` reports the arithmetic mean of per-prompt Prefill TPS; a group mean is not pooled token throughput. **Exception:** Jupiter 24B replacement means and sample counts were supplied directly by the user, not recalculated from current per-prompt rows. Its median TPS and latency fields remain blank because those metrics were not supplied. Planner time is retained in per-prompt details, not added to prefill time.


## 32B refresh (2026-11-27)

- Ours: 50 resident-server prompts, fixed G43/P21, session scratch reuse and locked-XC candidate deduplication (experiment setup from the preceding conversation). The active run replaces the earlier 32B server-plan measurements.
- Jupiter: 2026-11-26 L52 offline-DP run, CPU36/GPU16/HTP12, four sessions, resident autonomous mode, minimum chunk parameter b=128. All 50 prompts passed, replacing the previous 40-prompt baseline.
- [32B comparison CSV](32b-prefill-comparison-20261127.csv) contains untrimmed arithmetic mean, minimum-TPS-excluded mean (9 of 10 per group), trimmed speedups, Planner means and pooled token throughput. The ALL trimmed row averages the five group trimmed means; the ALL weighted columns use all 50 untrimmed observations. The main group-summary.csv retains untrimmed arithmetic means.
- Across all 50 prompts, previous/current Ours and current Jupiter pooled prefill TPS are 123.037 / 131.796 / 72.234. Ours total prefill latency changes from 222.835 s to 208.027 s; mean Planner latency changes from 362.539 ms to 64.722 ms.
- Raw results are matched by prompt filename, not exact token count: Jupiter has 49 prompts with 8 extra tokens and one with 13 extra tokens. Neither raw dataset is trimmed or rescaled. The source logs are chat excerpts, not verified binary/ZIP archives. Historical 32B results remain accessible at commit 91fe6a6c65bad073749ad1ab75a1fd6bc1380501.

## llama.cpp single-PC ShareGPT50 baseline

| Group | PC-only 14B | PC-only 24B | PC-only 32B |
| ---: | ---: | ---: | ---: |
| 0 | 57.6334 | 20.8682 | 21.4223 |
| 128 | 69.8149 | 21.7289 | 24.6974 |
| 256 | 71.3049 | 24.2536 | 25.8188 |
| 512 | 69.4976 | 31.5654 | 29.8023 |
| 1024 | 66.6422 | 32.0298 | 29.0208 |

Command: `python .\\tools\\sharegpt50\\bench_single_pc_sharegpt50.py --mode server --ngl auto --per-group 10 --batch 2048 --ubatch 512 --ctx 3000`.

`config.json` contains active run references, `archive-manifest.json` references available earlier ZIPs for runs whose measurements still derive from them, and `runs/llama-cpp-pc-only-sharegpt50/` contains the current single-PC baseline. The three Ours replacement SERVER_PROBE datasets and the refreshed Jupiter 32B data came from chat logs rather than verified ZIP archives; the earlier Jupiter 32B archive is marked superseded.
