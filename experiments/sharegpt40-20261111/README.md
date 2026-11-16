# ShareGPT benchmark snapshot

Ten experimental runs; **426 successful per-prompt results** in `results.csv` and **43 group summaries** in `group-summary.csv`. Prompt groups are sampling labels, not exact model token counts.

## Run inventory

| Method | Model | ID | Observations |
| --- | --- | --- | ---: |
| Prima.cpp | Qwen3-14B | `prima-14b` | 40 |
| Prima.cpp | Mistral Small 3.2-24B Q6_K | `prima-24b` | 40 |
| Prima.cpp | Qwen3-32B | `prima-32b` | 40 |
| Jupiter | Qwen3-14B | `jupiter-14b` | 40 |
| Jupiter | Mistral Small 3.2-24B Q6_K | `jupiter-24b` | 36 |
| Jupiter | Qwen3-32B | `jupiter-32b` | 40 |
| Galaxy | Qwen3-14B | `galaxy-14b` | 40 |
| Ours | Qwen3-14B | `ours-14b-autoplan` | 50 |
| Ours | Mistral Small 3.2-24B Q6_K | `ours-24b-server-plan` | 50 |
| Ours | Qwen3-32B | `ours-32b-server-plan` | 50 |

The replacement **Ours-32B** and **Ours-24B** results come from the two user-supplied `SERVER_PROBE` blocks (first 32B, second 24B). Each has 10 distinct prompts per group **0/128/256/512/1024**; `prompt_1024_10.txt` was run first as a long-prompt anchor, so rows are matched by filename, not output order. No old Ours 24B or 32B prompt measurements are retained.

Each new run's `runs/ours-*-server-plan/per-prompt-details.csv` keeps input tokens, prefill latency and TPS, cache=0, planned XC/XG/XP, actual XG/XP and planner latency. The physical layer placements were supplied separately by the user after the SERVER_PROBE logs: **24B** `G5,P3,G5,P3,G6,P3,G5,P4,G6` (40 layers) and **32B** `G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7` (64 layers). These layouts are recorded in the run configurations and in every per-prompt detail row; the underlying probe TPS, chunk sizes and planner times are unchanged. The explicit server launch command was provided only for the 24B block.

Ours 14B is unchanged: 50 observations over groups 128/256/512/1024/2048; best of two past results at the first four groups, with the 2048 group from a later user-supplied log. Its recorded `G10,P9,G12,P1,G8` layout is normalized metadata for some historical winners, **not** a claim that they were remeasured under this placement.

The six other complete runs each contain 40 observations over groups 128/256/512/1024. Jupiter 24B has 36 observations (group 1024: six), and its missing four are not imputed.

`group-summary.csv` computes ordinary arithmetic mean/median of recorded per-prompt Prefill TPS and latency; a group mean is not pooled token throughput. Planner time is retained in per-prompt details, not added to prefill time.

`config.json` contains active run references, `archive-manifest.json` references available earlier ZIPs for runs whose measurements still derive from them, and `runs/pc-only-14b` and `runs/pc-only-32b` preserve four separate incomplete single-PC reference observations. The two replacement SERVER_PROBE blocks came from chat logs, not uploaded ZIP archives, so they have no archive SHA record.
