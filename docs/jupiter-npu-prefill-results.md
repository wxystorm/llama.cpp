# Jupiter Prefill results (Phone NPU)

ShareGPT50, Native Autonomous, offline DP, Context/Batch 2048, 4 HTP sessions.

| Model | CPU layers | CUDA layers | NPU layers | Completed |
|---|---:|---:|---:|---:|
| Mistral Small 3.2 24B Q6_K | 25 | 7 | 8 | 50/50 PASS |
| Qwen3-32B Q4_K_M | 36 | 16 | 12 | 50/50 PASS |

Mean Prefill throughput (tok/s): discard the lowest throughput in each group of 10 and average the remaining 9; overall mean over the retained 45 prompts.

| Prompt group | 24B | 32B |
|---|---:|---:|
| 0-128 | 45.486 | 43.680 |
| 128 | 75.325 | 64.062 |
| 256 | 80.476 | 68.217 |
| 512 | 104.603 | 77.090 |
| 1024 | 111.395 | 84.463 |
| **Overall** | **83.457** | **67.502** |
