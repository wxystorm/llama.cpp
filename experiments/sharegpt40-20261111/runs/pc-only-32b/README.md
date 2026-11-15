# PC-only Qwen3-32B reference observations

Two **earlier partial reference observations** supplied on 2026-11-13, plus a third 2048-group lowest-TPS observation from 2026-11-15. These are **not** a complete ShareGPT40 run and are kept out of the root `results.csv` and `group-summary.csv`.

| Prompt group | Prompt name | Prefill tok/s | Prefill ms | Prompt tokens | Source |
| --- | --- | ---: | ---: | ---: | --- |
| 256 | Not specified | 21.58 | 12325.37 | 266 | `llama-server` prompt-eval log |
| 512 | `prompt_512_01.txt` | 32.9 | Not provided | Not provided | User-reported measurement |
| 2048 | `prompt_2048_02.txt` | **17.61** | **122146.17** | **2151** | Selected minimum from the uploaded 32B log |

The 256-group log also gives decode time 4872.29 ms (logged 3 tokens, 0.41 tok/s), total time 17197.66 ms (269 tokens), 2 graph reuses and 0 graph reallocations. The 512 prompt's `32.9 tok/s` is a distinct measurement and must not be attributed to the 256-group log.

Neither the exact per-prompt filename for the 256 group, nor the quantization/version, runtime flags, batch and context for this PC-only run have been confirmed. Use these numbers as preliminary reference points and rerun with matching model, prompt and settings before controlled speedup claims.

The 2048 entry is the **lowest of three completed 32B runs** in source `粘贴的文本 (1)(20261115-141500).txt` (SHA-256 `c26c795fd4263aa3048f00b6dffb722194d6fa45db72b231db91d21206145ca5`). It was run with `-ngl all -t 8 -c 3000 -b 128 -ub 128 -n 3` using build `b11897-fd8f8451a`; decode was 3 tokens / 11,215.84 ms / 0.18 tok/s, total time 133,362.01 ms. The other 2048 results are **not** added because the user asked to upload the minimum. See `2048-min-run-config.json`.
