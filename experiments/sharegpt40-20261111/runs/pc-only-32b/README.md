# PC-only Qwen3-32B reference observations

Two **partial reference observations** supplied on 2026-11-13. These are **not** a complete 40-prompt ShareGPT40 run and are kept out of the root `results.csv` and `group-summary.csv`.

| Prompt group | Prompt name | Prefill tok/s | Prefill ms | Prompt tokens | Source |
| --- | --- | ---: | ---: | ---: | --- |
| 256 | Not specified | 21.58 | 12325.37 | 266 | `llama-server` prompt-eval log |
| 512 | `prompt_512_01.txt` | 32.9 | Not provided | Not provided | User-reported measurement |

The 256-group log also gives decode time 4872.29 ms (logged 3 tokens, 0.41 tok/s), total time 17197.66 ms (269 tokens), 2 graph reuses and 0 graph reallocations. The 512 prompt's `32.9 tok/s` is a distinct measurement and must not be attributed to the 256-group log.

Neither the exact per-prompt filename for the 256 group, nor the quantization/version, runtime flags, batch and context for this PC-only run have been confirmed. Use these numbers as preliminary reference points and rerun with matching model, prompt and settings before controlled speedup claims.
