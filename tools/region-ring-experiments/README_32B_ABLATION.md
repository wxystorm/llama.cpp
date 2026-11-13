# Qwen3-32B bank / prefetch / cycle ablation

Run `run_32b_ablation.py` from the repository root. Each case runs **one** ShareGPT40 prompt by default (`--limit 1`), not the whole 40-prompt set.

| Case | GPU banks | Prefetch | Planning |
|---|---:|---|---|
| A0 | 2 | ON | Fixed `G8,P6,G8,P6,G8,P6,G8,P6,G8` |
| A1 | 2 | OFF | Exactly the same layout as A0 |
| A2 | 1 | OFF | Coarse G/C/N profile-guided 2-5 Cycle search within estimated memory |
| A3 | 1 | OFF | Single `G,[C],P` cycle; prioritize GPU, then CPU, Phone gets the remainder |

## Run

First obtain current Android available memory: `adb shell cat /proc/meminfo`. Convert `MemAvailable` from kB to MiB, leaving a safety margin, and supply `--phone-budget-mib`. There is intentionally no fabricated default Android memory budget.

```powershell
python .\tools\region-ring-experiments\run_32b_ablation.py `
  --model 'E:\llama\models\Qwen3-32B-Q4_K_M.gguf' `
  --rpc '10.39.52.114:50052' `
  --out '.\32b-ablation' `
  --cases A0 A1 A2 A3 `
  --phone-budget-mib 7800 `
  --limit 1
```

Replace 7800 with a budget based on actual Android MemAvailable. You can set `--only-prompt prompt_512_01.txt`, or use `--limit 0` for all 40 after the smoke tests pass.

The first A0 run produces an independent G/C/N profile once; A1, A2 and A3 reuse it. To run A2/A3 alone, provide `--profile-csv` pointing to a previously measured independent profile, e.g.:

```powershell
python .\tools\region-ring-experiments\run_32b_ablation.py `
  --cases A2 A3 `
  --profile-csv '.\32b-ablation\A0\independent-profile\profile.csv' `
  --phone-budget-mib 7800 `
  --limit 1
```

Use `--plan-only` to inspect layouts and memory estimates without running prompts (it still needs a valid independent profile for A2). The output contains `A0/`, `A1/`, `A2/`, `A3/` with separate normal runner logs, plus `A2-plan.json`, `A3-plan.json`, `plans.json` and `ablation-comparison.csv` (per-prompt joined metrics).

## Interpretation and limitations

- A0 and A1 preload the **same first two GPU Bank weights**; A1 disables only subsequent background refills, leaving a more comparable H2D baseline.
- A2 uses a physical **single GPU Bank**, keeps weight streaming on, and validates the peak GPU Region weights plus KV against native available VRAM.
- A2's search is a **coarse independent-profile-guided heuristic**, not the full C++ DAG planner. The C++ forced-plan path still calculates its own estimate under the existing two-bank simulation assumptions, so avoid presenting its predicted latency as a validated single-bank simulation.
- A3 is a **capacity-greedy** one-Cycle baseline. It does not optimize G/C/N speed. Both A2 and A3 change layer ownership relative to A0/A1.
- Python GGUF offset accounting and CPU/HTP memory estimates are conservative approximations. The native forced path has a single GPU Bank guard but still skips the full CPU/HTP memory feasibility check; only a successful model execution establishes feasibility.
- Check the runtime `[REGION_RING_ABLATION]` log and metrics before accepting timing data. For each case record exact layout, Prefill TPS/ms, GPU H2D and wait, Phone HTP time, CPU time, peak VRAM and model token count.
- Run repeated identical prompts and compare medians before drawing paper claims. A0 vs A1 isolates background prefetch best; A1 vs A2 also changes the layout/Bank memory budget, and A2 vs A3 is a system-level Cycle/capacity comparison.
