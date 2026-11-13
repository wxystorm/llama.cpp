# 32B Bank / Cycle ablation (experimental)

This experiment targets **Qwen3-32B-Q4_K_M.gguf (64 decoder layers)**, PC CUDA
and an Android Phone-owned HTP RPC server configured with the **same proven
four-session HTP-only startup** as `run_32b_profile5_plan.ps1`.

## Four controlled modes

| Mode | Physical GPU Banks | Mover | Planning | Layout |
|---|---:|---|---|---|
| A0 | 2 | enabled | forced | `G8,P6,G8,P6,G8,P6,G8,P6,G8` |
| A1 | 2 | disabled | forced (same as A0) | `G8,P6,G8,P6,G8,P6,G8,P6,G8` |
| A2 | 1 | disabled | profiled coarse, memory guarded | 2–5 G/[C]/P Cycles + G1 tail |
| A3 | 1 | disabled | memory-greedy, memory guarded | one G/C/P pass |

A2 scores a small, width-balanced Cycle family from an independent G/C/N
profile. It uses a **conservative demand-H2D estimate**, not the normal
two-Bank overlapping DAG prediction. It is an ablation heuristic, **not**
the production planner or a guarantee of globally optimal placement.

A3 allocates by **compute priority GPU > HTP (NPU) > CPU**, subject to the
verified per-device budgets. Its **physical layer order** is GPU, CPU, Phone.
At least one CPU layer is retained to satisfy the present Phone-owned
autonomous prefill requirement of 3+ stages. If large contiguous Region
performance cannot be estimated, the memory-safe greedy placement may still
run and its predicted time is reported as **unscored**.

A2 and A3 verify:
- available PC RAM including host-canonical GPU weights;
- physical GPU memory: *largest one-Bank GPU Region* + GPU KV/scratch;
- Phone shared RAM and each HTP Session's mapping/scratch budget.

If no feasible layout exists, the program reports NO_FEASIBLE_PLAN instead of
silently exceeding memory. This does not guarantee a runtime PASS: actual CUDA
allocations and Phone HTP final-head behaviour must still be tested.

## Running

On PC PowerShell, after checking out the feature branch and rebuilding
`llama-completion` and the Region Ring tests:

```powershell
.\tools\region-ring-experiments\run_32b_ablation.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -Out ".\32b-ablation-quick"
```

By default, only **prompt_128_01.txt** and **prompt_512_01.txt** are
run, once per mode. The suite uses `--no-mmap`, XG/XP/XC=128, 2048 batch,
2048 ubatch, `n_predict=1`, no forced full-40 sweep. It runs A0 first,
then reuses its **single independent G/C/N profile** in A1/A2/A3.

Examples:

```powershell
# Preview the commands without running any model:
.\tools\region-ring-experiments\run_32b_ablation.ps1 -DryRun

# Isolate two-Bank prefetch, fixed plan only:
.\tools\region-ring-experiments\run_32b_ablation.ps1 -Mode A0,A1

# More samples / repeat experiments:
.\tools\region-ring-experiments\run_32b_ablation.ps1 `
  -Prompt prompt_256_01.txt,prompt_1024_01.txt `
  -Repeat 3

# Reuse a previously verified independent C/G/N profile:
.\tools\region-ring-experiments\run_32b_ablation.ps1 `
  -ProfileCsv ".\qwen32b-profile\independent-profile\profile.csv"
```

**Do not reuse a Profile from another model** (e.g., Mistral-24B). Choose
a Qwen3-32B Profile captured on the same PC/Phone combination.

Results:
- `<Out>/A0-r01/summary.csv`, ..., `A3-r01/summary.csv` for individual
  runs with immutable run config and per-Prompt attempts.
- `<Out>/ablation_summary.csv` combined PASS rows.
- Per-Prompt `metrics.jsonl` includes `ablation_mode`,
  `gpu_bank_count`, `gpu_bank_streaming`, `bank_refills`,
  `bank_h2d_ms`, `bank_gpu_wait_ms`, `wall_ms`, and the selected layout.
  The driver fails if the engine ignores its mode or re-enables the Mover.

The server's HTP-only Plan-controlled mode, Phone graphless execution and
four HTP Sessions must be configured as in the existing successful 32B run.
Only PC code was changed for these ablation modes. **This branch has not
yet been verified on the actual PC/Phone hardware.**

## Interpreting the numbers

A0 vs A1 isolates **prefetch under a fixed two-Bank layout**. A1 vs A2
changes the physical Bank count *and* permits replanning: the difference is a
system-level capacity/replanning effect, not a pure Bank-count speedup. A2
vs A3 changes Cycle topology and device allocation while keeping single-Bank
demand loading.

Compare Prefill latency/TPS on the same prompts and full run configs,
report PASS rate and selected layer counts, peak VRAM, GPU H2D / Bank Wait,
HTP execution and Phone communication. For publication use 3+ independent
trials and medians, not a single warm-up run.
