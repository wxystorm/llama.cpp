# Qwen3-32B runtime ablation batch runner (50 ShareGPT prompts)

This runner is checked into `wxystorm/llama.cpp` on branch `feat/32b-runtime-ablation-singlepass-20261116`, based on `feat/region-ring-canonical-decode-reserve-20261116`. The C++ adjacent-GPU topology exception is already in this branch. No manual patch application is needed.

## Experiment design

| Case | Layout | GPU Banks | Background prefetch | Drain-first | Chunk choice |
|---|---|---|---|---|---|
| `full` | G8,P6,G8,P6,G8,P6,G8,P6,G8 | 2 A/B | ON | ON | Native locked-layout chunk-only |
| `no_drain` | Same | 2 A/B | ON | OFF | Native locked-layout chunk-only |
| `no_prefetch` | Same | 2 A/B | OFF | ON | Native locked-layout chunk-only |
| `no_multicycle` | G8,G8,G8,G8,P24,G8 | 2 A/B | ON | ON | Native locked-layout chunk-only |

The last layout has exactly two PC/Phone boundaries, five independent eight-layer GPU regions, and Bank order A, B, A, B, A. Both layouts assign 40 decoder layers to PC CUDA and 24 to Phone HTP. For each request the native C++ Planner selects XC/XG/XP independently but does not change the layout. This is **not** the forced-layout fixed-128 path.

## Checkout experimental branch and rebuild PC

From PowerShell in the existing repository:

```powershell
cd 'E:\llama\llama-tensor-v2'
git fetch origin
git switch feat/32b-runtime-ablation-singlepass-20261116
cmake --build .\build-tensor-v2 --config Release --target llama-completion -j 8
```

The opt-in `LLAMA_REGION_RING_ALLOW_ADJACENT_GPU=1` is set only for `no_multicycle`. The compiler rebuild is mandatory. Do **not** replace `G8,G8,G8,G8,P24,G8` with `G32,P24,G8`, as this would destroy the five GPU segments and A/B Bank order.

## Run one smoke prompt

```powershell
$root = 'E:\llama\llama-tensor-v2'
python "$root\tools\region-ring-experiments\run_32b_runtime_ablations.py" `
  --exe "$root\build-tensor-v2\bin\Release\llama-completion.exe" `
  --model 'E:\llama\models\Qwen3-32B-Q4_K_M.gguf' `
  --rpc '10.238.229.239:50052' `
  --prompt-dir 'E:\llama\datasets\sharegpt\sharegpt40' `
  --out "$root\32b-runtime-ablation-50" `
  --cases no_multicycle `
  --groups 128 256 512 1024 2048 `
  --batch 4096 --ubatch 4096 --threads 8 `
  --min-context 4096 --max-context 8192 `
  --smoke
```

## Run all 50 prompts

Remove `--smoke` from exactly the same command and rerun. The parent runner resumes successful prompts without rerunning them. A case writes `summary.csv`, `run-config.json`, `independent-profile/`, and individual `runs/prompt_*/attempt-*/result.json` files. The top-level `runtime-ablation-comparison.csv` joins all completed cases.

For all three Runtime ablations plus the matching fixed-layout Full baseline, use `--cases all` instead of `--cases no_multicycle`. This is **4 cases x 50 inputs = 200 input/case measurements**, not six Planner ablations. Each case has a separate output directory. If the first case produces a valid native profile, the next cases reuse an identical copy of the profile; no fake samples are generated.

To reuse an independent hardware profile acquired earlier, add:

```powershell
--profile-from 'E:\llama\llama-tensor-v2\previous-run\independent-profile\profile.csv'
```

If group 2048 is included, the profile must include the native long HTP KV anchors (2048 and 3072). The script will refuse an incomplete profile.

## Limitations and validation

- This branch only implements the **three Runtime ablations** and optional `full` case. Planner ablations (no dynamic chunk, no queueing simulation, and memory-proportional static placement) need separate planning implementation and are **not** silently simulated by environment flags.
- The matching Full comparator here uses the explicit `G8,P6` x4 + GPU tail layout, not the independent *automatic* Full Cycle layout if the main experiment chose something else.
- `no_multicycle` exposes a single 24-layer Phone Region. Four HTP sessions may stripe its layers, but native HTP session/shared memory guards or Phone graph registration can still reject it on a real device. Do not silently move layers back to PC or relabel an OOM as a performance number.
- The script checks the *actual* applied Planner layout and `locked_chunk_only` source in the output, and requires the double-Bank mover for the enabled-prefetch cases. A non-matching run is flagged as an audit failure.
- The code was syntax/unit tested with a mock runner. The physical 32B GGUF, Windows CUDA build, and Android HTP RPC server are not available in this environment; real execution has not been validated here.