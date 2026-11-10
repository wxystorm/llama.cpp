# Qwen3-32B ShareGPT40: independent resumable sweep

This is a new runner; it does **not** import or call any previous 32B sweep script. It invokes the existing local `llama-completion` executable directly (not the server-client `llama-cli`). Python 3.9+ and a running phone RPC server are required. By default weights stay memory-mapped, which is safer for a large 32B GGUF; `-NoMmap` is optional if host RAM is sufficient. The benchmark engine must support Region Ring JSONL metrics.

## Windows PowerShell

From the repository root (`E:\llama\llama-tensor-v2` in this example):

```powershell
.\tools\region-ring-experiments\run_32b_all40.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\32b-g8p6-x128-n1-all40" `
  -Batch 2048 -UBatch 2048 -NPredict 1 -Threads 8
```

The script discovers **exactly 40 files**: `prompt_128_01.txt` through `prompt_1024_10.txt`, with 10 files each in the 128/256/512/1024 groups. Missing files are a preflight error. `manifest.csv` provides per-file token estimates when possible; absent token counts use conservative bin ceilings (255, 511, 1023, 2000). `-c` is computed separately for each prompt as `ceil256(max(256, token_estimate + 1 generated token + 64 padding))` (minimum 256; default padding 64). For example, 167 input tokens receive `-c 256`, and 1855 input tokens receive `-c 2048`. The engine uses `-n 1` by default, so this is a prefill-focused run. Runtime-reported prefill token counts, not manifest estimates, are recorded in results. The default maximum context is 4096; override `-MaxContext` for unusually large prompts.

**Default fixed mode (no Planner/Profile):** `G8,P6,G8,P6,G8,P6,G8,P6,G8`, with `XC=XG=XP=128` and GPU weight streaming on. Native `apply-wide` is used to accept the forced layout, but its search/Profile are skipped.

## Custom layer layout and Chunk widths

```powershell
.\tools\region-ring-experiments\run_32b_all40.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Layout "G7,P6,G8,P6,G8,P6,G8,P6,G9" `
  -Chunk 128 -XG 64 -XP 96 `
  -Out ".\32b-layout-alt-xg64-xp96"
```

`-Layout` specifies all 64 model layers, beginning with G. Supported patterns are up to 8 cycles of `G -> [C] -> [P]`, optionally ending in one GPU tail. `C0` is allowed and canonicalized away. Invalid total layers or topology fail before launching the model. A valid topology does not guarantee that the model will fit in device memory.

`-Chunk` gives the uniform default width for XC/XG/XP; `-XC`, `-XG`, `-XP` override individual widths. Example: `-Chunk 128 -XG 64 -XP 96` means `XC=128,XG=64,XP=96`. Only widths supported by the native C++ forced Planner are accepted: `8,12,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192`.

`-Out` is optional: the existing default configuration keeps `32b-g8p6-x128-n1-all40`, while custom layout/Chunk settings automatically get a deterministic different output directory. When you specify `-Out` explicitly, use a separate directory for a different layout/Chunk; otherwise the existing run signature rejects mixing results. The optional `-AutoPlan` mode is preserved but cannot be combined with manual layout/Chunk options.

## A/B experiment: XG=96 and short-prefill GPU Bank Mover

Previously the CUDA Bank Mover started only when `tokens >= 2*XG`.
With `XG=96` that means 192 tokens, so the 167-182 token prompts
would still run in the slower `INACTIVE_SMALL_BATCH` / demand path.

The new **opt-in** `-ForceSmallBatchMover` flag sets
`LLAMA_REGION_RING_GPU_BANK_FORCE_SMALL_BATCH=1` in the PC engine,
waiving the two-full-Chunk gate for **prefill batches with more than one token**.
One-token decode and layouts with fewer than two GPU Regions retain the
legacy path. It does not change layer allocation, Chunk sizes, or Planner mode.
The native Bank Mover still has to finish each prior same-bank Region before
it can refill the next one. On very short prompts, asynchronous refill may
be slower due to initial Bank priming or contention; benchmark both modes.

**Rebuild the Windows PC executable first** (the RPC server need not change
for this PC-side scheduler option):

```powershell
cmake --build .\build-tensor-v2 --config Release --target llama-completion test-region-ring --parallel 4
ctest --test-dir .\build-tensor-v2 -C Release -R "^test-region-ring$" --output-on-failure
```

Try XG=96 with the threshold unchanged, then repeat with the override:

```powershell
$settings = @{
  Model     = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf"
  Rpc       = "10.39.52.114:50052"
  PromptDir = "E:\llama\datasets\sharegpt\sharegpt40"
  Layout    = "G8,P6,G8,P6,G8,P6,G8,P6,G8"
  Chunk     = 128
  XG        = 96
}

# Control: default XG96 threshold, no short-Batch mover.
.\tools\region-ring-experiments\run_32b_all40.ps1 @settings `
  -OnlyGroup 128 -Out ".\32b-xg96-mover-off"

# Experiment: allow short Prefill batches to use async GPU Bank refill.
.\tools\region-ring-experiments\run_32b_all40.ps1 @settings `
  -OnlyGroup 128 -Out ".\32b-xg96-mover-on" -ForceSmallBatchMover
```

`-ForceSmallBatchMover` requires fixed mode, streaming on, and at least
two GPU Regions. The flag is included in the run signature. Reuse the
**same** parameters and output path to resume; use `-OnlyPrompt` plus
`-RerunAll` to force rerun one previously successful prompt.

Verification: the `[REGION_RING_GPU_BANK] phase=INIT` log reports
`mover=CPU_SINGLE` with `override_applied=1` on short prompts.
The `region_ring` JSONL contains `gpu_bank_force_small_batch`,
`gpu_bank_small_batch_override_applied`, and `gpu_bank_streaming`; the
`summary.csv` also includes `bank_refills`, `bank_h2d_ms`,
`bank_gpu_wait_ms`, `bank_mover_wall_ms`, and `gpu_weight_stage_ms`.
`bank_refills>0` confirms that an actual refill completed.
The runner refuses to report PASS (`MOVER_UNVERIFIED` or `MOVER_INACTIVE`)
if the opt-in flag was ignored by an old build or the requested mover
did not activate.

**Tiny tail:** `LLAMA_REGION_RING_COALESCE_TINY_TAIL=1` invokes the native width adjustment on the actual UBatch token count, separately for each requested stage width. With X=128 a remainder of 1..32 may be coalesced by widening the uniform width slightly (at most 160). For example 513 tokens become `[129,129,129,126]` rather than `[128,128,128,128,1]`. The layout does not change. Check `[REGION_RING_TAIL_COALESCE]` in engine logs for the actual width. `summary.csv` records both the selected/forced `plan_xc/xg/xp` and `runtime_xc/xg/xp` reported by the engine (when available).

## Restart, retry, rerun

**Changed defaults:** the one-token, compact-context benchmark uses a new output directory (`32b-g8p6-x128-n1-all40`), separate from prior 32-token results. Existing output configured for `-NPredict 32` / `-MinContext 1024` / `-ContextPadding 256` cannot be resumed with these new settings because the signature checks prevent mixing incompatible experiments. Start a new `-Out` once, then **run the exact same command again with that same `-Out`** to resume. Successful (`PASS`) prompts are skipped; pending, interrupted and failed prompts are attempted. Existing attempts and logs are never overwritten. `-MaxRetries 1` (default) means up to two attempts per prompt in each invocation; `-RetryDelaySeconds 5` is the retry delay.

To **force rerun one previously successful Prompt** while retaining the same layout/Chunk and output directory:

```powershell
.\tools\region-ring-experiments\run_32b_all40.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Layout "G7,P6,G8,P6,G8,P6,G8,P6,G9" `
  -Chunk 128 -XG 64 -XP 96 `
  -Out ".\32b-layout-alt-xg64-xp96" `
  -OnlyPrompt "prompt_512_01.txt" -RerunAll
```

The rerun receives a new `attempt-XXXX` folder; older logs are retained. Without `-RerunAll`, a PASS prompt is skipped and a failed/interrupted prompt is retried.

Useful controls:

- `-DryRun` prints all planned commands without creating result files.
- `-Limit 2` runs at most two selected prompts for a smoke test. Omit it later to complete the same 40-prompt result directory.
- `-OnlyGroup 512` or `-OnlyPrompt prompt_512_01.txt` narrows one invocation, while still preserving the same 40-prompt configuration.
- `-RetryFailedOnly` skips never-run and successful prompts, targeting only failed/interrupted ones.
- `-RerunAll` reruns successful prompts too while retaining previous attempts.
- `-MaxRetries 0` disables automatic retries, useful while debugging.
- `-TimeoutSeconds 600` bounds an individual engine process, including model load. Default 0 means no timeout.
- `-StopOnFail` stops after a prompt exhausts its retries; otherwise the sweep continues to other prompts.
- `-EnvOverride "SOME_FLAG=1"` sets explicit engine environment variables and includes them in the run signature.

A run directory has a fixed configuration signature (`run-config.json`), including model and executable path/size/mtime, prompt hashes, fixed Ring settings and relevant inherited runtime environment variables. If any experiment-affecting setting changes, the runner **refuses to mix results**; use a **different `-Out`**. You can safely change only run controls (such as `-Limit`, `-MaxRetries`, `-RerunAll`, or `-RetryFailedOnly`) while resuming.

## Output layout

```text
32b-g8p6-x128-n1-all40/
  run-config.json
  summary.csv                  # 40 rows, written atomically after each attempt
  summary.json                 # counts by status
  .runner.lock                 # OS advisory lock (released even after crash)
  runs/
    prompt_512_01/
      attempt-0001/
        result.json            # RUNNING then final result
        metrics.jsonl
        output.stdout.txt
        engine.stderr.log
      attempt-0002/            # created only if retried / rerun
        ...
```

A run is **PASS** only with exit code 0, a native `perf` record with prefill data, completed `region_ring` records after the last `measurement_start`, and every reported runtime layout exactly matching the forced 64-layer layout. Otherwise it is marked `FAIL`, `NO_METRICS`, `NO_PERF`, `NO_RING`, `NO_PREFILL`, `INCOMPLETE_RING`, `WRONG_LAYOUT`, `MOVER_UNVERIFIED`, `MOVER_INACTIVE`, `TIMEOUT`, or `INTERRUPTED`. This avoids reporting false success when the process exits normally without real benchmark measurements.

The parent script updates `summary.csv` after each attempt. Stop with Ctrl+C and rerun the same command; an interrupted attempt stays in its own folder and will be retried. Do not modify the result files while a benchmark is running.

## Tests

```powershell
python -m unittest discover -s tools/region-ring-experiments -p "test_run_32b_all40.py" -v
```

The included tests simulate an engine without downloading model weights or contacting the phone. **Actual 32B inference must be tested on the PC/phone hardware.**
