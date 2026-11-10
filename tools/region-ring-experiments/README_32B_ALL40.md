# Qwen3-32B ShareGPT40: independent resumable sweep

This is a new runner; it does **not** import or call any previous 32B sweep script. It invokes the existing local `llama-completion` executable directly (not the server-client `llama-cli`). Python 3.9+ and a running phone RPC server are required. The benchmark engine must support Region Ring JSONL metrics.

## Windows PowerShell

From the repository root (`E:\llama\llama-tensor-v2` in this example):

```powershell
.\tools\region-ring-experiments\run_32b_all40.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\32b-g8p6-x128-all40" `
  -Batch 2048 -UBatch 2048 -NPredict 32 -Threads 8 `
  -NoMmap
```

The script discovers **exactly 40 files**: `prompt_128_01.txt` through `prompt_1024_10.txt`, with 10 files each in the 128/256/512/1024 groups. Missing files are a preflight error. `manifest.csv` provides per-file token estimates when possible; absent token counts use conservative bin ceilings (255, 511, 1023, 2000). `-c` is rounded to 256 tokens with padding and an initial minimum of 1024 tokens. Runtime-reported prefill token counts, not manifest estimates, are recorded in results. The default maximum context is 4096; override `-MaxContext` for unusually large prompts.

**Always fixed (no Planner/Profile):** `G8,P6,G8,P6,G8,P6,G8,P6,G8`, with `XC=XG=XP=128` and GPU weight streaming on by default. The native engine requires `LLAMA_HYBRID_REGION_RING_PLANNER=apply-wide` internally to accept a forced layout, but its search path reports `search=SKIPPED reason=FORCED_AB`; no automatic allocation or runtime profile is requested.

**Tiny tail only:** `LLAMA_REGION_RING_COALESCE_TINY_TAIL=1` invokes the engine's native `llama_region_ring_coalesced_chunk_width` on the **actual UBatch token count**, not the manifest estimate. If a remainder is between 1 and 32 tokens (<= 128/4), it slightly widens the common Chunk size to remove the tiny final Chunk, capped at 160; otherwise the requested width stays 128. When coalescing triggers, the UBatch uses a slightly widened uniform width. For example, 513 tokens become `[129,129,129,126]` instead of `[128,128,128,128,1]`. See `[REGION_RING_TAIL_COALESCE]` in the engine logs for actual runtime adjustments. No Python-side regrouping or GGUF padding is performed.

## Restart, retry, rerun

**Run the exact same command again with the exact same `-Out`** to resume. Successful (`PASS`) prompts are skipped; pending, interrupted and failed prompts are attempted. Existing attempts and logs are never overwritten. `-MaxRetries 1` (default) means up to two attempts per prompt in each invocation; `-RetryDelaySeconds 5` is the retry delay.

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
32b-g8p6-x128-all40/
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

A run is **PASS** only with exit code 0, a native `perf` record with prefill data, completed `region_ring` records after the last `measurement_start`, and every reported runtime layout exactly matching the forced 64-layer layout. Otherwise it is marked `FAIL`, `NO_METRICS`, `NO_PERF`, `NO_RING`, `NO_PREFILL`, `INCOMPLETE_RING`, `WRONG_LAYOUT`, `TIMEOUT`, or `INTERRUPTED`. This avoids reporting false success when the process exits normally without real benchmark measurements.

The parent script updates `summary.csv` after each attempt. Stop with Ctrl+C and rerun the same command; an interrupted attempt stays in its own folder and will be retried. Do not modify the result files while a benchmark is running.

## Tests

```powershell
python -m unittest discover -s tools/region-ring-experiments -p "test_run_32b_all40.py" -v
```

The included tests simulate an engine without downloading model weights or contacting the phone. **Actual 32B inference must be tested on the PC/phone hardware.**
