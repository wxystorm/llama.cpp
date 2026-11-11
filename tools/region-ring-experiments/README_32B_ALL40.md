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

`-Out` is optional: the normal default configuration keeps `32b-g8p6-x128-n1-all40`, while custom layout/Chunk or priority/no-mmap/cooldown settings automatically get a deterministic separate output directory. When you specify `-Out` explicitly, use a separate directory for a different layout/Chunk; otherwise the existing run signature rejects mixing results. The optional `-AutoPlan` mode is preserved but cannot be combined with manual layout/Chunk options.

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

## Preload all Phone HTP weights at model-load time, then warm up

**Fixed Phone Region layouts now default to two separate stages, before
any measured ShareGPT prompt:**

1. During **PC model loading**, `LLAMA_REGION_RING_PHONE_HTP_ONLY=1`
   and `LLAMA_REGION_RING_PHONE_EARLY_WEIGHT_RPC=<RPC>` enable the
   existing prefill-independent RPC command `PHONE_NPU_WEIGHT_PRELOAD`.
   The finalized `-Layout` determines every Phone Region, the exact
   Transformer layer IDs and its correct HTP0-3 session assignment.
   Phone loads/repackages these weights from its **local GGUF** and
   retains them in its HTP cache. The PC waits for all Region ACKs
   before Prefill; it fails the benchmark if the complete
   `START -> REGION* -> DONE -> JOIN` sequence cannot be verified.
   No manual `GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS` list is needed.
2. Execute a **warm-up Prefill**, using the first Prompt that would
   actually be measured (or the `-WarmupPrompt` override), with the
   same layout and runtime options. It lives under
   `<Out>\warmup\attempt-0001\`, with its own stdout, stderr,
   `metrics.jsonl`, `result.json`. It does not enter the 40-Prompt
   `summary.csv` or PASS counts. A warm-up failure aborts measured
   runs. The existing between-Prompt pause also applies once after
   warm-up.
3. Run measured Prompts. Model-load preload is checked in **every PC
   process**, but following runs should see resident Phone weights and
   can reuse them while the Phone RPC Server stays alive. On script
   resume, warm-up runs again *if* any new measured work remains, since
   the Phone RPC Server may have restarted between invocations.

For the current Qwen3-32B `G9,P5,G9,P5,G9,P5,G9,P5,G8` layout,
this means preloading **all 20 assigned Phone HTP Transformer layers**,
not just the first Region. This does *not* attempt to fit all
PC CUDA Region weights in 6 GiB VRAM: GPU Bank streaming remains.

```powershell
git pull --ff-only

.\tools\region-ring-experiments\run_32b_all40.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Layout "G9,P5,G9,P5,G9,P5,G9,P5,G8" `
  -Chunk 128 -XG 96 `
  -ForceSmallBatchMover -BoostBankMover -NoMmap `
  -InterPromptDelaySeconds 5 -NPredict 1 `
  -OnlyGroup 1024 `
  -Out ".\32b-xg96-early-weights-warmup-1024"
```

Expected **PC** log in the warm-up:
`[REGION_RING_PHONE_EARLY_WEIGHTS] event=DONE regions=4 ... status=PASS`
followed by `event=JOIN ... status=PASS`, and then
`[WARMUP] event=DONE status=PASS preload_status=PASS preload_regions=4 counted=0`.
The **Phone** server logs
`[PHONE_NPU_EARLY_WEIGHTS] event=LAYER ... cache=MISS`
on its first preload, and subsequent preloads should mostly report
`cache=HIT`. A `JOIN status=FAIL`, missing ACK, or unsupported
server must not silently turn a cold first Prompt into a PASS.
This requires the existing **patch-30-or-newer Phone RPC server and a
matching PC binary** with the Phone HTP preload function. No new RPC
wire command was introduced.

The warm-up still exercises per-RUN Graph/KV setup; it does not assume
those resources can be reused across different PC processes.

To disable stages separately (e.g., for an old baseline):
`-NoPhoneWeightPreload`, `-SkipWarmup`; optionally select a shorter
`-WarmupPrompt "prompt_128_01.txt"`. Both stages are disabled in
`-AutoPlan` mode, where the model-load-time Phone allocation is not
finalized. These choices are in the run-config signature. **Use a new
`-Out` directory** instead of mixing with the older first-Prompt-cold
results. Existing baseline can still resume with both new stages
disabled and otherwise-identical parameters.

## Drain-first Cycle scheduling: finish the earliest executable Cycle

For a fixed Ring such as `G0,P1,G2,P3,G4,P5,G6,P7,G8`, the PC GPU
worker now **prioritizes Chunks of the earliest unfinished runnable GPU
Region**, and the Phone worker independently prioritizes the earliest
runnable Phone Region. This gives G0 priority over G2 when both can run,
draining G0's Bank A earlier so G4's weights can transfer while G2
computes on Bank B. The same principle applies to G2/G4/G6 and later
Regions.

The choice is made **between Chunks**, not during a running Chunk; it
does not preempt kernels. It never delays a runnable downstream Chunk
just because the earliest Region is waiting for upstream Token intervals,
HTP input PUSH, an unavailable GPU Bank, or KV order. Instead it picks
the earliest **eligible** Region to avoid idle workers. Both GPU and
Phone continue to publish each completed Chunk to downstream immediately:
**there is no whole-Cycle barrier** and no change to HTP graphs or RPC.
The resource-aware eligibility filter and same-Bank lifetime constraints
remain in force.

**Default:** drain-first is enabled when resource-aware scheduling is
enabled. Override independently using
`LLAMA_REGION_RING_DRAIN_FIRST_CYCLE=1` (drain-first) or
`LLAMA_REGION_RING_DRAIN_FIRST_CYCLE=0` (resource-aware FIFO).
The PC log prints
`[REGION_RING_CYCLE_SCHED] mode=DRAIN_FIRST|FIFO ...`; the
`region_ring` JSONL record adds `drain_first_cycle`.

With the **same** `-Layout`, `-Chunk`, `-XG`, model, warm-up and
network conditions, compare the two strategies by appending one of
these options to the normal `run_32b_all40.ps1` command:

```powershell
# Drain-first Cycle priority:
-EnvOverride "LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED=1","LLAMA_REGION_RING_DRAIN_FIRST_CYCLE=1","LLAMA_REGION_RING_TASK_TRACE=1" `
-Out ".\\32b-g8p6-drain-first"

# Existing resource-aware FIFO control:
-EnvOverride "LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED=1","LLAMA_REGION_RING_DRAIN_FIRST_CYCLE=0","LLAMA_REGION_RING_TASK_TRACE=1" `
-Out ".\\32b-g8p6-resource-fifo"
```

The scheduler-only regressions in `tests/test-region-ring.cpp` verify:
(1) older G2 READY is skipped for a later-enqueued G0 Chunk;
(2) if G0 is ineligible, G2 is allowed to execute; (3) the same
upstream-first policy applies to Phone even with graph affinity;
(4) legacy FIFO/affinity behavior remains available. Performance
improvement must be established from whole-Prefill time plus the
GPU Compute/H2D overlap and Phone trace, not just the Bank Wait counter.

## Resource-aware Chunk scheduling (GPU Bank and Phone PUSH)

The native threaded Region Ring now keeps the same **Token-interval
dependency DAG, per-Region KV ordering, and double-Bank lifetime rules**,
but filters `READY` Chunk claims by their external resource readiness:

- **GPU:** a Chunk is claimable only if its parity Bank currently owns
  that GPU Region and is ready. If G4 is waiting for Bank A, a queued G2
  Chunk with resident Bank B can execute first. The Bank Mover signals the
  READY scheduler after refill completion; failure cancels READY.
- **Phone:** a Phone Chunk requiring PC->Phone PUSH is claimable only
  after that exact input PUSH succeeds. It can then run on the existing
  Phone-owned HTP executor; no new Graph, queue or RPC wire type is added.
  The PUSH worker signals READY when input is staged.
- READY is still conditioned on all **overlapping upstream Token
  intervals** having completed, and prior Chunk KV commits/deferred
  releases. **There is no whole-Cycle barrier.** The GPU Bank parity
  lifetime rule still prevents reloading Bank A/B until its previous
  owning GPU Region has fully drained.

**Default:** enabled when GPU Bank streaming or Phone PUSH is active.
A/B opt-out: `LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED=0`.
To compare with the prior behavior on the exact same prompt, pass
`-OnlyPrompt "prompt_1024_01.txt"` and give each variant its own
`-Out` directory. Example extra flags to append to the existing
32B launch command:

```powershell
# New scheduler, including PC-clock Chunk/Bank trace:
-EnvOverride "LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED=1", "LLAMA_REGION_RING_TASK_TRACE=1" `
-Out ".\32b-g8p6-resource-aware"

# Legacy scheduler control:
-EnvOverride "LLAMA_REGION_RING_RESOURCE_AWARE_CHUNK_SCHED=0", "LLAMA_REGION_RING_TASK_TRACE=1" `
-Out ".\32b-g8p6-resource-legacy"
```

The `region_ring` JSONL record includes
`resource_aware_chunk_sched`,
`gpu_resource_deferred_checks` and
`phone_resource_deferred_checks` (eligibility checks rejected,
**not unique tasks**). Existing `*_ready_wait_ms` includes waiting for
DAG tasks **and**, with the new filter, waiting for externally available
Bank/PUSH resources. The old `bank_gpu_wait_ms` can become nearly zero
because the GPU now waits *before claiming*, so **do not interpret a
lower Bank Wait counter alone as a throughput improvement**. Compare
full Prefill milliseconds / tok/s and the actual Chunk Trace's GPU
Compute vs Bank H2D overlap instead.

The three scheduler-only C++ regressions in
`tests/test-region-ring.cpp` cover skipping an unready Bank queue head,
preferring a Phone task with a completed PUSH, and waking blocked
workers on resource completion/cancellation. The actual RTX/HTP
performance and correctness still require running the user's
machine/phone A/B benchmark.

## Cooldown, raised transfer priority, and no-mmap A/B

The runner now **defaults to a 5-second pause between completed Prompt
runs**. It waits after the last attempt for a Prompt and before the next
actual Prompt execution (including when the last Prompt failed). It does
not wait before the first Prompt, after the last Prompt, when skipping
already-PASS prompts, or between retry attempts.
Use `-InterPromptDelaySeconds 0` to disable. Retry attempts have a separate
`-RetryDelaySeconds` (default 5 s).

For the requested XG96 experiment on Windows:

```powershell
.\tools\region-ring-experiments\run_32b_all40.ps1 `
  -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Layout "G8,P6,G8,P6,G8,P6,G8,P6,G8" `
  -Chunk 128 -XG 96 `
  -ForceSmallBatchMover `
  -BoostBankMover -NoMmap `
  -InterPromptDelaySeconds 5 `
  -OnlyGroup 128 `
  -Out ".\32b-xg96-mover-boost-nommap-pause5"
```

`-BoostBankMover` sets `LLAMA_REGION_RING_PRIORITY_BANK_MOVER=1` **only
for the PC engine subprocess**; the existing Windows C++ hook gives the
CPU Bank Mover thread `THREAD_PRIORITY_ABOVE_NORMAL`, not the whole
process or GPU device. Confirm
`[REGION_RING_THREAD_PRIORITY] role=BANK_MOVER requested=ABOVE_NORMAL status=PASS`
in `engine.stderr.log`. It cannot itself increase PCIe bandwidth.

`-NoMmap` passes `--no-mmap` to `llama-completion`: weights are loaded
into RAM rather than memory-mapped, potentially reducing mmap page-fault
variance. Ensure sufficient available RAM for the 32B model plus workspaces.
Each Prompt still runs in a separate engine process, so non-mmap model
loading can be slower; prefill time and model-load time are different metrics.

The new pause is included in `run-config.json`, and the boosted-thread
setting and no-mmap choice are also part of the configuration signature.
Older results that predate the pause setting can resume with
`-InterPromptDelaySeconds 0` and otherwise-identical settings; the original
`run-config.json` stays untouched. The 5-second setting deliberately cannot
be combined with older zero-pause measurements.
Use a **new -Out directory** so new results cannot be merged with old
ones. Keep the same flags and -Out to resume, or add
`-OnlyPrompt "prompt_128_06.txt" -RerunAll` to rerun one.

This experiment changes multiple variables (thread priority, mmap policy,
inter-Prompt cooling), so a throughput improvement would not isolate
which change helped. For a controlled comparison, hold two fixed and vary
the third.

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
