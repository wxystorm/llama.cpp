# ShareGPT50_short: 50 prompts, long-anchor placement lock, low Chunk search

This is a new experiment. Do not merge its results with the older forced
ShareGPT40 main table or 2048-4096-token group.

## Dataset / context

- Directory: E:\llama\datasets\sharegpt\sharegpt50_short
- Five groups: 0, 128, 256, 512, 1024; ten prompt_N_NN.txt each, 50 total.
- The group named 0 is really 64..128 tokens according to the tokenizer.
  Its ten counts MUST exist in manifest.csv under actual_tokens_tokenizer.
- Maximum context is 2048. The anchor prompt_1024_10.txt is always
  configured with context 2048; other contexts are rounded up to 256.
- Predict one token, and use --context-padding 0 to prevent an otherwise
  valid <2048-token prompt exceeding the ceiling.
- A NEW independent G/C/N Profile samples 16/24-token chunks, alongside
  32/64/96/128/192/256. **X=8 is not sampled or selected.** Phone HTP
  additionally samples REAL KV=2048 at the Context cap; KV=3072/4096
  is never sampled for this dataset. A previous short Profile with an
  extra X=8 point can be reused when it also has valid 16/24 samples.
- The full layer-placement Plan uses the normal chunk grid on 1024_10.
  Only AFTER this Plan is locked do prompts with <=256 planned tokens
  search 16/24 (and larger) widths; never X=8. The two-Bank
  2*XG <= tokens rule remains in effect.
- Planning-memory checks and real GPU/CPU/HTP allocations are not bypassed.

## Build

    $root = "E:\llama\llama-tensor-v2"
    cd $root
    git fetch origin
    git switch feat/region-ring-sharegpt50-short-first-long-20261116
    git pull --ff-only
    cmake --build .\build-tensor-v2 --config Release -j 8

## 32B example: two stages using the same output directory

    $root = "E:\llama\llama-tensor-v2"
    $data = "E:\llama\datasets\sharegpt\sharegpt50_short"
    $model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf"
    $layers = 64
    $out = "$root\32b-sharegpt50-short-fresh"
    $rpc = "10.238.229.239:50052"
    $opts = @(
      "--exe", "$root\build-tensor-v2\bin\Release\llama-completion.exe",
      "--model", $model, "--model-layers", "$layers",
      "--rpc", $rpc, "--prompt-dir", $data,
      "--repo-root", $root, "--out", $out,
      "--auto-plan", "--htp-only", "--main50-short",
      "--streaming", "on", "--profile-grid", "quick",
      "--batch", "2048", "--ubatch", "2048",
      "--n-predict", "1", "--threads", "8",
      "--min-context", "256", "--max-context", "2048",
      "--context-padding", "0", "--no-mmap",
      "--max-retries", "0", "--inter-prompt-delay-seconds", "5"
    )

    # Stage 1: only the longest 1024-group prompt; forces context 2048.
    python .\tools\region-ring-experiments\run_32b_all40.py @opts --only-prompt prompt_1024_10.txt
    Get-Content "$out\placement-lock.json"

    # Stage 2: same exact model/configuration/output; no --only-prompt.
    python .\tools\region-ring-experiments\run_32b_all40.py @opts

The first applied Plan is frozen to placement-lock.json (even if the
Runtime fails). A retry or the next prompt uses the exact same layer
placement, searching only XC/XG/XP. If stage 1 did not produce a Plan,
stage 2 fails closed rather than silently planning from a short prompt.

Use a distinct clean output directory for each model. For Qwen3-14B
use model Qwen3-14B-Q4_K_M.gguf and --model-layers 40; for Mistral
Small 3.2 24B use Mistral-Small-3.2-24B-Q6_K.gguf and --model-layers 40;
for Qwen3-32B use 64. Restart the phone RPC server with the matching
GGUF and compatible HTP build before switching models.

## Fixed 64-layer G/N topology, Chunk-only search (no estimated memory guard)

To use a *user-fixed* layout rather than the automatic long-anchor placement
search, use `--sharegpt50-short-dataset` plus `--locked-short-chunks`.
Do not add `--main50-short`: the latter deliberately prohibits a
user-fixed layout. For 64 Qwen3-32B layers, the originally typed final
`G6` must be **`G7`** (otherwise the layer counts sum to only 63).

    $root  = "E:\llama\llama-tensor-v2"
    $data  = "E:\llama\datasets\sharegpt\sharegpt50_short"
    $model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf"
    $out   = "$root\32b-fixed-gp-chunk16-no-memory-estimate"
    $rpc   = "10.238.229.239:50052"
    $optsFixed = @(
      "--exe", "$root\build-tensor-v2\bin\Release\llama-completion.exe",
      "--model", $model, "--model-layers", "64",
      "--rpc", $rpc, "--prompt-dir", $data,
      "--repo-root", $root, "--out", $out,
      "--auto-plan", "--htp-only", "--streaming", "on",
      "--sharegpt50-short-dataset",
      "--locked-layout", "G7,N4,G7,N4,G8,N5,G7,N4,G7,N4,G7",
      "--locked-short-chunks", "--skip-plan-memory-checks",
      "--profile-grid", "quick",
      "--batch", "2048", "--ubatch", "2048",
      "--n-predict", "1", "--threads", "8",
      "--min-context", "256", "--max-context", "2048",
      "--context-padding", "0",
      "--max-retries", "0", "--inter-prompt-delay-seconds", "5"
    )

    # For a new output directory, profile the first call or copy a
    # verified profile.csv + profile.json from a previous matching run
    # into $out\independent-profile before invoking the runner.
    python .\tools\region-ring-experiments\run_32b_all40.py @optsFixed --only-prompt prompt_1024_10.txt
    # Only after the first request genuinely succeeds, run the remaining 50.
    python .\tools\region-ring-experiments\run_32b_all40.py @optsFixed

The fixed layer allocation stays unchanged for every Prompt. Small
Prompts search Chunk **16/24/32+**, and large Prompts search **32+**.
The historical X=8 rows in older cached Profiles are accepted for
compatibility but never chosen by the Planner.

`--skip-plan-memory-checks` skips *estimated* CPU/GPU/Phone/HTP
capacity filtering only during locked Chunk search. The real CUDA,
host RAM and HTP allocations can still fail. Always use a separate
`--out` directory from guarded or automatic runs; compare real
Prefill PASS/TPS, not predicted_ms alone.

## Experimental resident llama-server smoke test

`llama-server` can keep the 64-layer G/P topology and HTP weights
resident across requests. The default server mode preserves its
model-load-time XC/XG/XP startup Plan. An additional opt-in
`--per-request-chunks` now calls the **native locked-layout Planner**
from `llama_context::decode()` before every real single-sequence
Prefill, using actual request token count and the same independent
G/C/N Profile/DAG; it applies the selected widths to this request's
runtime Stages only. Neither mode reallocates the model's PC/Phone layer
ownership. Keep startup-only and per-request runs in separate outputs;
the resident-server result is not directly the same experiment as
a separate process/model-load on each Prompt.

The smoke harness reuses this runner's HTP-only GPU Bank/RPC environment,
reuses a validated independent Profile (including an older X8 sample,
which is never selected), locks the 64-layer topology, opts out of the
Planner's memory ESTIMATES, and caps XG/XP at 32 so the 64-token short
group can retain two-Bank execution. It starts one server, submits the
longest Prompt, a short Prompt, then the longest Prompt again to check
request reset. The full 50 can be attempted with `--all50` AFTER this
three-request check passes.

    $root = "E:\llama\llama-tensor-v2"
    cd $root
    cmake --build .\build-tensor-v2 --config Release --target llama-server -j 8

    python .\tools\region-ring-experiments\probe_htp_llama_server.py `
      --server-exe "$root\build-tensor-v2\bin\Release\llama-server.exe" `
      --model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
      --rpc "10.238.229.239:50052" `
      --prompt-dir "E:\llama\datasets\sharegpt\sharegpt50_short" `
      --profile-source "$root\32b-sharegpt50-short-mmap-test\independent-profile" `
      --out "$root\32b-server-smoke" --port 18080

The harness forces `--parallel 1`, disables continuous batching, prompt
caching, on-disk KV checkpointing and ordinary warmup, and sends
`{"cache_prompt":false,"n_predict":1}` to `/completion`. It rejects
responses with `timings.cache_n != 0`, records the actual
`timings.prompt_n/prompt_ms/prompt_per_second`, and writes
`requests.jsonl` plus `llama-server.log`. A startup failure or later
HTP-only request failure does **not** establish server compatibility:
see the tail of `llama-server.log`. In particular, the current server
initialization uses a two-token context capability probe, while HTP-only
Graphless only supports substantive Prefill. That two-token probe remains a known diagnostic rejection, but
the three substantive requests above have passed on the target PC/Phone.

Offline checks:

    python .\tools\region-ring-experiments\test_probe_htp_llama_server.py

### Verified server smoke and full 50-Prompt run (2026-11-16)

After increasing the Phone Hexagon per-session fixed-Graph cache from
256 to 1024, the persistent HTP-only server passed three consecutive
requests. The first 1855-token prompt reached 85.629 tok/s; the 66-token
short prompt reached 22.442 tok/s; the repeated 1855-token prompt reached
80.250 tok/s. All three reported cache_n=0. This is a verified
three-request smoke test, not a measured full 50-prompt result.

The updated --all50 mode now uses exactly 50 UNIQUE prompts, including
prompt_1024_10.txt FIRST as the long anchor, and ten samples from each of
the five groups 0/128/256/512/1024. No duplicate long prompt is added.
It keeps one resident llama-server across all 50 HTTP requests, with
the same fixed topology and startup Chunk Plan (not per-request search).
The 0 group is 64..128 actual tokens according to manifest.csv.

Run from PowerShell with a NEW output directory (not 32b-server-smoke):

    $root = "E:\llama\llama-tensor-v2"
    cd $root
    git pull --ff-only
    python .\tools\region-ring-experiments\test_probe_htp_llama_server.py
    cmake --build .\build-tensor-v2 --config Release --target llama-server -j 8

    $serverOpts = @(
      "--server-exe", "$root\build-tensor-v2\bin\Release\llama-server.exe",
      "--model", "E:\llama\models\Qwen3-32B-Q4_K_M.gguf",
      "--rpc", "10.238.229.239:50052",
      "--prompt-dir", "E:\llama\datasets\sharegpt\sharegpt50_short",
      "--profile-source", "$root\32b-sharegpt50-short-mmap-test\independent-profile",
      "--out", "$root\32b-server-all50",
      "--port", "18080",
      "--all50"
    )

    python .\tools\region-ring-experiments\probe_htp_llama_server.py @serverOpts

To continue after an interruption, run the same command with --resume:

    python .\tools\region-ring-experiments\probe_htp_llama_server.py @serverOpts --resume

Optional staged sweep: add --limit 10 to the INITIAL invocation, then
use --resume without --limit to complete the remaining 40. A full
sweep defaults to a five-second gap between requests, excluded from
prefill_ms. To change the gap on a NEW experiment use
--inter-prompt-delay-seconds 0; changing it while resuming is rejected.

The script writes:
- requests.jsonl: one flushed PASS row per completed request, never
  overwritten by a resumed sweep.
- requests.csv: per-prompt actual prompt_n, cache_n, prefill_ms, tps,
  request_wall_ms, group, and difference from manifest tokenizer tokens.
- summary.csv and summary.json: each group and an ALL row, number
  completed/expected, median/mean/min/max tps, and weighted_tps equal to
  total processed tokens divided by total prefill time.
- server-run-config.json: fingerprint of the model, server binary,
  Profile, inputs, startup Plan, HTP flags and delay. A different
  fingerprint is NOT silently mixed into a partially completed run.
- llama-server.log for the first run, separate resume-numbered log files
  thereafter.

The script stops on a failed request or unexpectedly reused cache
instead of entering fake PASS results. Resumption starts a new resident
server but does not rerun completed requests. A successful 3/3 smoke
does not guarantee the entire 50 will pass.


### Native per-request Chunk-only Planner for resident llama-server

Use this **instead of** startup-only mode when you want every Prompt
to select its own XC/XG/XP. The Python script does NOT guess the Chunk
width based on token length; C++ re-invokes the existing native
`llama_hybrid_region_ring_planner_apply_wide()` search with actual
`n_tokens_all`, scores the fixed Region topology using the stored
Profile and full event DAG, and constructs new *local* runtime Stages
before the current decode executes. The model-owned layer placement,
HTP weight mappings and GPU Bank weights remain unchanged.

Opt-in is through `LLAMA_REGION_RING_PER_REQUEST_CHUNKS=1`,
set automatically by the Python harness when using
`--per-request-chunks`. The known two-token llama-server capability
probe is skipped (less than 32 tokens). A regular Prompt MUST be a
single fresh sequence processed in one UBatch. A request that cannot
be planned with the locked topology fails explicitly; it does not
quietly fall back to startup-only Chunk widths or repartition weights.

- For actual tokens <=256, use the independent short-width grid
  (16, 24, 32 and wider), with X=8 excluded.
- Longer Prompts retain the regular 32+ grid.
- The selected physical XG and XP are capped at 32; the unused XC
  model width may be larger because the fixed layout contains no
  CPU layer Regions.
- The two-Bank constraint 2*XG <= real tokens remains enforced.
- `--skip-plan-memory-checks` still skips ONLY estimates in this
  user-fixed locked layout; real allocation failures are still fatal.

For the 32B long-short-long regression with native re-planning:

    $root = "E:\llama\llama-tensor-v2"
    cd $root
    git pull --ff-only
    cmake --build .\build-tensor-v2 --config Release --target llama-server -j 8
    python .\tools\region-ring-experiments\test_probe_htp_llama_server.py

    $args = @(
      "--server-exe", "$root\build-tensor-v2\bin\Release\llama-server.exe",
      "--model", "E:\llama\models\Qwen3-32B-Q4_K_M.gguf",
      "--rpc", "10.238.229.239:50052",
      "--prompt-dir", "E:\llama\datasets\sharegpt\sharegpt50_short",
      "--profile-source", "$root\32b-sharegpt50-short-mmap-test\independent-profile",
      "--out", "$root\32b-server-dynamic-smoke",
      "--port", "18080",
      "--per-request-chunks"
    )
    python .\tools\region-ring-experiments\probe_htp_llama_server.py @args

To run the FULL 50 unique Prompts, use a separate output directory
and add `--all50`; the server remains resident for all 50:

    $full = @(
      "--server-exe", "$root\build-tensor-v2\bin\Release\llama-server.exe",
      "--model", "E:\llama\models\Qwen3-32B-Q4_K_M.gguf",
      "--rpc", "10.238.229.239:50052",
      "--prompt-dir", "E:\llama\datasets\sharegpt\sharegpt50_short",
      "--profile-source", "$root\32b-sharegpt50-short-mmap-test\independent-profile",
      "--out", "$root\32b-server-dynamic-all50",
      "--port", "18080",
      "--all50", "--per-request-chunks"
    )
    python .\tools\region-ring-experiments\probe_htp_llama_server.py @full

    # Resume after an interruption, with identical inputs and binary.
    python .\tools\region-ring-experiments\probe_htp_llama_server.py @full --resume

The harness validates EACH successful HTTP response against one native
`[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT` log from the same
request. A response without this marker is NOT counted as a valid
per-request benchmark. The CSV/JSONL rows contain
`plan_xc / plan_xg / plan_xp / planner_ms`, plus
`runtime_xc / runtime_xg / runtime_xp` from the matching
`[REGION_RING_SUM] ... complete=1 status=0` execution proof.
Runtime tiny-tail coalescing may increase a selected physical width by
up to 25%, so the selected and executed widths are deliberately separate.
Group summaries include the distribution of selected Chunk Plans and
mean/median Planner time.
The native Planner logs
`[REGION_RING_REQUEST_CHUNK_PLAN] event=REJECT ... status=FAIL`
if it cannot preserve the model-owned layout.

For A/B comparison run startup-only in a DIFFERENT `--out` directory
with the identical options EXCEPT omit `--per-request-chunks`.
The per-request implementation is experimental and needs PC+HTP
verification; do not treat the code change alone as a measured speedup.


## Consistent 32B A0..A3 ablations

The existing four-case ablation runner now supports the same 50-prompt
dataset for 40-layer 14B/24B or 64-layer 32B models, a 2048-context
ceiling and a CONTROLLED fixed Chunk width <=32
for valid short 0-group two-Bank execution. A0/A1 use the anchored Full
main-run placement lock, so they do not reuse an unrelated old G8/P6
layout. A2/A3 retain their own offline planner alternatives by design.

    python .\tools\region-ring-experiments\run_32b_ablation.py --model "$model" --model-layers "$layers" --exe "$root\build-tensor-v2\bin\Release\llama-completion.exe" --rpc "$rpc" --prompt-dir "$data" --out "$root\32b-ablation-sharegpt50-fresh" --main50-short --baseline-placement-lock "$out\placement-lock.json" --profile-csv "$out\independent-profile\profile.csv" --context 2048 --chunk 32 --limit 0 --phone-budget-mib 7800

7800 MiB is an illustrative phone capacity only: replace it with the
verified device's available-memory budget. Reusing the matched Full
main-run Profile prevents an unnecessary new calibration, especially for
40-layer 14B/24B models. If no --profile-csv is supplied, the A2 calibration
uses a valid 40/64-layer layout, never an incompatible legacy 64-layer layout. Run --model-layers 40 for Qwen3-14B or Mistral Small 24B, or 64
for Qwen3-32B; use a separate matching long-anchor placement-lock.json,
model GGUF and output directory for each model. A2/A3 retain their
conservative offline memory heuristic and are not the native full Planner.

## Offline regression tests

    python .\tools\region-ring-experiments\test_main50_short_workflow.py
    python .\tools\region-ring-experiments\test_first_plan_lock.py
    python .\tools\region-ring-experiments\test_run_32b_ablation.py

Tests validate CLI/dataset/lock logic without actual inference; hardware
integration is still required. Use run-config.json and summary.csv for
provenance, and keep the old forced-layout results separate.
