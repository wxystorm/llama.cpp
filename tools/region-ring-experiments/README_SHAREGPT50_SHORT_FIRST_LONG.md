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
resident across requests, but the current native Planner is a model-load
hook. The server **does not re-plan XC/XG/XP per HTTP request**. Treat
this mode as a separate fixed-Chunk feasibility/overhead test, never as
equivalent to the dynamic per-Prompt main experiment.

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
Graphless only supports substantive Prefill. That path is not yet
validated end to end on the target PC/Phone.

Offline checks:

    python .\tools\region-ring-experiments\test_probe_htp_llama_server.py

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
