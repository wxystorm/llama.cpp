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
- A NEW independent G/C/N Profile samples 8/16/24-token chunks, alongside
  32/64/96/128/192/256. The old Profile cache is deliberately incompatible.
- The full layer-placement Plan uses the normal chunk grid on 1024_10.
  Only AFTER this Plan is locked do prompts with <=256 planned tokens
  search 8/16/24 widths. The two-Bank 2*XG <= tokens rule remains in effect.
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

## Consistent 32B A0..A3 ablations

The existing four-case ablation runner now supports the same 50-prompt
dataset for 40-layer 14B/24B or 64-layer 32B models, a 2048-context
ceiling and a CONTROLLED fixed Chunk width <=32
for valid short 0-group two-Bank execution. A0/A1 use the anchored Full
main-run placement lock, so they do not reuse an unrelated old G8/P6
layout. A2/A3 retain their own offline planner alternatives by design.

    python .\tools\region-ring-experiments\run_32b_ablation.py --model "$model" --exe "$root\build-tensor-v2\bin\Release\llama-completion.exe" --rpc "$rpc" --prompt-dir "$data" --out "$root\32b-ablation-sharegpt50-fresh" --main50-short --baseline-placement-lock "$out\placement-lock.json" --context 2048 --chunk 32 --limit 0 --phone-budget-mib 7800

7800 MiB is an illustrative phone capacity only: replace it with the
device's observed available-memory budget. Run --model-layers 40 for Qwen3-14B or Mistral Small 24B, or 64
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
