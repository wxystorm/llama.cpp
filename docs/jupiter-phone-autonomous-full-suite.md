# One-command full Prompt evaluation: Jupiter Phone-owned autonomous Prefill

Branch: `feat/jupiter-phone-autonomous-prefill-20261123`.
Scope: Qwen3-14B only, fresh single-sequence **Prefill**, no Decode.

This is the formal all-Prompt extension of the already tested single-Prompt
`run_phone_owned_autonomous_ab.py`. It calls the same
`run_phone_owned_full_prefill.py` native runner per case, so
both Sync and Phone-autonomous modes use the same original PC CPU/CUDA prefix,
Phone local GGUF/HTP Graphs, private KV and PC output head.

## Windows: one command for all 50 ShareGPT Prompts

Keep the already-tested Phone RPC server running on the same branch.
The only new source in this commit is Python and PowerShell: rebuild the
PC only if `llama-tokenize.exe` is absent.

```powershell
cd E:\llama\llama-jupiter
git pull --ff-only

# Only when the tokenizer binary is missing:
cmake --build build-phone-owned --config Release --target llama-tokenize --parallel 8

# Preflight: count actual Qwen3 chat-template tokens and preview all plans.
.\tools\jupiter\run_phone_owned_full_suite.ps1 -DryRun

# Full 50-Prompt suite (default: L32..39 Phone, S4, 1 A/B pair per Prompt).
.\tools\jupiter\run_phone_owned_full_suite.ps1
```

The PowerShell wrapper defaults to these existing PC paths:
- `E:\llama\models\Qwen3-14B-Q4_K_M.gguf` (model)
- `E:\llama\datasets\sharegpt\sharegpt40` (Prompt directory)
- `10.71.109.52:10140` (Phone RPC endpoint)
- `build-phone-owned\bin\Release\llama-cli.exe`
- `build-phone-owned\bin\Release\llama-tokenize.exe`

All are overridable with `-Model`, `-PromptsDir`,
`-Rpc`, `-Exe` and `-Tokenizer`.
`-ExpectedCount 50` enforces full coverage; if a different
dataset is intentionally supplied, change this value or set it to 0.

Equivalent direct Python call:

```powershell
python .\tools\jupiter\run_phone_owned_full_suite.py `
  --exe ".\build-phone-owned\bin\Release\llama-cli.exe" `
  --tokenizer ".\build-phone-owned\bin\Release\llama-tokenize.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --rpc "10.71.109.52:10140" `
  --prompts-dir "E:\llama\datasets\sharegpt\sharegpt40" `
  --expected-count 50 --sessions 4 --layouts 32 `
  --chunk-size 88 --context-floor 512 --context-cap 4096 `
  --repeats 1 --revisit-first `
  --out-dir ".\jupiter-autonomous-full50"
```

## Exact workload / geometry policy

- Discover **every** file matching `prompt_*_*.txt` in the
  specified directory, sort by numeric length group and within-group index.
  Example expected groups: `0 / 128 / 256 / 512 / 1024`
  (10 Prompts per group for the 50-Prompt dataset). A
  `2048-4000` filename group is also supported.
- Count Qwen3 **chat-formatted** tokens with matching
  `llama-tokenize --stdin --chat-user --show-count`. Do not
  add a fixed chat-template offset or reuse raw text token counts.
- Choose Context as the smallest multiple of 256 that fits the actual
  token count, with a configurable minimum of 512 and maximum 4096.
  A request beyond 4096 is a preflight **error**, never silently skipped.
- Choose a 2–16 Chunk plan. Prefer 88 tokens (reproducing the validated
  [88,87] pilot for 175 tokens). For short Prompts, use 2 Chunks; for longer
  ones, increase Chunk size to keep at most 16 HTP Graph shapes.
  Every full Chunk has identical width; the last may be shorter.
- Keep `n_predict=1`. This is a Prefill first-token
  measurement, not Decode; no unsupported KV handoff.
- The Phone's HTP KV always belongs to the single active request.
  **Do not run multiple full-suite processes against the same Phone RPC
  server at once.**

## Per-Prompt experiment and cache comparability

For each Prompt and chosen layer layout, sequentially:
1. Run a **Sync Native warmup** to pre-load/rearm that Graph shape.
   The warmup is excluded from latency summaries.
2. Run full **Sync Native** and **Phone-autonomous Native** Prefill.
   Alternating which mode runs first reduces systematic run-order bias.
3. Compare **all F32 final logits**. PASS requires Top-1 equality,
   cosine >= 0.999 and relative RMSE <= 0.01. Keep actual error values
   for inspection; matching one generated token is not enough.
4. Mark a pair `warm_comparable=true` only when
   BOTH measured modes have zero model-load/Graph-build/buffer/precompile
   durations. All complete pairs remain recorded, but **only these
   comparable warm pairs** enter the speedup geometric mean.
5. Persist that case atomically before moving to the next Prompt.
   Failed Phone RPC leaves logs and halts by default (a failed stream
   can block the Phone's sole active slot); resume later with
   `-Resume`.

With default `-RevisitFirst`, the script also runs A -> B -> A
for two distinct token/Chunk shapes in each group, compares replay logits
to the initial Prompt for **both** modes, and records whether the *first*
revisit matched a previously resident template **before** its warmup.
A Graph cache MISS may be expected if the budget forced safe generation-wide
eviction; use the Phone `PREPARE_CACHE` and `CACHE_EVICT`
logs to explain it. Replay rows are excluded from throughput means.

## Output files and resuming

```text
jupiter-autonomous-full50/
  suite-manifest.json          input/config/binary provenance and plans
  token-cache.json             avoids retokenizing unchanged Prompt files
  all-results.csv              all cases, errors, modes, Context, Chunk, parity
  all-results.json             full machine-readable per-case results
  group-summary.csv            group-level mean, median and warm speedups
  summary.json                 total PASS/FAIL and warm geometric mean
  cases/
    l32/g128/prompt_128_01/r01/
      case-result.json
      attempt-001/
        warmup/                warmup log and timing; unused logits removed
        sync/
          native-full-prefill.log
          native-logits.f32
          results.json
          runner.stdout.log
        autonomous/
          native-full-prefill.log
          native-logits.f32
          results.json
          runner.stdout.log
```

Resume after interruption or correcting the Phone server:

```powershell
.\tools\jupiter\run_phone_owned_full_suite.ps1 -Resume
Get-Content .\jupiter-autonomous-full50\summary.json
Import-Csv .\jupiter-autonomous-full50\group-summary.csv | Format-Table
```

Only cases whose checkpoint says PASS and whose **two final logits
files are still present** are skipped. A failed or incomplete case gets
a new `attempt-NNN` subfolder; previous failure logs survive.
Resuming refuses changed Prompt contents, binary fingerprints, code,
layouts, Chunk/Context policy or mode options. Use a new output directory
when intentionally changing the experiment.

## Full dynamic-layout matrix / repeated measurements

After the baseline full50 suite succeeds, use the same entry point for
four Phone suffix sizes and 3 repeats per Prompt:

```powershell
.\tools\jupiter\run_phone_owned_full_suite.ps1 `
  -Layouts "32,34,36,39" -Repeats 3 `
  -OutDir ".\jupiter-autonomous-layout-matrix"
```

L32/34/36/39 correspond to Phone 8/6/4/1-layer
suffixes, while the PC CPU prefix remains L0..15. Each case automatically
gets its own Context based on actual Prompt tokens.

## Local tests / limits

```powershell
python -m unittest discover -s .\tools\jupiter -p test_run_phone_owned_full_suite.py -v
python -m py_compile .\tools\jupiter\run_phone_owned_full_suite.py
```

These are offline tests of planning, resume and summaries; only PC+Phone
can validate HTP Graph preparation, numerical accuracy and performance.

The Phone RPC server must remain running throughout the suite, with the
matching previously tested autonomous branch and local GGUF. If a long
Prompt causes HTP memory infeasibility, the suite records the failure and
halts by default instead of claiming all Prompts passed. Use the saved
case logs to adjust a NEW experiment or fix the Runtime, then resume
with identical measured settings.

There is **no Decode**, no independent multi-tenant Phone scheduling and
no claim of GPU/HTP kernel-level overlap from host timestamps alone.
