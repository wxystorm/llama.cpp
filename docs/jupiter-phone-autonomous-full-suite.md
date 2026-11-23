# Jupiter original-design full Prompt suite: Context=2048, measured q(x,y) DP

Branch: `feat/jupiter-phone-autonomous-prefill-20261123`
Workload: Qwen3-14B ShareGPT50, one-shot Prefill (`n_predict=1`), NO Decode.
The **same fixed Context=2048 and Batch=2048** are used for every Prompt.

This is the native Phone Graph migration of the *original Jupiter* planning
path, not a new fixed-88 heuristic. It preserves the original Stage2 minimum
efficient Chunk calibration, q(x,y) per-Prompt measurement, Sequence DP,
targeted q refinement, verification and Prefill benchmarking. Phone-side
Graph compilation and autonomous execution are different implementation
mechanisms, not a replacement for Jupiter's planner or layer assignment.

## One command on Windows

Update both the PC executable and Android Phone RPC server to **RPC 4.0.7**
from the same branch; old patch-6 binaries are wire-incompatible with the
extended nonuniform Sequence DP plan. Restart Phone server after deployment.
The original Native Sync and Phone Auto modes share this new Plan ABI.

```powershell
cd E:\llama\llama-jupiter
git fetch origin
git switch feat/jupiter-phone-autonomous-prefill-20261123
git pull --ff-only

cmake --build build-phone-owned --config Release --parallel 8 --target llama-cli llama-tokenize test-phone-owned-execution-plan test-phone-qwen3-local-builder

ctest --test-dir build-phone-owned -C Release --output-on-failure -R "test-phone-owned-execution-plan|test-phone-qwen3-local-builder|test-phone-owned-autonomous-queue"

# No inference; discover all Prompt files and exact Qwen3 chat-template tokens.
.\tools\jupiter\run_phone_owned_full_suite.ps1 -DryRun

# Stage2 b once per layer layout; Stage3 q(x,y)+DP for EVERY Prompt;
# then warmed Sync-vs-Phone autonomous full-logits A/B.
.\tools\jupiter\run_phone_owned_full_suite.ps1
```

Defaults are the already-tested paths:
- model `E:llamamodelsQwen3-14B-Q4_K_M.gguf`
- dataset `E:llamadatasetssharegptsharegpt40`
- Phone RPC `10.71.109.52:10140`
- PC first Phone layer L32 (Phone L32..39 / eight layers), CPU prefix L0..15,
  four HTP Sessions; Qwen3's PC Head remains unchanged
- 50 unique Prompt files are required by `-ExpectedCount 50`
- warm-up before measured pairs, alternating Sync/Phone-Auto run order
- per-group distinct shape A→B→A replay, complete F32 Logits parity check
- 1 measured pair per Prompt; stage2/q/refine calibration repeats are 2/2/2
- fixed 2048 Context and Batch, with max Chunk=128 and native limit 16

All parameters except fixed Context can be overridden. To use a preexisting
*matching* measured Stage2 b profile instead of remeasuring it:

```powershell
.\tools\jupiter\run_phone_owned_full_suite.ps1 -MinChunkJson ".\jupiter-min-chunk\min-chunk.json"
```

The Stage2 profile must have the same CPU layers, PC split, max Chunk,
hardware and workload configuration. One Stage2 JSON applies only to one
layer layout. Without a supplied JSON, the suite measures Stage2 on a
representative Prompt near 357 tokens, then reuses the measured hardware
efficiency curve as in Jupiter's original Stage5. The reuse across Prompt
lengths is explicitly recorded by Stage3, not presented as remeasurement.

## Planning is genuinely Jupiter q(x,y) + DP

1. Enumerate all `prompt_*_*.txt` files (sorted by numeric
   group and ordinal). Count **actual Chat Template tokens** using the
   matching `llama-tokenize --chat-user` tool, rather than
   assigning a guessed constant offset. Any Prompt exceeding fixed
   Context=2048 causes an explicit preflight error.
2. Use original `run_min_chunk_profile.py` to select
   the minimum efficient Chunk b for the given fixed model/device layout
   (PC CPU16 + GPU16 + Phone HTP8, S4).
3. Run original `run_sequence_pipeline.py` separately for
   each Prompt: initial q(x,y) stage timings, Sequence DP optimization,
   targeted missing-point q refinement, and application verification
   under the **original Jupiter RPC compute path**. Calibration samples
   synchronize devices; they are not formal throughput numbers.
4. Require `FULLY_MEASURED_AND_VERIFIED` and an exact
   `sequence.<actual-token-count>` plan. Do NOT accept
   an unmeasured DP guess, normalize a variable plan into fixed chunks, or
   silently change the total token count.
5. Transfer the exact ordered Chunk schedule into the Phone-local Graph
   Plan. Local Qwen3 graph dimensions, RoPE positions, private HTP KV slots,
   FIFO admission and the last P2C payload are indexed by cumulative
   prefix lengths, not by `chunk_index * chunk_width`.
   Different Chunk widths may coexist in the same run. The native path
   accepts **1–16 Chunks**: a short Jupiter DP result may contain just
   one Chunk and is still valid Prefill (without inter-Chunk overlap).
6. Run Sync Native warmup, then measured Sync Native and Phone autonomous
   for the exact same plan and fixed Context=2048. Only matched
   **hot-Graph** pairs enter reported geometric-mean performance ratios.
   Full F32 Logits equality is checked per Prompt.

The 16-Chunk cap is a **migration executor constraint**, not a claimed
original INFOCOM'25 Jupiter limitation. The Stage3 q(x,y) DP searches with
`--max-subsequences 16` and the profile's measured
min/max chunk bounds. If a measured DP cannot fit within 16 or preparation
exceeds the Phone's actual HTP memory budget, fail with the reason and do
not manufacture an incompatible baseline.

The original Jupiter **model-partition DP** still exists in
`jupiter_planner.py`; this full suite uses a specified
hardware-feasible, previously tested layer boundary L32 for both modes.
It does not claim to search every original Jupiter model-partition
alternative, because the Phone-owned native suffix presently supports
only 1..8 contiguous Transformer layers. This preserves the same
fixed placement in both A/B modes without misrepresenting its origin.

## Results and resume

```text
jupiter-autonomous-jupiter-dp-2048/
  suite-manifest.json
  token-cache.json
  calibration/
    l32/
      stage2/
        min-chunk.json
        chunk-samples.csv
      plans/
        prompt_128_01/
          final.plan           # original Jupiter sequence.N
          stage3/
            sequence-summary.json
            q-sweep/...        # measured q(x,y)
            refinement/...     # exact q calibration
  all-results.csv
  all-results.json
  group-summary.csv
  summary.json
  cases/
    l32/g128/prompt_128_01/r01/
      case-result.json
      attempt-001/
        warmup/native-full-prefill.log
        sync/native-full-prefill.log
        sync/native-logits.f32
        autonomous/native-full-prefill.log
        autonomous/native-logits.f32
```

The CSV has Prompt group, actual tokens, fixed Context, exact
Jupiter DP cuts, b, plan file/source, Sync/Auto Prefill latency,
warm cache status, throughput, complete Logits error and A-B-A replay
validation. The group CSV contains mean/median and **warm-only**
geometric mean speedup; revisit rows are excluded from throughput means.

After any interrupted calibration or Phone RPC failure:

```powershell
.\tools\jupiter\run_phone_owned_full_suite.ps1 -Resume
Get-Content ".\jupiter-autonomous-jupiter-dp-2048\summary.json"
Import-Csv ".\jupiter-autonomous-jupiter-dp-2048\group-summary.csv" | Format-Table
```

Only complete PASS checkpoints with both Logits files are skipped.
Incomplete/failed A/B attempts retain logs in incrementing attempt folders.
The input/runner signature is pinned, so changing Context, binary, Prompt
content, CPU/Phone split, DP settings or q-sweep settings requires a NEW
output directory. The Phone RPC Server must remain alive; do not run
two full suites against its sole active Native Graph run simultaneously.

## Diagnostics / explicit ablation, not the main result

To test the earlier uniform fixed128 strategy, opt in and name the result
an ablation. This intentionally **does not perform Jupiter DP**:

```powershell
.\tools\jupiter\run_phone_owned_full_suite.ps1 -SequencePolicy fixed-control -ChunkSize 128 -OutDir ".\jupiter-fixed128-control"
```

For repeated/alternative layer tests:

```powershell
.\tools\jupiter\run_phone_owned_full_suite.ps1 -Repeats 3 -OutDir ".\jupiter-dp-2048-repeat3"
.\tools\jupiter\run_phone_owned_full_suite.ps1 -Layouts "32,34,36,39" -OutDir ".\jupiter-dp-2048-layer-matrix"
```

A model-free unit-test entry point is available:

```powershell
python -m unittest discover -s .\tools\jupiter -p test_run_phone_owned_full_suite.py -v
python -m py_compile .\tools\jupiter\run_phone_owned_full_suite.py
```

Actual GPU/HTP correctness, memory viability and performance require
the Windows PC + Snapdragon Phone. No Decode and no KV handoff are involved.
