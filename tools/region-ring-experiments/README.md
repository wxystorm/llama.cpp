# Lightweight Region Ring prompt sweep

This runs each `prompt_N.txt` in the specified directory **once** in a fresh
`llama-completion` process, and saves everything to files. No Region/Chunk-level
logging is added to the hot path. Cumulative per-ubatch metrics are assembled after the worker DAG finishes, buffered in memory, and flushed at the final official performance report (not written to disk during device computation).

The current `tools/cli/llama-cli` entrypoint may be a **client**. Use the
local GGUF runner built from `tools/completion`:

```powershell
cmake --build build-tensor-v2 --config Release --target llama-completion
python .\tools\region-ring-experiments\run_prompt_sweep.py `
    --exe .\build-tensor-v2\bin\Release\llama-completion.exe `
    --model E:\llama\models\Qwen3-14B-Q4_K_M.gguf `
    --rpc 10.39.52.114:50052 `
    --prompt-dir D:\prima.cpp\prompts `
    --out .\experiment-results `
    --n-predict 32
```

For the shorter PowerShell entrypoint, use:

```powershell
.\tools\region-ring-experiments\run_prompt_sweep.ps1 `
    -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
    -Rpc "10.39.52.114:50052" `
    -PromptDir "D:\prima.cpp\prompts"
```

The wrapper defaults to `build-tensor-v2\bin\Release\llama-completion.exe`
and saves `summary.csv` under `region-ring-results`.

### Automatic per-model Region Ring baseline

An unset `LLAMA_HYBRID_REGION_RING_LAYOUT` is now resolved **inside the
C++ model loader** from the actual GGUF layer count (not the model filename).
The 40-layer Dense baseline is `G4,C10,C2P1,P8,P2C1,G4,C7,P5`; the
48-layer MoE baseline remains `G4,C7,P5,G4,C7,P5,G3,C8,P5`. For other
depths of at least eight layers, a conservative valid G/C/P baseline is
generated with 2-4 cycles. Its structural validity does not guarantee
runtime memory feasibility or good performance; the measured Planner can
replace it with a better candidate.

The sweep **clears inherited** `LLAMA_HYBRID_REGION_RING_LAYOUT` so a
stale PowerShell override from a different model cannot contaminate runs.
An explicit `--force-layout` continues to control a Planner A/B candidate.

`-BaselineChunkTokens 64` is the default PowerShell setting and is
passed as `LLAMA_HYBRID_REGION_RING_CHUNK_TOKENS`. This is the common
fallback Region Ring Chunk size, **not** Batch, UBatch, or context. When
`apply-wide` applies a Plan, its `XC/XG/XP` take precedence.

Replace the model filename, executable path and RPC address as appropriate.
The Phone `ggml-rpc-server` must be running. Set `--streaming off` to
compare a non-streaming configuration. For a fixed-layout A/B test, use
`--force-layout "G6,C2,P4,G6,C2,P4,G6,C2,P4,G6,C2,P4"` together with
`--force-xc 32 --force-xg 64 --force-xp 128` (the example has 48 layers;
use a layout matching your model).

The CSV parser can be checked without a model or Phone:

```powershell
python .\tools\region-ring-experiments\test_prompt_sweep.py
```

A command preview is available with `--dry-run`. Every actual run loads its
model once, uses the planner selected for that input length (filename token
number is a hint), and generates up to 32 tokens by default. Context size
grows with the filename token hint; increase `--context-padding` if the
tokenizer produces more tokens than expected.

### APPLY_WIDE first-Cycle Chunk search (fine-grid E + fixed-X prefix)

The apply-wide first-Cycle search considers six variables:
(G,C,P,XG,XC,XP). Its Chunk candidate set for **each device** is:

    8,12,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192

Chunks larger than the evaluated Prompt are not separately searched in this
first-Cycle pass. Layer compute and Prepare times for intermediate Chunk
sizes are interpolated from the Hybrid Profile; additional direct full-layer
probes at 24 and 48 tokens (and 96 on CUDA) improve that interpolation.

The existing minimax **E=max(e1,e2,e3)** remains the first-stage objective.
Its CPU/CUDA/Phone service terms are evaluated from cached per-device
Profile curves. The inter-device handoff terms explicitly account for
**overlapping absolute token intervals** for both producer and consumer X.
For example, G40 -> C24 incurs two intersections for the first G40
producer; any fixed transfer startup cost is charged for each simulated
overlap edge. The first-stage E handoff is a sum of those per-edge
estimated costs divided by Prompt tokens, consistent with the complete
event simulator's transfer model. These are estimates, not measured socket
timings; compare against real runs when evaluating performance.

The first-Cycle **E1(XG,XP)**, **E2(XC,XG)** and **E3(XP,XC)** constraints
allow exact pairwise bitmask screening. An initial E cutoff is derived from
the original four anchor sizes {16,32,64,128} and refined against the
fine-grid minimum. The search retains the best few X choices for each
Cycle geometry and the best geometry for each X triple, avoiding a giant
materialized six-dimensional candidate array.

The first-Cycle E screen evaluates **every** six-dimensional combination
within the 17-choice X grid that passes the E1/E2/E3 bitmask filter.
Before calculating the anchor E minimum, it now **rejects geometries that
cannot fit the GPU peak-memory budget**. With two-Bank weight streaming,
it evaluates Bank0/Bank1 peaks across the legal later-Cycle width range:
for the GPU-memory-only existence test, a one-layer GPU Region in each
future Cycle minimizes unavoidable GPU weights and GPU KV. If *no* legal
width sequence can fit that GPU-only minimum plus the optional XG slot
scratch, the entire first-Cycle seed is removed **before E ranking**.
The budget matches the final guard: **90% of measured free GPU memory**.
This proves GPU feasibility of at least one possible completion, but not
full CPU/Phone feasibility or fast end-to-end runtime.

There is no per-geometry Top-3 cutoff. The initial anchor-derived
E limit is a conservative screening bound **over GPU-viable candidates**;
actual seed eligibility is checked against the exact per-Cycle-count
`E_min + margin` after the scan. Streaming summaries retain the best
combinations by geometry, by X triple, and by global E so that millions of
screened combinations do not need to be materialized.

Up to **100 diverse first-Cycle seeds in total, shared across the 2/3/4
Cycle counts**, enter the prefix compensation beam (width **8**).
Selection retains a best seed for each feasible Cycle count, then diverse
geometries, diverse X triples and the remaining lowest-E seeds.
Each seed fixes **XC/XG/XP for every subsequent Cycle**.
Later Cycles adjust only G/C/P; a Phone stage can have any positive number
of layers subject to topology, profiled-cost and memory-feasibility checks,
rather than an arbitrary 8-layer limit. On each subsequent Cycle, the Planner recomputes the **maximum GPU Region
weight for each physical Bank** (even-Cycle and odd-Cycle GPU regions),
adds GPU KV, and prunes memory-impossible Beam expansions. Once the last
Cycle has been appended, the original full CPU/Phone/GPU memory feasibility
check runs **before** E-ranked Beam truncation, so low-E but impossible
layouts cannot occupy all eight Beam slots.

The original cumulative prefix E, history and Cycle-width soft-penalty
ranking remains in use. No Chunk re-search occurs after selecting the
first-Cycle seed. Final choices still come from the complete Prompt event
simulation and optional Tensor boundary comparison.

The original four-point uniform repeated-Cycle search stays as a separate
A/B control, including its historical 2..8 Phone-length sweep.
The unrestricted Phone search applies to the new first-Cycle prefix path.

Forced XC/XG/XP experiments accept the fine grid above; always specify
all three values together with a complete --force-layout.

### First-Cycle selected-seed diagnostics

With `--planner apply-wide`, each prompt directory now includes
`planner-first-cycle-top100.csv`. This is written **once after planning**,
before model execution, alongside `metrics.jsonl`; no per-task logging or
terminal output is added. The existing planner JSONL record also contains
`first_cycle_seed_csv` when the file was successfully written.

The CSV records **each of the at most 100 selected first-Cycle seeds**
(2/3/4 Cycle counts share the cap), in actual selection order:

- `rank`, `selected_by`: why it survived the global seed cap
  (`CYCLE_BEST_E`, `GEOMETRY_DIVERSITY`, `CHUNK_DIVERSITY`, `LOW_E_FILL`).
- `cycles,first_G,first_C,first_P,first_cycle_layers,XC,XG,XP,first_E`:
  exact initial geometry, independent Chunk choices, and E score.
- `W_*_mspt`, `T_*_mspt`: first-Cycle profiled station and handoff costs.
- `gpu_bank0_min_mib`, `gpu_bank1_min_mib`,
  `gpu_kv_min_mib`, `gpu_peak_min_mib`:
  optimistic first-Cycle GPU VRAM lower bounds (first CUDA Region, at
  least one mandatory second-Cycle GPU layer, and minimum future KV).
- `gpu_full_possible_min_mib`, `gpu_budget_mib`: the tightest full-Cycle
  **GPU-only** requirement attainable over legally sized later Cycles
  when their GPU Regions have one layer each, and the measured allowed
  GPU VRAM. The 100 seeds are selected only after this feasibility check.
  These are still not the actual sizes of a high-E/larger-G completion.
- `prefix_gpu_pruned`, `prefix_full_memory_pruned`: continuations
  eliminated *before* the Beam is limited to eight. The first covers GPU
  Bank/KV memory on every Cycle; the second covers full memory on the
  terminal Cycle, including PC and Phone.
- `prefix_completed`, `duplicates`, `memory_rejected`,
  `profile_rejected`, `dag_rejected`, `simulated`: how many complete
  continuations reached each outcome, with all counts scoped to that seed.
- `best_prefix_layout`, `best_prefix_predicted_ms`,
  `best_prefix_final_E`, `best_prefix_E`: the best fully simulated
  continuation **before** optional Tensor-boundary refinement. These are
  blank if the seed had no successfully simulated continuation.
- `best_gpu_bank0_peak_mib`, `best_gpu_bank1_peak_mib`,
  `best_gpu_kv_mib`, `best_gpu_total_mib`: **actual full-layout** peak
  GPU Bank requirements for this seed's best simulated continuation.
  Unlike first-Cycle lower bounds, these include every selected Cycle.
  Blank when no successful continuation or no VRAM budget was measured.

The Planner JSONL record also reports `gpu_first_pairs_pruned`,
`gpu_first_shapes_pruned`, `prefix_gpu_pruned`,
`prefix_full_memory_pruned`, and `gpu_memory_budget_mib`. The old
`memory_rejected` column is now only for full plans that unexpectedly
fail a final repeat memory check; most memory exclusions happen earlier.
`prefix_completed` may be **less than 8 × first_cycle_seeds** because
memory-infeasible continuations no longer occupy the Beam.

The CSV rows are *not* the 100 best full-Prompt layouts: they are the 100
first-Cycle seeds chosen before prefix compensation. Compare
`best_prefix_predicted_ms` with the Planner JSONL `predicted_ms` to
understand why a uniform base plan or Tensor-boundary refinement won.

## Results

```text
experiment-results/
  YYYYMMDD-HHMMSS-PID/
    summary.csv
    prompt_64/
      config.json
      planner-first-cycle-top100.csv
      metrics.jsonl
      engine.stderr.log
      output.stdout.txt
      result.json
    prompt_128/
      ...
```

The experiment data logger is opt-in via
`LLAMA_REGION_RING_METRICS_FILE=<path-to-jsonl>`; it appends:

- `type=planner`: exact proposed and baseline layouts; E, predicted time,
  selected Chunk sizes and whether the plan was actually applied.
- `type=region_ring`: **one in-memory record per complete ubatch** after workers join;
  accumulated CPU/CUDA/Phone compute-range, graph preparation,
  output readback/synchronization, ready/gate waits, overlapped execution,
  GPU Bank refill, H2D and GPU Bank blocking. Records are flushed when the
  formal `llama_perf_context_print` call occurs. Warmup records are discarded
  by the official performance reset.
- `type=measurement_start`: emitted after warmup at the official timing
  reset; the CSV excludes pre-reset/warmup Region Ring ubatches.
- `type=perf`: native `llama_perf_context` load/prefill/decode time and
  **actual** token counts, written by `llama_perf_context_print`.
- Meta copy profile: one snapshot at the start and one after all workers join.
  `meta_pc_to_phone_copy_ms`, `meta_phone_to_pc_copy_ms`,
  `meta_phone_backend_ms` and `meta_phone_rpc_ms` are measured
  **cumulative Meta service/copy times**, and may overlap. They do not
  measure pure TCP socket transfer time or bytes for every boundary.

`-lv 1` suppresses informational/verbose diagnostics, while fatal/error
messages are saved to `engine.stderr.log`. No raw CLI content or detailed
per-Chunk traces appear in the terminal.

## Timing interpretation

- **Official prefill throughput** is `actual_prefill_tokens * 1000 / prefill_ms`.
- `ring_wall_ms` sums individual Region Ring ubatch wall times. It is a
  diagnostic, **not** the official prompt evaluation time.
- CPU, CUDA and Phone `compute_range_ms` include backend submit/wait/sync
  and are **not pure device-kernel times**.
- `output_sync_ms` includes output readback and synchronization; it is
  **not** equivalent to pure socket/network transfer time.
- `meta_pc_to_phone_copy_ms` and `meta_phone_to_pc_copy_ms` are observed
  Meta copies (including any underlying RPC work). They are **not** socket-only
  times, and must not be added to total device service without overlap handling.
- Per-device busy intervals, Prepare, H2D, output sync and waits overlap.
  **Never add them up to obtain prefill wall time.**
- GPU Bank A/B preload is included in model/load time, but not in Prefill.
  Rolling replacement H2D and GPU Bank wait appear separately.
- `estimated_{gpu_to_cpu,cpu_to_gpu,pc_to_phone,phone_to_pc}_activation_mib`
  are **modeled** FP32 hidden-state handoff payloads derived from the actual
  full G/C/P layout and measured ubatch token count. They are not observed
  socket bytes and exclude all extra graph/control traffic. They are blank
  for layouts containing Directional C2P/P2C Tensor boundaries.
- Pure PC-to-Phone socket transmit/receive and Phone GPU device execution
  require a separate detailed RPC-side diagnostic run. This lightweight
  benchmark intentionally does **not** enable per-Chunk Meta tracing.

The sweep launches exactly one inference process for each matching prompt.
For a fixed-plan length study, keep the same layout across inputs using
manual `--force-layout`; for an auto-plan study, let `apply-wide` replan
per input. Keep these two experiment types separate in the final report.

## Experiment validity checks

The runner marks a row `PASS` only when the subprocess exits normally,
official `type=perf` data is present, and the measured Region Ring records
are complete and successful. `NO_PERF`, `NO_RING`, `NO_METRICS`, and
`INCOMPLETE_RING` are distinct failure statuses; inspect the per-prompt
`engine.stderr.log` for details. All subprocess stdout/stderr are redirected
to files, so the terminal only shows a short per-prompt completion line.

The screenshot's `prompt_64.txt` through `prompt_512.txt` fit the default
`-b 512 -ub 512` input size. For larger prompts, take care: the official
runner may split prefill across multiple ubatches. Record
`ring_ubatches` and verify GPU Bank reinitialization/correctness for
three-or-more GPU Regions before treating multi-ubatch streaming results as
validated. Increase `-Batch` and `-UBatch` with the PowerShell wrapper
only when your PC/Phone memory budget supports larger graphs.

To run a reproducible fixed-plan study with the PowerShell wrapper:

```powershell
.\tools\region-ring-experiments\run_prompt_sweep.ps1 `
    -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
    -Rpc "10.39.52.114:50052" `
    -ForceLayout "G6,C2,P4,G6,C2,P4,G6,C2,P4,G6,C2,P4" `
    -ForceXC 32 -ForceXG 64 -ForceXP 128
```

The fixed example contains 48 layers; substitute a layout whose counts
sum to your actual model depth.
