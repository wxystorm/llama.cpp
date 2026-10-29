# Lightweight Region Ring prompt sweep

This runs each `prompt_N.txt` in the specified directory **once** in a fresh
`llama-completion` process, and saves everything to files. No Region/Chunk-level
logging is added to the hot path.

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

## Results

```text
experiment-results/
  YYYYMMDD-HHMMSS-PID/
    summary.csv
    prompt_64/
      config.json
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
- `type=region_ring`: **one record per complete ubatch** after workers join;
  accumulated CPU/CUDA/Phone compute-range, graph preparation,
  output readback/synchronization, ready/gate waits, overlapped execution,
  GPU Bank refill, H2D and GPU Bank blocking.
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
- Pure PC-to-Phone socket transmit/receive and Phone GPU device execution
  require a separate detailed RPC-side diagnostic run. This lightweight
  benchmark intentionally does **not** enable per-Chunk Meta tracing.

The sweep launches exactly one inference process for each matching prompt.
For a fixed-plan length study, keep the same layout across inputs using
manual `--force-layout`; for an auto-plan study, let `apply-wide` replan
per input. Keep these two experiment types separate in the final report.
