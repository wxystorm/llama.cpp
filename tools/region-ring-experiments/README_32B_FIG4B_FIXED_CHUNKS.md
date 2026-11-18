# Fig. 4(b): resident Full Cycle 32B, fixed Chunk 64/128/256, no-mmap

This experimental branch is based on **`feat/region-ring-sharegpt50-short-first-long-20261116`**, the actual 32B ShareGPT50_short Full Cycle implementation. It does **not** use the older `llama-completion` experiment branch. Measurements use the same persistent `llama-server` harness as the main-study server results, with only **request-time Chunk choice** fixed.

The Fig. 4(b) wrapper runs 3 input-length groups (128, 512, 1024), one actual ShareGPT file per group, 3 fixed widths (64/128/256), and 3 repetitions by default: **9 configurations and 27 measurements**. Every `(Chunk, repetition)` gets its own resident server, an independent run fingerprint and audited request logs.

## Parity matrix

| Item | Fig. 4(b) policy |
|---|---|
| Model | `Qwen3-32B-Q4_K_M.gguf`, 64 layers |
| Placement | `G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7`, permanently locked (43 GPU + 21 Phone HTP) |
| Data | `sharegpt50_short` from the main study |
| Execution | Persistent `llama-server` with Phone HTP via RPC, same native request hook as main study |
| Model mapping | **`--no-mmap` always ON** |
| Context, Batch, UBatch | **2048 / 2048 / 2048** |
| CPU / concurrency | **8 threads**, parallel=1 |
| Flash attention / split | ON / tensor |
| Prefill | `n_predict=1`, same seed 1234, no conversation-style prompt prefix added |
| Prefix cache | `--no-cache-prompt`, `--cache-ram 0`, `--ctx-checkpoints 0`; every response must report cache_n=0 |
| Warmup | `--no-warmup` like the resident main-study server |
| Context shift / continuous batching | Both OFF |
| Runtime | Two GPU banks, asynchronous prefetch, Drain-First, resource-aware scheduling |
| Phone | Graphless autonomous HTP execution using the same profile, binary and RPC path as main study |
| Fixed width | Native `LLAMA_REGION_RING_FIXED_CHUNK_ABLATION=64/128/256` with a forced GPU mover for short prompts; **no dynamic request-time Chunk search** |
| Post-processing | Median of 3 measured `prompt_ms` per group/Chunk; groupwise normalization |

**Important historical distinction:** The original resident `probe_htp_llama_server.py` server command had *mmap enabled* because it did not pass `--no-mmap`. The earlier `run_32b_all40.py --main50-short` **CLI** workflow did use `--no-mmap`. The new Fig. 4(b) results will follow your requested **no-mmap** policy, but to compare against a previously collected **resident-server** Full Cycle baseline *strictly*, rerun that server baseline with `--no-mmap` too. Otherwise mmap policy differs even though the scheduler/layout/batch match.

## Build / run (Windows PowerShell)

Start the matching Phone Qwen3-32B HTP RPC service first (normally `10.238.229.239:50052`).

```powershell
$root = 'E:\llama\llama-tensor-v2'
cd $root
git fetch origin
git switch exp/fig4b-32b-main50-parity-20261118
git pull --ff-only
cmake --build .\build-tensor-v2 --config Release --target llama-server -j 8

$opts = @(
  '--server-exe', "$root\build-tensor-v2\bin\Release\llama-server.exe",
  '--model', 'E:\llama\models\Qwen3-32B-Q4_K_M.gguf',
  '--rpc', '10.238.229.239:50052',
  '--prompt-dir', 'E:\llama\datasets\sharegpt\sharegpt50_short',
  '--profile-source', "$root\32b-sharegpt50-short-mmap-test\independent-profile",
  '--out', "$root\fig4b-32b-chunk64-128-256-nommap",
  '--groups', '128', '512', '1024',
  '--chunks', '64', '128', '256',
  '--threads', '8',
  '--port', '18080'
)

# Inspect commands; nothing executes.
python .\tools\region-ring-experiments\run_32b_fig4b_fixed_chunks.py @opts --dry-run

# First pass: 9 measurements.
python .\tools\region-ring-experiments\run_32b_fig4b_fixed_chunks.py @opts --repeats 1

# Full run: 27 measurements, preserving previously completed PASS results.
python .\tools\region-ring-experiments\run_32b_fig4b_fixed_chunks.py @opts --repeats 3
```

The **profile source** must contain the validated native `profile.csv` and `profile.json` for this exact 32B Qwen3 hardware pair. The path shown matches the main-study README example; replace it if your measured Profile lives elsewhere. Do **not** switch Profiles across Chunk cases. The `--port` must be free; this harness starts/stops the resident PC server automatically.

This runner always sets `--no-mmap`. No user override, and audit rejects an executable command that lacks the flag. Existing results from the previous fixed 32/64/128 experiment under `fig4b-32b-main50-nommap` (and from the earlier CLI runner) must **not** be mixed into this new 64/128/256 sweep. The wrapper detects the prior `chunk_032` directory and refuses that `--out`; use the new directory shown above.

## Verify and plot

Each `chunk_064/repeat_01/` (and similarly named) case contains:
- `server-run-config.json` with server command, runtime flags, dataset/model/Profile fingerprints;
- `requests.csv` and `requests.jsonl` for three selected prompts;
- `llama-server.log` and resumed logs if needed;
- `summary.csv` and `summary.json` reporting 3 expected requests.

The top-level wrapper writes:
- `fig4b_raw.csv`: all 27 sample slots, real tokens, timings and validation errors;
- `fig4b_summary.csv`: median Prefill milliseconds by prompt group and fixed Chunk;
- `fig4b_plot_normalized.csv`: exactly 3 rows for the paper. Empty unless every requested sample passes validation.

The full server harness verifies actual `cache_n=0`, the exact locked layer placement, native `FIXED_CHUNK_ABLATION` decision and execution, and correct Drain-First/prefetch mode. Fig. 4(b) additionally checks the `--no-mmap` flag, session settings, profile configuration, real timing fields and cross-case token consistency. Native tiny-tail coalescing can make `runtime_xc/xg/xp` differ slightly from requested `plan_xc/xg/xp`; do not change the requested fixed Chunk to compensate.

Normalization for each input-group `g` and fixed width `c`:

```text
T(g,c) = median over repetitions of measured server prompt_ms
normalized(g,c) = T(g,c) / min(T(g,64), T(g,128), T(g,256))
```

**Short-prompt policy:** When the actual token count is smaller than the target Chunk width, the native fixed-width hook processes the entire prompt as a **single Chunk**. For example, the earlier `prompt_128_01.txt` had 167 actual tokens: target `256` becomes `XC=XG=XP=167` (the native selected plan), not a 256-token Chunk or an error. The GPU double-bank mover remains forced on. Runtime tiny-tail coalescing may still adjust the physical width within normal validation rules.

Crossing curves are **not** assumed; they are the hypothesis being measured.

## Rerun the resident Full Cycle main baseline with identical no-mmap

If your existing main-study server baseline was run with mmap, use a **new output directory** to obtain a genuinely comparable no-mmap baseline:

```powershell
$baseline = @(
  '--server-exe', "$root\build-tensor-v2\bin\Release\llama-server.exe",
  '--model', 'E:\llama\models\Qwen3-32B-Q4_K_M.gguf',
  '--rpc', '10.238.229.239:50052',
  '--prompt-dir', 'E:\llama\datasets\sharegpt\sharegpt50_short',
  '--profile-source', "$root\32b-sharegpt50-short-mmap-test\independent-profile",
  '--out', "$root\32b-server-dynamic-all50-nommap",
  '--all50', '--per-request-chunks', '--no-mmap'
)
python .\tools\region-ring-experiments\probe_htp_llama_server.py @baseline
```

Do not compare the previous mmap-on server Prefill figures directly with no-mmap results and describe them as all-other-settings-equal.

## Earlier Chunk=32 measurements are preserved

The preceding 32/64/128 sweep wrote to `fig4b-32b-main50-nommap`.
Its first 3 server measurements passed, but the wrapper initially rejected
`runtime_xc=0` because the layout has no CPU compute regions. That audit
bug was corrected separately, and **those older files should be preserved
unchanged**.

For the new 64/128/256 sweep, use the **new output folder** from the command
above, rebuild `llama-server` (native C++ now explicitly accepts fixed Chunk
256), then run `--repeats 1` or `--repeats 3`. Resuming within the
**new** folder uses the same verified `--resume` mechanism, but old Chunk=32
measurements are not reused.

## Offline tests

```powershell
python -m unittest discover -s .\tools\region-ring-experiments -p test_run_32b_fig4b_fixed_chunks.py -v
```

These tests are offline checks only; physical PC+Phone execution and the Phone's actual memory feasibility still must be verified on hardware.
