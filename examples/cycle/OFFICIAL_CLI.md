# Running the fixed physical Cycle via official `llama-cli`

The current `llama-cli` normally runs as an HTTP client backed by a
local `llama-server` process. This change adds an **explicit, opt-in**
Cycle-only local-completion path in that SAME executable.

- Standard argument parsing, `--rpc` registration and model loading.
- Standard `llama_context` / `llama_decode()` / Logits public API.
- Original five-queue 48-layer Cycle runtime and physical KV, now supporting
  explicit 256/512-token capacities (128 remains the default).
- Standard **external** sampler: honors `--temp`, `--seed`, top-k,
  top-p and configured penalties; not hardcoded greedy.
- **Does not start the server** or borrow the server's multi-slot KV.
- When opt-in is absent, normal `llama-cli` HTTP/server/UI mode is
  unchanged.

## Windows example

```powershell
git fetch origin
git switch feat/cycle-production-cuda-profile-20261024
git pull --ff-only origin feat/cycle-production-cuda-profile-20261024
cmake --build .\build-cycle-test --config Debug --target llama-cli

$env:LLAMA_HYBRID_CYCLE_PC_LAYERS = "4"
$env:LLAMA_HYBRID_CYCLE_PHONE_LAYERS = "2"
$env:LLAMA_HYBRID_CYCLE_STANDARD_DECODE = "1"

.\build-cycle-test\bin\Debug\llama-cli.exe `
    -m E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
    --rpc 10.39.52.114:50053 `
    --split-mode tensor `
    -c 128 -n 4 -t 8 `
    --temp 0 --seed 1234 `
    --no-mmap `
    -p "Hello, how are you?" `
    2>&1 | Tee-Object -FilePath ".\cycle-official-cli.log"
```

For **multiple RPC devices**, explicitly select the CPU and the intended
phone GPU with `--device CPU,<exact-RPC-device-name>`. Use
`llama-cli --list-devices` to inspect names after registering the
appropriate RPC server(s). The physical weight loader still requires
exactly PC CPU + Phone GPU.

Without `-p`, direct Cycle mode reads **one line** from terminal input.
This is not an interactive multi-turn chat or server UI. Single-turn
model Chat Template formatting is supported by the Cycle direct CLI.


## Single-turn model Chat Template: correctness and raw regression

The official Cycle CLI now formats a **single user message and an unfinished
assistant prefix** using the same `llama-common` model template renderer
as regular llama.cpp. In the default `AUTO` mode, it reads the GGUF model's
chat template if present; if no template is available, it preserves raw
completion. This does not enable an HTTP server, interactive chat, multiple
sequences or a second KV cache.

The CLI does **not** manually concatenate Qwen3 special markers. It asks
`common_chat_templates_apply()` for the model-formatted prompt with
`add_generation_prompt=true` and then calls `common_tokenize()` with
special-token parsing **enabled**. The result goes to the same physical
48-layer `llama_decode()` pipeline and the same external sampler.

For a normal single user instruction:

```powershell
# Baseline for output semantics; no PC CUDA until correctness is checked.
$env:LLAMA_HYBRID_CYCLE_CUDA_BUDGET_MIB = "0"
.\build-cycle-test\bin\Debug\llama-cli.exe `
    -m E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
    --rpc 10.39.52.114:50053 --split-mode tensor `
    -c 256 -n 32 -t 8 --temp 0 --seed 1234 `
    --no-mmap --no-display-prompt `
    -p "What is the capital of France?"
```

Look for `[CYCLE_OFFICIAL_CLI] phase=PROMPT_FORMAT
mode=MODEL_CHAT_TEMPLATE ... special=1 status=PASS`. The token count
may be greater than for the same raw text, because message delimiters and
the assistant prefix are part of the physical KV input.

For **exact historical raw prompt regression**, append
`--no-conversation` (or `-no-cnv`) to the same command. It selects
`mode=RAW_COMPLETION` and retains the earlier tokenization behavior.
`--no-display-prompt` only affects terminal echo; it does **not** change
the model's input tokens. `--conversation` explicitly requests a template;
if the model has none, this mode fails instead of inventing one. A
custom `--chat-template` may be supplied using the existing upstream flag.

To debug the exact rendered text, set
`LLAMA_HYBRID_CYCLE_VERBOSE=1`; the formatted prompt is printed on
stderr between `[CYCLE_CHAT_TEMPLATE] formatted_prompt_begin/end` markers.
Use `LLAMA_HYBRID_CYCLE_TOKEN_TRACE=1` independently to print exact
`[CYCLE_TOKEN_TRACE] phase=PROMPT/GENERATE index=... id=...` records, so
both backends can be checked against the same prompt Token IDs and generated
IDs (including EOG). The log never strips or rewrites generated text.
Do not use verbose or Token ID tracing during performance measurements.

**Important numerical comparison:** chat-formatted and raw prompts contain
*different Token IDs* and therefore naturally produce different logits.
For correctness tests compare Cycle against the CPU reference with
**the same formatted prompt and token sequence**, `--temp 0`, and the
same seed. If the only difference is an unwanted `A:` under raw mode,
that is not by itself evidence of a numerical bug. This template change
has not yet been compiled or hardware-verified.


## Exact present limitations

Only the 48-layer Qwen3-30B-A3B fixed `X=4,Y=2` layout, one sequence,
one single-turn **model Chat Template or raw completion** prompt, <=64 tokens per `llama_decode` invocation,
<=512 total physical KV, and only final-token Logits are supported.

The direct CLI evaluates the prompt in <=64-token batches and generates
up to the requested `-n`, provided `prompt_tokens + n_predict - 1 <=
n_ctx <= 512`. If the request exceeds capacity it **fails explicitly**;
it does not silently shift, truncate, replay or edit physical KV.

`--server-base`, multiple sequences, server-slot state, multimodal input,
LoRA/control vector hot-swap, and interactive chat are not yet integrated.
Warm-up, model-memory fitting and standard KV shift are intentionally
disabled for Cycle only. Standard `llama_memory` KV operations are NOT
interchangeable with Cycle physical KV.

`llama-cycle-standard` and `llama-cycle-cli` remain regression tools.
Before calling the official entry complete, verify the new
`[CYCLE_OFFICIAL_CLI]` trace on Windows/Phone. The official 128-token route has been reported to pass on real PC/Phone
hardware; the **256/512-token extension is not yet hardware-verified**. Existing CPU/GPU
numeric differences are not altered or reinterpreted by this migration.

## Larger physical KV: incremental verification

The former 128-token API and CLI behavior is preserved when using
`-c 128` or omitting `-c`. To allocate additional persistent physical K/V,
request `-c 256` or `-c 512` explicitly. The model is still loaded once;
each new Token appends K/V to its actual absolute position on all 48 layers.
The existing per-evaluation 64-token and per-Chunk 16-token limits remain.

```powershell
# Check the existing regression before the extension:
.\build-cycle-test\bin\Debug\llama-cli.exe `
    -m E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
    --rpc 10.39.52.114:50053 --split-mode tensor `
    -c 128 -n 4 -t 8 --temp 0 --seed 1234 --no-mmap `
    -p "Hello, how are you?" `
    2>&1 | Tee-Object -FilePath .\cycle-cli-128-regression.log

# Test a multi-Chunk Prompt crossing the old 128-token boundary.
# Inspect the printed prompt_tokens count; change 140 repetitions if
# tokenization yields <=128 or >256 tokens.
$longPrompt = ("hello " * 140).Trim()
.\build-cycle-test\bin\Debug\llama-cli.exe `
    -m E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
    --rpc 10.39.52.114:50053 --split-mode tensor `
    -c 256 -n 4 -t 8 --temp 0 --seed 1234 --no-mmap `
    -p $longPrompt `
    2>&1 | Tee-Object -FilePath .\cycle-cli-cross-128.log
```

For the second run, check that `prompt_tokens > 128` (and <=252) and
the final `[CYCLE_SESSION] phase=COMMIT` reaches the matching absolute
KV prefix. This validates historical K/V and causal masks beyond the old
128-token limit without requiring 140 single-token decoding steps.
A normal EOS token may end generation early; the long Prompt must itself
cross 128. After 256 succeeds, retry the same Prompt with `-c 512`.

For `n_ctx > 128`, old Attention Graph metadata caching is disabled for
monotonically increasing KV shapes (which cannot repeat within the same
session); the existing Attention Scratch Pool, Phone FFN cache and data-plane
Buffer Ownership remain active. This avoids retaining 16 large, never-reused
host graph metadata allocations per physical layer. No change is made to
128-token session reuse behavior.

## Official Prompt / Eval performance and UI

Cycle direct mode now uses the **same** `llama_perf_context()` counters
and `common_perf_print()` formatter as ordinary llama.cpp. Timed work
includes token Embedding, the complete synchronous 48-layer five-FIFO,
exposed PC/Phone RPC waits and output RMSNorm / LM Head. The one-way
`GRAPH_COMPUTE` socket-submit time is **not** used as eval duration.

With the default `--show-timings`, performance accounting enabled, and
`--color on` on a compatible terminal, the end of generation shows
official load / prompt eval / eval / sampling / total time and the same
purple UI summary as standard llama-cli:

    [ Prompt: 10.0 t/s | Generation: 3.0 t/s ]

These numbers are **examples**, not measured results. Detailed timing
comes from official `common_perf_print`; the purple line uses the
upstream `ui::show_info`. The ordinary `graphs reused` field does NOT
include Cycle's private Attention / FFN graph-cache hit counters.

The last generated token does not need a further decode, so 4 generated
tokens normally correspond to **3 eval runs**. Generation t/s refers
to those actual evaluated runs, following llama.cpp's performance
convention. If no incremental eval has occurred (such as `-n 1`),
the summary says `Generation: n/a` instead of displaying infinity.

`--no-show-timings` suppresses the timing report and UI summary.
`--no-perf` disables internal performance tracking, and the Cycle
CLI respects it. `--color off` disables terminal colors without
disabling performance text. When output is piped, color support is
also terminal-dependent.

The official CLI explicitly sets Prompt phase for *all* Prompt chunks,
including the final single-token chunk of a 64+1 split. This avoids
incorrectly attributing the one-token tail to Decode. Non-Cycle
contexts and the normal server-based CLI path are unchanged.

## Cycle physical memory reporting

Cycle Contexts deliberately omit the standard GGML Scheduler. The
original `common_perf_print()` nevertheless invokes
`common_memory_breakdown_print()` after reporting `graphs reused`.
The default `llama_context::memory_breakdown()` previously called
`ggml_backend_sched_get_buffer_type(nullptr, ...)`, causing
`GGML_ASSERT(sched)` at the end of a successful generation.

The Cycle branch now enumerates actual session-owned physical buffers
instead of consulting a nonexistent Scheduler. The standard memory
categories include persistent K/V as `context`, retained Attention
Scratch Pool buffers as `compute`, and optional CUDA weight mirrors
as `model`, grouped by their actual backend buffer types. Original
model weight usage remains included. This is *not* an exhaustive
accounting of transient graph allocations or OS/network overhead.

A passing run reports:

    [CYCLE_MEMORY_BREAKDOWN] scheduler=NOT_CREATED kv_bytes=... scratch_bytes=... cuda_mirror_bytes=... scope=SESSION_OWNED_BUFFERS status=PASS

and then reaches the official purple Prompt/Generation summary, without
the former scheduler assertion. The default Context's original
Scheduler-based breakdown remains unchanged.


## Experimental production CUDA and five-queue profiling

The production Cycle Context now accepts an **opt-in CUDA mirror budget**.
The unset or zero setting preserves the verified CPU + Phone GPU route.

```powershell
# Unset this for the existing tested CPU + Phone GPU baseline.
$env:LLAMA_HYBRID_CYCLE_CUDA_BUDGET_MIB = "1536"
# Run the same official llama-cli command as above.
```

The selected PC GPU is the first registered **local CUDA backend** (not the
Phone RPC GPU). The existing resource pool estimates weights for PC Full layers,
reserves 512 MiB of reported free VRAM, and mirrors **only** layers that fit the
requested budget. All unselected PC Full layers retain CPU execution. This is
**not** an automatic Planner, and the model's CPU/Phone physical weight layout
and 48-layer Cycle pattern do not change. For reproducibility, compare the
`[CYCLE_FULL_RESOURCE_PLAN]` line and record `cuda_mirror_layers`,
`cuda_estimated_mib`, and `pc_cpu_fallback`. A nonzero budget with no
registered CUDA backend fails instead of silently changing the experiment.

When `LLAMA_HYBRID_CYCLE_QUEUE_PROFILE=1` is set, the official forward
prints one `[CYCLE_CORE_QUEUE]` per logical worker after each FIFO invocation: task count, callback execution total/maximum,
queue wait total/maximum, and idle time. These are **PC host-side observations**:
they overlap across queues and must never be summed to produce end-to-end time.
A Phone GPU callback may include network and remote synchronization; it is
not a kernel-only device timer. The existing `llama_perf_context()` remains the
source of official end-to-end Prompt/Eval elapsed time.

**This optional CUDA path and new queue measurements require Windows and Phone
real-device regression before being treated as verified.** Test the zero-budget
baseline first, followed by a small nonzero budget that fits one or two mirrored
PC Full layers. Do not start with a budget intended for all 24 PC Full layers.
Confirm that all 48 KV prefixes, READY/PARTIAL ACKs and final logits continue
to complete and compare prompt/eval timings under the same input.


## Quiet generated text and opt-in diagnostics

Successful per-layer/per-Chunk Cycle tracing is now **off by default** so the
official CLI's sampled tokens on `stdout` are readable. The regular completion
status, resource-plan summary, standard Prompt/Eval statistics, error messages
and correctness/parity failures remain available. Cycle's direct
`std::printf` diagnostics were never controlled by the upstream `-lv`
argument; `-lv` only configures the normal logging system.

```powershell
# Quiet generation and correctness regression (default).
Remove-Item Env:LLAMA_HYBRID_CYCLE_VERBOSE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_CYCLE_QUEUE_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_CYCLE_STAGE_PROFILE -ErrorAction SilentlyContinue

# When investigating a wrong result, turn on all successful stage traces:
$env:LLAMA_HYBRID_CYCLE_VERBOSE = "1"
# When measuring bottlenecks, request real five-worker host-side queue metrics:
$env:LLAMA_HYBRID_CYCLE_QUEUE_PROFILE = "1"
# Optional existing Attention/FFN stage timing; unset to disable.
$env:LLAMA_HYBRID_CYCLE_STAGE_PROFILE = "1"
```

Unlike verbose logs, these options only change observation/printing; they do
not change KV ownership, worker scheduling, attention graphs, precision or
sampler output. The queue timeline is not collected unless queue profiling
is enabled. Hardware performance comparisons should leave detailed verbose
stage traces off to avoid log I/O overhead. `status=FAIL` from computational,
transport and ownership validation is still printed independently of
verbose mode.
