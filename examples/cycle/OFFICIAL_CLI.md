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
git pull --ff-only origin feat/cycle-runtime-integration-20261023
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
This is not yet the interactive multi-turn, chat-template-driven server UI.

## Exact present limitations

Only the 48-layer Qwen3-30B-A3B fixed `X=4,Y=2` layout, one sequence,
one raw completion prompt, <=64 tokens per `llama_decode` invocation,
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
