# Jupiter Phone-owned alignment with the Region Ring main design

Working branch: `feat/jupiter-phone-owned-main-align-20261123`
Base: `feat/jupiter-phone-owned-multishape-20261123`
Reference: `feat/phone-plan-overlap-multi-htp-20261109`

## Scope

The native Jupiter Qwen3 Prefill path constructs and executes the Phone suffix
from the Phone's local GGUF without transferring a PC GGML Graph. Its
`phone-qwen3-local-builder.h` is blob-identical to the reference branch;
`phone-npu-serial-executor.h` contains the same local graph precompiler.

This alignment branch preserves Jupiter's independent q(x,y)/DP and the
opt-in legacy RPC fallback. It DOES NOT replace Jupiter with Cycle Runtime.

## Implemented changes (requires rebuilding PC and Phone)

1. Native Phone-owned preparation uses the reference Region's one stable HTP
   scratch arena for all its Chunk Graphs; multiple Regions do **not** share
   the same scratch allocation. `GGML_RPC_PHONE_QWEN3_SHARED_SCRATCH=0`
   allows exact legacy per-Chunk allocation A/B. The shared arena is only
   reclaimed after the Hexagon fixed cache has been invalidated.
2. The PC graph boundary follows `LLAMA_JUPITER_PC_LAYERS` (32..39).
   The contiguous Phone suffix contains 1..8 Qwen3-14B Transformer layers.
   The default remains PC 0..31 and Phone 32..39.
3. The native Plan carries the actual layer count and Context/KV read span,
   with context 256..4096 tokens, divisible by 256, a maximum of 16 Chunk
   Graphs, and no more than 4096 tokens per Chunk. The benchmark runners
   expose `--pc-layers`, `--cpu-layers`, and `--context`.
4. S4 keeps its original mapping for the eight-layer pilot:
   `[0,0,1,1,2,2,3,3]`. A smaller suffix uses a contiguous, balanced
   static Session mapping. This is not a dynamic HTP scheduling optimizer.
5. The multi-Prompt harness no longer incorrectly equates a cache MISS with
   a confirmed eviction: server RESET/EVICT logs must establish the cause.

## Unchanged correctness restrictions

- Only dense Qwen3-14B, one fresh single-sequence Prefill, Flash Attention,
  PC final norm/head and one output token; **Decode remains unsupported**.
- A normal `PHONE_OWNED_RUN` still waits for the completed Phone suffix
  before the PC head can proceed. Phone-local per-Chunk multi-layer computation
  is autonomous within that call; the Region Ring task-queue/async PUSH-P2C
  transport is **not yet ported**.
- All HTP-owned KV allocations are private. No production Decode-KV handoff.
- Inference equivalence for new boundaries/contexts must be checked against
  the legacy Jupiter result with complete F32 logits; matching Top-1 alone is
  insufficient.
- Multi-shape Graph residency still depends on Phone memory budget and
  whole-generation cache invalidation; do not assume that A-B-A will HIT.

## Model-free check

```sh
cmake --build build-phone-owned --target llama-cli ggml-rpc-server \
  test-phone-owned-execution-plan test-phone-qwen3-local-builder
ctest --test-dir build-phone-owned --output-on-failure \
  -R 'test-phone-owned-execution-plan|test-phone-qwen3-local-builder'
python -m py_compile tools/jupiter/run_phone_owned_full_prefill.py \
  tools/jupiter/run_phone_owned_multi_prompt.py
```

## Example experimental PC/Phone smoke test

Build matching binaries on PC and Android from this branch.
Restart Phone RPC with `GGML_HEXAGON_NDEV=4`, the exact matching
`libggml-hexagon` and `ggml-rpc` binaries, and the correct local GGUF.

```powershell
python .\tools\jupiter\run_phone_owned_full_prefill.py `
  --exe ".\build-phone-owned\bin\Release\llama-cli.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --rpc "10.71.109.52:10140" `
  --prompt-file "E:\llama\datasets\sharegpt\sharegpt40\prompt_128_01.txt" `
  --sessions 4 --pc-layers 34 --cpu-layers 16 `
  --context 512 --target-tokens 175 --chunks 88,87 `
  --mode native --out-dir ".\jupiter-native-l34-175"
```

This example uses a **six-layer Phone suffix**, L34–39, while the PC
executes layers 0–33.

Check the Phone `[PHONE_OWNED_SCRATCH]` log for
`mode=REGION_SINGLE_SLOT`, `PREPARE_LAYER` for every assigned Layer,
`RUN_DONE` for every Chunk, and final PC `output=REAL_LOGITS`.

## Remaining integration to match the reference Runtime

The Region Ring reference uses `RPC_CMD_PHONE_INPUT_STREAM_RUN`,
Phone-owned task registration and input PUSH, session-aware execution
workers, and asynchronous P2C capture. The Jupiter native bridge instead
uses synchronous `RPC_CMD_PHONE_OWNED_RUN`. Porting that worker/task
interface requires separate, full integration of request IDs, readiness
fences, output lifetimes, queue cancellation/reset and KV ownership; it
is **not** supplied by the changes in this branch.

Do not replace the working Jupiter execution path or report that
asynchronous runtime alignment is complete until the new transport has
passed PC+Phone end-to-end parity and repeated-request tests.
