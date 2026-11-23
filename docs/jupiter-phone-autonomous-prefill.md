# Jupiter Phone-owned autonomous Prefill (no Decode)

Branch: `feat/jupiter-phone-autonomous-prefill-20261123`
Base: `feat/jupiter-phone-owned-main-align-20261123`
Reference semantics: `feat/phone-plan-overlap-multi-htp-20261109`

This migration deliberately keeps Jupiter's own CPU/CUDA/Phone layer planning
and the proven Phone-local Qwen3 HTP Graph Builder. It does not import Cycle
Runtime or the Region Ring planner, and **does not implement Decode**.

## New Prefill-only execution path

Enable on PC with `LLAMA_JUPITER_PHONE_OWNED_AUTONOMOUS=1` or
the harness flag `--autonomous-phone` (mutually exclusive with
`--async-phone`). Both binaries MUST come from this branch.

1. The PC builds its CPU/CUDA prefix normally; the Phone's suffix
   Graphs/HTP fixed compilations and private F16 KV are prepared from the
   locally loaded GGUF, as in the previous tested native Jupiter path.
2. Phone `AUTO_START` registers exactly one Prefill stream:
   run ID, independent P2C run ID, Region, Chunk count and bounded queue.
3. Each PC prefix Chunk sends `AUTO_PUSH` with an immutable
   F32 activation payload and its Chunk index. The ACK indicates **queue
   admission**, not HTP completion; a two-Chunk backpressure window stops
   unbounded host memory growth.
4. The **Phone** worker, not a PC worker issuing serial RUN RPCs, consumes
   requests in FIFO order. It calls the same local HTP multi-layer executor
   and preserves private KV dependencies.
5. Only the final Chunk's real hidden state is retained. The PC final
   custom node blocks once on the independent `AUTO_P2C_WAIT`
   channel; the Phone responds on the completion event. Earlier full
   hidden-state outputs are not sent over Wi-Fi.
6. The PC executes the original final RMSNorm/LM Head, producing the same
   vocabulary logits as synchronous Native mode.

The existing synchronous `PHONE_OWNED_RUN` path and
PC-side `--async-phone` worker remain intact for regressions.

## Task safety and Graph lifetime

- Registration is strict; PUSH validates run ID, P2C ID, Region, Chunk FIFO,
  activation dimensions and the declared input token count.
- One autonomous run is active at a time (the same restriction as the
  existing native Jupiter pilot); no overlapping independent Prompts.
- Queue depth is configurable at the RPC API and defaults to 2 in Jupiter.
- Phone task and P2C waits have a 120-second deadline. Failures cause a
  closed stream, not silent results from the next request.
- RESET joins the worker **before** the HTP fixed-Graph/KV/Scratch cache can
  be released or retained. Only completed runs are eligible for residency.
- On a later PREPARE, a failed stream or one idle for five minutes is
  canceled and reset to recover after a lost client or exception.
- Same-shape HTP Graph reuse, budgeted standby templates and session-wide
  invalidation remain under the existing Phone-owned cache policy. Graph
  cache eviction must be established from Phone logs; a PC-side MISS alone
  does not explain why the resource was released.
- A hard process/Phone crash may still require server restart. This is
  not general-purpose multi-tenant request scheduling.

## Protocol and compatibility

This branch increments Jupiter RPC patch from 4.0.5 to **4.0.6** and adds
three opt-in commands: AUTO_START, AUTO_PUSH and AUTO_P2C_WAIT. The
patch/HELLO check must reject old PC/Phone mixed binaries in native mode.
All old command numbers remain unchanged because the new commands are
appended. This transport is Jupiter's **Prefill-only equivalent of the
main design's Phone task stream semantics**, not a byte-for-byte copy of
the main Region Ring RPC wire format.

## Build and model-free tests

Rebuild BOTH PC and Android RPC binaries from this branch and restart the
Phone server with the same local Qwen3-14B GGUF and HTP 1–4 sessions.
On a PC configure/build tree with GGML_RPC=ON:

```sh
cmake -S . -B build-phone-owned -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-phone-owned --target llama-cli ggml-rpc-server \
  test-phone-owned-autonomous-queue test-phone-owned-execution-plan \
  test-phone-qwen3-local-builder
ctest --test-dir build-phone-owned --output-on-failure \
  -R 'test-phone-owned-autonomous-queue|test-phone-owned-execution-plan|test-phone-qwen3-local-builder'
python -m py_compile tools/jupiter/run_phone_owned_full_prefill.py \
  tools/jupiter/run_phone_owned_autonomous_ab.py \
  tools/jupiter/run_phone_owned_multi_prompt.py
```

Run the original 175-token shape on Windows. Keep the Phone RPC server
alive between sync and autonomous variants:

```powershell
python .\tools\jupiter\run_phone_owned_autonomous_ab.py `
  --exe ".\build-phone-owned\bin\Release\llama-cli.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --rpc "<phone-ip>:10140" `
  --prompt-file "E:\llama\datasets\sharegpt\sharegpt40\prompt_128_01.txt" `
  --sessions 4 --pc-layers 32 --cpu-layers 16 `
  --context 512 --target-tokens 175 --chunks 88,87 `
  --out-dir ".\jupiter-phone-autonomous-ab"
```

The report is `autonomous-ab-results.json`. Require
full F32 logits parity (top1/cosine/relative-RMSE), not merely identical
generated text. For performance analysis, verify both variants have
comparable PREPARE cache classes and repeat in reverse order with
`--reverse-order`. One warm pair is insufficient for a
statistically supported speedup claim.

On Phone inspect these events: `[PHONE_AUTO] START`,
`INPUT_PUSH`, `EXEC_DONE`,
`P2C_RETURN`, `STOP`, and the
existing `PREPARE_CACHE HIT/MISS/INVALID`,
`CACHE_EVICT`, `RESET graph_cache=...`.

## Next acceptance matrix (real Phone required)

1. Native Sync vs Auto: 14B, L32–39, S4, context 512, 175 tokens,
   two Chunks. Every final logit must agree.
2. Repeat with `--pc-layers 34 --context 1024` (six HTP
   layers); then 36 and 39 to exercise four- and one-layer suffixes.
3. Run 1024/2048/4096 context shapes with valid Chunk plans of at most 16
   Chunks, within actual Phone memory caps; compare full logits.
4. Run multiple Prompt shapes in A→B→A sequence using
   `run_phone_owned_multi_prompt.py --autonomous-phone --revisit-first`.
   Verify complete FIFO, KV isolation, cache HIT/MISS and eviction reasons.
5. Simulate a failed RPC/aborted input, then submit another Prompt to
   confirm worker join and Graph/KV recovery.
6. Compare measured Phone throughput and PC-prefix/Phone-suffix overlap
   against both synchronous native and PC-worker asynchronous native.

These tests have been added or described but **cannot be claimed PASS
until executed on the user's Windows PC and Snapdragon Phone**.

No Prefill-to-Decode KV handoff is implemented or required.
