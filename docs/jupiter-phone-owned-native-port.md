# Jupiter: Phone-owned Native Qwen3-14B Full Prefill (experimental)

Base: `feat/jupiter-htp-direct-rpc-20261112`  
Ported from: `feat/phone-plan-overlap-multi-htp-20261109`  
Working branch: `feat/jupiter-phone-owned-graphless-20261122`

## What is implemented

A complete **single fresh Prefill forward pass** is wired into the **real**
`llama-cli` / `llama_context::decode()` path, producing the final PC
Qwen3 logits for the first token:

1. PC CPU runs Transformer layers **0–15**.
2. PC CUDA runs Transformer layers **16–31**.
3. One GGML CPU custom operation transfers the exact F32 layer-31 hidden
   state to Phone and invokes `PHONE_OWNED_RUN`.
4. Phone-owned native HTP graphs run layers **32–39** in order, retaining
   per-layer private F16 KV across the Prompt chunks.
5. The last hidden state comes back to PC, which computes the original
   `output_norm` and `output.weight` LM Head on CUDA. The returned logits
   are consumed by the regular llama.cpp sampling code.

No GGML RPC Graph for layers 32–39 exists in the PC Scheduler during native
mode: the HTP suffix weights are instead loaded from Phone's local GGUF.
Unused PC model copies of those suffix weights are kept on CPU memory,
not allocated through RPC/HTP; final LM Head weights are placed on PC CUDA.

The native bridge is dynamically resolved via the GGML RPC backend registry.
It can be disabled by unsetting `LLAMA_JUPITER_PHONE_OWNED_NATIVE` (or
setting it to `0`), leaving the legacy Jupiter path unchanged.

### Limitations (important)

**The original S1/S4 175-token native Prefill has completed on physical
PC/Phone with the expected final Top-1. Cross-request Graph residency is a
new implementation and still requires a Phone rebuild and repeat-run validation.**

- Strict pilot: dense Qwen3-14B, 40 layers, hidden size 5120, 40 Q heads,
  8 KV heads, head dimension 128, context exactly **512**, Flash Attention
  enabled, fresh 0-based contiguous single-sequence Prompt, 1 logits output.
- HTP session count **1 or 4**. S4 assignment: L32–33/HTP0,
  L34–35/HTP1, L36–37/HTP2, L38–39/HTP3.
- The native Phone KV is not yet copied into the legacy llama.cpp Decode
  cache. **Generate only one token (`-n 1`)**; later Decode is intentionally
  rejected, rather than silently producing incorrect tokens.
- The PC still allocates CPU/CUDA Scheduler graphs for the first 32 layers
  and output head; only the **remote HTP suffix** is removed.
- The **first** native request still performs Phone-side weight load,
  Graph allocation and compilation. The **next matching** request can reuse
  the resident fixed Graphs after clearing the private F16 KV. Unmatched
  topologies fall back to cold Graph preparation. Model-startup prewarming
  remains separate work.
- Real logits output doesn't imply numerical parity. Compare with baseline
  using the supplied binary logits A/B runner before reporting speedup.
- No async overlap, prefix reuse, KV shift, additional sequences, LoRA,
  or automatic mixed-context decoding in the pilot.

## Cross-request Phone-owned HTP Graph residency

The Phone RPC process owns a persistent `rpc_server` shared by independent
client connections. When a **completed** native run sends `PHONE_OWNED_RESET`,
the server discards its logical request ID but retains one bounded set of
fixed-UID HTP Graphs, immutable weights, KV allocations and graph scratch.

For the next `PHONE_OWNED_PREPARE`:

- **HIT:** every scalar that affects graph construction matches (including
  token count, chunk size, KV capacity, layer and attention parameters),
  and the per-layer HTP Session mapping is identical. The server synchronizes
  each Session, **zeros every layer's F16 K/V buffer**, resets the per-layer
  chunk cursor, and maps the new logical run ID to the original physical
  Graph key. All model load, Graph build, allocation and precompile phases
  are skipped. The RPC message ABI is unchanged.
- **MISS:** a changed execution template or device mapping invalidates
  the old fixed Hexagon cache *before* releasing its Graph/KV/scratch buffers;
  the request then builds new Graphs. Only **one** template is resident at a
  time; this is not a multi-shape cache.
- **FAILED / INCOMPLETE:** RESET releases temporary Graphs rather than caching
  a partial run. The server rejects overlapping native requests until the
  previous run resets. The regular legacy RPC path is unchanged.

With persistence enabled (the default), Phone RPC logs:

```text
[JUPITER_PHONE_OWNED] RESET ... graph_cache=RESIDENT ...
[PHONE_OWNED_RESIDENCY] event=KV_RESET layer=32 ... status=PASS
...
[PHONE_OWNED_RESIDENCY] event=KV_RESET layer=39 ... status=PASS
[JUPITER_PHONE_OWNED] PREPARE_CACHE status=HIT ... graphs=16 ...
```

On a warm hit the PC `PREPARE_DONE` report must show
`load_ms=0.000 build_ms=0.000 buffer_ms=0.000 precompile_ms=0.000`,
with nonzero `graphs=16` and a small `pc_wall_ms` for RPC + KV reset.
Do not treat weight-pagecache hits alone as proof of Graph reuse.

To opt out and reproduce the original release-on-RESET path, set
`GGML_RPC_PHONE_OWNED_PERSIST=0` **before starting** `ggml-rpc-server`.
The Phone RPC process must stay running across requests; restarting it
discards all resident HTP resources.

### Cross-process validation (real PC + Phone)

After pulling the new code, **rebuild and restart the Phone RPC Server**;
rebuilding only the Windows client does not enable residency. Keep the same
Phone server alive while running three independent `llama-cli` processes:

```powershell
cd E:\llama\llama-jupiter
python .\tools\jupiter\test_phone_owned_residency.py `
  --exe ".\build-phone-owned\bin\Release\llama-cli.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --rpc "10.71.109.52:10140" `
  --prompt-file "E:\llama\datasets\sharegpt\sharegpt40\prompt_128_01.txt" `
  --sessions 4 --target-tokens 175 --chunks 88,87 --runs 3 `
  --out-dir ".\phone-owned-residency-s4"
```

The harness checks that subsequent runs report zero Graph preparation cost
and that **same-prompt final logits remain consistent across cleared KV**.
The first run may already be a hit if the server was prepared earlier.
Inspect the Phone terminal for `PREPARE_CACHE status=HIT` and all eight
`KV_RESET` entries. The harness does **not** claim cold-to-warm speedup
until those real-device checks pass.

## Build: Windows PC

```powershell
cd E:\llama\llama-jupiter
git fetch origin
git switch feat/jupiter-phone-owned-graphless-20261122
git pull --ff-only

cmake --build build-phone-owned --config Release --target llama-cli ggml-rpc --parallel 8
cmake --build build-phone-owned --config Release --target test-phone-owned-execution-plan test-phone-qwen3-local-builder test-phone-qwen3-local-parity --parallel 8
ctest --test-dir build-phone-owned -C Release --output-on-failure -R "test-phone-owned-execution-plan|test-phone-qwen3-local-builder|test-phone-qwen3-local-parity"
```

The legacy standalone native RPC smoke tool is still available as
`test-jupiter-phone-owned-rpc`, but **it is not the full-Prefill test**.

## Build and run: Phone Termux

Use the same experimental branch and the original working Android/HTP
toolchain and library paths. For example, assuming `build-htp` is your
previously configured Android build:

```bash
cd ~/llama-jupiter
git fetch origin
git switch feat/jupiter-phone-owned-graphless-20261122
git pull --ff-only

cmake --build build-htp --target ggml-rpc-server -j 4

export GGML_HEXAGON_NDEV=1
export LD_LIBRARY_PATH="$HOME/phone-stream-htp/lib:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$HOME/phone-stream-htp/lib:${ADSP_LIBRARY_PATH:-}"
export LD_PRELOAD=/system/lib64/liblzma.so

# Replace these with actual build and local GGUF paths:
./build-htp/bin/ggml-rpc-server \
  -H 0.0.0.0 -p 10140 -d HTP0 \
  -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf"
```

The RPC server must say `v4.0.5` and have direct access to the local
Qwen3-14B GGUF via `-m`. For S4, restart with
`GGML_HEXAGON_NDEV=4` and `-d HTP0,HTP1,HTP2,HTP3`.
Use a trusted private PC–Phone network; do not expose RPC to the internet.

## Run: complete Prompt Prefill + logits A/B

From the PC PowerShell:

```powershell
cd E:\llama\llama-jupiter

python .\tools\jupiter\run_phone_owned_full_prefill.py `
  --exe ".\build-phone-owned\bin\Release\llama-cli.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --rpc "10.71.109.52:10140" `
  --prompt-file "E:\llama\datasets\sharegpt\sharegpt40\prompt_128_01.txt" `
  --sessions 1 `
  --target-tokens 175 `
  --chunks 88,87 `
  --mode both `
  --out-dir ".\phone-owned-prefill-s1"
```

This runs **legacy first** and **native second**, using the same real
Prompt and two-Chunk plan. It disables CLI warmup and uses `-n 1`.
Its `results.json` contains the prompt-eval time, binary F32 logits
paths, exact Top-1 agreement, max-absolute error, RMSE, relative RMSE and
cosine. A numeric mismatch is reported, not silently labeled PASS.

For S4, restart Phone RPC with four HTP devices, then use `--sessions 4`
and an isolated output directory.

### Internal two-token probe (fixed on this branch)

The llama-server wrapper calls `common_context_can_seq_rm()` before serving
the real Prompt. Its capability check normally evaluates **two synthetic
tokens**, even when `--no-warmup` is set. For the one-shot native Phone-owned
KV/run this must NOT execute the model: otherwise it completes the Phone-owned
run with just two tokens and subsequent real Prefill fails with
`DECODE_KV_HANDOFF_NOT_IMPLEMENTED`.

When `LLAMA_JUPITER_PHONE_OWNED_NATIVE=1` and
`LLAMA_JUPITER_PREFILL=1`, the branch now checks the host memory API instead
of calling `llama_decode()` during this capability probe. Expected log:

```text
[JUPITER_PHONE_OWNED_PREFILL] phase=SEQ_RM_PROBE mode=MEMORY_ONLY result=1 synthetic_decode=SKIPPED
```

The real first native decode must be the 175-token Prompt, not the synthetic
2-token probe. This change is **PC-side only**: rebuild `llama-cli` and its
dependent `common` library; the Phone RPC server need not rebuild for this
specific fix. The A/B runner now rejects a missing probe marker and checks the
exact `PREPARE_BEGIN tokens=175` value.

### Expected log milestones

```text
[JUPITER_PHONE_OWNED] model_placement layer=32 ... role=PHONE_WEIGHT_PLACEHOLDER_CPU
[JUPITER_PHONE_OWNED] model_placement layer=40 ... role=PC_HEAD
[JUPITER_PHONE_OWNED_PREFILL] phase=PREPARE_BEGIN ...
[JUPITER_PHONE_OWNED_PREFILL] phase=PREPARE_DONE ...
[JUPITER_PHONE_OWNED_PREFILL] phase=CHUNK_DONE chunk=0 tokens=88 ...
[JUPITER_PHONE_OWNED_PREFILL] phase=CHUNK_DONE chunk=1 tokens=87 ...
[JUPITER_PHONE_OWNED_PREFILL] phase=COMPLETE tokens=175 chunks=2 output=REAL_LOGITS ...
[JUPITER_LOGITS_DUMP] status=PASS ...
```

Each client must rebuild its RPC library and the Phone must rebuild and
restart the v4.0.5 server. Never combine this RPC version with an older
Phone instance in native mode.

## Debug: full-Prefill numerical checkpoint parity

The first native real Prompt run produced finite logits but a different
Top-1 ID and cosine -0.07496 compared with the correct pure-CPU baseline.
A non-NaN output is not proof of model correctness.

The branch now provides two entirely optional debugging interfaces:

- PC: LLAMA_JUPITER_STAGE_DUMP_DIR=<absolute local directory> and
  LLAMA_JUPITER_STAGE_DUMP_TOKENS=175. GGML scheduler eval callback
  saves L31-L39 F32 stage outputs plus result_norm and result_output only during the 175-token request,
  not the synthetic two-token startup probe.
- Phone: GGML_RPC_PHONE_OWNED_STAGE_DUMP_DIR=<Termux directory> set before
  starting the rebuilt RPC server. When running native HTP Layers 32 through 39
  it writes l_out-{32..39}-chunk{0,1}.f32 (16 Phone F32 files).
- Native PC: add --stage-dump to run_phone_owned_full_prefill.py;
  this saves native-pc-stages/l_out-31.f32, phone_owned_hidden.f32,
  result_norm.f32 and result_output.f32.

Rebuild PC llama-cli and Phone ggml-rpc-server from this branch before
enabling checkpoints. The phone file export requires restarting the server
with the GGML_RPC_PHONE_OWNED_STAGE_DUMP_DIR environment variable.

To get reference intermediate tensors, run normal pure-CPU llama-cli on
the same real Prompt (175 tokens), with -ngl 0, --flash-attn on, -n 1,
context 512, and LLAMA_JUPITER_PREFILL=0, while the PC dump variables
are enabled. The expected files are l_out-31.f32 through l_out-39.f32, plus
result_norm.f32 and result_output.f32. Use a different directory for the native PC dump.
Transfer all 16 phone HTP checkpoint files into one local PC directory.

Use compare_phone_owned_stages.py with --cpu-dir, --native-dir and
--phone-dir. It prints cosine, relative RMSE, absolute error and per-chunk
(88 and 87 tokens) differences at four comparison points:

1. CPU L31 versus native PC L31 (full 175-token PC prefix correctness).
2. CPU L32 versus HTP L32 (full 175-token first Phone layer).
3. CPU L39 versus final token of HTP L39 (standard Qwen3 gathers only
   the selected output token during its final Transformer layer).
4. HTP L39 versus native PC phone_owned_hidden (all 175 tokens;
   verifies the Phone-to-PC RPC transfer).

Do NOT use these debug runs for performance claims. Scheduler eval
callbacks synchronize and split compute; phone dumps add file I/O.

Follow the FULL layer-by-layer progression through L32, L33, ..., L38,
then compare only the final output token at L39. Next compare the
output of PC Final RMSNorm (5120 F32s) and the raw LM Head logits
(151936 F32s) to isolate projection correctness. A large drop in
cosine between L39 and result_norm suggests a final norm issue;
a large drop between result_norm and result_output suggests the
LM Head weight/layout/device path. These are diagnostic hypotheses,
not automatically proven root causes.

## Next verification before performance claims

1. Check compilation on Windows MSVC and Termux HTP.
2. Check S1 native prepare and two real Prompt chunks.
3. Compare full F32 logits against baseline and inspect KV/Attention
   discrepancies if they differ.
4. Compare S1/S4 phase breakdowns and the PC `JUPITER_SCHED_ALLOC` timing.
5. Only after parity: measure request TTFT separately from HTP preparation.
6. Separately implement verified Phone-KV → Decode handoff for `-n > 1`.

**Do not claim the former ~1-second RPC `alloc_retry_ms` was removed until
the new branch runs and measures a real Prompt with this native graph path.**
