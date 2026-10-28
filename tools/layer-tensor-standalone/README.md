# Standalone Layer-vs-Tensor runtime

This experiment is isolated from `llama-cli`, Region Ring and `ggml-backend-meta.cpp`.

Two peer processes execute the same truncated Qwen3-MoE model:

- `llama-layer-tensor-bench` on the PC;
- `llama-layer-tensor-worker` on the Phone.

PC sends the plan once. During the measured interval it does not issue remote graph-compute commands. Both endpoints build GGML graphs locally from their own GGUF and exchange only activation/task events.

## Runtime

Each endpoint has:

```text
compute worker:
    OWNER_QUEUE        preferred
    REMOTE_FFN_QUEUE

reduce worker:
    REDUCE_QUEUE

I/O:
    RX loop
    TX worker
```

Tensor OWNER is real Qwen3-MoE compute:

```text
Attention
  -> residual
  -> FFN RMSNorm
  -> Router / TopK
       |
       +-- READY(residual, ffn_norm, topk, weights) --> peer
       |
       +-- local expert FFN shard
              |
              +-- PARTIAL --> peer
```

The non-Attention owner runs its expert shard from READY. REDUCE combines residual + both partials, then directly unlocks `(L+1,C)` on the same endpoint. Device-local continuation is preserved across REDUCE -> next OWNER and consecutive local Full-Layer stages; host/TCP transfer is used only at actual cross-device boundaries.

Attention chunks are admitted in increasing chunk order within each layer so persistent KV is causally valid. `--max-owner-ahead 1` is the default diagonal-wave admission gate: a layer may stay only one chunk ahead of READY progress from the downstream owner layer.

## Real model loading

Both processes use `LLAMA_HYBRID_EXPERIMENT_N_LAYERS=x` internally while loading the model, so only `blk.0 ... blk.(x-1)` are materialized. PC uses a local CPU backend. Phone defaults to the first local GPU/IGPU backend (`--device auto`), intended for Adreno OpenCL.

Model loading is outside the measured interval. Phone sends `MODEL_READY` only after its local GGUF/backend/KV state is ready. PC also tokenizes and computes the common token embeddings before starting `wall_ms`.

Local cross-layer continuation uses one persistent F32 activation arena shaped `[n_embd, total_tokens]`. Every chunk owns a fixed token range in that arena. `REDUCE(L,C)` or a local Full-Layer writes its output to that range with a backend-local tensor copy; the next local OWNER reads the same range directly. There are no per-chunk ping-pong slots and no GPU -> Host -> GPU bounce on this path.

## Build

PC:

```powershell
cmake --build build-tensor-bench --config Release --target llama-layer-tensor-bench llama-layer-tensor-worker
```

Phone, from an already configured OpenCL build:

```bash
cmake --build build-layer-tensor --target llama-layer-tensor-worker -j8
```

## Dependency self-test

The old deterministic reference engine remains only for scheduler regression testing:

```powershell
ctest --test-dir build-tensor-bench -C Release -R "layer-tensor-runtime-self-test" --output-on-failure
```

Reference timing is not used for performance conclusions.

## Real Phone worker

```bash
./build-layer-tensor/bin/llama-layer-tensor-worker \
  --listen 0.0.0.0:18061 \
  --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --device auto \
  --threads 8 \
  --trace
```

The worker prints `[LT_GGML_MODEL] ... status=READY` followed by `[LT_WORKER] event=MODEL_READY ...` after local model/backend setup.

## Real Tensor run

```powershell
.\build-tensor-bench\bin\Release\llama-layer-tensor-bench.exe `
  --connect PHONE_IP:18061 `
  --model E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
  --prompt-file .\prompt.txt `
  --mode tensor `
  --layers 6 `
  --chunk-tokens 64 `
  --pc-ratio 0.50 `
  --max-owner-ahead 1 `
  --threads 8 `
  --trace
```

`--c2p-ratio` and `--p2c-ratio` can tune the two directions independently. `--pc-ratio` sets both.

## Real Layer run

Restart the one-shot Phone worker, then run for example:

```powershell
.\build-tensor-bench\bin\Release\llama-layer-tensor-bench.exe `
  --connect PHONE_IP:18061 `
  --model E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
  --prompt-file .\prompt.txt `
  --mode layer `
  --layers 6 `
  --chunk-tokens 64 `
  --layer-owners pc,pc,phone,phone,pc,pc `
  --threads 8 `
  --trace
```

## Output

A successful performance run ends with:

```text
[LT_RUN] ... prelude_ms=... wall_ms=... tok_per_s=... final_chunks=... checksum=... status=PASS engine=GGML_REAL
```

`wall_ms` is the layer-stage interval used for Layer-vs-Tensor crossover. `prelude_ms` is reported separately and includes PC setup/embedding plus any wait for Phone `MODEL_READY`.

`checksum` is a deterministic checksum of the final hidden chunks accumulated on PC. Layer and Tensor candidates should be checked for numerical agreement before entering performance ranking.

## Current scope

The real adapter is intentionally Qwen3-MoE-only and rejects fused/scaled expert tensor variants it does not yet implement. The first target is `Qwen3-30B-A3B-Q4_K_M.gguf`.

Graphs currently use the standalone direct GGML backend and real model weights. Scheduler/protocol remain independent of Meta/RPC; no runtime graph is serialized from PC to Phone.
