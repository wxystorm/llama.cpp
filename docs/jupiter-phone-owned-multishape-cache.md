# Jupiter Phone-owned: multiple Prompt / Chunk shapes (experimental)

Branch: `feat/jupiter-phone-owned-multishape-20261123`  
Stable predecessor: `feat/jupiter-phone-owned-graphless-20261122`
(commit `25166d08416ed1ccc585aecc6e68e0d538cafade`).

## Goal and current scope

The stable branch proved Phone-owned native 14B prefill, S1/S4 execution,
and one resident Graph topology across independent `llama-cli` requests.
This branch adds **multiple simultaneously resident fixed Graph topologies**,
so Prompt A (e.g. 175 tokens / 88+87), Prompt B (e.g. another input length
or Chunk geometry), then A again can reuse A without rebuilding its
HTP Graphs. Each new logical request still gets fresh zeroed KV.

**This new multi-shape code is committed but has NOT yet been validated on
physical HTP.** Preserve the stable branch for fallbacks and comparisons.

The PC-side native pilot remains limited to:
- dense Qwen3-14B (40 layers): PC CPU 0–15, PC CUDA 16–31,
  Phone HTP 32–39, PC final head
- context exactly 512; fresh 2–512-token Prompt; one returned logit
  vector, `-n 1` (Phone-to-decode KV handoff remains unsupported)
- S1 or S4; chunks `<=128` tokens with a final tail and at most
  16 chunks. This is **not** a 2048/4096-context implementation.

## Correctness and cache design

The shared Phone RPC Server holds:
- immutable GGUF-derived HTP model weights across requests
- a set of resident Graph templates, each with its own fixed Graph UID,
  tensor metadata, Scratch storage, and private F16 KV
- at most one active logical Phone-owned request

The cache key matches **all graph-affecting plan fields** (Token count,
Chunk geometry, KV size/read span, layer, Qwen3/RoPE scalars) plus the
entire layer-to-HTP-Session mapping. `run_id` and `p2c_run_id` are
excluded from graph shape matching. A HIT maps a new logical run ID to
the original physical graph run key, synchronizes the owning HTP
Session, zeros every layer's F16 KV, resets its Chunk FIFO cursor, then
runs the existing fixed Graphs without touching its UIDs or allocations.

Different Prompt **text** with the same geometry can share one Graph.
Different Prompt **length or Chunk schedule** can coexist as different
Graph templates, subject to cache capacity.

### Bounded admission, eviction, and memory

Phone server environment variables (set **before server startup**):

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `GGML_RPC_PHONE_OWNED_PERSIST` | `1` | `0` disables retention |
| `GGML_RPC_PHONE_OWNED_CACHE_ENTRIES` | `2` | maximum 1–4 resident templates |
| `GGML_RPC_PHONE_OWNED_CACHE_BUDGET_MIB` | `2048` | 128–8192 MiB budget for **new Graph+KV allocations**, not model weights |

The server tracks actual HTP Graph/Scratch/KV bytes for every successful
new topology, then admits it to the resident pool only if the pool stays
within the budget. Before building a new topology it estimates the
additional bytes conservatively from existing cached shapes. If the
entry/budget limit is reached, it **flushes the entire generation** before
building the next shape.

This conservative eviction is intentional: the current Hexagon fixed
Graph cache only exposes **Session-wide invalidation**. Releasing the
tensor storage of one compiled template while retaining other compiled
opnodes could leave dangling pointers. This is NOT a per-entry LRU; a
future version needs safe per-UID invalidation before implementing LRU.

Memory tracked here excludes immutable model weights, Hexagon driver
internal memory, and PC memory. The budget is an additional bound, not
proof the Phone has sufficient free RAM. An incompatible Session mapping
is a different key; avoid changing the S1/S4 mapping while benchmarking.

An incomplete or failed request does not enter the cache; it causes a
safe full-generation release on RESET. A different valid Prompt is
never allowed to see another request's KV.

## Build and start the Phone RPC server

Stop the old Phone RPC Server first. On Termux:

```bash
cd ~/llama-jupiter
git fetch origin
git switch feat/jupiter-phone-owned-multishape-20261123
git pull --ff-only

cmake --build build-htp --target ggml-rpc-server -j 4

export GGML_RPC_PHONE_OWNED_PERSIST=1
export GGML_RPC_PHONE_OWNED_CACHE_ENTRIES=2
export GGML_RPC_PHONE_OWNED_CACHE_BUDGET_MIB=2048
export GGML_HEXAGON_NDEV=4
export LD_LIBRARY_PATH="$HOME/phone-stream-htp/lib:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$HOME/phone-stream-htp/lib:${ADSP_LIBRARY_PATH:-}"
export LD_PRELOAD=/system/lib64/liblzma.so

./build-htp/bin/ggml-rpc-server \
  -H 0.0.0.0 -p 10140 \
  -d HTP0,HTP1,HTP2,HTP3 \
  -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf"
```

Only bind to a private/trusted network. The GGML RPC Server is not an
authenticated Internet-facing service.

## Run A → B → A on PC (automated token count)

The test automatically tokenizes each file with the matching
`llama-tokenize.exe`, computes its actual tokens and 88-token chunks,
runs native Prefill in a separate `llama-cli.exe` process for each
Prompt, checks successful real final logits, and compares A with replayed
A after B. Do not restart the Phone RPC server between cases.

```powershell
cd E:\llama\llama-jupiter
git fetch origin
git switch feat/jupiter-phone-owned-multishape-20261123
git pull --ff-only

cmake --build build-phone-owned --config Release `
  --target llama-cli llama-tokenize test-phone-owned-execution-plan `
  --parallel 8

ctest --test-dir build-phone-owned -C Release --output-on-failure `
  -R "test-phone-owned-execution-plan"

python .\tools\jupiter\run_phone_owned_multi_prompt.py `
  --exe ".\build-phone-owned\bin\Release\llama-cli.exe" `
  --tokenizer ".\build-phone-owned\bin\Release\llama-tokenize.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --rpc "10.71.109.52:10140" `
  --prompts `
    "E:\llama\datasets\sharegpt\sharegpt40\prompt_128_01.txt" `
    "E:\llama\datasets\sharegpt\sharegpt40\prompt_128_02.txt" `
  --sessions 4 --chunk-size 88 `
  --revisit-first --require-revisit-hit `
  --out-dir ".\phone-owned-multi-prompt-s4"
```

**Select two Prompt files that have different actual Token counts**;
otherwise A and B may share a single identical template and this does
not test multi-shape residency. For another quick test, vary
`--chunk-size` in separate runs (but that replaces, rather than
interleaves, the template).

Expected Phone RPC sequence for two different geometries:

```text
PREPARE_CACHE status=MISS ... action=NEW_TEMPLATE
RESET ... graph_cache=RESIDENT ... entries=1
PREPARE_CACHE status=MISS ... retained_other_templates=1 action=NEW_TEMPLATE
RESET ... graph_cache=RESIDENT ... entries=2
KV_RESET layer=32 ...
...
KV_RESET layer=39 ...
PREPARE_CACHE status=HIT ... retained_other_templates=1
```

A warm HIT's PC `PREPARE_DONE` has `load_ms=0`,
`build_ms=0`, `buffer_ms=0`, `precompile_ms=0`.
The benchmark records complete `prompt_eval_ms`, which still includes
PC CPU/CUDA compute, not just HTP. Preserve separate PC cold/warm
comparisons when reporting end-to-end speedups.

Results: `phone-owned-multi-prompt-s4/multi-prompt-results.json`.
Inspect both the test output and the Phone RPC log; model-free contract
tests cannot establish HTP correctness.

## Rollback

The previously verified, single-template version is unchanged on
`feat/jupiter-phone-owned-graphless-20261122`. To roll back, build and
restart both PC/Phone from that branch.
