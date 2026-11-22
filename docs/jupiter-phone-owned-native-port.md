# Jupiter Phone-owned Native HTP Graph Port (experimental)

Source: `feat/phone-plan-overlap-multi-htp-20261109`; target base:
`feat/jupiter-htp-direct-rpc-20261112`.

## What is implemented

- The **original** immutable `region_plan`, Qwen3 local graph builder,
  output-parity helpers and Phone-owned HTP serial executor are imported.
- The phone's HTP executor can load immutable Qwen3 layer weights from its
  **local GGUF tensor source**, construct/allocate/compile native layer graphs
  and retain a private persistent F16 KV for each layer.
- New opt-in RPC opcodes:
  - `PHONE_OWNED_PREPARE`: one Plan + one explicit device ID per layer;
    prepares all layer/chunk Graphs on the phone, with preparation timings.
  - `PHONE_OWNED_RUN`: sends one F32 hidden state
    `[n_embd, n_tokens]`, computes every configured layer **on the phone**
    in Transformer order, and returns the last F32 hidden state.
  - `PHONE_OWNED_RESET`: drops native graphs/private KV for the active run.
- Client-side C++ entry points:
  `ggml_jupiter_phone_owned::{prepare,run_chunk,reset}` in
  `ggml/src/ggml-rpc/phone-owned-jupiter-api.h`.
- Three source-origin, model-free regression tests and a manual S1/S4
  native-graph prepare/run smoke tool.
- RPC version **4.0.5**: rebuild both the PC and phone RPC binaries for native
  mode. Legacy RPC execution remains the default and does not issue the new
  commands.

## Deliberate limitations / NOT yet implemented

**This is the native Phone-owned RPC execution bridge, not an end-to-end
`llama-cli` replacement or a demonstrated Prefill speedup.**

- Normal `llama-cli` still builds the complete PC/RPC Graph and executes
  the original Scheduler. Its `graph_alloc_ms` will not change until
  the CPU/CUDA Graph is cut at the layer-31 boundary and a delegate to
  `run_chunk` is inserted in the main inference path.
- The native endpoint currently returns **layer-39 hidden states**, not
  logits. The final output norm/head and generation/decode must be implemented
  or delegated to PC before enabling an end-to-end cutover.
- The original Plan supports only fresh monotonic Prefill (position base 0,
  F16 KV, no prefix reuse or KV shifting). Private Phone-owned KV is **not**
  synchronized to the legacy decode cache in this pilot.
- Synthetic-input smoke tests measure feasibility, not model correctness.
  Real same-boundary-activation comparisons against the unchanged legacy
  path, KV parity, logits parity and HTP memory-budget checks are mandatory
  before performance claims or enabling native mode in normal benchmarks.
- The first Phone-owned bridge is exclusive: one active run per phone RPC
  server. Mixing legacy compute and Phone-owned execution concurrently
  on the same HTP sessions is unsupported.
- Local GGUF access must be provided to the phone RPC server. A server
  without `local_tensor_source.get_info/read` fails closed during prepare.

## Model-free build and tests

On a Linux PC (no phone, CUDA or Hexagon required for static model-free tests):

```bash
cmake -S . -B build-phone-owned -DGGML_RPC=ON -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-phone-owned --parallel 4 --target ggml-rpc-server test-jupiter-phone-owned-rpc test-phone-owned-execution-plan test-phone-qwen3-local-builder test-phone-qwen3-local-parity
ctest --test-dir build-phone-owned --output-on-failure -R 'test-phone-owned-execution-plan|test-phone-qwen3-local-builder|test-phone-qwen3-local-parity'
```

On Windows, use your existing `build-jupiter` configuration if it enables
`GGML_RPC` and `LLAMA_BUILD_TESTS`, otherwise reconfigure.

## HTP smoke

Rebuild and restart the Phone RPC server from this branch; it needs its local
Qwen3-14B GGUF tensor source and HTP0 for S1, HTP0..HTP3 for S4.
The S1/S4 mapping matches the earlier 8-layer layout:
`S1: 0,0,0,0,0,0,0,0`;
`S4: 0,0,1,1,2,2,3,3`.

The smoke uses a 175-token Plan split into chunks **88 + 87**, layers 32–39,
fresh `n_kv=512` and F16 KV. It **does not** measure a language-model response.

```powershell
# prepare only: build and precompile Phone-owned Graphs
.\build-jupiter\bin\Release\test-jupiter-phone-owned-rpc.exe 10.71.109.52:10140 1

# also compute the two chunks from finite synthetic F32 hidden inputs
.\build-jupiter\bin\Release\test-jupiter-phone-owned-rpc.exe 10.71.109.52:10140 1 --run

# 4 sessions (requires all four HTP devices and enough session memory)
.\build-jupiter\bin\Release\test-jupiter-phone-owned-rpc.exe 10.71.109.52:10140 4 --run
```

The tool automatically sets `LLAMA_JUPITER_PHONE_OWNED_NATIVE=1`
so the HELLO handshake rejects phone RPC servers older than 4.0.5.
If the phone lacks a local GGUF loader or native fixed-graph precompile
entry point, look for `[JUPITER_PHONE_OWNED] PREPARE_REJECT` in the
phone log. There must be no silent fallback to old RPC Graph execution.

Expected log markers:

- `[JUPITER_PHONE_OWNED] PREPARE_LAYER`
- `[JUPITER_PHONE_OWNED] PREPARE_READY`
- `[JUPITER_PHONE_OWNED] RUN_DONE`
- `[JUPITER_PHONE_OWNED] RESET`

## Required next integration step

1. Gate the complete CPU/CUDA-local Graph at the layer-31/32 boundary.
2. Feed the boundary activation for each Chunk into the registered Phone-owned
   Region, without PC Scheduler RPC-layer split/alloc/INIT_TENSOR.
3. Restore final norm/head on PC or implement the phone's final head,
   and integrate the post-prefill decode KV state.
4. Compare outputs against the unmodified legacy path before evaluating
   TTFT; isolate prepare-at-load time from request-time execution.

**Do not report the previous 1-second `alloc_retry_ms` as eliminated until
the whole-Graph PC Scheduler allocation is actually bypassed.**
