# Phone-owned Qwen3 Layer Graph: phase 2 (shadow)

This branch moves the **graph description and construction logic** toward the
Phone without changing the currently working PC-built Graph execution path.

## Implemented

- RPC patch **28** has a versioned per-Region `region_plan`. No graph or
  tensor pointers cross this control RPC.
- The Phone records the plan for the active input stream, rejects conflicting
  re-registration, validates every legacy Chunk registration against it, and
  clears descriptors on stream teardown.
- `phone-qwen3-local-builder.h` builds a complete dense Qwen3 Transformer
  Layer *from Phone-resident GGUF weights*, using GGML operations:
  RMSNorm, Q/K/V matmul, per-head Q/K Norm, NeoX RoPE, K/V `SET_ROWS`,
  F16 Flash Attention, output projection, parallel SwiGLU FFN, and residuals.
- Side inputs (position, K/V indices, F16 causal mask) are generated locally
  from the immutable plan. No PC-supplied attention mask is required in this
  **fresh single-sequence prefill** subset.
- An optional Phone-side check creates one local graph per exact Chunk shape
  and verifies operator counts without HTP allocations.
- A **separate opt-in** HTP prepare step allocates persistent Phone-owned
  KV and fixed input/side-input/output buffers, fills positions/masks locally,
  and precompiles every local Chunk graph via the Hexagon fixed-graph API.
  It still does **not execute** these graphs or publish their results; the
  existing PC-built Graph path remains the only live execution path.

## Strict first-slice restrictions

Dense Qwen3, one complete Transformer Layer per Phone Region, F32 hidden
activations, F16 non-transposed K/V cache, Flash Attention, NeoX RoPE, one
sequence starting at position 0, no KV prefix reuse or shifting, no LoRA,
no additional per-layer features, and a single-output Phone Layer.

The PC still constructs and registers all old RPC Graphs. Keep the original
graph path for baseline performance and numerical parity.

## Enable both shadow layers

Build **both PC and Phone from the same commit** (patch 28 HELLO rejects
other versions). The Phone must start with a local copy of the *same model*
and have the matching layers resident in HTP before the PC registers plans.

Phone environment, shown for a four-Region layout with Phone Layers
9, 19, 29, 39 (adjust Region indices to match the actual runtime plan):

```sh
export GGML_RPC_PHONE_NPU_SERIAL_REGIONS=1,3,5,7
export GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS=9,19,29,39
export GGML_RPC_PHONE_QWEN3_LOCAL_BUILDER_SHADOW=1
# Additional memory/time-intensive HTP precompile smoke only:
# export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
```

PC environment (in addition to existing autonomous, input PUSH stream, and
Phone return stream experiment settings):

```powershell
$env:LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW = "1"
$env:LLAMA_REGION_RING_PHONE_ASYNC_GRAPH_SETUP = "1"
```

For the proposed first smoke test, use 512 prompt tokens, 128-token
macro-chunks, and `G9,P1,G9,P1,G9,P1,G9,P1`. Ensure the same GGUF weights
are available on both devices and Flash Attention is enabled.

Expected new Phone log:

```
[PHONE_QWEN3_LOCAL] event=GRAPH_BUILD ... mode=SHADOW_NO_HTP_ALLOC status=PASS
# Only with GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1:
[PHONE_QWEN3_LOCAL] event=HTP_PRECOMPILE ... mode=SHADOW_FIXED_STORAGE status=PASS
[PHONE_OWNED_PLAN] event=REGISTER ... graph_mode=SHADOW_PC_BUILT local_builder=GRAPH_BUILT status=READY
```

The GRAPH_BUILD message indicates **local graph construction validated**.
The optional HTP_PRECOMPILE message indicates backend allocation and graph
precompilation, **not** numerical parity or local compute execution.
Both modes retain the original PC-built Graph as the only compute owner.
Precompilation currently occurs synchronously during plan registration and
is therefore intentionally not a performance optimization yet.

## Model-free unit tests

```sh
cmake -S . -B build-phone-owned -DGGML_RPC=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-phone-owned --target test-phone-owned-execution-plan test-phone-qwen3-local-builder
ctest --test-dir build-phone-owned -R 'test-phone-(owned-execution-plan|qwen3-local-builder)' --output-on-failure
```

Tests cover plan validation/teardown, exact Chunk intervals including the tail,
per-position F16 causal masks, weight shapes, graph output dimensions and
presence of two `SET_ROWS`, two RoPE nodes, one Flash Attention and one SwiGLU.

## Remaining work before runtime cutover

1. Validate HTP precompilation on the real Phone and establish a
   Graph READY barrier, then move compilation off the synchronous RPC path.
2. Feed verified PC PUSH activations into the fixed Phone-owned input tensors;
   respect the existing per-Layer KV FIFO before executing the HTP Graph.
3. Publish the Phone-owned Graph output to P2C without looking up the old
   RPC-registered Graph, and implement safe slot reuse/backpressure.
4. Preserve PC decode compatibility (KV ownership/synchronization) and
   compare per-Layer activations and final logits against the old path.
5. Only then make the PC skip building/registering Phone RPC Graphs.
