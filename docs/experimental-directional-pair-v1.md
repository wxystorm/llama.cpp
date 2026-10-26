# Directional Pair V1 — PC → C2P → Phone → P2C → PC

Status: **topology selection and validation implemented; runtime reuses existing directional Meta path. Device build, numerical parity and Phone autonomous pipeline NOT yet verified on this branch.**

This feature is on a separate branch:
`exp/pc-c2p-phone-p2c-pc-20261026`.
The fixed C/G/P Ring, its Planner, and the previously validated Phone READY branch are unchanged. Do not enable Ring concurrently.

## Topology and exact ownership

On Qwen3-30B-A3B (48 layers) the initial, previously supported model policy is:

`PC_FULL [0,34) → TENSOR_C2P L34 → PHONE_FULL [35,46) → TENSOR_P2C L46 → PC_FULL [47,48)`.

- **C2P, layer 34**: PC owns Attention/Router, PC and Phone each compute a quantization-aligned FFN shard. PC→Phone sends the READY activation and the folded PC FFN-partial + residual. Phone locally reduces its FFN partial; Phone owns `l_out-34`. Do not require a Phone→PC return inside this Tensor layer.
- **Phone, layers 35..45**: Full Attention + FFN / KV reside on Phone. The existing optional full Region Graph / server MAIN implementation can optimize this area later; baseline still uses the established Meta path.
- **P2C, layer 46**: Phone owns Attention/Router; the two FFN shards run on their respective devices. Phone→PC sends READY and Phone partial + residual, and PC performs the final reduction. PC owns `l_out-46`. There is no PC→Phone return within this Tensor layer.
- **PC, layer 47**: Final PC region completes the remaining model computation and final logits. Optional CPU/CUDA assignment inside the PC-only parts follows `LLAMA_HYBRID_GPU_PC_LAYERS` and the existing PC Layer placement.

The two directional boundary layers need **only one direction each**, although the **whole block** uses both directions: C2P at the entrance and P2C at the exit.

`LLAMA_HYBRID_DIRECTIONAL_PAIR=1` is a single, explicit selector for the existing `LLAMA_HYBRID_REGION_BOUNDARY_V1` directional policy, with stricter checks:
- Requires supported hybrid model and `--split-mode tensor`.
- Rejects concurrent `LLAMA_HYBRID_REGION_RING_V1` and an already applied Hybrid Auto placement.
- Rejects legacy `LLAMA_HYBRID_REGION_C2P_ONLY`.
- Validates C2P/P2C geometry and assigns all layers exactly once.
- Logs the precise two directional boundaries, final output owner and PC CPU/CUDA placements.
- Verifies exactly one C2P and one P2C mask at every execution entry.

The pre-existing `LLAMA_HYBRID_REGION_BOUNDARY_V1` flag is still supported for rollback/comparison. Custom bounds use `LLAMA_HYBRID_REGION_C2P_LAYER` and `LLAMA_HYBRID_REGION_P2C_LAYER`, but the specialized Phone Region pipeline currently only recognizes **34,35–45,46**. First test should keep these defaults.

## A. Correctness-first baseline — do this before overlapping queues

PowerShell environment reset:

```powershell
# Disable the different fixed Ring and its Planner.
Remove-Item Env:LLAMA_HYBRID_REGION_RING_V1 -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_REGION_RING_LAYOUT -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_REGION_RING_PLANNER -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_RING_PHONE_FUSION -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_RING_PHONE_AUTOSTART -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_REGION_C2P_ONLY -ErrorAction SilentlyContinue

# Disable Phone Region optimizations only for the initial parity baseline.
Remove-Item Env:GGML_META_PHONE_REGION_PIPELINE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_PHONE_REGION_GRAPH -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_PHONE_REGION_LOCAL_MAIN -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_PHONE_REGION_BIDIR_PIPELINE -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_PHONE_REGION_PC_OVERLAP -ErrorAction SilentlyContinue

$env:LLAMA_HYBRID_DIRECTIONAL_PAIR = "1"
$env:LLAMA_HYBRID_REGION_C2P_LAYER = "34"
$env:LLAMA_HYBRID_REGION_P2C_LAYER = "46"
$env:LLAMA_HYBRID_REGION_C2P_PC_RATIO = "0.333333"
$env:LLAMA_HYBRID_REGION_P2C_PC_RATIO = "0.333333"

# Optional: number of PC_ONLY layers assigned to CUDA first in layer order.
# Verify actual GPU free memory before increasing.
$env:LLAMA_HYBRID_GPU_PC_LAYERS = "8"

# Diagnostic only (the self-check incurs overhead):
$env:GGML_META_PIPELINE_DEBUG = "1"
$env:GGML_META_REGION_C2P_SELF_CHECK = "1"
```

Use the existing `llama-cli` invocation with `--split-mode tensor`, the usual `--rpc` endpoint, the same model, prompt/seed, batch, ctx, and sampling settings as parity reference. Do **not** enable `--hybrid-auto` in this mode. Only build the PC from the new branch; the previously matched protocol-v17 Phone RPC server does not need code changes.

Build on PC:

```powershell
git fetch origin
git switch exp/pc-c2p-phone-p2c-pc-20261026
git pull --ff-only
cmake --build build-tensor-v2 --config Release --target llama-cli -j 8
```

Build policy test (if `LLAMA_BUILD_TESTS` is ON):

```powershell
cmake --build build-tensor-v2 --config Release --target test-directional-pair -j 8
ctest --test-dir build-tensor-v2 -C Release -R "^test-directional-pair$" --output-on-failure
```

Expected PC-side log headers:

```text
[DIRECTIONAL_PAIR_PLAN] pc0=[0,34) c2p=34 phone=[35,46) p2c=46 pc1=[47,48) c2p_owner=PHONE p2c_owner=PC ...
[DIRECTIONAL_PAIR_BACKENDS] pc_prefix_cpu=... pc_prefix_cuda=... phone_full=11 pc_suffix_cpu=... pc_suffix_cuda=... ...
[DIRECTIONAL_PAIR_EXEC] phase=PREFILL ... c2p=34 phone=[35,46) p2c=46 ... status=POLICY_OK
[META_C2P_POLICY_MASK] ... first=34 ...
[META_P2C_POLICY_MASK] ... first=46 ...
[REGION_C2P_PREFILL] ... owner=PHONE ... status=OK
[REGION_P2C_PREFILL] ... owner=PC ... status=OK
```

Filter full lines without PowerShell wrapping:

```powershell
Select-String -Path ".\region-pair.log" -Pattern '\[DIRECTIONAL_PAIR','\[REGION_BOUNDARY_V1\]','\[META_C2P_POLICY_MASK\]','\[META_P2C_POLICY_MASK\]','\[REGION_C2P_PREFILL\]','\[REGION_P2C_PREFILL\]','\[REGION_C2P_NUMERIC\]','\[REGION_P2C_NUMERIC\]' | ForEach-Object { $_.Line }
```

Check the output text/tokens/logits against a known-good same-model/same-seed reference. Shape placement and RPC success alone do not prove numerical parity. Run 2-token warmup, then 358-token Prefill, and only then longer inputs.

## B. Next optimization — not automatically enabled

The old Meta implementation supports a specialized **34 / Phone 35–45 / 46** region pipeline and Phone MAIN queue. Following baseline correctness, independently evaluate:
- `LLAMA_HYBRID_PHONE_REGION_GRAPH`, `GGML_META_PHONE_REGION_PIPELINE`: pre-allocated scratch and Region Graph, window=2, per-Chunk C2P / Phone / P2C continuation schedule.
- `GGML_META_PHONE_REGION_PC_OVERLAP`, `GGML_META_PHONE_REGION_BIDIR_PIPELINE`: upstream PC C2P preparation and directional returns overlap with Phone work.
- `GGML_META_PHONE_REGION_LOCAL_MAIN` together with the existing `GGML_META_PHONE_PREFILL_CHUNK_PIPELINE`, `GGML_META_PHONE_PREFILL_PRODUCER_ROUTE`, `LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER` and `LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE` prerequisites: Phone MAIN executes its locally queued graph and stages the return partial.

**Important:** the existing local-MAIN code still uses a small **PC-originated local-submit control RPC** per Chunk. That is not yet the fully READY-only autonomous Phone execution from the earlier Ring experiment. Converting that route to register-once / READY-trigger-next is a separate, testable follow-up. Do not claim it is implemented by this branch.

Do not turn on all the optimizations in a single first run: a failure would be impossible to localize. The initial aim is a validated C2P→Phone→P2C round trip with PC-owned final output.

## C. Planner integration later

Once a single directional pair is accurate, the Profile-based planner should search `(C/G prefix length, C2P layer, Phone full depth, P2C layer, C/G suffix, XC/XPhone/XT)` subject to memory and end-to-end Predicted Prefill. The current Ring Planner searches repeated fixed C/G/P cycles; it **must not** be applied directly to the directional-pair runtime.

Runtime prerequisites before differing PC/Phone Chunk sizes: exact token-interval dependency tracking, KV ordering, per-edge Buffer Slot references and completion credits. CPU/CUDA PC Gate / memory ownership remain real constraints.
