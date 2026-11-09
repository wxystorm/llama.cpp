# GCNA HTP memory policy (Snapdragon 8 Elite)

## Confirmed tests and selected policy

The standalone HTP memory probe has demonstrated a single 3072 MiB
RPCMEM/FastRPC arena and DSP-side ADD mapping, and two simultaneous 1536 MiB
arenas with DSP-side ADD. These are successful **probe working sets**, not
general NPU VRAM sizes or proof that a real multi-layer Qwen3 Graph can use
3072 MiB of weights without its own KV/activation allocations.

For the GCNA Planner/runtime on this device, use:

- **3072 MiB:** maximum NPU-owned committed allocation budget (weights,
  HTP-owned persistent KV, registered per-Chunk Graph/activation storage).
- **1536 MiB:** maximum allocation allowance per NPU Stage/physical Bank.
  A single Cycle's NPU weight bytes **plus its stage-specific KV, Graph and
  activation requirements** must fit this bound.
- **RESIDENT first:** if all selected NPU Stage weights + all persistent
  NPU KV + Graph/activation storage fit the 3072 MiB allowance, and the
  entire Phone Plan also fits available shared RAM, preload all HTP Stage
  weight arenas and **do not allocate rolling weight Banks**.
- **STREAMING_2BANK fallback:** only when RESIDENT is infeasible and at
  least two NPU Regions exist; rank-even/odd Bank sizes are the maximum
  weight arena size among their assigned Stages. Both Bank maxima plus
  persistent HTP KV and Graph/activation storage must fit 3072 MiB.
  Each Bank is at most 1536 MiB. Unselected excess NPU work may be
  moved to Adreno by the existing N->A candidate search.
- **NPU0:** allocate no HTP Bank/weight storage.
- **Independent Phone RAM budget:** Adreno resident weights/KV/workspace,
  CPU-side GGUF staging, and other shared overhead must still satisfy
  the Phone total shared RAM budget in addition to the NPU-only cap.

All allocations use bytes internally; these limits are MiB
(1 MiB = 1,048,576 bytes), not decimal MB.

## Where the restrictions are implemented

- ggml/include/ggml-phone-npu-budget.h: shared numerical constants.
- src/llama-region-ring-v2.h: pure NPU residency and two-Bank selection.
- src/llama-region-ring-planner.h: per-Stage NPU KV/Graph and joint Phone
  memory feasibility, plus selected NPU-only footprint reporting.
- src/llama-hybrid.cpp: native Profile-based memory estimates. Since the
  Phone prebuilds an independent Graph buffer for each Chunk, total Graph
  storage is estimated as the sum of per-Region workspace per Chunk.
  Actual Phone allocations are the final authority.
- ggml/src/ggml-hexagon/ggml-hexagon.cpp: in Plan-controlled Android
  HTP v79 mode, the default GGML_HEXAGON_MBUF becomes 1536 MiB unless
  explicitly set. Other Hexagon configurations retain their default.
- ggml/src/ggml-rpc/phone-npu-serial-executor.h: reject physical weight
  arenas exceeding 1536 MiB; track actual persistent NPU-owned KV,
  registered Graphs, and fixed activation arenas. Fail closed if
  3072 MiB globally or 1536 MiB per Stage would be exceeded.
- ggml/src/ggml-rpc/ggml-rpc.cpp: enforce the 3072 MiB Plan ceiling even
  with no GGML_RPC_PHONE_NPU_BANK_BUDGET_MIB override; lower user caps
  are honored.

The default Hexagon DSP active mapping limit GGML_HEXAGON_VMEM remains
3200 MiB. It is a **different** limit and is not increased here. A
successful admission does not substitute for HTP graph execution.

## Rebuild and verify

Rebuild BOTH PC Planner and Phone RPC/HTP backend from the same branch.
The mobile HTP library must pick up the new default MBUF logic before
running GCNA with 1536 MiB HTP Stage arenas.

On the PC after rebuilding:

    ctest --test-dir build-tensor-v2 -R "test-region-ring-v2-stage|test-region-ring-planner" --output-on-failure

Build directory names may differ from this example.

Inspect these logs when running a Plan:

    [REGION_RING_NATIVE_PROFILE]
    [REGION_RING_PLAN]
    [PHONE_NPU_PLAN]
    [NPU_BANK_MEMORY]
    [PHONE_OWNED_HTP_*]

Expected behavior:

- NPU weight policy RESIDENT with resident weight arenas and no rolling
  A/B weight buffers when all NPU allocations fit.
- STREAMING_2BANK only for multi-NPU-stage plans whose resident footprint
  does not fit but whose 2 Bank maxima and non-weights do fit.
- Distinct errors for oversized NPU Stage weights, actual Stage
  KV/Graph footprint, and actual global HTP allocation usage.

The Phone's real allocation limits and active DSP mappings always have
final authority. The standalone probe is a bound, not an exact performance
profile or a guarantee against DSP allocation failures under system load.
