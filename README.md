# Cycle / Region-Ring — Core Source Review Pack

**Purpose:** a *source-code review snapshot* focused on implementation ideas for collaborative PC + Phone LLM prefill. This is not a stand-alone buildable checkout.

- Upstream repository: https://github.com/wxystorm/llama.cpp
- Selected source branch: `exp/fig4b-32b-main50-parity-20261118`
- Pinned original commit: `0ea69b49c912dbf43ec64b0cbc81352e32d6596b` (2026-11-18)
- Included: 50 unmodified source/docs/test files, preserving repository-relative paths; 2 new navigation documents.
- Excluded: GGUF models, binaries, build artifacts, experimental result dumps, unrelated upstream code.
- **All existing source files are Git blob references from the pinned commit, without rewriting their content.**

## Context / 核心思路

This codebase implements a heterogenous PC (CUDA / CPU) + Phone (RPC / OpenCL / Hexagon HTP) prefill execution path. To understand the *actual* design, distinguish these layers:

1. **Profile & planner:** measured stage-cost inputs; G/C/P regions, independently chosen `XC/XG/XP` chunk sizes, memory checks, event-DAG prediction, and first-pass layer placement/lock logic.
2. **Cycle / Region-Ring runtime:** schedule activation chunks across multiple alternating regions and cycles; stage ordering, boundaries, ready queues and priorities, GPU weight transfer/double-bank/preload behavior.
3. **Cross-device execution:** actual integration with llama context/hybrid execution and Phone-side RPC, buffer binding, and optional HTP full-region execution.
4. **Experiments/tests:** entrypoints, ablations and invariants are useful to verify which features are actually wired into an experiment. The simulator is *not* a runtime measurement.

## Read these in order / 建议阅读顺序

1. `src/llama-region-ring.h` — region/task model, queues, scheduler invariants, execution policies.
2. `src/llama-region-ring-planner.h` — candidate placement, chunk selection, DAG score, memory budget.
3. `src/llama-region-ring-real-profile.h` — measured stage cost ingestion, lookup/rounding.
4. `src/llama-context.cpp` + `src/llama-hybrid.cpp` — actual hook-up to prefill, model layers, GPU banks, device handoff.
5. `ggml/src/ggml-rpc/ggml-rpc.cpp` + `phone-owned-execution-plan.h` — RPC-side execution protocol.
6. `ggml/src/ggml-rpc/phone-npu-serial-executor.h` and `ggml/src/ggml-hexagon/ggml-hexagon.cpp` — Phone HTP region execution.
7. `tools/region-ring-experiments/chunk_dag.py` and `probe_htp_llama_server.py` — offline estimation vs real observation.
8. `tests/test-region-ring-planner.cpp`, `tests/test-region-ring.cpp` — semantic assertions and examples.

## Questions to check in code / 给另一个 AI 的审查目标

- Where is the first-prompt region placement found, validated, and locked? Can later prompts change **only chunk widths** without silently altering placement?
- How are `XC/XG/XP` selected from independent measured G/C/N/A profiles, including KV-dependent shapes and memory accounting?
- How does multi-cycle execution maintain activation interval dependencies across G/C/Phone regions?
- Which execution paths implement *two GPU banks*, overlapping PCIe weight transfer and GPU compute? What events gate bank reuse?
- Where is **Drain-First** implemented and how does it differ from FIFO when chunks of different cycles compete?
- What is *actually executed on Phone HTP* versus OpenCL or CPU; who owns the KV cache and hidden-state handoff?
- Which timestamps are observed runtime measurements and which are simulation/forecast values?

## Important interpretation notes

- Don't call PC RAM-to-GPU VRAM transfers "paging"; use "PCIe weight transfer / weight copy".
- A profile predictor or DAG estimator is not evidence that an overlap occurs at runtime; inspect the execution and trace/event paths.
- `src/llama-context.cpp`, `src/llama-hybrid.cpp`, and `ggml-rpc.cpp` are long because they contain integration glue; search for `REGION_RING`, `PLANNER`, `BANK`, `PREFETCH`, `DRAIN`, `PHONE`, `HTP` and queue/scheduling names.
- This subset is intended for **architecture explanation / code review**, not compilation. Full upstream headers, CMake and unrelated libraries are intentionally omitted.
- See `FILE_MAP.tsv` for exact file manifest, blob hashes, and byte sizes.
