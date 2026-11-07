# Whole-layer Phone Adreno OpenCL + HTP0 mixed execution (experimental)

This branch is built on `feat/phone-npu-dual-bank-offline-plan-20261107`.
It adds an opt-in **execution smoke**, not yet a four-backend automatic
Planner. Only whole Transformer layers are assigned. The Phone has a
single compute worker, so OpenCL/HTP execution is **serial**, not concurrent.

## Mapping for first Qwen3-14B smoke

The 40-layer layout is:

    G8,C1,P1, G8,C1,P1, G8,C1,P1, G7,C2,P1

PC CPU: 5 layers (C1 x3, C2 x1)
PC CUDA: 31 layers (G8 x3, G7 x1)
Phone NPU HTP0: region 2/layer 9 and region 5/layer 19
Phone Adreno OpenCL: region 8/layer 29 and region 11/layer 39

The two NPU ranks occupy separate physical HTP Banks (A,B). This
first smoke deliberately uses **two** NPU regions, both primed at startup.
A third/deferred HTP rank needs an additional PC-Graph metadata-KV binding
lifecycle fix before mixed mode can test Bank refill; do NOT assume this
first milestone validates rank>=2.

Both HTP and OpenCL use the same Phone RPC Server. GPUOpenCL remains the
single visible RPC backend; HTP0 is a separate private executor, selected by
the ordered NPU Region list.

## 1. Phone Termux: before starting RPC

Use the SAME model (Qwen3-14B-Q4_K_M.gguf) already on the Phone.
Preserve your existing working LD_LIBRARY_PATH/OCL_ICD_VENDORS settings.

    export GGML_RPC_PHONE_NPU_SERIAL_REGIONS=2,5
    export GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS=9,19
    export GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2
    export GGML_RPC_PHONE_NPU_MIXED_OPENCL=1
    export GGML_RPC_PHONE_PERSISTENT_POOL_PREP=1
    export GGML_RPC_PHONE_STAGE_SLOT0_EXEC=1
    export GGML_RPC_PHONE_STAGE_DUAL_SLOT=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_INPUT=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_GRAPHLESS=1
    export GGML_RPC_PHONE_NPU_HOST_STAGE_MIB=512
    export GGML_RPC_REAL_STAGE_PROFILE=1

    ./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 \
      -d GPUOpenCL -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
      2>&1 | tee phone-mixed.log

Note: the Phone-side GRAPHLESS=1 is required by HTP two-Bank setup.
The PC-side global GRAPHLESS must be **0** because the GPUOpenCL Regions
still need PC-registered RPC Graphs. The mixed dispatcher nevertheless
selects **Phone-owned GRAPHLESS for the two HTP P1 Regions only**:
PC registers Plan + canonical KV, then skips PC HTP Chunk Graphs and
HTP auto-task registration. GPUOpenCL Regions retain PC Graphs and
OpenCL Slot0. The Phone server forbids PC HTP task registration.

Check the Phone boot log for HTP Region count=2 and both physical Banks.

## 2. PC PowerShell: fixed-layout mixed smoke

Compile the new PC branch and the matching Android RPC server first:

    cmake --build build-tensor-v2 --config Release --target llama-completion test-phone-mixed-region-selector -j 8
    ctest --test-dir build-tensor-v2 -C Release -R test-phone-mixed-region-selector --output-on-failure

Run one prompt_N.txt without exposing any layer/Chunk allocations as
command parameters:

    .\tools\region-ring-experiments\run_phone_mixed_smoke.ps1 `
      -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
      -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" `
      -Rpc "10.39.52.114:50052" -NoMmap

The script sets the **PC-side** mapping and restores previous environment
values on exit, including GRAPHLESS=0. It forces the internal smoke
layout and all three Chunk widths to 64. These are A/B TEST settings
only; the ordinary Planner will eventually determine layer/Chunk choices.
The smoke uses one generated token and preserves full HTP local-KV
writeback and OpenCL KV. Both Phone OpenCL Regions are P1. Earlier
P5 terminal graphs produced unsupported 2-way Phone backend splits.
HTP no longer needs OpenCL Slot0 at all: its Plan/KV-bound Graphless
jobs are native HTP. GPUOpenCL still requires OpenCL Slot0 and its
persistent pool; missing Slot0/pool options fail at Phone RPC startup.
Tiny warmup/decode can legally have two splits on the last Phone Stage:
Phone Meta last layer followed by PC CPU output Head; the mixed guard
allows only this exact two-backend split.

## What proves a real mixed run

Look for ALL of:

- PC `[PHONE_MIXED_BACKENDS] region=2/5 ... backend=HTP0`.
- PC `[PHONE_MIXED_BACKENDS] region=8/11 ... backend=GPUOpenCL`.
- PC `[PHONE_OWNED_KV] side=pc event=BIND ... status=PASS` for regions 2/5.
- PC `[PHONE_MIXED_BACKENDS] event=GRAPHLESS_READY ... backend=HTP0`.
- Phone `[PHONE_QWEN3_LOCAL] event=EXECUTE_ONLY ... backend=HTP0 ... status=PASS`.
- Phone `[PHONE_MIXED_BACKENDS] event=EXECUTE ... backend=GPUOpenCL ... status=PASS`.
- Phone `[REAL_STAGE_PROFILE] side=PHONE phase=CHUNK ... backend=HTP0/GPUOpenCL`.
- Complete P2C, prefill and KV synchronization without fallback.

Note that one Phone GPUOpenCL RPC-visible device still exists even when
the NPU-only mode is selected: visibility does not prove OpenCL compute.
For this smoke, require explicit GPUOpenCL COMPUTE PASS events.

## Scope and next steps

This branch proves mixed GPU/HTP execution only after the real-device smoke
passes. Code was reviewed, but **not Android/Windows compiled or hardware
verified** in the development environment.

The existing offline DAG Planner still models three device types:
PC CPU, PC CUDA, and aggregate PHONE. The current real Stage CSV has a
single `PHONE` cost class and cannot safely compare HTP with OpenCL
separately. Do not treat its automatic result as a four-backend decision.

To reach the requested automatic model-specific Profile -> Plan process:

1. Extend the separate Profile bundle to record Phone GPU/OpenCL and HTP0
   costs independently (compute, setup, KV, memory, transfer) with actual
   backend labels and model identity. Never silently reuse HTP as GPU cost.
2. Extend the DAG Planner to choose GPU/NPU Phone Region ownership as well
   as G/C/P layer counts and XC/XG/XP. Include both memories and HTP Bank
   weights, and prohibit NPU Region widths !=1 on the current backend.
3. Implement lazy canonical KV binding for deferred rank>=2 HTP Regions
   when PC-built metadata graphs are used; verify Bank refill and KV parity.
4. Optionally use per-Region graphless NPU when compatible with the GPU
   RPC graph lifecycle. Keep FIFO serial first; overlap is a later experiment.
5. Have saved plans output ordered NPU Region and absolute model Layer lists
   for Phone setup. Fail closed if server mapping and chosen Plan differ.

Separate Profile is a mandatory precondition for formal Plan. The
`run_phone_mixed_smoke.ps1` command performs no model-load profiling
(`-SkipProfile`); it is a fixed A/B hardware correctness experiment.
