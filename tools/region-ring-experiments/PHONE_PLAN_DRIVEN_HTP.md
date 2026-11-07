# Planner-driven Phone HTP assignment (RPC patch 30)

Branch: feat/phone-plan-driven-htp-map-20261107
Status: experimental; rebuild PC and Phone from the same commit.

## Why

Phone Region IDs and whole-Transformer-Layer IDs are outputs of the PC
plan, not persistent Phone environment settings. They now travel over
RPC_CMD_PHONE_NPU_PLAN_CONFIGURE after the input session START and before
the Phone-owned Graph/KV/Chunk registration.

The PC sends a sorted list of (Region, Layer) tuples. Phone validates
one Layer per HTP Region, two or more Regions in two-Bank mode, model
local GGUF weight availability, actual Bank allocation and correct
ownership. Rank 0/1 weights are primed after the Plan is accepted.
All other Phone Regions retain GPUOpenCL PC-built Graph ownership.
Any allocation/configuration failure aborts the run; no backend fallback.

## Phone: no Region/Layer exports

Rebuild Android RPC Server and unset any old per-layout variables:

    unset GGML_RPC_PHONE_NPU_SERIAL_REGIONS
    unset GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS

Set generic backend capabilities, independent of chosen layout:

    export GGML_RPC_PHONE_NPU_PLAN_CONTROLLED=1
    export GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2
    export GGML_RPC_PHONE_NPU_MIXED_OPENCL=1
    export GGML_RPC_PHONE_NPU_GPU_PIPELINE=1

    export GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_INPUT=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=1
    export GGML_RPC_PHONE_QWEN3_LOCAL_GRAPHLESS=1
    export GGML_RPC_PHONE_NPU_HOST_STAGE_MIB=512

    export GGML_RPC_PHONE_BINDING_PREFLIGHT=1
    export GGML_RPC_PHONE_STAGE_SLOT0_EXEC=1
    export GGML_RPC_PHONE_STAGE_DUAL_SLOT=1
    export GGML_RPC_PHONE_PERSISTENT_POOL_PREP=1
    export GGML_RPC_REAL_STAGE_PROFILE=1
    export GGML_RPC_PHONE_CHUNK_TIMELINE=1

    ./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d GPUOpenCL -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 2>&1 | tee phone-plan-controlled.log

At startup, expect PHONE_MIXED_BACKENDS plan_controlled=1,
htp_regions=0, status=READY.

After the PC selects the mapping and begins Prefill, Phone must log:

    [PHONE_NPU_PLAN] side=phone event=CONFIGURE ... status=PASS
    [NPU_BANK_PLAN] rank=0 bank=0 region=... layer=... origin=PC_PLAN
    [NPU_BANK_PLAN] rank=1 bank=1 region=... layer=... origin=PC_PLAN
    [NPU_BANK_MEMORY] event=ALLOCATED ... origin=PC_PLAN status=PASS

PC must log PHONE_NPU_PLAN event=CONFIGURE/INSTALL status=PASS.

## PC smoke

Synchronize PC and Phone to this branch: patch version 30 is incompatible
with the old patch 29, by design. PC Windows commands:

    git fetch origin
    git switch feat/phone-plan-driven-htp-map-20261107
    git pull --ff-only
    cmake --build build-tensor-v2 --config Release --target llama-completion test-phone-plan-region-map -j 8
    ctest --test-dir build-tensor-v2 -C Release -R test-phone-plan-region-map --output-on-failure

Run the existing validated mixed topology:

    .\tools\region-ring-experiments\run_phone_mixed_smoke.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" -Rpc "10.39.52.114:50052" -NoMmap -Chunk 64 -NPredict 1

This first test uses a fixed *PC* selection of HTP Region 2/L9
and Region 5/L19. The selected entries are automatically sent to the
Phone and used to initialize both physical Banks. No manual Phone
Region/Layer assignment remains.

## Planner integration and remaining limits

The offline four-backend DAG planner now emits phone_npu_mapping
and phone_gpu_mapping, with region/layer indices derived from the
selected G/C/N/A cycles; zero-width stages are omitted.

The offline schema=2 plan still has runtime_compatible=false:
contiguous NPU->Adreno stages within one Cycle require a separate
Region Ring runtime change. This branch delivers the mapping protocol
and verified fixed-topology experiment; it does NOT yet execute arbitrary
G/C/N/A plans. Do not pass schema=2 into run_saved_plan.ps1.

Different HTP mappings in the SAME Phone process currently reject
with PLAN_MAPPING_CHANGED_RESTART_PHONE_RPC_REQUIRED. Restarting
the same generic Phone RPC command requires zero layout edits. Safe
dynamic graph/Bank teardown and reconfigure remains future work.

The first Plan configuration moves HTP weight priming into startup of
the first Prefill run: benchmark setup separately from steady-state
compute. No real Snapdragon device verification has been performed.
