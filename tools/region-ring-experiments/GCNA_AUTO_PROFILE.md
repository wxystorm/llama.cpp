# G/C/N/A compact Profile with automatic Phone telemetry (RPC Patch33)

Branch: feat/four-backend-profile-rpc-gcna-20261107
Status: experimental code integrated; full Windows/Android build and real Snapdragon run are NOT verified.

## What it does

Give a sampled physical Cycle, for example G7,C1,N1,A1 repeated three times.
For original Qwen3-14B (40 Transformer layers), the Python model builder
creates a quantized compact 30-layer GGUF from exactly the original first
30 Transformer blocks. It preserves the original embedding, Head and
tokenizer metadata. It does NOT execute the other 10 layers on CPU.
The compact model is a real timed Stage probe; its logits are not those
of the original full 40-layer model and do NOT establish inference parity.

PC automatically converts G/C/N/A to separate G/C/P/P runtime Stages,
assigns consecutive layer offsets, derives HTP Region indices (2,6,10),
and sends those absolute layer indices to the Phone by Plan RPC.
The operator does not enter NPU Region/layer IDs. The Phone server
continues to use the original full model, with matching source weight
tensor bytes for the sampled prefix. Rebuild both sides from this commit.

Phone accumulates measured HTP and Adreno per-Chunk service data, then
exposes fixed-size paged Patch33 RPC snapshots after each Run END.
PC pulls the records once (not per Chunk), verifies Run IDs and logs them
inside its own engine.stderr.log. No manual copying of phone.log.
Phone additionally returns GPU/Host available memory, observed Stage
Arena peak, configured Host staging cap and actual third-Bank Host
Stage/Repack bytes and elapsed time. Missing measurements cause explicit
failure, never fabricated bandwidth.

## Phone launch

Phone must be rebuilt with RPC Patch33, the original GGUF and its usual
OpenCL/HTP runtime paths. The required environment settings are:

    unset GGML_RPC_PHONE_NPU_SERIAL_REGIONS
    unset GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS
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

    ./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d GPUOpenCL -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf"

## One command Profile (Windows PowerShell)

Use the ORIGINAL 40-layer GGUF for -Model. Python needs the bundled
gguf-py package and NumPy (e.g. python -m pip install -e .\gguf-py).
Keep PC compiled with Patch33; build llama-completion and llama-tokenize.

    .\tools\region-ring-experiments\run_gcna_profile.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" -Pattern "G7,C1,N1,A1" -Cycles 3 -NoMmap

The script creates a real compact GGUF from original blk.0..blk.29
and runs Chunk widths 16,32,64,128 with the current Stage executor.
G/C/N/A is the only input topology; PC calculates all physical Region IDs.
The Phone server stays connected for all tests and sends its Profile
directly to PC over RPC. It does not require a Phone log file on PC.

Output:

    model-profiles/Qwen3-14B-Q4_K_M/
       real-stage-mixed.csv             # CPU/GPU/PHONE/PHONE_GPU
       phone_mixed_calibration.json     # if base Profile already exists
       model_profile.json              # preexisting ORIGINAL-model base
       compact-probes/gcna-G7-C1-N1-A1-c3/
          *.gguf                       # compact timing-only GGUF
          *.probe.json                 # original/probe fingerprints
          runs/x16/, x32/, x64/, x128/

Existing full-model base Profile remains mandatory for CPU/CUDA weight
budget and PC<->Phone link calibration; it is NOT synthesized here.
If the base Profile is missing the Stage CSV is retained, but the runner
warns that final Plan requires the original-model model_profile.json.
The automated mixed calibration requires measured third-bank refill;
otherwise it refuses to invent NPU Host stage/repack bandwidth.

## Offline Plan on the ORIGINAL full model

    .\tools\region-ring-experiments\run_plan.ps1 -PhoneMixed -RuntimeCompatible -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" -Out ".\offline-plans\gcna-512"

Outputs plan.json (selected physical G/C/N/A Cycle and handoffs) and
candidates.csv. Offline Plan never contacts Phone or profiles again.

To run the Plan, use run_saved_plan.ps1 with -ConfirmBankedPhoneReady
only after rebuilding/validating matching Patch33 Phone and PC binaries.
Use ORIGINAL 40-layer GGUF for this full-model run, not compact probe.

## Verify

PC log: [PHONE_STAGE_PROFILE_RPC] event=COMPLETE ... status=PASS
PC log: [REAL_STAGE_PROFILE] side=PHONE phase=CHUNK source=RPC
PC log: [PHONE_PROFILE_RESOURCES] side=PHONE source=RPC
PC output: [GCNA_PROFILE] status=COMPLETE ... real-stage-mixed.csv
CSV contains CPU, GPU, PHONE and PHONE_GPU measured classes.

All these measurements are Stage service timings for the sampled shape.
Do not infer original full-model logits parity, successful Decode or
unmeasured multi-layer NPU behavior from them.
