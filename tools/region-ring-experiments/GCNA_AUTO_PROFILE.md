# G/C/N/A compact Profile with automatic Phone telemetry (RPC Patch32)

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
exposes fixed-size paged Patch32 RPC snapshots after each Run END.
PC pulls the records once (not per Chunk), verifies Run IDs and logs them
inside its own engine.stderr.log. No manual copying of phone.log.
Phone additionally returns GPU/Host available memory, observed Stage
Arena peak, configured Host staging cap and actual third-Bank Host
Stage/Repack bytes and elapsed time. Missing measurements cause explicit
failure, never fabricated bandwidth.

## Phone launch

Phone must be rebuilt with RPC Patch32, the original GGUF and its usual
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
Keep PC compiled with Patch32; build llama-completion and llama-tokenize.

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


## Backend-specific Stage depth matrix (experimental)

Run on Windows after rebuilding Patch32 on PC and phone:

    .\tools\region-ring-experiments\run_backend_profile.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" -G "1,2,4,6,8" -C "1,2,3,4" -N "1,2,3" -A "1,2,4,6" -Chunks "16,32,64,128" -Repeats 2 -NoMmap

For each depth/Chunk combination, the script executes the original
weights of a compact GGUF. It keeps minimum feeder G/C/N/A Stages to satisfy
the current hardware runtime, and extracts **only the named backend**
service timings. Thus curves are independent of a chosen full-model Plan
but measurements are **not standalone single-device microbenchmarks**.
The source manifest records `standalone_compute=false`. Phone N2/N3
are explicitly listed as unsupported; no fake linear NPU measurement.
An interrupted sweep resumes previously successful cases.

Output: `backend-matrix/backend-<G/C/N/A>-L<depth>/...`,
`backend-matrix/matrix_manifest.json`, and
`real-stage-mixed.csv` with independently selected target rows.
An existing original-model `model_profile.json` supplies measured
network/H2D/D2H rates. The legacy base Profile captures these from
real `HYBRID_PROFILE` transfer-only tensor-copy probes.

To export a separate per-transfer-size communication CSV from those
recorded real PC/RPC transfer probes:

    .\tools\region-ring-experiments\run_communication_profile.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf"

This produces `communication-profile.csv` with direction, payload bytes,
median service and wall time, sample count and effective MiB/s. It will
fail if a required direction was not actually measured. It is an
**exporter of existing transfer-only probes**, not a new standalone
network test. The Phone NPU->Adreno Host ownership/Upload and independent
Bank-only benchmark executables remain future work.

## Device-selected terminal Stage (experimental PC runtime)

The four-backend offline DAG now compares final Transformer ownership
on HTP (N), Phone OpenCL (A), PC CPU (C), or PC CUDA (G) and ranks
candidates by predicted critical-path latency instead of forcing Phone
at the end. After at least two full Phone Cycles it may assign the
remaining layers to a final G or G,C PC tail. It models the real
intermediate P2C transfer, PC CPU Head after final hidden-state
publication, and an additional measured CUDA->CPU readback for G-last.
Unknown Head/D2H costs **reject** PC-final candidates; they are not
assumed zero. `candidates.csv` includes `final_backend` and
`runtime_compatible`.

The experimental C++ runtime now accepts PC-terminal APPLY_WIDE layouts,
keeps Phone autonomous for intermediate Regions, and defers the
last G/C Stage's output Head into the shared canonical CPU Head. A
tiny (<32-token) PC-final run uses the serial staged fallback;
normal full prefill uses the worker DAG with an explicit terminal
hidden-state Host buffer. Saved schema-2 Plans can be applied after
`-ConfirmBankedPhoneReady` with an appropriately rebuilt server.

**Status: code committed; NOT built or proven on Windows/Android.**
The PC-final execution path is *experimental* and still requires
physical integration tests, full prefill/decode parity, GPU final D2H
validation, and phone-native two-bank synchronization checks. The
`runtime_compatible` flag is a topology/feature contract, **not PASS**.

The HTP executor remains restricted to one physical Transformer layer
per Region. An independent NPU multi-layer executor and the dedicated
communication-only sweep remain unfinished.

## Offline Plan on the ORIGINAL full model

    .\tools\region-ring-experiments\run_plan.ps1 -PhoneMixed -RuntimeCompatible -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" -Out ".\offline-plans\gcna-512"

Outputs plan.json (selected physical G/C/N/A Cycle and handoffs) and
candidates.csv. Offline Plan never contacts Phone or profiles again.

To run the Plan, use run_saved_plan.ps1 with -ConfirmBankedPhoneReady
only after rebuilding/validating matching Patch32 Phone and PC binaries.
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
