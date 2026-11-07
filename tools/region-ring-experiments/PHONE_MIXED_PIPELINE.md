# Phone HTP0 + Adreno GPUOpenCL: dual-lane Chunk pipeline (experiment)

Branch: `feat/phone-htp-opencl-dual-lane-pipeline-20261107`
Base: `feat/phone-opencl-htp-whole-layer-mix-20261107` (already passed
512-token prefill in serial mixed mode). This branch does not change the
40-layer model split or the offline planner.

## What changes

Previously the Phone had one autonomous compute thread. The selected
HTP0 layers (L9, L19) and GPUOpenCL layers (L29, L39) were serialized,
although they use different physical processing units.

With `GGML_RPC_PHONE_NPU_GPU_PIPELINE=1` on the Phone:

- One compute worker selects **only** HTP0 jobs (Region 2 and 5).
- A second compute worker selects **only** GPUOpenCL jobs (Region 8 and 11).
- Both share the existing *verified input* and *prepare* worker.
- Under the existing `phone_auto_mutex`, every Chunk is atomically
  claimed once and the **per-Region Chunk FIFO** is still checked.
- The HTP worker holds its private HTP backend mutex and directly uses
  Phone-local F32 input/KV/host P2C; OpenCL holds the graph registry mutex
  and its existing Slot0/dual-slot checks. There is **no new shared compute
  mutex**.
- The existing P2C Arena is thread-safe and can accept completed frames
  from either lane. The PC dependency scheduler still controls when input
  for each Region/Chunk is sent; this does not let downstream Layer 29
  consume Layer 9 output before the PC's intervening layers are ready.
- NPU physical two-weight-Bank refill and per-layer KV remain unchanged.
- Default or `GGML_RPC_PHONE_NPU_GPU_PIPELINE=0` preserves the known
  serial execution path.

This is **different-Chunk overlap**, not simultaneous GPU+NPU within a
single Transformer Layer and not a direct NPU->GPU activation transfer.
The actual layer order for one Chunk is:

    PC CUDA L0..7 -> PC CPU L8 -> Phone HTP L9
      -> PC CUDA L10..17 -> PC CPU L18 -> Phone HTP L19
      -> PC CUDA L20..27 -> PC CPU L28 -> Phone GPU L29
      -> PC CUDA L30..36 -> PC CPU L37..38 -> Phone GPU L39
      -> PC Head

## Phone Termux setup

Recompile the Android RPC server from this branch first. Start from the
previous working mixed setup and add the *new* pipeline switch:

    export GGML_RPC_PHONE_NPU_SERIAL_REGIONS=2,5
    export GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS=9,19
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
    export GGML_RPC_PHONE_CHUNK_TIMELINE=1
    export GGML_RPC_REAL_STAGE_PROFILE=1

    ./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 \
       -d GPUOpenCL -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
       2>&1 | tee phone-pipeline.log

Verify the boot log includes:

    [PHONE_MIXED_PIPELINE] event=START mode=DUAL_COMPUTE_LANES htp_workers=1 gpu_workers=1

If missing, the Phone RPC binary is stale or the switch was not read.

## PC setup and experiment

Compile matching Windows code and the host policy test:

    cmake --build build-tensor-v2 --config Release --target llama-completion test-phone-mixed-region-selector test-phone-mixed-pipeline -j 8
    ctest --test-dir build-tensor-v2 -C Release -R test-phone-mixed --output-on-failure

Use the existing Qwen3-14B `prompt_512.txt`:

    .\tools\region-ring-experiments\run_phone_mixed_smoke.ps1 `
      -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
      -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" `
      -Rpc "10.39.52.114:50052" -NoMmap -NPredict 1

This first smoke measures mixed Prefill only. To exercise Decode
and the KV sync/warmup transition, run a *second* test with
`-NPredict 32`. Do not assume Decode correctness from the 1-token PASS.

## How to prove real overlap

On Phone:

    grep -E '\[PHONE_MIXED_PIPELINE\]|\[PHONE_AUTO_SUM\]|\[PHONE_QWEN3_LOCAL\]|\[PHONE_STAGE_SLOT0\]' phone-pipeline.log | tail -n 120

For each accepted compute task, `CLAIM` is recorded before device lock
acquisition. Only `DEVICE_BEGIN`/`DEVICE_END` are used for overlap:

    [PHONE_MIXED_PIPELINE] event=DEVICE_BEGIN run=... region=2 chunk=... backend=HTP0 phone_us=...
    [PHONE_MIXED_PIPELINE] event=DEVICE_BEGIN run=... region=8 chunk=... backend=GPUOpenCL phone_us=...
    [PHONE_MIXED_PIPELINE] event=DEVICE_END   run=... region=2 chunk=... backend=HTP0 phone_us=...
    [PHONE_MIXED_PIPELINE] event=DEVICE_END   run=... region=8 chunk=... backend=GPUOpenCL phone_us=...

At RUN teardown:

    [PHONE_MIXED_PIPELINE] event=SUMMARY run=...
      htp_jobs=... gpu_jobs=...
      htp_device_service_ms=... gpu_device_service_ms=...
      htp_gpu_service_overlap_ms=... status=OVERLAP_MEASURED

The measured overlap uses **only Phone monotonic timestamps** and
excludes queue admission and waiting for the two device mutexes.
`htp_gpu_service_overlap_ms > 0` demonstrates simultaneous backend
*service calls* on two workers. Those calls include input/output transfers,
so this is **not proof that both GPU and HTP arithmetic kernels executed
at the same instant**. A kernel-level overlap claim needs separate
hardware-event profiling. `NO_OVERLAP_MEASURED` is not
automatically a correctness failure: causal readiness, short Chunks,
CPU/CUDA time, or lock contention may leave no simultaneous jobs.

For A/B comparison, restart Phone once with
`GGML_RPC_PHONE_NPU_GPU_PIPELINE=0` (serial baseline) and once with
`=1` (dual lanes), using the same Prompt, model, chunk layout, and
other runtime settings. Compare Prefill TPS, Phone compute intervals,
P2C timings, and graph/KV/P2C status, not only a single throughput
number. Preserve Android thermal and memory headroom.

## Safety and verification status

- Two separate Phone compute threads share one task map with atomic
  per-Chunk claim and per-Region FIFO. Output Arena is reservation-safe.
- The switch refuses to start without mixed OpenCL/HTP server mode.
- The implementation is code-complete for an initial experiment and
  includes a host policy test, but has **not been validated on your real
  Snapdragon HTP+Adreno device**. A PASS is not guaranteed.
- The NPU Bank mover already uses a third worker. It is not promoted to
  an independent model compute lane.
- Four-backend Profile/Plan remains separate future work; this is a
  fixed-layout correctness and overlap test.
