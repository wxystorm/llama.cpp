# V2 Phone native Profile over the existing ggml-rpc port 50052

Branch: `feat/region-ring-v2-stage-profile-rebuild-20261108`  
Protocol: **ggml-rpc 4.0.35**, V2 Profile wire JSON v1.

## What changed

There is **one listening server, one TCP port**. The Phone's normal
`ggml-rpc-server -p 50052` also accepts the bounded native Profile request
(command 57) after the existing GGML-RPC HELLO. The PC side reuses the
`-Rpc PHONE_IP:50052` argument of `run_prompt_sweep.ps1`.

The old independent Python TCP sidecar on `50177` is **not started**.
Instead, the RPC server launches the Python V2 Profile helper automatically
**on first Profile request**. That helper communicates over private inherited
stdio/UNIX socketpair, has no listening TCP socket, and executes the existing
`region-ring-v2-stage-profile` native binary for actual N/A measurements.
This reuses the tested model hashing, parity, and CSV schema; it is **not**
an in-process rewrite of the native profiler.

A worker or native profiler failure becomes a structured error returned to
the PC. No unrelated RPC connection is allowed to run arbitrary shell
commands or choose a different GGUF/profiler path.

## 1. Cross-compile Phone targets (Android arm64)

In the existing Android NDK r29 / Hexagon SDK build directory:

```sh
cmake --build build-phone-npu \
  --target ggml-rpc-server region-ring-v2-stage-profile -j 8
```

On Phone, package/deploy the following with **matching GGML backend and DSP
libraries**. Preserve the Qualcomm/HTP, `OCL_ICD_VENDORS`,
`LD_LIBRARY_PATH`, and `ADSP_LIBRARY_PATH` settings from your last
working NPU+OpenCL configuration.

```text
~/phone-v2-profile/
  bin/ggml-rpc-server
  bin/region-ring-v2-stage-profile
  lib/                    # compatible Android GGML/OpenCL/Hexagon/HTP .so
  v2_phone_profile_server.py
  v2_profile_wire.py
  v2_profile_transfer.py
```

The three Python files must be copied from the **same Git revision** as the
PC scripts; the `v2_profile_transfer.py` dependency is imported by the worker
but no raw-TCP Profile transfer service is started.

Install `python` in Termux if necessary. The RPC process automatically
locates these files relative to its executable (`../v2_phone_profile_server.py`
and sibling `region-ring-v2-stage-profile` under `bin/`). Optional
overrides: `GGML_RPC_V2_PROFILE_SCRIPT` and `GGML_RPC_V2_PROFILE_EXE`
are **local phone administrator** environment variables, never remotely
supplied by a Profile request.

## 2. Start *only* the usual Phone inference RPC

In Termux:

```sh
cd ~/phone-v2-profile
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$PWD/lib"
export GGML_HEXAGON_NDEV=1

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

python --version
./bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d GPUOpenCL \
  -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 2>&1 | tee phone-v2.log
```

Do **not** start `v2_phone_profile_server.py` manually. On the first
missing Profile sample the server should log:

```text
[V2_PROFILE_RPC] status=WORKER_STARTED port=SHARED_50052
[V2_PHONE_PROFILE] transport=RPC50052_STDIO status=READY
```

The Python worker runs only when needed, and the native profiler remains an
independent subprocess for each measured shape.

## 3. PC: run the existing sweep with one RPC endpoint

Build PC and Phone from protocol **4.0.34** source. In Windows PowerShell:

```powershell
git pull --ff-only
cmake --build build-tensor-v2 --config Release `
  --target llama-completion region-ring-v2-stage-profile -j 8

.\tools\region-ring-experiments\run_prompt_sweep.ps1 `
  -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir ".\single-prompt-512" `
  -Out ".\planner-test-results-v2" `
  -Planner apply-wide -Streaming on -NPredict 32 `
  -Threads 8 -Batch 512 -UBatch 512 -LogVerbosity 4
```

There is **no** `-V2ProfilePhone` argument and no `adb forward tcp:50177`.
The PC first checks a model-bound cached V2 Profile; on a miss, it calls
the native matrix with Phone endpoint `10.39.52.114:50052`.

A previously saved 50177 Profile cache is intentionally rejected with
`PHONE_PROFILE_TRANSPORT_MISMATCH`; G/C samples retain their own resume
records. The coordinator then samples missing N/A shapes. If N4 HTP parity
fails, that record is quarantined, not used by the Planner.

Standalone G/C/N/A matrix invocation:

```powershell
python .\tools\region-ring-experiments\run_v2_stage_matrix.py `
  --exe ".\build-tensor-v2\bin\Release\region-ring-v2-stage-profile.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --out ".\profile-v2\gcna-expanded" `
  --phone "10.39.52.114:50052" --backends GCNA `
  --layers "2,4" --chunks "64" --kvs "512" `
  --warmups 1 --repeats 3 --skip-tcp-transfer
```

## 4. Guarantees and limits

- One TCP endpoint; HELLO and native Profile are on the **same GGML RPC**
  transport. The Python control service on 50177 is no longer required.
- A Profile client sends only numerical shape parameters, a model fingerprint
  and an optional token. The server owns all executable/model paths.
- Every HTP result must pass CPU parity; `PARITY_FAIL` rows are excluded
  from the Planner-eligible CSV.
- Samples are serialized by the shared Phone worker lock. **External
  inference clients must remain idle during profiling**; the current RPC
  does not yet provide a global hardware-idle fence for all execution modes.
- The former raw-TCP Profile transfer test is skipped on the integrated
  port. Its stream timing is **not** a production GGML-RPC C2P/P2C timing.
  Retain production RPC transfer/Bank measurements for Planner calibration.
- The first worker startup hashes the full Phone GGUF once. Profile cache
  identity includes Phone hardware, native profiler binary, RPC Server,
  Python worker script and selected backend libraries.
- The production mixed HTP runtime still supports N1 per HTP Region; an
  independent N2/N4 Profile is not proof of production N2/N4 execution.
- Phone RPC is experimental and is not hardened for an untrusted network.
  Expose 50052 only on a trusted private LAN or authenticated tunnel.

**Source/test integration is not a Snapdragon runtime PASS.** Rebuild both
sides, run unit tests, then verify native parity and full Prefill/Decode on
the actual hardware before trusting a GCNA Plan.
