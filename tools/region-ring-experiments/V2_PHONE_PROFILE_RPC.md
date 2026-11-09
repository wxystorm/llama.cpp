# Native V2 Phone Profile inside ggml-rpc-server (one process, one port)

Branch: `feat/region-ring-v2-stage-profile-rebuild-20261108`.  
Wire: **GGML-RPC 4.0.35**, command 57, bounded V2 JSON payload v1.

## Architecture

There is **one Phone process**: `ggml-rpc-server`, listening only on the
existing inference RPC port `50052`.

```text
PC run_prompt_sweep.ps1 / run_v2_stage_matrix.py
    |
    | GGML-RPC HELLO 4.0.35 + command 57 (port 50052)
    v
Phone ggml-rpc-server (C++)
    |
    +-- parse, authenticate and validate bounded Profile shape
    +-- verify exact original GGUF SHA256
    +-- call v2_stage_profile_run_embedded() directly in-process
    |       N = HTP0 graph + CPU parity; A = Adreno OpenCL graph
    |       measured compute/prepare/weight refill and graph memory
    +-- return native CSV fields as JSON, PASS/PARITY_FAIL/ERROR
    |
    +-- existing inference/Plan/Phone-owned execution RPC commands
```

The original independent C++ profiler source is compiled into the Android
`ggml-rpc-server` binary by CMake. It uses the same HTP precompile, GGML
Graph, and CPU parity logic as the standalone tool, without copying or
reimplementing the measurement kernels.

**No Python, fork, exec, background worker, subprocess, standalone profiler
executable, second Termux window or TCP port 50177 is required on Phone.**
The PC-side Python orchestration scripts remain for cache/resume/scheduling
and for the G/C native executable; Phone execution itself is C++ only.

## Build and deploy

Cross-compile in the existing NDK r29 / Hexagon / OpenCL build environment:

```bash
cd /workspace
git pull --ff-only
cmake --build build-phone-npu --target ggml-rpc-server -j 8
```

The Phone deployment contains only:

```text
~/phone-v2-profile/
    bin/ggml-rpc-server
    lib/*.so                 (matching Android GGML/HTP/OpenCL libraries)
```

No Python scripts or `region-ring-v2-stage-profile` executable need to be
copied to Phone. Retain the working HTP DSP libraries and vendor runtime
dependencies from the validated NPU setup.

You must also rebuild the PC binary from protocol 4.0.35 source:

```powershell
git pull --ff-only
cmake --build build-tensor-v2 --config Release `
  --target llama-completion region-ring-v2-stage-profile -j 8
```

## Start Phone (Termux)

Keep the previously working OpenCL/HTP driver environment:

```bash
cd ~/phone-v2-profile
export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
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

./bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d GPUOpenCL \
    -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 2>&1 | tee phone-v2.log
```

Expect on startup:

```text
[V2_PROFILE_NATIVE] state=REGISTERED port=50052 process=RPC_ONLY
```

On an actual N/A profile request, expect:

```text
[V2_PROFILE_NATIVE] event=RUN backend=N layers=1 chunk=32 kv=512
[V2_PROFILE_NATIVE] event=DONE backend=N layers=1 status=PASS
```

If HTP numerical parity fails, the status must be `PARITY_FAIL`, never
silently treated as a valid Planner performance point. An N2/N4 profile PASS
does not prove that production runtime supports multi-layer HTP Regions.

## PC verification

In PowerShell, check the single-port hello first:

```powershell
python -c "import sys; sys.path.insert(0, 'tools/region-ring-experiments'); from v2_profile_wire import call; print(call('10.39.52.114:50052', 'hello'))"
```

Expected JSON fields: `status=OK`, `transport=V2_PHONE_PROFILE_RPC50052`,
`model_sha256`, `exe_sha256`, `runtime_sha256`, `device_id`.

The PC verifies the Phone SHA256 matches its own GGUF file before accepting
N/A samples. Phone runtime identity includes the **RPC server executable**
and selected GGML/HTP/OpenCL libraries; changes invalidate the N/A cache.
Previously valid G/C records are independently reusable.

Run a minimal N1/A1 independent matrix before a full 96-case sweep:

```powershell
python .\tools\region-ring-experiments\run_v2_stage_matrix.py `
  --exe ".\build-tensor-v2\bin\Release\region-ring-v2-stage-profile.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --phone "10.39.52.114:50052" `
  --out ".\profile-v2\gcna-native-smoke" `
  --backends NA --layers "1" --chunks "32" --kvs "512" `
  --warmups 1 --repeats 2 --skip-tcp-transfer
```

Only when N/A smoke succeeds, run the usual `run_prompt_sweep.ps1` with
`-Rpc "10.39.52.114:50052"`; do not supply `-V2ProfilePhone`, launch
a Python Phone service, or forward 50177.

## Limits

- Native in-process graph profiling currently serializes all N/A probes on
  Phone. Do not run a separate inference job during sampling; a global
  cross-command GPU/HTP idle fence is not yet present.
- The independent native sampler can load extra weights/graphs alongside
  the existing RPC Server, so memory feasibility must still be enforced.
- Unlike the old raw-TCP 50177 diagnostic, production transfer calibration
  must come from actual ggml-rpc Tensor/Bank timings. No synthetic TCP
  measurements are marked Planner-eligible.
- The RPC server is experimental and not hardened for an untrusted network.
- Compiled code is **not** a Snapdragon runtime PASS. Confirm NPU precompile,
  CPU parity, real GCNA plans and end-to-end logits on physical hardware.
