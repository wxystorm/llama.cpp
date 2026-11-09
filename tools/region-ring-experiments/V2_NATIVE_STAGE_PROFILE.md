# V2 native first-N layer Profile

Branch: feat/region-ring-v2-stage-profile-rebuild-20261108

Native code: tools/region-ring-experiments/v2_stage_profile.cpp
Runner: tools/region-ring-experiments/run_v2_stage_matrix.py

## What actually executes

For G4, X64, KV512:
- Read the real Layer 0..3 Transformer weights ONLY from the original
  GGUF to the selected CUDA backend. No Compact GGUF.
- Build one contiguous four-layer Qwen3 GGML Graph. Each layer has QKV,
  RoPE, Flash Attention, FFN, and its own persistent K/V.
- Allocate and populate historical KV. For KV512/X64, run the final
  64-token Chunk at positions 448..511.
- Execute that Graph only on CUDA; C4 uses CPU instead. No auxiliary
  Stage, Phone RPC, Embedding or final LM Head.
- Separately measure weight upload, Graph build, input prepare and
  synchronized native Compute; reject unsupported op/backend.

Initial scope: Qwen3 Dense, 1..32 consecutive layers, KV >= X,
and native supported RoPE. For any X (including 96) the graph
measures the exact final X tokens at positions KV-X .. KV-1 and
records token_begin. Unsupported model parameters fail closed.

## Windows PC

Build in an existing CUDA-enabled CMake configuration:

    cmake --build build-tensor-v2 --config Release --target region-ring-v2-stage-profile -j 4

A typical executable path is:
build-tensor-v2/bin/Release/region-ring-v2-stage-profile.exe

PowerShell (one command line):

    .\build-tensor-v2\bin\Release\region-ring-v2-stage-profile.exe --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" --backend G --layers 4 --chunk 64 --kv 512 --repeats 8

For CPU, use --backend C and optionally --threads 8.

Resumable G/C matrix (one line):

    python tools/region-ring-experiments/run_v2_stage_matrix.py --exe "build-tensor-v2/bin/Release/region-ring-v2-stage-profile.exe" --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" --out "profile-v2/pc" --backends GC --layers "1,2,4,6" --chunks "16,32,64,128" --kvs "128,512,1024"

The orchestrator saves each PASS sample and stage-profile-v2.csv.
Its cache key includes the full GGUF SHA256, executable SHA256,
device identity, backend, layers, X, KV, repeats and comparison tolerance.
--update forces reruns. Missing/failed samples are never synthesized.
PARITY_FAIL is cached as an observed inaccurate result, NOT as PASS.

Compute output includes compute_min_ms, compute_median_ms,
compute_p90_ms, compute_max_ms and compute_stddev_ms, plus independent
activation_upload_median_ms and activation_readback_median_ms measured
on each selected GGML physical backend. These isolated set/get times
are NOT the CPU/Phone network transport cost.

## Phone HTP N2 and N4

Build the same target in the Android/Termux configuration with the
Hexagon HTP backend enabled. The device must be HTP0. Quantized GGUF
weights allocate with the native HTP repack buffer, unquantized
weights with the native device buffer; no CPU/OpenCL fallback.

Phone shell:

    ./build/bin/region-ring-v2-stage-profile --model "$HOME/models/Qwen3-14B-Q4_K_M.gguf" --backend N --layers 2 --chunk 64 --kv 512 --warmups 1 --repeats 3 --verify-cpu

After N2 PASS, try changing --layers to 4.

The native runner builds a single N-layer Graph, allocates per-layer
KV, binds fixed Chunk inputs, invokes
ggml_backend_hexagon_precompile_fixed_graph, executes HTP Graph,
synchronizes, and clears the fixed Graph cache before freeing metadata.

For N, --verify-cpu is mandatory: the identical true-weight layer
prefix executes on CPU as a reference and compares final hidden
Activation using relative RMSE and max absolute error. Initial
thresholds are relative RMSE <= 0.02 and max abs <= 0.5, adjustable
with --relative-rmse and --max-abs-error. These are initial testing
policies, not measured Snapdragon tolerances.

Phone GPU uses --backend A and requires OpenCL.

To measure actual HTP host-staged weight-repack time without disk
IO, run --backend N with --bank-repeats 3 and optionally
--host-stage-cap-mib 1024. This stages the original raw GGUF weights
once in Phone Host RAM, repeatedly calls native HTP Tensor-set on
the SAME repack Buffer, synchronizes, and records
same_bank_refill_median_ms. weight_file_read_ms and
weight_backend_set_ms are independently measured, alongside the
initial aggregate weight_upload_ms.

THIS IS NOT A DUAL-BANK REFILL OR OVERLAP TEST. The native
Profile code does not allocate/rebind a second Bank here; do not
interpret the result as proof of dual-Bank execution readiness.
It can consume substantial Host memory and is disabled by default.
Every measured single-Stage Buffer allocation is tracked, while
two_weight_banks_allocation_sum_bytes in profile.json is only a
sizing estimate, never verified free memory.

## N4 accuracy diagnostics

On the user's Snapdragon 8 Elite / Hexagon v79, an initial native
Qwen3-14B Q4_K_M, X64, KV512 experiment produced:

- N2: HTP graph 70 nodes, native compute 21.914 ms, CPU parity
  relative RMSE 0.01235 and MaxAbs 0.14021: PASS.
- N4: HTP graph 140 nodes precompiled and executed successfully,
  but CPU parity relative RMSE 0.07100 and MaxAbs 2.28937: FAIL.

These are user-reported observations, not repo CI results. Do NOT relax
accuracy thresholds just to produce N4 PASS or infer that N4 cannot be
compiled/executed.

Use --diagnose-layers on the phone to capture each l_out from the **same
N4 Graph** and compare with the CPU Graph at Layer 0/1/2/3:

    ./bin/region-ring-v2-stage-profile --model "$HOME/models/Qwen3-14B-Q4_K_M.gguf" --backend N --layers 4 --chunk 64 --kv 512 --warmups 1 --repeats 3 --verify-cpu --diagnose-layers

Each [V2_LAYER_PARITY] event includes relative RMSE, MaxAbs, cosine,
reference RMS and the worst element. A diagnostic run marks all layer
outputs as GGML outputs, keeping their buffers resident. This changes
Graph allocation and can affect timing; the CSV is labeled with
NATIVE_SINGLE_BACKEND_GGUF_PREFIX_DIAGNOSTIC, not a regular planner sample.

For an execution-history A/B, also run with --warmups 0 --repeats 1:
a change in parity between single and repeated execution may indicate
KV update, reused allocation, fixed Graph state or synchronization issues.



## PC-driven Phone Profile control channel

The integrated versioned V2 Profile command on ggml-rpc:50052 lets PC request
actual N/A samples and collect their metrics without copying Phone CSV
or logs by hand. See V2_PHONE_PROFILE_RPC.md for the USB adb forward
setup and one-command GCNA sweep. The independent native sampler is NOT the
existing ggml-rpc inference port 50052, and is not yet merged into the
production Region Ring runtime.

Native parity failures now return exit code 2 and a complete CSV row
with parity=FAIL, so the PC can save the real N4 compute_ms for
diagnosis while keeping it out of the PASS/planner-eligible CSV.

## Scope and remaining work

The N4 standalone probe intentionally keeps its complete Stage weights
resident. It does NOT yet connect the N4 graph to production per-Cycle
HTP two-Bank refill/eviction, persistent Decode KV or a versioned
Phone inference RPC Stage command. The standalone Profile RPC path
exists, but production Phone initialization is still independent.

Host-only model-free V2 Stage and N2/N4 metadata tests, a synthetic
Qwen3 native CPU N1/N2 run, and cache/resume/parity tests are now
registered. Windows/Android compilation and real Qwen3 HTP N2/N4
parity have NOT been verified in this environment.

Next: run PC/Phone build and protocol tests, investigate N4 numerical
differences separately, bind N4 to the existing production HTP
two-Bank Runtime, and then add communication/Bank profiles before
V2 Planner consumes the approved compute samples.
