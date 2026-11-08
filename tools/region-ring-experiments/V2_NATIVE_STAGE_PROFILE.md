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

Initial scope: Qwen3 Dense, 1..32 consecutive layers, KV divisible by X,
and native supported RoPE. Unsupported model parameters fail closed.

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
backend, layers, X, KV, repeats and comparison tolerance. --update
forces reruns. Missing/failed samples are never synthesized.

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

## Scope and remaining work

The N4 standalone probe intentionally keeps its complete Stage weights
resident. It does NOT yet connect the N4 graph to production per-Cycle
HTP two-Bank refill/eviction, persistent Decode KV or a versioned
Phone RPC Stage command. Automatic PC-initiated Phone profiling has not
been wired into the main program.

Host-only model-free V2 Stage and N2/N4 metadata tests, a synthetic
Qwen3 native CPU N1/N2 run, and cache/resume/parity tests are now
registered. Windows/Android compilation and real Qwen3 HTP N2/N4
parity have NOT been verified in this environment.

Next: run PC/Phone build and numerical checks, bind N4 to existing
HTP two-Bank Runtime and RPC, then add communication/Bank profiles
and only afterwards the V2 Planner.
