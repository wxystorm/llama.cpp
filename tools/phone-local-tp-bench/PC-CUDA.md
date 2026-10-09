# PC-local CUDA + CPU MoE benchmark (Windows)

The PC branch reuses the phone benchmark's real Qwen3-MoE GGUF weights,
intermediate-width partition, CUDA/CPU parallel FFN, Stage/Join timing,
and correctness checks. The test is entirely local: no phone or RPC.

The historical PHONE_LOCAL_* and PHONE_FULL_* labels are deliberately kept
for comparing logs across platforms.

## Build and verify device enumeration

PowerShell, from the llama.cpp repo root:

~~~powershell
git fetch origin
git switch --track origin/pc-local-tensor-bench
cmake -S . -B build-pc-local -DGGML_CUDA=ON -DGGML_OPENCL=OFF -DGGML_RPC=OFF -DGGML_OPENMP=OFF
cmake --build build-pc-local --config Release --target llama-phone-local-tp-bench --parallel 8
$exe = ".\build-pc-local\bin\Release\llama-phone-local-tp-bench.exe"
$model = "E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf"
& $exe --list-devices
& $exe --gpu-backend cuda --probe-gpu
~~~

If the new branch already exists locally, use git switch pc-local-tensor-bench
and git pull --ff-only origin pc-local-tensor-bench instead of --track.
The device listing should show CUDA and CPU. The GPU probe allocates a small
buffer, but does not yet validate model operators.

If an existing build-cuda tree already enables CUDA, it can be reused by
rebuilding just this target, but a dedicated build directory avoids mixing
experiments with PC/phone RPC.

## First: FFN-only (isolates the local parallel expert computation)

~~~powershell
& $exe -m $model --gpu-backend cuda --layer 0 --layers 1 --tokens 400 --topk 8 --cpu-ratio 0.25 --threads 8 --warmup 2 --runs 5 2>&1 | Tee-Object -FilePath pc-ffn-only.log
~~~

Check PHONE_LOCAL_LAYER for actual CPU ratio and PHONE_LOCAL_SUM for the
GPU-only vs mixed speedup. Both modes use real quantized GGUF expert weights.
Deterministic activations, IDs, and mixtures are synthetic in this mode.

## Next: real Attention/Router + partitioned FFN across two layers

~~~powershell
& $exe -m $model --gpu-backend cuda --full-layer --layer 0 --layers 2 --tokens 400 --topk 8 --cpu-ratio 0.25 --threads 8 --warmup 2 --runs 5 2>&1 | Tee-Object -FilePath pc-full-layer.log
Get-Content pc-full-layer.log | Select-String '\[PHONE_FULL_(CONFIG|LAYER|STAGE_SUM|LAYER_CHECK|CHECK|SUM)\]'
~~~

Both modes execute Attention and real Router on CUDA, then:
- GPU_ONLY: all quantized MoE FFN weights on CUDA.
- CPU_GPU: aligned complementary FFN shards on CUDA and CPU concurrently;
  partial results are joined before passing to the next layer.

Read stage_ms, gpu_ffn_ms, cpu_ffn_ms, join_ms and total_ms in
PHONE_FULL_LAYER. The mixed FFN critical path is approximately
max(gpu_ffn_ms, cpu_ffn_ms), not their sum. CPU->GPU transfer and the
host-mediated join are included by the current benchmark.

## Ratio and thread sweeps

Try CPU ratios 0.10, 0.25, 0.45 and CPU thread counts 4, 8, and the
number of PC physical cores. Run the FFN-only test first, then the more
expensive full-layer test only for promising candidates.

Check PHONE_LAYER_LAYOUT actual_cpu_ratio every time: Q4_K quantization
alignment can make several requested ratios map to the same effective
split (the phone tests mapped 0.25 to about 1/3).

## How to interpret results

The desktop CUDA GPU can substantially outpace a CPU shard. If the CPU
is slow, the split loses, and the global planner should keep FFN on CUDA.
Unlike phone shared DRAM, transfers between CPU RAM and GPU VRAM cross
a discrete-device boundary, often PCIe.

These two-layer synthetic-activation tests are NOT full-model llama.cpp
tok/s, and do not include tokenizer, KV cache, Flash Attention, sampling,
PC-to-phone RPC or cross-layer pipeline scheduling. GPU-only also uses a
host-side residual join in this probe, so the absolute time differs from
production end-to-end CUDA inference.

The full-layer run must pass PHONE_FULL_CHECK and PHONE_FULL_LAYER_CHECK
before using its speedup figure as meaningful. CUDA/Windows execution
has not yet been compiled or benchmarked on the user's PC.
