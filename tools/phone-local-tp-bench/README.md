# Phone-local CPU + OpenCL MoE FFN benchmark

Independent benchmark on branch `phone-local-tensor-bench`, based on `planner-moe-stage`. It does not change the RPC protocol, planner, model graph, or existing PC/Phone paths.

## Scope

- Loads only 1-2 selected **MoE expert FFN layers** from one GGUF file (default 2 layers, starting at layer 0).
- Deterministic **synthetic activation for 400 tokens** and deterministic expert IDs/weights. This is not tokenizer-based end-to-end text generation, and excludes Attention, Norm, Router, KV cache and sampling.
- Uses real GGUF quantized expert matrices (gate, up, down); FFN width is split along aligned quantization blocks exactly as the existing tensor split design.
- Performs GPU-only baseline, frees those weights, loads complementary GPU/CPU weight shards, executes both concurrently, then joins the outputs on the GPU.
- Does **not** include disk loading or graph building in timed iterations. Includes GPU->CPU input staging and host-mediated output join in mixed-mode wall time.
- No multi-layer pipeline, inter-PC networking, RPC, chunking, or auto-planning.

## Build on the phone (Termux)

On this branch, in the repository root:

```bash
cmake -S . -B build-phone-local -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DGGML_RPC=OFF -DGGML_OPENMP=OFF
cmake --build build-phone-local --target llama-phone-local-tp-bench -j4
```

Check the GPU and CPU backend names in output. The benchmark automatically picks the first device whose name contains `OpenCL`, plus the CPU device.

Example using a local **single-file** Qwen3-30B-A3B Q4_K_M GGUF:

```bash
./build-phone-local/bin/llama-phone-local-tp-bench \
  -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  --layer 0 --layers 2 --tokens 400 --topk 8 \
  --threads 4 --cpu-ratio 0.25 --warmup 2 --runs 5 \
  | tee phone-local-25pct.log
```

Run the same configuration at CPU ratios `0.10`, `0.20`, `0.30`, `0.40`; ratio `0` is a GPU-only baseline. The program reports **actual** block-aligned ratios rather than claiming the requested ratio is exact.

Outputs:
- `[PHONE_LOCAL_CONFIG]`: device and workload configuration
- `[PHONE_LOCAL_LAYER]`: dimensions, quantization-aligned CPU/GPU split
- `[PHONE_LOCAL_TP] mode=GPU_ONLY`: per-run GPU-only latency
- `[PHONE_LOCAL_TP] mode=CPU_GPU`: per-run input staging, simultaneous compute, join and total
- `[PHONE_LOCAL_CHECK]`: relative L2 error compared with GPU-only output
- `[PHONE_LOCAL_SUM]`: per-layer medians and mixed-mode speedup

**Important limitations:** The `[PHONE_LOCAL_SUM]` timing represents a small set of independently benchmarked FFN layers, *not* actual 400-token prefill throughput. Each layer gets its own deterministic input, not the previous layer's output. Output check uses a diagnostic relative-L2 threshold and needs investigation if marked CHECK. It is not proof of token-level correctness. CPU/GPU share physical DRAM but the initial join uses tensor readback and an upload, not zero-copy. Quantization support and performance depend on the phone's OpenCL backend. This code has not been compiled or run on the target Android/Adreno phone here.
