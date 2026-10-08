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

## Diagnosing Adreno SIGSEGV while loading kernels

The OpenCL backend intentionally delays `load_cl_kernels()` until the **first GPU
buffer allocation**. A crash during `ggml_opencl: loading OpenCL kernels...`
is therefore an initialization failure; the benchmark may not yet have run
the FFN graph.

Build the updated branch and isolate kernel loading without opening the GGUF:

```bash
git pull --ff-only origin phone-local-tensor-bench
cmake --build build-phone-local --target llama-phone-local-tp-bench -j4

set -o pipefail
GGML_OPENCL_BUILD_TRACE=1 LD_LIBRARY_PATH=/vendor/lib64 \
  ./build-phone-local/bin/llama-phone-local-tp-bench --probe-opencl \
  2>&1 | tee phone-local-opencl-init.log
```

The `[PHONE_LOCAL_BOOT]` lines use stderr (so they are not buffered by the
`tee` stdout pipe). `[OPENCL_BUILD_TRACE]` prints the source hint, a
monotonic kernel compile ID, and the before/after of each `clBuildProgram`.
If it crashes after `phase=clBuildProgram_begin` but before the matching
`phase=clBuildProgram_end`, the vendor OpenCL compiler is the leading
suspect. Capture the final 30-50 lines from the combined log.

If the probe passes, rerun the regular two-layer benchmark using
`2>&1 | tee phone-local-25pct.log`. The last `[PHONE_LOCAL_BOOT]` marker
will distinguish GGUF loading, graph preparation, buffer allocation, and
weight loading.

Diagnostic workaround, **not comparable with the optimized Adreno baseline**:
rebuild with `-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF` and retry the probe.
If this changes the outcome, an Adreno-specific OpenCL kernel or compilation
path may be involved. Restore the option to ON for performance measurements.

## Quantized MoE OpenCL weight initialization

OpenCL `GGML_OPENCL_SOA_Q` converts the complete quantized tensor to its
device-side layout during `ggml_backend_tensor_set`. It is **not** safe to
upload one expert at a time with nonzero tensor offsets: the OpenCL conversion
can read `ggml_nbytes(tensor)` from the host pointer even if the requested
upload is smaller. This previously caused a SIGSEGV after
`[PHONE_LOCAL_BOOT] phase=weights_load_begin`.

The benchmark now builds each shard tensor in a contiguous host AoS array
for **all experts**, then uploads it with a single
`ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor))` call.
New `[PHONE_LOCAL_WEIGHT]` markers show assemble/upload boundaries and byte
counts. This increases temporary host memory but does not increase the number
of resident GPU layer weights.


## Continuous Qwen3-MoE layers: Attention + Router + CPU/GPU FFN

Pass --full-layer to select the new opt-in test; without it the original
FFN-only microbenchmark remains unchanged.

Each layer now executes real GGUF-weighted Attention on Adreno OpenCL:
attention RMSNorm; Q/K/V projection; Q/K RMSNorm and NeoX RoPE;
400-token causal, non-flash GQA Attention; output projection and residual;
FFN RMSNorm; real GGUF Router softmax/Top-K and renormalized expert weights.
Both comparison modes use exactly the same GPU Attention and Router.
GPU-only executes the whole quantized MoE FFN on the GPU.
CPU+GPU uses quant-block-aligned complementary FFN intermediate shards
in parallel, then adds their results with the Attention residual.
The resulting hidden states are passed to the NEXT layer.

Inputs remain fixed deterministic synthetic hidden states, not tokenizer
outputs. This is a continuous, prefill-shaped hidden-state benchmark, NOT
the complete model: no tokenizer/embedding, logits head, persistent KV
cache, sampling, RPC, pipeline or chunking. The causal attention graph is
non-flash, and the conservative GPU-to-FFN staging and CPU host joins
are included in both modes' timings. Full-model prefill performance
cannot be inferred directly from these measurements.

First, smoke-test one layer and one iteration in Termux:

    git pull --ff-only origin phone-local-tensor-bench
    cmake --build build-phone-local --target llama-phone-local-tp-bench -j4
    LD_LIBRARY_PATH=/vendor/lib64 ./build-phone-local/bin/llama-phone-local-tp-bench \
      -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
      --full-layer --layer 0 --layers 1 --tokens 400 --topk 8 \
      --cpu-ratio 0.25 --threads 8 --warmup 0 --runs 1 \
      2>&1 | tee phone-full-smoke.log

Then measure a contiguous two-layer pass (weights loaded outside timing):

    LD_LIBRARY_PATH=/vendor/lib64 ./build-phone-local/bin/llama-phone-local-tp-bench \
      -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
      --full-layer --layer 0 --layers 2 --tokens 400 --topk 8 \
      --cpu-ratio 0.25 --threads 8 --warmup 2 --runs 5 \
      2>&1 | tee phone-full-layer-2.log

Diagnostics:
- [PHONE_LAYER_BOOT] GPU Attention + Router construction/weight load.
- [PHONE_LAYER_LAYOUT] layer dimensions and quant-aligned FFN division.
- [PHONE_FULL_LAYER] per-layer GPU Attention+Router, stage, GPU FFN,
  CPU FFN, Join and total.
- [PHONE_FULL_RUN] wall-clock time across the entire requested layer range.
- [PHONE_FULL_CHECK] final hidden-state GPU-only versus split relative L2.
- [PHONE_FULL_SUM] median complete-layer-range latency and speedup.

A failed final-state relative-L2 comparison (>= 0.03) returns exit code 2.
Changes in Top-K decisions can amplify errors over several layers.
Both benchmark modes preload weights for all requested layers, and their
GPU allocations are released between GPU-only and CPU+GPU phases.
Start with 1-2 layers to keep phone peak memory usage manageable.
Mode order is currently GPU-only then mixed; later measurements should
alternate or randomize order to control DVFS and thermal drift.


Optional step-by-step debug for the first one-layer smoke test:

    PHONE_FULL_TRACE=1 LD_LIBRARY_PATH=/vendor/lib64 \
      ./build-phone-local/bin/llama-phone-local-tp-bench \
      -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
      --full-layer --layer 0 --layers 1 --tokens 400 \
      --topk 8 --cpu-ratio 0.25 --threads 8 --warmup 0 --runs 1 \
      2>&1 | tee phone-full-step.log

Trace phases: attention_router_begin/ok, ffn_stage_begin/ok,
ffn_parallel_begin/ok, join_begin/ok. No effect when PHONE_FULL_TRACE
is unset. This allows isolating an OpenCL/CPU failure without verbose
kernel-level logging.


## Stage and correctness breakdown

The --full-layer mode now times the six Router-to-FFN tensor copies
and two explicit backend synchronizations separately.

- [PHONE_FULL_STAGE_BYTES]: bytes for ffn_norm, expert IDs and expert mixture weights.
- [PHONE_FULL_STAGE]: per-repetition gpu_norm_ms, gpu_ids_ms, gpu_mix_ms,
  cpu_norm_ms, cpu_ids_ms, cpu_mix_ms, cpu_sync_ms, gpu_sync_ms,
  other_ms, accounted_ms and total_ms.
- [PHONE_FULL_STAGE_SUM]: per-layer medians for every copy, sync and total Stage.
- [PHONE_FULL_DIAG]: one additional, untimed pass per mode.
- [PHONE_FULL_LAYER_CHECK]: per-layer relative L2, maximum absolute error,
  changed_router_tokens, changed_router_slots and status.

Each copy duration is the wall-clock duration of ggml_backend_tensor_copy,
NOT necessarily direct DMA time. The backend can implicitly wait for device
work or fall back to host transfer. Individual medians therefore need not
sum to the median Stage total.

GPU-only and CPU+GPU each get one separate UNTIMED diagnostic pass. It
collects the hidden state after every layer and GPU Router Top-K IDs.
Any per-layer relative L2 >= 0.03 returns CHECK and exit code 2.
Changed expert selection is reported for diagnosis, not considered a
standalone failure.

Also, [PHONE_FULL_RUN] wall time is recorded BEFORE per-layer log printing;
terminal formatting and printing no longer contribute to the wall time.
Be aware of this accounting change when comparing against older logs.

### Run on Termux

    git pull --ff-only origin phone-local-tensor-bench
    cmake --build build-phone-local --target llama-phone-local-tp-bench -j4
    set -o pipefail
    LD_LIBRARY_PATH=/vendor/lib64 ./build-phone-local/bin/llama-phone-local-tp-bench \
      -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf \
      --full-layer --layer 0 --layers 2 --tokens 400 --topk 8 \
      --cpu-ratio 0.25 --threads 8 --warmup 2 --runs 5 \
      2>&1 | tee phone-full-stage-breakdown.log

    grep -E '\[PHONE_FULL_(STAGE_SUM|LAYER_CHECK|CHECK|SUM)\]' \
      phone-full-stage-breakdown.log

If Stage has spikes, compare CPU ffn_norm copies (large activation tensors)
against small ids/mix copies, then the two explicit synchronization calls.
Do not prematurely conclude that time attributed to a copy is only memcpy.
