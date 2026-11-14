# Phone-only CPU mmap + Adreno OpenCL weight streaming (experimental)

This branch adds a **phone-local** baseline for large GGUF models. No PC/RPC
participates in computation. It is independent of Region Ring, HTP, and the
cycle planner.

## How it works

Set `GGML_OPENCL_STREAM_WEIGHTS=1` and select the `GPUOpenCL` backend.
The model loader leaves **all model weights** in CPU file-backed mmap,
without upfront `MAP_POPULATE`/WILLNEED prefetch. It does not allocate a
persistent GPU model buffer.

The GGML scheduler selects supported `GGML_OP_MUL_MAT` operations with
`blk.*` weight names for OpenCL. When the graph reaches such an operation,
it copies the required weight from CPU mmap into a temporary OpenCL graph
split-input buffer, executes the operation, then reuses that buffer's
allocation for later splits. Other operators and output/embedding weights
use the normal CPU/backend assignment. KV cache is kept on CPU when the
runner uses `--no-kv-offload`.

**This is on-demand staging, not a disk I/O guarantee.** mmap maps the
complete GGUF into virtual address space; only touched pages must become
resident. Android's page cache may keep many previously touched pages,
and the kernel may reclaim them under memory pressure. The full model
still has to fit in file storage and virtual address space, and Android
might kill the process under severe RAM pressure.

This initial mode is synchronous, single-request, non-overlapped, and
optimized for a **dense-model baseline**. It does not implement GPU
double banking, NPU, automatic memory budgeting, MoE expert streaming,
or persistent GPU layer caches. Asynchronous prefetch is deferred.

## Build

Build llama.cpp with `GGML_OPENCL=ON` and a working Adreno OpenCL runtime.
The binary must be compiled from this branch. A standard upstream
binary does **not** recognize this streaming mode.

## Run on phone

First confirm the OpenCL device is available:

```bash
./build/bin/llama-cli --list-devices
```

Use the supplied script with a locally stored GGUF and prompt file:

```bash
LLAMA_BIN=./build/bin/llama-cli \
N_CTX=2048 N_BATCH=512 N_UBATCH=128 N_PREDICT=1 \
bash tools/phone-only-streaming/run_phone_opencl_stream.sh \
  /data/local/tmp/models/Qwen3-14B-Q4_K_M.gguf \
  /data/local/tmp/prompt.txt \
  ./phone-14b-prefill
```

For decode timing repeat with `N_PREDICT=32` and a **different output
directory**. Use the same settings with a larger GGUF for 24B/32B.

The important flags are `-dev GPUOpenCL -fit off -ngl 0 --mmap
--no-repack --no-host --no-warmup --no-kv-offload`.
`-ngl 0` here does **not** mean CPU-only compute: the modified
GGML scheduler explicitly places supported matrix multiplication
operations on OpenCL while their source weights stay in CPU mmap.

## Metrics and interpretation

`run.log` contains standard llama.cpp prompt evaluation and token
generation timings. `metrics.log` filters:
- `[PHONE_STREAM]` mode and OpenCL split statistics;
- `weight_copies` and `staged_weight_mib`: **requested CPU -> GPU
  staging**, potentially repeated across tokens; **not physical UFS
  bytes**;
- standard `prompt eval time` and `eval time` / tokens-per-second.

Confirm `opencl_splits > 0` and `weight_copies > 0` before claiming
Adreno computation. Repeating the run without clearing Android's page
cache is a **warm-cache** run, not a cold-UFS measurement.

## Known limits and validation

- **Not yet built or run on an Android/Adreno device.** Confirm numerical
  output against CPU-only execution on a small model before benchmarking.
- Some unsupported OpenCL operators will stay on CPU, potentially
  increasing activation transfers.
- The scheduler's graph allocator generally reuses weight staging
  allocations based on lifetimes, but the **actual GPU peak** must be
  confirmed on device; it is not hard-capped by this patch.
- The current mmap path may still trigger filesystem readahead, and
  page-cache pressure may cause device OOM/LMK on 24B/32B.
- For reliable comparisons record phone available memory, peak RSS,
  OpenCL buffer reservations, thermal state, model and prompt token count.
- Avoid `--mlock`, `--no-mmap`, `--direct-io` and `--repack` in this
  experimental mode.
- Use `-fit off` to prevent automatic settings from attempting to
  model full persistent VRAM residency.
