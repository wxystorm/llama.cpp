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
resident. Android's page cache may keep previously touched pages. The
full model must fit in file storage and virtual address space, while
the simultaneous live working set must fit in physical memory.

After each completed OpenCL split the scheduler synchronizes the device
and calls a model-owned reclaim hook for the split's CPU mmap weight
sources. On Linux/Android it invokes `madvise(MADV_DONTNEED)` and
`posix_fadvise(POSIX_FADV_DONTNEED)` on aligned interior pages. This
makes those CPU mappings re-fault on reuse and *requests* release of
clean page-cache pages. Both APIs are advisory: they cannot guarantee
physical page eviction or a hard global RAM limit. The GGUF descriptor
is retained until the model is destroyed.

The scheduler measures process-attributed `/proc/self/io read_bytes`
for each graph, plus major/minor faults via `getrusage`, to distinguish
hot-cache from storage-pressure cases. It also logs the CPU->OpenCL
transfer submission wall time, post-split advice time, and graph wall
time. **None of these are exact per-weight UFS read latency**; that
requires an explicit I/O pipeline or kernel tracing.

This initial mode is synchronous, single-request, non-overlapped, and
optimized for a **dense-model baseline**. It does not implement GPU
double banking, NPU, MoE expert streaming, or persistent GPU layer
caches. Asynchronous prefetch is deferred.

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
GPU_BUDGET_MIB=1536 RSS_LIMIT_MIB=7168 \
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

## Memory budgets

The runner provides two independent configurable guards:

- `GPU_BUDGET_MIB` (default `1536`): before a GGML graph allocation, a
  no-allocation GGML reserve pass measures its aggregate OpenCL compute
  and transient staging buffer requirement. If the estimate exceeds the
  budget, the graph reservation is rejected **before allocating those
  buffers**. This is a ceiling on the GGML scheduler's OpenCL compute
  buffers only, **not** a hard device-memory cap: OpenCL driver scratch,
  kernel allocations, and other consumers are not included.
- `RSS_LIMIT_MIB` (default `7168`): after each OpenCL split and its
  cache-reclamation attempt, the runner checks the process's current
  Linux `VmRSS` and aborts if it exceeds the budget. This is a
  *sampled guard* and may not prevent transient OOM or Android LMK
  before the check. Global file-cache RAM not mapped into process RSS
  is also outside this limit.

Both values should be tuned using device `MemAvailable`, actual OpenCL
reservations and measured peak RSS. The defaults are only initial
experiment settings, not universal safe values.

A failed allocation can also result from the Adreno maximum *single*
OpenCL buffer allocation, even if the overall GGML budget is respected.
There is not yet a tensor-slicing fallback for that failure.

## Metrics and interpretation

`run.log` contains standard llama.cpp prompt evaluation and token
generation timings. `metrics.log` filters:
- `[PHONE_STREAM]` mode and OpenCL split statistics;
- `weight_copies` and `staged_weight_mib`: **requested CPU -> GPU
  staging**, potentially repeated across tokens; **not physical UFS
  bytes**;
- `advised_weight_mib`: aligned file-mapped pages for which
  `posix_fadvise` completed; **not confirmed cache bytes evicted**;
- `process_read_bytes`: per-graph delta from `/proc/self/io`, or
  `UINT64_MAX` when unavailable; Linux-attributed storage reads, not
  UFS controller byte counters;
- `major_faults`, `minor_faults`, `peak_sampled_rss_mib`,
  `copy_submit_ms`, `advice_ms`, `graph_wall_ms`;
- set `TRACE_SPLITS=1` in the runner to emit `[PHONE_STREAM_SPLIT]`
  with per-OpenCL-split wall time and its attributed `read_bytes` delta.
  These values include OpenCL transfer/compute/synchronization and
  filesystem-cache advice, not isolated UFS service time;
- `[PHONE_STREAM_BUDGET]`: graph allocation estimate or RSS threshold
  rejection;
- standard `prompt eval time` and `eval time` / tokens-per-second.

Confirm `opencl_splits > 0` and `weight_copies > 0` before claiming
Adreno computation. Repeating the run without clearing Android's page
cache is a **warm-cache** run, not a cold-UFS measurement.

## Known limits and validation

- **Not yet built or run on an Android/Adreno device.** Confirm numerical
  output against CPU-only execution on a small model before benchmarking.
- Some unsupported OpenCL operators will stay on CPU, potentially
  increasing activation transfers.
- The graph's transient OpenCL allocation estimate is checked against
  the configured budget, but this does not cap OpenCL driver's hidden
  allocations or the maximum single `clCreateBuffer` size.
- The mmap path may still trigger filesystem readahead, and advisory
  page-cache reclamation may be delayed or refused. Android OOM/LMK
  remains possible on 24B/32B.
- For reliable comparisons record phone available memory, peak RSS,
  OpenCL buffer reservations, thermal state, model and prompt token count.
- Avoid `--mlock`, `--no-mmap`, `--direct-io` and `--repack` in this
  experimental mode.
- Use `-fit off` to prevent automatic settings from attempting to
  model full persistent VRAM residency.
