# Phone-only on-demand GGUF execution: CPU / Adreno / Hexagon (experimental)

Use the same phone-local GGUF model and prompt with one of three **mutually
exclusive** `llama-cli` options. All modes load weights on demand from a
file-backed CPU `mmap`; none requires a PC or RPC during inference.

| Option | Compute | Weight materialization | CPU fallback |
| --- | --- | --- | --- |
| `-cpu` | CPU | File-backed mmap pages, no GPU allocation | Not applicable |
| `-opencl` | Adreno OpenCL (supported block GEMMs) | CPU mmap -> temporary OpenCL split-input buffers | Unsupported operators |
| `-npu` | Qualcomm Hexagon HTP0 (supported block GEMMs) | CPU mmap -> temporary HTP0 shared buffers, with backend repack | Unsupported operators |

**Not all transformer operations are offloaded.** For OpenCL and HTP,
unsupported operators, output/embedding and KV may execute on CPU. The
experimental scheduler refuses a graph that has Transformer matmuls but
executes *zero* of them on the requested device, and logs the actual
`accelerated_matmuls` count. For NPU quant types unsupported by the
backend, loading is rejected outright rather than quietly running all
quantized layers on CPU.

## Basic commands

Build `llama-cli` from branch `feat/phone-only-opencl-streaming-20261114`.
Then choose one mode:

```bash
./llama-cli -cpu -m /path/to/model.gguf -f /path/to/prompt.txt -c 2048 -b 512 -ub 128 -n 1
./llama-cli -opencl -m /path/to/model.gguf -f /path/to/prompt.txt -c 2048 -b 512 -ub 128 -n 1
./llama-cli -npu -m /path/to/supported-q4_0.gguf -f /path/to/prompt.txt -c 2048 -b 512 -ub 128 -n 1
```

Each option sets `-ngl 0`, `-fit off`, `--mmap`, `--no-repack`,
`--no-host`, `--no-warmup`, CPU KV, and a single selected compute
backend automatically. `-ngl 0` means **weights are not resident** on
the accelerator; this branch's scheduler selects the supported matrix
multiplications for on-demand execution.

The benchmark runner also works from the checkout on the phone:

```bash
LLAMA_BIN=./build/bin/llama-cli \
  DEVICE_BUDGET_MIB=768 RSS_LIMIT_MIB=7168 TRACE_SPLITS=1 \
  bash tools/phone-only-streaming/run_phone_stream.sh \
  -npu /path/to/supported-q4_0.gguf prompt.txt ./out-htp
```

Use `-cpu` or `-opencl` in place of `-npu`. The runner uses a default
device compute reservation budget of 1536 MiB for OpenCL, 768 MiB for
HTP0; the CPU mode has no accelerator reservation. Both modes use
`RSS_LIMIT_MIB=7168` as a *sampled process RSS guard*, not a hard global
RAM or GPU driver limit. Customize limits for your device.

## Model support of the currently based-on-master Hexagon backend

The Hexagon code in this branch supports GEMM weights in
`F16`, `F32`, `Q4_0`, `Q4_1`, `Q8_0`, `IQ4_NL`, and `MXFP4`,
subject to per-operator shape and HTP resource limits. **It does not
support the `Q4_K` or `Q6_K` GEMM weights in your usual Q4_K_M/Q6_K
models.** Running `-npu` on those GGUFs fails with
`[PHONE_STREAM_NPU]` before the large model mappings are materialized;
this is intentional and avoids falsely reporting CPU fallback as HTP.

There is a separate experimental HTP implementation in
`feat/llama-mistral32-region-htp-20261112` containing Q4_K/Q6_K
kernels, but they have **not** been ported into this master-derived
phone-only branch. Use the supported GGUF formats for initial NPU smoke
tests. Running your 24B Q6_K or 32B Q4_K_M on HTP requires porting
those Hexagon host+DSP kernels (and retesting correct repacking).

For `-npu` build with `-DGGML_HEXAGON=ON`, the Qualcomm Hexagon SDK,
HTP DSP skeletons and required FastRPC deployment. For `-opencl` build
with `-DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON`.
For `-cpu` neither accelerator backend is needed.

## Scheduling / page cache

- CPU mode computes directly on the mapped GGUF tensors. Once a CPU
  graph split finishes, the model requests eviction of its consumed
  mapped weights.
- OpenCL and HTP mode stage selected weights when a matmul is reached,
  synchronize the device *before* advising eviction of the CPU source
  pages, and reuse GGML graph allocator staging space on later splits.
  The HTP path sets `GGML_HEXAGON_HOSTBUF=0` and `NDEV=1` so HTP
  tensor setters perform the backend's required quantized packing.
- Eviction uses `madvise(MADV_DONTNEED)` and
  `posix_fadvise(POSIX_FADV_DONTNEED)`. These are **best-effort hints**.
  The kernel may keep the clean UFS file pages cached. Large GGUF
  weights must still fit in virtual address space and storage; peak
  concurrent memory use must fit physical RAM.
- This version does not implement layer-by-layer reordering of all
  prompt chunks, asynchronous prefetch or multi-session HTP memory
  packing; it is a simple synchronous out-of-core baseline.

Metrics in `[PHONE_STREAM]` include `backend_splits`,
`accelerated_matmuls`, `total_blk_matmuls`, `weight_copies`,
`staged_weight_mib`, `advised_weight_mib`, `process_read_bytes`,
major/minor faults, and sampled RSS. `TRACE_SPLITS=1` adds
`[PHONE_STREAM_SPLIT]` entries. Process `read_bytes` is **attributed
Linux disk reads**, not a standalone UFS controller or per-weight disk
latency measurement. Use `-n 1` first, then `-n 32` for Decode.

## Validation status

Implemented on the GitHub branch and checked at source level; **not yet
compiled on an Android Adreno/Hexagon device**, nor validated numerically.
Before using these as benchmark data: compile, run the same prompt with
CPU as a reference, check nonzero `accelerated_matmuls` for the selected
accelerator, confirm no unexpected OOM, and compare the generated output.
