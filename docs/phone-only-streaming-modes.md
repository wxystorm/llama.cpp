# Phone-only on-demand GGUF execution: CPU / Adreno / Hexagon (experimental)

Use the same phone-local GGUF model and prompt with one of three **mutually
exclusive** `llama-cli` options. All modes load weights on demand from a
file-backed CPU `mmap`; none requires a PC or RPC during inference.

| Option | Compute | Weight materialization | CPU fallback |
| --- | --- | --- | --- |
| `-cpu` | CPU | File-backed mmap pages, no GPU allocation | Not applicable |
| `-opencl` | Adreno OpenCL (supported block GEMMs) | CPU mmap -> temporary OpenCL split-input buffers | Unsupported operators |
| `-npu` | Qualcomm Hexagon HTP0..HTP3 (supported block GEMMs) | CPU mmap -> per-session temporary HTP REPACK buffers | Unsupported operators |

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

## HTP Q4_K/Q6_K support (experimental)

The HTP Host and DSP paths have been selectively transplanted from
`feat/llama-mistral32-region-htp-20261112`: Q4_K and Q6_K K-quantized
block matmuls are now eligible for `-npu` and stage their tensors in
temporary **per-session HTP REPACK buffers** while the source GGUF stays on UFS/CPU mmap.
Q4_K uses Q4_1-compatible Tiled and Scale/Min layout; Q6_K uses dedicated
HVX Tiled/Flat and HMX dequantization kernels.

**Rebuild and deploy both the Android Host binary and Hexagon DSP skeleton.**
A Host-only rebuild does not update DSP kernels or its type table.
The DSP mmap-slot reclamation fix is included for repeated layer switches.
For wide Q6_K, HMX retries a non-pipelined solve; HVX Flat reduces
prefetch depth when necessary.

This port is source-integrated but **not yet compile-tested or device-tested**.
Check CPU/HTP parity and run a many-layer memory stress test before
recording throughput data. The existing Qwen3-32B-Q4_K_M and
Mistral-Small-24B-Q6_K GGUFs do not need re-quantization.


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
  prompt chunks or asynchronous prefetch; it uses sequential cross-session
  scheduling rather than concurrent four-way HTP execution.

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


## Four HTP FastRPC sessions (phone only)

The phone-only `-npu` mode now **defaults to four FastRPC sessions**.
It selects `HTP0,HTP1,HTP2,HTP3` into the scheduler, not just the
registry. Override with `GGML_HEXAGON_NDEV=1`, `2`, `3` or `4`
for a controlled comparison; any other count is rejected.

Each supported `blk.<layer>.*` matrix multiplication prefers session
`layer % n_sessions`, falling through to another selected session only
when that session does not support the operation. The ordinary GGML
scheduler handles activation transfers between sessions. Each session
owns its own HTP REPACK buffer and its own
`GGML_PHONE_STREAM_DEVICE_BUDGET_MIB` reservation guard (the runner
defaults to 768 MiB **per session**). The RSS guard is for the whole
process, not per session. These are FastRPC sessions sharing one
physical Hexagon subsystem, **not four independent NPUs**.

Use the installed paths on the Termux phone:

```bash
export LD_LIBRARY_PATH="$HOME/phone-stream-htp/lib:$PREFIX/lib"
export ADSP_LIBRARY_PATH="$HOME/phone-stream-htp/lib"

GGML_HEXAGON_NDEV=4 "$HOME/phone-stream-htp/bin/llama-cli" \
  -npu -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
  -f "$HOME/sharegpt40/prompt_512_01.txt" \
  -c 2048 -b 512 -ub 128 -n 1 -lv 3

# 4-session smoke test requires each session to execute a real matmul.
sh "$HOME/phone-stream-htp/smoke_htp4_sessions.sh" \
  "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
  "$HOME/sharegpt40/prompt_512_01.txt"

# The HTP subset of the full 14B/32B x ShareGPT50 sweep.
MODES=htp HTP_SESSIONS=4 LOG_ROOT="$HOME/log/htp4" \
  sh "$HOME/phone-stream-htp/run_sharegpt50_phone.sh" \
  "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
  "$HOME/models/Qwen3-32B-Q4_K_M.gguf" "$HOME/sharegpt40"
```

With `GGML_OPENCL_STREAM_TRACE_SPLITS=1` and `-lv 3`, inspect
`[PHONE_STREAM_HTP] device=HTP0/1/2/3 splits=...
accelerated_matmuls=...` and `[PHONE_STREAM_SPLIT] ... backend=HTP0/1/2/3`.
The test requires **nonzero matmuls on all four sessions**, not just
successful registration.

**Current limitations:** Rebuild and deploy the Android Host binary
and the matching Hexagon DSP skeleton. The phone must be able to open
HTP0 before other sessions can work. A previous error
`failed to open session 0 : error 0x80000406` is a separate
FastRPC/DSP deployment problem; changing NDEV alone cannot repair it.
The new scheduler has not yet passed Android compilation or on-device
execution. It implements **sequential multi-session distribution**,
not concurrent four-way parallel execution or verified numerical parity.
Use a separate `LOG_ROOT` for the one-session baseline because
the sweep runner refuses to mix changed configurations.
