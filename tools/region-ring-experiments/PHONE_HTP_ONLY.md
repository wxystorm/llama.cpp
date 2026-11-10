# Experimental Qwen3-32B HTP-only Phone prefill (Plan v4)

Branch: `feat/phone-htp-only-20261110`, fork of
`feat/phone-plan-overlap-multi-htp-20261109`.

This is an **opt-in, unverified runtime migration**, not a validated speedup.
Both PC and Phone must be built from the same branch. The known-good
OpenCL/HTP Graphless path is unaffected when the new flags are absent.

## Scope and boundaries

- Phone **does not initialize OpenCL**: start the RPC server with `-d CPU`.
  This backend transports RPC commands and is NOT the Transformer executor.
- Four independent Hexagon FastRPC sessions hold immutable weights, F16 KV,
  and fixed per-Region/Chunk HTP graphs. Each Phone Region may span multiple
  consecutive layers, distributed round-robin across sessions; compute remains
  serial within the Phone worker.
- The PC's Phone-only Transformer weights remain as CPU GGUF/mmap references
  solely for model tensor metadata. They are **not allocated/uploaded to the
  Phone RPC backend**. Phone loads required weights from the *same GGUF*
  stored locally; HTP owns the real execution weights.
- No PC-built Phone graph, no canonical OpenCL KV binding, no HTP-to-OpenCL
  KV copy on RUN end. Input and output use the existing verified host PUSH
  and P2C arena.
- **Fresh, single-sequence prefill only**: position base 0, identity F16 KV
  slots, flash attention, Qwen3, `-n 1`. Decode, KV reuse, multi-sequence,
  and warmup are **not supported** by this experiment. The runner appends
  `--no-warmup`. Do not use it for chatbot serving yet.
- `GGML_HEXAGON_NDEV=4` creates four virtual mapping domains, NOT four
  independent pools of physical RAM. All Session allocations consume the
  same phone RAM; mapping/address-space limits still apply.

## Phone: Termux startup (Qwen3-32B)

For the initial **16 Phone layers** layout `G8,P4,G8,P3,G8,P3,G8,P3,G8,P3,G8`,
the Phone regions are `1,3,5,7,9`.

```sh
cd ~/phone-381
export LD_LIBRARY_PATH="$PWD/lib:$PWD/bin:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$PWD/lib"

export GGML_HEXAGON_NDEV=4
export GGML_HEXAGON_HOSTBUF=1
export GGML_RPC_PHONE_HTP_ONLY=1
export GGML_RPC_PHONE_NPU_SERIAL_REGIONS=1,3,5,7,9

export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
export GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_INPUT=1
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=1
export GGML_RPC_PHONE_QWEN3_LOCAL_GRAPHLESS=1
unset GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS
unset GGML_RPC_PHONE_QWEN3_SHARED_SCRATCH

./bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d CPU \
    -m "$HOME/models/Qwen3-32B-Q4_K_M.gguf" \
    2>&1 | tee phone-32b-htp-only.log
```

Confirm four `[PHONE_NPU_SESSION] event=INIT` records and
`[PHONE_NPU_SERIAL] event=BACKEND_INIT ... rpc_visible=CPU_TRANSPORT_HTP_ONLY`.
If `-d CPU` is unavailable in the binary's `--help`, rebuild the Phone
binary with CPU and Hexagon support. Do NOT substitute `GPUOpenCL` in
HTP-only mode.

## PC: first smoke

```powershell
.\tools\region-ring-experiments\run_sharegpt40_32b.ps1 `
    -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
    -Rpc "10.39.52.114:50052" `
    -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
    -Out ".\sharegpt40-32b-htp-only-p16" `
    -Layout "G8,P4,G8,P3,G8,P3,G8,P3,G8,P3,G8" `
    -HtpOnly `
    -Only "prompt_512_01" `
    -Limit 1
```

The PC wrapper explicitly enables
`LLAMA_REGION_RING_PHONE_HTP_ONLY=1` for the child process, leaves
mmap enabled for the PC/GGUF weights, uses `-n 1`, and disables warmup.

Do not reuse the old output directory: it contains legacy Plan v2/v3
results with incompatible Phone memory placement.

## Scaling candidates

Only change the Phone `GGML_RPC_PHONE_NPU_SERIAL_REGIONS` **and** PC
`-Layout` together, restarting the Phone RPC server between experiments.

| Candidate | PC layout | Phone regions | PC / Phone layers |
|---|---|---|---|
| P16 | `G8,P4,G8,P3,G8,P3,G8,P3,G8,P3,G8` | `1,3,5,7,9` | 48 / 16 |
| P24 | `G8,P6,G8,P6,G8,P6,G8,P6,G8` | `1,3,5,7` | 40 / 24 |
| P32 | `G6,P8,G7,P8,G6,P8,G7,P8,G6` | `1,3,5,7` | 32 / 32 |

The fixed-layout ShareGPT runner requires an odd number of Regions,
strictly alternating GPU/Phone, **starting and ending with G**, and a
64-layer sum. P16 has six 8-layer GPU Regions and five Phone Regions
(P4 + P3 + P3 + P3 + P3), so both GPU Banks peak at eight layers.
The GPU-last requirement is intentional; do not append a G10 Region just
to make an invalid GPU/Phone layout pass validation.

Start with P16 and a short Prompt. **15 GiB MemAvailable is not a
guarantee** that 24 or 32 Phone layers fit: HTP repack, KV, per-layer
scratch, FastRPC memory mappings, address-space constraints, and host
memory all count. Track `/proc/meminfo`, per-session
`PHONE_OWNED_SCRATCH` telemetry, `PHONE_QWEN3_LOCAL HTP_PRECOMPILE`,
and model load logs. Reject any overbudget plan rather than swapping or
assuming that HTP0..HTP3 imply additive physical RAM.

## Concurrent PC/Phone model weight loading (Patch 30)

**Enabled automatically with `-HtpOnly`.** When the PC model loader
has finalized the model-owned G/P layout, it asynchronously sends
the Phone Region indices and complete Transformer layer ranges to the
RPC server. The Phone immediately loads those layers from its local
GGUF into their correct HTP FastRPC sessions, while the PC **continues
loading its G-layer model weights and optionally prewarming the PC
mmap page cache**. There is no additional Phone layer-ID environment
variable to configure (`GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS` stays
unset). The existing Phone Region allowlist must still match the PC
layout, e.g. `GGML_RPC_PHONE_NPU_SERIAL_REGIONS=1,3,5,7,9` for P16.

The PC joins the preload worker **before `load_tensors()` returns**,
so all of this work is outside the timed Prefill. If early preload
fails, it reports FAIL and preserves existing on-demand loading during
Phone-owned Plan registration; it never reports a silent cache hit.
The Phone keeps these immutable weights resident across RUNs; runtime
KV, activations and Chunk Graphs remain per-request and are not reused.

Required: rebuild the PC and Phone from the same commit and restart
the Phone RPC server because the protocol patch changes from 29 to 30.
On PC, the runner supplies `LLAMA_REGION_RING_PHONE_EARLY_WEIGHT_RPC`
automatically using its `-Rpc` address. The preload uses a separate
RPC socket from the normal COMPUTE and PUSH channels.

PC model-load logs:

- `[REGION_RING_PHONE_EARLY_WEIGHTS] event=START ... execution=CONCURRENT`
- `[REGION_RING_PHONE_EARLY_WEIGHTS] event=REGION ... status=PASS`
- `[REGION_RING_PHONE_EARLY_WEIGHTS] event=DONE ... status=PASS`
- `[REGION_RING_PHONE_EARLY_WEIGHTS] event=JOIN ... wait_ms=... status=PASS`

Phone log per layer: `[PHONE_NPU_EARLY_WEIGHTS] event=LAYER ...`
including HTP session, `cache=HIT|MISS`, `read_ms`, and `repack_ms`.
The 5 Region summaries end in `event=DONE status=PASS`.

**P16 test (PC RAM page-cache warmup in parallel):**

```powershell
.\tools\region-ring-experiments\run_sharegpt40_32b.ps1 `
    -HtpOnly -PreloadGpuRam `
    -Layout "G8,P4,G8,P3,G8,P3,G8,P3,G8,P3,G8" `
    -Only "prompt_512_01" `
    -Out ".\32b-p16-parallel-weight-load" `
    -Limit 1
```

For a controlled A/B test against the previous on-demand mode, add
`-NoEarlyHtpWeightPreload` and use a **different** `-Out` directory.
Track model initialization duration separately from `prefill_ms`.
The expected reduction in Phone-owned Plan registration is a test
hypothesis, not a measured speedup yet.
## Optional: PC GPU weight RAM preloading (experimental)

To pre-read the PC's G-layer weights into OS RAM page cache **before**
timed Prefill, add `-PreloadGpuRam` to the PC 32B PowerShell command:

```powershell
.\tools\region-ring-experiments\run_sharegpt40_32b.ps1 `
    -HtpOnly -PreloadGpuRam `
    -Layout "G8,P4,G8,P3,G8,P3,G8,P3,G8,P3,G8" `
    -Only "prompt_512_01" `
    -Out ".\32b-htp-only-p16-ram-preload" `
    -Limit 1
```

This opts into `LLAMA_REGION_RING_PC_GPU_RAM_PRELOAD=1`, which touches
each 4 KiB mmap page belonging to PC CUDA Transformer weights during
model initialization. The Phone-owned 16 layers, embedding and LM Head
are excluded. No new permanent heap copy is allocated. File-backed
pages may still be evicted, so this is **best-effort** RAM residency.

The implementation checks reported available physical RAM and reserves
2 GiB for other workloads. It logs either
`[REGION_RING_PC_RAM_PRELOAD] event=DONE ... mode=MMAP_PAGECACHE`
or `event=SKIP reason=INSUFFICIENT_AVAILABLE_RAM`.

For A/B, use the **same** Layout/Prompt with `-PreloadGpuRam` omitted
and a different `-Out` directory, and compare `prefill_ms`,
`bank_mover_wall_ms`, `bank_h2d_ms`, and `bank_gpu_wait_ms`.
The preload `wall_ms` is separate startup cost. Alternate run order
to control for the Windows OS file cache being warmed by the first run.

GPU weight refill currently defaults to direct Host->CUDA upload;
the old two-slot Pinned staging can be restored independently using
`GGML_SCHED_WEIGHT_STAGE_PINNED=1` for A/B. Do not vary both switches
in the same comparison.
## Pass criteria

- `[PHONE_HTP_ONLY] event=REGISTER ... opencl_kv=SKIPPED status=READY`
  per Phone Region.
- `[PHONE_QWEN3_LOCAL] event=EXECUTE_ONLY ... status=PASS` for all
  registered Chunk tasks.
- `[PHONE_HTP_ONLY] event=RUN_END ... opencl_sync=SKIPPED status=PASS`.
- PC `metrics.jsonl` reports full Prefill and no RPC/OpenCL model-weight
  transfer. Compare against the known-good baseline output and Prefill,
  not just a PASS process exit code.

For the descriptor-level host-only test:

```sh
c++ -std=c++17 -O2 -Wall -Wextra \
    tools/region-ring-experiments/test_phone_htp_only_plan.cpp \
    -o /tmp/test_phone_htp_only_plan
/tmp/test_phone_htp_only_plan
```

**Known unimplemented portions:** general Decode on HTP and reclaiming
the PC-side virtual GGUF mappings entirely. PC still needs the original
model GGUF for metadata and PC CUDA Banks; mmap avoids eagerly
duplicating the full model into small host RAM. This experiment changes
Phone GPU residency, not PC's entire model-loading architecture.
