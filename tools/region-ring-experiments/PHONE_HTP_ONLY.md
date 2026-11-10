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

For the initial **16 Phone layers** layout `G8,P3,G8,P3,G8,P3,G8,P3,G8,P2,G8,P2`,
the Phone regions are `1,3,5,7,9,11`.

```sh
cd ~/phone-381
export LD_LIBRARY_PATH="$PWD/lib:$PWD/bin:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$PWD/lib"

export GGML_HEXAGON_NDEV=4
export GGML_HEXAGON_HOSTBUF=1
export GGML_RPC_PHONE_HTP_ONLY=1
export GGML_RPC_PHONE_NPU_SERIAL_REGIONS=1,3,5,7,9,11

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
    -Layout "G8,P3,G8,P3,G8,P3,G8,P3,G8,P2,G8,P2" `
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
| P16 | `G8,P3,G8,P3,G8,P3,G8,P3,G8,P2,G8,P2` | `1,3,5,7,9,11` | 48 / 16 |
| P24 | `G8,P6,G8,P6,G8,P6,G8,P6,G8` | `1,3,5,7` | 40 / 24 |
| P32 | `G8,P8,G8,P8,G8,P8,G8,P8` | `1,3,5,7` | 32 / 32 |

Start with P16 and a short Prompt. **15 GiB MemAvailable is not a
guarantee** that 24 or 32 Phone layers fit: HTP repack, KV, per-layer
scratch, FastRPC memory mappings, address-space constraints, and host
memory all count. Track `/proc/meminfo`, per-session
`PHONE_OWNED_SCRATCH` telemetry, `PHONE_QWEN3_LOCAL HTP_PRECOMPILE`,
and model load logs. Reject any overbudget plan rather than swapping or
assuming that HTP0..HTP3 imply additive physical RAM.

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
