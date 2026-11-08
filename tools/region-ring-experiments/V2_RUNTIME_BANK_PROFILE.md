# V2 production Phone Bank / N->Adreno runtime telemetry

## What is measured

The independent G/C/N/A Stage profiler already measures Graph compute,
backend Activation copies, and optionally a SAME-BANK HTP reload.
It cannot measure real double-bank scheduling overlap.

This patch instruments the EXISTING production ggml-rpc server's two
physical HTP repack Banks and local HTP-to-Adreno handoff. It uses
the existing PHONE_STAGE_PROFILE_PULL RPC, extended to Patch33.
No new TCP bandwidth benchmark or Android log copying is needed.

The production Bank Mover already loads initial A/B arenas, queues
rank k+2 after rank k drains, stages raw GGUF on Host, takes the HTP
mutex, retires old Graphs, repacks the other weights, precompiles the
new Graph and makes the Bank READY. Per-layer KV remains separate.

Patch33 records:
- Bank enqueue/dequeue, Host staging, HTP lock acquisition, critical
  repack+compile and completion in the same Phone monotonic clock
- Host-stage overlap with actual HTP and Adreno device-service intervals
- HTP Bank-critical-section overlap with Adreno, NEVER with HTP compute
- NPU->Adreno Host F32 ownership transfer, Adreno upload queue delay,
  actual destination upload cost and time to downstream device service

These are OBSERVED interval overlaps, NOT inferred wall-time speedups.
The Phone HTP mutex intentionally serializes Bank mutation and HTP
compute. Unavailable device windows and incomplete runs are visible,
not fabricated as a successful Profile.

## Build both RPC endpoints at Patch33

The ggml/include/ggml-rpc.h protocol patch is now **33**.
Build both Windows PC and Android Phone RPC from the same commit:

    # PC PowerShell
    git pull --ff-only
    cmake --build .\build-tensor-v2 --config Release -j 4

    # Android Docker in /workspace
    cmake --build build-phone-npu --target ggml-rpc-server -j 8

Deploy the new Phone RPC server plus its compatible runtime libraries
using the existing successful installation method. The Phone's HTP
DSP kernels need no recompilation merely for the RPC schema change.
Old Patch32 and new Patch33 RPC peers must NOT be mixed.

Enable on the PHONE:

    export GGML_RPC_REAL_STAGE_PROFILE=1
    export GGML_RPC_PHONE_CHUNK_TIMELINE=1

Retain the working Plan-controlled dual-Bank and local HTP/OpenCL
runtime settings, such as:
GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2,
GGML_RPC_PHONE_NPU_PLAN_CONTROLLED=1,
GGML_RPC_PHONE_NPU_MIXED_OPENCL=1,
GGML_RPC_PHONE_NPU_GPU_PIPELINE=1,
GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1.
Do not manually invent Phone NPU Region IDs.

On PC, keep LLAMA_REGION_RING_REAL_STAGE_PROFILE=1 and
LLAMA_REGION_RING_PHONE_PROFILE_RPC=1. The existing
run_gcna_profile.ps1 helper sets these automatically.

**Requirements for real samples**
- Real HTP Bank refill: at least three HTP Region ranks in one run
  (the first two are initially resident in A and B).
- Real NPU->Adreno handoff: a Phone-owned graphless NPU Stage with
  an adjacent planned Adreno successor. Normal return-to-PC HTP
  Regions will have no local handoff, correctly NOT_EXERCISED.

## New PC-side structured log events

The PC automatically receives, after Input Stream END:

    [PHONE_BANK_RUNTIME_PROFILE] source=RPC_PATCH33 run=...
      refills=... failed=... inflight=... unstaged=...
      overflow=... device_windows=... htp_windows=...
      adreno_windows=... queue_ms=...
      host_stage_wall_ms=... htp_lock_wait_ms=...
      htp_critical_wall_ms=... precompile_ms=...
      host_stage_htp_overlap_ms=...
      host_stage_adreno_overlap_ms=...
      htp_critical_adreno_overlap_ms=...
      status=PASS|INCOMPLETE|NOT_EXERCISED
      overlap_state=MEASURED|NO_DEVICE_SERVICE_WINDOWS

    [PHONE_LOCAL_HANDOFF_RUNTIME_PROFILE] source=RPC_PATCH33 run=...
      edges=... incomplete=... bytes=...
      host_move_ms=... ready_to_upload_ms=...
      upload_ms=... ready_to_device_ms=...
      host_copy_bytes=0 intermediate_p2c_bytes=0
      status=PASS|INCOMPLETE|NOT_EXERCISED

The existing PHONE_PROFILE_RESOURCES event still provides raw
GGUF staged bytes and HTP repack bytes/ms.

All timing fields above are actual, same-Phone-clock measurements:
- queue_ms: Bank Mover dequeue latency after job enqueue
- host_stage_wall_ms: raw Host staging, including GGUF read
- htp_lock_wait_ms: time waiting on the HTP backend mutex
- htp_critical_wall_ms: exclusive Bank mutation, Graph evict,
  repack, compile, READY
- host_stage_htp_overlap_ms: Host staging intersects another HTP
  Stage's service interval
- htp_critical_adreno_overlap_ms: HTP mutation intersects Adreno
  service (distinct physical compute device)
- ready_to_upload_ms: NPU output published until Adreno input upload
  begins, including its scheduling wait
- upload_ms: Adreno's actual upload/preparation interval
- ready_to_device_ms: NPU output published until the Adreno device
  service starts, including queue/upload.

Any pending, failed, overflowing or unstaged Bank refill makes a
full real-overlap calibration INCOMPLETE. No refill => NOT_EXERCISED.
Zero device windows => overlap is NOT verified. The
htp_windows/adreno_windows counters distinguish the two physical lanes;
HTP weight staging overlap can be qualified only if there is at least
one measured HTP service window.

## Import to a model-bound Profile, using only PC files

Optional automatic import after a compact GCNA probe:

    .\tools\region-ring-experiments\run_gcna_profile.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" -V2ProfileJson ".\profile-v2\gcna-expanded\profile.json"

Or merge an existing PC engine.stderr.log or a directory tree:

    python .\tools\region-ring-experiments\merge_v2_runtime_profile.py --profile ".\profile-v2\gcna-expanded\profile.json" --pc-log ".\path\to\pc-run-root" --out ".\profile-v2\gcna-expanded\profile-with-runtime.json"

The output includes per-run exact real Bank/handoff costs, overlap
fractions, source PC log hashes and explicit qualification flags.
It never rewrites the original independent profile.json.

The PC engine log does not prove the runtime model SHA256 matches the
independent Stage Profile, or that the full production transfer/memory
and dependency DAG are calibrated. For that reason planner_ready=false
and planner_eligible=false are preserved.

A compact GCNA probe may validate Bank refills without exercising
graphless HTP->Adreno; this must remain NOT_EXERCISED rather than PASS.

## Validation

The changes are committed but new Patch33 Windows/Android builds,
Phone real dual-Bank overlap and N->Adreno end-to-end observations
are not yet verified. Run ctest for
test-region-ring-v2-runtime-profile-import and
test-phone-mixed-pipeline. Retest production mixed pipeline with
matching PC/Phone builds before Planner uses the data.
