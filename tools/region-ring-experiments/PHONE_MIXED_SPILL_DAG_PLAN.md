# Four-backend whole-layer Cycle Plan with HTP two-Bank spill

Branch: feat/phone-four-backend-spill-dag-20261107.
Status: Plan itself remains offline (no inference, model-load profiling or
RPC). The opt-in Patch31 experimental runtime now supports a validated
GPU-first/Phone-final subset with adjacent HTP -> Adreno activation handoff;
untested model/hardware combinations remain experimental. See
PHONE_NPU_ADRENO_LOCAL_RUNTIME.md for the exact requirements and smoke.
Do not treat a schema=2 Plan as proof of an Android PASS.

## Geometry and memory

Candidate Cycle is G(optional) -> C(optional) -> N(optional) -> A(optional):

- G: PC CUDA, rolling two weight Banks (may be zero)
- C: PC CPU (may be zero)
- N: Phone HTP NPU, current executor limits each Region to 1 layer (may be zero)
- A: Phone GPU Adreno OpenCL, whole Transformer layers (may be zero)

Create bounded balanced G,C,P_total Cycle seeds, including zero-depth
backend choices; then search N in [0,max_supported] and A=P_total-N per
Cycle. Even when the initial NPU allocation
overflows NPU memory, DO NOT reject the whole geometry: enumerate
spill into Phone GPU before applying memory pruning.

PC GPU memory = max(even G weights) + max(odd G weights) +
                ALL G KV + GPU scratch
HTP NPU memory = max(even actual N weights) + max(odd actual N weights) +
                 ALL N KV + NPU graph/scratch
Phone GPU memory = ALL A weights (resident) + ALL A KV + OpenCL scratch
Phone shared memory = HTP NPU memory + Phone GPU memory +
                      temporary Bank host-staging peak (rank k>=2)

The host-staging peak must also fit the measured
npu_host_stage_budget_mib (e.g. the configured Phone Host Stage cap).

Reject only when PC GPU, NPU, Phone GPU, or the shared Phone RAM budget
cannot accommodate the resulting split. Missing NPU Regions do not consume
a Bank rank. Current real HTP Profile supports P1; no invented P2 timings.
Phone GPU Pn service may be modeled as the sum of measured P1 services,
explicitly marked as a linear depth projection.

## Exact event DAG

The initial search uses 2..4 balanced Cycles and chooses a common Chunk
size from 16,32,64,128. A future iteration may choose independent XG,
XC, XN and XA. Events preserve inter-Layer dependencies and per-Region
KV FIFO, while using independent CPU/CUDA/HTP/Adreno compute resources.

Events include:
- PC local G->C copy, C/G->N/A over C2P, N->A local Phone upload,
  and A/N->next G over P2C
- PC GPU weight H2D: ranks 0/1 preloaded; rank k>=2 can start after
  rank k-2 finishes its LAST Chunk, on its own serial H2D lane
- HTP two Banks: ranks 0/1 preloaded; rank k>=2 host-stages weights
  after rank k-2 finishes LAST Chunk, on independent host IO;
  HTP repack then uses the HTP compute lane (cannot overlap HTP compute)
- Final Phone-to-PC return and PC Head

The predicted critical path includes bank mover events. Diagnostics in
plan.json include gpu_bank_h2d_ms, npu_bank_stage_ms,
npu_bank_repack_ms and per-resource busy_ms. These are simulation values,
not guaranteed hardware throughput.

## Separate, model-bound Profile bundle

    model-profiles/<GGUF-stem>/
       model_profile.json            existing base Profile
       real-stage-profile.csv        existing HTP-only CSV, untouched
       real-stage-mixed.csv          CPU/GPU/HTP/Adreno separate samples
       phone_mixed_calibration.json  measured Phone budgets & transfers

A mixed CSV has the same original ten columns:

    device,layers,tokens,kv_tokens,setup_ms,upload_ms,prepare_ms,compute_ms,terminal,samples

The device classes are CPU (PC CPU), GPU (PC CUDA), PHONE (Phone HTP0),
and PHONE_GPU (Adreno OpenCL). HTP Graphless prepare_ms=-1.000 means
no OpenCL Slot preparation; this sentinel is normalized to 0 in mixed
profile extraction. No fake Phone GPU data is generated.

After collecting a mixed smoke in the separate Profile phase:

    python .\tools\region-ring-experiments\build_real_stage_profile.py --run ".\phone-opencl-htp-mixed-smoke\<run-folder>" --rpc-log ".\phone-pipeline.log" --mixed --output ".\model-profiles\Qwen3-14B-Q4_K_M\real-stage-mixed.csv"

Also create the model-labelled phone_mixed_calibration.json, with schema=1,
model_fingerprint equal to model_profile.json:model.fingerprint,
stage_csv="real-stage-mixed.csv", and calibration containing
all of these measured positive numeric fields:

    phone_gpu_budget_mib
    phone_gpu_weight_layer_mib
    phone_gpu_scratch_mib
    phone_shared_budget_mib
    phone_local_mibps
    npu_host_stage_mibps
    npu_host_stage_budget_mib
    npu_bank_repack_mibps
    pc_head_ms (optional, >=0)

No guessed budget or copy bandwidth is silently substituted. The
two-region NPU smoke does NOT measure rank>=2 Bank host-stage/repack
throughput; those rates require an additional standalone calibration.
Actual Prompt KV memory is derived from the GGUF metadata and true
tokenizer count, not from a fixed Profile Prompt length. Absent CUDA,
HTP, or Adreno engines reserve no unnecessary scratch or weight Banks.
A CUDA-terminal candidate is scored only with an optional measured
base-calibration gpu_d2h_mibps for the final CPU Head.

## Formal Plan (never profiles)

    .\tools\region-ring-experiments\run_plan.ps1 -PhoneMixed -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -PromptFile "D:\prima.cpp\prompts\prompt_512.txt"

Outputs plan.json (schema=2, mixed_layout=G/C/N/A, runtime_compatible
flag determined by the current experimental runtime constraints)
and candidates.csv. The file now includes execution_stages (absolute layer
ranges, Region indices and backend) and execution_edges (handoff type and
resource lane). Zero-width stages are omitted before IDs are assigned.
HTP_TO_ADRENO now has an experimental Patch31 Phone host-F32 staging
implementation. PC still receives HTP P2C completion frames, and the DAG
models those side returns as a real transfer cost; NPU Bank reuse waits for
that final capture. No silent OpenCL fallback is allowed.
The old 3-device Profile/Plan flow stays unchanged.

To disable HTP search candidates, pass -MaxNpuRegionLayers 0. Depths greater
than 1 remain unsupported until measured HTP multi-layer timings and runtime
exist. The search is still based on balanced 2-4 Cycle seeds and one shared
Chunk width; full per-Cycle beam search remains future work.

run_saved_plan.ps1 only executes a schema=2 plan when its
runtime_compatible flag is true and -ConfirmBankedPhoneReady was explicitly
passed. It converts N/A to separate contiguous P Regions, never merging
or silently reassigning the physical backend; NPU assignments travel in the
Patch31 Plan RPC. Unsupported candidates fail before execution.

Test offline policy after syncing the branch:

    python -m unittest -v tools/region-ring-experiments/test_mixed_phone_dag_plan.py
    python -m unittest -v tools/region-ring-experiments/test_build_real_stage_profile.py

NOTE: the second command may require executing from the tools folder to
resolve local Python imports. No Android Phone is needed to test search.
