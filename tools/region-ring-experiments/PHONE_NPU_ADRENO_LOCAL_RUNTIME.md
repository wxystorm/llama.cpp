# Experimental Phone HTP -> Adreno contiguous Stage runtime (original RPC Patch 31)

> On branch feat/four-backend-profile-rpc-gcna-20261107, the current RPC
> protocol is Patch32 and automatic compact G/C/N/A Profile is documented in
> GCNA_AUTO_PROFILE.md. Rebuild both PC and Phone from the same commit.

Branch: `feat/phone-npu-adreno-no-mid-p2c-host-move-20261107`
Status: source integration only. **Not yet compiled on Windows/Android or verified on a Snapdragon phone.** Do not report a device PASS without logs and logits comparison.

## Contract

The PC four-backend Planner selects `G -> C? -> N? -> A?` per Cycle. The first runtime experiment covers only 2-4 **GPU-first, Phone-final** Cycles, each with a positive GPU stage and a positive Phone stage; CPU may be zero, NPU depth is 0/1, and at least two HTP P1 Regions and one directly adjacent `N -> A` edge are required. A `P1,Pn` pair in the C++ Region Ring is *not* two HTP Regions: the PC uses its selected ordered HTP map to designate the first as HTP and the next as GPUOpenCL. Only `N` (HTP) can have a direct successor `A` (Adreno). No device assignment is read from Phone layout-specific environment variables.

Patch31 PC plan configure sends `(HTP region, absolute layer, next Adreno region or zero)` after input-stream START and before owned Graph/KV registration. Phone accepts the HTP map only with two real weight Banks and validates that any local successor is exactly the next Region and is not HTP. Failure is fatal; no implicit OpenCL fallback.

HTP Graphless compute creates one F32 Host output vector. When its Plan contains an adjacent Adreno successor, the Phone **moves ownership of that same vector** to the bounded destination Host landing queue (`std::move`), wakes the OpenCL preparation Worker, and uploads from the owned vector directly into GPUOpenCL Graph/Slot0. There is no second Host memcpy, no pseudo-network packet or per-Chunk hash for these locally trusted bytes, and no PC->Phone Activation PUSH for this edge. This is **not** device-level shared-memory zero-copy: the necessary Host->OpenCL device upload remains. KV never travels with activations.

The internal HTP->Adreno edge sends **neither a full intermediate P2C frame nor a per-Chunk completion notification**. At PC dispatch time, Region Ring logically completes the N stage only to transfer ownership to the adjacent Phone A stage; this is **not** an assertion that HTP computation has finished. The Phone's own FIFO, output publication and KV rules gate Adreno. Actual **Adreno->PC** and final **Phone->PC Head** boundaries still return full, validated data, so the next PC GPU/CPU stage cannot advance until the real data arrives. NPU two-Bank reuse waits for the last local Host handoff (or actual P2C if N is terminal), not for any fictitious intermediate frame. The offline DAG now models this ownership boundary without inventing P2C bandwidth.

## Build on both sides

On Windows PowerShell, from the repo root:

```powershell
git fetch origin
git switch feat/phone-npu-adreno-no-mid-p2c-host-move-20261107
git pull --ff-only
cmake --build build-tensor-v2 --config Release --target llama-completion test-region-ring test-phone-plan-region-map test-phone-local-host-handoff -j 8
ctest --test-dir build-tensor-v2 -C Release -R "test-region-ring|test-phone-plan-region-map|test-phone-local-host-handoff" --output-on-failure
python -m unittest discover -s tools/region-ring-experiments -p "test_mixed_phone*.py" -v
```

Recompile the Android `ggml-rpc-server` and Hexagon-related binary components **from this same branch**, not Patch30. Keep your known-working OpenCL, `LD_LIBRARY_PATH`, `OCL_ICD_VENDORS` and HTP SDK settings.

On Phone Termux (Qwen3-14B example):

```sh
unset GGML_RPC_PHONE_NPU_SERIAL_REGIONS
unset GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS
export GGML_RPC_PHONE_NPU_PLAN_CONTROLLED=1
export GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2
export GGML_RPC_PHONE_NPU_MIXED_OPENCL=1
export GGML_RPC_PHONE_NPU_GPU_PIPELINE=1

export GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1
export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_INPUT=1
export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=1
export GGML_RPC_PHONE_QWEN3_LOCAL_GRAPHLESS=1
export GGML_RPC_PHONE_NPU_HOST_STAGE_MIB=512

export GGML_RPC_PHONE_BINDING_PREFLIGHT=1
export GGML_RPC_PHONE_STAGE_SLOT0_EXEC=1
export GGML_RPC_PHONE_STAGE_DUAL_SLOT=1
export GGML_RPC_PHONE_PERSISTENT_POOL_PREP=1
export GGML_RPC_REAL_STAGE_PROFILE=1
export GGML_RPC_PHONE_CHUNK_TIMELINE=1

./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d GPUOpenCL \
    -m "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 2>&1 | tee phone-local-handoff.log
```

## Profile -> Plan -> Run (PC)

A **real, matching** `model-profiles/Qwen3-14B-Q4_K_M` bundle with `real-stage-mixed.csv` and `phone_mixed_calibration.json` is mandatory. Calibrate HTP/Adreno compute, Phone-local transfer and NPU Bank repack on the actual devices. Never generate synthetic rates to manufacture a runnable Plan.

```powershell
.\tools\region-ring-experiments\run_plan.ps1 `
  -PhoneMixed -RuntimeCompatible `
  -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  -PromptFile "D:\prima.cpp\prompts\prompt_512.txt" `
  -Out ".\offline-plans\qwen14b-phone-local-512"

.\tools\region-ring-experiments\run_saved_plan.ps1 `
  -Plan ".\offline-plans\qwen14b-phone-local-512\plan.json" `
  -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir ".\single-prompt-512" `
  -Out ".\phone-local-handoff-smoke" `
  -NoMmap -NPredict 1 -ConfirmBankedPhoneReady
```

`-ConfirmBankedPhoneReady` is an operator acknowledgement of matching tested Phone binaries/configuration, **not** a probe or automatic health check. Do not use it to bypass known initialization failures. The saved-plan runner validates schema2, ordered Regions/layers, NPU mapping, adjacency and minimum two HTP Banks, then translates `N/A` to distinct `P,P` *runtime stages*, while preserving the PC-selected HTP map in RPC.

Start by validating `-NPredict 1` Prefill and parity against a nonmixed reference; only then try `-NPredict 32` to test Decode/KV synchronization. The current local handoff requires equal Chunk widths for the two Phone Stages; schema2 uses a shared X among all four engines.

## Expected evidence in logs

- PC: `[PHONE_NPU_PLAN] side=pc event=CONFIGURE/INSTALL ... status=PASS` and `[PHONE_LOCAL_HANDOFF] side=pc event=SKIP_PUSH ... status=PASS`
- Phone: `[PHONE_NPU_PLAN] side=phone event=CONFIGURE ... local_links=... status=PASS`, `[PHONE_LOCAL_HANDOFF] event=PUBLISH ... status=PASS`
- Phone: `[PHONE_MIXED_PIPELINE] event=DEVICE_BEGIN/DEVICE_END ... backend=HTP0/GPUOpenCL` for both physical engines
- PC: `[PHONE_LOCAL_HANDOFF] side=pc event=DELEGATE` for internal HTP and `[PHONE_INPUT_PUSH] event=SUMMARY ... local_skipped=N` (this field is separate from successful network PUSHs)
- Phone: `[PHONE_LOCAL_HANDOFF] event=NO_INTERMEDIATE_P2C ... payload_bytes=0 completion_frame=SKIPPED`, with **no P2C frame for those HTP Region/Chunk IDs**
- Phone: `[PHONE_INPUT_STREAM_SUM] ... local_published=N local_applied=N local_host_copy_bytes=0`; its `frames/verified/applied` counts refer only to network input, not internal HTP handoffs
- Both Phone and PC: complete P2C for every **cross-device** Adreno-to-PC or terminal Phone Region, `[PHONE_AUTO_SUM] ... failed=0` and no `REGISTER_REJECT`, Bank/Graph/KV mismatch or silent backend fallback
- End-to-end: Prefill PASS, correct logits, and (for Decode tests) successful KV synchronization

A nonzero `htp_gpu_service_overlap_ms` shows overlap between service intervals (including copies); kernel-level parallelism still needs separate hardware profiling. Under short Chunks, zero measured overlap need not mean a correctness failure.

## Current limits

This experimental runtime does **not** claim full arbitrary zero-width G/C/N/A execution. Offline planning can model zero G/Phone stages, but executable schema2 still requires GPU-first and Phone-final Cycles, at least two HTP P1 Regions, at least one Adreno and one adjacent HTP->Adreno edge. A multi-layer Adreno stage may need further device-specific Graph/Slot0 qualification; use P1 first when diagnosing hardware failures. The model is currently the Qwen3 whole-Transformer-layer path, and no Snapdragon/Windows device PASS has been collected yet. The Plan's predicted DAG is not a validated throughput guarantee.
