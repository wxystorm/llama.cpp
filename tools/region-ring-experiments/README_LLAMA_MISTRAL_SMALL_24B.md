# Mistral Small 3.2 24B Q6_K: dense Llama Region Ring + HTP experiment

Feature branch: `feat/llama-mistral32-region-htp-20261112`.
Start from the exact same PC/Phone commit. Do not mix it with the older
Qwen3-only RPC server binary.

## Target and compatibility boundary

Target GGUF: `mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q6_K.gguf`.
This quant is commonly published as an approximately 19.35 GB text-model
GGUF with metadata architecture `llama`, 40 decoder layers, 5120 hidden
channels, 32 query heads, 8 KV heads, head dimension 128. Always inspect
your own GGUF metadata; the filename is not an architecture proof.

This change enables the *dense LLM text inference* paths already used
by Qwen3: CPU/GPU/Phone Region ownership; GPU weight streaming via two
independently sized banks; independent G/C/N Profile V0; native APPLY_WIDE
DAG Plan; HTP-only/4-session local GGUF preload, Graphless execution,
Phone activation return and PC final output head.

The Qwen3 Graphless path retains head-wise Q/K RMSNorm and NeoX RoPE.
Dense Llama uses Normal RoPE, no Q/K head normalization, and optional
affine Q/K/V/O and FFN biases. The HTP backend already supports Q6_K
repacked weights and Normal RoPE. The Plan's low flag bit 1 identifies
Llama without changing RPC wire payload size; both sides must be rebuilt.

Only supported dense layouts using full-head RoPE, F16 KV and the
Graphless text-only contract are accepted. Special per-layer RoPE
frequency tensors, MoE experts, vision input / mmproj, unsupported
attention biases or other Llama-family variations are NOT asserted to
work. Do not treat a model-name prefix as blanket support for all Llama
architectures. HTP-only benchmark mode is fresh-prefill only with
`-NPredict 1`, not an independently verified multi-token decode path.

## 1. Update and build PC

```powershell
git fetch origin
git switch feat/llama-mistral32-region-htp-20261112
git pull --ff-only
cmake --build build-tensor-v2 --config Release --parallel 8 --target llama-completion test-region-ring-planner test-phone-owned-execution-plan test-phone-qwen3-local-builder
```

## 2. Launch Phone RPC server with SAME GGUF

Copy the exact local text GGUF to a path readable by the Phone server.
The Phone must have storage for the file in addition to HTP model caches.
Example Android shell environment (adapt the binary and GGUF paths):

```sh
export GGML_HEXAGON_NDEV=4
export GGML_RPC_PHONE_HTP_ONLY=1
export GGML_RPC_PHONE_NPU_PLAN_CONTROLLED=1
export GGML_RPC_REAL_STAGE_PROFILE=1
unset GGML_RPC_PHONE_NPU_SERIAL_REGIONS
unset GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS
./build/bin/ggml-rpc-server -H 0.0.0.0 -p 50052 -d CPU \
  -m "/sdcard/models/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q6_K.gguf"
```

The `-d CPU` backend is the RPC transport, NOT the HTP compute owner.
The private four FastRPC Sessions provide actual HTP computation.
All four Sessions share the phone's physical Android RAM.

## 3. One-prompt auto Profile + Plan smoke on PC

Use a new output directory; a Qwen3-32B V0 Profile is **not** reusable
for the different Mistral 24B model. The fixed-layout sweep wrapper
name is historic but accepts `-ModelLayers 40`.

```powershell
.\tools\region-ring-experiments\run_32b_profile5_plan.ps1 `
  -ModelLayers 40 `
  -Model "E:\llama\models\mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q6_K.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\mistral24b-q6k-htp-v0" `
  -ProfileGrid quick -ProfileEvery 0 `
  -Batch 2048 -UBatch 2048 -NPredict 1 -Threads 8 `
  -Limit 1 -StopOnFail
```

After PASS, repeat with the SAME parameters and output directory,
omitting `-Limit 1`, to resume and finish all 40 prompts. The first
run creates an independent `G/C/N` profile which is reused for subsequent
prompts, unless `-UpdateProfile` is explicitly set.

If you want to inspect one known 40-layer layout **without auto search**,
keep `-ModelLayers 40` and add:

```powershell
  -ForceLayout "G4,C5,N1,G4,C5,N1,G4,C5,N1,G4,C5,N1" `
  -ForceXC 64 -ForceXG 64 -ForceXP 64
```

This is an illustrative *balanced-memory* control layout, NOT a
benchmark result or a guarantee of memory fit. Automatic Plan remains
the recommended first experiment. HTP per-Session caps, shared RAM and
total GPU Bank+KV/scratch checks are still enforced.

## 4. Verify first PASS and numerical behavior

Search PC stderr for:
- `[INDEPENDENT_PROFILE]` with nonzero `CPU`, `GPU`, `PHONE` samples
- `[REGION_RING_PLAN_APPLY]` `applied=1`
- `[PHONE_HTP_ONLY]` / `[PHONE_OWNED_PLAN]` `status=READY`
- `[REGION_RING_PC_FINAL_HEAD]` `status=PASS`
- `[REGION_RING_PLAN_FORCE_HTP]` if a forced Plan is rejected

Run PC-only `llama-completion` on the same prompt and compare generated
logits or output after the HTP smoke. A PASS timing result is **not** a
numerical-parity proof. The full Mistral model is far larger than either
the 6 GiB PC GPU or the Phone HTP memory; weight streaming and mixed
placement are intentional.

## 5. Regression tests

```powershell
ctest --test-dir build-tensor-v2 -C Release -R "test-phone-(owned-execution-plan|qwen3-local-builder)|test-region-ring-planner" --output-on-failure
python -m unittest discover -s tools/region-ring-experiments -p "test_run_32b_profile5_runner.py" -v
```

This document records implementation and test commands, not a claim that
the user's PC/Phone have already passed execution.


## 6. Diagnose first-Prefill prediction gap (NOT a benchmark)

The DAG `predicted_ms` is a modeled makespan, not a measured kernel
duration. Before interpreting a large gap, compare actual Ring wall vs
prefill wall, GPU Bank H2D and exposed GPU waits, Phone push/return and
the **PC final output head**. The GPU-tail DAG currently does not score
the extra PC final head; do not assume that omission explains the entire
gap. A Profile with only a few HTP points can extrapolate P1/P3, even
when a real larger-depth NPU sample exists.

Analyze the original first attempt without any re-run or mutation:

```powershell
python .\tools\region-ring-experiments\diagnose_plan_gap.py `
  --out ".\mistral24b-q6k-htp-v0" --prompt prompt_128_01.txt
```

If no Chunk trace was captured in the original run, collect one
**separate diagnostic** with the *same* forced layout and old V0 source
from that run (do not replace original `-Out`):

```powershell
.\tools\region-ring-experiments\run_32b_profile5_plan.ps1 `
  -ModelLayers 40 `
  -Model "E:\llama\models\mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q6_K.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir "E:\llama\datasets\sharegpt\sharegpt40" `
  -Out ".\mistral24b-q6k-trace-only" `
  -RealStageProfileCsv ".\mistral24b-q6k-htp-v0\independent-profile\profile.csv" `
  -ForceLayout "G5,N4,G6,N2,G5,N3,G6,N2,G5,N1,G1" `
  -ForceXC 32 -ForceXG 32 -ForceXP 32 `
  -TraceTasks -ProfileGrid quick `
  -Batch 2048 -UBatch 2048 -NPredict 1 -Threads 8 `
  -Limit 1 -StopOnFail

python .\tools\region-ring-experiments\diagnose_plan_gap.py `
  --out ".\mistral24b-q6k-trace-only" `
  --profile-json ".\mistral24b-q6k-htp-v0\independent-profile\profile.json"
```

The new flag `-TraceTasks` enables an opt-in PC-clock
`region_ring_chunk_trace` timeline, which is serialized only after the
workers drain. The analysis prints per-Region prepare/compute/bank-wait
totals; those totals overlap and **must not be summed** into a predicted
critical path. Comparing the two runs is diagnostic, not official
benchmarking or a numerical correctness proof.
