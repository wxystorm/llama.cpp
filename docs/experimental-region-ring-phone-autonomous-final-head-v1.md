# Experimental Phone-owned final L39 and PC-only Qwen3 Head (V1)

This is a **separate opt-in experiment** extending Phone autonomous graph slots.
It does not change the validated final-Phone Meta path unless explicitly enabled.

## Ownership

Under a Qwen3 dense full-Region plan ending in Phone (for example
`G10,C1,P3,G11,C1,P1,G11,C1,P1`):

- PC builds and registers **all** non-CPU Phone Chunk graphs, including the
  terminal Region L39, with exact, private per-Chunk input and output buffers.
- The terminal Phone graph explicitly sets `hybrid_output_head=false`, so its
  only full-size output is `l_out-39`. It owns Attention, FFN, residual ADD,
  and KV writes; no LM Head or final RMSNorm is computed on Phone.
- After verified Input PUSH the Phone Prepare Worker uploads/binds an available
  dual slot and its Compute Worker executes in per-Region KV FIFO order.
- Phone Compute completes -> captures the exact F32 l_out to the P2C Arena ->
  the independent Phone Stream Sender pushes the frame to PC. There is no
  per-Chunk WAIT_CAPTURE or per-Chunk PC RPC Graph compute.
- PC binds **all** 18 output destinations before PUSH starts, and the P2C
  Receiver completes a deferred task only after the complete frame arrives.
- Once all Region Ring tasks finish, the PC main scheduler builds a dedicated
  **zero-layer Qwen3 Head-only graph** from the **last output token's l_out**:
  `inp_stage(F32) -> output RMSNorm -> LM Head -> logits`. This uses the
  canonical main `llm_graph_result` for the caller's normal output extraction.
- Fail closed if the Head graph has any Meta backend split: PC Head must stay
  on CPU/CUDA. No fallback to legacy Phone Meta on this opt-in path.

## Enable

First build this feature branch on **PC**. The RPC protocol is unchanged
from patch26 but the Phone RPC server should still be the matching current
autonomous-version build. Leave the other usual experiment settings in place:

```powershell
$env:GGML_META_PHONE_FULL_REGION_GRAPH = "1"
$env:LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES = "1"
$env:LLAMA_REGION_RING_PHONE_INPUT_STREAM = "1"
$env:LLAMA_REGION_RING_PHONE_AUTONOMOUS = "1"
$env:LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM = "1"
$env:LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL = "1"

.\tools\region-ring-experiments\run_prompt_sweep.ps1 `
  -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  -Rpc "10.39.52.114:50052" `
  -PromptDir ".\single-prompt" `
  -Out ".\planner-test-results-phone-auto-final" `
  -Planner "apply-wide" -Streaming "on" `
  -NPredict 32 -Threads 8 -Batch 512 -UBatch 512 `
  -ForceLayout "G10,C1,P3,G11,C1,P1,G11,C1,P1" `
  -ForceXC 16 -ForceXG 16 -ForceXP 64 -LogVerbosity 4
```

## Expected experiment observables

- PC: `[PHONE_AUTO_P2C] event=PREBIND ... outputs=18`.
- Phone: `[PHONE_AUTO_SUM] ... registered=18 completed=18 failed=0 published=18`.
- Phone: `[P2C_STREAM] side=PHONE phase=STOP ... frames=18 ... status=OK`.
- PC: `[PHONE_AUTO_FINAL_HEAD] event=PC_EXEC tokens=1 source_region=8
  source_token=357 ... status=PASS`.
- `summary.csv`: `phone_auto_final_enabled=True`,
  `phone_final_head_ms`, `phone_split_prepare_ms`, `prefill_ms`.
- With all three Phone Regions autonomous, the PC's per-Chunk final
  `PUSH_APPLY` and Meta rebuild calls should no longer be exercised.
- Compare final output tokens and, if available, logits with the same
  prompt/seed/config running the existing final Meta path
  (`LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL=0`).
  A `PASS` status alone is not a numerical parity check.

## Deliberate V1 limits

Only Qwen3 dense, single terminal output, fixed threaded full-Region
Prefill with the existing autonomous Graph and Input/P2C streams.
Warmup and one-token Decode stay on the validated legacy path.
The PC Head runs after **all** Ring outputs have arrived (simple correctness
barrier). Overlap of Head with other trailing Region tasks is not yet
optimized. The terminal Phone region currently publishes all 358 hidden
vectors even though the Head only uses the final one; the first version
reuses the existing full-size P2C capture contract unchanged.

This is an experimental implementation with **no PC/Android end-to-end
compile or numeric-parity test yet**. Keep the opt-in flag off for established
baseline runs until the new path is validated on actual hardware.
