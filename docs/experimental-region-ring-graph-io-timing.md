# Region Ring: Phone Graph and Split Prepare profiling

These counters are **diagnostic only**. They do not modify the planner, locking
policy, RPC protocol, tensor ownership, or the Phone double-slot executor.

Enable on the **PC process** before `run_prompt_sweep.ps1`:

```powershell
$env:LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES = "1"
$env:GGML_META_PHONE_FULL_REGION_GRAPH = "1"
$env:LLAMA_REGION_RING_TASK_TRACE = "1"
$env:LLAMA_REGION_RING_PREPARE_BREAKDOWN = "1"
$env:LLAMA_REGION_RING_RPC_GRAPH_PROFILE = "1"
```

The sweep already provides `LLAMA_REGION_RING_METRICS_FILE` and writes
`prompt_358/metrics.jsonl`, `prompt_358/engine.stderr.log`, and
`summary.csv`. A/B comparisons should keep layout, token counts, model,
Batch/UBatch, and graph/debug settings constant. Leave the two new profiling
switches equal in both arms.

## New summary columns

- `phone_split_target_wait_ms`: wall time spent *calling* destination
  event-wait/synchronize APIs inside `prepare_split()`. Event wait can enqueue
  an asynchronous dependency and does not necessarily wait for GPU completion.
- `phone_split_source_sync_ms`: explicit producer backend synchronization.
- `phone_split_copy_call_ms`: time in synchronous tensor-copy APIs or async
  copy enqueue APIs; not necessarily the eventual device-to-device transfer.
- `phone_split_profiled_ms`: timing of the instrumented prepare_split body.
  Compare to preexisting `phone_split_prepare_ms`; the latter also includes
  caller-side timer/getter overhead.
- `phone_split_copy_mib` and `phone_split_copy_calls`: requested copy payload,
  not observed network bytes. The same fields exist for CPU and GPU devices.
- `meta_phone_full_graph_assemble_ms`: subset of Meta
  `meta_graph_execute_ms` spent assembling Phone Full Region Graph node lists.
- `phone_rpc_graph_serialize_ms`: encoding complete `RPC_CMD_GRAPH_COMPUTE`
  payloads.
- `phone_rpc_graph_serialized_mib`: serialized graph request size.
- `phone_rpc_graph_compute_cmd_ms` and
  `phone_rpc_graph_recompute_cmd_ms`: client wall time during the corresponding
  RPC command calls. These may include server computation and acknowledgment;
  **do not** report them as pure registration or wire-transfer time.

`phone_rpc_regions_json` also has these RPC fields as raw `_us` and bytes.
When `LLAMA_REGION_RING_TASK_TRACE=1`, each
`region_ring_chunk_trace` record contains `split_target_wait_ms`,
`split_source_sync_ms`, `split_copy_call_ms`, `split_profiled_ms`,
`split_copy_bytes`, and `split_copy_calls`.

All timings are cumulative across the indicated chunks and may overlap with
other device workers. Do not sum CPU/GPU/Phone times into prefill wall time.
The existing `meta_graph_rebuild_ms` measures Meta subgraph rebuild and does
not include Phone Full Graph assembly or remote RPC graph registration.

## First analysis

```powershell
$run = "PATH_TO_RUN"
$fields = @(
    "status", "prefill_tps", "ring_wall_ms",
    "phone_split_prepare_ms", "phone_split_profiled_ms",
    "phone_split_target_wait_ms", "phone_split_source_sync_ms",
    "phone_split_copy_call_ms", "phone_split_copy_mib",
    "meta_phone_full_graph_assemble_ms",
    "phone_rpc_graph_serialize_ms", "phone_rpc_graph_serialized_mib",
    "phone_rpc_graph_compute_cmd_ms", "phone_rpc_graph_recompute_cmd_ms"
)
Import-Csv "$run\summary.csv" |
    Select-Object -Property $fields |
    Format-List
```
