# Region Ring: PC-driven Phone input PUSH (experimental)

**Opt-in and default-off.** This feature changes both the PC and Android RPC
server implementation. Both binaries must be rebuilt from the SAME commit:
the RPC patch version is now **22** (PUSH timing payload added).

On the PC, before invoking the ordinary prompt sweep:

```powershell
$env:LLAMA_REGION_RING_PHONE_INPUT_PUSH = "1"
$env:LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES = "1"
$env:GGML_META_PHONE_FULL_REGION_GRAPH = "1"
$env:LLAMA_REGION_RING_TASK_TRACE = "1"
$env:LLAMA_REGION_RING_PREPARE_BREAKDOWN = "1"
$env:LLAMA_REGION_RING_RPC_GRAPH_PROFILE = "1"
```

## What it implements

1. A Phone region's input becomes **pushable** as soon as every overlapping
   upstream token interval has completed. This is deliberately independent of
   its preceding Phone chunk's KV commit.
2. One independent PC pusher Worker reads immutable full-ubatch edge storage
   and uploads `n_tokens * n_embd * sizeof(float)` bytes via a persistent,
   dedicated RPC socket. Phone accepts the input into **host-only** landing
   storage: maximum **two staged chunks per Region**. It does not touch a live
   OpenCL graph tensor on receipt.
3. The existing Phone Compute Worker waits for its staged chunk before it
   acquires PREPARE. Its normal `prepare_split()` still binds other inputs.
   When the matching full hidden input reaches RPC `set_tensor()`, the client
   sends a small `APPLY` command. The server then sets the ORIGINAL RPC
   destination tensor from the stored bytes.
4. The ordinary Phone graph and KV order are unchanged. No new graph binding,
   registration or inference-time alternate input address is assumed.
   Non-matching input tensors use the legacy `SET_TENSOR` path.
5. After consumption, the PC releases the independent landing-slot credit.
   A missing match is reported as `phone_input_push_fallback` and the normal
   `SET_TENSOR` still supplies the input.

The Phone-side landing slots are **host staging**, not the existing Phone GPU
activation-slot pool. This first version moves the network upload out of the
Phone Compute Worker, but does **not** guarantee OpenCL upload or device compute
overlap. Do not confuse requested input size with actual network throughput.

## Preconditions and safety

- Only non-directional Full-Region threaded prefill with independent PC/Phone
  workers and full-ubatch edge storage is eligible. Decode, Tensor/C2P/P2C,
  bounded legacy output-slot mode and missing RPC capabilities keep the
  original execution path.
- The server never writes to current Graph input tensors from the pusher
  connection. The `APPLY` path validates run/region/chunk, bytes and a hash.
  A mismatch leaves the original `SET_TENSOR` enabled.
- The landing queue is bounded per Region. The separate scheduler credits
  block further pushes until a Phone Worker consumes an earlier staged input.
- Cancelled runs try to drop unused staged inputs. Correctness must be
  validated with output parity and full prefill/decode smoke tests, not only
  successful RPC acknowledgments.

## New metrics

The sweep exports:

- `phone_input_push_enabled`
- `phone_input_push_accepted`, `phone_input_push_applied`
- `phone_input_push_fallback`
- `phone_input_push_mib`, `phone_input_push_wall_ms`
- `phone_input_push_worker_wait_ms`

Inspect the first two new counters: a run with `accepted>0` but
`applied=0` did not remove the actual hidden input SET traffic.

```powershell
$run = "E:\llama\llama-tensor-v2\your-new-sweep-run"
Import-Csv "$run\summary.csv" |
    Select-Object status,prefill_tps,ring_wall_ms,
        phone_input_push_enabled,phone_input_push_accepted,
        phone_input_push_applied,phone_input_push_fallback,
        phone_input_push_mib,phone_input_push_wall_ms,
        phone_input_push_worker_wait_ms,
        phone_split_copy_call_ms,gpu_bank_wait_ms |
    Format-List
```

For A/B parity, run identical model, prompt, layout, XC/XG/XP,
Batch/UBatch, PC/Phone binaries and other flags with this one difference:

```powershell
$env:LLAMA_REGION_RING_PHONE_INPUT_PUSH = "0"  # baseline
$env:LLAMA_REGION_RING_PHONE_INPUT_PUSH = "1"  # opt-in
```

## ACK and transport timing breakdown (RPC patch 22)

The PC now measures PUSH in the *existing* call, without additional RPCs:
`phone_input_push_hash_ms`, `phone_input_push_pack_ms`,
`phone_input_push_header_send_ms`,
`phone_input_push_payload_send_ms`,
`phone_input_push_socket_send_ms` (sum of the two send calls),
`phone_input_push_ack_wait_ms`, and
`phone_input_push_socket_lock_ms`. Phone reports its own measured durations
*inside the same ACK*: `phone_input_push_phone_recv_header_ms`,
`phone_input_push_phone_recv_alloc_ms`,
`phone_input_push_phone_recv_payload_ms`,
`phone_input_push_phone_hash_ms`,
`phone_input_push_phone_slot_wait_ms`,
`phone_input_push_phone_host_stage_ms` and
`phone_input_push_phone_handler_ms`.

All values are cumulative across measured Push chunks. `ack_wait_ms` is
**not** pure network RTT: it includes any unfinished Phone receiver processing,
the Phone response, and scheduling. `payload_send_ms` is the PC-side duration
of blocking socket `send` calls and does not prove that bytes have already
arrived. `phone_recv_payload_ms` includes time spent blocked in Phone `recv`,
not just on-the-wire transmission. The Phone and PC use different clocks, so
only durations are compared; PC/Phone intervals may overlap and **must not be
summed** into `push_wall_ms`.

The existing trace flag writes a per-Chunk breakdown with no per-Chunk logging:

```powershell
$env:LLAMA_REGION_RING_TASK_TRACE = "1"
$env:LLAMA_REGION_RING_PHONE_INPUT_PUSH = "1"
# Re-run the existing 358-token prompt sweep.
$run = "E:\\llama\\llama-tensor-v2\\planner-test-results-full-region-v7\\YOUR-NEW-RUN"
Import-Csv "$run\\summary.csv" |
    Select-Object status,prefill_tps,phone_input_push_wall_ms,
        phone_input_push_hash_ms,phone_input_push_pack_ms,
        phone_input_push_socket_send_ms,phone_input_push_ack_wait_ms,
        phone_input_push_phone_recv_payload_ms,
        phone_input_push_phone_hash_ms,
        phone_input_push_phone_host_stage_ms |
    Format-List

Import-Csv "$run\\prompt_358\\phone_input_push_timeline.csv" |
    Select-Object region,chunk,tokens,push_wall_ms,push_hash_ms,
        push_pack_ms,push_socket_send_ms,push_ack_wait_ms,
        push_phone_recv_payload_ms,push_phone_hash_ms,
        push_phone_host_stage_ms |
    Format-Table -AutoSize
```

For optional *Phone-side* ACK-send timing, launch the Android RPC server
with `GGML_RPC_PHONE_INPUT_ACK_TRACE=1` and filter
`[PHONE_INPUT_PUSH_ACK]`. Its `ack_send_ms` **cannot** be included in the
ACK being timed and is therefore logged separately on the Phone only.

This profiling does not change the producer/consumer schedule, staged-input
semantics, Graph binding or ACK flow. Neither the old summary nor existing
server logs can reconstruct the new breakdown retroactively.

No speedup is claimed until Windows + Android runtime builds, output parity and
measured timelines have been checked.
