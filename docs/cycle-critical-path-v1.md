# Cycle V5 Critical-Path Trace (opt-in)

This trace is **observational only**. It does not change Phone MAIN priority,
C2P FFN/Reduce Graph execution, PC CUDA/CPU placement, network sends, KV
admission, or buffer leases. Use it on the **same** 64-token / 16-token-chunk
test and `max_inflight=2` that already runs successfully.

## Enable

On the PC (PowerShell), retain all known-good V5/P2C settings and add:

```powershell
$env:LLAMA_HYBRID_CYCLE_CRITICAL_PATH = "1"
$env:LLAMA_HYBRID_CYCLE_PERF_MODE = "1"
```

On the Phone (Termux), retain the current RPC server settings and add:

```sh
export GGML_CYCLE_CRITICAL_PATH=1
```

Rebuild both PC and Phone from this branch first. No RPC protocol version
change is required relative to V5 Patch 33. No extra device fences or
network control round-trips are used for the trace.

Extract the PC trace:

```powershell
Select-String -Path .\cycle-pc.log -Pattern 'CYCLE_PATH_TASK|CYCLE_PATH_CHUNK|CYCLE_PATH_CREDIT|CYCLE_PATH_SUM|CYCLE_PERIOD_PERF_SUM|CYCLE_PERIOD_PARITY'
```

Extract the Phone trace:

```sh
grep -E 'CYCLE_PHONE_PATH_JOIN|CYCLE_PHONE_PATH_MAIN|CYCLE_PHONE_EXECUTOR_V1|CYCLE_PHONE_V5_P2C' rpc.log
```

## What is measured

- `CYCLE_PATH_TASK`: PC-local monotonic timestamps relative to that period's
  FIFO start for **task queued, callback started and callback finished**.
  `queue_wait_ms` is the wait AFTER a task has entered its normal FIFO.
  The Phone GPU queue on PC is an **RPC proxy**, not the Phone GPU's clock.
- `CYCLE_PATH_CREDIT`: an oldest, causally ready Chunk was blocked before
  being queued because the layer had no free `max_inflight` entry credit.
  This differs from FIFO queue wait. Credit waits for different layers may
  overlap and MUST NOT be added together as if they were wall time.
- `CYCLE_PATH_CHUNK`: PC-observed C2P, L5/L6 completion ACK, Phone L7
  completion ACK, PC READY/PARTIAL reception, PC CPU FFN and final Reduce
  milestones on **one PC monotonic clock**. A remote ACK timestamp means
  the PC **observed** completion, not the exact Phone GPU finish time.
- `CYCLE_PATH_SUM`: first/last Chunk return, maximum C2P-to-P2C span,
  P2C's last ready dependency (`PC_CPU_FFN` vs `PHONE_PARTIAL_RX`),
  and PC CPU Reduce scheduler delay AFTER both inputs were already ready.
- `CYCLE_PHONE_PATH_JOIN`: Phone-local time from GPU READY, FFN completion,
  PC Partial device-readiness and Region registration to ordered MAIN Reduce
  admission. A high `region_to_admit_ms` is not by itself evidence that
  registration caused delay: the other join inputs may arrive later.
- `CYCLE_PHONE_PATH_MAIN`: on the **Phone clock only**, elapsed from
  C2P Reduce MAIN admission to Phone L7 READY staged, L7 READY to PARTIAL
  staged, and admission to PARTIAL staged. `STAGED` does **not** mean the
  peer received the network frame.

**Do not subtract Phone timestamps from PC timestamps.** Only match Chunk,
layer, phase and sequence IDs between logs. Likewise, sums of worker
callback times cannot be interpreted as end-to-end latency: workers overlap.

The current 8-layer physical-period test **ends at L7 P2C PC Reduce**.
It does not run the next PC CUDA cycle, so `next_cuda=NOT_PRESENT_IN_THIS_8_LAYER_PERIOD`
is an intentional coverage boundary, not a missing event.

## How to use the evidence

1. Compare `MEASURE` with the previous trace-disabled V5 run to estimate
   instrumentation overhead. Discard warmup when comparing steady execution.
2. Inspect `CYCLE_PATH_SUM` and `CYCLE_PATH_CHUNK`: determine whether the
   last ready P2C dependency is usually PC CPU FFN or Phone Partial reception.
3. If `join_ready_to_cpu_reduce_start_ms` is high, inspect PC_CPU's
   `CYCLE_PATH_TASK` queued/start records and the existing PC_CPU queue profile.
4. If `CYCLE_PATH_CREDIT` has large blocking spans, inspect the Phone
   and P2C task progression before modifying the window size. The current
   full-main implementation allows windows of 1 and 2, while the supplied
   run logs currently establish window 2;
   do not raise the window to 3 without verifying additional buffer leases.
5. If Phone `main_admit_to_partial_ms` is small while the overall PC
   Chunk span is large, the bottleneck is outside this local Phone chain.
6. If the host P2C receive duration is large, it includes *waiting for the
   frame and decoding*; it is NOT a pure network-wire latency measurement.
   Separate wire timing requires transport instrumentation on one local clock
   or a rigorously calibrated cross-machine clock protocol.

Numerical parity remains an independent gating issue. The previously
reported `max_abs=15.6350098` / strict `FAIL` must not be relabeled
as PASS simply because the executor and trace complete successfully.
