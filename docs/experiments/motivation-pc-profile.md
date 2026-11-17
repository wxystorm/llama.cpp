# PC-only Motivation profiling (experimental)

Base commit: `494d0f47bde61fd45018acab8334cdfd0d8b3ca0`.

Set `LLAMA_MOTI_PROFILE=1` to enable opt-in measurements. Unset it to run without the extra instrumentation.

Example (PowerShell, from the repository root):

```powershell
$env:LLAMA_MOTI_PROFILE = "1"
.\build\bin\Release\llama-cli.exe `
  -m "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" `
  -ngl auto -t 8 -c 3000 -b 2048 -ub 512 `
  -f "E:\llama\datasets\sharegpt\sharegpt40\prompt_1024_01.txt" `
  -n 1 --perf -lv 3 2>&1 | Tee-Object .\moti-32b.log
Remove-Item Env:LLAMA_MOTI_PROFILE
```

The profiler starts after context scheduling and output reservation for a multi-token logical batch (at least 2 tokens). It now logs `[MOTI_BEGIN]` when a profiling interval starts, `[MOTI_SYNC]` when synchronization occurs, and always emits `[MOTI_PREFILL]` when an active interval ends—even if the current queued-token count is 0 or 1. The `tokens` value is the token count queued at that synchronization, not necessarily the full prompt length. This makes premature synchronization visible rather than silently dropping the measurement. A run may produce multiple intervals.

The `[MOTI_PREFILL]` summary is deliberately split into three short log records with `section=timing`, `section=transfer`, and `section=memory` to avoid long-line truncation by logging consumers. Search for all `[MOTI_PREFILL]` records; `section=transfer` carries H2D/D2H MiB and call counts.

The `[MOTI_PREFILL]` records include:

- `wall_ms`: measured wall time over the instrumented decode window (not identical to application prompt-eval time).
- `cpu_ms`: sum of CPU backend graph-compute wall time, excluding separate graph-planning time.
- `gpu_stream_ms`: sum of CUDA graph stream intervals recorded with CUDA events; includes kernels, some device copies and possible stream gaps. **Not kernel-only GPU compute.**
- `h2d_ms`, `d2h_ms`, bytes and calls: covered CUDA tensor transfer paths, measured with CUDA events. Not exhaustive: batched/2D and internal CUDA copies are not yet included.
- `peak_working_set_mib`: Windows process *lifetime* peak working set, not just the prefill peak.
- `private_mib`: Windows private bytes at the end of the measurement.
- `gpu_device_used_peak_mib`: sampled device-wide VRAM in use, not process-only usage.
- `page_faults_all`: Windows *total* process page fault delta (hard + soft); **not disk reads**.

Interpretation: the intervals **may overlap**. Never sum them to reconstruct the wall-time bar in Fig. 2(a). With CUDA graphs the GPU stream intervals are not a clean kernel/copy decomposition. For a publishable breakdown, correlate with Nsight Systems and Windows Performance Recorder (ETW) hard faults + disk I/O during prefill. Record the application's `prompt eval time` alongside these raw measurements.

Profile overhead is nonzero, especially during the initial graph-building steps and VRAM sampling. Compare against uninstrumented throughput and run repetitions. The experimental profiler assumes one active context; it is not suitable for concurrent llama-server slots. GPU compilation and Windows runtime have not yet been validated on the target machine.
