# Jupiter Qwen3-14B on Phone Hexagon/HTP — direct RPC prototype

**Status: experimental, not yet hardware validated.** Do not report the
results as a verified NPU baseline until all runtime and parity checks below
have passed on the actual PC + Snapdragon 8 Elite. This prototype deliberately
lives on `feat/jupiter-htp-direct-rpc-20261112`; the normal Jupiter GPU
benchmark branch remains unchanged.

## Scope

The original Jupiter scheduling policy is preserved:

- Jupiter model partition: PC prefix, Phone suffix (optional PC CPU prefix)
- Stage-3 measured q(x,y), chunk DP and final Prefill benchmark unchanged
- Qwen3-14B-Q4_K_M, one output token (`-n 1`), reference contexts:
  group 128 => 512, group 256 => 768, group 512 => 1280, group 1024 => 2304
- The target comparison uses PC boundary 32 / Phone suffix 8 (L32–L39)
- The Phone RPC server exposes **HTP0**, rather than GPUOpenCL

This is **whole-layer Phone NPU offload**, NOT the Region Ring autonomous
Phone-owned execution pipeline. It uses Jupiter's existing RPC Graph upload,
Graph Recompute and KV scheduling. No HTP/GPU intraphone fallback is added.

## Changes in the prototype

1. Ported eleven related Hexagon host + HTP-side files from
   `feat/region-ring-phone-npu-layer-serial-20261104`, including
   Q4_K/Q6_K matrix multiply, weight tiling/repacking and backend support.
2. In `ggml-rpc.cpp`, request the actual remote backend allocation size
   for Q4_K and Q6_K tensor buffers rather than blindly using the original
   `ggml_nbytes()`. The Phone backend may require padded/repacked weight
   storage.
3. Added `start_phone_htp.sh` to expose four FastRPC sessions by default,
   `HTP0,HTP1,HTP2,HTP3`, on a single RPC port (10140).
   Set `GGML_HEXAGON_NDEV=1` to return to the proven single-session path.
   `GGML_HEXAGON_MBUF=1024` MiB is a per-buffer limit and
   `GGML_HEXAGON_VMEM=3072` MiB is configured per session.

**Important:** The generic GGML allocator uses each backend's reported
max-buffer-size to split large weight contexts into multiple buffers.
This addresses the single-buffer *size* limit but does not prove that
the total weight+KV+graph working set fits the HTP address-space budget.

## Enabling four FastRPC sessions (HTP0..HTP3)

The Phone Hexagon backend already implements `GGML_HEXAGON_NDEV` and per-device
FastRPC session reservation. The RPC server accepts the comma-separated
`--device` list. Jupiter's Phone launcher now defaults to
`GGML_HEXAGON_NDEV=4` and passes
`--device HTP0,HTP1,HTP2,HTP3`; set `GGML_HEXAGON_NDEV=1` to run only
HTP0, or `2`/`3` for smaller increments.

Before retesting on the Phone, rebuild and redeploy `libggml-hexagon.so`
and associated Android backend libraries from the same branch. The Hexagon
registry now enumerates *only successfully opened sessions*, preventing a
null-device assertion when HTP1/2/3 creation fails. The explicit server
`--device` list still fails closed if any requested session is missing; it
will not silently drop to one HTP or CPU.

```bash
cd "$HOME/jupiter-htp"
export LLAMA_JUPITER_PHONE_BIN_ROOT="$PWD"
export GGML_HEXAGON_NDEV=4
bash tools/jupiter/start_phone_htp.sh \
    "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 10140 \
    2>&1 | tee jupiter-phone-htp4.log
```

Expected startup evidence (not an assertion of hardware success):

```
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 4
ggml-hex: HTP0 new session ...
ggml-hex: HTP1 new session ...
ggml-hex: HTP2 new session ...
ggml-hex: HTP3 new session ...
ggml-hex: HTP sessions opened=4 requested=4
Devices:
  HTP0: Hexagon ...
  HTP1: Hexagon ...
  HTP2: Hexagon ...
  HTP3: Hexagon ...
```

**Important distinction:** four sessions are four FastRPC handles/address
spaces onto the same SoC HTP, not four separate NPUs or 4 times the throughput.
A 3 GiB VMEM setting per session does not automatically mean 12 GiB of
physically usable memory. Actual combined mappings are limited by phone RAM,
driver quotas, live allocations and session overhead.

**The Jupiter PC model placement still selects the first RPC device
(`RPC0`) for all Phone suffix layers.** All four sessions are available
over RPC, but *opening them alone does not distribute model weights,
KV, or Graph execution across HTP1..3*. That requires a separate
session-aware layer partition and memory budget change. Keep the currently
passing 40-layer model layout and `--pc-layers/--cpu-layers` unchanged until
four-session initialization is validated. After that, add cross-session
weight placement and compare its correctness/performance.

## Hexagon SDK 6.6 CMake configure fix

For Android ARM64, the Qualcomm SDK's `hexagon_fun.cmake` expects
`PREBUILT_LIB_DIR=android_aarch64`. Otherwise lines 99-106 may print
`string sub-command FIND requires 3 or 4 parameters` during configuration.
This prototype branch now supplies that default in
`ggml/src/ggml-hexagon/CMakeLists.txt` before the SDK is included, while
respecting explicit overrides.

For an already-created CMake build directory, reconfigure:

```sh
cmake -S . -B build-jupiter-htp-android \
  -DPREBUILT_LIB_DIR=android_aarch64
cmake --build build-jupiter-htp-android --parallel 8
```

The early `HAVE_DOTPROD - Failed`, `HAVE_SVE - Failed`, etc. lines are
nonfatal optional CPU feature probes, not this SDK failure.

## Android/Phone setup

Build the Android Phone RPC server **from this prototype branch**, with both
`GGML_RPC=ON` and `GGML_HEXAGON=ON`. Unlike ordinary OpenCL builds,
the Hexagon backend also requires the Qualcomm Hexagon SDK
(`HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`) and the v79 HTP skel
`libggml-htp-v79.so`. Reuse the proven cross-compilation and DSP library
installation procedure from the Region Ring Phone NPU build; the ordinary
Termux-only CMake configuration is not necessarily sufficient.

The repository now includes `.gitattributes` rules forcing `*.sh` and
`*.py` to LF on checkout, including Windows/WSL worktrees with
`core.autocrlf=true`. This prevents the Termux errors
`set: pipefa: invalid option name` and `syntax error near unexpected token


```sh
cd "$HOME/jupiter-phone"
export LLAMA_JUPITER_PHONE_BIN_ROOT="$PWD"
export GGML_HEXAGON_NDEV=1
export LLAMA_JUPITER_TIMELINE=1   # diagnostic only; disable for formal timing
bash tools/jupiter/start_phone_htp.sh \
  "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 10140 \
  2>&1 | tee jupiter-phone-htp.log
```

On startup, **confirm** the device list contains `HTP0: Hexagon`. If the
server reports `No devices found`, do not silently use `GPUOpenCL`;
fix the Phone HTP build/vendor driver before measuring.

The RPC server is not authenticated. Bind only on a trusted private
network and never expose the port to the public Internet.

## PC: first single-prompt smoke test

Build PC Jupiter from the *same* prototype branch; RPC changes are compatible
only with the corresponding source-version assumptions, so avoid mixing
binaries from different branches.

The PC has RTX 4050 6 GiB: 32 **CUDA** resident layers may not fit. For
first validation, keep the **PC boundary at 32** but place 16 of those
layers on PC CPU, leaving PC CUDA 16 and Phone HTP8. This is a three-stage
functional smoke test, **not** a directly comparable PC32-CUDA result.

```powershell
python .\tools\jupiter\run_sharegpt40_dp.py `
  --exe ".\build-jupiter\bin\Release\llama-cli.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --rpc "10.39.52.114:10140" `
  --pc-layers 32 --cpu-layers 16 `
  --context-policy group `
  --prompts-dir "E:\llama\datasets\sharegpt\sharegpt40" `
  --limit 1 `
  --max-rounds 1 --budgeted-refine `
  --out-dir ".\jupiter-14b-htp-smoke" `
  --resume --only-incomplete --fail-fast
```

If that succeeds, increase `--limit` and inspect all group sizes. To
attempt the requested PC-CUDA32 / HTP8 two-stage comparison, pass
`--pc-layers 32 --cpu-layers 0` in **a separate output directory**,
but first confirm 32 CUDA layers plus compute buffers fit actual PC VRAM.
Jupiter does **not** use the Region Ring double-bank GPU weight streaming
path. Even successful HTP offload does not make PC32 CUDA memory fit.

## Q4_K packed-weight PHYSICAL_BUFFER_OOB on PC-to-Phone RPC

Observed with Qwen3-14B Q4_K_M and 8 Phone layers:

```
[JUPITER_HTP_WEIGHT] phase=REJECT reason=PHYSICAL_BUFFER_OOB
name='blk.34.attn_output.weight' type=q4_K
required=16384000 remaining=14771200
```

This is a **real allocation contract mismatch**, not a request to disable
the safety check. The Phone Hexagon repacker needs 16,384,000 bytes for that
weight; only 14,771,200 bytes remain at its remote address. The logical GGUF
size is 14,745,600 bytes. The original Jupiter **Windows client** queries
the Phone for some quantized tensors only if `ne[0] % 512 != 0`, so this
Q4_K tensor can silently be allocated using `ggml_nbytes()`. The HTP
prototype branch extends `ggml_backend_rpc_buffer_type_get_alloc_size()`
on **PC** to query Q4_K and Q6_K unconditionally for non-views.

**Important: rebuild the Windows PC `llama-cli.exe` from this HTP
prototype branch, not only the Android Phone binary.** The Phone's new
bounds check prevents a crash, but does not change the offsets/sizes
already assigned by an older Windows executable.

From Windows PowerShell in the Jupiter checkout:

```powershell
git fetch origin
git switch feat/jupiter-htp-direct-rpc-20261112
git pull --ff-only
cmake --build .\build-jupiter --config Release --target llama-cli --parallel 8
$env:GGML_RPC_HTP_ALLOC_TRACE = "blk.34.attn_output.weight"
# Start the previous single-prompt Jupiter smoke run with this PC environment.
```

The RPC client now logs `[JUPITER_RPC_ALLOC] side=CLIENT` with both
`logical_bytes=14745600` and `remote_alloc_bytes=16384000`
for that weight when its requested filter matches. If this marker is
**missing**, check whether the PC process really uses the rebuilt executable
and loads the matching `ggml-rpc` DLL. On Windows, ensure the Release
build's `bin/Release` executable and DLLs came from the same HTP branch.
The Phone backend can independently log `side=PHONE` using the same
`GGML_RPC_HTP_ALLOC_TRACE` variable after a Phone rebuild, but that is
not required for the **PC allocation fix**.

After a successful rerun, `Remove-Item Env:GGML_RPC_HTP_ALLOC_TRACE`
in PowerShell to disable the client diagnostics.

## FastRPC HTP0 open error 0x80000406

If startup prints `ggml-hex: failed to open session 0 : error 0x80000406`,
the problem occurs in `htp_iface_open()` **before** model loading, graph
setup, or any HTP weight upload. The subsequent `GGML_ASSERT(device)`
is only a consequence of backend initialization returning no HTP device.

The launcher now pins `ADSP_LIBRARY_PATH` to a **single absolute
directory**, `$LLAMA_JUPITER_PHONE_BIN_ROOT/lib`, matching the previously
working startup configuration. The earlier launcher concatenated directories
using colon notation, which is appropriate for `LD_LIBRARY_PATH` but not
reliably interpreted by Qualcomm's FastRPC DSP search logic. Replacing the
script alone requires no C++ rebuild.

Check the v79 skel and re-test:

```sh
cd "$HOME/jupiter-htp"
ls -lh lib/libggml-htp-v79.so
bash -n tools/jupiter/start_phone_htp.sh
bash tools/jupiter/start_phone_htp.sh \
    "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 10140
```

The log now prints `adsp_library_path=...`. If the error persists,
check `LD_LIBRARY_PATH` for inherited `/vendor/lib64` entries and compare
a clean Termux path `$PWD/lib:$PWD/bin:$PREFIX/lib` on a new run.
Some Android SoCs lose access to the vendor DSP HAL when the entire
vendor library directory is forced to the front of the linker path.
Capture full Android logs for DSP library loader errors such as
`dlerror`, `signature verify`, `libggml-htp-v79.so` and `adsprpc`.
The code `0x80000406` alone cannot tell whether the issue is a missing
skel, an incompatible DSP binary or a signing/loader problem.

## Debugging a Phone SIGSEGV during GGUF weight upload

Observed: the Phone `ggml-rpc-server` exits with 139 (SIGSEGV), during
`blk.34.attn_output.weight`, before the first Graph. The Windows client then
hits `failed to load tensor ... from local RPC model file` and exits by
`GGML_ABORT` (0xc0000409). The Windows status is downstream, not evidence
of a separate PC failure.

The Phone RPC loader originally validated only the *logical* GGUF
`ggml_nbytes()` tensor range. However Q4_K/Q6_K are repacked into larger
Hexagon tiling layouts, and an incorrectly sized remote allocation can
overflow the end of an HTP buffer. The prototype now checks the backend's
physical allocation size before writing, and optionally emits phase markers.
These checks diagnose or reject unsafe layouts; they do **not** establish the
original SIGSEGV's root cause without reproducing on hardware.

Rebuild/redeploy the **entire Android `bin/` and `lib/` outputs** from the updated
experimental branch (not only the RPC executable: the changed code may be in
`libggml-rpc.so` and `libggml-hexagon.so`). The PC binary need not be rebuilt
for these diagnostics, because the RPC wire format is unchanged. On Termux:

```bash
cd "$HOME/jupiter-htp"
export LLAMA_JUPITER_PHONE_BIN_ROOT="$PWD"
export GGML_RPC_HTP_WEIGHT_DIAG=1
export GGML_HEXAGON_DIAG_TENSOR=blk.34.attn_output.weight
set +e
bash tools/jupiter/start_phone_htp.sh \
    "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 10140 \
    2>&1 | tee jupiter-phone-htp-diagnostic.log
rc=${PIPESTATUS[0]}
echo "PHONE_EXIT_CODE=$rc"
```

Inspect the last diagnostics:

```bash
grep -E 'JUPITER_HTP_WEIGHT|JUPITER_HEX_REPACK|PHYSICAL_BUFFER_OOB' \
    jupiter-phone-htp-diagnostic.log | tail -n 80
```

Interpretation:

- `phase=REJECT reason=PHYSICAL_BUFFER_OOB`: a real mismatch between
  remote packed weight allocation size and its containing HTP buffer.
- `phase=READ_BEGIN` without `READ_DONE`: crash inside the local GGUF read.
- `phase=SET_BEGIN` without `SET_DONE`: crash inside the backend repacker
  or mapped weight buffer. `JUPITER_HEX_REPACK` progress shows whether it
  reached the zeroing step or a particular matrix row.
- `SET_DONE` then SIGSEGV: inspect the next tensor and buffer metadata;
  a previous overwrite remains possible.

Also capture Android's native crash report immediately after the event:

```powershell
adb logcat -b crash -d -v threadtime
```

Look for the `ggml-rpc-server` faulting PC/frame, `SEGV_MAPERR` or
`SEGV_ACCERR`. Symbolization with the matching **unstripped** ARM64 build
libraries/binary is needed to distinguish an HTP repack write from a driver
mapping or `memcpy` failure. SIGSEGV (139) is not equivalent to the usual
OOM SIGKILL (137), even if memory pressure remains a possible contributing
factor.

When diagnostics finish, `unset GGML_RPC_HTP_WEIGHT_DIAG` and
`unset GGML_HEXAGON_DIAG_TENSOR` to restore normal benchmark behavior.

## Required checks before claiming an NPU result

1. Phone startup log shows `HTP0` (never `GPUOpenCL`).
2. 14B model loads with no unsupported quant tensor or repack failures.
3. Jupiter Remote Graph registrations/recomputes finish; no unsupported
   GGML op, invalid KV, allocator or transport failure.
4. Compare the generated token/logits with a known-good GPU Jupiter
   baseline on the **same prompt, quantization, and context**. Exact bitwise
   agreement is not required for different hardware, but discrepancies
   must be investigated.
5. Run both short- and long-context prompts. Confirm stable memory use
   and `-n 1` for all cases.
6. Only then run the complete 40-prompt benchmark and compare Prefill
   tok/s / latency / graph service times under matched PC placements.
   Disable timeline/trace for formal timings.

## Known limitations / next engineering work

- No Phone HTP hardware test or full Android rebuild has been performed
  while preparing this branch. Kernel build integration and correctness
  remain unverified.
- RPC currently advertises all remote ops as supported. If HTP rejects
  a Qwen3 op, the RPC Graph may fail rather than gracefully route it
  to Phone GPU. A backend-capability handshake and fail-closed validation
  are needed before considering the port production-ready.
- Hexagon backend currently reports 0 device free/total bytes through
  `get_memory()`. The fixed Jupiter layer cut does not itself rely on
  that estimate, but Quick8 auto-memory planning requires a proper
  HTP-specific budget (or explicit `--phone-free-mib`).
- This is direct HTP0 RPC, not the Region Ring private HTP serial executor:
  its weight residency, graph builder, and numerical behavior are
  potentially different.
- A full 8-layer Phone HTP suffix must be checked against the ~3 GiB
  VMEM budget, **including** weights, graph buffers and KV, not only
  per-buffer limits. If it fails, reduce Phone layers or implement a
  planned CPU/GPU/HTP spill/streaming design as separate work.
in\\r'` caused by a CRLF/corrupted copied script.

After pulling the branch, refresh the launcher from Git and validate it
**before creating the tarball**:

```sh
git pull --ff-only
git restore --source=HEAD -- tools/jupiter/start_phone_htp.sh
bash -n tools/jupiter/start_phone_htp.sh
python3 tools/jupiter/test_start_phone_htp.py
file tools/jupiter/start_phone_htp.sh
```

The `bash -n` check must print nothing and return 0; `file` should identify
ASCII/UTF-8 text with LF line terminators, not CRLF. Package from WSL after
this check rather than re-copying the script through a Windows text editor.

Copy `start_phone_htp.sh` alongside the Phone build, or invoke it
from the source tree. With the executable at `$HOME/jupiter-phone/bin/`:

```sh
cd "$HOME/jupiter-phone"
export LLAMA_JUPITER_PHONE_BIN_ROOT="$PWD"
export GGML_HEXAGON_NDEV=1
export LLAMA_JUPITER_TIMELINE=1   # diagnostic only; disable for formal timing
bash tools/jupiter/start_phone_htp.sh \
  "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 10140 \
  2>&1 | tee jupiter-phone-htp.log
```

On startup, **confirm** the device list contains `HTP0: Hexagon`. If the
server reports `No devices found`, do not silently use `GPUOpenCL`;
fix the Phone HTP build/vendor driver before measuring.

The RPC server is not authenticated. Bind only on a trusted private
network and never expose the port to the public Internet.

## PC: first single-prompt smoke test

Build PC Jupiter from the *same* prototype branch; RPC changes are compatible
only with the corresponding source-version assumptions, so avoid mixing
binaries from different branches.

The PC has RTX 4050 6 GiB: 32 **CUDA** resident layers may not fit. For
first validation, keep the **PC boundary at 32** but place 16 of those
layers on PC CPU, leaving PC CUDA 16 and Phone HTP8. This is a three-stage
functional smoke test, **not** a directly comparable PC32-CUDA result.

```powershell
python .\tools\jupiter\run_sharegpt40_dp.py `
  --exe ".\build-jupiter\bin\Release\llama-cli.exe" `
  --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" `
  --rpc "10.39.52.114:10140" `
  --pc-layers 32 --cpu-layers 16 `
  --context-policy group `
  --prompts-dir "E:\llama\datasets\sharegpt\sharegpt40" `
  --limit 1 `
  --max-rounds 1 --budgeted-refine `
  --out-dir ".\jupiter-14b-htp-smoke" `
  --resume --only-incomplete --fail-fast
```

If that succeeds, increase `--limit` and inspect all group sizes. To
attempt the requested PC-CUDA32 / HTP8 two-stage comparison, pass
`--pc-layers 32 --cpu-layers 0` in **a separate output directory**,
but first confirm 32 CUDA layers plus compute buffers fit actual PC VRAM.
Jupiter does **not** use the Region Ring double-bank GPU weight streaming
path. Even successful HTP offload does not make PC32 CUDA memory fit.

## Debugging a Phone SIGSEGV during GGUF weight upload

Observed: the Phone `ggml-rpc-server` exits with 139 (SIGSEGV), during
`blk.34.attn_output.weight`, before the first Graph. The Windows client then
hits `failed to load tensor ... from local RPC model file` and exits by
`GGML_ABORT` (0xc0000409). The Windows status is downstream, not evidence
of a separate PC failure.

The Phone RPC loader originally validated only the *logical* GGUF
`ggml_nbytes()` tensor range. However Q4_K/Q6_K are repacked into larger
Hexagon tiling layouts, and an incorrectly sized remote allocation can
overflow the end of an HTP buffer. The prototype now checks the backend's
physical allocation size before writing, and optionally emits phase markers.
These checks diagnose or reject unsafe layouts; they do **not** establish the
original SIGSEGV's root cause without reproducing on hardware.

Rebuild/redeploy the **entire Android `bin/` and `lib/` outputs** from the updated
experimental branch (not only the RPC executable: the changed code may be in
`libggml-rpc.so` and `libggml-hexagon.so`). The PC binary need not be rebuilt
for these diagnostics, because the RPC wire format is unchanged. On Termux:

```bash
cd "$HOME/jupiter-htp"
export LLAMA_JUPITER_PHONE_BIN_ROOT="$PWD"
export GGML_RPC_HTP_WEIGHT_DIAG=1
export GGML_HEXAGON_DIAG_TENSOR=blk.34.attn_output.weight
set +e
bash tools/jupiter/start_phone_htp.sh \
    "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 10140 \
    2>&1 | tee jupiter-phone-htp-diagnostic.log
rc=${PIPESTATUS[0]}
echo "PHONE_EXIT_CODE=$rc"
```

Inspect the last diagnostics:

```bash
grep -E 'JUPITER_HTP_WEIGHT|JUPITER_HEX_REPACK|PHYSICAL_BUFFER_OOB' \
    jupiter-phone-htp-diagnostic.log | tail -n 80
```

Interpretation:

- `phase=REJECT reason=PHYSICAL_BUFFER_OOB`: a real mismatch between
  remote packed weight allocation size and its containing HTP buffer.
- `phase=READ_BEGIN` without `READ_DONE`: crash inside the local GGUF read.
- `phase=SET_BEGIN` without `SET_DONE`: crash inside the backend repacker
  or mapped weight buffer. `JUPITER_HEX_REPACK` progress shows whether it
  reached the zeroing step or a particular matrix row.
- `SET_DONE` then SIGSEGV: inspect the next tensor and buffer metadata;
  a previous overwrite remains possible.

Also capture Android's native crash report immediately after the event:

```powershell
adb logcat -b crash -d -v threadtime
```

Look for the `ggml-rpc-server` faulting PC/frame, `SEGV_MAPERR` or
`SEGV_ACCERR`. Symbolization with the matching **unstripped** ARM64 build
libraries/binary is needed to distinguish an HTP repack write from a driver
mapping or `memcpy` failure. SIGSEGV (139) is not equivalent to the usual
OOM SIGKILL (137), even if memory pressure remains a possible contributing
factor.

When diagnostics finish, `unset GGML_RPC_HTP_WEIGHT_DIAG` and
`unset GGML_HEXAGON_DIAG_TENSOR` to restore normal benchmark behavior.

## Required checks before claiming an NPU result

1. Phone startup log shows `HTP0` (never `GPUOpenCL`).
2. 14B model loads with no unsupported quant tensor or repack failures.
3. Jupiter Remote Graph registrations/recomputes finish; no unsupported
   GGML op, invalid KV, allocator or transport failure.
4. Compare the generated token/logits with a known-good GPU Jupiter
   baseline on the **same prompt, quantization, and context**. Exact bitwise
   agreement is not required for different hardware, but discrepancies
   must be investigated.
5. Run both short- and long-context prompts. Confirm stable memory use
   and `-n 1` for all cases.
6. Only then run the complete 40-prompt benchmark and compare Prefill
   tok/s / latency / graph service times under matched PC placements.
   Disable timeline/trace for formal timings.

## Known limitations / next engineering work

- No Phone HTP hardware test or full Android rebuild has been performed
  while preparing this branch. Kernel build integration and correctness
  remain unverified.
- RPC currently advertises all remote ops as supported. If HTP rejects
  a Qwen3 op, the RPC Graph may fail rather than gracefully route it
  to Phone GPU. A backend-capability handshake and fail-closed validation
  are needed before considering the port production-ready.
- Hexagon backend currently reports 0 device free/total bytes through
  `get_memory()`. The fixed Jupiter layer cut does not itself rely on
  that estimate, but Quick8 auto-memory planning requires a proper
  HTP-specific budget (or explicit `--phone-free-mib`).
- This is direct HTP0 RPC, not the Region Ring private HTP serial executor:
  its weight residency, graph builder, and numerical behavior are
  potentially different.
- A full 8-layer Phone HTP suffix must be checked against the ~3 GiB
  VMEM budget, **including** weights, graph buffers and KV, not only
  per-buffer limits. If it fails, reduce Phone layers or implement a
  planned CPU/GPU/HTP spill/streaming design as separate work.
