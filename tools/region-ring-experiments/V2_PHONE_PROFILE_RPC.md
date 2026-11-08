# V2 Profile orchestration: G/C on PC, N/A on Phone

The V2 first-N real-GGUF Stage execution code is shared across CPU,
CUDA, HTP and Adreno OpenCL. This phase adds an explicit Profile control
channel so the PC can request N/A samples directly, receive real timing
records automatically and cache them with device identities. It does
NOT require copying CSV/log files off the Phone after each run.

**This is a dedicated V2 Profile sidecar on port 50177. It is NOT a
new opcode on the existing ggml-rpc inference port 50052.** This keeps
experimental benchmark subprocesses outside the legacy inference
scheduler. Production RPC integration is a separate cutover phase.

## 1. Prepare Phone (once after changing the native executable)

Cross-compile the target in the existing Snapdragon Android build:

    cmake --build build-phone-npu --target region-ring-v2-stage-profile -j 8

Deploy the Phone binary and compatible GGML Hexagon/OpenCL libraries
using the existing phone-v2-profile.tar.gz deployment method.

The Phone needs Python in Termux. If missing:

    pkg install python

From Windows PowerShell, copy the two small control-plane scripts:

    adb push .\tools\region-ring-experiments\v2_profile_wire.py /sdcard/Download/
    adb push .\tools\region-ring-experiments\v2_phone_profile_server.py /sdcard/Download/

In the Termux shell:

    cd ~/phone-v2-profile
    cp /sdcard/Download/v2_profile_wire.py .
    cp /sdcard/Download/v2_phone_profile_server.py .
    export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
    export ADSP_LIBRARY_PATH="$PWD/lib"
    export GGML_HEXAGON_NDEV=1
    python v2_phone_profile_server.py \
      --model "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
      --exe "$PWD/bin/region-ring-v2-stage-profile" \
      --bind 127.0.0.1 --port 50177

Keep this Termux process running during sampling. It reads and hashes
the actual original GGUF ONCE, fingerprints the native executable,
and exposes a fixed allowlist: first-N Qwen3 Stage Profile N/A, NOT
arbitrary shell commands. The server chooses its own executable and
model path; the PC sends only numeric workload parameters.

## 2. Connect from PC using USB (recommended)

After confirming the Phone server prints status=READY:

    adb forward tcp:50177 tcp:50177

This forwards 127.0.0.1:50177 on the PC to the Phone loopback port
over USB. No need to open the Phone to the LAN or assign a fixed phone
IP. It works alongside the existing inference RPC server.

For one smoke-test combination on all four backends (single PowerShell
command):

    python .\tools\region-ring-experiments\run_v2_stage_matrix.py --exe ".\build-tensor-v2\bin\Release\region-ring-v2-stage-profile.exe" --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" --out ".\profile-v2\gcna" --phone "127.0.0.1:50177" --backends GCNA --layers "2,4" --chunks "64" --kvs "512" --warmups 1 --repeats 3

For larger sweeps, vary layers/chunks/kvs and rerun. Existing PASS
cases with the same PC/Phone device fingerprint, original GGUF SHA256,
executable SHA256, parameters and thresholds are resumed. Add
--update to force resampling. The PC validates that its GGUF bytes
match the Phone GGUF exactly; a mismatch is a hard error.

The entire sweep does not abort on one failed case. The process exits
with a nonzero code when at least one case fails numerical parity or
execution. Inspect outputs regardless of that final exit code.

## 3. Output contract

- stage-profile-v2.csv: **only** PASS cases. N samples must have
  independently passed the native CPU reference check; N4 numerical
  failure cannot contaminate this file.
- stage-profile-v2-parity-fail.csv: actual compute_ms and accuracy
  measurements for native HTP/CUDA/OpenCL executions that completed
  but failed CPU comparison. These are diagnostic evidence, NOT inputs
  to Planner.
- G-L4-X64-KV512.json, C-L4-X64-KV512.json,
  N-L4-X64-KV512.json, A-L4-X64-KV512.json: full per-case records,
  keyed with model and executable hash and machine identity.
  Failed executions create a .failed.json diagnostic as well.
- Each record marks the transport as LOCAL_NATIVE or PHONE_RPC_V1,
  and only publishes measured rows with native provenance.

Phone N4 may currently execute successfully but return
PARITY_FAIL at X64/KV512; the PC still saves its real compute timing
without declaring that N4 is numerically correct. The first-N NPU
Graph is not a fake extrapolation from N1.

## 4. Optional trusted LAN transport

To avoid adb forward, start Phone server on its LAN interface:

    export V2_PROFILE_TOKEN="choose-a-random-string-of-16-or-more-chars"
    python v2_phone_profile_server.py \
      --model "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
      --exe "$PWD/bin/region-ring-v2-stage-profile" \
      --bind 0.0.0.0 --port 50177

Set the identical V2_PROFILE_TOKEN on the PC and pass
--phone PHONE_IP:50177. LAN token is sent in plaintext, so use
a trusted/private network only. For untrusted networks, prefer USB
adb forward rather than exposing a Profile service.

## 5. Scope limitations

- The Profile server is a separate long-lived Python process and
  launches the native executable for each N/A sample. Graph build,
  weight upload, HTP precompile and actual synchronized compute have
  SEPARATE timing fields; the new subprocess startup and transfer
  are NOT included in compute_median_ms.
- The server serializes all requests to avoid competing Phone Stage
  GPU/NPU workloads, but the concurrent legacy inference server
  should be left idle during benchmarking.
- Initial implementation accepts Qwen3 Dense and K >= X with K % X=0,
  up to 32 layers in the current shared Builder.
- NPU N4 numerical accuracy is not solved by this transport; its
  failed cases are quarantined. Planner must eventually consider
  only correctness-qualified profiles, plus separately measured
  Bank refill/transfer costs and memory feasibility.
- This step does not yet merge the N/A request into the production
  ggml-rpc message enum, dynamically drive the production Phone
  two-Bank executor, implement transfer Profile, or select a layout.
