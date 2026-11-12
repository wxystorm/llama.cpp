# Jupiter HTP0-HTP3: cross-session placement

Status: implementation committed; awaiting cross-session Phone inference and
logits verification on the actual Snapdragon 8 Elite.

## What changed

One Phone RPC endpoint exposes HTP0, HTP1, HTP2 and HTP3. The PC loader
now partitions its Phone suffix into contiguous groups, assigning
weights, KV and graphs to the corresponding RPC device. The LM head
is charged to and placed on the last active session.

A minimax memory planner uses GGUF weight metadata with the REMOTE
Hexagon physical allocation size, not the smaller raw quantized GGUF
size. It budgets per-layer KV and a per-session workspace reserve.
An additional check accounts for actual allocated GGML weight buffers.
An infeasible layout is rejected before reading weights.

For cross-HTP activations, the Phone RPC process stages non-quantized
activations in <=1 MiB host chunks rather than sending them back to
the PC over TCP and uploading them again.

Jupiter q calibration keeps a single aggregate Phone stage, using
the sum of the four serial HTP session compute times plus internal
copy costs. The original Jupiter sequence-DP optimizer is unchanged.
Four sessions are four FastRPC contexts, not four physical NPUs.

## Settings

The ShareGPT40 runner accepts these options:

- --htp-sessions 4
- --htp-global-budget-mib 8192
- --htp-session-vmem-mib 3072
- --htp-reserve-mib 512
- --htp-kv-element-bytes 2

It sets the context-aware KV budget for every child calibration and
benchmark process. The default --htp-sessions 1 preserves the old
one-session comparison. Four sessions must be explicitly selected.

For four sessions, the nominal per-session weight+KV budget is:
min(3072,8192/4)-512 = 1536 MiB. This does NOT guarantee that
the driver can map or retain all those buffers; verify on hardware.

## Build / deploy

Windows PC checkout:

    cd E:\llama\llama-jupiter
    git pull --ff-only
    cmake --build .\build-jupiter --config Release --target llama-cli --parallel 8

WSL Phone build (same branch and commit):

    cd /mnt/e/llama/llama-jupiter
    cmake --build build-jupiter-htp-android --parallel 8
    cmake --install build-jupiter-htp-android --prefix "$PWD/pkg-jupiter-htp"
    mkdir -p pkg-jupiter-htp/tools/jupiter
    cp tools/jupiter/start_phone_htp.sh pkg-jupiter-htp/tools/jupiter/
    tar -czf jupiter-htp-android.tar.gz -C pkg-jupiter-htp bin lib tools

Redeploy all Phone binary/shared libraries (including libggml-rpc.so
and libggml-hexagon.so) and the launcher. Updating only the launcher
is NOT sufficient for the Phone-local activation-copy path.

Termux:

    cd ~/jupiter-htp
    export LLAMA_JUPITER_PHONE_BIN_ROOT="$PWD"
    export GGML_HEXAGON_NDEV=4
    export GGML_HEXAGON_VMEM=3072
    export GGML_HEXAGON_MBUF=1024
    export LLAMA_JUPITER_HTP_COPY_TRACE=1
    bash tools/jupiter/start_phone_htp.sh \
        "$HOME/models/Qwen3-14B-Q4_K_M.gguf" 10140 \
        2>&1 | tee jupiter-phone-htp4.log

## PC smoke run

Example from the E:\llama\llama-jupiter checkout, in PowerShell:

    python .\tools\jupiter\run_sharegpt40_dp.py --exe ".\build-jupiter\bin\Release\llama-cli.exe" --model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" --rpc "10.39.52.114:10140" --prompts-dir "E:\llama\datasets\sharegpt\sharegpt40" --groups 128 --limit 1 --pc-layers 32 --cpu-layers 16 --htp-sessions 4 --htp-global-budget-mib 8192 --htp-session-vmem-mib 3072 --htp-reserve-mib 512 --max-rounds 1 --budgeted-refine --out-dir ".\jupiter-htp4-smoke" --resume --only-incomplete --fail-fast

Look for these log markers:

- JUPITER_HTP_SESSION_PLAN status=READY active=4
- JUPITER_HTP_SESSION rank=0..3 status=FIT
- JUPITER_HTP_MEMORY rank=0..3 status=FIT
- JUPITER_HTP_RUNTIME_MEMORY actual model/KV/compute per Session status=FIT
- JUPITER_STAGE_LAYER stage=PHONE_HTP0/1/2/3
- JUPITER_STAGE_PLAN output_device=RPC3
- JUPITER_STAGE_PROFILE backend=RPC0/1/2/3 in calibration
- JUPITER_HTP_SESSION_COPY path=PHONE_HOST_STAGING in Phone log
- final PASS with Prefill token/s, and no numerical or buffer errors

If that passes, try 12 Phone layers with --pc-layers 28
--cpu-layers 12 (CPU12/CUDA16/Phone12), then 16 Phone layers
with --pc-layers 24 --cpu-layers 8 (CPU8/CUDA16/Phone16).
Use separate output directories. The partitioner fails closed.

For full ShareGPT40, remove --groups 128 --limit 1; retain
--resume --only-incomplete. Disable tracing for formal timing,
and independently check numerical parity.

## Tests

    cmake --build .\build-jupiter --config Release --target test-jupiter-htp-placement
    ctest --test-dir .\build-jupiter -C Release -R test-jupiter-htp-placement --output-on-failure
    cd tools\jupiter
    python -m unittest test_refine_sequence_q test_run_sharegpt40_dp -v

No hardware test has yet confirmed the full cross-session pipeline.
