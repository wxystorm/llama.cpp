# PRIMA ShareGPT40: interruptible PC + Termux benchmark

Target: PC rank 0 plus Termux phone rank 1, using the Qwen3 Dense port on
branch baseline/prima-qwen3-dense-20261107.

## One-time setup on both nodes

Pull the same commit on **both** PC and phone, then rebuild llama-cli.

PC (WSL + CUDA 11.5 and GCC 10):

    cd /mnt/d/prima-qwen3
    git pull --ff-only
    cd baselines/prima.cpp
    make USE_HIGHS=1 GGML_CUDA=1 CUDA_DOCKER_ARCH=compute_86 CC=gcc-10 CXX=g++-10 NVCCFLAGS="-ccbin=/usr/bin/g++-10" -j4 llama-cli

Phone (Termux; CPU backend):

    cd ~/prima-qwen3
    git pull --ff-only
    cd baselines/prima.cpp
    make CC=clang CXX=clang++ -j4 llama-cli

The phone needs the **same GGUF model** as rank 0. Verify model paths before
running. If the phone checkout was sparse, tools/sharegpt40 is inside the
baselines/prima.cpp cone and will be available after pull.

## Start the phone first

    cd ~/prima-qwen3/baselines/prima.cpp
    bash tools/sharegpt40/phone_worker_loop.sh \
      "$HOME/models/Qwen3-14B-Q4_K_M.gguf" \
      10.39.52.214 10.39.52.214

Use the PC address the phone can reach, rather than blindly using the example
if your network has changed. Keep the phone awake (Termux:API wake-lock may
help). Logs are in ~/prima-sharegpt40-phone-logs/.

The worker script supervises one CLI process per prompt: after the PC's STOP
handshake, it restarts rank 1 for the next prompt. It does NOT preserve a
single TCP/ZMQ connection across prompt cases. A 5-second gap on PC allows
time for this restart.

## Run all 40 prompts from PC

    cd /mnt/d/prima-qwen3/baselines/prima.cpp

    python3 tools/sharegpt40/run_sweep.py \
      --binary ./llama-cli \
      --model /mnt/e/llama/models/Qwen3-14B-Q4_K_M.gguf \
      --prompt-dir /mnt/e/llama/datasets/sharegpt/sharegpt40 \
      --master 10.39.52.214 \
      --next 10.39.52.114 \
      --out /mnt/d/prima-sharegpt40-results \
      --ctx 4096 --predict 32 --delay 5 \
      -- -b 512 -ub 512 -t 8

The 40 file names MUST be prompt_{128,256,512,1024}_{01..10}.txt. The
script checks they exist. Change the IPs, model paths, context and batch sizes
to suit your experiment. Do not add an extra -f/-p after the separator.

To verify commands without actually running inference, add --dry-run.

## Resume / failure behavior

Press Ctrl+C to interrupt the PC runner; the current case is marked
INTERRUPTED, previous successful cases remain PASS. The phone supervises
successive rank-1 worker processes until you stop its loop separately.

Run the **exact same PC command again** to resume. Cases are skipped only if:

1. Their case JSON says PASS;
2. Their benchmark command/model metadata signature is unchanged;
3. The previous process exited 0 AND reported a valid prompt-evaluation time.

Failed, missing, or interrupted cases are rerun. By default the sweep stops
on the first failure for diagnosis. Add --continue-on-error to attempt other
cases despite failures. Add --force to rerun all 40.

Outputs:

    /mnt/d/prima-sharegpt40-results/
      run_config.json
      summary.csv
      cases/prompt_128_01.json
      logs/prompt_128_01.log
      ...

The CSV contains measured prompt-eval tokens, prefill milliseconds, prefill
tok/s, decode metrics, planner k/W, PC/Phone layer assignment if logged,
process exit code, duration and log paths. No estimated token counts are
silently substituted for real measurements.

## Idle phone with 0 assigned layers

In the two-rank case, HALDA may eliminate a weak phone. Previously topology
reconstruction assigned NODE_TYPE_EXIT. The patched source instead marks
rank 1 as NODE_TYPE_FORWARDER and retains it in the two-rank communication
ring for that **single prompt**. Phone does **not** compute any transformer
layers while idle, but it may relay messages and remains in the STOP ring.

This is intentionally limited to two ranks. It is not a persistent session
server; the Termux supervisor handles re-listening between separate prompts.
The forwarding path has not yet been validated against a real two-device
0-layer allocation. Please run a smoke test before a 40-prompt sweep.

## Validation

    python3 -m unittest discover -s tools/sharegpt40 -p 'test_*.py' -v

Then start the phone and first try a single prompt with your known-good
command. The batch script does not perform logits-parity verification.
