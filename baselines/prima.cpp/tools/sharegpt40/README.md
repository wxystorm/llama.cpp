# PRIMA ShareGPT40 / ShareGPT50: resumable PC + Termux benchmark

PRIMA rank 0 runs on the WSL PC; rank 1 runs on Android Termux.
The existing `tools/sharegpt40/` directory retains its name for backwards
compatibility. The sweep now supports both datasets automatically:

| Dataset | Prompt groups (ten files per group) | Expected files |
| --- | --- | --- |
| ShareGPT40 | 128, 256, 512, 1024 | 40 |
| ShareGPT50 | 0, 128, 256, 512, 1024 | 50 |

ShareGPT50's `prompt_0_XX.txt` files represent the shortest group (64-128
tokens); the group label in the filenames is simply `0`. Each group must
contain all ten files, with names `prompt_<group>_01.txt` through
`prompt_<group>_10.txt`. Missing, partial, or unexpected prompt files cause
an error instead of silently measuring the wrong dataset. The code discovers
the groups from `--prompt-dir`, so no `--per-group` or `--groups` flag is
necessary.

## Setup on both nodes

Pull this branch on both machines. After source-code changes, rebuild
`llama-cli`; for this sweep-script-only update, PC `git pull` is sufficient.

PC (WSL + CUDA 11.5 and GCC 10, when rebuilding):

    cd /mnt/d/prima-qwen3
    git pull --ff-only
    cd baselines/prima.cpp
    make USE_HIGHS=1 GGML_CUDA=1 CUDA_DOCKER_ARCH=compute_86 CC=gcc-10 CXX=g++-10 NVCCFLAGS="-ccbin=/usr/bin/g++-10" -j4 llama-cli

Phone (Termux, only needed to update/rebuild source):

    cd ~/prima-qwen3
    git pull --ff-only
    cd baselines/prima.cpp
    make CC=clang CXX=clang++ -j4 llama-cli

Both ranks must use the same GGUF model. The following IPs are examples
from one experiment: check them before starting. If WSL2 is using NAT,
the Windows-to-WSL portproxy for rank 0's ports 9000 and 10000 must also
be configured.

## Start the phone rank 1 first

    cd ~/prima-qwen3/baselines/prima.cpp

    bash tools/sharegpt40/phone_worker_loop.sh \
      "$HOME/models/Qwen3-32B-Q4_K_M.gguf" \
      10.238.229.214

Keep the phone loop running. It supervises a separate rank-1 CLI process
for each prompt, since the PC sends STOP after each session. This is not
a single persistent ZMQ connection. Logs are written under
`~/prima-sharegpt40-phone-logs/`. The script is **unchanged** for the
50-prompt dataset; it does not care how many prompts the PC will run.

## Run 50 prompts on PC WSL

Use a **new output directory** for ShareGPT50 so any older ShareGPT40
results and logs are preserved.

    cd /mnt/d/prima-qwen3/baselines/prima.cpp

    python3 tools/sharegpt40/run_sweep.py \
      --binary ./llama-cli \
      --model /mnt/e/llama/models/Qwen3-32B-Q4_K_M.gguf \
      --prompt-dir /mnt/e/llama/datasets/sharegpt/sharegpt50_short \
      --out /mnt/d/prima-sharegpt50-results-32b \
      --master 10.238.229.214 \
      --next 10.238.229.239 \
      --ctx 2048 --predict 32 \
      --delay 30 --timeout 3600 \
      -- -b 512 -ub 512 -t 8

For the first run, add `--max-new 1` before `--` to test
`prompt_0_01.txt` before launching all 50. After it passes, remove
`--max-new 1` and repeat the command to resume.

For the old ShareGPT40, pass its old `--prompt-dir` and
`--out` and the same script will run 40 instead. A quick dry-run,
which prints commands without executing them, is available with
`--dry-run --max-new 1`.

## Results, resume, and failures

Successful cases are skipped on reruns with the same model, binary,
dataset, prompt file metadata, context, and CLI parameters. If the prompt
directory or files are modified, the run signature changes; previous PASS
records are not silently reused. Existing case JSON and logs remain in
the output folder until the corresponding case is rerun, so use distinct
output folders for distinct benchmark datasets.

Use Ctrl+C to interrupt. A running case becomes INTERRUPTED, and running
the same command again resumes the missing/failed/interrupted cases.
Without `--continue-on-error`, the script stops at the first failure.
`--force` explicitly reruns every case.

Output files:

    /mnt/d/prima-sharegpt50-results-32b/
      run_config.json
      summary.csv
      cases/prompt_0_01.json
      cases/prompt_128_01.json
      logs/prompt_0_01.log
      ...

The CSV records exact prompt-evaluation tokens, prefill milliseconds/tok/s,
decode timing, planner k/W and PC/phone allocation when present, exit code,
wall clock and log paths. It only includes cases matching the current run
signature, to avoid mixing old and new datasets.

A failure in under a second (for example `EXIT_1`) is a separate
`llama-cli` initialization/connection issue, not evidence that the
40-versus-50 prompt enumeration was responsible. Inspect the PC log and
the phone's latest rank-1 log before diagnosing or changing delays.

    tail -n 120 /mnt/d/prima-sharegpt50-results-32b/logs/prompt_0_01.log

## Idle phone with 0 allocated layers

The two-rank topology patch keeps rank 1 as a zero-compute forwarder
inside that prompt session if HALDA allocates it no layers. The phone
supervisor recreates the next CLI connection after STOP. This path has
not been validated against every real two-device zero-layer plan.

## Validation

    python3 -m unittest discover -s tools/sharegpt40 -p 'test_*.py' -v

The tests cover ShareGPT40, ShareGPT50, missing first-group files,
resume behavior, and invalidating stale results after prompt edits.
They do not test real-device throughput or numerical parity.
