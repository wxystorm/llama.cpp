#!/usr/bin/env bash
# Phone-local, CPU mmap -> Adreno OpenCL on-demand weight staging.
# Usage: bash tools/phone-only-streaming/run_phone_opencl_stream.sh MODEL.gguf PROMPT.txt OUT_DIR
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: bash $0 MODEL.gguf PROMPT.txt OUT_DIR" >&2
    exit 2
fi

MODEL="$1"
PROMPT="$2"
OUT_DIR="$3"
LLAMA_BIN="${LLAMA_BIN:-./build/bin/llama-cli}"
N_CTX="${N_CTX:-2048}"
N_BATCH="${N_BATCH:-512}"
N_UBATCH="${N_UBATCH:-128}"
N_THREADS="${N_THREADS:-6}"
N_PREDICT="${N_PREDICT:-32}"
# This is a reservation budget for GGML's OpenCL compute/staging arena,
# NOT a global Adreno driver-memory ceiling.
GPU_BUDGET_MIB="${GPU_BUDGET_MIB:-1536}"
# This is a sampled process-RSS guard; file cache and the OpenCL driver
# can consume additional system-wide RAM beyond it.
RSS_LIMIT_MIB="${RSS_LIMIT_MIB:-7168}"

for limit in "$GPU_BUDGET_MIB" "$RSS_LIMIT_MIB"; do
    if ! [[ "$limit" =~ ^[1-9][0-9]*$ ]]; then
        echo "GPU_BUDGET_MIB and RSS_LIMIT_MIB must be positive integers" >&2
        exit 2
    fi
done

for file in "$MODEL" "$PROMPT" "$LLAMA_BIN"; do
    if [ ! -f "$file" ]; then
        echo "Missing file: $file" >&2
        exit 2
    fi
done
mkdir -p "$OUT_DIR"
export GGML_OPENCL_STREAM_WEIGHTS=1
export GGML_OPENCL_STREAM_GPU_BUDGET_MIB="$GPU_BUDGET_MIB"
export GGML_OPENCL_STREAM_RSS_LIMIT_MIB="$RSS_LIMIT_MIB"
export GGML_SCHED_DEBUG="${GGML_SCHED_DEBUG:-0}"

echo "[PHONE_STREAM_RUN] model=$MODEL prompt=$PROMPT out=$OUT_DIR"
echo "[PHONE_STREAM_RUN] ctx=$N_CTX batch=$N_BATCH ubatch=$N_UBATCH threads=$N_THREADS"
echo "[PHONE_STREAM_RUN] device=GPUOpenCL mmap=1 repack=0 fit=off mode=phone-only"
echo "[PHONE_STREAM_RUN] gpu_budget_mib=$GPU_BUDGET_MIB rss_limit_mib=$RSS_LIMIT_MIB"

# The first run is the only request. No synthetic warmup is included in timings.
# -ngl 0 is intentional: all model weights remain in file-backed CPU mmap.
# The opt-in scheduler places supported transformer matmuls on OpenCL and
# stages weights into transient GPU split-input buffers at execution time.
set +e
"$LLAMA_BIN" \
    -m "$MODEL" -f "$PROMPT" \
    -dev GPUOpenCL -fit off -ngl 0 \
    --mmap --no-repack --no-host \
    --no-warmup --no-kv-offload \
    -c "$N_CTX" -b "$N_BATCH" -ub "$N_UBATCH" -t "$N_THREADS" \
    -n "$N_PREDICT" \
    2>&1 | tee "$OUT_DIR/run.log"
status=${PIPESTATUS[0]}
set -e
grep -E 'PHONE_STREAM|prompt eval time|eval time|total time|model buffer size|compute buffer size' "$OUT_DIR/run.log" > "$OUT_DIR/metrics.log" || true
echo "[PHONE_STREAM_RUN] exit_code=$status metrics=$OUT_DIR/metrics.log"
exit "$status"
