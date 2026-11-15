#!/usr/bin/env bash
# Simple phone-local, file-backed streaming benchmark for CPU, Adreno or HTP.
# Usage: bash run_phone_stream.sh -cpu|-opencl|-npu MODEL.gguf PROMPT.txt OUT_DIR
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "Usage: bash $0 -cpu|-opencl|-npu MODEL.gguf PROMPT.txt OUT_DIR" >&2
    exit 2
fi
MODE="$1"
MODEL="$2"
PROMPT="$3"
OUT_DIR="$4"
case "$MODE" in
    -cpu|-opencl|-npu) ;;
    *) echo "Invalid backend: $MODE; use -cpu, -opencl or -npu" >&2; exit 2 ;;
esac

LLAMA_BIN="${LLAMA_BIN:-./build/bin/llama-cli}"
N_CTX="${N_CTX:-2048}"
N_BATCH="${N_BATCH:-512}"
N_UBATCH="${N_UBATCH:-128}"
N_THREADS="${N_THREADS:-6}"
N_PREDICT="${N_PREDICT:-1}"
RSS_LIMIT_MIB="${RSS_LIMIT_MIB:-7168}"
if [[ "$MODE" == "-npu" ]]; then
    DEVICE_BUDGET_MIB="${DEVICE_BUDGET_MIB:-768}"
else
    DEVICE_BUDGET_MIB="${DEVICE_BUDGET_MIB:-1536}"
fi

for value in "$RSS_LIMIT_MIB" "$DEVICE_BUDGET_MIB"; do
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "RSS_LIMIT_MIB and DEVICE_BUDGET_MIB must be positive integers" >&2
        exit 2
    fi
done
for file in "$MODEL" "$PROMPT" "$LLAMA_BIN"; do
    if [[ ! -f "$file" ]]; then
        echo "File not found: $file" >&2
        exit 2
    fi
done
mkdir -p "$OUT_DIR"

# The CLI flag determines the backend; the environment below only supplies
# budgets and metrics. There is no PC/RPC/HTP use in CPU and OpenCL modes.
export GGML_PHONE_STREAM_DEVICE_BUDGET_MIB="$DEVICE_BUDGET_MIB"
export GGML_OPENCL_STREAM_RSS_LIMIT_MIB="$RSS_LIMIT_MIB"
export GGML_OPENCL_STREAM_TRACE_SPLITS="${TRACE_SPLITS:-0}"

echo "[PHONE_STREAM_RUN] backend=$MODE model=$MODEL prompt=$PROMPT"
echo "[PHONE_STREAM_RUN] ctx=$N_CTX batch=$N_BATCH ubatch=$N_UBATCH predict=$N_PREDICT"
echo "[PHONE_STREAM_RUN] device_budget_mib=$DEVICE_BUDGET_MIB rss_limit_mib=$RSS_LIMIT_MIB"
echo "[PHONE_STREAM_RUN] npu_requires_native_supported_quant=1"

set +e
"$LLAMA_BIN" \
    "$MODE" \
    -m "$MODEL" -f "$PROMPT" \
    -c "$N_CTX" -b "$N_BATCH" -ub "$N_UBATCH" -t "$N_THREADS" \
    -n "$N_PREDICT" \
    2>&1 | tee "$OUT_DIR/run.log"
status=${PIPESTATUS[0]}
set -e
grep -E 'PHONE_STREAM|prompt eval time|eval time|total time|model buffer size|compute buffer size|llama_perf' "$OUT_DIR/run.log" > "$OUT_DIR/metrics.log" || true
echo "[PHONE_STREAM_RUN] exit_code=$status metrics=$OUT_DIR/metrics.log"
exit "$status"
