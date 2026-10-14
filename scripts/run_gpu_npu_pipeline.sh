#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

BIN="${BIN:-./build/bin/test-gpu-npu-pipeline}"
BACKEND_DIR="${BACKEND_DIR:-$(dirname "$(realpath "$BIN")")}"
OUT="${OUT:-pipeline-results}"
CHUNKS="${CHUNKS:-8}"
KV="${KV:-512}"
GPU_BACKEND="${GPU_BACKEND:-GPUOpenCL}"
NPU_BACKEND="${NPU_BACKEND:-HTP0}"

BIN_ABS="$(realpath "$BIN")"
BACKEND_DIR="$(realpath "$BACKEND_DIR")"
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"

export LD_LIBRARY_PATH="$BACKEND_DIR:/vendor/lib64:$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_HEXAGON_HOSTBUF=1

echo "[PIPE_RUN] bin=$BIN_ABS"
echo "[PIPE_RUN] backend_dir=$BACKEND_DIR"
echo "[PIPE_RUN] chunks=$CHUNKS kv=$KV hostbuf=$GGML_HEXAGON_HOSTBUF"

(
    cd "$BACKEND_DIR"
    "$BIN_ABS" \
        --gpu "$GPU_BACKEND" \
        --npu "$NPU_BACKEND" \
        --chunks "$CHUNKS" \
        --kv "$KV"
) 2>&1 | tee "$OUT_ABS/pipeline.log"

echo
echo "Summary:"
grep -E '^\[PIPE_(CONFIG|BUFFER|WARMUP|SERIAL|2SLOT|RESULT|DONE|SUPPORT_FAIL|ERROR)\]'     "$OUT_ABS/pipeline.log" || true
