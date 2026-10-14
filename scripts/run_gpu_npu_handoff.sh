#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

BIN="${BIN:-./build/bin/test-gpu-npu-handoff}"
BACKEND_DIR="${BACKEND_DIR:-$(dirname "$(realpath "$BIN")")}"
OUT="${OUT:-handoff-results}"
REPS="${REPS:-128}"
WARMUP="${WARMUP:-8}"
HIDDEN="${HIDDEN:-2048}"
GPU_BACKEND="${GPU_BACKEND:-GPUOpenCL}"
NPU_BACKEND="${NPU_BACKEND:-HTP0}"

BIN_ABS="$(realpath "$BIN")"
BACKEND_DIR="$(realpath "$BACKEND_DIR")"
mkdir -p "$OUT"
OUT_ABS="$(cd "$OUT" && pwd)"

export LD_LIBRARY_PATH="$BACKEND_DIR:/vendor/lib64:$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GGML_HEXAGON_HOSTBUF="${GGML_HEXAGON_HOSTBUF:-1}"

echo "[HANDOFF_RUN] bin=$BIN_ABS"
echo "[HANDOFF_RUN] backend_dir=$BACKEND_DIR"
echo "[HANDOFF_RUN] out=$OUT_ABS"
echo "[HANDOFF_RUN] GGML_HEXAGON_HOSTBUF=$GGML_HEXAGON_HOSTBUF"
echo "[HANDOFF_RUN] reps=$REPS warmup=$WARMUP hidden=$HIDDEN"

(
    cd "$BACKEND_DIR"
    "$BIN_ABS" \
        --gpu "$GPU_BACKEND" \
        --npu "$NPU_BACKEND" \
        --reps "$REPS" \
        --warmup "$WARMUP" \
        --hidden "$HIDDEN"
) 2>&1 | tee "$OUT_ABS/handoff.log"

echo
echo "Summary:"
grep -E '^\[HANDOFF_(CONFIG|BUFFER|VERIFY|SYNC|ASYNC_API|DONE)\]' "$OUT_ABS/handoff.log" || true
