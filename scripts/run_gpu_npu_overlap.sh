#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

BIN="${BIN:-./build/bin/test-backend-ops}"
GPU_BACKEND="${GPU_BACKEND:-GPUOpenCL}"
NPU_BACKEND="${NPU_BACKEND:-HTP0}"
OUT="${OUT:-overlap-results}"

BIN_ABS="$(realpath "$BIN")"
BACKEND_DIR="${BACKEND_DIR:-$(dirname "$BIN_ABS")}"
BACKEND_DIR="$(realpath "$BACKEND_DIR")"
OUT_ABS="$(mkdir -p "$OUT" && cd "$OUT" && pwd)"

# Dynamic GGML backends are discovered from the executable directory and the
# process current working directory. LD_LIBRARY_PATH alone is not a backend
# discovery path. Run every probe from BACKEND_DIR so a split bin/lib layout
# works without copying plugins beside the executable.
export LD_LIBRARY_PATH="$BACKEND_DIR:/vendor/lib64:$PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_test() {
    (
        cd "$BACKEND_DIR"
        "$BIN_ABS" "$@"
    )
}

echo "[OVERLAP_PROBE] bin=$BIN_ABS"
echo "[OVERLAP_PROBE] backend_dir=$BACKEND_DIR"
echo "[OVERLAP_PROBE] ld_library_path=$LD_LIBRARY_PATH"
echo "[OVERLAP_PROBE] backend files:"
find "$BACKEND_DIR" -maxdepth 1 -type f -name 'libggml-*.so' -print | sort || true
echo

GPU_FA_RE='hsk=128.*hsv=128.*nh=4.*nr23=\[8,1\].*nb=128'
GPU_MM_RE='type_a=q4_K.*type_b=f32.*n=128.*k=2048'
NPU_FFN_RE='type_a=q4_0.*type_b=f32.*n_mats=128.*n_used=8.*m=768.*k=2048'

echo "[1/5] Probe backend names/support"
run_test support -b "$GPU_BACKEND" -o FLASH_ATTN_EXT -p "$GPU_FA_RE" > "$OUT_ABS/gpu-support.log" 2>&1 || true
run_test support -b "$NPU_BACKEND" -o MUL_MAT_ID -p "$NPU_FFN_RE" > "$OUT_ABS/npu-support.log" 2>&1 || true

echo "[2/5] GPU solo: Flash Attention"
run_test perf -b "$GPU_BACKEND" -o FLASH_ATTN_EXT -p "$GPU_FA_RE" --output csv     > "$OUT_ABS/gpu-fa-solo.csv" 2> "$OUT_ABS/gpu-fa-solo.err"

echo "[3/5] NPU solo: Qwen3-shaped MoE FFN primitive (Q4_0 HTP proxy)"
run_test perf -b "$NPU_BACKEND" -o MUL_MAT_ID -p "$NPU_FFN_RE" --output csv     > "$OUT_ABS/npu-ffn-solo.csv" 2> "$OUT_ABS/npu-ffn-solo.err"

echo "[4/5] Concurrent: GPU Flash Attention + NPU FFN"
# Give HTP initialization a head start. Each matched perf set runs for multiple
# seconds, so the timed compute windows overlap even if backend init latencies differ.
run_test perf -b "$NPU_BACKEND" -o MUL_MAT_ID -p "$NPU_FFN_RE" --output csv     > "$OUT_ABS/npu-ffn-concurrent.csv" 2> "$OUT_ABS/npu-ffn-concurrent.err" &
NPU_PID=$!
sleep 2
run_test perf -b "$GPU_BACKEND" -o FLASH_ATTN_EXT -p "$GPU_FA_RE" --output csv     > "$OUT_ABS/gpu-fa-concurrent.csv" 2> "$OUT_ABS/gpu-fa-concurrent.err" &
GPU_PID=$!

wait "$GPU_PID"
wait "$NPU_PID"

echo "[5/5] Concurrent: GPU projection/router weight streaming + NPU FFN"
run_test perf -b "$NPU_BACKEND" -o MUL_MAT_ID -p "$NPU_FFN_RE" --output csv     > "$OUT_ABS/npu-ffn-concurrent-mm.csv" 2> "$OUT_ABS/npu-ffn-concurrent-mm.err" &
NPU_PID=$!
sleep 2
run_test perf -b "$GPU_BACKEND" -o MUL_MAT -p "$GPU_MM_RE" --output csv     > "$OUT_ABS/gpu-mm-concurrent.csv" 2> "$OUT_ABS/gpu-mm-concurrent.err" &
GPU_PID=$!

wait "$GPU_PID"
wait "$NPU_PID"

echo
echo "Done. Results are in: $OUT_ABS"
echo "Compare time_us for identical op_params between:"
echo "  gpu-fa-solo.csv          vs gpu-fa-concurrent.csv"
echo "  npu-ffn-solo.csv         vs npu-ffn-concurrent.csv"
echo "  npu-ffn-solo.csv         vs npu-ffn-concurrent-mm.csv"
echo
echo "Interpretation:"
echo "  slowdown <= 10% : overlap looks promising"
echo "  slowdown 10-25% : usable, but pipeline depth/backpressure will matter"
echo "  slowdown > 25%  : shared-memory/interconnect contention is a serious risk"
