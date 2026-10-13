#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

BIN="${BIN:-./build/bin/test-backend-ops}"
GPU_BACKEND="${GPU_BACKEND:-GPUOpenCL}"
NPU_BACKEND="${NPU_BACKEND:-HTP0}"
OUT="${OUT:-overlap-results}"

mkdir -p "$OUT"

GPU_FA_RE='hsk=128.*hsv=128.*nh=4.*nr23=\[8,1\].*nb=128'
GPU_MM_RE='type_a=q4_K.*type_b=f32.*n=128.*k=2048'
NPU_FFN_RE='type_a=q4_K.*type_b=f32.*n_mats=128.*n_used=8.*m=768.*k=2048'

echo "[1/5] Probe backend names/support"
"$BIN" support -b "$GPU_BACKEND" -o FLASH_ATTN_EXT -p "$GPU_FA_RE" > "$OUT/gpu-support.log" 2>&1 || true
"$BIN" support -b "$NPU_BACKEND" -o MUL_MAT_ID -p "$NPU_FFN_RE" > "$OUT/npu-support.log" 2>&1 || true

echo "[2/5] GPU solo: Flash Attention"
"$BIN" perf -b "$GPU_BACKEND" -o FLASH_ATTN_EXT -p "$GPU_FA_RE" --output csv     > "$OUT/gpu-fa-solo.csv" 2> "$OUT/gpu-fa-solo.err"

echo "[3/5] NPU solo: Qwen3 MoE FFN primitive"
"$BIN" perf -b "$NPU_BACKEND" -o MUL_MAT_ID -p "$NPU_FFN_RE" --output csv     > "$OUT/npu-ffn-solo.csv" 2> "$OUT/npu-ffn-solo.err"

echo "[4/5] Concurrent: GPU Flash Attention + NPU FFN"
# Give HTP initialization a head start. Each matched perf set runs for multiple
# seconds, so the timed compute windows overlap even if backend init latencies differ.
"$BIN" perf -b "$NPU_BACKEND" -o MUL_MAT_ID -p "$NPU_FFN_RE" --output csv     > "$OUT/npu-ffn-concurrent.csv" 2> "$OUT/npu-ffn-concurrent.err" &
NPU_PID=$!
sleep 2
"$BIN" perf -b "$GPU_BACKEND" -o FLASH_ATTN_EXT -p "$GPU_FA_RE" --output csv     > "$OUT/gpu-fa-concurrent.csv" 2> "$OUT/gpu-fa-concurrent.err" &
GPU_PID=$!

wait "$GPU_PID"
wait "$NPU_PID"

echo "[5/5] Concurrent: GPU projection/router weight streaming + NPU FFN"
"$BIN" perf -b "$NPU_BACKEND" -o MUL_MAT_ID -p "$NPU_FFN_RE" --output csv     > "$OUT/npu-ffn-concurrent-mm.csv" 2> "$OUT/npu-ffn-concurrent-mm.err" &
NPU_PID=$!
sleep 2
"$BIN" perf -b "$GPU_BACKEND" -o MUL_MAT -p "$GPU_MM_RE" --output csv     > "$OUT/gpu-mm-concurrent.csv" 2> "$OUT/gpu-mm-concurrent.err" &
GPU_PID=$!

wait "$GPU_PID"
wait "$NPU_PID"

echo
echo "Done. Results are in: $OUT"
echo "Compare time_us for identical op_params between:"
echo "  gpu-fa-solo.csv          vs gpu-fa-concurrent.csv"
echo "  npu-ffn-solo.csv         vs npu-ffn-concurrent.csv"
echo "  npu-ffn-solo.csv         vs npu-ffn-concurrent-mm.csv"
echo
echo "Interpretation:"
echo "  slowdown <= 10% : overlap looks promising"
echo "  slowdown 10-25% : usable, but pipeline depth/backpressure will matter"
echo "  slowdown > 25%  : shared-memory/interconnect contention is a serious risk"
