#!/system/bin/sh
# Confirm HTP0..HTP3 run real quantized transformer matmuls on a phone.
# Usage: sh smoke_htp4_sessions.sh MODEL.gguf PROMPT.txt
set -u
if [ "$#" -ne 2 ]; then
    echo "Usage: sh $0 MODEL.gguf PROMPT.txt" >&2
    exit 2
fi
MODEL="$1"
PROMPT="$2"
LLAMA_BIN="${LLAMA_BIN:-$HOME/phone-stream-htp/bin/llama-cli}"
OUT="${OUT:-$HOME/log/htp4-smoke}"
for f in "$MODEL" "$PROMPT" "$LLAMA_BIN"; do
    if [ ! -f "$f" ]; then echo "[HTP4_SMOKE_ERROR] missing $f" >&2; exit 2; fi
done
mkdir -p "$OUT" || exit 2
APP_DIR="$(cd "$(dirname "$LLAMA_BIN")/.." && pwd)"
export LD_LIBRARY_PATH="$APP_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$APP_DIR/lib${ADSP_LIBRARY_PATH:+:$ADSP_LIBRARY_PATH}"
export GGML_HEXAGON_NDEV=4
export GGML_PHONE_STREAM_DEVICE_BUDGET_MIB="${HTP_BUDGET_MIB:-768}"
export GGML_OPENCL_STREAM_RSS_LIMIT_MIB="${RSS_LIMIT_MIB:-7168}"
export GGML_OPENCL_STREAM_TRACE_SPLITS=1

echo "[HTP4_SMOKE] model=$MODEL prompt=$PROMPT log=$OUT/htp4.log"
"$LLAMA_BIN" -npu -m "$MODEL" -f "$PROMPT" \
    -c "${N_CTX:-2048}" -b "${N_BATCH:-512}" -ub "${N_UBATCH:-128}" \
    -t "${N_THREADS:-6}" -n "${N_PREDICT:-1}" -lv 3 \
    > "$OUT/htp4.log" 2>&1
status=$?
grep -E 'PHONE_STREAM|HTP[0-3]|ggml-hex|prompt eval time|eval time|llama_perf' "$OUT/htp4.log" \
    > "$OUT/htp4.metrics.log" || true
if [ "$status" -ne 0 ]; then
    echo "[HTP4_SMOKE_FAIL] llama-cli exited $status"
    grep -Ei 'failed|error|abort|assert|htp' "$OUT/htp4.log" | tail -n 50 || true
    echo "[HTP4_SMOKE_FAIL] full log: $OUT/htp4.log"
    exit "$status"
fi
missing=0
for i in 0 1 2 3; do
    if ! grep -Eq "\[PHONE_STREAM_HTP\] device=HTP${i} splits=[0-9]+ accelerated_matmuls=[1-9][0-9]*" "$OUT/htp4.log"; then
        echo "[HTP4_SMOKE_FAIL] HTP$i did not execute a transformer matmul" >&2
        missing=$((missing + 1))
    fi
done
grep '\[PHONE_STREAM_HTP\]' "$OUT/htp4.log" | tail -n 8 || true
if [ "$missing" -ne 0 ]; then
    echo "[HTP4_SMOKE_FAIL] $missing sessions missing matmul evidence; see $OUT/htp4.log" >&2
    exit 1
fi
echo "[HTP4_SMOKE] PASS: HTP0-HTP3 each executed at least one quantized transformer matmul"
echo "[HTP4_SMOKE] This is a scheduling smoke test, not numerical parity or concurrent-execution proof."
