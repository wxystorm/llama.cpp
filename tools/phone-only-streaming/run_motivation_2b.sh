#!/usr/bin/env bash
# Phone-only Motivation Fig. 2(b): weight stage versus HTP execution.
# Usage: bash run_motivation_2b.sh MODEL.gguf PROMPT.txt OUT_DIR
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: bash $0 MODEL.gguf PROMPT.txt OUT_DIR" >&2
    exit 2
fi

MODEL="$1"
PROMPT="$2"
OUT_DIR="$3"
LLAMA_BIN="${LLAMA_BIN:-$HOME/phone-stream-htp/bin/llama-cli}"
LAYER="${LAYER:-20}"
PRETOUCH="${PRETOUCH:-0}"
N_CTX="${N_CTX:-2048}"
N_BATCH="${N_BATCH:-512}"
N_UBATCH="${N_UBATCH:-128}"
N_THREADS="${N_THREADS:-6}"
N_PREDICT="${N_PREDICT:-1}"
DEVICE_BUDGET_MIB="${DEVICE_BUDGET_MIB:-768}"
RSS_LIMIT_MIB="${RSS_LIMIT_MIB:-7168}"

for f in "$MODEL" "$PROMPT" "$LLAMA_BIN"; do
    if [[ ! -f "$f" ]]; then
        echo "[PHONE_MOTI_ERROR] missing file: $f" >&2
        exit 2
    fi
done
if ! [[ "$LAYER" =~ ^[0-9]+$ && "$PRETOUCH" =~ ^[01]$ ]]; then
    echo "[PHONE_MOTI_ERROR] LAYER must be >=0 and PRETOUCH must be 0 or 1" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HOME/phone-stream-htp/lib:${LD_LIBRARY_PATH:-$PREFIX/lib}"
export ADSP_LIBRARY_PATH="${ADSP_LIBRARY_PATH:-$HOME/phone-stream-htp/lib}"
export GGML_HEXAGON_NDEV=1
export GGML_HEXAGON_HOSTBUF=0
export GGML_PHONE_MOTI_LAYER=1
export GGML_PHONE_MOTI_TARGET_LAYER="$LAYER"
export GGML_PHONE_MOTI_PRETOUCH="$PRETOUCH"
export GGML_PHONE_STREAM_DEVICE_BUDGET_MIB="$DEVICE_BUDGET_MIB"
export GGML_OPENCL_STREAM_RSS_LIMIT_MIB="$RSS_LIMIT_MIB"
export GGML_OPENCL_STREAM_TRACE_SPLITS=0

echo "[PHONE_MOTI_RUN] layer=$LAYER pretouch=$PRETOUCH model=$MODEL"
echo "[PHONE_MOTI_RUN] ctx=$N_CTX batch=$N_BATCH ubatch=$N_UBATCH threads=$N_THREADS"

set +e
"$LLAMA_BIN" -npu -m "$MODEL" -f "$PROMPT" \
    -c "$N_CTX" -b "$N_BATCH" -ub "$N_UBATCH" \
    -t "$N_THREADS" -n "$N_PREDICT" -lv 4 -st \
    </dev/null > "$OUT_DIR/run.log" 2>&1
status=$?
set -e

grep -E 'PHONE_MOTI|PHONE_STREAM|ggml-hex|prompt eval time|eval time|total time|REPACK|error|OOM|lmkd' \
    "$OUT_DIR/run.log" > "$OUT_DIR/metrics.log" || true
if command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/parse_motivation_2b.py" "$OUT_DIR/run.log" "$OUT_DIR" "$LAYER"
else
    echo "[PHONE_MOTI_RUN] python3 missing; inspect $OUT_DIR/metrics.log"
fi
echo "[PHONE_MOTI_RUN] exit_code=$status output=$OUT_DIR"
exit "$status"
