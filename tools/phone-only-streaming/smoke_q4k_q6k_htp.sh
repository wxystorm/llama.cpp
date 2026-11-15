#!/usr/bin/env bash
# On-device validation of the phone-only Q4_K/Q6_K Host+HTP port.
# Usage: LLAMA_BIN=./build/bin/llama-cli bash smoke_q4k_q6k_htp.sh Q4K.gguf Q6K.gguf prompt.txt output-dir
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "Usage: bash $0 Q4K.gguf Q6K.gguf prompt.txt output-dir" >&2
    exit 2
fi
Q4K_MODEL="$1"
Q6K_MODEL="$2"
PROMPT="$3"
OUT_ROOT="$4"

RUNNER="$(dirname "$0")/run_phone_stream.sh"
mkdir -p "$OUT_ROOT"
export N_PREDICT="${N_PREDICT:-1}"
export N_UBATCH="${N_UBATCH:-128}"
export N_BATCH="${N_BATCH:-512}"
export N_CTX="${N_CTX:-2048}"
export TRACE_SPLITS="${TRACE_SPLITS:-1}"

for entry in "q4k:$Q4K_MODEL" "q6k:$Q6K_MODEL"; do
    kind="${entry%%:*}"
    model="${entry#*:}"
    for mode in cpu npu; do
        out="$OUT_ROOT/$kind-$mode"
        echo "[QK_SMOKE] model=$kind backend=$mode out=$out"
        bash "$RUNNER" "-$mode" "$model" "$PROMPT" "$out"
        if [[ "$mode" == "npu" ]]; then
            if ! grep -q 'scheduler_buft=.*REPACK' "$out/run.log"; then
                echo "[QK_SMOKE_FAIL] HTP REPACK buffer was not selected for $kind" >&2
                exit 1
            fi
            if ! grep -Eq 'accelerated_matmuls=[1-9][0-9]*' "$out/run.log"; then
                echo "[QK_SMOKE_FAIL] no HTP transformer matmuls observed for $kind" >&2
                exit 1
            fi
        fi
    done
done

echo "[QK_SMOKE] PASS: both quantizations completed CPU and HTP; HTP offload was observed"
echo "[QK_SMOKE] Compare actual token/logits output separately to confirm numerical parity."
