#!/usr/bin/env bash
# Prima Fig. 3 / Qwen3-32B, fixed PC52 + Phone12, PC GPU14.
# Unique experiment. Never mix with earlier PC40/Phone24 automatic-profile runs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODE="${1:-}"
RUN_ID="prima-fig3-32b-pc52-phone12-gpu14-v1"
OUT="/mnt/d/fig3-prima32b-pc52-phone12-gpu14-v1"
MODEL="/mnt/e/llama/models/Qwen3-32B-Q4_K_M.gguf"
PROMPT="/mnt/e/llama/datasets/sharegpt/sharegpt50_short/prompt_1024_01.txt"

case "$MODE" in
  profile)
    EXTRA=(--profile-only)
    if [[ -e "$OUT/manifest.json" ]]; then
      echo "[PRIMA_FIG3_PC52] REFUSE: run exists at $OUT; use resume instead" >&2
      exit 2
    fi
    ;;
  finish|resume)
    EXTRA=(--resume)
    if [[ ! -s "$OUT/manifest.json" ]]; then
      echo "[PRIMA_FIG3_PC52] REFUSE: no saved Profile to resume at $OUT" >&2
      exit 2
    fi
    ;;
  full)
    EXTRA=()
    if [[ -e "$OUT/manifest.json" ]]; then
      echo "[PRIMA_FIG3_PC52] REFUSE: run exists at $OUT; use resume instead" >&2
      exit 2
    fi
    ;;
  analyze)
    EXTRA=(--analyze-only)
    ;;
  diagnostic)
    EXTRA=(--analyze-only --clock-mode piecewise-diagnostic)
    ;;
  *)
    echo "Usage: bash tools/fig3/run_32b_pc52_phone12.sh {profile|finish|full|analyze|diagnostic}" >&2
    echo "Phone run-id: $RUN_ID" >&2
    exit 2
    ;;
esac

if [[ ! -x "$ROOT/llama-cli" ]]; then
  echo "[PRIMA_FIG3_PC52] missing executable: $ROOT/llama-cli" >&2
  exit 2
fi
if [[ ! -f "$MODEL" || ! -f "$PROMPT" ]]; then
  echo "[PRIMA_FIG3_PC52] missing 32B model or ShareGPT prompt" >&2
  exit 2
fi

printf '[PRIMA_FIG3_PC52] mode=%s run=%s layout=PC52,Phone12 gpu14\n' "$MODE" "$RUN_ID"
printf '[PRIMA_FIG3_PC52] output=%s\n' "$OUT"
cd "$ROOT"
exec python3 tools/fig3/run_fig3_complete.py \
  --run-id "$RUN_ID" \
  --out "$OUT" \
  --binary "$ROOT/llama-cli" \
  --model "$MODEL" \
  --prompt "$PROMPT" \
  --master 10.238.229.214 \
  --next 10.238.229.239 \
  --context 2048 \
  --batch 512 \
  --ubatch 512 \
  --threads 8 \
  --model-layers 64 \
  --fixed-layers 52,12 \
  --pc-ngl 14 \
  --warmups 1 \
  --repeats 3 \
  --between 30 \
  "${EXTRA[@]}"
