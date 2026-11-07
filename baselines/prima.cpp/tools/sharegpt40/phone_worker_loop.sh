#!/usr/bin/env bash
# Keep a Termux rank-1 worker available for repeated single-prompt CLI sessions.
# This SUPERVISES individual processes; ZMQ connections are recreated between
# cases because upstream llama-cli sends STOP at the end of each process.
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="$HOME/models/Qwen3-14B-Q4_K_M.gguf"
MASTER_IP="10.39.52.214"
NEXT_IP=""
LOG_DIR="$HOME/prima-sharegpt40-phone-logs"
RESTART_DELAY=2
if [ -n "$(printenv PRIMA_PHONE_RESTART_DELAY 2>/dev/null)" ]; then RESTART_DELAY="$(printenv PRIMA_PHONE_RESTART_DELAY)"; fi
if [ "$#" -ge 1 ]; then MODEL="$1"; fi
if [ "$#" -ge 2 ]; then MASTER_IP="$2"; fi
NEXT_IP="$MASTER_IP"
if [ "$#" -ge 3 ]; then NEXT_IP="$3"; fi
if [ "$#" -ge 4 ]; then LOG_DIR="$4"; fi

if [ ! -x "$ROOT/llama-cli" ]; then
  echo "[PRIMA_PHONE_LOOP] llama-cli missing: $ROOT/llama-cli" >&2
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  echo "[PRIMA_PHONE_LOOP] model missing: $MODEL" >&2
  exit 2
fi
mkdir -p "$LOG_DIR" || exit 2
cd "$ROOT" || exit 2

running=1
child=""
stop_worker() {
  running=0
  if [ -n "$child" ]; then
    kill -TERM "$child" 2>/dev/null || true
  fi
}
trap 'stop_worker' INT TERM

attempt=0
while [ "$running" -eq 1 ]; do
  attempt=$((attempt + 1))
  ts="$(date '+%Y%m%d-%H%M%S')"
  log="$LOG_DIR/phone-rank1-$ts-$attempt.log"
  echo "[PRIMA_PHONE_LOOP] start session=$attempt log=$log"
  "$ROOT/llama-cli" -m "$MODEL" \
    --world 2 --rank 1 --master "$MASTER_IP" --next "$NEXT_IP" \
    --prefetch >"$log" 2>&1 &
  child=$!
  wait "$child"
  rc=$?
  child=""
  echo "[PRIMA_PHONE_LOOP] session=$attempt exit_code=$rc"
  if [ "$running" -ne 1 ]; then
    break
  fi
  # Rank-0 completion sends STOP; restart for the next independent prompt.
  # A rank assigned 0 layers is also kept available for the next session.
  sleep "$RESTART_DELAY"
done
echo "[PRIMA_PHONE_LOOP] stopped"
