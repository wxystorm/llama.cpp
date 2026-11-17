#!/usr/bin/env bash
# Prima Rank 1: the FIRST session is profiled, subsequent sessions are clean.
# Retains the existing two-rank Piped-ring + HiGHS behavior.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL="$HOME/models/Qwen3-14B-Q4_K_M.gguf"
MASTER="10.238.229.214"
RUN_ID="prima-fig3-14b"
if [ "$#" -gt 0 ]; then MODEL="$1"; fi
if [ "$#" -gt 1 ]; then MASTER="$2"; fi
if [ "$#" -gt 2 ]; then RUN_ID="$3"; fi
LOG_ROOT="$HOME/prima-fig3-phone/$RUN_ID"
if [ "$#" -gt 3 ]; then LOG_ROOT="$4"; fi
RESTART_DELAY="$(printenv PRIMA_FIG3_PHONE_DELAY 2>/dev/null || echo 20)"
CLOCK_PORT="$(printenv PRIMA_FIG3_CLOCK_PORT 2>/dev/null || echo 19139)"
case "$RUN_ID" in *[!a-zA-Z0-9._-]*|"") echo "unsafe RUN_ID" >&2; exit 2;; esac
if [ ! -x "$ROOT/llama-cli" ] || [ ! -f "$MODEL" ]; then
    echo "Missing llama-cli or model" >&2; exit 2
fi
mkdir -p "$LOG_ROOT/logs" || exit 2
TRACE="$LOG_ROOT/phone.csv"
if [ -e "$TRACE" ] && [ -s "$TRACE" ]; then
    echo "Existing Phone profile: $TRACE (choose new RUN_ID)" >&2; exit 2
fi
: > "$TRACE"
python3 -u "$ROOT/tools/fig3/phone_clock_server.py" \
    --port "$CLOCK_PORT" > "$LOG_ROOT/clock.log" 2>&1 &
clock_pid=$!
child=""
stop() {
    if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi
    kill -TERM "$clock_pid" 2>/dev/null || true
}
trap 'stop; exit 130' INT TERM
trap 'stop' EXIT
cd "$ROOT" || exit 2
attempt=0
echo "[PRIMA_FIG3_PHONE] run=$RUN_ID trace=$TRACE clock=$CLOCK_PORT"
while true; do
    attempt=$((attempt + 1))
    logfile="$LOG_ROOT/logs/rank1-$(printf '%02d' "$attempt").log"
    echo "[PRIMA_FIG3_PHONE] session=$attempt log=$logfile"
    if [ "$attempt" -eq 1 ]; then
        PRIMA_FIG3_FILE="$TRACE" PRIMA_FIG3_RANK=1 PRIMA_FIG3_RUN_ID="$RUN_ID" \
          "$ROOT/llama-cli" -m "$MODEL" --world 2 --rank 1 \
          --master "$MASTER" --next "$MASTER" --prefetch > "$logfile" 2>&1 &
    else
        env -u PRIMA_FIG3_FILE -u PRIMA_FIG3_RANK -u PRIMA_FIG3_RUN_ID \
          "$ROOT/llama-cli" -m "$MODEL" --world 2 --rank 1 \
          --master "$MASTER" --next "$MASTER" --prefetch > "$logfile" 2>&1 &
    fi
    child=$!
    wait "$child"
    rc=$?
    child=""
    echo "[PRIMA_FIG3_PHONE] session=$attempt status=$rc"
    sleep "$RESTART_DELAY"
done
