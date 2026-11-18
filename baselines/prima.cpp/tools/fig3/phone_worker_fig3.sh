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
RESUME="${PRIMA_FIG3_PHONE_RESUME:-0}"
case "$RESUME" in 0|1) ;; *) echo "PRIMA_FIG3_PHONE_RESUME must be 0 or 1" >&2; exit 2;; esac
case "$RUN_ID" in *[!a-zA-Z0-9._-]*|"") echo "unsafe RUN_ID" >&2; exit 2;; esac
if [ ! -x "$ROOT/llama-cli" ] || [ ! -f "$MODEL" ]; then
    echo "Missing llama-cli or model" >&2; exit 2
fi
mkdir -p "$LOG_ROOT/logs" || exit 2
TRACE="$LOG_ROOT/phone.csv"
attempt=0
clock_pid=""
if [ "$RESUME" = "1" ]; then
    # Only recover an already COMPLETE first, traced session. Reusing
    # the run ID must never truncate phone.csv or create a new profile.
    if [ ! -s "$TRACE" ]; then
        echo "Cannot resume: original Phone profile missing/empty: $TRACE" >&2
        exit 2
    fi
    IFS=, read -r saved_run_id _ < "$TRACE"
    if [ "$saved_run_id" != "$RUN_ID" ]; then
        echo "Cannot resume: Phone trace run-id $saved_run_id != $RUN_ID" >&2
        exit 2
    fi
    for prior in "$LOG_ROOT"/logs/rank1-*.log; do
        [ -f "$prior" ] || continue
        n="${prior##*/rank1-}"
        n="${n%.log}"
        case "$n" in *[!0-9]*|"") continue;; esac
        n=$((10#$n))
        if [ "$n" -gt "$attempt" ]; then attempt="$n"; fi
    done
    if [ "$attempt" -lt 1 ]; then
        echo "Cannot resume: no Rank1 session log; original Profile not verified" >&2
        exit 2
    fi
    # The PC --resume path reuses the previous before/during/after
    # calibrations. No clock responder is necessary for Normal runs.
    echo "[PRIMA_FIG3_PHONE] RESUME saved_phone_csv=$TRACE next_session=$((attempt+1))"
else
    if [ -e "$TRACE" ] && [ -s "$TRACE" ]; then
        echo "Existing Phone profile: $TRACE (use PRIMA_FIG3_PHONE_RESUME=1 or new RUN_ID)" >&2
        exit 2
    fi
    : > "$TRACE"
    python3 -u "$ROOT/tools/fig3/phone_clock_server.py" \
        --port "$CLOCK_PORT" > "$LOG_ROOT/clock.log" 2>&1 &
    clock_pid=$!
fi
child=""
stop() {
    if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; fi
    if [ -n "$clock_pid" ]; then kill -TERM "$clock_pid" 2>/dev/null || true; fi
}
trap 'stop; exit 130' INT TERM
trap 'stop' EXIT
cd "$ROOT" || exit 2
echo "[PRIMA_FIG3_PHONE] run=$RUN_ID trace=$TRACE clock=$CLOCK_PORT resume=$RESUME"
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
