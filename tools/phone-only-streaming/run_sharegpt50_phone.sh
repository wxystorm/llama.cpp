#!/system/bin/sh
# Phone-local ShareGPT50 sweep: 14B/32B x CPU/Adreno/HTP x 50 prompts.
# Positional args: MODEL_14B.gguf MODEL_32B.gguf PROMPT_DIR
# Run with: sh run_sharegpt50_phone.sh <14B> <32B> <prompts>
# Override: LLAMA_BIN, LOG_ROOT, MODES, PREDICT, BATCH, UBATCH_*, CONTEXT_*,
#           THREADS, RSS_LIMIT_MIB, GPU_BUDGET_MIB, HTP_BUDGET_MIB, HTP_SESSIONS,
#           PAUSE_SEC, TIMEOUT_SEC, LIMIT_CASES, RETRY_FAILED, FORCE_RERUN.
set -u

usage() {
    echo "Usage: sh $0 14B.gguf 32B.gguf /path/to/sharegpt40" >&2
    echo "Writes raw logs, key metrics and summary to ${LOG_ROOT:-$HOME/log}/sharegpt50" >&2
    exit 2
}

[ "$#" -eq 3 ] || usage
MODEL_14B="$1"
MODEL_32B="$2"
PROMPT_DIR="$3"

LLAMA_BIN="${LLAMA_BIN:-${HOME:-/data/local/tmp}/phone-stream-htp/bin/llama-cli}"
LOG_ROOT="${LOG_ROOT:-${HOME:-/data/local/tmp}/log}"
RUN_DIR="$LOG_ROOT/sharegpt50"
MODES="${MODES:-cpu gpu htp}"
PREDICT="${PREDICT:-1}"
BATCH="${BATCH:-512}"
THREADS="${THREADS:-6}"
UBATCH_128="${UBATCH_128:-64}"
UBATCH_256="${UBATCH_256:-128}"
UBATCH_512="${UBATCH_512:-128}"
UBATCH_1024="${UBATCH_1024:-128}"
UBATCH_2048="${UBATCH_2048:-128}"
CONTEXT_128="${CONTEXT_128:-1024}"
CONTEXT_256="${CONTEXT_256:-1024}"
CONTEXT_512="${CONTEXT_512:-2048}"
CONTEXT_1024="${CONTEXT_1024:-4096}"
CONTEXT_2048="${CONTEXT_2048:-8192}"
RSS_LIMIT_MIB="${RSS_LIMIT_MIB:-7168}"
GPU_BUDGET_MIB="${GPU_BUDGET_MIB:-1536}"
HTP_BUDGET_MIB="${HTP_BUDGET_MIB:-768}"
HTP_SESSIONS="${HTP_SESSIONS:-4}"
PAUSE_SEC="${PAUSE_SEC:-5}"
TIMEOUT_SEC="${TIMEOUT_SEC:-0}"
LIMIT_CASES="${LIMIT_CASES:-0}"
RETRY_FAILED="${RETRY_FAILED:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

for p in "$MODEL_14B" "$MODEL_32B" "$LLAMA_BIN"; do
    if [ ! -f "$p" ]; then
        echo "[SWEEP_ERROR] missing model/binary: $p" >&2
        exit 2
    fi
done
if [ ! -d "$PROMPT_DIR" ]; then
    echo "[SWEEP_ERROR] missing prompt directory: $PROMPT_DIR" >&2
    exit 2
fi
if [ ! -x "$LLAMA_BIN" ]; then
    echo "[SWEEP_ERROR] binary not executable: $LLAMA_BIN" >&2
    exit 2
fi

for n in "$PREDICT" "$BATCH" "$THREADS" \
         "$UBATCH_128" "$UBATCH_256" "$UBATCH_512" "$UBATCH_1024" "$UBATCH_2048" \
         "$CONTEXT_128" "$CONTEXT_256" "$CONTEXT_512" "$CONTEXT_1024" "$CONTEXT_2048" \
         "$RSS_LIMIT_MIB" "$GPU_BUDGET_MIB" "$HTP_BUDGET_MIB" "$HTP_SESSIONS" \
         "$PAUSE_SEC" "$TIMEOUT_SEC" "$LIMIT_CASES" "$RETRY_FAILED" "$FORCE_RERUN"; do
    case "$n" in
        ''|*[!0-9]*) echo "[SWEEP_ERROR] invalid numeric setting: $n" >&2; exit 2 ;;
    esac
done
if [ "$HTP_SESSIONS" -lt 1 ] || [ "$HTP_SESSIONS" -gt 4 ]; then
    echo "[SWEEP_ERROR] HTP_SESSIONS must be between 1 and 4" >&2
    exit 2
fi
if [ "$TIMEOUT_SEC" -gt 0 ] && ! command -v timeout >/dev/null 2>&1; then
    echo "[SWEEP_ERROR] TIMEOUT_SEC requires the timeout command" >&2
    exit 2
fi
for mode in $MODES; do
    case "$mode" in
        cpu|gpu|htp) ;;
        *) echo "[SWEEP_ERROR] invalid MODES entry: $mode" >&2; exit 2 ;;
    esac
done

# Fail early if the 50 requested txt prompts are not complete.
missing=0
for group in 128 256 512 1024 2048; do
    i=1
    while [ "$i" -le 10 ]; do
        two_digits="$(printf '%02d' "$i")"
        f="$PROMPT_DIR/prompt_${group}_${two_digits}.txt"
        if [ ! -f "$f" ]; then
            echo "[SWEEP_ERROR] missing prompt: $f" >&2
            missing=$((missing + 1))
        fi
        i=$((i + 1))
    done
done
[ "$missing" -eq 0 ] || exit 2

if ! mkdir -p "$RUN_DIR"; then
    echo "[SWEEP_ERROR] cannot write $RUN_DIR; set LOG_ROOT to a writable directory" >&2
    exit 2
fi
APP_DIR="$(cd "$(dirname "$LLAMA_BIN")/.." && pwd)"
export LD_LIBRARY_PATH="$APP_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$APP_DIR/lib${ADSP_LIBRARY_PATH:+:$ADSP_LIBRARY_PATH}"
# Set NDEV per experiment below. CPU/OpenCL must not open HTP sessions.
export GGML_OPENCL_STREAM_TRACE_SPLITS=1
export GGML_OPENCL_STREAM_RSS_LIMIT_MIB="$RSS_LIMIT_MIB"

# Keep one experiment configuration. Use a fresh LOG_ROOT when changing settings,
# so an old .ok cannot be mistaken for a result with different model/parameters.
CONFIG_NEXT="$RUN_DIR/.config.next"
{
    printf 'version=1\n'
    printf 'model_14b=%s\nmodel_32b=%s\nprompts=%s\nllama_bin=%s\n' "$MODEL_14B" "$MODEL_32B" "$PROMPT_DIR" "$LLAMA_BIN"
    printf 'modes=%s\npredict=%s\nbatch=%s\nthreads=%s\n' "$MODES" "$PREDICT" "$BATCH" "$THREADS"
    printf 'ubatch=%s,%s,%s,%s,%s\n' "$UBATCH_128" "$UBATCH_256" "$UBATCH_512" "$UBATCH_1024" "$UBATCH_2048"
    printf 'context=%s,%s,%s,%s,%s\n' "$CONTEXT_128" "$CONTEXT_256" "$CONTEXT_512" "$CONTEXT_1024" "$CONTEXT_2048"
    printf 'rss_limit_mib=%s\ngpu_budget_mib=%s\nhtp_budget_mib=%s\nhtp_sessions=%s\nverbosity=3\ntrace_splits=1\n' \
        "$RSS_LIMIT_MIB" "$GPU_BUDGET_MIB" "$HTP_BUDGET_MIB" "$HTP_SESSIONS"
} > "$CONFIG_NEXT"
if [ -f "$RUN_DIR/config.txt" ] &&
        [ "$(cat "$RUN_DIR/config.txt")" != "$(cat "$CONFIG_NEXT")" ]; then
    echo "[SWEEP_ERROR] configuration changed in $RUN_DIR" >&2
    echo "Use a new LOG_ROOT (or archive the old run) to avoid mixing different experiments." >&2
    rm -f "$CONFIG_NEXT"
    exit 2
fi
mv "$CONFIG_NEXT" "$RUN_DIR/config.txt"

SUMMARY="$RUN_DIR/summary.tsv"
if [ ! -f "$SUMMARY" ]; then
    printf 'time\tmodel\tbackend\tgroup\tprompt\tstatus\texit_code\tprefill_tokens\tprefill_ms\tprefill_tok_s\taccelerated_matmuls\tweight_copies\tstaged_weight_mib\tlog\n' > "$SUMMARY"
fi
# Preflight is informational: keep HTP0 disabled so a missing FastRPC DSP
# library cannot abort the entire sweep before CPU/GPU cases begin.
GGML_HEXAGON_NDEV=0 "$LLAMA_BIN" --list-devices -lv 3 > "$RUN_DIR/devices.log" 2>&1 || true

# Fail before the 100 HTP cases if the requested FastRPC sessions cannot
# open. The listed device names are authoritative; registration debug logs
# can mention HTP0 even when it failed to initialize.
case " $MODES " in
    *" htp "*)
        if ! GGML_HEXAGON_NDEV="$HTP_SESSIONS" GGML_HEXAGON_HOSTBUF=0 \
                "$LLAMA_BIN" --list-devices -lv 3 > "$RUN_DIR/htp-devices.log" 2>&1; then
            echo "[SWEEP_ERROR] HTP device preflight failed; see $RUN_DIR/htp-devices.log" >&2
            exit 2
        fi
        session=0
        while [ "$session" -lt "$HTP_SESSIONS" ]; do
            if ! grep -Eq "^  HTP${session}: " "$RUN_DIR/htp-devices.log"; then
                echo "[SWEEP_ERROR] HTP$session unavailable; see $RUN_DIR/htp-devices.log" >&2
                exit 2
            fi
            session=$((session + 1))
        done
        ;;
esac

echo "[SWEEP] 2 models x 50 prompts x 3 backends = 300 cases (when MODES is default)"
echo "[SWEEP] logs=$RUN_DIR verbose=3 trace_splits=1 htp_sessions=$HTP_SESSIONS"
echo "[SWEEP] resume skips PASS and FAIL; RETRY_FAILED=1 to retry failures"
ran=0
skipped=0
passed=0
failed=0

for mode in $MODES; do
    case "$mode" in
        cpu) flag="-cpu"; budget="$GPU_BUDGET_MIB"; export GGML_HEXAGON_NDEV=0 ;;
        gpu) flag="-opencl"; budget="$GPU_BUDGET_MIB"; export GGML_HEXAGON_NDEV=0 ;;
        htp) flag="-npu"; budget="$HTP_BUDGET_MIB"; export GGML_HEXAGON_NDEV="$HTP_SESSIONS" ;;
    esac
    export GGML_PHONE_STREAM_DEVICE_BUDGET_MIB="$budget"
    for model_tag in 14b 32b; do
        case "$model_tag" in
            14b) model="$MODEL_14B" ;;
            32b) model="$MODEL_32B" ;;
        esac
        for group in 128 256 512 1024 2048; do
            case "$group" in
                128) ctx="$CONTEXT_128"; ub="$UBATCH_128" ;;
                256) ctx="$CONTEXT_256"; ub="$UBATCH_256" ;;
                512) ctx="$CONTEXT_512"; ub="$UBATCH_512" ;;
                1024) ctx="$CONTEXT_1024"; ub="$UBATCH_1024" ;;
                2048) ctx="$CONTEXT_2048"; ub="$UBATCH_2048" ;;
            esac
            case_dir="$RUN_DIR/$model_tag/$mode"
            mkdir -p "$case_dir" || exit 2
            i=1
            while [ "$i" -le 10 ]; do
                two_digits="$(printf '%02d' "$i")"
                base="prompt_${group}_${two_digits}"
                prompt="$PROMPT_DIR/$base.txt"
                logfile="$case_dir/$base.log"
                keyfile="$case_dir/$base.metrics.log"
                ok="$case_dir/$base.ok"
                fail="$case_dir/$base.fail"
                if [ "$FORCE_RERUN" -ne 1 ] &&
                   { [ -f "$ok" ] || { [ -f "$fail" ] && [ "$RETRY_FAILED" -ne 1 ]; }; }; then
                    skipped=$((skipped + 1))
                    i=$((i + 1))
                    continue
                fi

                timestamp="$(date '+%Y-%m-%dT%H:%M:%S%z')"
                echo "[SWEEP_RUN] $model_tag $mode $base ctx=$ctx ub=$ub"
                {
                    printf '[SWEEP_BEGIN] timestamp=%s model=%s backend=%s prompt=%s\n' "$timestamp" "$model_tag" "$mode" "$base"
                    printf '[SWEEP_CONFIG] ctx=%s batch=%s ubatch=%s threads=%s predict=%s lv=3 rss_limit_mib=%s device_budget_mib=%s\n' \
                        "$ctx" "$BATCH" "$ub" "$THREADS" "$PREDICT" "$RSS_LIMIT_MIB" "$budget"
                } > "$logfile"

                if [ "$TIMEOUT_SEC" -gt 0 ]; then
                    timeout "$TIMEOUT_SEC" "$LLAMA_BIN" \
                        "$flag" -m "$model" -f "$prompt" \
                        -c "$ctx" -b "$BATCH" -ub "$ub" -t "$THREADS" \
                        -n "$PREDICT" -lv 3 </dev/null >> "$logfile" 2>&1
                else
                    "$LLAMA_BIN" \
                        "$flag" -m "$model" -f "$prompt" \
                        -c "$ctx" -b "$BATCH" -ub "$ub" -t "$THREADS" \
                        -n "$PREDICT" -lv 3 </dev/null >> "$logfile" 2>&1
                fi
                rc=$?
                printf '[SWEEP_END] exit_code=%s\n' "$rc" >> "$logfile"

                grep -E 'PHONE_STREAM|PHONE_STREAM_SPLIT|PHONE_STREAM_BUDGET|HTP_MMAP|HTP[0-3]|GPUOpenCL|ggml-hex|prompt eval time|eval time|total time|llama_perf|model buffer size|compute buffer size' \
                    "$logfile" > "$keyfile" || true
                perf_line="$(grep -E 'prompt eval time[[:space:]]*=' "$logfile" | tail -n 1)"
                prefill_ms="$(printf '%s\n' "$perf_line" | awk '{for (i=1;i<=NF;i++) if ($i=="=" && $(i+2)=="ms") {print $(i+1); exit}}')"
                prefill_tokens="$(printf '%s\n' "$perf_line" | awk '{for (i=1;i<=NF;i++) if ($i=="/" && $(i+2)=="tokens") {print $(i+1); exit}}')"
                prefill_tps="$(printf '%s\n' "$perf_line" | awk '{for (i=1;i<=NF;i++) if ($i=="tokens" && $(i+1)=="per") {print $(i-1); exit}}')"

                sums="$(awk '/\[PHONE_STREAM\] backend_splits=/ {
                    for (i=1; i<=NF; i++) {
                        split($i, kv, "=");
                        if (kv[1]=="accelerated_matmuls") accelerated+=kv[2];
                        if (kv[1]=="weight_copies") copies+=kv[2];
                        if (kv[1]=="staged_weight_mib") staged+=kv[2];
                    }
                } END {printf "%d %d %.2f", accelerated+0, copies+0, staged+0}' "$logfile")"
                set -- $sums
                accelerated="$1"
                copies="$2"
                staged="$3"

                status=PASS
                if [ "$rc" -ne 0 ]; then
                    status=FAIL_EXIT
                elif [ -z "$prefill_ms" ]; then
                    status=FAIL_NO_TIMING
                elif [ "$mode" != "cpu" ] && [ "$accelerated" -eq 0 ]; then
                    status=FAIL_NO_ACCELERATION
                fi

                rm -f "$ok" "$fail"
                if [ "$status" = PASS ]; then
                    : > "$ok"
                    passed=$((passed + 1))
                else
                    : > "$fail"
                    failed=$((failed + 1))
                fi
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$timestamp" "$model_tag" "$mode" "$group" "$base" "$status" "$rc" \
                    "$prefill_tokens" "$prefill_ms" "$prefill_tps" "$accelerated" "$copies" "$staged" "$logfile" \
                    >> "$SUMMARY"
                echo "[SWEEP_RESULT] $model_tag $mode $base status=$status exit=$rc prefill_tok_s=${prefill_tps:-NA} matmuls=$accelerated log=$logfile"
                ran=$((ran + 1))
                if [ "$LIMIT_CASES" -gt 0 ] && [ "$ran" -ge "$LIMIT_CASES" ]; then
                    echo "[SWEEP_LIMIT] reached LIMIT_CASES=$LIMIT_CASES; resume with LIMIT_CASES=0"
                    exit 0
                fi
                if [ "$PAUSE_SEC" -gt 0 ]; then sleep "$PAUSE_SEC"; fi
                i=$((i + 1))
            done
        done
    done
done

echo "[SWEEP_DONE] run=$ran skipped=$skipped pass=$passed fail=$failed"
echo "[SWEEP_DONE] summary=$SUMMARY; raw logs under $RUN_DIR/{14b,32b}/{cpu,gpu,htp}/"
