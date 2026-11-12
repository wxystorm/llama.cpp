#!/usr/bin/env bash
# Experimental direct HTP0 RPC server for Jupiter 14B on Android/Termux.
# Usage: LLAMA_JUPITER_PHONE_BIN_ROOT="$HOME/jupiter-phone" \
#        bash tools/jupiter/start_phone_htp.sh /path/to/Qwen3-14B-Q4_K_M.gguf [port]
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 PHONE_GGUF [PORT=10140]" >&2
    exit 2
fi

model="$1"
port="${2:-10140}"
root="${LLAMA_JUPITER_PHONE_BIN_ROOT:-$PWD}"
server="$root/bin/ggml-rpc-server"

case "$port" in
    *[!0-9]*|"") echo "invalid RPC port: $port" >&2; exit 2 ;;
esac
if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    echo "RPC port must be in [1,65535]" >&2
    exit 2
fi
if [ ! -f "$model" ]; then
    echo "Phone GGUF not found: $model" >&2
    exit 2
fi
if [ ! -x "$server" ]; then
    echo "Phone RPC server not executable: $server" >&2
    exit 2
fi

# Keep this first direct RPC experiment to one HTP session.
# The 1 GiB allocation cap is a per-buffer limit, NOT total HTP memory.
export GGML_HEXAGON_NDEV="${GGML_HEXAGON_NDEV:-1}"
export GGML_HEXAGON_HOSTBUF="${GGML_HEXAGON_HOSTBUF:-1}"
export GGML_HEXAGON_MBUF="${GGML_HEXAGON_MBUF:-1024}"
export GGML_HEXAGON_VMEM="${GGML_HEXAGON_VMEM:-3072}"
if [ "$GGML_HEXAGON_NDEV" != "1" ]; then
    echo "This prototype requires GGML_HEXAGON_NDEV=1" >&2
    exit 2
fi
case "$GGML_HEXAGON_MBUF" in
    ""|*[!0-9]*)
        echo "GGML_HEXAGON_MBUF must be an integer in [1,1024] MiB" >&2
        exit 2
        ;;
esac
if [ "$GGML_HEXAGON_MBUF" -gt 1024 ] || [ "$GGML_HEXAGON_MBUF" -lt 1 ]; then
    echo "Set GGML_HEXAGON_MBUF in [1,1024] MiB for this device" >&2
    exit 2
fi
case "$GGML_HEXAGON_VMEM" in
    ""|*[!0-9]*)
        echo "GGML_HEXAGON_VMEM must be a positive integer in MiB" >&2
        exit 2
        ;;
esac
if [ "$GGML_HEXAGON_VMEM" -lt 1 ]; then
    echo "GGML_HEXAGON_VMEM must be a positive integer in MiB" >&2
    exit 2
fi

# FastRPC searches this directory for libggml-htp-v79.so (DSP skel).
# Keep this a single absolute directory path. In particular, DO NOT
# concatenate LD_LIBRARY_PATH's colon-separated entries here: some
# Qualcomm FastRPC loaders interpret ADSP_LIBRARY_PATH differently and
# fail htp_iface_open() with 0x80000406.
export ADSP_LIBRARY_PATH="$root/lib"
export LD_LIBRARY_PATH="$root/lib:$root/bin:${LD_LIBRARY_PATH:-}"
if [ ! -f "$root/lib/libggml-htp-v79.so" ]; then
    echo "[JUPITER_PHONE_HTP] warning: v79 DSP skel missing: $root/lib/libggml-htp-v79.so" >&2
fi
echo "[JUPITER_PHONE_HTP] adsp_library_path=$ADSP_LIBRARY_PATH"
echo "[JUPITER_PHONE_HTP] backend=HTP0 rpc=0.0.0.0:$port model=$model"
echo "[JUPITER_PHONE_HTP] mbuf_mib=$GGML_HEXAGON_MBUF vmem_mib=$GGML_HEXAGON_VMEM"
echo "[JUPITER_PHONE_HTP] status=EXPERIMENTAL_NOT_PARITY_VERIFIED"
echo "[JUPITER_PHONE_HTP] verify that the RPC device list shows HTP0"
exec "$server" --device HTP0 --host 0.0.0.0 --port "$port" --model "$model"
