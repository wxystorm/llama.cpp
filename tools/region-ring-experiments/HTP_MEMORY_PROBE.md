# HTP0 memory allocation probe (Android / Termux)

This is a standalone diagnostic, not a Plan, Profile cache, NPU layer
executor or inference benchmark. It does not change the PC/Phone RPC protocol.
Run it only after stopping the phone ggml-rpc-server; the two processes
would otherwise compete for shared RAM and DSP resources.

## What it measures (and does not)

The executable obtains HTP0 and its actual REPACK buffer type through GGML.
Each allocation uses the production Hexagon code path:

    GGML buft allocation -> rpcmem_alloc2 -> rpcmem_to_fd -> fastrpc_mmap

It tries one large buffer (single) or two simultaneously live buffers (dual).
Buffers are freed after each attempt, before trying the next size. It reports
PASS, allocation/mapping failure or a preflight skip, the failed buffer index,
time and MemAvailable. --touch-pages optionally writes to every 4 KiB host
page; it costs substantially more RAM bandwidth.

A PASS means RPCMEM + FastRPC host-side mapping succeeded. It is NOT proof
that the DSP mapped all bytes at once using HAP_mmap or that a multi-layer
HTP graph will run: DSP mapping happens during HTP execution. The Hexagon
backend defaults to GGML_HEXAGON_MBUF=1024 MiB (single Buffer interface cap)
and GGML_HEXAGON_VMEM=3200 MiB (HTP active mapping budget). Neither value
is dedicated NPU DRAM. The HTP0 memory API currently reports free=total=0
(unsupported), so the real production limit requires a second graph/compute
test after this allocation probe.

## Build and deploy

On the PC in the existing Android cross-build environment:

    cd /mnt/e/llama/llama-tensor-v2
    cmake --build build-phone-npu --target ggml-htp-memory-probe -j 8
    cmake --install build-phone-npu --prefix pkg-phone/llama.cpp
    find pkg-phone/llama.cpp -name ggml-htp-memory-probe

On Windows, transfer the installed binary:

    adb push "E:\llama\llama-tensor-v2\pkg-phone\llama.cpp\bin\ggml-htp-memory-probe" /sdcard/Download/

On the phone (Termux), after stopping the normal RPC server:

    mkdir -p ~/phone-v2-profile/bin
    cp ~/storage/downloads/ggml-htp-memory-probe ~/phone-v2-profile/bin/
    chmod 755 ~/phone-v2-profile/bin/ggml-htp-memory-probe
    cd ~/phone-v2-profile
    export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
    export ADSP_LIBRARY_PATH="$PWD/lib"
    export GGML_HEXAGON_NDEV=1

    # This changes only this process's GGML software buffer ceiling.
    GGML_HEXAGON_MBUF=2048 ./bin/ggml-htp-memory-probe \
        --mode both \
        --sizes-mib 256,512,768,1024,1280,1536,1792 \
        2>&1 | tee htp-memory-probe.log

The default aggregate safety cap is min(MemAvailable / 3, 3072 MiB), and it
refuses to leave fewer than 2048 MiB available based on its pre-run snapshot.
Large dual allocations that exceed this conservative cap are logged as
SKIP_AGGREGATE_SAFETY_CAP, NOT evidence of HTP failure. For a deliberate
second pass, use e.g. --max-aggregate-mib 3584 after checking available
memory, preferably while the phone is cool, connected to power and idle.
This override remains bounded by MemAvailable - 2048 MiB.
All sizes are limited to 1984 MiB to avoid overflow when rpcmem_alloc2
falls back to signed 32-bit rpcmem_alloc (Hexagon adds a 4 KiB guard page).

Do not run the probe on a phone with active inference or important unsaved
work. Even an allocation attempt that returns failure can cause Android to
kill the process under memory pressure. The probe prints event=ATTEMPT before
allocating, making unexpected termination diagnosable.

## Read results

    grep '\[HTP_MEM_PROBE\]' htp-memory-probe.log
    grep -E 'failed to allocate|mapping failed|rpcmem_alloc' htp-memory-probe.log

For each mode, report the largest status=PASS along with the first failed
or skipped size. For dual mode, each_mib refers to each of the two live
buffers and total_mib to their sum.

## Planner follow-up

After the measurement, preserve independent single_buffer_limit,
two_bank_weight_budget, Phone shared RAM/KV/Adreno overhead and
DSP_active_mapping_limit. Do not assume one Buffer max equals all NPU
memory. Do not impose a hard 512 MiB per-stage ceiling without evidence.
A production Graph run must still test DSP-side active mappings and HTP
execution before relying on a Plan.
