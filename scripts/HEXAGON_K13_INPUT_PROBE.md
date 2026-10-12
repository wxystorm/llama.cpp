# Hexagon K13 SRC1 probe: controlled experiment

This branch is derived from test/hexagon-attn-ffn-20261012. It reuses the existing K14 DSP diagnostic bits to probe the exact weight blk.13.attn_k.weight. Only the Android host ggml-hexagon.cpp changes. The deployed libggml-htp-v79.so DSP library can be reused, with no ABI changes.

## Confirmed symptom

With Q/V13 pinned to CPU and K13 on HTP, candidate 15-output yields an all-zero Kcur-13 token column 2 in the 42-token graph (512/512 zero in candidate), whereas attn_norm-13 and CPU Q/V13 are identical. The baseline-repeat comparison is exact across 120 snapshots. An ordinary rounding discrepancy cannot explain the all-zero column.

## Phone shared configuration

```bash
export GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC=1
export GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1
export GGML_HEXAGON_OPFUSION=0
export GGML_HEXAGON_QKV_AUDIT=1
export GGML_HEXAGON_LINEAR_AUDIT=1
unset GGML_HEXAGON_KV14_SET_ROWS_FORCE_INPUT_FLUSH
unset GGML_HEXAGON_KV14_SET_ROWS_DIAG
unset GGML_HEXAGON_PROFILE
unset GGML_HEXAGON_SET_ROWS_SYNC_TIMING
unset GGML_HEXAGON_K14_SENTINEL
```

After building and deploying the **new Android host libggml-hexagon.so**, restart RPC Server for each condition. Existing DSP v79 library is unchanged.

## A: probe-only (no forced K13 input flush)

Phone:
```bash
export GGML_HEXAGON_K13_DIAG=1
unset GGML_HEXAGON_K13_FORCE_INPUT_FLUSH
./bin/ggml-rpc-server --device HTP0 --host 0.0.0.0 --port 50052 2>&1 | tee rpc-k13-diag.log
```

PC:
```powershell
git fetch origin
git switch test/hexagon-k13-input-probe-20261012
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -Ngl 4 -Tokens 1 `
    -Cases @("15-output") -FocusLayer 15 -BaselineControl `
    -PinQCpuLayer 13 -PinVCpuLayer 13 `
    -OutDir ".\hexagon-k13-diag-only"
```

## B: force-refresh just the K13 MatMul F32 input on DSP

Stop the old RPC Server; start a new one:
```bash
unset GGML_HEXAGON_K13_DIAG
export GGML_HEXAGON_K13_FORCE_INPUT_FLUSH=1
./bin/ggml-rpc-server --device HTP0 --host 0.0.0.0 --port 50052 2>&1 | tee rpc-k13-force.log
```

PC:
```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -Ngl 4 -Tokens 1 `
    -Cases @("15-output") -FocusLayer 15 -BaselineControl `
    -PinQCpuLayer 13 -PinVCpuLayer 13 `
    -OutDir ".\hexagon-k13-force-flush"
```

Compare both A and B using the existing diagnose_hexagon_token_columns.py, with --tokens 42 and name regex ^(attn_norm|Qcur|Kcur|Vcur)-13$. Check whether Kcur-13 token column 2 is all zero in each run and check overview.csv.

## MM_SELECT=2 experiment: the next comparison must cross DSP, Android host and PC

The user set `GGML_HEXAGON_MM_SELECT=2`. The K13 plan confirms `HVX_QUANT_ROW` (`kernel_id=9`) replaced `HVX_QUANT_ROW_FLAT` (`kernel_id=11`). At 42 tokens, DSP input FNV is identical across modes (`581aae06bf58a4cf`). The two kernels naturally return different full output FNVs:

| Kernel | K13 42-token DSP post-flush FNV across baseline / candidate / baseline-repeat |
| --- | --- |
| FLAT (11) | `4ea315c0e250df06` on seq 3/7/11 |
| ROW (9) | `008627493bc28585` on seq 3/7/11 |

The important observation is **within each kernel**, DSP's full output hash is the same across all three cases, despite the previously verified PC candidate K13 column-2 zero in the FLAT experiment. This is evidence to localize a possible **post-DSP output movement or capture issue**. But it does not prove the ROW mode PC capture is correct, and it cannot rule out a consistently faulty DSP result. Flush POST==FLUSH only compares DSP-visible bytes at two stages; it does not certify Android host or PC tensor readback.

For each experiment, compare these three exact hashes for the same 42-token MatMul:

1. `HEX_K13_DSP_PROBE output_flush_fnv64` for seq 3,7,11.
2. Android-host readback: the helper still prints `[HEX_K14_TENSOR] seq=3 phase=POST role=output ... fnv64=...` (also 7 and 11) for K13 because the old generic helper name was reused. Use:

```bash
grep -E '\[HEX_K14_TENSOR\] seq=(3|7|11) phase=POST role=output' rpc-k13-mm2.log
```

3. PC raw dumped `Kcur-13` F32 `512x42` from the existing folders, using the newly added script:

```powershell
git pull --ff-only
py -3 .\scripts\hash_hexagon_k13_snapshots.py `
    .\hexagon-k13-mm2\attn `
    --tensor Kcur-13 --ne0 512 --tokens 42 --column 2
```

Replace `hexagon-k13-mm2` with the actual OutDir used for the ROW test (no model rerun required). The script reads `baseline`, `15-output`, and `baseline-repeat` manifests, computes FNV64 over the exact whole `512x42xf32` binary, and shows whether token column 2 was zero.

Interpretation:

- DSP hashes equal, Android-host POST hashes equal, PC Dump hashes differ: investigate RPC tensor_get/transport, callback capture time, or PC buffer ownership/overwrites.
- DSP hashes equal, Android-host POST hashes differ: inspect DSP-to-host shared-buffer cache coherence or reused aliases.
- DSP hashes and PC hashes equal across all cases: ROW path may no longer show this problem, but verify per-token numerical accuracy against the expected K output before concluding anything.
- The FLAT/ROW hashes differ by kernel choice; **do not** compare FLAT DSP digest directly to ROW PC digest as if they should match.

Finally, changing `MM_SELECT` may alter prefetch, VTCM and timing. Do not claim it fixed the bug from phone-only batch ACKs; require matching PC capture and per-column evidence.

## Probe logs and caveats

```bash
grep -E 'HEX_K13_(PLAN|INPUT_PROBE|DSP_PROBE|OUTPUT_DELTA)|HEX_K14_DSP_PROBE' rpc-k13-force.log
```

Look for HEX_K13_PLAN weight=blk.13.attn_k.weight m=42 and HEX_K13_INPUT_PROBE stage=HTP_TAG force_flush=1. If available, HEX_K13_DSP_PROBE force_input_flush_requested=1 force_input_flush_ack=1 confirms the flagged DSP input flush. The old HEX_K14_TENSOR log prefixes may appear for a K13 seq because the original DSP diagnostic is being reused; correlate with the sequence number.

The DSP response contains only the **last** diagnostic probe per batch. If the log warns MULTIPLE_TAGGED_OPS, absence of a separate K13 ACK is inconclusive even if the K13 tag was present. Optionally set GGML_HEXAGON_OPBATCH=1 for a separate isolated ACK check, but batching changes graph timing and can mask this bug.

Interpretation: A fails / B passes with K13 DSP ACK favors an input cache visibility defect. A passes already means readback/probe timing changed the behavior, not proof flush fixed anything. A and B both fail despite ACK points to other host-to-DSP coherency, kernel write or output storage issues. Existing hex_l2flush on DSP cannot clean dirty cache lines still held by Android CPU.

The two modes are correctness experiments, not performance comparisons. Do not merge into plan-moe-edge until validated in the real hardware and capture-free Decode.