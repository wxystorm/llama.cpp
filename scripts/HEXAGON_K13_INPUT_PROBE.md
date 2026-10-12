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

## Probe logs and caveats

```bash
grep -E 'HEX_K13_(PLAN|INPUT_PROBE|DSP_PROBE|OUTPUT_DELTA)|HEX_K14_DSP_PROBE' rpc-k13-force.log
```

Look for HEX_K13_PLAN weight=blk.13.attn_k.weight m=42 and HEX_K13_INPUT_PROBE stage=HTP_TAG force_flush=1. If available, HEX_K13_DSP_PROBE force_input_flush_requested=1 force_input_flush_ack=1 confirms the flagged DSP input flush. The old HEX_K14_TENSOR log prefixes may appear for a K13 seq because the original DSP diagnostic is being reused; correlate with the sequence number.

The DSP response contains only the **last** diagnostic probe per batch. If the log warns MULTIPLE_TAGGED_OPS, absence of a separate K13 ACK is inconclusive even if the K13 tag was present. Optionally set GGML_HEXAGON_OPBATCH=1 for a separate isolated ACK check, but batching changes graph timing and can mask this bug.

Interpretation: A fails / B passes with K13 DSP ACK favors an input cache visibility defect. A passes already means readback/probe timing changed the behavior, not proof flush fixed anything. A and B both fail despite ACK points to other host-to-DSP coherency, kernel write or output storage issues. Existing hex_l2flush on DSP cannot clean dirty cache lines still held by Android CPU.

The two modes are correctness experiments, not performance comparisons. Do not merge into plan-moe-edge until validated in the real hardware and capture-free Decode.