# Selective Hexagon QKV input flush A/B (host-only experimental patch)

## Why this test exists

With `13-output` HTP offload and Q/K/V14 left on HTP, the first 2-token computation produced `attn_norm-14` very close to baseline yet Q14 and V14 nearly unrelated to reference. K14 alone was close while `GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1` was enabled. In the follow-up with `GGML_HEXAGON_K14_DIAG=1` but its force flush disabled, Q14 and V14 remained wrong and **K14 also became wrong**:

| Experiment | Q14 nRMSE | K14 nRMSE | V14 nRMSE |
| --- | ---: | ---: | ---: |
| K14 forced input flush ON | ~1.0745 | ~0.00131 | ~1.5441 |
| K14 force OFF, K14 probe ON | ~1.0745 | ~1.0336 | ~1.5441 |

This is strong intervention evidence for the K14 src1 flush path, conditional on confirming the phone binary, env and DSP ACK matched the intended conditions. It is NOT yet proof of a hardware cache coherence bug; timing and extra readbacks are confounders.

## New experimental switches

The host-side `ggml-hexagon.cpp` now supports exact QKV weight matches:

- `GGML_HEXAGON_QKV_PROBE_WEIGHTS`: comma-separated exact weight names, e.g. `blk.14.attn_q.weight,blk.14.attn_v.weight`; tags each selected standalone `HTP_OP_MUL_MAT` for the existing K14 diagnostic, without the extra forced DSP src1 flush.
- `GGML_HEXAGON_QKV_FORCE_INPUT_FLUSH_WEIGHTS`: comma-separated exact weight names; implies corresponding probe and additionally requests DSP `hex_l2flush(src1)` before each selected MatMul. Set the **same probe list** in A and B so diagnostics are held constant.

Valid names must match `blk.<digits>.attn_[qkv].weight`. Unmatched names are ignored with a warning. `GGML_HEXAGON_OPFUSION=0` is required to keep standalone QKV MatMuls; fused QKV operations will not match the standalone tagging.

No DSP code or ABI changes. Rebuild/redeploy **Android CPU-side `libggml-hexagon.so`** from the branch before running. Keep existing deployed `libggml-htp-v79.so`. If the server is launched from `./bin`, confirm that it loads the **new** host library from its runtime library search path, not an older copy in `./bin`.

## A/B experiment: keep K14 force ON and probe Q14/V14 consistently

On the phone, use the existing build and deployment workflow; restart the RPC server between A and B. Shared settings:

```bash
export GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1
export GGML_HEXAGON_QKV_PROBE_WEIGHTS='blk.14.attn_q.weight,blk.14.attn_v.weight'
export GGML_HEXAGON_OPFUSION=0
unset GGML_HEXAGON_K13_DIAG GGML_HEXAGON_K13_FORCE_INPUT_FLUSH
unset GGML_HEXAGON_MM_SELECT
```

**A — Q14/V14 probes only** (no extra Q/V flush):

```bash
unset GGML_HEXAGON_QKV_FORCE_INPUT_FLUSH_WEIGHTS
./bin/ggml-rpc-server --device HTP0 --host 0.0.0.0 --port 50052 2>&1 | tee rpc-attn13-qv-probe.log
```

On PC, run:

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @('13-output') -FocusLayer 13 -BaselineControl `
  -OutDir '.\hexagon-attn13-qv-probe'
```

**B — Q14/V14 probes + forced input flush** (restarting the phone server):

```bash
export GGML_HEXAGON_QKV_FORCE_INPUT_FLUSH_WEIGHTS='blk.14.attn_q.weight,blk.14.attn_v.weight'
./bin/ggml-rpc-server --device HTP0 --host 0.0.0.0 --port 50052 2>&1 | tee rpc-attn13-qv-force.log
```

On PC, run:

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @('13-output') -FocusLayer 13 -BaselineControl `
  -OutDir '.\hexagon-attn13-qv-force'
```

Inspect A and B:

```powershell
$root = '.\hexagon-attn13-qv-force\attn'
Import-Csv "$root\13-output\compare\tensor_metrics.csv" |
  Where-Object { $_.evaluation -eq '0' -and
    $_.name -in @('attn_norm-14','Qcur-14','Kcur-14','Vcur-14') } |
  Sort-Object { [int]$_.cpu_capture } |
  Select-Object -First 12 name,status,bad,normalized_rmse,cosine |
  Format-Table -AutoSize
```

Or with per-token comparison, repeat for either case:

```powershell
py -3 .\scripts\diagnose_hexagon_token_columns.py `
  .\hexagon-attn13-qv-force\attn\baseline `
  .\hexagon-attn13-qv-force\attn\13-output `
  --tokens 2 --top 2 --name-regex '^(attn_norm|Qcur|Kcur|Vcur)-14$'
```

Phone log:

```bash
grep -E 'HEX_QKV_(PROBE_CONFIG|PLAN|INPUT_PROBE|DSP_PROBE)|HEX_K14_DSP_PROBE|HEX_LINEAR_AUDIT' rpc-attn13-qv-force.log
```

Expected tag flags: probe-only `0x2`, force `0x6`. `force_input_flush_ack=1` acknowledges the DSP call only for the **last tagged operation of a batch**. The legacy DSP reply exposes one probe per batch; if multiple QKV ops were tagged together, the absence of per-Q/V ACK cannot demonstrate that a specific op was not flushed. `HEX_QKV_INPUT_PROBE stage=HTP_TAG` confirms only the host flag. Numerical snapshots are essential.

Interpretation:

- **A fails, B passes:** supports a DSP src1 flush/synchronization dependency; repeat A/B and investigate how shared CPU/RPC input cache data reach the DSP, before promoting to a generalized fix.
- **A passes, B passes:** diagnostic readback/timing already masks the error; neither forced flush nor kernel choice has been shown to be required.
- **A fails, B still fails:** identify exact first divergence and check per-operation tagging and input hashes before changing more flags.

**Important:** no actual Android compilation or on-device testing was performed when authoring this patch. These are controlled correctness experiments, not performance settings.
