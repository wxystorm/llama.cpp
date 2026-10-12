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

## FLAT retest result: zero column disappears when K13 probing/flush remains enabled

User repeated the original controlled placement (Q13/V13 on CPU, K13 on HTP, only candidate 15-output on HTP) in `hexagon-k13-flat-retest` after the ROW kernel comparison. This test's Android PLAN log has **not yet been supplied**, so the actual kernel for this particular retest should still be checked: `[HEX_K13_PLAN] ... kernel=HVX_QUANT_ROW_FLAT kernel_id=11`.

Results:

- `15-output` status `NUMERICALLY_CLOSE`, `compared=120`, `upstream_diff=0`, `reentry_diff=0`, no missing stages.
- `baseline-control` also `NUMERICALLY_CLOSE` with `diff=0`, `critical=0` across 120 snapshots, meaning the two identical baseline captures are strictly equal.
- PC 42-token `Kcur-13` exact F32 output hashes match across all cases: `e2e5cf4e374b1def` for baseline, candidate and baseline-repeat.
- Token column 2 zeros = 0/512, RMS=2.0578412 for all three cases.

This shows no single-column regression *in this new configuration*, even if the kernel has indeed reverted to FLAT. It disproves the claim that FLAT must fail for `m=42`. It does **not** yet identify the mechanism, because both K13 diagnostic host readback and the K13 forced-input flush can affect execution and cache state; absence of a flaw is not proof that the forced flush alone fixed it. The earlier FLAT DSP hash `4ea315c0e250df06` and the newest PC `e2e5cf4e374b1def` belong to different runs and must not be interpreted as equal-input controlled comparisons without checking exact plan/input hashes.

### Next experiment: exact same FLAT kernel, diagnostics ON, force-input flush OFF

Phone (preserve other variables and restart the RPC server):

```bash
unset GGML_HEXAGON_MM_SELECT
export GGML_HEXAGON_K13_DIAG=1
unset GGML_HEXAGON_K13_FORCE_INPUT_FLUSH

./bin/ggml-rpc-server --device HTP0 --host 0.0.0.0 --port 50052 2>&1 | tee rpc-k13-flat-diag-only.log
```

PC (new output directory):

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @("15-output") -FocusLayer 15 -BaselineControl `
  -PinQCpuLayer 13 -PinVCpuLayer 13 `
  -OutDir ".\hexagon-k13-flat-diag-only"
py -3 .\scripts\hash_hexagon_k13_snapshots.py `
  .\hexagon-k13-flat-diag-only\attn `
  --tokens 42 --column 2
```

**Verify** `[HEX_K13_PLAN] kernel_id=11` and `[HEX_K13_INPUT_PROBE] force_flush=0`, `[HEX_K13_DSP_PROBE] force_input_flush_requested=0`; the diagnostic's own pre/post tensor reads still occur.

- Probe-only FAIL, forced-flush PASS on same FLAT shape/config: strengthens cache visibility hypothesis, though the flush may also change timing.
- Probe-only PASS: probe logging/readbacks alone may perturb the bug; then repeat no-probe/no-flush original configuration to establish that failure still reproduces (fresh output directory), before attributing causation.
- Neither PASS consistently: consider repeating each configuration at least three times under identical binary/flags.

Also check `[HEX_LINEAR_AUDIT]` for `blk.15.attn_output.weight` to ensure the newly enabled projection actually ran on the HTP.


## No K13 diagnostics or force flush: another passing retest

User additionally ran `hexagon-k13-flat-noprobe` with both `GGML_HEXAGON_K13_DIAG` and `GGML_HEXAGON_K13_FORCE_INPUT_FLUSH` unset, with the same Q13/V13 pinned CPU and K13 on HTP:

- Candidate `15-output`: `NUMERICALLY_CLOSE`, 120 matched snapshots, first-evaluation upstream difference 0, reentry difference 0.
- `baseline-control`: `NUMERICALLY_CLOSE`, all 120 tensors bit-identical, diff=0 and critical=0.
- PC K13 output `(512,42,1,1)` F32 full FNV64 for baseline/candidate/baseline-repeat: `e2e5cf4e374b1def` in all three. Token column 2 zeros=0/512, RMS=2.0578412.

**Corrected causal conclusion:** In this new build/placement/run setup, neither K13 extra DSP input flush nor K13 diagnostic readback is necessary for the candidate to pass. Neither `HVX_QUANT_ROW_FLAT` nor the absence of a flush is established as intrinsically erroneous. The earlier FLAT single-column zero remains *unexplained and currently non-reproducible*. Host library rebuild, execution scheduling, allocator layout, process restarts, other environment variables or other unrecorded differences may be relevant. Different historical complete output hashes (`4ea315c0e250df06` versus `e2e5cf4e374b1def`) cannot be analyzed as same-input kernel output inconsistencies without validating both runs' exact input/weight digests and selected kernels. For no-probe runs there is no K13 PLAN diagnostic, so do not infer the kernel solely from the output folder name.

### Move to the more representative unpinned QKV test

Keep the same phone build and runtime settings, with `GGML_HEXAGON_K13_DIAG`, `GGML_HEXAGON_K13_FORCE_INPUT_FLUSH`, `GGML_HEXAGON_MM_SELECT` unset. Run the same test with **no CPU pins** so Q13/K13/V13 all remain on HTP:

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -Ngl 4 -Tokens 1 `
    -Cases @("15-output") -FocusLayer 15 -BaselineControl `
    -OutDir ".\hexagon-attn-original-qkv-retest"
```

Inspect `overview.csv`, `15-output/compare/summary.json`, and `placement.txt` to ensure there are **no** `blk.13.attn_q/k/v.weight=CPU` overrides. Verify the phone `[HEX_LINEAR_AUDIT]` indicates `blk.15.attn_output.weight` actually executed on HTP with `batch_status=OK`; a numerical pass without offload is not an offload correctness pass. If unpinned all-HTP QKV candidate is now clean, test `-Cases @("13-output", "14-output", "15-output", "all-output")` without `-FocusLayer` and preserve separate logs. Do not attribute root-cause resolution to the diagnostic patch without repeatable failing baseline/reproduction.

## Attention Output expanded gate: 13/14 downstream critical, all-output target critical

With CPU pins removed (Q13/K13/V13 intended HTP) and `-Stage attn -Ngl 4 -Cases @(13-output,14-output,all-output) -BaselineControl`:

| case | reported status | compared | upstream_diff | reentry_diff |
| --- | --- | ---: | ---: | ---: |
| `13-output` | `DOWNSTREAM_CRITICAL` | 120 | 0 | 0 |
| `14-output` | `DOWNSTREAM_CRITICAL` | 120 | 0 | 0 |
| `all-output` | `TARGET_CRITICAL` | 120 | 0 | 0 |
| `baseline-control` | `NUMERICALLY_CLOSE`, `critical=0`, `diff=0` | 120 | 0 | 0 |

These are **not passes**. In the comparator, critical is raised if nonfinite, `cosine<0.95`, or `NRMSE>0.05` on the selected CHECK_STAGES (plus undefined-cosine with nonzero element mismatches). `DOWNSTREAM_CRITICAL` does *not* mean target `attn_out` is identical, only that none of its rows meet critical criteria. For `all-output`, `TARGET_CRITICAL` can be at layer 13, 14 or 15; a later target failure may be propagated from an earlier modified layer. `diff=0` is tolerance-based, not necessarily bytewise equality.

Before modifying kernels or memory sync, extract exact target rows and first critical downstream rows from existing `tensor_metrics.csv` and `summary.json`. No rerun required:

```powershell
$root = '.\hexagon-attn-expanded\attn'
foreach ($case in @('13-output','14-output','all-output')) {
  $summary = Get-Content "$root\$case\compare\summary.json" -Raw | ConvertFrom-Json
  Write-Host "===== $case / $($summary.status) ====="
  Write-Host "target diff/critical=$($summary.region_counts.target.diff)/$($summary.region_counts.target.critical) downstream diff/critical=$($summary.region_counts.downstream.diff)/$($summary.region_counts.downstream.critical)"
  $rows = Import-Csv "$root\$case\compare\tensor_metrics.csv"
  $rows | Where-Object { $_.region -eq 'TARGET' -or $_.status -eq 'CRITICAL' } |
    Select-Object -First 24 evaluation,region,name,status,bad,total,rmse,normalized_rmse,cosine | Format-Table -AutoSize
}
```

Need to inspect which graph evaluation (2-token vs 42-token) first becomes severe, each `attn_out-N` target's RMSE/NRMSE/cosine, and `l_out-N` for residual propagation. Also validate fresh `[HEX_LINEAR_AUDIT]` records (correct weights, `batch_status=OK`, `compute_enabled=1`) from the **expanded test run**, not a prior K13-force log. Do not proceed to FFN until these errors are understood.


## Attention Output 13/14: small target error, catastrophic next-layer QKV

Fresh metrics from `hexagon-attn-expanded/attn` (un-pinned QKV on HTP, 2-token graph evaluation 0):

| Case | Target | target NRMSE | target cosine | first strongly divergent later projections |
| --- | --- | ---: | ---: | --- |
| 13-output | attn_out-13 | 0.010992 | 0.999940 | Qcur-14 NRMSE 1.074514 cosine -0.221969; Vcur-14 NRMSE 1.544102 cosine 0.000756 |
| 14-output | attn_out-14 | 0.007811 | 0.999971 | Qcur-15 NRMSE 1.021794 cosine 0.015517; Kcur-15 NRMSE 1.015586 cosine 0.016509; Vcur-15 NRMSE 1.316972 cosine -0.045330 |
| all-output | attn_out-13 | 0.010992 | 0.999940 | Qcur-14 same as 13-output; attn_out-14 NRMSE 1.316241 cosine -0.044907; attn_out-15 NRMSE 1.318228 cosine 0.043320 |

`13-output` downstream `attn_out-14` NRMSE 1.316644 cosine -0.045389 even though the layer14 attention output remained CPU; consequently, `all-output` Layer14 target critical is already explained *in large part* by prior layer13 input divergence, not necessarily a new independent attn-output kernel failure.

All three cases have upstream_diff=0, reentry_diff=0; baseline-control 120/120 strict identical. The earliest **CRITICAL** row being Qcur-14 or Qcur-15 does not mean it is the earliest **different** row. The printed filter omitted non-critical intermediary tensors. Need to first localize divergence between `attn_out-L` and next-layer QKV with saved CSV.

This is tested with `-sm layer`, so the custom tensor-split chunked FFN / cross-layer prepared QKV graph in `src/models/llama.cpp` is not enabled (its guard requires `LLAMA_SPLIT_MODE_TENSOR`). Inspect the ordinary FFN residual/RMS norm/backend hand-off first.

### Read-only first-evaluation chain extraction

```powershell
$root = ".\hexagon-attn-expanded\attn"
foreach ($case in @("13-output", "14-output")) {
    $layer = if ($case -eq "13-output") { 13 } else { 14 }
    $next = $layer + 1
    $names = @(
        "attn_out-$layer", "ffn_inp-$layer", "ffn_norm-$layer",
        "ffn_gate-$layer", "ffn_up-$layer", "ffn_swiglu-$layer",
        "ffn_down-$layer", "ffn_out-$layer", "l_out-$layer",
        "attn_norm-$next", "Qcur-$next", "Kcur-$next", "Vcur-$next"
    )
    Write-Host "==== $case : Layer $layer -> $next, evaluation 0 ===="
    Import-Csv "$root\$case\compare\tensor_metrics.csv" |
        Where-Object { $_.evaluation -eq "0" -and $names -contains $_.name } |
        Sort-Object { [int] $_.cpu_capture } |
        Select-Object cpu_capture,name,shape,status,bad,rmse,normalized_rmse,cosine |
        Format-Table -AutoSize
}
```

Interpretation:
- If `ffn_inp-L` is near target, but `ffn_gate/up/swiglu` or `l_out-L` first jumps: investigate FFN and residual after changed attention output.
- If `l_out-L` and `attn_norm-(L+1)` remain close but next `Qcur/Kcur/Vcur` jumps to NRMSE `1 with cosine near 0, investigate HTP next-layer QKV src1 buffer visibility, RPC scheduler hand-off and backend storage. **Do not attribute this to numerical amplification.**
- If `l_out-L` already differs severely, then next-layer projections may be correct for a badly different input: localize earliest pre-QKV discontinuity.
- For separate causal test, pin next layer's Q/K/V to CPU on both baseline and candidate, compare results; but note CPU pin changes graph layout, so do this only after reading existing CSV.

Current evidence does not establish which case holds. Verify corresponding run's `[HEX_LINEAR_AUDIT]` to establish the intended actual HTP MatMul.


### Concrete break: next-layer RMSNorm nearly identical, Q/K/V abruptly diverge

User printed previously hidden intermediate rows from 13-output and 14-output on the initial 2-token evaluation:

| Case | target attn_out NRMSE | own l_out NRMSE | next attn_norm NRMSE | next Q NRMSE/cos | next K NRMSE/cos | next V NRMSE/cos |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `13-output` | 0.010992 | 0.0000705 | 0.000298 | 1.074514 / -0.221969 | 0.001312 / 0.999999 | 1.544102 / 0.000756 |
| `14-output` | 0.007811 | 0.0000709 | 0.0002218 | 1.021794 / 0.015517 | 1.015586 / 0.016509 | 1.316972 / -0.045330 |

`ffn_inp`, `ffn_norm`, `ffn_gate`, `ffn_up`, `ffn_swiglu`, `l_out`, and next `attn_norm` are close; thus there is no critical jump in CPU FFN or interlayer RMSNorm. The first **large** numerical jump is in next-layer **Q/K/V HTP projection outputs**, in the 2-token eval. In 13-output K14 remains close while Q14/V14 are nearly unrelated to baseline, which is strong evidence against simple small numerical differences growing proportionally through the shared norm input. Physical DSP input visibility, opcode execution and buffer alias remain hypotheses, not confirmed mechanisms.

Next read-only gate from saved snapshots: per-token columns (2 tokens, not the older 42-token index-2 zero):

```powershell
$root = '.\hexagon-attn-expanded\attn'
foreach ($case in @('13-output','14-output')) {
  $next = if ($case -eq '13-output') { 14 } else { 15 }
  py -3 .\scripts\diagnose_hexagon_token_columns.py `
    "$root\baseline" "$root\$case" `
    --tokens 2 --top 2 `
    --name-regex "^(attn_norm|Qcur|Kcur|Vcur)-$next$"
}
```

Also verify same-run phone `[HEX_LINEAR_AUDIT]` for attention output L and `[HEX_QKV_AUDIT]` for next-layer Q/K/V; each must have `batch_status=OK`, `compute_enabled=1`, noting ACK alone does not validate values. Differentiate full/partial zero tokens from nonzero unrelated vectors.

If necessary, causal placement experiment: pin next-layer Q/V to CPU **in both** baseline and candidate for `13-output`, leaving K14 HTP:

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -Ngl 4 -Cases @('13-output') `
    -FocusLayer 13 -BaselineControl `
    -PinQCpuLayer 14 -PinVCpuLayer 14 `
    -OutDir '.\hexagon-attn13-qv14-cpu-control'
```

A pass there links failure to next-layer HTP projection or its memory/schedule dependency, but pinning changes graph placement, so it is not definitive proof of a specific cache or kernel defect. Do not start FFN target-offload tests while these errors persist.


### Per-token confirmation: next-layer errors are nonzero on BOTH 2-token columns

The latest saved-tensor ~diagnose_hexagon_token_columns.py --tokens 2~ comparison yields:

- ~13-output~: ~attn_norm-14~ matches; ~Qcur-14~ column 0/1 both badly different (2035/2048 and 2036/2048 bad), candidate RMS ~0.0676 versus reference RMS ~0.2945, cosine ~-0.22; ~Vcur-14~ columns 0/1 both severely wrong (509 or 510/512 bad), cosine ~0.0005-0.001; ~Kcur-14~ has very small error (nRMSE ~0.0012-0.0014, cosine ~0.999999).
- ~14-output~: ~attn_norm-15~ matches; Q15, K15 and V15 **both columns** badly diverge. Q15 candidate RMS ~0.059 versus reference ~0.261; K15 candidate ~0.069 versus reference ~0.355; V15 candidate ~0.0515 versus reference ~0.0634. No zeroed entire columns, and cosine generally near zero.

This rules out the *same* single missing token-column signature that appeared in earlier 42-token Q/K/V runs, and strongly favors a problem between near-identical host-side norm tensors and selected HTP projection values.

**A new useful conditional hypothesis**: The shared phone experiment env in this branch explicitly enables ~GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1~; its exact K14 input can be flushed before MatMul, while Q14/V14 and Q/K/V15 have no analogous targeted flush. This **could explain the selective K14 pass**, but is only a hypothesis until current-run phone logs confirm ~[HEX_K14_INPUT_FORCE_FLUSH] enabled=1~, a K14 tag and ~force_input_flush_ack=1~ for the actual 2-token evaluation. A K14 force flush also enables extra readbacks/probing, so correlation does not isolate flush from probe. Do not claim this is proved.

Minimal targeted test on the **same host+DSP binaries**: keep ~GGML_HEXAGON_K14_DIAG=1~ (diagnostic readbacks retained), but ~unset GGML_HEXAGON_K14_FORCE_INPUT_FLUSH~ to test whether K14 ceases to be protected. Restart RPC server, run only ~13-output~ in a fresh directory, and inspect same Q/K/V14 metrics. It is also useful to pin only next layer Q14/V14 to CPU in BOTH baseline and candidate to test backend-dependent corruption without reassigning attention output. These tests mutate graph timing/placement and should be interpreted accordingly.


## Probe logs and caveats

```bash
grep -E 'HEX_K13_(PLAN|INPUT_PROBE|DSP_PROBE|OUTPUT_DELTA)|HEX_K14_DSP_PROBE' rpc-k13-force.log
```

Look for HEX_K13_PLAN weight=blk.13.attn_k.weight m=42 and HEX_K13_INPUT_PROBE stage=HTP_TAG force_flush=1. If available, HEX_K13_DSP_PROBE force_input_flush_requested=1 force_input_flush_ack=1 confirms the flagged DSP input flush. The old HEX_K14_TENSOR log prefixes may appear for a K13 seq because the original DSP diagnostic is being reused; correlate with the sequence number.

The DSP response contains only the **last** diagnostic probe per batch. If the log warns MULTIPLE_TAGGED_OPS, absence of a separate K13 ACK is inconclusive even if the K13 tag was present. Optionally set GGML_HEXAGON_OPBATCH=1 for a separate isolated ACK check, but batching changes graph timing and can mask this bug.

Interpretation: A fails / B passes with K13 DSP ACK favors an input cache visibility defect. A passes already means readback/probe timing changed the behavior, not proof flush fixed anything. A and B both fail despite ACK points to other host-to-DSP coherency, kernel write or output storage issues. Existing hex_l2flush on DSP cannot clean dirty cache lines still held by Android CPU.

The two modes are correctness experiments, not performance comparisons. Do not merge into plan-moe-edge until validated in the real hardware and capture-free Decode.