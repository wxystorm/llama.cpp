# C2P L34 clean correctness baseline

This branch starts from `ca551103` and restores **only the serial, correctness-first**
C2P implementation from the pre-parallel Region Boundary work. It deliberately
excludes the later experimental Router/FFN overlap changes.

## Fixed layout for Qwen3-30B-A3B (48 layers)

- **PC Full**: L0-L33 (inclusive); PC can use CUDA/CPU subject to ordinary backend configuration.
- **C2P Tensor**: L34. PC owns Attention, residual, FFN norm, Router/TopK and its FFN shard.
- **Phone Full**: L35-L47 (inclusive).
- Default L34 FFN expert-weight share: PC one-third, Phone two-thirds.
- L34 final `l_out` is **Phone-owned**; do not mirror it to PC.

The C2P boundary stays PC-to-Phone:

1. PC computes Attention, the residual `x = ffn_inp`, FFN Norm and Router.
2. Send normalized hidden activation (the FFN input) and the matching Router TopK/weights PC-to-Phone.
3. Compute the two FFN shards. **This baseline serializes the FFN launch** to
   preserve correctness; simultaneous PC/Phone FFN is a separate future change.
4. PC folds `pc_ffn_partial + x` locally and sends that **one** tensor to Phone.
5. Phone adds its local `phone_ffn_partial`, materializes L34 `l_out`, and
   starts L35. No Phone-to-PC large activation return is allowed at this boundary.

TopK/weights are currently separate small messages. This is not yet a combined
READY packet, and this revision does not claim bandwidth or throughput gains.

## Windows PC environment (PowerShell)

Use the same new branch SHA on both PC and Phone; rebuild the PC RPC backend
and Phone RPC server, then restart the server before testing.

```powershell
$env:LLAMA_HYBRID_REGION_BOUNDARY_V1 = "1"
$env:LLAMA_HYBRID_REGION_C2P_ONLY = "1"
$env:LLAMA_HYBRID_REGION_C2P_LAYER = "34"
$env:LLAMA_HYBRID_REGION_C2P_PC_RATIO = "0.333333"
$env:GGML_META_PIPELINE_DEBUG = "1"
$env:GGML_META_REGION_C2P_NUMERIC_TRACE = "1"
```

Run the existing llama-cli command with tensor split and RPC endpoints. Avoid
reusing a Phone server process or build directory from a Cycle or V5 branch.

The `LLAMA_HYBRID_REGION_BOUNDARY_V1` switch is **off by default**. When
enabled here, only C2P-only layout is permitted; repeating C2P/P2C cycles are
intentionally disabled in this clean branch.

## RPC protocol fail-fast

The PC and Phone must both report `RPC_PROTO_MAJOR.MINOR.PATCH = 4.0.12`.
The client now refuses a different RPC patch version during HELLO and emits
`[RPC_PROTOCOL_MISMATCH]` instead of allowing graph serialization against a
different server implementation. Matching protocol numbers do not prove identical
binaries; verify both Git commit SHAs before runtime tests.

## Verification gates (not yet executed on hardware)

1. Confirm layout log `pc=[0,34) c2p_layer=34 phone=[35,48)`.
2. Confirm L34 FFN physical partition contains nonzero PC and Phone weight units.
3. Compare PC/Phone Router TopK/weights for every chunk; confirm normalized
   hidden arrives before Phone expert execution.
4. Verify `[REGION_C2P_PREFILL]` and `[REGION_C2P_NUMERIC]`, including all
   chunks and final `l_out-34`; follow with L35 input numerical parity.
5. Repeat with single-token decode, multiple chunks and an incremental decode
   sequence before enabling parallel FFN.
6. Only after correctness: measure per-stage PC, Phone, transfer and exposed
   wait times. Keep the existing serial path as the fallback for any future
   parallel experiment.

There is no claim that this commit was compiled, run on Windows/Termux or
validated for numerical parity yet.
