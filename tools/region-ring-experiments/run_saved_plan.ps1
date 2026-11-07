# Reuse a plan.json produced by region-ring-offline-plan. No planner search.
# The model, token-size hint and Phone NPU Region mapping are checked before
# handing off to the existing fixed-layout run_prompt_sweep.ps1 path.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Plan,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [Parameter(Mandatory=$true)][string]$PromptDir,
    [string]$Out = ".\offline-plan-runs",
    [string]$Exe = "",
    [int]$Threads = 8,
    [int]$NPredict = 32,
    [int]$Batch = 512,
    [int]$UBatch = 512,
    [switch]$NoMmap,
    [switch]$ConfirmBankedPhoneReady,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$planFile = (Resolve-Path $Plan).Path
$p = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
if ($p.schema -eq 2 -and $p.planner -eq 'phone-four-backend-dual-bank-exact-dag-v1') {
    throw "Four-backend G/C/N/A schema=2 plan is NOT runtime-compatible. Missing HTP->Adreno intra-Cycle handoff and N/A Region executor. Refusing legacy P reinterpretation or silent OpenCL fallback."
}
if ($p.schema -ne 1 -or $p.planner -ne 'offline-exact-dag') {
    throw "Invalid plan file schema or planner type: $planFile"
}
if ($p.npu_streaming_required -and -not $p.npu_streaming_runtime_ready) {
    throw "Plan requires two HTP weight Banks. Re-run offline planner with --npu-banked-runtime-confirmed 1 ONLY after the patched Phone HTP server has been built and tested."
}
if ($p.npu_streaming_required -and -not $ConfirmBankedPhoneReady) {
    throw "Banked Plan requires -ConfirmBankedPhoneReady after verifying the Phone server has GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2 and matching Region/Layer lists."
}
if ($p.max_npu_region_layers -gt 1) {
    throw "HTP local-only executor currently supports one Transformer layer per Phone Region."
}
if ([IO.Path]::GetFileName($Model) -ine [IO.Path]::GetFileName([string]$p.model_id)) {
    throw "Model file name does not match saved plan: expected $($p.model_id), got $Model. Re-plan for this model."
}
$prompts = @(Get-ChildItem -LiteralPath $PromptDir -File -Filter 'prompt_*.txt')
if ($prompts.Count -eq 0) { throw "No prompt_N.txt in $PromptDir" }
foreach ($f in $prompts) {
    if ($f.BaseName -notmatch '^prompt_([0-9]+)$' -or
            [int]$Matches[1] -ne [int]$p.tokens) {
        throw "Saved plan expects $($p.tokens)-token hints; this directory contains $($f.Name). Use a plan matching each prompt length."
    }
}
if ([int]$p.xc -le 0 -or [int]$p.xg -le 0 -or
        [int]$p.xp -le 0 -or [string]::IsNullOrWhiteSpace($p.layout)) {
    throw "Plan missing fixed layout or Chunk sizes."
}
Write-Host "[OFFLINE_PLAN_APPLY] model=$($p.model_id) tokens=$($p.tokens)"
Write-Host "[OFFLINE_PLAN_APPLY] layout=$($p.layout) XC=$($p.xc) XG=$($p.xg) XP=$($p.xp)"
Write-Host "[OFFLINE_PLAN_APPLY] Phone RPC MUST be started with GGML_RPC_PHONE_NPU_SERIAL_REGIONS=$($p.phone_npu_regions)"
if ($p.npu_streaming_required) {
    Write-Host "[OFFLINE_PLAN_APPLY] Phone MUST use GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2, the matching ordered GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS, Qwen3 LOCAL_ONLY+GRAPHLESS+DIRECT_INPUT+DIRECT_P2C+HTP_PRECOMPILE, and tested Android Hexagon binaries."
}
Write-Host "[OFFLINE_PLAN_APPLY] This runs APPLY_WIDE -SkipProfile. It does NOT run the Planner or collect new calibration."
$call = @{
    Model = $Model
    Rpc = $Rpc
    PromptDir = $PromptDir
    Out = $Out
    Threads = $Threads
    NPredict = $NPredict
    Batch = $Batch
    UBatch = $UBatch
    Planner = "apply-wide"
    SkipProfile = $true
    ForceLayout = [string]$p.layout
    ForceXC = [int]$p.xc
    ForceXG = [int]$p.xg
    ForceXP = [int]$p.xp
    LogVerbosity = 4
}
if ($Exe -ne "") { $call.Exe = $Exe }
if ($NoMmap) { $call.NoMmap = $true }
if ($DryRun) { $call.DryRun = $true }
& (Join-Path $PSScriptRoot "run_prompt_sweep.ps1") @call
exit $LASTEXITCODE
