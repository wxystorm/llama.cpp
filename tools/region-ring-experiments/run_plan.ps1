# Plan only from a saved, model-specific Profile. No profiling or inference.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [string]$ProfileDir = "",
    [string]$Out = "",
    [string]$PlannerExe = "",
    [string]$Tokenizer = "",
    [switch]$BankedRuntimeConfirmed
)
$ErrorActionPreference = "Stop"
# Mixed HTP/OpenCL cost classes are not yet modeled by the offline DAG.
# Reject instead of silently applying the old aggregate PHONE profile.
if ($env:LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP -eq "1") {
    throw "Four-backend Phone mixed Profile/Planner is not implemented yet. Use the dedicated run_phone_mixed_smoke.ps1 A/B test; do not use the HTP-only Profile as mixed data."
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$modelFile = (Resolve-Path -LiteralPath $Model).Path
$prompt = (Resolve-Path -LiteralPath $PromptFile).Path
$stem = [IO.Path]::GetFileNameWithoutExtension($modelFile)
if (!$ProfileDir) { $ProfileDir = Join-Path $repo "model-profiles\$stem" }
if (!$Out) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $Out = Join-Path $repo "offline-plans\$stem\$([IO.Path]::GetFileNameWithoutExtension($prompt))-$stamp"
}
if (!$PlannerExe) { $PlannerExe = Join-Path $repo "build-tensor-v2\bin\Release\region-ring-offline-plan.exe" }
if (!$Tokenizer) { $Tokenizer = Join-Path $repo "build-tensor-v2\bin\Release\llama-tokenize.exe" }
$py = (Get-Command python -ErrorAction Stop).Source
$argsPy = @(
    (Join-Path $PSScriptRoot "model_profile.py"), "plan",
    "--model", $modelFile, "--profile-dir", $ProfileDir,
    "--prompt", $prompt, "--tokenizer", $Tokenizer,
    "--planner", $PlannerExe, "--out", $Out
)
if ($BankedRuntimeConfirmed) { $argsPy += "--banked-confirmed" }
& $py @argsPy
if ($LASTEXITCODE -ne 0) { throw "Offline planning failed. Existing Profile was NOT updated." }
