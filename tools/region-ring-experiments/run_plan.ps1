# Plan only from a saved, model-specific Profile. No profiling or inference.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [string]$ProfileDir = "",
    [string]$Out = "",
    [string]$PlannerExe = "",
    [string]$Tokenizer = "",
    [switch]$BankedRuntimeConfirmed,
    [switch]$PhoneMixed
)
$ErrorActionPreference = "Stop"
# Keep the old three-device offline Planner from mistaking a mixed Phone
# configuration for the legacy aggregate PHONE backend. Select -PhoneMixed
# to use the separate measured four-resource DAG and Bank memory model.
if (!$PhoneMixed -and $env:LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP -eq "1") {
    throw "Mixed Phone mode requires -PhoneMixed (and the separate PHONE_GPU Profile); do not use the old 3-device DAG."
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
if ($PhoneMixed) {
    if ($BankedRuntimeConfirmed) {
        throw "-BankedRuntimeConfirmed applies only to the existing 3-backend executor. Four-backend mixed plans are offline-only until runtime cutover."
    }
    $argsMixed = @(
        (Join-Path $PSScriptRoot "mixed_phone_dag_plan.py"),
        "--model", $modelFile, "--prompt", $prompt,
        "--profile-dir", $ProfileDir, "--tokenizer", $Tokenizer,
        "--out", $Out
    )
    & $py @argsMixed
    if ($LASTEXITCODE -ne 0) { throw "Profile-only four-backend DAG planning failed. No profiling or inference was launched." }
    return
}
$argsPy = @(
    (Join-Path $PSScriptRoot "model_profile.py"), "plan",
    "--model", $modelFile, "--profile-dir", $ProfileDir,
    "--prompt", $prompt, "--tokenizer", $Tokenizer,
    "--planner", $PlannerExe, "--out", $Out
)
if ($BankedRuntimeConfirmed) { $argsPy += "--banked-confirmed" }
& $py @argsPy
if ($LASTEXITCODE -ne 0) { throw "Offline planning failed. Existing Profile was NOT updated." }
