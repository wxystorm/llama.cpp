# Real-path Stage calibration, not a synthetic GGML microbenchmark.
# One complete fixed-layout prefill per probe; NOT a Planner-candidate run.
# Restart the Phone RPC Server with GGML_RPC_REAL_STAGE_PROFILE=1.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [string]$Exe = "",
    [string]$Out = ".\real-stage-profile-results",
    [ValidateSet("small","medium")][string]$Grid = "small",
    [int]$Threads = 8,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$prompt = (Resolve-Path $PromptFile).Path
if ([IO.Path]::GetFileName($prompt) -notmatch '^prompt_\d+\.txt$') {
    throw "Expected a prompt_N.txt filename, e.g. prompt_358.txt."
}
$folder = [IO.Path]::GetFullPath((Join-Path $repo $Out))
$inputDir = Join-Path $folder "input"
New-Item -ItemType Directory -Force -Path $inputDir | Out-Null
Copy-Item $prompt (Join-Path $inputDir ([IO.Path]::GetFileName($prompt))) -Force
$keys = @(
    "LLAMA_REGION_RING_REAL_STAGE_PROFILE",
    "LLAMA_REGION_RING_TASK_TRACE",
    "GGML_META_PHONE_FULL_REGION_GRAPH",
    "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES",
    "LLAMA_REGION_RING_PHONE_INPUT_STREAM",
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS",
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL",
    "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM"
)
$old = @{}
foreach ($key in $keys) { $old[$key] = [Environment]::GetEnvironmentVariable($key) }
try {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,"1") }
    $probes = @(
        @("phone3","G10,C1,P3,G11,C1,P1,G11,C1,P1")
    )
    $sizes = @(64)
    if ($Grid -eq "medium") {
        $probes += @(
            @("phone5","G10,C1,P5,G11,C1,P1,G9,C1,P1"),
            @("cpu4","G10,C4,P3,G11,C1,P1,G8,C1,P1"),
            @("cpu8","G10,C8,P1,G10,C1,P1,G7,C1,P1")
        )
        $sizes = @(16,32,64,128)
    }
    foreach ($probe in $probes) {
        foreach ($x in $sizes) {
            $label = $probe[0]
            $layout = $probe[1]
            $opt = @{
                Model=$Model; Rpc=$Rpc; PromptDir=$inputDir
                Out=(Join-Path $folder "$label-x$x")
                Threads=$Threads; NPredict=1; Batch=512; UBatch=512
                Planner="apply-wide"; ForceLayout=$layout
                ForceXC=$x; ForceXG=$x; ForceXP=$x; LogVerbosity=4
            }
            if ($Exe -ne "") { $opt["Exe"] = $Exe }
            if ($DryRun) { $opt["DryRun"] = $true }
            Write-Host "[REAL_STAGE_PROFILE] profile=$label chunk=$x"
            & (Join-Path $PSScriptRoot "run_prompt_sweep.ps1") @opt
            if ($LASTEXITCODE -ne 0) {
                throw "Stage profile failed for $label X=$x, exit=$LASTEXITCODE"
            }
        }
    }
}
finally {
    foreach ($key in $keys) {
        [Environment]::SetEnvironmentVariable($key,$old[$key])
    }
}
Write-Host "[REAL_STAGE_PROFILE] logs=$folder"
Write-Host "Copy Phone rpc.log and run:"
Write-Host "python .\tools\region-ring-experiments\build_real_stage_profile.py --run $folder --rpc-log .\rpc.log --output .\real-stage-profile.csv"
