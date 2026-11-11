# ShareGPT40 incremental G/C/N(/A) Profile + independent APPLY_WIDE Plan.
# PC and Phone must both use the matching GCNA RPC patch 36 build.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [string]$PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string]$Out = ".\32b-gcna-profile5-plan-all40",
    [string]$Exe = "",
    [string]$NativeProfileExe = "",
    [string]$ProfileDir = "",
    [ValidateRange(1,40)][int]$ProfileEvery = 5,
    [string]$RefreshChunks = "96,128",
    [int]$Threads = 8,
    [int]$Batch = 2048,
    [int]$UBatch = 2048,
    [int]$NPredict = 1,
    [int]$MaxRetries = 1,
    [int]$Limit = 0,
    [switch]$NoMmap,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (!$Exe) {
    $Exe = Join-Path $repo "build-tensor-v2\bin\Release\llama-completion.exe"
}
if (!$NativeProfileExe) {
    $NativeProfileExe = Join-Path $repo "build-tensor-v2\bin\Release\region-ring-v2-stage-profile.exe"
}
$py = (Get-Command python -ErrorAction Stop).Source
$call = @(
    (Join-Path $PSScriptRoot "run_32b_profile5_plan.py"),
    "--exe", $Exe, "--native-profile-exe", $NativeProfileExe,
    "--model", $Model, "--rpc", $Rpc, "--prompt-dir", $PromptDir,
    "--out", $Out, "--profile-every", "$ProfileEvery",
    "--refresh-chunks", $RefreshChunks,
    "--threads", "$Threads", "--batch", "$Batch", "--ubatch", "$UBatch",
    "--n-predict", "$NPredict", "--max-retries", "$MaxRetries", "--limit", "$Limit"
)
if ($ProfileDir) { $call += @("--profile-dir", $ProfileDir) }
if ($NoMmap) { $call += "--no-mmap" }
if ($DryRun) { $call += "--dry-run" }
& $py @call
exit $LASTEXITCODE
