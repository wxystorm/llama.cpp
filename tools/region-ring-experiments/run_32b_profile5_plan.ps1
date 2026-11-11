# Native APPLY_WIDE Planner on the proven 32B CPU-transport / HTP-only
# backend. Each Prompt re-plans; genuine Stage timings roll every 5 PASS.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [string]$PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string]$Out = ".\32b-htp-only-profile5-plan-all40",
    [string]$Exe = "",
    [ValidateRange(1,40)][int]$ProfileEvery = 5,
    [string]$PhoneProfileLog = "",
    [string]$RealStageProfileCsv = "",
    [int]$Batch = 2048,
    [int]$UBatch = 2048,
    [int]$Threads = 8,
    [int]$NPredict = 1,
    [ValidateRange(0,40)][int]$Limit = 0,
    [int]$MaxRetries = 1,
    [switch]$NoMmap,
    [switch]$StopOnFail,
    [switch]$DryRun,
    [string[]]$EnvOverride = @()
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (!$Exe) {
    $Exe = Join-Path $root "build-tensor-v2\bin\Release\llama-completion.exe"
}
$python = (Get-Command python -ErrorAction Stop).Source
$arguments = @(
    (Join-Path $PSScriptRoot "run_32b_all40.py"),
    "--exe", $Exe, "--model", $Model, "--rpc", $Rpc,
    "--prompt-dir", $PromptDir, "--repo-root", $root, "--out", $Out,
    "--auto-plan", "--htp-only", "--profile-every", "$ProfileEvery",
    "--streaming", "on", "--batch", "$Batch", "--ubatch", "$UBatch",
    "--n-predict", "$NPredict", "--threads", "$Threads",
    "--max-retries", "$MaxRetries", "--limit", "$Limit",
    "--log-verbosity", "4", "--context-padding", "256"
)
if ($NoMmap) { $arguments += "--no-mmap" }
if ($StopOnFail) { $arguments += "--stop-on-fail" }
if ($DryRun) { $arguments += "--dry-run" }
if ($PhoneProfileLog) { $arguments += @("--phone-profile-log", $PhoneProfileLog) }
if ($RealStageProfileCsv) {
    $arguments += @("--real-stage-profile-csv", $RealStageProfileCsv)
}
foreach ($entry in $EnvOverride) { $arguments += @("--env", $entry) }
& $python @arguments
exit $LASTEXITCODE
