# Native APPLY_WIDE Planner on the proven 32B CPU-transport / HTP-only
# backend. Each Prompt re-plans using one cached G/C/N Profile (V0).
# Historical -ProfileEvery is accepted but ignored for safe old-command resume.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [string]$PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string]$Out = ".\32b-htp-only-profile5-plan-all40",
    [string]$Exe = "",
    [ValidateRange(0,40)][int]$ProfileEvery = 0,
    [ValidateSet("quick","full")][string]$ProfileGrid = "quick",
    [switch]$UpdateProfile,
    [string]$PhoneProfileLog = "",
    [string]$RealStageProfileCsv = "",
    [string]$ForceLayout = "",
    [int]$ForceXC = 128,
    [int]$ForceXG = 128,
    [int]$ForceXP = 128,
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
if ($ProfileEvery -ne 0) {
    Write-Warning "Rolling Profile5 is disabled in this entry point. Ignoring -ProfileEvery $ProfileEvery; the independent G/C/N profile is reused for all prompts."
}
$arguments = @(
    (Join-Path $PSScriptRoot "run_32b_all40.py"),
    "--exe", $Exe, "--model", $Model, "--rpc", $Rpc,
    "--prompt-dir", $PromptDir, "--repo-root", $root, "--out", $Out,
    "--auto-plan", "--htp-only", "--profile-every", "0",
    "--profile-grid", $ProfileGrid,
    "--streaming", "on", "--batch", "$Batch", "--ubatch", "$UBatch",
    "--n-predict", "$NPredict", "--threads", "$Threads",
    "--max-retries", "$MaxRetries", "--limit", "$Limit",
    "--log-verbosity", "4", "--context-padding", "256"
)
if ($ForceLayout) {
    $arguments += @(
        "--force-plan-layout", $ForceLayout,
        "--force-plan-xc", "$ForceXC",
        "--force-plan-xg", "$ForceXG",
        "--force-plan-xp", "$ForceXP"
    )
} elseif ($PSBoundParameters.ContainsKey("ForceXC") -or
          $PSBoundParameters.ContainsKey("ForceXG") -or
          $PSBoundParameters.ContainsKey("ForceXP")) {
    throw "-ForceXC/-ForceXG/-ForceXP require -ForceLayout"
}
if ($NoMmap) { $arguments += "--no-mmap" }
if ($UpdateProfile) { $arguments += "--update-profile" }
if ($StopOnFail) { $arguments += "--stop-on-fail" }
if ($DryRun) { $arguments += "--dry-run" }
if ($PhoneProfileLog) { $arguments += @("--phone-profile-log", $PhoneProfileLog) }
if ($RealStageProfileCsv) {
    $arguments += @("--real-stage-profile-csv", $RealStageProfileCsv)
}
foreach ($entry in $EnvOverride) { $arguments += @("--env", $entry) }
& $python @arguments
exit $LASTEXITCODE
