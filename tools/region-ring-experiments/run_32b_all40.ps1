# Qwen3-32B ShareGPT40: configurable forced layout/Chunk, safe per-config reruns.
# Run from any directory with PowerShell 5.1 or 7+ and Python 3.9+.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Model,
    [Parameter(Mandatory = $true)][string] $Rpc,
    [string] $Exe = "",
    [string] $PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string] $Out = "",
    [string] $Layout = "",
    [int] $Chunk = 128,
    [int] $XC = 0,
    [int] $XG = 0,
    [int] $XP = 0,
    [switch] $AutoPlan,
    [string] $RealStageProfileCsv = "",
    [ValidateSet("on", "off")][string] $Streaming = "on",
    [switch] $ForceSmallBatchMover,
    [switch] $BoostBankMover,
    [switch] $NoPhoneWeightPreload,
    [switch] $SkipWarmup,
    [string] $WarmupPrompt = "",
    [int] $Batch = 2048,
    [int] $UBatch = 2048,
    [int] $NPredict = 1,
    [int] $Threads = 8,
    [int] $Seed = 1234,
    [int] $MinContext = 256,
    [int] $MaxContext = 4096,
    [int] $ContextPadding = 64,
    [ValidateRange(0, 5)][int] $LogVerbosity = 4,
    [switch] $NoMmap,
    [ValidateRange(0, 50)][int] $MaxRetries = 1,
    [ValidateRange(0, 3600)][double] $RetryDelaySeconds = 5,
    [ValidateRange(0, 3600)][double] $InterPromptDelaySeconds = 5,
    [ValidateRange(0, 86400)][double] $TimeoutSeconds = 0,
    [int[]] $OnlyGroup = @(),
    [string[]] $OnlyPrompt = @(),
    [ValidateRange(0, 40)][int] $Limit = 0,
    [switch] $RerunAll,
    [switch] $RetryFailedOnly,
    [switch] $StopOnFail,
    [switch] $DryRun,
    [string[]] $EnvOverride = @()
)
$ErrorActionPreference = "Stop"
if (-not [string]::IsNullOrWhiteSpace($RealStageProfileCsv) -and -not $AutoPlan) {
    throw "-RealStageProfileCsv requires -AutoPlan."
}
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Exe)) {
    $Exe = Join-Path $repoRoot "build-tensor-v2\bin\Release\llama-completion.exe"
}
$python = Get-Command python -ErrorAction Stop
$arguments = @(
    (Join-Path $PSScriptRoot "run_32b_all40.py"),
    "--exe", $Exe, "--model", $Model, "--rpc", $Rpc,
    "--prompt-dir", $PromptDir, "--repo-root", $repoRoot,
    "--streaming", $Streaming,
    "--batch", "$Batch", "--ubatch", "$UBatch", "--n-predict", "$NPredict",
    "--threads", "$Threads", "--seed", "$Seed",
    "--min-context", "$MinContext", "--max-context", "$MaxContext",
    "--context-padding", "$ContextPadding", "--log-verbosity", "$LogVerbosity",
    "--max-retries", "$MaxRetries", "--retry-delay", "$RetryDelaySeconds",
    "--inter-prompt-delay-seconds", "$InterPromptDelaySeconds",
    "--timeout", "$TimeoutSeconds", "--limit", "$Limit"
)
if (-not [string]::IsNullOrWhiteSpace($Out)) { $arguments += @("--out", $Out) }
if ($PSBoundParameters.ContainsKey("Layout")) { $arguments += @("--layout", $Layout) }
if ($PSBoundParameters.ContainsKey("Chunk")) { $arguments += @("--chunk", "$Chunk") }
if ($PSBoundParameters.ContainsKey("XC")) { $arguments += @("--xc", "$XC") }
if ($PSBoundParameters.ContainsKey("XG")) { $arguments += @("--xg", "$XG") }
if ($PSBoundParameters.ContainsKey("XP")) { $arguments += @("--xp", "$XP") }
if ($AutoPlan) { $arguments += "--auto-plan" }
if ($ForceSmallBatchMover) { $arguments += "--force-small-batch-mover" }
if ($BoostBankMover) { $arguments += "--boost-bank-mover" }
if ($NoPhoneWeightPreload) { $arguments += "--no-phone-weight-preload" }
if ($SkipWarmup) { $arguments += "--skip-warmup" }
if (-not [string]::IsNullOrWhiteSpace($WarmupPrompt)) {
    $arguments += @("--warmup-prompt", $WarmupPrompt)
}
if (-not [string]::IsNullOrWhiteSpace($RealStageProfileCsv)) {
    $arguments += @("--real-stage-profile-csv", $RealStageProfileCsv)
}
if ($NoMmap) { $arguments += "--no-mmap" }
if ($RerunAll) { $arguments += "--rerun-all" }
if ($RetryFailedOnly) { $arguments += "--retry-failed-only" }
if ($StopOnFail) { $arguments += "--stop-on-fail" }
if ($DryRun) { $arguments += "--dry-run" }
foreach ($group in $OnlyGroup) { $arguments += @("--only-group", "$group") }
foreach ($prompt in $OnlyPrompt) { $arguments += @("--only-prompt", "$prompt") }
foreach ($entry in $EnvOverride) { $arguments += @("--env", $entry) }
& $python.Source @arguments
exit $LASTEXITCODE
