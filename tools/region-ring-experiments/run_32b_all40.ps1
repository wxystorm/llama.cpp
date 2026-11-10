# Qwen3-32B ShareGPT40 fixed G8/P6 x4 + G8, 128 Chunk, tiny-tail coalescing.
# Run from any directory with PowerShell 5.1 or 7+ and Python 3.9+.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Model,
    [Parameter(Mandatory = $true)][string] $Rpc,
    [string] $Exe = "",
    [string] $PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string] $Out = ".\32b-g8p6-x128-all40",
    [ValidateSet("on", "off")][string] $Streaming = "on",
    [int] $Batch = 2048,
    [int] $UBatch = 2048,
    [int] $NPredict = 32,
    [int] $Threads = 8,
    [int] $Seed = 1234,
    [int] $MinContext = 1024,
    [int] $MaxContext = 4096,
    [int] $ContextPadding = 256,
    [ValidateRange(0, 5)][int] $LogVerbosity = 4,
    [switch] $NoMmap,
    [ValidateRange(0, 50)][int] $MaxRetries = 1,
    [ValidateRange(0, 3600)][double] $RetryDelaySeconds = 5,
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
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Exe)) {
    $Exe = Join-Path $repoRoot "build-tensor-v2\bin\Release\llama-completion.exe"
}
$python = Get-Command python -ErrorAction Stop
$arguments = @(
    (Join-Path $PSScriptRoot "run_32b_all40.py"),
    "--exe", $Exe, "--model", $Model, "--rpc", $Rpc,
    "--prompt-dir", $PromptDir, "--out", $Out, "--repo-root", $repoRoot,
    "--streaming", $Streaming,
    "--batch", "$Batch", "--ubatch", "$UBatch", "--n-predict", "$NPredict",
    "--threads", "$Threads", "--seed", "$Seed",
    "--min-context", "$MinContext", "--max-context", "$MaxContext",
    "--context-padding", "$ContextPadding", "--log-verbosity", "$LogVerbosity",
    "--max-retries", "$MaxRetries", "--retry-delay", "$RetryDelaySeconds",
    "--timeout", "$TimeoutSeconds", "--limit", "$Limit"
)
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
