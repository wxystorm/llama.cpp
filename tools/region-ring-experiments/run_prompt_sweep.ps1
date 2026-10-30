# One run per prompt_N.txt. Output/diagnostics go to files, not the terminal.
# Call from PowerShell:
# .\tools\region-ring-experiments\run_prompt_sweep.ps1 -Model "E:\llama\models\Qwen3-14B-Q4_K_M.gguf" -Rpc "10.39.52.114:50052"
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,
    [Parameter(Mandatory = $true)]
    [string] $Rpc,
    [string] $Exe = "",
    [string] $PromptDir = "D:\prima.cpp\prompts",
    [string] $Out = ".\region-ring-results",
    [int] $NPredict = 32,
    [int] $Threads = 8,
    [int] $Batch = 512,
    [int] $UBatch = 512,
    [int] $Seed = 1234,
    [int] $BaselineChunkTokens = 64,
    [int] $MinContext = 1024,
    [int] $ContextPadding = 256,
    [string] $ForceLayout = "",
    [int] $ForceXC = 0,
    [int] $ForceXG = 0,
    [int] $ForceXP = 0,
    [ValidateSet("on", "off")]
    [string] $Streaming = "on",
    [string] $Planner = "apply-wide",
    [switch] $DryRun
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Exe)) {
    $Exe = Join-Path $repoRoot "build-tensor-v2\bin\Release\llama-completion.exe"
}
$python = Get-Command python -ErrorAction Stop
$callArgs = @(
    (Join-Path $PSScriptRoot "run_prompt_sweep.py"),
    "--exe", $Exe,
    "--model", $Model,
    "--rpc", $Rpc,
    "--prompt-dir", $PromptDir,
    "--out", $Out,
    "--n-predict", "$NPredict",
    "--threads", "$Threads",
    "--batch", "$Batch",
    "--ubatch", "$UBatch",
    "--seed", "$Seed",
    "--baseline-chunk-tokens", "$BaselineChunkTokens",
    "--min-context", "$MinContext",
    "--context-padding", "$ContextPadding",
    "--streaming", $Streaming,
    "--planner", $Planner
)
if (-not [string]::IsNullOrWhiteSpace($ForceLayout)) {
    if ($ForceXC -le 0 -or $ForceXG -le 0 -or $ForceXP -le 0) {
        throw "-ForceLayout requires positive -ForceXC, -ForceXG, and -ForceXP."
    }
    $callArgs += @(
        "--force-layout", $ForceLayout,
        "--force-xc", "$ForceXC",
        "--force-xg", "$ForceXG",
        "--force-xp", "$ForceXP"
    )
}
if ($DryRun) {
    $callArgs += "--dry-run"
}
& $python.Source @callArgs
exit $LASTEXITCODE
