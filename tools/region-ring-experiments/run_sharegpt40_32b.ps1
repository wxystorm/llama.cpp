# Qwen3-32B, 64-layer ShareGPT40 prefill-only sweep.
# GPU layout: G8,P1,G9,P1,G8,P1,G9,P1,G8,P1,G8,P1,G8
# Two CUDA weight Banks (peak 8 / 9 layers). Auto Context; -n 1; no Profile.
# By default GGUF weights stay mmap-backed to avoid exhausting PC host RAM.
# Re-run unchanged to resume. To compare settings, choose a new -Out.
[CmdletBinding()]
param(
    [string] $Model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf",
    [string] $Rpc = "10.39.52.114:50052",
    [string] $PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string] $Out = "E:\llama\sharegpt40-32b-g8g9-p1-prefill1",
    [string] $Exe = "",
    [string] $Layout = "G8,P1,G9,P1,G8,P1,G9,P1,G8,P1,G8,P1,G8",
    [int] $Context = 0,
    [int] $MinContext = 1024,
    [int] $ContextPadding = 64,
    [int] $XC = 128,
    [int] $XG = 128,
    [int] $XP = 128,
    [string] $ChunkOverrides = "",
    [string[]] $Only = @(),
    [string[]] $Rerun = @(),
    [double] $RerunBelow = 0,
    [int] $Limit = 0,
    [double] $Delay = 2,
    [switch] $NoMmap,
    [switch] $HtpOnly,
    [switch] $PreloadGpuRam,
    [switch] $NoEarlyHtpWeightPreload,
    [switch] $DryRun,
    [ValidateSet("model","manifest")][string] $TokenSource = "model",
    [string] $TokenizerExe = "",
    [switch] $UpdateTokens
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Exe)) {
    $Exe = Join-Path $repoRoot "build-tensor-v2\bin\Release\llama-completion.exe"
}
$python = Get-Command python -ErrorAction Stop
$callArgs = @(
    (Join-Path $PSScriptRoot "run_sharegpt40_32b.py"),
    "--exe", $Exe,
    "--model", $Model,
    "--rpc", $Rpc,
    "--prompt-dir", $PromptDir,
    "--out", $Out,
    "--layout", $Layout,
    "--min-context", "$MinContext",
    "--context-padding", "$ContextPadding",
    "--xc", "$XC", "--xg", "$XG", "--xp", "$XP",
    "--rerun-below", "$RerunBelow",
    "--limit", "$Limit",
    "--delay", "$Delay",
    "--token-source", $TokenSource
)
if ($Context -gt 0) { $callArgs += @("--context", "$Context") }
if ($ChunkOverrides -ne "") { $callArgs += @("--chunk-overrides", $ChunkOverrides) }
foreach ($name in $Only) { $callArgs += @("--only", $name) }
foreach ($name in $Rerun) { $callArgs += @("--rerun", $name) }
if ($DryRun) { $callArgs += "--dry-run" }
if ($NoMmap) { $callArgs += "--no-mmap" }
if ($HtpOnly) { $callArgs += "--htp-only" }
if ($PreloadGpuRam) { $callArgs += "--preload-gpu-ram" }
if ($NoEarlyHtpWeightPreload) { $callArgs += "--no-early-htp-weight-preload" }
if ($UpdateTokens) { $callArgs += "--update-tokens" }
if ($TokenizerExe -ne "") { $callArgs += @("--tokenizer-exe", $TokenizerExe) }
& $python.Source @callArgs
exit $LASTEXITCODE
