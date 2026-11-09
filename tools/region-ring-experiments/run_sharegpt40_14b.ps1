# Formal Qwen3-14B ShareGPT40 prefill-only sweep (G9/P1 x3 + G10).
# Per-Prompt minimum safe Context; one predicted token; no Profile by default.
# Re-run without switches to resume; -Rerun prompt_512_05 archives/retries.
[CmdletBinding()]
param(
    [string] $Model = "E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
    [string] $Rpc = "10.39.52.114:50052",
    [string] $PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string] $Out = "E:\llama\sharegpt40-14b-g9p1-prefill1",
    [string] $Exe = "",
    [string] $Layout = "G9,P1,G9,P1,G9,P1,G10",
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
    [switch] $DryRun,
    [switch] $FixedChunks,
    [switch] $ProfileChunks,
    [switch] $UpdateProfile,
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
    (Join-Path $PSScriptRoot "run_sharegpt40_14b.py"),
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
if ($FixedChunks) { $callArgs += "--fixed-chunks" }
if ($ProfileChunks) { $callArgs += "--profile-chunks" }
if ($UpdateProfile) { $callArgs += "--update-profile" }
if ($UpdateTokens) { $callArgs += "--update-tokens" }
if ($TokenizerExe -ne "") { $callArgs += @("--tokenizer-exe", $TokenizerExe) }
& $python.Source @callArgs
exit $LASTEXITCODE
