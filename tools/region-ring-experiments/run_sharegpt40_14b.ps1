# Fixed Qwen3-14B G9/P1 x3 + G10 ShareGPT40 sweep.
# Re-run without switches to resume; -Rerun prompt_512_05 forces one fresh attempt.
[CmdletBinding()]
param(
    [string] $Model = "E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
    [string] $Rpc = "10.39.52.114:50052",
    [string] $PromptDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string] $Out = "E:\llama\sharegpt40-14b-g10-results",
    [string] $Exe = "",
    [int] $Context = 4096,
    [string[]] $Only = @(),
    [string[]] $Rerun = @(),
    [double] $RerunBelow = 0,
    [int] $Limit = 0,
    [double] $Delay = 2,
    [switch] $DryRun
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
    "--context", "$Context",
    "--rerun-below", "$RerunBelow",
    "--limit", "$Limit",
    "--delay", "$Delay"
)
foreach ($name in $Only) { $callArgs += @("--only", $name) }
foreach ($name in $Rerun) { $callArgs += @("--rerun", $name) }
if ($DryRun) { $callArgs += "--dry-run" }
& $python.Source @callArgs
exit $LASTEXITCODE
