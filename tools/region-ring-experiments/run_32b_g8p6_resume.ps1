# Qwen3-32B G8/P6 resumable ShareGPT sweep. Always keeps previous best PASS.
# One pending prompt per call:
#   .\tools\region-ring-experiments\run_32b_g8p6_resume.ps1
# Rerun a specific prompt without overwriting the best result:
#   .\tools\region-ring-experiments\run_32b_g8p6_resume.ps1 -Rerun prompt_512_01.txt -Limit 1
[CmdletBinding()]
param(
    [string] $Model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf",
    [string] $Rpc = "10.39.52.114:50052",
    [string] $PromptDir = "",
    [string] $Out = ".\32b-g8p6-sharegpt40",
    [string] $Exe = "",
    [string] $Tokenizer = "",
    [ValidateRange(0, 100000)] [int] $Limit = 0,
    [switch] $ContinueOnFail,
    [string] $Rerun = "",
    [switch] $RerunAll,
    [ValidateRange(0, 100000)] [double] $MinTps = 0,
    [int] $Group = 0,
    [switch] $StatusOnly,
    [int] $Batch = 2048,
    [int] $UBatch = 2048,
    [int] $NPredict = 1,
    [int] $Threads = 8,
    [int] $Seed = 1234,
    [int] $MinContext = 1024,
    [int] $ContextPadding = 256,
    [ValidateRange(0, 5)] [int] $LogVerbosity = 4,
    [switch] $MoverPriority,
    [switch] $NoRamPreload,
    [switch] $NoTrace
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $Exe) {
    $Exe = Join-Path $root "build-tensor-v2\bin\Release\llama-completion.exe"
}
if (-not $Tokenizer) {
    $Tokenizer = Join-Path $root "build-tensor-v2\bin\Release\llama-tokenize.exe"
}
$python = (Get-Command python -ErrorAction Stop).Source
$callArgs = @(
    (Join-Path $PSScriptRoot "run_32b_g8p6_resume.py"),
    "--model", $Model, "--rpc", $Rpc,
    "--out", $Out, "--exe", $Exe, "--tokenizer", $Tokenizer,
    "--limit", "$Limit", "--min-tps", "$MinTps",
    "--batch", "$Batch", "--ubatch", "$UBatch",
    "--predict", "$NPredict", "--threads", "$Threads",
    "--seed", "$Seed", "--min-context", "$MinContext",
    "--padding", "$ContextPadding", "--log-verbosity", "$LogVerbosity"
)
if ($PromptDir) { $callArgs += @("--prompt-dir", $PromptDir) }
if ($Rerun) { $callArgs += @("--rerun", $Rerun) }
if ($RerunAll) { $callArgs += "--rerun-all" }
if ($ContinueOnFail) { $callArgs += "--continue-on-fail" }
if ($Group -gt 0) { $callArgs += @("--group", "$Group") }
if ($StatusOnly) { $callArgs += "--status-only" }
if ($MoverPriority) { $callArgs += "--mover-priority" }
if ($NoRamPreload) { $callArgs += "--no-ram-preload" }
if ($NoTrace) { $callArgs += "--no-trace" }
& $python @callArgs
if ($LASTEXITCODE -ne 0) {
    throw "32B G8/P6 sweep exited with code $LASTEXITCODE"
}
