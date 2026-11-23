<#
.SYNOPSIS
  Run the complete Qwen3-14B ShareGPT Phone-autonomous Prefill suite once.
.DESCRIPTION
  Exact Qwen3 chat-template token counts, dynamic Chunk/KV plans, warmed
  synchronous-vs-autonomous full-logits A/B, optional A-B-A, resume and CSV.
  All RPC tasks run serially; no Decode.
.EXAMPLE
  .\tools\jupiter\run_phone_owned_full_suite.ps1
.EXAMPLE
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -Resume
.EXAMPLE
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -Layouts "32,34,36,39" -Repeats 3 -OutDir ".\jupiter-full-matrix"
#>
[CmdletBinding()]
param(
    [string]$Exe = ".\build-phone-owned\bin\Release\llama-cli.exe",
    [string]$Tokenizer = ".\build-phone-owned\bin\Release\llama-tokenize.exe",
    [string]$Model = "E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
    [string]$Rpc = "10.71.109.52:10140",
    [string]$PromptsDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string]$OutDir = ".\jupiter-autonomous-full50",
    [string]$Layouts = "32",
    [int]$ExpectedCount = 50,
    [int]$Repeats = 1,
    [int]$Sessions = 4,
    [int]$CpuLayers = 16,
    [int]$ChunkSize = 88,
    [int]$ContextFloor = 512,
    [int]$ContextCap = 4096,
    [int]$TimeoutSeconds = 1800,
    [switch]$Resume,
    [switch]$DryRun,
    [switch]$NoRevisit,
    [switch]$NoWarmup,
    [switch]$ContinueOnError
)
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $repoRoot
try {
    foreach ($path in @($Exe, $Tokenizer, $Model, $PromptsDir)) {
        if (!(Test-Path -LiteralPath $path)) {
            throw "Required path not found: $path. Build llama-tokenize if missing."
        }
    }
    $driver = Join-Path $PSScriptRoot "run_phone_owned_full_suite.py"
    $argsList = @(
        $driver,
        "--exe", $Exe,
        "--tokenizer", $Tokenizer,
        "--model", $Model,
        "--rpc", $Rpc,
        "--prompts-dir", $PromptsDir,
        "--out-dir", $OutDir,
        "--layouts", $Layouts,
        "--expected-count", [string]$ExpectedCount,
        "--repeats", [string]$Repeats,
        "--sessions", [string]$Sessions,
        "--cpu-layers", [string]$CpuLayers,
        "--chunk-size", [string]$ChunkSize,
        "--context-floor", [string]$ContextFloor,
        "--context-cap", [string]$ContextCap,
        "--timeout-seconds", [string]$TimeoutSeconds
    )
    if ($Resume) { $argsList += "--resume" }
    if ($DryRun) { $argsList += "--dry-run" }
    if (!$NoRevisit) { $argsList += "--revisit-first" }
    if ($NoWarmup) { $argsList += "--no-warmup" }
    if ($ContinueOnError) { $argsList += "--continue-on-error" }

    Write-Host "[PHONE_SUITE_PS] repo=$repoRoot"
    Write-Host "[PHONE_SUITE_PS] expected_prompts=$ExpectedCount layouts=$Layouts repeats=$Repeats"
    & python @argsList
    $result = $LASTEXITCODE
    if ($result -ne 0) {
        throw "Full suite exited $result; check $OutDir\summary.json and the failed case logs. Use -Resume after fixing the problem."
    }
    Write-Host "[PHONE_SUITE_PS] status=PASS output=$OutDir"
}
finally {
    Pop-Location
}
