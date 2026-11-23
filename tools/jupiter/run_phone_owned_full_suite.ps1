<#
.SYNOPSIS
  Jupiter original q(x,y) + Sequence DP full-Prompt experiment, Context=2048.
.DESCRIPTION
  Per-Prompt: exact Qwen3 Chat Template tokens, measured Stage2 minimum
  efficient Chunk b, Stage3 q(x,y)/DP/refinement, Sync-vs-Phone-autonomous
  full-logits parity, cache comparability, A-B-A and resumable CSV.
  No Decode. PC and Phone RPC must be rebuilt from patch-7 branch.
.EXAMPLE
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -DryRun
  .\tools\jupiter\run_phone_owned_full_suite.ps1
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -Resume
.EXAMPLE
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -SequencePolicy fixed-control -ChunkSize 128 -OutDir .\jupiter-fixed128-ablation
#>
[CmdletBinding()]
param(
    [string]$Exe = ".\build-phone-owned\bin\Release\llama-cli.exe",
    [string]$Tokenizer = ".\build-phone-owned\bin\Release\llama-tokenize.exe",
    [string]$Model = "E:\llama\models\Qwen3-14B-Q4_K_M.gguf",
    [string]$Rpc = "10.71.109.52:10140",
    [string]$PromptsDir = "E:\llama\datasets\sharegpt\sharegpt40",
    [string]$OutDir = ".\jupiter-autonomous-jupiter-dp-2048",
    [ValidateSet("jupiter-dp", "fixed-control")]
    [string]$SequencePolicy = "jupiter-dp",
    [string]$Layouts = "32",
    [int]$ExpectedCount = 50,
    [int]$Repeats = 1,
    [int]$Sessions = 4,
    [int]$CpuLayers = 16,
    [int]$ChunkSize = 128,
    [int]$MaxChunk = 128,
    [string]$MinChunkJson = "",
    [int]$Stage2Repeats = 2,
    [int]$QRepeats = 2,
    [int]$RefineRepeats = 2,
    [int]$MaxRounds = 6,
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
    foreach ($required in @($Exe, $Tokenizer, $Model, $PromptsDir)) {
        if (!(Test-Path -LiteralPath $required)) {
            throw "Required path not found: $required. Compile llama-tokenize if missing."
        }
    }
    if ($MinChunkJson -and !(Test-Path -LiteralPath $MinChunkJson)) {
        throw "Original Jupiter Stage2 min-chunk.json missing: $MinChunkJson"
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
        "--sequence-policy", $SequencePolicy,
        "--layouts", $Layouts,
        "--expected-count", [string]$ExpectedCount,
        "--repeats", [string]$Repeats,
        "--sessions", [string]$Sessions,
        "--cpu-layers", [string]$CpuLayers,
        "--chunk-size", [string]$ChunkSize,
        "--max-chunk", [string]$MaxChunk,
        "--stage2-repeats", [string]$Stage2Repeats,
        "--q-repeats", [string]$QRepeats,
        "--refine-repeats", [string]$RefineRepeats,
        "--max-rounds", [string]$MaxRounds,
        "--timeout-seconds", [string]$TimeoutSeconds
    )
    if ($MinChunkJson) { $argsList += @("--min-chunk-json", $MinChunkJson) }
    if ($Resume) { $argsList += "--resume" }
    if ($DryRun) { $argsList += "--dry-run" }
    if (!$NoRevisit) { $argsList += "--revisit-first" }
    if ($NoWarmup) { $argsList += "--no-warmup" }
    if ($ContinueOnError) { $argsList += "--continue-on-error" }

    Write-Host "[JUPITER_FULL_SUITE] context=2048 batch=2048"
    Write-Host "[JUPITER_FULL_SUITE] policy=$SequencePolicy layouts=$Layouts expected=$ExpectedCount"
    Write-Host "[JUPITER_FULL_SUITE] stage2=$Stage2Repeats q=$QRepeats refine=$RefineRepeats"
    & python @argsList
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        throw "Jupiter full suite exited $rc. Inspect $OutDir\summary.json and calibration logs; use -Resume after fixing."
    }
    Write-Host "[JUPITER_FULL_SUITE] status=PASS output=$OutDir"
}
finally {
    Pop-Location
}
