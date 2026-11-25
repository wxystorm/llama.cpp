<#
.SYNOPSIS
  Jupiter q(x,y)/Sequence DP full Prefill for Qwen3-14B or Qwen3-32B.
.DESCRIPTION
  Fixed Context/Batch=2048; original measured Stage2 b, Stage3 q(x,y)/DP.
  UseServer now calibrates on the native Phone autonomous FIFO, not Legacy RPC.
  Rebuild both PC and Phone (RPC patch 8); use a NEW output for native profiles.
  Select a single native mode or compare Phone-autonomous P2C to native Sync.
  Single modes skip extra warmup and A-B-A revisits.
  Select model by -ModelSize or -Model, optionally choose CPU/CUDA/HTP
  layer counts or a sweep of Phone boundaries. No Decode.
.EXAMPLE
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -ModelSize 32b -DryRun
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -ModelSize 32b -UseServer
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -ModelSize 32b -CpuLayers 48 -Layouts 56
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -ModelSize 32b -CudaLayers 8 -PhoneLayers 8 -Resume
  .\tools\jupiter\run_phone_owned_full_suite.ps1 -ModelSize 32b -CpuLayers 44 -Layouts "56,58,60,63" -OutDir ".\jupiter-32b-layout-sweep"
#>
[CmdletBinding()]
param(
    [string]$Exe = ".\build-phone-owned\bin\Release\llama-cli.exe",
    [string]$Tokenizer = ".\build-phone-owned\bin\Release\llama-tokenize.exe",
    [string]$ServerExe = ".\build-phone-owned\bin\Release\llama-server.exe",
    [switch]$UseServer,
    [switch]$LegacyCalibration,
    [switch]$SkipLegacyLogitsCheck,
    [int]$ResidentStartTimeout = 1800,
    [int]$LegacyRequestTimeout = 120,
    [int]$LegacyRestartPrompts = 10,
    [int]$LegacyTimeoutRetries = 2,
    [ValidateSet("auto", "14b", "32b")]
    [string]$ModelSize = "auto",
    [string]$Model = "",
    [string]$Rpc = "10.71.109.52:10140",
    [string]$PromptsDir = "E:\llama\datasets\sharegpt\sharegpt50_short",
    [string]$Pattern = "prompt_*_*.txt",
    [string]$Groups = "0,128,256,512,1024",
    [string]$OutDir = "",
    [ValidateSet("jupiter-dp", "fixed-control")]
    [string]$SequencePolicy = "jupiter-dp",
    [ValidateSet("both", "sync", "autonomous")]
    [string]$RunMode = "both",
    [string]$Layouts = "",
    [int]$CpuLayers = -1,
    [int]$CudaLayers = -1,
    [int]$PhoneLayers = -1,
    [int]$ExpectedCount = 50,
    [int]$Repeats = 1,
    [int]$Sessions = 4,
    [int]$ChunkSize = 128,
    [int]$MaxChunk = 128,
    [string]$MinChunkJson = "",
    [int]$Stage2Repeats = 2,
    [int]$QRepeats = 2,
    [int]$RefineRepeats = 2,
    [int]$MaxRounds = 0,
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
    if (!$Model) {
        if ($ModelSize -eq "32b") {
            $Model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf"
        } else {
            $Model = "E:\llama\models\Qwen3-14B-Q4_K_M.gguf"
        }
    }
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
        "--model-size", $ModelSize,
        "--rpc", $Rpc,
        "--prompts-dir", $PromptsDir,
        "--pattern", $Pattern,
        "--groups", $Groups,
        "--sequence-policy", $SequencePolicy,
        "--run-mode", $RunMode,
        "--expected-count", [string]$ExpectedCount,
        "--repeats", [string]$Repeats,
        "--sessions", [string]$Sessions,
        "--chunk-size", [string]$ChunkSize,
        "--max-chunk", [string]$MaxChunk,
        "--stage2-repeats", [string]$Stage2Repeats,
        "--q-repeats", [string]$QRepeats,
        "--refine-repeats", [string]$RefineRepeats,
        "--max-rounds", [string]$MaxRounds,
        "--timeout-seconds", [string]$TimeoutSeconds
    )
    if ($UseServer) {
        if (!(Test-Path -LiteralPath $ServerExe)) {
            throw "Missing resident server: $ServerExe. Build llama-server first."
        }
        $argsList += @("--resident-server", $ServerExe,
                       "--resident-start-timeout", [string]$ResidentStartTimeout,
                       "--legacy-request-timeout", [string]$LegacyRequestTimeout,
                       "--legacy-restart-prompts", [string]$LegacyRestartPrompts,
                       "--legacy-timeout-retries", [string]$LegacyTimeoutRetries)
    }
    if ($LegacyCalibration) { $argsList += "--legacy-calibration" }
    if ($SkipLegacyLogitsCheck) { $argsList += "--skip-legacy-logits-check" }
    if ($OutDir) { $argsList += @("--out-dir", $OutDir) }
    if ($Layouts) { $argsList += @("--layouts", $Layouts) }
    if ($CpuLayers -ge 0) { $argsList += @("--cpu-layers", [string]$CpuLayers) }
    if ($CudaLayers -ge 0) { $argsList += @("--cuda-layers", [string]$CudaLayers) }
    if ($PhoneLayers -ge 0) { $argsList += @("--phone-layers", [string]$PhoneLayers) }
    if ($MinChunkJson) { $argsList += @("--min-chunk-json", $MinChunkJson) }
    if ($Resume) { $argsList += "--resume" }
    if ($DryRun) { $argsList += "--dry-run" }
    if ($RunMode -eq "both" -and !$NoRevisit) { $argsList += "--revisit-first" }
    if ($NoWarmup -or $RunMode -ne "both") { $argsList += "--no-warmup" }
    if ($ContinueOnError) { $argsList += "--continue-on-error" }

    Write-Host "[JUPITER_FULL_SUITE] model=$Model model_size=$ModelSize context=2048 batch=2048"
    Write-Host "[JUPITER_FULL_SUITE] groups=$Groups (default ShareGPT40: 0/128/256/512/1024, 10 each)"
    Write-Host "[JUPITER_FULL_SUITE] layouts=$Layouts cpu=$CpuLayers cuda=$CudaLayers phone=$PhoneLayers sessions=$Sessions run_mode=$RunMode"
    & python @argsList
    $rc = $LASTEXITCODE
    if ($rc -ne 0) {
        throw "Jupiter full suite exited $rc. Inspect the experiment output and calibration logs. Resume with the same arguments after fixing the problem."
    }
    Write-Host "[JUPITER_FULL_SUITE] status=PASS; see the model-specific output directory"
}
finally {
    Pop-Location
}
