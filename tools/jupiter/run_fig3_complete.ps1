<#
One-command Jupiter Fig.3 full collector.
Preflight, build PC, one Profile, clean normal Prefill timings, optional Phone
copy, strict 100% timeline. Analysis recovery never invokes inference.
#>
[CmdletBinding()]
param(
  [string]$Model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf",
  [string]$PromptFile = "E:\llama\datasets\sharegpt\sharegpt50_short\prompt_1024_01.txt",
  [string]$Rpc = "10.238.229.239:10139",
  [int]$PcLayers = 53,
  [int]$ModelLayers = 64,
  [int]$Context = 2048,
  [int]$Batch = 2048,
  [int]$UBatch = 128,
  [int]$Window = 2,
  [int]$Threads = 8,
  [int]$Warmups = 1,
  [int]$NormalRepeats = 3,
  [string]$PlanFile = "",
  [string]$SequencePlan = "",
  [string]$PhoneUser = "u0_a283",
  [int]$PhoneSshPort = 8022,
  [string]$PhoneEvents = "",
  [string]$PhoneRemoteFile = "/data/data/com.termux/files/home/jupiter-fig3-phone-32b.csv",
  [string]$OutputDir = ".\fig3-jupiter32b-complete",
  [switch]$AsyncC2P,
  [switch]$LocalGraphReuse,
  [switch]$SkipBuild,
  [switch]$SkipScp,
  [switch]$Resume,
  [switch]$AnalyzeOnly,
  [switch]$DryRun
)
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..\..")
try {
  $branch = (& git branch --show-current).Trim()
  if ($branch -ne "exp/jupiter-fig3-32b-20261118") {
    throw "Wrong Jupiter branch '$branch'. Switch to exp/jupiter-fig3-32b-20261118."
  }
  Write-Host "[JUPITER_FIG3] GitBranch=$branch"
  if (-not $AnalyzeOnly) {
    & python (Join-Path $PSScriptRoot "test_fig3_complete.py")
    if ($LASTEXITCODE -ne 0) { throw "Fig3 synthetic source/clock analyzer preflight failed" }
  }
  if (-not $SkipBuild -and -not $AnalyzeOnly -and -not $DryRun) {
    # The model experiment needs no Web UI assets.
    Remove-Item ".\build-jupiter\tools\ui\dist" -Recurse -Force -ErrorAction SilentlyContinue
    $cmakeArgs = @("-S",".","-B","build-jupiter",
      "-DLLAMA_BUILD_UI=OFF","-DLLAMA_USE_PREBUILT_UI=OFF",
      "-DLLAMA_UI_GZIP=OFF","-DGGML_CUDA=ON","-DGGML_RPC=ON")
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "Jupiter PC CMake configure failed" }
    & cmake --build build-jupiter --config Release --target llama-cli --parallel 4
    if ($LASTEXITCODE -ne 0) { throw "Jupiter PC llama-cli build failed" }
  }
  $cmd = @(
    (Join-Path $PSScriptRoot "run_fig3_complete.py"),
    "--exe", ".\build-jupiter\bin\Release\llama-cli.exe",
    "--model", $Model,
    "--prompt-file", $PromptFile,
    "--rpc", $Rpc,
    "--pc-layers", "$PcLayers", "--model-layers", "$ModelLayers",
    "--context", "$Context", "--batch", "$Batch", "--ubatch", "$UBatch",
    "--window", "$Window", "--threads", "$Threads",
    "--normal-repeats", "$NormalRepeats", "--warmups", "$Warmups",
    "--phone-user", $PhoneUser, "--phone-ssh-port", "$PhoneSshPort",
    "--phone-remote-file", $PhoneRemoteFile,
    "--out-dir", $OutputDir
  )
  if ($PlanFile) { $cmd += @("--plan-file", $PlanFile) }
  if ($SequencePlan) { $cmd += @("--sequence-plan", $SequencePlan) }
  if ($PhoneEvents) { $cmd += @("--phone-events", $PhoneEvents) }
  if ($AsyncC2P) { $cmd += "--async-c2p" }
  if ($LocalGraphReuse) { $cmd += "--local-graph-reuse" }
  if ($SkipScp) { $cmd += "--skip-scp" }
  if ($Resume) { $cmd += "--resume" }
  if ($AnalyzeOnly) { $cmd += "--analyze-only" }
  if ($DryRun) { $cmd += "--dry-run" }
  & python @cmd
  if ($LASTEXITCODE -ne 0) { throw "Jupiter Fig.3 collector failed; inspect $OutputDir" }
} finally {
  Pop-Location
}
