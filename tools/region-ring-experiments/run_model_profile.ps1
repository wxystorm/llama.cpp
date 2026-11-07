# Collect model-specific real CPU/CUDA/Phone HTP samples separately.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Rpc = "",
    [string]$PromptFile = "",
    [string]$PhoneLog = "",
    [string]$ProfileRoot = "",
    [string]$Exe = "",
    [string]$Tokenizer = "",
    [int]$Threads = 8,
    [switch]$ShowPhoneEnv,
    [switch]$Finalize,
    [switch]$UpdateProfile,
    [switch]$NoMmap,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
# Mixed HTP/OpenCL cost classes are not yet modeled by the offline DAG.
# Reject instead of silently applying the old aggregate PHONE profile.
if ($env:LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP -eq "1") {
    throw "Four-backend Phone mixed Profile/Planner is not implemented yet. Use the dedicated run_phone_mixed_smoke.ps1 A/B test; do not use the HTP-only Profile as mixed data."
}
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$py = (Get-Command python -ErrorAction Stop).Source
$tool = Join-Path $PSScriptRoot "model_profile.py"
$modelFile = (Resolve-Path -LiteralPath $Model).Path
if (!$ProfileRoot) { $ProfileRoot = Join-Path $repo "model-profiles" }
$folder = Join-Path ([IO.Path]::GetFullPath($ProfileRoot)) ([IO.Path]::GetFileNameWithoutExtension($modelFile))
$metaOutput = & $py $tool inspect --model $modelFile
if ($LASTEXITCODE -ne 0) { throw "Could not inspect GGUF model" }
$meta = ($metaOutput | Out-String | ConvertFrom-Json)

if ($ShowPhoneEnv) {
    Write-Host "[MODEL_PROFILE] model=$($meta.model.file) layers=$($meta.model.layers)"
    Write-Host "Start Phone RPC for this calibration mapping:"
    Write-Host "export GGML_RPC_PHONE_NPU_SERIAL_REGIONS='$($meta.probe.phone_regions)'"
    Write-Host "export GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS='$($meta.probe.phone_layers)'"
    Write-Host "export GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2"
    Write-Host "export GGML_RPC_REAL_STAGE_PROFILE=1"
    Write-Host "export GGML_RPC_PHONE_CHUNK_TIMELINE=1"
    Write-Host "export GGML_RPC_PHONE_QWEN3_LOCAL_ONLY=1"
    Write-Host "export GGML_RPC_PHONE_QWEN3_LOCAL_HTP_PRECOMPILE=1"
    Write-Host "export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_INPUT=1"
    Write-Host "export GGML_RPC_PHONE_QWEN3_LOCAL_DIRECT_P2C=1"
    Write-Host "export GGML_RPC_PHONE_QWEN3_LOCAL_GRAPHLESS=1"
    return
}
if ($Finalize) {
    if (!$PhoneLog) { throw "-Finalize requires -PhoneLog copied from the Phone server" }
    $rpcFile = (Resolve-Path -LiteralPath $PhoneLog).Path
    $runs = Join-Path $folder "runs"
    if (!(Test-Path -LiteralPath $runs)) { throw "No collected probe runs: $runs" }
    $csv = Join-Path $folder "real-stage-profile.csv"
    $stageBuilder = Join-Path $PSScriptRoot "build_real_stage_profile.py"
    & $py $stageBuilder --run $runs --rpc-log $rpcFile --output $csv
    if ($LASTEXITCODE -ne 0) { throw "Real Stage profile assembly failed" }
    & $py $tool finalize --model $modelFile --profile-dir $folder --phone-log $rpcFile --stage-csv $csv
    if ($LASTEXITCODE -ne 0) { throw "Model Profile validation failed" }
    return
}
if (!$Rpc -or !$PromptFile) { throw "Calibration requires -Rpc and -PromptFile" }
if (!$Exe) { $Exe = Join-Path $repo "build-tensor-v2\bin\Release\llama-completion.exe" }
if (!$Tokenizer) { $Tokenizer = Join-Path $repo "build-tensor-v2\bin\Release\llama-tokenize.exe" }
if (!(Test-Path -LiteralPath $Exe) -or !(Test-Path -LiteralPath $Tokenizer)) {
    throw "Build llama-completion and llama-tokenize first"
}
$prompt = (Resolve-Path -LiteralPath $PromptFile).Path
$tokOutput = & $Tokenizer -m $modelFile -f $prompt --no-escape --show-count --log-disable
if ($LASTEXITCODE -ne 0) { throw "Tokenizer failed" }
$tokMatch = [regex]::Match(($tokOutput -join [Environment]::NewLine), 'Total number of tokens:\s*(\d+)')
if (!$tokMatch.Success) { throw "Could not determine Prompt token count" }
$tokens = [int]$tokMatch.Groups[1].Value
$runs = Join-Path $folder "runs"
if ((Test-Path -LiteralPath $runs) -and @(Get-ChildItem $runs -Recurse -File -ErrorAction SilentlyContinue).Count -gt 0) {
    if (!$UpdateProfile) { throw "Existing probes in $runs. Pass -UpdateProfile or use -Finalize." }
    Remove-Item -LiteralPath $runs -Force -Recurse
    $manifest = Join-Path $folder "model_profile.json"
    if (Test-Path -LiteralPath $manifest) { Remove-Item -LiteralPath $manifest -Force }
}
$inputFolder = Join-Path $folder "input"
New-Item -ItemType Directory -Path $inputFolder -Force | Out-Null
Copy-Item -LiteralPath $prompt -Destination (Join-Path $inputFolder "prompt_$tokens.txt") -Force
$envNames = @(
    "LLAMA_REGION_RING_REAL_STAGE_PROFILE", "LLAMA_REGION_RING_TASK_TRACE",
    "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES", "LLAMA_REGION_RING_PHONE_AUTONOMOUS",
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL", "LLAMA_REGION_RING_PHONE_INPUT_PUSH",
    "LLAMA_REGION_RING_PHONE_INPUT_STREAM", "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM",
    "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW", "LLAMA_REGION_RING_PHONE_GRAPHLESS"
)
$old = @{}
foreach ($key in $envNames) { $old[$key] = [Environment]::GetEnvironmentVariable($key, "Process") }
try {
    foreach ($key in $envNames) { [Environment]::SetEnvironmentVariable($key, "1", "Process") }
    $first = $true
    $index = 0
    foreach ($layout in $meta.probe.layouts) {
        foreach ($x in $meta.probe.chunks) {
            $options = @{
                Model=$modelFile; Rpc=$Rpc; PromptDir=$inputFolder
                Out=(Join-Path $runs "layout$index-x$x"); Exe=$Exe
                Planner="apply-wide"; ForceLayout=[string]$layout
                ForceXC=[int]$x; ForceXG=[int]$x; ForceXP=[int]$x
                Threads=$Threads; Batch=512; UBatch=512; NPredict=1
                LogVerbosity=4
            }
            if (!$first) { $options.SkipProfile = $true }
            if ($NoMmap) { $options.NoMmap = $true }
            if ($DryRun) { $options.DryRun = $true }
            Write-Host "[MODEL_PROFILE] layout=$layout x=$x HYBRID_PROFILE=$first"
            & (Join-Path $PSScriptRoot "run_prompt_sweep.ps1") @options
            if ($LASTEXITCODE -ne 0) { throw "Probe failed layout=$layout x=$x" }
            $first = $false
        }
        $index++
    }
}
finally {
    foreach ($key in $envNames) { [Environment]::SetEnvironmentVariable($key, $old[$key], "Process") }
}
Write-Host "[MODEL_PROFILE] PC probe logs=$runs"
Write-Host "Copy this run's Phone RPC log to the PC; then use -Finalize -PhoneLog PATH"
