# Opt-in, fixed-layout whole-layer HTP+Adreno smoke for Qwen3-14B.
# This is deliberately not the auto 4-backend Planner.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [string]$Out = ".\phone-opencl-htp-mixed-smoke",
    [int]$Threads = 8,
    [switch]$NoMmap,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$modelFile = (Resolve-Path -LiteralPath $Model).Path
$prompt = (Resolve-Path -LiteralPath $PromptFile).Path
if ([IO.Path]::GetFileNameWithoutExtension($modelFile) -ne "Qwen3-14B-Q4_K_M") {
    throw "The initial fixed smoke only supports Qwen3-14B-Q4_K_M; do not apply its layer IDs to another model."
}
if ([IO.Path]::GetFileName($prompt) -notmatch '^prompt_[0-9]+\.txt$') {
    throw "Smoke requires a prompt_N.txt filename for the existing sweep runner."
}
$layout = "G7,C1,P1,G7,C1,P1,G7,C1,P1,G7,C1,P5"
$regions = "2,5"
$layers = "8,17"
Write-Host "[PHONE_MIXED_BACKENDS] expected_model=40-layer-Qwen3 backend_map=HTP0:$regions GPUOpenCL:8,11"
Write-Host "[PHONE_MIXED_BACKENDS] phone_server_requires:"
Write-Host "  GGML_RPC_PHONE_NPU_SERIAL_REGIONS=$regions"
Write-Host "  GGML_RPC_PHONE_NPU_MODEL_LOAD_LAYERS=$layers"
Write-Host "  GGML_RPC_PHONE_NPU_MIXED_OPENCL=1"
Write-Host "  GGML_RPC_PHONE_NPU_WEIGHT_BANKS=2"
$vars = @{
    "LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP" = "1"
    "LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS" = $regions
    "LLAMA_REGION_RING_PHONE_GRAPHLESS" = "0"
    "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW" = "1"
    "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES" = "1"
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS" = "1"
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL" = "1"
    "LLAMA_REGION_RING_PHONE_INPUT_PUSH" = "1"
    "LLAMA_REGION_RING_PHONE_INPUT_STREAM" = "1"
    "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM" = "1"
    "LLAMA_REGION_RING_PHONE_ASYNC_GRAPH_SETUP" = "0"
    "LLAMA_REGION_RING_REAL_STAGE_PROFILE" = "1"
    "LLAMA_REGION_RING_TASK_TRACE" = "1"
}
$old = @{}
foreach ($name in $vars.Keys) {
    $old[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$inputDir = Join-Path $repo ("phone-mixed-input-" + $stamp)
New-Item -ItemType Directory -Path $inputDir -Force | Out-Null
Copy-Item -LiteralPath $prompt -Destination (Join-Path $inputDir ([IO.Path]::GetFileName($prompt)))
try {
    foreach ($name in $vars.Keys) {
        [Environment]::SetEnvironmentVariable($name, $vars[$name], "Process")
    }
    $opts = @{
        Model = $modelFile
        Rpc = $Rpc
        PromptDir = $inputDir
        Out = $Out
        Planner = "apply-wide"
        SkipProfile = $true
        ForceLayout = $layout
        ForceXC = 64
        ForceXG = 64
        ForceXP = 64
        Batch = 512
        UBatch = 512
        NPredict = 1
        Threads = $Threads
        LogVerbosity = 4
    }
    if ($NoMmap) { $opts.NoMmap = $true }
    if ($DryRun) { $opts.DryRun = $true }
    & (Join-Path $PSScriptRoot "run_prompt_sweep.ps1") @opts
    if ($LASTEXITCODE -ne 0) { throw "Whole-layer HTP/OpenCL smoke failed; inspect the run's engine.stderr.log and Phone log." }
}
finally {
    foreach ($name in $vars.Keys) {
        [Environment]::SetEnvironmentVariable($name, $old[$name], "Process")
    }
    if (Test-Path -LiteralPath $inputDir) {
        Remove-Item -LiteralPath $inputDir -Recurse -Force
    }
}
