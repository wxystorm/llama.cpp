# GPU-only Qwen3 Region Ring: CUDA transformer layers + two streamed weight banks.
# Example (Qwen3-32B 64 layers, eight 8-layer windows):
# .\tools\region-ring-experiments\run_gpu_only_double_bank.ps1 -Model "E:\llama\models\Qwen3-32B-Q4_K_M.gguf" -Prompt ".\single-prompt-512\prompt_512.txt" -Layers 64 -LayersPerCycle 8
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Prompt,
    [string]$Exe = "",
    [string]$Out = ".\gpu-only-double-bank",
    [int]$Layers = 64,
    [int]$LayersPerCycle = 8,
    [int]$Chunk = 64,
    [int]$Batch = 2048,
    [int]$UBatch = 2048,
    [int]$Context = 4096,
    [int]$NPredict = 0,
    [int]$Threads = 8,
    [int]$LogVerbosity = 4,
    [switch]$NoMmap,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($Exe)) {
    $Exe = Join-Path $root "build-tensor-v2\bin\Release\llama-completion.exe"
}
if ($Layers -lt 2 -or $LayersPerCycle -le 0 -or $Chunk -le 0) {
    throw "Layers >= 2, LayersPerCycle > 0 and Chunk > 0 are required."
}
$cycleCount = [int][Math]::Ceiling($Layers / [double]$LayersPerCycle)
if ($cycleCount -lt 2 -or $cycleCount -gt 8) {
    throw "GPU-only mode requires 2..8 windows. Got $cycleCount; adjust -LayersPerCycle."
}
if ($UBatch -lt (2 * $Chunk) -or $Batch -lt $UBatch) {
    throw "Actual double-bank refill requires -UBatch >= 2 * -Chunk and -Batch >= -UBatch."
}
if ($NPredict -ne 0) {
    Write-Warning "Decode uses the smaller on-demand Bank path; this experiment verifies prefill double-bank streaming."
}
$layoutPieces = @()
$remaining = $Layers
while ($remaining -gt 0) {
    $take = [Math]::Min($LayersPerCycle, $remaining)
    $layoutPieces += "G$take"
    $remaining -= $take
}
$layout = $layoutPieces -join ","
$base = [IO.Path]::GetFullPath($Out)
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $base | Out-Null
}
$envNames = @(
    "LLAMA_HYBRID_REGION_RING_V1", "LLAMA_HYBRID_REGION_RING_EXEC",
    "LLAMA_HYBRID_REGION_RING_PLANNER", "LLAMA_HYBRID_REGION_RING_SKIP_PROFILE",
    "LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT", "LLAMA_HYBRID_REGION_RING_FORCE_XC",
    "LLAMA_HYBRID_REGION_RING_FORCE_XG", "LLAMA_HYBRID_REGION_RING_FORCE_XP",
    "LLAMA_HYBRID_REGION_RING_GPU_WEIGHT_STREAM", "LLAMA_HYBRID_REGION_RING_GPU_BANK_PEAK_RESERVE",
    "LLAMA_REGION_RING_TASK_TRACE", "LLAMA_REGION_RING_METRICS_FILE",
    "LLAMA_REGION_RING_COALESCE_TINY_TAIL", "LLAMA_REGION_RING_PHONE_HTP_ONLY",
    "LLAMA_REGION_RING_PHONE_GRAPHLESS", "LLAMA_REGION_RING_PHONE_AUTONOMOUS",
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL", "LLAMA_REGION_RING_PHONE_INPUT_PUSH",
    "LLAMA_REGION_RING_PHONE_INPUT_STREAM", "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM"
)
$before = @{}
foreach ($name in $envNames) {
    $before[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$overrides = @{
    LLAMA_HYBRID_REGION_RING_V1 = "1"
    LLAMA_HYBRID_REGION_RING_EXEC = "fifo-pc-phone"
    LLAMA_HYBRID_REGION_RING_PLANNER = "apply-wide"
    LLAMA_HYBRID_REGION_RING_SKIP_PROFILE = "1"
    LLAMA_HYBRID_REGION_RING_FORCE_LAYOUT = $layout
    LLAMA_HYBRID_REGION_RING_FORCE_XC = "$Chunk"
    LLAMA_HYBRID_REGION_RING_FORCE_XG = "$Chunk"
    LLAMA_HYBRID_REGION_RING_FORCE_XP = "$Chunk"
    LLAMA_HYBRID_REGION_RING_GPU_WEIGHT_STREAM = "1"
    LLAMA_HYBRID_REGION_RING_GPU_BANK_PEAK_RESERVE = "1"
    LLAMA_REGION_RING_TASK_TRACE = "1"
    LLAMA_REGION_RING_METRICS_FILE = (Join-Path $base "metrics.jsonl")
    LLAMA_REGION_RING_COALESCE_TINY_TAIL = "0"
}
$argsList = @(
    "-m", [IO.Path]::GetFullPath($Model),
    "--split-mode", "tensor", "--flash-attn", "on",
    "-c", "$Context", "-b", "$Batch", "-ub", "$UBatch",
    "-n", "$NPredict", "-t", "$Threads",
    "--no-conversation", "-f", [IO.Path]::GetFullPath($Prompt),
    "-lv", "$LogVerbosity"
)
if ($NoMmap) { $argsList += "--no-mmap" }
Write-Host "[GPU_ONLY] layout=$layout banks=A/B XG=$Chunk prefill_ubatch=$UBatch"
Write-Host "[GPU_ONLY] compute=CUDA0 layers=0..$($Layers - 1); CPU=embedding/head/weight-transfer"
Write-Host "[GPU_ONLY] engine=$Exe output=$base"
if ($DryRun) {
    Write-Host "[GPU_ONLY] dry-run: $Exe $($argsList -join ' ')"
    Write-Host "[GPU_ONLY] effective env:"
    $overrides.GetEnumerator() | Sort-Object Name | ForEach-Object { Write-Host "  $($_.Name)=$($_.Value)" }
    exit 0
}
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "Executable not found: $Exe" }
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model not found: $Model" }
if (-not (Test-Path -LiteralPath $Prompt -PathType Leaf)) { throw "Prompt not found: $Prompt" }
foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name, $null, "Process") }
foreach ($name in $overrides.Keys) { [Environment]::SetEnvironmentVariable($name, $overrides[$name], "Process") }
try {
    & $Exe @argsList 1> (Join-Path $base "output.stdout.txt") 2> (Join-Path $base "engine.stderr.log")
    $code = $LASTEXITCODE
} finally {
    foreach ($name in $envNames) {
        [Environment]::SetEnvironmentVariable($name, $before[$name], "Process")
    }
}
Write-Host "[GPU_ONLY] engine_exit=$code metrics=$(Join-Path $base 'metrics.jsonl')"
if ($code -eq 0) {
    & python (Join-Path $PSScriptRoot "verify_gpu_only_double_bank.py") --metrics (Join-Path $base "metrics.jsonl") --regions $cycleCount
    if ($LASTEXITCODE -ne 0) { exit 2 }
}
exit $code
