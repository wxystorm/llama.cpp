# Real four-backend compact Stage Profile. No manual NPU Region IDs and no
# copying Android logs. Runs only the sampled GGUF Transformer layers.
# Phone Patch33 RPC Server loads the original full GGUF for local HTP weights.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [string]$Pattern = "G7,C1,N1,A1",
    [ValidateRange(2,4)][int]$Cycles = 3,
    [ValidateSet("16,32,64,128", "16,32,64", "32,64,128", "64")][string]$Chunks = "16,32,64,128",
    [string]$ProfileRoot = "",
    [string]$V2ProfileJson = "",
    [int]$Threads = 8,
    [switch]$NoMmap,
    [switch]$UpdateCalibration,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$source = (Resolve-Path -LiteralPath $Model).Path
$prompt = (Resolve-Path -LiteralPath $PromptFile).Path
$python = (Get-Command python -ErrorAction Stop).Source
$builder = Join-Path $PSScriptRoot "gcna_probe_model.py"
$extractor = Join-Path $PSScriptRoot "build_real_stage_profile.py"
$stem = [IO.Path]::GetFileNameWithoutExtension($source)
if (!$ProfileRoot) { $ProfileRoot = Join-Path $repo "model-profiles\$stem" }
$folder = [IO.Path]::GetFullPath($ProfileRoot)
New-Item -ItemType Directory -Force -Path $folder | Out-Null

$specText = & $python $builder --describe --pattern $Pattern --cycles $Cycles
if ($LASTEXITCODE -ne 0) { throw "Invalid G/C/N/A profile pattern." }
$spec = ($specText | Out-String | ConvertFrom-Json)
$label = "gcna-$($spec.pattern.Replace(',', '-'))-c$Cycles"
$probeDir = Join-Path $folder "compact-probes\$label"
$probeModel = Join-Path $probeDir ("$stem-$label.gguf")
$manifestPath = [IO.Path]::ChangeExtension($probeModel, ".probe.json")
$runs = Join-Path $probeDir "runs"
$inputDir = Join-Path $probeDir "input"
New-Item -ItemType Directory -Force -Path $probeDir, $inputDir, $runs | Out-Null

if (!(Test-Path -LiteralPath $probeModel)) {
    if ($DryRun) {
        Write-Host "[GCNA_PROFILE] dry_run=1 compact_model=$probeModel (not yet generated)"
        return
    }
    $buildArgs = @($builder, "--model", $source, "--output", $probeModel,
                   "--pattern", $Pattern, "--cycles", "$Cycles",
                   "--manifest", $manifestPath)
    & $python @buildArgs
    if ($LASTEXITCODE -ne 0) { throw "Compact GGUF creation failed." }
} else {
    if (!(Test-Path -LiteralPath $manifestPath)) {
        throw "Compact model exists without a fingerprint manifest; refuse reuse."
    }
    $verifyArgs = @($builder, "--model", $source, "--output", $probeModel,
                    "--pattern", $Pattern, "--cycles", "$Cycles",
                    "--manifest", $manifestPath, "--verify")
    & $python @verifyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Existing compact GGUF fingerprint mismatch; refuse stale model cache."
    }
}
$tokenizer = Join-Path $repo "build-tensor-v2\bin\Release\llama-tokenize.exe"
if (!(Test-Path $tokenizer)) { throw "Build llama-tokenize first." }
$toks = & $tokenizer -m $probeModel -f $prompt --no-escape --show-count --log-disable
if ($LASTEXITCODE -ne 0) { throw "Tokenizer failed on compact Qwen3 model." }
$m = [regex]::Match(($toks -join [Environment]::NewLine), 'Total number of tokens:\s*(\d+)')
if (!$m.Success) { throw "Could not determine actual token count." }
$tokens = [int]$m.Groups[1].Value
$input = Join-Path $inputDir "prompt_$tokens.txt"
Copy-Item -LiteralPath $prompt -Destination $input -Force

# Runtime still uses G/C/P. The G/C/N/A descriptor determines mapping,
# Region IDs and absolute HTP layers; none are operator-entered.
$npuRegions = (@($spec.npu_regions | ForEach-Object { [string]$_ })) -join ","
$layout = [string]$spec.runtime_layout
Write-Host "[GCNA_PROFILE] stages=$($spec.pattern) cycles=$Cycles compact_layers=$($spec.layers)"
Write-Host "[GCNA_PROFILE] runtime_layout=$layout auto_htp_regions=$npuRegions"
Write-Host "[GCNA_PROFILE] Phone must load ORIGINAL GGUF $([IO.Path]::GetFileName($source))"
Write-Host "[GCNA_PROFILE] Only sampled first $($spec.layers) layers run on PC; no 40-layer CPU filler."
Write-Host "[GCNA_PROFILE] Phone samples returned via Patch33 RPC; DO NOT copy phone.log."

$vars = @{
    "LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP" = "1"
    "LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS" = $npuRegions
    "LLAMA_REGION_RING_PHONE_GRAPHLESS" = "0"
    "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW" = "1"
    "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES" = "1"
    "GGML_META_PHONE_FULL_REGION_GRAPH" = "1"
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS" = "1"
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL" = "1"
    "LLAMA_REGION_RING_PHONE_INPUT_PUSH" = "1"
    "LLAMA_REGION_RING_PHONE_INPUT_STREAM" = "1"
    "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM" = "1"
    "LLAMA_REGION_RING_PHONE_ASYNC_GRAPH_SETUP" = "0"
    "LLAMA_REGION_RING_REAL_STAGE_PROFILE" = "1"
    "LLAMA_REGION_RING_PHONE_PROFILE_RPC" = "1"
    "LLAMA_REGION_RING_TASK_TRACE" = "1"
}
$old = @{}
foreach ($key in $vars.Keys) {
    $old[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
}
try {
    foreach ($key in $vars.Keys) {
        [Environment]::SetEnvironmentVariable($key, $vars[$key], "Process")
    }
    foreach ($chunk in ($Chunks.Split(',') | ForEach-Object { [int]$_ })) {
        $output = Join-Path $runs "x$chunk"
        Write-Host "[GCNA_PROFILE] phase=RUN chunk=$chunk layout=$layout"
        $p = @{
            Model = $probeModel; Rpc = $Rpc; PromptDir = $inputDir
            Out = $output; Planner = "apply-wide"; SkipProfile = $true
            ForceLayout = $layout; ForceXC = $chunk
            ForceXG = $chunk; ForceXP = $chunk
            Batch = 512; UBatch = 512; NPredict = 1
            Threads = $Threads; LogVerbosity = 4
            NoMmap = [bool]$NoMmap; DryRun = [bool]$DryRun
        }
        & (Join-Path $PSScriptRoot "run_prompt_sweep.ps1") @p
        if ($LASTEXITCODE -ne 0) { throw "Profile probe failed at X=$chunk." }
    }
}
finally {
    foreach ($key in $vars.Keys) {
        [Environment]::SetEnvironmentVariable($key, $old[$key], "Process")
    }
}
if ($DryRun) { return }
$outCsv = Join-Path $folder "real-stage-mixed.csv"
& $python $extractor --run $runs --mixed --output $outCsv
if ($LASTEXITCODE -ne 0) {
    throw "Could not assemble four-backend Profile from PC logs/RPC samples."
}
$baseManifest = Join-Path $folder "model_profile.json"
$calibration = Join-Path $folder "phone_mixed_calibration.json"
if (Test-Path -LiteralPath $baseManifest) {
    if ((Test-Path -LiteralPath $calibration) -and !$UpdateCalibration) {
        Write-Warning "Existing phone_mixed_calibration.json preserved. Pass -UpdateCalibration to refresh from measured Patch32 RPC data."
    } else {
        $calibrationArgs = @(
            (Join-Path $PSScriptRoot "build_gcna_calibration.py"),
            "--model", $source, "--profile-dir", $folder,
            "--run", $runs, "--probe-manifest", $manifestPath,
            "--stage-csv", $outCsv, "--output", $calibration
        )
        if ($UpdateCalibration) { $calibrationArgs += "--update" }
        & $python @calibrationArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Measured Phone calibration generation failed; check RPC Bank REFILL and device memory samples."
        }
    }
} else {
    Write-Warning "model_profile.json (full Qwen3 base Profile) is absent. Four-backend Stage CSV is ready, but final mixed calibration/Plan requires this original-model base Profile; no synthetic base was created."
}
Write-Host "[GCNA_PROFILE] status=COMPLETE csv=$outCsv manifest=$manifestPath"
Write-Host "[GCNA_PROFILE] phone_calibration=$calibration"
if ($V2ProfileJson) {
    $v2Profile = (Resolve-Path -LiteralPath $V2ProfileJson).Path
    $merged = Join-Path $folder "profile-with-runtime.json"
    $mergeCmd = Join-Path $PSScriptRoot "merge_v2_runtime_profile.py"
    & $python $mergeCmd --profile $v2Profile --pc-log $runs --out $merged
    if ($LASTEXITCODE -ne 0) {
        throw "Could not import Patch33 real Bank/handoff telemetry from PC logs."
    }
    Write-Host "[GCNA_PROFILE] v2_runtime_profile=$merged"
}
Write-Host "[GCNA_PROFILE] No Android phone log was copied."
