# Collect backend-specific REAL Stage timing curves from compact mixed probes.
# Feeder G/C/N/A Stages are necessary in current runtime. Only the target
# backend/depth contributes to the Stage CSV. This is NOT a standalone kernel
# benchmark and NEVER claims unsupported NPU multi-layer measurements.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Rpc,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [string]$G="1,2,4,6,8",
    [string]$C="1,2,3,4",
    [string]$N="1,2,3",
    [string]$A="1,2,4,6",
    [string]$Chunks="16,32,64,128",
    [ValidateRange(2,4)][int]$Cycles=3,
    [ValidateRange(1,10)][int]$Repeats=1,
    [string]$ProfileRoot="",
    [int]$Threads=8,
    [switch]$NoMmap,
    [switch]$DryRun
)
$ErrorActionPreference="Stop"
$repo=(Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$source=(Resolve-Path -LiteralPath $Model).Path
$prompt=(Resolve-Path -LiteralPath $PromptFile).Path
$python=(Get-Command python -ErrorAction Stop).Source
$builder=Join-Path $PSScriptRoot "gcna_probe_model.py"
$extractor=Join-Path $PSScriptRoot "build_real_stage_profile.py"
$stem=[IO.Path]::GetFileNameWithoutExtension($source)
if (!$ProfileRoot) { $ProfileRoot=Join-Path $repo "model-profiles\$stem" }
$profile=[IO.Path]::GetFullPath($ProfileRoot)
$matrix=Join-Path $profile "backend-matrix"
function Parse-Depth([string]$raw,[string]$label) {
    $out=@()
    foreach ($part in $raw.Split(',')) {
        $value=0
        if (![int]::TryParse($part.Trim(),[ref]$value) -or $value -le 0) {
            throw "Invalid depth list $label=$raw"
        }
        if ($value -notin $out) { $out+= $value }
    }
    return @($out | Sort-Object)
}
$depths=@{G=@(Parse-Depth $G "G");C=@(Parse-Depth $C "C");N=@(Parse-Depth $N "N");A=@(Parse-Depth $A "A")}
$widths=@(Parse-Depth $Chunks "Chunks")
foreach ($x in $widths) { if ($x -notin @(16,32,64,128)) { throw "Unsupported X=$x" } }
$probes=@()
$unsupported=@()
foreach ($backend in @("G","C","N","A")) {
    foreach ($depth in $depths[$backend]) {
        if ($backend -eq "N" -and $depth -gt 1) {
            $unsupported+=@{backend=$backend;layers=$depth;status="UNSUPPORTED";reason="HTP_P1_ONLY"}
            continue
        }
        if ($depth -gt 8) {
            $unsupported+=@{backend=$backend;layers=$depth;status="UNSUPPORTED";reason="COMPACT_STAGE_DEPTH_MAX_8"}
            continue
        }
        $g=1;$c=1;$a=1
        switch ($backend) { "G" {$g=$depth} "C" {$c=$depth} "A" {$a=$depth} }
        $pattern="G$g,C$c,N1,A$a"
        $desc=& $python $builder --describe --pattern $pattern --cycles $Cycles
        if ($LASTEXITCODE -ne 0) { throw "GCNA shape unsupported: $pattern" }
        $spec=($desc | Out-String | ConvertFrom-Json)
        $probes+=@{backend=$backend;layers=$depth;pattern=$pattern;spec=$spec}
    }
}
if ($probes.Count -eq 0) { throw "No supported probes" }
$manifest=@{
    schema=1;mode="TARGET_STAGE_FROM_REAL_MIXED_PIPELINE"
    standalone_compute=$false;source_model=$source;chunks=$widths
    repeats=$Repeats;cycles=$Cycles
    supported=@($probes | ForEach-Object { @{backend=$_.backend;layers=$_.layers;pattern=$_.pattern} })
    unsupported=$unsupported
}
if ($DryRun) { $manifest | ConvertTo-Json -Depth 8; return }
$tokenizer=Join-Path $repo "build-tensor-v2\bin\Release\llama-tokenize.exe"
if (!(Test-Path $tokenizer)) { throw "Build llama-tokenize.exe first." }
$tok=& $tokenizer -m $source -f $prompt --no-escape --show-count --log-disable
if ($LASTEXITCODE -ne 0) { throw "Tokenizer failed" }
$match=[regex]::Match(($tok -join [Environment]::NewLine),'Total number of tokens:\s*(\d+)')
if (!$match.Success) { throw "Tokenizer did not report token count." }
$ntokens=[int]$match.Groups[1].Value
New-Item -ItemType Directory -Force -Path $matrix | Out-Null
$vars=@{
    "LLAMA_REGION_RING_PHONE_MIXED_OPENCL_HTP"="1"
    "LLAMA_REGION_RING_PHONE_GRAPHLESS"="0"
    "LLAMA_REGION_RING_PHONE_OWNED_PLAN_SHADOW"="1"
    "LLAMA_REGION_RING_PHONE_STATIC_TEMPLATES"="1"
    "GGML_META_PHONE_FULL_REGION_GRAPH"="1"
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS"="1"
    "LLAMA_REGION_RING_PHONE_AUTONOMOUS_FINAL"="1"
    "LLAMA_REGION_RING_PHONE_INPUT_PUSH"="1"
    "LLAMA_REGION_RING_PHONE_INPUT_STREAM"="1"
    "LLAMA_HYBRID_REGION_RING_PHONE_RETURN_STREAM"="1"
    "LLAMA_REGION_RING_PHONE_ASYNC_GRAPH_SETUP"="0"
    "LLAMA_REGION_RING_REAL_STAGE_PROFILE"="1"
    "LLAMA_REGION_RING_PHONE_PROFILE_RPC"="1"
    "LLAMA_REGION_RING_TASK_TRACE"="1"
}
$old=@{}
$oldNpu=[Environment]::GetEnvironmentVariable("LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS","Process")
foreach ($key in $vars.Keys) { $old[$key]=[Environment]::GetEnvironmentVariable($key,"Process") }
$failures=@()
try {
    foreach ($key in $vars.Keys) { [Environment]::SetEnvironmentVariable($key,$vars[$key],"Process") }
    foreach ($p in $probes) {
        $id="backend-$($p.backend)-L$($p.layers)"
        $folder=Join-Path $matrix $id
        $probe=Join-Path $folder "$stem-$id.gguf"
        $probeManifest=[IO.Path]::ChangeExtension($probe,".probe.json")
        $inputDir=Join-Path $folder "input"
        $runs=Join-Path $folder "runs"
        New-Item -ItemType Directory -Force -Path $folder,$inputDir,$runs | Out-Null
        $buildArgs=@($builder,"--model",$source,"--output",$probe,"--manifest",$probeManifest,"--pattern",$p.pattern,"--cycles","$Cycles")
        if (Test-Path $probe) { $buildArgs+="--verify" }
        & $python @buildArgs
        if ($LASTEXITCODE -ne 0) { throw "Probe build/verify failed: $id" }
        Copy-Item -LiteralPath $prompt -Destination (Join-Path $inputDir "prompt_$ntokens.txt") -Force
        $env:LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS=(@($p.spec.npu_regions | ForEach-Object { [string]$_ }) -join ",")
        foreach ($repeat in 1..$Repeats) {
            foreach ($x in $widths) {
                $out=Join-Path $runs "repeat$repeat-x$x"
                $pass=@(Get-ChildItem -LiteralPath $out -Filter result.json -Recurse -File -ErrorAction SilentlyContinue | Where-Object {
                    try { (Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json).status -eq "PASS" } catch { $false }
                })
                if ($pass.Count -gt 0) {
                    Write-Host "[BACKEND_PROFILE] skip=PASS backend=$($p.backend) layers=$($p.layers) X=$x repeat=$repeat"
                    continue
                }
                Write-Host "[BACKEND_PROFILE] run backend=$($p.backend) layers=$($p.layers) X=$x repeat=$repeat"
                $sweep=@{
                    Model=$probe;Rpc=$Rpc;PromptDir=$inputDir;Out=$out
                    Planner="apply-wide";SkipProfile=$true;ForceLayout=[string]$p.spec.runtime_layout
                    ForceXC=$x;ForceXG=$x;ForceXP=$x;Batch=512;UBatch=512;NPredict=1
                    Threads=$Threads;NoMmap=[bool]$NoMmap;LogVerbosity=4
                }
                & (Join-Path $PSScriptRoot "run_prompt_sweep.ps1") @sweep
                if ($LASTEXITCODE -ne 0) { $failures+="$id X=$x repeat=$repeat" }
            }
        }
    }
} finally {
    foreach ($key in $vars.Keys) { [Environment]::SetEnvironmentVariable($key,$old[$key],"Process") }
    [Environment]::SetEnvironmentVariable("LLAMA_REGION_RING_PHONE_NPU_SERIAL_REGIONS",$oldNpu,"Process")
}
$manifest.failures=$failures
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $matrix "matrix_manifest.json") -Encoding utf8
if ($failures.Count) { throw "Incomplete matrix runs: $($failures -join '; '); retry to resume." }
$csv=Join-Path $profile "real-stage-mixed.csv"
& $python $extractor --run $matrix --mixed --target-only --output $csv
if ($LASTEXITCODE -ne 0) { throw "Target Stage extraction failed." }
Write-Host "[BACKEND_PROFILE] status=COMPLETE csv=$csv; N2/N3 remain UNSUPPORTED"
