# Multi-layer Hexagon Q/K/V correctness isolation. Requires Windows PowerShell 5.1+.
# Run from the repository root. Not for performance measurement: tensor callbacks
# synchronously capture intermediate data and change GGML graph split boundaries.
#
# Phone RPC server (restart before test):
#   export GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC=1
#   export GGML_HEXAGON_QKV_AUDIT=1
#   export GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1
#   export GGML_HEXAGON_OPFUSION=0
#   unset GGML_HEXAGON_KV14_SET_ROWS_FORCE_INPUT_FLUSH
#   unset GGML_HEXAGON_KV14_SET_ROWS_DIAG
#   unset GGML_HEXAGON_PROFILE
#   unset GGML_HEXAGON_SET_ROWS_SYNC_TIMING
#
# Audit actual NPU placement in phone logs:
#   grep 'HEX_QKV_AUDIT.*weight=blk' rpc-multilayer.log
# A matching tensor result without a phone QKV AUDIT line is NOT proof of HTP
# execution; the planner/scheduler may have fallen back to CPU.

param(
    [ValidateRange(1, 512)] [int] $Tokens = 1,
    [ValidateRange(1, 128)] [int] $ModelLayers = 16,
    [ValidateRange(1, 128)] [int] $Ngl = 4,
    [ValidateRange(0, 127)] [int] $StartLayer = 13,
    [ValidateRange(0, 127)] [int] $EndLayer = 15,
    [string[]] $Cases = @("13-k", "14-k", "15-k", "14-qkv", "all-qkv"),
    [switch] $All,
    [switch] $IncludeFFN,
    [switch] $Overwrite,
    [string] $Exe = ".\build-hexagon-pc\bin\Release\llama-cli.exe",
    [string] $Model = "D:\model\Llama-3.2-1B-Instruct-Q4_0.gguf",
    [string] $Rpc = "10.39.52.114:50052",
    [string] $Prompt = "Explain what a computer is.",
    [string] $OutDir = ".\hexagon-qkv-validation"
)

$ErrorActionPreference = "Stop"
if ($StartLayer -gt $EndLayer) { throw "StartLayer must be <= EndLayer" }
if ($EndLayer -ge $ModelLayers) { throw "EndLayer must be below ModelLayers" }
if ($Ngl -gt $ModelLayers) { throw "Ngl must be <= ModelLayers" }
# In this llama-model.cpp, -ngl includes the output layer in the budget.
# First block on RPC = ModelLayers + 1 - Ngl (not ModelLayers - Ngl).
$firstRemoteLayer = [Math]::Max(0, $ModelLayers + 1 - $Ngl)
if ($StartLayer -lt $firstRemoteLayer) {
    $requiredNgl = $ModelLayers + 1 - $StartLayer
    throw "Layer $StartLayer is CPU at -ngl $Ngl. The first remote layer is $firstRemoteLayer; use at least -ngl $requiredNgl (includes output layer)."
}
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "Cannot find CLI: $Exe" }
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Cannot find model: $Model" }
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' is missing. Install Python and ensure py -3 works."
}

$Exe = (Resolve-Path -LiteralPath $Exe).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path

$layerRange = @($StartLayer..$EndLayer)
$layerRegex = $layerRange -join "|"
Write-Host "[QKV_OFFLOAD_RANGE] ngl=$Ngl model_layers=$ModelLayers first_remote_layer=$firstRemoteLayer test_range=$StartLayer..$EndLayer"

if ($All) {
    $Cases = @()
    foreach ($layer in $layerRange) {
        foreach ($role in @("q", "k", "v", "qkv")) {
            $Cases += ("{0}-{1}" -f $layer, $role)
        }
    }
    $Cases += "all-qkv"
}
if ($Cases.Count -eq 0) { throw "No test cases requested." }
$Cases = @($Cases | Select-Object -Unique)

function Get-EnabledWeights([string] $caseName) {
    $enabled = @{}
    if ($caseName -eq "cpu") { return $enabled }
    if ($caseName -eq "all-qkv") {
        foreach ($layer in $layerRange) {
            foreach ($role in @("q", "k", "v")) {
                $enabled["$layer-$role"] = $true
            }
        }
        return $enabled
    }
    if ($caseName -match '^(\d+)-(qkv|q|k|v)$') {
        $layer = [int] $Matches[1]
        $selection = $Matches[2]
    } else {
        throw "Invalid case '$caseName'. Use L-q, L-k, L-v, L-qkv or all-qkv."
    }
    if ($layer -lt $StartLayer -or $layer -gt $EndLayer) {
        throw "Case '$caseName' is outside [$StartLayer,$EndLayer]."
    }
    foreach ($role in @("q", "k", "v")) {
        if ($selection -eq "qkv" -or $selection -eq $role) {
            $enabled["$layer-$role"] = $true
        }
    }
    return $enabled
}

# Validate case list before launching a potentially expensive model.
foreach ($caseName in $Cases) {
    $null = Get-EnabledWeights $caseName
}

$env:GGML_RPC_MATMUL_ONLY = "1"
$env:GGML_RPC_DISABLE_F16_MATMUL = "1"
Remove-Item Env:GGML_SCHED_KV14_READBACK -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_TENSOR_COMPARE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_TENSOR_COMPARE_REGEX -ErrorAction SilentlyContinue

$stagePattern = 'Qcur|Kcur|Vcur|kq|kq_soft_max|kqv|kqv_out|attn_out|l_out'
if ($IncludeFFN) {
    $stagePattern += '|ffn_inp|ffn_norm|ffn_up|ffn_gate|ffn_swiglu|ffn_down|ffn_out'
}
$env:LLAMA_TENSOR_COMPARE_REGEX = '^(' + $stagePattern + ')-(' + $layerRegex + ')$'

$settings = @(
    "model=$Model", "exe=$Exe", "rpc=$Rpc", "model_layers=$ModelLayers",
    "ngl=$Ngl", "range=$StartLayer..$EndLayer", "tokens=$Tokens",
    "prompt=$Prompt", "regex=$env:LLAMA_TENSOR_COMPARE_REGEX",
    "GGML_RPC_MATMUL_ONLY=$env:GGML_RPC_MATMUL_ONLY",
    "GGML_RPC_DISABLE_F16_MATMUL=$env:GGML_RPC_DISABLE_F16_MATMUL",
    "CAUTION=Phone configuration must be checked independently; this script cannot change phone env."
)
Set-Content -LiteralPath (Join-Path $OutDir "settings.txt") -Value $settings -Encoding UTF8

function Run-Capture([string] $caseName) {
    $enabled = Get-EnabledWeights $caseName
    $folder = Join-Path $OutDir $caseName
    if (Test-Path -LiteralPath $folder) {
        if (-not $Overwrite) {
            throw "Output $folder exists. Use -Overwrite to replace existing snapshots."
        }
        Remove-Item -LiteralPath $folder -Recurse -Force
    }
    New-Item -ItemType Directory -Path $folder | Out-Null
    $env:LLAMA_TENSOR_COMPARE_DIR = $folder

    $rules = @(
        ('blk\.({0})\.(attn_output|ffn_(gate|up|down))\.weight=CPU' -f $layerRegex)
    )
    foreach ($layer in $layerRange) {
        foreach ($role in @("q", "k", "v")) {
            if (-not $enabled.ContainsKey("$layer-$role")) {
                $rules += ('blk\.{0}\.attn_{1}\.weight=CPU' -f $layer, $role)
            }
        }
    }
    $env:LLAMA_ARG_OVERRIDE_TENSOR = $rules -join ","

    $expected = @($enabled.Keys | Sort-Object | ForEach-Object {
        $pieces = $_ -split "-"
        "blk.$($pieces[0]).attn_$($pieces[1]).weight"
    })
    $notes = @(
        "case=$caseName",
        "target_NPU_weights=$($expected -join ',')",
        "override_tensor=$env:LLAMA_ARG_OVERRIDE_TENSOR",
        "filter=$env:LLAMA_TENSOR_COMPARE_REGEX",
        "NOTE=Check [HEX_QKV_AUDIT] phone log to confirm NPU execution."
    )
    Set-Content -LiteralPath (Join-Path $folder "placement.txt") -Value $notes -Encoding UTF8

    Write-Host "--- Case: $caseName ---"
    Write-Host "Expected HTP QKV weights: $($expected -join ', ')"
    $log = Join-Path $folder "run.log"
    $cliArgs = @(
        "-m", $Model, "--rpc", $Rpc, "--device", "RPC0",
        "-sm", "layer", "-ngl", "$Ngl", "-fa", "off",
        "--kv-offload", "--cache-type-k", "f16", "--cache-type-v", "f16",
        "-c", "512", "-n", "$Tokens", "-t", "8",
        "--no-warmup", "--single-turn", "--no-conversation",
        "--temp", "0", "--seed", "1234",
        "--perf", "-lv", "3", "-p", $Prompt
    )
    $ErrorActionPreference = "Continue"
    # Consume the pipeline so Run-Capture returns only the folder path.
    & $Exe @cliArgs 2>&1 | Tee-Object -FilePath $log | Out-Host
    $exit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($exit -ne 0) { throw "$caseName exit code $exit; see $log" }

    $manifest = Join-Path $folder "manifest.tsv"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "$caseName did not create manifest.tsv. Check run.log for TENSOR_COMPARE."
    }
    $snapshots = @((Import-Csv -LiteralPath $manifest -Delimiter ([char] 9)))
    if ($snapshots.Count -eq 0) {
        throw "$caseName created an empty tensor manifest: $manifest"
    }
    Write-Host "$caseName captures=$($snapshots.Count)"
    return $folder
}

$cpu = Run-Capture "cpu"
$overview = @()
$summaryScript = Join-Path $PSScriptRoot "summarize_hexagon_qkv_layers.py"
if (-not (Test-Path -LiteralPath $summaryScript)) {
    throw "Missing script $summaryScript (update the branch before running)."
}
foreach ($caseName in $Cases) {
    $candidate = Run-Capture $caseName
    $output = Join-Path $candidate "compare"
    Write-Host "--- Comparing $caseName with CPU reference ---"
    & py -3 $summaryScript $cpu $candidate --case $caseName --output $output
    $compareExit = $LASTEXITCODE
    if ($compareExit -ne 0 -and $compareExit -ne 2 -and $compareExit -ne 3) {
        throw "Python summarizer exited with $compareExit for $caseName"
    }
    $report = Get-Content -LiteralPath (Join-Path $output "summary.json") -Raw | ConvertFrom-Json
    $overview += [pscustomobject]@{
        case = $caseName
        status = $report.status
        compared = $report.compared
        only_cpu = $report.only_cpu
        only_npu = $report.only_npu
        critical = $report.critical
        diff = $report.diff
        min_cosine = $report.min_cosine
        max_rmse = $report.max_rmse
    }
}

$path = Join-Path $OutDir "overview.csv"
$overview | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
Write-Host "===== Multi-layer QKV results ====="
$overview | Format-Table -AutoSize
Write-Host "Overview CSV: $path"
Write-Host "Each case has tensor_metrics.csv, summary.json and placement.txt."
Write-Host "CRITICAL or CAPTURE_MISMATCH must be investigated, not treated as success."
Write-Host "NUMERICALLY_CLOSE requires confirming expected HTP weight execution in phone [HEX_QKV_AUDIT] logs."
