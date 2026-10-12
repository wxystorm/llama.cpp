# Stage-by-stage Hexagon RPC Attention-Output / FFN numerical correctness test.
# Requires Windows PowerShell 5.1+ and a running HTP0 RPC server.
# This is NOT a performance benchmark: tensor capture can change graph splits.
#
# Stage 'attn': baseline = Q/K/V on NPU; attn_output + FFN on CPU.
#                case = selected attn_output MatMuls moved to NPU.
# Stage 'ffn':  baseline = Q/K/V + attn_output on NPU; FFN on CPU.
#                case = selected gate/up/down MatMuls moved to NPU.
#
# In both stages, remote K/V cache remains enabled on both baseline and case.
# First validate attn before running ffn. Phone must enable the independent
# GGML_HEXAGON_LINEAR_AUDIT=1 to prove the target weight ran on HTP.
param(
    [ValidateSet("attn", "ffn")] [string] $Stage = "attn",
    [ValidateRange(1, 128)] [int] $ModelLayers = 16,
    [ValidateRange(1, 128)] [int] $Ngl = 4,
    [ValidateRange(0, 127)] [int] $StartLayer = 13,
    [ValidateRange(0, 127)] [int] $EndLayer = 15,
    [ValidateRange(1, 64)] [int] $Tokens = 1,
    [string[]] $Cases = @(),
    [switch] $All,
    [switch] $Overwrite,
    [switch] $RecompareOnly,
    [switch] $BaselineControl,
    [ValidateRange(-1, 127)] [int] $FocusLayer = -1,
    [string] $Exe = ".\build-hexagon-pc\bin\Release\llama-cli.exe",
    [string] $Model = "D:\model\Llama-3.2-1B-Instruct-Q4_0.gguf",
    [string] $Rpc = "10.39.52.114:50052",
    [string] $Prompt = "Explain what a computer is.",
    [string] $OutDir = ".\hexagon-linear-validation"
)

$ErrorActionPreference = "Stop"
if ($StartLayer -gt $EndLayer -or $EndLayer -ge $ModelLayers) {
    throw "Layer range $StartLayer..$EndLayer invalid for $ModelLayers-layer model"
}
if ($Ngl -gt ($ModelLayers + 1)) {
    throw "-Ngl $Ngl exceeds model layers + output layer ($($ModelLayers + 1))"
}
# Current llama-model.cpp counts the output layer as one of -ngl layers.
$firstRemoteLayer = [Math]::Max(0, $ModelLayers + 1 - $Ngl)
if ($StartLayer -lt $firstRemoteLayer) {
    $requiredNgl = $ModelLayers + 1 - $StartLayer
    throw "Layer $StartLayer is on CPU with -Ngl $Ngl. Use at least -Ngl $requiredNgl."
}
if ($StartLayer -gt $firstRemoteLayer) {
    Write-Warning "Layers $firstRemoteLayer..$($StartLayer-1) are on RPC but not controlled by this script."
}
if (-not $RecompareOnly) {
    if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
        throw "llama-cli not found: $Exe"
    }
    if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
        throw "Model not found: $Model"
    }
    $Exe = (Resolve-Path -LiteralPath $Exe).Path
    $Model = (Resolve-Path -LiteralPath $Model).Path
}
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Missing Python launcher (py -3)."
}
$layerRange = @($StartLayer..$EndLayer)
$layerRegex = $layerRange -join "|"
if ($RecompareOnly -and -not (Test-Path -LiteralPath $OutDir -PathType Container)) {
    throw "Existing output directory not found for recompare: $OutDir"
}
if (-not $RecompareOnly) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}
$OutDir = Join-Path (Resolve-Path -LiteralPath $OutDir).Path $Stage
if ($RecompareOnly) {
    if (-not (Test-Path -LiteralPath $OutDir -PathType Container)) {
        throw "Existing stage directory not found: $OutDir"
    }
} else {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}
$summaryScript = Join-Path $PSScriptRoot "summarize_hexagon_linear_layers.py"
if (-not (Test-Path -LiteralPath $summaryScript -PathType Leaf)) {
    throw "Summarizer missing: $summaryScript"
}

# A single case toggles only its target output/gate/up/down weights; there
# are no CPU-vs-NPU QKV changes within the same Stage.
function Get-EnabledWeights([string] $caseName) {
    $enabled = @{}
    if ($caseName -in @("baseline", "baseline-repeat")) { return $enabled }

    if ($Stage -eq "attn") {
        if ($caseName -eq "all-output") {
            foreach ($layer in $layerRange) {
                $enabled["${layer}-output"] = $true
            }
            return $enabled
        }
        if ($caseName -match '^(\d+)-output$') {
            $layer = [int] $Matches[1]
            $role = "output"
        } else {
            throw "Invalid attn case '$caseName'. Use N-output or all-output."
        }
        if ($layer -lt $StartLayer -or $layer -gt $EndLayer) {
            throw "Case '$caseName' outside selected layers $StartLayer..$EndLayer"
        }
        $enabled["${layer}-${role}"] = $true
        return $enabled
    }

    if ($caseName -eq "all-ffn") {
        foreach ($layer in $layerRange) {
            foreach ($role in @("gate", "up", "down")) {
                $enabled["${layer}-${role}"] = $true
            }
        }
        return $enabled
    }
    if ($caseName -match '^(\d+)-(gate|up|down|ffn)$') {
        $layer = [int] $Matches[1]
        $kind = $Matches[2]
    } else {
        throw "Invalid FFN case '$caseName'. Use N-gate/N-up/N-down/N-ffn or all-ffn."
    }
    if ($layer -lt $StartLayer -or $layer -gt $EndLayer) {
        throw "Case '$caseName' outside selected layers $StartLayer..$EndLayer"
    }
    foreach ($role in @("gate", "up", "down")) {
        if ($kind -eq "ffn" -or $kind -eq $role) {
            $enabled["${layer}-${role}"] = $true
        }
    }
    return $enabled
}

if ($RecompareOnly -and $Cases.Count -eq 0) {
    $Cases = @(Get-ChildItem -LiteralPath $OutDir -Directory |
                Where-Object { $_.Name -ne "baseline" -and
                               $_.Name -ne "baseline-repeat" } |
                ForEach-Object { $_.Name })
}
if ($Cases.Count -eq 0) {
    if ($Stage -eq "attn") {
        $Cases = @($layerRange | ForEach-Object { "$($_)-output" })
        $Cases += "all-output"
    } else {
        $Cases = @("${StartLayer}-gate", "${StartLayer}-up", "${StartLayer}-down",
                   "${StartLayer}-ffn", "all-ffn")
    }
}
if ($All) {
    $Cases = @()
    foreach ($layer in $layerRange) {
        if ($Stage -eq "attn") {
            $Cases += "${layer}-output"
        } else {
            foreach ($role in @("gate", "up", "down", "ffn")) {
                $Cases += "${layer}-${role}"
            }
        }
    }
    $Cases += $(if ($Stage -eq "attn") { "all-output" } else { "all-ffn" })
}
$Cases = @($Cases | Select-Object -Unique)
foreach ($caseName in $Cases) { $null = Get-EnabledWeights $caseName }
if ($Cases.Count -eq 0) { throw "No cases to run" }
if ($FocusLayer -ge 0 -and $Cases.Count -ne 1) {
    throw "-FocusLayer requires exactly one selected case; otherwise each case determines its own target layer."
}
if ($BaselineControl -and $RecompareOnly -and
    -not (Test-Path -LiteralPath (Join-Path $OutDir "baseline-repeat\manifest.tsv"))) {
    throw "Saved baseline-repeat/manifest.tsv missing. -BaselineControl with -RecompareOnly needs an existing control capture."
}

if (-not $RecompareOnly) {
$env:GGML_RPC_MATMUL_ONLY = "1"
$env:GGML_RPC_DISABLE_F16_MATMUL = "1"
Remove-Item Env:GGML_SCHED_KV14_READBACK -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_TENSOR_COMPARE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_TENSOR_COMPARE_REGEX -ErrorAction SilentlyContinue

$stages = "attn_norm|Qcur|Kcur|Vcur|kq|kq_soft_max|kqv|kqv_out|" +
          "attn_out|ffn_inp|ffn_norm|ffn_up|ffn_gate|ffn_swiglu|" +
          "ffn_down|ffn_out|l_out"
$env:LLAMA_TENSOR_COMPARE_REGEX = "^($stages)-($layerRegex)$"
$settings = @(
    "stage=$Stage", "exe=$Exe", "model=$Model", "rpc=$Rpc",
    "model_layers=$ModelLayers", "ngl=$Ngl",
    "layer_range=$StartLayer..$EndLayer", "tokens=$Tokens",
    "prompt=$Prompt", "filter=$env:LLAMA_TENSOR_COMPARE_REGEX",
    "cases=$($Cases -join ',')",
    "baseline=QKV_NPU;ATTN_OUTPUT_CPU+FFN_CPU if attn, ATTN_OUTPUT_NPU+FFN_CPU if ffn",
    "GGML_RPC_MATMUL_ONLY=$env:GGML_RPC_MATMUL_ONLY",
    "GGML_RPC_DISABLE_F16_MATMUL=$env:GGML_RPC_DISABLE_F16_MATMUL",
    "phone_needs_GGML_HEXAGON_LINEAR_AUDIT=1",
    "warning=tensor_dump_changes_graph_scheduling"
)
Set-Content -LiteralPath (Join-Path $OutDir "settings.txt") -Value $settings -Encoding UTF8
Write-Host "[HEX_LINEAR_TEST] stage=$Stage ngl=$Ngl first_remote=$firstRemoteLayer layers=$StartLayer..$EndLayer"
Write-Host "Baseline: QKV is on RPC/HTP (all stages). Attention output is on CPU only during 'attn' stage."
Write-Host "Validate [HEX_LINEAR_AUDIT] on phone for every target weight."
} else {
    Write-Host "[HEX_LINEAR_RECOMPARE] Reusing saved tensor snapshots in $OutDir; no CLI/RPC run"
}

function Run-Capture([string] $caseName) {
    $enabled = Get-EnabledWeights $caseName
    $folder = Join-Path $OutDir $caseName
    if (Test-Path -LiteralPath $folder) {
        if (-not $Overwrite) {
            throw "Existing case $folder. Use -Overwrite to replace (or new -OutDir)."
        }
        Remove-Item -LiteralPath $folder -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $folder | Out-Null

    $rules = @()
    foreach ($layer in $layerRange) {
        # Stage 'ffn' intentionally leaves ALL attn_output weights on HTP,
        # so FFN differences are isolated against that exact baseline.
        if ($Stage -eq "attn" -and
            -not $enabled.ContainsKey("${layer}-output")) {
            $rules += ('blk\.{0}\.attn_output\.weight=CPU' -f $layer)
        }
        foreach ($role in @("gate", "up", "down")) {
            if ($Stage -eq "attn" -or
                -not $enabled.ContainsKey("${layer}-${role}")) {
                $rules += ('blk\.{0}\.ffn_{1}\.weight=CPU' -f $layer, $role)
            }
        }
    }
    $env:LLAMA_ARG_OVERRIDE_TENSOR = $rules -join ","
    if (-not $env:LLAMA_ARG_OVERRIDE_TENSOR) {
        Remove-Item Env:LLAMA_ARG_OVERRIDE_TENSOR -ErrorAction SilentlyContinue
    }
    $expected = @($enabled.Keys | Sort-Object | ForEach-Object {
        $p = $_ -split "-"
        if ($Stage -eq "attn") {
            "blk.$($p[0]).attn_output.weight"
        } else {
            "blk.$($p[0]).ffn_$($p[1]).weight"
        }
    })
    $baselineExpected = @($layerRange | ForEach-Object {
        "blk.$($_).attn_output.weight"
    })
    $info = @(
        "case=$caseName",
        "stage=$Stage",
        "target_NPU_weights=$($expected -join ',')",
        "QKV_weights_all_target_layers=RPC",
        "attn_output_expected_on_HTP_baseline=$($Stage -eq 'ffn')",
        "attn_output_weights=$($baselineExpected -join ',')",
        "override_tensor=$env:LLAMA_ARG_OVERRIDE_TENSOR",
        "attention_output_and_FFN_target_audit=[HEX_LINEAR_AUDIT]"
    )
    Set-Content -LiteralPath (Join-Path $folder "placement.txt") -Value $info -Encoding UTF8

    Write-Host "---- stage=$Stage case=$caseName ----"
    Write-Host "Newly enabled HTP weights: $($expected -join ', ')"
    $env:LLAMA_TENSOR_COMPARE_DIR = $folder
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

    try {
        $ErrorActionPreference = "Continue"
        & $Exe @cliArgs 2>&1 | Tee-Object -FilePath $log | Out-Host
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = "Stop"
    }
    if ($exitCode -ne 0) { throw "Case $caseName exited $exitCode; see $log" }

    $manifest = Join-Path $folder "manifest.tsv"
    if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
        throw "$caseName no tensor manifest; see $log"
    }
    $records = @(Import-Csv -LiteralPath $manifest -Delimiter ([char] 9))
    if ($records.Count -eq 0) {
        throw "Empty tensor manifest in $folder"
    }
    Write-Host "$caseName tensor_captures=$($records.Count)"
    return $folder
}

$overview = @()
try {
    $baseline = if ($RecompareOnly) {
        Join-Path $OutDir "baseline"
    } else {
        Run-Capture "baseline"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $baseline "manifest.tsv"))) {
        throw "Baseline snapshot not found: $baseline"
    }
    foreach ($caseName in $Cases) {
        $candidate = if ($RecompareOnly) {
            Join-Path $OutDir $caseName
        } else {
            Run-Capture $caseName
        }
        if (-not (Test-Path -LiteralPath (Join-Path $candidate "manifest.tsv"))) {
            throw "Candidate snapshot not found: $candidate"
        }
        $output = Join-Path $candidate "compare"
        $compareArgs = @(
            $summaryScript, $baseline, $candidate,
            "--case", $caseName, "--output", $output,
            "--expected-layers", ($layerRange -join ",")
        )
        if ($FocusLayer -ge 0) {
            $compareArgs += @("--focus-layer", "$FocusLayer")
        }
        & py -3 @compareArgs
        $code = $LASTEXITCODE
        if ($code -notin @(0, 2, 3, 4)) {
            throw "Comparator error=$code for $caseName"
        }
        $report = Get-Content -LiteralPath (Join-Path $output "summary.json") -Raw | ConvertFrom-Json
        $worstName = $null
        $worstOutputNRMSE = $null
        if ($report.worst_output_stage) {
            $worstName = $report.worst_output_stage.name
            $worstOutputNRMSE = $report.worst_output_stage.normalized_rmse
        }
        $overview += [pscustomobject]@{
            stage = $Stage
            case = $caseName
            status = $report.status
            focus_layer = $report.focus_layer
            target_stage = $report.target_stage
            compared = $report.compared
            missing_stages = ($report.missing_stages -join ";")
            upstream_diff = $report.region_counts.upstream.diff
            upstream_critical = $report.region_counts.upstream.critical
            reentry_diff = $report.region_counts.reentry_upstream.diff
            reentry_critical = $report.region_counts.reentry_upstream.critical
            first_reentry_name = $report.first_reentry_diff.name
            first_reentry_nrmse = $report.first_reentry_diff.normalized_rmse
            target_diff = $report.region_counts.target.diff
            target_critical = $report.region_counts.target.critical
            downstream_diff = $report.region_counts.downstream.diff
            downstream_critical = $report.region_counts.downstream.critical
            first_upstream_name = $report.first_upstream_diff.name
            first_upstream_nrmse = $report.first_upstream_diff.normalized_rmse
            worst_target_name = $report.worst_target.name
            worst_target_nrmse = $report.worst_target.normalized_rmse
            only_baseline = $report.only_cpu
            only_candidate = $report.only_npu
            critical = $report.critical
            diff = $report.diff
            min_cosine = $report.min_cosine
            max_rmse = $report.max_rmse
            max_normalized_rmse = $report.max_normalized_rmse
            worst_output_name = $worstName
            worst_output_nrmse = $worstOutputNRMSE
        }
    }
    if ($BaselineControl) {
        $repeatFolder = if ($RecompareOnly) {
            Join-Path $OutDir "baseline-repeat"
        } else {
            Run-Capture "baseline-repeat"
        }
        $repeatOutput = Join-Path $repeatFolder "compare"
        & py -3 $summaryScript $baseline $repeatFolder --case "baseline-control" --output $repeatOutput --expected-layers ($layerRange -join ",")
        $controlExit = $LASTEXITCODE
        if ($controlExit -notin @(0, 2, 3, 4)) {
            throw "Baseline control compare failed: $controlExit"
        }
        $control = Get-Content -LiteralPath (Join-Path $repeatOutput "summary.json") -Raw | ConvertFrom-Json
        $overview += [pscustomobject]@{
            stage = $Stage
            case = "baseline-control"
            status = $control.status
            focus_layer = $control.focus_layer
            target_stage = "-"
            compared = $control.compared
            missing_stages = ($control.missing_stages -join ";")
            upstream_diff = $control.region_counts.upstream.diff
            upstream_critical = $control.region_counts.upstream.critical
            reentry_diff = $control.region_counts.reentry_upstream.diff
            reentry_critical = $control.region_counts.reentry_upstream.critical
            first_reentry_name = $control.first_reentry_diff.name
            first_reentry_nrmse = $control.first_reentry_diff.normalized_rmse
            target_diff = 0
            target_critical = 0
            downstream_diff = 0
            downstream_critical = 0
            first_upstream_name = $control.first_upstream_diff.name
            first_upstream_nrmse = $control.first_upstream_diff.normalized_rmse
            worst_target_name = $null
            worst_target_nrmse = $null
            only_baseline = $control.only_cpu
            only_candidate = $control.only_npu
            critical = $control.critical
            diff = $control.diff
            min_cosine = $control.min_cosine
            max_rmse = $control.max_rmse
            max_normalized_rmse = $control.max_normalized_rmse
            worst_output_name = $control.worst_output_stage.name
            worst_output_nrmse = $control.worst_output_stage.normalized_rmse
        }
    }
    $reportPath = Join-Path $OutDir "overview.csv"
    if ($overview.Count -gt 0) {
        $overview | Export-Csv -LiteralPath $reportPath -NoTypeInformation -Encoding UTF8
    }
    Write-Host "===== Incremental $Stage offload ====="
    $overview | Format-Table -AutoSize
    Write-Host "Output: $reportPath"
    Write-Host "Next gate: verify each target weight in PHONE [HEX_LINEAR_AUDIT] batch_status=OK."
    Write-Host "Then rerun with no Tensor Dump before interpreting real throughput."
} finally {
    if (-not $RecompareOnly) {
        Remove-Item Env:LLAMA_TENSOR_COMPARE_DIR -ErrorAction SilentlyContinue
        Remove-Item Env:LLAMA_TENSOR_COMPARE_REGEX -ErrorAction SilentlyContinue
        Remove-Item Env:LLAMA_ARG_OVERRIDE_TENSOR -ErrorAction SilentlyContinue
    }
}
