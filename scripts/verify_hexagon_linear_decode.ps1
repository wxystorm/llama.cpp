# Capture-free incremental attention-output / FFN offload Decode regression.
# Baseline always includes validated NPU QKV; for FFN stage baseline also
# includes NPU attention-output MatMul. Only the newly released weights differ.
# RPC KV F16 and FA-off settings are identical across modes.
# Phone requires GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC=1; for speed tests
# disable GGML_HEXAGON_QKV_AUDIT and GGML_HEXAGON_LINEAR_AUDIT.
# Tensor Dump and explicit KV readback are disabled on PC.
param(
    [ValidateSet("attn", "ffn")] [string] $Stage = "attn",
    [ValidateRange(1, 100)] [int] $Runs = 3,
    [ValidateRange(0, 10)] [int] $WarmupRuns = 1,
    [ValidateRange(1, 256)] [int] $Tokens = 64,
    [ValidateRange(1, 128)] [int] $ModelLayers = 16,
    [ValidateRange(1, 128)] [int] $Ngl = 4,
    [ValidateRange(0, 127)] [int] $StartLayer = 13,
    [ValidateRange(0, 127)] [int] $EndLayer = 15,
    [string] $Exe = ".\build-hexagon-pc\bin\Release\llama-cli.exe",
    [string] $Model = "D:\model\Llama-3.2-1B-Instruct-Q4_0.gguf",
    [string] $Rpc = "10.39.52.114:50052",
    [string] $Prompt = "Explain what a computer is.",
    [string] $OutDir = ".\hexagon-linear-decode"
)

$ErrorActionPreference = "Stop"
if ($StartLayer -gt $EndLayer -or $EndLayer -ge $ModelLayers) {
    throw "Invalid layer range: $StartLayer..$EndLayer for $ModelLayers layers"
}
# Current llama-model.cpp includes output layer in the -ngl budget.
$firstRemoteLayer = [Math]::Max(0, $ModelLayers + 1 - $Ngl)
if ($StartLayer -lt $firstRemoteLayer) {
    throw "Layer $StartLayer is not on RPC with -ngl $Ngl. Need at least -ngl $($ModelLayers + 1 - $StartLayer)."
}
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "CLI missing: $Exe" }
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model missing: $Model" }

$Exe = (Resolve-Path -LiteralPath $Exe).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = Join-Path (Resolve-Path -LiteralPath $OutDir).Path $Stage
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$layerRange = @($StartLayer..$EndLayer)
$layerRegex = $layerRange -join "|"
$attnOverride = 'blk\.({0})\.attn_output\.weight=CPU' -f $layerRegex
$ffnOverride  = 'blk\.({0})\.ffn_(gate|up|down)\.weight=CPU' -f $layerRegex
$candidateMode = if ($Stage -eq "attn") { "all-attn" } else { "all-ffn" }
$baselineOverride = if ($Stage -eq "attn") {
    "$attnOverride,$ffnOverride"
} else {
    $ffnOverride
}
$candidateOverride = if ($Stage -eq "attn") { $ffnOverride } else { "" }

$env:GGML_RPC_MATMUL_ONLY = "1"
$env:GGML_RPC_DISABLE_F16_MATMUL = "1"
# Crucial: no callback/readback (unlike verify_hexagon_qkv_layers.ps1).
Remove-Item Env:LLAMA_TENSOR_COMPARE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_TENSOR_COMPARE_REGEX -ErrorAction SilentlyContinue
Remove-Item Env:GGML_SCHED_KV14_READBACK -ErrorAction SilentlyContinue

$commonArgs = @(
    "-m", $Model, "--rpc", $Rpc, "--device", "RPC0",
    "-sm", "layer", "-ngl", "$Ngl", "-fa", "off",
    "--kv-offload", "--cache-type-k", "f16", "--cache-type-v", "f16",
    "-c", "512", "-n", "$Tokens", "-t", "8",
    "--ignore-eos", "--no-warmup", "--single-turn", "--no-conversation",
    "--temp", "0", "--seed", "1234", "--perf", "-lv", "3", "-p", $Prompt
)

$settings = @(
    "exe=$Exe", "model=$Model", "rpc=$Rpc", "prompt=$Prompt",
    "ngl=$Ngl", "model_layers=$ModelLayers", "range=$StartLayer..$EndLayer",
    "tokens=$Tokens", "runs=$Runs", "warmup_runs=$WarmupRuns",
    "stage=$Stage",
    "baseline=QKV_NPU+ATTN_CPU+FFN_CPU if attn; QKV_NPU+ATTN_NPU+FFN_CPU if ffn",
    "baseline_override=$baselineOverride",
    "candidate_override=$candidateOverride",
    "rpc_matmul_only=$env:GGML_RPC_MATMUL_ONLY",
    "rpc_disable_f16_matmul=$env:GGML_RPC_DISABLE_F16_MATMUL",
    "phone_flags_must_be_checked_manually=1",
    "tensor_dump_disabled=1"
)
Set-Content -LiteralPath (Join-Path $OutDir "settings.txt") -Value $settings -Encoding UTF8

# A round performs each mode once. Alternate order every round to reduce
# systematic temperature/frequency/drift bias.
$records = @()
$perfPrompt = '\bprompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens?\s*\(\s*(?:[^\r\n)]*?,\s*)?([0-9.]+)\s*tokens per second\s*\)'
$perfDecode = '(?<!prompt )\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*(?:runs?|tokens?)\s*\(\s*(?:[^\r\n)]*?,\s*)?([0-9.]+)\s*tokens per second\s*\)'

for ($round = 1; $round -le ($WarmupRuns + $Runs); ++$round) {
    $isWarmup = $round -le $WarmupRuns
    $number = if ($isWarmup) { $round } else { $round - $WarmupRuns }
    $tag = if ($isWarmup) { "warmup" } else { "run" }
    $modes = if (($round % 2) -eq 1) {
        @("baseline", $candidateMode)
    } else {
        @($candidateMode, "baseline")
    }

    foreach ($mode in $modes) {
        $rule = if ($mode -eq "baseline") { $baselineOverride } else { $candidateOverride }
        if ($rule) {
            $env:LLAMA_ARG_OVERRIDE_TENSOR = $rule
        } else {
            Remove-Item Env:LLAMA_ARG_OVERRIDE_TENSOR -ErrorAction SilentlyContinue
        }

        $log = Join-Path $OutDir ("{0}-{1}-{2:D2}.log" -f $mode, $tag, $number)
        Write-Host "===== $mode $tag $number/$Runs ====="
        Write-Host "NPU layer range: $StartLayer..$EndLayer; overrides: $env:LLAMA_ARG_OVERRIDE_TENSOR"
        $ErrorActionPreference = "Continue"
        & $Exe @commonArgs 2>&1 | Tee-Object -FilePath $log | Out-Host
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = "Stop"
        if ($exitCode -ne 0) {
            throw "Case $mode $tag $number failed with exit code $exitCode; inspect $log"
        }

        $raw = Get-Content -LiteralPath $log -Raw
        $p = [regex]::Matches($raw, $perfPrompt)
        $d = [regex]::Matches($raw, $perfDecode)
        if ($p.Count -eq 0 -or $d.Count -eq 0) {
            Write-Warning "Performance line not found in $log. Check --perf and -lv 3."
            throw "Missing timing report for $mode $tag $number"
        }
        $pm = $p[$p.Count - 1]
        $dm = $d[$d.Count - 1]
        $num = [Globalization.CultureInfo]::InvariantCulture
        $promptMs = [double]::Parse($pm.Groups[1].Value, $num)
        $promptTps = [double]::Parse($pm.Groups[3].Value, $num)
        $decodeMs = [double]::Parse($dm.Groups[1].Value, $num)
        $decodeTps = [double]::Parse($dm.Groups[3].Value, $num)

        if (-not $isWarmup) {
            $records += [pscustomobject]@{
                stage = $Stage
                mode = $mode
                run = $number
                prompt_tokens = [int]$pm.Groups[2].Value
                decode_runs = [int]$dm.Groups[2].Value
                prompt_ms = $promptMs
                prompt_tps = $promptTps
                decode_ms = $decodeMs
                decode_tps = $decodeTps
                logfile = $log
            }
        }
    }
}

$path = Join-Path $OutDir "results.csv"
$records | Export-Csv -LiteralPath $path -NoTypeInformation -Encoding UTF8
Write-Host "===== Capture-free $Stage Decode ====="
$records | Format-Table -AutoSize
Write-Host "Saved CSV: $path"
foreach ($mode in @("baseline", $candidateMode)) {
    $rows = @($records | Where-Object { $_.mode -eq $mode })
    if ($rows.Count -eq 0) { continue }
    $p = @($rows | ForEach-Object { $_.prompt_tps } | Sort-Object)
    $d = @($rows | ForEach-Object { $_.decode_tps } | Sort-Object)
    $idx = [int][Math]::Floor($rows.Count / 2)
    $promptMedian = if (($rows.Count % 2) -eq 1) { $p[$idx] } else { ($p[$idx-1] + $p[$idx]) / 2 }
    $decodeMedian = if (($rows.Count % 2) -eq 1) { $d[$idx] } else { ($d[$idx-1] + $d[$idx]) / 2 }
    Write-Host ("mode={0} prompt_median_tps={1:N2} decode_median_tps={2:N2}" -f $mode, $promptMedian, $decodeMedian)
}
Write-Host "For correctness: inspect exits, generation, and mobile HEX_LINEAR_AUDIT. Near-tie token forks can be normal."
Write-Host "Only weights in the tested range are controlled. Capture-free check does not prove numerical equivalence."
Remove-Item Env:LLAMA_ARG_OVERRIDE_TENSOR -ErrorAction SilentlyContinue
