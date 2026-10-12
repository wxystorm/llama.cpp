# Capture-free CPU-vs-HTP QKV decode regression and basic throughput sampling.
# Run from the repository root, after starting the mobile RPC server.
# Both modes use the same RPC KV cache, -ngl, model and input prompt:
#   cpu:     layer-range Q/K/V + attn_output + FFN weights pinned on CPU
#   all-qkv: layer-range Q/K/V allowed on RPC; attn_output + FFN pinned CPU
#
# Phone environment (restart server if changing):
#   export GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC=1
#   export GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1
#   export GGML_HEXAGON_OPFUSION=0
#   unset GGML_HEXAGON_KV14_SET_ROWS_FORCE_INPUT_FLUSH
#   unset GGML_HEXAGON_KV14_SET_ROWS_DIAG
#   unset GGML_HEXAGON_PROFILE
#   unset GGML_HEXAGON_SET_ROWS_SYNC_TIMING
# Optional: export GGML_HEXAGON_QKV_AUDIT=1 to confirm all 9 Q/K/V HTP
# weights. Audit logs add host overhead: unset for clean speed comparisons.
#
# Note: tiny top-logit differences can cause generation to fork. The generated
# text does not have to be byte-for-byte equal to pass a numerical smoke test.
# Check exit status, nonfinite/zero logits, HTP execution, and stability.
param(
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
    [string] $OutDir = ".\hexagon-qkv-decode"
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
$OutDir = (Resolve-Path -LiteralPath $OutDir).Path
$layerRange = @($StartLayer..$EndLayer)
$layerRegex = $layerRange -join "|"
$layerOverride = 'blk\.({0})\.(attn_output|ffn_(gate|up|down))\.weight=CPU' -f $layerRegex
$cpuQkvOverride = 'blk\.({0})\.attn_(q|k|v)\.weight=CPU' -f $layerRegex

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
    "cpu_QKV_override=$cpuQkvOverride",
    "both_modes_base_override=$layerOverride",
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
        @("cpu", "all-qkv")
    } else {
        @("all-qkv", "cpu")
    }

    foreach ($mode in $modes) {
        if ($mode -eq "cpu") {
            $env:LLAMA_ARG_OVERRIDE_TENSOR = "$layerOverride,$cpuQkvOverride"
        } else {
            $env:LLAMA_ARG_OVERRIDE_TENSOR = $layerOverride
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
Write-Host "===== Capture-free QKV Decode ====="
$records | Format-Table -AutoSize
Write-Host "Saved CSV: $path"
foreach ($mode in @("cpu", "all-qkv")) {
    $rows = @($records | Where-Object { $_.mode -eq $mode })
    if ($rows.Count -eq 0) { continue }
    $p = @($rows | ForEach-Object { $_.prompt_tps } | Sort-Object)
    $d = @($rows | ForEach-Object { $_.decode_tps } | Sort-Object)
    $idx = [int][Math]::Floor($rows.Count / 2)
    $promptMedian = if (($rows.Count % 2) -eq 1) { $p[$idx] } else { ($p[$idx-1] + $p[$idx]) / 2 }
    $decodeMedian = if (($rows.Count % 2) -eq 1) { $d[$idx] } else { ($d[$idx-1] + $d[$idx]) / 2 }
    Write-Host ("mode={0} prompt_median_tps={1:N2} decode_median_tps={2:N2}" -f $mode, $promptMedian, $decodeMedian)
}
Write-Host "For correctness: inspect generation text/HTP QKV audits. Near-tie token forks are allowed."
Write-Host "This is still QKV-only offload, NOT the full attention+FFN NPU execution path."
