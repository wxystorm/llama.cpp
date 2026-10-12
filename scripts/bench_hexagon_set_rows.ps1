# C-NPU performance test, with all PC tensor dumps/readbacks disabled.
# -Mode labels the run. It does not change the phone RPC server environment!
# Compare k14 vs generic for correctness-preserving throughput.
param(
    [ValidateSet("k14", "generic", "generic-profile")]
    [string] $Mode = "generic",
    [ValidateRange(1, 50)] [int] $Runs = 3,
    [ValidateRange(0, 10)] [int] $WarmupRuns = 1,
    [ValidateRange(1, 512)] [int] $Tokens = 64,
    [string] $Exe = ".\build-hexagon-pc\bin\Release\llama-cli.exe",
    [string] $Model = "D:\model\Llama-3.2-1B-Instruct-Q4_0.gguf",
    [string] $Rpc = "10.39.52.114:50052",
    [string] $Prompt = "Explain what a computer is.",
    [string] $OutDir = ".\bench-hexagon-sync"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) { throw "CLI not found: $Exe" }
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Model not found: $Model" }
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$Model = (Resolve-Path -LiteralPath $Model).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$ModeDir = Join-Path (Resolve-Path -LiteralPath $OutDir).Path $Mode
New-Item -ItemType Directory -Force -Path $ModeDir | Out-Null

$env:GGML_RPC_MATMUL_ONLY = "1"
$env:GGML_RPC_DISABLE_F16_MATMUL = "1"
$env:LLAMA_ARG_OVERRIDE_TENSOR = 'blk\.(14|15)\.(attn_output|ffn_(gate|up|down))\.weight=CPU,blk\.14\.attn_(q|v)\.weight=CPU,blk\.15\.attn_(q|k|v)\.weight=CPU'

Remove-Item Env:LLAMA_TENSOR_COMPARE_DIR -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_TENSOR_COMPARE_REGEX -ErrorAction SilentlyContinue
Remove-Item Env:GGML_SCHED_KV14_READBACK -ErrorAction SilentlyContinue

switch ($Mode) {
    "k14" {
        Write-Host "PHONE: GGML_HEXAGON_KV14_SET_ROWS_FORCE_INPUT_FLUSH=1"
        Write-Host "PHONE: GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC unset"
        Write-Host "PHONE: GGML_HEXAGON_PROFILE and GGML_HEXAGON_SET_ROWS_SYNC_TIMING unset"
    }
    "generic" {
        Write-Host "PHONE: GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC=1"
        Write-Host "PHONE: GGML_HEXAGON_KV14_SET_ROWS_FORCE_INPUT_FLUSH unset"
        Write-Host "PHONE: GGML_HEXAGON_PROFILE and GGML_HEXAGON_SET_ROWS_SYNC_TIMING unset"
    }
    "generic-profile" {
        Write-Host "PHONE: GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC=1"
        Write-Host "PHONE: GGML_HEXAGON_KV14_SET_ROWS_FORCE_INPUT_FLUSH unset"
        Write-Host "PHONE: GGML_HEXAGON_PROFILE=1, GGML_HEXAGON_SET_ROWS_SYNC_TIMING=1"
        Write-Host "PROFILING ADDS OVERHEAD. Do not compare this mode's throughput to k14/generic."
    }
}
Write-Host "ALL MODES on phone: GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1, GGML_HEXAGON_OPFUSION=0"
Write-Host "The phone RPC server must be restarted after changing its env vars."
Write-Host "Mode=$Mode RPC=$Rpc measured=$Runs warmups=$WarmupRuns tokens=$Tokens"

$cliArgs = @(
    "-m", $Model,
    "--rpc", $Rpc, "--device", "RPC0",
    "-sm", "layer", "-ngl", "3", "-fa", "off",
    "--kv-offload", "--cache-type-k", "f16", "--cache-type-v", "f16",
    "-c", "512", "-n", "$Tokens", "-t", "8",
    "--ignore-eos", "--no-warmup", "--single-turn",
    "--no-conversation", "--temp", "0", "--seed", "1234",
    "-lv", "3", "-p", $Prompt
)
$records = @()

for ($i = 1; $i -le ($WarmupRuns + $Runs); $i++) {
    $warm = $i -le $WarmupRuns
    $sequence = if ($warm) { $i } else { $i - $WarmupRuns }
    $kind = if ($warm) { "warmup" } else { "run" }
    $logPath = Join-Path $ModeDir ("{0}-{1:D2}.log" -f $kind, $sequence)

    Write-Host "--- $Mode $kind $sequence ---"
    # Native STDERR in Windows PowerShell 5.1 can be wrapped in ErrorRecords.
    $ErrorActionPreference = "Continue"
    & $Exe @cliArgs 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($exitCode -ne 0) {
        throw "Run failed (exit=$exitCode). See $logPath"
    }
    $raw = Get-Content -LiteralPath $logPath -Raw
    $p = [regex]::Matches($raw, 'prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens?\s*\(\s*([0-9.]+)\s*tokens per second\s*\)')
    $d = [regex]::Matches($raw, '(?<!prompt )eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs?\s*\(\s*([0-9.]+)\s*tokens per second\s*\)')
    $pMs = $null; $pTps = $null; $dMs = $null; $dTps = $null
    if ($p.Count -gt 0) {
        $m = $p[$p.Count - 1]
        $pMs  = [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
        $pTps = [double]::Parse($m.Groups[3].Value, [Globalization.CultureInfo]::InvariantCulture)
    }
    if ($d.Count -gt 0) {
        $m = $d[$d.Count - 1]
        $dMs  = [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
        $dTps = [double]::Parse($m.Groups[3].Value, [Globalization.CultureInfo]::InvariantCulture)
    }
    if ($null -eq $pTps -or $null -eq $dTps) {
        Write-Warning "Could not parse prompt/decode perf stats in $logPath"
    }
    if (-not $warm) {
        $records += [pscustomobject]@{
            mode       = $Mode
            run        = $sequence
            prompt_ms  = $pMs
            prompt_tps = $pTps
            decode_ms  = $dMs
            decode_tps = $dTps
            logfile    = $logPath
        }
    }
}
$csv = Join-Path $ModeDir "results.csv"
$records | Export-Csv -NoTypeInformation -Path $csv -Encoding UTF8
$records | Format-Table -AutoSize
Write-Host "Saved $csv"
Write-Host "Compare k14 vs generic with same phone conditions; use median of repeated t/s."
