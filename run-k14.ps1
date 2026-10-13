$exe = ".\build-hexagon-pc\bin\Release\llama-cli.exe"
# ===== 路径 =====

$model = "D:\model\Llama-3.2-1B-Instruct-Q4_0.gguf" # 修改为实际路径
$rpc = "10.39.52.114:50052"

# ===== 公共开关 =====
$env:GGML_RPC_MATMUL_ONLY = "1"
$env:GGML_RPC_DISABLE_F16_MATMUL = "1"

# 关闭 KV 回读诊断
Remove-Item Env:GGML_SCHED_KV14_READBACK -ErrorAction SilentlyContinue

# 必须指定我们需要的张量
$env:LLAMA_TENSOR_COMPARE_REGEX = '^(Qcur|Kcur|Vcur|kq|kq_soft_max|kqv|kqv_out|attn_out)-14$'

$baseOverride = 'blk\.(14|15)\.(attn_output|ffn_(gate|up|down))\.weight=CPU,blk\.14\.attn_(q|v)\.weight=CPU,blk\.15\.attn_(q|k|v)\.weight=CPU'

foreach ($tag in @("C-CPU", "C-NPU")) {

    # K14 计算位置
    if ($tag -eq "C-CPU") {
        $env:LLAMA_ARG_OVERRIDE_TENSOR = "${baseOverride},blk\.14\.attn_k\.weight=CPU"
    } else {
        $env:LLAMA_ARG_OVERRIDE_TENSOR = $baseOverride
    }

    # 使用绝对目录
    $env:LLAMA_TENSOR_COMPARE_DIR = Join-Path (Get-Location).Path "dump-$tag"

    Write-Host "Running $tag"
    Write-Host "Dump directory: $env:LLAMA_TENSOR_COMPARE_DIR"

    & $exe `
        -m $model `
        --rpc $rpc `
        --device RPC0 `
        -sm layer `
        -ngl 3 `
        -fa off `
        --kv-offload `
        --cache-type-k f16 `
        --cache-type-v f16 `
        -c 512 `
        -n 16 `
        -t 8 `
        --no-warmup `
        --single-turn `
        --no-conversation `
        --temp 0 `
        --seed 1234 `
        -lv 3 `
        -p "Explain what a computer is." `
        2>&1 | Tee-Object -FilePath ".\$tag.log"

    if ($LASTEXITCODE -ne 0) {
        throw "$tag failed with exit code $LASTEXITCODE"
    }
}

# ===== 检查 =====
Get-ChildItem .\dump-C-CPU
Get-ChildItem .\dump-C-NPU