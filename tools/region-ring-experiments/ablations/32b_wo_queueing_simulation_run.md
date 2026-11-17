# 32B Planner 消融：w/o Queueing Simulation（Linear Sum）

## 实验定义

- Qwen3-32B-Q4_K_M；ShareGPT50_short 50 条，0/128/256/512/1024 各 10 条。
- 独立 CPU/GPU/Phone HTP Profile 不变；Placement 重新搜索；XC=XG=XP 固定 64。
- 保留原来的 Uniform/Prefix/CPU-optional/Local/GPU-tail 候选搜索与内存约束，改变候选评分：所有 Region 的单步计算与 Prepare 乘以 tokens/64，加上边界传输的线性费用以及 GPU Bank rank≥2 的完整 H2D 权重替换耗时。不构造 DAG 事件、不估计并发重叠与流水气泡。
- 物理约束仍包括 GPU 两个 Bank 的峰值、GPU KV/图缓冲、Phone 实际可用 RAM、HTP Session 映射与共享预算以及 PC RAM。PC 使用已经存在的 --pc-mmap-budget（必须由引擎确认 GGUF 实际 mmap；只将文件映射的权重按并存窗口计费），而不是按无 mmap 的全部常驻权重计算。运行阶段保留 GPU 双 Bank 异步搬运、Drain-First、多轮 Cycle、Phone HTP。
- 选层时环境变量 LLAMA_REGION_RING_NO_DAG_PLANNER=1；必须处于 APPLY_WIDE + HTP-only + Streaming + Guarded-memory 模式。不可搭配 skip-plan-memory-checks。
- 第一条最长 Prompt 只负责产生并验证 Placement；Server 常驻后仍锁定该 Placement 执行相同 50 条。
- 旧 Profile 的 chunk_widths 可以不是新版完整网格，但只有本消融显式启用时才允许复用；要求 CSV 实测 CPU/GPU/PHONE X64 + Phone X64/KV2048，JSON 标记了 X64。正常动态 Chunk 实验继续严格检查全部网格；不伪造 Profile 元数据、不重新采样。

## 更新与测试

~~~powershell
cd E:\llama\llama-tensor-v2
git switch feat/region-ring-sharegpt50-short-first-long-20261116
git pull --ff-only origin feat/region-ring-sharegpt50-short-first-long-20261116

python -m unittest discover -s .\tools\region-ring-experiments -p "test_probe_htp_llama_server.py"
if ($LASTEXITCODE -ne 0) { throw "Python unit tests failed" }

cmake --build .\build-tensor-v2 --config Release --target llama-completion llama-server --parallel 8
if ($LASTEXITCODE -ne 0) { throw "C++ build failed" }

# Optional when CMake was configured with LLAMA_BUILD_TESTS=ON:
# cmake --build .\build-tensor-v2 --config Release --target test-region-ring-planner --parallel 8
# .\build-tensor-v2\bin\Release\test-region-ring-planner.exe
~~~

## Step 1：复用旧 Profile + 线性评分重新生成 Placement

以前由于旧采样网格不同触发 CACHE_CHUNK_GRID_MISMATCH；现在严格检查实际 X64 采样是否存在，合格才允许复用。源数据不修改。上次进一步出现 HTP_MEMORY_DIAG pass=0（session_map=1831、host_ram=2856、gpu_vram=644、phone_ram=1206），说明没有候选通过内存门槛；本次启用真实 mmap 权重窗口预算以避免无 mmap 口径的 PC 全部权重计费，同时不放松 GPU/HTP/Phone 限制。必须使用全新 `--out`，不能在旧实验配置中直接复跑。

沿用旧的 Full Cycle 独立 Profile，不重新采样，不使用本次重新采集的 32b-profile-fresh-v2。
不要传入 locked-layout 或 main50-short；它们会阻止重新选层。

~~~powershell
$root = "E:\llama\llama-tensor-v2"
$oldProfileCandidates = @(
    "$root\32b-planner-wo-profile-fixed64-all50-1\independent-profile",
    "$root\32b-server-chunk256-all50-new\independent-profile",
    "$root\32b-sharegpt50-short-mmap-test\independent-profile"
)
$oldProfile = $oldProfileCandidates |
    Where-Object {
        (Test-Path "$_\profile.csv") -and
        (Test-Path "$_\profile.json")
    } | Select-Object -First 1
if (-not $oldProfile) { throw "Old independent Profile not found" }
Write-Host "[OLD_PROFILE] $oldProfile"

$model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf"
$rpc = "10.238.229.239:50052"
$data = "E:\llama\datasets\sharegpt\sharegpt50_short"
$planOut = "$root\32b-wo-queue-sim-plan64-old-profile-mmap-v3"

$planOpts = @(
    "--exe", "$root\build-tensor-v2\bin\Release\llama-completion.exe",
    "--model", $model,
    "--model-layers", "64",
    "--rpc", $rpc,
    "--prompt-dir", $data,
    "--out", $planOut,
    "--auto-plan", "--htp-only", "--streaming", "on",
    "--pc-mmap-budget",
    "--sharegpt50-short-dataset",
    "--lock-layout-after-first",
    "--real-stage-profile-csv", "$oldProfile\profile.csv",
    "--env", "LLAMA_REGION_RING_NO_DAG_PLANNER=1",
    "--batch", "2048", "--ubatch", "2048",
    "--n-predict", "1", "--threads", "8",
    "--min-context", "256", "--max-context", "2048",
    "--context-padding", "0",
    "--max-retries", "0",
    "--only-prompt", "prompt_1024_10.txt"
)
python .\tools\region-ring-experiments\run_32b_all40.py @planOpts
if ($LASTEXITCODE -ne 0) { throw "Linear Planner failed" }

$lock = Get-Content "$planOut\placement-lock.json" -Raw | ConvertFrom-Json
$linearLayout = $lock.layout
Write-Host "[NO_DAG_SELECTED_LAYOUT] $linearLayout"
~~~

检查 [REGION_RING_PC_MEMORY_POLICY] mode=MMAP_WINDOWED ... status=CONFIGURED、[REGION_RING_NO_DAG_PLAN] mode=LINEAR_SUM_NO_QUEUE status=READY、[REGION_RING_PLAN_SELECTED] source=LINEAR_NO_DAG、第一条 PASS、placement-lock.json。若 PC 实际未启用 mmap，引擎会以 ACTUAL_MMAP_AND_HTP_STREAM_REQUIRED 拒绝；不得跳过护栏。若出现新拒绝，提取 REGION_RING_HTP_MEMORY_DIAG 和 PLAN_HTP_TAIL 等原生日志定位。

## Step 2：常驻 Server，固定 64 执行 50 条

~~~powershell
$serverOut = "$root\32b-wo-queue-sim-fixed64-all50-old-profile-mmap-v3"
$serverOpts5 = @(
    "--server-exe", "$root\build-tensor-v2\bin\Release\llama-server.exe",
    "--model", $model, "--model-layers", "64",
    "--locked-layout", $linearLayout,
    "--rpc", $rpc, "--prompt-dir", $data,
    "--profile-source", $oldProfile,
    "--port", "18080", "--threads", "8",
    "--all50", "--per-request-chunks",
    "--fixed-chunk", "64",
    "--no-dag-placement",
    "--guarded-memory-checks"
)

python .\tools\region-ring-experiments\probe_htp_llama_server.py @serverOpts5 `
    --out "$root\32b-wo-queue-sim-fixed64-smoke-old-profile-mmap-v3" --limit 2
if ($LASTEXITCODE -ne 0) { throw "No-DAG smoke failed" }

python .\tools\region-ring-experiments\probe_htp_llama_server.py @serverOpts5 `
    --out $serverOut
~~~

中断后同目录加 --resume。日志要求启动时 NO_DAG_PLAN READY、LOCK PASS 且内存策略 GUARDED；请求时 FIXED_CHUNK target=64 PASS、cache_n=0、50/50 PASS。GPU Bank refill 数目根据实际选定布局自动核对，不能固定假设 4 次。

## 归因注意

此前第 4 组 Fixed-64 采用原 Full Cycle Placement，并未在固定 64 的条件下由 DAG 重新选层。第 5 组重新搜索 Placement，因此要作严格的“评分器唯一变量”对比，需要相同固定 64 候选条件下的 DAG 选层对照。线性评分 predicted_ms 不应当作 DAG 预测值用于误差分析。
