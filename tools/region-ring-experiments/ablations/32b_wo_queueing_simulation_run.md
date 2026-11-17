# 32B Planner 消融：w/o Queueing Simulation（Linear Sum）

## 实验定义

- Qwen3-32B-Q4_K_M；ShareGPT50_short 50 条，0/128/256/512/1024 各 10 条。
- 独立 CPU/GPU/Phone HTP Profile 不变；Placement 重新搜索；XC=XG=XP 固定 64。
- 保留原来的 Uniform/Prefix/CPU-optional/Local/GPU-tail 候选搜索与内存约束，改变候选评分：所有 Region 的单步计算与 Prepare 乘以 tokens/64，加上边界传输的线性费用以及 GPU Bank rank≥2 的完整 H2D 权重替换耗时。不构造 DAG 事件、不估计并发重叠与流水气泡。
- 物理约束仍包括 GPU 两个 Bank 的峰值、GPU KV/图缓冲、Phone 实际可用 RAM、HTP Session 映射与共享预算以及 PC RAM。运行阶段保留 GPU 双 Bank 异步搬运、Drain-First、多轮 Cycle、Phone HTP。
- 选层时环境变量 LLAMA_REGION_RING_NO_DAG_PLANNER=1；必须处于 APPLY_WIDE + HTP-only + Streaming + Guarded-memory 模式。不可搭配 skip-plan-memory-checks。
- 第一条最长 Prompt 只负责产生并验证 Placement；Server 常驻后仍锁定该 Placement 执行相同 50 条。

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

## Step 1：独立 Profile + 线性评分重新生成 Placement

下面复用先前采集完成的 32b-profile-fresh-v2。不要传入 locked-layout 或 main50-short；它们会阻止重新选层。

~~~powershell
$root = "E:\llama\llama-tensor-v2"
$profileOut = "$root\32b-profile-fresh-v2"
$profileConfig = Get-Content "$profileOut\run-config.json" -Raw | ConvertFrom-Json
$model = $profileConfig.model.path
$rpc = $profileConfig.rpc
$data = $profileConfig.prompt_dir
$planOut = "$root\32b-wo-queue-sim-plan64"

$planOpts = @(
    "--exe", "$root\build-tensor-v2\bin\Release\llama-completion.exe",
    "--model", $model,
    "--model-layers", "64",
    "--rpc", $rpc,
    "--prompt-dir", $data,
    "--out", $planOut,
    "--auto-plan", "--htp-only", "--streaming", "on",
    "--sharegpt50-short-dataset",
    "--lock-layout-after-first",
    "--real-stage-profile-csv", "$profileOut\independent-profile\profile.csv",
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

检查 [REGION_RING_NO_DAG_PLAN] mode=LINEAR_SUM_NO_QUEUE status=READY、[REGION_RING_PLAN_SELECTED] source=LINEAR_NO_DAG、第一条 PASS、placement-lock.json。若内存拒绝或没有可行 Placement，应保留失败日志，不得跳过护栏。

## Step 2：常驻 Server，固定 64 执行 50 条

~~~powershell
$serverOut = "$root\32b-wo-queue-sim-fixed64-all50"
$serverOpts5 = @(
    "--server-exe", "$root\build-tensor-v2\bin\Release\llama-server.exe",
    "--model", $model, "--model-layers", "64",
    "--locked-layout", $linearLayout,
    "--rpc", $rpc, "--prompt-dir", $data,
    "--profile-source", "$profileOut\independent-profile",
    "--port", "18080", "--threads", "8",
    "--all50", "--per-request-chunks",
    "--fixed-chunk", "64",
    "--no-dag-placement",
    "--guarded-memory-checks"
)

python .\tools\region-ring-experiments\probe_htp_llama_server.py @serverOpts5 `
    --out "$root\32b-wo-queue-sim-fixed64-smoke" --limit 2
if ($LASTEXITCODE -ne 0) { throw "No-DAG smoke failed" }

python .\tools\region-ring-experiments\probe_htp_llama_server.py @serverOpts5 `
    --out $serverOut
~~~

中断后同目录加 --resume。日志要求启动时 NO_DAG_PLAN READY、LOCK PASS 且内存策略 GUARDED；请求时 FIXED_CHUNK target=64 PASS、cache_n=0、50/50 PASS。GPU Bank refill 数目根据实际选定布局自动核对，不能固定假设 4 次。

## 归因注意

此前第 4 组 Fixed-64 采用原 Full Cycle Placement，并未在固定 64 的条件下由 DAG 重新选层。第 5 组重新搜索 Placement，因此要作严格的“评分器唯一变量”对比，需要相同固定 64 候选条件下的 DAG 选层对照。线性评分 predicted_ms 不应当作 DAG 预测值用于误差分析。
