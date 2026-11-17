# 32B Planner 消融 5：w/o DAG / Queueing Simulation（锁层 + 动态 Chunk）

## 实验定义

- 模型：Qwen3-32B-Q4_K_M，ShareGPT50_short，0/128/256/512/1024 各 10 条，共 50 条。
- **锁定此前无 DAG 线性选层得到的布局**：`G8,P4,G6,C1,P5,G8,C1,P5,G6,P6,G8,C1,G5`（用户日志使用 `N`，与 `P` 同义）。GPU=41 层、Phone HTP=20 层、CPU=3 层，末尾 G5。
- Placement **不再变化**；每条 Prompt 基于旧独立 CPU/GPU/HTP Profile **重新搜索 XC/XG/XP**。不再固定 64。
- 线性评分器把 Region 计算、Prepare、边界传输及 GPU Bank H2D 相加；不创建 DAG 事件，不建模并发重叠和队列气泡。C++ 日志必须显示 `chunk_policy=ADAPTIVE_XC_XG_XP` 和 `source=LINEAR_NO_DAG_CHUNK_ONLY`。
- Runtime 保留双 Bank、后台异步预取、Drain-First、资源感知 Chunk 调度和 Phone HTP；保持 PC mmap 窗口、CUDA/HTP/Phone 物理内存校验，不使用 skip-plan-memory-checks。
- **复用旧 Profile**：`32b-planner-wo-profile-fixed64-all50-1/independent-profile`，不新采样、不篡改旧 `profile.csv` 或 `profile.json`。原始 Placement 搜索那一轮固定 Chunk=64 是为了确定该不可变分层；本次 50 条动态 Chunk 不受其固定宽度限制。旧 Profile 必须包含实际 X32 与 X64 的 CPU/GPU/PHONE 数据及 Phone X64/KV2048 样本，不能用新采样替代。

## 更新与编译

```powershell
cd E:\llama\llama-tensor-v2
git switch feat/region-ring-sharegpt50-short-first-long-20261116
git pull --ff-only origin feat/region-ring-sharegpt50-short-first-long-20261116

python -m unittest discover -s .\tools\region-ring-experiments -p "test_probe_htp_llama_server.py"
if ($LASTEXITCODE -ne 0) { throw "Python unittest failed" }

cmake --build .\build-tensor-v2 --config Release --target llama-server llama-completion --parallel 8
if ($LASTEXITCODE -ne 0) { throw "C++ build failed" }

# 可选：CMake 编译时启用 LLAMA_BUILD_TESTS=ON 的情况下：
# cmake --build .\build-tensor-v2 --config Release --target test-region-ring-planner --parallel 8
# .\build-tensor-v2\bin\Release\test-region-ring-planner.exe
```

## 运行 50 条：仅锁定 Placement，动态选择 XC/XG/XP

```powershell
$root = "E:\llama\llama-tensor-v2"
$model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf"
$rpc = "10.238.229.239:50052"  # 根据手机当前 RPC 地址调整
$data = "E:\llama\datasets\sharegpt\sharegpt50_short"
$oldProfile = "$root\32b-planner-wo-profile-fixed64-all50-1\independent-profile"
$linearLayout = "G8,P4,G6,C1,P5,G8,C1,P5,G6,P6,G8,C1,G5"

if (!(Test-Path "$oldProfile\profile.csv") -or
    !(Test-Path "$oldProfile\profile.json")) {
    throw "Old independent Profile missing"
}

$opts5 = @(
    "--server-exe", "$root\build-tensor-v2\bin\Release\llama-server.exe",
    "--model", $model, "--model-layers", "64",
    "--locked-layout", $linearLayout,
    "--rpc", $rpc, "--prompt-dir", $data,
    "--profile-source", $oldProfile,
    "--port", "18080", "--threads", "8",
    "--all50", "--per-request-chunks",
    "--no-dag-placement",
    "--guarded-memory-checks"
)

# 先试两条，使用独立输出目录
python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts5 `
    --out "$root\32b-wo-dag-dynamic-smoke-old-profile" --limit 2
if ($LASTEXITCODE -ne 0) { throw "No-DAG dynamic Chunk smoke failed" }

# 全 50 条；不要和之前的固定 64 实验共用 out
$serverOut = "$root\32b-wo-dag-dynamic-all50-old-profile"
python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts5 `
    --out $serverOut

# 若中断且原输出已有同一实验的 config，请改用 --resume
# python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts5 `
#     --out $serverOut --resume

Get-Content "$serverOut\summary.csv"
```

## 必须验证的日志

- Server 启动：`[REGION_RING_NO_DAG_PLAN] mode=LINEAR_SUM_NO_QUEUE chunk_policy=ADAPTIVE_XC_XG_XP ... placement=LOCKED ... event_simulation=OFF`
- Server 启动：`[REGION_RING_PLAN_LOCK] status=PASS source=LINEAR_NO_DAG_CHUNK_ONLY ... memory_policy=GUARDED`
- 每条请求：`[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT ... layout=G8,P4,...,G5 XC=... XG=... XP=... source=LINEAR_NO_DAG_CHUNK_ONLY layer_placement=IMMUTABLE`
- `XC/XG/XP` 可以随 Prompt 改变，不要求必须改变；不得强制全部等于 64。脚本核对请求实际执行 `REGION_RING_SUM` 与所选块宽、固定布局、`cache_n=0`，最终需要 50/50 PASS。
- 对 64–128 tokens 短 Prompt，双 GPU Bank 路径要求 `2*XG <= tokens`；旧 Profile 至少要有 X32 实测点。GPU bank refill 次数按布局中的 G Region 数量动态核对。

## 论文实验归因

本次实验用 **无 DAG 选出的固定 Placement + 无 DAG 动态 Chunk 评分**，而完整版 Full Cycle 使用 DAG 选层与 DAG 动态分块。因此两者变化不止评分器本身：若论文需要证明 *只有* DAG Chunk 评分器的净贡献，应再加一个 **相同 locked Placement、相同旧 Profile、动态 Chunk、开启 DAG 评分** 的对照。不要把该实验直接当成严格单变量的 DAG 消融。
