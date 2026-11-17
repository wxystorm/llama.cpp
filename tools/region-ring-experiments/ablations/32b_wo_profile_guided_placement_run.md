# 32B Planner 消融 6：w/o Profile-Guided Placement（按内存比例 G40/N24）

## 实验定义

- 输入：Qwen3-32B-Q4_K_M、ShareGPT50_short，五组各 10 条，50 条独立 Prompt；最长 1024_10 先作为启动锚点。
- **唯一替换的 Planner 决策**：不再由 Profile 搜索层 Placement，固定内存比例布局 `G7,N5,G7,N5,G7,N5,G7,N5,G6,N4,G6`（程序中 N=P，GPU=40、Phone HTP=24、CPU Compute=0）。
- 静态比例依据：PC 侧可用 CPU RAM + CUDA VRAM 与 Phone 可用 RAM 的占比，得到 G40/N24。PC CPU RAM 仍用于模型 mmap、GPU Bank source weight windows、网络与中间缓冲，不应解释为“CPU 内存未使用”。
- **仍保留**：独立 CPU/GPU/HTP Profile、每条 Prompt 的原生 DAG 动态 XC/XG/XP 搜索、GPU 双 Bank、后台异步预取、Drain-First、Phone HTP、单槽 `llama-server`、未缓存的 Prefill。
- **跳过 Planner 的容量估算检查**：`--memory-proportional-placement` 会自动开启 `LLAMA_REGION_RING_LOCKED_SKIP_MEMORY_CHECKS=1`；不会跳过 CUDA、系统 RAM、Android HTP 的真实内存分配。PC 端仍使用 mmap-windowed 账本。
- 失败记录：wrapper 在启动或请求失败后写 `failure.json` 和追加 `failures.jsonl`。只在日志或 Windows 内存错误退出码提供证据时记录 `status=OOM`；否则写 `CRASH_UNCONFIRMED`。失败请求不会编造成 0 tok/s 或 PASS。

## 更新、编译、单测

```powershell
$root = "E:\llama\llama-tensor-v2"
cd $root

# 注意：此前该分支刚压缩过提交历史。
# 如果本地已有旧分支提交，先备份当前分支，再与 origin 对齐；
# 请在无未提交修改时操作。不要盲目 reset --hard。
git fetch origin
git switch feat/region-ring-sharegpt50-short-first-long-20261116
git status --short
# 根据 git status 决定是否需要先提交/备份本地修改。

python -m unittest discover -s .\tools\region-ring-experiments -p "test_probe_htp_llama_server.py"
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }

cmake --build .\build-tensor-v2 --config Release --target llama-server --parallel 8
if ($LASTEXITCODE -ne 0) { throw "llama-server build failed" }
```

## 执行

```powershell
$root = "E:\llama\llama-tensor-v2"
$model = "E:\llama\models\Qwen3-32B-Q4_K_M.gguf"
$rpc = "10.238.229.239:50052"
$data = "E:\llama\datasets\sharegpt\sharegpt50_short"

# 复用 Full Cycle 的 independent Profile；必须有短句 X16/X24 的真实采样点。
$baseline = "$root\32b-server-chunk256-all50-new"
$profile = "$baseline\independent-profile"
if (!(Test-Path "$profile\profile.csv") -or
    !(Test-Path "$profile\profile.json")) {
    throw "Full Cycle independent Profile missing: $profile"
}

$out = "$root\32b-ablation-wo-profile-guided-placement-g40p24-all50"
$run = @(
    "--root", $root,
    "--model", $model,
    "--rpc", $rpc,
    "--prompt-dir", $data,
    "--profile-source", $profile,
    "--out", $out,
    "--port", "18080",
    "--threads", "8"
)

# 前两条：长 Prompt 锚点和 0–128 组短句。
python .\tools\region-ring-experiments\ablations\run_32b_memory_proportional.py @run --limit 2
if ($LASTEXITCODE -ne 0) {
    Get-Content "$out\failure.json"
    throw "Smoke failed; inspect OOM evidence before continuing"
}

# 成功后复用同一配置和 Profile，继续至 50/50，不重复头两条。
python .\tools\region-ring-experiments\ablations\run_32b_memory_proportional.py @run --resume

Get-Content "$out\summary.csv"
```

中途失败后的 `--resume` 仍可继续上次已 PASS 的 Prompt，保留之前的原始日志。

## 成功与失败验收

正常启动必须看到：
```text
[SERVER_PLANNER_ABLATION] mode=WO_PROFILE_GUIDED_PLACEMENT ... score=EVENT_DAG ... memory_estimates=SKIPPED ...
[REGION_RING_PLAN_MEMORY] ... policy=SKIP_ESTIMATES_ONLY ... status=SKIPPED
[REGION_RING_PLAN_LOCK] status=PASS source=CHUNK_ONLY ... memory_policy=SKIPPED
```

每条请求必须保留 `[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT ... source=LOCKED_CHUNK_ONLY` 和实际 `REGION_RING_SUM ... complete=1`；结果写入 `requests.jsonl`、`requests.csv`、`summary.csv`。如失败检查 `failure.json` 中的 `oom_evidence` 和 `server_log`，只有明确内存错误才判为 OOM；不把失败加入吞吐平均值。
