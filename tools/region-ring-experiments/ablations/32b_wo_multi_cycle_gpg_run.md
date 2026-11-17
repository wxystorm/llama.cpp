# 32B Runtime 消融：w/o Multi-Cycle（G–P–G）

## 单变量控制

- **Full Cycle：** `G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7`，43 GPU 层、21 Phone 层
- **消融逻辑布局：** `G36,P21,G7`，保持 43 GPU 层、21 Phone 层
- **消融实际 GPU Bank 分段：** `G7,G7,G8,G7,G7,P21,G7`，仍有 6 段 GPU Region、2 个 CUDA Bank
- **只取消多轮 PC↔Phone 穿插**：变成 GPU→Phone→GPU，一次进入 Phone、一次回 PC
- **保持开启：** GPU A/B 双 Bank、后台异步 Refill、Drain-First、资源感知调度、Phone HTP 多 Session、模型常驻 `llama-server`、不复用 KV
- **严格保留每条 Prompt 的 XC/XG/XP**：重放原 Full Cycle 的 `requests.jsonl`，不重新选择分块、不改变测试顺序
- `P21` 使用原有 Phone-owned Plan 的逐层 HTP 处理；真实 HTP 分配可行性必须在首条 Smoke 中验证，不能仅依据层数假定成功

## 更新 / 编译 / 检验

```powershell
cd E:\llama\llama-tensor-v2
git switch feat/region-ring-sharegpt50-short-first-long-20261116
git pull --ff-only
cmake --build .\build-tensor-v2 --config Release --target llama-server --parallel 8
if ($LASTEXITCODE -ne 0) { throw "llama-server 编译失败" }

python -m unittest discover -s .\tools\region-ring-experiments -p "test_probe_htp_llama_server.py"
if ($LASTEXITCODE -ne 0) { throw "测试失败" }
```

## 执行命令

```powershell
$root = "E:\llama\llama-tensor-v2"
$baseline = "$root\32b-server-chunk256-all50-new"
$source = "$baseline\requests.jsonl"
$cfg = Get-Content "$baseline\server-run-config.json" -Raw | ConvertFrom-Json

$opts3 = @(
    "--server-exe", "$root\build-tensor-v2\bin\Release\llama-server.exe",
    "--model", $cfg.configuration.model.path,
    "--model-layers", "64",
    "--locked-layout", "G36,P21,G7",
    "--rpc", $cfg.configuration.rpc,
    "--prompt-dir", $cfg.configuration.prompt_dir,
    "--profile-source", "$baseline\independent-profile",
    "--port", "18080",
    "--threads", "8",
    "--all50",
    "--per-request-chunks",
    "--max-plan-xg", "256",
    "--max-plan-xp", "256",
    "--replay-chunks-from", $source,
    "--no-multi-cycle"
)

# 首条长 Prompt 验证 HTP P21 / A+B 预取 / Planner Replay
python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts3 `
    --out "$root\32b-ablation-wo-multi-cycle-smoke" --limit 1
if ($LASTEXITCODE -ne 0) { throw "G-P-G Smoke 失败，不能运行全量" }

# 50 条正式运行
python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts3 `
    --out "$root\32b-ablation-wo-multi-cycle-all50"

# 意外中断时恢复：
# python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts3 `
#     --out "$root\32b-ablation-wo-multi-cycle-all50" --resume
```

## 逐条 PASS 的原生日志条件

```text
[REGION_RING_MULTI_CYCLE_ABLATION] mode=G_P_G logical=G36,P21,G7 physical=G7,G7,G8,G7,G7,P21,G7 gpu_layers=43 phone_layers=21 gpu_regions=6 phone_regions=1 gpu_banks=2 async_prefetch=1 drain_first=1 status=READY
[REGION_RING_ABLATION] gpu_banks=2 gpu_prefetch=1 gpu_bank_mover=1 sync_demand=0 status=CONFIGURED
[REGION_RING_CYCLE_SCHED] mode=DRAIN_FIRST
[REGION_RING_GPU_BANK_MOVER_SUM] worker=CPU_SINGLE started=4 completed=4 failures=0 ... thread_started=1 thread_joined=1 status=PASS
[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT ... source=LOCKED_CHUNK_REPLAY ... status=PASS
```

脚本会校验这些条件，以及每条请求的 token 数和 `XC/XG/XP` 是否与 Full Cycle 完全一致；任何一个条件不满足都不能标记该条 PASS。

## 文件

`32b-ablation-wo-multi-cycle-all50/requests.jsonl`、`requests.csv`、`summary.csv`、`metrics.jsonl`、`llama-server.log` 和 `server-run-config.json`。
