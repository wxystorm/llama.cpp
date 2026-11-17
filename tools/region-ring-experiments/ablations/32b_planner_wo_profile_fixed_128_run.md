# 32B Planner 消融：w/o Dynamic Chunk（固定 Chunk=128）

## 实验约束

- 仅关闭**逐请求的 Profile/DAG Chunk 搜索**，固定 `XC=XG=XP=128`；不调用请求期 `llama_hybrid_region_ring_planner_apply_wide`
- **用户指定的短句规则**：`tokens < 128` 时，不降低至 16/24/32/64 等候选格点，直接 `XC=XG=XP=tokens`；整个 Prompt 为一个 Chunk，不切分
- `tokens >= 128` 时按固定 128 执行；原先的 tiny-tail uniform coalescing 仍允许按既有规则微幅增大实际 Chunk，不改变其 Planner 选定值
- 保留 Full Cycle 的层数和拓扑：`G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7`；两端 GPU 43 层、Phone 21 层
- 保留两个 GPU Bank、后台异步 H2D Refill、Drain-First、Phone HTP Graphless、多 Session、模型常驻 llama-server 和未缓存的全部 ShareGPT50 请求
- 对单 Chunk 短句设置 `LLAMA_REGION_RING_GPU_BANK_FORCE_SMALL_BATCH=1`，强制保留原来的异步双 Bank Mover；不能为满足两个 GPU Chunk 的通常门槛而人为拆分 Prompt
- 模型加载阶段仍复用独立 Profile / 原始布局与初始化逻辑；消融范围是**请求期动态 Chunk 选择**，不宣称完全移除模型加载期 Profile

## 更新、编译、测试

```powershell
cd E:\llama\llama-tensor-v2
git switch feat/region-ring-sharegpt50-short-first-long-20261116
git pull --ff-only
cmake --build .\build-tensor-v2 --config Release --target llama-server --parallel 8
if ($LASTEXITCODE -ne 0) { throw "llama-server build failed" }

python -m unittest discover -s .\tools\region-ring-experiments -p "test_probe_htp_llama_server.py"
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
```

## 50 条 Prompt 运行命令

```powershell
$root = "E:\llama\llama-tensor-v2"
$baseline = "$root\32b-server-chunk256-all50-new"
$cfg = Get-Content "$baseline\server-run-config.json" -Raw | ConvertFrom-Json

$opts4 = @(
    "--server-exe", "$root\build-tensor-v2\bin\Release\llama-server.exe",
    "--model", $cfg.configuration.model.path,
    "--model-layers", "64",
    "--locked-layout", "G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7",
    "--rpc", $cfg.configuration.rpc,
    "--prompt-dir", $cfg.configuration.prompt_dir,
    "--profile-source", "$baseline\independent-profile",
    "--port", "18080",
    "--threads", "8",
    "--all50",
    "--per-request-chunks",
    "--fixed-chunk", "128"
)

# 前两条 Smoke: 第一条 1855 tokens，第二条 66 tokens（验证短句不拆分）
python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts4 `
    --out "$root\32b-planner-wo-profile-fixed128-smoke" --limit 2
if ($LASTEXITCODE -ne 0) { throw "fixed-128 smoke failed" }

# 50 条正式实验
python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts4 `
    --out "$root\32b-planner-wo-profile-fixed128-all50"

# 中断后恢复同一个结果目录
# python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts4 `
#     --out "$root\32b-planner-wo-profile-fixed128-all50" --resume
```

不要传入 `--replay-chunks-from`、`--no-multi-cycle`、`--no-drain-first` 或 `--no-async-prefetch`。

## 逐请求日志验证

```text
[SERVER_PLANNER_ABLATION] mode=WO_DYNAMIC_CHUNK fixed_chunk=128 short_prompt=WHOLE_PROMPT_SINGLE_CHUNK native_chunk_search=DISABLED_PER_REQUEST gpu_banks=2 async_prefetch=1 drain_first=1 ...
[REGION_RING_FIXED_CHUNK] target=128 tokens=66 XC=66 XG=66 XP=66 policy=WHOLE_PROMPT_SINGLE_CHUNK gpu_small_batch_mover=FORCED profile_queries=0 status=PASS
[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT tokens=66 ... XC=66 XG=66 XP=66 ... source=FIXED_CHUNK_ABLATION ... status=PASS
[REGION_RING_GPU_BANK] phase=INIT banks=2 ... force_small_batch=1 override_applied=1 status=READY
[REGION_RING_GPU_BANK_MOVER_SUM] worker=CPU_SINGLE started=4 completed=4 failures=0 ... thread_started=1 thread_joined=1 status=PASS
```

脚本强制核对 Chunk、原生 Runtime 完成日志、后台异步 Bank Mover、Drain-First 和原有固定层分配。任何条件不符都不记为 PASS。

正式结果位于 `32b-planner-wo-profile-fixed128-all50/requests.jsonl`、`requests.csv`、`summary.csv`、`metrics.jsonl` 及 `llama-server.log`。
