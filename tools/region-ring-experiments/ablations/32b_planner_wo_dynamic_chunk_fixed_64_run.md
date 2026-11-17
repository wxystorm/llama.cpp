# 32B Planner 消融：w/o Adaptive Chunk（固定 Chunk=64）

## 设计说明

- 本消融固定请求期 Chunk 为 XC=XG=XP=64；对于少于 64 tokens 的请求，采用整个 Prompt 作为单个 Chunk（XC=XG=XP=tokens）。
- 保留 Full Cycle 的固定 Placement：G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7，以及 GPU 双 Bank、异步预取、Drain-First 和 Phone HTP。
- 禁用**请求期** Profile/DAG Chunk 搜索（profile_queries=0）；模型加载与初始化时仍使用相同的 independent Profile。应在论文中称为 w/o Adaptive Chunk，而不是完全 w/o Profile。
- 此模式和固定 128 共用 --fixed-chunk 参数，仅允许 64 或 128。固定 128 的独立说明保留在 32b_planner_wo_profile_fixed_128_run.md。
- 0–128 组的实际 Prompt 通常有 64–128 tokens，因此固定 64 时这组大部分请求会按 64 分块；不能把 66-token 请求当成“小于 64、不拆分”的示例。

## Windows PowerShell：更新、编译、测试

~~~powershell
cd E:\llama\llama-tensor-v2
git switch feat/region-ring-sharegpt50-short-first-long-20261116
git pull --ff-only origin feat/region-ring-sharegpt50-short-first-long-20261116

python -m unittest discover -s .\tools\region-ring-experiments -p "test_probe_htp_llama_server.py"
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }

cmake --build .\build-tensor-v2 --config Release --target llama-server --parallel 8
if ($LASTEXITCODE -ne 0) { throw "llama-server build failed" }
~~~

## 固定 64：先 Smoke，再完整跑 50 条

~~~powershell
$root = "E:\llama\llama-tensor-v2"
$baseline = "$root\32b-server-chunk256-all50-new-1"
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
    "--fixed-chunk", "64"
)

python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts4 `
    --out "$root\32b-planner-wo-profile-fixed64-smoke" --limit 2
if ($LASTEXITCODE -ne 0) { throw "fixed-64 smoke failed" }

python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts4 `
    --out "$root\32b-planner-wo-profile-fixed64-all50-1"

# 中断后可用同一个目录恢复：
# python .\tools\region-ring-experiments\probe_htp_llama_server.py @opts4 `
#     --out "$root\32b-planner-wo-profile-fixed64-all50-1" --resume
~~~

避免传入 --replay-chunks-from、--no-multi-cycle、--no-drain-first 或 --no-async-prefetch；这些是独立消融因素。

## 预期验证日志

~~~text
[SERVER_PLANNER_ABLATION] mode=WO_DYNAMIC_CHUNK fixed_chunk=64 ... gpu_banks=2 async_prefetch=1 drain_first=1 ...
[REGION_RING_FIXED_CHUNK] target=64 tokens=66 XC=64 XG=64 XP=64 policy=FIXED_64 gpu_small_batch_mover=FORCED profile_queries=0 status=PASS
[REGION_RING_REQUEST_CHUNK_PLAN] event=SELECT tokens=66 ... XC=64 XG=64 XP=64 ... source=FIXED_CHUNK_ABLATION ... status=PASS
[REGION_RING_GPU_BANK_MOVER_SUM] worker=CPU_SINGLE started=4 completed=4 failures=0 ... status=PASS
~~~

脚本还会校验实际 Chunk 和双 Bank 异步搬运；严格比较 Full Cycle 结果前，请使用相同模型、Placement、Profile 来源和 50 条输入。