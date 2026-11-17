# 32B ShareGPT50 Planner 消融 6：w/o Profile-Guided Placement

## 实验配置

- 模型：Qwen3-32B-Q4_K_M；ShareGPT50_short，5 组共 50 条 Prompt
- 分层：`G7,N5,G7,N5,G7,N5,G7,N5,G7,N4,G5`（G40/N24/C0，脚本中 N=P）
- 方法：按内存比例静态分层；保留 DAG 动态 XC/XG/XP、GPU 双 Bank、异步预取、Drain-First、Phone HTP
- 内存政策：跳过 Planner **估算**检查，真实设备内存分配保持开启

## 2026-11-18 首次运行记录

| 实验 | 已完成 Prompt | Prefill 吞吐 (tok/s) | 状态 |
|---|---:|---:|---|
| Full Cycle | 50/50 | 123.037 | PASS |
| w/o Profile-Guided Placement（G40/N24） | 0/50 | — | **PHONE_OOM_REPORTED；待手机日志核对** |

**用户在手机端观察到 OOM 崩溃。** 当前先按「手机端 OOM（用户观察，待 Android 日志确认）」保留实验记录，不将该次结果当作 0 tok/s，也不计算提速倍数。

Windows `llama-server` 的可见报错发生在启动模型阶段，尚未进入任何 Prompt 的 Chunk 规划／计算。记录到的末端症状为：

```text
common_fit_params: failed to fit params to free device memory: llama_params_fit is not implemented for SPLIT_MODE_TENSOR, abort
llama_prepare_model_devices: skipping CPU (...) for tensor parallelism
llama_prepare_model_devices: LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices
llama_model_load_from_file_impl: failed to load model
llama_server: exiting due to model loading error
[MEMORY_PROPORTIONAL] status=CRASH_UNCONFIRMED passed=0
```

**因果说明：** Phone RPC/HTP 服务如果被 Android 的低内存机制杀死，确实可能导致 PC 侧远程设备不可见；但以上 Windows 日志本身只证明了无可用 Tensor Split 设备，不能独立证实手机 OOM。尚需对照手机 `logcat` 中的 `lmkd`、`lowmemorykiller`、`Out of memory`、`ENOMEM`、`HTP`／`FastRPC` 分配失败事件，以及 PC 端完整的 CUDA／RPC 初始化记录。

**待核对证据：** Android `logcat`（崩溃时间附近）及 Phone RPC 运行日志。若确认 LMKD kill 或原生分配失败，则将结果明确归档为 `PHONE_OOM_CONFIRMED`；若无法确认，则保留 `PHONE_OOM_REPORTED`。不要用 PC 侧无设备报错直接替代 OOM 证据。
