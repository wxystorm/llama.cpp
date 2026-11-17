# 32B ShareGPT50 Planner 消融 6：w/o Profile-Guided Placement

## 实验配置

- 模型：Qwen3-32B-Q4_K_M；ShareGPT50_short，5 组共 50 条 Prompt
- 分层：`G7,N5,G7,N5,G7,N5,G7,N5,G7,N4,G5`（G40/N24/C0；脚本中 N=P）
- 方法：静态内存比例 Placement；保留 DAG 动态 XC/XG/XP、GPU 双 Bank、异步预取、Drain-First、Phone HTP
- 内存政策：跳过 Planner 的**估算**容量检查，真实设备内存分配保持开启

## 2026-11-18 消融运行结果

| 实验 | 完成 Prompt | 整体 Prefill（tok/s） | 状态 |
|---|---:|---:|---|
| Full Cycle | 50/50 | 123.037 | PASS |
| w/o Profile-Guided Placement（G40/N24） | **0/50** | **—** | **PHONE_OOM（Android LOW_MEMORY 退出记录支持）** |

本次实验未完成模型加载，未运行任何 Chunk 或 Prompt。**OOM 属于可行性失败记录，不能写成吞吐为 0 tok/s，也不能据此计算速度比。**

## 证据

1. **Android 系统退出记录（本次补充）：** `adb shell dumpsys activity exit-info com.termux` 的筛选结果包含 **15 条** `process=com.termux reason=3 (LOW_MEMORY)`，其中 **5 条**含 `status=9`；这是系统层面确实发生过 Termux 低内存退出的直接证据。另有一条 `reason=10 (USER REQUESTED)`，与低内存退出不同，不能混淆。
2. **Android logcat 筛选结果：** 在 `11-18 03:38:16` 附近存在 `./bin/ggml-rpc-server` 相关 PackageUtils / MQSEventManager 消息，但所提供的片段未包含对应 PID 的 lmkd kill 明细。
3. **Windows 侧症状：** `llama-server` 在模型加载阶段报 `LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices`，随后 `failed to load model`，实验脚本记录 `CRASH_UNCONFIRMED passed=0`。这表明当时 PC 无可用 Tensor Split 设备；与手机 RPC 服务退出可能相符，但 PC 日志自身不足以确认 Phone OOM。

```text
process=com.termux reason=3 (LOW_MEMORY) subreason=0 (UNKNOWN) status=9
LLAMA_SPLIT_MODE_TENSOR needs >= 1 devices
[MEMORY_PROPORTIONAL] status=CRASH_UNCONFIRMED passed=0
```

**证据边界：** 本次提供的是 `Select-String` 筛选后的文本，而非含时间戳的完整 `termux-exit-info.txt`。系统低内存退出**已经确认发生过**，但尚无法仅据筛选结果将某一条 LOW_MEMORY 退出与 G40/N24 运行精确关联。此处实验状态综合用户当时观察到的手机 OOM 和 Android 系统低内存退出记录归档，不能声称已证明某个特定 HTP 内存分配失败。若需严格复现实验因果链，应保存崩溃时间戳、Termux 退出 PID 和 RPC 服务日志。

## 数据口径

此项消融为 **OOM / Infeasible（0/50 完成）**。主表宜使用 `OOM` 或 `N/A (OOM)`，不填虚构吞吐或加速比；其余 Full Cycle 与消融的成功吞吐结果保持不变。
