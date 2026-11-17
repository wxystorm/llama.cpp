# 32B ShareGPT50 Planner 消融：w/o DAG / Queueing Simulation

- 模型：Qwen3-32B-Q4_K_M；ShareGPT50_short，0 / 128 / 256 / 512 / 1024 五组，各 10 条，共 50 条、27,417 tokens。
- 对比：Full Cycle（DAG Planner）vs. w/o DAG（线性成本评分）。
- Planner：各 Prompt 动态选择 XC/XG/XP，Placement 固定；关闭 DAG/Queueing Simulation。
- Runtime：GPU 双 Bank、异步预取、Drain-First；关闭 Planner 内存估算检查，实际设备分配仍生效。
- Full Cycle 基线：`32b-server-chunk256-all50-new/summary.csv`。
- w/o DAG：本次补测五组平均 Prefill 吞吐数据。

## 1. 分组 Prefill 吞吐与相对提速

下表两侧均采用每组 10 条 Prompt 的**算术平均 tok/s**，按同一指标比较。吞吐下降 = 1 − w/o DAG / Full Cycle；Full Cycle 提速 = Full Cycle / w/o DAG。

| Token 组 | 条数 | Tokens | Full Cycle 平均 (tok/s) | w/o DAG 平均 (tok/s) | w/o DAG 吞吐下降 | Full Cycle 提速 |
|---|---:|---:|---:|---:|---:|---:|
| 0–128 | 10 | 911 | 50.776 | 41.627 | 18.02% | 1.220× |
| 128–256 | 10 | 2,040 | 87.919 | 65.094 | 25.96% | 1.351× |
| 256–512 | 10 | 3,283 | 109.842 | 62.283 | 43.30% | 1.764× |
| 512–1024 | 10 | 8,032 | 135.169 | 87.883 | 34.98% | 1.538× |
| 1024–2048 | 10 | 13,151 | 142.376 | 116.576 | 18.12% | 1.221× |
| **全部（50 条算术平均）** | **50** | **27,417** | **105.216** | **74.693** | **29.01%** | **1.409×** |

## 2. 整体 Prefill 吞吐对比

| 实验 | 整体 Prefill (tok/s) | 相对变化 |
|---|---:|---:|
| Full Cycle（DAG） | 123.037 | — |
| w/o DAG | 88.402 | 吞吐下降 28.15% |
| **Full Cycle 相对 w/o DAG 提速** | **1.392×** | — |

**整体结果：** w/o DAG 的整体吞吐为 **88.402 tok/s**，Full Cycle 为 **123.037 tok/s**，Full Cycle 相对 w/o DAG 提速 **1.392×**。五组平均吞吐依次为 **41.627 / 65.094 / 62.283 / 87.883 / 116.576 tok/s**。
