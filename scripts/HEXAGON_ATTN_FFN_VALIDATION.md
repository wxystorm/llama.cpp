# Hexagon HTP: 从 QKV 扩展到 Attention Output / FFN

分支：`test/hexagon-attn-ffn-20261012`；基于已经通过有限数值对照的 `test/hexagon-multilayer-qkv-20261012`。

之前的 `all-qkv` 在带 Tensor Dump 的 84 个张量上数值接近 CPU；尚未看到上一轮用户无 Dump 的最终 64-token Decode 报告，不能假定无 Dump 已通过。

## 实验设计

本阶段固定 16-layer Llama 3.2 1B Q4_0、`-ngl 4`、`-fa off`、`--kv-offload`、K/V F16。`ngl=4` 包含 Layer 13, 14, 15 和输出层。

| Stage | 同一阶段的 baseline | candidate |
| --- | --- | --- |
| `attn` | QKV=NPU、Attention Output=CPU、FFN=CPU | 在 QKV=NPU 基础上，只释放选定的 Attention Output MatMul 到 NPU |
| `ffn` | QKV=NPU、Attention Output=NPU、FFN=CPU | 在相同 QKV 与 Attention Output 配置下，只释放选定 FFN gate/up/down 到 NPU |

FFN 阶段必须在 Attention Output 阶段通过后再运行。这个方案能隔离新 MatMul 的数值影响，而不是再把早先 QKV 误差混进对照。

这里只放开权重对应的 MatMul，并**不意味着整个 Attention/FFN 图都由 NPU 计算**。Softmax、RoPE、RMSNorm、Residual 等仍可能在 CPU 或 RPC 后端处理。

## 手机 Termux 配置

此分支仅更改 Android Hexagon 宿主端审计代码，无 DSP ABI 变化。需要重编/重部署新版本 `libggml-hexagon.so`，原则上无需再次替换已匹配的 HTP v79 DSP 库。重新启动 RPC Server 以读取新环境变量。

```bash
export GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC=1
export GGML_HEXAGON_QKV_AUDIT=1
export GGML_HEXAGON_LINEAR_AUDIT=1
export GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1
export GGML_HEXAGON_OPFUSION=0

unset GGML_HEXAGON_KV14_SET_ROWS_FORCE_INPUT_FLUSH
unset GGML_HEXAGON_KV14_SET_ROWS_DIAG
unset GGML_HEXAGON_PROFILE
unset GGML_HEXAGON_SET_ROWS_SYNC_TIMING
unset GGML_HEXAGON_K14_SENTINEL

./bin/ggml-rpc-server --device HTP0 --host 0.0.0.0 --port 50052 \
    2>&1 | tee rpc-linear.log
```

新日志：`[HEX_LINEAR_AUDIT] session=... batch_id=... batch_status=OK weight=blk.13.attn_output.weight output=attn_out-13 ... compute_enabled=1`。
同理可见 `blk.N.ffn_gate.weight`、`blk.N.ffn_up.weight`、`blk.N.ffn_down.weight`。

```bash
grep -E 'HEX_LINEAR_AUDIT|HEX_QKV_AUDIT|HEX_KV_AUDIT' rpc-linear.log
```

**注意**：Batch OK 只证明已提交的 HTP 计算批次正常返回，不是该输出与 CPU 逐元素相等的证明。若数值显示通过但没有目标权重对应的审计行，先查是否实际 CPU fallback、是否正确加载手机新宿主库。

## PC 阶段 1：Attention Output

```powershell
git fetch origin
git switch test/hexagon-attn-ffn-20261012

.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -Ngl 4 -Tokens 1 `
    -Cases @("13-output","15-output") `
    -OutDir ".\hexagon-linear-validation"
```

两组正常、目标权重实际在 HTP 后，扩展到全部：

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -Ngl 4 -Tokens 1 -Overwrite
```

将运行 baseline、13-output、14-output、15-output、all-output。

输出在 `hexagon-linear-validation/attn/overview.csv`。每个用例包含 `run.log`、`placement.txt`、`manifest.tsv`、`compare/tensor_metrics.csv` 和 `compare/summary.json`。

## PC 阶段 2：FFN

等 `all-output` 数值与无 Dump Decode 稳定，再跑：

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage ffn -Ngl 4 -Tokens 1 `
    -Cases @("13-gate","13-up","13-down","13-ffn")
```

再扩展完整 13、14、15 层 FFN 的逐权重和合并测试：

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage ffn -Ngl 4 -Tokens 1 -All -Overwrite
```

`src/llama-graph.cpp` 仅在下投影带偏置时生成 `ffn_down-N` 的中间 callback，Llama 3.2 1B 可能没有这个张量名；以 `ffn_out-N`、`l_out-N` 及 `[HEX_LINEAR_AUDIT] weight=blk.N.ffn_down.weight` 判断。

## 正确性比较与无 Dump Decode

新脚本 `scripts/summarize_hexagon_linear_layers.py` 输出 RMSE、Cosine、Normalized RMSE = RMSE / max(RMS(reference), 1e-6)。默认阈值是主要输出相对 RMS 误差 5% 或 Cosine <0.95，作为探索性的 CRITICAL 预警，并非模型精度标准。

- `NUMERICALLY_CLOSE`：在启发式阈值内。还要检查 `worst_output_name`、`worst_output_nrmse` 和手机 HTP 审计。
- `CRITICAL`：可能存在真正的数值错误；定位首次偏差张量及上游输入。
- `CAPTURE_MISMATCH`：缺少某侧捕获或必要的阶段张量；不能当作通过。
- 在 Decode 中有接近的 top logits 时，两边选中不同 token 会使后续张量不可直接比较。先用 `-Tokens 1`。

Tensor Dump 会改变调度边界，不适合测速。因此数值和审计初步通过后，关闭手机的 `GGML_HEXAGON_LINEAR_AUDIT` / `GGML_HEXAGON_QKV_AUDIT`（重启 RPC Server），保留通用同步，然后做 64-token 无 Dump 验证：

```powershell
# Attention Output：QKV=NPU 基线对照 QKV+Attention Output=NPU
.\scripts\verify_hexagon_linear_decode.ps1 `
    -Stage attn -Ngl 4 -Tokens 64 -Runs 3 -WarmupRuns 1

# 仅在 Attention Output 通过后：增加 FFN=NPU
.\scripts\verify_hexagon_linear_decode.ps1 `
    -Stage ffn -Ngl 4 -Tokens 64 -Runs 3 -WarmupRuns 1
```

分别输出 `hexagon-linear-decode/attn/results.csv` 和 `hexagon-linear-decode/ffn/results.csv`，运行顺序交替，减少设备频率、温度、初始化带来的系统性偏差。检查是否崩溃、NAN/INF 或无意义输出。

后续如需扩大到 Layer 10～15，设置 `-Ngl 7 -StartLayer 10 -EndLayer 15` 并使用新目录；不要把不同模型/层数/放置方案的 Dump 混用。

目前仍是实验分支，**不要直接合并到 `plan-moe-edge`**。
