# Hexagon RPC 多层 Q/K/V 正确性验证（阶段 2）

目标：在已经验证 K14 + 通用 `SET_ROWS` 同步的基础上，将 HTP/QKV 计算从单一 K14 扩展到更多投影和更多层，并确保**实际执行位置**和**中间数值**均有证据。异常的 Combined tok/s 不在本阶段继续追踪。

## 与上一阶段的区别

- 固定在 PC 运行的 `llama-cli`、模型、上下文长度和 KV F16 设置。
- 一个基线 `cpu`：把参与实验的所有 Q/K/V 权重强制留在 CPU；KV Cache 仍按公共选项使用 RPC。
- 一个候选用例：仅释放一个投影（`L-q` / `L-k` / `L-v`）、释放整层 QKV（`L-qkv`），或释放范围内所有层 QKV（`all-qkv`）给 `-ngl` 所选 RPC/HTP 后端。其他投影均固定 CPU，FFN 和 attention output 权重固定 CPU。
- PC `LLAMA_TENSOR_COMPARE_DIR` 回调采集 layer 13–15 的 `Qcur/Kcur/Vcur/kq/kq_soft_max/kqv/kqv_out/attn_out/l_out`。
- Python `summarize_hexagon_qkv_layers.py` 用名称、四维形状、出现次数匹配；按 CPU 的 capture ID 顺序比较；输出 `tensor_metrics.csv` 和 `summary.json`。
- 手机的 `GGML_HEXAGON_QKV_AUDIT=1` 为已完成的 DSP opbatch 输出 `[HEX_QKV_AUDIT]`，核实目标 QKV 权重确实参与 HTP 提交和 batch ACK。这个标记不等于逐算子独立数值证明。

## 手机端部署

新分支：`test/hexagon-multilayer-qkv-20261012`，基于 `fix/hexagon-set-rows-input-sync-20261011`。

新增 QKV 审计只修改 **Android 端** `ggml-hexagon.cpp`，HTP DSP ABI 没有为本阶段改变。若已部署基分支的最新 DSP 库，原则上只需重新构建/部署宿主 `libggml-hexagon.so` 和对应 Server；务必保证实际启动进程加载的是新库。

手机 Termux，启动前设置：

```bash
export GGML_HEXAGON_SET_ROWS_HOST_INPUT_SYNC=1
export GGML_HEXAGON_QKV_AUDIT=1
export GGML_HEXAGON_K14_FORCE_INPUT_FLUSH=1
export GGML_HEXAGON_OPFUSION=0

unset GGML_HEXAGON_KV14_SET_ROWS_FORCE_INPUT_FLUSH
unset GGML_HEXAGON_KV14_SET_ROWS_DIAG
unset GGML_HEXAGON_PROFILE
unset GGML_HEXAGON_SET_ROWS_SYNC_TIMING

./bin/ggml-rpc-server --device HTP0 --host 0.0.0.0 --port 50052 \
  2>&1 | tee rpc-multilayer.log
```

检查设备日志（或在测试完成后）：

```bash
grep 'HEX_QKV_AUDIT' rpc-multilayer.log
```

成功的 NPU 投影应出现类似：

```text
[HEX_QKV_AUDIT] session=... batch_id=... batch_status=OK weight=blk.14.attn_k.weight ... compute_enabled=1
```

如果 PC 比较显示正确，但对应的 Q/K/V 权重始终没有该类记录，不能声称 HTP 实际执行了此算子；先检查实际的 RPC/CPU 放置、日志级别、是否部署新库以及 `GGML_HEXAGON_OPFUSION`。

## PC 运行

确认已经取到分支，PC 可以复用前一次构建的 `llama-cli.exe`（本阶段没有修改 PC 代码）。

```powershell
git fetch origin
git switch test/hexagon-multilayer-qkv-20261012

# 默认：CPU 基线、13-K、14-K、15-K、14-QKV、13~15 全 QKV
.\scripts\verify_hexagon_qkv_layers.ps1
```

可先只测试两个新增层：

```powershell
.\scripts\verify_hexagon_qkv_layers.ps1 `
    -Cases @("13-k", "15-k") `
    -Tokens 1 -Overwrite
```

一次完整覆盖三层所有 Q/K/V：

```powershell
.\scripts\verify_hexagon_qkv_layers.ps1 -All -Tokens 1 -Overwrite
```

当单 token Prefill 检查通过时，`-Tokens 8` 进一步验证 Decode，但注意：生成 token 若因极小数值差异发生采样分叉，后续张量差异不能直接视作错误。默认 `-Tokens 1` 是更清楚的首轮对照。

如要继续推广到 10–15 层，先改 `-ngl 6`、层范围和测试用例，再生成**新的** CPU 基线：

```powershell
.\scripts\verify_hexagon_qkv_layers.ps1 `
    -Ngl 6 -ModelLayers 16 -StartLayer 10 -EndLayer 15 `
    -Cases @("10-k","11-k","12-k","all-qkv") `
    -OutDir ".\hexagon-qkv-ngl6"
```

不建议在同一输出目录混合不同 `ngl`、模型或源代码版本的 Dump；旧目录存在时脚本默认拒绝覆盖，明确使用 `-Overwrite` 才覆盖。

## 结果

```text
hexagon-qkv-validation/
  settings.txt
  overview.csv
  cpu/
    manifest.tsv
    tensor_*.bin
    placement.txt
    run.log
  13-k/
    manifest.tsv
    tensor_*.bin
    placement.txt
    run.log
    compare/
      tensor_metrics.csv
      summary.json
  14-k/ ...
```

`overview.csv` 的三种 `status`：

- `NUMERICALLY_CLOSE`：没有达到严重错误的启发式阈值，但可能有精度正常的 `DIFF`；**必须再核对手机 QKV 执行审计**。
- `CRITICAL`：出现非有限值、余弦相似度低于 0.95，或部分 Attention 输出 RMSE 高于 0.05；这些是故障筛选门槛，不是普适数值正确性定理。
- `CAPTURE_MISMATCH`：两组没有匹配的张量或捕获集合不一致，需先排查参数/图结构。不可将未捕获的输出视为正确。

所有 `DIFF` 的原始 RMSE、cosine、bad/total 会保存在 `tensor_metrics.csv`；复现时应基于这些数字判别，避免“任意非零误差都是 bug”。

**限制：** Dump 回调主动等待生产者，可能掩盖调度 / 竞态条件；其结果必须在无 Dump 的端到端 `-n 30` 或更长 Decode 中复核。此阶段默认 `--kv-offload --cache-type-k f16 --cache-type-v f16 -fa off`，还不是 FlashAttention/NPU 全算子覆盖的证明。
