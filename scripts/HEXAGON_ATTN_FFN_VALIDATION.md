# Hexagon HTP: 从 QKV 扩展到 Attention Output / FFN

## 2026-10-12：固定 Layer13 Q 到 CPU 后，异常转移到 V

本轮在 `-PinQCpuLayer 13` 下重复 `baseline → 15-output → baseline-repeat`。新 `overview(6).csv`：

| case | compared | upstream_diff | reentry_diff | first_reentry_name | target_critical | overall_critical | max_normalized_rmse |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 15-output | 120 | 0 | 49 | Vcur-13 | 1 | 10 | 0.2309075 |
| baseline-control | 120 | 0 | 0 | none | 0 | 0 | 0 |

两组 `missing_stages` 为空，only_baseline/only_candidate 也均为 0。目标层 15-output 的 `worst_target_nrmse=0.057566`，不同于之前 Q13 on HTP 的 `0.089886`。需要注意所有下游误差可能受单列上游异常连带影响。

**这并不证明 V13 已全零**。`first_reentry_name=Vcur-13` 只说明它是第 2 次图计算最早超过阈值的捕获张量；应先在已有 Dump 上逐 token 检查 `candidate_all_zero`。

```powershell
git pull --ff-only
py -3 .\scripts\diagnose_hexagon_token_columns.py `
    .\hexagon-linear-attn-q13-cpu\attn\baseline `
    .\hexagon-linear-attn-q13-cpu\attn\15-output `
    --tokens 42 --top 2 `
    --name-regex '^(attn_norm|Qcur|Kcur|Vcur)-13
## 2026-10-12 token 列定位：Qcur-13 42-token 的第三列异常

最新保存的 `overview(5).csv` 和 `token_columns_42.csv` 给出明确的首次差异：

- 相同 QKV offload + Attention Output CPU baseline 运行两次，120 个张量完全一致；`baseline-control` 为 `NUMERICALLY_CLOSE`，无缺失及数值差异。
- 候选将 Layer15 `attn_output` 放到 NPU：`REENTRY_UPSTREAM_DIVERGENCE`，`upstream_diff=0`，`reentry_diff=47`，`first_reentry_name=Qcur-13`，`target_diff=2`、`target_critical=1`，`only_baseline=only_candidate=0`。上轮 CAPTURE_MISMATCH 是错误要求可缺失的 `ffn_out` callback，已消除。
- **第二组 42-token** `attn_norm-13` 每个 token 列都完全一致，Layer13 K/V 亦完全一致。`Qcur-13` 的 **token_column=2** 出现 2045/2048 个超差元素，NRMSE=1.0、cosine=NaN，而其余 41 个 token 的 Qcur 全部相同；三个 Qcur（预 RoPE、变形、后 RoPE）均在此列异常。
- 这组特征（输入一致 + Q 输出 NRMSE=1 + 候选零范数导致 cosine 未定义 + 无 nonfinite）**强烈表明候选的 Layer13 Q 投影第 2 列全部为零**。新版本 token-column 工具输出 baseline/candidate RMS、零元素数量、`candidate_all_zero`，可以用原始 `tensor_*.bin` 确认。
- `kqv_out-13`, `attn_out-13`, `l_out-13` 都只在 token_column=2 不同。到了 Layer14，因 token2 的 Key/Value 改变，后续列 2～41 的 causal attention 输出逐步扩散；因此 Layer15 的大 RMSE 很可能是此处**级联结果**，不能用来断定 Layer15 Attention Output MatMul 独立计算错误。
- 注意总体输入是否真正属于同一批次/同一 prefill 的不同阶段仍需 PC CLI 日志确定，不要仅依两次 graph callback 假设是两个连续 Decode token。

离线确认 Q13 异常列是否全零（只读旧 Dump，不重跑模型）：

```powershell
git pull --ff-only
py -3 .\scripts\diagnose_hexagon_token_columns.py `
  .\hexagon-linear-attn-control\attn\baseline `
  .\hexagon-linear-attn-control\attn\15-output `
  --tokens 42 --top 2 `
  --name-regex '^(attn_norm|Qcur|Kcur|Vcur)-13
## 2026-10-12 baseline-repeat 实测与二次修正

用户新的 `-Cases 15-output -FocusLayer 15 -BaselineControl` 结果表明：

- `baseline-control`：120/120 张量一一对应，120 个全零差异，所有 RMSE=0、cosine=1。这次实验中**相同放置运行两次严格一致**，不能再把异常解释成普通的随机波动。
- `15-output`：120/120 一一对应（两个侧的 only 均为 0），首个 2-token `attn_out-15` RMSE=0.000142 / NRMSE=0.003881；第二次 42-token `attn_out-15` RMSE=0.010720 / NRMSE=0.089886 / cosine=0.995966。
- 第二次 42-token `attn_out-13` RMSE=0.017045 / NRMSE=0.345685，超差元素 2038/86016，接近 *一个完整 token 列* 的 2048 元素；`ffn_gate-13` 也有 8168/344064 个超差，接近一个 8192 元素列。应定位具体 token 列、当次计算的上游张量和输入 token ID。
- 旧比较器错误要求 `ffn_out-13/14/15`。实际这三者在两个侧都没有捕获；已将 `ffn_out` 标记为可选，而仍要求每层 `attn_out` 与 `l_out`。同时 FFN down 的直接可观测输出改为 `l_out`。
- 旧比较器仅按整次运行最早的目标输出 ID=53 判断因果。实际基线有两组捕获：第 1 次 ne1=2，第 2 次 ne1=42。新版本按各次独立的 `attn_norm-13` 起点识别图计算组，分别检查该组 Layer15 的目标输出之前和之后的差异。

新报告新增 `REENTRY_UPSTREAM` 区域与 `REENTRY_UPSTREAM_DIVERGENCE` 状态。这不是“Layer 15 影响了同次计算的 Layer 13”，而是**下一次图计算在到达 Layer15 前已经与基线不同**。可能与上一轮状态、token 序列、KV cache 或 graph reuse 有关；无法单从两个 summary.json 证明具体原因。

### 再次离线重算原来的 Dump（不重跑模型）

```powershell
git pull --ff-only
py -3 .\scripts\test_hexagon_linear_focus.py

.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -RecompareOnly `
    -Cases @("15-output") -FocusLayer 15 `
    -OutDir ".\hexagon-linear-attn-control"
```

重点是结果中的 `graph_evaluation_starts`、`target_capture_ids_by_evaluation`、`reentry_diff`、`first_reentry_name`、`worst_target_nrmse`、`missing_stages`。预计 baseline-control 在修正后不再被误判 CAPTURE_MISMATCH；需要把该组也一起离线重算时，附加 `-BaselineControl`。

### 进一步确认是否只有某一个 token 列不同

```powershell
py -3 .\scripts\diagnose_hexagon_token_columns.py `
  .\hexagon-linear-attn-control\attn\baseline `
  .\hexagon-linear-attn-control\attn\15-output `
  --tokens 42 --top 5 `
  --csv .\hexagon-linear-attn-control\attn\token_columns_42.csv
```

`token_columns_42.csv` 按张量名、token 列下标给出 bad/total、RMSE、NRMSE、cosine，可直接区分：

1. `attn_norm-13` 和 `Qcur-13` 在某个 token 列已经不同 → 进入第 13 层前的输入不同或跨次状态变化；先查两个 PC 日志的 token ID、KV seq/pos。
2. 第 13 层前后输出不同但输入完全一致 → 检查 RPC 图执行、缓存和 K/V 张量同步。
3. `kqv_out-15` 两侧接近但 `attn_out-15` 显著不同 → 针对被迁移的 `attn_output` MatMul kernel/输入可见性继续隔离。

**禁止**仅凭通过的 batch ACK 或第一轮 2-token 低误差直接宣布 Attention Output 正确，FFN 阶段仍需等待这些问题解释清楚。


分支：`test/hexagon-attn-ffn-20261012`；基于已经通过有限数值对照的 `test/hexagon-multilayer-qkv-20261012`。

之前的 `all-qkv` 在带 Tensor Dump 的 84 个张量上数值接近 CPU；尚未看到上一轮用户无 Dump 的最终 64-token Decode 报告，不能假定无 Dump 已通过。

## 2026-10-12 Attention Output 测试问题与修正

已获得 HTP Batch OK 且 compute_enabled=1 的 blk.13.attn_output.weight、blk.15.attn_output.weight 审计。**这仅证明投影实际提交 HTP，不代表数值通过。**

前一轮 overview 结果：

| case | compared | only_baseline | only_candidate | critical | min_cosine | max_RMSE | worst_output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 13-output | 120 | 0 | 0 | 13 | -0.795295 | 0.316798 | attn_out-14 |
| 15-output | 120 | 0 | 0 | 11 | 0.944009 | 2.232093 | attn_out-13 |

**注意 1：** CAPTURE_MISMATCH 的含义是预期的某些阶段未被捕获，并不代表两组的 manifest key 集合不同（它们的 only count 均为 0）。必须检查报告中的 missing_stages。

**注意 2：** 仅迁移 15-output，却在 attn_out-13 发现明显差异，**不能只按层号判断因果关系**。如果这个快照出现在当前轮 Layer15 的目标输出之前，它是异常的上游偏差；如果出现在此前 Layer15 已改变采样 token 的下一次 Decode 中，它可能是正常的跨 token 传播。脚本以 baseline 的 **first_target_capture_id** 为近似时间屏障，只有屏障之前的差异列为 UPSTREAM；之后的早层快照归入 DOWNSTREAM，不能简单忽略它们。捕获 ID 是 callback 的顺序，不是 DSP 精确执行时间戳，仍需确认输入 token 序列。

修正后的比较器把张量划为三个区域：UPSTREAM（首次目标输出被捕获之前）、TARGET（被迁移算子的输出）、DOWNSTREAM（首次目标输出之后其他输出，**包括后续 Decode 的较低层**）。每个区域独立给出 diff、critical、首次差异和 worst NRMSE，保留 missing_stages 名单。可能状态包括 CAPTURE_MISMATCH、UPSTREAM_DIVERGENCE、TARGET_CRITICAL、DOWNSTREAM_CRITICAL、NUMERICALLY_CLOSE。

### 不重新运行模型，直接分析此前已有 Dump

在 PC 的对应分支更新：

```powershell
git pull --ff-only

# 使用已有 ./hexagon-linear-validation/attn/baseline,13-output,15-output
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -RecompareOnly `
  -Cases @("13-output","15-output")

# 聚焦 15-output，但不会隐藏它之前的差异
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -RecompareOnly `
  -Cases @("15-output") -FocusLayer 15
```

重新生成 attn/overview.csv 与 compare/summary.json、tensor_metrics.csv；不会清理原始 manifest 或重复执行 CLI。关注 missing_stages、upstream_diff、first_upstream_name、worst_target_name、worst_target_nrmse，并在 summary.json 查看 first_target_capture_id。若不同运行采样的 token 已经分叉，需要区分真实算子误差与后续输入不同。

### 复现时加 baseline 对照，识别跨次运行漂移

如果此前 15-output 仍表现为 UPSTREAM_DIVERGENCE：

```powershell
# 使用新目录而不是覆盖旧 Dump
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @("15-output") -FocusLayer 15 -BaselineControl `
  -OutDir ".\hexagon-linear-attn-control"
```

顺序是 baseline → 15-output → baseline-repeat。脚本会额外比较 baseline vs baseline-repeat，生成 baseline-control 行，并独立报告运行漂移。如果 baseline-control 都不一致，优先检查 RPC 缓存状态、调度、图回调和设备生命周期；如果 baseline-control 一致但 candidate 在 Layer 13 就分叉，检查目标权重放置或是否改变了执行图，使上游执行路径变化。

后续 FFN 阶段必须等到 Attention Output 真正数值可信、无 Dump Decode 稳定后再进行。

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

```

针对因果机制，重新运行一组小对照，**把 Layer13 Q 投影在 baseline 和 candidate 两边都固定 CPU**，其余 QKV/Attention output 放置保持原设置：

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @("15-output") -FocusLayer 15 -BaselineControl `
  -PinQCpuLayer 13 `
  -OutDir ".\hexagon-linear-attn-q13-cpu"
```

若本组异常消失，重点排查移动 `blk.15.attn_output.weight` 后图拆分/调度/分配变化如何影响 **仍然在 HTP 的 blk.13.attn_q.weight** 的某个 token 输出；若问题仍在，继续检查 Layer13 Q 后处理、host↔DSP cache 可见性和 RPC buffer 读写顺序。特别需要对应手机端的 `[HEX_QKV_AUDIT]` 记录，并在无 Dump 长 Decode 中再验证。

**不要在此阶段修改全局缓存刷新策略或推进 FFN**，先定位单列 Q 输出归零的机制。


## 2026-10-12 baseline-repeat 实测与二次修正

用户新的 `-Cases 15-output -FocusLayer 15 -BaselineControl` 结果表明：

- `baseline-control`：120/120 张量一一对应，120 个全零差异，所有 RMSE=0、cosine=1。这次实验中**相同放置运行两次严格一致**，不能再把异常解释成普通的随机波动。
- `15-output`：120/120 一一对应（两个侧的 only 均为 0），首个 2-token `attn_out-15` RMSE=0.000142 / NRMSE=0.003881；第二次 42-token `attn_out-15` RMSE=0.010720 / NRMSE=0.089886 / cosine=0.995966。
- 第二次 42-token `attn_out-13` RMSE=0.017045 / NRMSE=0.345685，超差元素 2038/86016，接近 *一个完整 token 列* 的 2048 元素；`ffn_gate-13` 也有 8168/344064 个超差，接近一个 8192 元素列。应定位具体 token 列、当次计算的上游张量和输入 token ID。
- 旧比较器错误要求 `ffn_out-13/14/15`。实际这三者在两个侧都没有捕获；已将 `ffn_out` 标记为可选，而仍要求每层 `attn_out` 与 `l_out`。同时 FFN down 的直接可观测输出改为 `l_out`。
- 旧比较器仅按整次运行最早的目标输出 ID=53 判断因果。实际基线有两组捕获：第 1 次 ne1=2，第 2 次 ne1=42。新版本按各次独立的 `attn_norm-13` 起点识别图计算组，分别检查该组 Layer15 的目标输出之前和之后的差异。

新报告新增 `REENTRY_UPSTREAM` 区域与 `REENTRY_UPSTREAM_DIVERGENCE` 状态。这不是“Layer 15 影响了同次计算的 Layer 13”，而是**下一次图计算在到达 Layer15 前已经与基线不同**。可能与上一轮状态、token 序列、KV cache 或 graph reuse 有关；无法单从两个 summary.json 证明具体原因。

### 再次离线重算原来的 Dump（不重跑模型）

```powershell
git pull --ff-only
py -3 .\scripts\test_hexagon_linear_focus.py

.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -RecompareOnly `
    -Cases @("15-output") -FocusLayer 15 `
    -OutDir ".\hexagon-linear-attn-control"
```

重点是结果中的 `graph_evaluation_starts`、`target_capture_ids_by_evaluation`、`reentry_diff`、`first_reentry_name`、`worst_target_nrmse`、`missing_stages`。预计 baseline-control 在修正后不再被误判 CAPTURE_MISMATCH；需要把该组也一起离线重算时，附加 `-BaselineControl`。

### 进一步确认是否只有某一个 token 列不同

```powershell
py -3 .\scripts\diagnose_hexagon_token_columns.py `
  .\hexagon-linear-attn-control\attn\baseline `
  .\hexagon-linear-attn-control\attn\15-output `
  --tokens 42 --top 5 `
  --csv .\hexagon-linear-attn-control\attn\token_columns_42.csv
```

`token_columns_42.csv` 按张量名、token 列下标给出 bad/total、RMSE、NRMSE、cosine，可直接区分：

1. `attn_norm-13` 和 `Qcur-13` 在某个 token 列已经不同 → 进入第 13 层前的输入不同或跨次状态变化；先查两个 PC 日志的 token ID、KV seq/pos。
2. 第 13 层前后输出不同但输入完全一致 → 检查 RPC 图执行、缓存和 K/V 张量同步。
3. `kqv_out-15` 两侧接近但 `attn_out-15` 显著不同 → 针对被迁移的 `attn_output` MatMul kernel/输入可见性继续隔离。

**禁止**仅凭通过的 batch ACK 或第一轮 2-token 低误差直接宣布 Attention Output 正确，FFN 阶段仍需等待这些问题解释清楚。


分支：`test/hexagon-attn-ffn-20261012`；基于已经通过有限数值对照的 `test/hexagon-multilayer-qkv-20261012`。

之前的 `all-qkv` 在带 Tensor Dump 的 84 个张量上数值接近 CPU；尚未看到上一轮用户无 Dump 的最终 64-token Decode 报告，不能假定无 Dump 已通过。

## 2026-10-12 Attention Output 测试问题与修正

已获得 HTP Batch OK 且 compute_enabled=1 的 blk.13.attn_output.weight、blk.15.attn_output.weight 审计。**这仅证明投影实际提交 HTP，不代表数值通过。**

前一轮 overview 结果：

| case | compared | only_baseline | only_candidate | critical | min_cosine | max_RMSE | worst_output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 13-output | 120 | 0 | 0 | 13 | -0.795295 | 0.316798 | attn_out-14 |
| 15-output | 120 | 0 | 0 | 11 | 0.944009 | 2.232093 | attn_out-13 |

**注意 1：** CAPTURE_MISMATCH 的含义是预期的某些阶段未被捕获，并不代表两组的 manifest key 集合不同（它们的 only count 均为 0）。必须检查报告中的 missing_stages。

**注意 2：** 仅迁移 15-output，却在 attn_out-13 发现明显差异，**不能只按层号判断因果关系**。如果这个快照出现在当前轮 Layer15 的目标输出之前，它是异常的上游偏差；如果出现在此前 Layer15 已改变采样 token 的下一次 Decode 中，它可能是正常的跨 token 传播。脚本以 baseline 的 **first_target_capture_id** 为近似时间屏障，只有屏障之前的差异列为 UPSTREAM；之后的早层快照归入 DOWNSTREAM，不能简单忽略它们。捕获 ID 是 callback 的顺序，不是 DSP 精确执行时间戳，仍需确认输入 token 序列。

修正后的比较器把张量划为三个区域：UPSTREAM（首次目标输出被捕获之前）、TARGET（被迁移算子的输出）、DOWNSTREAM（首次目标输出之后其他输出，**包括后续 Decode 的较低层**）。每个区域独立给出 diff、critical、首次差异和 worst NRMSE，保留 missing_stages 名单。可能状态包括 CAPTURE_MISMATCH、UPSTREAM_DIVERGENCE、TARGET_CRITICAL、DOWNSTREAM_CRITICAL、NUMERICALLY_CLOSE。

### 不重新运行模型，直接分析此前已有 Dump

在 PC 的对应分支更新：

```powershell
git pull --ff-only

# 使用已有 ./hexagon-linear-validation/attn/baseline,13-output,15-output
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -RecompareOnly `
  -Cases @("13-output","15-output")

# 聚焦 15-output，但不会隐藏它之前的差异
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -RecompareOnly `
  -Cases @("15-output") -FocusLayer 15
```

重新生成 attn/overview.csv 与 compare/summary.json、tensor_metrics.csv；不会清理原始 manifest 或重复执行 CLI。关注 missing_stages、upstream_diff、first_upstream_name、worst_target_name、worst_target_nrmse，并在 summary.json 查看 first_target_capture_id。若不同运行采样的 token 已经分叉，需要区分真实算子误差与后续输入不同。

### 复现时加 baseline 对照，识别跨次运行漂移

如果此前 15-output 仍表现为 UPSTREAM_DIVERGENCE：

```powershell
# 使用新目录而不是覆盖旧 Dump
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @("15-output") -FocusLayer 15 -BaselineControl `
  -OutDir ".\hexagon-linear-attn-control"
```

顺序是 baseline → 15-output → baseline-repeat。脚本会额外比较 baseline vs baseline-repeat，生成 baseline-control 行，并独立报告运行漂移。如果 baseline-control 都不一致，优先检查 RPC 缓存状态、调度、图回调和设备生命周期；如果 baseline-control 一致但 candidate 在 Layer 13 就分叉，检查目标权重放置或是否改变了执行图，使上游执行路径变化。

后续 FFN 阶段必须等到 Attention Output 真正数值可信、无 Dump Decode 稳定后再进行。

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

```

如果 V 同样出现某个 token 列全部清零，使用新增的三个独立 CPU 固定参数进一步隔离（请用各自独立新目录，避免覆盖旧 Dump）：

```powershell
# Q13+V13 在两边都 CPU，K13 仍走 HTP：
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -Ngl 4 -Tokens 1 `
    -Cases @("15-output") -FocusLayer 15 -BaselineControl `
    -PinQCpuLayer 13 -PinVCpuLayer 13 `
    -OutDir ".\hexagon-linear-attn-qv13-cpu"

# 如果仍有异常，再把 Q/K/V13 全部 pin 到 CPU：
.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -Ngl 4 -Tokens 1 `
    -Cases @("15-output") -FocusLayer 15 -BaselineControl `
    -PinQCpuLayer 13 -PinKCpuLayer 13 -PinVCpuLayer 13 `
    -OutDir ".\hexagon-linear-attn-qkv13-cpu"
```

核对新生成的 `placement.txt` 中的 override 规则；同时按每轮测试区间检查手机 `[HEX_QKV_AUDIT]`：CPU-pin 的权重应不再作为 HTP MatMul 出现。Batch 返回 OK 不等于实际输出非零。若异常从 Q 移到 V、再移到 K，优先调查同批 HTP QKV buffer 分配、列地址、图拆分和 RPC 缓存可见性；若全部留在 CPU 仍有差异，则转查 Attention/KV Cache 或其它上下游路径。变化也可能由图布局调整导致，不能直接把因果性归结到某个 kernel。

此阶段仍不继续 FFN。


## 2026-10-12 token 列定位：Qcur-13 42-token 的第三列异常

最新保存的 `overview(5).csv` 和 `token_columns_42.csv` 给出明确的首次差异：

- 相同 QKV offload + Attention Output CPU baseline 运行两次，120 个张量完全一致；`baseline-control` 为 `NUMERICALLY_CLOSE`，无缺失及数值差异。
- 候选将 Layer15 `attn_output` 放到 NPU：`REENTRY_UPSTREAM_DIVERGENCE`，`upstream_diff=0`，`reentry_diff=47`，`first_reentry_name=Qcur-13`，`target_diff=2`、`target_critical=1`，`only_baseline=only_candidate=0`。上轮 CAPTURE_MISMATCH 是错误要求可缺失的 `ffn_out` callback，已消除。
- **第二组 42-token** `attn_norm-13` 每个 token 列都完全一致，Layer13 K/V 亦完全一致。`Qcur-13` 的 **token_column=2** 出现 2045/2048 个超差元素，NRMSE=1.0、cosine=NaN，而其余 41 个 token 的 Qcur 全部相同；三个 Qcur（预 RoPE、变形、后 RoPE）均在此列异常。
- 这组特征（输入一致 + Q 输出 NRMSE=1 + 候选零范数导致 cosine 未定义 + 无 nonfinite）**强烈表明候选的 Layer13 Q 投影第 2 列全部为零**。新版本 token-column 工具输出 baseline/candidate RMS、零元素数量、`candidate_all_zero`，可以用原始 `tensor_*.bin` 确认。
- `kqv_out-13`, `attn_out-13`, `l_out-13` 都只在 token_column=2 不同。到了 Layer14，因 token2 的 Key/Value 改变，后续列 2～41 的 causal attention 输出逐步扩散；因此 Layer15 的大 RMSE 很可能是此处**级联结果**，不能用来断定 Layer15 Attention Output MatMul 独立计算错误。
- 注意总体输入是否真正属于同一批次/同一 prefill 的不同阶段仍需 PC CLI 日志确定，不要仅依两次 graph callback 假设是两个连续 Decode token。

离线确认 Q13 异常列是否全零（只读旧 Dump，不重跑模型）：

```powershell
git pull --ff-only
py -3 .\scripts\diagnose_hexagon_token_columns.py `
  .\hexagon-linear-attn-control\attn\baseline `
  .\hexagon-linear-attn-control\attn\15-output `
  --tokens 42 --top 2 `
  --name-regex '^(attn_norm|Qcur|Kcur|Vcur)-13
## 2026-10-12 baseline-repeat 实测与二次修正

用户新的 `-Cases 15-output -FocusLayer 15 -BaselineControl` 结果表明：

- `baseline-control`：120/120 张量一一对应，120 个全零差异，所有 RMSE=0、cosine=1。这次实验中**相同放置运行两次严格一致**，不能再把异常解释成普通的随机波动。
- `15-output`：120/120 一一对应（两个侧的 only 均为 0），首个 2-token `attn_out-15` RMSE=0.000142 / NRMSE=0.003881；第二次 42-token `attn_out-15` RMSE=0.010720 / NRMSE=0.089886 / cosine=0.995966。
- 第二次 42-token `attn_out-13` RMSE=0.017045 / NRMSE=0.345685，超差元素 2038/86016，接近 *一个完整 token 列* 的 2048 元素；`ffn_gate-13` 也有 8168/344064 个超差，接近一个 8192 元素列。应定位具体 token 列、当次计算的上游张量和输入 token ID。
- 旧比较器错误要求 `ffn_out-13/14/15`。实际这三者在两个侧都没有捕获；已将 `ffn_out` 标记为可选，而仍要求每层 `attn_out` 与 `l_out`。同时 FFN down 的直接可观测输出改为 `l_out`。
- 旧比较器仅按整次运行最早的目标输出 ID=53 判断因果。实际基线有两组捕获：第 1 次 ne1=2，第 2 次 ne1=42。新版本按各次独立的 `attn_norm-13` 起点识别图计算组，分别检查该组 Layer15 的目标输出之前和之后的差异。

新报告新增 `REENTRY_UPSTREAM` 区域与 `REENTRY_UPSTREAM_DIVERGENCE` 状态。这不是“Layer 15 影响了同次计算的 Layer 13”，而是**下一次图计算在到达 Layer15 前已经与基线不同**。可能与上一轮状态、token 序列、KV cache 或 graph reuse 有关；无法单从两个 summary.json 证明具体原因。

### 再次离线重算原来的 Dump（不重跑模型）

```powershell
git pull --ff-only
py -3 .\scripts\test_hexagon_linear_focus.py

.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -RecompareOnly `
    -Cases @("15-output") -FocusLayer 15 `
    -OutDir ".\hexagon-linear-attn-control"
```

重点是结果中的 `graph_evaluation_starts`、`target_capture_ids_by_evaluation`、`reentry_diff`、`first_reentry_name`、`worst_target_nrmse`、`missing_stages`。预计 baseline-control 在修正后不再被误判 CAPTURE_MISMATCH；需要把该组也一起离线重算时，附加 `-BaselineControl`。

### 进一步确认是否只有某一个 token 列不同

```powershell
py -3 .\scripts\diagnose_hexagon_token_columns.py `
  .\hexagon-linear-attn-control\attn\baseline `
  .\hexagon-linear-attn-control\attn\15-output `
  --tokens 42 --top 5 `
  --csv .\hexagon-linear-attn-control\attn\token_columns_42.csv
```

`token_columns_42.csv` 按张量名、token 列下标给出 bad/total、RMSE、NRMSE、cosine，可直接区分：

1. `attn_norm-13` 和 `Qcur-13` 在某个 token 列已经不同 → 进入第 13 层前的输入不同或跨次状态变化；先查两个 PC 日志的 token ID、KV seq/pos。
2. 第 13 层前后输出不同但输入完全一致 → 检查 RPC 图执行、缓存和 K/V 张量同步。
3. `kqv_out-15` 两侧接近但 `attn_out-15` 显著不同 → 针对被迁移的 `attn_output` MatMul kernel/输入可见性继续隔离。

**禁止**仅凭通过的 batch ACK 或第一轮 2-token 低误差直接宣布 Attention Output 正确，FFN 阶段仍需等待这些问题解释清楚。


分支：`test/hexagon-attn-ffn-20261012`；基于已经通过有限数值对照的 `test/hexagon-multilayer-qkv-20261012`。

之前的 `all-qkv` 在带 Tensor Dump 的 84 个张量上数值接近 CPU；尚未看到上一轮用户无 Dump 的最终 64-token Decode 报告，不能假定无 Dump 已通过。

## 2026-10-12 Attention Output 测试问题与修正

已获得 HTP Batch OK 且 compute_enabled=1 的 blk.13.attn_output.weight、blk.15.attn_output.weight 审计。**这仅证明投影实际提交 HTP，不代表数值通过。**

前一轮 overview 结果：

| case | compared | only_baseline | only_candidate | critical | min_cosine | max_RMSE | worst_output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 13-output | 120 | 0 | 0 | 13 | -0.795295 | 0.316798 | attn_out-14 |
| 15-output | 120 | 0 | 0 | 11 | 0.944009 | 2.232093 | attn_out-13 |

**注意 1：** CAPTURE_MISMATCH 的含义是预期的某些阶段未被捕获，并不代表两组的 manifest key 集合不同（它们的 only count 均为 0）。必须检查报告中的 missing_stages。

**注意 2：** 仅迁移 15-output，却在 attn_out-13 发现明显差异，**不能只按层号判断因果关系**。如果这个快照出现在当前轮 Layer15 的目标输出之前，它是异常的上游偏差；如果出现在此前 Layer15 已改变采样 token 的下一次 Decode 中，它可能是正常的跨 token 传播。脚本以 baseline 的 **first_target_capture_id** 为近似时间屏障，只有屏障之前的差异列为 UPSTREAM；之后的早层快照归入 DOWNSTREAM，不能简单忽略它们。捕获 ID 是 callback 的顺序，不是 DSP 精确执行时间戳，仍需确认输入 token 序列。

修正后的比较器把张量划为三个区域：UPSTREAM（首次目标输出被捕获之前）、TARGET（被迁移算子的输出）、DOWNSTREAM（首次目标输出之后其他输出，**包括后续 Decode 的较低层**）。每个区域独立给出 diff、critical、首次差异和 worst NRMSE，保留 missing_stages 名单。可能状态包括 CAPTURE_MISMATCH、UPSTREAM_DIVERGENCE、TARGET_CRITICAL、DOWNSTREAM_CRITICAL、NUMERICALLY_CLOSE。

### 不重新运行模型，直接分析此前已有 Dump

在 PC 的对应分支更新：

```powershell
git pull --ff-only

# 使用已有 ./hexagon-linear-validation/attn/baseline,13-output,15-output
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -RecompareOnly `
  -Cases @("13-output","15-output")

# 聚焦 15-output，但不会隐藏它之前的差异
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -RecompareOnly `
  -Cases @("15-output") -FocusLayer 15
```

重新生成 attn/overview.csv 与 compare/summary.json、tensor_metrics.csv；不会清理原始 manifest 或重复执行 CLI。关注 missing_stages、upstream_diff、first_upstream_name、worst_target_name、worst_target_nrmse，并在 summary.json 查看 first_target_capture_id。若不同运行采样的 token 已经分叉，需要区分真实算子误差与后续输入不同。

### 复现时加 baseline 对照，识别跨次运行漂移

如果此前 15-output 仍表现为 UPSTREAM_DIVERGENCE：

```powershell
# 使用新目录而不是覆盖旧 Dump
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @("15-output") -FocusLayer 15 -BaselineControl `
  -OutDir ".\hexagon-linear-attn-control"
```

顺序是 baseline → 15-output → baseline-repeat。脚本会额外比较 baseline vs baseline-repeat，生成 baseline-control 行，并独立报告运行漂移。如果 baseline-control 都不一致，优先检查 RPC 缓存状态、调度、图回调和设备生命周期；如果 baseline-control 一致但 candidate 在 Layer 13 就分叉，检查目标权重放置或是否改变了执行图，使上游执行路径变化。

后续 FFN 阶段必须等到 Attention Output 真正数值可信、无 Dump Decode 稳定后再进行。

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

```

针对因果机制，重新运行一组小对照，**把 Layer13 Q 投影在 baseline 和 candidate 两边都固定 CPU**，其余 QKV/Attention output 放置保持原设置：

```powershell
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @("15-output") -FocusLayer 15 -BaselineControl `
  -PinQCpuLayer 13 `
  -OutDir ".\hexagon-linear-attn-q13-cpu"
```

若本组异常消失，重点排查移动 `blk.15.attn_output.weight` 后图拆分/调度/分配变化如何影响 **仍然在 HTP 的 blk.13.attn_q.weight** 的某个 token 输出；若问题仍在，继续检查 Layer13 Q 后处理、host↔DSP cache 可见性和 RPC buffer 读写顺序。特别需要对应手机端的 `[HEX_QKV_AUDIT]` 记录，并在无 Dump 长 Decode 中再验证。

**不要在此阶段修改全局缓存刷新策略或推进 FFN**，先定位单列 Q 输出归零的机制。


## 2026-10-12 baseline-repeat 实测与二次修正

用户新的 `-Cases 15-output -FocusLayer 15 -BaselineControl` 结果表明：

- `baseline-control`：120/120 张量一一对应，120 个全零差异，所有 RMSE=0、cosine=1。这次实验中**相同放置运行两次严格一致**，不能再把异常解释成普通的随机波动。
- `15-output`：120/120 一一对应（两个侧的 only 均为 0），首个 2-token `attn_out-15` RMSE=0.000142 / NRMSE=0.003881；第二次 42-token `attn_out-15` RMSE=0.010720 / NRMSE=0.089886 / cosine=0.995966。
- 第二次 42-token `attn_out-13` RMSE=0.017045 / NRMSE=0.345685，超差元素 2038/86016，接近 *一个完整 token 列* 的 2048 元素；`ffn_gate-13` 也有 8168/344064 个超差，接近一个 8192 元素列。应定位具体 token 列、当次计算的上游张量和输入 token ID。
- 旧比较器错误要求 `ffn_out-13/14/15`。实际这三者在两个侧都没有捕获；已将 `ffn_out` 标记为可选，而仍要求每层 `attn_out` 与 `l_out`。同时 FFN down 的直接可观测输出改为 `l_out`。
- 旧比较器仅按整次运行最早的目标输出 ID=53 判断因果。实际基线有两组捕获：第 1 次 ne1=2，第 2 次 ne1=42。新版本按各次独立的 `attn_norm-13` 起点识别图计算组，分别检查该组 Layer15 的目标输出之前和之后的差异。

新报告新增 `REENTRY_UPSTREAM` 区域与 `REENTRY_UPSTREAM_DIVERGENCE` 状态。这不是“Layer 15 影响了同次计算的 Layer 13”，而是**下一次图计算在到达 Layer15 前已经与基线不同**。可能与上一轮状态、token 序列、KV cache 或 graph reuse 有关；无法单从两个 summary.json 证明具体原因。

### 再次离线重算原来的 Dump（不重跑模型）

```powershell
git pull --ff-only
py -3 .\scripts\test_hexagon_linear_focus.py

.\scripts\verify_hexagon_linear_layers.ps1 `
    -Stage attn -RecompareOnly `
    -Cases @("15-output") -FocusLayer 15 `
    -OutDir ".\hexagon-linear-attn-control"
```

重点是结果中的 `graph_evaluation_starts`、`target_capture_ids_by_evaluation`、`reentry_diff`、`first_reentry_name`、`worst_target_nrmse`、`missing_stages`。预计 baseline-control 在修正后不再被误判 CAPTURE_MISMATCH；需要把该组也一起离线重算时，附加 `-BaselineControl`。

### 进一步确认是否只有某一个 token 列不同

```powershell
py -3 .\scripts\diagnose_hexagon_token_columns.py `
  .\hexagon-linear-attn-control\attn\baseline `
  .\hexagon-linear-attn-control\attn\15-output `
  --tokens 42 --top 5 `
  --csv .\hexagon-linear-attn-control\attn\token_columns_42.csv
```

`token_columns_42.csv` 按张量名、token 列下标给出 bad/total、RMSE、NRMSE、cosine，可直接区分：

1. `attn_norm-13` 和 `Qcur-13` 在某个 token 列已经不同 → 进入第 13 层前的输入不同或跨次状态变化；先查两个 PC 日志的 token ID、KV seq/pos。
2. 第 13 层前后输出不同但输入完全一致 → 检查 RPC 图执行、缓存和 K/V 张量同步。
3. `kqv_out-15` 两侧接近但 `attn_out-15` 显著不同 → 针对被迁移的 `attn_output` MatMul kernel/输入可见性继续隔离。

**禁止**仅凭通过的 batch ACK 或第一轮 2-token 低误差直接宣布 Attention Output 正确，FFN 阶段仍需等待这些问题解释清楚。


分支：`test/hexagon-attn-ffn-20261012`；基于已经通过有限数值对照的 `test/hexagon-multilayer-qkv-20261012`。

之前的 `all-qkv` 在带 Tensor Dump 的 84 个张量上数值接近 CPU；尚未看到上一轮用户无 Dump 的最终 64-token Decode 报告，不能假定无 Dump 已通过。

## 2026-10-12 Attention Output 测试问题与修正

已获得 HTP Batch OK 且 compute_enabled=1 的 blk.13.attn_output.weight、blk.15.attn_output.weight 审计。**这仅证明投影实际提交 HTP，不代表数值通过。**

前一轮 overview 结果：

| case | compared | only_baseline | only_candidate | critical | min_cosine | max_RMSE | worst_output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 13-output | 120 | 0 | 0 | 13 | -0.795295 | 0.316798 | attn_out-14 |
| 15-output | 120 | 0 | 0 | 11 | 0.944009 | 2.232093 | attn_out-13 |

**注意 1：** CAPTURE_MISMATCH 的含义是预期的某些阶段未被捕获，并不代表两组的 manifest key 集合不同（它们的 only count 均为 0）。必须检查报告中的 missing_stages。

**注意 2：** 仅迁移 15-output，却在 attn_out-13 发现明显差异，**不能只按层号判断因果关系**。如果这个快照出现在当前轮 Layer15 的目标输出之前，它是异常的上游偏差；如果出现在此前 Layer15 已改变采样 token 的下一次 Decode 中，它可能是正常的跨 token 传播。脚本以 baseline 的 **first_target_capture_id** 为近似时间屏障，只有屏障之前的差异列为 UPSTREAM；之后的早层快照归入 DOWNSTREAM，不能简单忽略它们。捕获 ID 是 callback 的顺序，不是 DSP 精确执行时间戳，仍需确认输入 token 序列。

修正后的比较器把张量划为三个区域：UPSTREAM（首次目标输出被捕获之前）、TARGET（被迁移算子的输出）、DOWNSTREAM（首次目标输出之后其他输出，**包括后续 Decode 的较低层**）。每个区域独立给出 diff、critical、首次差异和 worst NRMSE，保留 missing_stages 名单。可能状态包括 CAPTURE_MISMATCH、UPSTREAM_DIVERGENCE、TARGET_CRITICAL、DOWNSTREAM_CRITICAL、NUMERICALLY_CLOSE。

### 不重新运行模型，直接分析此前已有 Dump

在 PC 的对应分支更新：

```powershell
git pull --ff-only

# 使用已有 ./hexagon-linear-validation/attn/baseline,13-output,15-output
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -RecompareOnly `
  -Cases @("13-output","15-output")

# 聚焦 15-output，但不会隐藏它之前的差异
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -RecompareOnly `
  -Cases @("15-output") -FocusLayer 15
```

重新生成 attn/overview.csv 与 compare/summary.json、tensor_metrics.csv；不会清理原始 manifest 或重复执行 CLI。关注 missing_stages、upstream_diff、first_upstream_name、worst_target_name、worst_target_nrmse，并在 summary.json 查看 first_target_capture_id。若不同运行采样的 token 已经分叉，需要区分真实算子误差与后续输入不同。

### 复现时加 baseline 对照，识别跨次运行漂移

如果此前 15-output 仍表现为 UPSTREAM_DIVERGENCE：

```powershell
# 使用新目录而不是覆盖旧 Dump
.\scripts\verify_hexagon_linear_layers.ps1 `
  -Stage attn -Ngl 4 -Tokens 1 `
  -Cases @("15-output") -FocusLayer 15 -BaselineControl `
  -OutDir ".\hexagon-linear-attn-control"
```

顺序是 baseline → 15-output → baseline-repeat。脚本会额外比较 baseline vs baseline-repeat，生成 baseline-control 行，并独立报告运行漂移。如果 baseline-control 都不一致，优先检查 RPC 缓存状态、调度、图回调和设备生命周期；如果 baseline-control 一致但 candidate 在 Layer 13 就分叉，检查目标权重放置或是否改变了执行图，使上游执行路径变化。

后续 FFN 阶段必须等到 Attention Output 真正数值可信、无 Dump Decode 稳定后再进行。

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
