# Phone–PC Hybrid Inference 使用与编译说明

## 1. 当前版本

仓库：

`wxystorm/llama.cpp`

分支：

`planner-moe-stage`


当前主要测试模型：

`Qwen3-30B-A3B-Q4_K_M.gguf`

当前运行架构为：

- PC：Windows + NVIDIA CUDA + CPU
- Phone：Android / Termux + Qualcomm Adreno OpenCL
- PC 作为主控端运行 `llama-cli`
- Phone 运行 `ggml-rpc-server`
- PC 通过 RPC 调用手机 GPU
- `--hybrid-auto` 自动 profile CPU / CUDA GPU / Phone RPC，并搜索混合执行方案


为了保证结果一致，PC 和手机建议使用同一份 GGUF 模型。

---

## 2. 获取代码

PC 和手机均切到同一个分支：

```bash
git fetch origin
git switch planner-moe-stage
git pull --ff-only origin planner-moe-stage
```

检查版本：

```bash
git rev-parse HEAD
```

本文档对应：

```text
8fa1ac8c66ba1da6bda6045d62e062a73715dc85
```

---

# 3. 手机端编译

环境为 Termux ARM64，手机 GPU 为 Qualcomm Adreno。

已经验证可用的配置为：

```bash
cmake -S . -B build-opencl -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DOpenCL_LIBRARY=$PREFIX/lib/libOpenCL.so \
  -DGGML_RPC=ON \
  -DGGML_OPENMP=OFF
```

然后编译 RPC server：

```bash
cmake --build build-opencl --target ggml-rpc-server -j4
```

生成：

```text
build-opencl/bin/ggml-rpc-server
```

如果之前已经完成 CMake configure，只修改了代码，则直接：

```bash
git pull origin planner-moe-stage

cmake --build build-opencl --target ggml-rpc-server -j4
```

即可。

---

# 4. 手机端启动 RPC Server

模型例如放在：

```text
~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

启动：

```bash
LD_LIBRARY_PATH=/vendor/lib64 \
./build-opencl/bin/ggml-rpc-server \
  --device GPUOpenCL \
  -H 0.0.0.0 \
  -p 50052 \
  -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

其中：

- `--device GPUOpenCL`：使用手机 Adreno OpenCL backend
- `-H 0.0.0.0`：监听所有网络接口
- `-p 50052`：RPC 端口
- `-m ...gguf`：手机本地模型

PC 和手机需要位于互相可访问的网络中。


---

# 5. PC 端编译

PC 使用 Windows + NVIDIA CUDA。

首次配置：

```powershell
cmake -S . -B build-cuda `
  -DGGML_CUDA=ON `
  -DGGML_RPC=ON
```

编译 Release：

```powershell
cmake --build build-cuda --config Release -j 8
```

主要程序位于：

```text
build-cuda\bin\Release\llama-cli.exe
```

修改代码以后一般只需要重新运行：

```powershell
cmake --build build-cuda --config Release -j 8
```

---

# 6. 当前 Hybrid / Phone-primary 环境变量

这些变量在 **PC 端 PowerShell** 设置。

## 6.1 默认：让 Hybrid Planner 全自动搜索

正常跑 `--hybrid-auto` 时，建议先清掉旧的固定 plan override：

```powershell
Remove-Item Env:LLAMA_HYBRID_FIXED_TENSOR_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_FIXED_PC_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_FIXED_PHONE_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_FIXED_TENSOR_PC_RATIO -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_FIXED_TENSOR_CHUNK_TOKENS -ErrorAction SilentlyContinue
```

这样 planner 会自己搜索：

```text
T  = Tensor layers
P  = Phone-only layers
C  = PC layers
G  = PC GPU layers
R  = Tensor PC ratio
XT = Tensor chunk tokens
XG / XC / XP = 各阶段 chunk / macro 参数
```

其中当前代码没有 `LLAMA_HYBRID_FIXED_GPU_LAYERS`、`FIXED_XG/XC/XP` 这类环境变量；`G/XG/XC/XP` 由 planner / score model 自动选择。

如果也希望 planner 自己决定 Tensor primary，不要设置：

```powershell
Remove-Item Env:LLAMA_HYBRID_TENSOR_PRIMARY -ErrorAction SilentlyContinue
```

只有在专门测试 Phone-primary 时才使用：

```powershell
$env:LLAMA_HYBRID_TENSOR_PRIMARY="phone"
```

## 6.2 当前 Phone-primary prefill chunk pipeline

Phone-primary 路径仍需要：

```powershell
$env:LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER="1"
$env:LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE="1"

$env:GGML_META_PHONE_PREFILL_CHUNK_PIPELINE="1"
$env:GGML_META_PHONE_PREFILL_PRODUCER_ROUTE="1"
$env:GGML_META_PHONE_PREFILL_ASYNC_RETURN="1"
$env:GGML_META_PHONE_PREFILL_ORDERED_RETURN="1"
$env:GGML_META_PHONE_PREFILL_CHUNK_JOIN="1"

Remove-Item Env:GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN -ErrorAction SilentlyContinue

Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FENCE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_ROUTE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_REDUCE -ErrorAction SilentlyContinue
$env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FFN_HANDOFF="1"
```

含义：

### `LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER=1`

Tensor 区域完整 activation / `l_out` 由 Phone 持有。

### `LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE=1`

FFN Tensor Parallel 使用单向归约：

```text
Phone route / local FFN
        |
        +--------------------+
        |                    |
        v                    v
   Phone FFN             PC FFN
                             |
                             v
                       PC partial
                             |
                             v
                          Phone
                             |
                             v
                            ADD
                             |
                             v
                    Phone owns full l_out
```

PC 不再要求拥有内部 Tensor 层的完整 `l_out`。

### `GGML_META_PHONE_PREFILL_CHUNK_PIPELINE=1`

启用 Phone-primary prefill chunk pipeline。

### `GGML_META_PHONE_PREFILL_PRODUCER_ROUTE=1`

Phone 作为 Router / Top-K 结果 producer。每个 chunk 的 route snapshot 通过独立 route lane 发送给 PC；当前 RPC route lane 数为 4。

### `GGML_META_PHONE_PREFILL_ASYNC_RETURN=1`

启用 PC partial -> Phone 的异步 return path。

### `GGML_META_PHONE_PREFILL_ORDERED_RETURN=1`

让 return 使用独立、有序的 RPC return 通道，避免把 return 和普通 compute socket 的控制流混在一起。

### `GGML_META_PHONE_PREFILL_CHUNK_JOIN=1`

启用当前 per-chunk join：

```text
Phone FFN_i ---------+
                     +--> ADD_i
PC FFN_i -> return --+
```

chunk 之间允许 overlap；但同一层所有 chunk 必须完成 join 后，才能进入下一层。

### `GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN`

**当前标准路径不要开启。** 之前 README 中要求设置它已经过时。

### `LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FFN_HANDOFF=1`

当前仍保留 FFN handoff 的 correctness fence。其它 `STRICT_FENCE / STRICT_ROUTE / STRICT_REDUCE` 默认关闭，避免重新引入不必要的全局同步。

RPC protocol 当前为 **v4.0.12**。如果 PC / Phone 一侧还停留在旧 RPC build，应两边更新并重新编译；尤其是 4-lane route snapshot 需要 patch 12。

---

# 7. PC 正常运行命令

例如手机地址为：

```text
192.168.1.100:50052
```

运行：

```powershell
.\build-cuda\bin\Release\llama-cli.exe `
  -m E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
  --rpc 192.168.1.100:50052 `
  --hybrid-auto `
  -c 512 `
  -n 60 `
  -t 8 `
  -lv 1
```

其中：

- `--rpc`：连接手机 RPC server
- `--hybrid-auto`：自动进行 CPU / GPU / Phone profile，并搜索 Hybrid plan
- `-c 512`：context size
- `-n 60`：生成 token 数
- `-t 8`：PC CPU thread 数
- `-lv 1`：减少普通调试输出，适合性能测试

启动后会先进行 Hybrid profile，然后输出选择的：

```text
T = Tensor layers
P = Phone-only layers
C = CPU layers
G = GPU layers
R = Tensor PC ratio

XG = GPU chunk
XC = CPU chunk
XT = Tensor chunk
XP = Phone chunk
```

之后进入正常 prefill / decode。

---

# 8. 性能测试时的环境变量

如果目的是看真正性能，不要打开 profiler 和 debug。

PC 端建议执行：

```powershell
Remove-Item Env:GGML_META_PIPELINE_DEBUG -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_TENSOR_PHONE_STAGE_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_TENSOR_EXPERT_LOAD_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_TP_ATTN_TRACE -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_TP_FFN_NUMERIC_TRACE -ErrorAction SilentlyContinue
```

但是当前 Phone-primary 功能变量要保留：

```powershell
$env:LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER="1"
$env:LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE="1"
$env:GGML_META_PHONE_PREFILL_CHUNK_PIPELINE="1"
$env:GGML_META_PHONE_PREFILL_PRODUCER_ROUTE="1"
$env:GGML_META_PHONE_PREFILL_ASYNC_RETURN="1"
$env:GGML_META_PHONE_PREFILL_ORDERED_RETURN="1"
$env:GGML_META_PHONE_PREFILL_CHUNK_JOIN="1"

Remove-Item Env:GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FENCE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_ROUTE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_REDUCE -ErrorAction SilentlyContinue
$env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FFN_HANDOFF="1"
```

如果是在测完整 autoplan，不要固定 `LLAMA_HYBRID_TENSOR_PRIMARY`；只有专门做 Phone-primary A/B 时再设置为 `phone`。

然后使用：

```text
-lv 1
```

进行性能测试。

---

# 9. 查看 Tensor 流水各阶段耗时时

如果需要分析性能瓶颈，可以额外开启：

```powershell
$env:GGML_META_TENSOR_PHONE_STAGE_PROFILE="1"
```

如果还需要完整 pipeline 调试：

```powershell
$env:GGML_META_PIPELINE_DEBUG="1"
```

这时建议运行：

```powershell
.\build-cuda\bin\Release\llama-cli.exe `
  -m E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
  --rpc 192.168.1.100:50052 `
  --hybrid-auto `
  -c 512 `
  -n 10 `
  -t 8 `
  -lv 3
```

`GGML_META_TENSOR_PHONE_STAGE_PROFILE=1` 主要会输出类似：

```text
[TENSOR_PHONE_STAGE_SG]
[TENSOR_PHONE_STAGE_COMM]
```

可以观察：

- Router handoff
- pre-route compute
- PC / Phone FFN
- return/reduce
- layer/chunk pipeline

的实际 wall time。

---

# 10. 手机 RPC / OpenCL 调试

正常性能运行时，手机不需要额外的 debug 环境变量。

如果需要排查 RPC/OpenCL 问题，可以临时开启：

```bash
export GGML_RPC_OPENCL_EXTRA_DEBUG=1
```

然后启动 RPC server。

另外之前用于验证量化权重传输正确性的：

```bash
export GGML_RPC_VERIFY_QUANT_WEIGHTS=1
```

只建议在 correctness debug 时使用，不应在性能测试中开启。

关闭：

```bash
unset GGML_RPC_OPENCL_EXTRA_DEBUG
unset GGML_RPC_VERIFY_QUANT_WEIGHTS
```

---

# 11. 环境变量关闭方式需要特别注意

当前很多实验变量的代码判断方式是：

```cpp
std::getenv("VARIABLE") != nullptr
```

所以：

```powershell
$env:GGML_META_PIPELINE_DEBUG="0"
```

**并不能真正关闭。**

因为变量仍然存在。

正确关闭方式是：

```powershell
Remove-Item Env:GGML_META_PIPELINE_DEBUG
```

手机 Termux 则使用：

```bash
unset GGML_RPC_OPENCL_EXTRA_DEBUG
```

因此，对于 `SINGLE_OWNER`、`ONEWAY_REDUCE`、`CHUNK_PIPELINE`、`DEFER_PHONE_FFN`、`ASYNC_RETURN` 等开关，也推荐：

- 开启：设置为 `1`
- 关闭：直接删除 / unset

而不是设置成 `0`。

---

# 12. 目前不需要设置的旧变量

当前 Phone-primary Tensor pipeline **不依赖**：

```text
LLAMA_HYBRID_STAGE_QUEUE
```

`LLAMA_HYBRID_STAGE_QUEUE` 是另一套 coarse GPU/CPU/TENSOR stage queue 执行路径。

当前 phone-primary Tensor 执行在代码中是独立处理的。

同样：

```text
LLAMA_PREFILL_STAGE_PIPELINE
```

属于旧的 legacy 路径，不建议为当前实验开启。

`LLAMA_HYBRID_RETURN_WAVEFRONT` 也是另一条 return-wavefront 路径，不是当前 Phone-primary chunk pipeline 的必要变量。

因此复现当前实验时，不要为了“多开流水线”把这些旧变量全部打开，否则反而容易混淆执行路径。

---

# 13. Decode

目前 Qwen3-MoE decode 的 chunk 策略仍固定为：

```text
LLAMA_CHUNKS = 1
```

当前代码中 planner 会输出：

```text
[DECODE_CHUNK] selected=1 mode=FIXED
```

因此正常使用时无需额外设置 `LLAMA_CHUNKS`。

设置更多的经过验证后并不占优。
---

# 14. 最简启动流程

## 手机

```bash
git pull --ff-only origin planner-moe-stage

cmake --build build-opencl --target ggml-rpc-server -j4

LD_LIBRARY_PATH=/vendor/lib64 \
./build-opencl/bin/ggml-rpc-server \
  --device GPUOpenCL \
  -H 0.0.0.0 \
  -p 50052 \
  -m ~/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

## PC

```powershell
git pull --ff-only origin planner-moe-stage

cmake --build build-cuda --config Release -j 8
```

清掉旧的固定 planner override：

```powershell
Remove-Item Env:LLAMA_HYBRID_FIXED_TENSOR_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_FIXED_PC_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_FIXED_PHONE_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_FIXED_TENSOR_PC_RATIO -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_FIXED_TENSOR_CHUNK_TOKENS -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_TENSOR_PRIMARY -ErrorAction SilentlyContinue
```

开启当前 Phone-primary pipeline 能力（当 planner 选择 Phone-primary 时生效）：

```powershell
$env:LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER="1"
$env:LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE="1"

$env:GGML_META_PHONE_PREFILL_CHUNK_PIPELINE="1"
$env:GGML_META_PHONE_PREFILL_PRODUCER_ROUTE="1"
$env:GGML_META_PHONE_PREFILL_ASYNC_RETURN="1"
$env:GGML_META_PHONE_PREFILL_ORDERED_RETURN="1"
$env:GGML_META_PHONE_PREFILL_CHUNK_JOIN="1"

Remove-Item Env:GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN -ErrorAction SilentlyContinue

Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FENCE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_ROUTE -ErrorAction SilentlyContinue
Remove-Item Env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_REDUCE -ErrorAction SilentlyContinue
$env:LLAMA_HYBRID_PHONE_PRIMARY_STRICT_FFN_HANDOFF="1"
```

如果要专门复现 Phone-primary 性能，再额外设置：

```powershell
$env:LLAMA_HYBRID_TENSOR_PRIMARY="phone"
```

关闭性能无关日志：

```powershell
Remove-Item Env:GGML_META_PIPELINE_DEBUG -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_TENSOR_PHONE_STAGE_PROFILE -ErrorAction SilentlyContinue
Remove-Item Env:GGML_META_TENSOR_EXPERT_LOAD_PROFILE -ErrorAction SilentlyContinue
```

运行：

```powershell
.\build-cuda\bin\Release\llama-cli.exe `
  -m E:\llama\models\Qwen3-30B-A3B-Q4_K_M.gguf `
  --rpc 192.168.1.100:50052 `
  --hybrid-auto `
  -c 512 `
  -n 60 `
  -t 8 `
  -lv 1
```

这样就是目前这套 Phone + PC Hybrid MoE 推理代码的标准复现方式。

---

## 当前源码位置

主要相关代码：

```text
src/llama-hybrid.cpp
src/llama-context.cpp
ggml/src/ggml-backend-meta.cpp
ggml/src/ggml-rpc/ggml-rpc.cpp
ggml/src/ggml-opencl/
tools/rpc/rpc-server.cpp
```

其中：

- `src/llama-hybrid.cpp`：profile、planner、Hybrid Auto、Tensor primary 选择
- `src/llama-context.cpp`：Hybrid runtime / stage 执行
- `ggml-backend-meta.cpp`：PC–Phone Tensor Parallel、Phone-primary owner、chunk pipeline、异步 return
- `ggml-rpc.cpp`：RPC 数据传输以及 OpenCL RPC 扩展
- `ggml-opencl`：手机 Adreno OpenCL backend
- `rpc-server.cpp`：手机端 RPC server
