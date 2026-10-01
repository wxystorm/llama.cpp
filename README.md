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

# 6. 当前 phone-primary Tensor 流水需要开启的变量

这些变量在 **PC 端 PowerShell** 设置。

```powershell
$env:LLAMA_HYBRID_TENSOR_PRIMARY="phone"

$env:LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER="1"
$env:LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE="1"

$env:GGML_META_PHONE_PREFILL_CHUNK_PIPELINE="1"
$env:GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN="1"
$env:GGML_META_PHONE_PREFILL_ASYNC_RETURN="1"
```

它们分别对应：

### `LLAMA_HYBRID_TENSOR_PRIMARY=phone`

让 Hybrid Planner 只考虑 **Phone-primary Tensor** 方案。

如果不设置，MoE planner 可以同时搜索 PC-primary 和 Phone-primary。

也可以设置成pc，这样再切分的时候，PC 将作为 Tensor 层 activation 的主要持有者。

### `LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER=1`

Tensor 区域中完整的 `l_out` 由 Phone 持有。

也就是 Phone 作为 Tensor 层 activation 的 owner。

### `LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE=1`

FFN Tensor Parallel 归约改为单向：

```text
PC partial
    |
    v
Phone ADD
    |
    v
Phone owns full l_out
```

不再每一层把完整结果重新 mirror 回 PC。

### `GGML_META_PHONE_PREFILL_CHUNK_PIPELINE=1`

启用目前的 prefill chunk pipeline。

### `GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN=1`

允许 Phone FFN 延后执行，从而与 PC 侧计算 / Router handoff 等阶段形成流水重叠。

### `GGML_META_PHONE_PREFILL_ASYNC_RETURN=1`

启用当前实现的异步 Phone → PC return path。



因此 **PC 和手机两边都需要使用最新代码重新编译**，否则 PC 即使打开 `ASYNC_RETURN`，旧 RPC server 也不一定支持对应接口。

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

但是下面这些功能变量要保留：

```powershell
$env:LLAMA_HYBRID_TENSOR_PRIMARY="phone"
$env:LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER="1"
$env:LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE="1"
$env:GGML_META_PHONE_PREFILL_CHUNK_PIPELINE="1"
$env:GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN="1"
$env:GGML_META_PHONE_PREFILL_ASYNC_RETURN="1"
```

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

开启当前 Phone-primary pipeline：

```powershell
$env:LLAMA_HYBRID_TENSOR_PRIMARY="phone" （还是建议设置成pc，因为这个更快）
$env:LLAMA_HYBRID_PHONE_PRIMARY_SINGLE_OWNER="1"
$env:LLAMA_HYBRID_PHONE_PRIMARY_ONEWAY_REDUCE="1"
$env:GGML_META_PHONE_PREFILL_CHUNK_PIPELINE="1"
$env:GGML_META_PHONE_PREFILL_DEFER_PHONE_FFN="1"
$env:GGML_META_PHONE_PREFILL_ASYNC_RETURN="1"
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
