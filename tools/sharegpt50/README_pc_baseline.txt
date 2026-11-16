单机原生 llama.cpp ShareGPT50 Prefill 基线 —— 修复版（2026-11-16）
===========================================================

1. 将 bench_single_pc_sharegpt50.py 覆盖到 E:\llama\bench_single_pc_sharegpt50.py
   原脚本报 UnboundLocalError: counter += 1；本版已经移除无定义的 counter。

2. 默认使用官方构建（不是 tensor-v2）：
   E:\llama\llama-official\build\bin\Release
   - 自动首选同目录 llama-server.exe（一个模型只加载一次）；
   - 如没有 llama-server.exe，则使用 llama-cli.exe / llama-completion.exe。
   - 若一定要使用你给出的 llama-cli.exe，加 --mode cli。

3. 本版默认参数：
   数据 E:\llama\datasets\sharegpt\sharegpt50_short
   五组 0/128/256/512/1024；每组前5条，共25条/模型、75条总计
   14B=Qwen3-14B-Q4_K_M.gguf
   24B=Mistral-Small-3.2-24B-Q6_K.gguf
   32B=Qwen3-32B-Q4_K_M.gguf
   -ngl all（所有模型）；-b 2048；-ub 512；-c 3000；-t 8；-tb 8
   CLI 模式另有 -lv 3、-n 1；不会额外附加 -st；Server 模式预热不计入成绩。
   注意保留 -b 2048，是你要求的论文主基线；示例命令 -b 256 不是这次的配置。

4. PowerShell（从 E:\llama 执行）：
   cd E:\llama
   python .\bench_single_pc_sharegpt50.py --per-group 5

   直接使用官方 llama-cli.exe（加载权重25次/模型，较慢）：
   python .\bench_single_pc_sharegpt50.py --per-group 5 --mode cli

   仅运行32B：
   python .\bench_single_pc_sharegpt50.py --models 32B --per-group 5

   补齐每组10条/全部50条（相同配置下自动续跑剩余条数）：
   python .\bench_single_pc_sharegpt50.py --per-group 10

   CUDA 显存不足时，手工指定每个模型层数（不会静默改变默认 all）：
   python .\bench_single_pc_sharegpt50.py --ngl14b 12 --ngl24b 8 --ngl32b 10 --per-group 5
   这属于另一套 offload 配置，不应误标为 -ngl all 实验。

5. 输出与续跑：
   E:\llama\results\llama_pc_sharegpt50_official_ngl_all\results.csv
   E:\llama\results\llama_pc_sharegpt50_official_ngl_all\summary_by_group.csv
   E:\llama\results\llama_pc_sharegpt50_official_ngl_all\summary_by_model.csv
   E:\llama\results\llama_pc_sharegpt50_official_ngl_all\logs\
   每条结束立即追加结果，按配置+模型+prompt内容+重复序号计算key并续跑。
   summary只计入 PASS；包含算术均值和按token/time加权速度。
   原先 tensor-v2 目录的结果不会被合并进本次 official baseline。

6. 显存提示：
   RTX 4050 6GiB VRAM 无法完整容纳这些大模型的权重。
   -ngl all 会尝试将所有模型层放到 GPU，实际可能 CUDA OOM 或 Windows 共享显存退化。
   本脚本严格照你的要求传 -ngl all，不自动降层数。失败时看 logs 下对应模型的启动日志，
   如果需要 CPU+GPU 的可运行原生单机基线，再使用单独的 --ngl14b/24b/32b 参数重跑。

7. 校验范围：已在模拟 llama-server 下完整检查75例 (3模型*5条*5组)；
   也已检查 CLI 参数与断点续跑；不是实际 RTX 4050/CUDA 性能结果。

8. 在官方仓库目录执行（和你的原始 llama-cli 命令路径一致）：
   cd E:\llama\llama-official
   python E:\llama\bench_single_pc_sharegpt50.py --mode cli --bin-dir "E:\llama\llama-official\build\bin\Release" --ngl all --per-group 5 --batch 2048 --ubatch 512 --ctx 3000 --threads 8 --threads-batch 8

   如同时编译了 llama-server.exe，推荐改用 --mode server 以避免每条 prompt 都重新加载大型模型；
   两种运行模式的统计需要分别报告，不要把两组数值混合在一起。