# Qwen3-32B ShareGPT50 — Fixed-64 平均 Prefill 结果

- 消融：w/o Adaptive Chunk（固定 `XC=XG=XP=64`）
- Placement：`G7,P4,G7,P4,G8,P5,G7,P4,G7,P4,G7`
- Profile：独立 G/C/N Profile；常驻 llama-server；双 GPU Bank、异步预取、Drain-First
- 运行：ShareGPT50_short，50/50 请求 PASS；汇总样本数见下表
- 运行指纹：`127bf5a70ce5`
- 统计口径：平均吞吐为逐请求 `prefill_tps` 的算术平均；加权吞吐为 `sum(prompt_n) / sum(prefill_ms) × 1000`

| Token 组 | 汇总样本数 | 平均 Prefill (tok/s) | 加权 Prefill (tok/s) |
|---|---:|---:|---:|
| 0–128 | 9 | 50.165 | 49.815 |
| 128–256 | 10 | 89.831 | 90.114 |
| 256–512 | 10 | 103.011 | 103.367 |
| 512–1024 | 10 | 129.056 | 130.001 |
| 1024–2048 | 10 | 129.907 | 129.888 |
| **全部** | **49** | **101.419** | **116.688** |
