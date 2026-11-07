# PRIMA.CPP — pinned upstream import + Qwen3 Dense port

**Upstream:** https://github.com/OpenCPIL/prima.cpp

**Upstream commit:** `288df6a2cc2120387e12d6f40798f30cc525c9ea`

**Original license:** MIT — when imported, retain upstream `LICENSE` and original authorship notices. This independently managed copy is not a claim of original authorship.

## Work branch

`baseline/prima-qwen3-dense-20261107`

The full upstream tree is imported under this folder by the bootstrap workflow, or locally:

```bash
bash baselines/_prima_bootstrap/import_and_port.sh --push
```

For safety the import checks out the exact upstream commit and refuses to modify other branches.

## Scope of the port

- Qwen3 Dense only (0.6B, 1.7B, 4B, 8B, 14B, 32B); **not** MoE/VL.
- Extends the distributed Qwen2 loader and graph with Q/K head-wise RMSNorm before RoPE.
- Handles Qwen3 bias-free Q/K/V and Q/K Norm tensors.
- Adapts HALDA profiling loader; preserves existing communication and scheduling.

## Validation status

**Unverified.** No model compilation, rank-0/rank-1 run, or logits parity check has passed yet. These tests are mandatory before using it for measurements.
