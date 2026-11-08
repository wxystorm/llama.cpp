#!/usr/bin/env python3
"""End-to-end native CPU first-N prefix Graph smoke with a tiny Qwen3 GGUF.

This builds genuine FP32 Qwen3 layers and executes the exact GGML
Attention/FFN/KV graph as production samples. No model download,
RPC, Phone or CUDA device is required.
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
from gguf import GGUFWriter  # noqa: E402


def create_model(path: Path, layers: int = 2, *, explicit_rope_dim: bool = False) -> None:
    writer = GGUFWriter(path, "qwen3")
    writer.add_block_count(layers)
    writer.add_context_length(128)
    writer.add_embedding_length(128)
    writer.add_head_count(1)
    writer.add_head_count_kv(1)
    writer.add_feed_forward_length(256)
    # This field is optional in real Qwen3 GGUFs. Test both forms.
    if explicit_rope_dim:
        writer.add_rope_dimension_count(128)
    writer.add_rope_freq_base(1000000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    rng = np.random.default_rng(42)
    def mat(shape):
        return (rng.standard_normal(shape).astype(np.float32) * 0.015)
    for l in range(layers):
        p = f"blk.{l}."
        for name in ("attn_norm", "ffn_norm"):
            writer.add_tensor(p + name + ".weight",
                              np.ones(128, dtype=np.float32))
        for name in ("attn_q_norm", "attn_k_norm"):
            writer.add_tensor(p + name + ".weight",
                              np.ones(128, dtype=np.float32))
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            writer.add_tensor(p + name + ".weight", mat((128, 128)))
        writer.add_tensor(p + "ffn_gate.weight", mat((256, 128)))
        writer.add_tensor(p + "ffn_up.weight", mat((256, 128)))
        writer.add_tensor(p + "ffn_down.weight", mat((128, 256)))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="v2_native_prefix_") as td:
        for explicit_rope_dim in (False, True):
            model = Path(td) / (
                "tiny-explicit.gguf" if explicit_rope_dim else "tiny-implicit.gguf")
            create_model(model, explicit_rope_dim=explicit_rope_dim)
            for n in (1, 2):
                command = [str(args.exe.resolve()), "--model", str(model),
                           "--backend", "C", "--layers", str(n),
                           "--chunk", "16", "--kv", "128",
                           "--threads", "2", "--warmups", "0",
                           "--repeats", "2", "--verify-cpu"]
                result = subprocess.run(command, check=False,
                                        capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"CPU_L{n}_ROPE_DIM_{explicit_rope_dim}_FAIL:"
                        f"{result.stderr[-4000:]}")
                rows = list(csv.DictReader(io.StringIO(result.stdout)))
                if len(rows) != 1 or rows[0]["backend"] != "C" or \
                        int(rows[0]["layers"]) != n or \
                        float(rows[0]["compute_median_ms"]) <= 0 or \
                        rows[0]["parity"] != "PASS":
                    raise RuntimeError(
                        f"CPU_L{n}_ROPE_DIM_{explicit_rope_dim}_RESULT:{rows}")
        print("[V2_NATIVE_PREFIX_TEST] status=PASS CPU_N1_N2_OPTIONAL_ROPE_DIM")


if __name__ == "__main__":
    main()
