#!/usr/bin/env python3
"""Exact Qwen3 dense 14B/32B Phone-owned Prefill model/layout contract.

Read GGUF metadata, not the filename. The HTP executor consumes the same
metadata-derived tensor shapes as the original Jupiter model. No Decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jupiter_memory_partition import read_gguf_weights


@dataclass(frozen=True)
class ModelLayout:
    family: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int

    @property
    def phone_first_min(self) -> int:
        return self.layers - 8

    @property
    def phone_first_max(self) -> int:
        return self.layers - 1

    @property
    def default_cpu_layers(self) -> int:
        return 16 if self.layers == 40 else 44

    @property
    def default_pc_layers(self) -> int:
        return self.layers - 8

    def validate_split(self, pc_layers: int, cpu_layers: int) -> None:
        if not self.phone_first_min <= pc_layers <= self.phone_first_max:
            raise ValueError(
                f"Qwen3-{self.family}: first Phone layer must be "
                f"{self.phone_first_min}..{self.phone_first_max}, got {pc_layers}")
        if not 0 <= cpu_layers < pc_layers:
            raise ValueError(
                f"Qwen3-{self.family}: require 0 <= CPU layers < "
                f"first Phone layer ({pc_layers}), got {cpu_layers}")


def inspect_model(path: Path, family: str = "auto") -> ModelLayout:
    """Verify dense Qwen3 GGUF dimensions before any Phone experiment."""
    model = read_gguf_weights(Path(path))
    if model.arch != "qwen3":
        raise ValueError(
            f"Phone-native supports dense Qwen3, got GGUF arch={model.arch!r}")
    layers = model.n_layers
    names = {40: "14b", 64: "32b"}
    if layers not in names:
        raise ValueError(
            f"Phone-native supports Qwen3-14B (40 layers) or "
            f"Qwen3-32B (64 layers), got {layers} layers")
    if family != "auto" and family != names[layers]:
        raise ValueError(
            f"--model-size={family} disagrees with GGUF {layers}-layer "
            f"Qwen3-{names[layers]}")
    heads = model.attention_head_count
    kv_heads = model.attention_head_count_kv
    hidden = model.embedding_length
    head_dim = model.attention_key_length or model.rope_dimension_count
    if (hidden != 5120 or
            (heads, kv_heads, head_dim) !=
            ((40, 8, 128) if layers == 40 else (64, 8, 128))):
        raise ValueError(
            f"Unsupported Qwen3 GGUF dimensions: layers={layers}, "
            f"hidden={hidden}, Q={heads}, KV={kv_heads}, head_dim={head_dim}")
    return ModelLayout(names[layers], layers, hidden, heads,
                       kv_heads, head_dim)
