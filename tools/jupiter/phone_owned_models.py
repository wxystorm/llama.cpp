#!/usr/bin/env python3
"""Exact Qwen3 14B/32B and dense Llama/Mistral 24B native contract.

Read GGUF metadata, not the filename. The HTP executor consumes the same
metadata-derived tensor shapes as the original Jupiter model. No Decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jupiter_memory_partition import read_gguf_weights

# ggml-rpc.cpp PREPARE request holds 64 layer-device indices. This is a
# wire-format capacity, not an experimentally chosen limit on HTP layers.
RPC_LAYER_DEVICE_SLOTS = 64


@dataclass(frozen=True)
class ModelLayout:
    family: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int

    @property
    def label(self) -> str:
        return "Mistral-Small-3.2-24B" if self.family == "24b" else f"Qwen3-{self.family.upper()}"

    @property
    def phone_first_min(self) -> int:
        # Preserve at least one PC Transformer layer and honor the RPC array.
        return max(1, self.layers - RPC_LAYER_DEVICE_SLOTS)

    @property
    def phone_first_max(self) -> int:
        return self.layers - 1

    @property
    def default_cpu_layers(self) -> int:
        return 25 if self.family == "24b" else 16 if self.layers == 40 else 44

    @property
    def default_pc_layers(self) -> int:
        return self.layers - 8

    def validate_split(self, pc_layers: int, cpu_layers: int) -> None:
        if not self.phone_first_min <= pc_layers <= self.phone_first_max:
            raise ValueError(
                f"{self.label}: first Phone layer must be "
                f"{self.phone_first_min}..{self.phone_first_max}, got {pc_layers}")
        if not 0 <= cpu_layers < pc_layers:
            raise ValueError(
                f"{self.label}: require 0 <= CPU layers < "
                f"first Phone layer ({pc_layers}), got {cpu_layers}")


def inspect_model(path: Path, family: str = "auto") -> ModelLayout:
    """Verify supported dense GGUF dimensions before any Phone experiment."""
    model = read_gguf_weights(Path(path))
    layers = model.n_layers
    heads = model.attention_head_count
    kv_heads = model.attention_head_count_kv
    hidden = model.embedding_length
    head_dim = model.attention_key_length or model.rope_dimension_count
    value_dim = model.attention_value_length or head_dim
    shape = (model.architecture, layers, hidden, heads, kv_heads, head_dim)
    supported = {
        ("qwen3", 40, 5120, 40, 8, 128): "14b",
        ("llama", 40, 5120, 32, 8, 128): "24b",
        ("qwen3", 64, 5120, 64, 8, 128): "32b",
    }
    actual = supported.get(shape)
    if actual is None or value_dim != head_dim or (
            model.rope_dimension_count not in (0, head_dim)):
        raise ValueError(f"Unsupported dense native GGUF dimensions: {shape}, "
                         f"value_dim={value_dim}, rope_dim={model.rope_dimension_count}")
    if family != "auto" and family != actual:
        raise ValueError(f"--model-size={family} disagrees with GGUF {actual}: {shape}")
    return ModelLayout(actual, layers, hidden, heads, kv_heads, head_dim)


def resolve_splits(model: ModelLayout, *, layouts: str | None = None,
                   cpu_layers: int | None = None,
                   cuda_layers: int | None = None,
                   phone_layers: int | None = None) -> tuple[list[int], int]:
    """Resolve model layers into CPU [0,c), CUDA [c,p), HTP [p,N).

    A multi-layout sweep varies p and keeps c fixed. An explicit CUDA or
    Phone count selects a single placement and cannot override a sweep.
    """
    max_phone = min(model.layers - 1, RPC_LAYER_DEVICE_SLOTS)
    if phone_layers is not None and not 1 <= phone_layers <= max_phone:
        raise ValueError(f"--phone-layers must be 1..{max_phone} for "
                         f"{model.label} (keep a PC prefix; "
                         f"RPC capacity={RPC_LAYER_DEVICE_SLOTS})")
    if cuda_layers is not None and cuda_layers < 1:
        raise ValueError("--cuda-layers must be positive")
    if layouts is not None and (phone_layers is not None or cuda_layers is not None):
        raise ValueError("Choose either --layouts or --phone-layers/"
                         "--cuda-layers, not both")
    if layouts:
        positions = list(dict.fromkeys(int(s.strip()) for s in layouts.split(",")))
    else:
        first_phone = (model.layers - phone_layers if phone_layers is not None
                       else model.default_pc_layers)
        positions = [first_phone]
    if not positions:
        raise ValueError("At least one PC/Phone boundary is required")
    if cpu_layers is None:
        if cuda_layers is not None:
            if len(positions) != 1:
                raise ValueError("CUDA layer count requires one Phone layout")
            cpu_layers = positions[0] - cuda_layers
        else:
            cpu_layers = model.default_cpu_layers
    if cuda_layers is not None and positions[0] - cpu_layers != cuda_layers:
        raise ValueError("--cuda-layers contradicts --cpu-layers"
                         " and Phone layer boundary")
    for position in positions:
        model.validate_split(position, cpu_layers)
    return positions, cpu_layers
