#!/usr/bin/env python3

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 使用当前 llama.cpp 仓库自带的 gguf-py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))

from gguf.gguf_reader import GGUFReader
from gguf.constants import GGML_QUANT_SIZES


MAGIC = b"FFNPACK1"
VERSION = 1

# header:
# magic[8]
# version
# group_size
# alignment
# n_entries
# data_offset
HEADER_FMT = "<8sIIIIQ"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

# entry:
# layer
# group
# neuron_begin
# neuron_count
# gate_type
# up_type
# down_type
# reserved
#
# bundle_offset
# bundle_size
# gate_offset
# gate_size
# up_offset
# up_size
# down_offset
# down_size
ENTRY_FMT = "<IIIIIIIIQQQQQQQQ"
ENTRY_SIZE = struct.calcsize(ENTRY_FMT)


def align_up(x: int, alignment: int) -> int:
    return (x + alignment - 1) // alignment * alignment


def raw_byte_matrix(tensor):
    """
    把 tensor.data 看成:
        [row, raw_bytes_per_row]

    不反量化，不改变 quant block 内容。
    """
    arr = np.asarray(tensor.data)

    if arr.ndim != 2:
        raise ValueError(
            f"{tensor.name}: expected 2-D tensor, "
            f"got data.shape={arr.shape}"
        )

    # quantized tensor 本身一般已经 uint8；
    # F16/F32 等也直接 view 成原始 bytes。
    return arr.view(np.uint8).reshape(arr.shape[0], -1)


@dataclass
class LayerFFN:
    layer: int
    gate: object
    up: object
    down: object


@dataclass
class PackEntry:
    layer: int
    group: int

    neuron_begin: int
    neuron_count: int

    gate_type: int
    up_type: int
    down_type: int

    bundle_offset: int = 0
    bundle_size: int = 0

    gate_offset: int = 0
    gate_size: int = 0

    up_offset: int = 0
    up_size: int = 0

    down_offset: int = 0
    down_size: int = 0

    # 仅转换过程中使用
    gate_tensor: object = None
    up_tensor: object = None
    down_tensor: object = None


def find_ffn_layers(reader: GGUFReader):
    pattern = re.compile(
        r"^blk\.(\d+)\.ffn_(gate|up|down)\.weight$"
    )

    layers = {}

    for tensor in reader.tensors:
        m = pattern.match(tensor.name)
        if not m:
            continue

        layer = int(m.group(1))
        kind = m.group(2)

        if layer not in layers:
            layers[layer] = {}

        layers[layer][kind] = tensor

    result = []

    for layer in sorted(layers):
        item = layers[layer]

        missing = {"gate", "up", "down"} - set(item.keys())
        if missing:
            raise RuntimeError(
                f"layer {layer}: missing FFN tensors: {missing}"
            )

        result.append(
            LayerFFN(
                layer=layer,
                gate=item["gate"],
                up=item["up"],
                down=item["down"],
            )
        )

    if not result:
        raise RuntimeError("No blk.N.ffn_{gate,up,down}.weight tensors found")

    return result


def check_layer_shapes(layer: LayerFFN):
    gate = layer.gate
    up = layer.up
    down = layer.down

    if len(gate.shape) != 2:
        raise RuntimeError(f"{gate.name}: shape={gate.shape}")

    hidden = int(gate.shape[0])
    ffn_dim = int(gate.shape[1])

    if list(map(int, up.shape)) != [hidden, ffn_dim]:
        raise RuntimeError(
            f"{up.name}: expected [{hidden}, {ffn_dim}], "
            f"got {up.shape}"
        )

    if list(map(int, down.shape)) != [ffn_dim, hidden]:
        raise RuntimeError(
            f"{down.name}: expected [{ffn_dim}, {hidden}], "
            f"got {down.shape}"
        )

    return hidden, ffn_dim


def make_entries(layers, group_size):
    entries = []

    for layer in layers:
        hidden, ffn_dim = check_layer_shapes(layer)

        gate_raw = raw_byte_matrix(layer.gate)
        up_raw = raw_byte_matrix(layer.up)
        down_raw = raw_byte_matrix(layer.down)

        # gate / up:
        #
        # GGML:
        #   ne[0] = hidden
        #   ne[1] = ffn_dim
        #
        # Python raw view:
        #   [ffn_dim, bytes(hidden)]
        if gate_raw.shape[0] != ffn_dim:
            raise RuntimeError(
                f"{layer.gate.name}: unexpected raw shape "
                f"{gate_raw.shape}"
            )

        if up_raw.shape[0] != ffn_dim:
            raise RuntimeError(
                f"{layer.up.name}: unexpected raw shape "
                f"{up_raw.shape}"
            )

        # down:
        #
        # GGML:
        #   ne[0] = ffn_dim
        #   ne[1] = hidden
        #
        # Python raw view:
        #   [hidden, bytes(ffn_dim)]
        if down_raw.shape[0] != hidden:
            raise RuntimeError(
                f"{layer.down.name}: unexpected raw shape "
                f"{down_raw.shape}"
            )

        down_block, down_type_size = GGML_QUANT_SIZES[
            layer.down.tensor_type
        ]

        # down 是沿 ne[0] / quantized row 内部切，
        # group 边界必须对齐 quant block。
        if group_size % down_block != 0:
            raise RuntimeError(
                f"group_size={group_size} is not aligned to "
                f"{layer.down.tensor_type.name} block={down_block}"
            )

        if ffn_dim % down_block != 0:
            raise RuntimeError(
                f"ffn_dim={ffn_dim} is not aligned to "
                f"down block={down_block}"
            )

        n_groups = (ffn_dim + group_size - 1) // group_size

        print(
            f"[layer {layer.layer}] "
            f"hidden={hidden} "
            f"ffn={ffn_dim} "
            f"groups={n_groups} "
            f"gate={layer.gate.tensor_type.name} "
            f"up={layer.up.tensor_type.name} "
            f"down={layer.down.tensor_type.name}"
        )

        for group in range(n_groups):
            begin = group * group_size
            end = min(begin + group_size, ffn_dim)
            count = end - begin

            if begin % down_block != 0 or end % down_block != 0:
                raise RuntimeError(
                    f"L{layer.layer} G{group}: "
                    f"[{begin},{end}) breaks quant block boundary"
                )

            # gate/up 是完整 neuron rows
            gate_size = count * gate_raw.shape[1]
            up_size = count * up_raw.shape[1]

            # down 是每个 output row 中的一段
            byte_begin = (
                begin // down_block
            ) * down_type_size

            byte_end = (
                end // down_block
            ) * down_type_size

            down_group_row_bytes = byte_end - byte_begin
            down_size = hidden * down_group_row_bytes

            gate_offset = 0
            up_offset = gate_offset + gate_size
            down_offset = up_offset + up_size

            bundle_size = (
                gate_size +
                up_size +
                down_size
            )

            entries.append(
                PackEntry(
                    layer=layer.layer,
                    group=group,

                    neuron_begin=begin,
                    neuron_count=count,

                    gate_type=int(layer.gate.tensor_type),
                    up_type=int(layer.up.tensor_type),
                    down_type=int(layer.down.tensor_type),

                    bundle_size=bundle_size,

                    gate_offset=gate_offset,
                    gate_size=gate_size,

                    up_offset=up_offset,
                    up_size=up_size,

                    down_offset=down_offset,
                    down_size=down_size,

                    gate_tensor=layer.gate,
                    up_tensor=layer.up,
                    down_tensor=layer.down,
                )
            )

    return entries


def assign_offsets(entries, alignment):
    index_end = HEADER_SIZE + len(entries) * ENTRY_SIZE
    data_offset = align_up(index_end, alignment)

    cur = data_offset

    for e in entries:
        cur = align_up(cur, alignment)
        e.bundle_offset = cur
        cur += e.bundle_size

    final_size = cur

    return data_offset, final_size


def write_entry(fp, e: PackEntry):
    fp.write(
        struct.pack(
            ENTRY_FMT,

            e.layer,
            e.group,

            e.neuron_begin,
            e.neuron_count,

            e.gate_type,
            e.up_type,
            e.down_type,
            0,

            e.bundle_offset,
            e.bundle_size,

            e.gate_offset,
            e.gate_size,

            e.up_offset,
            e.up_size,

            e.down_offset,
            e.down_size,
        )
    )


def write_bundle(fp, e: PackEntry):
    gate = raw_byte_matrix(e.gate_tensor)
    up = raw_byte_matrix(e.up_tensor)
    down = raw_byte_matrix(e.down_tensor)

    begin = e.neuron_begin
    end = begin + e.neuron_count

    # ---------------------------------------------------------
    # gate
    # 原 GGUF 中这部分本来就是连续 neuron rows
    # ---------------------------------------------------------
    gate_part = np.ascontiguousarray(
        gate[begin:end, :]
    )

    if gate_part.nbytes != e.gate_size:
        raise RuntimeError(
            f"L{e.layer} G{e.group}: gate size mismatch"
        )

    fp.write(gate_part.tobytes(order="C"))

    # ---------------------------------------------------------
    # up
    # ---------------------------------------------------------
    up_part = np.ascontiguousarray(
        up[begin:end, :]
    )

    if up_part.nbytes != e.up_size:
        raise RuntimeError(
            f"L{e.layer} G{e.group}: up size mismatch"
        )

    fp.write(up_part.tobytes(order="C"))

    # ---------------------------------------------------------
    # down
    #
    # 原布局:
    #
    # row0: G0 G1 G2 ...
    # row1: G0 G1 G2 ...
    #
    # 新布局把当前 group 从所有 row gather 后紧凑排列。
    # ---------------------------------------------------------
    block_size, type_size = GGML_QUANT_SIZES[
        e.down_tensor.tensor_type
    ]

    byte_begin = (
        begin // block_size
    ) * type_size

    byte_end = (
        end // block_size
    ) * type_size

    down_part = np.ascontiguousarray(
        down[:, byte_begin:byte_end]
    )

    if down_part.nbytes != e.down_size:
        raise RuntimeError(
            f"L{e.layer} G{e.group}: "
            f"down size mismatch "
            f"{down_part.nbytes} != {e.down_size}"
        )

    fp.write(down_part.tobytes(order="C"))


def convert(model_path, out_path, group_size, alignment):
    print(f"Loading GGUF metadata: {model_path}")

    reader = GGUFReader(model_path)

    layers = find_ffn_layers(reader)

    print(f"Found {len(layers)} FFN layers")

    entries = make_entries(
        layers,
        group_size,
    )

    data_offset, final_size = assign_offsets(
        entries,
        alignment,
    )

    total_payload = sum(
        e.bundle_size for e in entries
    )

    print()
    print(f"entries       : {len(entries)}")
    print(f"group size    : {group_size}")
    print(f"alignment     : {alignment}")
    print(f"payload       : {total_payload / 1024**3:.3f} GiB")
    print(f"output size   : {final_size / 1024**3:.3f} GiB")
    print()

    with open(out_path, "wb") as fp:
        # -------------------------
        # header
        # -------------------------
        fp.write(
            struct.pack(
                HEADER_FMT,
                MAGIC,
                VERSION,
                group_size,
                alignment,
                len(entries),
                data_offset,
            )
        )

        # -------------------------
        # index table
        # -------------------------
        for e in entries:
            write_entry(fp, e)

        # pad 到 data_offset
        if fp.tell() > data_offset:
            raise RuntimeError("Index larger than data_offset")

        fp.write(
            b"\0" * (data_offset - fp.tell())
        )

        # -------------------------
        # bundle data
        # -------------------------
        for i, e in enumerate(entries):
            if fp.tell() < e.bundle_offset:
                fp.write(
                    b"\0" * (
                        e.bundle_offset - fp.tell()
                    )
                )

            if fp.tell() != e.bundle_offset:
                raise RuntimeError(
                    f"bad bundle offset: "
                    f"{fp.tell()} != {e.bundle_offset}"
                )

            write_bundle(fp, e)

            expected_end = (
                e.bundle_offset +
                e.bundle_size
            )

            if fp.tell() != expected_end:
                raise RuntimeError(
                    f"L{e.layer} G{e.group}: "
                    f"bundle end mismatch"
                )

            if (
                i % 32 == 0
                or i + 1 == len(entries)
            ):
                print(
                    f"\rpacking "
                    f"{i + 1}/{len(entries)}",
                    end="",
                    flush=True,
                )

        fp.truncate(final_size)

    print()
    print(f"Done: {out_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "model",
        help="input GGUF",
    )

    ap.add_argument(
        "output",
        help="output .ffnpack",
    )

    ap.add_argument(
        "--group-size",
        type=int,
        default=512,
    )

    ap.add_argument(
        "--alignment",
        type=int,
        default=4096,
    )

    args = ap.parse_args()

    if args.group_size <= 0:
        raise ValueError("group-size must be > 0")

    if (
        args.alignment <= 0
        or args.alignment & (args.alignment - 1)
    ):
        raise ValueError(
            "alignment must be a power of two"
        )

    convert(
        args.model,
        args.output,
        args.group_size,
        args.alignment,
    )


if __name__ == "__main__":
    main()