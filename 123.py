import json
import os
import re
from pathlib import Path

from gguf import GGUFReader


# ============================================================
# 配置
# ============================================================

GGUF_PATH = r"E:\llama\models\Qwen2.5-32B-Instruct-Q4_K_M.gguf"

OUTPUT_BIN = r"E:\llama\models\qwen32b_ffn_stream.bin"
OUTPUT_INDEX = r"E:\llama\models\qwen32b_ffn_stream_index.json"

# 0~23 常驻，所以从 24 开始抽取
START_LAYER = 24

# Qwen2.5-32B 是 0~63
END_LAYER = 63

# 在 sidecar 文件中的排列顺序
FFN_TYPES = [
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
]

# 每次实际从 GGUF 复制多少数据
# 64 MiB，避免一次 bytes 对象过大
COPY_CHUNK_SIZE = 64 * 1024 * 1024


# ============================================================
# 工具函数
# ============================================================

def parse_ffn_tensor_name(name):
    """
    解析：
        blk.24.ffn_gate.weight
        blk.24.ffn_up.weight
        blk.24.ffn_down.weight

    返回：
        (layer, kind)

    例如：
        (24, "ffn_gate.weight")

    不是目标 tensor 则返回 None。
    """
    m = re.fullmatch(
        r"blk\.(\d+)\.(ffn_gate\.weight|ffn_up\.weight|ffn_down\.weight)",
        name
    )

    if m is None:
        return None

    layer = int(m.group(1))
    kind = m.group(2)

    return layer, kind


def copy_region(src, dst, src_offset, size):
    """
    从 src 文件的 src_offset 开始复制 size 字节到 dst。
    分块复制，避免一次性申请几百 MB Python bytes。
    """
    src.seek(src_offset)

    remain = size

    while remain > 0:
        n = min(remain, COPY_CHUNK_SIZE)

        data = src.read(n)

        if len(data) != n:
            raise RuntimeError(
                f"short read: expected {n} bytes, got {len(data)}"
            )

        dst.write(data)

        remain -= n


# ============================================================
# 主程序
# ============================================================

def main():

    gguf_path = Path(GGUF_PATH)

    if not gguf_path.exists():
        raise FileNotFoundError(GGUF_PATH)

    print("Reading GGUF metadata...")
    reader = GGUFReader(str(gguf_path))

    # --------------------------------------------------------
    # 1. 建立 tensor 名称 -> metadata 的映射
    # --------------------------------------------------------

    tensor_map = {}

    for tensor in reader.tensors:
        parsed = parse_ffn_tensor_name(tensor.name)

        if parsed is None:
            continue

        layer, kind = parsed

        if START_LAYER <= layer <= END_LAYER:
            tensor_map[(layer, kind)] = tensor

    # --------------------------------------------------------
    # 2. 检查需要的 tensor 是否全部存在
    # --------------------------------------------------------

    expected_count = (
        (END_LAYER - START_LAYER + 1)
        * len(FFN_TYPES)
    )

    print()
    print(f"Expected tensors : {expected_count}")
    print(f"Found tensors    : {len(tensor_map)}")

    missing = []

    for layer in range(START_LAYER, END_LAYER + 1):
        for kind in FFN_TYPES:

            if (layer, kind) not in tensor_map:
                missing.append(f"blk.{layer}.{kind}")

    if missing:
        print("\nMissing tensors:")

        for name in missing:
            print("  ", name)

        raise RuntimeError(
            f"{len(missing)} required FFN tensors are missing"
        )

    # --------------------------------------------------------
    # 3. 按实际执行顺序写 sidecar
    #
    # L24 gate
    # L24 up
    # L24 down
    # L25 gate
    # L25 up
    # L25 down
    # ...
    # --------------------------------------------------------

    index = {
        "source_gguf": str(gguf_path),
        "start_layer": START_LAYER,
        "end_layer": END_LAYER,
        "tensor_order": FFN_TYPES,
        "layers": {},
    }

    output_offset = 0

    with open(GGUF_PATH, "rb", buffering=0) as src, \
         open(OUTPUT_BIN, "wb", buffering=0) as dst:

        for layer in range(START_LAYER, END_LAYER + 1):

            print()
            print(f"========== layer {layer} ==========")

            layer_begin = output_offset

            layer_info = {
                "offset": layer_begin,
                "tensors": {}
            }

            for kind in FFN_TYPES:

                tensor = tensor_map[(layer, kind)]

                name = tensor.name
                src_offset = int(tensor.data_offset)
                size = int(tensor.n_bytes)

                dst_offset = output_offset

                print(
                    f"{name:<30} "
                    f"size={size / 1024 / 1024:8.3f} MiB  "
                    f"src={src_offset:12d}  "
                    f"dst={dst_offset:12d}"
                )

                # 从 GGUF 原始位置复制量化后的 bytes
                copy_region(
                    src,
                    dst,
                    src_offset,
                    size
                )

                layer_info["tensors"][kind] = {
                    "name": name,
                    "offset": dst_offset,
                    "relative_offset": dst_offset - layer_begin,
                    "size": size,
                    "source_offset": src_offset,
                    "ggml_type": str(tensor.tensor_type),
                    "shape": [
                        int(x) for x in tensor.shape
                    ],
                }

                output_offset += size

            # 整个 layer 的 gate + up + down 是严格连续的
            layer_info["size"] = output_offset - layer_begin

            index["layers"][str(layer)] = layer_info

            print(
                f"Layer {layer}: "
                f"{layer_info['size'] / 1024 / 1024:.3f} MiB"
            )

    # --------------------------------------------------------
    # 4. 保存 index
    # --------------------------------------------------------

    index["total_size"] = output_offset
    index["total_size_mib"] = output_offset / 1024 / 1024
    index["tensor_count"] = expected_count

    with open(
        OUTPUT_INDEX,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            index,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # 5. 最终信息
    # --------------------------------------------------------

    print()
    print("========================================")
    print("Done")
    print("========================================")

    print(f"Output bin   : {OUTPUT_BIN}")
    print(f"Output index : {OUTPUT_INDEX}")

    print(
        f"Total size   : "
        f"{output_offset / 1024 / 1024:.3f} MiB"
    )

    print(
        f"Tensor count : {expected_count}"
    )


if __name__ == "__main__":
    main()