#!/usr/bin/env python3
"""Print FNV-1a fingerprints matching the RPC attention input diagnostic.

Reads raw tensor snapshots emitted by LLAMA_TENSOR_COMPARE_DIR.  No external
Python packages required.  Match CPU hashes with [RPC_ATTN14_INPUT_RECEIVE]
and [RPC_ATTN14_INPUT_DEVICE] from the phone RPC server.
"""

import argparse
import csv
from pathlib import Path


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory", type=Path, help="CPU tensor dump directory")
    ap.add_argument("--name", default="attn_norm-14", help="tensor name")
    ap.add_argument("--tokens", type=int, default=2, help="ne1 token count")
    args = ap.parse_args()

    manifest = args.directory / "manifest.tsv"
    count = 0
    with manifest.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["name"] != args.name or int(row["ne1"]) != args.tokens:
                continue
            filename = args.directory / row["file"]
            data = filename.read_bytes()
            expected = int(row["bytes"])
            if len(data) != expected:
                raise ValueError(f"{filename}: expected {expected} bytes, got {len(data)}")
            print(f"id={row['id']} name={row['name']} ne1={row['ne1']} "
                  f"bytes={len(data)} fnv64={fnv1a64(data):016x}")
            count += 1
    print(f"matched={count}")


if __name__ == "__main__":
    main()
