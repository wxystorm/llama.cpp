#!/usr/bin/env python3
"""Compare CPU and native Phone-owned Qwen3 hidden-state checkpoints.

This is correctness diagnostics, not a timing/benchmark tool.
PC l_out-31/32/39 checkpoints are written by the GGML eval callback.
Phone l_out-32/39 checkpoints are written after native HTP layer execution.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import sys
from pathlib import Path


def load_f32(path: Path, expected: int) -> array.array:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    if len(raw) != expected * 4:
        raise ValueError(
            f"{path}: got {len(raw)} bytes; expected {expected * 4} "
            f"(expected {expected} F32 values)"
        )
    values = array.array("f")
    values.frombytes(raw)
    if len(values) != expected:
        raise ValueError(f"{path}: wrong F32 element count")
    return values


def get_phone_stage(directory: Path, layer: int, chunks: list[int],
                    embd: int) -> array.array:
    result = array.array("f")
    for index, n_tokens in enumerate(chunks):
        filename = f"l_out-{layer}-chunk{index}.f32"
        result.extend(load_f32(directory / filename, n_tokens * embd))
    return result


def compare(a: array.array, b: array.array) -> dict:
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("empty/mismatched vectors")
    bad_a = sum(not math.isfinite(x) for x in a)
    bad_b = sum(not math.isfinite(x) for x in b)
    if bad_a or bad_b:
        return {"elements": len(a), "nonfinite_ref": bad_a,
                "nonfinite_test": bad_b, "valid": False}
    n = len(a)
    sq_diff = math.fsum((float(x)-float(y))**2 for x, y in zip(a, b))
    sq_ref = math.fsum(float(x)**2 for x in a)
    sq_test = math.fsum(float(y)**2 for y in b)
    dot = math.fsum(float(x)*float(y) for x, y in zip(a, b))
    max_abs = max(abs(float(x)-float(y)) for x, y in zip(a, b))
    return {
        "elements": n,
        "valid": True,
        "rms_ref": math.sqrt(sq_ref/n),
        "rms_test": math.sqrt(sq_test/n),
        "rmse": math.sqrt(sq_diff/n),
        "relative_rmse": math.sqrt(sq_diff/sq_ref) if sq_ref else None,
        "cosine": dot/math.sqrt(sq_ref*sq_test) if sq_ref and sq_test else None,
        "max_abs": max_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-dir", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--phone-dir", type=Path, required=True)
    parser.add_argument("--chunks", default="88,87")
    parser.add_argument("--embd", type=int, default=5120)
    parser.add_argument("--out", type=Path, default=Path("phone-stage-parity.json"))
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("requires little-endian F32 files")
    chunks = [int(v) for v in args.chunks.split(",")]
    if not chunks or any(n <= 0 for n in chunks) or args.embd <= 0:
        parser.error("chunks and embd must be positive")
    n_elements = sum(chunks)*args.embd

    cpu31 = load_f32(args.cpu_dir / "l_out-31.f32", n_elements)
    cpu32 = load_f32(args.cpu_dir / "l_out-32.f32", n_elements)
    # Standard llama.cpp Qwen3 calls get_rows() inside layer 39 for
    # only the selected output token. All earlier layer outputs and
    # Phone-owned native outputs still contain the whole Prompt.
    cpu39_last = load_f32(args.cpu_dir / "l_out-39.f32", args.embd)
    native31 = load_f32(args.native_dir / "l_out-31.f32", n_elements)
    native39 = load_f32(args.native_dir / "phone_owned_hidden.f32", n_elements)
    phone32 = get_phone_stage(args.phone_dir, 32, chunks, args.embd)
    phone39 = get_phone_stage(args.phone_dir, 39, chunks, args.embd)

    comparisons = [
        ("L31 PC input parity", cpu31, native31, True),
        ("L32 first Phone layer", cpu32, phone32, True),
        ("L39 final-token Phone suffix",
         cpu39_last, phone39[-args.embd:], False),
        ("L39 phone-to-PC transfer", phone39, native39, True),
    ]
    result = {
        "chunks": chunks,
        "n_embd": args.embd,
        "measurements": {},
    }
    for title, reference, candidate, full_prompt in comparisons:
        summary = compare(reference, candidate)
        parts = []
        if full_prompt:
            offset = 0
            for index, tokens in enumerate(chunks):
                count = tokens * args.embd
                part = compare(
                    reference[offset:offset+count],
                    candidate[offset:offset+count])
                parts.append({"chunk": index, "tokens": tokens, **part})
                offset += count
        else:
            # L39 CPU reference contains only the final selected logits row.
            parts = [{"chunk": len(chunks)-1, "tokens": 1, **summary}]
        summary["per_chunk"] = parts
        result["measurements"][title] = summary
        print(f"[PHONE_PARITY] {title}")
        print(f"  cosine={summary.get('cosine')} "
              f"rRMSE={summary.get('relative_rmse')} "
              f"RMSE={summary.get('rmse')} maxabs={summary.get('max_abs')}")
        for part in parts:
            print(f"  chunk={part['chunk']} cosine={part.get('cosine')} "
                  f"rRMSE={part.get('relative_rmse')}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                        encoding="utf-8")
    print(f"[PHONE_PARITY] report={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as err:
        print(f"[PHONE_PARITY] ERROR: {err}", file=sys.stderr)
        raise SystemExit(1)
