#!/usr/bin/env python3
"""Compare CPU and native Phone-owned Qwen3 hidden-state checkpoints.

This is correctness diagnostics, not a timing/benchmark tool.
PC l_out-31..39 and result_norm/result_output checkpoints are
captured by the GGML eval callback. Phone layers 32..39 are captured
after actual native HTP layer execution.
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
    max_index = max(
        range(n), key=lambda i: abs(float(a[i])-float(b[i])))
    max_abs = abs(float(a[max_index])-float(b[max_index]))
    return {
        "elements": n,
        "max_abs_index": max_index,
        "max_abs_ref": float(a[max_index]),
        "max_abs_test": float(b[max_index]),
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
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--out", type=Path, default=Path("phone-stage-parity.json"))
    args = parser.parse_args()
    if sys.byteorder != "little":
        raise RuntimeError("requires little-endian F32 files")
    chunks = [int(v) for v in args.chunks.split(",")]
    if (not chunks or any(n <= 0 for n in chunks) or
            args.embd <= 0 or args.vocab <= 0):
        parser.error("chunks and embd must be positive")
    n_elements = sum(chunks)*args.embd

    comparisons = []
    cpu31 = load_f32(args.cpu_dir / "l_out-31.f32", n_elements)
    native31 = load_f32(args.native_dir / "l_out-31.f32", n_elements)
    comparisons.append(("L31 PC input parity", cpu31, native31, args.embd))

    # Standard Qwen3 gathers only the final selected logits row during
    # Transformer layer 39, whereas the native Phone computes every token.
    # Compare full 175-token tensors at layers 32..38, then the selected
    # final token at layer 39.
    for layer in range(32, 39):
        cpu_layer = load_f32(args.cpu_dir / f"l_out-{layer}.f32", n_elements)
        phone_layer = get_phone_stage(
            args.phone_dir, layer, chunks, args.embd)
        comparisons.append((
            f"L{layer} CPU vs HTP", cpu_layer, phone_layer, args.embd
        ))
    cpu39_last = load_f32(args.cpu_dir / "l_out-39.f32", args.embd)
    phone39 = get_phone_stage(args.phone_dir, 39, chunks, args.embd)
    comparisons.append((
        "L39 final-token Phone suffix", cpu39_last,
        phone39[-args.embd:], args.embd
    ))
    native39 = load_f32(
        args.native_dir / "phone_owned_hidden.f32", n_elements)
    comparisons.append((
        "L39 phone-to-PC transfer", phone39, native39, args.embd
    ))

    # These two post-Transformer checks identify whether final logits
    # diverge at RMSNorm or at the LM Head matrix multiplication.
    cpu_norm = load_f32(args.cpu_dir / "result_norm.f32", args.embd)
    native_norm = load_f32(args.native_dir / "result_norm.f32", args.embd)
    comparisons.append((
        "Final RMSNorm CPU vs Native", cpu_norm, native_norm, args.embd
    ))
    cpu_output = load_f32(args.cpu_dir / "result_output.f32", args.vocab)
    native_output = load_f32(
        args.native_dir / "result_output.f32", args.vocab)
    comparisons.append((
        "Raw LM Head logits CPU vs Native",
        cpu_output, native_output, args.vocab
    ))
    # In the native runner, the host logits dump and raw GGML output are
    # generated during the SAME request. Comparing these reveals an
    # output-transfer or synchronization issue independent of model parity.
    host_native_path = args.native_dir.parent / "native-logits.f32"
    if host_native_path.is_file():
        native_host = load_f32(host_native_path, args.vocab)
        comparisons.append((
            "Native raw LM Head vs native host logits",
            native_output, native_host, args.vocab
        ))
    else:
        print(f"[PHONE_PARITY] INFO: host logits not found at "
              f"{host_native_path}; skipping same-run raw-to-host check")
    result = {
        "chunks": chunks,
        "n_embd": args.embd,
        "measurements": {},
    }
    for title, reference, candidate, width in comparisons:
        summary = compare(reference, candidate)
        parts = []
        if len(reference) == n_elements and width == args.embd:
            offset = 0
            for index, tokens in enumerate(chunks):
                count = tokens * args.embd
                part = compare(
                    reference[offset:offset+count],
                    candidate[offset:offset+count])
                parts.append({"chunk": index, "tokens": tokens, **part})
                offset += count
        else:
            # The last Transformer layer / final head have only one
            # selected output token, not all 175 Prompt positions.
            parts = [{"chunk": len(chunks)-1, "tokens": 1, **summary}]
        summary["per_chunk"] = parts
        if title == "Native raw LM Head vs native host logits":
            summary["raw_host_exact_match"] = reference == candidate
            summary["raw_top1"] = max(
                range(args.vocab), key=lambda i: reference[i])
            summary["host_top1"] = max(
                range(args.vocab), key=lambda i: candidate[i])
        if len(reference) == n_elements and width == args.embd:
            summary["last_token"] = compare(
                reference[-args.embd:], candidate[-args.embd:])
            if summary.get("max_abs_index") is not None:
                i = summary["max_abs_index"]
                summary["max_abs_position"] = i // args.embd
                summary["max_abs_feature"] = i % args.embd
        result["measurements"][title] = summary
        print(f"[PHONE_PARITY] {title}")
        print(f"  cosine={summary.get('cosine')} "
              f"rRMSE={summary.get('relative_rmse')} "
              f"RMSE={summary.get('rmse')} maxabs={summary.get('max_abs')}")
        if "raw_host_exact_match" in summary:
            print(f"  raw_host_exact_match="
                  f"{summary['raw_host_exact_match']} "
                  f"raw_top1={summary['raw_top1']} "
                  f"host_top1={summary['host_top1']}")
        if "last_token" in summary:
            last = summary["last_token"]
            print(f"  final_input_token cosine={last.get('cosine')} "
                  f"rRMSE={last.get('relative_rmse')}")
            print(f"  worst_error_position={summary.get('max_abs_position')} "
                  f"feature={summary.get('max_abs_feature')}")
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
