#!/usr/bin/env python3
"""Create a *compact* Qwen3 GGUF for real G/C/N/A Stage profiling.

The output contains only the first L Transformer blocks, where L is the
number of sampled G+C+N+A layers. The embedding, output and tokenizer
tensors are retained. No remaining layers are executed by a hidden CPU
fallback. This is a timing probe, NOT an equivalent full-model inference.

Phone may keep the original full GGUF because the compact model references
the identical first L blocks, and PC sends their original absolute layer
indices (0..L-1) through the Patch32 Plan-controlled RPC.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

MIB = 1024 * 1024
TOKEN = re.compile(r"([GCNA])([1-9][0-9]*)\Z")
BLOCK = re.compile(r"^blk\.([0-9]+)\.")

@dataclass(frozen=True)
class Stage:
    backend: str
    layers: int
    region: int
    first_layer: int
    last_layer_exclusive: int

def describe(pattern: str, cycles: int) -> dict:
    if not 2 <= cycles <= 4:
        raise ValueError("PROFILE_REQUIRES_2_TO_4_CYCLES")
    atoms = pattern.split(",")
    if not 2 <= len(atoms) <= 4:
        raise ValueError("INVALID_GCNA_PROBE_PATTERN")
    parsed = []
    for item in atoms:
        item = item.strip()
        m = TOKEN.fullmatch(item)
        if not m:
            raise ValueError(f"INVALID_GCNA_PROBE_STAGE:{item}")
        parsed.append((m[1], int(m[2])))
    kinds = [k for k, _ in parsed]
    if kinds != sorted(kinds, key="GCNA".index) or kinds[0] != "G" or \
            len(kinds) != len(set(kinds)) or not any(k in kinds for k in ("N", "A")):
        raise ValueError("INVALID_GCNA_STAGE_ORDER")
    if "N" not in kinds or "A" not in kinds:
        raise ValueError("PROFILE_REQUIRES_ADJACENT_NPU_AND_ADRENO")
    if dict(parsed)["N"] != 1:
        raise ValueError("PHONE_HTP_PROFILE_REQUIRES_N1")
    if any(n > 8 for k,n in parsed if k in ("G", "C", "A")):
        raise ValueError("PROFILE_STAGE_DEPTH_MAX_8")
    stages: list[Stage] = []
    offset = 0
    for _ in range(cycles):
        for kind, depth in parsed:
            stages.append(Stage(kind, depth, len(stages), offset, offset+depth))
            offset += depth
    return {
        "schema": 1,
        "mode": "COMPACT_GCNA_PROBE_NOT_FULL_INFERENCE",
        "pattern": ",".join(f"{k}{n}" for k,n in parsed),
        "cycles": cycles,
        "layers": offset,
        "stages": [s.__dict__ for s in stages],
        "runtime_layout": ",".join(
            f"{'P' if s.backend in ('N', 'A') else s.backend}{s.layers}"
            for s in stages),
        "npu_regions": [s.region for s in stages if s.backend == "N"],
        "adreno_regions": [s.region for s in stages if s.backend == "A"],
        "sample_layers": {
            kind: sum(s.layers for s in stages if s.backend == kind)
            for kind in "GCNA",
        },
        "original_layer_indices": list(range(offset)),
    }

def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    n = path.stat().st_size
    h.update(str(n).encode("ascii"))
    with path.open("rb") as stream:
        h.update(stream.read(MIB))
        if n > MIB:
            stream.seek(max(0, n-MIB))
            h.update(stream.read(MIB))
    return h.hexdigest()

def _gguf():
    # Use the repository's matching GGUF reader/writer, including quantized
    # GGML tensor layouts; no pip installation and NO dequant/requant.
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "gguf-py"))
    import gguf  # noqa: PLC0415
    return gguf

def create_probe(source: Path, dest: Path, spec: dict) -> dict:
    gguf = _gguf()
    if not source.is_file() or source.resolve() == dest.resolve():
        raise ValueError("SOURCE_MUST_EXIST_AND_DIFFER_FROM_PROBE")
    if dest.exists():
        raise FileExistsError(
            "PROBE_ALREADY_EXISTS: remove only after you have verified the source")
    reader = gguf.GGUFReader(source)
    arch = reader.get_field("general.architecture")
    layers_field = reader.get_field("qwen3.block_count")
    if arch is None or arch.contents() != "qwen3" or layers_field is None:
        raise ValueError("COMPACT_PROFILE_QWEN3_DENSE_ONLY")
    source_layers = int(layers_field.contents())
    probe_layers = int(spec["layers"])
    if probe_layers >= source_layers:
        raise ValueError(
            "NOT_AN_ISOLATED_PROBE: sampled layer count must be smaller than model")
    if any(key.startswith("split.") for key in reader.fields):
        raise ValueError("SHARDED_GGUF_NOT_SUPPORTED_FOR_COMPACT_PROFILE")
    writer = gguf.GGUFWriter(dest.with_suffix(dest.suffix+".partial"), "qwen3",
                             endianess=reader.endianess)
    writer.data_alignment = int(reader.alignment)
    tensor_count = 0
    try:
        for name, field in reader.fields.items():
            if name.startswith("GGUF.") or name == "general.architecture":
                continue
            if name == "qwen3.block_count":
                writer.add_key_value(name, probe_layers, field.types[0])
                continue
            # Retain exact scalar types and the correct ARRAY element type,
            # including tokenizer tokens, merge scores and special tokens.
            typ = field.types[0]
            sub = field.types[-1] if typ == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(name, field.contents(), typ, sub)
        for tensor in reader.tensors:
            match = BLOCK.match(tensor.name)
            if match and int(match[1]) >= probe_layers:
                continue
            # Quantized tensors are still their original packed bytes.
            # GGUFWriter infers their logical shape from raw_dtype.
            writer.add_tensor(tensor.name, tensor.data,
                              raw_dtype=tensor.tensor_type)
            tensor_count += 1
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    part = dest.with_suffix(dest.suffix+".partial")
    test = gguf.GGUFReader(part)
    if int(test.get_field("qwen3.block_count").contents()) != probe_layers or \
            len(test.tensors) != tensor_count or \
            any(int(m[1]) >= probe_layers
                for t in test.tensors if (m := BLOCK.match(t.name))):
        raise ValueError("COMPACT_GGUF_VERIFICATION_FAILED")
    part.replace(dest)
    return {
        **spec,
        "source_model": source.name,
        "source_model_fingerprint": fingerprint(source),
        "source_model_layers": source_layers,
        "probe_model": dest.name,
        "probe_model_fingerprint": fingerprint(dest),
        "tensor_count": tensor_count,
        "phone_weight_source": "ORIGINAL_GGUF_FIRST_L_LAYERS",
        "model_equivalence": "NOT_EQUIVALENT_FOR_OUTPUTS_TIMING_ONLY",
    }

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="G7,C1,N1,A1")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    spec = describe(args.pattern, args.cycles)
    if args.describe:
        print(json.dumps(spec, ensure_ascii=False))
        return
    if not args.model or not args.output:
        ap.error("A compact real model probe needs --model and --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = create_probe(args.model.resolve(strict=True), args.output, spec)
    manifest = args.manifest or args.output.with_suffix(".probe.json")
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"[GCNA_PROFILE] original={result['source_model']} "
          f"compact={result['probe_model']} "
          f"sample_layers={result['layers']}/{result['source_model_layers']} "
          f"regions={len(result['stages'])} status=READY")
    print(f"[GCNA_PROFILE] manifest={manifest}")

if __name__ == "__main__":
    main()
