#!/usr/bin/env python3
"""Run/resume REAL isolated G/C/N/A first-N-layer Stage benchmarks.

Each sample invokes the compiled native v2-stage-profile program.
No Region Ring, no feeder Stages, no compact model, no fake multi-layer
estimates. For N, CPU parity is mandatory in the native binary.
Phone samples can be executed locally in Termux until RPC transport V2
is integrated. The result provenance is intentionally explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_positive(spec: str, name: str) -> list[int]:
    try:
        values = [int(s.strip()) for s in spec.split(",")]
    except ValueError as exc:
        raise ValueError(f"INVALID_{name}_LIST:{spec}") from exc
    if not values or any(x <= 0 for x in values):
        raise ValueError(f"INVALID_{name}_LIST:{spec}")
    return sorted(set(values))


def cases(backends: str, layers: list[int], chunks: list[int],
          kvs: list[int]) -> list[tuple[str, int, int, int]]:
    if not backends or not set(backends).issubset("GCNA"):
        raise ValueError("INVALID_BACKENDS:expected a subset of GCNA")
    return [(backend, n, x, kv) for backend, n, x, kv in
            itertools.product(dict.fromkeys(backends), layers, chunks, kvs)
            if kv >= x and kv % x == 0]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    os.replace(temp, path)


def valid_cached(path: Path, key: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
        return old.get("status") == "PASS" and old.get("key") == key
    except (OSError, ValueError):
        return False


def run_matrix(args: argparse.Namespace) -> int:
    exe = args.exe.expanduser().resolve(strict=True)
    model = args.model.expanduser().resolve(strict=True)
    if not exe.is_file() or not model.is_file():
        raise ValueError("EXECUTABLE_AND_GGUF_MUST_BE_FILES")
    layers = csv_positive(args.layers, "LAYERS")
    chunks = csv_positive(args.chunks, "CHUNKS")
    kvs = csv_positive(args.kvs, "KVS")
    combinations = cases(args.backends, layers, chunks, kvs)
    if not combinations:
        raise ValueError("NO_VALID_CHUNK_KV_COMBINATIONS")
    print("[V2_PROFILE_MATRIX] combinations=", len(combinations),
          " backends=", args.backends, sep="", flush=True)
    if args.dry_run:
        for item in combinations:
            print("[V2_PROFILE_MATRIX] shape=", item)
        return 0
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    identities = {
        "exe_sha256": sha256(exe),
        "model_sha256": sha256(model),
    }
    complete: list[dict[str, Any]] = []
    failures = 0
    for index, (backend, n, x, kv) in enumerate(combinations, 1):
        shape = {"backend": backend, "layers": n, "chunk": x,
                 "kv_tokens": kv, "warmups": args.warmups,
                 "repeats": args.repeats, "threads": args.threads,
                 "verify_cpu": backend == "N" or args.verify_cpu,
                 "max_relative_rmse": args.max_relative_rmse,
                 "max_abs_error": args.max_abs_error}
        key = {**identities, **shape}
        destination = out / f"{backend}-L{n}-X{x}-KV{kv}.json"
        if not args.update and valid_cached(destination, key):
            print(f"[V2_PROFILE_MATRIX] {index}/{len(combinations)} "
                  f"RESUME backend={backend} L={n} X={x} KV={kv}", flush=True)
            complete.append(json.loads(destination.read_text("utf-8")))
            continue
        argv = [str(exe), "--model", str(model), "--backend", backend,
                "--layers", str(n), "--chunk", str(x), "--kv", str(kv),
                "--warmups", str(args.warmups),
                "--repeats", str(args.repeats),
                "--threads", str(args.threads),
                "--relative-rmse", str(args.max_relative_rmse),
                "--max-abs-error", str(args.max_abs_error)]
        if shape["verify_cpu"]:
            argv.append("--verify-cpu")
        print(f"[V2_PROFILE_MATRIX] {index}/{len(combinations)} "
              f"RUN backend={backend} L={n} X={x} KV={kv}", flush=True)
        try:
            run = subprocess.run(argv, capture_output=True, text=True,
                                 check=False, timeout=args.timeout or None)
            if run.returncode != 0:
                raise RuntimeError(
                    f"NATIVE_EXIT_{run.returncode}: {run.stderr[-3000:]}")
            reader = csv.DictReader(io.StringIO(run.stdout))
            samples = list(reader)
            if len(samples) != 1 or not samples[0] or \
                    samples[0]["backend"] != backend or \
                    int(samples[0]["layers"]) != n or \
                    int(samples[0]["chunk"]) != x or \
                    int(samples[0]["kv_tokens"]) != kv or \
                    int(samples[0]["repeats"]) != args.repeats or \
                    samples[0].get("source") != "NATIVE_SINGLE_BACKEND_GGUF_PREFIX":
                raise ValueError("MISSING_OR_WRONG_NATIVE_RESULT_ROW")
            if backend == "N" and samples[0]["parity"] != "PASS":
                raise RuntimeError("HTP_RESULT_MISSING_CPU_PARITY_PASS")
            record = {"schema": 1, "status": "PASS", "key": key,
                      "sample": samples[0],
                      "origin": "NATIVE_SINGLE_BACKEND_GGUF_PREFIX"}
            atomic_json(destination, record)
            complete.append(record)
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            failures += 1
            failed = {"schema": 1, "status": "FAIL", "key": key,
                      "reason": str(exc)}
            # A forced refresh must never leave an older PASS selectable
            # after the same case failed. Preserve the error separately.
            atomic_json(destination, failed)
            atomic_json(destination.with_suffix(".failed.json"), failed)
            print("[V2_PROFILE_MATRIX] FAILED:", exc, file=sys.stderr,
                  flush=True)
    if complete:
        path = out / "stage-profile-v2.csv"
        fields = list(complete[0]["sample"].keys())
        temp = path.with_suffix(".csv.tmp")
        with temp.open("w", newline="", encoding="utf-8") as inp:
            writer = csv.DictWriter(inp, fieldnames=fields)
            writer.writeheader()
            for case in complete:
                if list(case["sample"]) != fields:
                    raise RuntimeError("PROFILE_SCHEMA_CHANGED_WITHIN_RUN")
                writer.writerow(case["sample"])
        os.replace(temp, path)
        print(f"[V2_PROFILE_MATRIX] csv={path}")
    print(f"[V2_PROFILE_MATRIX] pass={len(complete)} fail={failures}")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exe", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--backends", default="GC",
                   help="Subset of G,C,N,A. Only select locally available backends")
    p.add_argument("--layers", default="1,2,4,6")
    p.add_argument("--chunks", default="16,32,64,128")
    p.add_argument("--kvs", default="128,512,1024")
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-relative-rmse", type=float, default=0.02)
    p.add_argument("--max-abs-error", type=float, default=0.5)
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--verify-cpu", action="store_true",
                   help="Also compare G/C/A results against native CPU")
    p.add_argument("--update", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.threads < 1 or args.timeout < 0:
        p.error("INVALID_WARMUPS_REPEATS_THREADS_TIMEOUT")
    try:
        return run_matrix(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[V2_PROFILE_MATRIX] FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
