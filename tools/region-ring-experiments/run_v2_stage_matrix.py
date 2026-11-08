#!/usr/bin/env python3
"""Run/resume independent G/C/N/A Qwen3 Dense Stage Graph samples.

PC G/C runs through native local CPU/CUDA executable. With --phone,
N/A runs through a dedicated versioned V2 Phone profile control service.
No manual profile copying. Both sides use identical original GGUF bytes,
verified by SHA256. Only CPU-parity-approved HTP samples enter the PASS
performance CSV. PARITY_FAIL samples remain inspectable separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from v2_profile_wire import (
    ProfileProtocolError, SOURCE, WIRE_VERSION, call, validate_sample,
    validate_shape,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_device_id() -> str:
    # Include physical hardware, OS and driver in the local G/C cache
    # identity. A different PC (or GPU driver) must not reuse measurements.
    pieces = [
        platform.system(), platform.release(), platform.machine(),
        platform.node(), platform.processor(),
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    ]
    try:
        info = subprocess.run([
            "nvidia-smi", "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader"], capture_output=True, text=True,
            timeout=5, check=False)
        if info.returncode == 0:
            pieces.append(info.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()


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
            if kv >= x]


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


def read_native(text: str) -> dict[str, str]:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if len(rows) != 1 or not rows[0] or None in rows[0]:
        raise ProfileProtocolError("NATIVE_PROFILE_MISSING_OR_MULTIPLE_CSV_ROWS")
    return rows[0]


def strict_parity(row: dict[str, str],
                  shape: dict[str, Any]) -> None:
    if not shape["verify_cpu"]:
        if row.get("parity") == "FAIL":
            raise ProfileProtocolError("NATIVE_PARITY_FAILED_WITHOUT_REFERENCE")
        return
    if row.get("parity") not in ("PASS", "FAIL"):
        raise ProfileProtocolError("MISSING_MANDATORY_CPU_PARITY")
    try:
        numeric_pass = (
            float(row["relative_rmse"]) <= shape["max_relative_rmse"] and
            float(row["max_abs"]) <= shape["max_abs_error"])
    except (KeyError, ValueError) as exc:
        raise ProfileProtocolError("MISSING_PARITY_ERROR_METRICS") from exc
    if (row["parity"] == "PASS") != numeric_pass:
        raise ProfileProtocolError("PARITY_LABEL_CONTRADICTS_NUMERICAL_ERRORS")


def native_local(exe: Path, model: Path, shape: dict[str, Any],
                 timeout: int) -> tuple[str, dict[str, str], str]:
    argv = [
        str(exe), "--model", str(model), "--backend", shape["backend"],
        "--layers", str(shape["layers"]), "--chunk", str(shape["chunk"]),
        "--kv", str(shape["kv_tokens"]), "--warmups", str(shape["warmups"]),
        "--repeats", str(shape["repeats"]), "--threads", str(shape["threads"]),
        "--relative-rmse", str(shape["max_relative_rmse"]),
        "--max-abs-error", str(shape["max_abs_error"]),
        "--bank-repeats", str(shape.get("bank_repeats", 0)),
        "--host-stage-cap-mib", str(shape.get("host_stage_cap_mib", 1024)),
    ]
    if shape["verify_cpu"]:
        argv.append("--verify-cpu")
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              check=False, timeout=timeout or None)
    except subprocess.TimeoutExpired as exc:
        raise ProfileProtocolError("NATIVE_PROFILE_TIMEOUT") from exc
    if proc.returncode not in (0, 2):
        raise ProfileProtocolError(
            f"NATIVE_EXIT_{proc.returncode}: {proc.stderr[-3000:]}")
    row = read_native(proc.stdout)
    # Local G/C is validated with the same strict metrics as remote N/A.
    # Enforce source and parameter shape even if the native binary changed.
    expected = {
        "backend": shape["backend"], "layers": shape["layers"],
        "chunk": shape["chunk"], "kv_tokens": shape["kv_tokens"],
        "repeats": shape["repeats"],
    }
    if shape["backend"] in "NA":
        row = validate_sample(row, shape)
    else:
        try:
            if row["source"] != SOURCE or any(
                    str(row[k]) != str(v) for k, v in expected.items()):
                raise ProfileProtocolError("LOCAL_PROFILE_SHAPE_MISMATCH")
            for key in ("compute_median_ms", "prepare_median_ms",
                        "weight_upload_ms", "relative_rmse", "max_abs"):
                value = float(row[key])
                if not math.isfinite(value) or value < 0:
                    raise ProfileProtocolError("LOCAL_PROFILE_NONFINITE_METRIC")
            if float(row["compute_median_ms"]) <= 0:
                raise ProfileProtocolError("LOCAL_PROFILE_ZERO_COMPUTE")
        except (KeyError, ValueError) as exc:
            raise ProfileProtocolError("LOCAL_PROFILE_INVALID_ROW") from exc
    strict_parity(row, shape)
    if proc.returncode == 0 and (
            row.get("parity") != "FAIL" and
            (not shape["verify_cpu"] or row.get("parity") == "PASS")):
        return "PASS", row, proc.stderr[-3000:]
    if proc.returncode == 2 and row.get("parity") == "FAIL":
        return "PARITY_FAIL", row, proc.stderr[-3000:]
    raise ProfileProtocolError("NATIVE_RETURN_CODE_PARITY_INCONSISTENT")


def hello_phone(endpoint: str, token: str,
                timeout: int, model_sha: str) -> dict[str, Any]:
    info = call(endpoint, "hello", token=token, timeout=timeout)
    if info.get("status") != "OK" or \
            info.get("model_sha256") != model_sha or \
            not isinstance(info.get("exe_sha256"), str) or \
            len(info["exe_sha256"]) != 64 or \
            not isinstance(info.get("device_id"), str) or \
            len(info["device_id"]) != 64 or \
            info.get("transport") != "V2_PHONE_PROFILE_SIDECAR" or \
            not set(("N", "A")).issubset(info.get("backends", [])):
        raise ProfileProtocolError(
            "PHONE_MODEL_OR_RUNTIME_IDENTITY_MISMATCH: "
            "verify PC/Phone GGUF bytes and Phone native executable")
    return info


def phone_remote(endpoint: str, token: str, timeout: int,
                 model_sha: str, phone: dict[str, Any],
                 shape: dict[str, Any]) -> tuple[str, dict[str, str], str]:
    response = call(endpoint, "profile", {
        "model_sha256": model_sha,
        "shape": validate_shape(shape),
    }, token=token, timeout=timeout)
    if response.get("model_sha256") != model_sha or \
            response.get("exe_sha256") != phone["exe_sha256"] or \
            response.get("device_id") != phone["device_id"]:
        raise ProfileProtocolError("PHONE_PROFILE_IDENTITY_CHANGED_DURING_RUN")
    state = response.get("status")
    if state not in ("PASS", "PARITY_FAIL"):
        raise ProfileProtocolError("PHONE_PROFILE_INVALID_RESPONSE_STATUS")
    row = validate_sample(response.get("sample"), shape)
    strict_parity(row, shape)
    if state == "PASS" and (
            row["parity"] == "FAIL" or
            (shape["verify_cpu"] and row["parity"] != "PASS")):
        raise ProfileProtocolError("PHONE_PROFILE_FALSE_PARITY_PASS")
    if state == "PARITY_FAIL" and row["parity"] != "FAIL":
        raise ProfileProtocolError("PHONE_PROFILE_MISSING_PARITY_FAILURE")
    return state, row, str(response.get("diagnostics", ""))[-3000:]


def save_rows(path: Path, items: list[dict[str, Any]]) -> None:
    # Only records with the same actual native output schema can be combined.
    if not items:
        if path.exists():
            path.unlink()
        return
    fields = list(items[0]["sample"])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for record in items:
            if set(record["sample"]) != set(fields):
                raise RuntimeError("STAGE_PROFILE_SCHEMA_CHANGED_DURING_RUN")
            writer.writerow(record["sample"])
    os.replace(tmp, path)


def run_matrix(args: argparse.Namespace) -> int:
    layers = csv_positive(args.layers, "LAYERS")
    chunks = csv_positive(args.chunks, "CHUNKS")
    kvs = csv_positive(args.kvs, "KVS")
    combinations = cases(args.backends, layers, chunks, kvs)
    if not combinations:
        raise ValueError("NO_VALID_CHUNK_KV_COMBINATIONS")
    endpoint = getattr(args, "phone", "")
    use_phone = bool(endpoint and any(b in "NA" for b, _, _, _ in combinations))
    local_needed = any(b in "GC" or not (b in "NA" and use_phone)
                       for b, _, _, _ in combinations)
    print("[V2_PROFILE_MATRIX] combinations=", len(combinations),
          " backends=", args.backends,
          " phone_rpc=", int(use_phone), sep="", flush=True)
    if args.dry_run:
        for item in combinations:
            print("[V2_PROFILE_MATRIX] shape=", item)
        return 0
    model = args.model.expanduser().resolve(strict=True)
    if not model.is_file():
        raise ValueError("MODEL_MUST_BE_A_GGUF_FILE")
    exe = None
    if local_needed:
        if not getattr(args, "exe", None):
            raise ValueError("LOCAL_GC_PROFILE_REQUIRES_EXE")
        exe = args.exe.expanduser().resolve(strict=True)
        if not exe.is_file():
            raise ValueError("NATIVE_PROFILE_EXE_NOT_A_FILE")
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_sha = sha256(model)
    native_sha = sha256(exe) if exe else None
    local_device = host_device_id() if local_needed else None
    phone = None
    token = getattr(args, "phone_token", "") or \
        os.environ.get("V2_PROFILE_TOKEN", "")
    phone_timeout = int(getattr(args, "phone_timeout", 360))
    if phone_timeout <= 0:
        raise ValueError("INVALID_PHONE_TIMEOUT")
    phone_error = ""
    if use_phone:
        try:
            phone = hello_phone(endpoint, token, phone_timeout, model_sha)
            print("[V2_PROFILE_MATRIX] PHONE_CONNECTED device_id=" +
                  phone["device_id"][:16] + " exe_sha256=" +
                  phone["exe_sha256"][:16], flush=True)
        except (OSError, ValueError, RuntimeError) as exc:
            # Do not abandon PC G/C measurements if Phone is offline or
            # carries a different original GGUF; each N/A case will fail
            # explicitly and remain eligible for later resume.
            phone_error = str(exc)
            print("[V2_PROFILE_MATRIX] PHONE_UNAVAILABLE reason=" +
                  phone_error, file=sys.stderr, flush=True)
    complete: list[dict[str, Any]] = []
    parity_fails: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    failures = 0
    for index, (backend, n, x, kv) in enumerate(combinations, 1):
        remote = use_phone and backend in "NA"
        shape = {
            "backend": backend, "layers": n, "chunk": x,
            "kv_tokens": kv, "warmups": args.warmups,
            "repeats": args.repeats, "threads": args.threads,
            "verify_cpu": backend == "N" or args.verify_cpu,
            "max_relative_rmse": args.max_relative_rmse,
            "max_abs_error": args.max_abs_error,
            "bank_repeats": (getattr(args, "bank_repeats", 0)
                             if backend == "N" else 0),
            "host_stage_cap_mib": getattr(args, "host_stage_cap_mib", 1024),
        }
        identity = {
            "model_sha256": model_sha,
            "exe_sha256": (phone["exe_sha256"] if phone else None)
                if remote else native_sha,
            "device_id": (phone["device_id"] if phone else "PHONE_UNAVAILABLE")
                if remote else local_device,
            "rpc_version": WIRE_VERSION if remote else 0,
        }
        key = {**identity, **shape}
        destination = out / f"{backend}-L{n}-X{x}-KV{kv}.json"
        if not args.update and valid_cached(destination, key):
            print(f"[V2_PROFILE_MATRIX] {index}/{len(combinations)} "
                  f"RESUME backend={backend} L={n} X={x} KV={kv}", flush=True)
            complete.append(json.loads(destination.read_text("utf-8")))
            continue
        print(f"[V2_PROFILE_MATRIX] {index}/{len(combinations)} "
              f"RUN backend={backend} L={n} X={x} KV={kv} "
              f"device={'PHONE' if remote else 'PC'}", flush=True)
        try:
            if remote:
                if phone is None:
                    raise ProfileProtocolError(
                        "PHONE_PROFILE_NOT_CONNECTED:" + phone_error)
                state, sample, diag = phone_remote(
                    endpoint, token, phone_timeout, model_sha, phone, shape)
            else:
                state, sample, diag = native_local(
                    exe, model, shape, args.timeout)
            record = {
                "schema": 2, "status": state, "key": key,
                "sample": sample,
                "transport": "PHONE_RPC_V1" if remote else "LOCAL_NATIVE",
                "origin": SOURCE,
            }
            if diag:
                record["diagnostics"] = diag
            atomic_json(destination, record)
            if state == "PASS":
                complete.append(record)
                # A newer PASS invalidates any old error from the same case.
                failed_path = destination.with_suffix(".failed.json")
                if failed_path.exists():
                    failed_path.unlink()
            else:
                # Compute succeeded, but CPU reference validation failed.
                # This row stays separate and is NEVER planner-approved.
                failures += 1
                parity_fails.append(record)
                print(f"[V2_PROFILE_MATRIX] PARITY_FAIL "
                      f"{backend}-L{n}-X{x}-KV{kv} "
                      f"relative_rmse={sample['relative_rmse']} "
                      f"max_abs={sample['max_abs']}", file=sys.stderr,
                      flush=True)
        except (OSError, ValueError, RuntimeError, ProfileProtocolError) as exc:
            failures += 1
            failed = {"schema": 2, "status": "FAIL", "key": key,
                      "reason": str(exc)}
            atomic_json(destination, failed)
            atomic_json(destination.with_suffix(".failed.json"), failed)
            failed_records.append(failed)
            print("[V2_PROFILE_MATRIX] FAILED:", exc, file=sys.stderr,
                  flush=True)
    save_rows(out / "stage-profile-v2.csv", complete)
    save_rows(out / "stage-profile-v2-parity-fail.csv", parity_fails)
    # Unified structured model-specific Profile cache for the future V2
    # Planner. Never claim Planner readiness before transfer/Bank profiles
    # and all required numerical/compute backend samples are complete.
    aggregate = {
        "schema": 2,
        "model_sha256": model_sha,
        "devices": {
            "pc": {"device_id": local_device,
                   "exe_sha256": native_sha,
                   "backends": [b for b in "GC" if b in args.backends]},
            "phone": {
                "device_id": phone["device_id"] if phone else None,
                "exe_sha256": phone["exe_sha256"] if phone else None,
                "backends": [b for b in "NA" if b in args.backends],
                "transport": "V2_PHONE_PROFILE_SIDECAR" if use_phone else None,
            },
        },
        "request": {
            "backends": args.backends,
            "layers": layers, "chunks": chunks, "kvs": kvs,
            "warmups": args.warmups, "repeats": args.repeats,
            "threads": args.threads,
            "max_relative_rmse": args.max_relative_rmse,
            "max_abs_error": args.max_abs_error,
            "bank_repeats": getattr(args, "bank_repeats", 0),
            "host_stage_cap_mib": getattr(args, "host_stage_cap_mib", 1024),
        },
        "expected_cases": len(combinations),
        "counts": {"pass": len(complete),
                   "parity_fail": len(parity_fails),
                   "failed": len(failed_records)},
        "compute_complete": len(complete) == len(combinations),
        "planner_ready": False,
        "pending": ["transfer_only_profile", "htp_bank_refill_profile",
                    "runtime_memory_feasibility", "multi_backend_dag"],
        "compute_samples": [
            {"key": record["key"], "sample": record["sample"],
             "transport": record["transport"], "status": record["status"]}
            for record in complete
        ],
        "parity_fail_samples": [
            {"key": record["key"], "sample": record["sample"],
             "transport": record["transport"], "status": record["status"]}
            for record in parity_fails
        ],
        "failed_samples": [
            {"key": record["key"], "reason": record["reason"]}
            for record in failed_records
        ],
    }
    atomic_json(out / "profile.json", aggregate)
    print(f"[V2_PROFILE_MATRIX] profile_json={out / 'profile.json'}")
    print(f"[V2_PROFILE_MATRIX] pass={len(complete)} "
          f"parity_fail={len(parity_fails)} fail={failures-len(parity_fails)}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, help="Local PC native profiling executable")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--backends", default="GC",
                        help="Subset of GCNA. With --phone, N/A run on Phone")
    parser.add_argument("--phone", default="",
                        help="V2 phone Profile endpoint HOST:PORT, e.g. 127.0.0.1:50177")
    parser.add_argument("--phone-token", default="",
                        help="Shared token (prefer V2_PROFILE_TOKEN environment variable)")
    parser.add_argument("--phone-timeout", type=int, default=360)
    parser.add_argument("--layers", default="1,2,4,6")
    parser.add_argument("--chunks", default="16,32,64,128")
    parser.add_argument("--kvs", default="128,512,1024")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-relative-rmse", type=float, default=0.02)
    parser.add_argument("--max-abs-error", type=float, default=0.5)
    parser.add_argument("--bank-repeats", type=int, default=0,
                        help="N only: measure raw HTP weight reload into the same Bank, never dual-Bank overlap")
    parser.add_argument("--host-stage-cap-mib", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--verify-cpu", action="store_true",
                        help="Check G/C/A output against CPU (N is always verified)")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.warmups < 0 or args.repeats < 1 or args.threads < 1 or
            args.timeout < 0 or args.phone_timeout < 1 or
            args.bank_repeats < 0 or args.bank_repeats > 20 or
            args.host_stage_cap_mib < 1):
        parser.error("INVALID_PROFILE_TIMINGS")
    try:
        return run_matrix(args)
    except (OSError, ValueError, RuntimeError, ProfileProtocolError) as exc:
        print(f"[V2_PROFILE_MATRIX] FAIL {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
