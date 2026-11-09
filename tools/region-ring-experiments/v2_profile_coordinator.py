"""Model-bound native G/C/N/A Profile discovery and missing-case refresh.

This coordinator is run ONCE before a run_prompt_sweep, never once per
Prompt. Reuses the independent run_v2_stage_matrix.py sampler over the
existing inference RPC port 50052, with per-case resume logic.
N4 PARITY_FAIL is not a performance point:
only approved native PASS measurements may enter the C++ Planner.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from run_v2_stage_matrix import host_device_id, sha256


def select_profile_dir(root: Path, model: Path, model_hash: str,
                       preferred: Path | None = None) -> Path:
    if preferred is not None:
        return preferred.expanduser().resolve()
    previous = root / "profile-v2" / "gcna-expanded"
    try:
        manifest = json.loads((previous / "profile.json").read_text("utf-8"))
        if manifest.get("model_sha256") == model_hash:
            return previous
    except (OSError, ValueError):
        pass
    safe_stem = "".join(c if c.isalnum() or c in ".-_" else "_"
                        for c in model.stem)[:80]
    return root / "model-profiles" / safe_stem / "native-gcna-v2"


def validated_profile(directory: Path, model_sha: str,
                      pc_id: str, profiler_sha: str,
                      *, required_chunks: tuple[int, ...] = (32, 64, 96, 128),
                      required_kvs: tuple[int, ...] = (512, 1024)
                      ) -> tuple[bool, str]:
    path = directory / "profile.json"
    csv_path = directory / "stage-profile-v2.csv"
    if not path.is_file() or not csv_path.is_file():
        return False, "MISSING_NATIVE_PROFILE_FILES"
    try:
        profile = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return False, "BAD_NATIVE_PROFILE_JSON"
    if profile.get("schema") != 2 or profile.get("model_sha256") != model_sha:
        return False, "MODEL_FINGERPRINT_MISMATCH"
    devices = profile.get("devices", {})
    pc = devices.get("pc", {})
    phone = devices.get("phone", {})
    if pc.get("device_id") != pc_id or pc.get("exe_sha256") != profiler_sha:
        return False, "PC_BINARY_OR_DEVICE_MISMATCH"
    if (not isinstance(phone.get("device_id"), str) or
            not isinstance(phone.get("runtime_sha256"), str)):
        return False, "PHONE_PROFILE_IDENTITY_MISSING"
    eligible: dict[str, set[int]] = {b: set() for b in "GCNA"}
    for record in profile.get("compute_samples", []):
        if record.get("status") != "PASS":
            continue
        key = record.get("key", {})
        sample = record.get("sample", {})
        backend = key.get("backend")
        if backend not in eligible:
            continue
        if backend == "N" and sample.get("parity") != "PASS":
            continue
        try:
            layers = int(key["layers"])
            ms = float(sample["compute_median_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        if layers > 0 and 0 < ms < 1e8:
            eligible[backend].add(layers)
    if not eligible["G"] or not eligible["C"]:
        return False, "PC_GC_COMPUTE_PROFILE_MISSING"
    if 1 not in eligible["N"] or 1 not in eligible["A"]:
        return False, "N1_OR_A1_PROFILE_MISSING"
    bank = profile.get("npu_weight_refill_points", [])
    if not any(p.get("backend") == "N" and
               int(p.get("layers", 0)) == 1 and
               float(p.get("refill_median_ms", 0)) > 0
               for p in bank):
        return False, "N1_WEIGHT_REFILL_PROFILE_MISSING"
    try:
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            cols = set(reader.fieldnames or [])
            required = {"backend", "layers", "chunk", "kv_tokens",
                        "compute_median_ms", "prepare_median_ms",
                        "weight_bytes", "kv_bytes",
                        "graph_workspace_bytes", "parity", "source"}
            if not required.issubset(cols):
                return False, "NATIVE_STAGE_CSV_COLUMNS_MISSING"
            rows = list(reader)
    except (OSError, ValueError):
        return False, "CANNOT_READ_NATIVE_STAGE_CSV"
    if not rows:
        return False, "NATIVE_STAGE_CSV_EMPTY"
    if any(row.get("backend") == "N" and row.get("parity") != "PASS"
           for row in rows):
        return False, "UNSAFE_HTP_STAGE_IN_PASS_CSV"
    # Do not load a PASS CSV from another experiment merely because its
    # header happens to match. The matrix writes JSON and CSV together,
    # but a stopped run can leave a stale file behind.
    try:
        manifest_shapes = {
            (record["key"]["backend"],
             int(record["key"]["layers"]),
             int(record["key"]["chunk"]),
             int(record["key"]["kv_tokens"]))
            for record in profile.get("compute_samples", [])
            if record.get("status") == "PASS"
        }
        csv_shapes = {
            (row["backend"], int(row["layers"]),
             int(row["chunk"]), int(row["kv_tokens"]))
            for row in rows
        }
    except (KeyError, ValueError, TypeError):
        return False, "NATIVE_PROFILE_SHAPES_UNREADABLE"
    if manifest_shapes != csv_shapes:
        return False, "NATIVE_PASS_CSV_MANIFEST_MISMATCH"
    # The original apply-wide coarse and full-DAG phases can select any
    # supported chunk/KV combination. A single N1/A1 sample is NOT enough
    # to declare the cached matrix ready: the rest of the shapes must be
    # measured or the coordinator must resume their independent probes.
    # Only correctness-qualified NPU PASS samples appear in csv_shapes.
    # N2/N4 are optional until production can execute whole multi-layer
    # HTP Regions. Do not require their known PARITY_FAIL shapes.
    required_depths = {"G": 2, "C": 2, "N": 1, "A": 1}
    if not required_chunks or not required_kvs or any(
            type(x) is not int or x < 1 for x in
            (*required_chunks, *required_kvs)):
        return False, "INVALID_REQUIRED_PROFILE_GRID"
    for backend, depth in required_depths.items():
        for chunk in required_chunks:
            for kv in required_kvs:
                if kv < chunk:
                    continue
                if (backend, depth, chunk, kv) not in manifest_shapes:
                    return False, (
                        f"INCOMPLETE_{backend}{depth}_X{chunk}_KV{kv}")
    return True, "VERIFIED"


def prepare_native_profile(root: Path, model: Path, rpc: str,
                           threads: int, force: bool, dry_run: bool,
                           directory: Path | None = None,
                           profiler_exe: Path | None = None,
                           phone_endpoint: str = "",
                           required_kvs: tuple[int, ...] = (512, 1024)
                           ) -> Path:
    model_sha = sha256(model)
    folder = select_profile_dir(root, model, model_sha, directory)
    native_exe = (profiler_exe or (
        root / "build-tensor-v2" / "bin" / "Release" /
        "region-ring-v2-stage-profile.exe")).resolve()
    if not native_exe.is_file():
        raise RuntimeError(
            f"NATIVE_STAGE_PROFILER_MISSING: {native_exe}; "
            "build region-ring-v2-stage-profile first")
    native_sha = sha256(native_exe)
    pc_id = host_device_id()
    valid, reason = validated_profile(
        folder, model_sha, pc_id, native_sha, required_kvs=required_kvs)
    if valid and not force:
        print(f"[V2_PROFILE_CACHE] status=HIT source=NATIVE_GCNA "
              f"file={folder / 'profile.json'}", flush=True)
        return folder / "stage-profile-v2.csv"
    print(f"[V2_PROFILE_CACHE] status={'FORCE' if force else 'MISS'} "
          f"reason={reason} directory={folder}", flush=True)
    if dry_run:
        return folder / "stage-profile-v2.csv"
    if ":" not in rpc or not rpc.rsplit(":", 1)[1].isdigit():
        raise ValueError("INVALID_RPC_ADDRESS_FOR_PHONE_PROFILE")
    # The native Profile protocol now shares the main GGML inference RPC
    # listener. The old 50177 Python sidecar is not required.
    phone = phone_endpoint or rpc
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "tools" / "region-ring-experiments" /
            "run_v2_stage_matrix.py"),
        "--exe", str(native_exe),
        "--model", str(model),
        "--phone", phone,
        "--backends", "GCNA",
        "--layers-g", "2,4,6",
        "--layers-c", "2,4",
        "--layers-n", "1,2,4",
        "--layers-a", "1,2,3,4",
        "--chunks", "32,64,96,128",
        "--kvs", ",".join(str(kv) for kv in required_kvs),
        "--warmups", "2",
        "--repeats", "5",
        "--bank-repeats", "3",
        "--threads", str(threads),
        "--out", str(folder),
    ]
    if force:
        command.append("--update")
    result = subprocess.run(command, cwd=root, check=False)
    valid, reason = validated_profile(
        folder, model_sha, pc_id, native_sha, required_kvs=required_kvs)
    if not valid:
        raise RuntimeError(
            f"NATIVE_PROFILE_INCOMPLETE: {reason}; exit={result.returncode}. "
            f"Check {folder}; Phone profiling requires a matching Android RPC "
            f"server and local native profiler on {phone}. "
            "Known N4 parity failures are excluded, not silently accepted.")
    if result.returncode:
        manifest = json.loads((folder / "profile.json").read_text("utf-8"))
        # The matrix uses nonzero status for known NPU parity failures.
        # Those shapes are excluded from PASS CSV and cannot be planned.
        if manifest.get("counts", {}).get("failed", 0):
            raise RuntimeError("NATIVE_PROFILE_HAS_FAILED_PROBES")
        print("[V2_PROFILE_CACHE] status=READY_WITH_EXCLUDED_PARITY_FAIL "
              f"parity_fail={manifest.get('counts', {}).get('parity_fail', 0)}",
              flush=True)
    print(f"[V2_PROFILE_CACHE] status=READY source=NATIVE_GCNA "
          f"csv={folder / 'stage-profile-v2.csv'}", flush=True)
    return folder / "stage-profile-v2.csv"
