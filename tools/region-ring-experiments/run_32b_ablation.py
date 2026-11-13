#!/usr/bin/env python3
"""Qwen3-32B four-case ablation; one prompt by default, resumable by case.

A0: fixed G8/P6, two GPU banks with mover.
A1: same layout, two banks and demand loading (no mover).
A2: single GPU bank, measured-profile guided multi-cycle layout.
A3: single GPU bank, one G/C/P cycle, capacity-first GPU -> CPU -> HTP.
Does not silently treat a two-bank plan as a single-bank experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXED = "G8,P6,G8,P6,G8,P6,G8,P6,G8"
MIB = 1048576
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_layer_weights(path: Path, n_layers: int) -> list[int]:
    """Use actual GGUF tensor offsets, not the model file-size average."""
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF model")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise ValueError("unsupported GGUF version")
        n_tensor, n_kv = struct.unpack("<QQ", f.read(16))
        def u32():
            return struct.unpack("<I", f.read(4))[0]
        def u64():
            return struct.unpack("<Q", f.read(8))[0]
        def skip(n):
            f.seek(n, 1)
        def string():
            n = u64()
            if n > 10_000_000:
                raise ValueError("invalid GGUF string length")
            return f.read(n).decode("utf-8")
        def value(kind):
            if kind in SIZES:
                skip(SIZES[kind])
                return None
            if kind == 8:
                return string()
            if kind == 9:
                elem = u32()
                count = u64()
                if count > 10_000_000:
                    raise ValueError("oversized GGUF array")
                if elem in SIZES:
                    skip(SIZES[elem] * count)
                else:
                    for _ in range(count):
                        value(elem)
                return None
            raise ValueError(f"unsupported GGUF metadata type {kind}")
        alignment = 32
        for _ in range(n_kv):
            name = string()
            kind = u32()
            if name == "general.alignment" and kind in (4, 10):
                alignment = struct.unpack("<I" if kind == 4 else "<Q",
                                          f.read(SIZES[kind]))[0]
            else:
                value(kind)
        tensors = []
        for _ in range(n_tensor):
            name = string()
            ndim = u32()
            if ndim > 8:
                raise ValueError("invalid GGUF tensor shape")
            skip(ndim * 8)
            u32()  # ggml tensor type
            offset = u64()
            tensors.append((offset, name))
        if not tensors:
            raise ValueError("GGUF has no tensors")
        if alignment < 1 or alignment > 4096:
            raise ValueError("invalid GGUF alignment")
        base = (f.tell() + alignment - 1) // alignment * alignment
        f.seek(0, 2)
        tensor_bytes = f.tell() - base
    tensors.sort()
    out = [0] * n_layers
    for i, (offset, name) in enumerate(tensors):
        next_offset = tensors[i + 1][0] if i + 1 < len(tensors) else tensor_bytes
        length = next_offset - offset
        if length < 0:
            raise ValueError("invalid GGUF tensor offsets")
        match = re.match(r"blk\.(\d+)\.", name)
        if match and int(match.group(1)) < n_layers:
            out[int(match.group(1))] += length
    if any(w == 0 for w in out):
        raise ValueError("cannot determine weights for all decoder layers")
    return out


def free_vram_mib(override: int | None) -> int:
    if override is not None:
        return override
    cmd = ["nvidia-smi", "--query-gpu=memory.free",
           "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(cmd, text=True, timeout=10)
        return int(output.strip().splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ValueError("nvidia-smi unavailable; specify --gpu-free-mib") from exc


def free_ram_mib(override: int | None) -> int:
    if override is not None:
        return override
    if sys.platform == "win32":
        import ctypes
        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMORYSTATUS()
        m.dwLength = ctypes.sizeof(m)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return int(m.ullAvailPhys // MIB)
    raise ValueError("specify --cpu-free-mib on this host")


def phone_budget_mib(override: int | None) -> int:
    if override is None:
        raise ValueError("A2/A3 require --phone-budget-mib from Phone MemAvailable; "
                         "do not infer available Android RAM from total RAM")
    return override


def measured_ms_per_layer(path: Path, chunk: int) -> dict[str, float]:
    """Approximate independent backend service from actual C/G/N stage rows."""
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    values: dict[str, list[float]] = {"G": [], "C": [], "N": []}
    for row in rows:
        device = str(row.get("device", row.get("backend", ""))).upper()
        d = {"GPU": "G", "G": "G", "CPU": "C", "C": "C",
             "PHONE": "N", "N": "N", "HTP": "N"}.get(device)
        if not d:
            continue
        try:
            layers = int(row.get("layers", 0))
            tokens = int(row.get("chunk", row.get("chunk_tokens", 0)))
            ms = float(row.get("compute_ms", row.get("compute_median_ms", 0)))
            if layers and tokens and ms > 0 and math.isfinite(ms):
                values[d].append((abs(tokens - chunk), ms / layers * chunk / tokens))
        except (ValueError, TypeError):
            pass
    missing = [d for d in values if not values[d]]
    if missing:
        raise ValueError(f"independent stage profile missing {missing}: {path}")
    return {d: sorted(records, key=lambda x: x[0])[:6][0][1]
            for d, records in values.items()}


def balanced(total: int, n: int) -> list[int]:
    return [total // n + (i < total % n) for i in range(n)]


def phone_fits(layout: list[tuple[str, int]], weights: list[int],
               budget: int, session_budget: int) -> bool:
    total, stripe = 0, [0] * 4
    pos = 0
    phone_ordinal = 0
    for kind, count in layout:
        for j in range(pos, pos + count):
            if kind == "P":
                # Conservative weight-mapping factor plus per-layer scratch.
                needed = int(weights[j] * 1.10 + 32 * MIB)
                stripe[phone_ordinal % 4] += needed
                phone_ordinal += 1
                total += needed
        pos += count
    return total + 512 * MIB < budget * MIB and all(
        n + 64 * MIB < session_budget * MIB for n in stripe)


def layout_fits(layout: list[tuple[str, int]], weights: list[int],
                gpu_free: int, cpu_free: int, phone_budget: int,
                session_budget: int, gpu_reserve: int,
                cpu_reserve: int, context: int) -> bool:
    pos = 0
    gpu_layers, gpu_peak, cpu_weights = 0, 0, 0
    for kind, count in layout:
        segment = sum(weights[pos:pos + count])
        if kind == "G":
            gpu_peak = max(gpu_peak, segment)
            gpu_layers += count
        if kind == "C":
            cpu_weights += segment
        pos += count
    if pos != len(weights):
        return False
    # 8 KiB per GPU layer/token is deliberately conservative for Qwen3-32B KV;
    # this is an estimate, NOT the native runtime's authoritative allocator.
    gpu_kv = gpu_layers * context * 8192
    return (gpu_peak + gpu_kv + gpu_reserve * MIB < gpu_free * MIB
            and cpu_weights + cpu_reserve * MIB < cpu_free * MIB
            and phone_fits(layout, weights, phone_budget, session_budget))


def multi_cycle(weights: list[int], profile: dict[str, float], gpu_free: int,
                cpu_free: int, phone_budget: int, args) -> tuple[str, dict]:
    count = len(weights)
    scored = []
    for cycles in range(2, 6):
        for gpu_total in range(cycles, count - cycles + 1):
            for npu_total in range(cycles, count - gpu_total + 1):
                cpu_total = count - gpu_total - npu_total
                gs, cs, ps = [balanced(n, cycles) for n in
                               (gpu_total, cpu_total, npu_total)]
                layout = []
                for g, c, p in zip(gs, cs, ps):
                    layout.append(("G", g))
                    if c:
                        layout.append(("C", c))
                    layout.append(("P", p))
                if not layout_fits(layout, weights, gpu_free, cpu_free,
                                   phone_budget, args.htp_session_mib,
                                   args.gpu_reserve_mib, args.cpu_reserve_mib,
                                   args.context):
                    continue
                # Profile-derived, coarse *ranking only*. Native DAG is the
                # authoritative prediction after a concrete forced Plan.
                gpu_service = gpu_total * profile["G"]
                cpu_service = cpu_total * profile["C"]
                npu_service = npu_total * profile["N"]
                boundaries = len(layout) - 1
                # Only GPU-owned weights traverse H2D. Counting all model
                # weights would bias the candidate search against GPU layers.
                offset = 0
                gpu_weight_bytes = 0
                for kind, depth in layout:
                    if kind == "G":
                        gpu_weight_bytes += sum(weights[offset:offset + depth])
                    offset += depth
                h2d_ms = (gpu_weight_bytes / MIB) / 4.1
                score = max(gpu_service + h2d_ms, cpu_service, npu_service)
                score += 12.0 * boundaries + 0.25 * sum(
                    (g * profile["G"] + p * profile["N"])
                    for g, p in zip(gs, ps)) / cycles
                scored.append((score, layout, gpu_total, cpu_total, npu_total))
    if not scored:
        raise ValueError("no memory-feasible A2 multi-cycle candidate; "
                         "adjust actual memory budgets or lower context")
    score, selected, g, c, p = min(scored, key=lambda x: x[0])
    return ",".join(f"{kind}{depth}" for kind, depth in selected), {
        "rough_score_ms": round(score, 2), "gpu_layers": g,
        "cpu_layers": c, "phone_layers": p, "candidates": len(scored)}


def single_cycle(weights: list[int], gpu_free: int, cpu_free: int,
                 phone_budget: int, args) -> tuple[str, dict]:
    count = len(weights)
    # Capacity-first, no throughput optimizer: maximize GPU prefix, then CPU,
    # with at least one Phone layer to retain the G->C->P topology.
    for gpu in range(count - 1, 0, -1):
        for cpu in range(count - gpu - 1, -1, -1):
            phone = count - gpu - cpu
            if not phone:
                continue
            plan = [("G", gpu)]
            if cpu:
                plan.append(("C", cpu))
            plan.append(("P", phone))
            if layout_fits(plan, weights, gpu_free, cpu_free, phone_budget,
                           args.htp_session_mib, args.gpu_reserve_mib,
                           args.cpu_reserve_mib, args.context):
                return ",".join(f"{k}{n}" for k, n in plan), {
                    "gpu_layers": gpu, "cpu_layers": cpu, "phone_layers": phone,
                    "objective": "MAX_GPU_THEN_CPU_REMAINDER_TO_PHONE"}
    raise ValueError("no capacity-feasible single-cycle G,[C],P layout")


def run_one(case: str, layout: str, args, stage_csv: Path | None) -> int:
    out = args.out / case
    command = [
        sys.executable, str(Path(__file__).with_name("run_32b_all40.py")),
        "--model", str(args.model), "--model-layers", "64",
        "--exe", str(args.exe), "--rpc", args.rpc,
        "--prompt-dir", str(args.prompt_dir),
        "--out", str(out), "--auto-plan", "--htp-only",
        "--force-plan-layout", layout,
        "--force-plan-xc", str(args.chunk),
        "--force-plan-xg", str(args.chunk),
        "--force-plan-xp", str(args.chunk),
        "--streaming", "on", "--batch", "2048", "--ubatch", "2048",
        "--n-predict", "1", "--threads", str(args.threads),
        "--limit", str(args.limit), "--trace-tasks", "--no-mmap",
        "--skip-warmup", "--stop-on-fail",
        "--max-retries", "0",
        "--env", "LLAMA_REGION_RING_GPU_BANK_COUNT=" +
                   ("1" if case in ("A2", "A3") else "2"),
        "--env", "LLAMA_REGION_RING_GPU_BANK_PREFETCH=" +
                   ("1" if case == "A0" else "0"),
    ]
    if stage_csv is not None:
        command += ["--real-stage-profile-csv", str(stage_csv)]
    if args.only_prompt:
        command += ["--only-prompt", args.only_prompt]
    if args.dry_run:
        command += ["--dry-run"]
    print("[ABLATION_RUN]", case, layout, " ".join(f'"{p}"' if " " in p else p for p in command), flush=True)
    if args.plan_only:
        return 0
    return subprocess.call(command, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path(r"E:\llama\models\Qwen3-32B-Q4_K_M.gguf"))
    ap.add_argument("--exe", type=Path, default=ROOT / "build-tensor-v2/bin/Release/llama-completion.exe")
    ap.add_argument("--rpc", default="10.39.52.114:50052")
    ap.add_argument("--prompt-dir", type=Path, default=Path(r"E:\llama\datasets\sharegpt\sharegpt40"))
    ap.add_argument("--out", type=Path, default=ROOT / "32b-ablation")
    ap.add_argument("--cases", nargs="+", choices=("A0", "A1", "A2", "A3"), default=["A0", "A1", "A2", "A3"])
    ap.add_argument("--limit", type=int, default=1, help="only one prompt by default; 0 selects all")
    ap.add_argument("--only-prompt", default="")
    ap.add_argument("--profile-csv", type=Path, help="existing independent CPU/GPU/HTP profile.csv")
    ap.add_argument("--gpu-free-mib", type=int, help="override nvidia-smi free GPU memory")
    ap.add_argument("--cpu-free-mib", type=int, help="override free PC RAM")
    ap.add_argument("--phone-budget-mib", type=int, help="Android MemAvailable in MiB (required for A2/A3)")
    ap.add_argument("--htp-session-mib", type=int, default=2880)
    ap.add_argument("--gpu-reserve-mib", type=int, default=768)
    ap.add_argument("--cpu-reserve-mib", type=int, default=512)
    ap.add_argument("--context", type=int, default=4096, help="KV memory safety estimate")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="write plan manifest, do not invoke runner")
    args = ap.parse_args()
    if args.limit < 0 or args.limit > 40:
        ap.error("--limit must be 0..40")
    if args.model.exists() is False:
        ap.error(f"model not found: {args.model}")
    args.out.mkdir(parents=True, exist_ok=True)
    weights = gguf_layer_weights(args.model, 64)
    has_planning = any(c in ("A2", "A3") for c in args.cases)
    gpu = free_vram_mib(args.gpu_free_mib) if has_planning else None
    cpu = free_ram_mib(args.cpu_free_mib) if has_planning else None
    phone = phone_budget_mib(args.phone_budget_mib) if has_planning else None
    profile = args.profile_csv
    plans = {}
    for case in args.cases:
        if case in ("A0", "A1"):
            layout = FIXED
            details = {"gpu_layers": 40, "cpu_layers": 0, "phone_layers": 24}
        else:
            if case == "A2":
                # A0's C++ independent one-shot grid profile can be reused
                # for both A2 and A3 without rerunning calibration.
                if profile is None:
                    profile = args.out / "A0/independent-profile/profile.csv"
                if not profile.is_file():
                    ap.error(f"A2 requires independent profile CSV: {profile}. "
                             "Run A0 first or supply --profile-csv.")
                measures = measured_ms_per_layer(profile, args.chunk)
                layout, details = multi_cycle(weights, measures, gpu, cpu, phone, args)
            else:
                layout, details = single_cycle(weights, gpu, cpu, phone, args)
            details["gpu_free_mib"] = gpu
            details["cpu_free_mib"] = cpu
            details["phone_budget_mib"] = phone
            details["gpu_reserve_mib"] = args.gpu_reserve_mib
            details["memory_check"] = "OFFLINE_CONSERVATIVE_ESTIMATE_AND_NATIVE_GPU_GUARD"
        plans[case] = {"layout": layout, **details}
        print(f"[ABLATION_PLAN] case={case} layout={layout} info={json.dumps(details)}", flush=True)
        (args.out / f"{case}-plan.json").write_text(
            json.dumps(plans[case], indent=2), encoding="utf-8")
        ret = run_one(case, layout, args, profile)
        if ret == 0 and case == "A0" and profile is None:
            shared = args.out / "A0/independent-profile/profile.csv"
            if shared.is_file():
                profile = shared
                print(f"[ABLATION_PROFILE] source=A0_REUSE csv={profile}", flush=True)
        if ret != 0:
            print(f"[ABLATION_FAIL] case={case} exit_code={ret}", file=sys.stderr)
            return ret
    (args.out / "plans.json").write_text(
        json.dumps(plans, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
