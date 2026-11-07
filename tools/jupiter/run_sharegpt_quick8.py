#!/usr/bin/env python3
"""Select the first two ShareGPT40 prompts per group, then run Jupiter DP.

Selection is from the *existing* 40-prompt dataset, not the raw ShareGPT JSON.
The source is never modified. The downstream runner probes actual token counts,
calibrates q(x,y), refines a measured DP plan, and benchmarks DP only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from jupiter_memory_partition import (
    DeviceMemory, GIB, MIB, pc_cpu_memory, pc_gpu_memory, phone_rpc_memory,
    plan_memory_split, read_gguf_weights,
)

GROUPS = (128, 256, 512, 1024)


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def select_first(source: Path, per_group: int) -> list[dict]:
    if not source.is_dir():
        raise ValueError(f"source directory not found: {source}")
    if per_group < 1:
        raise ValueError("--per-group must be positive")
    picked = []
    for group in GROUPS:
        for index in range(1, per_group + 1):
            name = f"prompt_{group}_{index:02d}.txt"
            file = source / name
            if not file.is_file():
                raise ValueError(f"missing expected prompt: {file}")
            if not file.read_text(encoding="utf-8-sig").strip():
                raise ValueError(f"prompt is empty: {file}")
            picked.append({
                "group": group, "index": index, "prompt": name,
                "source": str(file.resolve()), "bytes": file.stat().st_size,
                "sha256": file_sha256(file),
            })
    return picked


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", default=r"E:\llama\datasets\sharegpt\sharegpt40")
    p.add_argument("--out-dir", default="jupiter-sharegpt-quick8-dp")
    p.add_argument("--per-group", type=int, default=2)
    p.add_argument("--limit", type=int, default=None,
                   help="limit total DP experiments after sampling (1 for smoke test)")
    p.add_argument("--fail-fast", action="store_true",
                   help="stop after the first failed prompt rather than retrying all 8")
    p.add_argument("--exe", default=r".\build-jupiter\bin\Release\llama-cli.exe")
    p.add_argument("--model", default=r"E:\llama\models\Qwen3-14B-Q4_K_M.gguf")
    p.add_argument("--rpc", default="10.39.52.114:10139")
    p.add_argument("--pc-layers", type=int, default=None,
                   help="optional fixed PC boundary, still validated against "
                        "GPU+RPC memory; omitted means auto split by free memory")
    p.add_argument("--cpu-free-mib", type=int, default=None,
                   help="optional PC free RAM override, MiB")
    p.add_argument("--cpu-reserve-mib", type=int, default=2048,
                   help="host RAM reserved for Windows, model cache, etc.")
    p.add_argument("--pc-free-mib", type=int, default=None,
                   help="override free PC GPU memory MiB (normally nvidia-smi)")
    p.add_argument("--phone-free-mib", type=int, default=None,
                   help="override free Phone GPU memory MiB (normally queried "
                        "directly through RPC GET_DEVICE_MEMORY)")
    p.add_argument("--pc-reserve-mib", type=int, default=512,
                   help="extra PC GPU buffer reserve beyond model weights")
    p.add_argument("--phone-reserve-mib", type=int, default=512,
                   help="fixed Phone memory reserve beyond all other allowances")
    p.add_argument("--cpu-workspace-mib", type=int, default=512,
                   help="minimum CPU graph/activation workspace estimate")
    p.add_argument("--gpu-workspace-mib", type=int, default=1024,
                   help="minimum PC GPU graph/activation workspace estimate")
    p.add_argument("--phone-workspace-mib", type=int, default=1536,
                   help="minimum remote graph/activation workspace estimate")
    p.add_argument("--memory-safety-percent", type=float, default=10.0,
                   help="additional percentage of each device's free memory "
                        "held back for allocator fragmentation and uncertainty")
    p.add_argument("--kv-bytes-per-element", type=float, default=2.0,
                   help="KV tensor element bytes, 2=f16 (default, matching CLI)")
    p.add_argument("--kv-per-layer-mib", type=float, default=None,
                   help="explicit KV cache MiB per layer if GGUF attention "
                        "metadata is absent or different from runtime")
    p.add_argument("--rpc-device-index", type=int, default=0)
    p.add_argument("--min-chunk", type=int, default=96)
    p.add_argument("--max-chunk", type=int, default=128)
    p.add_argument("--ubatch", type=int, default=128)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--context", type=int, default=2048)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--q-repeats", type=int, default=1)
    p.add_argument("--refine-repeats", type=int, default=1)
    p.add_argument("--bench-repeats", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=1)
    p.add_argument("--graph-prebuild", action="store_true")
    p.add_argument("--strict-refine", action="store_true",
                   help="require exact-q DP reoptimization convergence; slower")
    p.add_argument("--dry-run", action="store_true",
                   help="show selected source prompts without copying or running")
    p.add_argument("--sample-only", action="store_true",
                   help="copy selected prompts and write manifest, then exit")
    p.add_argument("--memory-check-only", action="store_true",
                   help="query CUDA/Phone free memory and print the proposed "
                        "layer split without starting any prompt inference")
    return p


def make_identity(args: argparse.Namespace, picked: list[dict]) -> dict:
    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    return {
        "version": 1, "selection": picked,
        "execution": {
            "exe": str(exe), "model": str(model), "rpc": args.rpc,
            "pc_layers": args.pc_layers,
            "cpu_layers": args.cpu_layers,
            "layer_allocation": "FREE_CPU_GPU_RPC_MEMORY",
            "cpu_reserve_mib": args.cpu_reserve_mib,
            "pc_reserve_mib": args.pc_reserve_mib,
            "phone_reserve_mib": args.phone_reserve_mib,
            "cpu_workspace_mib": args.cpu_workspace_mib,
            "gpu_workspace_mib": args.gpu_workspace_mib,
            "phone_workspace_mib": args.phone_workspace_mib,
            "memory_safety_percent": args.memory_safety_percent,
            "kv_bytes_per_element": args.kv_bytes_per_element,
            "kv_per_layer_mib": args.kv_per_layer_mib,
            "rpc_device_index": args.rpc_device_index,
            "min_chunk": args.min_chunk, "max_chunk": args.max_chunk,
            "ubatch": args.ubatch, "batch": args.batch,
            "context": args.context, "threads": args.threads,
            "window": args.window, "q_repeats": args.q_repeats,
            "refine_repeats": args.refine_repeats,
            "bench_repeats": args.bench_repeats,
            "max_rounds": args.max_rounds,
            "graph_prebuild": args.graph_prebuild,
            "strict_refine": args.strict_refine,
        },
    }


def prepare_sample(out: Path, picked: list[dict],
                   identity: dict | None) -> Path:
    manifest = out / "quick8-config.json"
    if identity is not None and manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous != identity:
            raise ValueError(
                f"existing run configuration differs: {manifest}; "
                "use a separate --out-dir for another model or settings"
            )
    prompt_dir = out / "sampled_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for entry in picked:
        dest = prompt_dir / entry["prompt"]
        if dest.exists():
            if file_sha256(dest) != entry["sha256"]:
                raise ValueError(f"sampled file has changed: {dest}")
        else:
            shutil.copy2(entry["source"], dest)
            if file_sha256(dest) != entry["sha256"]:
                raise RuntimeError(f"file copy checksum mismatch: {dest}")
    with (out / "sample-manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group", "index", "prompt", "source", "bytes", "sha256"
        ])
        writer.writeheader()
        writer.writerows(picked)
    if identity is not None:
        manifest.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return prompt_dir


def choose_layers_by_memory(args: argparse.Namespace):
    """Read free device memory; pick a contiguous boundary before any model run."""
    weights = read_gguf_weights(Path(args.model).resolve())
    pc = (pc_gpu_memory() if args.pc_free_mib is None else
          DeviceMemory(args.pc_free_mib*MIB, args.pc_free_mib*MIB))
    cpu = (pc_cpu_memory() if args.cpu_free_mib is None else
           DeviceMemory(args.cpu_free_mib*MIB, args.cpu_free_mib*MIB))
    phone = (phone_rpc_memory(args.rpc, args.rpc_device_index)
             if args.phone_free_mib is None else
             DeviceMemory(args.phone_free_mib*MIB, args.phone_free_mib*MIB))
    plan = plan_memory_split(
        weights, pc, phone,
        pc_reserve_mib=args.pc_reserve_mib,
        phone_reserve_mib=args.phone_reserve_mib,
        pc_layers_override=args.pc_layers,
        cpu=cpu,
        cpu_reserve_mib=args.cpu_reserve_mib,
        context=args.context,
        ubatch=args.ubatch,
        window=args.window,
        graph_prebuild=args.graph_prebuild,
        kv_dtype_bytes=args.kv_bytes_per_element,
        kv_per_layer_mib=args.kv_per_layer_mib,
        cpu_workspace_mib=args.cpu_workspace_mib,
        pc_workspace_mib=args.gpu_workspace_mib,
        phone_workspace_mib=args.phone_workspace_mib,
        safety_percent=args.memory_safety_percent,
    )
    if phone.free_bytes > 12*GIB:
        print(
            "[JUPITER_MEMORY_WARNING] Phone RPC reports over 12 GiB free. "
            "Confirm the selected RPC device is the intended accelerator "
            "and the reported memory is actually allocatable by that backend.",
            flush=True,
        )
    print(
        f"[JUPITER_MEMORY_SPLIT] architecture={weights.architecture} "
        f"total_layers={weights.n_layers} "
        f"PC_CPU={plan.cpu_layers} PC_GPU={plan.pc_gpu_layers} "
        f"Phone={plan.phone_layers} pc_boundary={plan.pc_layers} "
        f"pc_free_gib={pc.free_bytes/GIB:.2f} "
        f"phone_free_gib={phone.free_bytes/GIB:.2f} "
        f"cpu_free_gib={cpu.free_bytes/GIB:.2f} "
        f"cpu_budget_gib={plan.cpu_budget_bytes/GIB:.2f} "
        f"pc_budget_gib={plan.pc_budget_bytes/GIB:.2f} "
        f"phone_budget_gib={plan.phone_budget_bytes/GIB:.2f} "
        f"cpu_weights_gib={plan.cpu_weight_bytes/GIB:.2f} "
        f"pc_weights_gib={plan.pc_weight_bytes/GIB:.2f} "
        f"phone_weights_gib={plan.phone_weight_bytes/GIB:.2f} "
        f"kv_layer_mib={plan.kv_per_layer_bytes/MIB:.1f} "
        f"pc_kv_mib={plan.pc_kv_bytes/MIB:.1f} "
        f"phone_kv_mib={plan.phone_kv_bytes/MIB:.1f} "
        f"gpu_workspace_mib={plan.pc_workspace_bytes/MIB:.0f} "
        f"phone_workspace_mib={plan.phone_workspace_bytes/MIB:.0f} "
        f"safety_percent={args.memory_safety_percent:.1f} "
        f"mode={'OVERRIDE_CHECKED' if args.pc_layers is not None else 'AUTO_BY_MEMORY'}",
        flush=True,
    )
    for name, free, weights_size, kv, workspace, reserve, safety in (
        ("PC_CPU", cpu.free_bytes, plan.cpu_weight_bytes,
         plan.cpu_kv_bytes, plan.cpu_workspace_bytes,
         plan.cpu_reserve_bytes, plan.cpu_safety_bytes),
        ("PC_GPU", pc.free_bytes, plan.pc_weight_bytes,
         plan.pc_kv_bytes, plan.pc_workspace_bytes,
         plan.pc_reserve_bytes, plan.pc_safety_bytes),
        ("PHONE", phone.free_bytes, plan.phone_weight_bytes,
         plan.phone_kv_bytes, plan.phone_workspace_bytes,
         plan.phone_reserve_bytes, plan.phone_safety_bytes),
    ):
        remaining = free-weights_size-kv-workspace-reserve-safety
        print(
            f"[JUPITER_MEMORY_BUDGET] device={name} "
            f"free_gib={free/GIB:.2f} weights_gib={weights_size/GIB:.2f} "
            f"kv_gib={kv/GIB:.2f} workspace_gib={workspace/GIB:.2f} "
            f"fixed_reserve_gib={reserve/GIB:.2f} "
            f"safety_gib={safety/GIB:.2f} "
            f"remaining_gib={remaining/GIB:.2f} status="
            f"{'FITS_ESTIMATE' if remaining >= 0 else 'OVERCOMMITTED'}",
            flush=True,
        )
    return plan


def save_layer_plan(out: Path, plan) -> None:
    details = {
        "method": "GGUF_WEIGHTS_KV_CONTEXT_GRAPH_WORKSPACE_SAFETY",
        "notice": "Estimate only; backend max allocation, local RPC source "
                  "and actual graph buffers require runtime validation.",
        "context_tokens": plan.context_tokens,
        "kv_dtype_bytes": plan.kv_dtype_bytes,
        "kv_per_layer_mib": round(plan.kv_per_layer_bytes/MIB, 3),
        "cpu_kv_gib": round(plan.cpu_kv_bytes/GIB, 3),
        "pc_kv_gib": round(plan.pc_kv_bytes/GIB, 3),
        "phone_kv_gib": round(plan.phone_kv_bytes/GIB, 3),
        "cpu_workspace_gib": round(plan.cpu_workspace_bytes/GIB, 3),
        "pc_workspace_gib": round(plan.pc_workspace_bytes/GIB, 3),
        "phone_workspace_gib": round(plan.phone_workspace_bytes/GIB, 3),
        "cpu_safety_gib": round(plan.cpu_safety_bytes/GIB, 3),
        "pc_safety_gib": round(plan.pc_safety_bytes/GIB, 3),
        "phone_safety_gib": round(plan.phone_safety_bytes/GIB, 3),
        "estimated_remaining_gib": {
            "cpu": round((
                plan.cpu_available_bytes-plan.cpu_weight_bytes
                -plan.cpu_kv_bytes-plan.cpu_workspace_bytes
                -plan.cpu_reserve_bytes-plan.cpu_safety_bytes
            )/GIB, 3),
            "pc_gpu": round((
                plan.pc_available_bytes-plan.pc_weight_bytes
                -plan.pc_kv_bytes-plan.pc_workspace_bytes
                -plan.pc_reserve_bytes-plan.pc_safety_bytes
            )/GIB, 3),
            "phone": round((
                plan.phone_available_bytes-plan.phone_weight_bytes
                -plan.phone_kv_bytes-plan.phone_workspace_bytes
                -plan.phone_reserve_bytes-plan.phone_safety_bytes
            )/GIB, 3),
        },
        "architecture": plan.model.architecture,
        "total_layers": plan.model.n_layers,
        "pc_layers": plan.pc_layers,
        "cpu_layers": plan.cpu_layers,
        "pc_gpu_layers": plan.pc_gpu_layers,
        "phone_layers": plan.phone_layers,
        "cpu_layer_range": [0, plan.cpu_layers-1] if plan.cpu_layers else [],
        "pc_layer_range": [plan.cpu_layers, plan.pc_layers-1],
        "phone_layer_range": [plan.pc_layers, plan.model.n_layers-1],
        "total_tensor_weights_gib": round(plan.model.total_weights/GIB, 3),
        "nonblock_tensor_weights_gib": round(plan.model.other_bytes/GIB, 3),
        "cpu_weight_gib": round(plan.cpu_weight_bytes/GIB, 3),
        "cpu_free_gib": round(plan.cpu_available_bytes/GIB, 3),
        "cpu_reserve_gib": round(plan.cpu_reserve_bytes/GIB, 3),
        "cpu_budget_gib": round(plan.cpu_budget_bytes/GIB, 3),
        "pc_weight_gib": round(plan.pc_weight_bytes/GIB, 3),
        "phone_weight_gib": round(plan.phone_weight_bytes/GIB, 3),
        "pc_free_gib": round(plan.pc_available_bytes/GIB, 3),
        "phone_free_gib": round(plan.phone_available_bytes/GIB, 3),
        "pc_reserve_gib": round(plan.pc_reserve_bytes/GIB, 3),
        "phone_reserve_gib": round(plan.phone_reserve_bytes/GIB, 3),
        "pc_budget_gib": round(plan.pc_budget_bytes/GIB, 3),
        "phone_budget_gib": round(plan.phone_budget_bytes/GIB, 3),
    }
    (out/"memory-plan.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2)+"\n",
        encoding="utf-8",
    )


def build_command(args: argparse.Namespace, prompt_dir: Path, out: Path) -> list[str]:
    runner = Path(__file__).resolve().with_name("run_sharegpt40_dp.py")
    if not runner.is_file():
        raise ValueError(f"missing Jupiter runner beside this script: {runner}")
    cmd = [
        sys.executable, str(runner),
        "--prompts-dir", str(prompt_dir),
        "--out-dir", str(out / "dp"),
        "--groups", ",".join(str(g) for g in GROUPS),
        "--exe", str(Path(args.exe).resolve()),
        "--model", str(Path(args.model).resolve()),
        "--rpc", args.rpc,
        "--pc-layers", str(args.pc_layers), "--cpu-layers", str(args.cpu_layers),
        "--min-chunk", str(args.min_chunk),
        "--max-chunk", str(args.max_chunk),
        "--ubatch", str(args.ubatch), "--batch", str(args.batch),
        "--context", str(args.context), "--threads", str(args.threads),
        "--window", str(args.window),
        "--q-repeats", str(args.q_repeats),
        "--refine-repeats", str(args.refine_repeats),
        "--bench-repeats", str(args.bench_repeats),
        "--max-rounds", str(args.max_rounds),
        "--resume", "--only-incomplete",
    ]
    if args.graph_prebuild:
        cmd.append("--graph-prebuild")
    if getattr(args, "limit", None) is not None:
        cmd.extend(["--limit", str(args.limit)])
    if getattr(args, "fail_fast", False):
        cmd.append("--fail-fast")
    if not args.strict_refine:
        cmd.append("--budgeted-refine")
    return cmd


def main() -> int:
    args = make_parser().parse_args()
    if not (0 < args.min_chunk <= args.max_chunk <= args.ubatch <= args.batch
            <= args.context):
        raise ValueError("require 0 < min-chunk <= max-chunk <= ubatch <= batch <= context")
    if (min(args.threads, args.window, args.q_repeats, args.refine_repeats,
            args.bench_repeats, args.max_rounds) < 1
            or (args.limit is not None and args.limit < 1)
            or (args.pc_layers is not None and args.pc_layers < 1)
            or (args.cpu_free_mib is not None and args.cpu_free_mib < 1)
            or (args.pc_free_mib is not None and args.pc_free_mib < 1)
            or (args.phone_free_mib is not None and args.phone_free_mib < 1)
            or args.cpu_reserve_mib < 0
            or args.pc_reserve_mib < 0 or args.phone_reserve_mib < 0
            or min(args.cpu_workspace_mib, args.gpu_workspace_mib,
                   args.phone_workspace_mib) < 0
            or not (0 <= args.memory_safety_percent <= 50)
            or not (0 < args.kv_bytes_per_element <= 8)
            or (args.kv_per_layer_mib is not None
                and args.kv_per_layer_mib < 0)
            or args.rpc_device_index < 0):
        raise ValueError("invalid layer, memory or repetition parameters")
    source = Path(args.source_dir).resolve()
    picked = select_first(source, args.per_group)
    print(f"[JUPITER_QUICK8_SELECT] groups={GROUPS} per_group={args.per_group} "
          f"total={len(picked)}", flush=True)
    for item in picked:
        print(f"  {item['prompt']} ({item['bytes']} bytes)", flush=True)
    if args.dry_run:
        return 0
    out = Path(args.out_dir).resolve()
    if args.sample_only:
        prompt_dir = prepare_sample(out, picked, None)
        print(f"[JUPITER_QUICK8_SAMPLE] path={prompt_dir} "
              f"files={len(picked)} mode=SAMPLE_ONLY", flush=True)
        return 0
    if not Path(args.model).is_file():
        raise ValueError(f"--model not found: {args.model}")
    if not args.memory_check_only and not Path(args.exe).is_file():
        raise ValueError(f"--exe not found: {args.exe}")
    memory_plan = choose_layers_by_memory(args)
    if args.memory_check_only:
        print("[JUPITER_MEMORY_CHECK] status=FEASIBLE "
              "note=WEIGHT_ESTIMATE_NOT_RUNTIME_GUARANTEE", flush=True)
        return 0
    # Resolve the memory-dependent boundary before composing case identities.
    # A changed boundary must not silently reuse measurements from old runs.
    args.pc_layers = memory_plan.pc_layers
    args.cpu_layers = memory_plan.cpu_layers
    prompt_dir = prepare_sample(out, picked, make_identity(args, picked))
    save_layer_plan(out, memory_plan)
    print(f"[JUPITER_QUICK8_SAMPLE] path={prompt_dir} files={len(picked)}", flush=True)
    cmd = build_command(args, prompt_dir, out)
    print("[JUPITER_QUICK8_RUN] " + subprocess.list2cmdline(cmd), flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    print(f"[JUPITER_QUICK8_DONE] rc={rc} "
          f"results={out / 'dp' / 'dp-results.csv'} "
          f"groups={out / 'dp' / 'dp-groups.csv'}", flush=True)
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(f"[JUPITER_QUICK8_ERROR] {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("[JUPITER_QUICK8_STOP] interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130)
